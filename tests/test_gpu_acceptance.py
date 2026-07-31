import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)
def test_nvmolkit_gpu_workflow():
    import torch
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3

    from nvmolkit.clustering import fused_butina
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.similarity import crossTanimotoSimilarity
    from nvmolkit.types import CoordinateOutput, Device3DResult

    assert torch.cuda.is_available(), "A CUDA-capable NVIDIA GPU is required."

    smiles = ("CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC")
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles * 32]
    assert len(molecules) == 192
    assert all(molecule is not None for molecule in molecules)

    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=1024).GetFingerprints(
        molecules
    )
    similarity_result = crossTanimotoSimilarity(fingerprints)
    clusters, _cluster_sizes = fused_butina(fingerprints.torch(), cutoff=0.5)
    torch.cuda.synchronize()

    similarity = similarity_result.torch()
    assert tuple(similarity.shape) == (192, 192)
    assert torch.isfinite(similarity).all().item()
    assert ((similarity >= 0) & (similarity <= 1)).all().item()

    clustered_indices = [index for cluster in clusters for index in cluster]
    assert len(clustered_indices) == 192
    assert sorted(clustered_indices) == list(range(192))

    conformer_molecules = [Chem.AddHs(Chem.Mol(molecule)) for molecule in molecules[:4]]
    embedding_params = ETKDGv3()
    embedding_params.useRandomCoords = True
    embedding_params.randomSeed = 7
    EmbedMolecules(
        conformer_molecules,
        embedding_params,
        confsPerMolecule=2,
        maxIterations=-1,
    )
    optimization_result = MMFFOptimizeMoleculesConfs(
        conformer_molecules,
        maxIters=200,
        output=CoordinateOutput.DEVICE,
    )
    torch.cuda.synchronize()

    assert isinstance(optimization_result, Device3DResult)
    energies = optimization_result.energies.torch()
    converged = optimization_result.converged.torch()
    mol_indices = optimization_result.mol_indices.torch()
    conf_indices = optimization_result.conf_indices.torch()
    assert len(energies) == len(converged) == len(mol_indices) == len(conf_indices) == 8
    assert torch.isfinite(energies).all().item()
    assert torch.isfinite(converged).all().item()
    assert torch.isfinite(mol_indices).all().item()
    assert torch.isfinite(conf_indices).all().item()

    result_pairs = list(zip(mol_indices.tolist(), conf_indices.tolist()))
    assert all(0 <= mol_index < 4 for mol_index, _conf_index in result_pairs)
    assert all(0 <= conf_index < 2 for _mol_index, conf_index in result_pairs)
    assert len(set(result_pairs)) == 8

    coordinates_by_molecule = optimization_result.per_molecule()
    assert len(coordinates_by_molecule) == 4
    assert all(len(coordinates) == 2 for coordinates in coordinates_by_molecule)
    for molecule, conformer_coordinates in zip(
        conformer_molecules, coordinates_by_molecule
    ):
        for coordinates in conformer_coordinates:
            assert tuple(coordinates.shape) == (molecule.GetNumAtoms(), 3)
            assert torch.isfinite(coordinates).all().item()

    assert (converged == 1).any().item()
