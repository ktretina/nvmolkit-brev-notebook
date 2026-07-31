import os

import pytest


# Allow at most two deterministic MMFF non-convergences in the eight-conformer batch.
MIN_CONVERGED = 6


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)
def test_nvmolkit_gpu_workflow():
    import nvmolkit
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
    assert nvmolkit.__version__ == "0.5.0", (
        f"GPU acceptance requires nvMolKit 0.5.0; found {nvmolkit.__version__}"
    )

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
    expected_one = torch.tensor(1.0, dtype=similarity.dtype, device=similarity.device)
    assert torch.allclose(
        similarity.diagonal(),
        torch.ones_like(similarity.diagonal()),
        rtol=0,
        atol=1e-7,
    )
    assert torch.allclose(similarity, similarity.T, rtol=0, atol=1e-7)
    assert torch.isclose(similarity[0, 6], expected_one, rtol=0, atol=1e-7).item()

    clustered_indices = [index for cluster in clusters for index in cluster]
    assert len(clustered_indices) == 192
    assert sorted(clustered_indices) == list(range(192))
    cluster_by_index = [-1] * 192
    for cluster_id, cluster in enumerate(clusters):
        for molecule_index in cluster:
            cluster_by_index[molecule_index] = cluster_id
    assert len(clusters) > 1
    assert cluster_by_index[0] != cluster_by_index[3]
    for smiles_index in range(len(smiles)):
        repeated_cluster_ids = {
            cluster_by_index[smiles_index + repeat_index * len(smiles)]
            for repeat_index in range(32)
        }
        assert len(repeated_cluster_ids) == 1

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

    result_pairs = list(zip(mol_indices.tolist(), conf_indices.tolist()))
    assert all(0 <= mol_index < 4 for mol_index, _conf_index in result_pairs)
    assert all(0 <= conf_index < 2 for _mol_index, conf_index in result_pairs)
    assert len(set(result_pairs)) == 8
    convergence_values = [int(value) for value in converged.tolist()]
    assert set(convergence_values) <= {0, 1}
    unconverged_pairs = [
        pair
        for pair, did_converge in zip(result_pairs, convergence_values)
        if not did_converge
    ]
    converged_count = sum(convergence_values)
    assert any(convergence_values)
    assert converged_count >= MIN_CONVERGED, (
        f"MMFF converged {converged_count}/8; minimum is {MIN_CONVERGED}; "
        f"unconverged conformers: {unconverged_pairs}"
    )

    coordinates_by_molecule = optimization_result.per_molecule()
    assert len(coordinates_by_molecule) == 4
    assert all(len(coordinates) == 2 for coordinates in coordinates_by_molecule)
    for molecule, conformer_coordinates in zip(
        conformer_molecules, coordinates_by_molecule
    ):
        for coordinates in conformer_coordinates:
            assert tuple(coordinates.shape) == (molecule.GetNumAtoms(), 3)
            assert torch.isfinite(coordinates).all().item()
