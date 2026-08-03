import os
import json

import pytest


# Allow at most two deterministic MMFF non-convergences in the eight-conformer batch.
MIN_CONVERGED = 6


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)
def test_nvmolkit_gpu_workflow(tmp_path):
    import nvmolkit
    import pandas as pd
    import torch

    from chemistry_workflow import (
        RepresentativePolicy,
        WorkflowPhase,
        WorkflowState,
        build_workflow_report,
        discover_fused_butina_clusters,
        embed_representative_conformers,
        generate_morgan_fingerprints,
        inspect_library,
        measure_tanimoto_similarity,
        optimize_conformers_mmff94,
    )

    assert torch.cuda.is_available(), "A CUDA-capable NVIDIA GPU is required."
    assert "L4" in torch.cuda.get_device_name(0), (
        f"GPU acceptance requires an NVIDIA L4; found {torch.cuda.get_device_name(0)}"
    )
    assert torch.cuda.get_device_capability(0) >= (7, 0)
    assert nvmolkit.__version__ == "0.5.0", (
        f"GPU acceptance requires nvMolKit 0.5.0; found {nvmolkit.__version__}"
    )

    smiles = ("CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC")
    data_path = tmp_path / "gpu_acceptance_molecules.csv"
    pd.DataFrame(
        [
            {"molecule_id": f"mol-{index}", "smiles": smile}
            for index, smile in enumerate(smiles * 32)
        ]
    ).to_csv(data_path, index=False)
    state = WorkflowState()
    inspect_library(state, data_path, expected_rows=192)
    generate_morgan_fingerprints(
        state, fingerprint_radius=2, fingerprint_size=1024
    )
    measure_tanimoto_similarity(state)
    discover_fused_butina_clusters(state, cluster_cutoff=0.5)

    similarity = state.similarity.torch()
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

    clustered_indices = [index for cluster in state.clusters for index in cluster]
    assert len(clustered_indices) == 192
    assert sorted(clustered_indices) == list(range(192))
    cluster_by_index = [-1] * 192
    for cluster_id, cluster in enumerate(state.clusters):
        for molecule_index in cluster:
            cluster_by_index[molecule_index] = cluster_id
    assert len(state.clusters) > 1
    assert cluster_by_index[0] != cluster_by_index[3]
    for smiles_index in range(len(smiles)):
        repeated_cluster_ids = {
            cluster_by_index[smiles_index + repeat_index * len(smiles)]
            for repeat_index in range(32)
        }
        assert len(repeated_cluster_ids) == 1

    embed_representative_conformers(
        state,
        representative_count=4,
        representative_policy=RepresentativePolicy.INCLUDE_SINGLETON_IF_AVAILABLE,
        conformers_per_representative=3,
    )
    optimize_conformers_mmff94(state)
    report = build_workflow_report(state)

    assert state.phase is WorkflowPhase.OPTIMIZED
    assert [record.key for record in report.evidence] == [
        "E01", "E02", "E03", "E04", "E05", "E06"
    ]
    assert [record.provenance for record in report.evidence[1:]] == [
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    ]
    evidence = {record.key: json.loads(record.payload_json) for record in report.evidence}
    assert evidence["E01"]["valid_count"] == 192
    assert evidence["E02"]["packed_shape"] == [192, 32]
    assert evidence["E03"]["matrix_shape"] == [192, 192]
    assert evidence["E04"]["assignment_count"] == 192

    optimization_result = state.optimization_result
    energies = optimization_result.energies.torch()
    converged = optimization_result.converged.torch()
    mol_indices = optimization_result.mol_indices.torch()
    conf_indices = optimization_result.conf_indices.torch()
    attempted = evidence["E06"]["attempted_conformer_count"]
    assert len(energies) == len(converged) == len(mol_indices) == len(conf_indices) == attempted
    assert torch.isfinite(energies).all().item()

    result_pairs = list(zip(mol_indices.tolist(), conf_indices.tolist()))
    assert all(0 <= mol_index < len(state.conformer_molecules) for mol_index, _ in result_pairs)
    assert all(
        0 <= conf_index < state.conformer_molecules[mol_index].GetNumConformers()
        for mol_index, conf_index in result_pairs
    )
    assert len(set(result_pairs)) == attempted
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
        f"MMFF converged {converged_count}/{attempted}; minimum is {MIN_CONVERGED}; "
        f"unconverged conformers: {unconverged_pairs}"
    )

    coordinates_by_molecule = optimization_result.per_molecule()
    assert len(coordinates_by_molecule) == len(state.conformer_molecules)
    for molecule, conformer_coordinates in zip(
        state.conformer_molecules, coordinates_by_molecule
    ):
        assert len(conformer_coordinates) == molecule.GetNumConformers()
        for coordinates in conformer_coordinates:
            assert tuple(coordinates.shape) == (molecule.GetNumAtoms(), 3)
            assert torch.isfinite(coordinates).all().item()
