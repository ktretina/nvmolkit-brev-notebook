import os
import json
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest


# Request up to 12 conformers (4 representatives × 3); require at least half to converge.
MIN_CONVERGED = 6


def test_gpu_acceptance_source_gates_default_objective_challenge():
    source = inspect.getsource(test_nvmolkit_gpu_workflow)

    for required in (
        "sample_molecules.csv",
        "objective_required=True",
        "begin_objective_challenge",
        "benchmark_score > context.baseline_score",
        'objective_evidence.key == "O01"',
        'termination_reason == "target_achieved"',
    ):
        assert required in source


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)
def test_nvmolkit_gpu_workflow():
    import nvmolkit
    import torch

    from chemistry_workflow import (
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
    from demo_agent import BoundedWorkflowController, STAGES
    from objective_challenge import evaluate_diverse_panel

    assert torch.cuda.is_available(), "A CUDA-capable NVIDIA GPU is required."
    assert "L4" in torch.cuda.get_device_name(0), (
        f"GPU acceptance requires an NVIDIA L4; found {torch.cuda.get_device_name(0)}"
    )
    assert torch.cuda.get_device_capability(0) >= (7, 0)
    assert nvmolkit.__version__ == "0.5.0", (
        f"GPU acceptance requires nvMolKit 0.5.0; found {nvmolkit.__version__}"
    )

    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_molecules.csv"
    state = WorkflowState()

    stage_arguments = {
        "inspect_library": {},
        "generate_morgan_fingerprints": {
            "radius": 2,
            "size": 1024,
            "decision_basis": "Use the qualification fingerprint parameters.",
        },
        "measure_tanimoto_similarity": {},
        "discover_fused_butina_clusters": {
            "cutoff": 0.5,
            "decision_basis": "Use the qualification clustering cutoff.",
        },
        "embed_representative_conformers": {
            "representative_count": 4,
            "policy": "include_singleton_if_available",
            "conformers_per_representative": 3,
            "decision_basis": "Use the qualification conformer sample size.",
        },
        "optimize_conformers_mmff94": {},
    }
    plan_arguments = {
        "stages": [
            {
                "stage": stage,
                "rationale": f"Run {stage.replace('_', ' ')} after its prerequisite.",
            }
            for stage in STAGES
        ]
    }

    class ScriptedCompletions:
        def __init__(self):
            self.expected_names = ["submit_workflow_plan", *STAGES]
            self.arguments = [
                plan_arguments,
                *(stage_arguments[stage] for stage in STAGES),
            ]
            self.calls = []

        def create(self, **kwargs):
            call_index = len(self.calls)
            assert call_index < len(self.expected_names)
            expected_name = self.expected_names[call_index]
            assert kwargs["tool_choice"] == {
                "type": "function",
                "function": {"name": expected_name},
            }
            assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
                expected_name
            ]
            self.calls.append(kwargs)
            tool_call = SimpleNamespace(
                id=f"gpu-acceptance-{call_index}",
                type="function",
                function=SimpleNamespace(
                    name=expected_name,
                    arguments=json.dumps(self.arguments[call_index]),
                ),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tool_call])
                    )
                ]
            )

    completions = ScriptedCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    controller = BoundedWorkflowController.create(
        "Qualify the fixed nvMolKit workflow on the task-owned L4 GPU.",
        "nvapi-gpu-acceptance-placeholder",
        client=client,
        state=state,
        objective_required=True,
        executors={
            "inspect_library": lambda active_state: inspect_library(
                active_state, data_path, expected_rows=256
            ),
            "generate_morgan_fingerprints": generate_morgan_fingerprints,
            "measure_tanimoto_similarity": measure_tanimoto_similarity,
            "discover_fused_butina_clusters": discover_fused_butina_clusters,
            "embed_representative_conformers": embed_representative_conformers,
            "optimize_conformers_mmff94": optimize_conformers_mmff94,
            "build_workflow_report": build_workflow_report,
        },
    )
    plan = controller.request_plan()
    assert tuple(item.stage for item in plan.stages) == STAGES
    for stage in STAGES:
        proposal = controller.request_next_stage()
        assert proposal.stage == stage
        result = controller.execute_pending(proposal.arguments)
        assert result.stage == stage

    scientific = controller.scientific_result()
    assert scientific.turn_count == 7 == len(completions.calls)
    assert tuple(result.stage for result in scientific.stage_results) == STAGES
    assistant_messages = [
        message for message in scientific.messages if message["role"] == "assistant"
    ]
    tool_messages = [
        message for message in scientific.messages if message["role"] == "tool"
    ]
    assert len(assistant_messages) == len(tool_messages) == 7
    assert [message["tool_call_id"] for message in tool_messages] == [
        message["tool_calls"][0]["id"] for message in assistant_messages
    ]

    context = controller.begin_objective_challenge()
    assert len(context.candidates) == 8
    assert context.benchmark_score > context.baseline_score
    candidate_ids = tuple(candidate.molecule_id for candidate in context.candidates)
    best_panel = max(
        itertools.combinations(candidate_ids, 4),
        key=lambda panel: evaluate_diverse_panel(
            context,
            panel,
            attempt_number=2,
            decision_basis="Evaluate one bounded candidate panel.",
        ).score,
    )
    completions.expected_names.extend(["select_diverse_panel", "select_diverse_panel"])
    completions.arguments.extend(
        [
            {
                "selected_ids": list(context.baseline_ids),
                "decision_basis": "Measure the defined baseline before revising it.",
            },
            {
                "selected_ids": list(best_panel),
                "decision_basis": "Replace the measured limiting analogue pair.",
            },
        ]
    )
    for _ in range(2):
        proposal = controller.request_objective_attempt()
        controller.execute_objective_attempt(proposal)
    assert controller.objective_run is not None
    assert 1 <= len(controller.objective_run.attempts) <= 3
    assert controller.objective_run.termination_reason == "target_achieved"
    assert controller.objective_run.final_score >= context.target_score
    assert controller.objective_evidence.key == "O01"
    assert controller.session.turn_count == 9 == len(completions.calls)
    assistant_messages = [
        message for message in controller.session.messages if message["role"] == "assistant"
    ]
    tool_messages = [
        message for message in controller.session.messages if message["role"] == "tool"
    ]
    assert len(assistant_messages) == len(tool_messages) == 9
    assert [message["tool_call_id"] for message in tool_messages] == [
        message["tool_calls"][0]["id"] for message in assistant_messages
    ]

    fingerprints = state.fingerprints.torch()
    assert fingerprints.is_cuda
    assert fingerprints.device.type == "cuda"

    similarity = state.similarity.torch()
    assert similarity.is_cuda
    assert similarity.device.type == "cuda"
    assert tuple(similarity.shape) == (256, 256)
    assert torch.isfinite(similarity).all().item()
    assert ((similarity >= 0) & (similarity <= 1)).all().item()
    assert torch.allclose(
        similarity.diagonal(),
        torch.ones_like(similarity.diagonal()),
        rtol=0,
        atol=1e-7,
    )
    assert torch.allclose(similarity, similarity.T, rtol=0, atol=1e-7)

    clustered_indices = [index for cluster in state.clusters for index in cluster]
    assert len(clustered_indices) == 256
    assert sorted(clustered_indices) == list(range(256))
    cluster_by_index = [-1] * 256
    for cluster_id, cluster in enumerate(state.clusters):
        for molecule_index in cluster:
            cluster_by_index[molecule_index] = cluster_id
    assert len(state.clusters) >= 8
    assert len(set(cluster_by_index)) == len(state.clusters)
    assert len({candidate.cluster_id for candidate in context.candidates}) == 8

    report = scientific.report

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
    assert evidence["E01"]["valid_count"] == 256
    assert evidence["E02"]["packed_shape"] == [256, 32]
    assert evidence["E03"]["matrix_shape"] == [256, 256]
    assert evidence["E04"]["assignment_count"] == 256

    optimization_result = state.optimization_result
    energies = optimization_result.energies.torch()
    converged = optimization_result.converged.torch()
    mol_indices = optimization_result.mol_indices.torch()
    conf_indices = optimization_result.conf_indices.torch()
    assert energies.is_cuda
    assert converged.is_cuda
    assert mol_indices.is_cuda
    assert conf_indices.is_cuda
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
            assert coordinates.is_cuda
            assert tuple(coordinates.shape) == (molecule.GetNumAtoms(), 3)
            assert torch.isfinite(coordinates).all().item()
