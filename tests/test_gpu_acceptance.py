import json
import itertools
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# Request up to 20 conformers (4 representatives × 5); require at least half to converge.
MIN_CONVERGED = 10


def test_gpu_source_gate_covers_decision_ladder_and_persistent_conclusion():
    source = inspect.getsource(test_nvmolkit_gpu_workflow)
    for required in (
        "certify_argmax_reachability",
        "state_id",
        "accepted_maxima",
        "limiting_pairs",
        'objective_evidence.key == "O01"',
        "request_synthesis",
        "evidence_controlled_conclusion",
        'nvmolkit.__version__ == "0.6.0"',
        "normalize_fused_butina_result",
        "direct_labels",
        "direct_centroids",
    ):
        assert required in source


def test_cpu_acceptance_contract_covers_certified_objective_terminal_evidence():
    from objective_challenge import (
        TerminationReason,
        accepted_maxima,
        build_action_menu,
        build_objective_context,
        build_objective_evidence,
        certify_argmax_reachability,
        evaluate_selected_swap,
        measure_panel,
        terminal_objective_run,
    )
    from objective_fixtures import optimized_state

    context = build_objective_context(optimized_state())
    baseline = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, baseline, 0)
    maxima = accepted_maxima(menu)
    attempt = evaluate_selected_swap(context, menu, maxima[0], 1)
    run = terminal_objective_run(context, (attempt,), TerminationReason.TARGET_ACHIEVED)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert menu.state_id.startswith("state-")
    assert maxima and all(action.limiting_pairs for action in menu.actions)
    assert certify_argmax_reachability(context) is True
    assert run.termination_reason == "target_achieved"
    assert payload["attempts"][0]["state_id"] == menu.state_id
    assert payload["baseline"]["limiting_pairs"]
    assert payload["final_measurement"]["limiting_pairs"]


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)
def test_nvmolkit_gpu_workflow():
    import nvmolkit
    import torch
    from nvmolkit.clustering import fused_butina

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
    from demo_agent import (
        BoundedWorkflowController,
        EvidenceControlledConclusion,
        STAGES,
    )
    from objective_challenge import (
        accepted_maxima,
        build_action_menu,
        certify_argmax_reachability,
        evaluate_selected_swap,
        is_strict_improvement,
        measure_panel,
        target_is_achieved,
    )
    from notebooks.nvmolkit_compat import normalize_fused_butina_result

    assert torch.cuda.is_available(), "A CUDA-capable NVIDIA GPU is required."
    assert "L4" in torch.cuda.get_device_name(0), (
        f"GPU acceptance requires an NVIDIA L4; found {torch.cuda.get_device_name(0)}"
    )
    assert torch.cuda.get_device_capability(0) >= (7, 0)
    assert nvmolkit.__version__ == "0.6.0", (
        f"GPU acceptance requires nvMolKit 0.6.0; found {nvmolkit.__version__}"
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
            "cutoff": 0.4,
            "decision_basis": "Use the qualification clustering cutoff.",
        },
        "embed_representative_conformers": {
            "representative_count": 4,
            "policy": "largest_clusters_first",
            "conformers_per_representative": 5,
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
            arguments = self.arguments[call_index]
            if expected_name == "select_evidence_findings":
                findings = json.loads(kwargs["messages"][-1]["content"])["findings"]
                selected_ids = []
                seen_themes = set()
                for finding in findings:
                    if finding["theme"] not in seen_themes:
                        selected_ids.append(finding["finding_id"])
                        seen_themes.add(finding["theme"])
                arguments = {"ordered_finding_ids": selected_ids}
            tool_call = SimpleNamespace(
                id=f"gpu-acceptance-{call_index}",
                type="function",
                function=SimpleNamespace(
                    name=expected_name,
                    arguments=json.dumps(arguments),
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
    assert certify_argmax_reachability(context) is True
    baseline_measurement = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, baseline_measurement, 0)
    assert menu.state_id.startswith("state-")
    assert accepted_maxima(menu)
    assert all(action.limiting_pairs for action in menu.actions)
    assert is_strict_improvement(context.benchmark_score, context.baseline_score)
    candidate_ids = tuple(candidate.molecule_id for candidate in context.candidates)
    all_panels = tuple(itertools.combinations(candidate_ids, 4))
    assert len(all_panels) == 70
    below_target = []
    for panel in all_panels:
        first = measure_panel(context, panel)
        if first.achieved:
            continue
        below_target.append(first)
        first_menu = build_action_menu(context, first, 0)
        first_suggestions = accepted_maxima(first_menu)
        assert first_suggestions
        for selected_swap in first_suggestions:
            second = evaluate_selected_swap(
                context, first_menu, selected_swap, 1
            ).measurement
            if not second.achieved:
                next_suggestions = accepted_maxima(
                    build_action_menu(context, second, 1)
                )
                assert any(
                    target_is_achieved(suggestion.predicted_score, context.target_score)
                    for suggestion in next_suggestions
                )
    assert len(below_target) == 35

    accepted_panels = []
    accepted_scores = []
    while controller.objective_run is None:
        current_menu = controller.pending_action_menu
        assert current_menu is not None
        selected_swap = accepted_maxima(current_menu)[0]
        completions.expected_names.append("select_next_panel_swap")
        completions.arguments.append(
            {
                "state_id": current_menu.state_id,
                "swap_id": selected_swap.swap_id,
                "observed_limiting_pairs": [
                    list(pair) for pair in current_menu.source.limiting_pairs
                ],
                "decision_rule": "maximize_predicted_minimum_distance",
            }
        )
        proposal = controller.request_objective_attempt()
        attempt = controller.execute_objective_attempt(proposal)
        accepted_panels.append(tuple(sorted(attempt.selected_ids)))
        accepted_scores.append(attempt.score)
        if controller.objective_run is None:
            assert controller.objective_suggestions
            assert is_strict_improvement(
                accepted_maxima(controller.pending_action_menu)[0].predicted_score,
                attempt.score,
            )
    assert controller.objective_run is not None
    assert len(set(accepted_panels)) == len(accepted_panels)
    assert all(
        is_strict_improvement(later_score, earlier_score)
        for earlier_score, later_score in zip(accepted_scores, accepted_scores[1:])
    )
    assert 1 <= len(controller.objective_run.attempts) <= 3
    assert controller.objective_run.termination_reason == "target_achieved"
    assert target_is_achieved(
        controller.objective_run.final_score, context.target_score
    )
    assert controller.objective_evidence.key == "O01"
    objective_payload = json.loads(controller.objective_evidence.payload_json)
    assert "decision_basis" not in controller.objective_evidence.payload_json
    assert objective_payload["baseline"]["limiting_pairs"]
    assert objective_payload["final_measurement"]["limiting_pairs"]
    assert all(item["limiting_pairs"] for item in objective_payload["attempts"])
    assert controller.session.turn_count == 7 + len(accepted_panels)
    assert len(completions.calls) == 7 + len(accepted_panels)
    assistant_messages = [
        message
        for message in controller.session.messages
        if message["role"] == "assistant"
    ]
    tool_messages = [
        message for message in controller.session.messages if message["role"] == "tool"
    ]
    assert len(assistant_messages) == len(tool_messages) == 7 + len(accepted_panels)
    assert [message["tool_call_id"] for message in tool_messages] == [
        message["tool_calls"][0]["id"] for message in assistant_messages
    ]

    completions.expected_names.append("select_evidence_findings")
    completions.arguments.append(None)
    completed = controller.request_synthesis()
    assert type(completed.conclusion) is EvidenceControlledConclusion
    assert completed.conclusion is controller.evidence_controlled_conclusion
    assert completed.conclusion.finding_selection_status == "selected"
    assert len(completed.conclusion.ordered_findings) == 7
    assert completed.conclusion.measured_summary.achieved is True

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

    torch.cuda.synchronize()
    direct_result = fused_butina(
        fingerprints,
        cutoff=stage_arguments["discover_fused_butina_clusters"]["cutoff"],
        return_centroids=True,
    )
    torch.cuda.synchronize()
    direct_labels, direct_clusters, direct_centroids = normalize_fused_butina_result(
        direct_result, molecule_count=256
    )
    direct_label_values = [int(label) for label in direct_labels]
    direct_assigned_indices = [
        molecule_index for cluster in direct_clusters for molecule_index in cluster
    ]
    assert len(direct_label_values) == 256
    assert len(direct_assigned_indices) == 256
    assert sorted(direct_assigned_indices) == list(range(256))
    assert sorted(set(direct_label_values)) == list(range(len(direct_clusters)))
    assert all(
        int(direct_centroids[cluster_id]) in cluster
        for cluster_id, cluster in enumerate(direct_clusters)
    )
    assert all(
        (direct_label_values[left] == direct_label_values[right])
        == (cluster_by_index[left] == cluster_by_index[right])
        for left in range(256)
        for right in range(left + 1, 256)
    )

    report = scientific.report

    assert state.phase is WorkflowPhase.OPTIMIZED
    assert [record.key for record in report.evidence] == [
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
    ]
    assert [record.provenance for record in report.evidence[1:]] == [
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    ]
    evidence = {
        record.key: json.loads(record.payload_json) for record in report.evidence
    }
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
    assert (
        len(energies)
        == len(converged)
        == len(mol_indices)
        == len(conf_indices)
        == attempted
    )
    assert torch.isfinite(energies).all().item()

    result_pairs = list(zip(mol_indices.tolist(), conf_indices.tolist()))
    assert all(
        0 <= mol_index < len(state.conformer_molecules) for mol_index, _ in result_pairs
    )
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
