import json
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

import demo_agent
from chemistry_workflow import (
    EvidenceRecord,
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    WorkflowState,
)
from objective_challenge import rank_legal_swaps


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class FakeGpuResult:
    def __init__(self, values):
        self.tensor = FakeTensor(values)

    def torch(self):
        return self.tensor


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def response(name, arguments):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id=f"call-{name}",
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )],
        ))]
    )


def optimized_state(*, baseline_optimal=False):
    smiles = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCF", "C1CC1")
    distance = np.full((8, 8), 0.80, dtype=float)
    np.fill_diagonal(distance, 0.0)
    if not baseline_optimal:
        distance[0, 1] = distance[1, 0] = 0.35
    return WorkflowState(
        phase=WorkflowPhase.OPTIMIZED,
        records=[
            {"id": f"mol-{index}", "smiles": value, "source_row": index}
            for index, value in enumerate(smiles)
        ],
        molecules=[Chem.MolFromSmiles(value) for value in smiles],
        similarity=FakeGpuResult(1.0 - distance),
        clusters=[[index] for index in range(8)],
    )


def full_report():
    return WorkflowReport(tuple(
        EvidenceRecord(f"E0{number}", f"Evidence {number}", "{}", "test")
        for number in range(1, 7)
    ))


def completed_controller(objective_responses, *, baseline_optimal=False):
    completions = FakeCompletions(objective_responses)
    session = demo_agent.AgentSession(
        messages=[
            {"role": "system", "content": "bounded chemistry agent"},
            {"role": "user", "content": "analyze"},
        ],
        state=optimized_state(baseline_optimal=baseline_optimal),
        turn_count=7,
    )
    plan = demo_agent.WorkflowPlan(stages=[
        {"stage": stage, "rationale": f"Run {stage}."}
        for stage in demo_agent.STAGES
    ])
    controller = demo_agent.BoundedWorkflowController(
        session=session,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        executors={},
        plan=plan,
        stage_results=[StageResult(stage, stage, {}) for stage in demo_agent.STAGES],
        report=full_report(),
        objective_required=True,
    )
    return controller, completions


def proposal(selected_ids, basis):
    return response("select_diverse_panel", {
        "selected_ids": selected_ids,
        "decision_basis": basis,
    })


def objective_conclusion_arguments():
    themes_and_keys = (
        ("dataset_scope", ["E01"]),
        ("molecular_representation", ["E02"]),
        ("similarity_structure", ["E03"]),
        ("clustering", ["E04"]),
        ("conformational_sampling", ["E05", "E06"]),
        ("objective_driven_selection", ["O01"]),
        ("limitations_and_next_steps", ["E01", "E06", "O01"]),
    )
    return {
        "headline": "A bounded structural-diversity objective was measured",
        "sections": [
            {
                "theme": theme,
                "prose": f"Evidence-grounded interpretation for {theme}.",
                "evidence_keys": keys,
            }
            for theme, keys in themes_and_keys
        ],
    }


def test_objective_proposal_requires_four_unique_bounded_ids():
    valid = demo_agent.ObjectiveProposal(
        selected_ids=["mol-0", "mol-2", "mol-4", "mol-6"],
        decision_basis="Remove the limiting analogue.",
    )
    assert valid.selected_ids == ["mol-0", "mol-2", "mol-4", "mol-6"]

    with pytest.raises(Exception):
        demo_agent.ObjectiveProposal(
            selected_ids=["mol-0", "mol-0", "mol-4", "mol-6"],
            decision_basis="Duplicates are invalid.",
        )


def test_controller_returns_each_objective_proposal_before_execution_and_stops_on_success():
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-4", "mol-1", "mol-2", "mol-3"], "Remove the limiting analogue."),
    ])

    context = controller.begin_objective_challenge()
    first_proposal = controller.request_objective_attempt()
    assert controller.objective_attempts == []
    first = controller.execute_objective_attempt(first_proposal)
    second_proposal = controller.request_objective_attempt()
    selected_swap = controller.pending_objective_swap
    second = controller.execute_objective_attempt(second_proposal)

    assert context.baseline_score == pytest.approx(0.35)
    assert first.achieved is False
    assert first.selected_swap is None
    assert second.achieved is True
    assert second.selected_swap is selected_swap
    assert controller.objective_run.termination_reason == "target_achieved"
    assert controller.objective_evidence.key == "O01"
    assert controller.session.turn_count == 9
    assert len(completions.calls) == 2
    assert controller.session.messages[-1]["role"] == "tool"
    with pytest.raises(demo_agent.ToolCallError, match="already complete"):
        controller.request_objective_attempt()


def test_attempt_feedback_returns_score_target_and_limiting_pair_to_same_conversation():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()
    controller.execute_objective_attempt(pending)
    payload = json.loads(controller.session.messages[-1]["content"])

    assert payload["accepted"] is True
    assert payload["score"] == pytest.approx(0.35)
    assert payload["target_score"] == pytest.approx(0.71)
    assert payload["limiting_pair"] == ["mol-0", "mol-1"]
    assert payload["achieved"] is False
    assert payload["instruction"] == (
        "Select exactly one listed resulting_ids panel and explain how its limiting_pair "
        "and predicted_score compare with target_score."
    )
    expected = [
        {
            "replace_id": item.replace_id,
            "replacement_id": item.replacement_id,
            "resulting_ids": list(item.resulting_ids),
            "predicted_score": item.predicted_score,
            "score_delta": item.score_delta,
            "limiting_pair": list(item.limiting_pair),
        }
        for item in rank_legal_swaps(controller.objective_context, controller.objective_attempts[0])
    ]
    assert payload["legal_improving_swaps"] == expected
    assert set(payload["legal_improving_swaps"][0]) == {
        "replace_id",
        "replacement_id",
        "resulting_ids",
        "predicted_score",
        "score_delta",
        "limiting_pair",
    }


def test_duplicate_revision_is_rejected_then_corrected_without_scientific_attempt():
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-3", "mol-2", "mol-1", "mol-0"], "Repeat the same panel."),
        proposal(["mol-4", "mol-1", "mol-2", "mol-3"], "Use the ranked replacement."),
    ])
    controller.begin_objective_challenge()
    first = controller.request_objective_attempt()
    controller.execute_objective_attempt(first)

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-4", "mol-1", "mol-2", "mol-3"]
    assert corrected is controller.pending_objective
    assert controller.pending_objective_swap is controller.objective_suggestions[0]
    assert controller.objective_rejection_count == 1
    assert len(controller.objective_attempts) == 1
    assert len(completions.calls) == 3
    rejected = json.loads(controller.session.messages[-2]["content"])
    assert rejected["accepted"] is False
    assert rejected["reason"] == "duplicate_panel"
    assert rejected["corrections_remaining"] == 1
    assert "candidate_ids" not in rejected


def test_two_invalid_responses_exhaust_global_correction_budget_without_attempt():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-2", "mol-4", "outside-a"], "First invalid panel."),
        proposal(["mol-1", "mol-3", "mol-5", "outside-b"], "Second invalid panel."),
    ])
    controller.begin_objective_challenge()

    with pytest.raises(demo_agent.ToolCallError, match="correction limit"):
        controller.request_objective_attempt()

    assert controller.objective_rejection_count == demo_agent.MAX_OBJECTIVE_CORRECTIONS
    assert controller.pending_objective is None
    assert controller.pending_objective_swap is None
    assert controller.objective_attempts == []
    payload = json.loads(controller.session.messages[-1]["content"])
    assert payload["accepted"] is False
    assert payload["reason"] == "out_of_pool_panel"
    assert payload["corrections_remaining"] == 0


def test_schema_invalid_objective_response_is_paired_and_corrected_safely():
    secret = "RAW-SCHEMA-SECRET"
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2"], secret),
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Use four candidates."),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-0", "mol-1", "mol-2", "mol-3"]
    assert controller.objective_rejection_count == 1
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_assistant["role"] == "assistant"
    assert rejected_tool["role"] == "tool"
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    payload = json.loads(rejected_tool["content"])
    assert payload["reason"] == "invalid_objective_proposal"
    assert secret not in json.dumps(rejected_assistant)
    assert secret not in rejected_tool["content"]


def test_correction_budget_remains_global_across_accepted_attempts():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-2", "mol-4", "outside"], "Correct the initial pool."),
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure a valid initial panel."),
        proposal(["mol-0", "mol-2", "mol-4", "mol-6"], "Try an unlisted revision."),
    ])
    controller.begin_objective_challenge()
    initial = controller.request_objective_attempt()
    controller.execute_objective_attempt(initial)
    accepted_before = tuple(controller.objective_attempts)

    with pytest.raises(demo_agent.ToolCallError, match="correction limit"):
        controller.request_objective_attempt()

    assert controller.objective_rejection_count == 2
    assert tuple(controller.objective_attempts) == accepted_before
    assert controller.pending_objective is None
    assert controller.pending_objective_swap is None


def test_out_of_pool_initial_proposal_is_corrected_from_candidate_ids():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-2", "mol-4", "outside"], "Invalid outside candidate."),
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Use only candidates."),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-0", "mol-1", "mol-2", "mol-3"]
    assert controller.objective_attempts == []
    assert controller.objective_rejection_count == 1
    payload = json.loads(controller.session.messages[-2]["content"])
    assert payload["reason"] == "out_of_pool_panel"
    assert payload["candidate_ids"] == [f"mol-{index}" for index in range(8)]
    assert payload["instruction"] == "Select exactly four unique IDs from candidate_ids."


def test_unlisted_in_pool_revision_is_rejected_then_listed_swap_is_retained():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-0", "mol-2", "mol-4", "mol-6"], "Use an unlisted in-pool panel."),
        proposal(["mol-5", "mol-1", "mol-2", "mol-3"], "Use one listed swap."),
    ])
    controller.begin_objective_challenge()
    first = controller.request_objective_attempt()
    controller.execute_objective_attempt(first)

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-5", "mol-1", "mol-2", "mol-3"]
    assert controller.pending_objective_swap is controller.objective_suggestions[1]
    assert len(controller.objective_attempts) == 1
    payload = json.loads(controller.session.messages[-2]["content"])
    assert payload["reason"] == "panel_not_in_legal_improving_swaps"
    assert payload["instruction"] == (
        "Select exactly one listed resulting_ids panel and explain its limiting-pair rationale."
    )


def test_optimal_baseline_terminates_without_manufacturing_an_attempt():
    controller, completions = completed_controller([], baseline_optimal=True)

    context = controller.begin_objective_challenge()

    assert context.baseline_score == context.benchmark_score
    assert controller.objective_run.termination_reason == "baseline_already_optimal"
    assert controller.objective_run.attempts == ()
    assert completions.calls == []


def test_objective_prompt_contains_bounded_evidence_but_not_benchmark_panel():
    controller, _ = completed_controller([])
    controller.begin_objective_challenge()
    prompt = controller.session.messages[-1]["content"]

    assert "mol-0" in prompt and "cluster_id" in prompt
    assert "distance_matrix" in prompt
    assert "baseline_score" in prompt and "target_score" in prompt
    assert "benchmark_panel" not in prompt


def test_miss_without_legal_suggestions_fails_before_another_hosted_request():
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()
    controller.execute_objective_attempt(pending)
    controller.objective_suggestions = ()

    with pytest.raises(demo_agent.ToolCallError, match="legal improving"):
        controller.request_objective_attempt()

    assert len(controller.objective_attempts) == 1
    assert controller.objective_run is None
    assert len(completions.calls) == 1


def test_objective_required_controller_blocks_conclusion_until_termination():
    controller, _ = completed_controller([])

    with pytest.raises(demo_agent.ToolCallError, match="objective challenge"):
        controller.request_synthesis()

    controller.begin_objective_challenge()
    with pytest.raises(demo_agent.ToolCallError, match="objective challenge"):
        controller.request_synthesis()


def test_objective_conclusion_includes_o01_and_uses_the_extended_turn_budget():
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-2", "mol-4", "mol-6"], "Remove the limiting analogue."),
        response("submit_synthesis", objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()
    controller.execute_objective_attempt(pending)

    result = controller.request_synthesis()

    assert result.objective_run.achieved is True
    assert result.objective_evidence.key == "O01"
    assert len(result.conclusion.sections) == 7
    assert result.conclusion.sections[5].theme == "objective_driven_selection"
    assert result.turn_count == 9
    synthesis_call = completions.calls[-1]
    assert synthesis_call["tools"][0]["function"]["name"] == "submit_synthesis"
    supplied = controller.session.messages[-2]["content"]
    assert "O01" in supplied
