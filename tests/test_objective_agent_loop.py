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
        proposal(["mol-0", "mol-2", "mol-4", "mol-6"], "Remove the limiting analogue."),
    ])

    context = controller.begin_objective_challenge()
    first_proposal = controller.request_objective_attempt()
    assert controller.objective_attempts == []
    first = controller.execute_objective_attempt(first_proposal)
    second_proposal = controller.request_objective_attempt()
    second = controller.execute_objective_attempt(second_proposal)

    assert context.baseline_score == pytest.approx(0.35)
    assert first.achieved is False
    assert second.achieved is True
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


def test_three_valid_misses_terminate_without_claiming_success():
    misses = [
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], f"Attempt {number} remains bounded.")
        for number in range(1, 4)
    ]
    controller, _ = completed_controller(misses)
    controller.begin_objective_challenge()
    for _ in range(3):
        pending = controller.request_objective_attempt()
        controller.execute_objective_attempt(pending)

    assert controller.objective_run.achieved is False
    assert controller.objective_run.termination_reason == "attempt_limit_reached"
    assert len(controller.objective_run.attempts) == 3


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


def test_out_of_pool_proposal_is_rejected_without_scientific_attempt():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-2", "mol-4", "outside"], "Invalid outside candidate."),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()

    with pytest.raises(demo_agent.ToolCallError, match="rejected"):
        controller.execute_objective_attempt(pending)

    assert controller.objective_attempts == []
    assert controller.objective_run is None


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
