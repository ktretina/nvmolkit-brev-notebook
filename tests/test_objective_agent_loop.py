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
from objective_fixtures import quantized_baseline_target_context


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


def raw_response(name, raw_arguments):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id=f"call-{name}",
                type="function",
                function=SimpleNamespace(name=name, arguments=raw_arguments),
            )],
        ))]
    )


def content_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
            tool_calls=None,
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


def safe_objective_proposals():
    return [
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Measure deterministic baseline Step 0.",
        ),
        proposal(
            ["mol-4", "mol-1", "mol-2", "mol-3"],
            "Apply an accepted maximum replacement.",
        ),
    ]


def execute_safe_objective(controller):
    for _ in range(2):
        pending = controller.request_objective_attempt()
        controller.execute_objective_attempt(pending)


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


def live_invalid_objective_conclusion_arguments():
    arguments = objective_conclusion_arguments()
    arguments["sections"][4]["evidence_keys"] = ["E05"]
    arguments["sections"][5]["evidence_keys"] = ["E06"]
    arguments["sections"][6]["evidence_keys"] = ["E01", "E06"]
    return arguments


def schema_invalid_objective_conclusion_arguments():
    arguments = objective_conclusion_arguments()
    arguments["sections"][5]["theme"] = "bogus_theme"
    arguments["sections"][5]["evidence_keys"] = ["UNKNOWN"]
    return arguments


def objective_theme_evidence_contract():
    return (
        ("dataset_scope", ["E01"]),
        ("molecular_representation", ["E02"]),
        ("similarity_structure", ["E03"]),
        ("clustering", ["E04"]),
        ("conformational_sampling", ["E05", "E06"]),
        ("objective_driven_selection", ["O01"]),
        ("limitations_and_next_steps", ["E01", "E06", "O01"]),
    )


def test_objective_synthesis_tool_schema_pairs_themes_with_exact_evidence_arrays():
    parameters = demo_agent._tool_definition(
        "submit_synthesis", demo_agent.ObjectiveSubmitConclusionArgs
    )["function"]["parameters"]

    assert parameters["properties"]["sections"]["minItems"] == 7
    assert parameters["properties"]["sections"]["maxItems"] == 7
    branches = parameters["properties"]["sections"]["items"]["anyOf"]
    assert len(branches) == 7
    assert [branch["properties"]["theme"]["enum"][0] for branch in branches] == [
        theme for theme, _keys in objective_theme_evidence_contract()
    ]
    for branch, (theme, evidence_keys) in zip(
        branches, objective_theme_evidence_contract(), strict=True
    ):
        assert branch["type"] == "object"
        assert branch["additionalProperties"] is False
        assert branch["required"] == ["theme", "prose", "evidence_keys"]
        assert branch["properties"]["theme"] == {
            "type": "string",
            "enum": [theme],
        }
        assert branch["properties"]["prose"] == {
            "type": "string",
            "minLength": 1,
            "maxLength": 1200,
        }
        assert branch["properties"]["evidence_keys"] == {
            "type": "array",
            "enum": [evidence_keys],
        }


def test_live_objective_synthesis_failure_mapping_is_not_representable_by_tool_schema():
    parameters = demo_agent._tool_definition(
        "submit_synthesis", demo_agent.ObjectiveSubmitConclusionArgs
    )["function"]["parameters"]
    branches = parameters["properties"]["sections"]["items"]["anyOf"]
    invalid_sections = live_invalid_objective_conclusion_arguments()["sections"]

    for section in invalid_sections:
        matching_theme_branches = [
            branch
            for branch in branches
            if section["theme"] in branch["properties"]["theme"]["enum"]
        ]
        assert len(matching_theme_branches) == 1
        branch = matching_theme_branches[0]
        if section["evidence_keys"] != branch["properties"]["evidence_keys"]["enum"][0]:
            break
    else:
        pytest.fail("The live invalid theme/evidence mapping remained representable.")


def test_non_objective_synthesis_tool_schema_is_unchanged():
    expected = demo_agent.SubmitSynthesisArgs.model_json_schema()
    expected["additionalProperties"] = False
    expected["required"] = list(demo_agent.SubmitSynthesisArgs.model_fields)

    assert demo_agent._tool_definition(
        "submit_synthesis", demo_agent.SubmitSynthesisArgs
    )["function"]["parameters"] == expected


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


def test_revision_tool_schema_lists_exactly_current_legal_resulting_panels():
    accepted_panel = ["mol-0", "mol-1", "mol-2", "mol-3"]
    controller, completions = completed_controller([
        proposal(accepted_panel, "Measure the baseline panel."),
        proposal(
            ["mol-4", "mol-1", "mol-2", "mol-3"],
            "Use the measured legal swap.",
        ),
    ])
    controller.begin_objective_challenge()
    initial = controller.request_objective_attempt()
    controller.execute_objective_attempt(initial)
    expected_panels = [
        list(item.resulting_ids) for item in controller.objective_suggestions
    ]

    controller.request_objective_attempt()

    initial_schema = completions.calls[0]["tools"][0]["function"]["parameters"]
    revision_schema = completions.calls[1]["tools"][0]["function"]["parameters"]
    assert "enum" not in initial_schema["properties"]["selected_ids"]
    assert revision_schema["properties"]["selected_ids"]["enum"] == expected_panels
    assert accepted_panel not in revision_schema["properties"]["selected_ids"]["enum"]


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

    with pytest.raises(
        demo_agent.ObjectiveCorrectionLimitError, match="correction limit"
    ):
        controller.request_objective_attempt()

    assert controller.objective_rejection_count == demo_agent.MAX_OBJECTIVE_CORRECTIONS
    assert controller.pending_objective is None
    assert controller.pending_objective_swap is None
    assert controller.objective_attempts == []
    payload = json.loads(controller.session.messages[-1]["content"])
    assert payload["accepted"] is False
    assert payload["reason"] == "out_of_pool_panel"
    assert payload["corrections_remaining"] == 0


def test_preexhausted_objective_correction_budget_uses_dedicated_safe_error():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    controller.objective_rejection_count = demo_agent.MAX_OBJECTIVE_CORRECTIONS
    hosted_calls_before = len(completions.calls)

    with pytest.raises(
        demo_agent.ObjectiveCorrectionLimitError, match="correction limit"
    ):
        controller.request_objective_attempt()

    assert issubclass(
        demo_agent.ObjectiveCorrectionLimitError, demo_agent.ToolCallError
    )
    assert len(completions.calls) == hosted_calls_before
    assert controller.pending_objective is None
    assert controller.objective_attempts == []


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        {
            "selected_ids": ["mol-0", "mol-1", "mol-2"],
            "decision_basis": "RAW-SCHEMA-SECRET",
        },
        {
            "selected_ids": ["mol-0", "mol-0", "mol-2", "mol-3"],
            "decision_basis": "RAW-SCHEMA-SECRET",
        },
        {"decision_basis": "RAW-SCHEMA-SECRET"},
        {"selected_ids": ["mol-0", "mol-1", "mol-2", "mol-3"]},
    ],
)
def test_pydantic_invalid_objective_response_is_paired_and_corrected_safely(
    invalid_arguments,
):
    secret = "RAW-SCHEMA-SECRET"
    controller, _ = completed_controller([
        response("select_diverse_panel", invalid_arguments),
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


def test_placeholder_initial_rationale_is_paired_and_corrected_without_attempt():
    raw_placeholder = "selected_ids"
    valid_basis = "Measure all six pairwise panel distances."
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], raw_placeholder),
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], valid_basis),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.decision_basis == valid_basis
    assert controller.pending_objective is corrected
    assert controller.objective_attempts == []
    assert controller.objective_rejection_count == 1
    assert len(completions.calls) == 2
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_assistant["role"] == "assistant"
    assert rejected_tool["role"] == "tool"
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    payload = json.loads(rejected_tool["content"])
    assert payload["reason"] == "invalid_objective_proposal"
    assert payload["instruction"] == (
        "Select exactly four unique IDs from candidate_ids and provide a concise "
        "measured quantitative rationale."
    )
    assert f'"decision_basis":"{raw_placeholder}"' not in json.dumps(
        controller.session.messages, separators=(",", ":")
    )


def test_strict_objective_schema_publishes_objective_rationale_bounds():
    controller, completions = completed_controller([
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Measure all six pairwise panel distances.",
        ),
    ])
    controller.begin_objective_challenge()

    controller.request_objective_attempt()

    function = completions.calls[0]["tools"][0]["function"]
    rationale_schema = function["parameters"]["properties"]["decision_basis"]
    assert function["strict"] is True
    assert rationale_schema["minLength"] == 1
    assert rationale_schema["maxLength"] == 240
    assert rationale_schema["pattern"] == r"^[^\r\n`]+$"
    assert rationale_schema["description"] == (
        "Provide a concise measured quantitative reason for the selected panel. "
        "For a revision, compare its limiting_pair and predicted_score with "
        "target_score. Never repeat schema field names as the value."
    )


@pytest.mark.parametrize(
    "valid_basis",
    ["0.80 exceeds 0.71.", "Score 0.8 > 0.7."],
)
def test_short_quantitative_rationale_is_accepted_without_correction(valid_basis):
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], valid_basis),
    ])
    controller.begin_objective_challenge()

    pending = controller.request_objective_attempt()

    assert pending.decision_basis == valid_basis
    assert controller.pending_objective is pending
    assert controller.objective_rejection_count == 0
    assert controller.objective_attempts == []
    assert len(completions.calls) == 1


def test_malformed_json_objective_response_is_paired_and_corrected_safely():
    secret = "RAW-MALFORMED-SECRET"
    controller, _ = completed_controller([
        raw_response(
            "select_diverse_panel",
            '{"selected_ids":["mol-0","mol-1","mol-2","mol-3"],'
            f'"decision_basis":"{secret}"',
        ),
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Use the valid JSON panel proposal.",
        ),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-0", "mol-1", "mol-2", "mol-3"]
    assert controller.objective_rejection_count == 1
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    assert json.loads(rejected_tool["content"])["reason"] == "invalid_objective_proposal"
    assert secret not in json.dumps(controller.session.messages)


@pytest.mark.parametrize("raw_arguments", ["[]", "null", '"RAW-NONOBJECT-SECRET"'])
def test_non_object_json_objective_response_is_paired_and_corrected_safely(
    raw_arguments,
):
    controller, _ = completed_controller([
        raw_response("select_diverse_panel", raw_arguments),
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Use the valid JSON object proposal.",
        ),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-0", "mol-1", "mol-2", "mol-3"]
    assert controller.objective_rejection_count == 1
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    assert json.loads(rejected_tool["content"])["reason"] == "invalid_objective_proposal"
    sanitized = json.loads(
        rejected_assistant["tool_calls"][0]["function"]["arguments"]
    )
    assert sanitized == {
        "validation_issues": [
            {"field": "arguments", "error_type": "non_object_json"}
        ]
    }
    assert "RAW-NONOBJECT-SECRET" not in json.dumps(controller.session.messages)


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "null",
        '{"selected_ids":RAW-CONTENT-SECRET',
        "RAW-CONTENT-SECRET plain text",
    ],
)
def test_content_only_invalid_objective_response_is_immediately_paired_and_corrected(
    content,
):
    controller, completions = completed_controller([
        content_response(content),
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Use the corrected objective tool call.",
        ),
    ])
    controller.begin_objective_challenge()

    corrected = controller.request_objective_attempt()

    assert corrected.selected_ids == ["mol-0", "mol-1", "mol-2", "mol-3"]
    assert controller.objective_rejection_count == 1
    assert len(completions.calls) == 2
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    assert json.loads(rejected_tool["content"])["reason"] == "invalid_objective_proposal"
    assert json.loads(
        rejected_assistant["tool_calls"][0]["function"]["arguments"]
    ) == {
        "validation_issues": [
            {"field": "arguments", "error_type": "invalid_objective_content"}
        ]
    }
    assert "RAW-CONTENT-SECRET" not in json.dumps(controller.session.messages)


def test_two_content_only_objective_responses_exhaust_global_correction_budget():
    controller, completions = completed_controller([
        content_response("RAW-FIRST-CONTENT"),
        content_response('{"malformed":"RAW-SECOND-CONTENT"'),
    ])
    controller.begin_objective_challenge()

    with pytest.raises(
        demo_agent.ObjectiveCorrectionLimitError, match="correction limit"
    ):
        controller.request_objective_attempt()

    assert controller.objective_rejection_count == 2
    assert controller.objective_attempts == []
    assert controller.pending_objective is None
    assert len(completions.calls) == 2
    assert controller.session.turn_count == 9
    assert "RAW-FIRST-CONTENT" not in json.dumps(controller.session.messages)
    assert "RAW-SECOND-CONTENT" not in json.dumps(controller.session.messages)


def test_valid_content_only_objective_json_object_remains_compatible():
    arguments = {
        "selected_ids": ["mol-0", "mol-1", "mol-2", "mol-3"],
        "decision_basis": "Use the valid JSON object fallback.",
    }
    controller, completions = completed_controller([
        content_response(json.dumps(arguments)),
    ])
    controller.begin_objective_challenge()

    pending = controller.request_objective_attempt()

    assert pending.model_dump(mode="json") == arguments
    assert controller.objective_rejection_count == 0
    assert len(completions.calls) == 1
    assert controller.session.messages[-1]["tool_calls"][0]["id"].startswith("compat-")


def test_hosted_request_failure_does_not_consume_objective_correction_budget():
    controller, _ = completed_controller([])
    controller.begin_objective_challenge()

    with pytest.raises(demo_agent.ToolCallError, match="hosted Nemotron request failed"):
        controller.request_objective_attempt()

    assert controller.objective_rejection_count == 0
    assert controller.pending_objective is None
    assert controller.session.turn_count == 7


def test_objective_forced_call_normalizes_provider_metadata_and_decision_basis():
    raw = {
        "stage": "select_diverse_panel",
        "summary": "provider compatibility metadata",
        "selected_ids": ["mol-0", "mol-1", "mol-2", "mol-3"],
        "decision_basis": "  Use `ranked`\n candidates.  " + ("detail " * 80),
    }
    with pytest.raises(Exception):
        demo_agent.ObjectiveProposal.model_validate(raw)
    controller, _ = completed_controller([
        response("select_diverse_panel", raw),
    ])
    controller.begin_objective_challenge()

    pending = controller.request_objective_attempt()

    assert pending is controller.pending_objective
    assert controller.objective_rejection_count == 0
    assert 1 <= len(pending.decision_basis) <= 240
    assert "`" not in pending.decision_basis and "\n" not in pending.decision_basis
    assert pending.decision_basis.endswith("...")
    stored = json.loads(controller.session.messages[-1]["tool_calls"][0]["function"]["arguments"])
    assert set(stored) == {"selected_ids", "decision_basis"}
    assert stored["decision_basis"] == pending.decision_basis


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

    with pytest.raises(
        demo_agent.ObjectiveCorrectionLimitError, match="correction limit"
    ):
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
    assert payload["instruction"] == (
        "Select exactly four unique IDs from candidate_ids and provide a concise "
        "measured quantitative rationale."
    )


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
        "Select exactly one listed resulting_ids panel and provide a concise measured "
        "quantitative rationale comparing its limiting_pair and predicted_score with "
        "target_score."
    )


def test_placeholder_revision_rationale_is_corrected_with_exact_swap_provenance():
    raw_placeholder = "decision basis decision_basis"
    valid_basis = "Predicted score 0.80 exceeds the 0.71 target."
    controller, completions = completed_controller([
        proposal(
            ["mol-0", "mol-1", "mol-2", "mol-3"],
            "Measure the initial minimum pairwise distance.",
        ),
        proposal(["mol-5", "mol-1", "mol-2", "mol-3"], raw_placeholder),
        proposal(["mol-5", "mol-1", "mol-2", "mol-3"], valid_basis),
    ])
    controller.begin_objective_challenge()
    initial = controller.request_objective_attempt()
    controller.execute_objective_attempt(initial)
    selected_swap = controller.objective_suggestions[1]

    corrected = controller.request_objective_attempt()

    assert corrected.decision_basis == valid_basis
    assert controller.pending_objective_swap is selected_swap
    assert len(controller.objective_attempts) == 1
    assert controller.objective_rejection_count == 1
    assert len(completions.calls) == 3
    rejected_assistant = controller.session.messages[-3]
    rejected_tool = controller.session.messages[-2]
    assert rejected_tool["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    payload = json.loads(rejected_tool["content"])
    assert payload["reason"] == "invalid_objective_proposal"
    assert payload["instruction"] == (
        "Select exactly one listed resulting_ids panel and provide a concise measured "
        "quantitative rationale comparing its limiting_pair and predicted_score with "
        "target_score."
    )
    assert raw_placeholder not in json.dumps(controller.session.messages)


def test_revision_schema_allows_non_first_legal_panel_and_retains_exact_provenance():
    controller, completions = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-5", "mol-1", "mol-2", "mol-3"], "Choose the second legal swap."),
    ])
    controller.begin_objective_challenge()
    first = controller.request_objective_attempt()
    controller.execute_objective_attempt(first)
    selected_swap = controller.objective_suggestions[1]

    revision = controller.request_objective_attempt()
    accepted = controller.execute_objective_attempt(revision)

    revision_schema = completions.calls[1]["tools"][0]["function"]["parameters"]
    assert (
        list(selected_swap.resulting_ids)
        in revision_schema["properties"]["selected_ids"]["enum"]
    )
    assert accepted.selected_swap is selected_swap
    assert accepted.selected_ids == selected_swap.resulting_ids


def test_value_equal_revision_cannot_replace_exact_pending_proposal_or_swap():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-4", "mol-1", "mol-2", "mol-3"], "Use the ranked replacement."),
    ])
    controller.begin_objective_challenge()
    initial = controller.request_objective_attempt()
    controller.execute_objective_attempt(initial)
    pending = controller.request_objective_attempt()
    pending_swap = controller.pending_objective_swap
    copied = demo_agent.ObjectiveProposal.model_validate(pending.model_dump())
    assert copied == pending and copied is not pending

    with pytest.raises(demo_agent.ToolCallError, match="exact pending"):
        controller.execute_objective_attempt(copied)

    assert len(controller.objective_attempts) == 1
    assert controller.pending_objective is pending
    assert controller.pending_objective_swap is pending_swap


def test_ranking_failure_preserves_pending_objective_transaction(monkeypatch):
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()
    suggestions = controller.objective_suggestions
    original_rank = demo_agent.rank_legal_swaps

    def fail_ranking(*_args):
        raise RuntimeError("RAW-RANKING-SECRET")

    monkeypatch.setattr(demo_agent, "rank_legal_swaps", fail_ranking)
    with pytest.raises(demo_agent.ToolCallError, match="could not be ranked") as captured:
        controller.execute_objective_attempt(pending)

    assert controller.objective_attempts == []
    assert controller.pending_objective is pending
    assert controller.pending_objective_swap is None
    assert controller.objective_suggestions is suggestions
    assert controller.objective_run is None and controller.objective_evidence is None
    assert controller.session.messages[-1]["role"] == "assistant"
    assert "RAW-RANKING-SECRET" not in str(captured.value)

    monkeypatch.setattr(demo_agent, "rank_legal_swaps", original_rank)
    accepted = controller.execute_objective_attempt(pending)
    assert accepted is controller.objective_attempts[0]


def test_feedback_serialization_failure_preserves_revision_transaction(monkeypatch):
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the baseline panel."),
        proposal(["mol-4", "mol-1", "mol-2", "mol-3"], "Use the ranked replacement."),
    ])
    controller.begin_objective_challenge()
    initial = controller.request_objective_attempt()
    controller.execute_objective_attempt(initial)
    pending = controller.request_objective_attempt()
    pending_swap = controller.pending_objective_swap
    attempts = tuple(controller.objective_attempts)
    suggestions = controller.objective_suggestions
    original_serialize = demo_agent._serialize

    def fail_serialization(_value):
        raise demo_agent.ToolCallError("safe serialization failure")

    monkeypatch.setattr(demo_agent, "_serialize", fail_serialization)
    with pytest.raises(demo_agent.ToolCallError, match="serialization failure"):
        controller.execute_objective_attempt(pending)

    assert tuple(controller.objective_attempts) == attempts
    assert controller.pending_objective is pending
    assert controller.pending_objective_swap is pending_swap
    assert controller.objective_suggestions is suggestions
    assert controller.objective_run is None and controller.objective_evidence is None
    assert controller.session.messages[-1]["role"] == "assistant"

    monkeypatch.setattr(demo_agent, "_serialize", original_serialize)
    accepted = controller.execute_objective_attempt(pending)
    assert accepted.selected_swap is pending_swap


def test_optimal_baseline_terminates_without_manufacturing_an_attempt():
    controller, completions = completed_controller([], baseline_optimal=True)

    context = controller.begin_objective_challenge()

    assert context.baseline_score == context.benchmark_score
    assert controller.objective_run.termination_reason == "baseline_already_optimal"
    assert controller.objective_run.attempts == ()
    assert completions.calls == []


def test_quantized_baseline_target_short_circuits_without_hosted_attempt(monkeypatch):
    controller, completions = completed_controller([])
    context = quantized_baseline_target_context()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda state: context)

    controller.begin_objective_challenge()

    assert controller.objective_run.termination_reason == "target_achieved"
    assert controller.objective_run.attempts == ()
    assert controller.objective_evidence.key == "O01"
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
        *safe_objective_proposals(),
        response("submit_synthesis", objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    result = controller.request_synthesis()

    assert result.objective_run.achieved is True
    assert result.objective_evidence.key == "O01"
    assert len(result.conclusion.sections) == 7
    assert result.conclusion.sections[5].theme == "objective_driven_selection"
    assert result.turn_count == 10
    synthesis_call = completions.calls[-1]
    assert synthesis_call["tools"][0]["function"]["name"] == "submit_synthesis"
    supplied = controller.session.messages[-2]["content"]
    assert "O01" in supplied


def test_invalid_objective_conclusion_appends_sanitized_paired_feedback():
    controller, completions = completed_controller([
        *safe_objective_proposals(),
        response("submit_synthesis", live_invalid_objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    objective_run = controller.objective_run
    objective_evidence = controller.objective_evidence
    with pytest.raises(demo_agent.ConclusionValidationError) as error:
        controller.request_synthesis()

    assistant, feedback = controller.session.messages[-2:]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "submit_synthesis"
    assert feedback["role"] == "tool"
    assert feedback["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert json.loads(feedback["content"]) == {
        "accepted": False,
        "instruction": (
            "Resubmit all seven themes with their required evidence_keys; "
            "author the corrected evidence links without changing the evidence IDs."
        ),
        "validation_issues": {
            "duplicate_themes": [],
            "extra_evidence_keys": [],
            "extra_themes": [],
            "missing_evidence_keys": ["O01"],
            "missing_required_evidence": {
                "conformational_sampling": ["E06"],
                "limitations_and_next_steps": ["O01"],
                "objective_driven_selection": ["O01"],
            },
            "missing_themes": [],
        },
    }
    assert "prose" not in feedback["content"]
    assert error.value.report is controller.report
    assert controller.objective_run is objective_run
    assert controller.objective_evidence is objective_evidence
    assert controller.session.turn_count == 10
    assert len(completions.calls) == 3


def test_objective_conclusion_feedback_reports_missing_and_duplicate_themes():
    invalid = objective_conclusion_arguments()
    invalid["sections"][5] = {
        "theme": "dataset_scope",
        "prose": "A duplicate dataset section.",
        "evidence_keys": ["E01"],
    }
    controller, _ = completed_controller([
        *safe_objective_proposals(),
        response("submit_synthesis", invalid),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    with pytest.raises(demo_agent.ConclusionValidationError):
        controller.request_synthesis()

    issues = json.loads(controller.session.messages[-1]["content"])["validation_issues"]
    assert issues["missing_themes"] == ["objective_driven_selection"]
    assert issues["duplicate_themes"] == ["dataset_scope"]
    assert issues["extra_themes"] == []
    assert issues["missing_evidence_keys"] == []
    assert issues["extra_evidence_keys"] == []


def test_objective_conclusion_retry_uses_feedback_and_succeeds_with_exact_coverage():
    controller, completions = completed_controller([
        *safe_objective_proposals(),
        response("submit_synthesis", live_invalid_objective_conclusion_arguments()),
        response("submit_synthesis", objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    with pytest.raises(demo_agent.ConclusionValidationError):
        controller.request_synthesis()
    result = controller.request_synthesis()

    assert result.turn_count == 11 <= demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
    assert {key for section in result.conclusion.sections for key in section.evidence_keys} == {
        "E01", "E02", "E03", "E04", "E05", "E06", "O01",
    }
    rejected_assistant, rejected_feedback = result.messages[-3:-1]
    assert rejected_assistant["role"] == "assistant"
    assert rejected_feedback["role"] == "tool"
    assert rejected_feedback["tool_call_id"] == rejected_assistant["tool_calls"][0]["id"]
    assert completions.calls[-1]["messages"][-2] == rejected_feedback
    first_schema = completions.calls[-2]["tools"][0]["function"]["parameters"]
    retry_schema = completions.calls[-1]["tools"][0]["function"]["parameters"]
    assert retry_schema == first_schema
    assert len(retry_schema["properties"]["sections"]["items"]["anyOf"]) == 7


def test_valid_first_objective_conclusion_does_not_append_feedback():
    controller, _ = completed_controller([
        *safe_objective_proposals(),
        response("submit_synthesis", objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    result = controller.request_synthesis()

    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["tool_calls"][0]["function"]["name"] == "submit_synthesis"


def test_schema_invalid_objective_conclusion_gets_paired_feedback_then_retries():
    controller, completions = completed_controller([
        *safe_objective_proposals(),
        response("submit_synthesis", schema_invalid_objective_conclusion_arguments()),
        response("submit_synthesis", objective_conclusion_arguments()),
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    with pytest.raises(demo_agent.ConclusionValidationError):
        controller.request_synthesis()

    rejected, feedback = controller.session.messages[-2:]
    assert rejected["role"] == "assistant"
    assert rejected["tool_calls"][0]["id"] == "call-submit_synthesis"
    assert json.loads(rejected["tool_calls"][0]["function"]["arguments"]) == {
        "validation_issues": [
            {"error_type": "literal_error", "field": "sections.item.theme"},
            {
                "error_type": "literal_error",
                "field": "sections.item.evidence_keys.item",
            },
        ]
    }
    assert feedback["role"] == "tool"
    assert feedback["tool_call_id"] == rejected["tool_calls"][0]["id"]
    assert json.loads(feedback["content"]) == {
        "accepted": False,
        "instruction": (
            "Resubmit a valid seven-theme objective conclusion using only the "
            "allowed evidence_keys."
        ),
        "validation_issues": [
            {"error_type": "literal_error", "field": "sections.item.theme"},
            {
                "error_type": "literal_error",
                "field": "sections.item.evidence_keys.item",
            },
        ],
    }
    assert "bogus_theme" not in json.dumps((rejected, feedback))
    assert "UNKNOWN" not in json.dumps((rejected, feedback))
    assert controller.session.turn_count == 10

    result = controller.request_synthesis()

    assert result.turn_count == 11 <= demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
    assert result.messages[-3:-1] == (rejected, feedback)
    assert completions.calls[-1]["messages"][-2] == feedback


def test_schema_invalid_objective_conclusions_consume_the_bounded_turns():
    invalid_responses = [
        response("submit_synthesis", schema_invalid_objective_conclusion_arguments())
        for _ in range(5)
    ]
    controller, completions = completed_controller([
        *safe_objective_proposals(),
        *invalid_responses,
    ])
    controller.begin_objective_challenge()
    execute_safe_objective(controller)

    for expected_turn in range(10, demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS + 1):
        with pytest.raises(demo_agent.ConclusionValidationError):
            controller.request_synthesis()
        assert controller.session.turn_count == expected_turn
        assert controller.session.messages[-2]["role"] == "assistant"
        assert controller.session.messages[-1]["role"] == "tool"

    with pytest.raises(demo_agent.ToolCallError):
        controller.request_synthesis()

    assert controller.session.turn_count == demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
    assert len(completions.calls) == 6
