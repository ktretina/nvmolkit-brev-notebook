import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from openai import AuthenticationError, PermissionDeniedError
from rdkit import Chem

import demo_agent
from chemistry_workflow import (
    EvidenceRecord,
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    WorkflowState,
)
from objective_challenge import accepted_maxima, build_action_menu, measure_panel, rank_legal_swaps
from objective_fixtures import (
    controlled_context_with_ranked_swaps,
    quantized_baseline_target_context,
    two_revision_context,
)


class FailOnceList(list):
    def __init__(self, values):
        super().__init__(values)
        self.append_attempts = 0

    def append(self, value):
        self.append_attempts += 1
        if self.append_attempts == 1:
            raise RuntimeError("injected append failure")
        return super().append(value)


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
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if item is AUTO_SELECTION:
            properties = kwargs["tools"][0]["function"]["parameters"]["properties"]
            return response("select_next_panel_swap", {
                "state_id": properties["state_id"]["enum"][0],
                "swap_id": properties["swap_id"]["enum"][0],
                "observed_limiting_pairs": properties["observed_limiting_pairs"]["enum"][0],
                "decision_rule": "maximize_predicted_minimum_distance",
            })
        if item == "AUTO_STALE_SELECTION":
            properties = kwargs["tools"][0]["function"]["parameters"]["properties"]
            return response("select_next_panel_swap", {
                "state_id": "state-0000000000000000",
                "swap_id": properties["swap_id"]["enum"][0],
                "observed_limiting_pairs": properties["observed_limiting_pairs"]["enum"][0],
                "decision_rule": "maximize_predicted_minimum_distance",
            })
        return item


AUTO_SELECTION = object()


def connect_error(label="objective transport failure"):
    return httpx.ConnectError(
        label,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )


def request_error(status, error_type):
    response = httpx.Response(
        status,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )
    return error_type("objective authorization failure", response=response, body=None)


def response(name, arguments, *, content=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
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


def envelope_response(calls, *, content=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
            tool_calls=[SimpleNamespace(
                id=call.get("id"),
                type=call.get("type"),
                function=SimpleNamespace(
                    name=call.get("name"),
                    arguments=call.get("arguments"),
                ),
            ) for call in calls],
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


def initial_menu(controller):
    context = controller.objective_context
    assert context is not None
    return build_action_menu(context, measure_panel(context, context.baseline_ids), 0)


def selection(menu, index=0, **overrides):
    action = menu.actions[index]
    arguments = {
        "state_id": menu.state_id,
        "swap_id": action.swap_id,
        "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
        "decision_rule": "maximize_predicted_minimum_distance",
    }
    arguments.update(overrides)
    return response("select_next_panel_swap", arguments)


def assert_paired(messages):
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        assert index + 1 < len(messages)
        following = messages[index + 1]
        assert following["role"] == "tool"
        assert following["tool_call_id"] == message["tool_calls"][0]["id"]


def test_dynamic_objective_selection_schema_binds_exact_pending_state():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    assert menu is not None
    completions.responses.append(selection(menu))

    pending = controller.request_objective_selection()

    assert pending is controller.pending_objective_selection
    call = completions.calls[0]
    assert call["tool_choice"]["function"]["name"] == "select_next_panel_swap"
    parameters = call["tools"][0]["function"]["parameters"]
    assert parameters["required"] == [
        "state_id", "swap_id", "observed_limiting_pairs", "decision_rule"
    ]
    assert parameters["properties"]["state_id"]["enum"] == [menu.state_id]
    assert parameters["properties"]["swap_id"]["enum"] == [
        action.swap_id for action in menu.actions
    ]
    assert parameters["properties"]["observed_limiting_pairs"]["enum"] == [[
        list(pair) for pair in menu.source.limiting_pairs
    ]]
    Draft202012Validator.check_schema(parameters)
    Draft202012Validator(parameters).validate(pending.model_dump(mode="json"))


def test_invalid_invalid_terminalizes_with_one_correction_and_no_chemistry():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    assert menu is not None
    completions.responses.extend([
        selection(menu, state_id="state-0000000000000000"),
        selection(menu, state_id="state-1111111111111111"),
    ])

    with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
        controller.request_objective_selection()

    assert len(completions.calls) == 2
    assert controller.rejected_selection_count == 2
    assert controller.selection_response_count == 2
    assert controller.correction_prompts_sent == 1
    assert controller.accepted_attempt_count == 0
    assert controller.objective_run.termination_reason == "objective_correction_limit"
    correction = json.loads([
        item["content"] for item in controller.session.messages
        if item.get("role") == "user"
    ][-1])
    assert list(correction) == sorted(correction) == [
        "candidate_actions", "current_limiting_pairs", "decision_rule", "remaining_rejections"
    ]
    assert correction["remaining_rejections"] == 1
    assert_paired(controller.session.messages)


def test_transport_retry_counts_only_requests_then_valid_response():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.extend([connect_error(), selection(menu)])

    with pytest.raises(demo_agent.ToolCallError):
        controller.request_objective_selection()
    assert controller.objective_transport_retry_pending is True
    pending = controller.request_objective_selection(is_transport_retry=True)

    assert pending is controller.pending_objective_selection
    assert controller.objective_transport_retry_pending is False
    assert controller.provider_request_attempt_count == 2
    assert controller.selection_response_count == 1
    assert controller.rejected_selection_count == 0


@pytest.mark.parametrize(
    "invalid_response",
    [
        lambda menu: selection(menu, observed_limiting_pairs=[["mol-2", "mol-3"]]),
        lambda menu: response("select_diverse_panel", {"state_id": menu.state_id}),
        lambda menu: raw_response("select_next_panel_swap", "{malformed"),
    ],
)
def test_invalid_selection_kinds_reject_before_chemistry_then_accept(invalid_response):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.extend([invalid_response(menu), selection(menu)])

    pending = controller.request_objective_selection()

    assert pending is controller.pending_objective_selection
    assert controller.rejected_selection_count == 1
    assert controller.accepted_attempt_count == 0
    attempt = controller.execute_objective_selection(pending)
    assert attempt is controller.objective_attempts[0]
    assert controller.accepted_attempt_count == 1
    assert_paired(controller.session.messages)


def test_displayed_nonmax_selection_is_rejected_when_menu_has_lower_option(monkeypatch):
    controller, completions = completed_controller([])
    context = controlled_context_with_ranked_swaps()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    maxima = accepted_maxima(menu)
    lower = next(index for index, action in enumerate(menu.actions) if action not in maxima)
    completions.responses.extend([selection(menu, lower), selection(menu, menu.actions.index(maxima[0]))])

    controller.request_objective_selection()

    assert controller.rejected_selection_count == 1
    assert controller.accepted_attempt_count == 0


def test_two_transport_failures_terminalize_without_assistant_response():
    controller, completions = completed_controller([connect_error("first"), connect_error("second")])
    controller.begin_objective_challenge()
    with pytest.raises(demo_agent.ToolCallError):
        controller.request_objective_selection()
    with pytest.raises(demo_agent.ToolCallError):
        controller.request_objective_selection(is_transport_retry=True)

    assert controller.provider_request_attempt_count == 2
    assert controller.selection_response_count == 0
    assert controller.rejected_selection_count == 0
    assert controller.objective_run.termination_reason == "objective_provider_failure"
    assert controller.objective_evidence.key == "O01"


@pytest.mark.parametrize(
    "hosted_error",
    [
        request_error(401, AuthenticationError),
        request_error(403, PermissionDeniedError),
    ],
)
def test_objective_auth_errors_bypass_transport_retry_and_terminal_evidence(hosted_error):
    controller, completions = completed_controller([hosted_error])
    controller.begin_objective_challenge()

    with pytest.raises(ValueError, match="API key"):
        controller.request_objective_selection()

    assert len(completions.calls) == 1
    assert controller.provider_request_attempt_count == 1
    assert controller.selection_response_count == 0
    assert controller.rejected_selection_count == 0
    assert controller.correction_prompts_sent == 0
    assert controller.accepted_attempt_count == 0
    assert controller.objective_transport_retry_pending is False
    assert controller.objective_transport_retry_used is False
    assert controller.objective_run is None
    assert controller.objective_evidence is None


@pytest.mark.parametrize(
    "nontransport_failure",
    [
        RuntimeError("provider implementation failure"),
        SimpleNamespace(choices=[]),
    ],
)
def test_nontransport_objective_failures_are_not_retryable_or_terminalized(
    nontransport_failure,
):
    controller, completions = completed_controller([nontransport_failure])
    controller.begin_objective_challenge()

    with pytest.raises(demo_agent.ToolCallError):
        controller.request_objective_selection()

    assert len(completions.calls) == 1
    assert controller.provider_request_attempt_count == 1
    assert controller.selection_response_count == 0
    assert controller.rejected_selection_count == 0
    assert controller.correction_prompts_sent == 0
    assert controller.accepted_attempt_count == 0
    assert controller.objective_transport_retry_pending is False
    assert controller.objective_transport_retry_used is False
    assert controller.objective_run is None
    assert controller.objective_evidence is None


def test_objective_transport_retry_pending_is_read_only():
    controller, _ = completed_controller([])

    assert controller.objective_transport_retry_pending is False
    with pytest.raises(AttributeError):
        controller.objective_transport_retry_pending = True


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("accepted_attempt_count", 3),
        ("rejected_selection_count", 2),
        ("correction_prompts_sent", 1),
        ("selection_response_count", 5),
        ("provider_request_attempt_count", 6),
    ],
)
def test_each_objective_counter_bound_blocks_locally(field, maximum):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    if field == "accepted_attempt_count":
        controller.objective_attempts = [object(), object(), object()]
    setattr(controller, field, maximum)

    with pytest.raises(demo_agent.ToolCallError):
        controller.request_objective_selection()

    assert completions.calls == []


@pytest.mark.parametrize("sequence", [
    ("AUTO_STALE_SELECTION", AUTO_SELECTION, "AUTO_STALE_SELECTION"),
    (AUTO_SELECTION, "AUTO_STALE_SELECTION", "AUTO_STALE_SELECTION"),
])
def test_rejection_budget_never_resets_across_measured_attempts(monkeypatch, sequence):
    controller, completions = completed_controller(list(sequence))
    context = two_revision_context()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    controller.begin_objective_challenge()

    if sequence[0] == "AUTO_STALE_SELECTION":
        pending = controller.request_objective_selection()
    else:
        pending = controller.request_objective_selection()
        controller.execute_objective_selection(pending)
        with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
            controller.request_objective_selection()
        pending = None
    if pending is not None:
        controller.execute_objective_selection(pending)
        with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
            controller.request_objective_selection()

    assert controller.rejected_selection_count == 2
    assert controller.accepted_attempt_count == 1
    assert len(controller.objective_attempts) == 1
    assert controller.objective_run.termination_reason == "objective_correction_limit"
    assert controller.objective_evidence.key == "O01"
    assert len(completions.calls) == 3
    assert_paired(controller.session.messages)


def test_actual_old_state_and_swap_are_rejected_twice_after_state_transition(monkeypatch):
    controller, completions = completed_controller([])
    context = two_revision_context()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    controller.begin_objective_challenge()
    old_menu = controller.pending_action_menu
    completions.responses.append(selection(old_menu))

    pending = controller.request_objective_selection()
    controller.execute_objective_selection(pending)
    current_menu = controller.pending_action_menu
    assert current_menu.state_id != old_menu.state_id
    assert old_menu.actions[0].swap_id not in {action.swap_id for action in current_menu.actions}
    completions.responses.extend([selection(old_menu), selection(old_menu)])

    with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
        controller.request_objective_selection()

    assert controller.accepted_attempt_count == 1
    assert controller.rejected_selection_count == 2
    assert len(controller.objective_attempts) == 1
    rejected_tools = [
        json.loads(message["content"])
        for message in controller.session.messages
        if message.get("role") == "tool" and '"accepted":false' in message["content"]
    ]
    assert [item["reason"] for item in rejected_tools[-2:]] == [
        "stale_objective_state",
        "stale_objective_state",
    ]
    assert_paired(controller.session.messages)


def test_provider_content_is_transcript_only_and_never_enters_attempt_or_o01():
    model_prose = "MODEL PRIVATE EXPLANATION MUST NOT BECOME FACT"
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    action = accepted_maxima(menu)[0]
    arguments = {
        "state_id": menu.state_id,
        "swap_id": action.swap_id,
        "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
        "decision_rule": "maximize_predicted_minimum_distance",
    }
    completions.responses.append(
        response("select_next_panel_swap", arguments, content=model_prose)
    )

    pending = controller.request_objective_selection()
    attempt = controller.execute_objective_selection(pending)

    assert controller.session.messages[-2]["content"] == model_prose
    assert model_prose not in repr(attempt)
    assert model_prose not in attempt.decision_basis
    assert controller.objective_evidence is not None
    assert model_prose not in controller.objective_evidence.payload_json
    assert "decision_basis" not in controller.objective_evidence.payload_json


@pytest.mark.parametrize(
    ("sequence_name", "expected_counters", "expected_terminal"),
    [
        ("valid", (1, 0, 0, 1, 1), None),
        ("nonmax-valid", (1, 1, 1, 2, 2), "target_achieved"),
        ("wrong_pair-valid", (1, 1, 1, 2, 2), None),
        ("wrong_tool-valid", (1, 1, 1, 2, 2), None),
        ("malformed-valid", (1, 1, 1, 2, 2), None),
        ("valid-invalid-invalid", (1, 2, 1, 3, 3), "objective_correction_limit"),
        ("invalid-valid-invalid", (1, 2, 1, 3, 3), "objective_correction_limit"),
        ("invalid-invalid", (0, 2, 1, 2, 2), "objective_correction_limit"),
        ("transport-valid", (1, 0, 0, 1, 2), None),
        ("transport-transport", (0, 0, 0, 0, 2), "objective_provider_failure"),
    ],
)
def test_objective_selection_transition_table(
    monkeypatch, sequence_name, expected_counters, expected_terminal
):
    context = (
        controlled_context_with_ranked_swaps()
        if sequence_name == "nonmax-valid"
        else two_revision_context()
    )
    controller, completions = completed_controller([])
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu

    if sequence_name == "transport-transport":
        completions.responses.extend([connect_error("first"), connect_error("second")])
        with pytest.raises(demo_agent.ToolCallError):
            controller.request_objective_selection()
        with pytest.raises(demo_agent.ToolCallError):
            controller.request_objective_selection(is_transport_retry=True)
    elif sequence_name == "transport-valid":
        completions.responses.extend([connect_error(), selection(menu)])
        with pytest.raises(demo_agent.ToolCallError):
            controller.request_objective_selection()
        pending = controller.request_objective_selection(is_transport_retry=True)
        controller.execute_objective_selection(pending)
    else:
        invalid = selection(menu, state_id="state-0000000000000000")
        if sequence_name == "nonmax-valid":
            maxima = accepted_maxima(menu)
            lower = next(i for i, action in enumerate(menu.actions) if action not in maxima)
            completions.responses.extend([
                selection(menu, lower), selection(menu, menu.actions.index(maxima[0]))
            ])
        elif sequence_name == "wrong_pair-valid":
            completions.responses.extend([
                selection(menu, observed_limiting_pairs=[[menu.source.selected_ids[0], menu.source.selected_ids[2]]]),
                selection(menu),
            ])
        elif sequence_name == "wrong_tool-valid":
            completions.responses.extend([
                response("select_diverse_panel", {"legacy": True}), selection(menu)
            ])
        elif sequence_name == "malformed-valid":
            completions.responses.extend([
                raw_response("select_next_panel_swap", "{"), selection(menu)
            ])
        elif sequence_name in {"invalid-valid-invalid", "invalid-invalid"}:
            completions.responses.extend(
                [invalid, selection(menu)]
                if sequence_name == "invalid-valid-invalid"
                else [invalid, invalid]
            )
        else:
            completions.responses.append(selection(menu))

        if sequence_name == "invalid-invalid":
            with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
                controller.request_objective_selection()
        else:
            pending = controller.request_objective_selection()
            controller.execute_objective_selection(pending)
            if sequence_name in {"valid-invalid-invalid", "invalid-valid-invalid"}:
                next_menu = controller.pending_action_menu
                stale = selection(next_menu, state_id="state-0000000000000000")
                completions.responses.extend(
                    [stale, stale]
                    if sequence_name == "valid-invalid-invalid"
                    else [stale]
                )
                with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
                    controller.request_objective_selection()

    assert (
        controller.accepted_attempt_count,
        controller.rejected_selection_count,
        controller.correction_prompts_sent,
        controller.selection_response_count,
        controller.provider_request_attempt_count,
    ) == expected_counters
    assert len(controller.objective_attempts) == controller.accepted_attempt_count
    actual_terminal = (
        controller.objective_run.termination_reason
        if controller.objective_run is not None
        else None
    )
    assert actual_terminal == expected_terminal
    assert len(completions.calls) == controller.provider_request_attempt_count
    assert_paired(controller.session.messages)


def safe_objective_proposals():
    return [AUTO_SELECTION]


def execute_safe_objective(controller):
    while controller.objective_run is None:
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


def test_legacy_objective_proposal_shape_is_not_available_or_executable():
    assert "ObjectiveProposal" not in vars(demo_agent)
    controller, _ = completed_controller([])
    controller.begin_objective_challenge()
    legacy = {"selected_ids": ["mol-0", "mol-1", "mol-2", "mol-3"], "decision_basis": "model prose"}
    with pytest.raises((AttributeError, demo_agent.ToolCallError)):
        controller.execute_objective_attempt(legacy)


@pytest.mark.parametrize(
    "extra_field",
    ["selected_ids", "decision_basis", "rationale", "score", "delta", "summary"],
)
def test_objective_selection_forbids_legacy_or_model_authored_fields(extra_field):
    controller, _ = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    payload = {
        "state_id": menu.state_id,
        "swap_id": accepted_maxima(menu)[0].swap_id,
        "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
        "decision_rule": "maximize_predicted_minimum_distance",
        extra_field: "forbidden",
    }
    with pytest.raises(Exception):
        demo_agent.ObjectiveSelection.model_validate(payload)


def test_valid_selection_stays_pending_until_exact_execution_and_then_pairs():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.append(selection(menu))
    pending = controller.request_objective_selection()
    assert controller.accepted_attempt_count == 0 and controller.objective_attempts == []
    copied = demo_agent.ObjectiveSelection.model_validate(pending.model_dump())
    with pytest.raises(demo_agent.ToolCallError, match="exact pending"):
        controller.execute_objective_selection(copied)
    attempt = controller.execute_objective_selection(pending)
    assert attempt is controller.objective_attempts[0]
    assert controller.accepted_attempt_count == 1
    assert_paired(controller.session.messages)


@pytest.mark.parametrize(
    "overrides",
    [
        {"state_id": "state-0000000000000000"},
        {"observed_limiting_pairs": [["mol-2", "mol-3"]]},
        {"decision_rule": "wrong_rule"},
        {"swap_id": "mol-0->outside"},
    ],
)
def test_schema_or_state_invalid_selection_is_corrected_without_measurement(overrides):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.extend([selection(menu, **overrides), selection(menu)])
    pending = controller.request_objective_selection()
    assert controller.rejected_selection_count == 1
    assert controller.accepted_attempt_count == 0
    controller.execute_objective_selection(pending)
    assert controller.accepted_attempt_count == 1


@pytest.mark.parametrize("raw_arguments", ["{", "[]", "null", '"model prose"'])
def test_malformed_with_real_call_id_is_preserved_and_paired(raw_arguments):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.extend([
        raw_response("select_next_panel_swap", raw_arguments),
        selection(menu),
    ])
    controller.request_objective_selection()
    rejected, tool = controller.session.messages[-4:-2]
    assert rejected["tool_calls"][0]["function"]["arguments"] == raw_arguments
    assert tool["tool_call_id"] == rejected["tool_calls"][0]["id"]
    assert json.loads(tool["content"])["accepted"] is False


@pytest.mark.parametrize("content", [None, "plain model prose", "{}", "[]"])
def test_no_tool_assistant_is_preserved_without_phantom_pairing(content):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.extend([content_response(content), selection(menu)])
    controller.request_objective_selection()
    rejected = controller.session.messages[-3]
    assert rejected == {"role": "assistant", "content": content}
    assert controller.session.messages[-2]["role"] == "user"
    assert all(
        not (item.get("role") == "tool" and item.get("tool_call_id") is None)
        for item in controller.session.messages
    )


def test_missing_call_id_is_preserved_without_tool_result_then_corrected():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    missing = response("select_next_panel_swap", {
        "state_id": menu.state_id,
        "swap_id": menu.actions[0].swap_id,
        "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
        "decision_rule": "maximize_predicted_minimum_distance",
    })
    missing.choices[0].message.tool_calls[0].id = None
    completions.responses.extend([missing, selection(menu)])
    controller.request_objective_selection()
    rejected = controller.session.messages[-3]
    assert rejected["tool_calls"][0]["id"] is None
    assert controller.session.messages[-2]["role"] == "user"


def test_wrong_tool_with_real_id_is_preserved_and_paired():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    wrong = response("select_diverse_panel", {"selected_ids": ["model", "prose"]})
    completions.responses.extend([wrong, selection(menu)])
    controller.request_objective_selection()
    rejected, tool = controller.session.messages[-4:-2]
    assert rejected["tool_calls"][0]["function"]["name"] == "select_diverse_panel"
    assert tool["tool_call_id"] == rejected["tool_calls"][0]["id"]


def test_two_tool_calls_preserve_and_pair_both_real_ids_before_one_correction():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    valid_arguments = selection(menu).choices[0].message.tool_calls[0].function.arguments
    multiple = envelope_response([
        {"id": "call-first", "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
        {"id": "call-second", "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
    ], content="provider multi-call envelope")
    completions.responses.extend([multiple, selection(menu)])

    controller.request_objective_selection()

    rejected, first_tool, second_tool, correction, accepted = controller.session.messages[-5:]
    assert rejected["content"] == "provider multi-call envelope"
    assert [call["id"] for call in rejected["tool_calls"]] == ["call-first", "call-second"]
    assert [first_tool["tool_call_id"], second_tool["tool_call_id"]] == ["call-first", "call-second"]
    assert first_tool["role"] == second_tool["role"] == "tool"
    assert correction["role"] == "user"
    assert accepted["role"] == "assistant"
    assert controller.selection_response_count == 2
    assert controller.rejected_selection_count == 1
    assert controller.accepted_attempt_count == 0


def test_multi_call_pairs_only_nonempty_actual_ids_before_correction():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    valid_arguments = selection(menu).choices[0].message.tool_calls[0].function.arguments
    multiple = envelope_response([
        {"id": None, "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
        {"id": "call-real", "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
    ])
    completions.responses.extend([multiple, selection(menu)])

    controller.request_objective_selection()

    rejected, tool, correction, accepted = controller.session.messages[-4:]
    assert [call["id"] for call in rejected["tool_calls"]] == [None, "call-real"]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call-real"
    assert correction["role"] == "user" and accepted["role"] == "assistant"
    assert controller.selection_response_count == 2
    assert controller.rejected_selection_count == 1
    assert all(
        message.get("tool_call_id") is not None
        for message in controller.session.messages
        if message.get("role") == "tool"
    )


def test_single_non_function_call_is_rejected_paired_and_never_measured():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    valid_arguments = selection(menu).choices[0].message.tool_calls[0].function.arguments
    invalid = envelope_response([{
        "id": "call-non-function",
        "type": "custom",
        "name": "select_next_panel_swap",
        "arguments": valid_arguments,
    }])
    completions.responses.extend([invalid, selection(menu)])

    pending = controller.request_objective_selection()

    rejected, tool, correction, accepted = controller.session.messages[-4:]
    assert rejected["tool_calls"][0]["type"] == "custom"
    assert tool["tool_call_id"] == "call-non-function"
    assert correction["role"] == "user" and accepted["role"] == "assistant"
    assert controller.rejected_selection_count == 1
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []
    assert pending is controller.pending_objective_selection


def test_two_invalid_envelopes_terminalize_without_extra_prompt_or_third_request():
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    valid_arguments = selection(menu).choices[0].message.tool_calls[0].function.arguments
    completions.responses.extend([
        envelope_response([
            {"id": "call-a", "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
            {"id": "call-b", "type": "function", "name": "select_next_panel_swap", "arguments": valid_arguments},
        ]),
        envelope_response([{
            "id": "call-c",
            "type": "custom",
            "name": "select_next_panel_swap",
            "arguments": valid_arguments,
        }]),
    ])

    with pytest.raises(demo_agent.ObjectiveCorrectionLimitError):
        controller.request_objective_selection()

    assert len(completions.calls) == 2
    assert controller.selection_response_count == 2
    assert controller.rejected_selection_count == 2
    assert controller.correction_prompts_sent == 1
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []
    assert controller.objective_run.termination_reason == "objective_correction_limit"
    assert [message["role"] for message in controller.session.messages[-3:]] == [
        "user", "assistant", "tool"
    ]
    assert controller.session.messages[-1]["tool_call_id"] == "call-c"


@pytest.mark.parametrize("drift", ["state", "source", "actions", "pairs", "rule"])
def test_execute_revalidates_pending_selection_against_current_menu(drift):
    controller, completions = completed_controller([])
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.append(selection(menu))
    pending = controller.request_objective_selection()
    if drift == "state":
        controller.pending_action_menu = replace(menu, state_id="state-0000000000000000")
    elif drift == "source":
        controller.pending_action_menu = replace(menu, accepted_attempt_count=1)
    elif drift == "actions":
        controller.pending_action_menu = replace(menu, actions=menu.actions[1:])
    elif drift == "pairs":
        object.__setattr__(pending, "observed_limiting_pairs", [["mol-2", "mol-3"]])
    else:
        object.__setattr__(pending, "decision_rule", "wrong_rule")
    with pytest.raises(demo_agent.ToolCallError):
        controller.execute_objective_selection(pending)
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []


@pytest.mark.parametrize(
    "failure_site",
    ["evaluation", "next_menu", "terminal_run", "o01", "serialization", "append", "invariant"],
)
def test_objective_evaluation_failures_roll_back_and_terminalize_atomically(
    monkeypatch, failure_site
):
    context = (
        controlled_context_with_ranked_swaps()
        if failure_site in {"terminal_run", "o01"}
        else two_revision_context()
    )
    controller, completions = completed_controller([])
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.append(selection(menu))
    pending = controller.request_objective_selection()
    reached = []

    target = {
        "evaluation": "evaluate_selected_swap",
        "next_menu": "build_action_menu",
        "terminal_run": "terminal_objective_run",
        "o01": "build_objective_evidence",
        "serialization": "_serialize",
    }.get(failure_site)
    if target is not None:
        original = getattr(demo_agent, target)

        def fail_once(*args, **kwargs):
            reached.append(target)
            monkeypatch.setattr(demo_agent, target, original)
            raise RuntimeError(f"injected {failure_site} failure")

        monkeypatch.setattr(demo_agent, target, fail_once)
    elif failure_site == "append":
        controller.session.messages = FailOnceList(controller.session.messages)
    else:
        original = controller._validate_objective_commit

        def fail_invariant(transition):
            reached.append("invariant")
            controller._validate_objective_commit = original
            raise RuntimeError("injected invariant failure")

        controller._validate_objective_commit = fail_invariant

    with pytest.raises(demo_agent.ObjectiveEvaluationError):
        controller.execute_objective_selection(pending)

    if failure_site == "append":
        assert controller.session.messages[-1]["role"] == "tool"
        assert type(controller.session.messages) is list
    else:
        assert reached == [target or "invariant"]
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []
    assert controller.pending_objective_selection is None
    assert controller.pending_action_menu is None
    assert controller.objective_run.termination_reason == "evaluation_not_completed"
    assert controller.objective_run.attempts == ()
    assert controller.objective_evidence.key == "O01"
    error_result = json.loads(controller.session.messages[-1]["content"])
    assert error_result == {"accepted": False, "reason": "evaluation_not_completed"}
    assert_paired(controller.session.messages)


def test_commit_snapshot_restores_plain_messages_and_every_objective_field(monkeypatch):
    controller, completions = completed_controller([])
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: two_revision_context())
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.append(selection(menu))
    pending = controller.request_objective_selection()
    injected = FailOnceList(controller.session.messages)
    controller.session.messages = injected
    snapshot = controller._capture_objective_commit_snapshot()

    assert type(snapshot.messages) is tuple
    assert snapshot.messages == tuple(deepcopy(list(injected)))

    with pytest.raises(demo_agent.ObjectiveEvaluationError):
        controller.execute_objective_selection(pending)

    assert injected.append_attempts == 1
    assert type(controller.session.messages) is list
    assert controller.accepted_attempt_count == len(controller.objective_attempts) == 0
    assert controller.rejected_selection_count == 0
    assert controller.correction_prompts_sent == 0
    assert controller.selection_response_count == 1
    assert controller.provider_request_attempt_count == 1
    assert controller.objective_transport_retry_used is False
    assert controller.objective_transport_retry_pending is False


def test_double_failure_restores_deeply_isolated_pending_selection(monkeypatch):
    controller, completions = completed_controller([])
    monkeypatch.setattr(
        demo_agent, "build_objective_context", lambda _state: two_revision_context()
    )
    controller.begin_objective_challenge()
    menu = controller.pending_action_menu
    completions.responses.append(selection(menu))
    pending = controller.request_objective_selection()
    expected_dump = deepcopy(pending.model_dump())
    expected_json = demo_agent._serialize(expected_dump)
    expected_messages = deepcopy(controller.session.messages)
    expected_turn_count = controller.session.turn_count
    reached = []

    def mutate_then_fail(_transition):
        reached.append("commit")
        pending.observed_limiting_pairs[0][0] = "mutated-in-place"
        raise RuntimeError("injected commit invariant failure")

    def fail_terminal_recovery(*_args, **_kwargs):
        reached.append("terminal_recovery")
        raise RuntimeError("injected terminal recovery failure")

    monkeypatch.setattr(controller, "_validate_objective_commit", mutate_then_fail)
    monkeypatch.setattr(demo_agent, "terminal_objective_run", fail_terminal_recovery)

    with pytest.raises(
        demo_agent.ObjectiveEvaluationError,
        match="before a safe terminal result could be recorded",
    ):
        controller.execute_objective_selection(pending)

    restored = controller.pending_objective_selection
    assert reached == ["commit", "terminal_recovery"]
    assert demo_agent._serialize(restored.model_dump()) == expected_json
    assert restored.model_dump() == expected_dump
    assert restored is not pending
    assert restored.observed_limiting_pairs is not pending.observed_limiting_pairs
    assert restored.observed_limiting_pairs[0] is not pending.observed_limiting_pairs[0]
    assert controller.session.messages == expected_messages
    assert controller.session.messages is not expected_messages
    assert controller.session.turn_count == expected_turn_count
    assert controller.pending_action_menu == menu
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []
    assert controller.objective_run is None
    assert controller.objective_evidence is None


def test_unreachable_policy_maps_specific_eligibility_error_without_provider(monkeypatch):
    controller, completions = completed_controller([])
    context = two_revision_context()
    monkeypatch.setattr(demo_agent, "certify_argmax_reachability", lambda _context: False)
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    with pytest.raises(
        demo_agent.ObjectiveEligibilityError,
        match="not reachable under the bounded decision policy",
    ):
        controller.begin_objective_challenge()
    assert completions.calls == []


def test_no_legal_menu_terminalizes_without_provider(monkeypatch):
    from objective_fixtures import controlled_context_without_improving_swaps
    controller, completions = completed_controller([])
    context = controlled_context_without_improving_swaps()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    monkeypatch.setattr(demo_agent, "certify_argmax_reachability", lambda _context: True)
    controller.begin_objective_challenge()
    assert controller.objective_run.termination_reason == "no_legal_improving_swap"
    assert completions.calls == []

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

    assert "candidate_actions" in prompt
    assert "current_limiting_pairs" in prompt and "state_id" in prompt
    assert "decision_basis" not in prompt and "rationale" not in prompt


def test_empty_production_menu_stops_before_any_hosted_request(monkeypatch):
    from objective_fixtures import controlled_context_without_improving_swaps

    controller, completions = completed_controller([])
    context = controlled_context_without_improving_swaps()
    monkeypatch.setattr(demo_agent, "build_objective_context", lambda _state: context)
    monkeypatch.setattr(demo_agent, "certify_argmax_reachability", lambda _context: True)

    controller.begin_objective_challenge()

    assert controller.objective_attempts == []
    assert controller.objective_run.termination_reason == "no_legal_improving_swap"
    assert completions.calls == []


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
    assert result.turn_count == 9
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
    assert controller.session.turn_count == 9
    assert len(completions.calls) == 2


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

    assert result.turn_count == 10 <= demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
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
    assert controller.session.turn_count == 9

    result = controller.request_synthesis()

    assert result.turn_count == 10 <= demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
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

    for expected_turn in range(9, demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS + 1):
        with pytest.raises(demo_agent.ConclusionValidationError):
            controller.request_synthesis()
        assert controller.session.turn_count == expected_turn
        assert controller.session.messages[-2]["role"] == "assistant"
        assert controller.session.messages[-1]["role"] == "tool"

    with pytest.raises(demo_agent.ToolCallError):
        controller.request_synthesis()

    assert controller.session.turn_count == demo_agent.MAX_OBJECTIVE_SYNTHESIS_TURNS
    assert len(completions.calls) == 6
