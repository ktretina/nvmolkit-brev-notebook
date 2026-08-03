import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, PermissionDeniedError
from pydantic import ValidationError

import demo_agent
from chemistry_workflow import (
    EvidenceRecord,
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    WorkflowState,
)


VALID_API_KEY = "nvapi-test-key"
SECRET = "do-not-leak-this-secret"
STAGES = (
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
)
PHASES = (
    WorkflowPhase.INSPECTED,
    WorkflowPhase.FINGERPRINTED,
    WorkflowPhase.COMPARED,
    WorkflowPhase.CLUSTERED,
    WorkflowPhase.EMBEDDED,
    WorkflowPhase.OPTIMIZED,
)
VALID_ARGS = {
    "inspect_library": {},
    "generate_morgan_fingerprints": {
        "radius": 2,
        "size": 1024,
        "decision_basis": "Use a compact standard fingerprint.",
    },
    "measure_tanimoto_similarity": {},
    "discover_fused_butina_clusters": {
        "cutoff": 0.5,
        "decision_basis": "Use the observed similarity spread.",
    },
    "embed_representative_conformers": {
        "representative_count": 4,
        "policy": "include_singleton_if_available",
        "conformers_per_representative": 4,
        "decision_basis": "Sample diverse clusters and one singleton.",
    },
    "optimize_conformers_mmff94": {},
}


def call(name, arguments, call_id=None, call_type="function"):
    return SimpleNamespace(
        id=call_id or f"call-{name}",
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def response(name, arguments, **kwargs):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=kwargs.get("content"),
                    tool_calls=[
                        call(
                            name,
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments),
                            call_id=kwargs.get("call_id"),
                            call_type=kwargs.get("call_type", "function"),
                        )
                    ],
                )
            )
        ]
    )


def plan_arguments(stages=STAGES):
    return {
        "stages": [
            {
                "stage": stage,
                "rationale": f"Run {stage.replace('_', ' ')} after its prerequisite.",
            }
            for stage in stages
        ]
    }


def valid_responses():
    return [response("submit_workflow_plan", plan_arguments())] + [
        response(stage, VALID_ARGS[stage]) for stage in STAGES
    ]


class FakeCompletions:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        if not self.responses:
            return SimpleNamespace(choices=[])
        return self.responses.pop(0)


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def fake_executors(record=None):
    calls = record if record is not None else []
    executors = {}
    for stage, next_phase in zip(STAGES, PHASES):
        def execute(state, _stage=stage, _phase=next_phase, **kwargs):
            calls.append((_stage, kwargs))
            summary = {"stage": _stage, "sequence": len(calls)}
            state.summaries[_stage] = summary
            state.phase = _phase
            return StageResult(_stage, _stage, summary)

        executors[stage] = execute
    executors["build_workflow_report"] = lambda state: WorkflowReport(
        evidence=(EvidenceRecord("E01", "test", "{}", "test"),)
    )
    return executors


def run(responses=None, **kwargs):
    completions = FakeCompletions(responses or valid_responses())
    result = demo_agent.run_scientific_loop(
        "Analyze the bundled molecular library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=fake_executors(),
        **kwargs,
    )
    return result, completions


def request_error(status, error_type):
    http_response = httpx.Response(
        status,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )
    return error_type(SECRET, response=http_response, body=None)


def test_stateful_loop_uses_one_history_and_exactly_seven_hosted_turns():
    result, completions = run()

    assert result.turn_count == 7 == len(completions.calls)
    assert result.report.evidence[0].key == "E01"
    roles = [message["role"] for message in result.messages]
    assert roles[:2] == ["system", "user"]
    assert roles.count("assistant") == 7
    assert roles.count("tool") == 7
    for index, hosted_call in enumerate(completions.calls):
        assert hosted_call["messages"] == list(result.messages[: 2 + 2 * index])


def test_every_assistant_call_has_a_matching_canonical_tool_result():
    result, _ = run()

    for index, message in enumerate(result.messages):
        if message["role"] != "assistant":
            continue
        tool_message = result.messages[index + 1]
        tool_call = message["tool_calls"][0]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == tool_call["id"]
        decoded = json.loads(tool_message["content"])
        assert tool_message["content"] == json.dumps(
            decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def test_skill_is_exact_initial_grounding_not_an_artificial_tool():
    result, completions = run()
    skill = Path("skills/nvmolkit/SKILL.md").read_text()
    system = result.messages[0]["content"]

    assert "skills/nvmolkit/SKILL.md" in system
    assert skill in system
    assert "RDKit input validation" in system
    assert "nvMolKit GPU" in system
    assert "concise decision summaries" in system
    assert "hidden chain-of-thought" in system
    assert "read_nvmolkit_skill" not in system
    exposed_names = {
        hosted["tools"][0]["function"]["name"] for hosted in completions.calls
    }
    assert exposed_names == {"submit_workflow_plan", *STAGES}
    assert "read_nvmolkit_skill" not in exposed_names


def test_each_turn_exposes_and_forces_only_the_phase_eligible_schema():
    _, completions = run()

    expected = ("submit_workflow_plan", *STAGES)
    for hosted, tool_name in zip(completions.calls, expected):
        assert len(hosted["tools"]) == 1
        function = hosted["tools"][0]["function"]
        assert function["name"] == tool_name
        assert function["strict"] is True
        assert function["parameters"]["additionalProperties"] is False
        assert hosted["tool_choice"] == {
            "type": "function",
            "function": {"name": tool_name},
        }


def test_plan_requires_exact_dependency_order_and_nonempty_rationales():
    parsed = demo_agent.WorkflowPlan.model_validate(plan_arguments())
    assert tuple(stage.stage for stage in parsed.stages) == STAGES
    with pytest.raises(ValidationError):
        demo_agent.WorkflowPlan.model_validate(plan_arguments(reversed(STAGES)))
    bad = plan_arguments()
    bad["stages"][2]["rationale"] = ""
    with pytest.raises(ValidationError):
        demo_agent.WorkflowPlan.model_validate(bad)


@pytest.mark.parametrize(
    "value",
    ["", "short", "valid rationale\ncontinued", "use `code` here", "x" * 241],
)
def test_decision_basis_rejects_unpresentable_or_unbounded_text(value):
    with pytest.raises(ValidationError):
        demo_agent.FingerprintArgs.model_validate(
            {"radius": 2, "size": 1024, "decision_basis": value}
        )


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        ("InspectionArgs", {"extra": True}),
        ("SimilarityArgs", {"extra": True}),
        ("OptimizationArgs", {"extra": True}),
        ("FingerprintArgs", {"radius": 1, "size": 1024, "decision_basis": "A valid scientific reason."}),
        ("FingerprintArgs", {"radius": 2, "size": "1024", "decision_basis": "A valid scientific reason."}),
        ("ClusterArgs", {"cutoff": 0.39, "decision_basis": "A valid scientific reason."}),
        ("ClusterArgs", {"cutoff": "0.5", "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 7, "policy": "largest_clusters_first", "conformers_per_representative": 4, "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 4, "policy": "unknown", "conformers_per_representative": 4, "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 4, "policy": "largest_clusters_first", "conformers_per_representative": 2, "decision_basis": "A valid scientific reason."}),
    ],
)
def test_argument_models_are_strict_bounded_frozen_and_forbid_extras(model, arguments):
    model = getattr(demo_agent, model)
    assert model.model_config["strict"] is True
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        model.model_validate(arguments)


def test_adaptive_choices_receive_prerequisite_summaries_in_same_history():
    _, completions = run()
    cluster_history = completions.calls[4]["messages"]
    embed_history = completions.calls[5]["messages"]

    assert any(
        message["role"] == "tool"
        and "measure_tanimoto_similarity" in message["content"]
        for message in cluster_history
    )
    assert any(
        message["role"] == "tool"
        and "discover_fused_butina_clusters" in message["content"]
        for message in embed_history
    )


def test_decision_basis_is_retained_in_history_but_not_passed_to_executors():
    recorded = []
    completions = FakeCompletions(valid_responses())
    demo_agent.run_scientific_loop(
        "Analyze the library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=fake_executors(recorded),
    )

    fingerprint_call = next(
        message
        for message in completions.calls[3]["messages"]
        if message["role"] == "assistant"
        and message["tool_calls"][0]["function"]["name"]
        == "generate_morgan_fingerprints"
    )
    assert json.loads(
        fingerprint_call["tool_calls"][0]["function"]["arguments"]
    )["decision_basis"]
    assert recorded == [
        ("inspect_library", {}),
        ("generate_morgan_fingerprints", {"fingerprint_radius": 2, "fingerprint_size": 1024}),
        ("measure_tanimoto_similarity", {}),
        ("discover_fused_butina_clusters", {"cluster_cutoff": 0.5}),
        ("embed_representative_conformers", {"representative_count": 4, "representative_policy": "include_singleton_if_available", "conformers_per_representative": 4}),
        ("optimize_conformers_mmff94", {}),
    ]


@pytest.mark.parametrize(
    "bad_tool_calls",
    [
        None,
        [],
        [call("inspect_library", "{}"), call("inspect_library", "{}")],
        [call("wrong_stage", "{}")],
        [call("inspect_library", "{}", call_id=" ")],
        [call("inspect_library", "{}", call_type="custom")],
        [call("inspect_library", "not-json")],
        [call("inspect_library", "[]")],
        [call("inspect_library", '{"extra":true}')],
    ],
)
def test_invalid_scientific_calls_fail_before_any_executor(bad_tool_calls):
    plan = response("submit_workflow_plan", plan_arguments())
    bad = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="early text", tool_calls=bad_tool_calls))]
    )
    completions = FakeCompletions([plan, bad])
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze the library.",
            VALID_API_KEY,
            client=fake_client(completions),
            executors=fake_executors(executor_calls),
        )

    assert executor_calls == []
    assert len(completions.calls) == 2


def test_repeated_or_out_of_phase_call_stops_before_wrong_executor():
    responses = valid_responses()
    responses[2] = response("inspect_library", {})
    calls = []
    completions = FakeCompletions(responses)

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors(calls)
        )

    assert calls == [("inspect_library", {})]


def test_executor_failure_is_secret_safe_and_stops_later_calls():
    executors = fake_executors()
    executors["inspect_library"] = lambda state: (_ for _ in ()).throw(RuntimeError(SECRET))
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )

    assert SECRET not in str(error.value)
    assert len(completions.calls) == 2


def test_executor_cannot_report_success_without_exact_phase_advance():
    class SerializationProbe:
        calls = 0

        def tolist(self):
            self.calls += 1
            return ["serialized"]

    probe = SerializationProbe()
    executors = fake_executors()
    executors["inspect_library"] = lambda state: StageResult(
        "inspect_library", "inspect", {"probe": probe}
    )
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError, match="phase"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )

    assert probe.calls == 0
    assert len(completions.calls) == 2


def test_nonfinite_tool_summary_is_rejected_before_next_hosted_turn():
    executors = fake_executors()
    def nonfinite(state):
        state.phase = WorkflowPhase.INSPECTED
        return StageResult("inspect_library", "inspect", {"value": float("nan")})
    executors["inspect_library"] = nonfinite
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError, match="serialized safely"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )
    assert len(completions.calls) == 2


def test_hosted_configuration_disables_retries_and_thinking(monkeypatch):
    constructor_calls = []
    monkeypatch.setattr(demo_agent, "OpenAI", lambda **kwargs: constructor_calls.append(kwargs) or object())
    assert demo_agent._client(VALID_API_KEY) is not None
    assert constructor_calls == [{"base_url": demo_agent.NVIDIA_BASE_URL, "api_key": VALID_API_KEY, "max_retries": 0}]

    _, completions = run()
    for call_record in completions.calls:
        assert call_record["model"] == demo_agent.DEFAULT_MODEL
        assert call_record["temperature"] == 0.2
        assert call_record["stream"] is False
        assert call_record["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("api_key", ["", "ngc-key", f"nvapi-{SECRET} "])
def test_invalid_key_is_secret_safe_and_stops_before_network(api_key):
    completions = FakeCompletions(valid_responses())
    with pytest.raises(ValueError) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", api_key, client=fake_client(completions), executors=fake_executors()
        )
    assert api_key not in str(error.value) if api_key else True
    assert completions.calls == []


@pytest.mark.parametrize(
    ("hosted_error", "exception_type"),
    [
        (request_error(401, AuthenticationError), ValueError),
        (request_error(403, PermissionDeniedError), ValueError),
        (RuntimeError(SECRET), demo_agent.ToolCallError),
    ],
)
def test_hosted_errors_are_secret_safe(hosted_error, exception_type):
    completions = FakeCompletions(error=hosted_error)
    with pytest.raises(exception_type) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors()
        )
    assert SECRET not in str(error.value)
    assert VALID_API_KEY not in str(error.value)
    assert len(completions.calls) == 1


def test_empty_hosted_response_fails_closed_without_retry():
    completions = FakeCompletions([])
    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors()
        )
    assert len(completions.calls) == 1


def test_default_dispatcher_calls_only_fixed_chemistry_facade(monkeypatch):
    calls = []
    for stage, phase in zip(STAGES, PHASES):
        def facade(state, *args, _stage=stage, _phase=phase, **kwargs):
            calls.append((_stage, kwargs))
            state.phase = _phase
            return StageResult(_stage, _stage, {"stage": _stage})
        monkeypatch.setattr(demo_agent, stage, facade)
    report = WorkflowReport((EvidenceRecord("E01", "test", "{}", "test"),))
    monkeypatch.setattr(demo_agent, "build_workflow_report", lambda state: report)
    completions = FakeCompletions(valid_responses())

    result = demo_agent.run_scientific_loop(
        "Analyze.", VALID_API_KEY, client=fake_client(completions)
    )

    assert result.report is report
    assert [name for name, _ in calls] == list(STAGES)


def test_empty_executor_registry_is_rejected_without_using_production_tools(monkeypatch):
    default_calls = []
    monkeypatch.setattr(
        demo_agent,
        "_default_executors",
        lambda: default_calls.append("called") or fake_executors(),
    )
    completions = FakeCompletions(valid_responses())

    with pytest.raises(ValueError, match="fixed scientific workflow"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors={}
        )

    assert default_calls == []
    assert completions.calls == []


def test_session_turn_limit_fails_closed():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
        turn_count=7,
    )
    completions = FakeCompletions([response("submit_workflow_plan", plan_arguments())])
    with pytest.raises(demo_agent.ToolCallError, match="turn limit"):
        demo_agent._request_call(
            session,
            fake_client(completions),
            "submit_workflow_plan",
            demo_agent.WorkflowPlan,
            demo_agent.DEFAULT_MODEL,
        )
    assert completions.calls == []
