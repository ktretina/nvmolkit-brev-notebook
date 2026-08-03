import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, PermissionDeniedError
from pydantic import BaseModel, ValidationError

import demo_agent


TOOL_NAMES = (
    "read_nvmolkit_skill",
    "prepare_molecular_sample",
    "compute_morgan_fingerprints",
    "compute_tanimoto_similarity",
    "cluster_with_fused_butina",
    "generate_and_optimize_conformers",
)

MODEL_NAMES = {
    "read_nvmolkit_skill": "ReadSkillArgs",
    "prepare_molecular_sample": "PrepareSampleArgs",
    "compute_morgan_fingerprints": "FingerprintArgs",
    "compute_tanimoto_similarity": "SimilarityArgs",
    "cluster_with_fused_butina": "ClusterArgs",
    "generate_and_optimize_conformers": "ConformerArgs",
}

VALID_ARGUMENTS = {
    "read_nvmolkit_skill": {},
    "prepare_molecular_sample": {"preview_count": 24},
    "compute_morgan_fingerprints": {
        "fingerprint_radius": 2,
        "fingerprint_size": 1024,
    },
    "compute_tanimoto_similarity": {},
    "cluster_with_fused_butina": {"cluster_cutoff": 0.5},
    "generate_and_optimize_conformers": {
        "representative_count": 4,
        "conformers_per_representative": 4,
    },
}

INVALID_ARGUMENTS = {
    "read_nvmolkit_skill": {"unexpected": True},
    "prepare_molecular_sample": {"preview_count": 23},
    "compute_morgan_fingerprints": {
        "fingerprint_radius": 4,
        "fingerprint_size": 1024,
    },
    "compute_tanimoto_similarity": {"unexpected": True},
    "cluster_with_fused_butina": {"cluster_cutoff": 0.39},
    "generate_and_optimize_conformers": {
        "representative_count": 7,
        "conformers_per_representative": 4,
    },
}

VALID_API_KEY = "nvapi-test-key"
SECRET_MARKER = "do-not-leak-this-secret"
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted Developer API key. Generate it from the "
    "Nemotron build.nvidia.com model page, then paste only the bare key; it "
    "starts with nvapi-. An NGC personal key is a different credential and "
    "must not be substituted."
)
GENERIC_ERROR = "The hosted Nemotron request failed. Check network access and model availability."


class FakeCompletions:
    def __init__(self, *, content=None, tool_calls=None, error=None):
        self.content = content
        self.tool_calls = tool_calls
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content,
                        tool_calls=self.tool_calls,
                    )
                )
            ]
        )


class SecretConversion:
    def tolist(self):
        raise RuntimeError(SECRET_MARKER)


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def tool_call(
    arguments,
    *,
    name="cluster_with_fused_butina",
    call_id="call-123",
    call_type="function",
    function_marker="valid",
):
    function = (
        SimpleNamespace(name=name, arguments=arguments)
        if function_marker == "valid"
        else function_marker
    )
    return SimpleNamespace(id=call_id, type=call_type, function=function)


def request(tool_name, completions, **kwargs):
    return demo_agent.request_tool_call(
        VALID_API_KEY,
        tool_name=tool_name,
        task_prompt=f"Run the bounded {tool_name} stage.",
        context={"stage": tool_name, "count": 24},
        client=fake_client(completions),
        **kwargs,
    )


def valid_decision(tool_name="cluster_with_fused_butina"):
    raw = json.dumps(VALID_ARGUMENTS[tool_name])
    completions = FakeCompletions(
        tool_calls=[tool_call(raw, name=tool_name, call_id=f"call-{tool_name}")]
    )
    return request(tool_name, completions)


def response_error(status, error_type):
    response = httpx.Response(
        status,
        request=httpx.Request(
            "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
        ),
    )
    return error_type(SECRET_MARKER, response=response, body=None)


def test_client_disables_openai_sdk_retries(monkeypatch):
    calls = []
    sentinel = object()

    def recording_openai(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(demo_agent, "OpenAI", recording_openai)

    result = demo_agent._client(VALID_API_KEY)

    assert result is sentinel
    assert calls == [
        {
            "base_url": demo_agent.NVIDIA_BASE_URL,
            "api_key": VALID_API_KEY,
            "max_retries": 0,
        }
    ]


def test_public_models_and_tool_mappings_are_exact():
    assert set(demo_agent.TOOL_ARGUMENT_MODELS) == set(TOOL_NAMES)
    assert set(demo_agent.TOOL_DESCRIPTIONS) == set(TOOL_NAMES)
    for tool_name, model_name in MODEL_NAMES.items():
        model = getattr(demo_agent, model_name)
        assert demo_agent.TOOL_ARGUMENT_MODELS[tool_name] is model
        description = demo_agent.TOOL_DESCRIPTIONS[tool_name]
        assert isinstance(description, str) and description.strip()
        assert len(description.split()) <= 24


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_all_six_models_accept_valid_arguments_and_forbid_extras(tool_name):
    model = demo_agent.TOOL_ARGUMENT_MODELS[tool_name]

    parsed = model.model_validate(VALID_ARGUMENTS[tool_name])

    assert model.model_config["frozen"] is True
    assert isinstance(parsed, BaseModel)
    assert parsed.model_dump() == VALID_ARGUMENTS[tool_name]
    with pytest.raises(ValidationError):
        model.model_validate({**VALID_ARGUMENTS[tool_name], "unexpected": True})


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("prepare_molecular_sample", {}),
        ("prepare_molecular_sample", {"preview_count": "24"}),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_size": 1024},
        ),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": 2},
        ),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": "2", "fingerprint_size": 1024},
        ),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": 2, "fingerprint_size": "1024"},
        ),
        ("cluster_with_fused_butina", {}),
        ("cluster_with_fused_butina", {"cluster_cutoff": "0.5"}),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 4},
        ),
        (
            "generate_and_optimize_conformers",
            {"conformers_per_representative": 4},
        ),
        (
            "generate_and_optimize_conformers",
            {"representative_count": "4", "conformers_per_representative": 4},
        ),
    ],
)
def test_models_reject_missing_or_non_strict_arguments(tool_name, arguments):
    with pytest.raises(ValidationError):
        demo_agent.TOOL_ARGUMENT_MODELS[tool_name].model_validate(arguments)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("prepare_molecular_sample", {"preview_count": 23}),
        ("prepare_molecular_sample", {"preview_count": 25}),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": 1, "fingerprint_size": 1024},
        ),
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": 2, "fingerprint_size": 4096},
        ),
        ("cluster_with_fused_butina", {"cluster_cutoff": 0.399}),
        ("cluster_with_fused_butina", {"cluster_cutoff": 0.601}),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 2, "conformers_per_representative": 4},
        ),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 4, "conformers_per_representative": 2},
        ),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 4, "conformers_per_representative": 9},
        ),
    ],
)
def test_models_reject_out_of_range_arguments(tool_name, arguments):
    with pytest.raises(ValidationError):
        demo_agent.TOOL_ARGUMENT_MODELS[tool_name].model_validate(arguments)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "compute_morgan_fingerprints",
            {"fingerprint_radius": 3, "fingerprint_size": 2048},
        ),
        ("cluster_with_fused_butina", {"cluster_cutoff": 0.4}),
        ("cluster_with_fused_butina", {"cluster_cutoff": 0.6}),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 3, "conformers_per_representative": 3},
        ),
        (
            "generate_and_optimize_conformers",
            {"representative_count": 6, "conformers_per_representative": 8},
        ),
    ],
)
def test_models_accept_all_allowed_literals_and_range_boundaries(tool_name, arguments):
    assert (
        demo_agent.TOOL_ARGUMENT_MODELS[tool_name].model_validate(arguments).model_dump()
        == arguments
    )


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_request_forces_exactly_one_strict_named_tool_schema(tool_name):
    raw = json.dumps(VALID_ARGUMENTS[tool_name])
    completions = FakeCompletions(
        tool_calls=[tool_call(raw, name=tool_name, call_id=f"call-{tool_name}")]
    )

    decision = request(tool_name, completions)

    call = completions.calls[0]
    assert call["model"] == demo_agent.DEFAULT_MODEL
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 400
    assert call["stream"] is False
    assert call["extra_body"] == demo_agent.NEMOTRON_TOOL_EXTRA_BODY == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert call["tool_choice"] == {
        "type": "function",
        "function": {"name": tool_name},
    }
    assert len(call["tools"]) == 1
    function = call["tools"][0]["function"]
    assert function["name"] == tool_name
    assert function["description"] == demo_agent.TOOL_DESCRIPTIONS[tool_name]
    assert function["strict"] is True
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == set(VALID_ARGUMENTS[tool_name])
    assert parameters["required"] == list(parameters["properties"])
    assert decision.arguments.model_dump() == VALID_ARGUMENTS[tool_name]
    user_message = call["messages"][-1]["content"]
    assert f"Run the bounded {tool_name} stage." in user_message
    assert json.dumps(
        {"stage": tool_name, "count": 24}, separators=(",", ":"), sort_keys=True
    ) in user_message


def test_valid_tool_call_returns_frozen_secret_free_decision():
    raw = '{"cluster_cutoff": 0.5}'
    completions = FakeCompletions(tool_calls=[tool_call(raw)])

    decision = request("cluster_with_fused_butina", completions)

    assert decision.source == "nemotron"
    assert decision.error is None
    assert decision.tool_name == "cluster_with_fused_butina"
    assert decision.tool_call_id == "call-123"
    assert decision.raw_arguments == raw
    assert decision.arguments == demo_agent.ClusterArgs(cluster_cutoff=0.5)
    assert "api_key" not in vars(decision)
    assert VALID_API_KEY not in repr(decision)
    with pytest.raises((FrozenInstanceError, ValidationError, AttributeError)):
        decision.tool_name = "read_nvmolkit_skill"


def test_validated_tool_arguments_cannot_be_mutated_after_decision():
    decision = valid_decision()

    with pytest.raises(ValidationError, match="(?i)frozen"):
        decision.arguments.cluster_cutoff = 0.55

    assert decision.arguments.cluster_cutoff == 0.5


@pytest.mark.parametrize(
    "tool_calls",
    [
        None,
        [],
        [tool_call('{"cluster_cutoff": 0.5}'), tool_call('{"cluster_cutoff": 0.5}')],
        [tool_call('{"cluster_cutoff": 0.5}', function_marker=None)],
        [SimpleNamespace(id="call-123", type="function", function=SimpleNamespace())],
        [tool_call('{"cluster_cutoff": 0.5}', call_type=None)],
        [tool_call('{"cluster_cutoff": 0.5}', call_type="custom")],
        [tool_call('{"cluster_cutoff": 0.5}', name="run_python")],
        [tool_call('{"cluster_cutoff": 0.5}', call_id="")],
        [tool_call('{"cluster_cutoff": 0.5}', call_id="   ")],
        [tool_call("")],
        [tool_call("   ")],
        [tool_call("not-json")],
        [tool_call("[]")],
        [tool_call("null")],
        [tool_call("{}")],
        [tool_call('{"cluster_cutoff": 0.5, "extra": true}')],
        [tool_call('{"cluster_cutoff": 0.39}')],
    ],
)
def test_invalid_hosted_response_stops_before_executor(tool_calls):
    completions = FakeCompletions(tool_calls=tool_calls)
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.request_and_execute_step(
            VALID_API_KEY,
            tool_name="cluster_with_fused_butina",
            task_prompt="Cluster the validated fingerprints.",
            context={"molecule_count": 24},
            executor=lambda arguments: executor_calls.append(arguments),
            client=fake_client(completions),
        )

    assert executor_calls == []
    assert len(completions.calls) == 1


@pytest.mark.parametrize("tool_calls", [{"x": object()}, "x", object()])
def test_non_sequence_tool_call_collections_stop_before_executor(tool_calls):
    completions = FakeCompletions(tool_calls=tool_calls)
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError) as exc_info:
        demo_agent.request_and_execute_step(
            VALID_API_KEY,
            tool_name="cluster_with_fused_butina",
            task_prompt="Cluster the validated fingerprints.",
            context={"molecule_count": 24},
            executor=lambda arguments: executor_calls.append(arguments),
            client=fake_client(completions),
        )

    assert str(exc_info.value) == "The hosted tool call collection was malformed."
    assert SECRET_MARKER not in str(exc_info.value)
    assert VALID_API_KEY not in str(exc_info.value)
    assert executor_calls == []
    assert len(completions.calls) == 1


def test_valid_response_executes_exactly_once_with_validated_model():
    completions = FakeCompletions(
        tool_calls=[tool_call('{"cluster_cutoff": 0.5}')]
    )
    executor_calls = []

    decision, result = demo_agent.request_and_execute_step(
        VALID_API_KEY,
        tool_name="cluster_with_fused_butina",
        task_prompt="Cluster the validated fingerprints.",
        context={"molecule_count": 24},
        executor=lambda arguments: executor_calls.append(arguments) or {"clusters": 3},
        client=fake_client(completions),
    )

    assert executor_calls == [decision.arguments]
    assert isinstance(executor_calls[0], demo_agent.ClusterArgs)
    assert result == {"clusters": 3}
    assert len(completions.calls) == 1


def test_allow_list_rejection_happens_before_network_or_executor():
    completions = FakeCompletions(tool_calls=[])
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.request_and_execute_step(
            VALID_API_KEY,
            tool_name="run_python",
            task_prompt="Run arbitrary code.",
            context={},
            executor=lambda arguments: executor_calls.append(arguments),
            client=fake_client(completions),
        )

    assert completions.calls == []
    assert executor_calls == []


def test_request_failure_stops_before_executor_and_hides_exception_and_key():
    completions = FakeCompletions(error=RuntimeError(f"offline {SECRET_MARKER}"))
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError) as exc_info:
        demo_agent.request_and_execute_step(
            VALID_API_KEY,
            tool_name="cluster_with_fused_butina",
            task_prompt="Cluster.",
            context={},
            executor=lambda arguments: executor_calls.append(arguments),
            client=fake_client(completions),
        )

    assert str(exc_info.value) == GENERIC_ERROR
    assert SECRET_MARKER not in str(exc_info.value)
    assert VALID_API_KEY not in str(exc_info.value)
    assert executor_calls == []
    assert len(completions.calls) == 1


@pytest.mark.parametrize("request_kind", ["tool_call", "execute", "brief", "final"])
def test_conversion_failures_are_secret_safe_and_stop_before_network_or_executor(
    request_kind,
):
    completions = FakeCompletions(content="unused", tool_calls=[])
    executor_calls = []
    secret_value = SecretConversion()

    with pytest.raises(demo_agent.ToolCallError) as exc_info:
        if request_kind == "tool_call":
            demo_agent.request_tool_call(
                VALID_API_KEY,
                tool_name="read_nvmolkit_skill",
                task_prompt="Read the skill.",
                context={"unsafe": secret_value},
                client=fake_client(completions),
            )
        elif request_kind == "execute":
            demo_agent.request_and_execute_step(
                VALID_API_KEY,
                tool_name="read_nvmolkit_skill",
                task_prompt="Read the skill.",
                context={"unsafe": secret_value},
                executor=lambda arguments: executor_calls.append(arguments),
                client=fake_client(completions),
            )
        elif request_kind == "brief":
            demo_agent.request_brief_interpretation(
                VALID_API_KEY,
                valid_decision(),
                {"unsafe": secret_value},
                {"description": "unused"},
                client=fake_client(completions),
            )
        else:
            demo_agent.request_final_synthesis(
                VALID_API_KEY,
                {"unsafe": secret_value},
                client=fake_client(completions),
            )

    assert str(exc_info.value) == "The scientific result could not be serialized safely."
    assert SECRET_MARKER not in str(exc_info.value)
    assert VALID_API_KEY not in str(exc_info.value)
    assert completions.calls == []
    assert executor_calls == []


@pytest.mark.parametrize("api_key", ["", "ngc-personal-key", f"nvapi-{SECRET_MARKER} "])
def test_bad_hosted_key_is_secret_safe_and_rejected_before_network(api_key):
    completions = FakeCompletions(tool_calls=[])

    with pytest.raises(ValueError) as exc_info:
        demo_agent.request_tool_call(
            api_key,
            tool_name="read_nvmolkit_skill",
            task_prompt="Read the skill.",
            context={},
            client=fake_client(completions),
        )

    assert str(exc_info.value) == AUTH_GUIDANCE
    if api_key:
        assert api_key not in str(exc_info.value)
    assert completions.calls == []


@pytest.mark.parametrize(
    "error",
    [
        response_error(401, AuthenticationError),
        response_error(403, PermissionDeniedError),
    ],
)
def test_hosted_auth_errors_become_secret_safe_guidance(error):
    completions = FakeCompletions(error=error)

    with pytest.raises(ValueError) as exc_info:
        request("read_nvmolkit_skill", completions)

    assert str(exc_info.value) == AUTH_GUIDANCE
    assert SECRET_MARKER not in str(exc_info.value)
    assert VALID_API_KEY not in str(exc_info.value)


def test_brief_interpretation_continues_tool_exchange_with_figure_contract():
    decision = valid_decision()
    completions = FakeCompletions(content="Three clusters indicate bounded library diversity.")
    tool_result = {"method": "Butina", "clusters": 3, "sizes": [12, 8, 4]}
    figure_context = {
        "description": "Bar chart of cluster membership",
        "axes": {"x": "cluster rank", "y": "molecule count"},
        "scale": "linear",
        "salient_values": {"largest_cluster": 12},
    }

    result = demo_agent.request_brief_interpretation(
        VALID_API_KEY,
        decision,
        tool_result,
        figure_context,
        client=fake_client(completions),
    )

    call = completions.calls[0]
    assert call["stream"] is False
    assert call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    messages = call["messages"]
    system_prompt = messages[0]["content"]
    assert "2-4 sentences" in system_prompt
    assert "text-only" in system_prompt
    assert "figure pixels" in system_prompt
    assert all(
        field in system_prompt
        for field in ("description", "axes", "scale", "salient values")
    )
    assert all(
        boundary in system_prompt.lower()
        for boundary in (
            "binding",
            "biological activity",
            "admet",
            "efficacy",
            "safety",
            "synthesizability",
            "clinical relevance",
            "experimentally validated conformations",
        )
    )
    assert messages[-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": decision.tool_call_id,
                "type": "function",
                "function": {
                    "name": decision.tool_name,
                    "arguments": decision.raw_arguments,
                },
            }
        ],
    }
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == decision.tool_call_id
    assert json.loads(messages[-1]["content"]) == {
        "tool_result": tool_result,
        "figure_context": figure_context,
    }
    assert result == "Three clusters indicate bounded library diversity."


def test_final_synthesis_serializes_all_stages_and_enforces_scientific_boundaries():
    analysis_summary = {
        tool_name: {
            "result": {"stage_index": index, "count": 24 + index},
            "figure_context": {"salient_values": {"value": index}},
        }
        for index, tool_name in enumerate(TOOL_NAMES)
    }
    completions = FakeCompletions(content="A bounded 500-word synthesis.")

    result = demo_agent.request_final_synthesis(
        VALID_API_KEY,
        analysis_summary,
        client=fake_client(completions),
    )

    call = completions.calls[0]
    assert call["stream"] is False
    assert call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    system_prompt = call["messages"][0]["content"]
    assert (
        "PhD-level scientific synthesis that remains readable in a presentation"
        in system_prompt
    )
    assert "450-650 words" in system_prompt
    assert all(
        theme in system_prompt.lower()
        for theme in (
            "dataset validity and scope",
            "molecular representation",
            "pairwise similarity structure",
            "clustering and library diversity",
            "conformational sampling and mmff94 convergence",
            "limitations and appropriate next analyses",
        )
    )
    assert "quantitative" in system_prompt.lower()
    assert "figure_context" in system_prompt
    assert all(
        boundary in system_prompt.lower()
        for boundary in (
            "binding",
            "biological activity",
            "admet",
            "efficacy",
            "safety",
            "synthesizability",
            "clinical relevance",
            "experimentally validated conformations",
        )
    )
    assert "within-molecule sampled force-field minima" in system_prompt.lower()
    assert "global or experimental conformations" in system_prompt.lower()
    payload = json.loads(call["messages"][-1]["content"])
    assert payload == analysis_summary
    assert set(payload) == set(TOOL_NAMES)
    assert result == "A bounded 500-word synthesis."


@pytest.mark.parametrize(
    "request_kind",
    ["brief", "final"],
)
@pytest.mark.parametrize(
    ("error", "expected_type", "expected_message"),
    [
        (response_error(401, AuthenticationError), ValueError, AUTH_GUIDANCE),
        (response_error(403, PermissionDeniedError), ValueError, AUTH_GUIDANCE),
        (
            RuntimeError(f"network failed {SECRET_MARKER}"),
            None,
            GENERIC_ERROR,
        ),
    ],
)
def test_narrative_request_errors_are_secret_safe(
    request_kind, error, expected_type, expected_message
):
    completions = FakeCompletions(error=error)
    exception_type = expected_type or demo_agent.ToolCallError

    with pytest.raises(exception_type) as exc_info:
        if request_kind == "brief":
            demo_agent.request_brief_interpretation(
                VALID_API_KEY,
                valid_decision(),
                {"clusters": 3},
                {"description": "cluster bars"},
                client=fake_client(completions),
            )
        else:
            demo_agent.request_final_synthesis(
                VALID_API_KEY,
                {"read_nvmolkit_skill": {"loaded": True}},
                client=fake_client(completions),
            )

    assert str(exc_info.value) == expected_message
    assert SECRET_MARKER not in str(exc_info.value)
    assert VALID_API_KEY not in str(exc_info.value)


def test_preserves_hosted_constants_and_guidance():
    assert demo_agent.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert demo_agent.DEFAULT_MODEL
    assert demo_agent.NEMOTRON_TOOL_EXTRA_BODY == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert demo_agent.AUTH_GUIDANCE == AUTH_GUIDANCE
