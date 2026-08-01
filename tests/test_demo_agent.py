import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, PermissionDeniedError
from pydantic import ValidationError

from demo_agent import (
    DEFAULT_MODEL,
    DEFAULT_PLAN,
    WorkflowPlan,
    parse_plan,
    request_explanation,
    request_tool_call,
)


EXPECTED_DEFAULT_PLAN = {
    "fingerprint_radius": 2,
    "fingerprint_size": 1024,
    "cluster_cutoff": 0.5,
    "representative_count": 4,
    "conformers_per_representative": 4,
}

VALID_API_KEY = "nvapi-"
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted Developer API key. Generate it from the "
    "Nemotron build.nvidia.com model page, then paste only the bare key; it "
    "starts with nvapi-. An NGC personal key is a different credential and "
    "must not be substituted."
)


def test_accepts_default_plan():
    parsed_plan = parse_plan(json.dumps(DEFAULT_PLAN))

    assert DEFAULT_PLAN == EXPECTED_DEFAULT_PLAN
    assert parsed_plan.model_dump() == EXPECTED_DEFAULT_PLAN
    assert WorkflowPlan() == parsed_plan


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("fingerprint_radius", 4),
        ("fingerprint_size", 4096),
        ("cluster_cutoff", 0.0),
        ("representative_count", 7),
        ("conformers_per_representative", 9),
        ("execute_python", True),
    ],
)
def test_rejects_out_of_contract_plan_fields(field, invalid_value):
    raw_plan = {**EXPECTED_DEFAULT_PLAN, field: invalid_value}

    with pytest.raises(ValidationError):
        parse_plan(json.dumps(raw_plan))


def test_rejects_prose_wrapped_json():
    raw = f"Here is the plan:\n{json.dumps(EXPECTED_DEFAULT_PLAN)}"

    with pytest.raises(ValueError, match="validation"):
        parse_plan(raw)


class FakeCompletions:
    def __init__(self, content=None, tool_calls=None, error=None):
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


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def tool_call(
    arguments,
    name="analyze_molecule_library",
    call_id="call-123",
    call_type="function",
):
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def valid_decision(completions=None):
    raw = json.dumps({**EXPECTED_DEFAULT_PLAN, "fingerprint_radius": 3})
    completions = completions or FakeCompletions(tool_calls=[tool_call(raw)])
    return request_tool_call(VALID_API_KEY, client=fake_client(completions))


def test_request_tool_call_forces_one_strict_nvmolkit_function():
    raw = json.dumps({**EXPECTED_DEFAULT_PLAN, "fingerprint_radius": 3})
    completions = FakeCompletions(tool_calls=[tool_call(raw)])

    decision = request_tool_call(VALID_API_KEY, client=fake_client(completions))

    assert decision.source == "nemotron"
    assert decision.error is None
    assert decision.tool_name == "analyze_molecule_library"
    assert decision.tool_call_id == "call-123"
    assert decision.raw_arguments == raw
    assert decision.plan.fingerprint_radius == 3
    assert "api_key" not in vars(decision)
    call = completions.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 400
    assert call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert call["tool_choice"] == {
        "type": "function",
        "function": {"name": "analyze_molecule_library"},
    }
    assert len(call["tools"]) == 1
    tool = call["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "analyze_molecule_library"
    parameters = tool["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == set(EXPECTED_DEFAULT_PLAN)
    assert set(parameters["required"]) == set(EXPECTED_DEFAULT_PLAN)
    prompt = call["messages"][0]["content"].lower()
    assert "call analyze_molecule_library exactly once" in prompt
    assert "return exact json" not in prompt
    assert "do not request code execution" in prompt
    assert all(
        constraint in prompt
        for constraint in (
            "fingerprint_radius: 2 or 3",
            "fingerprint_size: 1024 or 2048",
            "cluster_cutoff: 0.2 through 0.8",
            "representative_count: 1 through 6",
            "conformers_per_representative: 1 through 8",
        )
    )
    assert all(
        guidance in prompt
        for guidance in (
            "nvmolkit is for gpu-accelerated batched operations",
            "no cpu fallback",
            "rdkit is used for molecule parsing, display",
            "isolated/single-molecule cpu utilities",
            "this demo uses a batch",
            "gpu path makes sense",
        )
    )
    assert "scientific overclaims" in prompt
    assert "outputs do not establish" in prompt
    assert all(
        term in prompt
        for term in (
            "binding",
            "activity",
            "admet",
            "efficacy",
            "safety",
            "synthesizability",
            "clinical relevance",
            "experimentally validated conformations",
        )
    )


def test_request_tool_call_falls_back_for_invalid_arguments():
    invalid_arguments = (
        ("not JSON", "validation"),
        ("{}", "missing required plan fields"),
        (
            json.dumps({"fingerprint_radius": 2}),
            "fingerprint_size",
        ),
    )

    for arguments, expected_error in invalid_arguments:
        completions = FakeCompletions(tool_calls=[tool_call(arguments)])
        decision = request_tool_call(VALID_API_KEY, client=fake_client(completions))

        assert decision.source == "default_after_error"
        assert decision.plan.model_dump() == EXPECTED_DEFAULT_PLAN
        assert expected_error in decision.error.lower()
        assert decision.tool_name == "analyze_molecule_library"
        assert decision.tool_call_id.startswith("default-")
        assert json.loads(decision.raw_arguments) == EXPECTED_DEFAULT_PLAN

    with pytest.raises(ValueError, match="(?i)missing required plan fields"):
        parse_plan("{}")


@pytest.mark.parametrize(
    ("tool_calls", "expected_error"),
    [
        (None, "missing"),
        ([], "missing"),
        ([SimpleNamespace(id="call-123", type="function", function=None)], "malformed"),
        (
            [
                SimpleNamespace(
                    id="call-123",
                    function=SimpleNamespace(
                        name="analyze_molecule_library",
                        arguments=json.dumps(EXPECTED_DEFAULT_PLAN),
                    ),
                )
            ],
            "tool call type",
        ),
        (
            [tool_call(json.dumps(EXPECTED_DEFAULT_PLAN), call_type="custom")],
            "tool call type",
        ),
        ([tool_call(json.dumps(EXPECTED_DEFAULT_PLAN), name="run_python")], "unexpected tool"),
        ([tool_call(json.dumps(EXPECTED_DEFAULT_PLAN)), tool_call(json.dumps(EXPECTED_DEFAULT_PLAN))], "exactly one"),
    ],
)
def test_request_tool_call_falls_back_for_missing_malformed_or_wrong_call(
    tool_calls, expected_error
):
    completions = FakeCompletions(tool_calls=tool_calls)

    decision = request_tool_call(VALID_API_KEY, client=fake_client(completions))

    assert decision.source == "default_after_error"
    assert decision.plan.model_dump() == EXPECTED_DEFAULT_PLAN
    assert expected_error in decision.error.lower()
    assert decision.tool_name == "analyze_molecule_library"
    assert json.loads(decision.raw_arguments) == EXPECTED_DEFAULT_PLAN


def test_request_tool_call_falls_back_when_offline():
    completions = FakeCompletions(error=RuntimeError("offline"))

    decision = request_tool_call(VALID_API_KEY, client=fake_client(completions))

    assert decision.source == "default_after_error"
    assert decision.plan.model_dump() == EXPECTED_DEFAULT_PLAN
    assert "offline" in decision.error


def test_request_tool_call_rejects_empty_api_key():
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        request_tool_call("")


@pytest.mark.parametrize("request_name", ["tool_call", "explanation"])
def test_requests_reject_non_hosted_key_before_network_call(request_name):
    completions = FakeCompletions(content="unused")
    secret = "ngc-secret-marker"

    with pytest.raises(ValueError, match="hosted Developer API key") as exc_info:
        if request_name == "tool_call":
            request_tool_call(secret, client=fake_client(completions))
        else:
            request_explanation(secret, valid_decision(), {}, client=fake_client(completions))

    assert str(exc_info.value) == AUTH_GUIDANCE
    assert secret not in str(exc_info.value)
    assert completions.calls == []


@pytest.mark.parametrize("request_name", ["tool_call", "explanation"])
def test_requests_translate_authentication_error_to_hosted_key_guidance(request_name):
    response = httpx.Response(
        401,
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )
    auth_error = AuthenticationError(
        "Authentication failed",
        response=response,
        body=None,
    )
    completions = FakeCompletions(error=auth_error)

    with pytest.raises(ValueError) as exc_info:
        if request_name == "tool_call":
            request_tool_call(VALID_API_KEY, client=fake_client(completions))
        else:
            request_explanation(
                VALID_API_KEY, valid_decision(), {}, client=fake_client(completions)
            )

    assert str(exc_info.value) == AUTH_GUIDANCE
    assert completions.calls


@pytest.mark.parametrize("request_name", ["tool_call", "explanation"])
def test_requests_translate_permission_denied_to_hosted_key_guidance(request_name):
    response = httpx.Response(
        403,
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )
    permission_error = PermissionDeniedError(
        "Permission denied",
        response=response,
        body=None,
    )
    completions = FakeCompletions(error=permission_error)

    with pytest.raises(ValueError) as exc_info:
        if request_name == "tool_call":
            request_tool_call(VALID_API_KEY, client=fake_client(completions))
        else:
            request_explanation(
                VALID_API_KEY, valid_decision(), {}, client=fake_client(completions)
            )

    assert str(exc_info.value) == AUTH_GUIDANCE
    assert completions.calls


def test_request_explanation_round_trips_only_summary_as_tool_result():
    completions = FakeCompletions(content="A bounded explanation.")
    summary = {"method": "Butina", "clusters": 3}
    decision = valid_decision()

    result = request_explanation(
        VALID_API_KEY, decision, summary, client=fake_client(completions)
    )

    messages = completions.calls[0]["messages"]
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
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
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": decision.tool_call_id,
        "content": json.dumps(summary),
    }
    prompt = messages[0]["content"]
    assert all(
        term in prompt
        for term in (
            "binding",
            "activity",
            "ADMET",
            "efficacy",
            "safety",
            "synthesizability",
            "clinical relevance",
            "experimentally validated conformations",
        )
    )
    assert result == "A bounded explanation."
