import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from demo_agent import (
    DEFAULT_PLAN,
    WorkflowPlan,
    parse_plan,
    request_explanation,
    request_plan,
)


EXPECTED_DEFAULT_PLAN = {
    "fingerprint_radius": 2,
    "fingerprint_size": 1024,
    "cluster_cutoff": 0.5,
    "representative_count": 4,
    "conformers_per_representative": 4,
}


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

    with pytest.raises(ValidationError):
        parse_plan(raw)


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_request_plan_accepts_valid_nemotron_json():
    raw = json.dumps({**EXPECTED_DEFAULT_PLAN, "fingerprint_radius": 3})
    completions = FakeCompletions(content=raw)

    decision = request_plan("test-key", client=fake_client(completions))

    assert decision.source == "nemotron"
    assert decision.error is None
    assert decision.raw == raw
    assert decision.plan.fingerprint_radius == 3
    call = completions.calls[0]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 400
    assert "arbitrary code" in call["messages"][0]["content"]
    assert "scientific overclaims" in call["messages"][0]["content"]


def test_request_plan_falls_back_for_invalid_json():
    completions = FakeCompletions(content="not JSON")

    decision = request_plan("test-key", client=fake_client(completions))

    assert decision.source == "default_after_error"
    assert decision.plan.model_dump() == EXPECTED_DEFAULT_PLAN
    assert "validation" in decision.error.lower()
    assert decision.raw == "not JSON"


def test_request_plan_falls_back_when_offline():
    completions = FakeCompletions(error=RuntimeError("offline"))

    decision = request_plan("test-key", client=fake_client(completions))

    assert decision.source == "default_after_error"
    assert decision.plan.model_dump() == EXPECTED_DEFAULT_PLAN
    assert "offline" in decision.error


def test_request_plan_rejects_empty_api_key():
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        request_plan("")


def test_request_explanation_states_claim_boundary_and_returns_content():
    completions = FakeCompletions(content="A bounded explanation.")

    result = request_explanation(
        "test-key", "Computed 3 clusters.", client=fake_client(completions)
    )

    prompt = completions.calls[0]["messages"][-1]["content"]
    assert "not evidence of binding" in prompt
    assert result == "A bounded explanation."
