import json

import pytest
from pydantic import ValidationError

from demo_agent import DEFAULT_PLAN, WorkflowPlan, parse_plan


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
