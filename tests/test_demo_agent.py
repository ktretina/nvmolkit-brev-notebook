import json

import pytest
from pydantic import ValidationError

from demo_agent import DEFAULT_PLAN, parse_plan


def test_accepts_default_plan():
    assert parse_plan(json.dumps(DEFAULT_PLAN)).model_dump() == DEFAULT_PLAN


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("radius", 4),
        ("size", 4096),
        ("cutoff", 0.0),
        ("representative_count", 7),
        ("conformers", 9),
        ("execute_python", True),
    ],
)
def test_rejects_out_of_contract_plan_fields(field, invalid_value):
    raw_plan = {**DEFAULT_PLAN, field: invalid_value}

    with pytest.raises(ValidationError):
        parse_plan(json.dumps(raw_plan))


def test_rejects_prose_wrapped_json():
    raw = f"Here is the plan:\n{json.dumps(DEFAULT_PLAN)}"

    with pytest.raises(ValidationError):
        parse_plan(raw)
