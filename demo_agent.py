from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PLAN = {
    "radius": 2,
    "size": 1024,
    "cutoff": 0.5,
    "representative_count": 4,
    "conformers": 4,
}


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    radius: Literal[2, 3]
    size: Literal[1024, 2048]
    cutoff: float = Field(ge=0.2, le=0.8)
    representative_count: int = Field(ge=1, le=6)
    conformers: int = Field(ge=1, le=8)


def parse_plan(raw: str) -> WorkflowPlan:
    return WorkflowPlan.model_validate_json(raw)
