from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PLAN = {
    "fingerprint_radius": 2,
    "fingerprint_size": 1024,
    "cluster_cutoff": 0.5,
    "representative_count": 4,
    "conformers_per_representative": 4,
}


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fingerprint_radius: Literal[2, 3]
    fingerprint_size: Literal[1024, 2048]
    cluster_cutoff: float = Field(ge=0.2, le=0.8)
    representative_count: int = Field(ge=1, le=6)
    conformers_per_representative: int = Field(ge=1, le=8)


def parse_plan(raw: str) -> WorkflowPlan:
    return WorkflowPlan.model_validate_json(raw)
