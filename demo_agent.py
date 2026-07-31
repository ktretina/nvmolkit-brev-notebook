from dataclasses import dataclass
from typing import Literal

from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
PLAN_SYSTEM_PROMPT = """You plan parameters for one fixed molecular workflow:
Morgan fingerprints, Tanimoto similarity, Butina clustering, ETKDGv3 conformer
generation, and MMFF94 minimization. Return exact JSON containing only these five
keys: fingerprint_radius, fingerprint_size, cluster_cutoff,
representative_count, conformers_per_representative. Do not request code execution
or propose arbitrary code.
Forbid scientific overclaims: computed descriptors or geometries do not establish
biological or clinical outcomes."""


DEFAULT_PLAN = {
    "fingerprint_radius": 2,
    "fingerprint_size": 1024,
    "cluster_cutoff": 0.5,
    "representative_count": 4,
    "conformers_per_representative": 4,
}


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fingerprint_radius: Literal[2, 3] = 2
    fingerprint_size: Literal[1024, 2048] = 1024
    cluster_cutoff: float = Field(default=0.5, ge=0.2, le=0.8)
    representative_count: int = Field(default=4, ge=1, le=6)
    conformers_per_representative: int = Field(default=4, ge=1, le=8)


@dataclass(frozen=True)
class PlanDecision:
    plan: WorkflowPlan
    source: Literal["nemotron", "default_after_error"]
    error: str | None
    raw: str | None


def parse_plan(raw: str) -> WorkflowPlan:
    return WorkflowPlan.model_validate_json(raw)


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def request_plan(
    api_key: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> PlanDecision:
    if not api_key:
        raise ValueError("NVIDIA_API_KEY must not be empty")

    raw = None
    try:
        active_client = client or _client(api_key)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Choose valid parameters for the fixed workflow and return exact JSON only.",
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = response.choices[0].message.content
        return PlanDecision(
            plan=parse_plan(raw), source="nemotron", error=None, raw=raw
        )
    except (
        APIError,
        ValidationError,
        RuntimeError,
        ValueError,
        IndexError,
        AttributeError,
    ) as exc:
        return PlanDecision(
            plan=WorkflowPlan.model_validate(DEFAULT_PLAN),
            source="default_after_error",
            error=str(exc),
            raw=raw,
        )


def request_explanation(
    api_key: str,
    summary: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> str:
    if not api_key:
        raise ValueError("NVIDIA_API_KEY must not be empty")

    active_client = client or _client(api_key)
    response = active_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Explain workflow results accurately and conservatively in 120 words or fewer.",
            },
            {
                "role": "user",
                "content": (
                    f"Explain this summary in no more than 120 words: {summary}\n"
                    "Computed descriptors and geometries are not evidence of binding, "
                    "activity, ADMET, efficacy, safety, or clinical relevance."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=220,
    )
    return response.choices[0].message.content or ""
