import json
from dataclasses import dataclass
from typing import Literal

from openai import APIError, AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NEMOTRON_TOOL_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False}
}
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted Developer API key. Generate it from the "
    "Nemotron build.nvidia.com model page, then paste only the bare key; it "
    "starts with nvapi-. An NGC personal key is a different credential and "
    "must not be substituted."
)
PLAN_SYSTEM_PROMPT = """You plan parameters for one fixed molecular workflow:
Morgan fingerprints, Tanimoto similarity, Butina clustering, ETKDGv3 conformer
generation, and MMFF94 minimization. Call analyze_molecule_library exactly once
with only these five arguments: fingerprint_radius, fingerprint_size, cluster_cutoff,
representative_count, conformers_per_representative. Do not request code execution
or propose arbitrary code. Allowed values are fingerprint_radius: 2 or 3;
fingerprint_size: 1024 or 2048; cluster_cutoff: 0.2 through 0.8;
representative_count: 1 through 6; conformers_per_representative: 1 through 8.
nvMolKit is for GPU-accelerated batched operations and has no CPU fallback.
RDKit is used for molecule parsing, display, and isolated/single-molecule CPU utilities.
This demo uses a batch, so the GPU path makes sense.
Forbid scientific overclaims: outputs do not establish binding, activity, ADMET,
efficacy, safety, synthesizability, or clinical relevance. Outputs also do not
establish experimentally validated conformations."""


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


REQUIRED_PLAN_KEYS = frozenset(WorkflowPlan.model_fields)
TOOL_NAME = "analyze_molecule_library"
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Run the fixed nvMolKit molecular-library analysis workflow.",
        "strict": True,
        "parameters": {
            **WorkflowPlan.model_json_schema(),
            "required": sorted(REQUIRED_PLAN_KEYS),
        },
    },
}


@dataclass(frozen=True)
class PlanDecision:
    plan: WorkflowPlan
    source: Literal["nemotron", "default_after_error"]
    error: str | None
    tool_name: str
    tool_call_id: str
    raw_arguments: str


def parse_plan(raw: str) -> WorkflowPlan:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plan JSON validation failed: {exc}") from exc

    if not isinstance(decoded, dict):
        raise ValueError("Plan JSON must be a top-level object")

    missing_fields = REQUIRED_PLAN_KEYS.difference(decoded)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing required plan fields: {missing}")

    return WorkflowPlan.model_validate_json(raw)


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def _validate_api_key(api_key: str) -> None:
    if not api_key or not api_key.startswith("nvapi-"):
        raise ValueError(AUTH_GUIDANCE)


def _default_after_error(exc: Exception) -> PlanDecision:
    return PlanDecision(
        plan=WorkflowPlan.model_validate(DEFAULT_PLAN),
        source="default_after_error",
        error=str(exc),
        tool_name=TOOL_NAME,
        tool_call_id=f"default-{TOOL_NAME}",
        raw_arguments=json.dumps(DEFAULT_PLAN),
    )


def request_tool_call(
    api_key: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> PlanDecision:
    _validate_api_key(api_key)

    try:
        active_client = client or _client(api_key)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Choose valid parameters and request the fixed "
                        "analyze_molecule_library tool."
                    ),
                },
            ],
            tools=[TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=400,
        )
    except (AuthenticationError, PermissionDeniedError):
        raise ValueError(AUTH_GUIDANCE) from None
    except (APIError, RuntimeError) as exc:
        return _default_after_error(exc)

    try:
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("Missing required tool call")
        if len(tool_calls) != 1:
            raise ValueError("Expected exactly one tool call")
        tool_call = tool_calls[0]
        if getattr(tool_call, "type", None) != "function":
            raise ValueError("Invalid or missing tool call type; expected function")
        if tool_call.function is None:
            raise ValueError("Malformed tool call")
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id
        raw_arguments = tool_call.function.arguments
        if not all(isinstance(value, str) and value for value in (tool_call_id, raw_arguments)):
            raise ValueError("Malformed tool call")
        if tool_name != TOOL_NAME:
            raise ValueError(f"Unexpected tool requested: {tool_name}")
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        return _default_after_error(exc)

    try:
        plan = parse_plan(raw_arguments)
    except (ValidationError, ValueError) as exc:
        return _default_after_error(exc)

    return PlanDecision(
        plan=plan,
        source="nemotron",
        error=None,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        raw_arguments=raw_arguments,
    )


def request_explanation(
    api_key: str,
    decision: PlanDecision,
    summary: dict[str, int | float | str],
    model: str = DEFAULT_MODEL,
    client=None,
) -> str:
    _validate_api_key(api_key)

    serialized_summary = json.dumps(summary)
    active_client = client or _client(api_key)
    try:
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Explain the completed workflow results accurately and conservatively in "
                        "120 words or fewer. Computed descriptors and geometries are not evidence "
                        "of binding, activity, ADMET, efficacy, safety, synthesizability, or "
                        "clinical relevance, and they are not experimentally validated conformations."
                    ),
                },
                {
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
                },
                {
                    "role": "tool",
                    "tool_call_id": decision.tool_call_id,
                    "content": serialized_summary,
                },
            ],
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=220,
        )
    except (AuthenticationError, PermissionDeniedError):
        raise ValueError(AUTH_GUIDANCE) from None
    return response.choices[0].message.content or ""
