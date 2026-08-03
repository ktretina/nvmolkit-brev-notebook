"""One bounded Nemotron conversation over the nvMolKit chemistry workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from chemistry_workflow import (
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    WorkflowState,
    build_workflow_report,
    discover_fused_butina_clusters,
    eligible_stage,
    embed_representative_conformers,
    generate_morgan_fingerprints,
    inspect_library,
    measure_tanimoto_similarity,
    optimize_conformers_mmff94,
)


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NEMOTRON_TOOL_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted Developer API key. Generate it from the "
    "Nemotron build.nvidia.com model page, then paste only the bare key; it "
    "starts with nvapi-. An NGC personal key is a different credential and "
    "must not be substituted."
)
_REQUEST_ERROR = (
    "The hosted Nemotron request failed. Check network access and model availability."
)
_SERIALIZATION_ERROR = "The scientific result could not be serialized safely."
_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "nvmolkit" / "SKILL.md"
_DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_molecules.csv"

STAGES = (
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
)
POST_STAGE_PHASES = dict(
    zip(
        STAGES,
        (
            WorkflowPhase.INSPECTED,
            WorkflowPhase.FINGERPRINTED,
            WorkflowPhase.COMPARED,
            WorkflowPhase.CLUSTERED,
            WorkflowPhase.EMBEDDED,
            WorkflowPhase.OPTIMIZED,
        ),
        strict=True,
    )
)
StageName = Literal[
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
]
DecisionBasis = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=12,
        max_length=240,
        pattern=r"^[^\r\n`]+$",
    ),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class InspectionArgs(_StrictModel):
    pass


class FingerprintArgs(_StrictModel):
    radius: Literal[2, 3]
    size: Literal[1024, 2048]
    decision_basis: DecisionBasis


class SimilarityArgs(_StrictModel):
    pass


class ClusterArgs(_StrictModel):
    cutoff: float = Field(ge=0.40, le=0.60)
    decision_basis: DecisionBasis


class EmbedArgs(_StrictModel):
    representative_count: int = Field(ge=3, le=6)
    policy: Literal[
        "largest_clusters_first", "include_singleton_if_available"
    ]
    conformers_per_representative: int = Field(ge=3, le=8)
    decision_basis: DecisionBasis


class OptimizationArgs(_StrictModel):
    pass


class PlanStage(_StrictModel):
    stage: StageName
    rationale: DecisionBasis


class WorkflowPlan(_StrictModel):
    stages: list[PlanStage] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def dependency_order_is_exact(self) -> "WorkflowPlan":
        if tuple(item.stage for item in self.stages) != STAGES:
            raise ValueError("The workflow plan must use the exact dependency order.")
        return self


TOOL_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "inspect_library": InspectionArgs,
    "generate_morgan_fingerprints": FingerprintArgs,
    "measure_tanimoto_similarity": SimilarityArgs,
    "discover_fused_butina_clusters": ClusterArgs,
    "embed_representative_conformers": EmbedArgs,
    "optimize_conformers_mmff94": OptimizationArgs,
}
TOOL_DESCRIPTIONS = {
    "submit_workflow_plan": "Submit the six-stage scientific plan in dependency order.",
    "inspect_library": "Validate the fixed molecular library with RDKit and report invalid SMILES.",
    "generate_morgan_fingerprints": "Choose bounded parameters and run nvMolKit Morgan fingerprints on the GPU.",
    "measure_tanimoto_similarity": "Run nvMolKit all-pairs Tanimoto similarity on the GPU.",
    "discover_fused_butina_clusters": "Choose a bounded cutoff and run nvMolKit fused Butina clustering on the GPU.",
    "embed_representative_conformers": "Choose bounded sampling parameters and run nvMolKit conformer embedding on the GPU.",
    "optimize_conformers_mmff94": "Run nvMolKit MMFF94 conformer optimization on the GPU.",
}


class ToolCallError(RuntimeError):
    """A secret-safe failure in the bounded hosted or scientific loop."""


@dataclass
class AgentSession:
    messages: list[dict[str, Any]]
    state: WorkflowState
    turn_count: int = 0

    def eligible_tool_name(self) -> str:
        return eligible_stage(self.state)


@dataclass(frozen=True)
class ScientificLoopResult:
    messages: tuple[dict[str, Any], ...]
    report: WorkflowReport
    turn_count: int


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, max_retries=0)


def _validate_api_key(api_key: str) -> None:
    if (
        not isinstance(api_key, str)
        or not api_key.startswith("nvapi-")
        or api_key != api_key.strip()
    ):
        raise ValueError(AUTH_GUIDANCE)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise ToolCallError(_SERIALIZATION_ERROR)


def _serialize(value: Any) -> str:
    try:
        return json.dumps(
            _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except Exception:
        raise ToolCallError(_SERIALIZATION_ERROR) from None


def _tool_definition(name: str, model: type[BaseModel]) -> dict[str, Any]:
    parameters = model.model_json_schema()
    parameters["additionalProperties"] = False
    parameters["required"] = list(model.model_fields)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "strict": True,
            "parameters": parameters,
        },
    }


def _raise_request_error(error: Exception) -> None:
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        raise ValueError(AUTH_GUIDANCE) from None
    raise ToolCallError(_REQUEST_ERROR) from None


def _request_call(
    session: AgentSession,
    client: Any,
    expected_name: str,
    argument_model: type[BaseModel],
    model: str,
) -> BaseModel:
    """Request, validate, and append exactly one forced hosted call."""
    if session.turn_count >= 7:
        raise ToolCallError("The bounded hosted turn limit was reached.")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=session.messages,
            tools=[_tool_definition(expected_name, argument_model)],
            tool_choice={
                "type": "function",
                "function": {"name": expected_name},
            },
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=900 if expected_name == "submit_workflow_plan" else 400,
            stream=False,
        )
    except Exception as error:
        _raise_request_error(error)

    try:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        calls = getattr(message, "tool_calls", None)
        if isinstance(content, str) and content.strip():
            raise ToolCallError("Hosted text was returned before the required tool call.")
        if not isinstance(calls, (list, tuple)) or len(calls) != 1:
            raise ToolCallError("Expected exactly one hosted tool call.")
        call = calls[0]
        function = getattr(call, "function", None)
        call_id = getattr(call, "id", None)
        if getattr(call, "type", None) != "function" or function is None:
            raise ToolCallError("The hosted tool call was malformed.")
        if getattr(function, "name", None) != expected_name:
            raise ToolCallError("The hosted tool call was out of phase.")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ToolCallError("The hosted tool call ID was missing.")
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            raise ToolCallError("The hosted tool arguments were missing.")
        decoded = json.loads(raw_arguments)
        if not isinstance(decoded, dict):
            raise ToolCallError("The hosted tool arguments must be a JSON object.")
        arguments = argument_model.model_validate(decoded)
    except ToolCallError:
        raise
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError, ValidationError):
        raise ToolCallError("The hosted tool call failed strict validation.") from None

    session.messages.append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": expected_name,
                        "arguments": raw_arguments,
                    },
                }
            ],
        }
    )
    session.turn_count += 1
    return arguments


def _append_tool_result(session: AgentSession, content: Any) -> None:
    assistant = session.messages[-1]
    session.messages.append(
        {
            "role": "tool",
            "tool_call_id": assistant["tool_calls"][0]["id"],
            "content": _serialize(content),
        }
    )


def _system_grounding() -> str:
    skill = _SKILL_PATH.read_text(encoding="utf-8")
    return f"""You are a bounded chemistry workflow agent. The exact vendored
BioNeMo Agent Toolkit skill snapshot below is grounding, not a callable tool.
Grounding provenance: skills/nvmolkit/SKILL.md

Follow this dependency order: inspect, fingerprint, compare, cluster, embed,
then optimize. RDKit input validation, MMFF94 eligibility, and
representative selection; each named nvMolKit GPU operation executes the batched
science. Choose only parameters in the supplied schema. Provide concise decision summaries,
not hidden chain-of-thought. These computational descriptors
and sampled force-field geometries do not establish binding, activity, ADMET,
efficacy, safety, synthesizability, clinical relevance, or experimental truth.

--- exact skills/nvmolkit/SKILL.md content ---
{skill}"""


def _default_executors() -> dict[str, Any]:
    return {
        "inspect_library": lambda state: inspect_library(state, _DATA_PATH),
        "generate_morgan_fingerprints": lambda state, **args: generate_morgan_fingerprints(state, **args),
        "measure_tanimoto_similarity": lambda state: measure_tanimoto_similarity(state),
        "discover_fused_butina_clusters": lambda state, **args: discover_fused_butina_clusters(state, **args),
        "embed_representative_conformers": lambda state, **args: embed_representative_conformers(state, **args),
        "optimize_conformers_mmff94": lambda state: optimize_conformers_mmff94(state),
        "build_workflow_report": build_workflow_report,
    }


def _executor_arguments(name: str, arguments: BaseModel) -> dict[str, Any]:
    if name == "generate_morgan_fingerprints":
        return {
            "fingerprint_radius": arguments.radius,
            "fingerprint_size": arguments.size,
        }
    if name == "discover_fused_butina_clusters":
        return {"cluster_cutoff": arguments.cutoff}
    if name == "embed_representative_conformers":
        return {
            "representative_count": arguments.representative_count,
            "representative_policy": arguments.policy,
            "conformers_per_representative": arguments.conformers_per_representative,
        }
    return {}


def run_scientific_loop(
    user_goal: str,
    api_key: str,
    *,
    client: Any = None,
    executors: dict[str, Any] | None = None,
    state: WorkflowState | None = None,
) -> ScientificLoopResult:
    """Run one plan and six phase-gated scientific tool calls in one history."""
    _validate_api_key(api_key)
    if not isinstance(user_goal, str) or not user_goal.strip():
        raise ValueError("A non-empty scientific goal is required.")
    active_client = client or _client(api_key)
    active_executors = _default_executors() if executors is None else executors
    allowed_executor_keys = set(STAGES) | {"build_workflow_report"}
    if set(active_executors) != allowed_executor_keys or not all(
        callable(active_executors[key]) for key in allowed_executor_keys
    ):
        raise ValueError("Executors must match the fixed scientific workflow.")

    session = AgentSession(
        messages=[
            {"role": "system", "content": _system_grounding()},
            {"role": "user", "content": user_goal.strip()},
        ],
        state=state or WorkflowState(),
    )
    plan = _request_call(
        session, active_client, "submit_workflow_plan", WorkflowPlan, DEFAULT_MODEL
    )
    _append_tool_result(
        session,
        {
            "accepted": True,
            "stages": [item.model_dump(mode="json") for item in plan.stages],
        },
    )

    for stage in STAGES:
        if session.eligible_tool_name() != stage:
            raise ToolCallError("The scientific workflow state is out of phase.")
        arguments = _request_call(
            session, active_client, stage, TOOL_ARGUMENT_MODELS[stage], DEFAULT_MODEL
        )
        try:
            result = active_executors[stage](
                session.state, **_executor_arguments(stage, arguments)
            )
        except Exception:
            raise ToolCallError("The scientific executor failed.") from None
        if not isinstance(result, StageResult) or result.stage != stage:
            raise ToolCallError("The scientific executor returned an invalid stage result.")
        if session.state.phase is not POST_STAGE_PHASES[stage]:
            raise ToolCallError("The scientific executor left the workflow out of phase.")
        _append_tool_result(
            session,
            {
                "stage": stage,
                "decision_basis": getattr(arguments, "decision_basis", None),
                "summary": result.summary,
            },
        )

    if session.state.phase is not WorkflowPhase.OPTIMIZED or session.turn_count != 7:
        raise ToolCallError("The scientific workflow did not complete exactly once.")
    try:
        report = active_executors["build_workflow_report"](session.state)
    except Exception:
        raise ToolCallError("The scientific report could not be built.") from None
    if not isinstance(report, WorkflowReport):
        raise ToolCallError("The scientific report was invalid.")
    return ScientificLoopResult(tuple(session.messages), report, session.turn_count)
