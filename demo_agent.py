"""One bounded nvMolKit workflow with a grounded, schema-checked Nemotron conclusion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from chemistry_workflow import (
    EvidenceRecord,
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
from objective_challenge import (
    MAX_ATTEMPTS,
    ObjectiveAttempt,
    ObjectiveContext,
    ObjectiveRun,
    build_objective_context,
    build_objective_evidence,
    evaluate_diverse_panel,
    finalize_objective_run,
    no_improvement_run,
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
_SYNTHESIS_PROMPT = (
    "Produce one detailed PhD-level but presentation-readable integrated interpretation. Collectively cite every supplied evidence key in the structured evidence_keys fields, but do not mention evidence IDs or add evidence-citation labels in the headline or prose. "
    "Do not infer binding, activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimental truth. State that force-field energies compare conformers only within each molecule. "
    "Three-dimensional methods use ETKDGv3 and MMFF94."
)
_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "nvmolkit" / "SKILL.md"
_DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_molecules.csv"

_NVMOLKIT_CAPABILITIES = (
    ("nvmolkit.fingerprints", "MorganFingerprintGenerator"),
    ("nvmolkit.similarity", "crossTanimotoSimilarity"),
    ("nvmolkit.clustering", "fused_butina"),
    ("nvmolkit.embedMolecules", "EmbedMolecules"),
    ("nvmolkit.mmffOptimization", "MMFFOptimizeMoleculesConfs"),
)


def notebook_preflight() -> str:
    """Validate the supported GPU runtime and return a hidden hosted API key."""
    import getpass
    import os
    import sys

    assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12), (
        "This notebook requires CPython 3.12."
    )
    import torch

    assert torch.cuda.is_available(), "This notebook requires an NVIDIA CUDA GPU."
    for module_name, entry_point in _NVMOLKIT_CAPABILITIES:
        module = __import__(module_name, fromlist=(entry_point,))
        assert hasattr(module, entry_point), f"Missing nvMolKit capability: {entry_point}"

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass(
            "Hosted NVIDIA Developer API key (nvapi-; input hidden): "
        ).strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is required.")
    return api_key

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
        min_length=1,
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


MoleculeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[^\s\r\n`]+$",
    ),
]


class ObjectiveProposal(_StrictModel):
    selected_ids: list[MoleculeId] = Field(min_length=4, max_length=4)
    decision_basis: DecisionBasis

    @field_validator("selected_ids")
    @classmethod
    def selected_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Objective proposal molecule IDs must be unique.")
        return value


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


ConclusionTheme = Literal[
    "dataset_scope", "molecular_representation", "similarity_structure",
    "clustering", "conformational_sampling", "limitations_and_next_steps",
]
EvidenceKey = Literal["E01", "E02", "E03", "E04", "E05", "E06"]


class ConclusionSection(_StrictModel):
    theme: ConclusionTheme
    prose: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
    evidence_keys: list[EvidenceKey] = Field(min_length=1)


class SubmitSynthesisArgs(_StrictModel):
    headline: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    sections: list[ConclusionSection] = Field(min_length=6, max_length=6)


ObjectiveConclusionTheme = Literal[
    "dataset_scope",
    "molecular_representation",
    "similarity_structure",
    "clustering",
    "conformational_sampling",
    "objective_driven_selection",
    "limitations_and_next_steps",
]
ObjectiveEvidenceKey = Literal["E01", "E02", "E03", "E04", "E05", "E06", "O01"]


class ObjectiveConclusionSection(_StrictModel):
    theme: ObjectiveConclusionTheme
    prose: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)
    ]
    evidence_keys: list[ObjectiveEvidenceKey] = Field(min_length=1)


class ObjectiveSubmitConclusionArgs(_StrictModel):
    headline: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
    ]
    sections: list[ObjectiveConclusionSection] = Field(min_length=7, max_length=7)


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
    "select_diverse_panel": "Select exactly four supplied candidate IDs to reach the maximin Tanimoto-distance objective.",
    "submit_synthesis": "Submit one grounded qualitative synthesis; keep evidence IDs only in evidence_keys fields.",
}


class ToolCallError(RuntimeError):
    """A secret-safe failure in the bounded hosted or scientific loop."""


class _HostedArgumentsValidationError(ToolCallError):
    def __init__(self, stage: str, issues: tuple[tuple[str, str], ...]):
        self.stage = stage
        self.issues = issues
        signature = ", ".join(
            f"{field}:{error_type}" for field, error_type in issues
        )
        super().__init__(f"{stage} arguments failed validation: {signature}")


class ConclusionValidationError(ToolCallError):
    def __init__(self, report: WorkflowReport):
        super().__init__("Nemotron synthesis failed validation; structured results are preserved.")
        self.report = report
        self.rejected_prose = None


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
    plan: WorkflowPlan
    stage_results: tuple[StageResult, ...]
    turn_count: int


@dataclass(frozen=True)
class WorkflowResult:
    messages: tuple[dict[str, Any], ...]
    report: WorkflowReport
    plan: WorkflowPlan
    conclusion: SubmitSynthesisArgs | ObjectiveSubmitConclusionArgs
    stage_results: tuple[StageResult, ...]
    turn_count: int = 8
    objective_run: ObjectiveRun | None = None
    objective_evidence: EvidenceRecord | None = None


@dataclass(frozen=True)
class StageProposal:
    stage: StageName
    arguments: BaseModel


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


_REQUIRED_CONCLUSION_EVIDENCE = {
    "dataset_scope": {"E01"},
    "molecular_representation": {"E02"},
    "similarity_structure": {"E03"},
    "clustering": {"E04"},
    "conformational_sampling": {"E05", "E06"},
    "limitations_and_next_steps": {"E01", "E06"},
}

_REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE = {
    **_REQUIRED_CONCLUSION_EVIDENCE,
    "objective_driven_selection": {"O01"},
    "limitations_and_next_steps": {"E01", "E06", "O01"},
}


def validate_conclusion(conclusion: SubmitSynthesisArgs, report: WorkflowReport) -> SubmitSynthesisArgs:
    """Check the synthesis schema and evidence links, not the truth of qualitative prose."""
    themes = [section.theme for section in conclusion.sections]
    cited = {key for section in conclusion.sections for key in section.evidence_keys}
    report_keys = tuple(record.key for record in report.evidence)
    known = set(report_keys)
    valid = set(themes) == set(_REQUIRED_CONCLUSION_EVIDENCE) and len(themes) == len(set(themes))
    valid &= report_keys == EvidenceKey.__args__ and cited == known
    if not valid:
        raise ConclusionValidationError(report)
    return conclusion


def validate_objective_conclusion(
    conclusion: ObjectiveSubmitConclusionArgs,
    report: WorkflowReport,
    objective_evidence: EvidenceRecord,
) -> ObjectiveSubmitConclusionArgs:
    """Validate exact E01-E06 plus O01 coverage for the objective-aware conclusion."""
    themes = [section.theme for section in conclusion.sections]
    cited = {key for section in conclusion.sections for key in section.evidence_keys}
    report_keys = tuple(record.key for record in report.evidence)
    known = set(report_keys) | {objective_evidence.key}
    valid = (
        set(themes) == set(_REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE)
        and len(themes) == len(set(themes))
        and report_keys == EvidenceKey.__args__
        and objective_evidence.key == "O01"
        and cited == known
    )
    valid &= all(
        _REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE[section.theme].issubset(
            set(section.evidence_keys)
        )
        for section in conclusion.sections
    )
    if not valid:
        raise ConclusionValidationError(report)
    return conclusion


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
    *,
    _text_only_retries_remaining: int = 2,
    _max_turns: int = 8,
) -> BaseModel:
    """Request, validate, and append exactly one forced hosted call."""
    if session.turn_count >= _max_turns or (
        _max_turns == 8
        and session.turn_count == 7
        and expected_name != "submit_synthesis"
    ):
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
            temperature=0.0,
            max_tokens={"submit_synthesis": 1800, "submit_workflow_plan": 900}.get(expected_name, 400),
            stream=False,
        )
    except Exception as error:
        _raise_request_error(error)

    try:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        calls = getattr(message, "tool_calls", None)
        if content is not None and not isinstance(content, str):
            raise ToolCallError(
                "Hosted assistant content was returned before the required tool call."
            )
        no_calls = calls is None or (
            isinstance(calls, (list, tuple)) and not calls
        )
        content_arguments = None
        if no_calls and isinstance(content, str):
            try:
                candidate, _end = json.JSONDecoder().raw_decode(content.lstrip())
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict):
                content_arguments = candidate
        content = None
        if (
            no_calls
            and content_arguments is None
            and _text_only_retries_remaining
        ):
            return _request_call(
                session,
                client,
                expected_name,
                argument_model,
                model,
                _text_only_retries_remaining=_text_only_retries_remaining - 1,
                _max_turns=_max_turns,
            )
        if content_arguments is not None:
            call_id = f"compat-{session.turn_count + 1}-{expected_name}"
            decoded = content_arguments
            raw_arguments = _serialize(decoded)
        else:
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
        declared_fields = argument_model.model_fields
        if (
            content_arguments is not None
            and "stage" in decoded
            and "stage" not in declared_fields
        ):
            if decoded["stage"] != expected_name:
                raise ToolCallError("The hosted tool call was out of phase.")
        if declared_fields and any(key not in declared_fields for key in decoded):
            decoded = {
                key: decoded[key] for key in declared_fields if key in decoded
            }
            raw_arguments = _serialize(decoded)
        cutoff = decoded.get("cutoff")
        if expected_name == "discover_fused_butina_clusters" and isinstance(
            cutoff, str
        ):
            compact_cutoff = cutoff.strip()
            if (
                compact_cutoff
                and compact_cutoff.count(".") <= 1
                and compact_cutoff.replace(".", "").isdigit()
            ):
                decoded = {**decoded, "cutoff": float(compact_cutoff)}
                raw_arguments = _serialize(decoded)
        decision_basis = decoded.get("decision_basis")
        if isinstance(decision_basis, str):
            compact_basis = " ".join(decision_basis.replace("`", "").split())
            if len(compact_basis) > 240:
                compact_basis = compact_basis[:237].rstrip() + "..."
            if compact_basis != decision_basis:
                decoded = {**decoded, "decision_basis": compact_basis}
                raw_arguments = _serialize(decoded)
        arguments = argument_model.model_validate(decoded)
    except ToolCallError:
        raise
    except ValidationError as error:
        issues = tuple(
            (
                ".".join(str(part) for part in item["loc"]),
                item["type"],
            )
            for item in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise _HostedArgumentsValidationError(
            expected_name,
            issues,
        ) from None
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
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


def _system_grounding(skill: str | None = None) -> str:
    bundled_skill = _SKILL_PATH.read_text(encoding="utf-8")
    if skill is None:
        skill = bundled_skill
    if skill != bundled_skill:
        raise ValueError("The agent requires the exact bundled nvMolKit skill.")
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


ProgressCallback = Callable[[str, Any], None]


def _emit_progress(callback: ProgressCallback | None, event: str, payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except ToolCallError:
        raise
    except Exception:
        raise ToolCallError("Local progress display failed.") from None


@dataclass
class BoundedWorkflowController:
    """Expose one bounded hosted proposal and deterministic execution at a time."""

    session: AgentSession
    client: Any
    executors: dict[str, Any]
    plan: WorkflowPlan | None = None
    pending: StageProposal | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    report: WorkflowReport | None = None
    synthesis_prompt_appended: bool = False
    objective_required: bool = False
    objective_context: ObjectiveContext | None = None
    pending_objective: ObjectiveProposal | None = None
    objective_attempts: list[ObjectiveAttempt] = field(default_factory=list)
    objective_run: ObjectiveRun | None = None
    objective_evidence: EvidenceRecord | None = None
    objective_prompt_appended: bool = False

    @classmethod
    def create(
        cls,
        user_goal: str,
        api_key: str,
        *,
        client: Any = None,
        executors: dict[str, Any] | None = None,
        state: WorkflowState | None = None,
        skill: str | None = None,
        objective_required: bool = False,
    ) -> "BoundedWorkflowController":
        """Validate and initialize the fixed workflow without making a hosted request."""
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
                {"role": "system", "content": _system_grounding(skill)},
                {"role": "user", "content": user_goal.strip()},
            ],
            state=state or WorkflowState(),
        )
        return cls(
            session=session,
            client=active_client,
            executors=active_executors,
            objective_required=objective_required,
        )

    def request_plan(self) -> WorkflowPlan:
        if self.plan is not None or self.session.turn_count != 0:
            raise ToolCallError("The workflow plan can be requested exactly once.")
        plan = _request_call(
            self.session,
            self.client,
            "submit_workflow_plan",
            WorkflowPlan,
            DEFAULT_MODEL,
        )
        self.plan = WorkflowPlan.model_validate(plan.model_dump())
        _append_tool_result(
            self.session,
            {
                "accepted": True,
                "stages": [
                    item.model_dump(mode="json") for item in self.plan.stages
                ],
            },
        )
        return self.plan

    def request_next_stage(self) -> StageProposal:
        if self.plan is None:
            raise ToolCallError("A workflow plan is required before requesting a stage.")
        if self.pending is not None:
            raise ToolCallError("A stage proposal is already pending approval.")
        if self.session.state.phase is WorkflowPhase.OPTIMIZED:
            raise ToolCallError("The scientific stages are already complete.")
        stage = self.session.eligible_tool_name()
        if stage not in TOOL_ARGUMENT_MODELS:
            raise ToolCallError("The scientific workflow state is out of phase.")
        arguments = _request_call(
            self.session,
            self.client,
            stage,
            TOOL_ARGUMENT_MODELS[stage],
            DEFAULT_MODEL,
        )
        proposal = StageProposal(stage, arguments)
        self.pending = proposal
        return proposal

    def execute_pending(self, approved: BaseModel) -> StageResult:
        proposal = self.pending
        if proposal is None:
            raise ToolCallError("No stage proposal is pending approval.")
        stage = proposal.stage
        argument_model = TOOL_ARGUMENT_MODELS[stage]
        if type(approved) is not argument_model:
            raise ToolCallError(
                "The approved arguments must use the exact model for the pending stage."
            )
        if self.session.eligible_tool_name() != stage:
            raise ToolCallError("The scientific workflow state is out of phase.")
        try:
            executed = argument_model.model_validate(approved.model_dump())
        except ValidationError:
            raise ToolCallError("Approved stage arguments failed strict validation.") from None
        if (
            "decision_basis" in argument_model.model_fields
            and executed.decision_basis != proposal.arguments.decision_basis
        ):
            raise ToolCallError("The Nemotron decision summary cannot be changed.")
        proposed_arguments = proposal.arguments.model_dump(mode="json")
        executed_arguments = executed.model_dump(mode="json")
        proposed_executor_args = _executor_arguments(stage, proposal.arguments)
        executed_executor_args = _executor_arguments(stage, executed)
        try:
            result = self.executors[stage](
                self.session.state, **executed_executor_args
            )
        except Exception:
            raise ToolCallError("The scientific executor failed.") from None
        if not isinstance(result, StageResult) or result.stage != stage:
            raise ToolCallError("The scientific executor returned an invalid stage result.")
        if self.session.state.phase is not POST_STAGE_PHASES[stage]:
            raise ToolCallError("The scientific executor left the workflow out of phase.")
        _append_tool_result(
            self.session,
            {
                "stage": stage,
                "decision_basis": getattr(executed, "decision_basis", None),
                "proposed_arguments": proposed_arguments,
                "executed_arguments": executed_arguments,
                "user_override": proposed_executor_args != executed_executor_args,
                "summary": result.summary,
            },
        )
        self.stage_results.append(result)
        self.pending = None
        return result

    def scientific_result(self) -> ScientificLoopResult:
        if self.plan is None:
            raise ToolCallError("The scientific workflow did not complete exactly once.")
        if self.pending is not None:
            raise ToolCallError("A stage proposal is still pending approval.")
        turn_count_is_valid = (
            self.session.turn_count == 7
            if self.objective_context is None
            else 7 <= self.session.turn_count <= 10
        )
        if (
            tuple(result.stage for result in self.stage_results) != STAGES
            or not turn_count_is_valid
            or self.session.state.phase is not WorkflowPhase.OPTIMIZED
        ):
            raise ToolCallError("The scientific workflow did not complete exactly once.")
        if self.report is None:
            try:
                report = self.executors["build_workflow_report"](self.session.state)
            except Exception:
                raise ToolCallError("The scientific report could not be built.") from None
            if not isinstance(report, WorkflowReport):
                raise ToolCallError("The scientific report was invalid.")
            self.report = report
        return ScientificLoopResult(
            tuple(self.session.messages),
            self.report,
            self.plan,
            tuple(self.stage_results),
            self.session.turn_count,
        )

    def begin_objective_challenge(self) -> ObjectiveContext:
        """Initialize the post-workflow objective from immutable scientific evidence."""
        self.scientific_result()
        if self.objective_context is not None or self.objective_prompt_appended:
            raise ToolCallError("The objective challenge can be initialized exactly once.")
        try:
            context = build_objective_context(self.session.state)
        except Exception:
            raise ToolCallError("The objective challenge could not be constructed.") from None
        candidates = [
            {
                "molecule_id": candidate.molecule_id,
                "cluster_id": candidate.cluster_id,
            }
            for candidate in context.candidates
        ]
        prompt_payload = {
            "objective": "Select four candidates that maximize the minimum pairwise Morgan/Tanimoto distance.",
            "candidate_pool": candidates,
            "distance_matrix": context.distance_matrix.tolist(),
            "distance_matrix_order": [
                candidate.molecule_id for candidate in context.candidates
            ],
            "baseline_ids": list(context.baseline_ids),
            "baseline_score": context.baseline_score,
            "benchmark_score": context.benchmark_score,
            "target_score": context.target_score,
            "target_rule": "baseline plus 80 percent of attainable improvement",
            "constraints": [
                "exactly four unique supplied molecule IDs",
                "one molecule per fused Butina cluster",
                "all candidates are RDKit MMFF94-parameter eligible",
                "at most three accepted attempts",
            ],
            "instruction": (
                "Call select_diverse_panel with one panel and a concise decision summary. "
                "Use measured limiting-pair feedback to revise if the target is missed."
            ),
        }
        self.session.messages.append(
            {"role": "user", "content": _serialize(prompt_payload)}
        )
        self.objective_context = context
        self.objective_prompt_appended = True
        if abs(context.benchmark_score - context.baseline_score) <= 1e-12:
            self.objective_run = no_improvement_run(context)
            self.objective_evidence = build_objective_evidence(self.objective_run)
        return context

    def request_objective_attempt(self) -> ObjectiveProposal:
        """Request one strict panel proposal without evaluating it."""
        if self.objective_context is None or not self.objective_prompt_appended:
            raise ToolCallError("The objective challenge has not been initialized.")
        if self.objective_run is not None:
            raise ToolCallError("The objective challenge is already complete.")
        if self.pending_objective is not None:
            raise ToolCallError("An objective proposal is already pending evaluation.")
        if len(self.objective_attempts) >= MAX_ATTEMPTS:
            raise ToolCallError("The objective attempt limit was reached.")
        proposal = _request_call(
            self.session,
            self.client,
            "select_diverse_panel",
            ObjectiveProposal,
            DEFAULT_MODEL,
            _max_turns=11,
        )
        self.pending_objective = ObjectiveProposal.model_validate(
            proposal.model_dump()
        )
        return self.pending_objective

    def execute_objective_attempt(
        self, proposal: ObjectiveProposal
    ) -> ObjectiveAttempt:
        """Evaluate the exact pending proposal and append its measured feedback."""
        if proposal is not self.pending_objective or type(proposal) is not ObjectiveProposal:
            raise ToolCallError("The exact pending objective proposal is required.")
        context = self.objective_context
        if context is None or self.objective_run is not None:
            raise ToolCallError("The objective challenge is not awaiting evaluation.")
        try:
            attempt = evaluate_diverse_panel(
                context,
                tuple(proposal.selected_ids),
                attempt_number=len(self.objective_attempts) + 1,
                decision_basis=proposal.decision_basis,
            )
        except Exception:
            raise ToolCallError("The objective evaluator rejected the proposed panel.") from None
        self.objective_attempts.append(attempt)
        self.pending_objective = None
        _append_tool_result(
            self.session,
            {
                "accepted": True,
                "attempt_number": attempt.attempt_number,
                "selected_ids": list(attempt.selected_ids),
                "score": attempt.score,
                "target_score": context.target_score,
                "limiting_pair": list(attempt.limiting_pair),
                "constraints_passed": attempt.constraints_passed,
                "achieved": attempt.achieved,
                "attempts_remaining": MAX_ATTEMPTS - len(self.objective_attempts),
            },
        )
        if attempt.achieved or len(self.objective_attempts) == MAX_ATTEMPTS:
            self.objective_run = finalize_objective_run(
                context, tuple(self.objective_attempts)
            )
            self.objective_evidence = build_objective_evidence(self.objective_run)
        return attempt

    def request_synthesis(self) -> WorkflowResult:
        if (
            self.objective_required or self.objective_context is not None
        ) and self.objective_run is None:
            raise ToolCallError(
                "The objective challenge must terminate before the conclusion."
            )
        scientific = self.scientific_result()
        objective_active = self.objective_required or self.objective_context is not None
        objective_evidence = self.objective_evidence if objective_active else None
        if objective_active and objective_evidence is None:
            raise ToolCallError("The objective evidence record is missing.")
        if not self.synthesis_prompt_appended:
            evidence_records = list(scientific.report.evidence)
            if objective_evidence is not None:
                evidence_records.append(objective_evidence)
            evidence = _serialize(
                {"evidence": [item.__dict__ for item in evidence_records]}
            )
            prompt = _SYNTHESIS_PROMPT
            if objective_active:
                prompt += (
                    " Add one objective_driven_selection section grounded in O01. "
                    "Describe structural-library coverage only, and preserve whether the "
                    "target was achieved or the attempt limit was reached."
                )
            self.session.messages.append(
                {"role": "user", "content": prompt + "\n" + evidence}
            )
            self.synthesis_prompt_appended = True
        try:
            conclusion_model = (
                ObjectiveSubmitConclusionArgs if objective_active else SubmitSynthesisArgs
            )
            conclusion = _request_call(
                self.session,
                self.client,
                "submit_synthesis",
                conclusion_model,
                DEFAULT_MODEL,
                _max_turns=11 if objective_active else 8,
            )
            if objective_active:
                assert objective_evidence is not None
                conclusion = validate_objective_conclusion(
                    conclusion, scientific.report, objective_evidence
                )
            else:
                conclusion = validate_conclusion(conclusion, scientific.report)
        except _HostedArgumentsValidationError:
            raise ConclusionValidationError(scientific.report) from None
        return WorkflowResult(
            tuple(self.session.messages),
            scientific.report,
            scientific.plan,
            conclusion,
            scientific.stage_results,
            self.session.turn_count,
            self.objective_run,
            objective_evidence,
        )


def _complete_scientific_loop(
    controller: BoundedWorkflowController,
    progress_callback: ProgressCallback | None,
) -> ScientificLoopResult:
    try:
        plan = controller.request_plan()
    except Exception as error:
        _emit_progress(progress_callback, "failure", str(error))
        raise
    _emit_progress(progress_callback, "plan", plan)

    for stage in STAGES:
        if controller.session.eligible_tool_name() != stage:
            raise ToolCallError("The scientific workflow state is out of phase.")
        try:
            proposal = controller.request_next_stage()
        except Exception as error:
            _emit_progress(progress_callback, "failure", str(error))
            raise
        try:
            result = controller.execute_pending(proposal.arguments)
        except Exception as error:
            _emit_progress(progress_callback, "failure", str(error))
            raise
        _emit_progress(
            progress_callback,
            "stage",
            {"result": result, "arguments": proposal.arguments},
        )

    try:
        return controller.scientific_result()
    except Exception as error:
        if "scientific report" in str(error):
            _emit_progress(progress_callback, "failure", str(error))
        raise


def run_scientific_loop(
    user_goal: str,
    api_key: str,
    *,
    client: Any = None,
    executors: dict[str, Any] | None = None,
    state: WorkflowState | None = None,
    progress_callback: ProgressCallback | None = None,
    skill: str | None = None,
) -> ScientificLoopResult:
    """Run one plan and six phase-gated scientific tool calls in one history."""
    controller = BoundedWorkflowController.create(
        user_goal,
        api_key,
        client=client,
        executors=executors,
        state=state,
        skill=skill,
    )
    return _complete_scientific_loop(controller, progress_callback)


_STAGE_METRICS = {
    "inspect_library": ("raw_count", "valid_count", "invalid_count", "invalid_ids", "preview_count"),
    "generate_morgan_fingerprints": ("molecule_count", "active_bits_min", "active_bits_median", "active_bits_max"),
    "measure_tanimoto_similarity": ("q1", "median", "q3", "p90", "max", "most_similar_nonidentical_pair"),
    "discover_fused_butina_clusters": ("cluster_cutoff", "cluster_count", "singleton_count", "largest_cluster_sizes"),
    "embed_representative_conformers": ("selected_representative_count", "selection_shortfall", "generated_conformer_count", "partial_embedding_ids", "zero_embedding_ids"),
    "optimize_conformers_mmff94": ("attempted_conformer_count", "converged_conformer_count", "unconverged_conformer_count"),
}


def _display_figure(figure: Any) -> None:
    from io import BytesIO

    from IPython.display import Image, display
    from matplotlib.figure import Figure

    if isinstance(figure, Figure):
        png = BytesIO()
        figure.savefig(png, format="png", dpi=120, bbox_inches="tight")
        display(Image(data=png.getvalue(), format="png"))
        return
    display(figure)


def _display_progress_event(event: str, payload: Any) -> None:
    from IPython.display import Markdown, display

    if event == "plan":
        lines = "\n".join(f"- `{item.stage}` — {item.rationale}" for item in payload.stages)
        display(Markdown(f"## Nemotron plan\n{lines}"))
        return
    if event == "failure":
        display(Markdown(f"**Workflow stopped:** {payload}"))
        return
    result, arguments = payload["result"], payload["arguments"]
    chosen = arguments.model_dump(mode="json", exclude={"decision_basis"})
    lines = [f"- `{key}`: `{value}`" for key, value in chosen.items() if value not in (None, "", [], {})]
    if not lines:
        lines.append("- No model-selected parameters for this fixed step.")
    if basis := getattr(arguments, "decision_basis", None):
        lines.append(f"- Decision summary: {basis}")
    metrics = {key: result.summary[key] for key in _STAGE_METRICS[result.stage] if key in result.summary}
    if metrics:
        lines.extend(["", "Result:", *(f"- `{key}`: `{value}`" for key, value in metrics.items())])
    display(Markdown(f"### Nemotron → {result.display_label}\n" + "\n".join(lines)))
    for figure in result.figures:
        _display_figure(figure)


_EVIDENCE_CITATION_KEY = r"(?:E0[1-6]|O01)"
_EVIDENCE_CITATION_GROUP = (
    rf"{_EVIDENCE_CITATION_KEY}"
    rf"(?:\s*(?:,|and|[-–—])\s*{_EVIDENCE_CITATION_KEY})*"
)


def _presentation_text(text: str) -> str:
    text = re.sub(rf"\s*\({_EVIDENCE_CITATION_GROUP}\)", "", text)
    text = re.sub(
        rf"\s*Evidence:\s*{_EVIDENCE_CITATION_GROUP}\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:E0[1-6]|O01)\b", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return " ".join(text.split())


def _display_conclusion(result: WorkflowResult) -> None:
    from IPython.display import Markdown, display

    sections = "\n\n".join(
        f"### {section.theme.replace('_', ' ').title()}\n{_presentation_text(section.prose)}"
        for section in result.conclusion.sections
    )
    rendered = f"## Schema-checked scientific conclusion\n### {_presentation_text(result.conclusion.headline)}\nPython checks the response structure before rendering; Nemotron's qualitative interpretation is not automatically fact-verified.\nPython-rendered methods: 3D conformers use ETKDGv3; energies use MMFF94.\n\n{sections}"
    display(Markdown(rendered))


def run_workflow(
    user_goal: str,
    api_key: str,
    display_events: bool = True,
    *,
    skill: str | None = None,
    client: Any = None,
    executors: dict[str, Any] | None = None,
    state: WorkflowState | None = None,
) -> WorkflowResult:
    """Run the chain, then schema-check one evidence-linked synthesis without fact-verifying its prose."""
    progress_callback = _display_progress_event if display_events else None
    controller = BoundedWorkflowController.create(
        user_goal,
        api_key,
        client=client,
        executors=executors,
        state=state,
        skill=skill,
    )
    _complete_scientific_loop(controller, progress_callback)
    try:
        result = controller.request_synthesis()
    except ConclusionValidationError:
        _emit_progress(progress_callback, "failure", "Nemotron synthesis failed validation.")
        raise
    except Exception as error:
        _emit_progress(progress_callback, "failure", str(error))
        raise
    if display_events:
        _display_conclusion(result)
    return result
