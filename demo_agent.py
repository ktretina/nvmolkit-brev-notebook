"""One bounded nvMolKit workflow with a grounded, schema-checked Nemotron conclusion."""

from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import httpx
from openai import APIConnectionError, AuthenticationError, OpenAI, PermissionDeniedError
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
    ObjectiveActionMenu,
    ObjectiveAttempt,
    ObjectiveContext,
    ObjectiveRun,
    ObjectiveSwap,
    TerminationReason,
    accepted_maxima,
    build_action_menu,
    build_objective_context,
    build_objective_evidence,
    certify_argmax_reachability,
    evaluate_selected_swap,
    finalize_no_legal_swap,
    measure_panel,
    no_improvement_run,
    resolve_menu_action,
    score_key,
    target_is_achieved,
    terminal_objective_run,
)
from objective_findings import (
    CONCLUSION_THEMES,
    EvidenceFinding,
    EvidenceSnapshot,
    FindingCatalog,
    MeasuredSummary,
    build_evidence_snapshot,
    build_finding_catalog_from_snapshot,
    validate_finding,
)


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NEMOTRON_TOOL_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_OBJECTIVE_CORRECTIONS = 2
MAX_OBJECTIVE_HOSTED_TURNS = 12
MAX_OBJECTIVE_SYNTHESIS_TURNS = 13
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


DecisionRule = Literal["maximize_predicted_minimum_distance"]
MoleculeSwapId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=170,
        pattern=r"^[^\s\r\n`]+->[^\s\r\n`]+$",
    ),
]


class ObjectiveSelection(_StrictModel):
    state_id: Annotated[str, StringConstraints(pattern=r"^state-[0-9a-f]{16}$")]
    swap_id: MoleculeSwapId
    observed_limiting_pairs: list[list[MoleculeId]] = Field(min_length=1, max_length=6)
    decision_rule: DecisionRule

    @field_validator("swap_id")
    @classmethod
    def swap_id_has_one_reserved_delimiter(cls, value: str) -> str:
        if value.count("->") != 1:
            raise ValueError("Objective swap ID must contain one reserved delimiter.")
        return value

    @field_validator("observed_limiting_pairs")
    @classmethod
    def pairs_are_exact_pairs(cls, value: list[list[str]]) -> list[list[str]]:
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in value):
            raise ValueError("Objective limiting pairs must contain two distinct IDs.")
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


class FindingSelection(_StrictModel):
    ordered_finding_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=7, max_length=7)


@dataclass(frozen=True)
class EvidenceControlledConclusion:
    evidence_snapshot: EvidenceSnapshot
    measured_summary: MeasuredSummary
    ordered_findings: tuple[EvidenceFinding, ...]
    finding_selection_status: Literal["selected", "finding_selection_unavailable"]

    def __post_init__(self) -> None:
        if type(self.evidence_snapshot) is not EvidenceSnapshot:
            raise ValueError("Conclusion requires an exact evidence snapshot.")
        if self.measured_summary is not self.evidence_snapshot.summary:
            raise ValueError("Conclusion summary must come from its evidence snapshot.")
        if type(self.ordered_findings) is not tuple or len(self.ordered_findings) != 7:
            raise ValueError("Conclusion requires exactly seven findings.")
        if len({finding.finding_id for finding in self.ordered_findings}) != 7:
            raise ValueError("Conclusion finding IDs must be unique.")
        if {finding.theme for finding in self.ordered_findings} != set(CONCLUSION_THEMES):
            raise ValueError("Conclusion requires exactly one finding per theme.")
        for finding in self.ordered_findings:
            validate_finding(finding, self.evidence_snapshot)
        catalog = build_finding_catalog_from_snapshot(self.evidence_snapshot)
        if any(finding.finding_id not in catalog.ids for finding in self.ordered_findings):
            raise ValueError("Conclusion findings must belong to the current catalog.")
        if self.finding_selection_status not in {
            "selected", "finding_selection_unavailable"
        }:
            raise ValueError("Conclusion selection status is invalid.")
        if (
            self.finding_selection_status == "finding_selection_unavailable"
            and tuple(finding.finding_id for finding in self.ordered_findings)
            != tuple(catalog.ids_for_theme(theme)[0] for theme in CONCLUSION_THEMES)
        ):
            raise ValueError("Fallback findings must use canonical theme-order choices.")


FindingSelectionFailureReason = Literal[
    "authentication_failure",
    "permission_failure",
    "transport_failure",
    "provider_failure",
    "malformed_response",
    "local_validation_failure",
]


@dataclass(frozen=True)
class FindingSelectionAudit:
    provider_response_count: int
    failure_reason: FindingSelectionFailureReason | None
    failure_detail: str | None

    def __post_init__(self) -> None:
        if self.provider_response_count not in (0, 1):
            raise ValueError("Finding-selection response count must be zero or one.")
        if (self.failure_reason is None) != (self.failure_detail is None):
            raise ValueError("Finding-selection failure reason and detail must agree.")


@dataclass(frozen=True)
class _FindingSelectionOutcome:
    findings: tuple[EvidenceFinding, ...] | None
    audit: FindingSelectionAudit


def validate_evidence_controlled_conclusion(
    conclusion: EvidenceControlledConclusion,
) -> EvidenceControlledConclusion:
    """Re-run every immutable-container invariant at a later trust boundary."""
    if type(conclusion) is not EvidenceControlledConclusion:
        raise ValueError("Conclusion validation requires the exact conclusion type.")
    EvidenceControlledConclusion(
        evidence_snapshot=conclusion.evidence_snapshot,
        measured_summary=conclusion.measured_summary,
        ordered_findings=conclusion.ordered_findings,
        finding_selection_status=conclusion.finding_selection_status,
    )
    return conclusion


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
    "select_next_panel_swap": "Select one state-bound tied-maximum action from the current deterministic menu.",
    "select_evidence_findings": "Select exactly one measured finding for each conclusion theme and order the selected finding IDs by evidence emphasis.",
    "submit_synthesis": "Submit one grounded qualitative synthesis; keep evidence IDs only in evidence_keys fields.",
}


class ToolCallError(RuntimeError):
    """A secret-safe failure in the bounded hosted or scientific loop."""


class ObjectiveCorrectionLimitError(ToolCallError):
    """The model exhausted its bounded objective-response correction budget."""


class ObjectiveEligibilityError(ToolCallError):
    """The measured workflow cannot enter the certified bounded policy."""


class ObjectiveEvaluationError(ToolCallError):
    """The selected objective action could not be evaluated and committed atomically."""


class _HostedArgumentsValidationError(ToolCallError):
    def __init__(
        self,
        stage: str,
        issues: tuple[tuple[str, str], ...],
        call_id: str | None = None,
    ):
        self.stage = stage
        self.issues = issues
        self.call_id = call_id
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
    conclusion: SubmitSynthesisArgs | ObjectiveSubmitConclusionArgs | EvidenceControlledConclusion
    stage_results: tuple[StageResult, ...]
    turn_count: int = 8
    objective_run: ObjectiveRun | None = None
    objective_evidence: EvidenceRecord | None = None
    finding_selection_audit: FindingSelectionAudit | None = None


@dataclass(frozen=True)
class StageProposal:
    stage: StageName
    arguments: BaseModel


_CANONICAL_REPORT_ROWS = (
    ("E01", "Library inspection", "RDKit input validation"),
    ("E02", "Morgan fingerprints", "MorganFingerprintGenerator"),
    ("E03", "Tanimoto similarity", "crossTanimotoSimilarity"),
    ("E04", "Fused Butina clusters", "fused_butina"),
    ("E05", "Representative embedding", "EmbedMolecules"),
    ("E06", "MMFF94 optimization", "MMFFOptimizeMoleculesConfs"),
)
_CANONICAL_REPORT_REQUIRED_FIELDS = {
    "E01": {"raw_count", "valid_count", "invalid_count", "invalid_ids", "preview_count", "count_unit"},
    "E02": {"fingerprint_radius", "fingerprint_size_bits", "packed_shape", "molecule_count", "active_bits_min", "active_bits_median", "active_bits_max", "executor", "size_unit"},
    "E03": {"matrix_shape", "q1", "median", "q3", "p90", "max_off_diagonal", "most_similar_pair", "similarity_unit"},
    "E04": {"cutoff", "cluster_count", "singleton_count", "singleton_fraction", "largest_cluster_sizes", "assignment_count", "cutoff_unit"},
    "E05": {"requested_representative_count", "selected_representative_count", "selection_shortfall", "representative_policy", "representatives", "requested_conformers_per_representative", "generated_conformer_count", "partial_embedding_ids", "zero_embedding_ids", "count_unit"},
    "E06": {"attempted_conformer_count", "converged_conformer_count", "unconverged_conformer_count", "per_conformer_records", "selected_conformer_records", "energy_unit", "comparison_scope"},
}


def _validate_prepared_messages(messages: Any) -> None:
    if type(messages) is not tuple or len(messages) != 16:
        raise ValueError("A prepared snapshot requires exactly 16 messages.")
    if [item.get("role") for item in messages[:2]] != ["system", "user"]:
        raise ValueError("A prepared snapshot requires system and user grounding.")
    expected_names = ("submit_workflow_plan", *STAGES)
    for index, expected_name in enumerate(expected_names):
        assistant, tool = messages[2 + index * 2:4 + index * 2]
        calls = assistant.get("tool_calls") if type(assistant) is dict else None
        if (
            assistant.get("role") != "assistant"
            or type(calls) is not list
            or len(calls) != 1
            or type(calls[0]) is not dict
            or calls[0].get("type") != "function"
            or type(calls[0].get("id")) is not str
            or not calls[0]["id"].strip()
            or type(calls[0].get("function")) is not dict
            or calls[0]["function"].get("name") != expected_name
            or type(tool) is not dict
            or tool.get("role") != "tool"
            or tool.get("tool_call_id") != calls[0]["id"]
        ):
            raise ValueError("Prepared hosted messages must be exactly paired.")


def _validate_prepared_report(report: Any) -> None:
    if type(report) is not WorkflowReport or len(report.evidence) != 6:
        raise ValueError("A prepared snapshot requires the canonical E01-E06 report.")
    for record, expected in zip(report.evidence, _CANONICAL_REPORT_ROWS, strict=True):
        if (
            type(record) is not EvidenceRecord
            or (record.key, record.label, record.provenance) != expected
            or type(record.payload_json) is not str
        ):
            raise ValueError("A prepared snapshot requires the canonical E01-E06 report.")
        try:
            payload = json.loads(record.payload_json)
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Prepared report evidence must contain canonical JSON.") from None
        if (
            type(payload) is not dict
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != record.payload_json
        ):
            raise ValueError("Prepared report evidence must contain canonical JSON.")
        if not _CANONICAL_REPORT_REQUIRED_FIELDS[record.key].issubset(payload):
            raise ValueError("Prepared report evidence must be production-shaped.")


def _prepared_snapshot_digest(
    messages: tuple[dict[str, Any], ...],
    state: WorkflowState,
    plan: WorkflowPlan,
    stage_results: tuple[StageResult, ...],
    report: WorkflowReport,
    turn_count: int,
) -> str:
    payload = {
        "messages": _json_safe(messages),
        "state": _canonical_snapshot_value(state),
        "plan": plan.model_dump(mode="json"),
        "stage_results": [
            {
                "stage": item.stage,
                "display_label": item.display_label,
                "summary": _json_safe(item.summary),
                "figure_types": [type(figure).__qualname__ for figure in item.figures],
            }
            for item in stage_results
        ],
        "report": [record.__dict__ for record in report.evidence],
        "turn_count": turn_count,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_snapshot_value(value: Any) -> Any:
    """Losslessly normalize retained mutable artifacts for tamper detection."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return _canonical_snapshot_value(value.model_dump(mode="python"))
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): _canonical_snapshot_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_snapshot_value(item) for item in value]
    if is_dataclass(value):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_snapshot_value(getattr(value, item.name))
                for item in dataclass_fields(value)
            },
        }
    to_binary = getattr(value, "ToBinary", None)
    if callable(to_binary):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "binary": bytes(to_binary()).hex(),
        }
    optimization_fields = ("energies", "converged", "mol_indices", "conf_indices")
    if all(hasattr(value, name) for name in optimization_fields):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "optimization_buffers": {
                name: _canonical_snapshot_value(getattr(value, name))
                for name in optimization_fields
            },
        }
    as_tensor = getattr(value, "torch", None)
    if callable(as_tensor):
        tensor = as_tensor()
        if tensor is not value:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "tensor": _canonical_snapshot_value(tensor),
            }
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
            "values": _canonical_snapshot_value(to_list()),
        }
    if hasattr(value, "__dict__"):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": _canonical_snapshot_value(vars(value)),
        }
    raise ValueError(
        f"Prepared snapshot contains an unsupported artifact: {type(value).__qualname__}."
    )


def _validate_prepared_scientific_semantics(
    state: WorkflowState, report: WorkflowReport
) -> None:
    try:
        rebuilt_report = build_workflow_report(deepcopy(state))
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Prepared state cannot rebuild the canonical E01-E06 report."
        ) from error
    if rebuilt_report != report:
        raise ValueError("Prepared state and E01-E06 report are inconsistent.")
    context = build_objective_context(state)
    current = measure_panel(context, context.baseline_ids)
    attempts: list[ObjectiveAttempt] = []
    while not target_is_achieved(current.score, context.target_score) and len(attempts) < 3:
        menu = build_action_menu(context, current, len(attempts))
        maxima = accepted_maxima(menu)
        if not maxima:
            raise ValueError("Prepared objective state has no certified argmax action.")
        attempt = evaluate_selected_swap(context, menu, maxima[0], len(attempts) + 1)
        attempts.append(attempt)
        current = attempt.measurement
    reason = (
        TerminationReason.TARGET_ACHIEVED
        if target_is_achieved(current.score, context.target_score)
        else TerminationReason.ATTEMPT_LIMIT_REACHED
    )
    run = terminal_objective_run(context, tuple(attempts), reason)
    build_evidence_snapshot(report, run)


@dataclass(frozen=True)
class PreparedScientificSnapshot:
    """A validated seven-turn scientific boundary reusable only by deep cloning."""

    messages: tuple[dict[str, Any], ...]
    state: WorkflowState
    plan: WorkflowPlan
    stage_results: tuple[StageResult, ...]
    report: WorkflowReport
    turn_count: int

    def __post_init__(self) -> None:
        messages = tuple(deepcopy(self.messages))
        state = deepcopy(self.state)
        plan = WorkflowPlan.model_validate(deepcopy(self.plan.model_dump()))
        stage_results = tuple(deepcopy(self.stage_results))
        report = deepcopy(self.report)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "stage_results", stage_results)
        object.__setattr__(self, "report", report)
        if self.turn_count != 7:
            raise ValueError("A prepared snapshot requires exactly seven hosted turns.")
        _validate_prepared_messages(self.messages)
        if type(self.state) is not WorkflowState or self.state.phase is not WorkflowPhase.OPTIMIZED:
            raise ValueError("A prepared snapshot requires an optimized scientific state.")
        if (
            type(self.plan) is not WorkflowPlan
            or tuple(item.stage for item in self.plan.stages) != STAGES
            or type(self.stage_results) is not tuple
            or tuple(item.stage for item in self.stage_results) != STAGES
            or any(type(item) is not StageResult for item in self.stage_results)
            or any(
                type(item.display_label) is not str
                or not item.display_label
                or type(item.summary) is not dict
                or not item.summary
                for item in self.stage_results
            )
        ):
            raise ValueError("A prepared snapshot requires the exact six ordered stages.")
        _validate_prepared_report(self.report)
        _validate_prepared_scientific_semantics(self.state, self.report)
        object.__setattr__(self, "_canonical_digest", _prepared_snapshot_digest(
            self.messages, self.state, self.plan, self.stage_results,
            self.report, self.turn_count,
        ))


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


def _swap_payload(item: ObjectiveSwap) -> dict[str, Any]:
    return {
        "replace_id": item.replace_id,
        "replacement_id": item.replacement_id,
        "resulting_ids": list(item.resulting_ids),
        "predicted_score": item.predicted_score,
        "score_delta": item.score_delta,
        "limiting_pair": list(item.limiting_pair),
    }


def _objective_action_payload(item: ObjectiveSwap) -> dict[str, Any]:
    """Return the one canonical provider-visible action row."""
    return {
        "limiting_pairs": [list(pair) for pair in item.limiting_pairs],
        "predicted_score": item.predicted_score,
        "replace_id": item.replace_id,
        "replacement_id": item.replacement_id,
        "resulting_ids": list(item.resulting_ids),
        "score_delta": item.score_delta,
        "swap_id": item.swap_id,
        "target_status": item.target_status,
    }


def _objective_selection_tool(menu: ObjectiveActionMenu) -> dict[str, Any]:
    if type(menu) is not ObjectiveActionMenu or not menu.actions:
        raise ToolCallError("The current objective action menu is invalid.")
    parameters = ObjectiveSelection.model_json_schema()
    properties = parameters["properties"]
    properties["state_id"]["enum"] = [menu.state_id]
    properties["swap_id"]["enum"] = [action.swap_id for action in menu.actions]
    properties["observed_limiting_pairs"]["enum"] = [[
        list(pair) for pair in menu.source.limiting_pairs
    ]]
    properties["decision_rule"]["enum"] = [
        "maximize_predicted_minimum_distance"
    ]
    parameters["additionalProperties"] = False
    parameters["required"] = list(ObjectiveSelection.model_fields)
    return {
        "type": "function",
        "function": {
            "name": "select_next_panel_swap",
            "description": TOOL_DESCRIPTIONS["select_next_panel_swap"],
            "strict": True,
            "parameters": parameters,
        },
    }


def _panel_key(ids: Any) -> tuple[str, ...]:
    return tuple(sorted(ids))


_REQUIRED_CONCLUSION_EVIDENCE = {
    "dataset_scope": {"E01"},
    "molecular_representation": {"E02"},
    "similarity_structure": {"E03"},
    "clustering": {"E04"},
    "conformational_sampling": {"E05", "E06"},
    "limitations_and_next_steps": {"E01", "E06"},
}

_REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE = {
    "dataset_scope": {"E01"},
    "molecular_representation": {"E02"},
    "similarity_structure": {"E03"},
    "clustering": {"E04"},
    "conformational_sampling": {"E05", "E06"},
    "objective_driven_selection": {"O01"},
    "limitations_and_next_steps": {"E01", "E06", "O01"},
}


def _objective_conclusion_validation_feedback(
    conclusion: ObjectiveSubmitConclusionArgs,
    report: WorkflowReport,
    objective_evidence: EvidenceRecord,
) -> dict[str, Any]:
    """Describe rejected objective metadata without retaining model-authored prose."""
    expected_themes = tuple(_REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE)
    actual_themes = [section.theme for section in conclusion.sections]
    actual_theme_set = set(actual_themes)
    duplicate_themes = [
        theme for theme in expected_themes if actual_themes.count(theme) > 1
    ]
    missing_required: dict[str, list[str]] = {}
    for theme in expected_themes:
        sections = [section for section in conclusion.sections if section.theme == theme]
        missing = {
            key
            for section in sections
            for key in _REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE[theme]
            if key not in section.evidence_keys
        }
        if missing:
            missing_required[theme] = sorted(missing)

    expected_keys = {record.key for record in report.evidence} | {
        objective_evidence.key
    }
    cited_keys = {
        key for section in conclusion.sections for key in section.evidence_keys
    }
    return {
        "accepted": False,
        "validation_issues": {
            "missing_themes": [
                theme for theme in expected_themes if theme not in actual_theme_set
            ],
            "extra_themes": sorted(actual_theme_set - set(expected_themes)),
            "duplicate_themes": duplicate_themes,
            "missing_required_evidence": missing_required,
            "missing_evidence_keys": sorted(expected_keys - cited_keys),
            "extra_evidence_keys": sorted(cited_keys - expected_keys),
        },
        "instruction": (
            "Resubmit all seven themes with their required evidence_keys; "
            "author the corrected evidence links without changing the evidence IDs."
        ),
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


def _tool_definition(
    name: str,
    model: type[BaseModel],
    *,
    finding_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    parameters = model.model_json_schema()
    parameters["additionalProperties"] = False
    parameters["required"] = list(model.model_fields)
    if name == "submit_synthesis" and model is ObjectiveSubmitConclusionArgs:
        section_branches = []
        for theme, required_evidence in _REQUIRED_OBJECTIVE_CONCLUSION_EVIDENCE.items():
            evidence_keys = [
                key for key in ObjectiveEvidenceKey.__args__ if key in required_evidence
            ]
            section_branches.append({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "theme": {"type": "string", "enum": [theme]},
                    "prose": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1200,
                    },
                    "evidence_keys": {
                        "type": "array",
                        "enum": [evidence_keys],
                    },
                },
                "required": ["theme", "prose", "evidence_keys"],
            })
        parameters["properties"]["sections"]["items"] = {
            "anyOf": section_branches
        }
        parameters.pop("$defs", None)
    if name == "select_evidence_findings" and model is FindingSelection:
        if (
            type(finding_ids) is not tuple
            or not finding_ids
            or any(type(finding_id) is not str for finding_id in finding_ids)
            or len(finding_ids) != len(set(finding_ids))
        ):
            raise ValueError("Finding selection schema requires unique current catalog IDs.")
        parameters["properties"]["ordered_finding_ids"]["items"] = {
            "type": "string",
            "enum": list(finding_ids),
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "strict": True,
            "parameters": parameters,
        },
    }


def _canonical_findings(
    catalog: FindingCatalog,
    snapshot: EvidenceSnapshot,
) -> tuple[EvidenceFinding, ...]:
    by_id = {finding.finding_id: finding for finding in catalog.findings}
    findings = tuple(
        by_id[catalog.ids_for_theme(theme)[0]] for theme in CONCLUSION_THEMES
    )
    for finding in findings:
        validate_finding(finding, snapshot)
    return findings


def _validate_finding_selection(
    selection: FindingSelection,
    catalog: FindingCatalog,
    snapshot: EvidenceSnapshot,
) -> tuple[EvidenceFinding, ...]:
    ids = tuple(selection.ordered_finding_ids)
    if len(ids) != 7 or len(set(ids)) != 7:
        raise ValueError("Finding selection requires exactly seven unique IDs.")
    by_id = {finding.finding_id: finding for finding in catalog.findings}
    try:
        findings = tuple(by_id[finding_id] for finding_id in ids)
    except KeyError as error:
        raise ValueError("Finding selection used an ID outside the current catalog.") from error
    if {finding.theme for finding in findings} != set(CONCLUSION_THEMES):
        raise ValueError("Finding selection requires exactly one finding per theme.")
    for finding in findings:
        validate_finding(finding, snapshot)
    return findings


def _finding_catalog_prompt(catalog: FindingCatalog) -> str:
    return _serialize({
        "instruction": (
            "Select exactly one current finding per conclusion theme. Order the seven "
            "IDs by the evidence emphasis you want presented. Do not write prose."
        ),
        "findings": [
            {
                "finding_id": finding.finding_id,
                "theme": finding.theme,
                "evidence_keys": list(finding.evidence_keys),
                "text": finding.text,
            }
            for finding in catalog.findings
        ],
    })


def _request_finding_selection(
    session: AgentSession,
    client: Any,
    catalog: FindingCatalog,
    snapshot: EvidenceSnapshot,
) -> _FindingSelectionOutcome:
    """Make one hosted request; pair every representable real call and never retry."""
    try:
        tool = _tool_definition(
            "select_evidence_findings",
            FindingSelection,
            finding_ids=catalog.ids,
        )
    except Exception:
        return _FindingSelectionOutcome(None, FindingSelectionAudit(
            0,
            "local_validation_failure",
            "The finding-selection request could not be validated locally.",
        ))
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=session.messages,
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": "select_evidence_findings"},
            },
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.0,
            max_tokens=400,
            stream=False,
        )
    except Exception as error:
        if isinstance(error, AuthenticationError):
            reason: FindingSelectionFailureReason = "authentication_failure"
            try:
                _raise_request_error(error)
            except ValueError as safe_error:
                detail = str(safe_error)
        elif isinstance(error, PermissionDeniedError):
            reason = "permission_failure"
            try:
                _raise_request_error(error)
            except ValueError as safe_error:
                detail = str(safe_error)
        elif isinstance(error, (httpx.TransportError, APIConnectionError)):
            reason = "transport_failure"
            detail = _REQUEST_ERROR
        else:
            reason = "provider_failure"
            detail = _REQUEST_ERROR
        return _FindingSelectionOutcome(
            None, FindingSelectionAudit(0, reason, detail)
        )

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return _FindingSelectionOutcome(None, FindingSelectionAudit(
            1,
            "malformed_response",
            "The finding-selection response did not contain an assistant message.",
        ))
    assistant = BoundedWorkflowController._objective_assistant_payload(message)
    session.messages.append(assistant)
    session.turn_count += 1
    calls = getattr(message, "tool_calls", None)
    call_ids = tuple(
        call_id
        for call in calls
        if isinstance((call_id := getattr(call, "id", None)), str)
        and call_id.strip()
    ) if isinstance(calls, (list, tuple)) else ()

    selected: tuple[EvidenceFinding, ...] | None = None
    accepted_id: str | None = None
    failure_reason: FindingSelectionFailureReason | None = "malformed_response"
    failure_detail: str | None = (
        "The finding-selection response failed strict envelope or schema validation."
    )
    if isinstance(calls, (list, tuple)) and len(calls) == 1:
        call = calls[0]
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        raw_arguments = getattr(function, "arguments", None)
        try:
            if (
                getattr(call, "type", None) != "function"
                or not isinstance(call_id, str)
                or not call_id.strip()
                or getattr(function, "name", None) != "select_evidence_findings"
                or not isinstance(raw_arguments, str)
            ):
                raise ValueError
            decoded = json.loads(raw_arguments)
            selection = FindingSelection.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            selected = None
        else:
            try:
                selected = _validate_finding_selection(selection, catalog, snapshot)
            except ValueError:
                failure_reason = "local_validation_failure"
                failure_detail = (
                    "The finding selection did not satisfy the current evidence catalog."
                )
            else:
                accepted_id = call_id
                failure_reason = None
                failure_detail = None

    for call_id in call_ids:
        accepted = selected is not None and call_id == accepted_id
        session.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": _serialize({
                "accepted": accepted,
                "status": "selected" if accepted else "rejected",
            }),
        })
    return _FindingSelectionOutcome(
        selected,
        FindingSelectionAudit(1, failure_reason, failure_detail),
    )


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
            tools=[
                _tool_definition(
                    expected_name,
                    argument_model,
                )
            ],
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
            raise _HostedArgumentsValidationError(
                expected_name,
                (("arguments", "non_object_json"),),
                call_id,
            )
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
            call_id,
        ) from None
    except json.JSONDecodeError:
        raise _HostedArgumentsValidationError(
            expected_name,
            (("arguments", "invalid_json"),),
            call_id,
        ) from None
    except (AttributeError, IndexError, TypeError):
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


def _append_rejected_hosted_call(
    session: AgentSession, error: _HostedArgumentsValidationError
) -> None:
    if not isinstance(error.call_id, str) or not error.call_id.strip():
        raise ToolCallError("The rejected hosted tool call ID was missing.")
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": error.call_id,
                    "type": "function",
                    "function": {
                        "name": error.stage,
                        "arguments": _serialize(
                            {
                                "validation_issues": [
                                    {"field": field, "error_type": error_type}
                                    for field, error_type in error.issues
                                ]
                            }
                        ),
                    },
                }
            ],
        }
    )
    session.turn_count += 1


def _bounded_objective_synthesis_issues(
    issues: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Reduce hosted schema errors to a small allowlisted retry vocabulary."""
    allowed_error_types = {
        "extra_forbidden",
        "invalid_json",
        "list_type",
        "literal_error",
        "missing",
        "non_object_json",
        "string_too_long",
        "string_too_short",
        "string_type",
        "too_long",
        "too_short",
    }
    bounded: list[tuple[str, str]] = []
    for field, error_type in issues:
        parts = field.split(".")
        if parts in (["arguments"], ["headline"], ["sections"]):
            bounded_field = field
        elif len(parts) >= 2 and parts[0] == "sections" and parts[1].isdigit():
            if len(parts) == 2:
                bounded_field = "sections.item"
            elif len(parts) == 3 and parts[2] in {"theme", "prose", "evidence_keys"}:
                bounded_field = f"sections.item.{parts[2]}"
            elif (
                len(parts) == 4
                and parts[2] == "evidence_keys"
                and parts[3].isdigit()
            ):
                bounded_field = "sections.item.evidence_keys.item"
            else:
                bounded_field = "arguments"
        else:
            bounded_field = "arguments"
        bounded_error = (
            error_type if error_type in allowed_error_types else "invalid_value"
        )
        issue = (bounded_field, bounded_error)
        if issue not in bounded:
            bounded.append(issue)
        if len(bounded) == 12:
            break
    return tuple(bounded)


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


@dataclass(frozen=True)
class _ObjectiveTransition:
    attempt: ObjectiveAttempt
    objective_attempts: tuple[ObjectiveAttempt, ...]
    accepted_attempt_count: int
    rejected_selection_count: int
    correction_prompts_sent: int
    selection_response_count: int
    provider_request_attempt_count: int
    objective_transport_retry_used: bool
    objective_transport_retry_pending: bool
    pending_action_menu: ObjectiveActionMenu | None
    pending_objective_selection: ObjectiveSelection | None
    objective_failure_reason: TerminationReason | None
    objective_run: ObjectiveRun | None
    objective_evidence: EvidenceRecord | None
    serialized_tool_message: str
    serialized_next_prompt: str | None


@dataclass(frozen=True)
class _ObjectiveCommitSnapshot:
    messages: tuple[Any, ...]
    turn_count: int
    objective_required: bool
    objective_context: ObjectiveContext | None
    pending_action_menu: ObjectiveActionMenu | None
    pending_objective_selection: ObjectiveSelection | None
    objective_attempts: tuple[ObjectiveAttempt, ...]
    accepted_attempt_count: int
    rejected_selection_count: int
    correction_prompts_sent: int
    selection_response_count: int
    provider_request_attempt_count: int
    objective_transport_retry_used: bool
    objective_transport_retry_pending: bool
    objective_failure_reason: TerminationReason | None
    objective_run: ObjectiveRun | None
    objective_evidence: EvidenceRecord | None
    objective_prompt_appended: bool


def _copy_objective_selection(
    selection: ObjectiveSelection | None,
) -> ObjectiveSelection | None:
    """Reconstruct the shallow-frozen selection so nested pair lists cannot alias."""
    if selection is None:
        return None
    if type(selection) is not ObjectiveSelection:
        raise TypeError("Pending objective selection must use the exact schema type.")
    return ObjectiveSelection.model_validate(deepcopy(selection.model_dump()))


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
    pending_action_menu: ObjectiveActionMenu | None = None
    pending_objective_selection: ObjectiveSelection | None = None
    objective_attempts: list[ObjectiveAttempt] = field(default_factory=list)
    accepted_attempt_count: int = 0
    rejected_selection_count: int = 0
    correction_prompts_sent: int = 0
    selection_response_count: int = 0
    provider_request_attempt_count: int = 0
    objective_transport_retry_used: bool = False
    _objective_transport_retry_pending: bool = False
    objective_failure_reason: TerminationReason | None = None
    objective_run: ObjectiveRun | None = None
    objective_evidence: EvidenceRecord | None = None
    objective_prompt_appended: bool = False
    evidence_controlled_conclusion: EvidenceControlledConclusion | None = None
    finding_selection_audit: FindingSelectionAudit | None = None

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
            else 7 <= self.session.turn_count <= MAX_OBJECTIVE_HOSTED_TURNS
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

    def _terminalize_objective(self, reason: TerminationReason, *, menu=None) -> None:
        context = self.objective_context
        if context is None:
            raise ToolCallError("The objective challenge has not been initialized.")
        self.objective_failure_reason = reason if reason in {
            TerminationReason.OBJECTIVE_CORRECTION_LIMIT,
            TerminationReason.OBJECTIVE_PROVIDER_FAILURE,
        } else None
        self.objective_run = terminal_objective_run(
            context, tuple(self.objective_attempts), reason, menu=menu
        )
        self.objective_evidence = build_objective_evidence(self.objective_run)
        self.pending_action_menu = None
        self.pending_objective_selection = None
        self._objective_transport_retry_pending = False

    def _check_objective_bounds(self) -> None:
        values = (
            (self.accepted_attempt_count, 0, 3),
            (self.rejected_selection_count, 0, 2),
            (self.correction_prompts_sent, 0, 1),
            (self.selection_response_count, 0, 5),
            (self.provider_request_attempt_count, 0, 6),
        )
        if any(type(value) is not int or not low <= value <= high for value, low, high in values):
            raise ToolCallError("The objective controller counters are invalid.")
        if self.accepted_attempt_count != len(self.objective_attempts):
            raise ToolCallError("The objective accepted-attempt ledger is inconsistent.")
        if self.correction_prompts_sent != min(self.rejected_selection_count, 1):
            raise ToolCallError("The objective correction-prompt ledger is inconsistent.")

    def begin_objective_challenge(self) -> ObjectiveContext:
        """Measure Step 0 and publish the first certified production action menu."""
        self.scientific_result()
        if self.objective_context is not None or self.objective_prompt_appended:
            raise ToolCallError("The objective challenge can be initialized exactly once.")
        try:
            context = build_objective_context(self.session.state)
            if not certify_argmax_reachability(context):
                raise RuntimeError(
                    "Objective target is not reachable under the bounded decision policy."
                )
        except RuntimeError as error:
            if str(error) == "Objective target is not reachable under the bounded decision policy.":
                raise ObjectiveEligibilityError(str(error)) from None
            raise ToolCallError("The objective challenge could not be constructed.") from None
        except Exception:
            raise ToolCallError("The objective challenge could not be constructed.") from None
        self.objective_context = context
        self.objective_prompt_appended = True
        baseline = measure_panel(context, context.baseline_ids)
        if target_is_achieved(baseline.score, context.target_score):
            if baseline.score_key == score_key(context.benchmark_score):
                self.objective_run = no_improvement_run(context)
            else:
                self.objective_run = terminal_objective_run(
                    context, (), TerminationReason.TARGET_ACHIEVED
                )
            self.objective_evidence = build_objective_evidence(self.objective_run)
            return context
        menu = build_action_menu(context, baseline, 0)
        if not menu.actions:
            self.pending_action_menu = menu
            self._terminalize_objective(TerminationReason.NO_LEGAL_IMPROVING_SWAP, menu=menu)
            return context
        self.pending_action_menu = menu
        self.session.messages.append({"role": "user", "content": _serialize({
            "candidate_actions": [_objective_action_payload(item) for item in menu.actions],
            "current_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
            "decision_rule": "maximize_predicted_minimum_distance",
            "state_id": menu.state_id,
        })})
        return context

    @staticmethod
    def _objective_assistant_payload(message: Any) -> dict[str, Any]:
        """Preserve the provider's actual assistant shape without inventing calls."""
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None),
        }
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, (list, tuple)):
            payload["tool_calls"] = [
                {
                    "id": getattr(call, "id", None),
                    "type": getattr(call, "type", None),
                    "function": {
                        "name": getattr(getattr(call, "function", None), "name", None),
                        "arguments": getattr(
                            getattr(call, "function", None), "arguments", None
                        ),
                    },
                }
                for call in calls
            ]
        return payload

    def _append_objective_assistant(self, payload: dict[str, Any]) -> None:
        self.session.messages.append(payload)
        self.session.turn_count += 1

    def _reject_objective_selection(
        self, assistant: dict[str, Any], call_ids: tuple[str, ...], reason: str
    ) -> None:
        self._append_objective_assistant(assistant)
        for call_id in call_ids:
            self.session.messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _serialize({"accepted": False, "reason": reason}),
            })
        self.rejected_selection_count += 1
        if self.rejected_selection_count == 2:
            self._terminalize_objective(TerminationReason.OBJECTIVE_CORRECTION_LIMIT)
            raise ObjectiveCorrectionLimitError("The objective correction limit was reached.")
        menu = self.pending_action_menu
        assert menu is not None
        self.session.messages.append({"role": "user", "content": _serialize({
            "candidate_actions": [_objective_action_payload(item) for item in menu.actions],
            "current_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
            "decision_rule": "maximize_predicted_minimum_distance",
            "remaining_rejections": 1,
            "state_id": menu.state_id,
        })})
        self.correction_prompts_sent += 1

    def request_objective_selection(self, *, is_transport_retry: bool = False) -> ObjectiveSelection:
        """Request one state-bound deterministic argmax action selection."""
        self._check_objective_bounds()
        menu = self.pending_action_menu
        context = self.objective_context
        if context is None or not self.objective_prompt_appended:
            raise ToolCallError("The objective challenge has not been initialized.")
        if self.objective_run is not None:
            raise ToolCallError("The objective challenge is already complete.")
        if menu is None or not menu.actions:
            raise ToolCallError("No current objective action menu is available.")
        if self.pending_objective_selection is not None:
            raise ToolCallError("An objective selection is already pending evaluation.")
        if self.accepted_attempt_count >= 3:
            raise ToolCallError("The objective attempt limit was reached.")
        if self.rejected_selection_count >= 2:
            self._terminalize_objective(TerminationReason.OBJECTIVE_CORRECTION_LIMIT)
            raise ObjectiveCorrectionLimitError("The objective correction limit was reached.")
        if self.selection_response_count >= 5:
            raise ToolCallError("The objective response limit was reached.")
        if self.provider_request_attempt_count >= 6:
            raise ToolCallError("The objective provider request limit was reached.")
        if is_transport_retry != self._objective_transport_retry_pending:
            raise ToolCallError("The objective transport retry state is invalid.")
        if is_transport_retry:
            self._objective_transport_retry_pending = False
            self.objective_transport_retry_used = True

        while True:
            self.provider_request_attempt_count += 1
            try:
                response = self.client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=self.session.messages,
                    tools=[_objective_selection_tool(menu)],
                    tool_choice={"type": "function", "function": {"name": "select_next_panel_swap"}},
                    extra_body=NEMOTRON_TOOL_EXTRA_BODY,
                    temperature=0.0,
                    max_tokens=400,
                    stream=False,
                )
            except Exception as error:
                if not isinstance(error, (httpx.TransportError, APIConnectionError)):
                    self._terminalize_objective(
                        TerminationReason.OBJECTIVE_PROVIDER_FAILURE
                    )
                    _raise_request_error(error)
                if self.objective_transport_retry_used:
                    self._terminalize_objective(TerminationReason.OBJECTIVE_PROVIDER_FAILURE)
                else:
                    self._objective_transport_retry_pending = True
                raise ToolCallError(_REQUEST_ERROR) from None

            try:
                message = response.choices[0].message
            except (AttributeError, IndexError, TypeError) as error:
                self._terminalize_objective(
                    TerminationReason.OBJECTIVE_PROVIDER_FAILURE
                )
                _raise_request_error(error)
            self.selection_response_count += 1
            assistant = self._objective_assistant_payload(message)
            calls = getattr(message, "tool_calls", None)
            call_ids = tuple(
                call_id
                for call in calls
                if isinstance((call_id := getattr(call, "id", None)), str)
                and call_id.strip()
            ) if isinstance(calls, (list, tuple)) else ()
            decoded: Any = None
            reason = "schema_invalid_selection"
            try:
                if not isinstance(calls, (list, tuple)) or len(calls) != 1:
                    reason = "missing_objective_tool"
                    raise ValueError
                call = calls[0]
                if getattr(call, "type", None) != "function":
                    reason = "wrong_objective_call_type"
                    raise ValueError
                call_id = getattr(call, "id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    reason = "missing_objective_call_id"
                    raise ValueError
                function = getattr(call, "function", None)
                name = getattr(function, "name", None)
                raw = getattr(function, "arguments", "")
                if name != "select_next_panel_swap":
                    reason = "wrong_objective_tool"
                    raise ValueError
                decoded = json.loads(raw)
                selection = ObjectiveSelection.model_validate(decoded)
                expected_pairs = [list(pair) for pair in menu.source.limiting_pairs]
                action = next((item for item in menu.actions if item.swap_id == selection.swap_id), None)
                if selection.state_id != menu.state_id:
                    reason = "stale_objective_state"
                elif selection.observed_limiting_pairs != expected_pairs:
                    reason = "wrong_limiting_pairs"
                elif selection.decision_rule != "maximize_predicted_minimum_distance":
                    reason = "wrong_decision_rule"
                elif action is None:
                    reason = "unavailable_objective_swap"
                elif action not in accepted_maxima(menu):
                    reason = "nonmax_objective_swap"
                elif not all(action.replace_id in pair for pair in menu.source.limiting_pairs):
                    reason = "invalid_co_limiter_effect"
                else:
                    self._append_objective_assistant(assistant)
                    self.pending_objective_selection = ObjectiveSelection.model_validate(selection.model_dump())
                    return self.pending_objective_selection
            except (ValidationError, ValueError, TypeError, AttributeError, IndexError, json.JSONDecodeError):
                pass
            self._reject_objective_selection(assistant, call_ids, reason)
            self._check_objective_bounds()
            if self.selection_response_count >= 5 or self.provider_request_attempt_count >= 6:
                raise ToolCallError("The objective hosted request bound was reached.")

    def _capture_objective_commit_snapshot(self) -> _ObjectiveCommitSnapshot:
        return _ObjectiveCommitSnapshot(
            messages=tuple(deepcopy(list(self.session.messages))),
            turn_count=self.session.turn_count,
            objective_required=self.objective_required,
            objective_context=self.objective_context,
            pending_action_menu=self.pending_action_menu,
            pending_objective_selection=_copy_objective_selection(
                self.pending_objective_selection
            ),
            objective_attempts=tuple(self.objective_attempts),
            accepted_attempt_count=self.accepted_attempt_count,
            rejected_selection_count=self.rejected_selection_count,
            correction_prompts_sent=self.correction_prompts_sent,
            selection_response_count=self.selection_response_count,
            provider_request_attempt_count=self.provider_request_attempt_count,
            objective_transport_retry_used=self.objective_transport_retry_used,
            objective_transport_retry_pending=self._objective_transport_retry_pending,
            objective_failure_reason=self.objective_failure_reason,
            objective_run=self.objective_run,
            objective_evidence=self.objective_evidence,
            objective_prompt_appended=self.objective_prompt_appended,
        )

    def _restore_objective_commit_snapshot(
        self, snapshot: _ObjectiveCommitSnapshot
    ) -> None:
        self.session.messages = list(deepcopy(snapshot.messages))
        self.session.turn_count = snapshot.turn_count
        self.objective_required = snapshot.objective_required
        self.objective_context = snapshot.objective_context
        self.pending_action_menu = snapshot.pending_action_menu
        self.pending_objective_selection = _copy_objective_selection(
            snapshot.pending_objective_selection
        )
        self.objective_attempts = list(snapshot.objective_attempts)
        self.accepted_attempt_count = snapshot.accepted_attempt_count
        self.rejected_selection_count = snapshot.rejected_selection_count
        self.correction_prompts_sent = snapshot.correction_prompts_sent
        self.selection_response_count = snapshot.selection_response_count
        self.provider_request_attempt_count = snapshot.provider_request_attempt_count
        self.objective_transport_retry_used = snapshot.objective_transport_retry_used
        self._objective_transport_retry_pending = snapshot.objective_transport_retry_pending
        self.objective_failure_reason = snapshot.objective_failure_reason
        self.objective_run = snapshot.objective_run
        self.objective_evidence = snapshot.objective_evidence
        self.objective_prompt_appended = snapshot.objective_prompt_appended

    def _build_objective_transition(
        self,
        context: ObjectiveContext,
        menu: ObjectiveActionMenu,
        action: ObjectiveSwap,
    ) -> _ObjectiveTransition:
        attempt = evaluate_selected_swap(
            context, menu, action, self.accepted_attempt_count + 1
        )
        attempts = (*self.objective_attempts, attempt)
        accepted_count = self.accepted_attempt_count + 1
        next_menu = None
        run = None
        evidence = None
        next_prompt = None
        if attempt.achieved:
            run = terminal_objective_run(
                context, attempts, TerminationReason.TARGET_ACHIEVED
            )
        elif accepted_count == MAX_ATTEMPTS:
            run = terminal_objective_run(
                context, attempts, TerminationReason.ATTEMPT_LIMIT_REACHED
            )
        else:
            next_menu = build_action_menu(context, attempt.measurement, accepted_count)
            if not next_menu.actions:
                run = terminal_objective_run(
                    context,
                    attempts,
                    TerminationReason.NO_LEGAL_IMPROVING_SWAP,
                    menu=next_menu,
                )
                next_menu = None
            else:
                next_prompt = _serialize({
                    "candidate_actions": [
                        _objective_action_payload(item) for item in next_menu.actions
                    ],
                    "current_limiting_pairs": [
                        list(pair) for pair in next_menu.source.limiting_pairs
                    ],
                    "decision_rule": "maximize_predicted_minimum_distance",
                    "state_id": next_menu.state_id,
                })
        if run is not None:
            evidence = build_objective_evidence(run)
        tool_message = _serialize({
            "accepted": True,
            "achieved": attempt.achieved,
            "attempt_number": attempt.attempt_number,
            "limiting_pairs": [list(pair) for pair in attempt.limiting_pairs],
            "score": attempt.score,
            "selected_ids": list(attempt.selected_ids),
        })
        return _ObjectiveTransition(
            attempt=attempt,
            objective_attempts=attempts,
            accepted_attempt_count=accepted_count,
            rejected_selection_count=self.rejected_selection_count,
            correction_prompts_sent=self.correction_prompts_sent,
            selection_response_count=self.selection_response_count,
            provider_request_attempt_count=self.provider_request_attempt_count,
            objective_transport_retry_used=self.objective_transport_retry_used,
            objective_transport_retry_pending=self._objective_transport_retry_pending,
            pending_action_menu=next_menu,
            pending_objective_selection=None,
            objective_failure_reason=None,
            objective_run=run,
            objective_evidence=evidence,
            serialized_tool_message=tool_message,
            serialized_next_prompt=next_prompt,
        )

    def _validate_objective_commit(self, transition: _ObjectiveTransition) -> None:
        self._check_objective_bounds()
        if (
            type(self.session.messages) is not list
            or tuple(self.objective_attempts) != transition.objective_attempts
            or self.accepted_attempt_count != transition.accepted_attempt_count
            or self.rejected_selection_count != transition.rejected_selection_count
            or self.correction_prompts_sent != transition.correction_prompts_sent
            or self.selection_response_count != transition.selection_response_count
            or self.provider_request_attempt_count
            != transition.provider_request_attempt_count
            or self.objective_transport_retry_used
            is not transition.objective_transport_retry_used
            or self._objective_transport_retry_pending
            is not transition.objective_transport_retry_pending
            or self.pending_action_menu != transition.pending_action_menu
            or self.pending_objective_selection is not None
            or self.objective_failure_reason != transition.objective_failure_reason
            or self.objective_run != transition.objective_run
            or self.objective_evidence != transition.objective_evidence
            or self.session.messages[-1].get("role")
            != ("user" if transition.serialized_next_prompt is not None else "tool")
        ):
            raise RuntimeError("Objective transition commit invariant failed.")

    def _commit_objective_transition(self, transition: _ObjectiveTransition) -> None:
        snapshot = self._capture_objective_commit_snapshot()
        try:
            self.objective_attempts = list(transition.objective_attempts)
            self.accepted_attempt_count = transition.accepted_attempt_count
            self.rejected_selection_count = transition.rejected_selection_count
            self.correction_prompts_sent = transition.correction_prompts_sent
            self.selection_response_count = transition.selection_response_count
            self.provider_request_attempt_count = transition.provider_request_attempt_count
            self.objective_transport_retry_used = transition.objective_transport_retry_used
            self._objective_transport_retry_pending = transition.objective_transport_retry_pending
            self.pending_action_menu = transition.pending_action_menu
            self.pending_objective_selection = transition.pending_objective_selection
            self.objective_failure_reason = transition.objective_failure_reason
            self.objective_run = transition.objective_run
            self.objective_evidence = transition.objective_evidence
            assistant = self.session.messages[-1]
            self.session.messages.append({
                "role": "tool",
                "tool_call_id": assistant["tool_calls"][0]["id"],
                "content": transition.serialized_tool_message,
            })
            if transition.serialized_next_prompt is not None:
                self.session.messages.append({
                    "role": "user", "content": transition.serialized_next_prompt
                })
            self._validate_objective_commit(transition)
        except Exception:
            self._restore_objective_commit_snapshot(snapshot)
            raise

    def _fail_objective_evaluation(
        self, snapshot: _ObjectiveCommitSnapshot
    ) -> None:
        self._restore_objective_commit_snapshot(snapshot)
        context = self.objective_context
        if context is None:
            raise RuntimeError("Objective evaluation failure lost its context.")
        run = terminal_objective_run(
            context,
            tuple(self.objective_attempts),
            TerminationReason.EVALUATION_NOT_COMPLETED,
        )
        evidence = build_objective_evidence(run)
        error_content = _serialize({
            "accepted": False, "reason": "evaluation_not_completed"
        })
        assistant = self.session.messages[-1]
        self.objective_run = run
        self.objective_evidence = evidence
        self.objective_failure_reason = TerminationReason.EVALUATION_NOT_COMPLETED
        self.pending_action_menu = None
        self.pending_objective_selection = None
        self.session.messages.append({
            "role": "tool",
            "tool_call_id": assistant["tool_calls"][0]["id"],
            "content": error_content,
        })

    def execute_objective_selection(self, selection: ObjectiveSelection) -> ObjectiveAttempt:
        """Evaluate prospectively, then atomically commit the exact pending action."""
        self._check_objective_bounds()
        menu = self.pending_action_menu
        context = self.objective_context
        if selection is not self.pending_objective_selection or type(selection) is not ObjectiveSelection:
            raise ToolCallError("The exact pending objective selection is required.")
        if context is None or menu is None or self.objective_run is not None:
            raise ToolCallError("The objective challenge is not awaiting evaluation.")
        try:
            action = resolve_menu_action(
                context,
                menu,
                state_id=selection.state_id,
                swap_id=selection.swap_id,
                observed_limiting_pairs=tuple(
                    tuple(pair) for pair in selection.observed_limiting_pairs
                ),
                decision_rule=selection.decision_rule,
            )
        except Exception:
            raise ToolCallError(
                "The pending objective selection no longer matches the current menu."
            ) from None
        snapshot = self._capture_objective_commit_snapshot()
        try:
            transition = self._build_objective_transition(context, menu, action)
            self._commit_objective_transition(transition)
        except Exception:
            try:
                self._fail_objective_evaluation(snapshot)
            except Exception:
                self._restore_objective_commit_snapshot(snapshot)
                raise ObjectiveEvaluationError(
                    "The objective evaluation failed before a safe terminal result could be recorded."
                ) from None
            raise ObjectiveEvaluationError(
                "The objective evaluation was not completed; no attempt was accepted."
            ) from None
        return transition.attempt

    # Task-5 compatibility aliases: deterministic selection only, no hosted rationale.
    def request_objective_attempt(self) -> ObjectiveSelection:
        return self.request_objective_selection(
            is_transport_retry=self._objective_transport_retry_pending
        )

    def execute_objective_attempt(self, selection: ObjectiveSelection) -> ObjectiveAttempt:
        return self.execute_objective_selection(selection)

    @property
    def pending_objective(self) -> ObjectiveSelection | None:
        return self.pending_objective_selection

    @property
    def pending_objective_swap(self) -> ObjectiveSwap | None:
        menu = self.pending_action_menu
        selection = self.pending_objective_selection
        if menu is None or selection is None:
            return None
        return next((item for item in menu.actions if item.swap_id == selection.swap_id), None)

    @property
    def objective_suggestions(self) -> tuple[ObjectiveSwap, ...]:
        return () if self.pending_action_menu is None else self.pending_action_menu.actions

    @property
    def objective_rejection_count(self) -> int:
        return self.rejected_selection_count

    @property
    def objective_transport_retry_pending(self) -> bool:
        """Report whether the controller classified the last failure as retryable transport."""
        return self._objective_transport_retry_pending

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
        if objective_active:
            assert self.objective_run is not None
            if self.evidence_controlled_conclusion is None:
                snapshot = build_evidence_snapshot(scientific.report, self.objective_run)
                catalog = build_finding_catalog_from_snapshot(snapshot)
                if not self.synthesis_prompt_appended:
                    self.session.messages.append({
                        "role": "user",
                        "content": _finding_catalog_prompt(catalog),
                    })
                    self.synthesis_prompt_appended = True
                outcome = _request_finding_selection(
                    self.session, self.client, catalog, snapshot
                )
                selected = outcome.findings
                self.finding_selection_audit = outcome.audit
                status: Literal["selected", "finding_selection_unavailable"]
                if selected is None:
                    selected = _canonical_findings(catalog, snapshot)
                    status = "finding_selection_unavailable"
                else:
                    status = "selected"
                self.evidence_controlled_conclusion = EvidenceControlledConclusion(
                    evidence_snapshot=snapshot,
                    measured_summary=snapshot.summary,
                    ordered_findings=selected,
                    finding_selection_status=status,
                )
            return WorkflowResult(
                tuple(self.session.messages),
                scientific.report,
                scientific.plan,
                self.evidence_controlled_conclusion,
                scientific.stage_results,
                self.session.turn_count,
                self.objective_run,
                objective_evidence,
                self.finding_selection_audit,
            )
        if not self.synthesis_prompt_appended:
            evidence_records = list(scientific.report.evidence)
            evidence = _serialize(
                {"evidence": [item.__dict__ for item in evidence_records]}
            )
            prompt = _SYNTHESIS_PROMPT
            self.session.messages.append(
                {"role": "user", "content": prompt + "\n" + evidence}
            )
            self.synthesis_prompt_appended = True
        try:
            conclusion_model = SubmitSynthesisArgs
            conclusion = _request_call(
                self.session,
                self.client,
                "submit_synthesis",
                conclusion_model,
                DEFAULT_MODEL,
                _max_turns=8,
            )
            conclusion = validate_conclusion(conclusion, scientific.report)
        except _HostedArgumentsValidationError as error:
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


def clone_prepared_controller(
    snapshot: PreparedScientificSnapshot,
    *,
    client: Any,
    executors: dict[str, Any],
) -> BoundedWorkflowController:
    """Create one objective-clean controller from a deep-isolated scientific snapshot."""
    if type(snapshot) is not PreparedScientificSnapshot:
        raise TypeError("An exact prepared scientific snapshot is required.")
    current_digest = _prepared_snapshot_digest(
        snapshot.messages, snapshot.state, snapshot.plan,
        snapshot.stage_results, snapshot.report, snapshot.turn_count,
    )
    if current_digest != getattr(snapshot, "_canonical_digest", None):
        raise ValueError("Prepared scientific snapshot was tampered after construction.")
    PreparedScientificSnapshot(
        messages=snapshot.messages,
        state=snapshot.state,
        plan=snapshot.plan,
        stage_results=snapshot.stage_results,
        report=snapshot.report,
        turn_count=snapshot.turn_count,
    )
    required_executors = set(STAGES) | {"build_workflow_report"}
    if (
        type(executors) is not dict
        or set(executors) != required_executors
        or not all(callable(value) for value in executors.values())
    ):
        raise ValueError("Prepared executors must match the fixed workflow.")
    session = AgentSession(
        messages=list(deepcopy(snapshot.messages)),
        state=deepcopy(snapshot.state),
        turn_count=snapshot.turn_count,
    )
    plan = WorkflowPlan.model_validate(deepcopy(snapshot.plan.model_dump()))
    stage_results = list(deepcopy(snapshot.stage_results))
    report = deepcopy(snapshot.report)
    return BoundedWorkflowController(
        session=session,
        client=client,
        executors=dict(executors),
        plan=plan,
        stage_results=stage_results,
        report=report,
        objective_required=True,
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
