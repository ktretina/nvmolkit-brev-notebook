"""One bounded nvMolKit workflow with an evidence-linked, schema-checked Nemotron conclusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

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
_SYNTHESIS_PROMPT = (
    "Produce one detailed PhD-level but presentation-readable integrated interpretation. Cite only supplied evidence keys; use no numeric literals in authored text. "
    "Do not infer binding, activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimental truth. State that force-field energies compare conformers only within each molecule. "
    "Exact quantities, three-dimensional methods, ETKDGv3, and MMFF94 are rendered separately from canonical evidence."
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
    "submit_synthesis": "Submit one evidence-cited qualitative scientific synthesis.",
}


class ToolCallError(RuntimeError):
    """A secret-safe failure in the bounded hosted or scientific loop."""


class _HostedArgumentsValidationError(ToolCallError):
    pass


class ConclusionValidationError(ToolCallError):
    def __init__(self, report: WorkflowReport):
        super().__init__("Nemotron synthesis failed validation; canonical evidence is preserved.")
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
    conclusion: SubmitSynthesisArgs
    stage_results: tuple[StageResult, ...]
    turn_count: int = 8


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


def validate_conclusion(conclusion: SubmitSynthesisArgs, report: WorkflowReport) -> SubmitSynthesisArgs:
    """Check the synthesis schema and evidence links, not the truth of qualitative prose."""
    themes = [section.theme for section in conclusion.sections]
    cited = {key for section in conclusion.sections for key in section.evidence_keys}
    report_keys = tuple(record.key for record in report.evidence)
    known = set(report_keys)
    prose = conclusion.headline + "".join(section.prose for section in conclusion.sections)
    valid = set(themes) == set(_REQUIRED_CONCLUSION_EVIDENCE) and len(themes) == len(set(themes))
    valid &= report_keys == EvidenceKey.__args__ and cited == known
    valid &= all(_REQUIRED_CONCLUSION_EVIDENCE[item.theme] <= set(item.evidence_keys) for item in conclusion.sections)
    valid &= not any(character.isdigit() for character in prose)
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
) -> BaseModel:
    """Request, validate, and append exactly one forced hosted call."""
    if session.turn_count >= 8 or (session.turn_count == 7 and expected_name != "submit_synthesis"):
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
            max_tokens={"submit_synthesis": 1800, "submit_workflow_plan": 900}.get(expected_name, 400),
            stream=False,
        )
    except Exception as error:
        _raise_request_error(error)

    try:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        calls = getattr(message, "tool_calls", None)
        if content is not None and (
            not isinstance(content, str) or content.strip()
        ):
            raise ToolCallError(
                "Hosted assistant content was returned before the required tool call."
            )
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
    except ValidationError:
        raise _HostedArgumentsValidationError(
            "The hosted tool arguments failed strict validation."
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


def run_scientific_loop(
    user_goal: str,
    api_key: str,
    *,
    client: Any = None,
    executors: dict[str, Any] | None = None,
    state: WorkflowState | None = None,
    progress_callback: ProgressCallback | None = None,
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
    stage_results: list[StageResult] = []
    try:
        plan = _request_call(
            session, active_client, "submit_workflow_plan", WorkflowPlan, DEFAULT_MODEL
        )
    except Exception as error:
        _emit_progress(progress_callback, "failure", str(error))
        raise
    _emit_progress(progress_callback, "plan", plan)
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
        try:
            arguments = _request_call(
                session, active_client, stage, TOOL_ARGUMENT_MODELS[stage], DEFAULT_MODEL
            )
        except Exception as error:
            _emit_progress(progress_callback, "failure", str(error))
            raise
        try:
            result = active_executors[stage](
                session.state, **_executor_arguments(stage, arguments)
            )
        except Exception:
            _emit_progress(progress_callback, "failure", "The scientific executor failed.")
            raise ToolCallError("The scientific executor failed.") from None
        if not isinstance(result, StageResult) or result.stage != stage:
            _emit_progress(progress_callback, "failure", "The scientific executor returned an invalid stage result.")
            raise ToolCallError("The scientific executor returned an invalid stage result.")
        if session.state.phase is not POST_STAGE_PHASES[stage]:
            _emit_progress(progress_callback, "failure", "The scientific executor left the workflow out of phase.")
            raise ToolCallError("The scientific executor left the workflow out of phase.")
        stage_results.append(result)
        try:
            _append_tool_result(
                session,
                {
                    "stage": stage,
                    "decision_basis": getattr(arguments, "decision_basis", None),
                    "summary": result.summary,
                },
            )
        except Exception as error:
            _emit_progress(progress_callback, "failure", str(error))
            raise
        _emit_progress(progress_callback, "stage", {"result": result, "arguments": arguments})

    if session.state.phase is not WorkflowPhase.OPTIMIZED or session.turn_count != 7:
        raise ToolCallError("The scientific workflow did not complete exactly once.")
    try:
        report = active_executors["build_workflow_report"](session.state)
    except Exception:
        _emit_progress(progress_callback, "failure", "The scientific report could not be built.")
        raise ToolCallError("The scientific report could not be built.") from None
    if not isinstance(report, WorkflowReport):
        _emit_progress(progress_callback, "failure", "The scientific report was invalid.")
        raise ToolCallError("The scientific report was invalid.")
    return ScientificLoopResult(tuple(session.messages), report, plan, tuple(stage_results), session.turn_count)


_STAGE_METRICS = {
    "inspect_library": ("raw_count", "valid_count", "invalid_count", "invalid_ids", "preview_count"),
    "generate_morgan_fingerprints": ("molecule_count", "active_bits_min", "active_bits_median", "active_bits_max"),
    "measure_tanimoto_similarity": ("q1", "median", "q3", "p90", "max", "most_similar_nonidentical_pair"),
    "discover_fused_butina_clusters": ("cluster_cutoff", "cluster_count", "singleton_count", "largest_cluster_sizes"),
    "embed_representative_conformers": ("selected_representative_count", "selection_shortfall", "generated_conformer_count", "partial_embedding_ids", "zero_embedding_ids"),
    "optimize_conformers_mmff94": ("attempted_conformer_count", "converged_conformer_count", "unconverged_conformer_count"),
}


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
        display(figure)


def _compact_evidence(report: WorkflowReport) -> str:
    try:
        e = {record.key: json.loads(record.payload_json) for record in report.evidence}
    except (TypeError, json.JSONDecodeError):
        raise ToolCallError(_SERIALIZATION_ERROR) from None
    p1, p2, p3, p4, p5, p6 = (e.get(f"E0{i}", {}) for i in range(1, 7))
    representatives = [item.get("molecule_id") for item in p5.get("representatives", [])]
    lines = [
        "## Canonical evidence",
        f"### E01\nRaw/valid/invalid/previewed: {p1.get('raw_count', 'n/a')} / {p1.get('valid_count', 'n/a')} / {p1.get('invalid_count', 'n/a')} / {p1.get('preview_count', 'n/a')}; invalid IDs: {p1.get('invalid_ids', [])}.",
        f"### E02\nRadius/size/molecules: {p2.get('fingerprint_radius', 'n/a')} / {p2.get('fingerprint_size_bits', 'n/a')} bits / {p2.get('molecule_count', 'n/a')}; active bits min/median/max: {p2.get('active_bits_min', 'n/a')} / {p2.get('active_bits_median', 'n/a')} / {p2.get('active_bits_max', 'n/a')}.",
        f"### E03\nQ1/median/Q3/P90/max: {p3.get('q1', 'n/a')} / {p3.get('median', 'n/a')} / {p3.get('q3', 'n/a')} / {p3.get('p90', 'n/a')} / {p3.get('max_off_diagonal', 'n/a')}; most-similar pair: {p3.get('most_similar_pair', {})}.",
        f"### E04\nCutoff/clusters/singletons: {p4.get('cutoff', 'n/a')} / {p4.get('cluster_count', 'n/a')} / {p4.get('singleton_count', 'n/a')}; largest sizes: {p4.get('largest_cluster_sizes', [])}.",
        f"### E05\nRequested/selected/shortfall: {p5.get('requested_representative_count', 'n/a')} / {p5.get('selected_representative_count', 'n/a')} / {p5.get('selection_shortfall', 'n/a')}; policy: {p5.get('representative_policy', 'n/a')}; generated: {p5.get('generated_conformer_count', 'n/a')}; partial/zero: {p5.get('partial_embedding_ids', [])} / {p5.get('zero_embedding_ids', [])}; representatives: {representatives}.",
        f"### E06\nAttempted/converged/unconverged: {p6.get('attempted_conformer_count', 'n/a')} / {p6.get('converged_conformer_count', 'n/a')} / {p6.get('unconverged_conformer_count', 'n/a')}.",
        "| Molecule | Conformer | Energy (kcal/mol) |", "|---|---:|---:|",
    ]
    lines.extend(f"| {row.get('molecule_id', 'n/a')} | {row.get('conformer_index', 'n/a')} | {row.get('energy_kcal_mol', 'n/a')} |" for row in p6.get("selected_conformer_records", []))
    return "\n".join(lines)


def _display_conclusion(result: WorkflowResult) -> None:
    from IPython.display import Markdown, display

    sections = "\n\n".join(
        f"### {section.theme.replace('_', ' ').title()}\n{section.prose} "
        f"*Evidence: {', '.join(section.evidence_keys)}*"
        for section in result.conclusion.sections
    )
    rendered = f"## Evidence-linked, schema-checked conclusion\n### {result.conclusion.headline}\nPython verifies its schema, evidence references, and exact rendered metrics; Nemotron's qualitative interpretation is not automatically fact-verified.\nPython-rendered methods: 3D conformers use ETKDGv3; energies use MMFF94.\n{_compact_evidence(result.report)}\n\n{sections}"
    display(Markdown(rendered))


def run_workflow(
    user_goal: str,
    api_key: str,
    display_events: bool = True,
    *,
    client: Any = None,
    executors: dict[str, Any] | None = None,
    state: WorkflowState | None = None,
) -> WorkflowResult:
    """Run the chain, then schema-check one evidence-linked synthesis without fact-verifying its prose."""
    _validate_api_key(api_key)
    active_client = client or _client(api_key)
    progress_callback = _display_progress_event if display_events else None
    scientific = run_scientific_loop(
        user_goal, api_key, client=active_client, executors=executors, state=state,
        progress_callback=progress_callback,
    )
    session = AgentSession(list(scientific.messages), state or WorkflowState(), 7)
    evidence = _serialize({"evidence": [item.__dict__ for item in scientific.report.evidence]})
    session.messages.append({"role": "user", "content": _SYNTHESIS_PROMPT + "\n" + evidence})
    try:
        conclusion = _request_call(session, active_client, "submit_synthesis", SubmitSynthesisArgs, DEFAULT_MODEL)
        conclusion = validate_conclusion(conclusion, scientific.report)
    except ConclusionValidationError:
        if display_events:
            from IPython.display import Markdown, display
            display(Markdown(_compact_evidence(scientific.report)))
        raise
    except _HostedArgumentsValidationError:
        if display_events:
            from IPython.display import Markdown, display
            display(Markdown(_compact_evidence(scientific.report)))
        raise ConclusionValidationError(scientific.report) from None
    except Exception as error:
        _emit_progress(progress_callback, "failure", str(error))
        raise
    result = WorkflowResult(tuple(session.messages), scientific.report, scientific.plan, conclusion, scientific.stage_results, session.turn_count)
    if display_events:
        _display_conclusion(result)
    return result
