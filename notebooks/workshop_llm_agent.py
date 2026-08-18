"""Bounded hosted policy and analysis helpers for ACS workshop Modules 2 and 3.

Module 2 accepts only structured policy values, then renders its executable
function in Python. Module 3 remains a separately bounded analysis controller.
"""

from __future__ import annotations

import ast
import csv
import getpass
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NEMOTRON_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
WORKSHOP_AGENT_VERSION = "2026.08.18.4"
WORKSHOP_MODE_ENV = "NVMOLKIT_WORKSHOP_MODE"
_PANEL_CHILD_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "VIRTUAL_ENV",
    }
)
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted NVIDIA Developer API key beginning with "
    "nvapi-. Generate it from the Nemotron model page on build.nvidia.com."
)


class WorkshopAgentError(RuntimeError):
    """Raised when a bounded hosted or local agent step cannot be completed."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NeighborhoodPolicy(_StrictModel):
    """The complete, non-executable hosted decision surface for Module 2."""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, populate_by_name=True
    )

    missing_anchor: Literal["raise", "skip"] = Field(alias="MISSING_ANCHOR")
    invalid_matrix: Literal["raise", "skip"] = Field(alias="INVALID_MATRIX")
    missing_anchor_explanation: str = Field(
        min_length=12, max_length=240, alias="MISSING_ANCHOR_EXPLANATION"
    )
    invalid_matrix_explanation: str = Field(
        min_length=12, max_length=240, alias="INVALID_MATRIX_EXPLANATION"
    )


@dataclass(frozen=True)
class NeighborhoodImplementation:
    """A Python-rendered Module 2 implementation and its policy receipt."""

    label: Literal["reference", "hosted_nemotron"]
    policy: NeighborhoodPolicy
    function_source: str


class PanelStrategy(_StrictModel):
    title: str = Field(min_length=5)
    approach: str = Field(min_length=40)
    property_coverage_measure: str = Field(min_length=20)
    cluster_balance: str = Field(min_length=20)
    tradeoff: str = Field(min_length=20)


class PanelPlan(_StrictModel):
    data_observations: list[str] = Field(min_length=2, max_length=5)
    strategies: list[PanelStrategy] = Field(min_length=2, max_length=2)
    recommended_strategy: int = Field(ge=1, le=2)
    recommendation_reason: str = Field(min_length=30)


class PanelAudit(_StrictModel):
    result_assessment: str = Field(min_length=40)
    surprising_result: str = Field(min_length=20)
    scientific_boundaries: str = Field(min_length=30)
    next_iteration: str = Field(min_length=30)


@dataclass(frozen=True)
class AgentAttempt:
    number: int
    source_file: str
    return_code: int | None
    elapsed_seconds: float
    passed: bool
    message: str
    stdout_tail: str
    stderr_tail: str
    implementation_summary: str
    expected_tradeoffs: tuple[str, ...]


@dataclass(frozen=True)
class PanelAgentRun:
    success: bool
    approved_strategy: int
    attempts: tuple[AgentAttempt, ...]
    analysis_path: Path
    panel_path: Path
    report_path: Path
    trace_path: Path
    audit: PanelAudit | None


_TOOL_DESCRIPTIONS = {
    "submit_neighborhood_policy": (
        "Return only the two bounded policies and one concise explanation for each. "
        "Do not return executable source."
    ),
    "submit_panel_plan": (
        "Return exactly two scientifically defensible panel-design strategies and "
        "recommend one after inspecting the supplied bounded data profile."
    ),
    "submit_panel_audit": (
        "Inspect validated panel-design receipts and report the result, one surprising "
        "or important observation, the scientific boundaries, and a next iteration."
    ),
}


def get_workshop_api_key(api_key: str | None = None, *, prompt: bool = True) -> str:
    """Return a hidden hosted key without imposing demo_agent's full GPU preflight."""
    candidate = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
    candidate = candidate.strip()
    if not candidate and prompt:
        candidate = getpass.getpass(
            "Hosted NVIDIA Developer API key (nvapi-; input hidden): "
        ).strip()
    if not candidate or not candidate.startswith("nvapi-"):
        raise ValueError(AUTH_GUIDANCE)
    return candidate


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, max_retries=0)


def _tool_definition(name: str, response_model: type[BaseModel]) -> dict[str, Any]:
    schema = response_model.model_json_schema(by_alias=True)
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _TOOL_DESCRIPTIONS[name],
            "strict": True,
            "parameters": schema,
        },
    }


def _structured_request(
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    tool_name: str,
    response_model: type[_StrictModel],
    max_tokens: int,
    client: Any = None,
) -> _StrictModel:
    """Make one forced schema-checked Nemotron tool call."""
    active_client = client or _client(get_workshop_api_key(api_key, prompt=False))
    request_failure: str | None = None
    response = None
    try:
        response = active_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[_tool_definition(tool_name, response_model)],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            extra_body=NEMOTRON_EXTRA_BODY,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
        )
    except (AuthenticationError, PermissionDeniedError):
        request_failure = AUTH_GUIDANCE
    except Exception:
        request_failure = (
            "The hosted Nemotron request failed; check network and model availability."
        )
    if request_failure == AUTH_GUIDANCE:
        raise ValueError(request_failure)
    if request_failure:
        raise WorkshopAgentError(request_failure)

    schema_failure = False
    result = None
    try:
        assert response is not None
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, (list, tuple)) and len(calls) == 1:
            call = calls[0]
            function = getattr(call, "function", None)
            if function is None or getattr(function, "name", None) != tool_name:
                raise WorkshopAgentError(
                    "The hosted response called an unexpected tool."
                )
            payload = json.loads(function.arguments)
        else:
            # Some compatible endpoints return the forced arguments as JSON content.
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                raise WorkshopAgentError(
                    "The hosted response did not contain tool arguments."
                )
            payload = json.loads(content)
        result = response_model.model_validate(payload)
    except (AttributeError, IndexError, json.JSONDecodeError, ValidationError):
        schema_failure = True
    if schema_failure:
        raise WorkshopAgentError(
            "The hosted response did not satisfy the workshop's required schema."
        )
    assert result is not None
    return result


def workshop_mode() -> Literal["interactive", "reference"]:
    """Return the only supported Module 2 execution mode."""
    mode = os.environ.get(WORKSHOP_MODE_ENV, "interactive").strip().lower()
    if mode not in {"interactive", "reference"}:
        raise ValueError(f"{WORKSHOP_MODE_ENV} must be interactive or reference.")
    return mode  # type: ignore[return-value]


_REFERENCE_NEIGHBORHOOD_POLICY = NeighborhoodPolicy.model_validate(
    {
        "MISSING_ANCHOR": "raise",
        "INVALID_MATRIX": "raise",
        "MISSING_ANCHOR_EXPLANATION": "Stop when a named anchor is absent from the fixed sample.",
        "INVALID_MATRIX_EXPLANATION": "Stop when similarity rows cannot be matched to the input records.",
    }
)


def _render_neighborhood_function(policy: NeighborhoodPolicy) -> str:
    """Render the sole executable Module 2 function from two validated choices."""
    missing_action = (
        "raise ValueError(f'Anchor not found: {term}')"
        if policy.missing_anchor == "raise"
        else "continue"
    )
    invalid_action = (
        "raise RuntimeError(f'Unexpected similarity matrix: {similarities.shape}')"
        if policy.invalid_matrix == "raise"
        else "continue"
    )
    return textwrap.dedent(
        f'''\
        def build_neighborhood_atlas(records, anchor_terms, radii=(2, 3), fp_bits=1024, top_k=10):
            """Build a deterministic, multi-radius structural-neighborhood atlas."""
            required = {{"_mol", "name", "canonical_ikey", "reframedb_url"}}
            missing = required - set(records.columns)
            if missing:
                raise ValueError(f"records is missing {{sorted(missing)}}")
            query_indices = []
            for term in anchor_terms:
                matches = records[records["name"].str.contains(term, case=False, regex=False)]
                if matches.empty:
                    {missing_action}
                else:
                    query_indices.append(int(matches.index[0]))
            molecules = records["_mol"].tolist()
            query_molecules = records.loc[query_indices, "_mol"].tolist()
            rows = []
            for radius in radii:
                library_fps = make_fingerprints(molecules, radius=radius, fp_bits=fp_bits)
                query_fps = make_fingerprints(query_molecules, radius=radius, fp_bits=fp_bits)
                similarities = tanimoto_matrix(query_fps, library_fps)
                if similarities.shape != (len(query_indices), len(records)) or not np.isfinite(similarities).all() or not ((similarities >= 0) & (similarities <= 1)).all():
                    {invalid_action}
                for query_position, query_index in enumerate(query_indices):
                    query = records.loc[query_index]
                    order = np.argsort(-similarities[query_position], kind="stable")
                    order = [int(index) for index in order if records.iloc[index]["canonical_ikey"] != query["canonical_ikey"]][:top_k]
                    for rank, library_index in enumerate(order, start=1):
                        neighbor = records.iloc[library_index]
                        rows.append({{"radius": int(radius), "query": query["name"], "query_ikey": query["canonical_ikey"], "rank": rank, "neighbor": neighbor["name"], "neighbor_ikey": neighbor["canonical_ikey"], "tanimoto": float(similarities[query_position, library_index]), "profile": neighbor["reframedb_url"]}})
            columns = ["radius", "query", "query_ikey", "rank", "neighbor", "neighbor_ikey", "tanimoto", "profile"]
            return pd.DataFrame(rows, columns=columns).sort_values(["radius", "query", "rank"], ignore_index=True)
        '''
    )


def validate_rendered_neighborhood_function(
    policy: NeighborhoodPolicy, function_source: str
) -> str:
    """Validate only the exact source rendered locally for a validated policy."""
    expected = _render_neighborhood_function(policy)
    if function_source != expected:
        raise WorkshopAgentError(
            "Module 2 accepts only locally rendered function source."
        )
    try:
        tree = ast.parse(function_source)
    except SyntaxError as exc:
        raise WorkshopAgentError(
            "The local Module 2 renderer produced invalid syntax."
        ) from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise WorkshopAgentError(
            "The local Module 2 renderer produced an invalid function."
        )
    return function_source


def select_neighborhood_implementation(
    prompt: str,
    *,
    mode: Literal["interactive", "reference"] | None = None,
    api_key: str | None = None,
    client: Any = None,
) -> NeighborhoodImplementation:
    """Choose a bounded policy and return only Python-owned rendered source."""
    active_mode = workshop_mode() if mode is None else mode
    if active_mode not in {"interactive", "reference"}:
        raise ValueError(f"{WORKSHOP_MODE_ENV} must be interactive or reference.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty Module 2 policy prompt is required.")
    if active_mode == "reference":
        policy = _REFERENCE_NEIGHBORHOOD_POLICY
        label: Literal["reference", "hosted_nemotron"] = "reference"
    else:
        protected_key = get_workshop_api_key(api_key, prompt=True)
        policy = _structured_request(
            api_key=protected_key,
            client=client,
            system_prompt=(
                "You are a bounded chemistry policy assistant. Return exactly two "
                "raise-or-skip policies and concise explanations. Never return code."
            ),
            user_prompt=prompt.strip(),
            tool_name="submit_neighborhood_policy",
            response_model=NeighborhoodPolicy,
            max_tokens=500,
        )
        label = "hosted_nemotron"
    function_source = _render_neighborhood_function(policy)
    return NeighborhoodImplementation(
        label=label,
        policy=policy,
        function_source=validate_rendered_neighborhood_function(
            policy, function_source
        ),
    )


def bind_neighborhood_builder(
    implementation: NeighborhoodImplementation, namespace: dict[str, Any]
) -> Any:
    """Execute only the exact, Python-rendered Module 2 function."""
    source = validate_rendered_neighborhood_function(
        implementation.policy, implementation.function_source
    )
    local_namespace = dict(namespace)
    exec(compile(source, "<workshop-neighborhood-renderer>", "exec"), local_namespace)
    return local_namespace["build_neighborhood_atlas"]


def _quantile(values: list[float], fraction: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - position) + clean[upper] * (position - lower)


def profile_candidate_csv(path: Path) -> dict[str, Any]:
    """Create a small deterministic profile without sending molecule rows wholesale."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise WorkshopAgentError("The Module 3 candidate CSV is empty.")
    columns = list(rows[0])
    keys = [row.get("canonical_ikey", "") for row in rows]
    statuses = Counter(row.get("status", "<missing>") or "<missing>" for row in rows)
    descriptor_quantiles: dict[str, dict[str, float | None]] = {}
    for column in ("MolWt", "cLogP", "TPSA"):
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue
        descriptor_quantiles[column] = {
            "p05": _quantile(values, 0.05),
            "median": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
        }
    return {
        "row_count": len(rows),
        "columns": columns,
        "unique_canonical_ikeys": len(set(keys)),
        "duplicate_canonical_ikey_rows": len(keys) - len(set(keys)),
        "blank_canonical_ikey_rows": sum(not key.strip() for key in keys),
        "status_counts": dict(statuses.most_common(10)),
        "descriptor_quantiles": descriptor_quantiles,
    }


def minimum_pairwise_distance(similarity_matrix: Any) -> float:
    """Return the minimum upper-triangle value of one minus similarity."""
    try:
        rows = [list(row) for row in similarity_matrix]
    except TypeError:
        raise TypeError("similarity_matrix must be a square numeric matrix.") from None
    size = len(rows)
    if size < 2 or any(len(row) != size for row in rows):
        raise ValueError("similarity_matrix must be square with at least two rows.")
    distances: list[float] = []
    for row_index in range(size):
        for column_index in range(row_index + 1, size):
            value = rows[row_index][column_index]
            if isinstance(value, bool):
                raise TypeError("similarity_matrix values must be numeric.")
            try:
                similarity = float(value)
            except (TypeError, ValueError):
                raise TypeError("similarity_matrix values must be numeric.") from None
            if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
                raise ValueError(
                    "similarity_matrix values must be finite and in [0, 1]."
                )
            distances.append(1.0 - similarity)
    return min(distances)


def _descriptor_values(records: Any, column: str) -> list[float]:
    try:
        raw_values = records[column]
    except (KeyError, TypeError):
        try:
            raw_values = [row[column] for row in records]
        except (KeyError, TypeError):
            raise ValueError(
                "Descriptor records must contain MolWt, cLogP, and TPSA."
            ) from None
    values: list[float] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TypeError("Descriptor values must be numeric.")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise TypeError("Descriptor values must be numeric.") from None
        if not math.isfinite(numeric):
            raise ValueError("Descriptor values must be finite.")
        values.append(numeric)
    if not values:
        raise ValueError("Descriptor records must not be empty.")
    return values


def descriptor_range_coverage(candidates: Any, selected: Any) -> float:
    """Return mean selected/candidate range for MolWt, cLogP, and TPSA."""
    normalized_ranges: list[float] = []
    for column in ("MolWt", "cLogP", "TPSA"):
        candidate_values = _descriptor_values(candidates, column)
        selected_values = _descriptor_values(selected, column)
        candidate_range = max(candidate_values) - min(candidate_values)
        selected_range = max(selected_values) - min(selected_values)
        if selected_range > candidate_range + 1e-12:
            raise ValueError(
                "Selected descriptor values must come from the candidates."
            )
        normalized_ranges.append(
            1.0 if candidate_range == 0.0 else selected_range / candidate_range
        )
    return sum(normalized_ranges) / len(normalized_ranges)


def validate_panel_analysis_source(
    source: str, *, approved_strategy: int, expected_panel_size: int
) -> str:
    """Accept only the exact controller-rendered implementation."""
    expected = _render_panel_analysis(approved_strategy, expected_panel_size)
    if source != expected:
        raise WorkshopAgentError(
            "Module 3 accepts only the exact controller-rendered implementation."
        )
    try:
        compile(source, "<workshop-panel-renderer>", "exec")
    except SyntaxError:
        raise WorkshopAgentError(
            "The exact controller-rendered implementation is not valid Python."
        ) from None
    return source


def _require_plain_artifact(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise WorkshopAgentError(
            "Module 3 artifacts must be present as regular files in the workspace."
        )


def _strict_report_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkshopAgentError(f"report.json {field} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise WorkshopAgentError(f"report.json {field} must be finite.")
    return number


def _rdkit_similarity_matrix(
    rows: list[dict[str, str]], radius: int, fp_bits: int
) -> list[list[float]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        raise WorkshopAgentError(
            "RDKit is required for independent Module 3 artifact validation."
        ) from None
    molecules = [Chem.MolFromSmiles(str(row.get("smile", ""))) for row in rows]
    if any(molecule is None for molecule in molecules):
        raise WorkshopAgentError("Module 3 artifacts contain an invalid SMILES value.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    fingerprints = list(generator.GetFingerprints(molecules, numThreads=0))
    return [
        [
            float(value)
            for value in DataStructs.BulkTanimotoSimilarity(query, fingerprints)
        ]
        for query in fingerprints
    ]


def _validate_panel_artifacts_snapshot(
    workdir: Path,
    *,
    expected_panel_size: int,
    not_before_ns: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate artifacts and return a receipt plus the exact report snapshot."""
    if type(expected_panel_size) is not int or expected_panel_size != 24:
        raise ValueError("expected_panel_size must be exactly 24.")
    if not_before_ns is not None and (
        type(not_before_ns) is not int or not_before_ns < 0
    ):
        raise TypeError("not_before_ns must be a non-negative integer or None.")

    resolved = Path(workdir).resolve()
    candidate_path = resolved / "reframe_candidates.csv"
    panel_path = resolved / "panel.csv"
    report_path = resolved / "report.json"
    for path in (candidate_path, panel_path, report_path):
        _require_plain_artifact(path)
    if not_before_ns is not None and any(
        path.stat(follow_symlinks=False).st_mtime_ns < not_before_ns
        for path in (panel_path, report_path)
    ):
        raise WorkshopAgentError(
            "Module 3 output artifacts are not from the current execution."
        )

    try:
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            candidates = list(csv.DictReader(handle))
        with panel_path.open(newline="", encoding="utf-8") as handle:
            panel = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        raise WorkshopAgentError("Module 3 CSV artifacts could not be read.") from None

    if len(candidates) != 96:
        raise WorkshopAgentError("reframe_candidates.csv must contain exactly 96 rows.")
    candidate_required_columns = {
        "smile",
        "canonical_ikey",
        "name",
        "status",
        "reframedb_url",
        "MolWt",
        "cLogP",
        "TPSA",
    }
    if not candidates or candidate_required_columns - set(candidates[0]):
        raise WorkshopAgentError(
            "reframe_candidates.csv is missing required candidate columns."
        )
    candidate_keys = [row.get("canonical_ikey", "").strip() for row in candidates]
    if any(not key for key in candidate_keys) or len(set(candidate_keys)) != 96:
        raise WorkshopAgentError(
            "The candidate input must contain exactly 96 unique connectivity keys."
        )
    if len(panel) != expected_panel_size:
        raise WorkshopAgentError(
            f"panel.csv has {len(panel)} rows; expected {expected_panel_size}."
        )
    required_columns = {
        "smile",
        "canonical_ikey",
        "name",
        "reframedb_url",
        "MolWt",
        "cLogP",
        "TPSA",
        "selection_reason",
        "method_cluster",
        "selection_order",
    }
    if not panel or required_columns - set(panel[0]):
        raise WorkshopAgentError("panel.csv is missing required panel-design columns.")
    panel_keys = [row.get("canonical_ikey", "").strip() for row in panel]
    if len(set(panel_keys)) != expected_panel_size or not set(panel_keys) < set(
        candidate_keys
    ):
        raise WorkshopAgentError(
            "Panel connectivity keys must be a unique strict subset of the input."
        )
    candidate_by_key = {row["canonical_ikey"].strip(): row for row in candidates}
    for row in panel:
        source_row = candidate_by_key[row["canonical_ikey"].strip()]
        if (
            row.get("smile") != source_row.get("smile")
            or row.get("name") != source_row.get("name")
            or row.get("reframedb_url", "").strip()
            != source_row.get("reframedb_url", "").strip()
        ):
            raise WorkshopAgentError(
                "Panel membership or ReFRAME provenance does not match the input."
            )
        try:
            descriptors_match = all(
                math.isclose(
                    float(row[column]),
                    float(source_row[column]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                for column in ("MolWt", "cLogP", "TPSA")
            )
        except (TypeError, ValueError):
            descriptors_match = False
        if not descriptors_match:
            raise WorkshopAgentError(
                "Panel descriptors do not match the candidate input."
            )
    try:
        orders = sorted(int(row["selection_order"]) for row in panel)
    except (TypeError, ValueError):
        raise WorkshopAgentError("selection_order must contain integers.") from None
    if orders != list(range(1, expected_panel_size + 1)):
        raise WorkshopAgentError("selection_order must be contiguous and one-based.")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise WorkshopAgentError("report.json is not valid JSON.") from None
    required_report_keys = {
        "seed",
        "backend",
        "parameters",
        "candidate_count",
        "panel_count",
        "unique_ikeys",
        "descriptor_quantiles",
        "pairwise_similarity",
        "cluster_coverage",
        "acceptance",
        "limitations",
        "files",
    }
    if not isinstance(report, dict) or set(report) != required_report_keys:
        raise WorkshopAgentError(
            "report.json must contain exactly the required fields."
        )
    if (
        report["candidate_count"] != 96
        or report["panel_count"] != 24
        or report["unique_ikeys"] != 24
        or type(report["candidate_count"]) is not int
        or type(report["panel_count"]) is not int
        or type(report["unique_ikeys"]) is not int
    ):
        raise WorkshopAgentError("report.json row counts do not match the artifacts.")
    if report["seed"] != 2026 or type(report["seed"]) is not int:
        raise WorkshopAgentError("report.json seed must equal 2026.")
    if report["backend"] not in {
        "nvmolkit-gpu",
        "rdkit-cpu-reference (not GPU evidence)",
    }:
        raise WorkshopAgentError("report.json backend is not approved.")
    if report["files"] != {
        "analysis": "analysis.py",
        "panel": "panel.csv",
        "report": "report.json",
    }:
        raise WorkshopAgentError(
            "report.json files must name only the required artifacts."
        )
    parameters = report["parameters"]
    if not isinstance(parameters, dict):
        raise WorkshopAgentError("report.json parameters must be a mapping.")
    radius = parameters.get("radius")
    fp_bits = parameters.get("fp_bits")
    if type(radius) is not int or radius not in (2, 3):
        raise WorkshopAgentError("report.json radius is not approved.")
    if type(fp_bits) is not int or fp_bits not in (1024, 2048):
        raise WorkshopAgentError("report.json fp_bits is not approved.")
    strategy = parameters.get("strategy")
    approved_parameters = {
        "cluster_aware_max_min": (2, 1024),
        "descriptor_seeded_max_min": (3, 2048),
    }
    if approved_parameters.get(strategy) != (radius, fp_bits):
        raise WorkshopAgentError(
            "report.json strategy parameters are not an allow-listed combination."
        )
    if parameters.get("baseline") != "first_24_stable_source_rows":
        raise WorkshopAgentError("report.json does not name the fixed baseline.")
    if not isinstance(report["descriptor_quantiles"], dict) or set(
        report["descriptor_quantiles"]
    ) != {"candidate", "panel"}:
        raise WorkshopAgentError("report.json descriptor_quantiles are malformed.")
    if not isinstance(report["pairwise_similarity"], dict):
        raise WorkshopAgentError("report.json pairwise_similarity is malformed.")
    if not isinstance(report["limitations"], list) or not all(
        isinstance(item, str) and item.strip() for item in report["limitations"]
    ):
        raise WorkshopAgentError("report.json limitations are malformed.")

    candidate_similarity = _rdkit_similarity_matrix(candidates, radius, fp_bits)
    selected_similarity = _rdkit_similarity_matrix(panel, radius, fp_bits)
    baseline = candidates[:expected_panel_size]
    baseline_similarity = [
        row[:expected_panel_size] for row in candidate_similarity[:expected_panel_size]
    ]
    baseline_minimum_distance = minimum_pairwise_distance(baseline_similarity)
    selected_minimum_distance = minimum_pairwise_distance(selected_similarity)
    baseline_descriptor_coverage = descriptor_range_coverage(candidates, baseline)
    selected_descriptor_coverage = descriptor_range_coverage(candidates, panel)
    tolerance = 1e-12
    passed = (
        selected_minimum_distance + tolerance >= baseline_minimum_distance
        and selected_descriptor_coverage + tolerance >= baseline_descriptor_coverage
        and (
            selected_minimum_distance > baseline_minimum_distance + tolerance
            or selected_descriptor_coverage > baseline_descriptor_coverage + tolerance
        )
    )
    if not passed:
        raise WorkshopAgentError(
            "The selected panel does not meet the fixed first-24 baseline contract."
        )

    acceptance = report["acceptance"]
    expected_acceptance = {
        "baseline_minimum_distance": baseline_minimum_distance,
        "selected_minimum_distance": selected_minimum_distance,
        "baseline_descriptor_coverage": baseline_descriptor_coverage,
        "selected_descriptor_coverage": selected_descriptor_coverage,
    }
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
        raise WorkshopAgentError("report.json acceptance receipt did not pass.")
    for field, expected in expected_acceptance.items():
        observed = _strict_report_number(acceptance.get(field), f"acceptance.{field}")
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
            raise WorkshopAgentError(
                "report.json acceptance metrics do not match independent validation."
            )
    reported_minimum_distance = _strict_report_number(
        report["pairwise_similarity"].get("minimum_distance"),
        "pairwise_similarity.minimum_distance",
    )
    if not math.isclose(
        reported_minimum_distance,
        selected_minimum_distance,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise WorkshopAgentError(
            "report.json minimum distance does not match independent validation."
        )
    receipt = {
        "candidate_count": 96,
        "panel_count": 24,
        "strict_subset": True,
        "baseline_minimum_distance": baseline_minimum_distance,
        "selected_minimum_distance": selected_minimum_distance,
        "baseline_descriptor_coverage": baseline_descriptor_coverage,
        "selected_descriptor_coverage": selected_descriptor_coverage,
        "acceptance_passed": True,
    }
    report_snapshot = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return receipt, report_snapshot


def validate_panel_artifacts(
    workdir: Path,
    *,
    expected_panel_size: int,
    not_before_ns: int | None = None,
) -> dict[str, Any]:
    """Validate the fixed 96-to-24 artifact contract independently."""
    receipt, _ = _validate_panel_artifacts_snapshot(
        workdir,
        expected_panel_size=expected_panel_size,
        not_before_ns=not_before_ns,
    )
    return receipt


def _render_panel_analysis(approved_strategy: int, expected_panel_size: int) -> str:
    """Render one exact, tested 96-to-24 panel analysis."""
    if approved_strategy not in (1, 2):
        raise ValueError("approved_strategy must be 1 or 2.")
    if type(expected_panel_size) is not int or expected_panel_size != 24:
        raise ValueError("expected_panel_size must be exactly 24.")
    radius = 2 if approved_strategy == 1 else 3
    fp_bits = 1024 if approved_strategy == 1 else 2048
    strategy_name = (
        "cluster_aware_max_min"
        if approved_strategy == 1
        else "descriptor_seeded_max_min"
    )
    return textwrap.dedent(
        f"""\
        from pathlib import Path
        import json

        import numpy as np
        import pandas as pd
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.ML.Cluster import Butina


        SEED = 2026
        CANDIDATE_COUNT = 96
        PANEL_SIZE = 24
        RADIUS = {radius}
        FP_BITS = {fp_bits}
        DISTANCE_CUTOFF = 0.55
        SIMILARITY_TOLERANCE = 1e-5
        STRATEGY = {strategy_name!r}


        def descriptor_summary(frame):
            return {{
                column: {{
                    "p05": float(frame[column].quantile(0.05)),
                    "median": float(frame[column].quantile(0.50)),
                    "p95": float(frame[column].quantile(0.95)),
                }}
                for column in ("MolWt", "cLogP", "TPSA")
            }}


        def minimum_distance(matrix):
            upper = matrix[np.triu_indices(len(matrix), k=1)]
            if len(upper) == 0:
                raise ValueError("At least two rows are required for distance")
            return float(np.min(1.0 - upper))


        def descriptor_coverage(candidates, selected):
            normalized_ranges = []
            for column in ("MolWt", "cLogP", "TPSA"):
                candidate_range = float(
                    candidates[column].max() - candidates[column].min()
                )
                selected_range = float(
                    selected[column].max() - selected[column].min()
                )
                normalized_ranges.append(
                    1.0
                    if candidate_range == 0.0
                    else selected_range / candidate_range
                )
            return float(np.mean(normalized_ranges))


        def rdkit_fingerprints_and_similarity(molecules):
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=RADIUS, fpSize=FP_BITS
            )
            fingerprints = list(
                generator.GetFingerprints(molecules, numThreads=0)
            )
            similarity = np.asarray(
                [
                    DataStructs.BulkTanimotoSimilarity(query, fingerprints)
                    for query in fingerprints
                ],
                dtype=float,
            )
            return fingerprints, similarity


        def rdkit_clusters(fingerprints):
            distances = []
            for row_index in range(1, len(fingerprints)):
                distances.extend(
                    DataStructs.BulkTanimotoSimilarity(
                        fingerprints[row_index],
                        fingerprints[:row_index],
                        returnDistance=True,
                    )
                )
            return Butina.ClusterData(
                distances,
                len(fingerprints),
                DISTANCE_CUTOFF,
                isDistData=True,
                reordering=True,
            )


        df = pd.read_csv("reframe_candidates.csv")
        required_columns = {{
            "smile",
            "canonical_ikey",
            "name",
            "status",
            "reframedb_url",
            "MolWt",
            "cLogP",
            "TPSA",
        }}
        if required_columns - set(df.columns):
            raise ValueError("Candidate input is missing required columns")
        if len(df) != CANDIDATE_COUNT:
            raise ValueError("Candidate input must contain exactly 96 rows")
        if df["canonical_ikey"].isna().any() or df["canonical_ikey"].nunique() != 96:
            raise ValueError("Candidate connectivity keys must be unique")
        if not np.isfinite(df[["MolWt", "cLogP", "TPSA"]].to_numpy(dtype=float)).all():
            raise ValueError("Candidate descriptors must be finite")

        molecules = [Chem.MolFromSmiles(str(smile)) for smile in df["smile"]]
        if any(molecule is None for molecule in molecules):
            raise ValueError("Every candidate SMILES must parse")

        nvmolkit_ready = False
        nvmolkit_fingerprints = None
        try:
            import torch
            from nvmolkit.clustering import fused_butina
            from nvmolkit.fingerprints import MorganFingerprintGenerator
            from nvmolkit.similarity import crossTanimotoSimilarity

            nvmolkit_ready = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):
            nvmolkit_ready = False

        if nvmolkit_ready:
            generator = MorganFingerprintGenerator(radius=RADIUS, fpSize=FP_BITS)
            nvmolkit_fingerprints = generator.GetFingerprints(
                molecules, num_threads=0
            )
            fingerprint_tensor = (
                nvmolkit_fingerprints
                if isinstance(nvmolkit_fingerprints, torch.Tensor)
                else nvmolkit_fingerprints.torch()
            )
            similarity_matrix = crossTanimotoSimilarity(
                nvmolkit_fingerprints, nvmolkit_fingerprints
            ).numpy()
            backend = "nvmolkit-gpu"
            rdkit_fingerprints = None
        else:
            rdkit_fingerprints, similarity_matrix = (
                rdkit_fingerprints_and_similarity(molecules)
            )
            backend = "rdkit-cpu-reference (not GPU evidence)"

        if similarity_matrix.shape != (CANDIDATE_COUNT, CANDIDATE_COUNT):
            raise ValueError("Unexpected all-pairs similarity shape")
        if not np.isfinite(similarity_matrix).all():
            raise ValueError("Similarity matrix contains non-finite values")
        raw_similarity_min = float(similarity_matrix.min())
        raw_similarity_max = float(similarity_matrix.max())
        if (
            raw_similarity_min < -SIMILARITY_TOLERANCE
            or raw_similarity_max > 1.0 + SIMILARITY_TOLERANCE
        ):
            raise ValueError("Similarity matrix is outside the Tanimoto range")
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        descriptor_extrema = []
        for column in ("MolWt", "cLogP", "TPSA"):
            for index in (int(df[column].idxmin()), int(df[column].idxmax())):
                if index not in descriptor_extrema:
                    descriptor_extrema.append(index)

        cluster_labels = np.full(CANDIDATE_COUNT, -1, dtype=int)
        cluster_sizes = np.ones(CANDIDATE_COUNT, dtype=int)
        centroid_indices = list(range(CANDIDATE_COUNT))
        if {approved_strategy} == 1:
            if nvmolkit_ready:
                clusters, _, _ = fused_butina(
                    fingerprint_tensor,
                    cutoff=DISTANCE_CUTOFF,
                    return_centroids=True,
                )
            else:
                clusters = rdkit_clusters(rdkit_fingerprints)
            centroid_indices = []
            for cluster_id, members in enumerate(clusters):
                member_indices = [int(index) for index in members]
                if not member_indices:
                    raise ValueError("Cluster output contains an empty cluster")
                cluster_labels[member_indices] = cluster_id
                cluster_sizes[member_indices] = len(member_indices)
                centroid_indices.append(member_indices[0])
            if (cluster_labels < 0).any():
                raise ValueError("Every candidate must receive a cluster label")

        status_text = df["status"].fillna("").str.lower()
        availability_rank = np.where(
            status_text.str.contains("available", regex=False), 0, 1
        )
        selected_indices = list(descriptor_extrema)
        selected_set = set(selected_indices)
        maximum_similarity = similarity_matrix[:, selected_indices].max(axis=1)

        while len(selected_indices) < PANEL_SIZE:
            if {approved_strategy} == 1:
                pool = [
                    index
                    for index in centroid_indices
                    if index not in selected_set
                ]
                if not pool:
                    pool = [
                        index
                        for index in range(CANDIDATE_COUNT)
                        if index not in selected_set
                    ]
                next_index = min(
                    pool,
                    key=lambda index: (
                        float(maximum_similarity[index]),
                        -int(cluster_sizes[index]),
                        int(availability_rank[index]),
                        str(df.iloc[index]["canonical_ikey"]),
                    ),
                )
            else:
                pool = [
                    index
                    for index in range(CANDIDATE_COUNT)
                    if index not in selected_set
                ]
                next_index = min(
                    pool,
                    key=lambda index: (
                        float(maximum_similarity[index]),
                        int(availability_rank[index]),
                        str(df.iloc[index]["canonical_ikey"]),
                    ),
                )
            selected_indices.append(int(next_index))
            selected_set.add(int(next_index))
            maximum_similarity = np.maximum(
                maximum_similarity, similarity_matrix[:, next_index]
            )

        panel_df = df.iloc[selected_indices].copy().reset_index(drop=True)
        panel_df["selection_order"] = np.arange(1, PANEL_SIZE + 1)
        if {approved_strategy} == 1:
            panel_df["method_cluster"] = cluster_labels[selected_indices]
            panel_df["selection_reason"] = (
                "Descriptor-extrema seed plus cluster-aware max-min selection"
            )
            cluster_coverage = {{
                "candidate_clusters": int(len(set(cluster_labels.tolist()))),
                "panel_clusters": int(
                    len(set(cluster_labels[selected_indices].tolist()))
                ),
            }}
        else:
            panel_df["method_cluster"] = "not_clustered"
            panel_df["selection_reason"] = (
                "Descriptor-extrema seed plus deterministic max-min selection"
            )
            cluster_coverage = {{
                "method": "not_clustered",
                "selected_compounds": PANEL_SIZE,
            }}

        panel_similarity = similarity_matrix[np.ix_(selected_indices, selected_indices)]
        baseline_similarity = similarity_matrix[:PANEL_SIZE, :PANEL_SIZE]
        baseline_df = df.iloc[:PANEL_SIZE]
        baseline_minimum_distance = minimum_distance(baseline_similarity)
        selected_minimum_distance = minimum_distance(panel_similarity)
        baseline_descriptor_coverage = descriptor_coverage(df, baseline_df)
        selected_descriptor_coverage = descriptor_coverage(df, panel_df)
        tolerance = 1e-12
        acceptance_passed = (
            selected_minimum_distance + tolerance >= baseline_minimum_distance
            and selected_descriptor_coverage + tolerance
            >= baseline_descriptor_coverage
            and (
                selected_minimum_distance > baseline_minimum_distance + tolerance
                or selected_descriptor_coverage
                > baseline_descriptor_coverage + tolerance
            )
        )
        if not acceptance_passed:
            raise ValueError(
                "Selected panel did not meet the fixed first-24 baseline contract"
            )
        if len(panel_df) != PANEL_SIZE or panel_df["canonical_ikey"].nunique() != 24:
            raise ValueError("Panel must contain 24 unique connectivity keys")
        if not set(panel_df["canonical_ikey"]) < set(df["canonical_ikey"]):
            raise ValueError("Panel connectivity keys must be a strict subset")

        panel_df.to_csv("panel.csv", index=False)
        upper_triangle = panel_similarity[
            np.triu_indices(PANEL_SIZE, k=1)
        ]
        report = {{
            "seed": SEED,
            "backend": backend,
            "parameters": {{
                "strategy": STRATEGY,
                "radius": RADIUS,
                "fp_bits": FP_BITS,
                "distance_cutoff": (
                    DISTANCE_CUTOFF if {approved_strategy} == 1 else None
                ),
                "baseline": "first_24_stable_source_rows",
                "raw_similarity_range": [
                    raw_similarity_min,
                    raw_similarity_max,
                ],
            }},
            "candidate_count": CANDIDATE_COUNT,
            "panel_count": PANEL_SIZE,
            "unique_ikeys": int(panel_df["canonical_ikey"].nunique()),
            "descriptor_quantiles": {{
                "candidate": descriptor_summary(df),
                "panel": descriptor_summary(panel_df),
            }},
            "pairwise_similarity": {{
                "pair_count": int(len(upper_triangle)),
                "median": float(np.median(upper_triangle)),
                "p95": float(np.quantile(upper_triangle, 0.95)),
                "maximum": float(np.max(upper_triangle)),
                "minimum_distance": selected_minimum_distance,
            }},
            "cluster_coverage": cluster_coverage,
            "acceptance": {{
                "baseline_minimum_distance": baseline_minimum_distance,
                "selected_minimum_distance": selected_minimum_distance,
                "baseline_descriptor_coverage": baseline_descriptor_coverage,
                "selected_descriptor_coverage": selected_descriptor_coverage,
                "passed": True,
            }},
            "limitations": [
                "Fingerprint diversity is representation- and parameter-dependent.",
                "The first 24 rows are a deterministic teaching baseline, not an optimum.",
                "Structural diversity does not establish biological activity or safety.",
            ],
            "files": {{
                "analysis": "analysis.py",
                "panel": "panel.csv",
                "report": "report.json",
            }},
        }}
        Path("report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"Selected {{PANEL_SIZE}} of {{CANDIDATE_COUNT}} candidates "
            f"with {{STRATEGY}} using {{backend}}"
        )
        """
    )


def reference_panel_plan() -> PanelPlan:
    """Return the fixed, Python-owned Module 3 reference plan."""
    return PanelPlan.model_validate(
        {
            "data_observations": [
                "The fixed teaching input contains 96 unique connectivity keys.",
                "The audit compares every selection with the first 24 stable source rows.",
                "MolWt, cLogP, and TPSA ranges measure bounded descriptor coverage.",
            ],
            "strategies": [
                {
                    "title": "Cluster-aware max-min",
                    "approach": (
                        "Seed descriptor extrema, then select separated Butina "
                        "representatives with deterministic cluster-aware tie breaks."
                    ),
                    "property_coverage_measure": (
                        "Mean normalized MolWt, cLogP, and TPSA ranges."
                    ),
                    "cluster_balance": (
                        "Prefer separated cluster representatives and larger clusters "
                        "only after the max-similarity criterion."
                    ),
                    "tradeoff": (
                        "Cluster membership depends on the fingerprint and cutoff."
                    ),
                },
                {
                    "title": "Descriptor-seeded max-min",
                    "approach": (
                        "Seed descriptor extrema, then add deterministic farthest "
                        "fingerprint points from the complete fixed candidate set."
                    ),
                    "property_coverage_measure": (
                        "Mean normalized MolWt, cLogP, and TPSA ranges."
                    ),
                    "cluster_balance": (
                        "Use fingerprint separation without assigning cluster labels."
                    ),
                    "tradeoff": (
                        "Farthest-point selection can favor unusual structural features."
                    ),
                },
            ],
            "recommended_strategy": 2,
            "recommendation_reason": (
                "The descriptor-seeded max-min strategy directly targets the fixed "
                "distance contract while preserving all three measured ranges."
            ),
        }
    )


def reference_panel_audit(report: dict[str, Any]) -> PanelAudit:
    """Build the fixed reference audit from an already validated report."""
    try:
        acceptance = report["acceptance"]
        if (
            report["candidate_count"] != 96
            or report["panel_count"] != 24
            or acceptance["passed"] is not True
        ):
            raise KeyError
        minimum_distance = float(acceptance["selected_minimum_distance"])
        coverage = float(acceptance["selected_descriptor_coverage"])
        if not math.isfinite(minimum_distance) or not math.isfinite(coverage):
            raise KeyError
    except (KeyError, TypeError, ValueError):
        raise WorkshopAgentError(
            "The validated report cannot support the reference audit."
        ) from None
    return PanelAudit.model_validate(
        {
            "result_assessment": (
                "The fixed 24-member panel passed the independently checked "
                "first-24 baseline contract."
            ),
            "surprising_result": (
                f"The selected minimum distance is {minimum_distance:.3f} and the "
                f"normalized descriptor coverage is {coverage:.3f}."
            ),
            "scientific_boundaries": (
                "The result measures fingerprint separation and three computed "
                "descriptors; it does not establish biological activity or safety."
            ),
            "next_iteration": (
                "Repeat the bounded comparison with another fingerprint representation "
                "and add declared assay constraints before experimental selection."
            ),
        }
    )


def _remove_previous_regular_output(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise WorkshopAgentError(
            "Module 3 output paths must be regular workspace files."
        )
    if path.exists():
        path.unlink()


def _panel_child_environment() -> dict[str, str]:
    """Return only the fixed runtime variables needed by the analysis child."""
    return {
        name: os.environ[name]
        for name in _PANEL_CHILD_ENVIRONMENT_ALLOWLIST
        if name in os.environ
    }


class PanelDesignAgent:
    """Plan, render, execute, validate, and audit one bounded Module 3 analysis."""

    def __init__(
        self,
        *,
        workdir: Path,
        mission: str,
        mode: Literal["hosted", "reference"] = "hosted",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        if type(mode) is not str or mode not in {"hosted", "reference"}:
            raise ValueError("mode must be 'hosted' or 'reference'.")
        resolved = Path(workdir).resolve()
        candidate_path = resolved / "reframe_candidates.csv"
        if (
            not resolved.is_dir()
            or candidate_path.is_symlink()
            or not candidate_path.is_file()
        ):
            raise ValueError(
                "The agent workspace must contain a regular reframe_candidates.csv."
            )
        mission_text = mission.strip()
        if not mission_text:
            raise ValueError("mission must not be empty.")
        self.workdir = resolved
        self.mission = mission_text
        self.mode: Literal["hosted", "reference"] = mode
        self.data_profile = profile_candidate_csv(candidate_path)
        if (
            self.data_profile["row_count"] != 96
            or self.data_profile["unique_canonical_ikeys"] != 96
            or self.data_profile["duplicate_canonical_ikey_rows"] != 0
            or self.data_profile["blank_canonical_ikey_rows"] != 0
        ):
            raise WorkshopAgentError(
                "Module 3 requires exactly 96 unique candidate connectivity keys."
            )
        self.plan: PanelPlan | None = None

        if mode == "reference":
            if api_key is not None or client is not None:
                raise ValueError("Reference mode requires no API key or client.")
            self.api_key: str | None = None
            self.client = None
        else:
            if client is None:
                protected_key = get_workshop_api_key(api_key)
                self.api_key = protected_key
                self.client = _client(protected_key)
            else:
                self.api_key = (
                    get_workshop_api_key(api_key, prompt=False)
                    if api_key is not None
                    else None
                )
                self.client = client

    def request_plan(self) -> PanelPlan:
        """Return the fixed reference plan or request the strict hosted plan."""
        if self.mode == "reference":
            self.plan = reference_panel_plan()
            return self.plan
        prompt = (
            f"Mission:\n{self.mission}\n\n"
            "Deterministic input profile:\n"
            f"{json.dumps(self.data_profile, indent=2, sort_keys=True)}\n\n"
            "Compare exactly these controller-owned strategies. Strategy 1 is "
            "descriptor-extrema seeding plus cluster-aware max-min selection with "
            "Morgan radius 2, 1024 bits, and Butina distance cutoff 0.55. Strategy 2 "
            "is descriptor-extrema seeding plus greedy max-min selection with Morgan "
            "radius 3 and 2048 bits. The fixed baseline is the first 24 stable source "
            "rows. Do not write code; the controller owns all executable source."
        )
        self.plan = _structured_request(
            api_key=self.api_key or "",
            client=self.client,
            system_prompt=(
                "You are a bounded chemistry planning assistant. Return only the "
                "strict PanelPlan schema. Fingerprint separation is not activity."
            ),
            user_prompt=prompt,
            tool_name="submit_panel_plan",
            response_model=PanelPlan,
            max_tokens=1800,
        )
        return self.plan

    def _request_audit(
        self, approved_strategy: int, validated_report_snapshot: str
    ) -> PanelAudit:
        if self.plan is None:
            raise WorkshopAgentError("Request a plan before auditing a strategy.")
        try:
            report = json.loads(validated_report_snapshot)
        except (TypeError, json.JSONDecodeError):
            raise WorkshopAgentError(
                "The validated report is unavailable for audit."
            ) from None
        if self.mode == "reference":
            return reference_panel_audit(report)
        prompt = (
            "Review only this independently validated panel report. State the measured "
            "tradeoff and scientific boundaries. Do not infer binding, activity, "
            "efficacy, safety, or clinical relevance.\n\n"
            f"Approved strategy {approved_strategy}:\n"
            f"{json.dumps(self.plan.strategies[approved_strategy - 1].model_dump(mode='json'), sort_keys=True)}\n\n"
            f"Validated report:\n{json.dumps(report, sort_keys=True)}"
        )
        return _structured_request(
            api_key=self.api_key or "",
            client=self.client,
            system_prompt=(
                "You are a bounded scientific reviewer. Return only the strict "
                "PanelAudit schema and use only the validated receipt."
            ),
            user_prompt=prompt,
            tool_name="submit_panel_audit",
            response_model=PanelAudit,
            max_tokens=1200,
        )

    def run(
        self,
        *,
        approved_strategy: int,
        expected_panel_size: int,
        max_revisions: int = 0,
        timeout_seconds: int = 300,
        python_executable: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PanelAgentRun:
        """Render and execute one exact controller-owned strategy."""
        if self.plan is None:
            raise WorkshopAgentError(
                "Request and review a plan before running the agent."
            )
        if approved_strategy not in (1, 2):
            raise ValueError("approved_strategy must be 1 or 2.")
        if type(expected_panel_size) is not int or expected_panel_size != 24:
            raise ValueError("expected_panel_size must be exactly 24.")
        if type(max_revisions) is not int or max_revisions != 0:
            raise ValueError("max_revisions must be 0 for controller-owned source.")
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        executable = python_executable or sys.executable
        strategy = self.plan.strategies[approved_strategy - 1]

        def notify(event: str, **payload: Any) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event, payload)
            except Exception:
                pass

        notify(
            "run_started",
            approved_strategy=approved_strategy,
            expected_panel_size=expected_panel_size,
        )
        source = _render_panel_analysis(approved_strategy, expected_panel_size)
        source = validate_panel_analysis_source(
            source,
            approved_strategy=approved_strategy,
            expected_panel_size=expected_panel_size,
        )
        analysis_path = self.workdir / "analysis.py"
        rendered_path = self.workdir / "analysis_attempt_1_rendered.py"
        trace_path = self.workdir / "agent_trace.json"
        panel_path = self.workdir / "panel.csv"
        report_path = self.workdir / "report.json"
        for path in (analysis_path, rendered_path, trace_path, panel_path, report_path):
            _remove_previous_regular_output(path)
        analysis_path.write_text(source, encoding="utf-8")
        rendered_path.write_text(source, encoding="utf-8")
        notify(
            "source_rendered",
            attempt=1,
            source_file=rendered_path.name,
            source=source,
            implementation_summary=strategy.approach,
            expected_tradeoffs=[strategy.tradeoff],
        )
        notify("source_validated", attempt=1, source_file=rendered_path.name)

        child_environment = _panel_child_environment()
        not_before_ns = time.time_ns()
        notify(
            "execution_started",
            attempt=1,
            timeout_seconds=timeout_seconds,
        )
        started = time.perf_counter()
        completed = None
        attempt: AgentAttempt
        try:
            completed = subprocess.run(
                [executable, "-I", "analysis.py"],
                cwd=self.workdir,
                env=child_environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode:
                raise WorkshopAgentError(
                    f"analysis.py exited with code {completed.returncode}."
                )
            receipt, validated_report_snapshot = _validate_panel_artifacts_snapshot(
                self.workdir,
                expected_panel_size=expected_panel_size,
                not_before_ns=not_before_ns,
            )
            attempt = AgentAttempt(
                number=1,
                source_file=rendered_path.name,
                return_code=completed.returncode,
                elapsed_seconds=elapsed,
                passed=True,
                message=json.dumps(receipt, sort_keys=True),
                stdout_tail=completed.stdout[-3000:],
                stderr_tail=completed.stderr[-3000:],
                implementation_summary=strategy.approach,
                expected_tradeoffs=(strategy.tradeoff,),
            )
            notify(
                "attempt_passed",
                attempt=1,
                elapsed_seconds=elapsed,
                receipt=receipt,
                stdout_tail=completed.stdout[-3000:],
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            message = f"analysis.py exceeded the {timeout_seconds}-second limit."
            attempt = AgentAttempt(
                number=1,
                source_file=rendered_path.name,
                return_code=None,
                elapsed_seconds=elapsed,
                passed=False,
                message=message,
                stdout_tail="",
                stderr_tail="",
                implementation_summary=strategy.approach,
                expected_tradeoffs=(strategy.tradeoff,),
            )
            notify(
                "attempt_failed",
                attempt=1,
                elapsed_seconds=elapsed,
                message=message,
                will_revise=False,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            message = f"{type(error).__name__}: {error}"
            attempt = AgentAttempt(
                number=1,
                source_file=rendered_path.name,
                return_code=(completed.returncode if completed is not None else None),
                elapsed_seconds=elapsed,
                passed=False,
                message=message[-6000:],
                stdout_tail=(completed.stdout[-3000:] if completed is not None else ""),
                stderr_tail=(completed.stderr[-3000:] if completed is not None else ""),
                implementation_summary=strategy.approach,
                expected_tradeoffs=(strategy.tradeoff,),
            )
            notify(
                "attempt_failed",
                attempt=1,
                elapsed_seconds=elapsed,
                message=message[-6000:],
                will_revise=False,
            )

        success = attempt.passed
        audit: PanelAudit | None = None
        audit_error = ""
        if success:
            notify("audit_started")
            try:
                audit = self._request_audit(
                    approved_strategy, validated_report_snapshot
                )
                notify("audit_completed", audit=audit.model_dump(mode="json"))
            except Exception:
                audit_error = "The optional scientific audit was unavailable."
                notify("audit_failed", message=audit_error)

        trace_payload = {
            "workshop_agent_version": WORKSHOP_AGENT_VERSION,
            "mode": self.mode,
            "model": DEFAULT_MODEL if self.mode == "hosted" else None,
            "analysis_transport": "exact_controller_renderer",
            "data_profile": self.data_profile,
            "plan": self.plan.model_dump(mode="json"),
            "approved_strategy": approved_strategy,
            "attempts": [asdict(attempt)],
            "success": success,
            "audit": audit.model_dump(mode="json") if audit is not None else None,
            "audit_error": audit_error,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
        result = PanelAgentRun(
            success=success,
            approved_strategy=approved_strategy,
            attempts=(attempt,),
            analysis_path=analysis_path,
            panel_path=panel_path,
            report_path=report_path,
            trace_path=trace_path,
            audit=audit,
        )
        notify(
            "run_completed",
            success=success,
            attempt_count=1,
            trace_path=str(trace_path),
        )
        return result
