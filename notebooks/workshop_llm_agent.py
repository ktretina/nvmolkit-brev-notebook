"""Bounded hosted coding agents for ACS workshop Modules 2 and 3.

The hosted LLM proposes Python source, but deterministic local code decides what
may execute and whether the resulting artifacts pass.  This is intentionally a
small workshop controller rather than a general shell-capable coding agent.
"""

from __future__ import annotations

import ast
import builtins
import csv
import getpass
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NEMOTRON_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
WORKSHOP_AGENT_VERSION = "2026.08.14.22"
NEIGHBORHOOD_FUNCTION_LINE_CAP = 70
AUTH_GUIDANCE = (
    "NVIDIA_API_KEY must be a hosted NVIDIA Developer API key beginning with "
    "nvapi-. Generate it from the Nemotron model page on build.nvidia.com."
)


class WorkshopAgentError(RuntimeError):
    """Raised when a bounded hosted or local agent step cannot be completed."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GeneratedFunction(_StrictModel):
    function_source: str = Field(min_length=80)
    design_choices: list[str] = Field(min_length=2, max_length=2)


class NeighborhoodCodePlan(_StrictModel):
    missing_anchor: Literal["raise", "skip"]
    invalid_matrix: Literal["raise", "skip"]
    design_choices: list[str] = Field(min_length=2, max_length=2)


class NeighborhoodReview(_StrictModel):
    software_failure_mode: str = Field(min_length=20)
    representation_limitation: str = Field(min_length=20)
    unsupported_biological_inference: str = Field(min_length=20)
    proposed_tests: list[str] = Field(min_length=3, max_length=3)


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


class GeneratedAnalysis(_StrictModel):
    analysis_source: str = Field(min_length=500)
    implementation_summary: str = Field(min_length=40)
    expected_tradeoffs: list[str] = Field(min_length=1, max_length=4)


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
    generated_source: str = ""


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
    "submit_neighborhood_review": (
        "Return a skeptical review with one software failure mode, one molecular-"
        "representation limitation, one unsupported biological inference, and "
        "exactly three concrete proposed tests."
    ),
    "submit_panel_plan": (
        "Return exactly two scientifically defensible panel-design strategies and "
        "recommend one after inspecting the supplied bounded data profile."
    ),
    "submit_panel_analysis": (
        "Return a complete standalone analysis.py implementation, a short summary, "
        "and its main expected tradeoffs. Return Python source without fences."
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
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(response_model.model_fields)
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
        raise ValueError(AUTH_GUIDANCE) from None
    except Exception as exc:
        raise WorkshopAgentError(
            "The hosted Nemotron request failed; check network and model availability."
        ) from exc

    try:
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, (list, tuple)) and len(calls) == 1:
            call = calls[0]
            function = getattr(call, "function", None)
            if function is None or getattr(function, "name", None) != tool_name:
                raise WorkshopAgentError("The hosted response called an unexpected tool.")
            payload = json.loads(function.arguments)
        else:
            # Some compatible endpoints return the forced arguments as JSON content.
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                raise WorkshopAgentError("The hosted response did not contain tool arguments.")
            payload = json.loads(content)
        return response_model.model_validate(payload)
    except (AttributeError, IndexError, json.JSONDecodeError, ValidationError) as exc:
        raise WorkshopAgentError(
            "The hosted response did not satisfy the workshop's required schema."
        ) from exc


def _plain_text_request(
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    client: Any = None,
) -> str:
    """Request ordinary multiline text without a structured-tool string cap."""
    active_client = client or _client(get_workshop_api_key(api_key, prompt=False))
    try:
        response = active_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body=NEMOTRON_EXTRA_BODY,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
        )
    except (AuthenticationError, PermissionDeniedError):
        raise ValueError(AUTH_GUIDANCE) from None
    except Exception as exc:
        raise WorkshopAgentError(
            "The hosted Nemotron request failed; check network and model availability."
        ) from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise WorkshopAgentError(
            "The hosted response did not contain a text message."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise WorkshopAgentError("The hosted response did not contain multiline text.")
    return content.strip()


def _neighborhood_text_request(
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    client: Any = None,
) -> str:
    """Backward-compatible Module 2 wrapper around ordinary text generation."""
    return _plain_text_request(
        api_key=api_key,
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )


def _parse_neighborhood_text_response(content: str) -> NeighborhoodCodePlan:
    """Extract two bounded policies and their explanations."""
    policies = {}
    for field in ("MISSING_ANCHOR", "INVALID_MATRIX"):
        match = re.search(rf"(?im)^\s*{field}\s*:\s*(raise|skip)\s*$", content)
        if match:
            policies[field] = match.group(1).lower()
    choices_by_number: dict[int, str] = {}
    for match in re.finditer(
        r"(?im)^\s*(?:[-*]\s*)?(?:DESIGN_?)?CHOICE[_ ]?([12])\s*:\s*(.+?)\s*$",
        content,
    ):
        choices_by_number[int(match.group(1))] = match.group(2).strip()
    if set(policies) != {"MISSING_ANCHOR", "INVALID_MATRIX"}:
        raise WorkshopAgentError(
            "The hosted response must select raise or skip for both bounded policies."
        )
    if set(choices_by_number) != {1, 2}:
        raise WorkshopAgentError(
            "The hosted response did not include exactly two marked design choices."
        )

    try:
        return NeighborhoodCodePlan(
            missing_anchor=policies["MISSING_ANCHOR"],
            invalid_matrix=policies["INVALID_MATRIX"],
            design_choices=[choices_by_number[1], choices_by_number[2]],
        )
    except ValidationError as exc:
        raise WorkshopAgentError(
            "The hosted response contained an invalid policy or design choice."
        ) from exc


def _render_neighborhood_function(plan: NeighborhoodCodePlan) -> str:
    """Render a tested neighborhood workflow from the agent's bounded policies."""
    missing_action = (
        "raise ValueError(f'Anchor not found: {term}')"
        if plan.missing_anchor == "raise"
        else "continue"
    )
    invalid_action = (
        "raise RuntimeError('Unexpected similarity matrix')"
        if plan.invalid_matrix == "raise"
        else "continue"
    )
    return f'''def build_neighborhood_atlas(records, anchor_terms, radii=(2, 3), fp_bits=1024, top_k=10):
    """Build a tidy, multi-radius structural-neighborhood atlas."""
    required = {{'_mol', 'name', 'canonical_ikey', 'reframedb_url'}}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f'records is missing {{sorted(missing)}}')
    query_indices = []
    for term in anchor_terms:
        matches = records[records['name'].str.contains(term, case=False, regex=False)]
        if matches.empty:
            {missing_action}
        query_indices.append(int(matches.index[0]))
    if not query_indices:
        raise ValueError('No anchor compounds were found')
    rows = []
    molecules = records['_mol'].tolist()
    query_molecules = records.loc[query_indices, '_mol'].tolist()
    # Compare the same queries with molecular context at each radius.
    for radius in radii:
        library_fps = make_fingerprints(molecules, radius=radius, fp_bits=fp_bits)
        query_fps = make_fingerprints(query_molecules, radius=radius, fp_bits=fp_bits)
        similarities = tanimoto_matrix(query_fps, library_fps)
        expected_shape = (len(query_indices), len(records))
        invalid_matrix = (similarities.shape != expected_shape or
                          not np.isfinite(similarities).all() or
                          similarities.min() < 0 or similarities.max() > 1)
        if invalid_matrix:
            {invalid_action}
        for query_position, query_index in enumerate(query_indices):
            query = records.loc[query_index]
            order = np.argsort(-similarities[query_position], kind='stable')
            # Exclude the reference itself before selecting the reported neighbors.
            order = [int(idx) for idx in order
                     if records.iloc[idx]['canonical_ikey'] != query['canonical_ikey']][:top_k]
            for rank, library_index in enumerate(order, start=1):
                neighbor = records.iloc[library_index]
                rows.append({{'radius': int(radius), 'query': query['name'],
                             'query_ikey': query['canonical_ikey'], 'rank': rank,
                             'neighbor': neighbor['name'],
                             'neighbor_ikey': neighbor['canonical_ikey'],
                             'tanimoto': float(similarities[query_position, library_index]),
                             'profile': neighbor['reframedb_url']}})
    columns = ['radius', 'query', 'query_ikey', 'rank', 'neighbor',
               'neighbor_ikey', 'tanimoto', 'profile']
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(['radius', 'query', 'rank'], ignore_index=True)
'''


def _strip_code_fence(source: str) -> str:
    """Extract Python from common hosted-response envelopes."""
    source = source.lstrip("\ufeff").strip()
    fenced_blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_blocks:
        # Prefer the block that contains the required chemistry entry point.
        source = next(
            (
                block
                for block in fenced_blocks
                if "MorganFingerprintGenerator" in block
            ),
            fenced_blocks[0],
        ).strip()

    lines = source.splitlines()
    while lines and lines[0].strip().lower() in {
        "python",
        "python:",
        "analysis.py",
        "analysis.py:",
        "source:",
        "code:",
    }:
        lines.pop(0)
    return "\n".join(lines).strip() + "\n"


_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_ALWAYS_FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "chown",
    "hardlink_to",
    "popen",
    "rmdir",
    "spawn",
    "symlink_to",
    "system",
    "unlink",
}
_PATH_MUTATION_ATTRIBUTES = {"rename", "replace"}


def _is_safe_report_open(call: ast.Call) -> bool:
    """Allow only a literal, text-mode write of the required JSON artifact."""
    if not call.args or len(call.args) > 2:
        return False
    target = call.args[0]
    if not isinstance(target, ast.Constant) or target.value != "report.json":
        return False
    mode_node = call.args[1] if len(call.args) == 2 else None
    for keyword in call.keywords:
        if keyword.arg == "mode" and mode_node is None:
            mode_node = keyword.value
        elif keyword.arg == "encoding":
            if not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value in {"utf-8", "UTF-8"}
            ):
                return False
        else:
            return False
    return (
        isinstance(mode_node, ast.Constant)
        and mode_node.value in {"w", "wt"}
    )


def _looks_like_path_receiver(node: ast.AST) -> bool:
    """Recognize direct Path construction and conventional path variable names."""
    if isinstance(node, ast.Name):
        lowered = node.id.lower()
        return lowered == "path" or lowered.endswith(("_path", "_file"))
    if isinstance(node, ast.Call):
        function = node.func
        return (
            isinstance(function, ast.Name) and function.id == "Path"
        ) or (
            isinstance(function, ast.Attribute) and function.attr == "Path"
        )
    if isinstance(node, ast.Attribute):
        return _looks_like_path_receiver(node.value)
    return False


def _validate_safe_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if not _is_safe_report_open(node):
                    raise WorkshopAgentError(
                        "Generated code may call open() only to write the literal "
                        "report.json artifact in text mode. The controller writes "
                        "analysis.py itself."
                    )
            elif isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise WorkshopAgentError(f"Generated code may not call {node.func.id}().")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ALWAYS_FORBIDDEN_ATTRIBUTES
            ):
                raise WorkshopAgentError(
                    f"Generated code may not call .{node.func.attr}()."
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _PATH_MUTATION_ATTRIBUTES
                and _looks_like_path_receiver(node.func.value)
            ):
                raise WorkshopAgentError(
                    f"Generated code may not call Path.{node.func.attr}()."
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise WorkshopAgentError("Generated code may not access dunder attributes.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise WorkshopAgentError("Generated code may not mutate global or nonlocal state.")


def _module_bindings_before(tree: ast.AST, line_number: int) -> set[str]:
    """Collect names available before one top-level generated-code statement."""
    bindings = set(dir(builtins))

    def bind_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            bindings.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind_target(element)

    class BindingVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            if node.lineno < line_number:
                for alias in node.names:
                    bindings.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.lineno < line_number:
                for alias in node.names:
                    bindings.add(alias.asname or alias.name)

        def visit_Assign(self, node: ast.Assign) -> None:
            if node.lineno < line_number:
                for target in node.targets:
                    bind_target(target)
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.lineno < line_number:
                bind_target(node.target)
            if node.value is not None:
                self.visit(node.value)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            if node.lineno < line_number:
                bind_target(node.target)
            self.visit(node.value)

        def visit_For(self, node: ast.For) -> None:
            if node.lineno < line_number:
                bind_target(node.target)
            self.visit(node.iter)
            for statement in (*node.body, *node.orelse):
                self.visit(statement)

        visit_AsyncFor = visit_For

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None and node.lineno < line_number:
                    bind_target(item.optional_vars)
            for statement in node.body:
                self.visit(statement)

        visit_AsyncWith = visit_With

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name and node.lineno < line_number:
                bindings.add(node.name)
            for statement in node.body:
                self.visit(statement)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.lineno < line_number:
                bindings.add(node.name)
            # Function-local assignments are not module bindings.

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.lineno < line_number:
                bindings.add(node.name)
            # Class-body assignments are not module bindings.

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_comprehension(self, node: ast.comprehension) -> None:
            # Comprehension targets do not leak into module scope in Python 3.
            self.visit(node.iter)
            for condition in node.ifs:
                self.visit(condition)

    visitor = BindingVisitor()
    for statement in tree.body:
        if getattr(statement, "lineno", line_number) >= line_number:
            break
        visitor.visit(statement)
    return bindings


def _panel_api_issues(tree: ast.AST) -> list[str]:
    """Report common generated-code mistakes before spending a GPU execution attempt."""
    issues: list[str] = []
    name_assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    dict_assignments = {
        name: value for name, value in name_assignments.items()
        if isinstance(value, ast.Dict)
    }
    if "valid_mols" in assigned_names and "valid_df" not in assigned_names:
        issues.append(
            "when invalid molecules are removed, create an aligned valid_df and reset "
            "its index so rows, molecules, fingerprints, and similarities stay aligned"
        )

    available_position_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "available_indices"
        ):
            available_position_names.add(node.targets[0].id)
    if any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "available_indices"
        and isinstance(node.slice, ast.Name)
        and node.slice.id in available_position_names
        for node in ast.walk(tree)
    ):
        issues.append(
            "do not index available_indices twice; keep the argmin as a position and "
            "convert it to one candidate index exactly once"
        )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and node.iter.args
            and isinstance(node.iter.args[0], ast.Constant)
            and node.iter.args[0].value == 96
            and any(isinstance(child, ast.Continue) for child in ast.walk(node))
        ):
            issues.append(
                "a for range(96) loop with continue can select fewer than 96 compounds; "
                "continue until 96 are selected or the candidate pool is exhausted"
            )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "max_similarity"
            for target in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "full"
                and len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
                and value.args[1].value in {1, 1.0}
            ):
                issues.append(
                    "initialize max_similarity with zeros for greedy max-min selection; "
                    "starting at 1 prevents np.maximum updates"
                )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "max_similarity"
            and not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
                and not isinstance(node.value.value, bool)
            )
            for target in node.targets
        ):
            issues.append(
                "do not assign a similarity row or vector to one max_similarity element; "
                "update the complete vector with max_similarity = np.maximum("
                "max_similarity, similarity_matrix[:, selected_idx]) and only use a "
                "numeric scalar when masking selected indices"
            )
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "max_similarity"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "maximum"
            and not (
                len(node.value.args) == 2
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == "max_similarity"
                and isinstance(node.value.args[1], ast.Subscript)
                and isinstance(node.value.args[1].value, ast.Name)
                and node.value.args[1].value.id == "similarity_matrix"
                and isinstance(node.value.args[1].slice, ast.Tuple)
                and len(node.value.args[1].slice.elts) == 2
                and isinstance(node.value.args[1].slice.elts[0], ast.Slice)
                and isinstance(node.value.args[1].slice.elts[1], ast.Name)
                and node.value.args[1].slice.elts[1].id == "selected_idx"
            )
        ):
            issues.append(
                "update greedy similarities with the full candidate column "
                "similarity_matrix[:, selected_idx], which has the same 1-D length as "
                "max_similarity"
            )
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "max_similarity"
                for target in node.targets
            )
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "similarity_matrix"
        ):
            issues.append(
                "do not replace max_similarity with a similarity-matrix row or subset; "
                "retain its full candidate length and update it with np.maximum"
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) > 2
        ):
            root = node.value
            while isinstance(root, ast.Subscript):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "similarity_matrix":
                issues.append(
                    "similarity_matrix is 2-D; extract the panel matrix with "
                    "similarity_matrix[np.ix_(selected_indices, selected_indices)]"
                )
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"report", "report_data"}
                for target in node.targets
            )
        ):
            if node not in tree.body:
                issues.append(
                    "construct report at module scope after all report inputs are defined"
                )
            else:
                report_bindings = _module_bindings_before(tree, node.lineno)
                report_local_names = {
                    child.id
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                }
                undefined_report_names = sorted({
                    child.id
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id not in report_bindings
                    and child.id not in report_local_names
                })
                if undefined_report_names:
                    issues.append(
                        "define every report input before constructing report; names used "
                        "before definition: " + ", ".join(undefined_report_names)
                    )
            report_fields = {
                key.value: value
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            seed_value = report_fields.get("seed")
            if not (
                isinstance(seed_value, ast.Constant) and seed_value.value == 2026
            ):
                issues.append("report must record seed=2026 at the top level")
            parameters = report_fields.get("parameters")
            if isinstance(parameters, ast.Name):
                parameters = dict_assignments.get(parameters.id)
            if not isinstance(parameters, ast.Dict) or not parameters.keys:
                issues.append("report parameters must be a non-empty mapping")
            descriptor_quantiles = report_fields.get("descriptor_quantiles")
            if isinstance(descriptor_quantiles, ast.Name):
                descriptor_quantiles = dict_assignments.get(descriptor_quantiles.id)
            descriptor_keys = {
                key.value
                for key in descriptor_quantiles.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            } if isinstance(descriptor_quantiles, ast.Dict) else set()
            if not {"candidate", "panel"}.issubset(descriptor_keys):
                issues.append(
                    "report descriptor_quantiles must contain separate candidate and "
                    "panel mappings"
                )
            candidate_count = report_fields.get("candidate_count")
            if candidate_count is not None and any(
                token in ast.unparse(candidate_count).lower()
                for token in ("available", "filtered", "valid")
            ):
                issues.append(
                    "report candidate_count must use the complete input table, not an "
                    "available, filtered, or valid-molecule subset"
                )
            unique_ikeys = report_fields.get("unique_ikeys")
            if isinstance(unique_ikeys, ast.Name):
                unique_ikeys = name_assignments.get(unique_ikeys.id, unique_ikeys)
            expected_unique_expression = False
            if (
                isinstance(unique_ikeys, ast.Call)
                and isinstance(unique_ikeys.func, ast.Name)
                and unique_ikeys.func.id == "int"
                and len(unique_ikeys.args) == 1
            ):
                count_call = unique_ikeys.args[0]
                expected_unique_expression = (
                    isinstance(count_call, ast.Call)
                    and isinstance(count_call.func, ast.Attribute)
                    and count_call.func.attr == "nunique"
                    and "panel_df" in ast.unparse(count_call.func.value)
                    and "canonical_ikey" in ast.unparse(count_call.func.value)
                )
            if not expected_unique_expression:
                issues.append(
                    "set report unique_ikeys exactly to "
                    "int(panel_df['canonical_ikey'].nunique()), an integer count for "
                    "the final panel rather than a boolean, list, string, or candidate "
                    "table count"
                )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "panel_df"
            for target in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "copy"
            ):
                value = value.func.value
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id in {"df", "records"}
                and "selected_indices" in ast.unparse(value.slice)
            ):
                issues.append(
                    "build panel_df from the aligned valid_df used for fingerprints, not "
                    "from the original unfiltered table with filtered-table indices"
                )
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "numpy"
            and isinstance(node.func.value, ast.Name)
            and "tensor" in node.func.value.id.lower()
        ):
            issues.append(
                "do not call .numpy() directly on a CUDA torch tensor; use the "
                "nvMolKit result's .numpy() or tensor.detach().cpu().numpy()"
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "fused_butina"
            and node.args
            and "numpy" in ast.unparse(node.args[0]).lower()
        ):
            issues.append(
                "fused_butina requires the packed CUDA fingerprint tensor, not NumPy"
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes"}
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "Path"
            and node.func.value.args
            and isinstance(node.func.value.args[0], ast.Constant)
            and node.func.value.args[0].value == "analysis.py"
        ):
            issues.append(
                "the generated script must not write analysis.py; the controller owns it"
            )
    string_values = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    optional_artifacts = {
        "panel_representatives.sdf",
        "geometry_summary.csv",
    }
    if string_values.intersection(optional_artifacts):
        issues.append(
            "omit optional 3D artifacts from the bounded agent run; the simplified "
            "notebook uses a bounded 2D structure gallery in Step 4"
        )
    has_similarity_panel_slice = any(
        isinstance(node, ast.Subscript)
        and "similarity_matrix" in ast.unparse(node)
        and "selected_indices" in ast.unparse(node)
        for node in ast.walk(tree)
    )
    has_np_ix = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ix_"
        for node in ast.walk(tree)
    )
    if has_similarity_panel_slice and not has_np_ix:
        issues.append(
            "extract the square panel similarity matrix with "
            "similarity_matrix[np.ix_(selected_indices, selected_indices)]"
        )
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"mean", "min", "max", "median"}
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "panel_similarity_matrix"
        for node in ast.walk(tree)
    ):
        issues.append(
            "summarize only the off-diagonal upper triangle of the panel similarity "
            "matrix so self-similarities of 1.0 are excluded"
        )
    return list(dict.fromkeys(issues))


def _repair_missing_function_body_indent(source: str) -> str:
    """Repair a generated function body shifted to column zero."""
    lines = source.splitlines()
    if not lines or not lines[0].startswith("def build_neighborhood_atlas("):
        return source

    # Locate the signature's final colon, including when arguments span several lines.
    signature_end_line = None
    nesting_depth = 0
    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
        for token in tokens:
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                nesting_depth += 1
            elif token.string in ")]}":
                nesting_depth -= 1
            elif token.string == ":" and nesting_depth == 0:
                signature_end_line = token.end[0]
                if lines[signature_end_line - 1][token.end[1] :].strip():
                    return source
                break
    except (IndentationError, tokenize.TokenError):
        return source

    if signature_end_line is None:
        return source
    first_body_line = next(
        (line for line in lines[signature_end_line:] if line.strip()), ""
    )
    if not first_body_line or first_body_line[0].isspace():
        return source
    repaired_lines = lines[:signature_end_line] + [
        f"    {line}" if line.strip() else line
        for line in lines[signature_end_line:]
    ]
    repaired = "\n".join(repaired_lines) + "\n"
    try:
        ast.parse(repaired)
    except SyntaxError:
        return source
    return repaired


def _compact_neighborhood_source(source: str) -> str:
    """Remove presentation padding while preserving executable Python."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return source

    function = functions[0]
    doc_node = (
        function.body[0]
        if function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
        else None
    )
    doc_start = doc_node.lineno if doc_node else None
    doc_end = doc_node.end_lineno if doc_node else None
    doc_summary = ""
    if doc_node:
        first_paragraph = ast.get_docstring(function, clean=True).split("\n\n", 1)[0]
        doc_summary = " ".join(first_paragraph.split())
    protected_string_lines = {
        number
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node is not getattr(doc_node, "value", None)
        and node.end_lineno
        and node.end_lineno > node.lineno
        for number in range(node.lineno, node.end_lineno + 1)
    }

    compacted = []
    for number, line in enumerate(source.splitlines(), start=1):
        if doc_start and doc_start <= number <= doc_end:
            if number == doc_start:
                indentation = line[: len(line) - len(line.lstrip())]
                compacted.append(f"{indentation}{json.dumps(doc_summary, ensure_ascii=False)}")
            continue
        if number in protected_string_lines:
            compacted.append(line.rstrip())
            continue
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            decoration = stripped[1:].strip()
            if decoration and set(decoration) <= {"-", "="}:
                continue
        compacted.append(line.rstrip())
    return "\n".join(compacted) + "\n"


def validate_neighborhood_function_source(source: str) -> str:
    """Validate the single function generated for Module 2."""
    source = _strip_code_fence(source)
    source = _repair_missing_function_body_indent(source)
    source = _compact_neighborhood_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkshopAgentError(f"Generated function has invalid syntax: {exc}") from exc

    issues = []
    if len(source.splitlines()) > NEIGHBORHOOD_FUNCTION_LINE_CAP:
        issues.append(
            "the compacted function exceeds the "
            f"{NEIGHBORHOOD_FUNCTION_LINE_CAP}-line safety cap"
        )
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        issues.append("do not add imports; the notebook already provides pd and np")
    top_level = [node for node in tree.body if not isinstance(node, ast.Expr)]
    functions = [node for node in top_level if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(top_level) != 1:
        issues.append("return exactly one top-level function")
    else:
        function = functions[0]
        if function.name != "build_neighborhood_atlas" or function.decorator_list:
            issues.append(
                "the function must be undecorated and named build_neighborhood_atlas"
            )
        positional = [item.arg for item in function.args.args]
        if positional[:2] != ["records", "anchor_terms"]:
            issues.append("the first arguments must be records and anchor_terms")
        if not ast.get_docstring(function):
            issues.append("include a short function docstring")

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {
        node.func.id for node in calls if isinstance(node.func, ast.Name)
    }
    for required_helper in ("make_fingerprints", "tanimoto_matrix"):
        if required_helper not in called_names:
            issues.append(f"call the supplied {required_helper} helper")
    for call in calls:
        if isinstance(call.func, ast.Name) and call.func.id == "make_fingerprints":
            unsupported = {
                keyword.arg
                for keyword in call.keywords
                if keyword.arg not in {"radius", "fp_bits"}
            }
            if unsupported:
                issues.append(
                    "make_fingerprints only accepts molecules, radius, and fp_bits"
                )
                break

    anchor_terms_as_string = any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "anchor_terms"
        and node.attr in {"lower", "casefold", "strip"}
        for node in ast.walk(tree)
    )
    if anchor_terms_as_string:
        issues.append(
            "anchor_terms is a sequence; iterate over each term instead of calling "
            "a string method on the collection"
        )
    iterates_over_anchor_terms = any(
        (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "anchor_terms"
        )
        or (
            isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp))
            and any(
                isinstance(generator.iter, ast.Name)
                and generator.iter.id == "anchor_terms"
                for generator in node.generators
            )
        )
        for node in ast.walk(tree)
    )
    if not iterates_over_anchor_terms:
        issues.append("iterate over anchor_terms to locate one query row per term")

    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    scaffold_names = {
        "query_indices",
        "molecules",
        "query_molecules",
        "library_fps",
        "query_fps",
        "similarities",
        "rows",
        "columns",
    }
    missing_scaffold_names = sorted(scaffold_names - assigned_names)
    if missing_scaffold_names:
        issues.append(
            "follow the workshop scaffold; missing variables: "
            + ", ".join(missing_scaffold_names)
        )

    literal_anchor_match = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "contains"
        and call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "term"
        and any(
            keyword.arg == "case"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
        and any(
            keyword.arg == "regex"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
        for call in calls
    )
    if not literal_anchor_match:
        issues.append(
            "locate anchors with .str.contains(term, case=False, regex=False); "
            "exact name equality is not reliable for ReFRAME labels"
        )

    full_library_assignment = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "molecules"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "tolist"
        and isinstance(node.value.func.value, ast.Subscript)
        and isinstance(node.value.func.value.value, ast.Name)
        and node.value.func.value.value.id == "records"
        and isinstance(node.value.func.value.slice, ast.Constant)
        and node.value.func.value.slice.value == "_mol"
        for node in ast.walk(tree)
    )
    if not full_library_assignment:
        issues.append(
            "assign molecules = records['_mol'].tolist() so the fingerprint library "
            "stays aligned with records.iloc"
        )

    nested_radius_loop = any(
        isinstance(anchor_loop, ast.For)
        and isinstance(anchor_loop.iter, ast.Name)
        and anchor_loop.iter.id == "anchor_terms"
        and any(
            isinstance(descendant, ast.For)
            and descendant is not anchor_loop
            and isinstance(descendant.iter, ast.Name)
            and descendant.iter.id == "radii"
            for descendant in ast.walk(anchor_loop)
        )
        for anchor_loop in ast.walk(tree)
    )
    if nested_radius_loop:
        issues.append(
            "collect all query_indices first, then run the radius loop outside the "
            "anchor_terms loop"
        )

    fingerprinted_inputs = {
        call.args[0].id
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "make_fingerprints"
        and call.args
        and isinstance(call.args[0], ast.Name)
    }
    if not {"molecules", "query_molecules"}.issubset(fingerprinted_inputs):
        issues.append(
            "call make_fingerprints separately for molecules and query_molecules"
        )
    expected_similarity_call = any(
        isinstance(call.func, ast.Name)
        and call.func.id == "tanimoto_matrix"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "query_fps"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "library_fps"
        for call in calls
    )
    if not expected_similarity_call:
        issues.append("call tanimoto_matrix(query_fps, library_fps)")

    required_fields = {
        "_mol",
        "name",
        "canonical_ikey",
        "reframedb_url",
        "radius",
        "query",
        "query_ikey",
        "rank",
        "neighbor",
        "neighbor_ikey",
        "tanimoto",
        "profile",
    }
    string_constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing_fields = sorted(required_fields - string_constants)
    if missing_fields:
        issues.append(
            "use the required record and output fields: " + ", ".join(missing_fields)
        )
    query_uses_collection = any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "query"
            and isinstance(value, ast.Name)
            and value.id == "anchor_terms"
            for key, value in zip(node.keys, node.values)
        )
        for node in ast.walk(tree)
    )
    if query_uses_collection:
        issues.append("write the current query name, not the anchor_terms collection")
    stable_argsort = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "argsort"
        and any(
            keyword.arg == "kind"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "stable"
            for keyword in call.keywords
        )
        for call in calls
    )
    if not stable_argsort:
        issues.append("use np.argsort(..., kind='stable') for deterministic ranking")
    sorts_one_query_row = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "argsort"
        and call.args
        and isinstance(call.args[0], ast.UnaryOp)
        and isinstance(call.args[0].op, ast.USub)
        and isinstance(call.args[0].operand, ast.Subscript)
        and isinstance(call.args[0].operand.value, ast.Name)
        and call.args[0].operand.value.id == "similarities"
        and isinstance(call.args[0].operand.slice, ast.Name)
        and call.args[0].operand.slice.id == "query_position"
        for call in calls
    )
    if not sorts_one_query_row:
        issues.append(
            "rank one query row with np.argsort(-similarities[query_position], "
            "kind='stable')"
        )

    forbidden_internals = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"fp_data", "ikey_list", "numpy"}
    }
    if forbidden_internals:
        issues.append(
            "do not access fingerprint internals or call .numpy(); the helpers return "
            "backend-ready fingerprints and a host similarity matrix"
        )
    if called_names.intersection({"MorganFingerprintGenerator", "crossTanimotoSimilarity"}):
        issues.append("use the supplied helpers instead of direct nvMolKit calls")
    try:
        _validate_safe_tree(tree)
    except WorkshopAgentError as exc:
        issues.append(str(exc))

    if issues:
        unique_issues = list(dict.fromkeys(issues))
        details = "\n".join(f"- {issue}" for issue in unique_issues)
        raise WorkshopAgentError(f"Generated function failed local checks:\n{details}")
    return source


def generate_neighborhood_function(
    prompt: str,
    api_key: str,
    *,
    client: Any = None,
    max_repairs: int = 2,
) -> GeneratedFunction:
    """Generate one Module 2 function, requesting bounded repairs when needed."""
    if max_repairs not in range(0, 4):
        raise ValueError("max_repairs must be between 0 and 3.")

    system_prompt = (
        "You are a bounded coding partner for chemists. Select two implementation "
        "policies that the local workshop agent will render into tested Python. "
        "Return ordinary text in exactly this format:\n"
        "MISSING_ANCHOR: raise or skip\n"
        "INVALID_MATRIX: raise or skip\n"
        "CHOICE_1: one brief design choice\n"
        "CHOICE_2: one brief design choice\n"
        "Explain the scientific or software consequence of each policy without "
        "making biological claims. Do not return code, JSON, or Markdown fences."
    )
    request_prompt = prompt
    last_error: WorkshopAgentError | None = None

    for attempt in range(max_repairs + 1):
        content = _neighborhood_text_request(
            api_key=api_key,
            client=client,
            system_prompt=system_prompt,
            user_prompt=request_prompt,
            max_tokens=800,
        )
        try:
            plan = _parse_neighborhood_text_response(content)
            rendered = _render_neighborhood_function(plan)
            validated = validate_neighborhood_function_source(rendered)
            return GeneratedFunction(
                function_source=validated,
                design_choices=plan.design_choices,
            )
        except WorkshopAgentError as exc:
            last_error = exc
            if attempt == max_repairs:
                break
            request_prompt = (
                f"{prompt}\n\n"
                "Your previous response failed local validation. Return a corrected "
                "complete response using MISSING_ANCHOR, INVALID_MATRIX, CHOICE_1, "
                "and CHOICE_2 exactly as requested. Select only raise or skip for "
                "each policy. Do not return code or JSON.\n\n"
                f"Validation error:\n{exc}\n\n"
                "Rejected response:\n"
                f"{content}"
            )

    raise WorkshopAgentError(
        f"Generated function remained invalid after {max_repairs + 1} attempts: "
        f"{last_error}"
    ) from last_error


def review_neighborhood_function(
    *,
    prompt: str,
    function_source: str,
    result_summary: str,
    api_key: str,
    client: Any = None,
) -> NeighborhoodReview:
    """Ask the same embedded agent for a bounded skeptical review."""
    user_prompt = (
        f"Original contract:\n{prompt}\n\n"
        f"Implementation:\n```python\n{function_source}\n```\n\n"
        f"Observed result summary:\n{result_summary}\n\n"
        "Review without changing or executing the code."
    )
    return _structured_request(
        api_key=api_key,
        client=client,
        system_prompt=(
            "You are a skeptical cheminformatics reviewer. Separate software "
            "correctness, molecular representation, and biological inference. "
            "Propose tests that could fail for concrete defects."
        ),
        user_prompt=user_prompt,
        tool_name="submit_neighborhood_review",
        response_model=NeighborhoodReview,
        max_tokens=1400,
    )


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
        "status_counts": dict(statuses.most_common(10)),
        "descriptor_quantiles": descriptor_quantiles,
    }


_ALLOWED_IMPORT_ROOTS = {
    "collections",
    "itertools",
    "json",
    "math",
    "nvmolkit",
    "numpy",
    "pandas",
    "pathlib",
    "rdkit",
    "torch",
}


# The controller owns this small, allowlisted preamble.  Generated analyses can
# therefore focus on the scientific method without failing because an import was
# accidentally omitted from an otherwise usable proposal.
_PANEL_REQUIRED_IMPORTS = (
    ("pathlib", "Path", "Path", "from pathlib import Path"),
    ("json", None, "json", "import json"),
    ("numpy", None, "np", "import numpy as np"),
    ("pandas", None, "pd", "import pandas as pd"),
    ("torch", None, "torch", "import torch"),
    ("rdkit", "Chem", "Chem", "from rdkit import Chem"),
    (
        "nvmolkit.fingerprints",
        "MorganFingerprintGenerator",
        "MorganFingerprintGenerator",
        "from nvmolkit.fingerprints import MorganFingerprintGenerator",
    ),
    (
        "nvmolkit.clustering",
        "fused_butina",
        "fused_butina",
        "from nvmolkit.clustering import fused_butina",
    ),
    (
        "nvmolkit.similarity",
        "crossTanimotoSimilarity",
        "crossTanimotoSimilarity",
        "from nvmolkit.similarity import crossTanimotoSimilarity",
    ),
)


def _ensure_panel_imports(source: str) -> str:
    """Prepend any missing imports from the controller's fixed safe preamble."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Preserve the original syntax error for the main validator's clearer receipt.
        return source

    imports: set[tuple[str, str | None, str]] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                imports.add((alias.name, None, bound_name))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                imports.add((node.module or "", alias.name, bound_name))

    missing_lines = [
        line
        for module, imported_name, bound_name, line in _PANEL_REQUIRED_IMPORTS
        if (module, imported_name, bound_name) not in imports
    ]
    if not missing_lines:
        return source
    return "\n".join(missing_lines) + "\n\n" + source.lstrip()


def validate_panel_analysis_source(source: str) -> str:
    """Apply conservative static checks to the standalone Module 3 script."""
    source = _strip_code_fence(source)
    source = _ensure_panel_imports(source)
    if len(source.splitlines()) > 550:
        raise WorkshopAgentError("The generated analysis exceeds the 550-line cap.")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkshopAgentError(f"Generated analysis has invalid syntax: {exc}") from exc
    local_issues = _panel_api_issues(tree)
    try:
        _validate_safe_tree(tree)
    except WorkshopAgentError as exc:
        local_issues.insert(0, str(exc))
    if local_issues:
        details = "\n".join(f"- {issue}" for issue in local_issues)
        raise WorkshopAgentError(
            f"Generated analysis failed local checks:\n{details}"
        )

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise WorkshopAgentError("Generated analysis may not use relative imports.")
            imported_roots.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value.startswith(("/", "~")) or ".." in Path(value).parts:
                raise WorkshopAgentError(
                    "Generated analysis may not contain absolute or parent-relative paths."
                )
    disallowed = imported_roots - _ALLOWED_IMPORT_ROOTS
    if disallowed:
        raise WorkshopAgentError(
            f"Generated analysis uses disallowed imports: {sorted(disallowed)}"
        )
    if "nvmolkit" not in imported_roots:
        raise WorkshopAgentError("Generated analysis must import installed nvMolKit.")
    required_api_names = {"MorganFingerprintGenerator"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if not required_api_names.issubset(names):
        raise WorkshopAgentError("Generated analysis must use MorganFingerprintGenerator.")
    second_operations = {
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    }
    if not names.intersection(second_operations):
        raise WorkshopAgentError(
            "Generated analysis must use a second approved nvMolKit batch operation."
        )
    string_values = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required_files = {"reframe_candidates.csv", "panel.csv", "report.json"}
    if not required_files.issubset(string_values):
        raise WorkshopAgentError(
            "Generated analysis must use the three required workspace filenames."
        )
    return source


def _similarity_values(value: Any, key: str = "") -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            lowered = str(child_key).lower()
            if any(token in lowered for token in ("count", "pairs", "size", "n_")):
                continue
            values.extend(_similarity_values(child, lowered))
    elif isinstance(value, list):
        for child in value:
            values.extend(_similarity_values(child, key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values.append(float(value))
    return values


def validate_panel_artifacts(
    workdir: Path, *, expected_panel_size: int
) -> dict[str, Any]:
    """Validate agent artifacts independently of the generated analysis."""
    candidate_path = workdir / "reframe_candidates.csv"
    panel_path = workdir / "panel.csv"
    report_path = workdir / "report.json"
    missing = [path.name for path in (panel_path, report_path) if not path.exists()]
    if missing:
        raise WorkshopAgentError(f"Missing required artifacts: {missing}")

    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    with panel_path.open(newline="", encoding="utf-8") as handle:
        panel = list(csv.DictReader(handle))
    if not panel:
        raise WorkshopAgentError("panel.csv is empty.")
    required_columns = {
        "smile",
        "canonical_ikey",
        "name",
        "reframedb_url",
        "selection_reason",
        "method_cluster",
        "selection_order",
    }
    missing_columns = required_columns - set(panel[0])
    if missing_columns:
        raise WorkshopAgentError(
            f"panel.csv is missing columns: {sorted(missing_columns)}"
        )
    if len(panel) != expected_panel_size:
        raise WorkshopAgentError(
            f"panel.csv has {len(panel)} rows; expected {expected_panel_size}."
        )
    candidate_keys = {row.get("canonical_ikey", "") for row in candidates}
    panel_keys = [row.get("canonical_ikey", "") for row in panel]
    if len(set(panel_keys)) != expected_panel_size or not set(panel_keys) <= candidate_keys:
        raise WorkshopAgentError("Panel connectivity keys are not unique input members.")
    if any(not row.get("reframedb_url", "").strip() for row in panel):
        raise WorkshopAgentError("Every selected compound must retain its ReFRAME URL.")
    try:
        orders = sorted(int(row["selection_order"]) for row in panel)
    except (TypeError, ValueError) as exc:
        raise WorkshopAgentError("selection_order must contain integers.") from exc
    if orders != list(range(1, expected_panel_size + 1)):
        raise WorkshopAgentError("selection_order must be contiguous and one-based.")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkshopAgentError("report.json is not valid JSON.") from exc
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
        "limitations",
        "files",
    }
    missing_report = required_report_keys - set(report)
    if missing_report:
        raise WorkshopAgentError(
            f"report.json is missing keys: {sorted(missing_report)}"
        )
    if report["candidate_count"] != len(candidates):
        raise WorkshopAgentError("report.json candidate_count does not match the input.")
    if report["panel_count"] != expected_panel_size:
        raise WorkshopAgentError("report.json panel_count does not match panel.csv.")
    unique_value = report["unique_ikeys"]
    if type(unique_value) is not int or unique_value != expected_panel_size:
        raise WorkshopAgentError(
            "report.json unique_ikeys must be the integer count of distinct panel "
            f"canonical_ikey values ({expected_panel_size})."
        )
    similarities = _similarity_values(report["pairwise_similarity"])
    if not similarities:
        raise WorkshopAgentError("report.json has no numeric similarity summaries.")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in similarities):
        raise WorkshopAgentError("Similarity summaries must be finite and in [0, 1].")
    if report["seed"] != 2026:
        raise WorkshopAgentError("report.json seed must equal the workshop seed 2026.")
    if not isinstance(report["parameters"], dict) or not report["parameters"]:
        raise WorkshopAgentError("report.json parameters must be a non-empty mapping.")
    descriptor_quantiles = report["descriptor_quantiles"]
    if not isinstance(descriptor_quantiles, dict) or not {
        "candidate",
        "panel",
    }.issubset(descriptor_quantiles):
        raise WorkshopAgentError(
            "report.json descriptor_quantiles must contain candidate and panel entries."
        )
    return {
        "candidate_count": len(candidates),
        "panel_count": len(panel),
        "similarity_summaries_checked": len(similarities),
    }


def _render_panel_analysis(approved_strategy: int, expected_panel_size: int) -> str:
    """Render a tested standalone analysis for one sponsor-approved strategy."""
    if approved_strategy not in (1, 2):
        raise ValueError("approved_strategy must be 1 or 2.")
    if expected_panel_size < 1:
        raise ValueError("expected_panel_size must be positive.")

    common_prefix = textwrap.dedent(
        f'''\
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        import torch
        from rdkit import Chem
        from nvmolkit.fingerprints import MorganFingerprintGenerator
        from nvmolkit.clustering import fused_butina
        from nvmolkit.similarity import crossTanimotoSimilarity

        SEED = 2026
        PANEL_SIZE = {int(expected_panel_size)}
        RADIUS = {2 if approved_strategy == 1 else 3}
        FP_BITS = {1024 if approved_strategy == 1 else 2048}
        DISTANCE_CUTOFF = 0.55
        SIMILARITY_TOLERANCE = 1e-5


        def summarize_descriptors(frame):
            return {{
                column: {{
                    "p05": float(frame[column].quantile(0.05)),
                    "median": float(frame[column].quantile(0.50)),
                    "p95": float(frame[column].quantile(0.95)),
                }}
                for column in ("MolWt", "cLogP", "TPSA")
            }}


        df = pd.read_csv("reframe_candidates.csv")
        required_columns = {{
            "smile", "canonical_ikey", "name", "status", "reframedb_url",
            "MolWt", "cLogP", "TPSA",
        }}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing input columns: {{sorted(missing_columns)}}")

        parsed_molecules = [Chem.MolFromSmiles(str(smile)) for smile in df["smile"]]
        valid_mask = np.asarray([molecule is not None for molecule in parsed_molecules])
        valid_df = df.loc[valid_mask].copy().reset_index(drop=True)
        molecules = [molecule for molecule in parsed_molecules if molecule is not None]
        if len(valid_df) < PANEL_SIZE:
            raise ValueError("Too few valid molecules for the requested panel size")

        status_text = valid_df["status"].fillna("").str.lower()
        valid_df["_availability_rank"] = np.select(
            [
                status_text.eq("plated and available for follow-up"),
                status_text.eq("not plated but available for one-off testing"),
            ],
            [0, 1],
            default=2,
        )

        fingerprint_generator = MorganFingerprintGenerator(radius=RADIUS, fpSize=FP_BITS)
        fingerprint_result = fingerprint_generator.GetFingerprints(
            molecules, num_threads=0
        )
        fingerprint_tensor = fingerprint_result.torch()
        if fingerprint_tensor.shape[0] != len(valid_df):
            raise ValueError("Fingerprint rows are not aligned with valid_df")
        similarity_matrix = crossTanimotoSimilarity(
            fingerprint_result, fingerprint_result
        ).numpy()
        if similarity_matrix.shape != (len(valid_df), len(valid_df)):
            raise ValueError("Unexpected all-pairs similarity shape")
        if not np.isfinite(similarity_matrix).all():
            raise ValueError("Similarity matrix contains non-finite values")
        raw_similarity_min = float(similarity_matrix.min())
        raw_similarity_max = float(similarity_matrix.max())
        if (
            raw_similarity_min < -SIMILARITY_TOLERANCE
            or raw_similarity_max > 1.0 + SIMILARITY_TOLERANCE
        ):
            raise ValueError(
                "Similarity matrix is meaningfully outside [0, 1]: "
                f"observed range [{{raw_similarity_min:.8g}}, "
                f"{{raw_similarity_max:.8g}}]"
            )
        # Tanimoto is mathematically bounded; remove only harmless GPU roundoff.
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        '''
    )

    if approved_strategy == 1:
        strategy_block = textwrap.dedent(
            '''\
            clusters, _, _ = fused_butina(
                fingerprint_tensor,
                cutoff=DISTANCE_CUTOFF,
                return_centroids=True,
            )
            cluster_labels = np.full(len(valid_df), -1, dtype=int)
            centroid_indices = []
            for cluster_id, members in enumerate(clusters):
                member_indices = [int(index) for index in members]
                cluster_labels[member_indices] = cluster_id
                centroid_indices.append(member_indices[0])
            if (cluster_labels < 0).any():
                raise ValueError("Every valid molecule must receive a cluster label")

            valid_df["method_cluster"] = cluster_labels
            cluster_sizes = np.bincount(cluster_labels)
            valid_df["_cluster_size"] = cluster_sizes[cluster_labels]
            valid_df["_mw_band"] = pd.cut(
                valid_df["MolWt"],
                [-np.inf, 300.0, 500.0, np.inf],
                labels=["low", "middle", "high"],
            )

            centroid_frame = valid_df.iloc[centroid_indices].copy()
            band_queues = {}
            for band in ("low", "middle", "high"):
                ordered = centroid_frame.loc[centroid_frame["_mw_band"] == band].sort_values(
                    ["_availability_rank", "_cluster_size", "canonical_ikey"],
                    ascending=[True, False, True],
                )
                band_queues[band] = [int(index) for index in ordered.index]

            selected_indices = []
            while len(selected_indices) < min(PANEL_SIZE, len(centroid_indices)):
                selected_this_round = False
                for band in ("low", "middle", "high"):
                    if band_queues[band] and len(selected_indices) < PANEL_SIZE:
                        selected_indices.append(band_queues[band].pop(0))
                        selected_this_round = True
                if not selected_this_round:
                    break

            if len(selected_indices) < PANEL_SIZE:
                remaining = valid_df.loc[~valid_df.index.isin(selected_indices)].sort_values(
                    ["_availability_rank", "_cluster_size", "canonical_ikey"],
                    ascending=[True, False, True],
                )
                selected_indices.extend(
                    int(index)
                    for index in remaining.index[: PANEL_SIZE - len(selected_indices)]
                )

            centroid_set = set(centroid_indices)
            panel_df = valid_df.iloc[selected_indices].copy()
            panel_df["selection_reason"] = [
                "Butina centroid with molecular-weight-band coverage"
                if index in centroid_set
                else "Deterministic availability-ranked cluster fill"
                for index in selected_indices
            ]
            strategy_name = "cluster_first_butina"
            '''
        )
    else:
        strategy_block = textwrap.dedent(
            '''\
            preferred_mask = valid_df["_availability_rank"].to_numpy() <= 1
            if int(preferred_mask.sum()) < PANEL_SIZE:
                preferred_mask = np.ones(len(valid_df), dtype=bool)

            descriptor_columns = ["MolWt", "cLogP", "TPSA"]
            descriptor_center = valid_df[descriptor_columns].median()
            descriptor_scale = valid_df[descriptor_columns].std().replace(0.0, 1.0)
            property_distance = (
                (valid_df[descriptor_columns] - descriptor_center) / descriptor_scale
            ).abs().sum(axis=1)

            selected_indices = []
            selected_mask = np.zeros(len(valid_df), dtype=bool)
            max_similarity = np.zeros(len(valid_df), dtype=float)
            while len(selected_indices) < PANEL_SIZE:
                available_indices = np.flatnonzero(preferred_mask & ~selected_mask)
                if len(available_indices) == 0:
                    available_indices = np.flatnonzero(~selected_mask)
                if len(available_indices) == 0:
                    raise ValueError("Candidate pool was exhausted before filling the panel")

                if not selected_indices:
                    selected_idx = int(min(
                        available_indices,
                        key=lambda index: (
                            float(property_distance.iloc[int(index)]),
                            str(valid_df.iloc[int(index)]["canonical_ikey"]),
                        ),
                    ))
                else:
                    scores = max_similarity[available_indices]
                    minimum_score = float(scores.min())
                    tied_indices = available_indices[
                        np.isclose(scores, minimum_score, rtol=0.0, atol=1e-12)
                    ]
                    selected_idx = int(min(
                        tied_indices,
                        key=lambda index: str(
                            valid_df.iloc[int(index)]["canonical_ikey"]
                        ),
                    ))

                selected_indices.append(selected_idx)
                selected_mask[selected_idx] = True
                max_similarity = np.maximum(
                    max_similarity, similarity_matrix[:, selected_idx]
                )
                max_similarity[selected_indices] = 1.0

            panel_df = valid_df.iloc[selected_indices].copy()
            panel_df["method_cluster"] = "not_clustered"
            panel_df["selection_reason"] = (
                "Greedy max-min fingerprint diversity with availability preference"
            )
            strategy_name = "greedy_max_min_similarity"
            '''
        )

    common_suffix = textwrap.dedent(
        f'''\
        if len(selected_indices) != PANEL_SIZE:
            raise ValueError("Selection did not produce the requested panel size")
        panel_df["selection_order"] = np.arange(1, PANEL_SIZE + 1)
        panel_df = panel_df.drop(
            columns=["_availability_rank", "_cluster_size", "_mw_band"],
            errors="ignore",
        ).reset_index(drop=True)
        if panel_df["canonical_ikey"].nunique() != PANEL_SIZE:
            raise ValueError("Selected connectivity keys are not unique")
        panel_df.to_csv("panel.csv", index=False)

        panel_similarity_matrix = similarity_matrix[
            np.ix_(selected_indices, selected_indices)
        ]
        upper_triangle = panel_similarity_matrix[
            np.triu_indices(PANEL_SIZE, k=1)
        ]
        candidate_descriptor_quantiles = summarize_descriptors(df)
        panel_descriptor_quantiles = summarize_descriptors(panel_df)

        if {approved_strategy} == 1:
            cluster_coverage = {{
                "candidate_clusters": int(len(clusters)),
                "panel_clusters": int(panel_df["method_cluster"].nunique()),
            }}
        else:
            cluster_coverage = {{
                "method": "not_clustered",
                "selected_compounds": int(len(panel_df)),
            }}

        report = {{
            "seed": 2026,
            "backend": "nvmolkit",
            "parameters": {{
                "strategy": strategy_name,
                "radius": RADIUS,
                "fp_bits": FP_BITS,
                "distance_cutoff": DISTANCE_CUTOFF if {approved_strategy} == 1 else None,
                "similarity_tolerance": SIMILARITY_TOLERANCE,
                "raw_similarity_range": [
                    raw_similarity_min,
                    raw_similarity_max,
                ],
            }},
            "candidate_count": int(len(df)),
            "panel_count": int(len(panel_df)),
            "unique_ikeys": int(panel_df["canonical_ikey"].nunique()),
            "descriptor_quantiles": {{
                "candidate": candidate_descriptor_quantiles,
                "panel": panel_descriptor_quantiles,
            }},
            "pairwise_similarity": {{
                "pair_count": int(len(upper_triangle)),
                "median": float(np.median(upper_triangle)),
                "p95": float(np.quantile(upper_triangle, 0.95)),
                "maximum": float(np.max(upper_triangle)),
            }},
            "cluster_coverage": cluster_coverage,
            "limitations": [
                "Fingerprint diversity is representation- and parameter-dependent.",
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
        print(f"Selected {{len(panel_df)}} compounds with {{strategy_name}}")
        '''
    )
    return common_prefix + "\n" + strategy_block + "\n" + common_suffix


class PanelDesignAgent:
    """Plan, render, execute, validate, and audit one bounded Module 3 analysis."""

    def __init__(
        self,
        *,
        workdir: Path,
        mission: str,
        api_key: str,
        client: Any = None,
    ) -> None:
        resolved = Path(workdir).resolve()
        if not resolved.is_dir() or not (resolved / "reframe_candidates.csv").is_file():
            raise ValueError("The agent workspace must contain reframe_candidates.csv.")
        self.workdir = resolved
        self.mission = mission.strip()
        self.api_key = get_workshop_api_key(api_key, prompt=False)
        self.client = client or _client(self.api_key)
        self.data_profile = profile_candidate_csv(resolved / "reframe_candidates.csv")
        self.plan: PanelPlan | None = None

    def request_plan(self) -> PanelPlan:
        """Ask Nemotron for two strategies without writing or running code."""
        prompt = (
            f"Mission:\n{self.mission}\n\n"
            "Deterministic input profile:\n"
            f"{json.dumps(self.data_profile, indent=2, sort_keys=True)}\n\n"
            "Evaluate exactly these two implementation families against the observed "
            "data profile. Strategy 1 is cluster-first: Morgan radius 2, 1024 bits, "
            "fused Butina distance cutoff 0.55, followed by molecular-weight-band "
            "interleaving. Strategy 2 is greedy max-min: Morgan radius 3, 2048 bits, "
            "all-pairs Tanimoto similarity, availability preference, and deterministic "
            "farthest-point selection. Explain the scientific tradeoffs, recommend one, "
            "and do not write code. The local controller owns the tested implementation."
        )
        self.plan = _structured_request(
            api_key=self.api_key,
            client=self.client,
            system_prompt=(
                "You are a scientific coding agent proposing an auditable chemical-"
                "library panel design. A human sponsor must approve a strategy before "
                "the local controller renders code. Molecular similarity is not "
                "biological activity."
            ),
            user_prompt=prompt,
            tool_name="submit_panel_plan",
            response_model=PanelPlan,
            max_tokens=2200,
        )
        return self.plan

    def _generation_prompt(self, approved_strategy: int) -> str:
        if self.plan is None:
            raise WorkshopAgentError("Request a plan before approving a strategy.")
        strategy = self.plan.strategies[approved_strategy - 1]
        return (
            "The sponsor approved this agent-proposed strategy. Render the matching "
            "tested local implementation and preserve the scientific rationale as a "
            "receipt. No hosted model will write executable source at this stage.\n\n"
            f"Approved strategy {approved_strategy}:\n"
            f"{json.dumps(strategy.model_dump(mode='json'), indent=2)}"
        )

    def _request_analysis(
        self,
        prompt: str,
        approved_strategy: int,
        expected_panel_size: int = 96,
    ) -> GeneratedAnalysis:
        del prompt
        source = _render_panel_analysis(approved_strategy, expected_panel_size)
        strategy = self.plan.strategies[approved_strategy - 1]
        return GeneratedAnalysis.model_construct(
            analysis_source=source,
            implementation_summary=(
                "Controller-rendered implementation of the sponsor-approved strategy: "
                + strategy.approach
            ),
            expected_tradeoffs=[strategy.tradeoff],
        )

    def _request_audit(self, approved_strategy: int) -> PanelAudit:
        report_text = (self.workdir / "report.json").read_text(encoding="utf-8")
        prompt = (
            "Review the independently validated result of the approved panel-design "
            "strategy. Identify an important or surprising observation even if the "
            "result is acceptable. Do not infer binding, activity, efficacy, safety, "
            "or clinical relevance.\n\n"
            f"Approved strategy {approved_strategy}:\n"
            f"{json.dumps(self.plan.strategies[approved_strategy - 1].model_dump(mode='json'), indent=2)}\n\n"
            f"Validated report.json:\n{report_text[:16000]}"
        )
        return _structured_request(
            api_key=self.api_key,
            client=self.client,
            system_prompt=(
                "You are the final scientific reviewer for a computational chemistry "
                "workshop. Interpret only the supplied structural and physicochemical "
                "receipts, state limitations explicitly, and propose a bounded next step."
            ),
            user_prompt=prompt,
            tool_name="submit_panel_audit",
            response_model=PanelAudit,
            max_tokens=1600,
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
        """Render, run, and independently validate one approved strategy."""
        if self.plan is None:
            raise WorkshopAgentError("Request and review a plan before running the agent.")
        if approved_strategy not in (1, 2):
            raise ValueError("approved_strategy must be 1 or 2.")
        if max_revisions not in range(0, 4):
            raise ValueError("max_revisions must be between 0 and 3.")
        executable = python_executable or sys.executable
        attempts: list[AgentAttempt] = []
        source = ""
        failure = ""

        def notify(event: str, **payload: Any) -> None:
            """Send optional presentation updates without changing scientific execution."""
            if progress_callback is None:
                return
            try:
                progress_callback(event, payload)
            except Exception:
                # A notebook rendering problem must not alter the agent's analysis.
                pass

        notify(
            "run_started",
            approved_strategy=approved_strategy,
            expected_panel_size=expected_panel_size,
            max_attempts=1,
        )

        # Executable source is controller-rendered, so model repair attempts no longer
        # exist.  The parameter remains accepted for older notebook callers.
        for attempt_number in (1,):
            elapsed = 0.0
            generated_summary = ""
            generated_tradeoffs: tuple[str, ...] = ()
            attempt_generated_source = ""
            attempt_source_name = "(no source file)"
            prompt = self._generation_prompt(approved_strategy)
            notify(
                "generation_started",
                attempt=attempt_number,
                is_revision=False,
            )
            try:
                generated = self._request_analysis(
                    prompt, approved_strategy, expected_panel_size
                )
                attempt_generated_source = generated.analysis_source
                generated_summary = generated.implementation_summary
                generated_tradeoffs = tuple(generated.expected_tradeoffs)
                rendered_source_path = (
                    self.workdir / f"analysis_attempt_{attempt_number}_rendered.py"
                )
                rendered_source_path.write_text(
                    attempt_generated_source, encoding="utf-8"
                )
                attempt_source_name = rendered_source_path.name
                notify(
                    "source_received",
                    attempt=attempt_number,
                    source_file=rendered_source_path.name,
                    source=attempt_generated_source,
                    implementation_summary=generated_summary,
                    expected_tradeoffs=list(generated_tradeoffs),
                )
                source = validate_panel_analysis_source(attempt_generated_source)
                rendered_source_path.write_text(source, encoding="utf-8")
                attempt_source = rendered_source_path
                (self.workdir / "analysis.py").write_text(source, encoding="utf-8")
                notify(
                    "source_generated",
                    attempt=attempt_number,
                    source_file=attempt_source.name,
                    source=source,
                    implementation_summary=generated_summary,
                    expected_tradeoffs=list(generated_tradeoffs),
                )

                # Exact outputs are cleared so stale artifacts cannot pass a later attempt.
                for artifact_name in ("panel.csv", "report.json"):
                    artifact = self.workdir / artifact_name
                    if artifact.exists():
                        artifact.unlink()

                child_environment = dict(os.environ)
                child_environment.pop("NVIDIA_API_KEY", None)
                notify(
                    "execution_started",
                    attempt=attempt_number,
                    timeout_seconds=timeout_seconds,
                )
                started = time.perf_counter()
                try:
                    completed = subprocess.run(
                        [executable, "analysis.py"],
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
                            f"analysis.py exited with code {completed.returncode}.\n"
                            f"STDERR:\n{completed.stderr[-6000:]}"
                        )
                    receipt = validate_panel_artifacts(
                        self.workdir, expected_panel_size=expected_panel_size
                    )
                    attempts.append(
                        AgentAttempt(
                            attempt_number,
                            attempt_source.name,
                            completed.returncode,
                            elapsed,
                            True,
                            json.dumps(receipt, sort_keys=True),
                            completed.stdout[-3000:],
                            completed.stderr[-3000:],
                            generated_summary,
                            generated_tradeoffs,
                            attempt_generated_source,
                        )
                    )
                    notify(
                        "attempt_passed",
                        attempt=attempt_number,
                        elapsed_seconds=elapsed,
                        receipt=receipt,
                        stdout_tail=completed.stdout[-3000:],
                    )
                    failure = ""
                    break
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.perf_counter() - started
                    raise WorkshopAgentError(
                        f"analysis.py exceeded the {timeout_seconds}-second limit."
                    ) from exc
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    AgentAttempt(
                        attempt_number,
                        attempt_source_name,
                        None,
                        float(elapsed),
                        False,
                        failure[-6000:],
                        "",
                        "",
                        generated_summary,
                        generated_tradeoffs,
                        attempt_generated_source,
                    )
                )
                notify(
                    "attempt_failed",
                    attempt=attempt_number,
                    elapsed_seconds=elapsed,
                    message=failure[-6000:],
                    will_revise=False,
                )
                break

        success = bool(attempts and attempts[-1].passed)
        audit: PanelAudit | None = None
        audit_error = ""
        if success:
            notify("audit_started")
            try:
                audit = self._request_audit(approved_strategy)
                notify("audit_completed", audit=audit.model_dump(mode="json"))
            except Exception as exc:
                # Valid artifacts remain usable even if the qualitative audit is unavailable.
                audit_error = f"{type(exc).__name__}: {exc}"
                notify("audit_failed", message=audit_error)
        trace_path = self.workdir / "agent_trace.json"
        trace_payload = {
            "workshop_agent_version": WORKSHOP_AGENT_VERSION,
            "model": DEFAULT_MODEL,
            "analysis_transport": "deterministic_strategy_renderer",
            "data_profile": self.data_profile,
            "plan": self.plan.model_dump(mode="json"),
            "approved_strategy": approved_strategy,
            "attempts": [asdict(item) for item in attempts],
            "success": success,
            "audit": audit.model_dump(mode="json") if audit is not None else None,
            "audit_error": audit_error,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
        result = PanelAgentRun(
            success=success,
            approved_strategy=approved_strategy,
            attempts=tuple(attempts),
            analysis_path=self.workdir / "analysis.py",
            panel_path=self.workdir / "panel.csv",
            report_path=self.workdir / "report.json",
            trace_path=trace_path,
            audit=audit,
        )
        notify(
            "run_completed",
            success=success,
            attempt_count=len(attempts),
            trace_path=str(trace_path),
        )
        return result
