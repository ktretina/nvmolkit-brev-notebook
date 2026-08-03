"""Strict hosted-Nemotron helpers for the guided nvMolKit notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
_REQUEST_ERROR = (
    "The hosted Nemotron request failed. Check network access and model availability."
)
_EMPTY_NARRATIVE_ERROR = "The hosted Nemotron narrative response was empty."


class ReadSkillArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class PrepareSampleArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    preview_count: Literal[24]


class FingerprintArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    fingerprint_radius: Literal[2, 3]
    fingerprint_size: Literal[1024, 2048]


class SimilarityArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ClusterArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    cluster_cutoff: float = Field(ge=0.40, le=0.60)


class ConformerArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    representative_count: int = Field(ge=3, le=6)
    conformers_per_representative: int = Field(ge=3, le=8)


TOOL_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "read_nvmolkit_skill": ReadSkillArgs,
    "prepare_molecular_sample": PrepareSampleArgs,
    "compute_morgan_fingerprints": FingerprintArgs,
    "compute_tanimoto_similarity": SimilarityArgs,
    "cluster_with_fused_butina": ClusterArgs,
    "generate_and_optimize_conformers": ConformerArgs,
}

TOOL_DESCRIPTIONS = {
    "read_nvmolkit_skill": "Read the pinned nvMolKit skill before scientific execution.",
    "prepare_molecular_sample": "Load and preview the bundled molecular sample.",
    "compute_morgan_fingerprints": "Compute bounded Morgan fingerprints for the prepared sample.",
    "compute_tanimoto_similarity": "Compute pairwise Tanimoto similarity from validated fingerprints.",
    "cluster_with_fused_butina": "Cluster the molecular library with fused Butina.",
    "generate_and_optimize_conformers": "Generate ETKDG conformers and optimize them with MMFF94.",
}

_TOOL_SYSTEM_PROMPT = """You select arguments for exactly one named, allow-listed
scientific function in a guided nvMolKit notebook. Call only the supplied tool and
call it exactly once. Use only the supplied task and JSON context. Do not request
code execution, dynamic imports, arbitrary tools, or additional functions. These
computational outputs do not establish binding, biological activity, ADMET,
efficacy, safety, synthesizability, clinical relevance, or experimentally
validated conformations."""

_BRIEF_SYSTEM_PROMPT = (
    "Give only 2-4 sentences interpreting the supplied scientific tool result "
    "conservatively. You are text-only: you do not receive figure pixels. You "
    "receive a figure_context description, axes, scale, and salient values; refer "
    "only to that supplied evidence. Do not claim binding, biological activity, "
    "ADMET, efficacy, safety, synthesizability, clinical relevance, or experimentally "
    "validated conformations. Distinguish computational descriptors and sampled "
    "force-field geometries from biological or experimental evidence."
)

_FINAL_SYSTEM_PROMPT = (
    "Write a PhD-level scientific synthesis that remains readable in a presentation. "
    "Use 450-650 words and all real stage summaries supplied in the JSON payload. "
    "Address these six themes explicitly: dataset "
    "validity and scope; molecular representation; pairwise similarity structure; "
    "clustering and library diversity; conformational sampling and MMFF94 "
    "convergence; limitations and appropriate next analyses. Make quantitative "
    "references only to supplied results and figure_context evidence. Do not claim "
    "binding, biological activity, ADMET, efficacy, safety, synthesizability, "
    "clinical relevance, or experimentally validated conformations. Distinguish "
    "within-molecule sampled force-field minima from global or experimental "
    "conformations."
)


class ToolCallError(RuntimeError):
    """A secret-safe failure to obtain or validate a hosted tool call."""


@dataclass(frozen=True)
class ToolDecision:
    arguments: BaseModel
    source: Literal["nemotron"]
    error: None
    tool_name: str
    tool_call_id: str
    raw_arguments: str


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, max_retries=0)


def _validate_api_key(api_key: str) -> None:
    if (
        not isinstance(api_key, str)
        or not api_key.startswith("nvapi-")
        or api_key != api_key.strip()
    ):
        raise ValueError(AUTH_GUIDANCE)


def _tool_definition(tool_name: str) -> dict[str, Any]:
    model = TOOL_ARGUMENT_MODELS[tool_name]
    parameters = model.model_json_schema()
    parameters["additionalProperties"] = False
    parameters["required"] = list(model.model_fields)
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": TOOL_DESCRIPTIONS[tool_name],
            "strict": True,
            "parameters": parameters,
        },
    }


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
    raise ToolCallError("The scientific result could not be serialized safely.")


def _serialize(value: Any) -> str:
    try:
        return json.dumps(
            _json_safe(value),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except Exception:
        raise ToolCallError(
            "The scientific result could not be serialized safely."
        ) from None


def _raise_request_error(exc: Exception) -> None:
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        raise ValueError(AUTH_GUIDANCE) from None
    raise ToolCallError(_REQUEST_ERROR) from None


def _validate_narrative_content(content: Any) -> str:
    if not isinstance(content, str) or not content.strip():
        raise ToolCallError(_EMPTY_NARRATIVE_ERROR)
    return content


def request_tool_call(
    api_key: str,
    *,
    tool_name: str,
    task_prompt: str,
    context: Any,
    model: str = DEFAULT_MODEL,
    client=None,
) -> ToolDecision:
    """Request and strictly validate one forced, allow-listed function call."""

    _validate_api_key(api_key)
    if tool_name not in TOOL_ARGUMENT_MODELS:
        raise ToolCallError("The requested tool is not allow-listed.")

    serialized_context = _serialize(context)
    try:
        active_client = client or _client(api_key)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TOOL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Task: {task_prompt}\nContext JSON: {serialized_context}",
                },
            ],
            tools=[_tool_definition(tool_name)],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=400,
            stream=False,
        )
    except Exception as exc:
        _raise_request_error(exc)

    try:
        tool_calls = response.choices[0].message.tool_calls
        if not isinstance(tool_calls, (list, tuple)):
            raise ToolCallError("The hosted tool call collection was malformed.")
        if len(tool_calls) != 1:
            raise ToolCallError("Expected exactly one hosted tool call.")
        tool_call = tool_calls[0]
        if getattr(tool_call, "type", None) != "function":
            raise ToolCallError("The hosted tool call type was invalid.")
        function = getattr(tool_call, "function", None)
        if function is None:
            raise ToolCallError("The hosted function call was malformed.")
        returned_name = getattr(function, "name", None)
        if returned_name != tool_name:
            raise ToolCallError("The hosted tool call named an unexpected function.")
        tool_call_id = getattr(tool_call, "id", None)
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ToolCallError("The hosted tool call ID was missing.")
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            raise ToolCallError("The hosted tool arguments were missing.")
        try:
            decoded = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            raise ToolCallError("The hosted tool arguments were not valid JSON.") from None
        if not isinstance(decoded, dict):
            raise ToolCallError("The hosted tool arguments must be a JSON object.")
        arguments = TOOL_ARGUMENT_MODELS[tool_name].model_validate(decoded)
    except ToolCallError:
        raise
    except (AttributeError, IndexError, TypeError, ValidationError):
        raise ToolCallError("The hosted tool arguments failed strict validation.") from None

    return ToolDecision(
        arguments=arguments,
        source="nemotron",
        error=None,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        raw_arguments=raw_arguments,
    )


def request_and_execute_step(
    api_key: str,
    *,
    tool_name: str,
    task_prompt: str,
    context: Any,
    executor,
    model: str = DEFAULT_MODEL,
    client=None,
) -> tuple[ToolDecision, Any]:
    """Execute once, and only after the hosted call passes strict validation."""

    decision = request_tool_call(
        api_key,
        tool_name=tool_name,
        task_prompt=task_prompt,
        context=context,
        model=model,
        client=client,
    )
    result = executor(decision.arguments)
    return decision, result


def request_brief_interpretation(
    api_key: str,
    decision: ToolDecision,
    tool_result: Any,
    figure_context: Any,
    model: str = DEFAULT_MODEL,
    client=None,
) -> str:
    """Continue the tool-call exchange for a bounded section interpretation."""

    _validate_api_key(api_key)
    serialized_result = _serialize(
        {"tool_result": tool_result, "figure_context": figure_context}
    )
    try:
        active_client = client or _client(api_key)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _BRIEF_SYSTEM_PROMPT},
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
                    "content": serialized_result,
                },
            ],
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=240,
            stream=False,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        _raise_request_error(exc)
    return _validate_narrative_content(content)


def request_final_synthesis(
    api_key: str,
    analysis_summary: Any,
    model: str = DEFAULT_MODEL,
    client=None,
) -> str:
    """Request a detailed synthesis from JSON-safe summaries of completed stages."""

    _validate_api_key(api_key)
    serialized_summary = _serialize(analysis_summary)
    try:
        active_client = client or _client(api_key)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": serialized_summary},
            ],
            extra_body=NEMOTRON_TOOL_EXTRA_BODY,
            temperature=0.2,
            max_tokens=1000,
            stream=False,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        _raise_request_error(exc)
    return _validate_narrative_content(content)
