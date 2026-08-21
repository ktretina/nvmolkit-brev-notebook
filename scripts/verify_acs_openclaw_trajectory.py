import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


RUNNER_PREFIX: Final = (
    "env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 "
    "/sandbox/.openclaw/workspace/acs_workshop_runner.py"
)
LESSON_COMMANDS: Final = (
    f"{RUNNER_PREFIX} run-lesson data-and-representation",
    f"{RUNNER_PREFIX} run-lesson relationships-and-groups",
    f"{RUNNER_PREFIX} run-lesson sampled-3d-geometry",
)
OBJECTIVE_START: Final = f"{RUNNER_PREFIX} objective-start"
OBJECTIVE_STEP_RE: Final = re.compile(
    re.escape(RUNNER_PREFIX)
    + r" objective-step --state-id '([^'\n]+)' --swap-id '([^'\n]+)'\Z"
)
PROMPT_IDS: Final = (
    "01-data-and-representation",
    "02-relationships-and-groups",
    "03-sampled-3d-geometry",
    "04-objective",
)
PROMPT_SHA256: Final = (
    "39ca26c1b494dbe01bcbaabf27d72d755b444915e9ff26c874e629f09610bf22",
    "5d556991910812a24bb09b23cd250fd4a7157986948082fb8cc05cb3d52c1f5e",
    "6779b1bfbe141a72c795d5e648ad33a5e7ddd55a8bc953b0c1ae116f757be34a",
    "ec93fcfa236b6000980178626b322aeb0786a52a53a0132338784221c24550ea",
)
HEADINGS: Final = (
    "## Question",
    "## What ran",
    "## Measured result",
    "## Meaning",
    "## Scientific limit",
    "## Image and download location",
)
_DOWNLOAD_SENTENCE: Final = (
    "The current bundle is in **Download Results** at `workshop/results.zip`."
)
_OBJECTIVE_MEANING_INCREASED: Final = (
    "A larger `D_min` means the least separated pair in the selected panel became "
    "more separated in this fingerprint space."
)
_OBJECTIVE_MEANING_UNCHANGED: Final = (
    "`D_min` did not increase; the weakest-link separation remained unchanged in "
    "this fingerprint space."
)
_OBJECTIVE_SCIENTIFIC_LIMIT: Final = (
    "`D_min` is the minimum pairwise Tanimoto distance, "
    "`min(1 - Tanimoto similarity)`, and the weakest-link diversity score within "
    "eight fixed candidates. This structural-descriptor objective does not "
    "demonstrate unrestricted autonomous design or biological performance."
)
PROMPT_MEDIA: Final = (
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "01-inspection/library_preview.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "06-mmff94/optimized_structures.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png",
)
REQUIRED_ZIP_MEMBERS: Final = (
    "README.md",
    "data/sample_molecules.csv",
    "data/PROVENANCE.md",
    "01-inspection/README.md",
    "01-inspection/summary.json",
    "01-inspection/library_preview.png",
    "02-fingerprints/README.md",
    "02-fingerprints/summary.json",
    "02-fingerprints/fingerprint_density.png",
    "03-similarity/README.md",
    "03-similarity/summary.json",
    "03-similarity/similarity_heatmap.png",
    "03-similarity/top_similarity_pairs.csv",
    "03-similarity/similarity_matrix.csv",
    "04-clusters/README.md",
    "04-clusters/summary.json",
    "04-clusters/cluster_sizes.png",
    "04-clusters/cluster_assignments.csv",
    "05-conformers/README.md",
    "05-conformers/summary.json",
    "05-conformers/embedding_counts.png",
    "06-mmff94/README.md",
    "06-mmff94/summary.json",
    "06-mmff94/conformer_energies.png",
    "06-mmff94/optimized_structures.png",
    "06-mmff94/mmff94_energies.csv",
    "06-mmff94/optimized_conformers.sdf",
    "06-mmff94/workflow_evidence.json",
    "07-objective/README.md",
    "07-objective/objective_summary.json",
    "07-objective/objective_evidence.json",
    "07-objective/score_trajectory.png",
    "07-objective/final_panel.png",
    "07-objective/final_similarity_heatmap.png",
)
REQUIRED_CHAT_PNGS: Final = (
    "01-inspection/library_preview.png",
    "04-clusters/cluster_sizes.png",
    "06-mmff94/optimized_structures.png",
    "07-objective/final_panel.png",
)
MAX_TRAJECTORY_BYTES: Final = 16 * 1024 * 1024
MAX_TRAJECTORY_LINES: Final = 4096
MAX_MEMBER_BYTES: Final = 8 * 1024 * 1024
MAX_EXPANDED_BYTES: Final = 32 * 1024 * 1024
_MAX_PAGE_BYTES: Final = 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 64 * 1024 * 1024
_SCORE_SCALE: Final = 10**12
_EVENT_KEYS: Final = {
    "data",
    "modelApi",
    "modelId",
    "provider",
    "runId",
    "schemaVersion",
    "seq",
    "sessionId",
    "sessionKey",
    "source",
    "sourceSeq",
    "traceId",
    "traceSchema",
    "ts",
    "type",
    "workspaceDir",
}
_EVENT_STRING_KEYS: Final = (
    "modelApi",
    "modelId",
    "provider",
    "runId",
    "sessionId",
    "sessionKey",
    "source",
    "traceId",
    "type",
    "workspaceDir",
)
_ASSISTANT_KEYS: Final = {
    "api",
    "content",
    "model",
    "provider",
    "responseId",
    "role",
    "stopReason",
    "timestamp",
    "usage",
}
_TOOL_RESULT_KEYS: Final = {
    "content",
    "isError",
    "role",
    "timestamp",
    "toolCallId",
    "toolName",
}
_ACTION_KEYS: Final = {
    "swap_id",
    "replace_id",
    "replacement_id",
    "resulting_ids",
    "predicted_score",
    "score_delta",
    "limiting_pairs",
    "target_status",
}
_MEASUREMENT_KEYS: Final = {
    "selected_ids",
    "score",
    "limiting_pairs",
    "achieved",
}
_ATTEMPT_KEYS: Final = {
    "attempt_number",
    "state_id",
    "selected_ids",
    "score",
    "limiting_pairs",
    "achieved",
    "selected_swap",
}
_TARGET_STATUSES: Final = {"below_target", "meets_target"}
_TERMINATION_REASONS: Final = {
    "target_achieved",
    "baseline_already_optimal",
    "attempt_limit_reached",
    "no_legal_improving_swap",
    "objective_correction_limit",
    "objective_provider_failure",
    "evaluation_not_completed",
}
_LESSON_RESULT_KEYS: Final = {
    "schema_version",
    "status",
    "lesson",
    "completed_stages",
    "results_zip_path",
    "artifact_relative_zip_path",
    "answer_markdown",
}
_LESSON_STAGES: Final = {
    "data-and-representation": (
        "inspect_library",
        "generate_morgan_fingerprints",
    ),
    "relationships-and-groups": (
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
    ),
    "sampled-3d-geometry": (
        "embed_representative_conformers",
        "optimize_conformers_mmff94",
    ),
}
_LESSON_QUESTIONS: Final = {
    "data-and-representation": (
        "What is in the fixed molecule library, and how is it represented for "
        "comparison?"
    ),
    "relationships-and-groups": (
        "Which molecules are similar, and how does Butina group them from distances "
        "derived from GPU-computed Tanimoto similarities?"
    ),
    "sampled-3d-geometry": ("What sampled 3D geometries were generated and optimized?"),
}
_LESSON_MEANINGS: Final = {
    "data-and-representation": (
        "The validated molecules were converted into fixed-length structural "
        "descriptors that support comparisons within this exercise."
    ),
    "relationships-and-groups": (
        "The similarity stage compares the fixed fingerprints; Butina then groups "
        "molecules whose Tanimoto distances satisfy the fixed rule."
    ),
    "sampled-3d-geometry": (
        "The fixed representatives received deterministic ETKDGv3 conformer "
        "samples followed by within-molecule MMFF94 optimization."
    ),
}
_LESSON_SCIENTIFIC_LIMITS: Final = {
    "data-and-representation": (
        "This is a deterministic 256-record ChEMBL convenience sample, not "
        "representative chemical space. Morgan and Tanimoto conclusions depend "
        "on the radius-2, 1024-bit hashed fingerprint."
    ),
    "relationships-and-groups": (
        "The cutoff 0.40 is Tanimoto distance, not similarity. Results depend on "
        "the radius-2, 1024-bit hashed fingerprint, and similarity 1.0 does not "
        "prove molecular identity or biological behavior."
    ),
    "sampled-3d-geometry": (
        "The selected molecules are not centroids, medoids, or globally optimal "
        "representatives. Sampled conformers are not experimental structures, "
        "and MMFF94 energies compare sampled conformers within one molecule only."
    ),
}
_STAGE_EXECUTION: Final = {
    "inspect_library": (
        "CPU",
        "RDKit",
        "library parsing and validation",
        None,
    ),
    "generate_morgan_fingerprints": (
        "GPU",
        "nvMolKit",
        "Morgan fingerprint generation",
        None,
    ),
    "measure_tanimoto_similarity": (
        "GPU",
        "nvMolKit",
        "Tanimoto similarity calculation",
        None,
    ),
    "discover_fused_butina_clusters": (
        "CPU",
        "RDKit",
        "Butina clustering",
        "measure_tanimoto_similarity",
    ),
    "embed_representative_conformers": (
        "GPU",
        "nvMolKit",
        "ETKDGv3 conformer embedding",
        None,
    ),
    "optimize_conformers_mmff94": (
        "GPU",
        "nvMolKit",
        "MMFF94 conformer optimization",
        None,
    ),
}
_STAGE_ITEM_KEYS: Final = {
    "stage",
    "result",
    "execution",
    "image_paths",
    "summary_path",
    "readme_path",
    "artifact_directory",
}
_EXECUTION_KEYS: Final = {
    "placement",
    "software",
    "operation",
    "upstream",
    "gpu",
}
_GPU_KEYS: Final = {
    "name",
    "device",
    "torch_version",
    "nvmolkit_version",
}
_UPSTREAM_KEYS: Final = {"stage", "placement", "software", "operation"}
_STAGE_IMAGE_COUNTS: Final = {
    "inspect_library": 1,
    "generate_morgan_fingerprints": 1,
    "measure_tanimoto_similarity": 1,
    "discover_fused_butina_clusters": 1,
    "embed_representative_conformers": 1,
    "optimize_conformers_mmff94": 2,
}
_UINT_PATTERN: Final = r"(?:0|[1-9][0-9]*)"
_FIXED_2_PATTERN: Final = rf"{_UINT_PATTERN}\.[0-9]{{2}}"
_FIXED_3_PATTERN: Final = rf"{_UINT_PATTERN}\.[0-9]{{3}}"
_SIGNED_FIXED_3_PATTERN: Final = rf"-?{_FIXED_3_PATTERN}"
_VERSION_TOKEN_RE: Final = re.compile(r"\A[A-Za-z0-9.+_-]{1,64}\Z")
_SAFE_IDENTIFIER_RE: Final = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")
_FORBIDDEN_IDENTIFIER_FRAGMENTS: Final = (
    "accelerat",
    "speedup",
    "faster",
    "similarityscore",
    "intermediate",
    "predicted",
    "target",
    "perstep",
)
_GpuIdentity = tuple[str, str, str, str]
_JSON_STRING_PATTERN: Final = (
    r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
)
_INSPECT_RESULT_RE: Final = re.compile(
    rf"\A(?P<raw>{_UINT_PATTERN}) raw rows; "
    rf"(?P<valid>{_UINT_PATTERN}) valid molecules; "
    rf"(?P<invalid>{_UINT_PATTERN}) invalid molecules; "
    rf"(?P<preview>{_UINT_PATTERN}) molecules in the preview\.\Z"
)
_MORGAN_RESULT_RE: Final = re.compile(
    rf"\AMorgan radius (?P<radius>{_UINT_PATTERN}) with "
    rf"(?P<bits>{_UINT_PATTERN}) bits produced packed shape "
    rf"(?P<rows>{_UINT_PATTERN}) x (?P<columns>{_UINT_PATTERN}); "
    rf"active bits min (?P<minimum>{_UINT_PATTERN}), median "
    rf"(?P<median>{_FIXED_3_PATTERN}), max (?P<maximum>{_UINT_PATTERN})\.\Z"
)
_SIMILARITY_RESULT_RE: Final = re.compile(
    rf"\Atop non-self pair (?P<first>{_JSON_STRING_PATTERN}) and "
    rf"(?P<second>{_JSON_STRING_PATTERN}) had Tanimoto similarity "
    rf"(?P<similarity>{_FIXED_3_PATTERN}); q1 "
    rf"(?P<q1>{_FIXED_3_PATTERN}), median (?P<median>{_FIXED_3_PATTERN}), "
    rf"q3 (?P<q3>{_FIXED_3_PATTERN}), p90 (?P<p90>{_FIXED_3_PATTERN})\.\Z"
)
_CLUSTER_RESULT_RE: Final = re.compile(
    rf"\Acutoff (?P<cutoff>{_FIXED_2_PATTERN}) produced "
    rf"(?P<clusters>{_UINT_PATTERN}) clusters with "
    rf"(?P<singletons>{_UINT_PATTERN}) singletons; largest cluster sizes: "
    rf"(?P<sizes>(?:{_UINT_PATTERN}(?:, {_UINT_PATTERN})*)?)\.\Z"
)
_EMBED_RESULT_RE: Final = re.compile(
    rf"\Aselected (?P<selected>{_UINT_PATTERN}) of "
    rf"(?P<requested_representatives>{_UINT_PATTERN}) representatives and "
    rf"generated (?P<generated>{_UINT_PATTERN}) of "
    rf"(?P<requested_conformers>{_UINT_PATTERN}) requested conformers; "
    rf"(?P<partial>{_UINT_PATTERN}) partial ID, "
    rf"(?P<zero>{_UINT_PATTERN}) zero IDs; ETKDGv3 seed 7\.\Z"
)
_MINIMUM_ENTRY_PATTERN: Final = (
    rf"{_JSON_STRING_PATTERN}={_SIGNED_FIXED_3_PATTERN} kcal/mol"
)
_OPTIMIZE_RESULT_RE: Final = re.compile(
    rf"\A(?P<attempted>{_UINT_PATTERN}) conformers attempted; "
    rf"(?P<converged>{_UINT_PATTERN}) converged; "
    rf"(?P<unconverged>{_UINT_PATTERN}) unconverged; "
    rf"within-molecule minima: (?P<minima>none|{_MINIMUM_ENTRY_PATTERN}"
    rf"(?:, {_MINIMUM_ENTRY_PATTERN})*); maximum iterations 500\.\Z"
)
_MINIMUM_ENTRY_RE: Final = re.compile(
    rf"(?P<identifier>{_JSON_STRING_PATTERN})="
    rf"(?P<energy>{_SIGNED_FIXED_3_PATTERN}) kcal/mol"
)
_PENDING_RESULT_KEYS: Final = {
    "schema_version",
    "status",
    "terminal",
    "attempt_count",
    "attempt_limit",
    "state_id",
    "current",
    "target_score",
    "actions",
    "achieved",
    "termination_reason",
    "image_paths",
    "artifact_directory",
    "results_zip_path",
    "artifact_relative_zip_path",
}
_TERMINAL_RESULT_KEYS: Final = {
    "schema_version",
    "status",
    "terminal",
    "attempt_count",
    "attempt_limit",
    "baseline",
    "target_score",
    "final",
    "attempts",
    "achieved",
    "termination_reason",
    "image_paths",
    "artifact_directory",
    "results_zip_path",
    "artifact_relative_zip_path",
    "answer_markdown",
}


class VerificationError(RuntimeError):
    ALLOWED = {
        "invalid_evidence",
        "prompt_contract",
        "command_contract",
        "objective_contract",
        "answer_contract",
        "archive_contract",
    }

    def __init__(self, code: str) -> None:
        if code not in self.ALLOWED:
            code = "invalid_evidence"
        super().__init__(code)
        self.code = code


def _read_regular(path: Path, maximum: int, code: str) -> bytes:
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise VerificationError(code)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(code) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not 0 <= metadata.st_size <= maximum
        ):
            raise VerificationError(code)
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise VerificationError(code)
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise VerificationError(code)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size:
            raise VerificationError(code)
        return raw
    except OSError as error:
        raise VerificationError(code) from error
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _json_object(text: str, code: str) -> dict[str, Any]:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(code)
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise VerificationError(code)

    try:
        value = json.loads(
            text,
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise VerificationError(code) from error
    if type(value) is not dict:
        raise VerificationError(code)
    return value


def load_prompt_contracts(page_path: Path) -> tuple[tuple[str, str, str], ...]:
    raw = _read_regular(page_path, _MAX_PAGE_BYTES, "prompt_contract")
    try:
        source = raw.decode("utf-8")
    except UnicodeError as error:
        raise VerificationError("prompt_contract") from error
    expected_markers = tuple(
        marker
        for prompt_id in PROMPT_IDS
        for marker in (
            f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->",
            f"<!-- ACS_PROMPT:{prompt_id}:END -->",
        )
    )
    observed_markers = tuple(
        re.findall(r"<!-- ACS_PROMPT:[a-z0-9-]+:(?:BEGIN|END) -->", source)
    )
    if observed_markers != expected_markers or source.count("<!-- ACS_PROMPT:") != len(
        expected_markers
    ):
        raise VerificationError("prompt_contract")
    contracts: list[tuple[str, str, str]] = []
    prior_end = -1
    for prompt_id, media_line, expected_digest in zip(
        PROMPT_IDS, PROMPT_MEDIA, PROMPT_SHA256, strict=True
    ):
        begin = f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->"
        end = f"<!-- ACS_PROMPT:{prompt_id}:END -->"
        begin_index = source.find(begin)
        end_index = source.find(end, begin_index + len(begin))
        if begin_index <= prior_end or end_index <= begin_index:
            raise VerificationError("prompt_contract")
        region = source[begin_index + len(begin) : end_index]
        opening_fence = "~~~text\n"
        closing_fence = "\n~~~\n"
        if region.count(opening_fence) != 1 or region.count(closing_fence) != 1:
            raise VerificationError("prompt_contract")
        opening_index = region.find(opening_fence)
        prompt_start = opening_index + len(opening_fence)
        prompt_end = region.find(closing_fence, prompt_start)
        if opening_index < 0 or prompt_end < prompt_start:
            raise VerificationError("prompt_contract")
        if (
            region[:opening_index].strip()
            or region[prompt_end + len(closing_fence) :].strip()
        ):
            raise VerificationError("prompt_contract")
        prompt = region[prompt_start:prompt_end]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if not prompt.endswith(media_line) or digest != expected_digest:
            raise VerificationError("prompt_contract")
        contracts.append((prompt_id, prompt, digest))
        prior_end = end_index
    return tuple(contracts)


def load_messages_snapshot(path: Path) -> list[dict[str, Any]]:
    raw = _read_regular(path, MAX_TRAJECTORY_BYTES, "invalid_evidence")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise VerificationError("invalid_evidence") from error
    if not 1 <= len(lines) <= MAX_TRAJECTORY_LINES or any(not line for line in lines):
        raise VerificationError("invalid_evidence")
    latest: list[dict[str, Any]] | None = None
    latest_identity: tuple[str, str, str] | None = None
    for line in lines:
        event = _json_object(line, "invalid_evidence")
        data = event.get("data")
        if (
            set(event) != _EVENT_KEYS
            or type(data) is not dict
            or event.get("schemaVersion") != 1
            or type(event.get("schemaVersion")) is not int
            or event.get("traceSchema") != "openclaw-trajectory"
            or any(
                type(event.get(key)) is not str or not event[key]
                for key in _EVENT_STRING_KEYS
            )
            or any(
                type(event.get(key)) is not int or event[key] < 0
                for key in ("seq", "sourceSeq", "ts")
            )
        ):
            raise VerificationError("invalid_evidence")
        if "messagesSnapshot" not in data:
            continue
        if set(data) != {"messagesSnapshot"}:
            raise VerificationError("invalid_evidence")
        snapshot = data["messagesSnapshot"]
        if (
            type(snapshot) is not list
            or not snapshot
            or any(type(message) is not dict for message in snapshot)
        ):
            raise VerificationError("invalid_evidence")
        latest = snapshot
        latest_identity = (event["modelApi"], event["modelId"], event["provider"])
    if latest is None or latest_identity is None:
        raise VerificationError("invalid_evidence")
    expected_api, expected_model, expected_provider = latest_identity
    for message in latest:
        if message.get("role") == "assistant" and (
            message.get("api") != expected_api
            or message.get("model") != expected_model
            or message.get("provider") != expected_provider
        ):
            raise VerificationError("invalid_evidence")
    return latest


def _single_text(message: Mapping[str, Any], code: str) -> str:
    content = message.get("content")
    if type(content) is str:
        return content
    if (
        type(content) is list
        and len(content) == 1
        and type(content[0]) is dict
        and set(content[0]) == {"type", "text"}
        and content[0]["type"] == "text"
        and type(content[0]["text"]) is str
    ):
        return content[0]["text"]
    raise VerificationError(code)


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _assistant_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if (
        set(message) != _ASSISTANT_KEYS
        or message.get("role") != "assistant"
        or any(
            type(message.get(key)) is not str or not message[key]
            for key in (
                "api",
                "model",
                "provider",
                "responseId",
                "stopReason",
            )
        )
        or not _nonnegative_int(message.get("timestamp"))
        or type(message.get("usage")) is not dict
        or type(content) is not list
        or len(content) != 1
        or type(content[0]) is not dict
    ):
        raise VerificationError("command_contract")
    block = content[0]
    block_type = block.get("type")
    expected_stop_reason = (
        "toolUse"
        if block_type == "toolCall"
        else "stop"
        if block_type == "text"
        else None
    )
    if (
        expected_stop_reason is None
        or message.get("stopReason") != expected_stop_reason
    ):
        raise VerificationError("command_contract")
    return [block]


def _validate_user_message(message: Mapping[str, Any]) -> str:
    if (
        set(message) != {"content", "role", "timestamp"}
        or message.get("role") != "user"
        or not _nonnegative_int(message.get("timestamp"))
    ):
        raise VerificationError("prompt_contract")
    return _single_text(message, "prompt_contract")


def _tool_result_object(message: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = set(message)
    content = message.get("content")
    if (
        keys not in (_TOOL_RESULT_KEYS, _TOOL_RESULT_KEYS | {"details"})
        or message.get("role") != "toolResult"
        or message.get("toolName") != "exec"
        or message.get("isError") is not False
        or not _nonnegative_int(message.get("timestamp"))
        or type(message.get("toolCallId")) is not str
        or not message["toolCallId"]
        or ("details" in message and type(message["details"]) is not dict)
        or type(content) is not list
        or len(content) != 1
        or type(content[0]) is not dict
        or set(content[0]) != {"type", "text"}
        or content[0].get("type") != "text"
        or type(content[0].get("text")) is not str
    ):
        raise VerificationError("command_contract")
    return (
        message["toolCallId"],
        _json_object(content[0]["text"], "command_contract"),
    )


@dataclass(frozen=True)
class _CallResult:
    call_id: str
    command: str
    result: dict[str, Any]


def _tool_call(block: Mapping[str, Any]) -> tuple[str, str]:
    if (
        set(block) != {"type", "name", "id", "arguments", "partialArgs"}
        or block.get("type") != "toolCall"
        or block.get("name") != "exec"
        or type(block.get("id")) is not str
        or not block["id"]
        or type(block.get("arguments")) is not dict
        or set(block["arguments"]) != {"command"}
        or type(block["arguments"].get("command")) is not str
        or not block["arguments"]["command"]
        or type(block.get("partialArgs")) is not str
    ):
        raise VerificationError("command_contract")
    partial = _json_object(block["partialArgs"], "command_contract")
    if partial != block["arguments"]:
        raise VerificationError("command_contract")
    return block["id"], block["arguments"]["command"]


def _turn_evidence(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[tuple[_CallResult, ...], str]:
    call_order: list[str] = []
    commands: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    text_events: list[tuple[int, str]] = []
    first_result_position = -1
    last_result_position = -1
    pending_call_id: str | None = None
    position = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            block = _assistant_blocks(message)[0]
            position += 1
            if pending_call_id is not None:
                raise VerificationError("command_contract")
            if block.get("type") == "text":
                if set(block) != {"type", "text"} or type(block.get("text")) is not str:
                    raise VerificationError("command_contract")
                text_events.append((position, block["text"]))
                continue
            call_id, command = _tool_call(block)
            if (
                call_id in commands
                or pending_call_id is not None
                or any(text.strip() for _, text in text_events)
            ):
                raise VerificationError("command_contract")
            call_order.append(call_id)
            commands[call_id] = command
            pending_call_id = call_id
        elif role == "toolResult":
            position += 1
            call_id, result = _tool_result_object(message)
            if (
                call_id not in commands
                or call_id in results
                or call_id != pending_call_id
            ):
                raise VerificationError("command_contract")
            results[call_id] = result
            if first_result_position < 0:
                first_result_position = position
            last_result_position = position
            pending_call_id = None
        else:
            raise VerificationError("command_contract")
    if pending_call_id is not None or set(results) != set(commands):
        raise VerificationError("command_contract")
    if any(
        event_position > first_result_position and not text.strip()
        for event_position, text in text_events
    ):
        raise VerificationError("answer_contract")
    nonempty = [
        (event_position, text) for event_position, text in text_events if text.strip()
    ]
    if (
        len(nonempty) != 1
        or nonempty[0][0] <= last_result_position
        or first_result_position < 0
    ):
        raise VerificationError("answer_contract")
    calls = tuple(
        _CallResult(call_id, commands[call_id], results[call_id])
        for call_id in call_order
    )
    return calls, nonempty[0][1]


def _finite_score(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(code)
    number = float(value)
    if not math.isfinite(number):
        raise VerificationError(code)
    return number


def _finite_float(value: object, code: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise VerificationError(code)
    return value


def _objective_score(value: object, code: str) -> float:
    number = _finite_float(value, code)
    if not 0.0 <= number <= 1.0:
        raise VerificationError(code)
    return number


def _score_key(value: float) -> int:
    numerator, denominator = value.as_integer_ratio()
    return (2 * numerator * _SCORE_SCALE + denominator) // (2 * denominator)


def _target_achieved(score: float, target: float) -> bool:
    return _score_key(score) >= _score_key(target)


def _schema_version_one(result: Mapping[str, Any]) -> bool:
    return type(result.get("schema_version")) is int and result["schema_version"] == 1


def _nonempty_string(result: Mapping[str, Any], key: str) -> bool:
    return type(result.get(key)) is str and bool(result[key])


def _valid_artifact_locations(result: Mapping[str, Any]) -> bool:
    return (
        _nonempty_string(result, "artifact_directory")
        and _nonempty_string(result, "results_zip_path")
        and result.get("artifact_relative_zip_path") == "workshop/results.zip"
    )


def _validate_gpu_identity(value: object, *, required: bool) -> _GpuIdentity | None:
    if not required:
        if value is not None:
            raise VerificationError("command_contract")
        return None
    if type(value) is not dict or set(value) != _GPU_KEYS:
        raise VerificationError("command_contract")
    torch_version = value.get("torch_version")
    nvmolkit_version = value.get("nvmolkit_version")
    if (
        value.get("name") != "NVIDIA L4"
        or value.get("device") != "cuda:0"
        or type(torch_version) is not str
        or _VERSION_TOKEN_RE.fullmatch(torch_version) is None
        or type(nvmolkit_version) is not str
        or _VERSION_TOKEN_RE.fullmatch(nvmolkit_version) is None
    ):
        raise VerificationError("command_contract")
    return ("NVIDIA L4", "cuda:0", torch_version, nvmolkit_version)


def _validate_stage_execution(value: object, stage: str) -> _GpuIdentity | None:
    if type(value) is not dict or set(value) != _EXECUTION_KEYS:
        raise VerificationError("command_contract")
    placement, software, operation, upstream_stage = _STAGE_EXECUTION[stage]
    if (
        value.get("placement") != placement
        or value.get("software") != software
        or value.get("operation") != operation
    ):
        raise VerificationError("command_contract")
    upstream = value.get("upstream")
    if upstream_stage is None:
        if upstream is not None:
            raise VerificationError("command_contract")
    else:
        if type(upstream) is not dict or set(upstream) != _UPSTREAM_KEYS:
            raise VerificationError("command_contract")
        upstream_placement, upstream_software, upstream_operation, _ = _STAGE_EXECUTION[
            upstream_stage
        ]
        if upstream != {
            "stage": upstream_stage,
            "placement": upstream_placement,
            "software": upstream_software,
            "operation": upstream_operation,
        }:
            raise VerificationError("command_contract")
    return _validate_gpu_identity(value.get("gpu"), required=stage != "inspect_library")


def _result_match(pattern: re.Pattern[str], value: object) -> re.Match[str]:
    if type(value) is not str or not 1 <= len(value) <= 4096:
        raise VerificationError("command_contract")
    match = pattern.fullmatch(value)
    if match is None:
        raise VerificationError("command_contract")
    return match


def _result_int(match: re.Match[str], group: str) -> int:
    try:
        return int(match.group(group))
    except (ValueError, OverflowError) as error:
        raise VerificationError("command_contract") from error


def _result_float(match: re.Match[str], group: str) -> float:
    try:
        value = float(match.group(group))
    except (ValueError, OverflowError) as error:
        raise VerificationError("command_contract") from error
    if not math.isfinite(value):
        raise VerificationError("command_contract")
    return value


def _result_json_string(token: str) -> str:
    try:
        value = json.loads(token)
    except (json.JSONDecodeError, TypeError) as error:
        raise VerificationError("command_contract") from error
    if type(value) is not str or not value:
        raise VerificationError("command_contract")
    return value


def _safe_identifier(value: object, code: str) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise VerificationError(code)
    normalized = re.sub(r"[._-]", "", value).lower()
    if any(fragment in normalized for fragment in _FORBIDDEN_IDENTIFIER_FRAGMENTS):
        raise VerificationError(code)
    return value


def _validate_stage_result(value: object, stage: str) -> dict[str, int]:
    if stage == "inspect_library":
        match = _result_match(_INSPECT_RESULT_RE, value)
        raw = _result_int(match, "raw")
        valid = _result_int(match, "valid")
        invalid = _result_int(match, "invalid")
        preview = _result_int(match, "preview")
        if (
            raw != 256
            or valid < 6
            or raw != valid + invalid
            or preview != min(valid, 24)
        ):
            raise VerificationError("command_contract")
        return {"valid": valid}
    if stage == "generate_morgan_fingerprints":
        match = _result_match(_MORGAN_RESULT_RE, value)
        radius = _result_int(match, "radius")
        bits = _result_int(match, "bits")
        rows = _result_int(match, "rows")
        columns = _result_int(match, "columns")
        minimum = _result_int(match, "minimum")
        median = _result_float(match, "median")
        maximum = _result_int(match, "maximum")
        if (
            radius != 2
            or bits != 1024
            or columns != (bits + 7) // 8
            or not 0 <= minimum <= median <= maximum <= bits
            or not (median * 2.0).is_integer()
            or (rows % 2 == 1 and not median.is_integer())
            or (rows == 1 and not minimum == median == maximum)
        ):
            raise VerificationError("command_contract")
        return {"rows": rows}
    if stage == "measure_tanimoto_similarity":
        match = _result_match(_SIMILARITY_RESULT_RE, value)
        first = _result_json_string(match.group("first"))
        second = _result_json_string(match.group("second"))
        _safe_identifier(first, "command_contract")
        _safe_identifier(second, "command_contract")
        similarity = _result_float(match, "similarity")
        q1 = _result_float(match, "q1")
        median = _result_float(match, "median")
        q3 = _result_float(match, "q3")
        p90 = _result_float(match, "p90")
        if (
            first == second
            or not 0.0 <= q1 <= median <= q3 <= p90 <= 1.0
            or not p90 <= similarity <= 1.0
        ):
            raise VerificationError("command_contract")
        return {}
    if stage == "discover_fused_butina_clusters":
        match = _result_match(_CLUSTER_RESULT_RE, value)
        cutoff = _result_float(match, "cutoff")
        clusters = _result_int(match, "clusters")
        singletons = _result_int(match, "singletons")
        sizes_text = match.group("sizes")
        sizes = (
            tuple(int(item) for item in sizes_text.split(", ")) if sizes_text else ()
        )
        listed_singletons = sum(size == 1 for size in sizes)
        if (
            cutoff != 0.40
            or not 1 <= clusters <= 256
            or not 0 <= singletons <= clusters
            or len(sizes) != min(clusters, 15)
            or any(not 1 <= size <= 256 for size in sizes)
            or any(first < second for first, second in zip(sizes, sizes[1:]))
        ):
            raise VerificationError("command_contract")
        if clusters <= 15:
            if listed_singletons != singletons or sum(sizes) > 256:
                raise VerificationError("command_contract")
            minimum_population = sum(sizes)
            maximum_population = minimum_population
        else:
            omitted_clusters = clusters - len(sizes)
            omitted_singletons = singletons - listed_singletons
            if not 0 <= omitted_singletons <= omitted_clusters:
                raise VerificationError("command_contract")
            omitted_non_singletons = omitted_clusters - omitted_singletons
            if sizes[-1] == 1 and omitted_non_singletons:
                raise VerificationError("command_contract")
            minimum_population = (
                sum(sizes) + omitted_singletons + 2 * omitted_non_singletons
            )
            maximum_population = (
                sum(sizes) + omitted_singletons + sizes[-1] * omitted_non_singletons
            )
            if minimum_population > 256:
                raise VerificationError("command_contract")
        return {
            "cluster_count": clusters,
            "population_min": minimum_population,
            "population_max": maximum_population,
        }
    if stage == "embed_representative_conformers":
        match = _result_match(_EMBED_RESULT_RE, value)
        selected = _result_int(match, "selected")
        requested_representatives = _result_int(match, "requested_representatives")
        generated = _result_int(match, "generated")
        requested_conformers = _result_int(match, "requested_conformers")
        partial = _result_int(match, "partial")
        zero = _result_int(match, "zero")
        if (
            requested_representatives != 6
            or selected != 6
            or generated < 1
            or requested_conformers != selected * 5
            or partial + zero > selected
        ):
            raise VerificationError("command_contract")
        full = selected - partial - zero
        if not 5 * full + partial <= generated <= 5 * full + 4 * partial:
            raise VerificationError("command_contract")
        return {
            "selected": selected,
            "full": full,
            "partial": partial,
            "zero": zero,
            "generated": generated,
        }
    if stage == "optimize_conformers_mmff94":
        match = _result_match(_OPTIMIZE_RESULT_RE, value)
        attempted = _result_int(match, "attempted")
        converged = _result_int(match, "converged")
        unconverged = _result_int(match, "unconverged")
        minima = match.group("minima")
        minimum_count = 0
        seen_identifiers: set[str] = set()
        if minima != "none":
            entries = tuple(_MINIMUM_ENTRY_RE.finditer(minima))
            if not entries or ", ".join(entry.group(0) for entry in entries) != minima:
                raise VerificationError("command_contract")
            for entry in entries:
                identifier = _result_json_string(entry.group("identifier"))
                _safe_identifier(identifier, "command_contract")
                if identifier in seen_identifiers:
                    raise VerificationError("command_contract")
                seen_identifiers.add(identifier)
                _result_float(entry, "energy")
            minimum_count = len(entries)
        if attempted != converged + unconverged or minimum_count > converged:
            raise VerificationError("command_contract")
        return {
            "attempted": attempted,
            "converged": converged,
            "minima": minimum_count,
        }
    raise VerificationError("command_contract")


def _validate_completed_stage(
    value: object, expected_stage: str
) -> tuple[dict[str, int], _GpuIdentity | None]:
    if (
        type(value) is not dict
        or set(value) != _STAGE_ITEM_KEYS
        or value.get("stage") != expected_stage
        or not _nonempty_string(value, "result")
        or not _nonempty_string(value, "summary_path")
        or not _nonempty_string(value, "readme_path")
        or not _nonempty_string(value, "artifact_directory")
    ):
        raise VerificationError("command_contract")
    facts = _validate_stage_result(value.get("result"), expected_stage)
    image_paths = value.get("image_paths")
    if (
        type(image_paths) is not list
        or len(image_paths) != _STAGE_IMAGE_COUNTS[expected_stage]
        or any(type(path) is not str or not path for path in image_paths)
    ):
        raise VerificationError("command_contract")
    gpu_identity = _validate_stage_execution(value.get("execution"), expected_stage)
    return facts, gpu_identity


def _validate_lesson_result(
    result: Mapping[str, Any], lesson: str, library_valid_count: int | None = None
) -> tuple[int, _GpuIdentity]:
    completed_stages = result.get("completed_stages")
    if (
        set(result) != _LESSON_RESULT_KEYS
        or not _schema_version_one(result)
        or result.get("status") != "complete"
        or result.get("lesson") != lesson
        or type(completed_stages) is not list
        or not _nonempty_string(result, "results_zip_path")
        or result.get("artifact_relative_zip_path") != "workshop/results.zip"
        or type(result.get("answer_markdown")) is not str
    ):
        raise VerificationError("command_contract")
    expected_stages = _LESSON_STAGES[lesson]
    if len(completed_stages) != len(expected_stages):
        raise VerificationError("command_contract")
    facts: dict[str, dict[str, int]] = {}
    gpu_identities: list[_GpuIdentity] = []
    for item, expected_stage in zip(completed_stages, expected_stages, strict=True):
        stage_facts, gpu_identity = _validate_completed_stage(item, expected_stage)
        facts[expected_stage] = stage_facts
        if gpu_identity is not None:
            gpu_identities.append(gpu_identity)
    if not gpu_identities or any(
        identity != gpu_identities[0] for identity in gpu_identities[1:]
    ):
        raise VerificationError("command_contract")
    lesson_gpu_identity = gpu_identities[0]
    if lesson == "data-and-representation":
        valid_count = facts["inspect_library"]["valid"]
        if facts["generate_morgan_fingerprints"].get("rows") != valid_count:
            raise VerificationError("command_contract")
        return valid_count, lesson_gpu_identity
    if library_valid_count is None:
        raise VerificationError("command_contract")
    if lesson == "relationships-and-groups":
        clusters = facts["discover_fused_butina_clusters"]
        if clusters["cluster_count"] < 6 or not (
            clusters["population_min"]
            <= library_valid_count
            <= clusters["population_max"]
        ):
            raise VerificationError("command_contract")
    if lesson == "sampled-3d-geometry":
        embed = facts["embed_representative_conformers"]
        optimize = facts["optimize_conformers_mmff94"]
        selected = embed["selected"]
        full = embed["full"]
        partial = embed["partial"]
        zero = embed["zero"]
        generated = embed["generated"]
        attempted = optimize["attempted"]
        converged = optimize["converged"]
        minima = optimize["minima"]
        if attempted != generated:
            raise VerificationError("command_contract")
        if converged == 0:
            if minima != 0:
                raise VerificationError("command_contract")
        else:
            partial_total = generated - 5 * full
            active = selected - zero
            minimum_active = active + 1
            for active_count in range(1, active + 1):
                used_full = min(active_count, full)
                used_partial = active_count - used_full
                maximum_capacity = 5 * used_full
                if used_partial:
                    maximum_capacity += min(
                        4 * used_partial,
                        partial_total - (partial - used_partial),
                    )
                if maximum_capacity >= converged:
                    minimum_active = active_count
                    break
            if not minimum_active <= minima <= min(converged, active):
                raise VerificationError("command_contract")
    return library_valid_count, lesson_gpu_identity


def _lesson_execution_sentence(item: Mapping[str, Any]) -> str:
    execution = item.get("execution")
    if type(execution) is not dict:
        raise VerificationError("answer_contract")
    placement = execution.get("placement")
    software = execution.get("software")
    operation = execution.get("operation")
    gpu = execution.get("gpu")
    upstream = execution.get("upstream")
    if (
        type(placement) is not str
        or type(software) is not str
        or type(operation) is not str
    ):
        raise VerificationError("answer_contract")
    if placement == "GPU":
        if type(gpu) is not dict:
            raise VerificationError("answer_contract")
        name = gpu.get("name")
        device = gpu.get("device")
        if type(name) is not str or type(device) is not str:
            raise VerificationError("answer_contract")
        return f"{software} ran {operation} on GPU {name} ({device})."
    if upstream is None:
        return f"{software} ran {operation} on CPU."
    if type(upstream) is not dict or type(gpu) is not dict:
        raise VerificationError("answer_contract")
    upstream_software = upstream.get("software")
    upstream_operation = upstream.get("operation")
    name = gpu.get("name")
    device = gpu.get("device")
    if any(
        type(value) is not str
        for value in (upstream_software, upstream_operation, name, device)
    ):
        raise VerificationError("answer_contract")
    return (
        f"{software} ran {operation} on CPU using {upstream_software} "
        f"{upstream_operation} results computed on GPU {name} ({device})."
    )


def _canonical_lesson_answer(
    result: Mapping[str, Any], lesson: str, media_line: str
) -> str:
    completed_stages = result.get("completed_stages")
    if type(completed_stages) is not list or not 1 <= len(completed_stages) <= 3:
        raise VerificationError("answer_contract")
    stages: list[dict[str, Any]] = []
    measured: list[str] = []
    for item in completed_stages:
        if type(item) is not dict or type(item.get("result")) is not str:
            raise VerificationError("answer_contract")
        stages.append(item)
        measured.append(item["result"])
    what_ran = " ".join(_lesson_execution_sentence(item) for item in stages)
    if lesson == "relationships-and-groups":
        execution = stages[0].get("execution")
        if type(execution) is not dict or type(execution.get("gpu")) is not dict:
            raise VerificationError("answer_contract")
        gpu = execution["gpu"]
        name = gpu.get("name")
        device = gpu.get("device")
        if type(name) is not str or type(device) is not str:
            raise VerificationError("answer_contract")
        what_ran = (
            "nvMolKit generated Morgan fingerprints and computed Tanimoto "
            f"similarities on GPU {name} ({device}). RDKit ran Butina clustering "
            "on CPU using those GPU-computed similarities."
        )
    elif lesson == "sampled-3d-geometry":
        what_ran = f"One command returned both stages. {what_ran}"
    return (
        f"## Question\n{_LESSON_QUESTIONS[lesson]}\n\n"
        f"## What ran\n{what_ran}\n\n"
        "## Measured result\n"
        + "\n".join(f"- {item}" for item in measured)
        + f"\n\n## Meaning\n{_LESSON_MEANINGS[lesson]}\n\n"
        f"## Scientific limit\n{_LESSON_SCIENTIFIC_LIMITS[lesson]}\n\n"
        f"## Image and download location\n{_DOWNLOAD_SENTENCE}\n\n"
        f"{media_line}"
    )


def _validated_ids(value: object) -> tuple[str, ...]:
    if type(value) is not list or len(value) != 4:
        raise VerificationError("objective_contract")
    identifiers = tuple(_safe_identifier(item, "objective_contract") for item in value)
    if len(set(identifiers)) != 4:
        raise VerificationError("objective_contract")
    return identifiers


def _validated_pairs(
    value: object, selected_ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    if type(value) is not list or not value:
        raise VerificationError("objective_contract")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or any(type(identifier) is not str or not identifier for identifier in item)
            or item[0] == item[1]
            or any(identifier not in selected_ids for identifier in item)
        ):
            raise VerificationError("objective_contract")
        pairs.append((item[0], item[1]))
    if len(pairs) != len(set(pairs)):
        raise VerificationError("objective_contract")
    return tuple(pairs)


def _validate_measurement(value: object, target_score: float) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _MEASUREMENT_KEYS:
        raise VerificationError("objective_contract")
    selected_ids = _validated_ids(value.get("selected_ids"))
    score = _objective_score(value.get("score"), "objective_contract")
    _validated_pairs(value.get("limiting_pairs"), selected_ids)
    if type(value.get("achieved")) is not bool or value.get(
        "achieved"
    ) != _target_achieved(score, target_score):
        raise VerificationError("objective_contract")
    return value


def _validate_action(
    value: object,
    *,
    current: Mapping[str, Any] | None,
    target_score: float,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _ACTION_KEYS:
        raise VerificationError("objective_contract")
    replace_id = _safe_identifier(value.get("replace_id"), "objective_contract")
    replacement_id = _safe_identifier(value.get("replacement_id"), "objective_contract")
    swap_id = value.get("swap_id")
    if replace_id == replacement_id or swap_id != f"{replace_id}->{replacement_id}":
        raise VerificationError("objective_contract")
    resulting_ids = _validated_ids(value.get("resulting_ids"))
    predicted_score = _objective_score(
        value.get("predicted_score"), "objective_contract"
    )
    score_delta = _finite_float(value.get("score_delta"), "objective_contract")
    _validated_pairs(value.get("limiting_pairs"), resulting_ids)
    target_status = value.get("target_status")
    if (
        type(target_status) is not str
        or target_status not in _TARGET_STATUSES
        or target_status
        != (
            "meets_target"
            if _target_achieved(predicted_score, target_score)
            else "below_target"
        )
        or score_delta <= 0.0
    ):
        raise VerificationError("objective_contract")
    if current is not None:
        current_ids = _validated_ids(current.get("selected_ids"))
        current_score = _objective_score(current.get("score"), "objective_contract")
        if (
            replace_id not in current_ids
            or replacement_id in current_ids
            or resulting_ids
            != tuple(
                replacement_id if identifier == replace_id else identifier
                for identifier in current_ids
            )
            or not math.isclose(
                score_delta,
                predicted_score - current_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise VerificationError("objective_contract")
    return value


def _measurement_matches_action(
    measurement: Mapping[str, Any], action: Mapping[str, Any]
) -> bool:
    return (
        measurement.get("selected_ids") == action.get("resulting_ids")
        and measurement.get("score") == action.get("predicted_score")
        and measurement.get("limiting_pairs") == action.get("limiting_pairs")
        and measurement.get("achieved")
        == (action.get("target_status") == "meets_target")
    )


def _record_measurement_candidates(
    universe: set[str], measurement: Mapping[str, Any]
) -> None:
    universe.update(_validated_ids(measurement.get("selected_ids")))
    if len(universe) > 8:
        raise VerificationError("objective_contract")


def _record_action_candidates(universe: set[str], action: Mapping[str, Any]) -> None:
    universe.add(_safe_identifier(action.get("replace_id"), "objective_contract"))
    universe.add(_safe_identifier(action.get("replacement_id"), "objective_contract"))
    universe.update(_validated_ids(action.get("resulting_ids")))
    if len(universe) > 8:
        raise VerificationError("objective_contract")


def _validate_pending_result(
    result: Mapping[str, Any], expected_attempt_count: int
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if (
        set(result) != _PENDING_RESULT_KEYS
        or not _schema_version_one(result)
        or result.get("status") != "pending"
        or result.get("terminal") is not False
        or type(result.get("attempt_count")) is not int
        or result.get("attempt_count") != expected_attempt_count
        or result.get("attempt_limit") != 3
        or not 0 <= expected_attempt_count < 3
        or type(result.get("actions")) is not list
        or result.get("achieved") is not None
        or result.get("termination_reason") is not None
        or result.get("image_paths") != []
        or not _valid_artifact_locations(result)
    ):
        raise VerificationError("objective_contract")
    _safe_identifier(result.get("state_id"), "objective_contract")
    target_score = _objective_score(result.get("target_score"), "objective_contract")
    current = _validate_measurement(result.get("current"), target_score)
    if current.get("achieved") is not False:
        raise VerificationError("objective_contract")
    raw_actions = result.get("actions")
    if type(raw_actions) is not list or not 1 <= len(raw_actions) <= 3:
        raise VerificationError("objective_contract")
    actions = tuple(
        _validate_action(
            action,
            current=current,
            target_score=target_score,
        )
        for action in raw_actions
    )
    return current, actions


def _validate_terminal_result(
    result: Mapping[str, Any], expected_attempt_count: int
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    image_paths = result.get("image_paths")
    attempts = result.get("attempts")
    if (
        set(result) != _TERMINAL_RESULT_KEYS
        or not _schema_version_one(result)
        or result.get("status") != "complete"
        or result.get("terminal") is not True
        or type(result.get("attempt_count")) is not int
        or result.get("attempt_count") != expected_attempt_count
        or result.get("attempt_limit") != 3
        or not 0 <= expected_attempt_count <= 3
        or type(attempts) is not list
        or len(attempts) != expected_attempt_count
        or type(result.get("achieved")) is not bool
        or type(result.get("termination_reason")) is not str
        or result.get("termination_reason") not in _TERMINATION_REASONS
        or type(image_paths) is not list
        or len(image_paths) != 3
        or any(type(path) is not str or not path for path in image_paths)
        or not _valid_artifact_locations(result)
        or type(result.get("answer_markdown")) is not str
    ):
        raise VerificationError("objective_contract")
    target_score = _objective_score(result.get("target_score"), "objective_contract")
    baseline = _validate_measurement(result.get("baseline"), target_score)
    final = _validate_measurement(result.get("final"), target_score)
    validated_attempts: list[dict[str, Any]] = []
    current = baseline
    seen_states: set[str] = set()
    for position, value in enumerate(attempts, start=1):
        if type(value) is not dict or set(value) != _ATTEMPT_KEYS:
            raise VerificationError("objective_contract")
        state_id = _safe_identifier(value.get("state_id"), "objective_contract")
        if (
            value.get("attempt_number") != position
            or type(value.get("attempt_number")) is not int
            or state_id in seen_states
        ):
            raise VerificationError("objective_contract")
        seen_states.add(state_id)
        measurement = _validate_measurement(
            {
                "selected_ids": value.get("selected_ids"),
                "score": value.get("score"),
                "limiting_pairs": value.get("limiting_pairs"),
                "achieved": value.get("achieved"),
            },
            target_score,
        )
        action = _validate_action(
            value.get("selected_swap"),
            current=current,
            target_score=target_score,
        )
        if not _measurement_matches_action(measurement, action):
            raise VerificationError("objective_contract")
        current = measurement
        validated_attempts.append(value)
    if final != current or result.get("achieved") != final.get("achieved"):
        raise VerificationError("objective_contract")
    reason = result.get("termination_reason")
    if (
        (
            reason == "baseline_already_optimal"
            and (expected_attempt_count != 0 or result.get("achieved") is not True)
        )
        or (
            reason == "target_achieved"
            and (expected_attempt_count == 0 or result.get("achieved") is not True)
        )
        or (
            reason == "attempt_limit_reached"
            and (expected_attempt_count != 3 or result.get("achieved") is not False)
        )
        or (
            reason
            in {
                "no_legal_improving_swap",
                "objective_correction_limit",
                "objective_provider_failure",
                "evaluation_not_completed",
            }
            and (result.get("achieved") is not False or expected_attempt_count >= 3)
        )
    ):
        raise VerificationError("objective_contract")
    return baseline, final, tuple(validated_attempts)


def _quoted_ids(value: object) -> str:
    identifiers = _validated_ids(value)
    return ", ".join(json.dumps(identifier) for identifier in identifiers)


def _canonical_objective_answer(result: Mapping[str, Any], media_line: str) -> str:
    baseline = result.get("baseline")
    final = result.get("final")
    attempts = result.get("attempts")
    if (
        type(baseline) is not dict
        or type(final) is not dict
        or type(attempts) is not list
    ):
        raise VerificationError("answer_contract")
    baseline_score = _finite_float(baseline.get("score"), "answer_contract")
    final_score = _finite_float(final.get("score"), "answer_contract")
    limiting_pairs = _validated_pairs(
        baseline.get("limiting_pairs"), _validated_ids(baseline.get("selected_ids"))
    )
    swap_ids: list[str] = []
    for attempt in attempts:
        if type(attempt) is not dict:
            raise VerificationError("answer_contract")
        selected_swap = attempt.get("selected_swap")
        if type(selected_swap) is not dict or not _nonempty_string(
            selected_swap, "swap_id"
        ):
            raise VerificationError("answer_contract")
        swap_ids.append(selected_swap["swap_id"])
    swaps = ", ".join(json.dumps(swap_id) for swap_id in swap_ids) or "none"
    meaning = (
        _OBJECTIVE_MEANING_INCREASED
        if final_score > baseline_score
        else _OBJECTIVE_MEANING_UNCHANGED
    )
    what_ran = (
        f"Baseline panel: {_quoted_ids(baseline.get('selected_ids'))}. "
        "Baseline limiting pair: "
        f"{', '.join(json.dumps(identifier) for identifier in limiting_pairs[0])}. "
        f"Accepted swaps: {swaps}. Final panel: "
        f"{_quoted_ids(final.get('selected_ids'))}. Python validated each "
        "displayed maximum-score action against the fixed Tanimoto distance "
        "matrix derived from nvMolKit GPU-computed Morgan fingerprints and "
        "similarities."
    )
    measured = "\n".join(
        (
            f"- Baseline `D_min`: {baseline_score:.3f}.",
            f"- Final `D_min`: {final_score:.3f}.",
            f"- Change in `D_min`: {final_score - baseline_score:+.3f}.",
        )
    )
    return (
        "## Question\nCan a bounded agent improve the weakest-link diversity of a "
        "four-molecule panel?\n\n"
        f"## What ran\n{what_ran}\n\n"
        f"## Measured result\n{measured}\n\n"
        f"## Meaning\n{meaning}\n\n"
        f"## Scientific limit\n{_OBJECTIVE_SCIENTIFIC_LIMIT}\n\n"
        f"## Image and download location\n{_DOWNLOAD_SENTENCE}\n\n"
        f"{media_line}"
    )


def _validate_answer(
    answer: str,
    result: Mapping[str, Any],
    media_line: str,
    *,
    objective: bool,
    expected: str | None = None,
) -> None:
    canonical = result.get("answer_markdown")
    if type(canonical) is not str or answer != canonical:
        raise VerificationError("answer_contract")
    observed_headings = tuple(re.findall(r"(?m)^## [^\r\n]+$", answer))
    if observed_headings != HEADINGS:
        raise VerificationError("answer_contract")
    answer_lines = answer.splitlines()
    if (
        answer.count(media_line) != 1
        or not answer_lines
        or answer_lines[-1] != media_line
        or not answer.endswith(media_line)
    ):
        raise VerificationError("answer_contract")
    if (
        answer.count("**Download Results**") != 1
        or answer.count("`workshop/results.zip`") != 1
    ):
        raise VerificationError("answer_contract")
    download_tail = (
        f"## Image and download location\n{_DOWNLOAD_SENTENCE}\n\n{media_line}"
    )
    if not answer.endswith(download_tail):
        raise VerificationError("answer_contract")
    independent = (
        _canonical_objective_answer(result, media_line) if objective else expected
    )
    if independent is None or answer != independent:
        raise VerificationError("answer_contract")


def validate_trajectory(
    messages: Sequence[Mapping[str, Any]],
    contracts: Sequence[tuple[str, str, str]],
) -> tuple[int, int]:
    if len(contracts) != 4 or not messages:
        raise VerificationError("prompt_contract")
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if len(user_indices) != 4 or user_indices[0] != 0:
        raise VerificationError("prompt_contract")
    turns: list[Sequence[Mapping[str, Any]]] = []
    for turn_index, (message_index, contract) in enumerate(
        zip(user_indices, contracts, strict=True)
    ):
        if (
            type(contract) is not tuple
            or len(contract) != 3
            or contract[0] != PROMPT_IDS[turn_index]
            or contract[2] != PROMPT_SHA256[turn_index]
        ):
            raise VerificationError("prompt_contract")
        prompt = _validate_user_message(messages[message_index])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (
            prompt != contract[1]
            or digest != contract[2]
            or digest != PROMPT_SHA256[turn_index]
        ):
            raise VerificationError("prompt_contract")
        end = user_indices[turn_index + 1] if turn_index + 1 < 4 else len(messages)
        turns.append(messages[message_index + 1 : end])

    exec_count = 0
    seen_call_ids: set[str] = set()
    lesson_names = (
        "data-and-representation",
        "relationships-and-groups",
        "sampled-3d-geometry",
    )
    library_valid_count: int | None = None
    trajectory_gpu_identity: _GpuIdentity | None = None
    for turn_index in range(3):
        calls, answer = _turn_evidence(turns[turn_index])
        if (
            len(calls) != 1
            or calls[0].command != LESSON_COMMANDS[turn_index]
            or calls[0].call_id in seen_call_ids
        ):
            raise VerificationError("command_contract")
        seen_call_ids.add(calls[0].call_id)
        result = calls[0].result
        library_valid_count, lesson_gpu_identity = _validate_lesson_result(
            result, lesson_names[turn_index], library_valid_count
        )
        if trajectory_gpu_identity is None:
            trajectory_gpu_identity = lesson_gpu_identity
        elif lesson_gpu_identity != trajectory_gpu_identity:
            raise VerificationError("command_contract")
        _validate_answer(
            answer,
            result,
            PROMPT_MEDIA[turn_index],
            objective=False,
            expected=_canonical_lesson_answer(
                result, lesson_names[turn_index], PROMPT_MEDIA[turn_index]
            ),
        )
        exec_count += 1

    calls, answer = _turn_evidence(turns[3])
    if not 1 <= len(calls) <= 4 or calls[0].command != OBJECTIVE_START:
        raise VerificationError("objective_contract")
    for call in calls:
        if call.call_id in seen_call_ids:
            raise VerificationError("command_contract")
        seen_call_ids.add(call.call_id)
    current = calls[0].result
    exec_count += 1
    seen_states: set[str] = set()
    candidate_universe: set[str] = set()
    selected_steps: list[tuple[str, dict[str, Any]]] = []
    initial_measurement: dict[str, Any] | None = None
    objective_target_score: float | None = None
    for call in calls[1:]:
        current_measurement, actions = _validate_pending_result(
            current, len(selected_steps)
        )
        _record_measurement_candidates(candidate_universe, current_measurement)
        for action in actions:
            _record_action_candidates(candidate_universe, action)
        if initial_measurement is None:
            initial_measurement = current_measurement
        pending_target_score = _finite_float(
            current.get("target_score"), "objective_contract"
        )
        if objective_target_score is None:
            objective_target_score = pending_target_score
        elif pending_target_score != objective_target_score:
            raise VerificationError("objective_contract")
        if selected_steps and not _measurement_matches_action(
            current_measurement, selected_steps[-1][1]
        ):
            raise VerificationError("objective_contract")
        state_id = _safe_identifier(current.get("state_id"), "objective_contract")
        match = OBJECTIVE_STEP_RE.fullmatch(call.command)
        if state_id in seen_states or match is None:
            raise VerificationError("objective_contract")
        seen_states.add(state_id)
        command_state, swap_id = match.groups()
        _safe_identifier(command_state, "objective_contract")
        if command_state != state_id or not 1 <= len(actions) <= 3:
            raise VerificationError("objective_contract")
        scored_actions: list[tuple[dict[str, Any], float]] = []
        observed_swaps: set[str] = set()
        for action in actions:
            if (
                type(action.get("swap_id")) is not str
                or not action["swap_id"]
                or action["swap_id"] in observed_swaps
            ):
                raise VerificationError("objective_contract")
            observed_swaps.add(action["swap_id"])
            scored_actions.append(
                (
                    action,
                    _finite_float(action.get("predicted_score"), "objective_contract"),
                )
            )
        selected = [
            (action, score)
            for action, score in scored_actions
            if action.get("swap_id") == swap_id
        ]
        if len(selected) != 1 or selected[0][1] != max(
            score for _, score in scored_actions
        ):
            raise VerificationError("objective_contract")
        selected_steps.append((state_id, selected[0][0]))
        current = call.result
        exec_count += 1
    baseline, final, attempts = _validate_terminal_result(current, len(selected_steps))
    _record_measurement_candidates(candidate_universe, baseline)
    _record_measurement_candidates(candidate_universe, final)
    for attempt in attempts:
        _record_measurement_candidates(candidate_universe, attempt)
        selected_swap = attempt.get("selected_swap")
        if type(selected_swap) is not dict:
            raise VerificationError("objective_contract")
        _record_action_candidates(candidate_universe, selected_swap)
    if selected_steps:
        if (
            baseline != initial_measurement
            or current.get("target_score") != objective_target_score
            or not _measurement_matches_action(final, selected_steps[-1][1])
        ):
            raise VerificationError("objective_contract")
    for attempt, (state_id, action) in zip(attempts, selected_steps, strict=True):
        if (
            attempt.get("state_id") != state_id
            or attempt.get("selected_swap") != action
        ):
            raise VerificationError("objective_contract")
    _validate_answer(answer, current, PROMPT_MEDIA[3], objective=True)
    return exec_count, len(calls) - 1


def _validate_png(contents: bytes) -> None:
    if len(contents) < 57 or contents[:8] != b"\x89PNG\r\n\x1a\n":
        raise VerificationError("archive_contract")
    offset = 8
    chunk_index = 0
    width = 0
    height = 0
    channels = 0
    seen_ihdr = False
    seen_plte = False
    idat_started = False
    idat_ended = False
    seen_iend = False
    idat_parts: list[bytes] = []
    while offset < len(contents):
        if seen_iend or offset + 12 > len(contents):
            raise VerificationError("archive_contract")
        length = struct.unpack_from(">I", contents, offset)[0]
        kind = contents[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        record_end = data_end + 4
        if (
            length > MAX_MEMBER_BYTES
            or record_end > len(contents)
            or len(kind) != 4
            or any(
                byte not in range(ord("A"), ord("Z") + 1)
                and byte not in range(ord("a"), ord("z") + 1)
                for byte in kind
            )
            or kind[2] & 0x20
        ):
            raise VerificationError("archive_contract")
        payload = contents[data_start:data_end]
        expected_crc = struct.unpack_from(">I", contents, data_end)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise VerificationError("archive_contract")
        if chunk_index == 0 and kind != b"IHDR":
            raise VerificationError("archive_contract")
        if kind == b"IHDR":
            if seen_ihdr or chunk_index != 0 or length != 13:
                raise VerificationError("archive_contract")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression_method,
                filter_method,
                interlace_method,
            ) = struct.unpack(">IIBBBBB", payload)
            if (
                not 0 < width <= 0x7FFFFFFF
                or not 0 < height <= 0x7FFFFFFF
                or bit_depth != 8
                or color_type not in {2, 6}
                or compression_method != 0
                or filter_method != 0
                or interlace_method != 0
            ):
                raise VerificationError("archive_contract")
            channels = 3 if color_type == 2 else 4
            seen_ihdr = True
        elif kind == b"PLTE":
            if (
                not seen_ihdr
                or seen_plte
                or idat_started
                or not 0 < length <= 768
                or length % 3
            ):
                raise VerificationError("archive_contract")
            seen_plte = True
        elif kind == b"IDAT":
            if not seen_ihdr or idat_ended:
                raise VerificationError("archive_contract")
            idat_started = True
            idat_parts.append(payload)
        elif kind == b"IEND":
            if not seen_ihdr or not idat_started or length != 0:
                raise VerificationError("archive_contract")
            seen_iend = True
            if record_end != len(contents):
                raise VerificationError("archive_contract")
        else:
            if not seen_ihdr or kind[0] & 0x20 == 0:
                raise VerificationError("archive_contract")
            if idat_started:
                idat_ended = True
        offset = record_end
        chunk_index += 1
    if not seen_ihdr or not idat_started or not seen_iend or offset != len(contents):
        raise VerificationError("archive_contract")
    row_size = 1 + width * channels
    expected_size = height * row_size
    if expected_size > MAX_EXPANDED_BYTES:
        raise VerificationError("archive_contract")
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(b"".join(idat_parts), expected_size + 1)
        if decompressor.unconsumed_tail:
            raise VerificationError("archive_contract")
        decoded += decompressor.flush()
    except zlib.error as error:
        raise VerificationError("archive_contract") from error
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(decoded) != expected_size
        or any(decoded[row * row_size] > 4 for row in range(height))
    ):
        raise VerificationError("archive_contract")


def _validate_local_header(raw: bytes, info: zipfile.ZipInfo) -> tuple[int, int]:
    offset = info.header_offset
    if (
        offset < 0
        or offset + 30 > len(raw)
        or raw[offset : offset + 4] != b"PK\x03\x04"
    ):
        raise VerificationError("archive_contract")
    version_needed = struct.unpack_from("<H", raw, offset + 4)[0]
    flags = struct.unpack_from("<H", raw, offset + 6)[0]
    compression = struct.unpack_from("<H", raw, offset + 8)[0]
    dos_time = struct.unpack_from("<H", raw, offset + 10)[0]
    dos_date = struct.unpack_from("<H", raw, offset + 12)[0]
    crc = struct.unpack_from("<I", raw, offset + 14)[0]
    compressed_size = struct.unpack_from("<I", raw, offset + 18)[0]
    file_size = struct.unpack_from("<I", raw, offset + 22)[0]
    name_length = struct.unpack_from("<H", raw, offset + 26)[0]
    extra_length = struct.unpack_from("<H", raw, offset + 28)[0]
    name_start = offset + 30
    name_end = name_start + name_length
    if (
        name_end + extra_length + compressed_size > len(raw)
        or version_needed != 20
        or version_needed != info.extract_version
        or flags != 0
        or flags != info.flag_bits
        or compression != zipfile.ZIP_DEFLATED
        or compression != info.compress_type
        or dos_time != 0
        or dos_date != 33
        or crc != info.CRC
        or compressed_size != info.compress_size
        or file_size != info.file_size
        or raw[name_start:name_end] != info.filename.encode("ascii")
        or extra_length != 0
    ):
        raise VerificationError("archive_contract")
    data_start = name_end + extra_length
    return int(data_start), int(data_start + compressed_size)


def _validate_raw_member(
    raw: bytes,
    info: zipfile.ZipInfo,
    data_start: int,
    data_end: int,
    output_limit: int,
) -> bytes:
    compressed = raw[data_start:data_end]
    if info.compress_type == zipfile.ZIP_STORED:
        contents = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        try:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            contents = decompressor.decompress(compressed, output_limit + 1)
        except zlib.error as error:
            raise VerificationError("archive_contract") from error
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise VerificationError("archive_contract")
    else:
        raise VerificationError("archive_contract")
    if (
        len(contents) > output_limit
        or len(contents) != info.file_size
        or zlib.crc32(contents) & 0xFFFFFFFF != info.CRC
    ):
        raise VerificationError("archive_contract")
    return contents


def _validate_central_header(raw: bytes, info: zipfile.ZipInfo, offset: int) -> int:
    if (
        offset < 0
        or offset + 46 > len(raw)
        or raw[offset : offset + 4] != b"PK\x01\x02"
    ):
        raise VerificationError("archive_contract")
    version_made_by = struct.unpack_from("<H", raw, offset + 4)[0]
    version_needed = struct.unpack_from("<H", raw, offset + 6)[0]
    flags = struct.unpack_from("<H", raw, offset + 8)[0]
    compression = struct.unpack_from("<H", raw, offset + 10)[0]
    dos_time = struct.unpack_from("<H", raw, offset + 12)[0]
    dos_date = struct.unpack_from("<H", raw, offset + 14)[0]
    crc = struct.unpack_from("<I", raw, offset + 16)[0]
    compressed_size = struct.unpack_from("<I", raw, offset + 20)[0]
    file_size = struct.unpack_from("<I", raw, offset + 24)[0]
    name_length = struct.unpack_from("<H", raw, offset + 28)[0]
    extra_length = struct.unpack_from("<H", raw, offset + 30)[0]
    comment_length = struct.unpack_from("<H", raw, offset + 32)[0]
    disk_number = struct.unpack_from("<H", raw, offset + 34)[0]
    internal_attr = struct.unpack_from("<H", raw, offset + 36)[0]
    external_attr = struct.unpack_from("<I", raw, offset + 38)[0]
    local_offset = struct.unpack_from("<I", raw, offset + 42)[0]
    name_start = offset + 46
    name_end = name_start + name_length
    record_end = name_end + extra_length + comment_length
    if (
        record_end > len(raw)
        or version_made_by != ((3 << 8) | 20)
        or version_made_by & 0xFF != info.create_version
        or version_made_by >> 8 != info.create_system
        or version_needed != 20
        or version_needed != info.extract_version
        or flags != 0
        or flags != info.flag_bits
        or compression != zipfile.ZIP_DEFLATED
        or compression != info.compress_type
        or dos_time != 0
        or dos_date != 33
        or crc != info.CRC
        or compressed_size != info.compress_size
        or file_size != info.file_size
        or raw[name_start:name_end] != info.filename.encode("ascii")
        or extra_length != 0
        or comment_length != 0
        or disk_number != 0
        or internal_attr != info.internal_attr
        or external_attr != info.external_attr
        or local_offset != info.header_offset
    ):
        raise VerificationError("archive_contract")
    return int(record_end)


def validate_results_zip(path: Path) -> tuple[str, int, int]:
    raw = _read_regular(path, _MAX_ARCHIVE_BYTES, "archive_contract")
    eocd = raw.rfind(b"PK\x05\x06")
    if eocd >= 0 and eocd + 22 <= len(raw):
        (
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack_from("<HHHHIIH", raw, eocd + 4)
    else:
        disk_number = central_disk = disk_entries = total_entries = -1
        central_size = central_offset = comment_length = -1
    if (
        eocd < 0
        or eocd + 22 != len(raw)
        or disk_number != 0
        or central_disk != 0
        or disk_entries != len(REQUIRED_ZIP_MEMBERS)
        or total_entries != len(REQUIRED_ZIP_MEMBERS)
        or comment_length != 0
        or central_offset < 0
        or central_size < 0
        or central_offset + central_size != eocd
    ):
        raise VerificationError("archive_contract")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                archive.comment
                or len(names) != len(set(names))
                or tuple(names) != REQUIRED_ZIP_MEMBERS
            ):
                raise VerificationError("archive_contract")
            expanded = 0
            local_cursor = 0
            for info in infos:
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or info.flag_bits != 0
                    or info.extra
                    or info.comment
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or mode != stat.S_IFREG | 0o644
                    or info.filename.startswith("/")
                    or "\\" in info.filename
                    or any(part in {"", ".", ".."} for part in info.filename.split("/"))
                    or info.file_size > MAX_MEMBER_BYTES
                ):
                    raise VerificationError("archive_contract")
                if info.header_offset != local_cursor:
                    raise VerificationError("archive_contract")
                data_start, local_cursor = _validate_local_header(raw, info)
                contents = _validate_raw_member(
                    raw,
                    info,
                    data_start,
                    local_cursor,
                    min(MAX_MEMBER_BYTES, MAX_EXPANDED_BYTES - expanded),
                )
                expanded += len(contents)
                if info.filename in REQUIRED_CHAT_PNGS:
                    _validate_png(contents)
            if local_cursor != central_offset:
                raise VerificationError("archive_contract")
            central_cursor = central_offset
            for info in infos:
                central_cursor = _validate_central_header(raw, info, central_cursor)
            if central_cursor != eocd:
                raise VerificationError("archive_contract")
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError("archive_contract") from error
    return hashlib.sha256(raw).hexdigest(), len(raw), len(REQUIRED_CHAT_PNGS)


def verify_acceptance(
    trajectory_path: Path,
    results_zip_path: Path,
    page_path: Path,
) -> dict[str, int | str]:
    contracts = load_prompt_contracts(page_path)
    messages = load_messages_snapshot(trajectory_path)
    exec_count, objective_step_count = validate_trajectory(messages, contracts)
    archive_sha256, archive_size, png_count = validate_results_zip(results_zip_path)
    return {
        "schema_version": 1,
        "status": "pass",
        "prompt_count": 4,
        "exec_call_count": exec_count,
        "objective_step_count": objective_step_count,
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "required_png_count": png_count,
    }


class _ClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        raise VerificationError("invalid_evidence")


def build_parser() -> argparse.ArgumentParser:
    parser = _ClosedArgumentParser(add_help=False)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--results-zip", required=True, type=Path)
    parser.add_argument(
        "--page",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "acs-fall-2026-workshop.md",
    )
    return parser


def _emit(receipt: Mapping[str, int | str]) -> None:
    sys.stdout.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        receipt = verify_acceptance(
            arguments.trajectory,
            arguments.results_zip,
            arguments.page,
        )
    except VerificationError as error:
        _emit(
            {
                "schema_version": 1,
                "status": "fail",
                "issue_code": error.code,
            }
        )
        return 2
    except Exception:
        _emit(
            {
                "schema_version": 1,
                "status": "fail",
                "issue_code": "invalid_evidence",
            }
        )
        return 2
    _emit(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
