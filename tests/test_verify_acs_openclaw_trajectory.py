from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import struct
import subprocess
import sys
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_acs_openclaw_trajectory.py"
FIXTURE = ROOT / "tests" / "fixtures" / "acs_openclaw_2026_7_1_trajectory.jsonl"
PAGE = ROOT / "docs" / "acs-fall-2026-workshop.md"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acs_trajectory_verifier", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


APPROVED_ZIP_MEMBERS = (
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

APPROVED_CHAT_PNGS = (
    "01-inspection/library_preview.png",
    "04-clusters/cluster_sizes.png",
    "06-mmff94/optimized_structures.png",
    "07-objective/final_panel.png",
)

EVENT_KEYS = {
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


def _event(data: dict[str, object], seq: int = 1) -> dict[str, object]:
    return {
        "data": data,
        "modelApi": "synthetic-api",
        "modelId": "synthetic-model",
        "provider": "synthetic-provider",
        "runId": "synthetic-run",
        "schemaVersion": 1,
        "seq": seq,
        "sessionId": "synthetic-session",
        "sessionKey": "synthetic-session-key",
        "source": "synthetic-fixture",
        "sourceSeq": seq,
        "traceId": "synthetic-trace",
        "traceSchema": "openclaw-trajectory",
        "ts": 1_786_000_000_000 + seq,
        "type": "synthetic-event",
        "workspaceDir": "/synthetic/workspace",
    }


def _tool_call(call_id: str, command: str) -> dict[str, object]:
    arguments = {"command": command}
    return {
        "api": "synthetic-api",
        "model": "synthetic-model",
        "provider": "synthetic-provider",
        "responseId": f"response-{call_id}",
        "role": "assistant",
        "stopReason": "toolUse",
        "timestamp": 1_786_000_000_000,
        "usage": {},
        "content": [
            {
                "type": "toolCall",
                "name": "exec",
                "id": call_id,
                "arguments": arguments,
                "partialArgs": json.dumps(arguments, separators=(",", ":")),
            }
        ],
    }


def _tool_result(
    call_id: str, result: dict[str, object], *, include_details: bool = False
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": "exec",
        "isError": False,
        "timestamp": 1_786_000_000_000,
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, sort_keys=True, separators=(",", ":")),
            }
        ],
    }
    if include_details:
        message["details"] = {"synthetic": True}
    return message


def _assistant_answer(answer: str, response_id: str = "answer") -> dict[str, object]:
    return {
        "api": "synthetic-api",
        "content": [{"type": "text", "text": answer}],
        "model": "synthetic-model",
        "provider": "synthetic-provider",
        "responseId": f"response-{response_id}",
        "role": "assistant",
        "stopReason": "stop",
        "timestamp": 1_786_000_000_001,
        "usage": {},
    }


def _user_prompt(prompt: str, index: int) -> dict[str, object]:
    content: object = [{"type": "text", "text": prompt}] if index == 3 else prompt
    return {
        "content": content,
        "role": "user",
        "timestamp": 1_786_000_000_000 + index,
    }


LESSON_STAGES = (
    ("inspect_library", "generate_morgan_fingerprints"),
    ("measure_tanimoto_similarity", "discover_fused_butina_clusters"),
    ("embed_representative_conformers", "optimize_conformers_mmff94"),
)
STAGE_EXECUTION = {
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

LESSON_QUESTIONS = (
    "What is in the fixed molecule library, and how is it represented for comparison?",
    "Which molecules are similar, and how does Butina group them from distances "
    "derived from GPU-computed Tanimoto similarities?",
    "What sampled 3D geometries were generated and optimized?",
)
LESSON_MEANINGS = (
    "The validated molecules were converted into fixed-length structural "
    "descriptors that support comparisons within this exercise.",
    "The similarity stage compares the fixed fingerprints; Butina then groups "
    "molecules whose Tanimoto distances satisfy the fixed rule.",
    "The fixed representatives received deterministic ETKDGv3 conformer samples "
    "followed by within-molecule MMFF94 optimization.",
)
LESSON_LIMITS = (
    "This is a deterministic 256-record ChEMBL convenience sample, not "
    "representative chemical space. Morgan and Tanimoto conclusions depend on "
    "the radius-2, 1024-bit hashed fingerprint.",
    "The cutoff 0.40 is Tanimoto distance, not similarity. Results depend on the "
    "radius-2, 1024-bit hashed fingerprint, and similarity 1.0 does not prove "
    "molecular identity or biological behavior.",
    "The selected molecules are not centroids, medoids, or globally optimal "
    "representatives. Sampled conformers are not experimental structures, and "
    "MMFF94 energies compare sampled conformers within one molecule only.",
)
STAGE_RESULTS = {
    "inspect_library": (
        "256 raw rows; 256 valid molecules; 0 invalid molecules; 24 molecules in "
        "the preview."
    ),
    "generate_morgan_fingerprints": (
        "Morgan radius 2 with 1024 bits produced packed shape 256 x 128; active "
        "bits min 4, median 18.000, max 45."
    ),
    "measure_tanimoto_similarity": (
        'top non-self pair "mol-001" and "mol-002" had Tanimoto similarity '
        "0.875; q1 0.100, median 0.200, q3 0.300, p90 0.400."
    ),
    "discover_fused_butina_clusters": (
        "cutoff 0.40 produced 42 clusters with 18 singletons; largest cluster "
        "sizes: 21, 18, 17, 16, 15, 14, 13, 12, 12, 11, 11, 10, 9, 8, 6."
    ),
    "embed_representative_conformers": (
        "selected 6 of 6 representatives and generated 30 of 30 requested "
        "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7."
    ),
    "optimize_conformers_mmff94": (
        "30 conformers attempted; 29 converged; 1 unconverged; within-molecule "
        'minima: "mol-001"=-12.345 kcal/mol, "mol-002"=-11.234 kcal/mol, '
        '"mol-003"=-10.123 kcal/mol, "mol-004"=-9.012 kcal/mol, '
        '"mol-005"=-8.901 kcal/mol, "mol-006"=-7.890 kcal/mol; maximum '
        "iterations 500."
    ),
}


def _compact_stage(stage: str) -> dict[str, object]:
    placement, software, operation, upstream_stage = STAGE_EXECUTION[stage]
    gpu: dict[str, object] | None = None
    if stage != "inspect_library":
        gpu = {
            "name": "NVIDIA L4",
            "device": "cuda:0",
            "torch_version": "synthetic-torch",
            "nvmolkit_version": "synthetic-nvmolkit",
        }
    upstream: dict[str, object] | None = None
    if upstream_stage is not None:
        upstream_placement, upstream_software, upstream_operation, _ = STAGE_EXECUTION[
            upstream_stage
        ]
        upstream = {
            "stage": upstream_stage,
            "placement": upstream_placement,
            "software": upstream_software,
            "operation": upstream_operation,
        }
    image_count = 2 if stage == "optimize_conformers_mmff94" else 1
    return {
        "stage": stage,
        "result": STAGE_RESULTS[stage],
        "execution": {
            "placement": placement,
            "software": software,
            "operation": operation,
            "upstream": upstream,
            "gpu": gpu,
        },
        "image_paths": [
            f"/synthetic/workshop/{stage}/image-{index}.png"
            for index in range(image_count)
        ],
        "summary_path": f"/synthetic/workshop/{stage}/summary.json",
        "readme_path": f"/synthetic/workshop/{stage}/README.md",
        "artifact_directory": f"/synthetic/workshop/{stage}",
    }


def _lesson_answer(index: int, stages: Sequence[dict[str, object]]) -> str:
    sentences: list[str] = []
    measured: list[str] = []
    for stage in stages:
        execution = stage["execution"]
        assert isinstance(execution, dict)
        placement = execution["placement"]
        software = execution["software"]
        operation = execution["operation"]
        gpu = execution["gpu"]
        upstream = execution["upstream"]
        assert isinstance(placement, str)
        assert isinstance(software, str)
        assert isinstance(operation, str)
        if placement == "GPU":
            assert isinstance(gpu, dict)
            sentences.append(
                f"{software} ran {operation} on GPU {gpu['name']} ({gpu['device']})."
            )
        elif upstream is None:
            sentences.append(f"{software} ran {operation} on CPU.")
        else:
            assert isinstance(upstream, dict)
            assert isinstance(gpu, dict)
            sentences.append(
                f"{software} ran {operation} on CPU using {upstream['software']} "
                f"{upstream['operation']} results computed on GPU {gpu['name']} "
                f"({gpu['device']})."
            )
        result = stage["result"]
        assert isinstance(result, str)
        measured.append(result)
    what_ran = " ".join(sentences)
    if index == 1:
        first_execution = stages[0]["execution"]
        assert isinstance(first_execution, dict)
        gpu = first_execution["gpu"]
        assert isinstance(gpu, dict)
        what_ran = (
            "nvMolKit generated Morgan fingerprints and computed Tanimoto "
            f"similarities on GPU {gpu['name']} ({gpu['device']}). RDKit ran "
            "Butina clustering on CPU using those GPU-computed similarities."
        )
    elif index == 2:
        what_ran = f"One command returned both stages. {what_ran}"
    return (
        f"## Question\n{LESSON_QUESTIONS[index]}\n\n"
        f"## What ran\n{what_ran}\n\n"
        "## Measured result\n"
        + "\n".join(f"- {item}" for item in measured)
        + f"\n\n## Meaning\n{LESSON_MEANINGS[index]}\n\n"
        f"## Scientific limit\n{LESSON_LIMITS[index]}\n\n"
        "## Image and download location\nThe current bundle is in **Download "
        "Results** at `workshop/results.zip`.\n\n"
        f"{verifier.PROMPT_MEDIA[index]}"
    )


def _lesson_result(index: int) -> dict[str, object]:
    lessons = (
        "data-and-representation",
        "relationships-and-groups",
        "sampled-3d-geometry",
    )
    completed_stages = [_compact_stage(stage) for stage in LESSON_STAGES[index]]
    return {
        "schema_version": 1,
        "status": "complete",
        "lesson": lessons[index],
        "completed_stages": completed_stages,
        "results_zip_path": "/synthetic/workshop/results.zip",
        "artifact_relative_zip_path": "workshop/results.zip",
        "answer_markdown": _lesson_answer(index, completed_stages),
    }


def _panel_ids(attempt_count: int) -> list[str]:
    ids = ["mol-a", "mol-b", "mol-c", "mol-d"]
    for index, replacement in enumerate(("mol-e", "mol-f", "mol-g")):
        if index >= attempt_count:
            break
        ids[index] = replacement
    return ids


def _objective_score(attempt_count: int) -> float:
    return 0.5 if attempt_count == 0 else 0.55 + 0.05 * attempt_count


def _measurement(attempt_count: int, *, achieved: bool = False) -> dict[str, object]:
    ids = _panel_ids(attempt_count)
    return {
        "selected_ids": ids,
        "score": _objective_score(attempt_count),
        "limiting_pairs": [[ids[-2], ids[-1]]],
        "achieved": achieved,
    }


def _objective_action(number: int, *, maximum: bool) -> dict[str, object]:
    current_ids = _panel_ids(number - 1)
    replace_index = number - 1 if maximum else 3
    replacement = ("mol-e", "mol-f", "mol-g")[number - 1] if maximum else "mol-h"
    resulting_ids = list(current_ids)
    replace_id = resulting_ids[replace_index]
    resulting_ids[replace_index] = replacement
    current_score = _objective_score(number - 1)
    predicted_score = _objective_score(number) if maximum else current_score + 0.01
    return {
        "swap_id": f"{replace_id}->{replacement}",
        "replace_id": replace_id,
        "replacement_id": replacement,
        "resulting_ids": resulting_ids,
        "predicted_score": predicted_score,
        "score_delta": predicted_score - current_score,
        "limiting_pairs": [[resulting_ids[-2], resulting_ids[-1]]],
        "target_status": "below_target",
    }


def _pending_objective(number: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pending",
        "terminal": False,
        "attempt_count": number - 1,
        "attempt_limit": 3,
        "state_id": f"state-{number:03d}",
        "current": _measurement(number - 1),
        "target_score": 0.75,
        "actions": [
            _objective_action(number, maximum=True),
            _objective_action(number, maximum=False),
        ],
        "achieved": None,
        "termination_reason": None,
        "image_paths": [],
        "artifact_directory": "/synthetic/workshop/07-objective",
        "results_zip_path": "/synthetic/workshop/results.zip",
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def _terminal_objective(objective_step_count: int = 1) -> dict[str, object]:
    baseline = _measurement(0, achieved=objective_step_count == 0)
    final = _measurement(objective_step_count, achieved=objective_step_count == 0)
    baseline_ids = baseline["selected_ids"]
    baseline_pairs = baseline["limiting_pairs"]
    final_ids = final["selected_ids"]
    assert isinstance(baseline_ids, list)
    assert isinstance(baseline_pairs, list)
    assert isinstance(baseline_pairs[0], list)
    assert isinstance(final_ids, list)
    swap_ids = [
        str(_objective_action(number, maximum=True)["swap_id"])
        for number in range(1, objective_step_count + 1)
    ]

    def quoted(values: Sequence[object]) -> str:
        return ", ".join(json.dumps(value) for value in values)

    swap_text = quoted(swap_ids) if swap_ids else "none"
    baseline_score = _objective_score(0)
    final_score = _objective_score(objective_step_count)
    meaning = (
        "A larger `D_min` means the least separated pair in the selected panel "
        "became more separated in this fingerprint space."
        if final_score > baseline_score
        else "`D_min` did not increase; the weakest-link separation remained "
        "unchanged in this fingerprint space."
    )
    answer = (
        "## Question\nCan a bounded agent improve the weakest-link diversity of a "
        "four-molecule panel?\n\n"
        f"## What ran\nBaseline panel: {quoted(baseline_ids)}. Baseline "
        f"limiting pair: {quoted(baseline_pairs[0])}. Accepted swaps: "
        f"{swap_text}. Final panel: {quoted(final_ids)}. Python validated each "
        "displayed maximum-score action against the fixed Tanimoto distance "
        "matrix derived from nvMolKit GPU-computed Morgan fingerprints and "
        "similarities.\n\n"
        "## Measured result\n"
        f"- Baseline `D_min`: {baseline_score:.3f}.\n"
        f"- Final `D_min`: {final_score:.3f}.\n"
        f"- Change in `D_min`: {final_score - baseline_score:+.3f}.\n\n"
        f"## Meaning\n{meaning}\n\n"
        "## Scientific limit\n`D_min` is the minimum pairwise Tanimoto distance, "
        "`min(1 - Tanimoto similarity)`, and the weakest-link diversity score "
        "within eight fixed candidates. This structural-descriptor objective "
        "does not demonstrate unrestricted autonomous design or biological "
        "performance.\n\n"
        "## Image and download location\nThe current bundle is in **Download "
        "Results** at `workshop/results.zip`.\n\n"
        f"{verifier.PROMPT_MEDIA[3]}"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "terminal": True,
        "attempt_count": objective_step_count,
        "attempt_limit": 3,
        "baseline": baseline,
        "final": final,
        "target_score": 0.5 if objective_step_count == 0 else 0.75,
        "attempts": [
            {
                "attempt_number": number,
                "state_id": f"state-{number:03d}",
                **_measurement(number),
                "selected_swap": _objective_action(number, maximum=True),
            }
            for number in range(1, objective_step_count + 1)
        ],
        "achieved": objective_step_count == 0,
        "termination_reason": (
            "baseline_already_optimal"
            if objective_step_count == 0
            else (
                "attempt_limit_reached"
                if objective_step_count == 3
                else "no_legal_improving_swap"
            )
        ),
        "image_paths": [
            "/synthetic/workshop/07-objective/score_trajectory.png",
            "/synthetic/workshop/07-objective/final_panel.png",
            "/synthetic/workshop/07-objective/final_similarity_heatmap.png",
        ],
        "artifact_directory": "/synthetic/workshop/07-objective",
        "results_zip_path": "/synthetic/workshop/results.zip",
        "artifact_relative_zip_path": "workshop/results.zip",
        "answer_markdown": answer,
    }


def _valid_messages(
    prompts: tuple[str, ...], objective_step_count: int = 1
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for index in range(3):
        call_id = f"lesson-{index + 1}"
        result = _lesson_result(index)
        messages.extend(
            (
                _user_prompt(prompts[index], index),
                _tool_call(call_id, verifier.LESSON_COMMANDS[index]),
                _tool_result(call_id, result, include_details=index == 1),
                _assistant_answer(str(result["answer_markdown"]), call_id),
            )
        )
    first_result = (
        _terminal_objective(0) if objective_step_count == 0 else _pending_objective(1)
    )
    messages.extend(
        (
            _user_prompt(prompts[3], 3),
            _tool_call("objective-start", verifier.OBJECTIVE_START),
            _tool_result("objective-start", first_result),
        )
    )
    for number in range(1, objective_step_count + 1):
        pending = _pending_objective(number)
        actions = pending["actions"]
        assert isinstance(actions, list)
        best_action = actions[0]
        assert isinstance(best_action, dict)
        command = (
            f"{verifier.RUNNER_PREFIX} objective-step "
            f"--state-id '{pending['state_id']}' --swap-id '{best_action['swap_id']}'"
        )
        result = (
            _terminal_objective(objective_step_count)
            if number == objective_step_count
            else _pending_objective(number + 1)
        )
        call_id = f"objective-step-{number}"
        messages.extend(
            (
                _tool_call(call_id, command),
                _tool_result(call_id, result, include_details=number == 1),
            )
        )
    terminal = _terminal_objective(objective_step_count)
    messages.append(
        _assistant_answer(str(terminal["answer_markdown"]), "objective-final")
    )
    return messages


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def _png(width: int = 1, height: int = 1) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixels = b"\x00" + b"\x00\x00\x00\xff" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(pixels * height))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk_bounds(contents: bytes, wanted: bytes) -> tuple[int, int, int]:
    offset = 8
    while offset + 12 <= len(contents):
        length = struct.unpack_from(">I", contents, offset)[0]
        end = offset + 12 + length
        if contents[offset + 4 : offset + 8] == wanted:
            return offset, offset + 8, end
        offset = end
    raise AssertionError(f"PNG chunk {wanted!r} is missing")


def _replace_png_chunk(contents: bytes, kind: bytes, payload: bytes) -> bytes:
    start, _, end = _png_chunk_bounds(contents, kind)
    return contents[:start] + _png_chunk(kind, payload) + contents[end:]


def _corrupt_png_crc(contents: bytes, kind: bytes) -> bytes:
    _, _, end = _png_chunk_bounds(contents, kind)
    changed = bytearray(contents)
    changed[end - 1] ^= 1
    return bytes(changed)


def _write_archive(
    path: Path,
    *,
    names: tuple[str, ...] = APPROVED_ZIP_MEMBERS,
    contents: dict[str, bytes] | None = None,
    sizes: dict[str, int] | None = None,
    modes: dict[str, int] | None = None,
    extras: dict[str, bytes] | None = None,
    timestamps: dict[str, tuple[int, int, int, int, int, int]] | None = None,
    compressions: dict[str, int] | None = None,
    archive_comment: bytes = b"",
    entry_comments: dict[str, bytes] | None = None,
) -> None:
    contents = {} if contents is None else contents
    sizes = {} if sizes is None else sizes
    modes = {} if modes is None else modes
    extras = {} if extras is None else extras
    timestamps = {} if timestamps is None else timestamps
    compressions = {} if compressions is None else compressions
    entry_comments = {} if entry_comments is None else entry_comments
    with zipfile.ZipFile(path, "w", strict_timestamps=True) as archive:
        for name in names:
            info = zipfile.ZipInfo(
                name, date_time=timestamps.get(name, (1980, 1, 1, 0, 0, 0))
            )
            info.compress_type = compressions.get(name, zipfile.ZIP_DEFLATED)
            info.external_attr = modes.get(name, stat.S_IFREG | 0o644) << 16
            info.extra = extras.get(name, b"")
            info.comment = entry_comments.get(name, b"")
            default = _png() if name.endswith(".png") else b"validated fixture\n"
            payload = contents.get(
                name, b"x" * sizes[name] if name in sizes else default
            )
            archive.writestr(info, payload)
        archive.comment = archive_comment


def _write_valid_archive(path: Path) -> None:
    _write_archive(path)


def _latest_snapshot(path: Path) -> list[dict[str, object]]:
    loaded: object = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert isinstance(loaded, dict)
    data = loaded["data"]
    assert isinstance(data, dict)
    snapshot = data["messagesSnapshot"]
    assert isinstance(snapshot, list)
    messages: list[dict[str, object]] = []
    for message in snapshot:
        assert isinstance(message, dict)
        assert all(isinstance(key, str) for key in message)
        messages.append(message)
    return messages


def _write_snapshot(path: Path, snapshot: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(_event({"messagesSnapshot": snapshot})) + "\n",
        encoding="utf-8",
    )


def _user_indices(snapshot: list[dict[str, object]]) -> list[int]:
    return [index for index, item in enumerate(snapshot) if item.get("role") == "user"]


def _third_user_index(snapshot: list[dict[str, object]]) -> int:
    return _user_indices(snapshot)[2]


def _next_user_index(snapshot: list[dict[str, object]], current: int) -> int:
    return next(index for index in _user_indices(snapshot) if index > current)


def _final_answer_index(snapshot: list[dict[str, object]], turn: int) -> int:
    users = _user_indices(snapshot)
    return (users[turn + 1] if turn < 3 else len(snapshot)) - 1


def _first_final_answer_index(snapshot: list[dict[str, object]]) -> int:
    return _final_answer_index(snapshot, 0)


def _objective_step_block(snapshot: list[dict[str, object]]) -> dict[str, object]:
    for message in snapshot:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            arguments = block.get("arguments")
            if (
                block.get("type") == "toolCall"
                and isinstance(arguments, dict)
                and " objective-step " in str(arguments.get("command", ""))
            ):
                return block
    raise AssertionError("objective-step fixture block is missing")


def _find_message(snapshot: list[dict[str, object]], call_id: str) -> int:
    return next(
        index
        for index, message in enumerate(snapshot)
        if message.get("role") == "toolResult" and message.get("toolCallId") == call_id
    )


def _result_payload(
    snapshot: list[dict[str, object]], call_id: str
) -> dict[str, object]:
    message = snapshot[_find_message(snapshot, call_id)]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    loaded: object = json.loads(str(block["text"]))
    assert isinstance(loaded, dict)
    result: dict[str, object] = {}
    for key, value in loaded.items():
        assert isinstance(key, str)
        result[key] = value
    return result


def _set_result_payload(
    snapshot: list[dict[str, object]], call_id: str, payload: dict[str, object]
) -> None:
    message = snapshot[_find_message(snapshot, call_id)]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    block["text"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _set_canonical_answer(
    snapshot: list[dict[str, object]], call_id: str, turn: int, answer: str
) -> None:
    payload = _result_payload(snapshot, call_id)
    payload["answer_markdown"] = answer
    _set_result_payload(snapshot, call_id, payload)
    message = snapshot[_final_answer_index(snapshot, turn)]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    block["text"] = answer


def _set_lesson_stage_result(
    snapshot: list[dict[str, object]],
    *,
    call_id: str,
    turn: int,
    stage_index: int,
    stage_result: str,
) -> None:
    result = _result_payload(snapshot, call_id)
    completed_stages = result["completed_stages"]
    assert isinstance(completed_stages, list)
    stages: list[dict[str, object]] = []
    for item in completed_stages:
        assert isinstance(item, dict)
        assert all(isinstance(key, str) for key in item)
        stages.append(item)
    stages[stage_index]["result"] = stage_result
    answer = _lesson_answer(turn, stages)
    result["answer_markdown"] = answer
    _set_result_payload(snapshot, call_id, result)
    message = snapshot[_final_answer_index(snapshot, turn)]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    block["text"] = answer


def _set_lesson_gpu_field(
    snapshot: list[dict[str, object]],
    *,
    call_id: str,
    turn: int,
    stage_index: int,
    field: str,
    value: str,
) -> None:
    result = _result_payload(snapshot, call_id)
    completed_stages = result["completed_stages"]
    assert isinstance(completed_stages, list)
    stages: list[dict[str, object]] = []
    for item in completed_stages:
        assert isinstance(item, dict)
        assert all(isinstance(key, str) for key in item)
        stages.append(item)
    execution = stages[stage_index]["execution"]
    assert isinstance(execution, dict)
    gpu = execution["gpu"]
    assert isinstance(gpu, dict)
    gpu[field] = value
    answer = _lesson_answer(turn, stages)
    result["answer_markdown"] = answer
    _set_result_payload(snapshot, call_id, result)
    message = snapshot[_final_answer_index(snapshot, turn)]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    block["text"] = answer


def _fixture_prompts() -> tuple[str, ...]:
    prompts: list[str] = []
    for message in _latest_snapshot(FIXTURE):
        if message.get("role") != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            prompts.append(content)
            continue
        assert isinstance(content, list)
        assert len(content) == 1
        block = content[0]
        assert isinstance(block, dict)
        text = block["text"]
        assert isinstance(text, str)
        prompts.append(text)
    assert len(prompts) == 4
    return tuple(prompts)


def _fixture_contracts() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (prompt_id, prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        for prompt_id, prompt in zip(
            verifier.PROMPT_IDS, _fixture_prompts(), strict=True
        )
    )


def _write_pinned_page(path: Path) -> None:
    regions = []
    prompts = tuple(contract[1] for contract in verifier.load_prompt_contracts(PAGE))
    for prompt_id, prompt in zip(verifier.PROMPT_IDS, prompts, strict=True):
        regions.append(
            f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->\n"
            f"~~~text\n{prompt}\n~~~\n"
            f"<!-- ACS_PROMPT:{prompt_id}:END -->\n"
        )
    path.write_text("\n".join(regions), encoding="utf-8")


def _valid_evidence(
    tmp_path: Path, objective_step_count: int = 1
) -> tuple[Path, Path, Path]:
    page = tmp_path / "pinned-page.md"
    _write_pinned_page(page)
    prompts = tuple(contract[1] for contract in verifier.load_prompt_contracts(page))
    trajectory = tmp_path / "trajectory.jsonl"
    archive = tmp_path / "results.zip"
    _write_snapshot(trajectory, _valid_messages(prompts, objective_step_count))
    _write_valid_archive(archive)
    return trajectory, archive, page


def _patch_zip_flag(path: Path, mask: int) -> None:
    raw = bytearray(path.read_bytes())
    local = raw.find(b"PK\x03\x04")
    central = raw.find(b"PK\x01\x02")
    assert local >= 0
    assert central >= 0
    local_flags = struct.unpack_from("<H", raw, local + 6)[0]
    central_flags = struct.unpack_from("<H", raw, central + 8)[0]
    struct.pack_into("<H", raw, local + 6, local_flags | mask)
    struct.pack_into("<H", raw, central + 8, central_flags | mask)
    path.write_bytes(raw)


def _patch_central_crc(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    central = raw.find(b"PK\x01\x02")
    assert central >= 0
    crc = struct.unpack_from("<I", raw, central + 16)[0]
    struct.pack_into("<I", raw, central + 16, crc ^ 0xFFFFFFFF)
    path.write_bytes(raw)


def _forge_one_byte_declared_member(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    local = raw.find(b"PK\x03\x04")
    central = raw.find(b"PK\x01\x02")
    assert local >= 0
    assert central >= 0
    declared = b"x"
    crc = zlib.crc32(declared) & 0xFFFFFFFF
    struct.pack_into("<I", raw, local + 14, crc)
    struct.pack_into("<I", raw, local + 22, len(declared))
    struct.pack_into("<I", raw, central + 16, crc)
    struct.pack_into("<I", raw, central + 24, len(declared))
    path.write_bytes(raw)


def _patch_zip_version(path: Path, mutation: str) -> None:
    raw = bytearray(path.read_bytes())
    local = raw.find(b"PK\x03\x04")
    central = raw.find(b"PK\x01\x02")
    assert local >= 0
    assert central >= 0
    if mutation == "local-needed":
        struct.pack_into("<H", raw, local + 4, 21)
    elif mutation == "central-needed":
        struct.pack_into("<H", raw, central + 6, 21)
    else:
        struct.pack_into("<H", raw, central + 4, 20)
    path.write_bytes(raw)


def _patch_local_dos_date(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    local = raw.find(b"PK\x03\x04")
    assert local >= 0
    dos_date = struct.unpack_from("<H", raw, local + 12)[0]
    struct.pack_into("<H", raw, local + 12, dos_date + 1)
    path.write_bytes(raw)


def _assert_archive_rejected(path: Path) -> None:
    with pytest.raises(verifier.VerificationError, match="^archive_contract$"):
        verifier.validate_results_zip(path)


def test_required_archive_contract_is_pinned() -> None:
    assert verifier.REQUIRED_ZIP_MEMBERS == APPROVED_ZIP_MEMBERS
    assert verifier.REQUIRED_CHAT_PNGS == APPROVED_CHAT_PNGS


def test_valid_trajectory_and_archive_emit_closed_receipt(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    receipt = verifier.verify_acceptance(trajectory, archive, page)
    assert receipt == {
        "schema_version": 1,
        "status": "pass",
        "prompt_count": 4,
        "exec_call_count": 5,
        "objective_step_count": 1,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_size": archive.stat().st_size,
        "required_png_count": 4,
    }


@pytest.mark.parametrize("objective_step_count", range(4))
def test_accepts_every_bounded_objective_step_count(
    tmp_path: Path, objective_step_count: int
) -> None:
    trajectory, archive, page = _valid_evidence(
        tmp_path, objective_step_count=objective_step_count
    )
    receipt = verifier.verify_acceptance(trajectory, archive, page)
    assert receipt["exec_call_count"] == 4 + objective_step_count
    assert receipt["objective_step_count"] == objective_step_count


def test_rejects_more_than_eight_global_objective_candidates(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path, objective_step_count=3)
    snapshot = _latest_snapshot(trajectory)
    pending = _result_payload(snapshot, "objective-start")
    actions = pending["actions"]
    assert isinstance(actions, list)
    alternative = actions[1]
    assert isinstance(alternative, dict)
    alternative["replacement_id"] = "mol-i"
    alternative["swap_id"] = "mol-d->mol-i"
    alternative["resulting_ids"] = ["mol-a", "mol-b", "mol-c", "mol-i"]
    alternative["limiting_pairs"] = [["mol-c", "mol-i"]]
    _set_result_payload(snapshot, "objective-start", pending)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_accepts_a_tied_maximum_objective_action(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    pending = _result_payload(snapshot, "objective-start")
    actions = pending["actions"]
    assert isinstance(actions, list)
    first = actions[0]
    second = actions[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["predicted_score"] = first["predicted_score"]
    second["score_delta"] = first["score_delta"]
    _set_result_payload(snapshot, "objective-start", pending)
    block = _objective_step_block(snapshot)
    block["arguments"] = {
        "command": (
            f"{verifier.RUNNER_PREFIX} objective-step --state-id 'state-001' "
            f"--swap-id '{second['swap_id']}'"
        )
    }
    block["partialArgs"] = json.dumps(block["arguments"], separators=(",", ":"))
    terminal = _result_payload(snapshot, "objective-step-1")
    resulting_ids = second["resulting_ids"]
    score = second["predicted_score"]
    limiting_pairs = second["limiting_pairs"]
    final = {
        "selected_ids": resulting_ids,
        "score": score,
        "limiting_pairs": limiting_pairs,
        "achieved": False,
    }
    terminal["final"] = final
    terminal["attempts"] = [
        {
            "attempt_number": 1,
            "state_id": "state-001",
            **final,
            "selected_swap": second,
        }
    ]
    terminal["answer_markdown"] = verifier._canonical_objective_answer(
        terminal, verifier.PROMPT_MEDIA[3]
    )
    _set_canonical_answer(
        snapshot,
        "objective-step-1",
        3,
        str(terminal["answer_markdown"]),
    )
    _set_result_payload(snapshot, "objective-step-1", terminal)
    _write_snapshot(trajectory, snapshot)
    verifier.verify_acceptance(trajectory, archive, page)


def test_sanitized_openclaw_2026_7_1_fixture_has_six_causal_pairs() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert set(event) == EVENT_KEYS
    assert event["schemaVersion"] == 1
    assert event["traceSchema"] == "openclaw-trajectory"
    contracts = _fixture_contracts()
    messages = verifier.load_messages_snapshot(FIXTURE)
    assert len(messages) == 20
    assert [message["role"] for message in messages].count("user") == 4
    assert [message["role"] for message in messages].count("assistant") == 10
    assert [message["role"] for message in messages].count("toolResult") == 6
    assert verifier.validate_trajectory(messages, contracts) == (6, 2)


@pytest.mark.parametrize("mutation", ["changed", "reordered"])
def test_rejects_changed_or_reordered_prompt(tmp_path: Path, mutation: str) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    if mutation == "changed":
        snapshot[0]["content"] = str(snapshot[0]["content"]) + " changed"
    else:
        first = snapshot[0]["content"]
        second_user = _user_indices(snapshot)[1]
        snapshot[0]["content"] = snapshot[second_user]["content"]
        snapshot[second_user]["content"] = first
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^prompt_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_page_and_trajectory_changed_together(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    changed_page = tmp_path / "changed-page.md"
    changed_page.write_bytes(
        page.read_bytes().replace(b"Question:", b"Question changed:", 1)
    )
    snapshot = _latest_snapshot(trajectory)
    snapshot[0]["content"] = str(snapshot[0]["content"]).replace(
        "Question:", "Question changed:", 1
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^prompt_contract$"):
        verifier.verify_acceptance(trajectory, archive, changed_page)


def test_rejects_duplicate_prompt_3_command(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    third_turn_end = _next_user_index(snapshot, _third_user_index(snapshot))
    snapshot[third_turn_end:third_turn_end] = [
        _tool_call("duplicate-p3", verifier.LESSON_COMMANDS[2]),
        _tool_result("duplicate-p3", _lesson_result(2)),
    ]
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_next_call_before_prior_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    objective_start = _user_indices(snapshot)[3] + 1
    step_call = snapshot.pop(objective_start + 2)
    snapshot.insert(objective_start + 1, step_call)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_whitespace_between_call_and_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    snapshot.insert(2, _assistant_answer("  \n", "pre-result-whitespace"))
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_partial_args_that_do_not_match_arguments(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    first_call = snapshot[1]["content"]
    assert isinstance(first_call, list)
    assert isinstance(first_call[0], dict)
    first_call[0]["partialArgs"] = '{"command":"synthetic mismatch"}'
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize("assistant_kind", ["tool-call", "final-text"])
def test_rejects_swapped_assistant_stop_reason(
    tmp_path: Path, assistant_kind: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = (
        snapshot[1]
        if assistant_kind == "tool-call"
        else snapshot[_first_final_answer_index(snapshot)]
    )
    message["stopReason"] = "stop" if assistant_kind == "tool-call" else "toolUse"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_error_tool_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = snapshot[_find_message(snapshot, "lesson-1")]
    message["isError"] = True
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_string_instead_of_tool_result_text_block(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = snapshot[_find_message(snapshot, "lesson-1")]
    content = message["content"]
    assert isinstance(content, list)
    assert isinstance(content[0], dict)
    message["content"] = content[0]["text"]
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_malformed_tool_result_json(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = snapshot[_find_message(snapshot, "lesson-1")]
    content = message["content"]
    assert isinstance(content, list)
    assert isinstance(content[0], dict)
    content[0]["text"] = "{"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_duplicate_json_keys_in_tool_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = snapshot[_find_message(snapshot, "lesson-1")]
    content = message["content"]
    assert isinstance(content, list)
    assert isinstance(content[0], dict)
    content[0]["text"] = '{"status":"complete","status":"complete"}'
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_incomplete_lesson(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "lesson-2")
    result["status"] = "pending"
    _set_result_payload(snapshot, "lesson-2", result)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "mutation",
    [
        "sequence",
        "stage-extra",
        "result-empty",
        "execution-null",
        "execution-extra",
        "gpu-empty",
        "upstream-string",
        "image-type",
        "path-empty",
    ],
)
def test_rejects_open_or_malformed_completed_stage_items(
    tmp_path: Path, mutation: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    call_id = "lesson-2" if mutation == "upstream-string" else "lesson-1"
    result = _result_payload(snapshot, call_id)
    stages = result["completed_stages"]
    assert isinstance(stages, list)
    assert isinstance(stages[0], dict)
    assert isinstance(stages[1], dict)
    if mutation == "sequence":
        stages.reverse()
    elif mutation == "stage-extra":
        stages[0]["extra"] = "drift"
    elif mutation == "result-empty":
        stages[0]["result"] = ""
    elif mutation == "execution-null":
        stages[0]["execution"] = None
    elif mutation == "execution-extra":
        execution = stages[0]["execution"]
        assert isinstance(execution, dict)
        execution["extra"] = "drift"
    elif mutation == "gpu-empty":
        execution = stages[1]["execution"]
        assert isinstance(execution, dict)
        gpu = execution["gpu"]
        assert isinstance(gpu, dict)
        gpu["device"] = ""
    elif mutation == "upstream-string":
        execution = stages[1]["execution"]
        assert isinstance(execution, dict)
        execution["upstream"] = "GPU stage"
    elif mutation == "image-type":
        stages[0]["image_paths"] = "image.png"
    else:
        stages[0]["summary_path"] = ""
    _set_result_payload(snapshot, call_id, result)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_hostile_fabricated_lesson_result_and_matching_answers(
    tmp_path: Path,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=0,
        stage_result="GPU accelerated the workload with a 9x speedup and made it faster.",
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "NVIDIA L4 GPU accelerated 9x speedup and faster"),
        ("device", "cuda:0 faster"),
        ("torch_version", "bad version"),
        ("nvmolkit_version", "v" * 65),
    ],
)
def test_rejects_hostile_or_unsafe_gpu_identity_with_matching_answer(
    tmp_path: Path, field: str, value: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_gpu_field(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        field=field,
        value=value,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize("stage_indices", [(0,), (0, 1)])
def test_rejects_gpu_version_provenance_drift_with_matching_answer(
    tmp_path: Path, stage_indices: tuple[int, ...]
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    for stage_index in stage_indices:
        _set_lesson_gpu_field(
            snapshot,
            call_id="lesson-2",
            turn=1,
            stage_index=stage_index,
            field="nvmolkit_version",
            value="synthetic-drift",
        )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    ("call_id", "turn", "stage_index", "stage_result"),
    [
        (
            "lesson-1",
            0,
            0,
            "10 raw rows; 9 valid molecules; 0 invalid molecules; 2 molecules in "
            "the preview.",
        ),
        (
            "lesson-1",
            0,
            1,
            "Morgan radius 2 with 1024 bits produced packed shape 256 x 128; "
            "active bits min 20, median 18.000, max 45.",
        ),
        (
            "lesson-1",
            0,
            1,
            "Morgan radius 2 with 1024 bits produced packed shape 255 x 128; "
            "active bits min 4, median 18.000, max 45.",
        ),
        (
            "lesson-2",
            1,
            0,
            'top non-self pair "mol-001" and "mol-002" had Tanimoto similarity '
            "1.100; q1 0.100, median 0.200, q3 0.300, p90 0.400.",
        ),
        (
            "lesson-2",
            1,
            1,
            "cutoff 0.40 produced 2 clusters with 3 singletons; largest cluster "
            "sizes: 2, 1.",
        ),
        (
            "lesson-3",
            2,
            0,
            "selected 6 of 6 representatives and generated 31 of 30 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7.",
        ),
        (
            "lesson-3",
            2,
            0,
            "selected 6 of 6 representatives and generated 30 of 30 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 8.",
        ),
        (
            "lesson-3",
            2,
            1,
            "30 conformers attempted; 29 converged; 0 unconverged; "
            'within-molecule minima: "mol-001"=-12.345 kcal/mol; maximum '
            "iterations 500.",
        ),
        (
            "lesson-3",
            2,
            1,
            "30 conformers attempted; 29 converged; 1 unconverged; "
            'within-molecule minima: "mol-001"=-12.345 kcal/mol; maximum '
            "iterations 499.",
        ),
    ],
)
def test_rejects_lesson_stage_result_grammar_or_relation_drift(
    tmp_path: Path,
    call_id: str,
    turn: int,
    stage_index: int,
    stage_result: str,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id=call_id,
        turn=turn,
        stage_index=stage_index,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    ("call_id", "turn", "stage_index", "stage_result"),
    [
        (
            "lesson-1",
            0,
            1,
            "Morgan radius 3 with 1024 bits produced packed shape 256 x 128; "
            "active bits min 4, median 18.000, max 45.",
        ),
        (
            "lesson-1",
            0,
            1,
            "Morgan radius 2 with 2048 bits produced packed shape 256 x 256; "
            "active bits min 4, median 18.000, max 45.",
        ),
        (
            "lesson-2",
            1,
            1,
            "cutoff 0.50 produced 42 clusters with 18 singletons; largest cluster "
            "sizes: 20, 15, 11.",
        ),
        (
            "lesson-3",
            2,
            0,
            "selected 6 of 7 representatives and generated 30 of 30 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7.",
        ),
        (
            "lesson-3",
            2,
            0,
            "selected 6 of 6 representatives and generated 30 of 36 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7.",
        ),
    ],
)
def test_rejects_fixed_lesson_profile_drift(
    tmp_path: Path,
    call_id: str,
    turn: int,
    stage_index: int,
    stage_result: str,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id=call_id,
        turn=turn,
        stage_index=stage_index,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_consistent_non_256_record_sample(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=0,
        stage_result=(
            "255 raw rows; 255 valid molecules; 0 invalid molecules; 12 molecules "
            "in the preview."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        stage_result=(
            "Morgan radius 2 with 1024 bits produced packed shape 255 x 128; "
            "active bits min 4, median 18.000, max 45."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_accepts_exact_fixed_p3_runner_profile(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_legacy_4_by_8_p3_profile(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-3",
        turn=2,
        stage_index=0,
        stage_result=(
            "selected 4 of 4 representatives and generated 32 of 32 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_zero_valid_fixed_sample(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=0,
        stage_result=(
            "256 raw rows; 0 valid molecules; 256 invalid molecules; 0 molecules "
            "in the preview."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        stage_result=(
            "Morgan radius 2 with 1024 bits produced packed shape 0 x 128; "
            "active bits min 0, median 0.000, max 0."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_self_consistent_population_too_small_for_p3(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=0,
        stage_result=(
            "256 raw rows; 1 valid molecules; 255 invalid molecules; 1 molecules "
            "in the preview."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        stage_result=(
            "Morgan radius 2 with 1024 bits produced packed shape 1 x 128; "
            "active bits min 4, median 4.000, max 4."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-2",
        turn=1,
        stage_index=1,
        stage_result=(
            "cutoff 0.40 produced 1 clusters with 1 singletons; largest cluster "
            "sizes: 1."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_cluster_population_mismatch_with_p1_valid_count(
    tmp_path: Path,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-2",
        turn=1,
        stage_index=1,
        stage_result=(
            "cutoff 0.40 produced 1 clusters with 1 singletons; largest cluster "
            "sizes: 1."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_too_few_clusters_for_six_p3_representatives(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-2",
        turn=1,
        stage_index=1,
        stage_result=(
            "cutoff 0.40 produced 1 clusters with 0 singletons; largest cluster "
            "sizes: 256."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "stage_result",
    [
        (
            "cutoff 0.40 produced 1 clusters with 0 singletons; largest cluster "
            "sizes: 999."
        ),
        (
            "cutoff 0.40 produced 42 clusters with 18 singletons; largest cluster "
            "sizes: 21, 18, 17."
        ),
        (
            "cutoff 0.40 produced 42 clusters with 30 singletons; largest cluster "
            "sizes: 21, 18, 17, 16, 15, 14, 13, 12, 12, 11, 11, 10, 9, 8, 6."
        ),
    ],
)
def test_rejects_infeasible_cluster_population(
    tmp_path: Path, stage_result: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-2",
        turn=1,
        stage_index=1,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "stage_result",
    [
        (
            "selected 4 of 6 representatives and generated 20 of 20 requested "
            "conformers; 0 partial ID, 0 zero IDs; ETKDGv3 seed 7."
        ),
        (
            "selected 6 of 6 representatives and generated 30 of 30 requested "
            "conformers; 0 partial ID, 6 zero IDs; ETKDGv3 seed 7."
        ),
        (
            "selected 6 of 6 representatives and generated 0 of 30 requested "
            "conformers; 0 partial ID, 6 zero IDs; ETKDGv3 seed 7."
        ),
        (
            "selected 6 of 6 representatives and generated 25 of 30 requested "
            "conformers; 1 partial ID, 0 zero IDs; ETKDGv3 seed 7."
        ),
        (
            "selected 6 of 6 representatives and generated 30 of 30 requested "
            "conformers; 1 partial ID, 0 zero IDs; ETKDGv3 seed 7."
        ),
    ],
)
def test_rejects_inconsistent_embedding_counts(
    tmp_path: Path, stage_result: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-3",
        turn=2,
        stage_index=0,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "stage_result",
    [
        STAGE_RESULTS["optimize_conformers_mmff94"].replace(
            "30 conformers attempted; 29 converged; 1 unconverged;",
            "999 conformers attempted; 29 converged; 970 unconverged;",
        ),
        (
            "30 conformers attempted; 29 converged; 1 unconverged; "
            "within-molecule minima: none; maximum iterations 500."
        ),
    ],
)
def test_rejects_inconsistent_optimization_counts(
    tmp_path: Path, stage_result: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-3",
        turn=2,
        stage_index=1,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_too_few_minima_for_partial_embedding_capacity(
    tmp_path: Path,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-3",
        turn=2,
        stage_index=0,
        stage_result=(
            "selected 6 of 6 representatives and generated 24 of 30 requested "
            "conformers; 6 partial ID, 0 zero IDs; ETKDGv3 seed 7."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-3",
        turn=2,
        stage_index=1,
        stage_result=(
            "24 conformers attempted; 20 converged; 4 unconverged; "
            'within-molecule minima: "mol-001"=-12.345 kcal/mol, '
            '"mol-002"=-11.234 kcal/mol, "mol-003"=-10.123 kcal/mol, '
            '"mol-004"=-9.012 kcal/mol; maximum iterations 500.'
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_non_half_integer_morgan_active_bit_median(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        stage_result=(
            "Morgan radius 2 with 1024 bits produced packed shape 256 x 128; "
            "active bits min 4, median 18.123, max 45."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_half_integer_morgan_median_for_odd_row_count(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=0,
        stage_result=(
            "256 raw rows; 255 valid molecules; 1 invalid molecules; 24 molecules "
            "in the preview."
        ),
    )
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-1",
        turn=0,
        stage_index=1,
        stage_result=(
            "Morgan radius 2 with 1024 bits produced packed shape 255 x 128; "
            "active bits min 4, median 18.500, max 45."
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    ("call_id", "turn", "stage_index", "stage_result"),
    [
        (
            "lesson-2",
            1,
            0,
            'top non-self pair "mol-001 GPU accelerated 9x speedup" and '
            '"mol-002" had Tanimoto similarity 0.875; q1 0.100, median 0.200, '
            "q3 0.300, p90 0.400.",
        ),
        (
            "lesson-3",
            2,
            1,
            STAGE_RESULTS["optimize_conformers_mmff94"].replace(
                '"mol-001"',
                '"mol-001 GPU accelerated 9x speedup"',
            ),
        ),
    ],
)
def test_rejects_matching_answer_claim_injection_in_stage_identifier(
    tmp_path: Path,
    call_id: str,
    turn: int,
    stage_index: int,
    stage_result: str,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id=call_id,
        turn=turn,
        stage_index=stage_index,
        stage_result=stage_result,
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "identifier",
    ["GPU-Accelerated", "speed_up", "FASTER", "similarity-score"],
)
def test_rejects_safe_token_claim_injection_in_stage_identifier(
    tmp_path: Path, identifier: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    _set_lesson_stage_result(
        snapshot,
        call_id="lesson-2",
        turn=1,
        stage_index=0,
        stage_result=STAGE_RESULTS["measure_tanimoto_similarity"].replace(
            '"mol-001"', json.dumps(identifier)
        ),
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "identifier",
    ["GPU-Accelerated", "speed_up", "FASTER", "similarity-score"],
)
def test_rejects_safe_token_claim_injection_in_objective_answer(
    tmp_path: Path, identifier: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path, objective_step_count=0)
    snapshot = _latest_snapshot(trajectory)
    terminal = _result_payload(snapshot, "objective-start")
    for field in ("baseline", "final"):
        measurement = terminal[field]
        assert isinstance(measurement, dict)
        selected_ids = measurement["selected_ids"]
        assert isinstance(selected_ids, list)
        selected_ids[0] = identifier
    answer = str(terminal["answer_markdown"]).replace('"mol-a"', json.dumps(identifier))
    terminal["answer_markdown"] = answer
    _set_result_payload(snapshot, "objective-start", terminal)
    _set_canonical_answer(snapshot, "objective-start", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "identifier",
    [
        "predicted-score-0.777",
        "target_score_0750",
        "target-0750",
        "intermediate-score-0.650",
        "per-step-score-0.600",
    ],
)
def test_rejects_p4_score_detail_identifier_in_matching_answer(
    tmp_path: Path, identifier: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path, objective_step_count=0)
    snapshot = _latest_snapshot(trajectory)
    terminal = _result_payload(snapshot, "objective-start")
    for field in ("baseline", "final"):
        measurement = terminal[field]
        assert isinstance(measurement, dict)
        selected_ids = measurement["selected_ids"]
        assert isinstance(selected_ids, list)
        selected_ids[0] = identifier
    answer = str(terminal["answer_markdown"]).replace('"mol-a"', json.dumps(identifier))
    terminal["answer_markdown"] = answer
    _set_result_payload(snapshot, "objective-start", terminal)
    _set_canonical_answer(snapshot, "objective-start", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    ("call_id", "mutation", "issue_code"),
    [
        ("lesson-1", "extra", "command_contract"),
        ("lesson-1", "missing", "command_contract"),
        ("objective-start", "extra", "objective_contract"),
        ("objective-start", "missing", "objective_contract"),
        ("objective-step-1", "extra", "objective_contract"),
        ("objective-step-1", "missing", "objective_contract"),
    ],
)
def test_rejects_open_result_object_schemas(
    tmp_path: Path, call_id: str, mutation: str, issue_code: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, call_id)
    if mutation == "extra":
        result["unexpected"] = "synthetic"
    else:
        del result["schema_version"]
    _set_result_payload(snapshot, call_id, result)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match=f"^{issue_code}$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_missing_terminal_objective_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    start_result = _find_message(snapshot, "objective-start")
    del snapshot[start_result + 1 : start_result + 3]
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_call_after_terminal_objective_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    final_result = _find_message(snapshot, "objective-step-1")
    command = (
        f"{verifier.RUNNER_PREFIX} objective-step --state-id 'state-after' "
        "--swap-id 'mol-a->mol-z'"
    )
    snapshot[final_result + 1 : final_result + 1] = [
        _tool_call("after-terminal", command),
        _tool_result("after-terminal", _terminal_objective()),
    ]
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_empty_fragment_after_successful_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result_index = _find_message(snapshot, "lesson-1")
    snapshot.insert(result_index + 1, _assistant_answer("  \n"))
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_extra_assistant_prose_fragment(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    final_answer = _first_final_answer_index(snapshot)
    snapshot.insert(final_answer, _assistant_answer("Synthetic extra prose."))
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_split_canonical_answer_fragments(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    final_answer = _first_final_answer_index(snapshot)
    message = snapshot[final_answer]
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    answer = str(block["text"])
    midpoint = len(answer) // 2
    block["text"] = answer[:midpoint]
    snapshot.insert(final_answer + 1, _assistant_answer(answer[midpoint:]))
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_answer_that_differs_from_answer_markdown(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    message = snapshot[_first_final_answer_index(snapshot)]
    content = message["content"]
    assert isinstance(content, list)
    assert isinstance(content[0], dict)
    content[0]["text"] = str(content[0]["text"]) + " extra"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_matching_lesson_answer_drift_from_closed_result(
    tmp_path: Path,
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "lesson-1")
    answer = str(result["answer_markdown"]).replace(
        LESSON_QUESTIONS[0],
        "What generic workflow result was returned?",
    )
    _set_canonical_answer(snapshot, "lesson-1", 0, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_forbidden_acceleration_claim(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "lesson-1")
    answer = str(result["answer_markdown"]).replace(
        "RDKit ran library parsing and validation on CPU.",
        "RDKit ran accelerated library parsing and validation on CPU.",
    )
    _set_canonical_answer(snapshot, "lesson-1", 0, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_d_min_called_a_similarity_score(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "objective-step-1")
    answer = str(result["answer_markdown"]).replace(
        "weakest-link diversity score", "weakest-link similarity score"
    )
    _set_canonical_answer(snapshot, "objective-step-1", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_fourth_prompt_4_measured_score(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "objective-step-1")
    answer = str(result["answer_markdown"]).replace(
        "- Change in `D_min`: +0.100.\n\n## Meaning",
        "- Change in `D_min`: +0.100.\n- Predicted score: 0.610.\n\n## Meaning",
    )
    _set_canonical_answer(snapshot, "objective-step-1", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_accepts_nonmetric_numeric_panel_ids_in_canonical_answer() -> None:
    terminal = _terminal_objective(1)
    baseline = terminal["baseline"]
    final = terminal["final"]
    attempts = terminal["attempts"]
    assert isinstance(baseline, dict)
    assert isinstance(final, dict)
    assert isinstance(attempts, list)
    assert isinstance(attempts[0], dict)
    action = attempts[0]["selected_swap"]
    assert isinstance(action, dict)
    baseline["selected_ids"] = ["0", "2", "3", "4"]
    baseline["limiting_pairs"] = [["3", "4"]]
    action["swap_id"] = "0->1"
    action["replace_id"] = "0"
    action["replacement_id"] = "1"
    action["resulting_ids"] = ["1", "2", "3", "4"]
    action["limiting_pairs"] = [["3", "4"]]
    attempts[0]["selected_ids"] = action["resulting_ids"]
    attempts[0]["limiting_pairs"] = action["limiting_pairs"]
    final["selected_ids"] = action["resulting_ids"]
    final["limiting_pairs"] = action["limiting_pairs"]
    answer = verifier._canonical_objective_answer(terminal, verifier.PROMPT_MEDIA[3])
    terminal["answer_markdown"] = answer
    verifier._validate_answer(
        answer, terminal, verifier.PROMPT_MEDIA[3], objective=True
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "similarity-hyphen",
        "similarity-spaces",
        "contradictory-definition",
        "extra-score-fact",
        "media-prefix",
        "download-label",
    ],
)
def test_rejects_strengthened_answer_contract_mutations(
    tmp_path: Path, mutation: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "objective-step-1")
    answer = str(result["answer_markdown"])
    meaning = (
        "A larger `D_min` means the least separated pair in the selected panel "
        "became more separated in this fingerprint space."
    )
    if mutation == "similarity-hyphen":
        answer = answer.replace(
            meaning,
            meaning + " This is not a similarity-score.",
        )
    elif mutation == "similarity-spaces":
        answer = answer.replace(
            meaning,
            meaning + " This is not a similarity  score.",
        )
    elif mutation == "contradictory-definition":
        answer = answer.replace(
            "within eight fixed candidates.",
            "within eight fixed candidates. `D_min` is also the maximum "
            "pairwise Tanimoto distance.",
        )
    elif mutation == "extra-score-fact":
        answer = answer.replace(
            meaning,
            meaning + " A diagnostic score was 0.777.",
        )
    elif mutation == "media-prefix":
        answer = answer.replace("\n\nMEDIA:", "\n\nImage: MEDIA:")
    else:
        answer = answer.replace("**Download Results**", "Download Results")
    _set_canonical_answer(snapshot, "objective-step-1", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "mutation",
    [
        "negated-minimum",
        "lower-increases",
        "leading-dot-metric",
        "integer-metric",
        "download-outside-section",
    ],
)
def test_rejects_canonical_p4_section_mutations(tmp_path: Path, mutation: str) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    result = _result_payload(snapshot, "objective-step-1")
    answer = str(result["answer_markdown"])
    meaning = (
        "A larger `D_min` means the least separated pair in the selected panel "
        "became more separated in this fingerprint space."
    )
    download = (
        "The current bundle is in **Download Results** at `workshop/results.zip`."
    )
    if mutation == "negated-minimum":
        answer = answer.replace(
            "`D_min` is the minimum pairwise Tanimoto distance",
            "`D_min` is not the minimum pairwise Tanimoto distance",
        )
    elif mutation == "lower-increases":
        answer = answer.replace(
            meaning,
            "A lower `D_min` means the least separated pair in the selected "
            "panel became more separated in this fingerprint space.",
        )
    elif mutation == "leading-dot-metric":
        answer = answer.replace(meaning, meaning + " A diagnostic metric was .777.")
    elif mutation == "integer-metric":
        answer = answer.replace(meaning, meaning + " A diagnostic metric was 7.")
    else:
        answer = answer.replace(
            "## Meaning\n",
            f"## Meaning\n{download}\n\n",
            1,
        ).replace(
            f"## Image and download location\n{download}",
            "## Image and download location\nSynthetic location.",
            1,
        )
    _set_canonical_answer(snapshot, "objective-step-1", 3, answer)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^answer_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "mutation",
    [
        "pending-current-extra",
        "action-type",
        "terminal-baseline-extra",
        "attempt-null",
        "selected-swap-extra",
    ],
)
def test_rejects_nested_objective_schema_drift(tmp_path: Path, mutation: str) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    pending = _result_payload(snapshot, "objective-start")
    terminal = _result_payload(snapshot, "objective-step-1")
    if mutation == "pending-current-extra":
        current = pending["current"]
        assert isinstance(current, dict)
        current["extra"] = "drift"
        _set_result_payload(snapshot, "objective-start", pending)
    elif mutation == "action-type":
        actions = pending["actions"]
        assert isinstance(actions, list)
        assert isinstance(actions[0], dict)
        actions[0]["resulting_ids"] = "mol-a,mol-b,mol-c,mol-d"
        _set_result_payload(snapshot, "objective-start", pending)
    elif mutation == "terminal-baseline-extra":
        baseline = terminal["baseline"]
        assert isinstance(baseline, dict)
        baseline["extra"] = False
        _set_result_payload(snapshot, "objective-step-1", terminal)
    elif mutation == "attempt-null":
        attempts = terminal["attempts"]
        assert isinstance(attempts, list)
        attempts[0] = None
        _set_result_payload(snapshot, "objective-step-1", terminal)
    else:
        attempts = terminal["attempts"]
        assert isinstance(attempts, list)
        assert isinstance(attempts[0], dict)
        selected_swap = attempts[0]["selected_swap"]
        assert isinstance(selected_swap, dict)
        selected_swap["extra"] = "drift"
        _set_result_payload(snapshot, "objective-step-1", terminal)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize(
    "mutation",
    [
        "terminal-count",
        "pending-transition",
        "target-transition",
        "terminal-transition",
        "baseline-transition",
        "baseline-reason-status",
        "failure-after-limit",
        "below-target-success",
        "out-of-range-target",
    ],
)
def test_rejects_objective_history_or_transition_drift(
    tmp_path: Path, mutation: str
) -> None:
    step_count = {
        "pending-transition": 2,
        "target-transition": 2,
        "baseline-reason-status": 0,
        "failure-after-limit": 3,
    }.get(mutation, 1)
    trajectory, archive, page = _valid_evidence(tmp_path, step_count)
    snapshot = _latest_snapshot(trajectory)
    if mutation == "terminal-count":
        terminal = _result_payload(snapshot, "objective-step-1")
        terminal["attempt_count"] = 0
        terminal["attempts"] = []
        _set_result_payload(snapshot, "objective-step-1", terminal)
    elif mutation == "pending-transition":
        pending = _result_payload(snapshot, "objective-step-1")
        current = pending["current"]
        assert isinstance(current, dict)
        current["score"] = 0.999
        _set_result_payload(snapshot, "objective-step-1", pending)
    elif mutation == "target-transition":
        pending = _result_payload(snapshot, "objective-step-1")
        pending["target_score"] = 0.8
        _set_result_payload(snapshot, "objective-step-1", pending)
    elif mutation == "terminal-transition":
        terminal = _result_payload(snapshot, "objective-step-1")
        final = terminal["final"]
        assert isinstance(final, dict)
        final["selected_ids"] = ["mol-a", "mol-b", "mol-c", "mol-x"]
        _set_result_payload(snapshot, "objective-step-1", terminal)
    elif mutation == "baseline-transition":
        terminal = _result_payload(snapshot, "objective-step-1")
        baseline = terminal["baseline"]
        assert isinstance(baseline, dict)
        baseline["limiting_pairs"] = [["mol-a", "mol-b"]]
        terminal["answer_markdown"] = verifier._canonical_objective_answer(
            terminal, verifier.PROMPT_MEDIA[3]
        )
        _set_canonical_answer(
            snapshot,
            "objective-step-1",
            3,
            str(terminal["answer_markdown"]),
        )
        _set_result_payload(snapshot, "objective-step-1", terminal)
    elif mutation == "baseline-reason-status":
        terminal = _result_payload(snapshot, "objective-start")
        baseline = terminal["baseline"]
        final = terminal["final"]
        assert isinstance(baseline, dict)
        assert isinstance(final, dict)
        baseline["achieved"] = False
        final["achieved"] = False
        terminal["achieved"] = False
        _set_result_payload(snapshot, "objective-start", terminal)
    elif mutation == "failure-after-limit":
        terminal = _result_payload(snapshot, "objective-step-3")
        terminal["termination_reason"] = "objective_provider_failure"
        _set_result_payload(snapshot, "objective-step-3", terminal)
    elif mutation == "below-target-success":
        pending = _result_payload(snapshot, "objective-start")
        actions = pending["actions"]
        assert isinstance(actions, list)
        assert isinstance(actions[0], dict)
        actions[0]["target_status"] = "meets_target"
        terminal = _result_payload(snapshot, "objective-step-1")
        attempts = terminal["attempts"]
        final = terminal["final"]
        assert isinstance(attempts, list)
        assert isinstance(attempts[0], dict)
        assert isinstance(attempts[0]["selected_swap"], dict)
        assert isinstance(final, dict)
        attempts[0]["selected_swap"]["target_status"] = "meets_target"
        attempts[0]["achieved"] = True
        final["achieved"] = True
        terminal["achieved"] = True
        terminal["termination_reason"] = "target_achieved"
        _set_result_payload(snapshot, "objective-start", pending)
        _set_result_payload(snapshot, "objective-step-1", terminal)
    else:
        pending = _result_payload(snapshot, "objective-start")
        terminal = _result_payload(snapshot, "objective-step-1")
        pending["target_score"] = 1.1
        terminal["target_score"] = 1.1
        _set_result_payload(snapshot, "objective-start", pending)
        _set_result_payload(snapshot, "objective-step-1", terminal)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_zero_step_achieved_terminal_requires_baseline_reason(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path, objective_step_count=0)
    snapshot = _latest_snapshot(trajectory)
    terminal = _result_payload(snapshot, "objective-start")
    terminal["termination_reason"] = "target_achieved"
    _set_result_payload(snapshot, "objective-start", terminal)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize("mutation", ["wrong-state", "lower-score"])
def test_rejects_nonmaximum_or_wrong_state_objective_step(
    tmp_path: Path, mutation: str
) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    command_block = _objective_step_block(snapshot)
    state = "wrong" if mutation == "wrong-state" else "state-001"
    swap = "mol-a->mol-e" if mutation == "wrong-state" else "mol-d->mol-h"
    command_block["arguments"] = {
        "command": (
            f"{verifier.RUNNER_PREFIX} objective-step --state-id '{state}' "
            f"--swap-id '{swap}'"
        )
    }
    command_block["partialArgs"] = json.dumps(
        command_block["arguments"], separators=(",", ":")
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


@pytest.mark.parametrize("unsafe_id", ["mol bad", "mol\nbad", "x" * 65])
def test_rejects_unsafe_objective_panel_identifier(unsafe_id: str) -> None:
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier._validated_ids([unsafe_id, "mol-b", "mol-c", "mol-d"])


def test_rejects_unsafe_objective_replacement_identifier() -> None:
    action = _objective_action(1, maximum=True)
    resulting_ids = action["resulting_ids"]
    assert isinstance(resulting_ids, list)
    resulting_ids[0] = "mol injected claim"
    action["replacement_id"] = "mol injected claim"
    action["swap_id"] = "mol-a->mol injected claim"
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier._validate_action(
            action,
            current=_measurement(0),
            target_score=0.75,
        )


def test_rejects_unsafe_objective_state_identifier(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    pending = _result_payload(snapshot, "objective-start")
    pending["state_id"] = "state injected claim"
    _set_result_payload(snapshot, "objective-start", pending)
    command_block = _objective_step_block(snapshot)
    command_block["arguments"] = {
        "command": (
            f"{verifier.RUNNER_PREFIX} objective-step "
            "--state-id 'state injected claim' --swap-id 'mol-a->mol-e'"
        )
    }
    command_block["partialArgs"] = json.dumps(
        command_block["arguments"], separators=(",", ":")
    )
    terminal = _result_payload(snapshot, "objective-step-1")
    attempts = terminal["attempts"]
    assert isinstance(attempts, list)
    assert isinstance(attempts[0], dict)
    attempts[0]["state_id"] = "state injected claim"
    _set_result_payload(snapshot, "objective-step-1", terminal)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^objective_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_duplicate_call_id_across_turns(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    second_call = _user_indices(snapshot)[1] + 1
    block = snapshot[second_call]["content"]
    assert isinstance(block, list)
    assert isinstance(block[0], dict)
    block[0]["id"] = "lesson-1"
    second_result = _find_message(snapshot, "lesson-2")
    snapshot[second_result]["toolCallId"] = "lesson-1"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="^command_contract$"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_load_prompt_contracts_rejects_extra_region_text(tmp_path: Path) -> None:
    page = tmp_path / "pinned.md"
    _write_pinned_page(page)
    changed = tmp_path / "page.md"
    changed.write_bytes(
        page.read_bytes().replace(b"\n~~~text\n", b"\nextra\n~~~text\n", 1)
    )
    with pytest.raises(verifier.VerificationError, match="^prompt_contract$"):
        verifier.load_prompt_contracts(changed)


def test_load_prompt_contracts_rejects_symlink(tmp_path: Path) -> None:
    page = tmp_path / "pinned.md"
    _write_pinned_page(page)
    link = tmp_path / "page.md"
    link.symlink_to(page)
    with pytest.raises(verifier.VerificationError, match="^prompt_contract$"):
        verifier.load_prompt_contracts(link)


def test_load_messages_snapshot_selects_last_snapshot(tmp_path: Path) -> None:
    prompts = _fixture_prompts()
    expected = _valid_messages(prompts)
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        json.dumps(_event({"synthetic": True}, seq=1))
        + "\n"
        + json.dumps(_event({"messagesSnapshot": expected}, seq=2))
        + "\n",
        encoding="utf-8",
    )
    assert verifier.load_messages_snapshot(path) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-envelope-key",
        "missing-envelope-key",
        "schema-version",
        "trace-schema",
        "negative-sequence",
        "string-sequence",
        "empty-identity",
        "snapshot-data-extra",
    ],
)
def test_load_messages_snapshot_rejects_event_envelope_drift(
    tmp_path: Path, mutation: str
) -> None:
    event = _event({"messagesSnapshot": _valid_messages(_fixture_prompts())})
    if mutation == "extra-envelope-key":
        event["extra"] = "drift"
    elif mutation == "missing-envelope-key":
        del event["modelApi"]
    elif mutation == "schema-version":
        event["schemaVersion"] = 2
    elif mutation == "trace-schema":
        event["traceSchema"] = 1
    elif mutation == "negative-sequence":
        event["sourceSeq"] = -1
    elif mutation == "string-sequence":
        event["seq"] = "1"
    elif mutation == "empty-identity":
        event["sessionId"] = ""
    else:
        data = event["data"]
        assert isinstance(data, dict)
        data["extra"] = False
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="^invalid_evidence$"):
        verifier.load_messages_snapshot(path)


@pytest.mark.parametrize(
    ("message_field", "envelope_field", "mutate_envelope"),
    [
        ("api", "modelApi", False),
        ("model", "modelId", False),
        ("provider", "provider", False),
        ("model", "modelId", True),
    ],
)
def test_load_messages_snapshot_rejects_assistant_envelope_identity_drift(
    tmp_path: Path,
    message_field: str,
    envelope_field: str,
    mutate_envelope: bool,
) -> None:
    trajectory, _, _ = _valid_evidence(tmp_path)
    loaded: object = json.loads(trajectory.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    if mutate_envelope:
        loaded[envelope_field] = "drifted-envelope-identity"
    else:
        data = loaded["data"]
        assert isinstance(data, dict)
        snapshot = data["messagesSnapshot"]
        assert isinstance(snapshot, list)
        assistant = next(
            message
            for message in snapshot
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
        assistant[message_field] = "drifted-assistant-identity"
    trajectory.write_text(json.dumps(loaded) + "\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="^invalid_evidence$"):
        verifier.load_messages_snapshot(trajectory)


@pytest.mark.parametrize(
    "contents",
    [
        '{"data":{"messagesSnapshot":[]},"data":{}}\n',
        "[]\n",
        '{"data":{"messagesSnapshot":[]}}\n'
        '{"data":{"messagesSnapshot":[{"role":"user"}]}}\n',
    ],
)
def test_load_messages_snapshot_rejects_invalid_or_open_events(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="^invalid_evidence$"):
        verifier.load_messages_snapshot(path)


def test_rejects_archive_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, names=APPROVED_ZIP_MEMBERS + ("../escape",))
    _assert_archive_rejected(archive)


def test_rejects_duplicate_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_archive(archive, names=APPROVED_ZIP_MEMBERS + ("README.md",))
    _assert_archive_rejected(archive)


def test_rejects_symlink_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, modes={APPROVED_ZIP_MEMBERS[0]: stat.S_IFLNK | 0o777})
    _assert_archive_rejected(archive)


def test_rejects_encrypted_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    _patch_zip_flag(archive, 1)
    _assert_archive_rejected(archive)


def test_rejects_archive_comment(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, archive_comment=b"synthetic comment")
    _assert_archive_rejected(archive)


def test_rejects_archive_entry_comment(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, entry_comments={APPROVED_ZIP_MEMBERS[0]: b"comment"})
    _assert_archive_rejected(archive)


def test_rejects_archive_extra_field(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, extras={APPROVED_ZIP_MEMBERS[0]: b"\x01\x00\x00\x00"})
    _assert_archive_rejected(archive)


def test_rejects_wrong_archive_timestamp(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, timestamps={APPROVED_ZIP_MEMBERS[0]: (1980, 1, 2, 0, 0, 0)})
    _assert_archive_rejected(archive)


def test_rejects_wrong_local_header_dos_date(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    _patch_local_dos_date(archive)
    _assert_archive_rejected(archive)


def test_rejects_unsupported_archive_compression(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, compressions={APPROVED_ZIP_MEMBERS[0]: zipfile.ZIP_STORED})
    _assert_archive_rejected(archive)


@pytest.mark.parametrize(
    "mutation", ["local-needed", "central-needed", "central-made-by"]
)
def test_rejects_invalid_or_mismatched_zip_versions(
    tmp_path: Path, mutation: str
) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    _patch_zip_version(archive, mutation)
    _assert_archive_rejected(archive)


def test_rejects_missing_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, names=APPROVED_ZIP_MEMBERS[:-1])
    _assert_archive_rejected(archive)


def test_rejects_extra_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, names=APPROVED_ZIP_MEMBERS + ("extra.txt",))
    _assert_archive_rejected(archive)


def test_rejects_corrupt_archive_crc(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    _patch_central_crc(archive)
    _assert_archive_rejected(archive)


def test_rejects_forged_small_size_for_large_raw_deflate_stream(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(
        archive,
        sizes={APPROVED_ZIP_MEMBERS[0]: 9 * 1024 * 1024},
    )
    _forge_one_byte_declared_member(archive)
    _assert_archive_rejected(archive)


def test_rejects_trailing_archive_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    archive.write_bytes(archive.read_bytes() + b"trailing")
    _assert_archive_rejected(archive)


def test_rejects_prepended_archive_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_valid_archive(archive)
    archive.write_bytes(b"prepended" + archive.read_bytes())
    _assert_archive_rejected(archive)


def test_rejects_invalid_required_png(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, contents={APPROVED_CHAT_PNGS[0]: b"not a png"})
    _assert_archive_rejected(archive)


def test_rejects_zero_width_required_png(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, contents={APPROVED_CHAT_PNGS[0]: _png(width=0)})
    _assert_archive_rejected(archive)


@pytest.mark.parametrize(
    "mutation",
    [
        "ihdr-crc",
        "idat-crc",
        "illegal-fields",
        "truncated",
        "missing-iend",
        "duplicate-iend",
        "invalid-zlib",
        "wrong-scanline-length",
        "invalid-filter",
    ],
)
def test_rejects_malformed_png_chunk_stream(tmp_path: Path, mutation: str) -> None:
    contents = _png()
    if mutation == "ihdr-crc":
        contents = _corrupt_png_crc(contents, b"IHDR")
    elif mutation == "idat-crc":
        contents = _corrupt_png_crc(contents, b"IDAT")
    elif mutation == "illegal-fields":
        _, payload_start, _ = _png_chunk_bounds(contents, b"IHDR")
        ihdr = bytearray(contents[payload_start : payload_start + 13])
        ihdr[9] = 3
        contents = _replace_png_chunk(contents, b"IHDR", bytes(ihdr))
    elif mutation == "truncated":
        contents = contents[:-1]
    elif mutation == "missing-iend":
        iend_start, _, _ = _png_chunk_bounds(contents, b"IEND")
        contents = contents[:iend_start]
    elif mutation == "duplicate-iend":
        contents += _png_chunk(b"IEND", b"")
    elif mutation == "invalid-zlib":
        contents = _replace_png_chunk(contents, b"IDAT", b"not-zlib")
    elif mutation == "wrong-scanline-length":
        contents = _replace_png_chunk(contents, b"IDAT", zlib.compress(b"\x00\x00"))
    else:
        contents = _replace_png_chunk(
            contents,
            b"IDAT",
            zlib.compress(b"\x05\x00\x00\x00\xff"),
        )
    archive = tmp_path / "results.zip"
    _write_archive(archive, contents={APPROVED_CHAT_PNGS[0]: contents})
    _assert_archive_rejected(archive)


def test_rejects_member_larger_than_eight_mib(tmp_path: Path) -> None:
    archive = tmp_path / "results.zip"
    _write_archive(archive, sizes={APPROVED_ZIP_MEMBERS[0]: 8 * 1024 * 1024 + 1})
    _assert_archive_rejected(archive)


def test_rejects_total_expansion_larger_than_thirty_two_mib(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "results.zip"
    sizes = {name: 8 * 1024 * 1024 for name in APPROVED_ZIP_MEMBERS[:4]}
    _write_archive(archive, sizes=sizes)
    _assert_archive_rejected(archive)


def test_cli_success_prints_only_closed_receipt(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trajectory",
            str(trajectory),
            "--results-zip",
            str(archive),
            "--page",
            str(page),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    receipt = json.loads(completed.stdout)
    assert set(receipt) == {
        "schema_version",
        "status",
        "prompt_count",
        "exec_call_count",
        "objective_step_count",
        "archive_sha256",
        "archive_size",
        "required_png_count",
    }
    assert (
        completed.stdout
        == json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_cli_failure_prints_only_safe_issue_code(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    rejected = tmp_path / "TOP_SECRET_TOKEN.jsonl"
    rejected.write_text("not-json\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trajectory",
            str(rejected),
            "--results-zip",
            str(archive),
            "--page",
            str(page),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert completed.stdout == (
        '{"issue_code":"invalid_evidence","schema_version":1,"status":"fail"}\n'
    )
    combined = completed.stdout + completed.stderr
    for forbidden in (
        "TOP_SECRET_TOKEN",
        "Question:",
        "answer_markdown",
        "acs_workshop_runner.py",
        "http",
        "Traceback",
        "JSONDecodeError",
    ):
        assert forbidden not in combined
    assert trajectory.exists()


def test_cli_argument_failure_does_not_echo_rejected_value() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--unknown", "TOP_SECRET_TOKEN"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert completed.stdout == (
        '{"issue_code":"invalid_evidence","schema_version":1,"status":"fail"}\n'
    )
    assert "TOP_SECRET_TOKEN" not in completed.stdout


@pytest.mark.parametrize("arguments", [[], ["--help"], ["-h"]])
def test_cli_non_verification_invocation_emits_only_closed_failure(
    arguments: list[str],
) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert completed.stderr == ""
    assert completed.stdout == (
        '{"issue_code":"invalid_evidence","schema_version":1,"status":"fail"}\n'
    )
