#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Final, Mapping, NamedTuple, NoReturn, Sequence


SANDBOX: Final = "acs-chemistry-agent"
WORKSPACE: Final = Path("/sandbox/.openclaw/workspace")
RESULTS_ZIP: Final = WORKSPACE / "outputs" / "workshop" / "results.zip"
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
PROMPT_MEDIA: Final = (
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "01-inspection/library_preview.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "06-mmff94/optimized_structures.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png",
)
AGENT_TIMEOUT_SECONDS: Final = 600
PROCESS_TIMEOUT_SECONDS: Final = 660
TRANSFER_TIMEOUT_SECONDS: Final = 120
MAX_PAGE_BYTES: Final = 256 * 1024
MAX_TRAJECTORY_BYTES: Final = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES: Final = 40 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES: Final = 16 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES: Final = 128 * 1024 * 1024
FAILURE_EXIT: Final = 70
TIMEOUT_EXIT: Final = 75
_CREDENTIAL_KEY: Final = re.compile(
    rb"(?i)(?:nvidia_inference_api_key|gateway[_-]?token|api[_-]?key|"
    rb"access[_-]?token|authorization|password|client[_-]?secret)"
    rb"[\"']?\s*[:=]\s*[\"']?[^\s\"']+"
)
_CREDENTIAL_NAMES: Final = {
    "access_token",
    "accesstoken",
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "clientsecret",
    "gateway_token",
    "gatewaytoken",
    "nvidia_inference_api_key",
    "nvidiainferenceapikey",
    "password",
}


class QAError(Exception):
    """A closed, non-timeout QA failure."""


class QATimeout(QAError):
    """A prompt timed out and the entire session must be discarded."""


VerifyAcceptance = Callable[[Path, Path, Path], dict[str, int | str]]


class _VerifierBoundary(NamedTuple):
    error: type[Exception]
    required_members: tuple[str, ...]
    verify: VerifyAcceptance


class _ClosedParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        raise QAError


def _canonical_json(payload: Mapping[str, int | str]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _emit_failure(code: str) -> None:
    sys.stdout.buffer.write(
        _canonical_json({"code": code, "schema_version": 1, "status": "fail"})
    )


def _safe_lstat(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise QAError
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if index != len(parts) - 1:
                raise QAError from None
            raise
        if stat.S_ISLNK(mode):
            raise QAError
    return os.lstat(path)


def _read_regular(path: Path, limit: int) -> bytes:
    try:
        before = _safe_lstat(path)
    except (FileNotFoundError, OSError) as error:
        raise QAError from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or before.st_size > limit
    ):
        raise QAError
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise QAError
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise QAError
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise QAError
        return b"".join(chunks)
    except OSError as error:
        raise QAError from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_output_directory(path: Path) -> None:
    try:
        metadata = _safe_lstat(path)
    except (FileNotFoundError, OSError) as error:
        raise QAError from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise QAError


def _parse_session_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise QAError from error
    if str(parsed) != value or parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise QAError
    return value


def load_prompts(page: Path) -> tuple[str, ...]:
    raw = _read_regular(page, MAX_PAGE_BYTES)
    try:
        source = raw.decode("utf-8")
    except UnicodeError as error:
        raise QAError from error
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
    if observed_markers != expected_markers or source.count("<!-- ACS_PROMPT:") != 8:
        raise QAError
    prompts: list[str] = []
    prior_end = -1
    for prompt_id, expected_digest, media_line in zip(
        PROMPT_IDS, PROMPT_SHA256, PROMPT_MEDIA, strict=True
    ):
        begin = f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->"
        end = f"<!-- ACS_PROMPT:{prompt_id}:END -->"
        begin_index = source.find(begin)
        end_index = source.find(end, begin_index + len(begin))
        if begin_index <= prior_end or end_index <= begin_index:
            raise QAError
        region = source[begin_index + len(begin) : end_index]
        opening = "~~~text\n"
        closing = "\n~~~\n"
        if region.count(opening) != 1 or region.count(closing) != 1:
            raise QAError
        opening_index = region.find(opening)
        prompt_start = opening_index + len(opening)
        prompt_end = region.find(closing, prompt_start)
        if (
            opening_index < 0
            or prompt_end < prompt_start
            or region[:opening_index].strip()
            or region[prompt_end + len(closing) :].strip()
        ):
            raise QAError
        prompt = region[prompt_start:prompt_end]
        if (
            not prompt.endswith(media_line)
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != expected_digest
        ):
            raise QAError
        prompts.append(prompt)
        prior_end = end_index
    return tuple(prompts)


def _load_verifier(page: Path, prompts: tuple[str, ...]) -> _VerifierBoundary:
    try:
        from verify_acs_openclaw_trajectory import (
            REQUIRED_ZIP_MEMBERS,
            VerificationError,
            load_prompt_contracts,
            verify_acceptance,
        )
    except ImportError as error:
        raise QAError from error
    parameters = inspect.signature(verify_acceptance).parameters
    if tuple(parameters) != ("trajectory_path", "results_zip_path", "page_path"):
        raise QAError
    if not isinstance(VerificationError, type) or not issubclass(
        VerificationError, Exception
    ):
        raise QAError
    if (
        type(REQUIRED_ZIP_MEMBERS) is not tuple
        or len(REQUIRED_ZIP_MEMBERS) != 34
        or len(set(REQUIRED_ZIP_MEMBERS)) != len(REQUIRED_ZIP_MEMBERS)
        or any(type(name) is not str or not name for name in REQUIRED_ZIP_MEMBERS)
    ):
        raise QAError
    try:
        contracts = load_prompt_contracts(page)
    except (VerificationError, OSError) as error:
        raise QAError from error
    expected = tuple(
        (prompt_id, prompt, digest)
        for prompt_id, prompt, digest in zip(
            PROMPT_IDS, prompts, PROMPT_SHA256, strict=True
        )
    )
    if contracts != expected:
        raise QAError
    return _VerifierBoundary(
        error=VerificationError,
        required_members=REQUIRED_ZIP_MEMBERS,
        verify=verify_acceptance,
    )


def _run_quiet(
    command: Sequence[str],
    *,
    timeout: int | None = None,
    provider_timeout: bool = False,
) -> int:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.TimeoutExpired) and provider_timeout:
            raise QATimeout from error
        raise QAError from error
    return completed.returncode


def _submit_prompts(
    nemoclaw: str, sandbox: str, session_id: str, prompts: Sequence[str]
) -> None:
    for prompt in prompts:
        status = _run_quiet(
            (
                nemoclaw,
                sandbox,
                "agent",
                "--session-id",
                session_id,
                "--json",
                "--timeout",
                str(AGENT_TIMEOUT_SECONDS),
                "-m",
                prompt,
            ),
            timeout=PROCESS_TIMEOUT_SECONDS,
            provider_timeout=True,
        )
        if status == 124:
            raise QATimeout
        if status != 0:
            raise QAError


def _bounded_entries(directory: Path, expected_count: int) -> tuple[Path, ...]:
    entries: list[Path] = []
    iterator = directory.iterdir()
    for _ in range(expected_count + 1):
        try:
            entries.append(next(iterator))
        except StopIteration:
            break
    return tuple(entries)


def _select_exact_regular(directory: Path, expected: Path) -> Path:
    try:
        metadata = _safe_lstat(directory)
    except (FileNotFoundError, OSError) as error:
        raise QAError from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise QAError
    try:
        entries = _bounded_entries(directory, 1)
        candidate_metadata = os.lstat(expected)
    except OSError as error:
        raise QAError from error
    if (
        entries != (expected,)
        or not stat.S_ISREG(candidate_metadata.st_mode)
        or candidate_metadata.st_nlink != 1
        or candidate_metadata.st_uid != os.getuid()
    ):
        raise QAError
    return expected


def _normalize_export_directory(directory: Path) -> None:
    try:
        parent = _safe_lstat(directory.parent)
        before = _safe_lstat(directory)
    except (FileNotFoundError, OSError) as error:
        raise QAError from error
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_IMODE(parent.st_mode) != 0o700
        or parent.st_uid != os.getuid()
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) not in {0o700, 0o775}
        or before.st_uid != os.getuid()
    ):
        raise QAError
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(directory, flags)
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
        ) != (before.st_dev, before.st_ino, before.st_mode, before.st_uid):
            raise QAError
        os.fchmod(descriptor, 0o700)
        normalized = os.fstat(descriptor)
        if (
            normalized.st_dev,
            normalized.st_ino,
            stat.S_IMODE(normalized.st_mode),
            normalized.st_uid,
        ) != (before.st_dev, before.st_ino, 0o700, os.getuid()):
            raise QAError
    except OSError as error:
        raise QAError from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    final = _safe_lstat(directory)
    if (
        final.st_dev,
        final.st_ino,
        stat.S_IMODE(final.st_mode),
        final.st_uid,
    ) != (before.st_dev, before.st_ino, 0o700, os.getuid()):
        raise QAError


def _select_exact_trajectory(directory: Path, session_id: str) -> Path:
    _normalize_export_directory(directory)
    session = directory / f"{session_id}.jsonl"
    trajectory = directory / f"{session_id}.trajectory.jsonl"
    try:
        entries = _bounded_entries(directory, 2)
        metadata = (os.lstat(session), os.lstat(trajectory))
    except OSError as error:
        raise QAError from error
    if len(entries) != 2 or set(entries) != {session, trajectory}:
        raise QAError
    for observed in metadata:
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_nlink != 1
            or observed.st_uid != os.getuid()
        ):
            raise QAError
    return trajectory


def _closed_json(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise QAError
        result[key] = value
    return result


def _normalized_credential_name(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _reject_json_credential_keys(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str or _normalized_credential_name(key) in _CREDENTIAL_NAMES:
                raise QAError
            _reject_json_credential_keys(child)
    elif type(value) is list:
        for child in value:
            _reject_json_credential_keys(child)


def _scan_for_credentials(
    trajectory: bytes,
    archive_bytes: bytes,
    required_members: tuple[str, ...],
) -> None:
    if _CREDENTIAL_KEY.search(trajectory):
        raise QAError
    if _CREDENTIAL_KEY.search(archive_bytes):
        raise QAError
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            for name in required_members:
                content = archive.read(name)
                if _CREDENTIAL_KEY.search(content):
                    raise QAError
                if name.endswith(".json"):
                    parsed = json.loads(
                        content.decode("utf-8"),
                        object_pairs_hook=_closed_json,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            QAError(value)
                        ),
                    )
                    _reject_json_credential_keys(parsed)
    except QAError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise QAError from error


def _write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
    if path.exists():
        raise QAError
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        published = False
    except OSError as error:
        if published:
            try:
                path.unlink()
            except OSError:
                pass
        raise QAError from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_evidence_bytes(
    trajectory: bytes,
    archive: bytes,
    directory: Path,
    page: Path,
    verifier: _VerifierBoundary,
) -> dict[str, int | str]:
    trajectory_path = directory / ".trajectory-validation.jsonl"
    archive_path = directory / ".archive-validation.zip"
    _write_atomic(trajectory_path, trajectory)
    _write_atomic(archive_path, archive)
    try:
        receipt = verifier.verify(trajectory_path, archive_path, page)
    except (verifier.error, OSError) as error:
        raise QAError from error
    expected_keys = {
        "archive_sha256",
        "archive_size",
        "exec_call_count",
        "objective_step_count",
        "prompt_count",
        "required_png_count",
        "schema_version",
        "status",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "pass"
        or receipt.get("prompt_count") != 4
        or receipt.get("required_png_count") != 4
        or receipt.get("archive_sha256") != hashlib.sha256(archive).hexdigest()
        or receipt.get("archive_size") != len(archive)
        or any(
            type(receipt.get(key)) is not int or int(receipt[key]) < 0
            for key in (
                "exec_call_count",
                "objective_step_count",
                "prompt_count",
                "required_png_count",
                "archive_size",
            )
        )
    ):
        raise QAError
    return receipt


def _cleanup_stage(stage: Path) -> None:
    try:
        metadata = os.lstat(stage)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        shutil.rmtree(stage)


def _copy_evidence(
    nemoclaw: str,
    openshell: str,
    sandbox: str,
    session_id: str,
    output_dir: Path,
    page: Path,
    verifier: _VerifierBoundary,
) -> dict[str, int | str]:
    stage = output_dir / ".qa-export-pending"
    if stage.exists() or stage.is_symlink():
        raise QAError
    stage.mkdir(mode=0o700)
    export_dir = stage / "session-export"
    archive_dir = stage / "archive-export"
    archive_dir.mkdir(mode=0o700)
    try:
        status = _run_quiet(
            (
                nemoclaw,
                sandbox,
                "sessions",
                "export",
                f"agent:main:{session_id}",
                "--include-trajectory",
                "--out",
                str(export_dir),
            ),
            timeout=TRANSFER_TIMEOUT_SECONDS,
        )
        if status != 0:
            raise QAError
        trajectory_path = _select_exact_trajectory(export_dir, session_id)
        status = _run_quiet(
            (
                openshell,
                "sandbox",
                "download",
                sandbox,
                str(RESULTS_ZIP),
                str(archive_dir),
            ),
            timeout=TRANSFER_TIMEOUT_SECONDS,
        )
        if status != 0:
            raise QAError
        archive_path = _select_exact_regular(archive_dir, archive_dir / "results.zip")
        trajectory = _read_regular(trajectory_path, MAX_TRAJECTORY_BYTES)
        archive = _read_regular(archive_path, MAX_ARCHIVE_BYTES)
        verified = _verify_evidence_bytes(
            trajectory, archive, stage, page, verifier
        )
        _scan_for_credentials(trajectory, archive, verifier.required_members)
        trajectory_destination = output_dir / f"{session_id}.trajectory.jsonl"
        archive_destination = output_dir / "results.zip"
        acceptance = output_dir / "acceptance.json"
        if any(
            path.exists() or path.is_symlink()
            for path in (
                trajectory_destination,
                archive_destination,
                acceptance,
            )
        ):
            raise QAError
        _write_atomic(trajectory_destination, trajectory)
        try:
            _write_atomic(archive_destination, archive)
            receipt: dict[str, int | str] = {
                "prompt_count": 4,
                "exec_call_count": verified["exec_call_count"],
                "objective_step_count": verified["objective_step_count"],
                "required_png_count": verified["required_png_count"],
                "results_zip_sha256": hashlib.sha256(archive).hexdigest(),
                "results_zip_size": len(archive),
                "schema_version": 1,
                "status": "pass",
                "trajectory_sha256": hashlib.sha256(trajectory).hexdigest(),
                "trajectory_size": len(trajectory),
            }
            _write_atomic(acceptance, _canonical_json(receipt))
        except QAError:
            trajectory_destination.unlink(missing_ok=True)
            archive_destination.unlink(missing_ok=True)
            acceptance.unlink(missing_ok=True)
            raise
        return receipt
    finally:
        _cleanup_stage(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = _ClosedParser(add_help=False)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--page", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sandbox", required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, int | str]:
    session_id = _parse_session_id(str(arguments.session_id))
    sandbox = str(arguments.sandbox)
    if sandbox != SANDBOX:
        raise QAError
    page = Path(arguments.page)
    output_dir = Path(arguments.output_dir)
    _validate_output_directory(output_dir)
    destinations = (
        output_dir / f"{session_id}.trajectory.jsonl",
        output_dir / "results.zip",
        output_dir / "acceptance.json",
        output_dir / ".qa-export-pending",
        output_dir / f".{session_id}.trajectory.jsonl.tmp",
        output_dir / ".results.zip.tmp",
        output_dir / ".acceptance.json.tmp",
    )
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise QAError
    prompts = load_prompts(page)
    verifier = _load_verifier(page, prompts)
    nemoclaw = shutil.which("nemoclaw")
    openshell = shutil.which("openshell")
    if nemoclaw is None or openshell is None:
        raise QAError
    _submit_prompts(nemoclaw, sandbox, session_id, prompts)
    return _copy_evidence(
        nemoclaw, openshell, sandbox, session_id, output_dir, page, verifier
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        receipt = run(arguments)
    except QATimeout:
        _emit_failure("timeout")
        return TIMEOUT_EXIT
    except (Exception, KeyboardInterrupt, SystemExit):
        _emit_failure("qa_failed")
        return FAILURE_EXIT
    sys.stdout.buffer.write(_canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
