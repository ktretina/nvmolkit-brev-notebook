#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path


EXPECTED_SANDBOX = "acs-chemistry-agent"
HEX = re.compile(r"[0-9a-f]{64}\Z")
INVOCATION = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
MODES = {"apply", "reset-between-qa", "rollback"}
MAX_RECEIPT_BYTES = 16 * 1024
MANAGED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
REQUEST_KEYS = {
    "bundle_dir",
    "mode",
    "patch_script",
    "sandbox",
    "schema_version",
    "state_dir",
}


class ControllerError(Exception):
    pass


def canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def closed_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ControllerError
        payload[key] = value
    return payload


def decode_canonical(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        raise ControllerError
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=closed_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ControllerError(value)),
        )
    except (ControllerError, UnicodeError, json.JSONDecodeError) as error:
        raise ControllerError from error
    if type(payload) is not dict or canonical(payload) != raw:
        raise ControllerError
    return payload


def emit(payload: object) -> None:
    sys.stdout.buffer.write(canonical(payload))
    sys.stdout.buffer.flush()


def preflight_failure() -> int:
    emit(
        {
            "code": "preflight_failed",
            "patch_started": False,
            "schema_version": 1,
            "status": "fail",
        }
    )
    return 70


def parse_arguments(arguments: list[str]) -> tuple[str, dict[str, str]]:
    full_names = {
        "--bundle-dir": "bundle_dir",
        "--invocation-id": "invocation_id",
        "--invocation-root": "invocation_root",
        "--mode": "mode",
        "--patch-script": "patch_script",
        "--sandbox": "sandbox",
        "--state-dir": "state_dir",
    }
    interface = "full"
    names = full_names
    if arguments[:1] == ["--receipt"]:
        interface = "receipt"
        names = {
            "--invocation-id": "invocation_id",
            "--invocation-root": "invocation_root",
        }
        arguments = arguments[1:]
    parsed: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag not in names or index + 1 >= len(arguments):
            raise ControllerError
        name = names[flag]
        value = arguments[index + 1]
        if name in parsed or not value:
            raise ControllerError
        parsed[name] = value
        index += 2
    if set(parsed) != set(names.values()):
        raise ControllerError
    return interface, parsed


def safe_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    if not path.is_absolute():
        raise ControllerError
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise ControllerError
        if stat.S_ISLNK(metadata.st_mode):
            raise ControllerError
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ControllerError


def private_directory(path: Path) -> Path:
    safe_components(path)
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ControllerError
    return path.resolve(strict=True)


def normalized_state(path: Path) -> Path:
    safe_components(path, allow_missing_leaf=True)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        parent = path.parent.resolve(strict=True)
        return parent / path.name
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ControllerError
    return path.resolve(strict=True)


def owned_patch(path: Path) -> Path:
    safe_components(path)
    metadata = os.lstat(path)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or mode & 0o022
        or not mode & stat.S_IXUSR
    ):
        raise ControllerError
    return path.resolve(strict=True)


def invocation_boundary(parsed: dict[str, str]) -> tuple[Path, str]:
    invocation_id = parsed["invocation_id"]
    if INVOCATION.fullmatch(invocation_id) is None:
        raise ControllerError
    invocation_root = private_directory(Path(parsed["invocation_root"]))
    return invocation_root, invocation_id


def bound_path(value: str) -> str:
    if not Path(value).is_absolute():
        return value
    return os.path.normpath(value)


def bind_request(parsed: dict[str, str]) -> dict[str, object]:
    request: dict[str, object] = {
        "bundle_dir": bound_path(parsed["bundle_dir"]),
        "mode": parsed["mode"],
        "patch_script": bound_path(parsed["patch_script"]),
        "sandbox": parsed["sandbox"],
        "schema_version": 1,
        "state_dir": bound_path(parsed["state_dir"]),
    }
    return request


def execution_request(request: dict[str, object]) -> dict[str, object]:
    if request["mode"] not in MODES or request["sandbox"] != EXPECTED_SANDBOX:
        raise ControllerError
    patch_script = owned_patch(Path(str(request["patch_script"])))
    bundle_dir = private_directory(Path(str(request["bundle_dir"])))
    state_dir = normalized_state(Path(str(request["state_dir"])))
    return {
        **request,
        "bundle_dir": str(bundle_dir),
        "patch_script": str(patch_script),
        "state_dir": str(state_dir),
    }


def regular_bytes(path: Path) -> bytes:
    safe_components(path)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size > MAX_RECEIPT_BYTES
    ):
        raise ControllerError
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != identity:
            raise ControllerError
        raw = os.read(descriptor, MAX_RECEIPT_BYTES + 1)
        if len(raw) != before.st_size:
            raise ControllerError
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
        ) != identity:
            raise ControllerError
        return raw
    finally:
        os.close(descriptor)


def atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp-" + str(uuid.uuid4()))
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise ControllerError
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if regular_bytes(temporary) != raw:
            raise ControllerError
        os.replace(temporary, path)
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_request(payload: dict[str, object]) -> None:
    if (
        set(payload) != REQUEST_KEYS
        or payload["schema_version"] != 1
        or any(
            type(payload[key]) is not str or not payload[key]
            for key in ("bundle_dir", "mode", "patch_script", "sandbox", "state_dir")
        )
    ):
        raise ControllerError


def validate_patch_receipt(
    payload: dict[str, object], mode: str, returncode: int
) -> None:
    if payload.get("schema_version") != 1 or payload.get("main_session_touched") is not False:
        raise ControllerError
    if payload.get("status") == "fail":
        if (
            set(payload)
            != {
                "code",
                "main_session_touched",
                "rollback",
                "schema_version",
                "status",
            }
            or returncode == 0
            or type(payload["code"]) is not str
            or re.fullmatch(r"[a-z0-9_]{1,64}", payload["code"]) is None
            or type(payload["rollback"]) is not bool
        ):
            raise ControllerError
        return
    if payload.get("status") != "pass" or returncode != 0 or payload.get("mode") != mode:
        raise ControllerError
    if mode == "apply":
        if (
            set(payload)
            != {
                "loop_detection",
                "main_session_touched",
                "manifest_files",
                "mode",
                "rollback_ready",
                "runner_hash",
                "schema_version",
                "status",
                "tools_hash",
                "workshop_reset",
            }
            or payload["loop_detection"] is not True
            or payload["manifest_files"] != 6
            or payload["rollback_ready"] is not True
            or payload["workshop_reset"] is not True
            or type(payload["runner_hash"]) is not str
            or type(payload["tools_hash"]) is not str
            or HEX.fullmatch(payload["runner_hash"]) is None
            or HEX.fullmatch(payload["tools_hash"]) is None
        ):
            raise ControllerError
    elif mode == "rollback":
        if (
            set(payload)
            != {
                "idempotent",
                "main_session_touched",
                "mode",
                "restored",
                "schema_version",
                "status",
            }
            or type(payload["idempotent"]) is not bool
            or payload["restored"] is not True
        ):
            raise ControllerError
    elif (
        set(payload)
        != {
            "loop_detection",
            "main_session_touched",
            "mode",
            "schema_version",
            "status",
            "workshop_reset",
        }
        or payload["loop_detection"] is not True
        or payload["workshop_reset"] is not True
    ):
        raise ControllerError


def validate_terminal(
    payload: dict[str, object], invocation_id: str, request_mode: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("invocation_id") != invocation_id:
        raise ControllerError
    if set(payload) == {
        "inner_exit_code",
        "invocation_id",
        "patch_receipt",
        "schema_version",
        "status",
    }:
        if (
            type(payload["inner_exit_code"]) is not int
            or type(payload["patch_receipt"]) is not dict
            or payload["status"] not in {"fail", "pass"}
            or payload["patch_receipt"].get("status") != payload["status"]
        ):
            raise ControllerError
        validate_patch_receipt(
            payload["patch_receipt"], request_mode, payload["inner_exit_code"]
        )
        return
    if set(payload) == {
        "code",
        "inner_exit_code",
        "invocation_id",
        "patch_receipt",
        "schema_version",
        "status",
    }:
        if (
            payload["code"] != "invalid_patch_result"
            or type(payload["inner_exit_code"]) is not int
            or payload["patch_receipt"] is not None
            or payload["status"] != "fail"
        ):
            raise ControllerError
        return
    if set(payload) == {
        "code",
        "inner_exit_code",
        "invocation_id",
        "patch_receipt",
        "patch_started",
        "schema_version",
        "status",
    }:
        if (
            payload["code"] != "preflight_failed"
            or payload["inner_exit_code"] is not None
            or payload["patch_receipt"] is not None
            or payload["patch_started"] is not False
            or payload["status"] != "fail"
        ):
            raise ControllerError
        return
    raise ControllerError


def progress_response(invocation_id: str) -> int:
    emit(
        {
            "invocation_id": invocation_id,
            "schema_version": 1,
            "status": "in_progress",
        }
    )
    return 0


def invalid_record_response(invocation_id: str) -> int:
    emit(
        {
            "code": "invocation_record_invalid",
            "invocation_id": invocation_id,
            "schema_version": 1,
            "status": "fail",
        }
    )
    return 0


def receipt_response(
    invocation_dir: Path,
    invocation_id: str,
    expected_request: dict[str, object] | None,
) -> int:
    try:
        private_directory(invocation_dir)
    except (ControllerError, OSError):
        return invalid_record_response(invocation_id)
    request_path = invocation_dir / "request.json"
    try:
        os.lstat(request_path)
    except FileNotFoundError:
        return progress_response(invocation_id)
    try:
        request_raw = regular_bytes(request_path)
        observed_request = decode_canonical(request_raw)
        validate_request(observed_request)
    except (ControllerError, OSError):
        return invalid_record_response(invocation_id)
    if expected_request is not None and observed_request != expected_request:
        emit(
            {
                "code": "invocation_request_mismatch",
                "invocation_id": invocation_id,
                "schema_version": 1,
                "status": "fail",
            }
        )
        return 0
    terminal_path = invocation_dir / "terminal.json"
    try:
        os.lstat(terminal_path)
    except FileNotFoundError:
        return progress_response(invocation_id)
    try:
        terminal_raw = regular_bytes(terminal_path)
        terminal = decode_canonical(terminal_raw)
        validate_terminal(terminal, invocation_id, str(observed_request["mode"]))
    except (ControllerError, OSError):
        return invalid_record_response(invocation_id)
    sys.stdout.buffer.write(terminal_raw)
    sys.stdout.buffer.flush()
    return 0


def nonstarted_preflight(invocation_dir: Path, invocation_id: str) -> int:
    terminal = {
        "code": "preflight_failed",
        "inner_exit_code": None,
        "invocation_id": invocation_id,
        "patch_receipt": None,
        "patch_started": False,
        "schema_version": 1,
        "status": "fail",
    }
    terminal_raw = canonical(terminal)
    atomic(invocation_dir / "terminal.json", terminal_raw)
    sys.stdout.buffer.write(terminal_raw)
    sys.stdout.buffer.flush()
    return 0


class ManagedSignalRelay:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.previous_handlers: dict[int, object] = {}
        self.previous_mask: set[signal.Signals] = set()
        self.active_mask: set[signal.Signals] = set()

    def __enter__(self) -> ManagedSignalRelay:
        self.previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, MANAGED_SIGNALS
        )
        self.active_mask = self.previous_mask.difference(MANAGED_SIGNALS)
        self.previous_handlers = {
            number: signal.getsignal(number) for number in MANAGED_SIGNALS
        }
        for number in MANAGED_SIGNALS:
            signal.signal(number, self.forward)
        return self

    def forward(self, number: int, _frame: object) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, number)
        except ProcessLookupError:
            pass

    def restore_child_mask(self) -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, self.active_mask)

    def spawn(self, arguments: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            arguments,
            preexec_fn=self.restore_child_mask,
            **kwargs,
        )
        self.process = process
        signal.pthread_sigmask(signal.SIG_SETMASK, self.active_mask)
        return process

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
        signal.pthread_sigmask(signal.SIG_SETMASK, self.previous_mask)
        for number, handler in self.previous_handlers.items():
            signal.signal(number, handler)


def run_claimed_with_relay(
    invocation_dir: Path,
    invocation_id: str,
    request: dict[str, object],
    relay: ManagedSignalRelay,
) -> int:
    stdout_path = invocation_dir / ".patch.stdout"
    stderr_path = invocation_dir / ".patch.stderr"
    stdout_descriptor = os.open(stdout_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stderr_descriptor = os.open(stderr_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        process = relay.spawn(
            [
                str(request["patch_script"]),
                "--mode",
                str(request["mode"]),
                "--bundle-dir",
                str(request["bundle_dir"]),
                "--state-dir",
                str(request["state_dir"]),
                "--sandbox",
                str(request["sandbox"]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout_descriptor,
            stderr=stderr_descriptor,
            close_fds=True,
            start_new_session=True,
        )
        returncode = process.wait()
    finally:
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)
    try:
        stdout_raw = regular_bytes(stdout_path)
        stderr_raw = regular_bytes(stderr_path)
        if stderr_raw:
            raise ControllerError
        patch_receipt = decode_canonical(stdout_raw)
        validate_patch_receipt(patch_receipt, str(request["mode"]), returncode)
        terminal: dict[str, object] = {
            "inner_exit_code": returncode,
            "invocation_id": invocation_id,
            "patch_receipt": patch_receipt,
            "schema_version": 1,
            "status": patch_receipt["status"],
        }
    except (ControllerError, FileNotFoundError):
        terminal = {
            "code": "invalid_patch_result",
            "inner_exit_code": returncode,
            "invocation_id": invocation_id,
            "patch_receipt": None,
            "schema_version": 1,
            "status": "fail",
        }
    terminal_raw = canonical(terminal)
    atomic(invocation_dir / "terminal.json", terminal_raw)
    for path in (stdout_path, stderr_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    sys.stdout.buffer.write(terminal_raw)
    sys.stdout.buffer.flush()
    return 0


def run_claimed(
    invocation_dir: Path, invocation_id: str, request: dict[str, object]
) -> int:
    with ManagedSignalRelay() as relay:
        return run_claimed_with_relay(
            invocation_dir, invocation_id, request, relay
        )


def main() -> int:
    os.umask(0o077)
    try:
        interface, parsed = parse_arguments(sys.argv[1:])
        invocation_root, invocation_id = invocation_boundary(parsed)
    except (ControllerError, OSError, ValueError):
        return preflight_failure()
    invocation_dir = invocation_root / invocation_id
    if interface == "receipt":
        return receipt_response(invocation_dir, invocation_id, None)
    request = bind_request(parsed)
    try:
        os.mkdir(invocation_dir, 0o700)
    except FileExistsError:
        return receipt_response(invocation_dir, invocation_id, request)
    except OSError:
        return preflight_failure()
    try:
        if private_directory(invocation_dir) != invocation_dir:
            raise ControllerError
        fsync_directory(invocation_root)
        atomic(invocation_dir / "request.json", canonical(request))
        try:
            validated_request = execution_request(request)
        except (ControllerError, OSError, ValueError):
            return nonstarted_preflight(invocation_dir, invocation_id)
        return run_claimed(invocation_dir, invocation_id, validated_request)
    except BaseException:
        terminal = {
            "code": "invalid_patch_result",
            "inner_exit_code": -1,
            "invocation_id": invocation_id,
            "patch_receipt": None,
            "schema_version": 1,
            "status": "fail",
        }
        try:
            atomic(invocation_dir / "terminal.json", canonical(terminal))
        except BaseException:
            pass
        emit(terminal)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
