from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts" / "run_acs_live_instance_patch_once.py"
SANDBOX = "acs-chemistry-agent"
INVOCATION_ID = "11111111-1111-4111-8111-111111111111"
FAIL_RECEIPT = {
    "code": "operation_failed",
    "main_session_touched": False,
    "rollback": True,
    "schema_version": 1,
    "status": "fail",
}
PASS_RECEIPT = {
    "loop_detection": True,
    "main_session_touched": False,
    "manifest_files": 6,
    "mode": "apply",
    "rollback_ready": True,
    "runner_hash": "a" * 64,
    "schema_version": 1,
    "status": "pass",
    "tools_hash": "b" * 64,
    "workshop_reset": True,
}


def _canonical(payload: object) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def _write_fake_patch(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


counter = Path(os.environ["ACS_CONTROLLER_COUNTER"])
count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
counter.write_text(str(count), encoding="utf-8")
Path(os.environ["ACS_CONTROLLER_ARGS"]).write_text(
    json.dumps(sys.argv[1:], separators=(",", ":")) + "\\n", encoding="utf-8"
)
started = os.environ.get("ACS_CONTROLLER_STARTED")
release = os.environ.get("ACS_CONTROLLER_RELEASE")
if started and release:
    Path(started).touch()
    deadline = time.monotonic() + 10
    while not Path(release).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
receipt = os.environ["ACS_CONTROLLER_PATCH_RECEIPT"]
sys.stdout.write(receipt)
if os.environ.get("ACS_CONTROLLER_SECOND_RECEIPT") == "1":
    sys.stdout.write(receipt)
raise SystemExit(int(os.environ["ACS_CONTROLLER_PATCH_RC"]))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _environment(
    tmp_path: Path, receipt: dict[str, object], returncode: int
) -> tuple[dict[str, str], Path, Path]:
    counter = tmp_path / "counter"
    args_file = tmp_path / "args.json"
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_CONTROLLER_ARGS": str(args_file),
            "ACS_CONTROLLER_COUNTER": str(counter),
            "ACS_CONTROLLER_PATCH_RC": str(returncode),
            "ACS_CONTROLLER_PATCH_RECEIPT": _canonical(receipt),
        }
    )
    return environment, counter, args_file


def _command(
    invocation_root: Path,
    patch_script: Path,
    bundle_dir: Path,
    state_dir: Path,
    *,
    invocation_id: str = INVOCATION_ID,
) -> list[str]:
    return [
        sys.executable,
        str(CONTROLLER),
        "--invocation-root",
        str(invocation_root),
        "--invocation-id",
        invocation_id,
        "--patch-script",
        str(patch_script),
        "--mode",
        "apply",
        "--bundle-dir",
        str(bundle_dir),
        "--state-dir",
        str(state_dir),
        "--sandbox",
        SANDBOX,
    ]


def _run(
    command: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _receipt_command(
    invocation_root: Path, *, invocation_id: str = INVOCATION_ID
) -> list[str]:
    return [
        sys.executable,
        str(CONTROLLER),
        "--receipt",
        "--invocation-root",
        str(invocation_root),
        "--invocation-id",
        invocation_id,
    ]


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    invocation_root = tmp_path / "invocations"
    patch_script = tmp_path / "fake-patch.py"
    bundle_dir = tmp_path / "bundle"
    state_dir = tmp_path / "state"
    invocation_root.mkdir(mode=0o700)
    bundle_dir.mkdir(mode=0o700)
    _write_fake_patch(patch_script)
    return invocation_root, patch_script, bundle_dir, state_dir


def _run_signal_race_driver(
    tmp_path: Path, phase: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    driver = tmp_path / f"signal-race-{phase}.py"
    invocation_dir = tmp_path / f"signal-race-{phase}-invocation"
    forwarded = tmp_path / f"signal-race-{phase}-forwarded"
    invocation_dir.mkdir(mode=0o700)
    driver.write_text(
        """from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path


controller_path = Path(sys.argv[1])
phase = sys.argv[2]
invocation_dir = Path(sys.argv[3])
forwarded = Path(sys.argv[4])
spec = importlib.util.spec_from_file_location("controller_under_signal_test", controller_path)
assert spec is not None and spec.loader is not None
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
receipt = {
    "loop_detection": True,
    "main_session_touched": False,
    "manifest_files": 6,
    "mode": "apply",
    "rollback_ready": True,
    "runner_hash": "a" * 64,
    "schema_version": 1,
    "status": "pass",
    "tools_hash": "b" * 64,
    "workshop_reset": True,
}
request = {
    "bundle_dir": str(invocation_dir),
    "mode": "apply",
    "patch_script": str(controller_path),
    "sandbox": "acs-chemistry-agent",
    "schema_version": 1,
    "state_dir": str(invocation_dir / "state"),
}


class FakePatchProcess:
    def __init__(self, _arguments: object, **kwargs: object) -> None:
        self.pid = 999999
        os.write(int(kwargs["stdout"]), controller.canonical(receipt))
        if phase == "early":
            os.kill(os.getpid(), signal.SIGTERM)

    def wait(self) -> int:
        return 0


def record_forward(_pid: int, number: int) -> None:
    forwarded.write_text(str(number), encoding="utf-8")
    raise ProcessLookupError


controller.subprocess.Popen = FakePatchProcess
controller.os.killpg = record_forward
if phase == "late":
    original_regular = controller.regular_bytes
    fired = False

    def signal_before_receipt_read(path: Path) -> bytes:
        global fired
        if not fired and path.name == ".patch.stdout":
            fired = True
            os.kill(os.getpid(), signal.SIGTERM)
        return original_regular(path)

    controller.regular_bytes = signal_before_receipt_read
raise SystemExit(
    controller.run_claimed(
        invocation_dir,
        "11111111-1111-4111-8111-111111111111",
        request,
    )
)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(CONTROLLER),
            phase,
            str(invocation_dir),
            str(forwarded),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, invocation_dir, forwarded


def test_controller_records_failed_patch_once_and_duplicate_reuses_terminal(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, args_file = _environment(tmp_path, FAIL_RECEIPT, 17)
    command = _command(invocation_root, patch_script, bundle_dir, state_dir)

    first = _run(command, environment)
    second = _run(command, environment)

    expected = {
        "inner_exit_code": 17,
        "invocation_id": INVOCATION_ID,
        "patch_receipt": FAIL_RECEIPT,
        "schema_version": 1,
        "status": "fail",
    }
    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == _canonical(expected)
    assert counter.read_text(encoding="utf-8") == "1"
    assert json.loads(args_file.read_text(encoding="utf-8")) == [
        "--mode",
        "apply",
        "--bundle-dir",
        str(bundle_dir),
        "--state-dir",
        str(state_dir),
        "--sandbox",
        SANDBOX,
    ]
    invocation_dir = invocation_root / INVOCATION_ID
    request = invocation_dir / "request.json"
    terminal = invocation_dir / "terminal.json"
    assert stat.S_IMODE(invocation_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(request.stat().st_mode) == 0o600
    assert stat.S_IMODE(terminal.stat().st_mode) == 0o600
    assert terminal.read_text(encoding="utf-8") == _canonical(expected)
    assert sorted(path.name for path in invocation_dir.iterdir()) == [
        "request.json",
        "terminal.json",
    ]


def test_controller_records_successful_patch_with_inner_exit_code(tmp_path: Path) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)

    completed = _run(
        _command(invocation_root, patch_script, bundle_dir, state_dir), environment
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == {
        "inner_exit_code": 0,
        "invocation_id": INVOCATION_ID,
        "patch_receipt": PASS_RECEIPT,
        "schema_version": 1,
        "status": "pass",
    }
    assert counter.read_text(encoding="utf-8") == "1"


def test_receipt_lookup_does_not_revalidate_mutable_patch_or_bundle_paths(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    completed = _run(
        _command(invocation_root, patch_script, bundle_dir, state_dir), environment
    )
    patch_script.unlink()
    bundle_dir.rmdir()

    receipt = _run(_receipt_command(invocation_root), environment)

    assert completed.returncode == receipt.returncode == 0
    assert receipt.stdout == completed.stdout
    assert receipt.stderr == ""
    assert counter.read_text(encoding="utf-8") == "1"


def test_controller_concurrent_duplicate_never_reexecutes_patch(tmp_path: Path) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    started = tmp_path / "started"
    release = tmp_path / "release"
    environment.update(
        {
            "ACS_CONTROLLER_STARTED": str(started),
            "ACS_CONTROLLER_RELEASE": str(release),
        }
    )
    command = _command(invocation_root, patch_script, bundle_dir, state_dir)
    first = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not started.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()

    duplicate = _run(command, environment)

    assert duplicate.returncode == 0
    assert json.loads(duplicate.stdout) == {
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "in_progress",
    }
    assert counter.read_text(encoding="utf-8") == "1"
    release.touch()
    first_stdout, first_stderr = first.communicate(timeout=20)
    assert first.returncode == 0, (first_stdout, first_stderr)
    terminal_duplicate = _run(command, environment)
    assert terminal_duplicate.returncode == 0
    assert terminal_duplicate.stdout == first_stdout
    assert counter.read_text(encoding="utf-8") == "1"


def test_controller_rejects_duplicate_id_with_different_normalized_request(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    original = _command(invocation_root, patch_script, bundle_dir, state_dir)
    first = _run(original, environment)
    other_state = tmp_path / "other-state"

    mismatch = _run(
        _command(invocation_root, patch_script, bundle_dir, other_state), environment
    )
    exact_duplicate = _run(original, environment)

    assert first.returncode == mismatch.returncode == exact_duplicate.returncode == 0
    assert json.loads(mismatch.stdout) == {
        "code": "invocation_request_mismatch",
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "fail",
    }
    assert exact_duplicate.stdout == first.stdout
    assert counter.read_text(encoding="utf-8") == "1"
    request_path = invocation_root / INVOCATION_ID / "request.json"
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o600
    assert json.loads(request_path.read_text(encoding="utf-8")) == {
        "bundle_dir": str(bundle_dir.resolve()),
        "mode": "apply",
        "patch_script": str(patch_script.resolve()),
        "sandbox": SANDBOX,
        "schema_version": 1,
        "state_dir": str(state_dir.resolve()),
    }


def test_controller_revalidates_nested_patch_receipt_from_terminal(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    command = _command(invocation_root, patch_script, bundle_dir, state_dir)
    completed = _run(command, environment)
    assert completed.returncode == 0
    terminal_path = invocation_root / INVOCATION_ID / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["patch_receipt"]["unexpected"] = True
    terminal_path.write_text(_canonical(terminal), encoding="utf-8")
    terminal_path.chmod(0o600)

    duplicate = _run(command, environment)

    assert duplicate.returncode == 0
    assert json.loads(duplicate.stdout) == {
        "code": "invocation_record_invalid",
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "fail",
    }
    assert counter.read_text(encoding="utf-8") == "1"


@pytest.mark.parametrize("malformation", ["directory-mode", "request"])
def test_receipt_lookup_distinguishes_malformed_claim_from_missing_request(
    tmp_path: Path, malformation: str
) -> None:
    invocation_root, _, _, _ = _setup(tmp_path)
    invocation_dir = invocation_root / INVOCATION_ID
    invocation_dir.mkdir(mode=0o700)
    environment = os.environ.copy()

    missing = _run(_receipt_command(invocation_root), environment)
    assert missing.returncode == 0
    assert json.loads(missing.stdout) == {
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "in_progress",
    }

    if malformation == "directory-mode":
        invocation_dir.chmod(0o750)
    else:
        request = invocation_dir / "request.json"
        request.write_text("{}\n", encoding="utf-8")
        request.chmod(0o600)
    malformed = _run(_receipt_command(invocation_root), environment)

    assert malformed.returncode == 0
    assert json.loads(malformed.stdout) == {
        "code": "invocation_record_invalid",
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "fail",
    }


def test_controller_claims_and_terminalizes_nonstarted_preflight_failure(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    patch_script.unlink()
    command = _command(invocation_root, patch_script, bundle_dir, state_dir)

    first = _run(command, environment)
    duplicate = _run(command, environment)
    mismatch = _run(
        _command(
            invocation_root,
            patch_script,
            bundle_dir,
            tmp_path / "other-state",
        ),
        environment,
    )
    receipt = _run(_receipt_command(invocation_root), environment)

    expected = {
        "code": "preflight_failed",
        "inner_exit_code": None,
        "invocation_id": INVOCATION_ID,
        "patch_receipt": None,
        "patch_started": False,
        "schema_version": 1,
        "status": "fail",
    }
    assert first.returncode == duplicate.returncode == mismatch.returncode == 0
    assert receipt.returncode == 0
    assert first.stdout == duplicate.stdout == receipt.stdout == _canonical(expected)
    assert json.loads(mismatch.stdout) == {
        "code": "invocation_request_mismatch",
        "invocation_id": INVOCATION_ID,
        "schema_version": 1,
        "status": "fail",
    }
    assert not counter.exists()
    invocation_dir = invocation_root / INVOCATION_ID
    assert stat.S_IMODE(invocation_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((invocation_dir / "request.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((invocation_dir / "terminal.json").stat().st_mode) == 0o600


def test_controller_forwards_term_to_patch_group_and_writes_terminal(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    started = tmp_path / "signal-started"
    release = tmp_path / "signal-release"
    environment.update(
        {
            "ACS_CONTROLLER_STARTED": str(started),
            "ACS_CONTROLLER_RELEASE": str(release),
        }
    )
    process = subprocess.Popen(
        _command(invocation_root, patch_script, bundle_dir, state_dir),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not started.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)

    assert process.returncode == 0, (stdout, stderr)
    terminal = json.loads(stdout)
    assert terminal == {
        "code": "invalid_patch_result",
        "inner_exit_code": -signal.SIGTERM,
        "invocation_id": INVOCATION_ID,
        "patch_receipt": None,
        "schema_version": 1,
        "status": "fail",
    }
    receipt = _run(_receipt_command(invocation_root), environment)
    assert receipt.returncode == 0
    assert receipt.stdout == stdout
    assert counter.read_text(encoding="utf-8") == "1"


def test_controller_blocks_early_managed_signal_until_patch_group_exists(
    tmp_path: Path,
) -> None:
    completed, invocation_dir, forwarded = _run_signal_race_driver(
        tmp_path, "early"
    )

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert json.loads(completed.stdout)["status"] == "pass"
    assert json.loads(
        (invocation_dir / "terminal.json").read_text(encoding="utf-8")
    )["status"] == "pass"
    assert forwarded.read_text(encoding="utf-8") == str(signal.SIGTERM)


def test_controller_handles_late_managed_signal_until_terminal_is_durable(
    tmp_path: Path,
) -> None:
    completed, invocation_dir, forwarded = _run_signal_race_driver(
        tmp_path, "late"
    )

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert json.loads(completed.stdout)["status"] == "pass"
    terminal = invocation_dir / "terminal.json"
    assert json.loads(terminal.read_text(encoding="utf-8"))["status"] == "pass"
    assert stat.S_IMODE(terminal.stat().st_mode) == 0o600
    assert forwarded.read_text(encoding="utf-8") == str(signal.SIGTERM)


def test_controller_terminalizes_invalid_patch_output_without_retry(tmp_path: Path) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)
    environment["ACS_CONTROLLER_SECOND_RECEIPT"] = "1"
    command = _command(invocation_root, patch_script, bundle_dir, state_dir)

    first = _run(command, environment)
    duplicate = _run(command, environment)

    expected = {
        "code": "invalid_patch_result",
        "inner_exit_code": 0,
        "invocation_id": INVOCATION_ID,
        "patch_receipt": None,
        "schema_version": 1,
        "status": "fail",
    }
    assert first.returncode == duplicate.returncode == 0
    assert first.stdout == duplicate.stdout == _canonical(expected)
    assert counter.read_text(encoding="utf-8") == "1"


def test_controller_rejects_noncanonical_invocation_id_before_claim(
    tmp_path: Path,
) -> None:
    invocation_root, patch_script, bundle_dir, state_dir = _setup(tmp_path)
    environment, counter, _ = _environment(tmp_path, PASS_RECEIPT, 0)

    completed = _run(
        _command(
            invocation_root,
            patch_script,
            bundle_dir,
            state_dir,
            invocation_id="not-a-uuid",
        ),
        environment,
    )

    assert completed.returncode == 70
    assert json.loads(completed.stdout) == {
        "code": "preflight_failed",
        "patch_started": False,
        "schema_version": 1,
        "status": "fail",
    }
    assert completed.stderr == ""
    assert not counter.exists()
    assert tuple(invocation_root.iterdir()) == ()
