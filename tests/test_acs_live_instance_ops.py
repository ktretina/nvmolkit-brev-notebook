from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = ROOT / "scripts" / "acs_live_instance_patch.sh"
QA_SCRIPT = ROOT / "scripts" / "run_acs_openclaw_live_qa.py"
PAGE = ROOT / "docs" / "acs-fall-2026-workshop.md"
TRAJECTORY_FIXTURE = (
    ROOT / "tests" / "fixtures" / "acs_openclaw_2026_7_1_trajectory.jsonl"
)
SANDBOX = "acs-chemistry-agent"
WORKSPACE = Path("sandbox/.openclaw/workspace")
LOOP_MISSING_DIAGNOSTIC = (
    "Config path not found: tools.loopDetection.enabled. "
    "Run openclaw config validate to inspect config shape."
)
PROMPT_SHA256 = (
    "39ca26c1b494dbe01bcbaabf27d72d755b444915e9ff26c874e629f09610bf22",
    "5d556991910812a24bb09b23cd250fd4a7157986948082fb8cc05cb3d52c1f5e",
    "6779b1bfbe141a72c795d5e648ad33a5e7ddd55a8bc953b0c1ae116f757be34a",
    "ec93fcfa236b6000980178626b322aeb0786a52a53a0132338784221c24550ea",
)
BUNDLE_FILES = (
    "acs_workshop_runner.py",
    "launchable/acs_workspace_tools.md",
    "scripts/verify_acs_openclaw_trajectory.py",
    "scripts/acs_live_instance_patch.sh",
    "scripts/run_acs_openclaw_live_qa.py",
    "docs/acs-fall-2026-workshop.md",
)
PROTECTED_FILES = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
    "data/PROVENANCE.md",
    "objective_challenge.py",
)
REQUIRED_ZIP_MEMBERS = (
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


def _load_qa_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acs_live_qa_under_test", QA_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts = str(QA_SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path[0] == scripts
        sys.path.pop(0)
    return module


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)


def _fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.jsonl"
    loop_state = tmp_path / "loop-state.json"
    _write_executable(
        fake_bin / "id",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "case \"${1-}\" in\n"
        "  -un) printf '%s\\n' \"${ACS_FAKE_USER:-ubuntu}\" ;;\n"
        "  -u) printf '%s\\n' \"${ACS_FAKE_UID}\" ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
    )
    _write_executable(
        fake_bin / "nemoclaw",
        r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


args = sys.argv[1:]
log = Path(os.environ["ACS_FAKE_COMMAND_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(["nemoclaw", *args], separators=(",", ":")) + "\n")

if len(args) < 2 or args[0] != "acs-chemistry-agent":
    raise SystemExit(64)

root = Path(os.environ["ACS_FAKE_SANDBOX_ROOT"])
loop_path = Path(os.environ["ACS_FAKE_LOOP_STATE"])


def mapped(value: str) -> str:
    value = value.replace(
        "/sandbox/.openclaw/workspace", str(root / "sandbox/.openclaw/workspace")
    )
    return value.replace(
        "/sandbox/.acs-prompt-reliability-20260821",
        str(root / "sandbox/.acs-prompt-reliability-20260821"),
    )


def load_loop() -> dict[str, object]:
    return json.loads(loop_path.read_text(encoding="utf-8"))


def save_loop(presence: str, value: bool | None = None) -> None:
    payload: dict[str, object] = {"presence": presence}
    if presence == "present":
        payload["value"] = value
    loop_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


operation = args[1]
if operation == "agent":
    counter = Path(os.environ["ACS_FAKE_AGENT_COUNTER"])
    count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
    counter.write_text(str(count), encoding="utf-8")
    failure_index = int(os.environ.get("ACS_FAKE_AGENT_FAILURE_INDEX", "0"))
    if count == failure_index:
        mode = os.environ.get("ACS_FAKE_AGENT_FAILURE_MODE", "failure")
        print("answer-secret-canary")
        print("exception-secret-canary", file=sys.stderr)
        raise SystemExit(124 if mode == "timeout" else 75 if mode == "rc75" else 17)
    print("answer-secret-canary")
    print("exception-secret-canary", file=sys.stderr)
    raise SystemExit(0)

if operation == "sessions" and args[2:3] == ["export"]:
    key = args[3]
    out = Path(args[args.index("--out") + 1])
    out.mkdir(mode=0o775)
    out.chmod(0o775)
    session_id = key.rsplit(":", 1)[-1]
    if os.environ.get("ACS_FAKE_EXPORT_OTHER") == "1":
        session_id = "00000000-0000-4000-8000-000000000000"
    destination = out / f"{session_id}.trajectory.jsonl"
    shutil.copyfile(os.environ["ACS_FAKE_TRAJECTORY_FIXTURE"], destination)
    destination.chmod(0o600)
    session = out / f"{session_id}.jsonl"
    session_mutation = os.environ.get("ACS_FAKE_SESSION_EXPORT_MUTATION")
    if session_mutation == "symlink":
        session.symlink_to(destination.name)
    elif session_mutation == "hardlink":
        os.link(destination, session)
    else:
        session.write_text('api_key="session-jsonl-must-not-be-read"\n', encoding="utf-8")
        session.chmod(0o644 if session_mutation == "mode" else 0o600)
    if os.environ.get("ACS_FAKE_EXPORT_EXTRA") == "1":
        extra = out / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o600)
    raise SystemExit(0)

if operation == "gateway" and args[2:] == ["restart", "--quiet"]:
    raise SystemExit(0)

if operation != "exec" or "--" not in args:
    raise SystemExit(64)

command = args[args.index("--") + 1 :]
if command[:3] == ["env", "NO_COLOR=1", "NODE_NO_WARNINGS=1"]:
    command = command[3:]
if command[:3] == ["openclaw", "config", "get"]:
    state = load_loop()
    if state["presence"] == "absent":
        if os.environ.get("ACS_FAKE_ABSENT_STDOUT_NEWLINE") == "1":
            print()
        if os.environ.get("ACS_FAKE_ABSENT_FATAL_PREFIX") == "1":
            print("fatal: transport authentication failed", file=sys.stderr)
        print(
            "\x1b[1m\x1b[32m✓\x1b[39m\x1b[0m "
            "Active gateway set to 'nemoclaw'",
            file=sys.stderr,
        )
        print(
            "Config path not found: tools.loopDetection.enabled. "
            "Run openclaw config validate to inspect config shape.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("true" if state["value"] is True else "false")
    raise SystemExit(0)
if command[:3] == ["openclaw", "config", "unset"]:
    if command[-1] == "--help":
        raise SystemExit(
            0 if os.environ.get("ACS_FAKE_UNSET_SUPPORTED", "1") == "1" else 2
        )
    save_loop("absent")
    raise SystemExit(0)
if command[:3] == ["openclaw", "config", "set"]:
    if os.environ.get("ACS_FAKE_TERM_ON_CONFIG_SET") == "1":
        os.kill(os.getppid(), signal.SIGTERM)
        raise SystemExit(143)
    save_loop("present", command[4] == "true")
    raise SystemExit(0)

mapped_command = [mapped(value) for value in command]
if "-c" in mapped_command and len(mapped_command) > mapped_command.index("-c") + 2:
    code_index = mapped_command.index("-c") + 1
    helper_action = mapped_command[code_index + 1]
    if helper_action == "backup" and os.environ.get("ACS_FAKE_BACKUP_STARTED"):
        started = Path(os.environ["ACS_FAKE_BACKUP_STARTED"])
        release = Path(os.environ["ACS_FAKE_BACKUP_RELEASE"])
        try:
            descriptor = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            concurrent = os.environ.get("ACS_FAKE_BACKUP_CONCURRENT")
            if concurrent:
                Path(concurrent).touch()
            raise SystemExit(67)
        else:
            os.close(descriptor)
        deadline = time.monotonic() + 10
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not release.exists():
            raise SystemExit(66)
    if (
        os.environ.get("ACS_FAKE_FAIL_FINAL_VERIFY") == "1"
        and helper_action == "verify-installed"
        and load_loop() == {"presence": "present", "value": True}
    ):
        raise SystemExit(31)
    if (
        os.environ.get("ACS_FAKE_FAIL_RESTORE_COPY") == "1"
        and helper_action == "rollback"
    ):
        needle = 'raw = contents["."]'
        replacement = '(_ for _ in ()).throw(OSError("injected restore copy"))'
        if needle not in mapped_command[code_index]:
            raise SystemExit(65)
        mapped_command[code_index] = mapped_command[code_index].replace(
            needle, replacement, 1
        )
    if (
        os.environ.get("ACS_FAKE_TERM_DURING_BACKUP") == "1"
        and helper_action == "backup"
    ):
        os.kill(os.getppid(), signal.SIGTERM)
        raise SystemExit(143)
if (
    os.environ.get("ACS_FAKE_FAIL_HELPER_ACTION")
    and "-c" in mapped_command
    and len(mapped_command) > mapped_command.index("-c") + 2
    and mapped_command[mapped_command.index("-c") + 2]
    == os.environ["ACS_FAKE_FAIL_HELPER_ACTION"]
):
    raise SystemExit(29)
completed = subprocess.run(mapped_command, env=os.environ, check=False)
raise SystemExit(completed.returncode)
""",
    )
    _write_executable(
        fake_bin / "openshell",
        r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


args = sys.argv[1:]
log = Path(os.environ["ACS_FAKE_COMMAND_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(["openshell", *args], separators=(",", ":")) + "\n")
if len(args) != 5 or args[0] != "sandbox":
    raise SystemExit(64)
operation, sandbox, source_raw, destination_raw = args[1:]
if sandbox != "acs-chemistry-agent":
    raise SystemExit(64)
root = Path(os.environ["ACS_FAKE_SANDBOX_ROOT"])


def mapped(value: str) -> Path:
    value = value.replace(
        "/sandbox/.openclaw/workspace", str(root / "sandbox/.openclaw/workspace")
    )
    value = value.replace(
        "/sandbox/.acs-prompt-reliability-20260821",
        str(root / "sandbox/.acs-prompt-reliability-20260821"),
    )
    return Path(value)


if operation == "upload":
    if (
        os.environ.get("ACS_FAKE_FAIL_UPLOAD") == "1"
        and Path(source_raw).name != "backup-package.json"
    ):
        raise SystemExit(23)
    source = Path(source_raw)
    destination = mapped(destination_raw)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / source.name)
elif operation == "download":
    source = mapped(source_raw)
    destination = Path(destination_raw)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if os.environ.get("ACS_FAKE_DOWNLOAD_SYMLINK") == "1":
        target.symlink_to(Path(os.environ["ACS_FAKE_SYMLINK_TARGET"]))
    else:
        shutil.copy2(source, target)
        if os.environ.get("ACS_FAKE_DOWNLOAD_EXTRA") == "1":
            (destination / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
else:
    raise SystemExit(64)
""",
    )
    return fake_bin, command_log, loop_state


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_workspace(root: Path) -> Path:
    workspace = root / WORKSPACE
    (workspace / "data").mkdir(parents=True)
    (workspace / "outputs/workshop/nested").mkdir(parents=True)
    (workspace / ".acs-workshop-state").mkdir(mode=0o700)
    contents = {
        "TOOLS.md": b"old tools\n",
        "acs_workshop_runner.py": b"old runner\n",
        "chemistry_workflow.py": b"old workflow\n",
        "data/sample_molecules.csv": b"id,smiles\nold,C\n",
        "data/PROVENANCE.md": b"old provenance\n",
        "objective_challenge.py": b"old objective\n",
    }
    for relative, raw in contents.items():
        path = workspace / relative
        path.write_bytes(raw)
        path.chmod(0o444)
    manifest = {
        "schema_version": 1,
        "files": {
            relative: _sha256(workspace / relative) for relative in PROTECTED_FILES
        },
    }
    manifest_path = workspace / ".acs-workshop-state/manifest.json"
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o444)
    context = workspace / ".acs-workshop-state/context.json"
    history = workspace / ".acs-workshop-state/history.json"
    context.write_bytes(b'{"old":"context"}\n')
    history.write_bytes(b'{"old":"history"}\n')
    context.chmod(0o600)
    history.chmod(0o600)
    artifact = workspace / "outputs/workshop/nested/original.bin"
    artifact.write_bytes(b"original artifact\n")
    artifact.chmod(0o640)
    (workspace / "outputs/workshop").chmod(0o750)
    (workspace / "unrelated.txt").write_bytes(b"must remain\n")
    return workspace


def _seed_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    staged = {
        "acs_workshop_runner.py": (
            b"import argparse\n"
            b"parser = argparse.ArgumentParser()\n"
            b"parser.parse_args()\n"
        ),
        "launchable/acs_workspace_tools.md": b"new tools\n",
        "scripts/verify_acs_openclaw_trajectory.py": b"verifier\n",
        "scripts/acs_live_instance_patch.sh": b"patch\n",
        "scripts/run_acs_openclaw_live_qa.py": b"qa\n",
        "docs/acs-fall-2026-workshop.md": b"page\n",
    }
    for relative, raw in staged.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o600)
    manifest = {
        "schema_version": 1,
        "files": {relative: _sha256(bundle / relative) for relative in BUNDLE_FILES},
    }
    manifest_path = bundle / "bundle-manifest.json"
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o600)
    return bundle


def _snapshot(root: Path) -> dict[str, tuple[str, int, str | None]]:
    snapshot: dict[str, tuple[str, int, str | None]] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted([*directories, *files]):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = os.lstat(path).st_mode
            if stat.S_ISDIR(mode):
                snapshot[relative] = ("dir", stat.S_IMODE(mode), None)
            elif stat.S_ISREG(mode):
                snapshot[relative] = ("file", stat.S_IMODE(mode), _sha256(path))
            elif stat.S_ISLNK(mode):
                snapshot[relative] = ("link", stat.S_IMODE(mode), os.readlink(path))
            else:
                snapshot[relative] = ("special", stat.S_IMODE(mode), None)
    return snapshot


def _read_commands(path: Path) -> tuple[tuple[str, ...], ...]:
    if not path.exists():
        return ()
    return tuple(tuple(json.loads(line)) for line in path.read_text().splitlines())


def _base_environment(
    tmp_path: Path,
    fake_bin: Path,
    command_log: Path,
    loop_state: Path,
    sandbox_root: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_FAKE_AGENT_COUNTER": str(tmp_path / "agent-counter"),
            "ACS_FAKE_COMMAND_LOG": str(command_log),
            "ACS_FAKE_LOOP_STATE": str(loop_state),
            "ACS_FAKE_SANDBOX_ROOT": str(sandbox_root),
            "ACS_FAKE_TRAJECTORY_FIXTURE": str(TRAJECTORY_FIXTURE),
            "ACS_FAKE_UNSET_SUPPORTED": "1",
            "ACS_FAKE_UID": str(
                int(hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12], 16)
            ),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    return environment


def _run_patch(
    mode: str,
    bundle: Path,
    state_dir: Path,
    environment: dict[str, str],
    *,
    sandbox: str = SANDBOX,
) -> subprocess.CompletedProcess[str]:
    assert PATCH_SCRIPT.is_file(), "the live patch script is missing"
    return subprocess.run(
        [
            "bash",
            str(PATCH_SCRIPT),
            "--mode",
            mode,
            "--bundle-dir",
            str(bundle),
            "--state-dir",
            str(state_dir),
            "--sandbox",
            sandbox,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def _valid_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\xff"))
        + _png_chunk(b"IEND", b"")
    )


def _write_valid_zip(
    path: Path, contents: dict[str, bytes] | None = None
) -> None:
    overrides = {} if contents is None else contents
    with zipfile.ZipFile(path, "w", strict_timestamps=True) as archive:
        for name in REQUIRED_ZIP_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            default = (
                _valid_png()
                if name.endswith(".png")
                else b"{}\n"
                if name.endswith(".json")
                else b"validated evidence\n"
            )
            archive.writestr(info, overrides.get(name, default))
    path.chmod(0o600)


def _write_invalid_zip(path: Path, content: bytes = b"invalid evidence\n") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("evidence.txt", content)
    path.chmod(0o600)


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
    path.chmod(0o600)


def _run_qa(
    tmp_path: Path,
    environment: dict[str, str],
    *,
    session_id: str = "11111111-1111-4111-8111-111111111111",
    page: Path = PAGE,
    output_dir: Path | None = None,
    sandbox: str = SANDBOX,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    assert QA_SCRIPT.is_file(), "the exact-session QA script is missing"
    if output_dir is None:
        output_dir = tmp_path / "qa-output"
        output_dir.mkdir(mode=0o700)
    completed = subprocess.run(
        [
            sys.executable,
            str(QA_SCRIPT),
            "--session-id",
            session_id,
            "--page",
            str(page),
            "--output-dir",
            str(output_dir),
            "--sandbox",
            sandbox,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, output_dir


@pytest.fixture
def fake_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin, command_log, loop_state = _fake_tools(tmp_path)
    sandbox_root = tmp_path / "fake-sandbox"
    workspace = _seed_workspace(sandbox_root)
    loop_state.write_bytes(_canonical({"presence": "present", "value": False}))
    results = workspace / "outputs/workshop/results.zip"
    _write_valid_zip(results)
    environment = _base_environment(
        tmp_path, fake_bin, command_log, loop_state, sandbox_root
    )
    return environment, command_log, loop_state, sandbox_root


def test_live_operation_sources_pin_safe_interfaces() -> None:
    patch_source = PATCH_SCRIPT.read_text(encoding="utf-8")
    qa_source = QA_SCRIPT.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in patch_source
    assert "umask 077" in patch_source
    assert "rm -rf" not in patch_source
    assert "pkill" not in patch_source
    assert "brev " not in patch_source
    assert " session" not in patch_source
    assert "brev " not in qa_source
    assert "--include-trajectory" in qa_source
    assert "agent:main:" in qa_source
    for digest in PROMPT_SHA256:
        assert qa_source.count(digest) == 1
    assert stat.S_IMODE(PATCH_SCRIPT.stat().st_mode) == 0o755
    assert stat.S_IMODE(QA_SCRIPT.stat().st_mode) == 0o755


def test_patch_invalid_arguments_emit_one_closed_receipt(
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, _ = fake_environment

    completed = subprocess.run(
        ["bash", str(PATCH_SCRIPT), "--unknown"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "code": "preflight_failed",
        "main_session_touched": False,
        "rollback": False,
        "schema_version": 1,
        "status": "fail",
    }


@pytest.mark.parametrize(
    ("prior_state", "expected_after_rollback"),
    [
        ({"presence": "absent"}, {"presence": "absent"}),
        (
            {"presence": "present", "value": False},
            {"presence": "present", "value": False},
        ),
        (
            {"presence": "present", "value": True},
            {"presence": "present", "value": True},
        ),
    ],
)
def test_patch_apply_and_idempotent_rollback_restore_exact_prior_state(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    prior_state: dict[str, object],
    expected_after_rollback: dict[str, object],
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    loop_state.write_bytes(_canonical(prior_state))
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"

    applied = _run_patch("apply", bundle, state_dir, environment)

    assert applied.returncode == 0, (applied.stdout, applied.stderr)
    receipt = json.loads(applied.stdout)
    assert set(receipt) == {
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
    assert receipt["status"] == "pass"
    assert receipt["mode"] == "apply"
    assert receipt["loop_detection"] is True
    assert receipt["manifest_files"] == 6
    assert receipt["main_session_touched"] is False
    assert receipt["rollback_ready"] is True
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }
    assert (workspace / "acs_workshop_runner.py").read_bytes() == (
        bundle / "acs_workshop_runner.py"
    ).read_bytes()
    assert (workspace / "TOOLS.md").read_bytes() == (
        bundle / "launchable/acs_workspace_tools.md"
    ).read_bytes()
    assert not (workspace / ".acs-workshop-state/context.json").exists()
    assert not (workspace / ".acs-workshop-state/history.json").exists()
    assert list((workspace / "outputs/workshop").iterdir()) == []
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    for current, directories, files in os.walk(state_dir):
        for directory in directories:
            assert stat.S_IMODE((Path(current) / directory).stat().st_mode) == 0o700
        for filename in files:
            assert stat.S_IMODE((Path(current) / filename).stat().st_mode) == 0o600
    remote_state = sandbox_root / "sandbox/.acs-prompt-reliability-20260821"
    assert stat.S_IMODE(remote_state.stat().st_mode) == 0o700
    for current, directories, files in os.walk(remote_state):
        for directory in directories:
            assert stat.S_IMODE((Path(current) / directory).stat().st_mode) == 0o700
        for filename in files:
            assert stat.S_IMODE((Path(current) / filename).stat().st_mode) == 0o600
    apply_calls = _read_commands(command_log)
    assert (
        sum(call[2:] == ("gateway", "restart", "--quiet") for call in apply_calls) == 1
    )
    assert (
        sum(
            call[-6:]
            == (
                "openclaw",
                "config",
                "set",
                "tools.loopDetection.enabled",
                "true",
                "--strict-json",
            )
            for call in apply_calls
        )
        == 1
    )

    rolled_back = _run_patch("rollback", bundle, state_dir, environment)
    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)
    assert json.loads(rolled_back.stdout)["idempotent"] is False

    second_rollback = _run_patch("rollback", bundle, state_dir, environment)

    assert second_rollback.returncode == 0
    assert json.loads(second_rollback.stdout)["idempotent"] is True
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == expected_after_rollback
    calls = _read_commands(command_log)
    assert sum(call[2:] == ("gateway", "restart", "--quiet") for call in calls) == 2
    if prior_state["presence"] == "absent":
        assert any("unset" in call for call in calls)


def test_patch_accepts_standard_sticky_sandbox_tmp_parent(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, sandbox_root = fake_environment
    sandbox_tmp = sandbox_root / "tmp"
    sandbox_tmp.mkdir()
    sandbox_tmp.chmod(0o1777)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"

    applied = _run_patch("apply", bundle, state_dir, environment)

    assert applied.returncode == 0, (applied.stdout, applied.stderr)
    assert stat.S_IMODE(sandbox_tmp.stat().st_mode) == 0o1777
    rolled_back = _run_patch("rollback", bundle, state_dir, environment)
    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)


def test_trusted_backup_transfer_uses_private_sandbox_path(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, _, sandbox_root = fake_environment
    sandbox_tmp = sandbox_root / "tmp"
    sandbox_tmp.mkdir()
    sandbox_tmp.chmod(0o1777)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"

    applied = _run_patch("apply", bundle, state_dir, environment)
    assert applied.returncode == 0, (applied.stdout, applied.stderr)
    rolled_back = _run_patch("rollback", bundle, state_dir, environment)
    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)

    calls = _read_commands(command_log)
    downloads = [
        call for call in calls if call[:3] == ("openshell", "sandbox", "download")
    ]
    uploads = [
        call for call in calls if call[:3] == ("openshell", "sandbox", "upload")
    ]
    expected_root = "/sandbox/.acs-prompt-reliability-20260821/"
    assert len(downloads) == 1
    assert downloads[0][4].startswith(expected_root)
    assert len(uploads) == 3
    assert all(call[5].startswith(expected_root) for call in uploads)
    assert all(
        "/tmp/acs-prompt-reliability-20260821" not in argument
        for call in calls
        for argument in call
    )


def test_apply_stops_before_sandbox_mutation_when_absence_cannot_be_unset(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    loop_state.write_bytes(_canonical({"presence": "absent"}))
    environment["ACS_FAKE_UNSET_SUPPORTED"] = "0"
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode != 0
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {"presence": "absent"}
    calls = _read_commands(command_log)
    assert all(call[0] != "openshell" for call in calls)
    assert not any("set" in call and "config" in call for call in calls)


def test_backup_failure_clears_only_prepared_host_state_for_exact_retry(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    environment["ACS_FAKE_FAIL_HELPER_ACTION"] = "backup"

    failed = _run_patch("apply", bundle, state_dir, environment)

    assert failed.returncode == 70
    assert json.loads(failed.stdout)["code"] == "preflight_failed"
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }

    environment.pop("ACS_FAKE_FAIL_HELPER_ACTION")
    retried = _run_patch("apply", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)
    assert json.loads(retried.stdout)["status"] == "pass"


def test_signal_during_backup_leaves_no_wedge_and_exact_retry_succeeds(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    environment["ACS_FAKE_TERM_DURING_BACKUP"] = "1"

    interrupted = _run_patch("apply", bundle, state_dir, environment)

    assert interrupted.returncode != 0
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }

    environment.pop("ACS_FAKE_TERM_DURING_BACKUP")
    retried = _run_patch("apply", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)
    assert json.loads(retried.stdout)["status"] == "pass"


def test_process_kill_during_backup_is_reconciled_by_exact_retry(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    started = tmp_path / "backup-started"
    release = tmp_path / "backup-release"
    environment["ACS_FAKE_BACKUP_STARTED"] = str(started)
    environment["ACS_FAKE_BACKUP_RELEASE"] = str(release)
    process = subprocess.Popen(
        [
            "bash",
            str(PATCH_SCRIPT),
            "--mode",
            "apply",
            "--bundle-dir",
            str(bundle),
            "--state-dir",
            str(state_dir),
            "--sandbox",
            SANDBOX,
        ],
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
    process.kill()
    release.touch()
    process.communicate(timeout=10)
    assert _snapshot(workspace) == before

    environment.pop("ACS_FAKE_BACKUP_STARTED")
    environment.pop("ACS_FAKE_BACKUP_RELEASE")
    retried = _run_patch("apply", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)
    assert json.loads(retried.stdout)["status"] == "pass"
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }


def test_host_global_active_operation_rejects_cross_state_apply_and_wrong_order(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    original = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    first_state = tmp_path / "first-state"
    second_state = tmp_path / "second-state"
    first = _run_patch("apply", bundle, first_state, environment)
    assert first.returncode == 0, (first.stdout, first.stderr)
    after_first = _snapshot(workspace)

    second = _run_patch("apply", bundle, second_state, environment)
    wrong_rollback = _run_patch("rollback", bundle, second_state, environment)

    assert second.returncode == 70
    assert wrong_rollback.returncode == 70
    assert _snapshot(workspace) == after_first
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }

    rolled_back = _run_patch("rollback", bundle, first_state, environment)

    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)
    assert _snapshot(workspace) == original
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


def test_host_global_lock_prevents_concurrent_backup_from_another_state(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, _ = fake_environment
    bundle = _seed_bundle(tmp_path)
    started = tmp_path / "backup-started"
    release = tmp_path / "backup-release"
    concurrent = tmp_path / "backup-concurrent"
    environment.update(
        {
            "ACS_FAKE_BACKUP_STARTED": str(started),
            "ACS_FAKE_BACKUP_RELEASE": str(release),
            "ACS_FAKE_BACKUP_CONCURRENT": str(concurrent),
        }
    )
    first_state = tmp_path / "first-state"
    first = subprocess.Popen(
        [
            "bash",
            str(PATCH_SCRIPT),
            "--mode",
            "apply",
            "--bundle-dir",
            str(bundle),
            "--state-dir",
            str(first_state),
            "--sandbox",
            SANDBOX,
        ],
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

    second = _run_patch("apply", bundle, tmp_path / "second-state", environment)

    assert second.returncode == 70
    assert not concurrent.exists()
    release.touch()
    first_stdout, first_stderr = first.communicate(timeout=20)
    assert first.returncode == 0, (first_stdout, first_stderr)
    for name in (
        "ACS_FAKE_BACKUP_STARTED",
        "ACS_FAKE_BACKUP_RELEASE",
        "ACS_FAKE_BACKUP_CONCURRENT",
    ):
        environment.pop(name)
    rolled_back = _run_patch("rollback", bundle, first_state, environment)
    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)


def test_host_global_lock_prevents_concurrent_backup_from_same_state(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, _ = fake_environment
    bundle = _seed_bundle(tmp_path)
    started = tmp_path / "backup-started"
    release = tmp_path / "backup-release"
    concurrent = tmp_path / "backup-concurrent"
    environment.update(
        {
            "ACS_FAKE_BACKUP_STARTED": str(started),
            "ACS_FAKE_BACKUP_RELEASE": str(release),
            "ACS_FAKE_BACKUP_CONCURRENT": str(concurrent),
        }
    )
    state_dir = tmp_path / "shared-state"
    first = subprocess.Popen(
        [
            "bash",
            str(PATCH_SCRIPT),
            "--mode",
            "apply",
            "--bundle-dir",
            str(bundle),
            "--state-dir",
            str(state_dir),
            "--sandbox",
            SANDBOX,
        ],
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

    second = _run_patch("apply", bundle, state_dir, environment)

    assert second.returncode == 70
    assert not concurrent.exists()
    release.touch()
    first_stdout, first_stderr = first.communicate(timeout=20)
    assert first.returncode == 0, (first_stdout, first_stderr)
    for name in (
        "ACS_FAKE_BACKUP_STARTED",
        "ACS_FAKE_BACKUP_RELEASE",
        "ACS_FAKE_BACKUP_CONCURRENT",
    ):
        environment.pop(name)
    rolled_back = _run_patch("rollback", bundle, state_dir, environment)
    assert rolled_back.returncode == 0, (rolled_back.stdout, rolled_back.stderr)


def test_apply_requires_absent_config_get_stdout_to_be_zero_bytes(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    loop_state.write_bytes(_canonical({"presence": "absent"}))
    environment["ACS_FAKE_ABSENT_STDOUT_NEWLINE"] = "1"
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode == 70
    assert json.loads(completed.stdout)["code"] == "preflight_failed"
    assert completed.stderr == ""
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {"presence": "absent"}
    assert all(call[0] != "openshell" for call in _read_commands(command_log))


def test_absent_config_diagnostic_rejects_masqueraded_fatal_error_and_retries(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    loop_state.write_bytes(_canonical({"presence": "absent"}))
    environment["ACS_FAKE_ABSENT_FATAL_PREFIX"] = "1"
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"

    rejected = _run_patch("apply", bundle, state_dir, environment)

    assert rejected.returncode == 70
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {"presence": "absent"}
    assert all(call[0] != "openshell" for call in _read_commands(command_log))

    environment.pop("ACS_FAKE_ABSENT_FATAL_PREFIX")
    retried = _run_patch("apply", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)


def test_rollback_restores_from_host_backup_after_sandbox_backup_is_deleted(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    operation = next(
        (sandbox_root / "tmp/acs-prompt-reliability-20260821").iterdir()
    )
    (operation / "backup-package.json").unlink()

    completed = _run_patch("rollback", bundle, state_dir, environment)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


@pytest.mark.parametrize("failure_mode", ["upload", "term-after-install"])
def test_apply_failure_after_backup_rolls_back_without_exposing_output(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    failure_mode: str,
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    if failure_mode == "upload":
        environment["ACS_FAKE_FAIL_UPLOAD"] = "1"
    else:
        environment["ACS_FAKE_TERM_ON_CONFIG_SET"] = "1"

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode != 0
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }
    combined = completed.stdout + completed.stderr
    assert "old runner" not in combined
    assert "old tools" not in combined
    receipt = json.loads(completed.stdout)
    assert set(receipt) == {
        "code",
        "main_session_touched",
        "rollback",
        "schema_version",
        "status",
    }
    assert receipt["rollback"] is True


def test_apply_failure_after_successful_loop_readback_still_rolls_back(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    environment["ACS_FAKE_FAIL_FINAL_VERIFY"] = "1"
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode != 0
    assert completed.stderr == ""
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }
    assert json.loads(completed.stdout)["rollback"] is True


def test_stale_atomic_temp_cannot_corrupt_failure_rollback(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    environment["ACS_FAKE_TERM_ON_CONFIG_SET"] = "1"
    workspace = sandbox_root / WORKSPACE
    stale = workspace / ".acs_workshop_runner.py.patch-tmp"
    stale.write_bytes(b"pre-poisoned\n")
    stale.chmod(0o600)
    before = _snapshot(workspace)

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode != 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["rollback"] is True
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


def test_reset_between_qa_changes_only_the_three_approved_targets(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    (workspace / "outputs/workshop/result.txt").write_text("qa", encoding="utf-8")
    (workspace / ".acs-workshop-state/context.json").write_text("{}\n")
    (workspace / ".acs-workshop-state/history.json").write_text("{}\n")
    protected_before = {
        relative: _snapshot(workspace)[relative]
        for relative in [*PROTECTED_FILES, ".acs-workshop-state/manifest.json"]
    }

    completed = _run_patch("reset-between-qa", bundle, state_dir, environment)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    receipt = json.loads(completed.stdout)
    assert receipt == {
        "loop_detection": True,
        "main_session_touched": False,
        "mode": "reset-between-qa",
        "schema_version": 1,
        "status": "pass",
        "workshop_reset": True,
    }
    current = _snapshot(workspace)
    assert {
        relative: current[relative] for relative in protected_before
    } == protected_before
    assert list((workspace / "outputs/workshop").iterdir()) == []
    assert not (workspace / ".acs-workshop-state/context.json").exists()
    assert not (workspace / ".acs-workshop-state/history.json").exists()
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }
    calls = _read_commands(command_log)
    assert sum(call[2:] == ("gateway", "restart", "--quiet") for call in calls) == 1


def test_reset_failure_uses_the_same_full_rollback(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    before = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    (workspace / "outputs/workshop/partial.txt").write_text("partial\n")
    environment["ACS_FAKE_FAIL_HELPER_ACTION"] = "reset"

    completed = _run_patch("reset-between-qa", bundle, state_dir, environment)

    assert completed.returncode != 0
    assert json.loads(completed.stdout)["rollback"] is True
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


def test_reset_rejects_corrupt_rollback_source_before_mutation(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    partial = workspace / "outputs/workshop/partial.txt"
    partial.write_text("partial\n", encoding="utf-8")
    operation_id = json.loads(
        (state_dir / "current.json").read_text(encoding="utf-8")
    )["operation_id"]
    backup = state_dir / operation_id / "backup-package.json"
    backup.write_bytes(b"corrupt host backup\n")
    backup.chmod(0o600)
    before = _snapshot(workspace)

    completed = _run_patch("reset-between-qa", bundle, state_dir, environment)

    assert completed.returncode == 70
    assert json.loads(completed.stdout)["rollback"] is False
    assert completed.stderr == ""
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-user",
        "wrong-sandbox",
        "bundle-symlink",
        "hash-drift",
        "target-symlink",
        "target-ancestor-symlink",
        "protected-hash-drift",
        "protected-mode-drift",
        "state-directory-mode-drift",
    ],
)
def test_patch_rejects_identity_path_and_hash_violations_before_install(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    mutation: str,
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    bundle = _seed_bundle(tmp_path)
    sandbox = SANDBOX
    if mutation == "wrong-user":
        environment["ACS_FAKE_USER"] = "root"
    elif mutation == "wrong-sandbox":
        sandbox = "other"
    elif mutation == "bundle-symlink":
        target = bundle / "acs_workshop_runner.py"
        original = bundle / "runner-real.py"
        target.rename(original)
        target.symlink_to(original)
    elif mutation == "hash-drift":
        (bundle / "acs_workshop_runner.py").write_bytes(b"drift\n")
    elif mutation == "target-symlink":
        target = workspace / "TOOLS.md"
        target.unlink()
        target.symlink_to(workspace / "unrelated.txt")
    elif mutation == "target-ancestor-symlink":
        data = workspace / "data"
        outside = sandbox_root / "outside-data"
        data.rename(outside)
        data.symlink_to(outside, target_is_directory=True)
    elif mutation == "protected-hash-drift":
        protected = workspace / "data/sample_molecules.csv"
        protected.chmod(0o600)
        protected.write_bytes(b"id,smiles\ndrift,N\n")
        protected.chmod(0o444)
    elif mutation == "protected-mode-drift":
        (workspace / "chemistry_workflow.py").chmod(0o640)
    else:
        (workspace / ".acs-workshop-state").chmod(0o750)
    before = _snapshot(workspace)
    prior_loop = json.loads(loop_state.read_text())

    completed = _run_patch(
        "apply", bundle, tmp_path / "state", environment, sandbox=sandbox
    )

    assert completed.returncode != 0
    assert _snapshot(workspace) == before
    assert json.loads(loop_state.read_text()) == prior_loop


def test_corrupt_backup_is_rejected_before_rollback_mutation(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    patched = _snapshot(workspace)
    operation_id = json.loads(
        (state_dir / "current.json").read_text(encoding="utf-8")
    )["operation_id"]
    backup = state_dir / operation_id / "backup-package.json"
    backup.write_bytes(b"corrupt host backup\n")
    backup.chmod(0o600)

    completed = _run_patch("rollback", bundle, state_dir, environment)

    assert completed.returncode != 0
    assert _snapshot(workspace) == patched
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }


def test_forged_sandbox_backup_and_manifest_cannot_replace_host_trusted_backup(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    original = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    operation = next(
        (sandbox_root / "tmp/acs-prompt-reliability-20260821").iterdir()
    )
    forged = b"same-uid sandbox forgery\n"
    package_path = operation / "backup-package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["targets"]["runner"]["contents"]["."] = base64.b64encode(
        forged
    ).decode("ascii")
    package["targets"]["runner"]["descriptor"]["sha256"] = hashlib.sha256(
        forged
    ).hexdigest()
    package_path.write_bytes(_canonical(package))
    package_path.chmod(0o600)

    completed = _run_patch("rollback", bundle, state_dir, environment)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert _snapshot(workspace) == original
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


def test_restore_copy_failure_keeps_current_targets_and_allows_retry(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    original = _snapshot(workspace)
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"
    assert _run_patch("apply", bundle, state_dir, environment).returncode == 0
    patched = _snapshot(workspace)
    environment["ACS_FAKE_FAIL_RESTORE_COPY"] = "1"

    failed = _run_patch("rollback", bundle, state_dir, environment)

    assert failed.returncode != 0
    assert failed.stderr == ""
    assert json.loads(failed.stdout)["rollback"] is False
    assert _snapshot(workspace) == patched
    for relative in (
        "acs_workshop_runner.py",
        "TOOLS.md",
        ".acs-workshop-state/manifest.json",
        "outputs/workshop",
    ):
        assert (workspace / relative).exists()
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": True,
    }

    environment.pop("ACS_FAKE_FAIL_RESTORE_COPY")
    retried = _run_patch("rollback", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)
    assert _snapshot(workspace) == original
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


def test_patch_rejects_special_target_without_mutating_it(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, loop_state, sandbox_root = fake_environment
    target = sandbox_root / WORKSPACE / "TOOLS.md"
    target.unlink()
    os.mkfifo(target, mode=0o600)

    completed = _run_patch(
        "apply", _seed_bundle(tmp_path), tmp_path / "state", environment
    )

    assert completed.returncode != 0
    assert stat.S_ISFIFO(os.lstat(target).st_mode)
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }


@pytest.mark.parametrize("violation", ["oversize", "depth", "entry-count"])
def test_backup_bounds_reject_before_mutation_and_allow_exact_retry(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    violation: str,
) -> None:
    environment, command_log, loop_state, sandbox_root = fake_environment
    workspace = sandbox_root / WORKSPACE
    output = workspace / "outputs/workshop"
    bounded = output / "bounded-adversary"
    bounded.mkdir()
    if violation == "oversize":
        oversized = bounded / "oversized.bin"
        descriptor = os.open(oversized, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(descriptor, 64 * 1024 * 1024 + 1)
        finally:
            os.close(descriptor)
    elif violation == "depth":
        current = bounded
        for index in range(17):
            current = current / f"d{index:02d}"
            current.mkdir()
    else:
        for index in range(513):
            (bounded / f"entry-{index:03d}").write_bytes(b"")
    runner_before = (workspace / "acs_workshop_runner.py").read_bytes()
    bundle = _seed_bundle(tmp_path)
    state_dir = tmp_path / "state"

    rejected = _run_patch("apply", bundle, state_dir, environment)

    assert rejected.returncode == 70
    assert (workspace / "acs_workshop_runner.py").read_bytes() == runner_before
    assert bounded.exists()
    assert json.loads(loop_state.read_text()) == {
        "presence": "present",
        "value": False,
    }
    assert all(call[0] != "openshell" for call in _read_commands(command_log))

    shutil.rmtree(bounded)
    retried = _run_patch("apply", bundle, state_dir, environment)

    assert retried.returncode == 0, (retried.stdout, retried.stderr)


def test_qa_submits_exact_prompts_once_and_exports_closed_evidence(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, _, _ = fake_environment

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stderr == ""
    receipt = json.loads(completed.stdout)
    assert set(receipt) == {
        "exec_call_count",
        "objective_step_count",
        "prompt_count",
        "required_png_count",
        "results_zip_sha256",
        "results_zip_size",
        "schema_version",
        "status",
        "trajectory_sha256",
        "trajectory_size",
    }
    assert receipt["status"] == "pass"
    acceptance = output_dir / "acceptance.json"
    trajectory = output_dir / "11111111-1111-4111-8111-111111111111.trajectory.jsonl"
    archive = output_dir / "results.zip"
    assert acceptance.read_text(encoding="utf-8") == completed.stdout
    assert stat.S_IMODE(acceptance.stat().st_mode) == 0o600
    assert stat.S_IMODE(trajectory.stat().st_mode) == 0o600
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert not (
        output_dir / "11111111-1111-4111-8111-111111111111.jsonl"
    ).exists()
    calls = _read_commands(command_log)
    agent_calls = [call for call in calls if len(call) > 2 and call[2] == "agent"]
    assert len(agent_calls) == 4
    for call in agent_calls:
        assert call[:9] == (
            "nemoclaw",
            SANDBOX,
            "agent",
            "--session-id",
            "11111111-1111-4111-8111-111111111111",
            "--json",
            "--timeout",
            "600",
            "-m",
        )
        assert len(call) == 10
    observed_hashes = tuple(
        hashlib.sha256(call[call.index("-m") + 1].encode()).hexdigest()
        for call in agent_calls
    )
    assert observed_hashes == PROMPT_SHA256
    exports = [
        call for call in calls if len(call) > 3 and call[2:4] == ("sessions", "export")
    ]
    assert len(exports) == 1
    assert exports[0][4] == "agent:main:11111111-1111-4111-8111-111111111111"
    assert exports[0][5:7] == ("--include-trajectory", "--out")
    downloads = [
        call for call in calls if call[:3] == ("openshell", "sandbox", "download")
    ]
    assert len(downloads) == 1
    assert downloads[0][3:5] == (
        SANDBOX,
        "/sandbox/.openclaw/workspace/outputs/workshop/results.zip",
    )
    assert len(downloads[0]) == 6
    assert "answer-secret-canary" not in completed.stdout
    assert "exception-secret-canary" not in completed.stdout
    assert "11111111" not in completed.stdout
    assert str(output_dir) not in completed.stdout


@pytest.mark.parametrize(
    ("failure_mode", "expected_code"),
    [("timeout", 75), ("rc75", 70), ("failure", 70)],
)
def test_qa_stops_at_first_agent_failure_without_accepted_receipt(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    failure_mode: str,
    expected_code: int,
) -> None:
    environment, command_log, _, _ = fake_environment
    environment["ACS_FAKE_AGENT_FAILURE_INDEX"] = "2"
    environment["ACS_FAKE_AGENT_FAILURE_MODE"] = failure_mode

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == expected_code
    assert not (output_dir / "acceptance.json").exists()
    calls = _read_commands(command_log)
    agent_calls = [call for call in calls if len(call) > 2 and call[2] == "agent"]
    assert len(agent_calls) == 2
    assert not any("export" in call for call in calls)
    assert "answer-secret-canary" not in completed.stdout + completed.stderr
    receipt = json.loads(completed.stdout)
    assert set(receipt) == {"code", "schema_version", "status"}
    assert receipt["status"] == "fail"


def test_qa_rejects_wrong_trajectory_basename(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, _ = fake_environment
    environment["ACS_FAKE_EXPORT_OTHER"] = "1"

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert not (output_dir / "acceptance.json").exists()
    assert not list(output_dir.glob("*.trajectory.jsonl"))


def test_qa_rejects_syntactically_valid_but_semantically_wrong_trajectory(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, _ = fake_environment
    mutated = tmp_path / "semantically-wrong.jsonl"
    source = TRAJECTORY_FIXTURE.read_text(encoding="utf-8")
    changed = source.replace(
        "A larger `D_min` means the least separated pair",
        "A smaller `D_min` means the least separated pair",
        1,
    )
    assert changed != source
    mutated.write_text(changed, encoding="utf-8")
    environment["ACS_FAKE_TRAJECTORY_FIXTURE"] = str(mutated)

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert list(output_dir.iterdir()) == []


def test_qa_rejects_loadable_but_noncontract_archive(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, sandbox_root = fake_environment
    _write_invalid_zip(sandbox_root / WORKSPACE / "outputs/workshop/results.zip")

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("source", ["trajectory", "archive"])
def test_qa_rejects_unexpected_files_in_exact_export_directories(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    source: str,
) -> None:
    environment, _, _, _ = fake_environment
    if source == "trajectory":
        environment["ACS_FAKE_EXPORT_EXTRA"] = "1"
    else:
        environment["ACS_FAKE_DOWNLOAD_EXTRA"] = "1"

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "code": "qa_failed",
        "schema_version": 1,
        "status": "fail",
    }
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("mutation", ["mode", "symlink", "hardlink"])
def test_qa_rejects_unsafe_exact_session_export_companion(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    mutation: str,
) -> None:
    environment, _, _, _ = fake_environment
    environment["ACS_FAKE_SESSION_EXPORT_MUTATION"] = mutation

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "code": "qa_failed",
        "schema_version": 1,
        "status": "fail",
    }
    assert list(output_dir.iterdir()) == []


def test_qa_rejects_credential_names_and_symlinked_downloads(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, sandbox_root = fake_environment
    results = sandbox_root / WORKSPACE / "outputs/workshop/results.zip"
    _write_valid_zip(
        results,
        {"README.md": b'NVIDIA_INFERENCE_API_KEY="secret-canary"\n'},
    )

    credential_failure, first_output = _run_qa(tmp_path, environment)

    assert credential_failure.returncode == 70
    assert "secret-canary" not in credential_failure.stdout + credential_failure.stderr
    assert not (first_output / "acceptance.json").exists()

    environment["ACS_FAKE_AGENT_COUNTER"] = str(tmp_path / "second-agent-counter")
    environment["ACS_FAKE_DOWNLOAD_SYMLINK"] = "1"
    environment["ACS_FAKE_SYMLINK_TARGET"] = str(results)
    _write_valid_zip(results)
    second_output = tmp_path / "second-output"
    second_output.mkdir(mode=0o700)
    symlink_failure, _ = _run_qa(
        tmp_path,
        environment,
        output_dir=second_output,
        session_id="22222222-2222-4222-8222-222222222222",
    )

    assert symlink_failure.returncode == 70
    assert not (second_output / "acceptance.json").exists()


def test_qa_rejects_json_escaped_credential_key_after_full_verification(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, sandbox_root = fake_environment
    results = sandbox_root / WORKSPACE / "outputs/workshop/results.zip"
    _write_valid_zip(
        results,
        {"01-inspection/summary.json": b'{"api\\u005fkey":"secret-canary"}\n'},
    )

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert "secret-canary" not in completed.stdout + completed.stderr
    assert list(output_dir.iterdir()) == []


def test_qa_rejects_duplicate_keys_in_verified_json_member(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _, _, sandbox_root = fake_environment
    results = sandbox_root / WORKSPACE / "outputs/workshop/results.zip"
    _write_valid_zip(
        results,
        {"01-inspection/summary.json": b'{"safe":1,"safe":2}\n'},
    )

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("mutation", ["many-members", "forged-size"])
def test_qa_full_verifier_rejects_hostile_zip_before_credential_scan(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    mutation: str,
) -> None:
    environment, _, _, sandbox_root = fake_environment
    results = sandbox_root / WORKSPACE / "outputs/workshop/results.zip"
    if mutation == "many-members":
        with zipfile.ZipFile(results, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(2000):
                archive.writestr(f"empty-{index:04d}", b"")
        results.chmod(0o600)
    else:
        _write_valid_zip(results, {"README.md": b"x" * (9 * 1024 * 1024)})
        _forge_one_byte_declared_member(results)

    completed, output_dir = _run_qa(tmp_path, environment)

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("violation", ["uuid", "page-hash", "output-mode", "existing"])
def test_qa_rejects_invalid_inputs_before_first_submission(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
    violation: str,
) -> None:
    environment, command_log, _, _ = fake_environment
    session_id = "33333333-3333-4333-8333-333333333333"
    page = PAGE
    output = tmp_path / "qa-output"
    output.mkdir(mode=0o700)
    if violation == "uuid":
        session_id = "not-a-uuid"
    elif violation == "page-hash":
        page = tmp_path / "page.md"
        page.write_text(
            PAGE.read_text().replace("Scientific objective:", "Objective:", 1)
        )
    elif violation == "output-mode":
        output.chmod(0o755)
    else:
        (output / "acceptance.json").write_text("old\n", encoding="utf-8")

    completed, _ = _run_qa(
        tmp_path,
        environment,
        session_id=session_id,
        page=page,
        output_dir=output,
    )

    assert completed.returncode == 70
    assert not any(
        len(call) > 2 and call[2] == "agent" for call in _read_commands(command_log)
    )


def test_qa_rejects_stale_legacy_publication_temp_before_live_commands(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, _, _ = fake_environment
    output = tmp_path / "qa-output"
    output.mkdir(mode=0o700)
    stale = output / ".acceptance.json.tmp"
    stale.write_bytes(b"pre-poisoned\n")
    stale.chmod(0o600)

    completed, _ = _run_qa(tmp_path, environment, output_dir=output)

    assert completed.returncode == 70
    assert tuple(output.iterdir()) == (stale,)
    assert not any(
        len(call) > 2 and call[2] == "agent" for call in _read_commands(command_log)
    )


def test_qa_regular_reader_rejects_hardlinks(tmp_path: Path) -> None:
    module = _load_qa_module()
    source = tmp_path / "source"
    source.write_bytes(b"evidence\n")
    os.link(source, tmp_path / "second-name")

    with pytest.raises(module.QAError):
        module._read_regular(source, 64)


def test_qa_regular_reader_rejects_non_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_qa_module()
    source = tmp_path / "source"
    source.write_bytes(b"evidence\n")
    metadata = os.lstat(source)
    fake_metadata = SimpleNamespace(
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_mode=metadata.st_mode,
        st_nlink=metadata.st_nlink,
        st_size=metadata.st_size,
        st_uid=os.getuid() + 1,
    )
    monkeypatch.setattr(module, "_safe_lstat", lambda _path: fake_metadata)

    with pytest.raises(module.QAError):
        module._read_regular(source, 64)


def test_qa_regular_reader_rejects_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_qa_module()
    source = tmp_path / "source"
    source.write_bytes(b"original\n")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement\n")
    safe_lstat = module._safe_lstat

    def swap_after_lstat(path: Path) -> os.stat_result:
        metadata = safe_lstat(path)
        os.replace(replacement, path)
        return metadata

    monkeypatch.setattr(module, "_safe_lstat", swap_after_lstat)

    with pytest.raises(module.QAError):
        module._read_regular(source, 64)


def test_qa_trajectory_selection_stops_after_expected_count_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_qa_module()
    session_id = "66666666-6666-4666-8666-666666666666"
    tmp_path.chmod(0o700)
    directory = tmp_path / "export"
    directory.mkdir(mode=0o775)
    directory.chmod(0o775)
    session = directory / f"{session_id}.jsonl"
    trajectory = directory / f"{session_id}.trajectory.jsonl"
    extra = directory / "extra.jsonl"
    for path in (session, trajectory, extra):
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):  # type: ignore[no-untyped-def]
        if path != directory:
            yield from original_iterdir(path)
            return
        yield session
        yield trajectory
        yield extra
        raise AssertionError("directory enumeration exceeded the closed bound")

    monkeypatch.setattr(type(directory), "iterdir", guarded_iterdir)

    with pytest.raises(module.QAError):
        module._select_exact_trajectory(directory, session_id)


@pytest.mark.parametrize("provider_timeout", [False, True])
def test_qa_host_process_timeout_is_provider_specific(
    monkeypatch: pytest.MonkeyPatch, provider_timeout: bool
) -> None:
    module = _load_qa_module()

    def expire(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["closed-command"], 1)

    monkeypatch.setattr(module.subprocess, "run", expire)
    expected = module.QATimeout if provider_timeout else module.QAError
    with pytest.raises(expected) as raised:
        module._run_quiet(
            ("closed-command",), timeout=1, provider_timeout=provider_timeout
        )
    if not provider_timeout:
        assert not isinstance(raised.value, module.QATimeout)


def test_qa_credential_scan_checks_the_exact_archive_bytes(tmp_path: Path) -> None:
    module = _load_qa_module()
    archive = tmp_path / "secret.zip"
    _write_valid_zip(archive, {"README.md": b'api_key="secret-canary"\n'})

    with pytest.raises(module.QAError):
        module._scan_for_credentials(
            b"safe trajectory\n", archive.read_bytes(), REQUIRED_ZIP_MEMBERS
        )


def test_qa_missing_verifier_dependency_still_emits_closed_failure(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, _, _ = fake_environment
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    isolated_script = isolated / QA_SCRIPT.name
    shutil.copy2(QA_SCRIPT, isolated_script)
    output = isolated / "output"
    output.mkdir(mode=0o700)
    isolated_environment = environment.copy()
    isolated_environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(isolated_script),
            "--session-id",
            "44444444-4444-4444-8444-444444444444",
            "--page",
            str(PAGE),
            "--output-dir",
            str(output),
            "--sandbox",
            SANDBOX,
        ],
        cwd=isolated,
        env=isolated_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "code": "qa_failed",
        "schema_version": 1,
        "status": "fail",
    }
    assert not any(
        len(call) > 2 and call[2] == "agent" for call in _read_commands(command_log)
    )


def test_qa_broken_verifier_api_fails_before_first_live_command(
    tmp_path: Path,
    fake_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, command_log, _, _ = fake_environment
    isolated = tmp_path / "broken-verifier"
    isolated.mkdir()
    shutil.copy2(QA_SCRIPT, isolated / QA_SCRIPT.name)
    (isolated / "verify_acs_openclaw_trajectory.py").write_text(
        "class VerificationError(RuntimeError):\n"
        "    pass\n"
        "REQUIRED_ZIP_MEMBERS = tuple(str(i) for i in range(34))\n"
        "def load_prompt_contracts(page_path):\n"
        "    return ()\n",
        encoding="utf-8",
    )
    output = isolated / "output"
    output.mkdir(mode=0o700)
    completed = subprocess.run(
        [
            sys.executable,
            str(isolated / QA_SCRIPT.name),
            "--session-id",
            "55555555-5555-4555-8555-555555555555",
            "--page",
            str(PAGE),
            "--output-dir",
            str(output),
            "--sandbox",
            SANDBOX,
        ],
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 70
    assert completed.stderr == ""
    assert not any(
        len(call) > 2 and call[2] == "agent" for call in _read_commands(command_log)
    )
