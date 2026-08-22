#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
exec 2>/dev/null

readonly expected_sandbox="acs-chemistry-agent"
readonly workspace="/sandbox/.openclaw/workspace"
readonly sandbox_state_root="/tmp/acs-prompt-reliability-20260821"
readonly loop_key="tools.loopDetection.enabled"
readonly missing_loop_diagnostic="Config path not found: tools.loopDetection.enabled. Run openclaw config validate to inspect config shape."

emit_preflight_failure() {
  printf '%s\n' '{"code":"preflight_failed","main_session_touched":false,"rollback":false,"schema_version":1,"status":"fail"}'
}

mode=""
bundle_dir=""
state_dir=""
sandbox=""
while (( $# > 0 )); do
  case "$1" in
    --mode)
      [[ -z "${mode}" && $# -ge 2 ]] || {
        emit_preflight_failure
        exit 70
      }
      mode="$2"
      shift 2
      ;;
    --bundle-dir)
      [[ -z "${bundle_dir}" && $# -ge 2 ]] || {
        emit_preflight_failure
        exit 70
      }
      bundle_dir="$2"
      shift 2
      ;;
    --state-dir)
      [[ -z "${state_dir}" && $# -ge 2 ]] || {
        emit_preflight_failure
        exit 70
      }
      state_dir="$2"
      shift 2
      ;;
    --sandbox)
      [[ -z "${sandbox}" && $# -ge 2 ]] || {
        emit_preflight_failure
        exit 70
      }
      sandbox="$2"
      shift 2
      ;;
    *)
      emit_preflight_failure
      exit 70
      ;;
  esac
done

[[ "${mode}" == "apply" || "${mode}" == "rollback" || "${mode}" == "reset-between-qa" ]] || {
  emit_preflight_failure
  exit 70
}
[[ "${bundle_dir}" == /* && "${state_dir}" == /* ]] || {
  emit_preflight_failure
  exit 70
}
[[ "${sandbox}" == "${expected_sandbox}" ]] || {
  emit_preflight_failure
  exit 70
}
[[ "$(id -un)" == "ubuntu" ]] || {
  emit_preflight_failure
  exit 70
}

nemoclaw="$(command -v nemoclaw || true)"
python="$(command -v python3 || true)"
openshell="$(command -v openshell || true)"
host_uid="$(id -u || true)"
host_tmp="$("${python}" -c 'from pathlib import Path; print(Path("/tmp").resolve(strict=True))' 2>/dev/null || true)"
[[ -n "${nemoclaw}" && -n "${python}" && -n "${openshell}" && "${host_uid}" =~ ^[0-9]+$ && "${host_tmp}" == /* ]] || {
  emit_preflight_failure
  exit 70
}
readonly host_global_state="${host_tmp}/acs-prompt-reliability-20260821-host-${host_uid}"
readonly nemoclaw openshell python host_uid host_tmp

read -r -d '' host_helper <<'PY' || true
import base64
import binascii
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path

BUNDLE_FILES = (
    "acs_workshop_runner.py",
    "launchable/acs_workspace_tools.md",
    "scripts/verify_acs_openclaw_trajectory.py",
    "scripts/acs_live_instance_patch.sh",
    "scripts/run_acs_openclaw_live_qa.py",
    "docs/acs-fall-2026-workshop.md",
)
BACKUP_LABELS = ("context", "history", "manifest", "outputs", "runner", "tools")
HEX = re.compile(r"[0-9a-f]{64}\Z")
OPERATION = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 512
MAX_DEPTH = 16
MAX_PACKAGE_BYTES = 176 * 1024 * 1024


def closed(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=closed,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def encoded(payload):
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def safe_components(path, allow_missing_leaf=False):
    if not path.is_absolute():
        raise ValueError
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise
        if stat.S_ISLNK(mode) or (index < len(parts) - 1 and not stat.S_ISDIR(mode)):
            raise ValueError


def regular(path, *, mode=None, limit=MAX_PACKAGE_BYTES):
    safe_components(path)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid() or before.st_size > limit:
        raise ValueError
    if mode is not None and stat.S_IMODE(before.st_mode) != mode:
        raise ValueError
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_uid, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_uid, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != identity:
            raise ValueError
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_uid, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise ValueError
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_raw(path, raw, *, exclusive=False):
    temporary = path.with_name("." + path.name + ".tmp-" + str(uuid.uuid4()))
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if regular(temporary, mode=0o600) != raw:
            raise ValueError
        if exclusive:
            os.link(temporary, path, follow_symlinks=False)
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic(path, payload):
    write_raw(path, encoded(payload))


def exclusive(path, payload):
    write_raw(path, encoded(payload), exclusive=True)


def prepare_directory(path):
    safe_components(path, allow_missing_leaf=True)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        os.mkdir(path, 0o700)
        metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.getuid():
        raise ValueError


def validate_bundle(path):
    prepare_directory(path)
    raw = regular(path / "bundle-manifest.json", mode=0o600)
    payload = decode(raw)
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != {"schema_version", "files"} or payload["schema_version"] != 1:
        raise ValueError
    files = payload["files"]
    if type(files) is not dict or tuple(files) != tuple(sorted(BUNDLE_FILES)) or set(files) != set(BUNDLE_FILES):
        raise ValueError
    observed = {}
    for relative in BUNDLE_FILES:
        expected = files[relative]
        if type(expected) is not str or HEX.fullmatch(expected) is None:
            raise ValueError
        target = path / relative
        raw_file = regular(target)
        if target.resolve().is_relative_to(path.resolve()) is False:
            raise ValueError
        digest = hashlib.sha256(raw_file).hexdigest()
        if digest != expected:
            raise ValueError
        observed[relative] = digest
    print(observed["acs_workshop_runner.py"] + "\t" + observed["launchable/acs_workspace_tools.md"])


def valid_mode(value):
    return type(value) is int and 0 <= value <= 0o7777


def validate_descriptor(descriptor):
    if type(descriptor) is not dict or type(descriptor.get("present")) is not bool:
        raise ValueError
    if descriptor["present"] is False:
        if descriptor != {"present": False}:
            raise ValueError
        return ()
    if descriptor.get("kind") == "file":
        if set(descriptor) != {"kind", "mode", "present", "sha256"} or not valid_mode(descriptor["mode"]) or type(descriptor["sha256"]) is not str or HEX.fullmatch(descriptor["sha256"]) is None:
            raise ValueError
        return (".",)
    if descriptor.get("kind") != "directory" or set(descriptor) != {"entries", "kind", "present"} or type(descriptor["entries"]) is not dict:
        raise ValueError
    entries = descriptor["entries"]
    if not 1 <= len(entries) <= MAX_ENTRIES or "." not in entries:
        raise ValueError
    files = []
    for relative, item in entries.items():
        if type(relative) is not str or type(item) is not dict:
            raise ValueError
        candidate = Path(relative)
        if relative != "." and (candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative or len(candidate.parts) > MAX_DEPTH):
            raise ValueError
        if item.get("kind") == "dir":
            if set(item) != {"kind", "mode"} or not valid_mode(item["mode"]):
                raise ValueError
        elif item.get("kind") == "file":
            if set(item) != {"kind", "mode", "sha256"} or not valid_mode(item["mode"]) or type(item["sha256"]) is not str or HEX.fullmatch(item["sha256"]) is None:
                raise ValueError
            files.append(relative)
        else:
            raise ValueError
        if relative == "." and item.get("kind") != "dir":
            raise ValueError
        if relative != ".":
            parent = candidate.parent.as_posix()
            if parent == "":
                parent = "."
            if parent not in entries or entries[parent].get("kind") != "dir":
                raise ValueError
    return tuple(sorted(files))


def validate_package(raw):
    payload = decode(raw)
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != {"schema_version", "targets"} or payload["schema_version"] != 1:
        raise ValueError
    targets = payload["targets"]
    if type(targets) is not dict or tuple(targets) != BACKUP_LABELS:
        raise ValueError
    total_bytes = 0
    total_entries = 0
    for label in BACKUP_LABELS:
        target = targets[label]
        if type(target) is not dict or set(target) != {"contents", "descriptor"} or type(target["contents"]) is not dict:
            raise ValueError
        expected_files = validate_descriptor(target["descriptor"])
        contents = target["contents"]
        if tuple(contents) != expected_files:
            raise ValueError
        if target["descriptor"]["present"] is not False:
            total_entries += 1 if target["descriptor"]["kind"] == "file" else len(target["descriptor"]["entries"])
        for relative in expected_files:
            value = contents[relative]
            if type(value) is not str:
                raise ValueError
            try:
                raw_file = base64.b64decode(value.encode("ascii"), validate=True)
            except (UnicodeError, binascii.Error) as error:
                raise ValueError from error
            if len(raw_file) > MAX_FILE_BYTES:
                raise ValueError
            total_bytes += len(raw_file)
            if total_bytes > MAX_TOTAL_BYTES:
                raise ValueError
            descriptor = target["descriptor"]
            expected_hash = descriptor["sha256"] if descriptor["kind"] == "file" else descriptor["entries"][relative]["sha256"]
            if hashlib.sha256(raw_file).hexdigest() != expected_hash:
                raise ValueError
    if total_entries > MAX_ENTRIES:
        raise ValueError
    return payload


def package_digest(path):
    raw = regular(path, mode=0o600, limit=MAX_PACKAGE_BYTES)
    validate_package(raw)
    return hashlib.sha256(raw).hexdigest()


def state_identity(state):
    prepare_directory(state)
    metadata = os.lstat(state)
    return {
        "state_dev": metadata.st_dev,
        "state_ino": metadata.st_ino,
        "state_path": str(state.resolve(strict=True)),
    }


def expected_reservation(state, sandbox, operation):
    identity = state_identity(state)
    return {
        "backup_sha256": None,
        "operation_id": operation,
        "phase": "preparing",
        "sandbox": sandbox,
        "schema_version": 1,
        **identity,
    }


def expected_active(state, sandbox, operation, backup_hash):
    payload = expected_reservation(state, sandbox, operation)
    payload["backup_sha256"] = backup_hash
    payload["phase"] = "active"
    return payload


def load_active(global_state):
    path = global_state / "active.json"
    try:
        raw = regular(path, mode=0o600)
    except FileNotFoundError:
        return None
    payload = decode(raw)
    expected_keys = {"backup_sha256", "operation_id", "phase", "sandbox", "schema_version", "state_dev", "state_ino", "state_path"}
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != expected_keys or payload["schema_version"] != 1 or type(payload["sandbox"]) is not str or type(payload["state_path"]) is not str or type(payload["state_dev"]) is not int or type(payload["state_ino"]) is not int or type(payload["operation_id"]) is not str or OPERATION.fullmatch(payload["operation_id"]) is None or payload["phase"] not in {"preparing", "active"}:
        raise ValueError
    if payload["phase"] == "preparing":
        if payload["backup_sha256"] is not None:
            raise ValueError
    elif type(payload["backup_sha256"]) is not str or HEX.fullmatch(payload["backup_sha256"]) is None:
        raise ValueError
    return payload


def require_free(global_state):
    prepare_directory(global_state)
    if load_active(global_state) is not None:
        raise ValueError


def create_operation(state, operation, presence, value, runner_hash, tools_hash):
    if OPERATION.fullmatch(operation) is None or presence not in {"absent", "present"} or HEX.fullmatch(runner_hash) is None or HEX.fullmatch(tools_hash) is None:
        raise ValueError
    current = state / "current.json"
    if current.exists() or current.is_symlink():
        raise ValueError
    child = state / operation
    os.mkdir(child, 0o700)
    loop = {"presence": presence}
    if presence == "present":
        if value not in {"true", "false"}:
            raise ValueError
        loop["value"] = value == "true"
    elif value != "none":
        raise ValueError
    payload = {
        "backup_sha256": None,
        "loop_state": loop,
        "operation_id": operation,
        "phase": "prepared",
        "rolled_back": False,
        "runner_hash": runner_hash,
        "schema_version": 1,
        "tools_hash": tools_hash,
    }
    atomic(child / "operation.json", payload)
    atomic(current, {"operation_id": operation, "schema_version": 1})


def reserve_operation(state, global_state, sandbox, operation):
    _path, payload = operation_payload(state, operation)
    if payload["phase"] != "prepared" or payload["rolled_back"] is not False:
        raise ValueError
    require_free(global_state)
    exclusive(global_state / "active.json", expected_reservation(state, sandbox, operation))


def operation_payload(state, operation=None):
    prepare_directory(state)
    current_raw = regular(state / "current.json", mode=0o600)
    current = decode(current_raw)
    if encoded(current) != current_raw or type(current) is not dict or set(current) != {"operation_id", "schema_version"} or current["schema_version"] != 1:
        raise ValueError
    observed_operation = current["operation_id"]
    if type(observed_operation) is not str or OPERATION.fullmatch(observed_operation) is None or (operation is not None and observed_operation != operation):
        raise ValueError
    child = state / observed_operation
    child_mode = os.lstat(child).st_mode
    if not stat.S_ISDIR(child_mode) or stat.S_IMODE(child_mode) != 0o700 or os.lstat(child).st_uid != os.getuid():
        raise ValueError
    path = child / "operation.json"
    raw = regular(path, mode=0o600)
    payload = decode(raw)
    expected_keys = {"backup_sha256", "loop_state", "operation_id", "phase", "rolled_back", "runner_hash", "schema_version", "tools_hash"}
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != expected_keys or payload["schema_version"] != 1 or payload["operation_id"] != observed_operation or payload["phase"] not in {"prepared", "backup_ready"} or type(payload["rolled_back"]) is not bool or type(payload["runner_hash"]) is not str or HEX.fullmatch(payload["runner_hash"]) is None or type(payload["tools_hash"]) is not str or HEX.fullmatch(payload["tools_hash"]) is None:
        raise ValueError
    if payload["phase"] == "prepared":
        if payload["backup_sha256"] is not None or payload["rolled_back"] is not False:
            raise ValueError
    elif type(payload["backup_sha256"]) is not str or HEX.fullmatch(payload["backup_sha256"]) is None:
        raise ValueError
    loop = payload["loop_state"]
    if type(loop) is not dict or loop.get("presence") not in {"absent", "present"}:
        raise ValueError
    if loop["presence"] == "absent":
        if set(loop) != {"presence"}:
            raise ValueError
    elif set(loop) != {"presence", "value"} or type(loop["value"]) is not bool:
        raise ValueError
    return path, payload


def anchor_activate(state, global_state, sandbox, operation, remote_hash):
    if HEX.fullmatch(remote_hash) is None:
        raise ValueError
    path, payload = operation_payload(state, operation)
    if payload["phase"] != "prepared":
        raise ValueError
    observed_hash = package_digest(state / operation / "backup-package.json")
    if observed_hash != remote_hash:
        raise ValueError
    reservation = expected_reservation(state, sandbox, operation)
    active = expected_active(state, sandbox, operation, observed_hash)
    if load_active(global_state) != reservation:
        raise ValueError
    atomic(global_state / "active.json", active)
    try:
        payload["backup_sha256"] = observed_hash
        payload["phase"] = "backup_ready"
        atomic(path, payload)
    except BaseException:
        if load_active(global_state) == active:
            atomic(global_state / "active.json", reservation)
        raise
    print(observed_hash)


def load_operation(state, global_state, sandbox):
    _, payload = operation_payload(state)
    if payload["phase"] != "backup_ready":
        raise ValueError
    operation = payload["operation_id"]
    backup_hash = package_digest(state / operation / "backup-package.json")
    if backup_hash != payload["backup_sha256"]:
        raise ValueError
    expected = expected_active(state, sandbox, operation, backup_hash)
    active = load_active(global_state)
    if payload["rolled_back"] is False:
        if active != expected:
            raise ValueError
    elif active is not None and active != expected:
        raise ValueError
    loop = payload["loop_state"]
    value = "none" if loop["presence"] == "absent" else ("true" if loop["value"] else "false")
    print("\t".join((operation, loop["presence"], value, "true" if payload["rolled_back"] else "false", payload["runner_hash"], payload["tools_hash"], backup_hash)))


def clear_prepared(state, global_state, sandbox, operation):
    path, payload = operation_payload(state, operation)
    if payload["phase"] != "prepared" or payload["rolled_back"] is not False:
        raise ValueError
    active = load_active(global_state)
    reservation = expected_reservation(state, sandbox, operation)
    if active == reservation:
        (global_state / "active.json").unlink()
    elif active is not None and active.get("operation_id") == operation and active.get("state_path") == reservation["state_path"]:
        raise ValueError
    package = state / operation / "backup-package.json"
    try:
        metadata = os.lstat(package)
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != os.getuid():
            raise ValueError
        package.unlink()
    (state / "current.json").unlink()
    path.unlink()
    (state / operation).rmdir()


def complete_rollback(state, global_state, sandbox, operation):
    path, payload = operation_payload(state, operation)
    if payload["phase"] != "backup_ready":
        raise ValueError
    backup_hash = package_digest(state / operation / "backup-package.json")
    if backup_hash != payload["backup_sha256"]:
        raise ValueError
    expected = expected_active(state, sandbox, operation, backup_hash)
    active = load_active(global_state)
    if active is not None and active != expected:
        raise ValueError
    if payload["rolled_back"] is False and active != expected:
        raise ValueError
    if payload["rolled_back"] is False:
        payload["rolled_back"] = True
        atomic(path, payload)
    if load_active(global_state) == expected:
        (global_state / "active.json").unlink()


def parse_loop(stdout_path, stderr_path, status, expected):
    stdout = regular(stdout_path, mode=0o600)
    stderr = regular(stderr_path, mode=0o600)
    if status == "0":
        if stdout == b"true\n":
            print("present\ttrue")
            return
        if stdout == b"false\n":
            print("present\tfalse")
            return
        raise ValueError
    gateway = b"\x1b[1m\x1b[32m\xe2\x9c\x93\x1b[39m\x1b[0m Active gateway set to 'nemoclaw'\n"
    expected_stderr = gateway + expected.encode("utf-8") + b"\n"
    if status != "1" or stdout != b"" or stderr != expected_stderr:
        raise ValueError
    print("absent\tnone")


action = sys.argv[1]
if action == "prepare":
    prepare_directory(Path(sys.argv[2]))
elif action == "prepare-global":
    prepare_directory(Path(sys.argv[2]))
elif action == "free":
    require_free(Path(sys.argv[2]))
elif action == "bundle":
    validate_bundle(Path(sys.argv[2]))
elif action == "create":
    create_operation(Path(sys.argv[2]), *sys.argv[3:])
elif action == "reserve":
    reserve_operation(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
elif action == "anchor-activate":
    anchor_activate(Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:])
elif action == "load":
    load_operation(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
elif action == "complete":
    complete_rollback(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
elif action == "clear-prepared":
    clear_prepared(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
elif action == "parse-loop":
    parse_loop(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
else:
    raise ValueError
PY
readonly host_helper

read -r -d '' sandbox_helper <<'PY' || true
import base64
import binascii
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path

WORKSPACE = Path("/sandbox/.openclaw/workspace")
BASE = Path("/tmp/acs-prompt-reliability-20260821")
OPERATION = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 512
MAX_DEPTH = 16
MAX_PACKAGE_BYTES = 176 * 1024 * 1024
PROTECTED = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
    "data/PROVENANCE.md",
    "objective_challenge.py",
)
BACKUP_TARGETS = {
    "context": WORKSPACE / ".acs-workshop-state/context.json",
    "history": WORKSPACE / ".acs-workshop-state/history.json",
    "manifest": WORKSPACE / ".acs-workshop-state/manifest.json",
    "outputs": WORKSPACE / "outputs/workshop",
    "runner": WORKSPACE / "acs_workshop_runner.py",
    "tools": WORKSPACE / "TOOLS.md",
}


class Budget:
    def __init__(self):
        self.entries = 0
        self.total_bytes = 0

    def entry(self, depth):
        if depth > MAX_DEPTH:
            raise ValueError
        self.entries += 1
        if self.entries > MAX_ENTRIES:
            raise ValueError

    def file(self, size):
        if size > MAX_FILE_BYTES:
            raise ValueError
        self.total_bytes += size
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise ValueError


def closed(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=closed,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def encoded(payload):
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def safe_components(path, allow_missing_leaf=False):
    if not path.is_absolute():
        raise ValueError
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise
        if stat.S_ISLNK(mode) or (index < len(parts) - 1 and not stat.S_ISDIR(mode)):
            raise ValueError


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_regular(path, *, limit=MAX_FILE_BYTES, budget=None):
    safe_components(path)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid() or before.st_size > limit:
        raise ValueError
    if budget is not None:
        budget.file(before.st_size)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_uid, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_uid, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != identity:
            raise ValueError
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_uid, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise ValueError
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def safe_regular(path, *, limit=MAX_FILE_BYTES):
    return read_regular(path, limit=limit)[0]


def prepare_bytes(path, raw, mode, purpose, *, limit=MAX_FILE_BYTES):
    if len(raw) > limit:
        raise ValueError
    safe_components(path.parent)
    safe_components(path, allow_missing_leaf=True)
    temporary = path.with_name("." + path.name + "." + purpose + "-" + str(uuid.uuid4()))
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError
            remaining = remaining[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        observed, metadata = read_regular(temporary, limit=limit)
        if observed != raw or stat.S_IMODE(metadata.st_mode) != mode:
            raise ValueError
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_bytes(path, raw, mode, *, limit=MAX_FILE_BYTES):
    temporary = prepare_bytes(path, raw, mode, "patch-tmp", limit=limit)
    try:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture(path, budget=None):
    if budget is None:
        budget = Budget()
    try:
        safe_components(path, allow_missing_leaf=True)
        metadata = os.lstat(path)
    except FileNotFoundError:
        return {"present": False}, {}
    if metadata.st_uid != os.getuid() or stat.S_ISLNK(metadata.st_mode):
        raise ValueError
    if stat.S_ISREG(metadata.st_mode):
        budget.entry(0)
        raw, opened = read_regular(path, budget=budget)
        descriptor = {"kind": "file", "mode": stat.S_IMODE(opened.st_mode), "present": True, "sha256": hashlib.sha256(raw).hexdigest()}
        return descriptor, {".": raw}
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError
    budget.entry(0)
    entries = {".": {"kind": "dir", "mode": stat.S_IMODE(metadata.st_mode)}}
    contents = {}
    pending = [(path, Path("."), 0)]
    while pending:
        current, relative_root, depth = pending.pop()
        children = []
        with os.scandir(current) as iterator:
            for child in iterator:
                child_metadata = child.stat(follow_symlinks=False)
                budget.entry(depth + 1)
                if child_metadata.st_uid != os.getuid() or stat.S_ISLNK(child_metadata.st_mode) or not (stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISREG(child_metadata.st_mode)):
                    raise ValueError
                children.append((child.name, child_metadata))
        for name, child_metadata in sorted(children, reverse=True):
            child = current / name
            relative = Path(name) if relative_root == Path(".") else relative_root / name
            relative_text = relative.as_posix()
            if stat.S_ISDIR(child_metadata.st_mode):
                entries[relative_text] = {"kind": "dir", "mode": stat.S_IMODE(child_metadata.st_mode)}
                pending.append((child, relative, depth + 1))
            else:
                raw, opened = read_regular(child, budget=budget)
                entries[relative_text] = {"kind": "file", "mode": stat.S_IMODE(opened.st_mode), "sha256": hashlib.sha256(raw).hexdigest()}
                contents[relative_text] = raw
    return {"entries": entries, "kind": "directory", "present": True}, contents


def description(path):
    return capture(path)[0]


def valid_mode(value):
    return type(value) is int and 0 <= value <= 0o7777


def validate_descriptor(descriptor):
    if type(descriptor) is not dict or type(descriptor.get("present")) is not bool:
        raise ValueError
    if descriptor["present"] is False:
        if descriptor != {"present": False}:
            raise ValueError
        return ()
    if descriptor.get("kind") == "file":
        if set(descriptor) != {"kind", "mode", "present", "sha256"} or not valid_mode(descriptor["mode"]) or type(descriptor["sha256"]) is not str or HEX.fullmatch(descriptor["sha256"]) is None:
            raise ValueError
        return (".",)
    if descriptor.get("kind") != "directory" or set(descriptor) != {"entries", "kind", "present"} or type(descriptor["entries"]) is not dict:
        raise ValueError
    entries = descriptor["entries"]
    if not 1 <= len(entries) <= MAX_ENTRIES or "." not in entries:
        raise ValueError
    files = []
    for relative, item in entries.items():
        if type(relative) is not str or type(item) is not dict:
            raise ValueError
        candidate = Path(relative)
        if relative != "." and (candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative or len(candidate.parts) > MAX_DEPTH):
            raise ValueError
        if item.get("kind") == "dir":
            if set(item) != {"kind", "mode"} or not valid_mode(item["mode"]):
                raise ValueError
        elif item.get("kind") == "file":
            if set(item) != {"kind", "mode", "sha256"} or not valid_mode(item["mode"]) or type(item["sha256"]) is not str or HEX.fullmatch(item["sha256"]) is None:
                raise ValueError
            files.append(relative)
        else:
            raise ValueError
        if relative == "." and item.get("kind") != "dir":
            raise ValueError
        if relative != ".":
            parent = candidate.parent.as_posix()
            if parent == "":
                parent = "."
            if parent not in entries or entries[parent].get("kind") != "dir":
                raise ValueError
    return tuple(sorted(files))


def load_package(path, expected_hash=None):
    raw = safe_regular(path, limit=MAX_PACKAGE_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError
    payload = decode(raw)
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != {"schema_version", "targets"} or payload["schema_version"] != 1 or type(payload["targets"]) is not dict or tuple(payload["targets"]) != tuple(BACKUP_TARGETS):
        raise ValueError
    budget = Budget()
    decoded_targets = {}
    for label, target in payload["targets"].items():
        if type(target) is not dict or set(target) != {"contents", "descriptor"} or type(target["contents"]) is not dict:
            raise ValueError
        descriptor = target["descriptor"]
        expected_files = validate_descriptor(descriptor)
        contents = target["contents"]
        if tuple(contents) != expected_files:
            raise ValueError
        if descriptor["present"] is not False:
            for relative in ((".",) if descriptor["kind"] == "file" else tuple(descriptor["entries"])):
                depth = 0 if relative == "." else len(Path(relative).parts)
                budget.entry(depth)
        raw_contents = {}
        for relative in expected_files:
            value = contents[relative]
            if type(value) is not str:
                raise ValueError
            try:
                content = base64.b64decode(value.encode("ascii"), validate=True)
            except (UnicodeError, binascii.Error) as error:
                raise ValueError from error
            budget.file(len(content))
            expected = descriptor["sha256"] if descriptor["kind"] == "file" else descriptor["entries"][relative]["sha256"]
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError
            raw_contents[relative] = content
        decoded_targets[label] = {"contents": raw_contents, "descriptor": descriptor}
    return digest, decoded_targets


def operation_root(operation):
    if OPERATION.fullmatch(operation) is None:
        raise ValueError
    return BASE / operation


def ensure_directory(path):
    safe_components(path, allow_missing_leaf=True)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        path.mkdir(mode=0o700)
        metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.getuid():
        raise ValueError


def ensure_base():
    safe_components(BASE.parent)
    if not stat.S_ISDIR(os.lstat(BASE.parent).st_mode):
        raise ValueError
    ensure_directory(BASE)


def verify_protected_manifest():
    state = WORKSPACE / ".acs-workshop-state"
    safe_components(state)
    state_metadata = os.lstat(state)
    if not stat.S_ISDIR(state_metadata.st_mode) or stat.S_IMODE(state_metadata.st_mode) != 0o700 or state_metadata.st_uid != os.getuid():
        raise ValueError
    raw, manifest_metadata = read_regular(state / "manifest.json")
    if stat.S_IMODE(manifest_metadata.st_mode) != 0o444:
        raise ValueError
    payload = decode(raw)
    if encoded(payload) != raw or type(payload) is not dict or set(payload) != {"files", "schema_version"} or payload["schema_version"] != 1 or type(payload["files"]) is not dict or set(payload["files"]) != set(PROTECTED):
        raise ValueError
    for relative in PROTECTED:
        digest = payload["files"][relative]
        protected_raw, protected_metadata = read_regular(WORKSPACE / relative)
        if type(digest) is not str or HEX.fullmatch(digest) is None or stat.S_IMODE(protected_metadata.st_mode) != 0o444 or hashlib.sha256(protected_raw).hexdigest() != digest:
            raise ValueError


def backup(operation):
    verify_protected_manifest()
    ensure_base()
    root = operation_root(operation)
    root.mkdir(mode=0o700)
    (root / "stage").mkdir(mode=0o700)
    (root / "quarantine").mkdir(mode=0o700)
    (root / "restore").mkdir(mode=0o700)
    budget = Budget()
    targets = {}
    for label, source in BACKUP_TARGETS.items():
        descriptor, contents = capture(source, budget)
        targets[label] = {
            "contents": {relative: base64.b64encode(raw).decode("ascii") for relative, raw in contents.items()},
            "descriptor": descriptor,
        }
    raw = encoded({"schema_version": 1, "targets": targets})
    atomic_bytes(root / "backup-package.json", raw, 0o600, limit=MAX_PACKAGE_BYTES)
    digest, decoded_targets = load_package(root / "backup-package.json")
    for label, source in BACKUP_TARGETS.items():
        if description(source) != decoded_targets[label]["descriptor"]:
            raise ValueError
    print(digest)


def prepare_restore(operation, restore_id):
    if OPERATION.fullmatch(restore_id) is None:
        raise ValueError
    root = operation_root(operation)
    ensure_directory(root)
    restore = root / "restore"
    ensure_directory(restore)
    child = restore / restore_id
    child.mkdir(mode=0o700)


def restore_package(operation, restore_id, expected_hash):
    if OPERATION.fullmatch(restore_id) is None or HEX.fullmatch(expected_hash) is None:
        raise ValueError
    path = operation_root(operation) / "restore" / restore_id / "backup-package.json"
    return load_package(path, expected_hash)[1]


def privatize(path, descriptor):
    if descriptor["kind"] == "file":
        safe_regular(path)
        os.chmod(path, 0o600)
        return
    for relative, item in sorted(descriptor["entries"].items(), key=lambda pair: pair[0].count("/"), reverse=True):
        target = path if relative == "." else path / relative
        os.chmod(target, 0o600 if item["kind"] == "file" else 0o700)


def quarantine_targets(operation, label):
    destination = operation_root(operation) / "quarantine" / label
    destination.mkdir(mode=0o700)
    for name in ("outputs", "context", "history"):
        target = BACKUP_TARGETS[name]
        descriptor = description(target)
        if descriptor["present"] is not False:
            quarantined = destination / name
            os.replace(target, quarantined)
            fsync_directory(target.parent)
            privatize(quarantined, descriptor)
    (WORKSPACE / "outputs/workshop").mkdir(mode=0o755)
    fsync_directory(WORKSPACE / "outputs")


def rebuild_manifest():
    files = {relative: hashlib.sha256(safe_regular(WORKSPACE / relative)).hexdigest() for relative in PROTECTED}
    state = WORKSPACE / ".acs-workshop-state"
    state_mode = os.lstat(state).st_mode
    if not stat.S_ISDIR(state_mode) or stat.S_IMODE(state_mode) != 0o700:
        raise ValueError
    atomic_bytes(state / "manifest.json", encoded({"files": files, "schema_version": 1}), 0o444)


def directory_is_empty(path):
    safe_components(path)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError
    with os.scandir(path) as iterator:
        return next(iterator, None) is None


def verify_installed(runner_hash, tools_hash):
    if HEX.fullmatch(runner_hash) is None or HEX.fullmatch(tools_hash) is None:
        raise ValueError
    verify_protected_manifest()
    for relative, digest in {"acs_workshop_runner.py": runner_hash, "TOOLS.md": tools_hash}.items():
        raw, metadata = read_regular(WORKSPACE / relative)
        if hashlib.sha256(raw).hexdigest() != digest or stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ValueError
    if not directory_is_empty(WORKSPACE / "outputs/workshop") or description(WORKSPACE / ".acs-workshop-state/context.json")["present"] is not False or description(WORKSPACE / ".acs-workshop-state/history.json")["present"] is not False:
        raise ValueError


def verify_sources(runner_hash, tools_hash):
    verify_protected_manifest()
    for relative, digest in {"acs_workshop_runner.py": runner_hash, "TOOLS.md": tools_hash}.items():
        raw, metadata = read_regular(WORKSPACE / relative)
        if hashlib.sha256(raw).hexdigest() != digest or stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ValueError


def install(operation, runner_hash, tools_hash):
    stage = operation_root(operation) / "stage"
    runner_raw = safe_regular(stage / "acs_workshop_runner.py")
    tools_raw = safe_regular(stage / "acs_workspace_tools.md")
    if hashlib.sha256(runner_raw).hexdigest() != runner_hash or hashlib.sha256(tools_raw).hexdigest() != tools_hash:
        raise ValueError
    safe_regular(WORKSPACE / "acs_workshop_runner.py")
    safe_regular(WORKSPACE / "TOOLS.md")
    atomic_bytes(WORKSPACE / "acs_workshop_runner.py", runner_raw, 0o444)
    atomic_bytes(WORKSPACE / "TOOLS.md", tools_raw, 0o444)
    rebuild_manifest()
    quarantine_targets(operation, "apply-reset")
    verify_installed(runner_hash, tools_hash)


def prepare_restore_file(destination, descriptor, contents):
    raw = contents["."]
    temporary = prepare_bytes(destination, raw, descriptor["mode"], "restore-file")
    if description(temporary) != descriptor:
        temporary.unlink()
        raise ValueError
    return temporary


def quarantine_prepared(path, quarantine, label):
    try:
        descriptor = description(path)
    except FileNotFoundError:
        return
    candidate = quarantine / ("prepared-" + label + "-" + str(uuid.uuid4()))
    os.replace(path, candidate)
    fsync_directory(path.parent)
    privatize(candidate, descriptor)


def prepare_restore_directory(destination, descriptor, contents, quarantine, label):
    temporary = destination.with_name("." + destination.name + ".restore-" + str(uuid.uuid4()))
    temporary.mkdir(mode=0o700)
    try:
        directories = [(relative, item) for relative, item in descriptor["entries"].items() if item["kind"] == "dir" and relative != "."]
        for relative, _item in sorted(directories, key=lambda pair: len(Path(pair[0]).parts)):
            (temporary / relative).mkdir(mode=0o700)
        for relative, item in descriptor["entries"].items():
            if item["kind"] == "file":
                atomic_bytes(temporary / relative, contents[relative], item["mode"])
        for relative, item in sorted(directories, key=lambda pair: len(Path(pair[0]).parts), reverse=True):
            os.chmod(temporary / relative, item["mode"])
            fsync_directory(temporary / relative)
        os.chmod(temporary, descriptor["entries"]["."]["mode"])
        fsync_directory(temporary)
        if description(temporary) != descriptor:
            raise ValueError
        return temporary
    except BaseException:
        quarantine_prepared(temporary, quarantine, label)
        raise


def move_current_to_quarantine(destination, quarantine, label, current):
    candidate = quarantine / label
    os.replace(destination, candidate)
    fsync_directory(destination.parent)
    privatize(candidate, current)
    return candidate


def install_prepared(destination, temporary, quarantine, label, current):
    temporary_mode = os.lstat(temporary).st_mode
    if current["present"] is False or (current["kind"] == "file" and stat.S_ISREG(temporary_mode)):
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        return
    candidate = move_current_to_quarantine(destination, quarantine, label, current)
    try:
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            os.lstat(destination)
        except FileNotFoundError:
            os.replace(candidate, destination)
            fsync_directory(destination.parent)
        raise


def verify_restored_targets(targets):
    for label, destination in BACKUP_TARGETS.items():
        if description(destination) != targets[label]["descriptor"]:
            raise ValueError
    verify_protected_manifest()


def rollback(operation, restore_id, expected_hash):
    targets = restore_package(operation, restore_id, expected_hash)
    root = operation_root(operation)
    quarantine = root / "quarantine" / ("rollback-current-" + str(uuid.uuid4()))
    quarantine.mkdir(mode=0o700)
    prepared = {}
    current_targets = {}
    try:
        for label, destination in BACKUP_TARGETS.items():
            current_targets[label] = description(destination)
            target = targets[label]
            descriptor = target["descriptor"]
            if descriptor["present"] is False:
                continue
            if descriptor["kind"] == "file":
                prepared[label] = prepare_restore_file(destination, descriptor, target["contents"])
            else:
                prepared[label] = prepare_restore_directory(destination, descriptor, target["contents"], quarantine, label)
    except BaseException:
        for label, temporary in prepared.items():
            quarantine_prepared(temporary, quarantine, label)
        raise
    try:
        for label, destination in BACKUP_TARGETS.items():
            descriptor = targets[label]["descriptor"]
            current = current_targets[label]
            if descriptor["present"] is False:
                if current["present"] is not False:
                    move_current_to_quarantine(destination, quarantine, label, current)
                continue
            install_prepared(destination, prepared.pop(label), quarantine, label, current)
    except BaseException:
        for label, temporary in prepared.items():
            quarantine_prepared(temporary, quarantine, label)
        raise
    verify_restored_targets(targets)


def verify_restored(operation, restore_id, expected_hash):
    verify_restored_targets(restore_package(operation, restore_id, expected_hash))


def reset(operation, restore_id, expected_hash, reset_id, runner_hash, tools_hash):
    if OPERATION.fullmatch(reset_id) is None:
        raise ValueError
    restore_package(operation, restore_id, expected_hash)
    verify_sources(runner_hash, tools_hash)
    quarantine_targets(operation, "between-qa-" + reset_id)
    verify_installed(runner_hash, tools_hash)


action = sys.argv[1]
if action == "backup":
    backup(sys.argv[2])
elif action == "prepare-restore":
    prepare_restore(*sys.argv[2:])
elif action == "install":
    install(*sys.argv[2:])
elif action == "rollback":
    rollback(*sys.argv[2:])
elif action == "verify-restored":
    verify_restored(*sys.argv[2:])
elif action == "verify-installed":
    verify_installed(*sys.argv[2:])
elif action == "reset":
    reset(*sys.argv[2:])
else:
    raise ValueError
PY
readonly sandbox_helper

if ! "${python}" -c "${host_helper}" prepare "${state_dir}" >/dev/null || ! \
  "${python}" -c "${host_helper}" prepare-global "${host_global_state}" >/dev/null; then
  emit_preflight_failure
  exit 70
fi

query_loop_state() {
  local stderr_file="$1"
  local stdout_file="${stderr_file}.stdout"
  local parsed status parse_status
  install -m 600 /dev/null "${stderr_file}"
  install -m 600 /dev/null "${stdout_file}"
  if (
    trap - ERR
    NO_COLOR=1 NODE_NO_WARNINGS=1 "${nemoclaw}" "${sandbox}" exec \
      --workdir "${workspace}" -- env NO_COLOR=1 NODE_NO_WARNINGS=1 \
      openclaw config get "${loop_key}" --json \
      >"${stdout_file}" 2>"${stderr_file}"
  ); then
    status=0
  else
    status=$?
  fi
  if parsed="$("${python}" -c "${host_helper}" parse-loop \
    "${stdout_file}" "${stderr_file}" "${status}" "${missing_loop_diagnostic}")"; then
    parse_status=0
  else
    parse_status=$?
  fi
  install -m 600 /dev/null "${stderr_file}"
  install -m 600 /dev/null "${stdout_file}"
  if (( parse_status != 0 )); then
    return 11
  fi
  printf '%s\n' "${parsed}"
}

read_operation() {
  "${python}" -c "${host_helper}" load \
    "${state_dir}" "${host_global_state}" "${sandbox}"
}

clear_prepared_operation() {
  "${python}" -c "${host_helper}" clear-prepared \
    "${state_dir}" "${host_global_state}" "${sandbox}" "${operation_id}" \
    >/dev/null
}

sandbox_action() {
  local action="$1"
  shift
  "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
    python3 -c "${sandbox_helper}" "${action}" "$@" >/dev/null 2>&1
}

sandbox_action_capture() {
  local action="$1"
  shift
  "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
    python3 -c "${sandbox_helper}" "${action}" "$@" 2>/dev/null
}

restore_loop_state() {
  local presence="$1"
  local value="$2"
  if [[ "${presence}" == "absent" ]]; then
    "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
      openclaw config unset "${loop_key}" >/dev/null 2>&1
  else
    "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
      openclaw config set "${loop_key}" "${value}" --strict-json >/dev/null 2>&1
  fi
  "${nemoclaw}" "${sandbox}" gateway restart --quiet >/dev/null 2>&1
  local observed observed_presence observed_value
  local query_status
  if observed="$(query_loop_state "${state_dir}/.loop-restore-readback")"; then
    query_status=0
  else
    query_status=$?
    failure_code="rollback_loop_query_${query_status}"
    return 1
  fi
  if ! IFS=$'\t' read -r observed_presence observed_value <<<"${observed}"; then
    failure_code="rollback_loop_parse"
    return 1
  fi
  if [[ "${observed_presence}" != "${presence}" || "${observed_value}" != "${value}" ]]; then
    failure_code="rollback_loop_mismatch"
    return 1
  fi
}

operation_id=""
prior_presence=""
prior_value=""
rolled_back=""
runner_hash=""
tools_hash=""
backup_hash=""
restore_id=""
rollback_ready=0
failure_code="operation_failed"

stage_trusted_backup() {
  restore_id="$("${python}" -c 'import uuid; print(uuid.uuid4())')"
  sandbox_action prepare-restore "${operation_id}" "${restore_id}" || return 1
  "${openshell}" sandbox upload "${sandbox}" \
    "${state_dir}/${operation_id}/backup-package.json" \
    "${sandbox_state_root}/${operation_id}/restore/${restore_id}" \
    >/dev/null 2>&1 || return 1
}

rollback_internal() {
  failure_code="rollback_transfer"
  stage_trusted_backup || return 1
  failure_code="rollback_files"
  sandbox_action rollback \
    "${operation_id}" "${restore_id}" "${backup_hash}" || return 1
  failure_code="rollback_loop"
  restore_loop_state "${prior_presence}" "${prior_value}" || return 1
  failure_code="rollback_verify"
  sandbox_action verify-restored \
    "${operation_id}" "${restore_id}" "${backup_hash}" || return 1
  failure_code="rollback_state"
  "${python}" -c "${host_helper}" complete \
    "${state_dir}" "${host_global_state}" "${sandbox}" "${operation_id}" \
    >/dev/null || return 1
}

handle_failure() {
  local status="$1"
  local original_code="${failure_code}"
  trap - ERR INT TERM
  set +e
  local restored=false
  if (( rollback_ready == 1 )) && rollback_internal; then
    restored=true
  fi
  printf '{"code":"%s","main_session_touched":false,"rollback":%s,"schema_version":1,"status":"fail"}\n' "${original_code}" "${restored}"
  if (( status == 0 )); then
    status=70
  fi
  exit "${status}"
}

if [[ "${mode}" == "apply" ]]; then
  if ! "${python}" -c "${host_helper}" free \
    "${host_global_state}" >/dev/null; then
    emit_preflight_failure
    exit 70
  fi
  if ! hashes="$("${python}" -c "${host_helper}" bundle "${bundle_dir}")"; then
    emit_preflight_failure
    exit 70
  fi
  IFS=$'\t' read -r runner_hash tools_hash <<<"${hashes}"
  if ! IFS=$'\t' read -r prior_presence prior_value < <(
    query_loop_state "${state_dir}/.loop-preflight"
  ); then
    emit_preflight_failure
    exit 70
  fi
  if [[ "${prior_presence}" == "absent" ]] && ! \
    "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
      openclaw config unset --help >/dev/null 2>&1; then
    emit_preflight_failure
    exit 70
  fi
  operation_id="$("${python}" -c 'import uuid; print(uuid.uuid4())')"
  if ! "${python}" -c "${host_helper}" create "${state_dir}" "${operation_id}" \
    "${prior_presence}" "${prior_value}" "${runner_hash}" "${tools_hash}" >/dev/null; then
    emit_preflight_failure
    exit 70
  fi
  if ! "${python}" -c "${host_helper}" reserve \
    "${state_dir}" "${host_global_state}" "${sandbox}" "${operation_id}" \
    >/dev/null; then
    clear_prepared_operation || true
    emit_preflight_failure
    exit 70
  fi
  if ! remote_backup_hash="$(sandbox_action_capture backup "${operation_id}")"; then
    clear_prepared_operation || true
    emit_preflight_failure
    exit 70
  fi
  if [[ ! "${remote_backup_hash}" =~ ^[0-9a-f]{64}$ ]] || ! \
    "${openshell}" sandbox download "${sandbox}" \
      "${sandbox_state_root}/${operation_id}/backup-package.json" \
      "${state_dir}/${operation_id}" >/dev/null 2>&1; then
    clear_prepared_operation || true
    emit_preflight_failure
    exit 70
  fi
  if ! backup_hash="$("${python}" -c "${host_helper}" anchor-activate \
    "${state_dir}" "${host_global_state}" "${sandbox}" "${operation_id}" \
    "${remote_backup_hash}")"; then
    clear_prepared_operation || true
    emit_preflight_failure
    exit 70
  fi
  rollback_ready=1
  trap 'handle_failure $?' ERR
  trap 'handle_failure 130' INT
  trap 'handle_failure 143' TERM

  "${openshell}" sandbox upload "${sandbox}" \
    "${bundle_dir}/acs_workshop_runner.py" \
    "${sandbox_state_root}/${operation_id}/stage" >/dev/null 2>&1
  "${openshell}" sandbox upload "${sandbox}" \
    "${bundle_dir}/launchable/acs_workspace_tools.md" \
    "${sandbox_state_root}/${operation_id}/stage" >/dev/null 2>&1
  sandbox_action install "${operation_id}" "${runner_hash}" "${tools_hash}"
  "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
    env PYTHONDONTWRITEBYTECODE=1 python3 \
    "${workspace}/acs_workshop_runner.py" --help >/dev/null 2>&1
  sandbox_action verify-installed "${runner_hash}" "${tools_hash}"
  "${nemoclaw}" "${sandbox}" exec --workdir "${workspace}" -- \
    openclaw config set "${loop_key}" true --strict-json >/dev/null 2>&1
  "${nemoclaw}" "${sandbox}" gateway restart --quiet >/dev/null 2>&1
  observed_presence=""
  observed_value=""
  IFS=$'\t' read -r observed_presence observed_value < <(
    query_loop_state "${state_dir}/.loop-apply-readback"
  )
  [[ "${observed_presence}" == "present" && "${observed_value}" == "true" ]]
  sandbox_action verify-installed "${runner_hash}" "${tools_hash}"
  trap - ERR INT TERM
  rollback_ready=0
  printf '{"loop_detection":true,"main_session_touched":false,"manifest_files":6,"mode":"apply","rollback_ready":true,"runner_hash":"%s","schema_version":1,"status":"pass","tools_hash":"%s","workshop_reset":true}\n' \
    "${runner_hash}" "${tools_hash}"
  exit 0
fi

if ! operation_record="$(read_operation)"; then
  emit_preflight_failure
  exit 70
fi
IFS=$'\t' read -r operation_id prior_presence prior_value rolled_back runner_hash tools_hash backup_hash <<<"${operation_record}"

if [[ "${mode}" == "rollback" ]]; then
  if [[ "${rolled_back}" == "true" ]]; then
    if ! stage_trusted_backup || ! \
      sandbox_action verify-restored \
        "${operation_id}" "${restore_id}" "${backup_hash}" || ! \
      IFS=$'\t' read -r observed_presence observed_value < <(
        query_loop_state "${state_dir}/.loop-idempotent-readback"
      ) || [[ "${observed_presence}" != "${prior_presence}" || "${observed_value}" != "${prior_value}" ]] || ! \
      "${python}" -c "${host_helper}" complete \
        "${state_dir}" "${host_global_state}" "${sandbox}" "${operation_id}" \
        >/dev/null; then
      emit_preflight_failure
      exit 70
    fi
    printf '%s\n' '{"idempotent":true,"main_session_touched":false,"mode":"rollback","restored":true,"schema_version":1,"status":"pass"}'
    exit 0
  fi
  rollback_ready=1
  trap 'handle_failure $?' ERR
  trap 'handle_failure 130' INT
  trap 'handle_failure 143' TERM
  rollback_internal
  trap - ERR INT TERM
  rollback_ready=0
  printf '%s\n' '{"idempotent":false,"main_session_touched":false,"mode":"rollback","restored":true,"schema_version":1,"status":"pass"}'
  exit 0
fi

[[ "${rolled_back}" == "false" ]] || {
  emit_preflight_failure
  exit 70
}
if ! IFS=$'\t' read -r observed_presence observed_value < <(
  query_loop_state "${state_dir}/.loop-reset-readback"
) || [[ "${observed_presence}" != "present" || "${observed_value}" != "true" ]]; then
  emit_preflight_failure
  exit 70
fi
reset_id="$("${python}" -c 'import uuid; print(uuid.uuid4())')"
if ! stage_trusted_backup; then
  emit_preflight_failure
  exit 70
fi
rollback_ready=1
failure_code="reset_failed"
trap 'handle_failure $?' ERR
trap 'handle_failure 130' INT
trap 'handle_failure 143' TERM
sandbox_action reset "${operation_id}" "${restore_id}" "${backup_hash}" \
  "${reset_id}" "${runner_hash}" "${tools_hash}"
trap - ERR INT TERM
rollback_ready=0
printf '%s\n' '{"loop_detection":true,"main_session_touched":false,"mode":"reset-between-qa","schema_version":1,"status":"pass","workshop_reset":true}'
