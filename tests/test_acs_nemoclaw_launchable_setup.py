from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchable" / "acs_nemoclaw_launchable_setup.sh"
CONFIG_ARGV = (
    (
        "acs-chemistry-agent",
        "exec",
        "--workdir",
        "/sandbox/.openclaw/workspace",
        "--",
        "openclaw",
        "config",
        "set",
        "models.providers.inference.timeoutSeconds",
        "300",
        "--strict-json",
    ),
    (
        "acs-chemistry-agent",
        "exec",
        "--workdir",
        "/sandbox/.openclaw/workspace",
        "--",
        "openclaw",
        "config",
        "set",
        "tools.loopDetection.enabled",
        "true",
        "--strict-json",
    ),
    ("acs-chemistry-agent", "gateway", "restart", "--quiet"),
    (
        "acs-chemistry-agent",
        "exec",
        "--workdir",
        "/sandbox/.openclaw/workspace",
        "--",
        "openclaw",
        "config",
        "get",
        "models.providers.inference.timeoutSeconds",
        "--json",
    ),
    (
        "acs-chemistry-agent",
        "exec",
        "--workdir",
        "/sandbox/.openclaw/workspace",
        "--",
        "openclaw",
        "config",
        "get",
        "tools.loopDetection.enabled",
        "--json",
    ),
)


def _source() -> str:
    assert SCRIPT.is_file(), "the unified ACS NemoClaw setup script is missing"
    return SCRIPT.read_text(encoding="utf-8")


def _fake_nvidia_smi(tmp_path: Path, inventory: str) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable = fake_bin / "nvidia-smi"
    executable.write_text(
        f"#!/usr/bin/env bash\nprintf '%b' {inventory!r}\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return fake_bin


def _run_preflight(
    tmp_path: Path,
    inventory: str,
    *,
    inference_key: str | None,
) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_nvidia_smi(tmp_path, inventory)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    if inference_key is None:
        environment.pop("NVIDIA_INFERENCE_API_KEY", None)
    else:
        environment["NVIDIA_INFERENCE_API_KEY"] = inference_key
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_nemoclaw(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-nemoclaw"
    executable.write_text(
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
call_number=0
if [[ -f "${ACS_FAKE_COUNTER}" ]]; then
  call_number="$(<"${ACS_FAKE_COUNTER}")"
fi
call_number=$((call_number + 1))
printf '%s\n' "${call_number}" > "${ACS_FAKE_COUNTER}"
{
  printf '%s' "$1"
  shift
  for argument in "$@"; do
    printf '\t%s' "${argument}"
  done
  printf '\n'
} >> "${ACS_FAKE_LOG}"
printf '%s\n' "${ACS_SECRET_CANARY}" >&2
printf '%s\n' "${ACS_RAW_CANARY}" >&2
if [[ "${ACS_FAIL_CALL}" == "${call_number}" ]]; then
  exit 17
fi
arguments=" $* "
if [[ "${arguments}" == *" openclaw config get models.providers.inference.timeoutSeconds --json "* ]]; then
  printf '%s\n' "${ACS_PROVIDER_JSON}"
elif [[ "${arguments}" == *" openclaw config get tools.loopDetection.enabled --json "* ]]; then
  printf '%s\n' "${ACS_LOOP_JSON}"
else
  printf '%s\n' "${ACS_RAW_CANARY}"
fi
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _run_runtime_config(
    tmp_path: Path,
    *,
    fail_call: int = 0,
    provider_json: str = "300",
    loop_json: str = "true",
) -> tuple[subprocess.CompletedProcess[str], tuple[tuple[str, ...], ...], Path]:
    fake_nemoclaw = _fake_nemoclaw(tmp_path)
    call_log = tmp_path / "nemoclaw-argv.log"
    counter = tmp_path / "nemoclaw-counter"
    ready_marker = tmp_path / "ready"
    harness = tmp_path / "configure-runtime.sh"
    harness.write_text(
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
source "${ACS_SETUP_SCRIPT}"
configure_openclaw_runtime \
  "${ACS_FAKE_NEMOCLAW}" \
  "acs-chemistry-agent" \
  "/sandbox/.openclaw/workspace"
printf 'ready\n' > "${ACS_READY_MARKER}"
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_FAIL_CALL": str(fail_call),
            "ACS_FAKE_COUNTER": str(counter),
            "ACS_FAKE_LOG": str(call_log),
            "ACS_FAKE_NEMOCLAW": str(fake_nemoclaw),
            "ACS_LOOP_JSON": loop_json,
            "ACS_PROVIDER_JSON": provider_json,
            "ACS_RAW_CANARY": "raw-config-output-canary",
            "ACS_READY_MARKER": str(ready_marker),
            "ACS_SECRET_CANARY": "nvapi-config-secret-canary",
            "ACS_SETUP_SCRIPT": str(SCRIPT),
            "HOME": str(tmp_path / "home"),
        }
    )
    completed = subprocess.run(
        ["bash", str(harness)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = ()
    if call_log.exists():
        calls = tuple(
            tuple(line.split("\t"))
            for line in call_log.read_text(encoding="utf-8").splitlines()
        )
    return completed, calls, ready_marker


def test_setup_script_has_the_complete_secret_safe_contract() -> None:
    source = _source()
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "set -Eeuo pipefail" in source
    assert "set -x" not in source
    assert "NVIDIA_INFERENCE_API_KEY is required" in source
    assert "chmod 600" in source
    stored_key = source.index("chmod 600")
    key_unset = source.index("unset NVIDIA_INFERENCE_API_KEY", stored_key)
    assert stored_key < key_unset
    assert key_unset < source.index("ACS_PHASE_ZERO_DETACHED=1")
    assert "ACS_PHASE_ZERO_DETACHED=1" in source
    assert 'bash "${phase_zero_script}"' in source
    assert '"${phase_zero_status}"' in source
    assert "dashboard-url" not in source
    assert "gateway-token --quiet" in source


def test_setup_prepends_local_bin_before_first_nemoclaw_config_command() -> None:
    source = _source()
    nemoclaw_path = 'readonly nemoclaw="${HOME}/.local/bin/nemoclaw"'
    openshell_path = 'readonly openshell="${HOME}/.local/bin/openshell"'
    executable_check = (
        '[[ -x "${nemoclaw}" && -x "${openshell}" ]] || '
        'die "NemoClaw or OpenShell is missing."'
    )
    path_export = 'export PATH="${HOME}/.local/bin:${PATH}"'
    first_config_command = (
        'configure_openclaw_runtime "${nemoclaw}" "${sandbox_name}" "${workspace}"'
    )

    assert source.count(path_export) == 1
    assert source.index(nemoclaw_path) < source.index(executable_check)
    assert source.index(openshell_path) < source.index(executable_check)
    assert source.index(executable_check) < source.index(path_export)
    assert source.index(path_export) < source.index(first_config_command)


def test_setup_applies_verified_openclaw_config_before_runtime_checks() -> None:
    source = _source()
    executable_check = (
        '[[ -x "${nemoclaw}" && -x "${openshell}" ]] || '
        'die "NemoClaw or OpenShell is missing."'
    )
    tools_install = (
        '"${nemoclaw}" "${sandbox_name}" exec -- mv -- \\\n'
        '  "${workspace}/acs_workspace_tools.md" "${workspace}/TOOLS.md"'
    )
    config_call = (
        'configure_openclaw_runtime "${nemoclaw}" "${sandbox_name}" "${workspace}"'
    )
    listener_check = 'ss -H -ltn "sport = :18789"'

    assert tools_install in source
    assert config_call in source
    positions = tuple(
        source.index(fragment)
        for fragment in (
            executable_check,
            tools_install,
            config_call,
            listener_check,
        )
    )
    assert positions == tuple(sorted(positions))
    assert source.count(config_call) == 1
    assert '"${nemoclaw}" "${sandbox_name}" config set' not in source
    assert '"${nemoclaw}" "${sandbox_name}" config get' not in source
    assert "--config-accept-new-path" not in source
    assert "--format json" not in source
    assert "agent --session-id" not in source
    assert "HIGHLIGHT_THRESHOLD" not in source
    assert source.count("threshold-080") == 1
    assert "shields down" not in source


def test_setup_reads_config_without_printing_values_or_secrets() -> None:
    source = _source()
    config_start = source.index("configure_openclaw_runtime() {")
    config_end = source.index("\n}\n\nif [[", config_start)
    block = source[config_start:config_end]

    assert "NVIDIA_INFERENCE_API_KEY" not in block
    assert "printf" not in block
    assert "echo" not in block
    assert block.count(">/dev/null 2>&1") == 3
    assert block.count("--json 2>/dev/null") == 2
    assert "|| true" not in block


def test_runtime_config_executes_exact_order_and_suppresses_cli_output(
    tmp_path: Path,
) -> None:
    completed, calls, ready_marker = _run_runtime_config(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert calls == CONFIG_ARGV
    assert ready_marker.read_text(encoding="utf-8") == "ready\n"
    combined = completed.stdout + completed.stderr
    assert "Phase: Configure OpenClaw runtime" in completed.stdout
    assert "raw-config-output-canary" not in combined
    assert "nvapi-config-secret-canary" not in combined


@pytest.mark.parametrize(
    ("fail_call", "expected_error"),
    (
        (1, "could not set the inference provider timeout."),
        (2, "could not enable OpenClaw tool-loop detection."),
        (3, "could not restart the OpenClaw gateway."),
        (4, "could not read back the inference provider timeout."),
        (5, "could not read back OpenClaw tool-loop detection."),
    ),
)
def test_runtime_config_nonzero_gate_stops_before_ready(
    tmp_path: Path,
    fail_call: int,
    expected_error: str,
) -> None:
    completed, calls, ready_marker = _run_runtime_config(
        tmp_path,
        fail_call=fail_call,
    )

    assert completed.returncode != 0
    assert calls == CONFIG_ARGV[:fail_call]
    assert not ready_marker.exists()
    assert expected_error in completed.stderr
    combined = completed.stdout + completed.stderr
    assert "raw-config-output-canary" not in combined
    assert "nvapi-config-secret-canary" not in combined


def test_runtime_config_failed_restart_rejects_stale_valid_values(
    tmp_path: Path,
) -> None:
    completed, calls, ready_marker = _run_runtime_config(
        tmp_path,
        fail_call=3,
        provider_json="300",
        loop_json="true",
    )

    assert completed.returncode != 0
    assert calls == CONFIG_ARGV[:3]
    assert not ready_marker.exists()
    assert "could not restart the OpenClaw gateway." in completed.stderr


@pytest.mark.parametrize(
    ("provider_json", "loop_json"),
    (
        ('"300"', "true"),
        ("300.0", "true"),
        ("300", '"true"'),
        ("300", "false"),
    ),
)
def test_runtime_config_accepts_only_exact_json_number_and_boolean(
    tmp_path: Path,
    provider_json: str,
    loop_json: str,
) -> None:
    completed, calls, ready_marker = _run_runtime_config(
        tmp_path,
        provider_json=provider_json,
        loop_json=loop_json,
    )

    assert completed.returncode != 0
    assert calls in (CONFIG_ARGV[:4], CONFIG_ARGV)
    assert not ready_marker.exists()


def test_setup_script_orchestrates_only_the_lean_workshop_assets() -> None:
    source = _source()

    required_assets = (
        "launchable/pytorch-cu128-policy.yaml",
        "launchable/install_nvmolkit_in_sandbox.sh",
        "launchable/nvmolkit_gpu_probe.py",
        "skills/nvmolkit",
        "data/sample_molecules.csv",
        "data/PROVENANCE.md",
        "acs_workshop_runner.py",
        "objective_challenge.py",
        "chemistry_workflow.py",
        "launchable/acs_workspace_tools.md",
        "launchable/start_artifact_server.sh",
        "launchable/openclaw_secure_link_proxy.mjs",
    )
    for asset in required_assets:
        assert asset in source

    assert "policy add" in source
    assert "--from-file" in source
    assert "--yes" in source
    assert source.count("sandbox upload") >= 8
    assert "nvmolkit-install.exit" in source
    assert "PYTHONPATH=/tmp/.local/lib/python3.13/site-packages" in source
    assert "skill install" in source
    assert '"${workspace}/TOOLS.md"' in source
    assert source.count("sandbox upload") == 9
    assert "agent --session-id" not in source
    assert source.count("acs_chemistry_task.py") == 1
    assert source.count("acs_task_prompt.txt") == 1


def test_retired_task_prompt_is_absent_but_remote_cleanup_is_retained() -> None:
    assert not (ROOT / "launchable" / "acs_task_prompt.txt").exists()
    assert _source().count("acs_task_prompt.txt") == 1


def test_setup_uploads_files_to_existing_parent_directories() -> None:
    source = _source()
    cleanup = source.index('"${nemoclaw}" "${sandbox_name}" exec -- rm -rf --')
    expected_uploads = (
        '"${openshell}" sandbox upload "${sandbox_name}" "${sandbox_installer}" \\\n'
        '  "/tmp/acs-setup"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${gpu_probe}" \\\n'
        '  "/tmp/acs-setup"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${dataset}" \\\n'
        '  "${workspace}/data"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${provenance}" \\\n'
        '  "${workspace}/data"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${workshop_runner}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${objective_challenge}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_workflow}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${workspace_tools}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${artifact_server}" \\\n'
        '  "${workspace}"',
    )

    for path in (
        '"${workspace}/outputs/workshop"',
        '"${workspace}/.acs-workshop-state"',
        '"${workspace}/outputs/threshold-080"',
        '"${workspace}/acs_chemistry_task.py"',
        '"${workspace}/acs_task_prompt.txt"',
    ):
        assert path in source[cleanup : source.index(expected_uploads[0])]
    assert 'rm -rf -- "${workspace}"' not in source
    assert (
        '"${workspace}/outputs"'
        not in source[cleanup : source.index(expected_uploads[0])]
    )
    assert cleanup < source.index(expected_uploads[0])
    assert source.count("sandbox upload") == len(expected_uploads)
    for upload in expected_uploads:
        assert upload in source
    assert (
        '"${nemoclaw}" "${sandbox_name}" exec -- mv -- \\\n'
        '  "${workspace}/acs_workspace_tools.md" "${workspace}/TOOLS.md"' in source
    )


def test_setup_creates_the_exact_atomic_read_only_manifest_before_quiet_help() -> None:
    source = _source()
    last_upload = source.rindex("sandbox upload")
    state_create = source.index('mkdir -m 700 -- "${workspace}/.acs-workshop-state"')
    manifest_create = source.index('ACS_EXPECTED_RUNNER_SHA="${expected_runner_sha}"')
    help_smoke = source.index("acs_workshop_runner.py --help >/dev/null 2>&1")

    assert last_upload < state_create < manifest_create < help_smoke
    assert 'expected_runner_sha="$(host_sha256 "${workshop_runner}")"' in source
    assert 'expected_objective_sha="$(host_sha256 "${objective_challenge}")"' in source
    assert 'expected_workflow_sha="$(host_sha256 "${chemistry_workflow}")"' in source
    assert 'expected_dataset_sha="$(host_sha256 "${dataset}")"' in source
    assert 'expected_provenance_sha="$(host_sha256 "${provenance}")"' in source
    assert 'expected_tools_sha="$(host_sha256 "${workspace_tools}")"' in source
    for name in (
        '"TOOLS.md": os.environ["ACS_EXPECTED_TOOLS_SHA"]',
        '"acs_workshop_runner.py": os.environ["ACS_EXPECTED_RUNNER_SHA"]',
        '"chemistry_workflow.py": os.environ["ACS_EXPECTED_WORKFLOW_SHA"]',
        '"data/sample_molecules.csv": os.environ["ACS_EXPECTED_DATASET_SHA"]',
        '"data/PROVENANCE.md": os.environ["ACS_EXPECTED_PROVENANCE_SHA"]',
        '"objective_challenge.py": os.environ["ACS_EXPECTED_OBJECTIVE_SHA"]',
    ):
        assert name in source
    assert "os.O_CREAT | os.O_EXCL | os.O_WRONLY" in source
    assert "os.fsync(descriptor)" in source
    assert "os.replace(temporary, manifest)" in source
    assert "os.chmod(manifest, 0o444)" in source
    assert source.count('"schema_version": 1') == 1
    assert source.count("ACS_EXPECTED_") == 12
    assert "chmod 0444 -- \\\n" in source
    assert source.index("chmod 0444 -- \\\n") < help_smoke


def test_setup_clears_inherited_state_directory_special_bits_before_use() -> None:
    source = _source()
    state_create = source.index('mkdir -m 700 -- "${workspace}/.acs-workshop-state"')
    clear_setgid = source.index(
        'exec -- chmod g-s -- "${workspace}/.acs-workshop-state"', state_create
    )
    normalize_mode = source.index(
        'exec -- chmod 0700 -- "${workspace}/.acs-workshop-state"', clear_setgid
    )
    verify_mode = source.index(
        'test "$(stat -c "%a" '
        '/sandbox/.openclaw/workspace/.acs-workshop-state)" = "700"',
        normalize_mode,
    )
    manifest_create = source.index('ACS_EXPECTED_RUNNER_SHA="${expected_runner_sha}"')

    assert state_create < clear_setgid < normalize_mode < verify_mode < manifest_create


def test_setup_runs_full_workflow_smoke_after_manifest_before_services() -> None:
    source = _source()
    manifest = source.index("os.chmod(manifest, 0o444)")
    smoke_import = source.index("import acs_workshop_runner as runner")
    smoke_command = source.rfind('"${nemoclaw}"', manifest, smoke_import)
    smoke_end = source.index('>"${workflow_smoke_log}" 2>&1; then', smoke_import)
    artifact_sentinel = source.index("ACS_ARTIFACT_SENTINEL_CONTENT=", smoke_end)
    services = source.index('phase "Start attendee services"', artifact_sentinel)
    ready = source.index('mv -f -- "${ready_temp}" "${ready_marker}"', services)
    block = source[smoke_command:smoke_end]

    assert manifest < smoke_command < smoke_import < smoke_end
    assert smoke_end < artifact_sentinel < services < ready
    assert "PYTHONDONTWRITEBYTECODE=1" in block
    assert 'PYTHONPATH="${workspace}:/tmp/.local/lib/python3.13/site-packages"' in block
    assert "runner.verify_manifest(runner.DEFAULT_PATHS)" in block
    assert 'runner.execute_workflow_prefix("optimize_conformers_mmff94")' in block
    assert "from chemistry_workflow import WorkflowPhase" in block
    assert "execution.state.phase is not WorkflowPhase.OPTIMIZED" in block
    assert "runner.WorkflowPhase" not in block
    assert "torch.cuda.device_count() != 1" in block
    assert 'torch.cuda.get_device_name(0) != "NVIDIA L4"' in block
    for stage_name in (
        "inspect_library",
        "generate_morgan_fingerprints",
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
        "embed_representative_conformers",
        "optimize_conformers_mmff94",
    ):
        assert f'"{stage_name}"' in block
    assert "_publish" not in block
    assert "run_lesson" not in block
    assert source[smoke_end:].startswith('>"${workflow_smoke_log}" 2>&1; then')


def test_setup_keeps_private_full_smoke_diagnostics() -> None:
    source = _source()
    log_declaration = source.index(
        'readonly workflow_smoke_log="${state_dir}/workflow-smoke.log"'
    )
    log_creation = source.index(
        'install -m 600 /dev/null "${workflow_smoke_log}"', log_declaration
    )
    smoke_command = source.index(
        'if ! "${nemoclaw}" "${sandbox_name}" exec --no-tty --timeout 600',
        log_creation,
    )
    log_redirect = source.index('>"${workflow_smoke_log}" 2>&1; then', smoke_command)
    safe_failure = source.index(
        'die "the full workshop smoke failed; inspect ${workflow_smoke_log}."',
        log_redirect,
    )
    artifact_sentinel = source.index("ACS_ARTIFACT_SENTINEL_CONTENT=", safe_failure)

    assert (
        log_creation < smoke_command < log_redirect < safe_failure < artifact_sentinel
    )
    assert "NVIDIA_INFERENCE_API_KEY" not in source[smoke_command:log_redirect]


def test_setup_script_requires_a_loopback_dashboard_and_exposes_only_the_proxy() -> (
    None
):
    source = _source()
    phase_zero = (ROOT / "launchable" / "nemoclaw_phase_zero.sh").read_text(
        encoding="utf-8"
    )

    start_artifacts = (
        '"${openshell}" forward start 0.0.0.0:8765 "${sandbox_name}" '
        "</dev/null >/dev/null 2>&1 &"
    )
    forbidden_bind = "NEMOCLAW_DASHBOARD_BIND" + "=0.0.0.0"
    assert forbidden_bind not in phase_zero + source
    assert "forward stop 18789" not in source
    assert "0.0.0.0:18789" not in source
    assert "127.0.0.1:18789" in source
    assert 'ss -H -ltn "sport = :18789"' in source
    assert "ipaddress.ip_address(host).is_loopback" in source
    assert 'ACS_DASHBOARD_TOKEN="${gateway_token}"' in source
    assert '"${node_bin}" "${dashboard_proxy}"' in source
    assert start_artifacts in source
    assert "forward start --background" not in source
    assert source.count("/usr/bin/setsid -f") == 0
    assert "http://127.0.0.1:18788/" in source
    assert 'wait_for_http "http://127.0.0.1:8765/${artifact_sentinel_name}"' in source
    assert source.count("threshold-080") == 1
    assert source.index("gateway-token --quiet") < source.index(
        'ACS_DASHBOARD_TOKEN="${gateway_token}"'
    )
    assert source.index('ACS_DASHBOARD_TOKEN="${gateway_token}"') < source.index(
        "unset gateway_token"
    )
    assert "for attempt in {1..60}" in source
    assert source.index("http://127.0.0.1:8765/") < source.index("printf 'ready\\n'")


def test_rerun_stops_exact_tracked_services_before_workspace_reset() -> None:
    source = _source()
    cleanup = source.index('"${nemoclaw}" "${sandbox_name}" exec -- rm -rf --')
    stop_proxy = source.index("stop_tracked_proxy\n")
    stop_forward = source.index("stop_tracked_artifact_forward\n")
    proxy_listener = source.index('ss -H -ltn "sport = :18788"', stop_proxy)
    forward_listener = source.index('ss -H -ltn "sport = :8765"', stop_forward)

    assert (
        'readonly artifact_forward_pid_file="${state_dir}/artifact-forward.pid"'
        in source
    )
    assert "proxy_process_matches" in source
    assert "artifact_forward_process_matches" in source
    assert '"/proc/${pid}/stat"' in source
    assert 'local expected_start_ticks="$2"' in source
    assert '"${process_argv[1]}" == "forward"' in source
    assert '"${process_argv[2]}" == "start"' in source
    assert '"${process_argv[3]}" == "0.0.0.0:8765"' in source
    assert '"${process_argv[4]}" == "${sandbox_name}"' in source
    assert stop_proxy < stop_forward < proxy_listener < forward_listener < cleanup
    assert 'die "the recorded proxy PID file is unsafe."' in source
    assert 'die "the recorded artifact-forward PID file is unsafe."' in source
    assert 'die "port 8765 is already owned by an untracked process."' in source


def test_nonzero_exit_rolls_back_only_new_validated_services_until_ready() -> None:
    source = _source()
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    proxy_spawn = source.index(
        '/usr/bin/setsid "${node_bin}" "${dashboard_proxy}" '
        "</dev/null >/dev/null 2>&1 &"
    )
    proxy_pid = source.index("proxy_pid=$!", proxy_spawn)
    proxy_tick = source.index(
        'if ! proxy_start_ticks="$(process_start_ticks "${proxy_pid}")"', proxy_pid
    )
    proxy_arm = source.index("proxy_rollback_armed=1", proxy_pid)
    proxy_publish = source.index(
        'mv -f -- "${proxy_pid_temp}" "${proxy_pid_file}"', proxy_arm
    )
    forward_spawn = source.index(
        '/usr/bin/setsid "${openshell}" forward start 0.0.0.0:8765 '
        '"${sandbox_name}" </dev/null >/dev/null 2>&1 &'
    )
    forward_pid = source.index("artifact_forward_pid=$!", forward_spawn)
    forward_tick = source.index(
        'if ! artifact_forward_start_ticks="$(process_start_ticks "${artifact_forward_pid}")"',
        forward_pid,
    )
    forward_arm = source.index("artifact_forward_rollback_armed=1", forward_pid)
    forward_publish = source.index(
        'mv -f -- "${artifact_forward_pid_temp}" "${artifact_forward_pid_file}"',
        forward_arm,
    )
    assert "local exit_code=$?" in cleanup
    assert "rollback_new_artifact_forward" in cleanup
    assert "rollback_new_proxy" in cleanup
    assert proxy_pid < proxy_tick < proxy_arm < proxy_publish
    assert forward_pid < forward_tick < forward_arm < forward_publish
    assert '"${proxy_pid}:${proxy_start_ticks}"' in source
    assert '"${artifact_forward_pid}:${artifact_forward_start_ticks}"' in source
    for function_name, validator, pid in (
        ("rollback_new_proxy", "proxy_process_matches", "proxy_pid"),
        (
            "rollback_new_artifact_forward",
            "artifact_forward_process_matches",
            "artifact_forward_pid",
        ),
    ):
        function_start = source.index(f"{function_name}() {{")
        function_end = source.index("\n}", function_start)
        body = source[function_start:function_end]
        normalized_body = " ".join(body.split())
        assert f'{validator} "${{{pid}}}"' in normalized_body
        assert f'kill -TERM "${{{pid}}}"' in body
        assert f'kill -KILL "${{{pid}}}"' in body


def test_failed_start_tick_read_terminates_and_reaps_the_direct_child() -> None:
    source = _source()
    helper_start = source.index("terminate_just_spawned_child() {")
    helper_end = source.index("\n}", helper_start)
    helper = source[helper_start:helper_end]

    assert 'kill -TERM "${pid}"' in helper
    assert 'kill -KILL "${pid}"' in helper
    assert 'wait "${pid}"' in helper
    assert "for attempt in {1..10}" in helper
    for ticks, pid, error in (
        (
            "proxy_start_ticks",
            "proxy_pid",
            "could not record the new proxy process identity.",
        ),
        (
            "artifact_forward_start_ticks",
            "artifact_forward_pid",
            "could not record the new artifact-forward process identity.",
        ),
    ):
        guard = source.index(f'if ! {ticks}="$(process_start_ticks "${{{pid}}}")"')
        guard_end = source.index("\nfi", guard)
        body = source[guard:guard_end]
        arm = source.index(f"{pid.removesuffix('_pid')}_rollback_armed=1", guard)
        assert f'terminate_just_spawned_child "${{{pid}}}"' in body
        assert f'die "{error}"' in body
        assert guard_end < arm


def test_ready_marker_and_final_output_remain_failure_atomic() -> None:
    source = _source()
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    ready_publish = source.index('mv -f -- "${ready_temp}" "${ready_marker}"')
    ready_output = source.index("printf 'ACS chemistry workspace is ready.\\n'")
    success = source.index("setup_succeeded=1")

    assert "setup_succeeded=0" in source
    assert '"${exit_code}" != "0"' in cleanup
    assert '"${setup_succeeded}" != "1"' in cleanup
    assert 'rm -f -- "${ready_marker}"' in cleanup
    assert cleanup.index('rm -f -- "${ready_marker}"') < cleanup.index(
        "rollback_new_artifact_forward"
    )
    assert "trap 'exit 129' HUP" in source
    assert "trap 'exit 130' INT" in source
    assert "trap 'exit 143' TERM" in source
    assert ready_publish < ready_output < success
    assert source.count("proxy_rollback_armed=0") == 1
    assert source.count("artifact_forward_rollback_armed=0") == 1


def test_artifact_forward_serves_an_exact_setup_sentinel_before_ready() -> None:
    source = _source()
    create = source.index(
        'ACS_ARTIFACT_SENTINEL_CONTENT="${artifact_sentinel_content}"'
    )
    fetch = source.index(
        'wait_for_http "http://127.0.0.1:8765/${artifact_sentinel_name}"'
    )
    compare = source.index(
        '[[ "$(<"${artifact_probe}")" == "${artifact_sentinel_content}" ]]'
    )
    ready = source.index('mv -f -- "${ready_temp}" "${ready_marker}"')

    assert 'readonly artifact_sentinel_name=".acs-artifact-service-ready"' in source
    assert (
        'readonly artifact_sentinel_content="acs-artifact-service-ready-v1"' in source
    )
    assert '"${workspace}/outputs/${artifact_sentinel_name}"' in source
    assert create < fetch < compare < ready
    assert 'rm -f -- "${artifact_probe}"' in source


def test_setup_has_short_progress_without_raw_tool_output() -> None:
    source = _source()

    for label in (
        "Validate hardware",
        "Install OpenClaw runtime",
        "Configure OpenClaw runtime",
        "Verify private dashboard",
        "Install chemistry tools",
        "Prepare workshop files",
        "Start attendee services",
    ):
        assert f'phase "{label}"' in source
    assert "set -x" not in source
    assert source.count("--json 2>/dev/null") == 2
    assert "--json" not in source.replace("--json 2>/dev/null", "")
    assert source.count(">/dev/null 2>&1") >= 12


def test_setup_tracks_and_validates_proxy_process_ownership() -> None:
    source = _source()

    assert (
        'readonly proxy_pid_file="${state_dir}/openclaw-secure-link-proxy.pid"'
        in source
    )
    assert '"/proc/${pid}/exe"' in source
    assert '"/proc/${pid}/cmdline"' in source
    assert 'kill -TERM "${old_proxy_pid}"' in source
    assert 'kill -0 "${old_proxy_pid}"' in source
    assert 'kill -KILL "${old_proxy_pid}"' not in source
    assert 'ss -H -ltn "sport = :18788"' in source
    assert '/usr/bin/setsid "${node_bin}" "${dashboard_proxy}"' in source
    assert '/usr/bin/setsid -f "${node_bin}" "${dashboard_proxy}"' not in source
    assert "&\nproxy_pid=$!" in source
    assert 'kill -0 "${proxy_pid}"' in source
    assert 'mv -f -- "${proxy_pid_temp}" "${proxy_pid_file}"' in source
    assert source.index('export ACS_DASHBOARD_TOKEN="${gateway_token}"') < source.index(
        '/usr/bin/setsid "${node_bin}" "${dashboard_proxy}"'
    )
    assert source.index(
        '/usr/bin/setsid "${node_bin}" "${dashboard_proxy}"'
    ) < source.index("unset ACS_DASHBOARD_TOKEN")


def test_setup_resolves_exactly_one_executable_nvm_node_without_ambient_path() -> None:
    source = _source()

    assert '"${HOME}"/.nvm/versions/node/*/bin/node' in source
    assert "node_candidates=()" in source
    assert '[[ -x "${candidate}" ]]' in source
    assert 'node_candidates+=("${candidate}")' in source
    assert '[[ "${#node_candidates[@]}" == "1" ]]' in source
    assert 'readonly node_bin="${node_candidates[0]}"' in source
    assert '/usr/bin/setsid "${node_bin}" "${dashboard_proxy}"' in source
    assert "/usr/bin/setsid -f node" not in source
    assert "command -v node" not in source
    assert "/home/ubuntu/.nvm" not in source


@pytest.mark.parametrize(
    "inventory",
    (
        "",
        "NVIDIA A100-SXM4-80GB\n",
        "NVIDIA L4\nNVIDIA L4\n",
    ),
)
def test_setup_fails_closed_before_secret_storage_without_exactly_one_l4(
    tmp_path: Path, inventory: str
) -> None:
    canary = "nvapi-test-secret-canary"
    completed = _run_preflight(tmp_path, inventory, inference_key=canary)

    assert completed.returncode != 0
    assert "exactly one NVIDIA L4" in completed.stderr
    assert canary not in completed.stdout + completed.stderr
    assert not (
        tmp_path / "home/.config/acs-phase-zero/NVIDIA_INFERENCE_API_KEY"
    ).exists()
    assert not (tmp_path / "home/.local/state/acs-nemoclaw-launchable/ready").exists()


def test_setup_requires_the_inference_key_before_phase_zero(tmp_path: Path) -> None:
    completed = _run_preflight(tmp_path, "NVIDIA L4\n", inference_key=None)

    assert completed.returncode != 0
    assert "NVIDIA_INFERENCE_API_KEY is required" in completed.stderr
    assert not (
        tmp_path / "home/.config/acs-phase-zero/NVIDIA_INFERENCE_API_KEY"
    ).exists()
    assert not (tmp_path / "home/.local/state/acs-nemoclaw-launchable/ready").exists()
