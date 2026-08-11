from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchable" / "acs_nemoclaw_launchable_setup.sh"


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


def test_setup_applies_verified_host_provider_timeout_before_runtime_checks() -> None:
    source = _source()
    executable_check = (
        '[[ -x "${nemoclaw}" && -x "${openshell}" ]] || '
        'die "NemoClaw or OpenShell is missing."'
    )
    config_set = (
        '"${nemoclaw}" "${sandbox_name}" config set \\\n'
        "  --key models.providers.inference.timeoutSeconds \\\n"
        "  --value 300 \\\n"
        "  --config-accept-new-path \\\n"
        "  --restart"
    )
    config_get = (
        'provider_timeout="$("${nemoclaw}" "${sandbox_name}" config get \\\n'
        "  --key models.providers.inference.timeoutSeconds \\\n"
        '  --format json)"'
    )
    timeout_check = '[[ "${provider_timeout}" == "300" ]] ||'
    listener_check = 'ss -H -ltn "sport = :18789"'
    seed_turn = "agent --session-id main --json --timeout 600 -m"

    assert config_set in source
    assert config_get in source
    assert timeout_check in source
    assert source.index(executable_check) < source.index(config_set)
    assert source.index(config_set) < source.index(config_get)
    assert source.index(config_get) < source.index(timeout_check)
    assert source.index(timeout_check) < source.index(listener_check)
    assert source.index(timeout_check) < source.index(seed_turn)
    assert "shields down" not in source
    assert "openclaw config set" not in source


def test_setup_script_orchestrates_fixed_assets_and_one_time_bounded_agent_turn() -> (
    None
):
    source = _source()

    required_assets = (
        "launchable/pytorch-cu128-policy.yaml",
        "launchable/install_nvmolkit_in_sandbox.sh",
        "launchable/nvmolkit_gpu_probe.py",
        "skills/nvmolkit",
        "data/sample_molecules.csv",
        "acs_chemistry_task.py",
        "chemistry_workflow.py",
        "launchable/acs_workspace_tools.md",
        "launchable/acs_task_prompt.txt",
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
    assert source.count("agent --session-id main --json --timeout 600 -m") == 1
    assert "outputs/threshold-080/similarity_heatmap.png" in source
    assert "outputs/threshold-080/results.zip" in source
    assert "outputs/threshold-080/summary.json" in source
    assert '"highlight_threshold"' in source
    assert "0.80" in source


def test_setup_uploads_files_to_existing_parent_directories() -> None:
    source = _source()
    stale_path_cleanup = (
        '"${nemoclaw}" "${sandbox_name}" exec -- rm -rf -- \\\n'
        '  "/tmp/acs-setup/install_nvmolkit_in_sandbox.sh" \\\n'
        '  "/tmp/acs-setup/nvmolkit_gpu_probe.py" \\\n'
        '  "${workspace}/data/sample_molecules.csv" \\\n'
        '  "${workspace}/acs_chemistry_task.py" \\\n'
        '  "${workspace}/chemistry_workflow.py" \\\n'
        '  "${workspace}/TOOLS.md" \\\n'
        '  "${workspace}/acs_workspace_tools.md" \\\n'
        '  "${workspace}/acs_task_prompt.txt" \\\n'
        '  "${workspace}/start_artifact_server.sh"'
    )
    expected_uploads = (
        '"${openshell}" sandbox upload "${sandbox_name}" "${sandbox_installer}" \\\n'
        '  "/tmp/acs-setup"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${gpu_probe}" \\\n'
        '  "/tmp/acs-setup"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${dataset}" \\\n'
        '  "${workspace}/data"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_task}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_workflow}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${workspace_tools}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${task_prompt}" \\\n'
        '  "${workspace}"',
        '"${openshell}" sandbox upload "${sandbox_name}" "${artifact_server}" \\\n'
        '  "${workspace}"',
    )

    assert stale_path_cleanup in source
    assert source.index(stale_path_cleanup) < source.index(expected_uploads[0])
    assert source.count("sandbox upload") == len(expected_uploads)
    for upload in expected_uploads:
        assert upload in source
    assert (
        '"${nemoclaw}" "${sandbox_name}" exec -- mv -- \\\n'
        '  "${workspace}/acs_workspace_tools.md" "${workspace}/TOOLS.md"' in source
    )


def test_setup_script_requires_a_loopback_dashboard_and_exposes_only_the_proxy() -> (
    None
):
    source = _source()
    phase_zero = (ROOT / "launchable" / "nemoclaw_phase_zero.sh").read_text(
        encoding="utf-8"
    )

    start_artifacts = (
        'forward start --background 0.0.0.0:8765 "${sandbox_name}" '
        "</dev/null >/dev/null 2>&1"
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
    assert source.count("/usr/bin/setsid -f") == 1
    assert "http://127.0.0.1:18788/" in source
    assert "http://127.0.0.1:8765/threshold-080/results.zip" in source
    assert source.index("gateway-token --quiet") < source.index(
        'ACS_DASHBOARD_TOKEN="${gateway_token}"'
    )
    assert source.index('ACS_DASHBOARD_TOKEN="${gateway_token}"') < source.index(
        "unset gateway_token"
    )
    assert "for attempt in {1..60}" in source
    assert source.index(
        "http://127.0.0.1:8765/threshold-080/results.zip"
    ) < source.index("printf 'ready\\n'")


def test_setup_verifies_the_exact_post_turn_source_inputs_and_artifact_set() -> None:
    source = _source()
    agent_turn = source.index("agent --session-id main --json --timeout 600 -m")

    assert "acs_chemistry_task.original.py" not in source
    assert "protected.sha256" not in source
    assert 'expected_task_sha="$(python3 - "${chemistry_task}"' in source
    assert 'expected_dataset_sha="$(host_sha256 "${dataset}")"' in source
    assert 'expected_workflow_sha="$(host_sha256 "${chemistry_workflow}")"' in source
    assert 'expected_tools_sha="$(host_sha256 "${workspace_tools}")"' in source
    assert 'expected_prompt_sha="$(host_sha256 "${task_prompt}")"' in source
    assert 'expected_server_sha="$(host_sha256 "${artifact_server}")"' in source
    assert (
        source.index('expected_task_sha="$(python3 - "${chemistry_task}"') < agent_turn
    )
    assert source.index('rm -rf -- "${result_dir}"') < agent_turn
    assert source.index('mkdir -p -- "${result_dir}"') < agent_turn
    assert source.index('rm -rf -- "${result_dir}"') < source.index(
        'mkdir -p -- "${result_dir}"'
    )
    assert 'old = b"HIGHLIGHT_THRESHOLD = 0.70\\n"' in source
    assert 'new = b"HIGHLIGHT_THRESHOLD = 0.80\\n"' in source
    assert 'ACS_EXPECTED_TASK_SHA="${expected_task_sha}"' in source
    assert 'ACS_EXPECTED_SERVER_SHA="${expected_server_sha}"' in source
    assert '"start_artifact_server.sh": os.environ["ACS_EXPECTED_SERVER_SHA"]' in source
    assert "hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash" in source
    assert "expected_files = {" in source
    for artifact in (
        '"README.md"',
        '"summary.json"',
        '"top_10_pairs.csv"',
        '"similarity_heatmap.png"',
        '"results.zip"',
    ):
        assert artifact in source
    assert "set(output.iterdir())" not in source
    assert "actual_files != expected_files" in source


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
    assert "kill -KILL" not in source
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
