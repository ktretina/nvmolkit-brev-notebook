import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchable" / "nemoclaw_phase_zero.sh"
PYTORCH_POLICY = ROOT / "launchable" / "pytorch-cu128-policy.yaml"
NVMOLKIT_INSTALLER = ROOT / "launchable" / "install_nvmolkit_in_sandbox.sh"
GPU_PROBE = ROOT / "launchable" / "nvmolkit_gpu_probe.py"
WORKSPACE_TOOLS = ROOT / "launchable" / "acs_workspace_tools.md"
ACCEPTANCE_PROMPT = ROOT / "launchable" / "acs_task_prompt.txt"
ARTIFACT_SERVER = ROOT / "launchable" / "start_artifact_server.sh"


def test_phase_zero_setup_is_pinned_and_removes_transport_key_before_install():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = SCRIPT.read_text()
    assert "0d1cb93888c817daec44b2cc996afa75eebcbd46" in source
    assert "b52f053a550fab90ab1dff4ab7f3a0b55b2506aeafd2062832e65632fdbcae70" in source
    assert "NEMOCLAW_PROVIDER=build" in source
    assert "NEMOCLAW_MODEL=nvidia/nemotron-3-super-120b-a12b" in source
    assert "NEMOCLAW_SANDBOX_GPU=1" in source
    assert source.index('rm -f -- "${key_file}"') < source.index('bash "${installer}"')


def test_phase_zero_setup_detaches_before_reading_the_key(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux_log = tmp_path / "tmux.log"
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$TMUX_LOG"\n'
        'if [[ "${1:-}" == has-session ]]; then exit 1; fi\n'
    )
    fake_tmux.chmod(fake_tmux.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TMUX_LOG": str(tmux_log),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = tmux_log.read_text()
    assert "new-session -d -s acs-phase-zero-install" in commands
    assert "send-keys -t acs-phase-zero-install" in commands
    assert "NemoClaw installation started" in completed.stdout


def test_pytorch_policy_is_narrow_and_read_only():
    source = PYTORCH_POLICY.read_text()

    assert "host: download.pytorch.org" in source
    assert "host: download-r2.pytorch.org" in source
    assert "host: pypi.nvidia.com" in source
    assert "port: 443" in source
    assert "protocol: rest" in source
    assert "enforcement: enforce" in source
    assert "method: GET" in source
    assert "method: HEAD" in source
    assert "path: /usr/bin/python3*" in source
    assert "method: POST" not in source
    assert source.count("host:") == 3


def test_sandbox_installer_uses_only_the_minimal_pinned_chemistry_stack():
    completed = subprocess.run(
        ["bash", "-n", str(NVMOLKIT_INSTALLER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = NVMOLKIT_INSTALLER.read_text()
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "torch==2.7.1+cu128" in source
    assert "nvmolkit==0.5.0" in source
    assert "pandas==2.3.1" in source
    assert "matplotlib==3.10.3" in source
    assert "install.exit" in source
    assert "jupyter" not in source.lower()
    assert "seaborn" not in source.lower()


def test_gpu_probe_exercises_real_morgan_fingerprints_on_cuda():
    source = GPU_PROBE.read_text()
    compile(source, str(GPU_PROBE), "exec")

    assert "MorganFingerprintGenerator" in source
    assert "GetFingerprints" in source
    assert "torch.cuda.synchronize()" in source
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.get_device_name(0)" in source
    assert "(3, 32)" in source


def test_workspace_note_routes_agent_to_the_preinstalled_chemistry_runtime():
    source = WORKSPACE_TOOLS.read_text()

    assert "Do not install or upgrade packages" in source
    assert "nvmolkit-usage" in source
    assert "PYTHONPATH=/tmp/.local/lib/python3.13/site-packages" in source
    assert "/sandbox/.openclaw/workspace/data/sample_molecules.csv" in source
    assert "/sandbox/.openclaw/workspace/outputs" in source
    assert "not evidence of biological activity" in source


def test_acceptance_prompt_requests_one_bounded_edit_and_one_gpu_run():
    source = ACCEPTANCE_PROMPT.read_text()

    assert "HIGHLIGHT_THRESHOLD = 0.70" in source
    assert "HIGHLIGHT_THRESHOLD = 0.80" in source
    assert source.count("env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages") == 1
    assert "--output /sandbox/.openclaw/workspace/outputs/threshold-080" in source
    assert "Do not install packages" in source
    assert (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/threshold-080/similarity_heatmap.png"
        in source
    )
    assert (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/threshold-080/results.zip" in source
    )


def test_artifact_server_is_retry_safe_and_serves_only_outputs():
    completed = subprocess.run(
        ["bash", "-n", str(ARTIFACT_SERVER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = ARTIFACT_SERVER.read_text()
    assert "tmux has-session -t acs-artifacts" in source
    assert "python3 -m http.server 8765" in source
    assert "--bind 0.0.0.0" in source
    assert "--directory /sandbox/.openclaw/workspace/outputs" in source
