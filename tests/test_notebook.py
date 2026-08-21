import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_preserves_launch_and_separate_acceptance_gates():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    for instructions in (readme, fields):
        assert "Brev-managed Jupyter" in instructions
        assert "Only my organization" in instructions
        assert "Secure Link" in instructions
        assert "CPython 3.12" in instructions
        assert "mode `0600`" in instructions
        assert "hidden prompt" not in instructions.lower()
    lowered = readme.lower()
    for gate in ("local deterministic acceptance", "gpu acceptance", "hosted inference acceptance", "rendered deployment acceptance"):
        assert gate in lowered
    assert "bounded policy" in lowered
    assert "strict plan" in lowered
    assert "strict audit" in lowered
    assert "minimum tanimoto distance" in lowered
    assert "pytest -q" in readme
    assert "RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q" in readme
    assert "not yet live-qualified" in lowered


def test_setup_uses_brev_managed_python_and_leaves_jupyter_to_brev():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert setup.splitlines()[0] == "#!/bin/bash"
    assert '${HOME}/.venv/bin/python3' in setup
    assert "command -v python3.12" not in setup
    assert '"${PYTHON}" -m pip --version' in setup
    assert '"${PYTHON}" -m ensurepip --upgrade' in setup
    assert '"${PYTHON}" -m pip install --upgrade pip' in setup
    assert '"${PYTHON}" -m pip install -r requirements.txt' in setup
    assert 'url = "http://127.0.0.1:8888/api"' in setup
    assert '${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY' in setup
    assert '${HOME}/.jupyter/lab/user-settings/@jupyter-widgets/jupyterlab-manager' in setup
    assert '"saveState": true' in setup
    assert 'chmod 600 "${api_key_temp}"' in setup
    assert "NEMOTRON_MODEL" not in setup
    assert "JUPYTER_PORT" not in setup
    assert len(setup.encode("utf-8")) <= 16_384
    source_cleanup = setup.index("unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY")
    assert source_cleanup < setup.index('install -d -m 700 "${api_key_directory}"')
    assert setup.index("unset launch_api_key") < setup.index(
        'widget_settings_directory="${HOME}/.jupyter'
    )
    assert "export launch_api_key" not in setup
    assert all(forbidden not in setup for forbidden in ("jupyter lab", "nohup", "PID_FILE", "kill ", "-m venv"))
    assert "jupyterlab==" not in requirements


def _run_setup(tmp_path, setup_values):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / "nvmolkit-brev-notebook").symlink_to(REPO_ROOT, target_is_directory=True)
    managed_bin = fake_home / ".venv" / "bin"
    fake_bin = tmp_path / "bin"
    managed_bin.mkdir(parents=True)
    fake_bin.mkdir()
    log = tmp_path / "invocations.log"
    managed_python = managed_bin / "python3"
    managed_python.write_text("""#!/bin/bash
case "${1:-}" in
  -c) printf 'VERSION_CHECK %s\n' "${2:-}" >>"${INVOCATION_LOG}" ;;
  --version) printf 'Python 3.12.13\n' ;;
  -m) printf 'MODULE %s %s %s\n' "${2:-}" "${3:-}" "${4:-}" >>"${INVOCATION_LOG}"; [[ "${2:-} ${3:-}" == "pip --version" ]] && exit 1; [[ "${4:-}" == "-r" && ! -f "${5:-}" ]] && exit 93; true ;;
  -) payload="$(</dev/stdin)"; [[ "$payload" == *"torch.cuda.is_available"* ]] && printf 'SMOKE\n' >>"${INVOCATION_LOG}" || printf 'HEALTH\n' >>"${INVOCATION_LOG}" ;;
  *) exit 92 ;;
esac
""", encoding="utf-8")
    managed_python.chmod(0o755)
    (fake_bin / "uname").write_text("""#!/bin/bash
set -euo pipefail
[[ -z "${NVIDIA_INFERENCE_API_KEY+x}" ]]
[[ -z "${NVIDIA_API_KEY+x}" ]]
[[ -z "${launch_api_key+x}" ]]
printf 'ENV_CLEAN\n' >>"${INVOCATION_LOG}"
[[ "${1:-}" == "-s" ]] && printf 'Linux\n' || printf 'x86_64\n'
""", encoding="utf-8")
    (fake_bin / "uname").chmod(0o755)
    env = os.environ | {
        "HOME": str(fake_home),
        "INVOCATION_LOG": str(log),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
    } | setup_values
    copied_setup = tmp_path / "brev-generated-setup.sh"
    copied_setup.write_bytes((REPO_ROOT / "launchable" / "setup.sh").read_bytes())
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    result = subprocess.run(
        ["bash", str(copied_setup)],
        cwd=execution_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, fake_home, log


def test_setup_prefers_inference_key_and_runs_only_managed_runtime(tmp_path):
    primary_key = "sk-primary-setup-sentinel-must-not-leak"
    legacy_key = "sk-legacy-setup-sentinel-must-not-leak"
    result, fake_home, log = _run_setup(
        tmp_path,
        {
            "NVIDIA_INFERENCE_API_KEY": primary_key,
            "NVIDIA_API_KEY": legacy_key,
        },
    )
    assert result.returncode == 0, result.stderr
    key_directory = fake_home / ".config" / "nvmolkit"
    key_file = key_directory / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == primary_key
    assert key_directory.stat().st_mode & 0o777 == 0o700
    assert key_file.stat().st_mode & 0o777 == 0o600
    widget_settings = (
        fake_home
        / ".jupyter"
        / "lab"
        / "user-settings"
        / "@jupyter-widgets"
        / "jupyterlab-manager"
        / "plugin.jupyterlab-settings"
    )
    assert json.loads(widget_settings.read_text(encoding="utf-8")) == {
        "saveState": True
    }
    assert primary_key not in result.stdout + result.stderr
    assert legacy_key not in result.stdout + result.stderr
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert invocations.count("ENV_CLEAN") == 2
    assert any("sys.implementation.name" in line for line in invocations)
    assert invocations.index("MODULE ensurepip --upgrade ") < invocations.index("MODULE pip install --upgrade")
    assert invocations.index("MODULE pip install --upgrade") < invocations.index("MODULE pip install -r")
    assert invocations.index("MODULE pip install -r") < invocations.index("SMOKE") < invocations.index("HEALTH")


def test_setup_accepts_legacy_variable_name_only_for_sk_key(tmp_path):
    legacy_key = "sk-legacy-name-sentinel-must-not-leak"
    result, fake_home, _ = _run_setup(
        tmp_path,
        {"NVIDIA_API_KEY": legacy_key},
    )
    assert result.returncode == 0, result.stderr
    key_file = fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == legacy_key
    assert legacy_key not in result.stdout + result.stderr


def test_setup_rejects_invalid_primary_without_falling_back(tmp_path):
    invalid_primary = "nvapi-primary-must-not-leak"
    valid_legacy = "sk-legacy-must-not-be-used"
    result, fake_home, _ = _run_setup(
        tmp_path,
        {
            "NVIDIA_INFERENCE_API_KEY": invalid_primary,
            "NVIDIA_API_KEY": valid_legacy,
        },
    )
    assert result.returncode != 0
    assert "Inference Hub key beginning with sk-" in result.stderr
    assert invalid_primary not in result.stdout + result.stderr
    assert valid_legacy not in result.stdout + result.stderr
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_setup_rejects_empty_primary_without_falling_back(tmp_path):
    valid_legacy = "sk-legacy-must-not-be-used"
    result, fake_home, _ = _run_setup(
        tmp_path,
        {
            "NVIDIA_INFERENCE_API_KEY": "",
            "NVIDIA_API_KEY": valid_legacy,
        },
    )
    assert result.returncode != 0
    assert "Inference Hub key beginning with sk-" in result.stderr
    assert valid_legacy not in result.stdout + result.stderr
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_setup_rejects_legacy_build_key(tmp_path):
    legacy_build_key = "nvapi-legacy-build-key-must-not-leak"
    result, fake_home, _ = _run_setup(
        tmp_path,
        {"NVIDIA_API_KEY": legacy_build_key},
    )
    assert result.returncode != 0
    assert "Inference Hub key beginning with sk-" in result.stderr
    assert legacy_build_key not in result.stdout + result.stderr
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_launchable_contract_fixes_storage_model_port_and_one_setup_value():
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    assert "75 GiB" in fields
    assert "50 GiB" not in fields
    assert "required Text parameter `NVIDIA_INFERENCE_API_KEY`" in fields
    assert "required" in fields.lower()
    assert "no default" in fields.lower()
    assert "`nvidia/nvidia/nemotron-3-nano-30b-a3b`" in fields
    assert "`https://inference-api.nvidia.com/v1`" in fields
    assert "port `8888`" in fields
    assert "Remove `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT`" in fields
    assert "paste the current contents of `launchable/setup.sh`" in fields


def health_probe_source():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    start = setup.index("import json\nimport time\nimport urllib.error")
    return setup[start:setup.index("\nPY", start)]


def run_health_probe(mode):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if mode == "redirect":
                self.send_response(302 if self.path == "/api" else 200)
                if self.path == "/api":
                    self.send_header("Location", "/login")
                self.end_headers()
                return
            body = b'{"version": "4.4.5"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, format, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe = re.sub(r'url = "http://127\.0\.0\.1:8888/[^"]+"', f'url = "http://127.0.0.1:{server.server_port}/api"', health_probe_source())
        probe = probe.replace("deadline = time.monotonic() + 60", "deadline = time.monotonic() + 0.2").replace("time.sleep(1)", "time.sleep(0.01)")
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_health_probe_rejects_redirect_to_login_html():
    result = run_health_probe("redirect")
    assert result.returncode != 0
    assert "did not become healthy" in result.stderr


def test_health_probe_accepts_versioned_api_json():
    result = run_health_probe("valid")
    assert result.returncode == 0, result.stderr
    assert "health probe passed" in result.stdout
