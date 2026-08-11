#!/bin/bash
set -euo pipefail

if [[ -f "${PWD}/requirements.txt" && -f "${PWD}/demo_agent.py" ]]; then
  PROJECT_DIR="${PWD}"
elif [[ -f "${HOME}/nvmolkit-brev-notebook/requirements.txt" ]]; then
  PROJECT_DIR="${HOME}/nvmolkit-brev-notebook"
else
  echo "Error: could not find the cloned nvmolkit-brev-notebook project." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "Error: NVIDIA_API_KEY is required in Brev Setup values." >&2
  exit 1
fi

api_key_directory="${HOME}/.config/nvmolkit"
api_key_path="${HOME}/.config/nvmolkit/NVIDIA_API_KEY"
install -d -m 700 "${api_key_directory}"
chmod 700 "${api_key_directory}"
umask 077
api_key_temp="$(mktemp "${api_key_path}.tmp.XXXXXX")"
cleanup_api_key_temp() {
  rm -f -- "${api_key_temp}"
}
trap cleanup_api_key_temp EXIT
printf '%s' "${NVIDIA_API_KEY}" >"${api_key_temp}"
chmod 600 "${api_key_temp}"
mv -f -- "${api_key_temp}" "${api_key_path}"
trap - EXIT
unset NVIDIA_API_KEY

widget_settings_directory="${HOME}/.jupyter/lab/user-settings/@jupyter-widgets/jupyterlab-manager"
widget_settings_path="${widget_settings_directory}/plugin.jupyterlab-settings"
install -d -m 700 "${widget_settings_directory}"
widget_settings_temp="$(mktemp "${widget_settings_path}.tmp.XXXXXX")"
cleanup_widget_settings_temp() {
  rm -f -- "${widget_settings_temp}"
}
trap cleanup_widget_settings_temp EXIT
printf '%s\n' '{"saveState": true}' >"${widget_settings_temp}"
chmod 600 "${widget_settings_temp}"
mv -f -- "${widget_settings_temp}" "${widget_settings_path}"
trap - EXIT

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: this Launchable requires Linux x86-64 with CPython 3.12; found OS $(uname -s)." >&2
  exit 1
fi

machine="$(uname -m)"
if [[ "${machine}" != "x86_64" && "${machine,,}" != "amd64" ]]; then
  echo "Error: this Launchable requires Linux x86-64 with CPython 3.12; found architecture ${machine}." >&2
  exit 1
fi

is_cpython_312() {
  "$1" -c 'import sys; raise SystemExit(not (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)))'
}

managed_python="${HOME}/.venv/bin/python3"
if [[ ! -x "${managed_python}" ]] || ! is_cpython_312 "${managed_python}"; then
  echo "Error: Brev-managed ${managed_python} must be CPython 3.12." >&2
  exit 1
fi
PYTHON="${managed_python}"

echo "Installing into $("${PYTHON}" -c 'import sys; print(sys.executable)') ($("${PYTHON}" --version 2>&1))."
if ! "${PYTHON}" -m pip --version >/dev/null 2>&1; then
  "${PYTHON}" -m ensurepip --upgrade
fi
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r requirements.txt

"${PYTHON}" - <<'PY'
import nvmolkit
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required, but torch.cuda.is_available() is false")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"nvMolKit: {nvmolkit.__version__}")
PY

"${PYTHON}" - <<'PY'
import json
import time
import urllib.error
import urllib.request

url = "http://127.0.0.1:8888/api"
deadline = time.monotonic() + 60
last_error = "not ready"

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.geturl() != url:
                last_error = f"redirected to {response.geturl()}"
            elif response.status != 200:
                last_error = f"HTTP {response.status}"
            else:
                try:
                    payload = json.load(response)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    last_error = f"invalid JSON: {exc}"
                else:
                    version = payload.get("version") if isinstance(payload, dict) else None
                    if isinstance(version, str) and version.strip():
                        print(
                            "Brev-managed Jupyter health probe passed "
                            f"(version {version})."
                        )
                        break
                    last_error = "JSON response lacks a nonempty string version"
    except (OSError, urllib.error.URLError) as exc:
        last_error = str(exc)
    time.sleep(1)
else:
    raise RuntimeError(
        f"Brev-managed Jupyter did not become healthy at {url} within 60 seconds: "
        f"{last_error}"
    )
PY

echo "Setup complete. The launch credential is protected for notebook use. Brev manages JupyterLab and its Secure Link on port 8888."
