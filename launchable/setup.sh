#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

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

url = "http://127.0.0.1:8888/api/"
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

echo "Setup complete. Brev manages JupyterLab and its Secure Link on port 8888."
