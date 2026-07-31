#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR=".venv"
PORT="${JUPYTER_PORT:-8888}"

cd "${PROJECT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python 3.11 or newer is required." >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Error: Python 3.11 or newer is required; found $(python3 --version 2>&1)." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

"${VENV_DIR}/bin/python" - <<'PY'
import nvmolkit
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required, but torch.cuda.is_available() is false")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"nvMolKit: {nvmolkit.__version__}")
PY

nohup "${VENV_DIR}/bin/jupyter" lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --no-browser \
  --ServerApp.root_dir="${PROJECT_DIR}" \
  --IdentityProvider.token='' \
  --PasswordIdentityProvider.hashed_password='' \
  >"${PROJECT_DIR}/jupyter.log" 2>&1 &
jupyter_pid=$!

echo "JupyterLab started (PID ${jupyter_pid}) on port ${PORT}. Open it only through the organization-only Brev Secure Link; do not expose this port publicly."
