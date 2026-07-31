#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR=".venv"
PORT="${JUPYTER_PORT:-8888}"
PID_FILE="${PROJECT_DIR}/.jupyter.pid"
LOG_FILE="${PROJECT_DIR}/jupyter.log"
JUPYTER_EXEC="${PROJECT_DIR}/${VENV_DIR}/bin/jupyter"

cd "${PROJECT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python >=3.11 and <3.15 is required." >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 15)))'; then
  echo "Error: Python >=3.11 and <3.15 is required; found $(python3 --version 2>&1)." >&2
  exit 1
fi

if [[ ! "${PORT}" =~ ^[0-9]{1,5}$ ]] || (( 10#${PORT} < 1 || 10#${PORT} > 65535 )); then
  echo "Error: JUPYTER_PORT must be an integer from 1 through 65535." >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  IFS= read -r existing_pid <"${PID_FILE}" || existing_pid=""
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    if ! python3 - "${existing_pid}" "${JUPYTER_EXEC}" "${PROJECT_DIR}" "${PORT}" <<'PY'
import os
import pathlib
import sys

pid, jupyter_exec, project_dir, port = sys.argv[1:]
try:
    argv = [os.fsdecode(value) for value in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
except OSError:
    raise SystemExit(1)

required = {
    jupyter_exec,
    f"--ServerApp.root_dir={project_dir}",
    f"--port={port}",
}
raise SystemExit(not required.issubset(argv))
PY
    then
      echo "Error: .jupyter.pid names a live process that cannot be verified as this Launchable's JupyterLab; refusing to stop it." >&2
      exit 1
    fi

    kill "${existing_pid}"
    for _ in {1..50}; do
      if ! kill -0 "${existing_pid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Error: the verified prior JupyterLab process did not stop; refusing to start another." >&2
      exit 1
    fi
  fi
  rm -f "${PID_FILE}"
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

nohup "${JUPYTER_EXEC}" lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --ServerApp.port_retries=0 \
  --no-browser \
  --ServerApp.root_dir="${PROJECT_DIR}" \
  --IdentityProvider.token='' \
  --PasswordIdentityProvider.hashed_password='' \
  >"${LOG_FILE}" 2>&1 &
jupyter_pid=$!
printf '%s\n' "${jupyter_pid}" >"${PID_FILE}"

if ! "${VENV_DIR}/bin/python" - "${jupyter_pid}" "${PORT}" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

pid = int(sys.argv[1])
url = f"http://127.0.0.1:{sys.argv[2]}/api"
deadline = time.monotonic() + 30

while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(2)
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(0.5)

raise SystemExit(1)
PY
then
  echo "Error: JupyterLab did not become ready on the configured port; recent log output follows." >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  kill "${jupyter_pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  exit 1
fi

echo "JupyterLab started (PID ${jupyter_pid}) on port ${PORT}. Open it only through the organization-only Brev Secure Link; do not expose this port publicly."
