#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly state_dir="/sandbox/.openclaw/workspace/.acs-phase-zero"
readonly status_file="${state_dir}/nvmolkit-install.exit"

mkdir -p -- "${state_dir}"
rm -f -- "${status_file}"

record_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${status_file}"
}
trap record_exit EXIT

python3 -m pip install \
  --user \
  --break-system-packages \
  --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 \
  nvmolkit==0.5.0 \
  pandas==2.3.1 \
  matplotlib==3.10.3

python3 - <<'PY'
import nvmolkit
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the OpenClaw sandbox")
if nvmolkit.__version__ != "0.5.0":
    raise RuntimeError(f"Unexpected nvMolKit version: {nvmolkit.__version__}")

print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
print(f"nvMolKit {nvmolkit.__version__}")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY
