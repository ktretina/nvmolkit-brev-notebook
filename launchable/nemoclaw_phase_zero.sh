#!/usr/bin/env bash
set -Eeuo pipefail

readonly session_name="acs-phase-zero-install"
readonly script_path="${ACS_PHASE_ZERO_SCRIPT_PATH:-${HOME}/.config/acs-phase-zero/install.sh}"

if [[ "${ACS_PHASE_ZERO_DETACHED:-0}" != "1" ]]; then
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    printf 'NemoClaw installation session already exists.\n'
    exit 0
  fi
  tmux new-session -d -s "${session_name}"
  printf -v launch_command 'ACS_PHASE_ZERO_DETACHED=1 %q' "${script_path}"
  tmux send-keys -t "${session_name}" "${launch_command}" C-m
  printf 'NemoClaw installation started in tmux session %s.\n' "${session_name}"
  exit 0
fi

umask 077

readonly key_file="${ACS_PHASE_ZERO_KEY_FILE:-${HOME}/.config/acs-phase-zero/NVIDIA_INFERENCE_API_KEY}"
readonly state_dir="${ACS_PHASE_ZERO_STATE_DIR:-${HOME}/.local/state/acs-phase-zero}"
readonly status_file="${state_dir}/install.exit"
readonly first_status_file="${state_dir}/install.first.exit"
readonly resume_status_file="${state_dir}/install.resume.exit"
readonly install_ref="0d1cb93888c817daec44b2cc996afa75eebcbd46"
readonly installer_sha="b52f053a550fab90ab1dff4ab7f3a0b55b2506aeafd2062832e65632fdbcae70"
readonly install_url="https://raw.githubusercontent.com/NVIDIA/NemoClaw/${install_ref}/install.sh"
installer=""

mkdir -p -- "${state_dir}"
rm -f -- "${status_file}" "${first_status_file}" "${resume_status_file}"

record_status() {
  local path="$1"
  local exit_code="$2"
  rm -f -- "${path}"
  printf '%s\n' "${exit_code}" > "${path}"
  chmod 600 "${path}"
}

is_interrupted_provider_selection() {
  local session_file="${HOME}/.nemoclaw/onboard-session.json"
  [[ -f "${session_file}" && ! -L "${session_file}" && -O "${session_file}" ]] || return 1
  python3 - "${session_file}" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
try:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ValueError("unsafe session file")
        session = json.load(stream)
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

if not isinstance(session, dict):
    raise SystemExit(1)
failure = session.get("failure")
if not isinstance(failure, dict):
    raise SystemExit(1)
if not (
    session.get("status") == "failed"
    and session.get("resumable") is True
    and failure.get("step") == "provider_selection"
    and failure.get("interrupted") is True
):
    raise SystemExit(1)
PY
}

cleanup() {
  local exit_code=$?
  rm -f -- "${key_file}"
  if [[ -n "${installer}" ]]; then
    rm -f -- "${installer}"
  fi
  unset COMPATIBLE_API_KEY NVIDIA_INFERENCE_API_KEY || true
  record_status "${status_file}" "${exit_code}"
}
trap cleanup EXIT

unset COMPATIBLE_API_KEY NVIDIA_INFERENCE_API_KEY || true

if [[ ! -f "${key_file}" || -L "${key_file}" || ! -O "${key_file}" ]]; then
  printf 'Protected NVIDIA inference key file is missing or unsafe.\n' >&2
  exit 2
fi
if [[ "$(stat -c '%a' "${key_file}")" != "600" || ! -s "${key_file}" ]]; then
  printf 'Protected NVIDIA inference key file has the wrong mode or is empty.\n' >&2
  exit 2
fi

if ! python3 - "${key_file}" <<'PY'
from pathlib import Path
import sys

payload = Path(sys.argv[1]).read_bytes()
if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
    raise SystemExit(1)
key = payload[:-1]
if (
    not key
    or key == b"__NVIDIA_INFERENCE_API_KEY__"
    or any(byte < 0x21 or byte == 0x7F for byte in key)
):
    raise SystemExit(1)
PY
then
  printf 'Protected NVIDIA inference key is malformed.\n' >&2
  exit 2
fi

installer="$(mktemp "${TMPDIR:-/tmp}/nemoclaw-install.XXXXXX")"
curl -fsSL "${install_url}" -o "${installer}"
printf '%s  %s\n' "${installer_sha}" "${installer}" | sha256sum -c -

IFS= read -r COMPATIBLE_API_KEY < "${key_file}"
if [[ -z "${COMPATIBLE_API_KEY}" \
  || "${COMPATIBLE_API_KEY}" == __NVIDIA_INFERENCE_API_KEY_[_] \
  || "${COMPATIBLE_API_KEY}" =~ [[:space:]] \
  || "${COMPATIBLE_API_KEY}" =~ [[:cntrl:]] ]]; then
  printf 'Protected NVIDIA inference key is malformed.\n' >&2
  exit 2
fi
export COMPATIBLE_API_KEY
rm -f -- "${key_file}"

export NEMOCLAW_INSTALL_REF="${install_ref}"
export NEMOCLAW_NON_INTERACTIVE=1
export NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1
export NEMOCLAW_AGENT=openclaw
export NEMOCLAW_PROVIDER=custom
export NEMOCLAW_ENDPOINT_URL=https://inference-api.nvidia.com/v1
export NEMOCLAW_SANDBOX_NAME=acs-chemistry-agent
export NEMOCLAW_MODEL=nvidia/nvidia/nemotron-3-super-120b-a12b
export NEMOCLAW_PREFERRED_API=openai-completions
export NEMOCLAW_POLICY_TIER=balanced
export NEMOCLAW_POLICY_MODE=suggested
export NEMOCLAW_WEB_SEARCH_PROVIDER=none
export NEMOCLAW_SANDBOX_GPU=1
export NEMOCLAW_DASHBOARD_PORT=18789

if bash "${installer}"; then
  install_exit=0
else
  install_exit=$?
fi
record_status "${first_status_file}" "${install_exit}"

final_exit="${install_exit}"
if [[ "${install_exit}" == 1 \
  && -x "${HOME}/.local/bin/nemoclaw" ]] \
  && is_interrupted_provider_selection; then
  export PATH="${HOME}/.local/bin:${PATH}"
  if NEMOCLAW_ONBOARD_VALIDATION_TIMEOUT_SECONDS=60 \
    nemoclaw onboard --resume --non-interactive --yes \
      --yes-i-accept-third-party-software; then
    resume_exit=0
  else
    resume_exit=$?
  fi
  record_status "${resume_status_file}" "${resume_exit}"
  final_exit="${resume_exit}"
fi

exit "${final_exit}"
