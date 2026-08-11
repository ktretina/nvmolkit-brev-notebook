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
readonly install_ref="0d1cb93888c817daec44b2cc996afa75eebcbd46"
readonly installer_sha="b52f053a550fab90ab1dff4ab7f3a0b55b2506aeafd2062832e65632fdbcae70"
readonly install_url="https://raw.githubusercontent.com/NVIDIA/NemoClaw/${install_ref}/install.sh"
installer=""

mkdir -p -- "${state_dir}"
rm -f -- "${status_file}"

cleanup() {
  local exit_code=$?
  rm -f -- "${key_file}"
  if [[ -n "${installer}" ]]; then
    rm -f -- "${installer}"
  fi
  unset NVIDIA_INFERENCE_API_KEY || true
  printf '%s\n' "${exit_code}" > "${status_file}"
}
trap cleanup EXIT

if [[ ! -f "${key_file}" || -L "${key_file}" || ! -O "${key_file}" ]]; then
  printf 'Protected NVIDIA inference key file is missing or unsafe.\n' >&2
  exit 2
fi
if [[ "$(stat -c '%a' "${key_file}")" != "600" || ! -s "${key_file}" ]]; then
  printf 'Protected NVIDIA inference key file has the wrong mode or is empty.\n' >&2
  exit 2
fi

installer="$(mktemp "${TMPDIR:-/tmp}/nemoclaw-install.XXXXXX")"
curl -fsSL "${install_url}" -o "${installer}"
printf '%s  %s\n' "${installer_sha}" "${installer}" | sha256sum -c -

IFS= read -r NVIDIA_INFERENCE_API_KEY < "${key_file}"
if [[ "${NVIDIA_INFERENCE_API_KEY}" != nvapi-* ]]; then
  printf 'NVIDIA inference key does not have the expected prefix.\n' >&2
  exit 2
fi
export NVIDIA_INFERENCE_API_KEY
rm -f -- "${key_file}"

export NEMOCLAW_INSTALL_REF="${install_ref}"
export NEMOCLAW_NON_INTERACTIVE=1
export NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1
export NEMOCLAW_AGENT=openclaw
export NEMOCLAW_PROVIDER=build
export NEMOCLAW_SANDBOX_NAME=acs-chemistry-agent
export NEMOCLAW_MODEL=nvidia/nemotron-3-super-120b-a12b
export NEMOCLAW_POLICY_TIER=balanced
export NEMOCLAW_POLICY_MODE=suggested
export NEMOCLAW_WEB_SEARCH_PROVIDER=none
export NEMOCLAW_SANDBOX_GPU=1
export NEMOCLAW_DASHBOARD_PORT=18789

bash "${installer}"
