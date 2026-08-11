#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly sandbox_name="acs-chemistry-agent"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly project_dir="$(dirname -- "${script_dir}")"
readonly workspace="/sandbox/.openclaw/workspace"
readonly result_dir="${workspace}/outputs/threshold-080"
readonly result_png="${workspace}/outputs/threshold-080/similarity_heatmap.png"
readonly result_zip="${workspace}/outputs/threshold-080/results.zip"
readonly result_summary="${workspace}/outputs/threshold-080/summary.json"

readonly phase_zero_script="${project_dir}/launchable/nemoclaw_phase_zero.sh"
readonly phase_zero_state="${HOME}/.local/state/acs-phase-zero"
readonly phase_zero_status="${phase_zero_state}/install.exit"
readonly key_dir="${HOME}/.config/acs-phase-zero"
readonly key_file="${key_dir}/NVIDIA_INFERENCE_API_KEY"
readonly state_dir="${HOME}/.local/state/acs-nemoclaw-launchable"
readonly ready_marker="${state_dir}/ready"
readonly proxy_pid_file="${state_dir}/openclaw-secure-link-proxy.pid"

readonly pytorch_policy="${project_dir}/launchable/pytorch-cu128-policy.yaml"
readonly sandbox_installer="${project_dir}/launchable/install_nvmolkit_in_sandbox.sh"
readonly gpu_probe="${project_dir}/launchable/nvmolkit_gpu_probe.py"
readonly skill_dir="${project_dir}/skills/nvmolkit"
readonly dataset="${project_dir}/data/sample_molecules.csv"
readonly chemistry_task="${project_dir}/acs_chemistry_task.py"
readonly chemistry_workflow="${project_dir}/chemistry_workflow.py"
readonly workspace_tools="${project_dir}/launchable/acs_workspace_tools.md"
readonly task_prompt="${project_dir}/launchable/acs_task_prompt.txt"
readonly artifact_server="${project_dir}/launchable/start_artifact_server.sh"
readonly dashboard_proxy="${project_dir}/launchable/openclaw_secure_link_proxy.mjs"

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

cleanup() {
  rm -f -- "${key_file}"
  unset NVIDIA_INFERENCE_API_KEY gateway_token ACS_DASHBOARD_TOKEN || true
}
trap cleanup EXIT

gpu_inventory=""
if ! gpu_inventory="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null)"; then
  die "this Launchable requires exactly one NVIDIA L4."
fi
gpu_count="$(printf '%s\n' "${gpu_inventory}" | awk 'NF { count += 1 } END { print count + 0 }')"
if [[ "${gpu_count}" != "1" || "${gpu_inventory}" != "NVIDIA L4" ]]; then
  die "this Launchable requires exactly one NVIDIA L4."
fi

if [[ -z "${NVIDIA_INFERENCE_API_KEY:-}" ]]; then
  die "NVIDIA_INFERENCE_API_KEY is required."
fi

required_assets=(
  "${phase_zero_script}"
  "${pytorch_policy}"
  "${sandbox_installer}"
  "${gpu_probe}"
  "${skill_dir}/SKILL.md"
  "${dataset}"
  "${chemistry_task}"
  "${chemistry_workflow}"
  "${workspace_tools}"
  "${task_prompt}"
  "${artifact_server}"
  "${dashboard_proxy}"
)
for asset in "${required_assets[@]}"; do
  [[ -f "${asset}" ]] || die "required project asset is missing: ${asset}"
done

install -d -m 700 -- "${key_dir}" "${phase_zero_state}" "${state_dir}"
rm -f -- "${ready_marker}"
printf '%s\n' "${NVIDIA_INFERENCE_API_KEY}" > "${key_file}"
chmod 600 "${key_file}"
unset NVIDIA_INFERENCE_API_KEY

if ! ACS_PHASE_ZERO_DETACHED=1 \
  ACS_PHASE_ZERO_KEY_FILE="${key_file}" \
  ACS_PHASE_ZERO_STATE_DIR="${phase_zero_state}" \
  bash "${phase_zero_script}" >/dev/null 2>&1; then
  die "the pinned NemoClaw installer failed; inspect ${phase_zero_status}."
fi
[[ -f "${phase_zero_status}" ]] || die "the NemoClaw installer status is missing."
[[ "$(<"${phase_zero_status}")" == "0" ]] || die "the NemoClaw installer status is not zero."

readonly nemoclaw="${HOME}/.local/bin/nemoclaw"
readonly openshell="${HOME}/.local/bin/openshell"
[[ -x "${nemoclaw}" && -x "${openshell}" ]] || die "NemoClaw or OpenShell is missing."
node_candidates=()
for candidate in "${HOME}"/.nvm/versions/node/*/bin/node; do
  [[ -x "${candidate}" ]] && node_candidates+=("${candidate}")
done
[[ "${#node_candidates[@]}" == "1" ]] || die "exactly one executable NVM Node binary is required."
readonly node_bin="${node_candidates[0]}"

dashboard_listeners="$(ss -H -ltn "sport = :18789")"
[[ -n "${dashboard_listeners}" ]] || die "the private OpenClaw dashboard is not listening."
ACS_DASHBOARD_LISTENERS="${dashboard_listeners}" python3 -c '
import ipaddress
import os

for line in os.environ["ACS_DASHBOARD_LISTENERS"].splitlines():
    fields = line.split()
    if len(fields) < 4:
        raise SystemExit("invalid dashboard listener record")
    endpoint = fields[3]
    if endpoint.startswith("["):
        host = endpoint[1 : endpoint.rfind("]")]
    else:
        host = endpoint.rsplit(":", 1)[0]
    if not ipaddress.ip_address(host).is_loopback:
        raise SystemExit("the raw OpenClaw dashboard is not loopback-only")
'

host_sha256() {
  sha256sum -- "$1" | awk '{ print $1 }'
}

expected_task_sha="$(python3 - "${chemistry_task}" <<'PY'
import hashlib
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_bytes()
old = b"HIGHLIGHT_THRESHOLD = 0.70\n"
new = b"HIGHLIGHT_THRESHOLD = 0.80\n"
if source.count(old) != 1:
    raise SystemExit("the prepared chemistry task has the wrong edit point")
print(hashlib.sha256(source.replace(old, new, 1)).hexdigest())
PY
)"
expected_dataset_sha="$(host_sha256 "${dataset}")"
expected_workflow_sha="$(host_sha256 "${chemistry_workflow}")"
expected_tools_sha="$(host_sha256 "${workspace_tools}")"
expected_prompt_sha="$(host_sha256 "${task_prompt}")"
expected_server_sha="$(host_sha256 "${artifact_server}")"
readonly expected_task_sha expected_dataset_sha expected_workflow_sha
readonly expected_tools_sha expected_prompt_sha expected_server_sha

"${nemoclaw}" "${sandbox_name}" policy add \
  --from-file "${pytorch_policy}" --yes

"${nemoclaw}" "${sandbox_name}" exec -- mkdir -p \
  "${workspace}/data" "${workspace}/outputs" "/tmp/acs-setup"
"${nemoclaw}" "${sandbox_name}" exec -- rm -rf -- \
  "/tmp/acs-setup/install_nvmolkit_in_sandbox.sh" \
  "/tmp/acs-setup/nvmolkit_gpu_probe.py" \
  "${workspace}/data/sample_molecules.csv" \
  "${workspace}/acs_chemistry_task.py" \
  "${workspace}/chemistry_workflow.py" \
  "${workspace}/TOOLS.md" \
  "${workspace}/acs_workspace_tools.md" \
  "${workspace}/acs_task_prompt.txt" \
  "${workspace}/start_artifact_server.sh"
"${openshell}" sandbox upload "${sandbox_name}" "${sandbox_installer}" \
  "/tmp/acs-setup"
"${openshell}" sandbox upload "${sandbox_name}" "${gpu_probe}" \
  "/tmp/acs-setup"
"${nemoclaw}" "${sandbox_name}" exec -- bash \
  "/tmp/acs-setup/install_nvmolkit_in_sandbox.sh"
"${nemoclaw}" "${sandbox_name}" exec -- bash -c \
  'test -f /sandbox/.openclaw/workspace/.acs-phase-zero/nvmolkit-install.exit && test "$(cat /sandbox/.openclaw/workspace/.acs-phase-zero/nvmolkit-install.exit)" = 0'
"${nemoclaw}" "${sandbox_name}" exec -- env \
  PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /tmp/acs-setup/nvmolkit_gpu_probe.py

"${nemoclaw}" "${sandbox_name}" skill install "${skill_dir}"
"${openshell}" sandbox upload "${sandbox_name}" "${dataset}" \
  "${workspace}/data"
"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_task}" \
  "${workspace}"
"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_workflow}" \
  "${workspace}"
"${openshell}" sandbox upload "${sandbox_name}" "${workspace_tools}" \
  "${workspace}"
"${nemoclaw}" "${sandbox_name}" exec -- mv -- \
  "${workspace}/acs_workspace_tools.md" "${workspace}/TOOLS.md"
"${openshell}" sandbox upload "${sandbox_name}" "${task_prompt}" \
  "${workspace}"
"${openshell}" sandbox upload "${sandbox_name}" "${artifact_server}" \
  "${workspace}"

"${nemoclaw}" "${sandbox_name}" exec -- rm -rf -- "${result_dir}"
"${nemoclaw}" "${sandbox_name}" exec -- mkdir -p -- "${result_dir}"

"${nemoclaw}" "${sandbox_name}" agent --session-id main --json --timeout 600 -m \
  "$(<"${task_prompt}")" >/dev/null

"${nemoclaw}" "${sandbox_name}" exec -- env \
  ACS_OUTPUT_DIR="${result_dir}" \
  ACS_EXPECTED_TASK_SHA="${expected_task_sha}" \
  ACS_EXPECTED_DATASET_SHA="${expected_dataset_sha}" \
  ACS_EXPECTED_WORKFLOW_SHA="${expected_workflow_sha}" \
  ACS_EXPECTED_TOOLS_SHA="${expected_tools_sha}" \
  ACS_EXPECTED_PROMPT_SHA="${expected_prompt_sha}" \
  ACS_EXPECTED_SERVER_SHA="${expected_server_sha}" \
  python3 -c '
import hashlib, json, os, zipfile
from pathlib import Path

output = Path(os.environ["ACS_OUTPUT_DIR"])
workspace = Path("/sandbox/.openclaw/workspace")
workspace_hashes = {
    "acs_chemistry_task.py": os.environ["ACS_EXPECTED_TASK_SHA"],
    "data/sample_molecules.csv": os.environ["ACS_EXPECTED_DATASET_SHA"],
    "chemistry_workflow.py": os.environ["ACS_EXPECTED_WORKFLOW_SHA"],
    "TOOLS.md": os.environ["ACS_EXPECTED_TOOLS_SHA"],
    "acs_task_prompt.txt": os.environ["ACS_EXPECTED_PROMPT_SHA"],
    "start_artifact_server.sh": os.environ["ACS_EXPECTED_SERVER_SHA"],
}
for name, expected_hash in workspace_hashes.items():
    path = workspace / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise SystemExit(f"protected workspace file changed: {name}")
expected_files = {
    output / "README.md",
    output / "summary.json",
    output / "top_10_pairs.csv",
    output / "similarity_heatmap.png",
    output / "results.zip",
}
actual_files = {path for path in output.iterdir()}
if actual_files != expected_files or any(not path.is_file() for path in actual_files):
    raise SystemExit("threshold-080 contains an unexpected artifact set")
png = output / "similarity_heatmap.png"
archive = output / "results.zip"
summary_path = output / "summary.json"
if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
    raise SystemExit("invalid threshold-080 PNG")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_threshold = 0.80
if summary["parameters"]["highlight_threshold"] != expected_threshold:
    raise SystemExit("threshold-080 summary has the wrong threshold")
if summary["gpu"]["gpu_name"] != "NVIDIA L4":
    raise SystemExit("threshold-080 summary has the wrong GPU")
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None or set(bundle.namelist()) != {
        "README.md", "summary.json", "top_10_pairs.csv", "similarity_heatmap.png"
    }:
        raise SystemExit("threshold-080 results.zip is invalid")
'

"${nemoclaw}" "${sandbox_name}" exec -- bash "${workspace}/start_artifact_server.sh"

# Keep the stock dashboard forward private on 127.0.0.1:18789.
proxy_process_matches() {
  local pid="$1"
  local process_exe="/proc/${pid}/exe"
  local process_cmdline="/proc/${pid}/cmdline"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "${process_exe}" && -r "${process_cmdline}" ]] || return 1
  [[ "$(stat -c '%u' "/proc/${pid}")" == "$(id -u)" ]] || return 1
  [[ "$(readlink -f "${process_exe}")" == "$(readlink -f "${node_bin}")" ]] || return 1
  mapfile -d '' -t process_argv < "${process_cmdline}"
  [[ "${#process_argv[@]}" -ge 2 ]] || return 1
  [[ "${process_argv[1]}" == "${dashboard_proxy}" ]]
}

if [[ -f "${proxy_pid_file}" ]]; then
  old_proxy_pid="$(<"${proxy_pid_file}")"
  if kill -0 "${old_proxy_pid}" 2>/dev/null; then
    proxy_process_matches "${old_proxy_pid}" ||
      die "the recorded proxy PID is owned by another process."
    kill -TERM "${old_proxy_pid}"
    for attempt in {1..30}; do
      if ! kill -0 "${old_proxy_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    ! kill -0 "${old_proxy_pid}" 2>/dev/null ||
      die "the prior OpenClaw Secure Link proxy did not stop."
  fi
  rm -f -- "${proxy_pid_file}"
fi
[[ -z "$(ss -H -ltn "sport = :18788")" ]] ||
  die "port 18788 is already owned by an untracked process."

gateway_token="$("${nemoclaw}" "${sandbox_name}" gateway-token --quiet 2>/dev/null)"
[[ -n "${gateway_token}" ]] || die "the OpenClaw gateway token is unavailable."
export ACS_DASHBOARD_TOKEN="${gateway_token}"
unset gateway_token
/usr/bin/setsid "${node_bin}" "${dashboard_proxy}" </dev/null >/dev/null 2>&1 &
proxy_pid=$!
unset ACS_DASHBOARD_TOKEN
sleep 1
kill -0 "${proxy_pid}" 2>/dev/null || die "the OpenClaw Secure Link proxy exited."
proxy_process_matches "${proxy_pid}" || die "the new proxy process identity is invalid."
proxy_pid_temp="$(mktemp "${state_dir}/openclaw-secure-link-proxy.pid.XXXXXX")"
printf '%s\n' "${proxy_pid}" > "${proxy_pid_temp}"
chmod 600 "${proxy_pid_temp}"
mv -f -- "${proxy_pid_temp}" "${proxy_pid_file}"

/usr/bin/setsid -f "${openshell}" forward start --background 0.0.0.0:8765 "${sandbox_name}" </dev/null >/dev/null 2>&1

wait_for_http() {
  local url="$1"
  local destination="$2"
  local label="$3"
  for attempt in {1..60}; do
    if curl -fsSL --max-time 2 --output "${destination}" "${url}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  die "${label} did not become ready."
}

wait_for_http "http://127.0.0.1:18788/" /dev/null "OpenClaw Secure Link proxy"
kill -0 "${proxy_pid}" 2>/dev/null || die "the OpenClaw Secure Link proxy stopped."
download_probe="$(mktemp "${state_dir}/results-zip.XXXXXX")"
wait_for_http \
  "http://127.0.0.1:8765/threshold-080/results.zip" \
  "${download_probe}" \
  "artifact download"
python3 -c '
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as bundle:
    if bundle.testzip() is not None or "summary.json" not in bundle.namelist():
        raise SystemExit("downloaded results.zip is invalid")
' "${download_probe}"
rm -f -- "${download_probe}"

ready_temp="${ready_marker}.tmp"
printf 'ready\n' > "${ready_temp}"
chmod 644 "${ready_temp}"
mv -f -- "${ready_temp}" "${ready_marker}"
printf 'ACS chemistry workspace is ready.\n'
