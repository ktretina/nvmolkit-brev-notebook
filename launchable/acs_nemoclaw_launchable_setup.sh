#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly sandbox_name="acs-chemistry-agent"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly project_dir="$(dirname -- "${script_dir}")"
readonly workspace="/sandbox/.openclaw/workspace"

readonly phase_zero_script="${project_dir}/launchable/nemoclaw_phase_zero.sh"
readonly phase_zero_state="${HOME}/.local/state/acs-phase-zero"
readonly phase_zero_status="${phase_zero_state}/install.exit"
readonly key_dir="${HOME}/.config/acs-phase-zero"
readonly key_file="${key_dir}/NVIDIA_INFERENCE_API_KEY"
readonly state_dir="${HOME}/.local/state/acs-nemoclaw-launchable"
readonly ready_marker="${state_dir}/ready"
readonly workflow_smoke_log="${state_dir}/workflow-smoke.log"
readonly proxy_pid_file="${state_dir}/openclaw-secure-link-proxy.pid"
readonly artifact_forward_pid_file="${state_dir}/artifact-forward.pid"
readonly artifact_sentinel_name=".acs-artifact-service-ready"
readonly artifact_sentinel_content="acs-artifact-service-ready-v1"

readonly pytorch_policy="${project_dir}/launchable/pytorch-cu128-policy.yaml"
readonly sandbox_installer="${project_dir}/launchable/install_nvmolkit_in_sandbox.sh"
readonly gpu_probe="${project_dir}/launchable/nvmolkit_gpu_probe.py"
readonly skill_dir="${project_dir}/skills/nvmolkit"
readonly dataset="${project_dir}/data/sample_molecules.csv"
readonly provenance="${project_dir}/data/PROVENANCE.md"
readonly workshop_runner="${project_dir}/acs_workshop_runner.py"
readonly objective_challenge="${project_dir}/objective_challenge.py"
readonly chemistry_workflow="${project_dir}/chemistry_workflow.py"
readonly workspace_tools="${project_dir}/launchable/acs_workspace_tools.md"
readonly artifact_server="${project_dir}/launchable/start_artifact_server.sh"
readonly dashboard_proxy="${project_dir}/launchable/openclaw_secure_link_proxy.mjs"

proxy_pid=""
proxy_start_ticks=""
proxy_rollback_armed=0
artifact_forward_pid=""
artifact_forward_start_ticks=""
artifact_forward_rollback_armed=0
artifact_probe=""
setup_succeeded=0

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

phase() {
  printf 'Phase: %s\n' "$1"
}

cleanup() {
  local exit_code=$?
  set +e
  rm -f -- "${key_file}"
  unset NVIDIA_INFERENCE_API_KEY gateway_token ACS_DASHBOARD_TOKEN || true
  if [[ "${exit_code}" != "0" || "${setup_succeeded}" != "1" ]]; then
    rm -f -- "${ready_marker}"
    if [[ "${artifact_forward_rollback_armed}" == "1" ]]; then
      rollback_new_artifact_forward
    fi
    if [[ "${proxy_rollback_armed}" == "1" ]]; then
      rollback_new_proxy
    fi
  fi
  if [[ -n "${artifact_probe}" ]]; then
    rm -f -- "${artifact_probe}"
  fi
  trap - EXIT
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

phase "Validate hardware"
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
  "${provenance}"
  "${workshop_runner}"
  "${objective_challenge}"
  "${chemistry_workflow}"
  "${workspace_tools}"
  "${artifact_server}"
  "${dashboard_proxy}"
)
for asset in "${required_assets[@]}"; do
  [[ -f "${asset}" ]] || die "required project asset is missing: ${asset}"
done

phase "Install OpenClaw runtime"
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
export PATH="${HOME}/.local/bin:${PATH}"
node_candidates=()
for candidate in "${HOME}"/.nvm/versions/node/*/bin/node; do
  [[ -x "${candidate}" ]] && node_candidates+=("${candidate}")
done
[[ "${#node_candidates[@]}" == "1" ]] || die "exactly one executable NVM Node binary is required."
readonly node_bin="${node_candidates[0]}"

process_start_ticks() {
  local pid="$1"
  local stat_record=""
  local stat_fields=""
  local -a fields=()
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat_record < "/proc/${pid}/stat" || return 1
  stat_fields="${stat_record##*) }"
  [[ "${stat_fields}" != "${stat_record}" ]] || return 1
  read -r -a fields <<< "${stat_fields}"
  [[ "${#fields[@]}" -ge 20 && "${fields[19]}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${fields[19]}"
}

terminate_just_spawned_child() {
  local pid="$1"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    for attempt in {1..10}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  wait "${pid}" 2>/dev/null || true
}

proxy_process_matches() {
  local pid="$1"
  local expected_start_ticks="$2"
  local process_exe="/proc/${pid}/exe"
  local process_cmdline="/proc/${pid}/cmdline"
  local -a process_argv=()
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${expected_start_ticks}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "${process_exe}" && -r "${process_cmdline}" ]] || return 1
  [[ "$(stat -c '%u' "/proc/${pid}")" == "$(id -u)" ]] || return 1
  [[ "$(process_start_ticks "${pid}")" == "${expected_start_ticks}" ]] || return 1
  [[ "$(readlink -f "${process_exe}")" == "$(readlink -f "${node_bin}")" ]] || return 1
  mapfile -d '' -t process_argv < "${process_cmdline}"
  [[ "${#process_argv[@]}" -eq 2 ]] || return 1
  [[ "${process_argv[1]}" == "${dashboard_proxy}" ]]
}

artifact_forward_process_matches() {
  local pid="$1"
  local expected_start_ticks="$2"
  local process_exe="/proc/${pid}/exe"
  local process_cmdline="/proc/${pid}/cmdline"
  local -a process_argv=()
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${expected_start_ticks}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "${process_exe}" && -r "${process_cmdline}" ]] || return 1
  [[ "$(stat -c '%u' "/proc/${pid}")" == "$(id -u)" ]] || return 1
  [[ "$(process_start_ticks "${pid}")" == "${expected_start_ticks}" ]] || return 1
  [[ "$(readlink -f "${process_exe}")" == "$(readlink -f "${openshell}")" ]] || return 1
  mapfile -d '' -t process_argv < "${process_cmdline}"
  [[ "${#process_argv[@]}" -eq 5 ]] || return 1
  [[ "${process_argv[1]}" == "forward" ]] || return 1
  [[ "${process_argv[2]}" == "start" ]] || return 1
  [[ "${process_argv[3]}" == "0.0.0.0:8765" ]] || return 1
  [[ "${process_argv[4]}" == "${sandbox_name}" ]]
}

stop_tracked_proxy() {
  local record=""
  local old_proxy_pid=""
  local old_proxy_start_ticks=""
  if [[ ! -e "${proxy_pid_file}" && ! -L "${proxy_pid_file}" ]]; then
    return 0
  fi
  [[ -f "${proxy_pid_file}" && ! -L "${proxy_pid_file}" && -O "${proxy_pid_file}" ]] ||
    die "the recorded proxy PID file is unsafe."
  [[ "$(stat -c '%a' "${proxy_pid_file}")" == "600" ]] ||
    die "the recorded proxy PID file is unsafe."
  record="$(<"${proxy_pid_file}")"
  [[ "${record}" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] ||
    die "the recorded proxy PID file is unsafe."
  old_proxy_pid="${BASH_REMATCH[1]}"
  old_proxy_start_ticks="${BASH_REMATCH[2]}"
  if kill -0 "${old_proxy_pid}" 2>/dev/null; then
    proxy_process_matches "${old_proxy_pid}" "${old_proxy_start_ticks}" ||
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
}

stop_tracked_artifact_forward() {
  local record=""
  local old_artifact_forward_pid=""
  local old_artifact_forward_start_ticks=""
  if [[ ! -e "${artifact_forward_pid_file}" && ! -L "${artifact_forward_pid_file}" ]]; then
    return 0
  fi
  [[ -f "${artifact_forward_pid_file}" && ! -L "${artifact_forward_pid_file}" && -O "${artifact_forward_pid_file}" ]] ||
    die "the recorded artifact-forward PID file is unsafe."
  [[ "$(stat -c '%a' "${artifact_forward_pid_file}")" == "600" ]] ||
    die "the recorded artifact-forward PID file is unsafe."
  record="$(<"${artifact_forward_pid_file}")"
  [[ "${record}" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] ||
    die "the recorded artifact-forward PID file is unsafe."
  old_artifact_forward_pid="${BASH_REMATCH[1]}"
  old_artifact_forward_start_ticks="${BASH_REMATCH[2]}"
  if kill -0 "${old_artifact_forward_pid}" 2>/dev/null; then
    artifact_forward_process_matches "${old_artifact_forward_pid}" "${old_artifact_forward_start_ticks}" ||
      die "the recorded artifact-forward PID is owned by another process."
    kill -TERM "${old_artifact_forward_pid}"
    for attempt in {1..30}; do
      if ! kill -0 "${old_artifact_forward_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    ! kill -0 "${old_artifact_forward_pid}" 2>/dev/null ||
      die "the prior artifact forward did not stop."
  fi
  rm -f -- "${artifact_forward_pid_file}"
}

rollback_new_proxy() {
  if kill -0 "${proxy_pid}" 2>/dev/null &&
    proxy_process_matches "${proxy_pid}" "${proxy_start_ticks}"; then
    kill -TERM "${proxy_pid}" 2>/dev/null
    for attempt in {1..10}; do
      if ! kill -0 "${proxy_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${proxy_pid}" 2>/dev/null &&
      proxy_process_matches "${proxy_pid}" "${proxy_start_ticks}"; then
      kill -KILL "${proxy_pid}" 2>/dev/null
    fi
    wait "${proxy_pid}" 2>/dev/null
  fi
  if [[ -f "${proxy_pid_file}" && ! -L "${proxy_pid_file}" ]] &&
    [[ "$(<"${proxy_pid_file}")" == "${proxy_pid}:${proxy_start_ticks}" ]]; then
    rm -f -- "${proxy_pid_file}"
  fi
}

rollback_new_artifact_forward() {
  if kill -0 "${artifact_forward_pid}" 2>/dev/null &&
    artifact_forward_process_matches "${artifact_forward_pid}" "${artifact_forward_start_ticks}"; then
    kill -TERM "${artifact_forward_pid}" 2>/dev/null
    for attempt in {1..10}; do
      if ! kill -0 "${artifact_forward_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${artifact_forward_pid}" 2>/dev/null &&
      artifact_forward_process_matches "${artifact_forward_pid}" "${artifact_forward_start_ticks}"; then
      kill -KILL "${artifact_forward_pid}" 2>/dev/null
    fi
    wait "${artifact_forward_pid}" 2>/dev/null
  fi
  if [[ -f "${artifact_forward_pid_file}" && ! -L "${artifact_forward_pid_file}" ]] &&
    [[ "$(<"${artifact_forward_pid_file}")" == "${artifact_forward_pid}:${artifact_forward_start_ticks}" ]]; then
    rm -f -- "${artifact_forward_pid_file}"
  fi
}

stop_tracked_proxy
stop_tracked_artifact_forward
[[ -z "$(ss -H -ltn "sport = :18788")" ]] ||
  die "port 18788 is already owned by an untracked process."
[[ -z "$(ss -H -ltn "sport = :8765")" ]] ||
  die "port 8765 is already owned by an untracked process."

"${nemoclaw}" "${sandbox_name}" config set \
  --key models.providers.inference.timeoutSeconds \
  --value 300 \
  --config-accept-new-path \
  --restart >/dev/null 2>&1
provider_timeout="$("${nemoclaw}" "${sandbox_name}" config get \
  --key models.providers.inference.timeoutSeconds \
  --format json 2>/dev/null)"
[[ "${provider_timeout}" == "300" ]] ||
  die "the inference provider timeout was not set to 300 seconds."
readonly provider_timeout

phase "Verify private dashboard"
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
' >/dev/null 2>&1

host_sha256() {
  sha256sum -- "$1" | awk '{ print $1 }'
}

expected_runner_sha="$(host_sha256 "${workshop_runner}")"
expected_objective_sha="$(host_sha256 "${objective_challenge}")"
expected_workflow_sha="$(host_sha256 "${chemistry_workflow}")"
expected_dataset_sha="$(host_sha256 "${dataset}")"
expected_provenance_sha="$(host_sha256 "${provenance}")"
expected_tools_sha="$(host_sha256 "${workspace_tools}")"
readonly expected_runner_sha expected_objective_sha expected_workflow_sha
readonly expected_dataset_sha expected_provenance_sha expected_tools_sha

phase "Install chemistry tools"
"${nemoclaw}" "${sandbox_name}" policy add \
  --from-file "${pytorch_policy}" --yes >/dev/null 2>&1

"${nemoclaw}" "${sandbox_name}" exec -- mkdir -p \
  "${workspace}/data" "${workspace}/outputs" "/tmp/acs-setup" >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- rm -rf -- \
  "/tmp/acs-setup/install_nvmolkit_in_sandbox.sh" \
  "/tmp/acs-setup/nvmolkit_gpu_probe.py" \
  "${workspace}/data/sample_molecules.csv" \
  "${workspace}/data/PROVENANCE.md" \
  "${workspace}/acs_workshop_runner.py" \
  "${workspace}/objective_challenge.py" \
  "${workspace}/chemistry_workflow.py" \
  "${workspace}/TOOLS.md" \
  "${workspace}/acs_workspace_tools.md" \
  "${workspace}/start_artifact_server.sh" \
  "${workspace}/outputs/workshop" \
  "${workspace}/.acs-workshop-state" \
  "${workspace}/outputs/${artifact_sentinel_name}" \
  "${workspace}/outputs/threshold-080" \
  "${workspace}/acs_chemistry_task.py" \
  "${workspace}/acs_task_prompt.txt" >/dev/null 2>&1

"${openshell}" sandbox upload "${sandbox_name}" "${sandbox_installer}" \
  "/tmp/acs-setup" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${gpu_probe}" \
  "/tmp/acs-setup" >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- bash \
  "/tmp/acs-setup/install_nvmolkit_in_sandbox.sh" >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- bash -c \
  'test -f /sandbox/.openclaw/workspace/.acs-phase-zero/nvmolkit-install.exit && test "$(cat /sandbox/.openclaw/workspace/.acs-phase-zero/nvmolkit-install.exit)" = 0' \
  >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- env \
  PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /tmp/acs-setup/nvmolkit_gpu_probe.py >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" skill install "${skill_dir}" >/dev/null 2>&1

phase "Prepare workshop files"
"${openshell}" sandbox upload "${sandbox_name}" "${dataset}" \
  "${workspace}/data" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${provenance}" \
  "${workspace}/data" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${workshop_runner}" \
  "${workspace}" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${objective_challenge}" \
  "${workspace}" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${chemistry_workflow}" \
  "${workspace}" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${workspace_tools}" \
  "${workspace}" >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- mv -- \
  "${workspace}/acs_workspace_tools.md" "${workspace}/TOOLS.md" >/dev/null 2>&1
"${openshell}" sandbox upload "${sandbox_name}" "${artifact_server}" \
  "${workspace}" >/dev/null 2>&1

"${nemoclaw}" "${sandbox_name}" exec -- mkdir -m 700 -- "${workspace}/.acs-workshop-state" \
  >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- chmod g-s -- "${workspace}/.acs-workshop-state" \
  >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- chmod 0700 -- "${workspace}/.acs-workshop-state" \
  >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- bash -c \
  'test "$(stat -c "%a" /sandbox/.openclaw/workspace/.acs-workshop-state)" = "700"' \
  >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- env \
  ACS_EXPECTED_RUNNER_SHA="${expected_runner_sha}" \
  ACS_EXPECTED_OBJECTIVE_SHA="${expected_objective_sha}" \
  ACS_EXPECTED_WORKFLOW_SHA="${expected_workflow_sha}" \
  ACS_EXPECTED_DATASET_SHA="${expected_dataset_sha}" \
  ACS_EXPECTED_PROVENANCE_SHA="${expected_provenance_sha}" \
  ACS_EXPECTED_TOOLS_SHA="${expected_tools_sha}" \
  python3 -c '
import json
import os
import stat
from pathlib import Path

state = Path("/sandbox/.openclaw/workspace/.acs-workshop-state")
manifest = state / "manifest.json"
temporary = state / f".manifest-{os.getpid()}.tmp"
try:
    mode = os.lstat(manifest).st_mode
except FileNotFoundError:
    pass
else:
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SystemExit("unsafe workshop manifest")
payload = {
    "schema_version": 1,
    "files": {
        "TOOLS.md": os.environ["ACS_EXPECTED_TOOLS_SHA"],
        "acs_workshop_runner.py": os.environ["ACS_EXPECTED_RUNNER_SHA"],
        "chemistry_workflow.py": os.environ["ACS_EXPECTED_WORKFLOW_SHA"],
        "data/sample_molecules.csv": os.environ["ACS_EXPECTED_DATASET_SHA"],
        "data/PROVENANCE.md": os.environ["ACS_EXPECTED_PROVENANCE_SHA"],
        "objective_challenge.py": os.environ["ACS_EXPECTED_OBJECTIVE_SHA"],
    },
}
encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written < 1:
            raise OSError("short manifest write")
        remaining = remaining[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
try:
    os.replace(temporary, manifest)
    os.chmod(manifest, 0o444)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
' >/dev/null 2>&1

"${nemoclaw}" "${sandbox_name}" exec -- chmod 0444 -- \
  "${workspace}/acs_workshop_runner.py" \
  "${workspace}/objective_challenge.py" \
  "${workspace}/chemistry_workflow.py" \
  "${workspace}/data/sample_molecules.csv" \
  "${workspace}/data/PROVENANCE.md" \
  "${workspace}/TOOLS.md" \
  "${workspace}/.acs-workshop-state/manifest.json" >/dev/null 2>&1
"${nemoclaw}" "${sandbox_name}" exec -- env \
  PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py --help >/dev/null 2>&1
install -m 600 /dev/null "${workflow_smoke_log}"
if ! "${nemoclaw}" "${sandbox_name}" exec --no-tty --timeout 600 -- env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${workspace}:/tmp/.local/lib/python3.13/site-packages" \
  MPLCONFIGDIR=/tmp/acs-workshop-smoke-mpl \
  python3 -c '
import acs_workshop_runner as runner
import torch
from chemistry_workflow import WorkflowPhase

runner.verify_manifest(runner.DEFAULT_PATHS)
execution = runner.execute_workflow_prefix("optimize_conformers_mmff94")
expected_stages = (
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
)
if tuple(result.stage for result in execution.stage_results) != expected_stages:
    raise SystemExit("the workshop smoke did not execute the exact six stages")
if execution.state.phase is not WorkflowPhase.OPTIMIZED:
    raise SystemExit("the workshop smoke did not reach the optimized phase")
if (
    torch.cuda.device_count() != 1
    or torch.cuda.get_device_name(0) != "NVIDIA L4"
    or execution.gpu is None
    or execution.gpu.name != "NVIDIA L4"
):
    raise SystemExit("the workshop smoke did not use exactly one NVIDIA L4")
' >"${workflow_smoke_log}" 2>&1; then
  die "the full workshop smoke failed; inspect ${workflow_smoke_log}."
fi

"${nemoclaw}" "${sandbox_name}" exec -- env \
  ACS_ARTIFACT_SENTINEL_CONTENT="${artifact_sentinel_content}" \
  python3 -c '
import os
from pathlib import Path

sentinel = Path("/sandbox/.openclaw/workspace/outputs/.acs-artifact-service-ready")
sentinel.write_text(os.environ["ACS_ARTIFACT_SENTINEL_CONTENT"] + "\n", encoding="utf-8")
sentinel.chmod(0o444)
' >/dev/null 2>&1

"${nemoclaw}" "${sandbox_name}" exec -- bash "${workspace}/start_artifact_server.sh" \
  >/dev/null 2>&1

phase "Start attendee services"
# Keep the stock dashboard forward private on 127.0.0.1:18789.
gateway_token="$("${nemoclaw}" "${sandbox_name}" gateway-token --quiet 2>/dev/null)"
[[ -n "${gateway_token}" ]] || die "the OpenClaw gateway token is unavailable."
export ACS_DASHBOARD_TOKEN="${gateway_token}"
unset gateway_token
/usr/bin/setsid "${node_bin}" "${dashboard_proxy}" </dev/null >/dev/null 2>&1 &
proxy_pid=$!
if ! proxy_start_ticks="$(process_start_ticks "${proxy_pid}")"; then
  terminate_just_spawned_child "${proxy_pid}"
  die "could not record the new proxy process identity."
fi
proxy_rollback_armed=1
unset ACS_DASHBOARD_TOKEN
sleep 1
kill -0 "${proxy_pid}" 2>/dev/null || die "the OpenClaw Secure Link proxy exited."
proxy_process_matches "${proxy_pid}" "${proxy_start_ticks}" ||
  die "the new proxy process identity is invalid."
proxy_pid_temp="$(mktemp "${state_dir}/openclaw-secure-link-proxy.pid.XXXXXX")"
printf '%s\n' "${proxy_pid}:${proxy_start_ticks}" > "${proxy_pid_temp}"
chmod 600 "${proxy_pid_temp}"
mv -f -- "${proxy_pid_temp}" "${proxy_pid_file}"

# Keep OpenShell in foreground; this shell owns the background process and PID.
/usr/bin/setsid "${openshell}" forward start 0.0.0.0:8765 "${sandbox_name}" </dev/null >/dev/null 2>&1 &
artifact_forward_pid=$!
if ! artifact_forward_start_ticks="$(process_start_ticks "${artifact_forward_pid}")"; then
  terminate_just_spawned_child "${artifact_forward_pid}"
  die "could not record the new artifact-forward process identity."
fi
artifact_forward_rollback_armed=1
sleep 1
kill -0 "${artifact_forward_pid}" 2>/dev/null || die "the artifact forward exited."
artifact_forward_process_matches \
  "${artifact_forward_pid}" "${artifact_forward_start_ticks}" ||
  die "the new artifact-forward process identity is invalid."
artifact_forward_pid_temp="$(mktemp "${state_dir}/artifact-forward.pid.XXXXXX")"
printf '%s\n' "${artifact_forward_pid}:${artifact_forward_start_ticks}" > "${artifact_forward_pid_temp}"
chmod 600 "${artifact_forward_pid_temp}"
mv -f -- "${artifact_forward_pid_temp}" "${artifact_forward_pid_file}"

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
kill -0 "${artifact_forward_pid}" 2>/dev/null || die "the artifact forward stopped."
artifact_probe="$(mktemp "${state_dir}/artifact-sentinel.XXXXXX")"
wait_for_http "http://127.0.0.1:8765/${artifact_sentinel_name}" \
  "${artifact_probe}" "artifact download"
[[ "$(<"${artifact_probe}")" == "${artifact_sentinel_content}" ]] ||
  die "the artifact download sentinel is invalid."
rm -f -- "${artifact_probe}"
artifact_probe=""

ready_temp="${ready_marker}.tmp"
printf 'ready\n' > "${ready_temp}"
chmod 644 "${ready_temp}"
mv -f -- "${ready_temp}" "${ready_marker}"
printf 'ACS chemistry workspace is ready.\n'
setup_succeeded=1
