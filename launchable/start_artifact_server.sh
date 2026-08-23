#!/usr/bin/env bash
set -Eeuo pipefail

readonly session_name="acs-artifacts"
readonly output_dir="/sandbox/.openclaw/workspace/outputs"

mkdir -p -- "${output_dir}"
if tmux has-session -t acs-artifacts 2>/dev/null; then
  exit 0
fi

tmux new-session -d -s "${session_name}" \
  "exec python3 -m http.server 8765 --bind 0.0.0.0 --directory /sandbox/.openclaw/workspace/outputs"
