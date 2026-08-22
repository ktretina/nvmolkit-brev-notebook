#!/usr/bin/env bash
set -Eeuo pipefail
set +x +v

umask 077

readonly repo_url="https://github.com/ktretina/nvmolkit-brev-notebook.git"
readonly repo_commit="ccd3d80093a7c161c4572a04e5661429c7eb8b87"
readonly source_root="${HOME}/.local/share/acs-nemoclaw-launchable"
readonly checkout_dir="${source_root}/source-${repo_commit}"

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

require_clean_checkout() {
  local checkout_status
  if ! checkout_status="$(
    git -C "${checkout_dir}" status --porcelain=v1 --untracked-files=all
  )"; then
    die "could not verify the pinned checkout state."
  fi
  [[ -z "${checkout_status}" ]] || die "the pinned checkout is not clean."
}

unset launch_key
launch_key=__NVIDIA_INFERENCE_API_KEY__
unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY COMPATIBLE_API_KEY
if [[ "${launch_key}" == __NVIDIA_INFERENCE_API_KEY_[_] ]]; then
  unset launch_key
  die "the ACS bootstrap is unrendered; use the private rendered bootstrap."
fi
if [[ -z "${launch_key}" || "${launch_key}" =~ [[:space:]] ]]; then
  unset launch_key
  die "the ACS bootstrap key must not be empty or contain whitespace."
fi
if [[ "${launch_key}" =~ [[:cntrl:]] ]]; then
  unset launch_key
  die "the ACS bootstrap key must not contain control characters."
fi

[[ "${repo_commit}" =~ ^[0-9a-f]{40}$ ]] ||
  die "repo_commit must be a full commit SHA."
command -v git >/dev/null 2>&1 || die "git is required."

install -d -m 700 -- "${source_root}"
reused_checkout=0
if [[ ! -e "${checkout_dir}" ]]; then
  git clone --quiet --no-checkout "${repo_url}" "${checkout_dir}"
elif [[ ! -d "${checkout_dir}/.git" ]]; then
  die "checkout path exists but is not a Git repository."
else
  reused_checkout=1
fi
chmod 700 "${checkout_dir}"

origin_url="$(git -C "${checkout_dir}" remote get-url --all origin)"
[[ "${origin_url}" == "${repo_url}" ]] || die "unexpected source repository."
if [[ "${reused_checkout}" == "1" ]]; then
  require_clean_checkout
fi

git -C "${checkout_dir}" fetch --quiet --tags origin
git -C "${checkout_dir}" cat-file -e "${repo_commit}^{commit}" ||
  die "pinned commit is unavailable."
git -C "${checkout_dir}" checkout --quiet --detach "${repo_commit}"
if git -C "${checkout_dir}" symbolic-ref --quiet HEAD >/dev/null 2>&1; then
  die "pinned checkout is not detached."
fi
[[ "$(git -C "${checkout_dir}" rev-parse --verify HEAD)" == "${repo_commit}" ]] ||
  die "pinned commit verification failed."
require_clean_checkout

export NVIDIA_INFERENCE_API_KEY="${launch_key}"
exec /bin/bash "${checkout_dir}/launchable/acs_nemoclaw_launchable_setup.sh"
