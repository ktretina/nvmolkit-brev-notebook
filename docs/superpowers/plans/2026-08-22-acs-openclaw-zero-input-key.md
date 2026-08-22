# ACS OpenClaw Zero-Input Credential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove attendee API-key input from the ACS OpenClaw Launchable while preserving the existing protected NemoClaw/OpenShell credential flow and completing one fresh four-prompt browser acceptance run.

**Architecture:** Reuse the accepted zero-input notebook pattern. Git contains one secret-free bootstrap template and one ACS-specific offline renderer; the Brev Console stores the private rendered bootstrap and exposes no Launch parameters. Existing unified setup, Phase Zero, chemistry, verifier, and live-operation code remain unchanged.

**Tech Stack:** Bash, Python 3, pytest, Brev Console/CLI, NemoClaw/OpenShell, Playwright with an isolated browser profile.

---

### Task 1: Protect the zero-input bootstrap boundary

**Files:**
- Create: `launchable/render_acs_console_bootstrap.py`
- Modify: `launchable/acs_console_bootstrap.sh.in`
- Modify: `launchable/acs_console_bootstrap.sh`
- Test: `tests/test_acs_console_bootstrap.py`

- [ ] **Step 1: Write the failing bootstrap and renderer tests**

Add assertions that the public template contains exactly one
`__NVIDIA_INFERENCE_API_KEY__` sentinel, contains no credential, and fails
before Git when run unrendered. Add renderer tests for:

```python
rendered = renderer.render_bootstrap(template, "nvapi-test-canary")
assert "__NVIDIA_INFERENCE_API_KEY__" not in rendered
assert "nvapi-test-canary" in rendered
```

Also require `nvapi-` with nonempty suffix, reject CR/LF/NUL, reject an absent
or repeated sentinel, create only a new mode-`0600` file outside the repository,
reject symlinked parents and existing targets, and prove the canary never enters
stdout, stderr, Git children, or install children.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_acs_console_bootstrap.py
```

Expected: failure because the renderer does not exist and the current template
still reads an attendee-supplied environment variable.

- [ ] **Step 3: Implement the bounded renderer**

Create an ACS-only renderer based on the already accepted
`origin/main:launchable/render_setup.py` pattern. It must:

```python
SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
TEMPLATE_PATH = SCRIPT_DIR / "acs_console_bootstrap.sh"

def render_bootstrap(template_text: str, key: str) -> str:
    if template_text.count(SENTINEL) != 1:
        raise ValueError("bootstrap template must contain exactly one credential sentinel")
    if not key.startswith("nvapi-") or key == "nvapi-":
        raise ValueError("credential must be an NVIDIA API key beginning with nvapi-")
    if any(character in key for character in ("\r", "\n", "\x00")):
        raise ValueError("credential must be one line without NUL")
    return template_text.replace(SENTINEL, shlex.quote(key), 1)
```

Use `getpass.getpass`, `os.open` with `O_EXCL` and `O_NOFOLLOW` where available,
an opened parent-directory descriptor, exact device/inode revalidation, mode
`0600`, and cleanup of a partially created output on failure. Print only the
output path and a handling warning.

- [ ] **Step 4: Replace the bootstrap input with the sentinel**

At the top of both bootstrap files, use:

```bash
set +x +v
launch_key=__NVIDIA_INFERENCE_API_KEY__
unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY

if [[ "${launch_key}" == __NVIDIA_* ]]; then
  unset launch_key
  die "render a private Brev Console bootstrap before deployment."
fi
if [[ "${launch_key}" != nvapi-* || "${launch_key}" == nvapi- ]]; then
  unset launch_key
  die "the saved workshop credential is invalid."
fi
```

Keep the existing Git child shielding and final
`export NVIDIA_INFERENCE_API_KEY="${launch_key}"` handoff unchanged.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
python3 -m pytest -q tests/test_acs_console_bootstrap.py
```

Expected: all tests pass and no canary appears in output.

### Task 2: Remove attendee key instructions

**Files:**
- Modify: `launchable/ACS_LAUNCHABLE_FIELDS.md`
- Modify: `docs/acs-fall-2026-workshop.md`
- Test: `tests/test_acs_console_bootstrap.py`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Write the failing instruction tests**

Require the OpenClaw authoring sheet to say **No Launch parameters or Setup
values**, name the private renderer command, prohibit source/default storage,
and preserve the existing two Secure Links. Require the conversational attendee
section to start with deployment, waiting, opening OpenClaw, and creating a new
session, with no API-key field or entry step.

Keep the separate notebook Launchable instructions unchanged.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_fall_2026_workshop_page.py
```

Expected: failures on the current required `NVIDIA_INFERENCE_API_KEY` field and
the current attendee entry step.

- [ ] **Step 3: Apply the minimum instruction changes**

Replace only the OpenClaw launch-parameter block and conversational deployment
steps. State that the organizer provisions the workshop-only key in the saved
private setup body, attendees enter no key, and the key must be monitored and
revoked after the workshop. Do not change the four prompt blocks or hashes.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same two-file command. Expected: all tests pass and all four prompt
hash locks remain unchanged.

### Task 3: Verify and commit the reviewed source

**Files:**
- Verify only the files from Tasks 1 and 2

- [ ] **Step 1: Run the focused setup and attendee suites once**

```bash
python3 -m pytest -q \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_fall_2026_workshop_page.py
```

- [ ] **Step 2: Run the existing verifier and live-operation suites once**

```bash
python3 -m pytest -q \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py
```

- [ ] **Step 3: Verify the public files are secret-free**

Run fixed-name scans for credential sentinels and credential-shaped literals.
Do not read an actual credential or scan environment values. Confirm the setup
payload is no larger than 16,384 UTF-8 bytes and passes `bash -n`.

- [ ] **Step 4: Commit the reviewed source**

```bash
git add \
  launchable/render_acs_console_bootstrap.py \
  launchable/acs_console_bootstrap.sh.in \
  launchable/acs_console_bootstrap.sh \
  launchable/ACS_LAUNCHABLE_FIELDS.md \
  docs/acs-fall-2026-workshop.md \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_fall_2026_workshop_page.py
git commit -m "fix: remove ACS attendee credential input"
```

### Task 4: Repin and publish the bootstrap

**Files:**
- Modify: `launchable/acs_console_bootstrap.sh`
- Modify: `tests/test_acs_console_bootstrap.py`

- [ ] **Step 1: Record the reviewed source commit**

Run `git rev-parse HEAD` and use that exact 40-character value as the only
`repo_commit` in the public bootstrap and the test constant. Apply the two
literal replacements with `apply_patch`; do not render a credential.

- [ ] **Step 2: Verify the repin**

```bash
python3 -m pytest -q tests/test_acs_console_bootstrap.py
bash -n launchable/acs_console_bootstrap.sh
LC_ALL=C wc -c < launchable/acs_console_bootstrap.sh
```

Expected: tests pass, Bash syntax passes, and size is at most 16,384 bytes.

- [ ] **Step 3: Commit and push without force**

```bash
git add launchable/acs_console_bootstrap.sh tests/test_acs_console_bootstrap.py
git commit -m "chore: repin zero-input ACS bootstrap"
git push origin acs-fall-2026-launchable
git ls-remote origin refs/heads/acs-fall-2026-launchable
```

### Task 5: Update the exact saved Launchable

**Files:**
- Private temporary output only: `/private/tmp/acs-openclaw-workshop-setup.sh`

- [ ] **Step 1: Render one private Console body**

Run interactively:

```bash
python3 launchable/render_acs_console_bootstrap.py \
  /private/tmp/acs-openclaw-workshop-setup.sh
```

Enter the workshop-only `nvapi-` key only at the hidden prompt. Validate only
that the output is a regular owned mode-`0600` file, has one reviewed source
pin, passes `bash -n`, and is within the size limit. Do not print its content.

- [ ] **Step 2: Update only `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`**

In the authenticated Brev Console, keep the existing hardware, access, source,
and Secure Links. Replace only the saved setup body with the private rendered
file and remove every Launch parameter. Save once, then use Preview to confirm
that no Setup value is shown. Do not use an undocumented API or retry a failed
save without inspecting its visible state.

- [ ] **Step 3: Remove the private render**

After the Console save is confirmed, delete only
`/private/tmp/acs-openclaw-workshop-setup.sh` and verify it is absent.

### Task 6: Run fresh attendee acceptance and stop the instance

**Files:**
- Temporary browser profile, screenshots, DOM snapshots, and download under one
  task-specific `/private/tmp/acs-openclaw-acceptance-*` directory

- [ ] **Step 1: Deploy one fresh L4 instance**

Reconfirm organization `agents-in-ls`, exact Launchable ID, unique instance
name, one `g6.xlarge` L4 price, and stop authority. Dry-run without `--param` and
require no missing parameter. Create exactly once.

- [ ] **Step 2: Wait for the Launchable ready marker**

Check only fixed status files and service readiness. Validate credential file
presence, owner, type, and mode without reading its content. Require no API-key
environment variable in the post-setup shell.

- [ ] **Step 3: Run the real browser journey once**

Use Playwright from the installed local package with a new temporary browser
profile and download directory. Open the attendee OpenClaw Secure Link, require
automatic access with no token/password page, create one new session, and paste
the four canonical prompt blocks once in order. Capture one screenshot and DOM
snapshot after each prompt.

- [ ] **Step 4: Verify science, images, download, safety, and usability**

Require the four expected native PNGs, CPU/GPU explanations, bounded runner
receipts, safe repeat handling where specified, and clicked `results.zip`
download. Run the existing trajectory and archive verifier against the browser
evidence. Scan rendered text and downloaded bytes for credential names,
credential-shaped values, internal paths, raw dashboard controls, debug UI, and
admin actions. Follow only the written attendee instructions.

- [ ] **Step 5: Publish attendee instructions only after acceptance**

After the fresh browser pass, synchronize the accepted local workshop page to
`NVIDIA/digital-biology-examples:gh-pages/acsfall26/README.md`, verify the four
prompt regions remain byte-identical, commit, and push without force.

- [ ] **Step 6: Stop the exact paid instance and verify `STOPPED`**

Send one stop command for the task-owned instance, then poll read-only state
until Brev reports `STOPPED`. On the first unresolved acceptance blocker, stop
the instance and report NOT READY without expanding scope.
