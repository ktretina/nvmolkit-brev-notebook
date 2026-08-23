# ACS Final Fresh Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one ACS Brev Launchable that preserves the 8192-token live setting, uses the approved natural prompts, and passes one fresh four-prompt attendee journey.

**Architecture:** Keep the merged OpenShell filesystem-policy fix in the bounded workshop runner. Persist only the required OpenClaw model limit in setup, make the repository prompt/verifier contract match the already-published attendee prompts, then pin the private Console bootstrap to the resulting public source commit. Do not repair the failed diagnostic instance or expand the live updater.

**Tech Stack:** Bash, Python 3, pytest, OpenClaw/NemoClaw, Brev Console and CLI, GitHub.

---

### Task 1: Persist the live model-output limit

**Files:**
- Modify: `tests/test_acs_nemoclaw_launchable_setup.py`
- Modify: `launchable/acs_nemoclaw_launchable_setup.sh`

- [ ] Add a focused source-contract test that requires one strict `models.providers.inference.models.0.maxTokens` write of `8192` and one exact JSON readback check.
- [ ] Run the focused test and confirm it fails because setup does not persist the setting.
- [ ] Extend `configure_openclaw_runtime` with only that write and readback validation.
- [ ] Run the focused setup test and confirm it passes.

### Task 2: Synchronize the approved natural prompts

**Files:**
- Modify: `docs/acs-fall-2026-workshop.md`
- Modify: `scripts/run_acs_openclaw_live_qa.py`
- Modify: `scripts/verify_acs_openclaw_trajectory.py`
- Modify: `tests/test_acs_fall_2026_workshop_page.py`
- Modify: `tests/test_verify_acs_openclaw_trajectory.py`
- Modify: `tests/test_acs_live_instance_ops.py`
- Modify only if byte-locked expectations require it: `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl`

- [ ] Copy the four published natural prompt blocks exactly into the canonical workshop page without changing their scientific or safety constraints.
- [ ] Update only the corresponding prompt hashes and byte-locked test/fixture expectations.
- [ ] Run the page, verifier, and live-operation tests and confirm the accepted trajectory uses all four prompts in order.

### Task 3: Publish and repin source

**Files:**
- Modify after the source merge: `launchable/acs_console_bootstrap.sh`
- Modify after the source merge: `tests/test_acs_console_bootstrap.py`

- [ ] Review the combined diff, run the scoped suites and static checks, commit, push, and merge the source corrections.
- [ ] Replace the bootstrap pin and its test constant with that exact public merge commit.
- [ ] Run the bootstrap test, `bash -n`, the 16,384-byte limit, and `git diff --check`; then commit, push, and merge the repin.

### Task 4: Save and qualify the Launchable

**Resources:**
- Launchable: `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`
- Organization: `agents-in-ls`

- [ ] Render one private setup script with the workshop `inference.nvidia.com` key and validate it without displaying the key.
- [ ] Save only that setup script in the existing Launchable through the supported Brev Console; preserve the L4 and Secure Links and expose no attendee Setup values.
- [ ] Deploy one fresh task-owned instance and verify automatic browser authentication.
- [ ] Run Prompts 1-4 once in order, verify scientific explanations, CPU/GPU distinctions, images, download, repeat safety, and credential/interface safety.
- [ ] Stop and verify `STOPPED` for every task-owned paid test instance after the acceptance decision.
