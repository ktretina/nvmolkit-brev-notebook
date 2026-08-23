# ACS Optional Attendee Text Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the notebook and OpenClaw instructions optional, accurate, science-first, and usable by an ACS chemist while preserving the tested four-prompt execution contract.

**Architecture:** Keep the four short attendee prompts in the canonical Markdown page. Keep commands, retry limits, output formatting, and scientific guardrails in the protected OpenClaw workspace guidance. Continue to bind the exact public prompt bytes with SHA-256, but stop requiring attendee prompts to contain internal `MEDIA:` paths. The OpenClaw interface is unchanged and remains a known acceptance blocker.

**Tech Stack:** Markdown, Python 3, pytest, SHA-256 prompt bindings, synthetic OpenClaw trajectory fixture, Git.

---

## Scope guard

- Change attendee text and the smallest prompt-loading code needed for that text.
- Do not change the Launchable setup, OpenClaw UI, runner behavior, scientific verifier rules, credentials, transaction code, or deployment code.
- Do not run live mutation or publish the Launchable in this text-only change.
- Run the existing verifier and live-operation suites once, after all local changes are complete.
- Record new non-Critical findings as residual risks. Stop for a Critical failure.

### Task 1: Replace stale guide-contract tests with the approved attendee contract

**Files:**

- Modify: `tests/test_acs_fall_2026_workshop_page.py`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Write failing tests for the optional two-journey structure**

Replace required-lab headings and assertions with these ordered sections:

```python
REQUIRED_SECTIONS = (
    "## Before you begin",
    "## Choose either journey",
    "## Notebook journey",
    "## OpenClaw journey",
    "## Four OpenClaw prompts",
    "## Results and downloads",
    "## Finish",
    "## Scientific limits",
    "## Official links",
)
```

Assert that the page:

- says both journeys are optional and independent;
- names `nvmolkit_nemotron_demo.ipynb`, **Start Agent**, **Approve & Run**, and **Run Objective Challenge**;
- says the notebook key is an NVIDIA inference API key and is entered only in the named Launchable field;
- says OpenClaw Setup is empty and no API key, gateway token, or password is requested;
- contains no `required lab`, `complete both`, `hosted mode`, stale module, or companion wording.

- [ ] **Step 2: Write failing tests for safe attendee language**

For each marked prompt, reject these attendee-visible implementation terms:

```python
FORBIDDEN_ATTENDEE_TERMS = (
    "/sandbox/",
    "PYTHONPATH=",
    "acs_workshop_runner.py",
    "MEDIA:",
    "answer_markdown",
    "state_id",
    "swap_id",
    "one-call budget",
    "Run only this exact command",
)
```

Assert that the recovery text tells the attendee to wait, explains safe repeated-prompt behavior, and says to take a screenshot and continue with another optional activity after an error.

- [ ] **Step 3: Keep the scientific content tests**

Retain assertions that the four prompts, in order:

- distinguish RDKit CPU work and nvMolKit GPU work correctly;
- state the fingerprint, Butina, sampled-conformer, MMFF94, and `D_min` limits;
- request one image and the workshop download;
- ask for the existing result after an accidental repeat.

Remove tests that require commands, media paths, six response headings, or objective state IDs in attendee prompts. Keep prompt marker uniqueness and SHA-256 tests, with temporary placeholder digests until Task 3 calculates the final values.

- [ ] **Step 4: Run the page test and confirm RED**

Run:

```bash
python3 -m pytest -q tests/test_acs_fall_2026_workshop_page.py
```

Expected: FAIL because the page still uses required-lab language and exposes commands and paths.

### Task 2: Rewrite the canonical attendee guide and four prompts

**Files:**

- Modify: `docs/acs-fall-2026-workshop.md`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Rewrite the journey instructions**

Use the approved design in `docs/superpowers/specs/2026-08-23-acs-optional-attendee-text-design.md`.

The notebook path must tell the attendee to:

1. create or copy an API key from `https://inference.nvidia.com`;
2. enter it only in the notebook Launchable's `NVIDIA_API_KEY` field;
3. open JupyterLab, open `nvmolkit_nemotron_demo.ipynb`, and run cells top to bottom;
4. use **Start Agent**, then **Approve & Run**, then **Run Objective Challenge**;
5. download wanted results and stop the environment.

Do not claim that the key must start with a specific prefix. Do not call it an NGC key.

The OpenClaw path must tell the attendee to leave Setup empty, open **Open Chemistry Agent**, start one new chat, send the prompts in order, wait for each answer and image, and use **Download Results** after Prompt 4.

- [ ] **Step 2: Replace the four marked prompt blocks exactly**

Copy the four approved prompt texts from the design specification without adding commands, paths, response-schema language, or hidden execution details.

- [ ] **Step 3: Add expected-result and recovery text**

Name the expected image or visible result for each prompt and the `results.zip` download, without a filesystem path. Add the approved repeated-prompt and screenshot recovery rule.

- [ ] **Step 4: Run the page test and inspect the remaining failures**

Run:

```bash
python3 -m pytest -q tests/test_acs_fall_2026_workshop_page.py
```

Expected: only prompt-digest assertions fail.

### Task 3: Preserve prompt binding without exposing internal media paths

**Files:**

- Modify: `scripts/run_acs_openclaw_live_qa.py`
- Modify: `scripts/verify_acs_openclaw_trajectory.py`
- Modify: `tests/test_verify_acs_openclaw_trajectory.py`
- Modify: `tests/test_acs_live_instance_ops.py`
- Modify: `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl`
- Modify: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Add failing loader tests**

Add focused tests that prove both prompt loaders accept the approved prompts even though those prompts do not end with `PROMPT_MEDIA`. Keep tests that reject wrong hashes, extra marker-region text, duplicate markers, malformed fences, symlinks, and oversized pages.

Expected contract:

```python
digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
if digest != expected_digest:
    raise ...
```

The answer verifier must continue to require and validate the correct `MEDIA:` line in assistant output. Only the attendee-prompt suffix requirement is removed.

- [ ] **Step 2: Run the focused loader tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tests/test_verify_acs_openclaw_trajectory.py -k 'load_prompt_contracts' \
  tests/test_acs_live_instance_ops.py -k 'qa_submits_exact_prompts'
```

Expected: FAIL because both loaders still require `prompt.endswith(media_line)`.

- [ ] **Step 3: Make the smallest loader change**

In `load_prompts()` and `load_prompt_contracts()`, remove `PROMPT_MEDIA` from the prompt-extraction loop and remove only the prompt-suffix check. Keep `PROMPT_MEDIA` for assistant-answer and artifact verification.

- [ ] **Step 4: Calculate and install the four exact hashes**

Use one small read-only Python command to extract the four marked prompt strings with the production fence rules and print their SHA-256 values. Copy the same four values into:

- `scripts/run_acs_openclaw_live_qa.py`;
- `scripts/verify_acs_openclaw_trajectory.py`;
- `tests/test_acs_live_instance_ops.py`;
- `tests/test_acs_fall_2026_workshop_page.py`.

Do not calculate hashes from manually copied text.

- [ ] **Step 5: Update the synthetic trajectory fixture**

Change only the four user-message contents in `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl` to the exact marked prompt bytes. Keep all assistant tool calls, tool results, scientific answers, media lines, and artifact evidence unchanged.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
python3 -m pytest -q \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_verify_acs_openclaw_trajectory.py -k 'load_prompt_contracts or accepts_valid_evidence' \
  tests/test_acs_live_instance_ops.py -k 'sources_pin_safe_interfaces or qa_submits_exact_prompts'
```

Expected: PASS.

### Task 4: Sync the public attendee page

**Files:**

- Modify: `/Users/ktretina/Desktop/Codex Working Folder/digital-biology-examples/acsfall26/README.md`
- Verify: `docs/acs-fall-2026-workshop.md`

- [ ] **Step 1: Fail closed on repository state**

In the public checkout, verify branch `gh-pages`, a clean worktree, and that local `HEAD` equals `origin/gh-pages`. Fetch only if needed to verify current remote state. Stop if the public branch moved or contains unrelated changes.

- [ ] **Step 2: Replace the public page**

Use `apply_patch` to make `acsfall26/README.md` byte-identical to the canonical attendee guide. Do not change other public files.

- [ ] **Step 3: Verify byte identity and public-page safety**

Run:

```bash
cmp \
  '/Users/ktretina/Desktop/Codex Working Folder/nvmolkit-brev-notebook/.worktrees/acs-fall-2026-launchable/docs/acs-fall-2026-workshop.md' \
  '/Users/ktretina/Desktop/Codex Working Folder/digital-biology-examples/acsfall26/README.md'
```

Expected: exit 0 with no output.

Search the public page for credentials and internal terms. Expected: no key value, `/sandbox/`, `MEDIA:`, runner command, state ID, or swap ID.

### Task 5: Review and verify once

**Files:**

- Review all files changed in Tasks 1–4.

- [ ] **Step 1: Review the diff against the approved scope**

Confirm:

- both journeys are optional and independent;
- notebook labels match the current notebook;
- all four prompt blocks match the approved specification;
- hidden command reliability remains in `launchable/acs_workspace_tools.md` and is not copied to attendee text;
- no setup, runner, verifier-science, UI, deployment, or transaction behavior changed.

- [ ] **Step 2: Run the existing verifier and live-operation suites once**

Run one command:

```bash
python3 -m pytest -q \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py
```

Expected: PASS. Do not repeat this full command unless a Critical test failure requires a narrow correction. Record new non-Critical findings as residual risks.

- [ ] **Step 3: Run static checks**

Run:

```bash
git diff --check
python3 -m py_compile \
  scripts/run_acs_openclaw_live_qa.py \
  scripts/verify_acs_openclaw_trajectory.py
```

Expected: PASS.

- [ ] **Step 4: Commit each repository separately**

In the implementation repository:

```bash
git add docs/acs-fall-2026-workshop.md \
  scripts/run_acs_openclaw_live_qa.py \
  scripts/verify_acs_openclaw_trajectory.py \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py \
  tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl
git commit -m "docs: simplify optional ACS workshop journeys"
```

In the public repository:

```bash
git add acsfall26/README.md
git commit -m "Update ACS Fall 2026 attendee guide"
```

Do not push either repository until the user gives separate publish approval.

## Residual acceptance risk

These text changes improve clarity but cannot make the workshop pass complete target-user acceptance. The unchanged OpenClaw operator interface can still expose implementation-oriented interaction and can still fail the intended attendee experience. A later UI/configuration decision and a fresh browser-based four-prompt journey remain necessary before claiming workshop acceptance.
