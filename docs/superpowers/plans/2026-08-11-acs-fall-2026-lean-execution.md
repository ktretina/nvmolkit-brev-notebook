# ACS Fall 2026 Lean Workshop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one reliable ACS chemist workshop path that runs real nvMolKit work on an NVIDIA L4 through OpenClaw and Nemotron, shows native images, and provides useful downloadable chemistry files.

**Architecture:** Keep the fixed runner and stage artifacts already implemented. Add only useful chemistry exports, a simple safe results bundle, a small three-action objective state, four copy-paste prompts, and one fresh end-to-end Brev acceptance run. The long plan at `docs/superpowers/plans/2026-08-11-acs-fall-2026-attendee-workshop.md` remains historical; this plan supersedes its Tasks 4 through 13.

**Tech Stack:** Python 3.13 in the sandbox, nvMolKit 0.5.0, RDKit, pandas, Matplotlib, OpenClaw/NemoClaw, hosted Nemotron, Brev L4, Bash, pytest.

---

## Cut line

Keep fixed inputs and commands, API-key secrecy, private raw port 18789, safe artifact paths, no attendee package installation or arbitrary network use, one NVIDIA L4, at most three objective actions, native images, real downloads, a fresh live acceptance run, and exact instance cleanup.

Do not build transcript receipts, state-log receipt schemas, byte-identical prompt auditing, multi-target rollback recovery, exhaustive hostile-filesystem mutation matrices, or unrelated baseline-test repairs.

## Fixed attendee flow

1. Inspect the 256-molecule library and generate Morgan fingerprints.
2. Measure Tanimoto similarity and discover fused Butina clusters.
3. Generate representative conformers and optimize them with MMFF94.
4. Complete one bounded four-molecule diversity challenge with at most three actions.

### Task 1: Finish useful chemistry downloads

**Files:**

- Modify: `acs_workshop_runner.py`
- Modify: `tests/test_acs_workshop_runner.py`

- [ ] **Step 1: Preserve the existing Task 3 RED and GREEN evidence**

The current unstaged diff must retain the real-workflow tests for:

- `top_similarity_pairs.csv` and `similarity_matrix.csv`;
- `cluster_assignments.csv`;
- `mmff94_energies.csv` and `optimized_conformers.sdf`; and
- parsed `workflow_evidence.json` records E01 through E06.

- [ ] **Step 2: Run the focused and adjacent tests**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "similarity_csv or cluster_assignments or mmff94_csv or workflow_evidence"

env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py tests/test_chemistry_workflow.py
```

Expected: all selected tests pass. Do not add more export types.

- [ ] **Step 3: Run static and secret gates, then commit**

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff format --check \
  acs_workshop_runner.py tests/test_acs_workshop_runner.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff check \
  acs_workshop_runner.py tests/test_acs_workshop_runner.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m py_compile \
  acs_workshop_runner.py tests/test_acs_workshop_runner.py
git diff --check
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add downloadable workshop chemistry data"
```

### Task 2: Add a simple safe bundle and bounded objective

**Files:**

- Modify: `acs_workshop_runner.py`
- Modify: `tests/test_acs_workshop_runner.py`

- [ ] **Step 1: Write failing lean bundle tests**

Require stage output to be built under one task-owned temporary directory and renamed into place only after every declared file validates. A valid existing completed stage returns its existing closed envelope. Build `results.zip` through a temporary regular file and `os.replace`; include only public stage/objective files with safe relative names. Reject symlinks at the output root, selected stage directory, temporary directory, and ZIP target. Do not implement backup restoration or injected multi-replace rollback.

- [ ] **Step 2: Write failing bounded-objective tests**

After stage 6, store one mode-0600, regular, non-symlink JSON state with the dataset hash, current four molecule indices, current score, displayed actions, and attempt count. `objective-start` returns the current panel, score, state ID, and fixed displayed actions. `objective-step --state-id ID --swap-id ID` accepts only a currently displayed action, rejects stale IDs, and stops after success, no legal action, or three accepted actions.

Terminal output is limited to:

```text
README.md
objective_summary.json
objective_evidence.json
score_trajectory.png
final_panel.png
final_similarity_heatmap.png
```

The final ZIP is rebuilt after terminal output. Do not add history replay, matrix reconstruction, or transcript receipts.

- [ ] **Step 3: Implement only the tested behavior**

Reuse `objective_challenge.py` for scoring, legal actions, and figures. Keep public output below `outputs/workshop`; keep private objective state below `.acs-workshop-state` and exclude it from the ZIP.

- [ ] **Step 4: Verify and commit**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py tests/test_objective_challenge.py
```

Run Ruff, scoped strict mypy, compilation, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Add bounded workshop objective and results bundle"
```

### Task 3: Wire the Launchable and create four prompts

**Files:**

- Modify: `launchable/acs_nemoclaw_launchable_setup.sh`
- Modify: `launchable/acs_workspace_tools.md`
- Create: `launchable/acs_workshop_prompts.md`
- Modify: `tests/test_acs_nemoclaw_launchable_setup.py`
- Create: `tests/test_acs_workshop_prompts.py`

- [ ] **Step 1: Write failing setup tests**

Require setup to upload the runner and `objective_challenge.py`, remove only task-owned stale workshop output/state on full setup, create the closed read-only manifest after the seed turn, run `acs_workshop_runner.py --help`, and preserve the working threshold-0.80 seed, ports 18788/8765, proxy, key handling, and private 18789 behavior.

- [ ] **Step 2: Write failing four-prompt tests**

Each marked prompt must use only fixed runner commands and the installed nvMolKit skill. Prompts 1 through 3 each run their two named stages in order and end with one approved `MEDIA:` image. Prompt 4 uses `objective-start`, zero to three quoted `objective-step` calls from displayed IDs, stops at terminal status, shows `final_panel.png`, and directs the attendee to the Download Results Secure Link.

Reject package installation, `curl`, `wget`, arbitrary paths/options, raw port 18789, tokenized URLs, and secret-shaped values.

- [ ] **Step 3: Implement, verify, and commit**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_workshop_prompts.py
bash -n launchable/acs_nemoclaw_launchable_setup.sh
node --test tests/openclaw_secure_link_proxy.test.mjs
```

Run Ruff, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Add the ACS workshop prompts"
```

### Task 4: Run lean local acceptance and final review

**Files:**

- Verify all files changed in Tasks 1 through 3.

- [ ] **Step 1: Run the focused workshop suite**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  tests/test_chemistry_workflow.py \
  tests/test_objective_challenge.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_workshop_prompts.py \
  tests/test_openclaw_secure_link_proxy.py
```

- [ ] **Step 2: Run repository and static gates**

Run the repository suite once and record any unchanged baseline failures without repairing unrelated tests. Run Ruff, Bash syntax, Node proxy tests, Python compilation, `git diff --check`, and Gitleaks for the implementation range.

- [ ] **Step 3: Run one final specification and code-quality review**

Review the complete implementation against this lean plan. Fix only Critical or Important issues that affect attendee completion, secrets, bounded execution, scientific correctness, downloads, or live reliability.

### Task 5: Publish and qualify one fresh L4

**External approval gate:** Before public push or billable compute, show the exact commit, repository/branch, Launchable ID, organization, L4 configuration, displayed price, and stop/delete contract. Do not switch organizations.

- [ ] **Step 1: Push the reviewed branch and update the Console bootstrap**

Use the exact public 40-character commit in the existing bootstrap. Do not expose the inference key in output, files, process arguments, or chat.

- [ ] **Step 2: Deploy one fresh L4 and run the attendee journey**

Wait for setup readiness. Open the protected Open Chemistry Agent link, start one new session, and run the four exact prompt blocks. Confirm the hosted model is `inference/nvidia/nemotron-3-super-120b-a12b`, the GPU is one NVIDIA L4, the four native images render, and the downloaded ZIP contains the CSV, SDF, JSON, README, and PNG outputs.

- [ ] **Step 3: Check secrets and clean up**

Check setup output and the four assistant answers for key/token/tokenized-URL patterns without retaining raw secrets. Preserve only bounded non-secret diagnostics. Stop and delete the exact fresh instance on success, failure, timeout, or interruption.

### Task 6: Create the attendee reference sheet and hand off

**Files:**

- Create: `docs/acs-fall-2026-workshop.md`
- Create: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Write the concise page**

Include Brev signup, NVIDIA account and Nemotron API-key instructions, the repository, the notebook Launchable, the OpenClaw Launchable, setup/readiness guidance, the four byte-identical prompt blocks, download instructions, scientific limits, troubleshooting for model timeout, and instructions to stop or delete every workshop environment.

- [ ] **Step 2: Verify links, prompt identity, secrets, and layout**

The page test must check required URLs, prompt order/identity, absence of secret-shaped values and `BuildDoneVideo`, and concise section order. Perform read-only live checks of the public URLs immediately before handoff.

- [ ] **Step 3: Commit and deliver**

Run the focused page and workshop tests, static gates, and Gitleaks. Commit the page locally. Give the user the clickable local Markdown path, implementation and page commits, two Launchable URLs, fresh-L4 result, ZIP hash, and cleanup result. Do not publish a GitHub Page without separate approval.
