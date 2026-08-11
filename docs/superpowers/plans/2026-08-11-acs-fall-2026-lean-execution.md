# ACS Fall 2026 Lean Workshop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one reliable ACS chemist workshop path that runs real nvMolKit work on an NVIDIA L4 through OpenClaw and Nemotron, shows native images, and provides useful downloadable chemistry files.

**Architecture:** Keep the fixed runner, stage artifacts, and chemistry exports already implemented. Three paired lessons each execute one dependency prefix and publish two stages. A concise tool envelope keeps large facts in files, while one immutable eight-candidate context and one small mutable objective state support at most three retry-safe actions without GPU recomputation. The OpenClaw Launchable is the only required hands-on path; the notebook is an optional instructor companion. The long plan at `docs/superpowers/plans/2026-08-11-acs-fall-2026-attendee-workshop.md` remains historical; this plan supersedes its Tasks 4 through 13.

**Tech Stack:** Python 3.13 in the sandbox, nvMolKit 0.5.0, RDKit, pandas, Matplotlib, OpenClaw/NemoClaw, hosted Nemotron, Brev L4, Bash, pytest.

---

## Cut line

Keep fixed inputs and commands, API-key secrecy, private raw port 18789, safe artifact paths, no attendee package installation or arbitrary network use, one NVIDIA L4, at most three objective actions, native images, real downloads, a fresh clean-browser acceptance run, and exact instance cleanup.

Do not build transcript receipts, state-log receipt schemas, live transcript byte matching, multi-target rollback recovery, exhaustive hostile-filesystem mutation matrices, or unrelated baseline-test repairs. Do not claim representative chemical space, GPU speedup, unrestricted autonomous design, biological activity, or cross-molecule MMFF94 energy comparability.

## Fixed attendee flow

1. Inspect the 256-molecule convenience sample and generate radius-2, 1024-bit Morgan fingerprints in one GPU lesson.
2. Measure Tanimoto similarity and discover fused Butina clusters at a 0.40 Tanimoto-distance cutoff in one GPU lesson.
3. Request up to five conformers for each deterministic MMFF94-eligible selection and optimize the generated conformers with MMFF94 in one GPU lesson.
4. Complete one bounded four-molecule diversity challenge with at most three actions.

Each attendee answer uses this fixed structure: Question; What ran; Measured result with at most three facts; Meaning; Scientific limit; Image and download location. Prompt 4 is the primary wow moment: baseline panel and weakest-link `D_min`, selected swap, final panel, and measured improvement.

### Task 1: Finish useful chemistry downloads — complete

**Files:**

- Modified: `acs_workshop_runner.py`
- Modified: `tests/test_acs_workshop_runner.py`

- [x] Added `top_similarity_pairs.csv`, `similarity_matrix.csv`, `cluster_assignments.csv`, `mmff94_energies.csv`, `optimized_conformers.sdf`, and parsed E01–E06 `workflow_evidence.json`.
- [x] Verified raw similarity ordering, complete cluster coverage, source-row provenance, CSV/SDF record identity, finite MMFF94 values, and SDF coordinate round-trip behavior.
- [x] Verified 8 focused export tests, 157 combined runner/workflow tests, Ruff, compilation, diff checks, and Gitleaks.
- [x] Committed as `9598080434e93091098fb985cd97263d10fcde8a`.

### Task 2: Add three paired lessons, concise output, and a safe bundle — complete

**Files:**

- Modify: `acs_workshop_runner.py`
- Modify: `tests/test_acs_workshop_runner.py`

- [x] **Step 1: Write failing paired-lesson tests**

Add one fixed `run-lesson` command with exactly these choices and publications:

```text
data-and-representation -> inspect_library, generate_morgan_fingerprints
relationships-and-groups -> measure_tanimoto_similarity, discover_fused_butina_clusters
sampled-3d-geometry -> embed_representative_conformers, optimize_conformers_mmff94
```

Each command must call `execute_workflow_prefix` once at the pair's terminal stage and publish both retained `StageResult` objects. Prompt-facing code must not execute six separate stage commands.

- [x] **Step 2: Write failing concise-envelope tests**

Return only this information:

```text
schema_version, status, lesson, completed_stages,
results_zip_path, artifact_relative_zip_path
```

Each `completed_stages` item contains only `stage`, one concise measured `result`, `image_paths`, `summary_path`, `readme_path`, and `artifact_directory`. It must not contain the full summary or per-cluster/per-conformer records. Full facts remain in `summary.json`.

- [x] **Step 3: Write failing atomic-publication and ZIP tests**

Build each new stage in one task-owned temporary directory, validate all declared files, and rename it into place only when complete. Reuse a valid completed stage; reject an invalid or symlinked target. Do not implement backup restoration across stages.

Build `results.zip` through one temporary regular file and `os.replace`. Include only safe public paths:

```text
README.md
data/sample_molecules.csv
data/PROVENANCE.md
01-inspection/ through 06-mmff94/
07-objective/ when terminal
```

The root README maps the four workshop questions to artifact directories. Private state and the ZIP itself are excluded.

- [x] **Step 4: Implement, verify, and commit**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py tests/test_chemistry_workflow.py
```

Run Ruff, compilation, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Add paired ACS workshop lessons"
```

Completed as `1817b22eef5137e218c191de9d0c18a754f2eb62`: 167 focused and adjacent tests passed; Ruff, compilation, diff checks, and staged Gitleaks passed; specification and code-quality reviews found no remaining Critical or Important issue. This is local implementation evidence only.

### Task 3: Add the bounded diversity objective — complete

**Files:**

- Modify: `acs_workshop_runner.py`
- Modify: `tests/test_acs_workshop_runner.py`
- Read: `objective_challenge.py`
- Read: `tests/objective_fixtures.py`

- [x] **Step 1: Write failing private-state tests**

At the end of the third lesson, write two mode-0600 regular non-symlink files atomically:

1. Immutable context: eight candidate IDs, molecule indices, source rows, cluster IDs, the validated 8×8 Tanimoto-distance matrix, baseline, attainable benchmark, target, dataset hash, and fixed profile.
2. Mutable state: current panel, current action menu, accepted attempt count, terminal status, up to three measured attempt records for the score trajectory and accepted-swap report, and the exact last accepted `(state_id, swap_id)` plus its result envelope.

Objective steps must not rerun nvMolKit, RDKit embedding, or MMFF94.

After this state exists, extend third-lesson cache validation: a cached third lesson is valid only when its immutable objective context and compatible mutable state also validate. Preserve valid initial, progressed, or terminal objective state; create initial state only when no valid objective progress exists.

- [x] **Step 2: Write failing action and retry tests**

`objective-start` returns the baseline panel, weakest-link `D_min`, limiting pair, target, state ID, and at most three fixed legal actions. `objective-step --state-id ID --swap-id ID` accepts only a displayed action tied for the maximum predicted `D_min` and stops at target, no legal improvement, or three accepted actions.

An exact duplicate of the most recently committed `(state_id, swap_id)` returns the already committed current or terminal envelope. Every other stale, invented, or fourth action fails without mutation. This is retry safety, not a replay log.

- [x] **Step 3: Write failing terminal-artifact tests**

Terminal output is exactly:

```text
README.md
objective_summary.json
objective_evidence.json
score_trajectory.png
final_panel.png
final_similarity_heatmap.png
```

The summary reports baseline, final, target, limiting pair, accepted swaps, and terminal reason. Python computes every score; Nemotron only selects a displayed action and explains the measured result. Rebuild the public ZIP after terminal publication.

- [x] **Step 4: Implement, verify, and commit**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py tests/test_objective_challenge.py
```

Run Ruff, compilation, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Add the bounded ACS diversity objective"
```

Completed as `2c9c022e29c9150d1d9e7695d5c3bf77eeb7a0b1`: the bounded objective persists one immutable eight-candidate context and one small retry-safe state, performs no objective-time GPU recomputation, publishes the exact six-file terminal result, and extends the hash-bound lesson archive. The final repair rejects skipped attempt sequences, preserves a terminal ZIP when lesson rerun validation fails, validates the bound archive before a cached pending retry, and enforces 8 MiB per-member and 32 MiB aggregate archive limits before decompression. Fresh verification passed 198 runner/objective tests, Ruff, compilation, diff checks, and staged Gitleaks. Independent specification and code-quality reviews found no remaining Critical or Important issue. This is local implementation evidence only.

### Task 4: Rewire the Launchable without a hidden model turn — complete

**Files:**

- Modify: `launchable/acs_nemoclaw_launchable_setup.sh`
- Modify: `launchable/acs_workspace_tools.md`
- Modify: `tests/test_acs_nemoclaw_launchable_setup.py`
- Modify: `tests/test_nemoclaw_phase_zero_setup.py`

- [x] **Step 1: Write failing setup-source tests**

Require setup to upload the runner, `objective_challenge.py`, `chemistry_workflow.py`, the fixed CSV, `data/PROVENANCE.md`, `TOOLS.md`, and the artifact server. Create the closed read-only manifest after upload and before runner smoke.

Remove the hidden threshold-0.80 agent turn, its source edit, its threshold artifacts, and its validation. Do not replace it with another setup-time Nemotron request. Keep the pinned install, real small nvMolKit CUDA probe, 300-second provider timeout, runner manifest/help smoke, artifact-server readiness, protected proxy on 18788, download service on 8765, secret handling, and private raw 18789.

- [x] **Step 2: Add safe progress and rerun behavior**

Print short phase names without raw installer output, keys, tokens, or tokenized URLs. On a full setup rerun, remove only setup-owned workshop output/state and recreate the manifest. Do not delete unrelated attendee files.

- [x] **Step 3: Implement, verify, and commit**

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py \
  tests/test_openclaw_secure_link_proxy.py
bash -n launchable/acs_nemoclaw_launchable_setup.sh
node --test tests/openclaw_secure_link_proxy.test.mjs
```

Run Ruff, compilation, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Wire the lean ACS workshop Launchable"
```

Completed as `e0d98d7332cf5b030ef16f8d00c9fc3022305133`: setup now uploads the reviewed workshop runner, objective domain, workflow, fixed data and provenance, tools note, and artifact server; creates the exact read-only six-file manifest; runs only deterministic L4, CUDA, runner, proxy, and download-service checks; and makes no setup-time model request. Exact process identity, failure rollback, a setup-owned download sentinel, and coherent ready-state cleanup support fresh runs and bounded reruns. Fresh verification passed 31 setup tests, three proxy tests, Bash syntax, Ruff, compilation, diff checks, and staged Gitleaks. Independent specification and code-quality reviews found no remaining Critical or Important issue. This is local implementation evidence only.

### Task 5: Create the canonical attendee page and four prompts

**Files:**

- Create: `docs/acs-fall-2026-workshop.md`
- Create: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Write the concise page**

Use this page as the only prompt source. Label the notebook Launchable as an optional instructor-led companion and the OpenClaw Launchable as the required hands-on path. Put Brev account creation and NVIDIA/Nemotron key generation in pre-work.

Each of four marked prompt blocks must be self-contained, use only the fixed runner CLI, and require the fixed answer structure. Prompt 1 reads the installed nvMolKit skill once. Prompts 1–3 call one paired lesson each and display respectively `library_preview.png`, `cluster_sizes.png`, and `optimized_structures.png`. Prompt 4 uses `objective-start`, zero to three quoted displayed actions tied for the maximum predicted `D_min`, and displays `final_panel.png`.

- [ ] **Step 2: Add exact scientific framing**

The page and prompts must state:

- the ChEMBL data are a deterministic 256-record convenience sample, not representative chemical space;
- cutoff 0.40 is Tanimoto distance;
- deterministic selected molecules are not centroids, medoids, or globally optimal representatives;
- Morgan/Tanimoto conclusions depend on the radius-2, 1024-bit hashed fingerprint and similarity 1.0 does not prove molecular identity;
- MMFF94 energies compare sampled conformers within one molecule only;
- `D_min` is the weakest-link diversity score within eight fixed candidates; and
- the run proves real GPU execution, not acceleration or speedup.

Reject install/network instructions, arbitrary runner options, raw port 18789, tokenized URLs, secret-shaped values, `BuildDoneVideo`, and claims of unrestricted autonomous design.

- [ ] **Step 3: Verify and commit the draft page**

Tests check required URLs, section order, exactly four prompt markers, fixed runner commands, response structure, scientific limits, secrets, optional-vs-required lab wording, download instructions, and stop/delete guidance.

Run focused page tests, Ruff, `git diff --check`, and staged Gitleaks. Commit with:

```bash
git commit -m "Add the ACS workshop attendee guide"
```

### Task 6: Run lean local acceptance and final implementation review

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
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_openclaw_secure_link_proxy.py
```

- [ ] **Step 2: Run proportional repository and static gates**

Run the repository suite once and record unchanged baseline failures without repairing unrelated tests. Run Ruff, Bash syntax, Node proxy tests, Python compilation, `git diff --check`, and Gitleaks for the implementation range. Do not add strict mypy without a repository configuration.

- [ ] **Step 3: Run one final specification and code-quality review**

Review the complete implementation against this plan. Fix only Critical or Important issues that affect attendee completion, secrets, bounded execution, scientific correctness, downloads, or live reliability.

### Task 7: Publish and qualify one fresh L4

**External approval gate:** Before public push or billable compute, show the exact commit, repository and branch, Launchable ID, organization, L4 configuration, displayed price, and stop/delete contract. Do not switch organizations.

- [ ] **Step 1: Push and update the Console bootstrap**

Use the exact public 40-character commit. Do not expose the inference key in output, files, process arguments, or chat.

- [ ] **Step 2: Run one clean-browser attendee journey**

Use only `docs/acs-fall-2026-workshop.md`; do not use SSH, a terminal, or repository knowledge to complete the four prompts. Confirm real Nemotron, OpenClaw, nvMolKit, and one NVIDIA L4; four native images; a ZIP that opens and contains the expected CSV, SDF, JSON, README, PNG, input CSV, and provenance; and no facilitator repair.

Before deployment, record the allocated hands-on lab duration. Record setup time, each prompt time, any retry, model, GPU, and ZIP hash. Pass the rehearsal only when all four prompts finish without facilitator repair in at most half of that allocated period, leaving the other half for account issues and one recovery attempt. Do not publish timing guidance unless measured unambiguously.

- [ ] **Step 3: Check secrets and clean up**

Check setup output and four assistant answers for key, token, and tokenized-URL patterns without retaining raw secrets. Preserve only bounded non-secret diagnostics. Stop and delete the exact fresh instance on success, failure, timeout, or interruption.

### Task 8: Finalize the attendee record and hand off

- [ ] Add only measured live evidence, confirmed troubleshooting, and any demonstrated attendee blocker to the page. Do not change the four accepted prompt blocks without rerunning live acceptance.
- [ ] Recheck public links, focused page/workshop tests, static gates, Gitleaks, branch status, and the exact stopped/deleted instance evidence.
- [ ] Commit the final local page and deliver its clickable path, implementation/page commits, both Launchable URLs, fresh-L4 result, measured timing scope, ZIP hash, and cleanup result. Do not publish a GitHub Page without separate approval.
