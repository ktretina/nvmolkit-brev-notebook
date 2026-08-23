# ACS Live Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the reviewed OpenClaw workshop with current `main`, correct the optional notebook journey, and publish only after one fresh attendee validation succeeds.

**Architecture:** Preserve the nvMolKit 0.6 normalized fused-Butina path used by the notebooks and the explicit RDKit CPU clustering path used by OpenClaw. Keep the four hash-bound OpenClaw prompts unchanged. Treat GitHub publication, Brev Console persistence, fresh deployment, browser acceptance, and paid-instance cleanup as separate gates.

**Tech Stack:** Python 3, nvMolKit 0.6, RDKit, pytest, GitHub Actions, Brev Console/CLI, OpenClaw.

---

### Task 1: Correct the optional notebook journey

**Files:**
- Modify: `docs/acs-fall-2026-workshop.md`
- Modify: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] Change the page test first to require the three current notebook filenames, Module 1 direct nvMolKit work, hosted Modules 2–3, and zero attendee key entry.
- [ ] Run the page test and confirm RED because it still names `nvmolkit_nemotron_demo.ipynb` and asks attendees for a key.
- [ ] Replace only the notebook-journey text. Keep both journeys optional and keep all four OpenClaw prompt bytes unchanged.
- [ ] Run the page test and confirm GREEN.

### Task 2: Merge current `main` with both clustering backends

**Files:**
- Modify during merge: `chemistry_workflow.py`
- Modify during merge: `tests/test_chemistry_workflow.py`

- [ ] Merge `origin/main` without rewriting history and reproduce the two conflicts.
- [ ] Resolve `_fused_butina` with the nvMolKit 0.6 `return_centroids` argument.
- [ ] For `backend="fused"`, call with `return_centroids=True` and normalize with `normalize_fused_butina_result`; for `backend="rdkit"`, retain `_rdkit_butina_clusters(validated_similarity_matrix(state), cutoff=cutoff)`.
- [ ] Preserve tests for the normalized fused result and add/retain tests for the RDKit CPU backend.
- [ ] Run `tests/test_chemistry_workflow.py` and `tests/test_acs_workshop_runner.py`.
- [ ] Commit the merge and push the existing PR branch.

### Task 3: Verify and merge the pull request

**Files:**
- Verify repository test suites and PR checks.

- [ ] Run the page, bootstrap, chemistry, OpenClaw verifier, and live-operation suites once after integration.
- [ ] Inspect PR #5; fix only failures caused by this integration.
- [ ] Merge PR #5 without force-push and record the published `main` commit.
- [ ] Repin and render the OpenClaw Console bootstrap to the reviewed public source commit if the merge commit changes the source pin.

### Task 4: Persist and validate the exact Launchable

**Resources:**
- Launchable: `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`
- Organization: resolve and pin from current authenticated Brev metadata before mutation.

- [ ] Save the reviewed rendered bootstrap in the Brev Console using a supported authenticated control surface. If no callable authoring surface exists, stop and request only this exact user action.
- [ ] Confirm the Launchable has no attendee Setup values and uses the intended L4/default configuration.
- [ ] Deploy one fresh task-owned instance, record its exact name/ID and price, and wait for setup.
- [ ] Through the browser, confirm automatic authentication and run the four prompts once in order.
- [ ] Verify CPU/GPU explanations, images, `results.zip`, repeat safety, and credential/interface exposure.
- [ ] Stop the exact paid test instance and verify `STOPPED`.

### Task 5: Publish the attendee guide

**Files:**
- Publish: `/Users/ktretina/Desktop/Codex Working Folder/digital-biology-examples/acsfall26/README.md`

- [ ] Only after Task 4 passes, make the public page byte-identical to the accepted canonical guide.
- [ ] Push `gh-pages` and verify the published page and four prompt hashes.
- [ ] Report PASS or NOT READY with the fresh deployment, four prompt outcomes, browser/artifact/credential/usability results, commits, Launchable state, and stopped-instance evidence.
