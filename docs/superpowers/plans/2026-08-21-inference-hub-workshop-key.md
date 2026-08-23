# Inference Hub Workshop Key Implementation Plan

> **Scope:** Update only the `nvMolKit + Nemotron Notebook` source and its active three-notebook workshop experience. Do not restore the removed demo notebook, modify the conversational project, or change any other Brev Launchable.

**Goal:** Let workshop attendees use the organizer-provided Inference Hub credential through one Launchable setup value while keeping secrets out of notebooks, logs, child processes, and repository artifacts.

**Runtime contract:** Use `https://inference-api.nvidia.com/v1` with model `nvidia/nvidia/nemotron-3-nano-30b-a3b`. Prefer `NVIDIA_INFERENCE_API_KEY`; accept `NVIDIA_API_KEY` only as a legacy variable name for an `sk-` Inference Hub key. Reject `nvapi-` Build keys with actionable guidance. Set `parallel_tool_calls=False` on every structured tool request.

## Task 1: Lock the active helper contract with regression tests

**Files:**
- Modify: `tests/test_workshop_llm_agent.py`

1. Add tests for the Inference Hub base URL and exact model ID.
2. Add tests that the new environment variable wins over the legacy variable name, and that an invalid primary value fails closed.
3. Add tests for secure canonical-file fallback and legacy-name fallback with `sk-` values.
4. Add tests that `nvapi-` values are rejected with Inference Hub guidance.
5. Add redaction cases for `sk-` values and `NVIDIA_INFERENCE_API_KEY`, including quoted mappings and safe left-boundary behavior.
6. Assert that neither credential variable reaches child processes.
7. Assert `parallel_tool_calls=False` for each structured request path.
8. Run the focused tests and confirm the new tests fail for the intended reasons.

## Task 2: Implement the active helper changes

**Files:**
- Modify: `notebooks/workshop_llm_agent.py`

1. Change the endpoint and model constants to the verified Inference Hub values.
2. Resolve the new environment variable and canonical key file first.
3. Retain the old environment-variable name as a migration alias for `sk-` values only.
4. Preserve existing file ownership, permission, symlink, and size checks.
5. Extend secret redaction to both variable names and both key-shaped prefixes, without matching ordinary hyphenated text.
6. Remove both credential variables from child-process environments.
7. Add `parallel_tool_calls=False` to the shared structured-request call.
8. Bump the helper version and run the focused test file to green.

## Task 3: Align the Launchable setup and active workshop content

**Files:**
- Modify: `launchable/setup.sh`
- Modify: `launchable/fields.md`
- Modify: `README.md`
- Modify: `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
- Modify: `notebooks/03_full_agent_reframe_panel_design.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `tests/test_workshop_notebook_execution.py`
- Modify: `tests/test_workshop_notebook_inventory.py`

1. Add failing tests for one required `NVIDIA_INFERENCE_API_KEY` Launchable field, secure persistence to `~/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY`, primary precedence, legacy-name fallback for `sk-`, invalid-primary fail-closed behavior, environment cleanup, and the 16 KiB script limit.
2. Update setup to write the preferred `sk-` value atomically with mode `0600`, never print it, and unset both variable names.
3. Update attendee copy to explain that the organizer supplies the workshop Inference Hub key; attendees do not create an NVIDIA API key.
4. Update Module 2 key guidance and both embedded helper-version locks. Module 1 needs no credential change.
5. Update the current inventory to describe the existing three notebooks and keep the deleted demo notebook absent.
6. Run the focused setup, notebook-execution, and inventory tests to green.

## Task 4: Verify the release candidate

1. Run focused helper, setup, notebook-copy, notebook-execution, and inventory tests with the qualified Python 3.12 interpreter.
2. Measure `launchable/setup.sh` exactly and require no more than 16,384 bytes.
3. Scan tracked release files for credential values and stale attendee-facing Build-key instructions.
4. Use the locally configured Inference Hub key for a minimal live smoke test without printing or persisting the secret.
5. Run independent specification and code-quality reviews; fix material findings and re-run affected tests.
6. Do not claim full-suite success locally if the known deleted-demo inventory tests remain outside the corrected current release contract.

## Task 5: Prepare the Notebook Launchable update

1. Report the exact source commit and tested setup payload.
2. Treat the saved Brev Console field definition and setup-script body as a separate deployment surface; repository edits alone do not update them.
3. Update and requalify only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` after the external-write boundary is confirmed.
4. Return the Notebook test URL and clearly separate source validation, Launchable configuration, deployment, and live browser qualification.
