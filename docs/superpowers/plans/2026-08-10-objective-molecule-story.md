# Objective Molecule Story Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense Objective-Driven Agent Challenge receipt presentation with the approved five-step, molecule-first attempt story and an exact progress bar after every attempt.

**Architecture:** Keep the objective controller, action menus, validation, execution, evidence ledger, and conclusion contracts unchanged. Change only the `interactive_workflow.py` presentation layer so it draws retained RDKit molecules as inline SVG and renders each attempt as Observe, Candidate menu, Agent chooses, Execute panel, and Measure panel. Keep detailed receipts in controller-owned data, but do not place them in the visible objective interface.

**Tech Stack:** Python 3.12, RDKit SVG drawing, ipywidgets HTML/VBox, pytest, nbformat.

---

### Task 1: Lock the approved visual and scientific contract with failing tests

**Files:**
- Modify: `tests/test_interactive_workflow.py`

- [ ] **Step 1: Write the failing molecule-story tests**

Add tests that call the real objective renderer with `optimized_state().molecules`. Require five ordered step sections, inline SVG molecule tiles, the larger Molecular change panel, the Why this choice section, and one progress bar after the attempt.

```python
assert rendered.count("aria-label='Objective step'") == 5
assert "Observe panel" in rendered
assert "Candidate menu" in rendered
assert "Agent chooses" in rendered
assert "Execute panel" in rendered
assert "Measure panel" in rendered
assert rendered.count("data-molecule-id=") >= 4
assert "<svg" in rendered
assert "Molecular change" in rendered
assert "Why this choice?" in rendered
assert "data-objective-progress" in rendered
assert "required improvement achieved" in rendered
```

Add negative assertions that visible attempt HTML does not contain `ObjectiveSelection(`, `PanelMeasurement(`, `data-receipt`, `Full candidate menu`, or the obsolete molecule-state footer sentence. Assert that Step 2 shows at most four scored candidates and a remaining-action count when the menu is longer.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/pytest -q tests/test_interactive_workflow.py -k 'molecule_story or progress_bar or concise_candidate'
```

Expected: FAIL because the current renderer has no molecule SVG or per-attempt progress bar and displays raw receipts.

### Task 2: Implement the molecule-first attempt renderer

**Files:**
- Modify: `interactive_workflow.py`
- Test: `tests/test_interactive_workflow.py`

- [ ] **Step 1: Add guarded RDKit molecule rendering**

Add a helper that resolves a molecule ID through the objective candidate provenance, draws the corresponding retained RDKit molecule as SVG, strips the XML preamble, and returns a labeled molecule tile. Reject missing, duplicated, out-of-range, or non-RDKit artifacts.

- [ ] **Step 2: Render the five exact steps**

Replace the receipt-heavy row markup with five ordered sections: Observe panel, Candidate menu, Agent chooses, Execute panel, and Measure panel. Show the source panel and limiting pair; up to four ordered replacement molecules and calculated scores; the outgoing/incoming pair; the resulting four-molecule panel; and the measured limiting pair, D_min, delta, and target gap.

- [ ] **Step 3: Add change, reason, and progress presentation**

For the latest attempt, show a larger outgoing/incoming Molecular change and three deterministic reasons: accepted argmax, removal from every prior limiting pair, and preservation of four unique cluster representatives. End every measured attempt with a progress bar derived from `(score - baseline) / (target - baseline)`. For an unmeasured attempt, show the last measured score and state clearly that evaluation was not completed.

- [ ] **Step 4: Keep validation fail-closed**

Retain the existing exact-type, state ID, argmax, score-key, panel, limiting-pair, target, and independent measurement validation before rendering. Do not change the controller, objective domain, receipts, or evidence contracts.

- [ ] **Step 5: Run the focused tests and verify GREEN**

```bash
PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/pytest -q tests/test_interactive_workflow.py
```

Expected: all interactive workflow tests pass after obsolete visible-receipt expectations are replaced.

### Task 3: Replace the live card layout and persistent notebook copy

**Files:**
- Modify: `interactive_workflow.py`
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `README.md`
- Modify: `tests/test_interactive_workflow.py`
- Modify: `tests/test_notebook.py`

- [ ] **Step 1: Add failing live and persistence acceptance tests**

Require two visible attempt stories, ten ordered step sections, inline molecule SVG, two end-of-attempt progress bars, the final score, and the evidence-controlled conclusion after widget embedding. Assert that no visible objective HTML contains raw JSON, Python evaluation source, measurement receipts, or obsolete collapsed detail labels.

- [ ] **Step 2: Run the acceptance tests and verify RED**

```bash
PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/pytest -q tests/test_interactive_workflow.py -k 'completed_real_workflow_root or objective_attempts_render'
```

Expected: FAIL until the live workflow passes retained state molecules to the renderer and uses visible HTML attempt cards instead of receipt accordions.

- [ ] **Step 3: Wire the live state into the renderer**

Pass `controller.session.state.molecules` into summary and attempt rendering. Render attempts in a `VBox` in chronological order. Keep the Run Objective Challenge button, failure handling, terminal outcome, figures, and conclusion flow unchanged.

- [ ] **Step 4: Update the notebook and README narrative**

Explain the five molecule-aware steps, the LLM input menu and output swap, and the progress bar after every attempt. Remove claims that the visible objective card shows the full menu, JSON, executed Python, or measurement receipt. Preserve the scientific boundary language and the separate six-stage command receipts.

- [ ] **Step 5: Run notebook and persistence tests and verify GREEN**

```bash
PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/pytest -q tests/test_interactive_workflow.py tests/test_notebook.py
```

Expected: all tests pass.

### Task 4: Verify the complete local change

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run syntax and notebook validation**

```bash
/private/tmp/nvmolkit-ui-venv312/bin/python -m py_compile interactive_workflow.py
/private/tmp/nvmolkit-ui-venv312/bin/python -c "import nbformat; nbformat.read('notebooks/nvmolkit_nemotron_demo.ipynb', as_version=4); print('notebook valid')"
```

Expected: exit 0 and `notebook valid`.

- [ ] **Step 2: Run the supported full local suite**

```bash
PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/pytest -q -k 'not test_notebook_preflight_fails_closed_when_launch_key_is_missing and not test_notebook_preflight_rejects_launch_key_with_unsafe_permissions'
```

Expected: all supported local tests pass; the live GPU test remains skipped. The two deselected preflight tests require the Brev CUDA runtime because `notebook_preflight()` imports CUDA-enabled PyTorch before checking the protected key file.

- [ ] **Step 3: Inspect the final diff**

```bash
git diff --check
git status --short
git diff --stat
git diff -- interactive_workflow.py tests/test_interactive_workflow.py tests/test_notebook.py README.md notebooks/nvmolkit_nemotron_demo.ipynb
```

Expected: no whitespace errors and only the approved presentation, copy, tests, and plan are changed.
