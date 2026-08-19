# Live-Browser Notebook Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Modules 2 and 3 clear, truthful, and readable in live JupyterLab without changing their model, scientific computation, data, or request bounds.

**Architecture:** Keep the current bounded agent and scientific controllers. Repair Module 2 in the notebook presentation only. For Module 3, add one live callback output region and truthful audit status in the widget helper, then make a normal notebook cell the authoritative same-kernel receipt that revalidates the retained run before displaying metrics or chemistry.

**Tech Stack:** Python 3.12, Jupyter notebooks (`nbformat`), `ipywidgets`, pandas, RDKit, nvMolKit, pytest, Ruff.

---

## Fixed scope and file responsibilities

- `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`: Module 2 roles, normal-path evidence wording, compact table, and discussion.
- `notebooks/module3_interactive_workflow.py`: Module 3 live output capture and widget/transcript audit-status language.
- `notebooks/03_full_agent_reframe_panel_design.ipynb`: canonical same-kernel receipt, replay instructions, evidence consistency, scientific interpretation, and gallery width.
- `tests/test_workshop_llm_agent.py`: source-level notebook contracts and direct widget behavior.
- `tests/test_workshop_notebook_execution.py`: clean-kernel and replay behavior.
- `tests/test_workshop_notebook_inventory.py`: existing notebook-cleanliness boundary; change only if a new focused assertion is necessary.

Do not modify `notebooks/workshop_llm_agent.py`, Module 1, the companion demo,
the fixed model or tool schemas, the data snapshot, setup, Launchable fields, or
another Brev Launchable.

Use this interpreter for local gates:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

Run one heavy command at a time.

### Task 1: Make Module 2 truthful and readable

**Files:**
- Modify: `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb` cells `cell-65b035d14880`, `cell-a341837a7935`, `cell-89be8eb5140f`, `student-copy-generated-code`, `cell-a511e060a724`, `cell-b766b437bdb5`, `cell-5e6cb5fb12e0`, `embedded-agent-review`, `cell-17376c167229`, `2db02500`
- Modify: `tests/test_workshop_llm_agent.py`
- Modify: `tests/test_workshop_notebook_execution.py`
- Modify for review repair: `docs/superpowers/specs/2026-08-18-live-browser-notebook-experience-design.md`
- Modify for review repair: `docs/superpowers/plans/2026-08-18-live-browser-notebook-experience.md`

- [ ] **Step 1: Add failing Module 2 role, evidence, and table tests**

Add this helper to `tests/test_workshop_llm_agent.py`:

```python
def _notebook_cell_source(path: Path, cell_id: str) -> str:
    stored = json.loads(path.read_text(encoding="utf-8"))
    cell = next(cell for cell in stored["cells"] if cell["id"] == cell_id)
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source
```

Add these tests:

```python
def test_module2_roles_distinguish_hosted_selection_from_local_reference_values():
    goal = _notebook_cell_source(NOTEBOOK_PATH, "cell-65b035d14880")
    receipt = _notebook_cell_source(NOTEBOOK_PATH, "embedded-agent-review")

    assert "Hosted mode asks Nemotron to choose" in goal
    assert "Reference mode uses fixed local reference policy values" in goal
    assert "no hosted selection" in goal
    assert 'if implementation.label == "hosted_nemotron":' in receipt
    assert 'elif implementation.label == "reference":' in receipt
    assert "fixed local reference policy values were used" in receipt
    assert "no hosted selection occurred" in receipt
    assert "hosted receipt" not in receipt.lower()
    assert "policy receipt" in receipt.lower()


def test_module2_validation_copy_is_limited_to_normal_path_invariants():
    notebook_source = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    check_cell = _notebook_cell_source(NOTEBOOK_PATH, "cell-b766b437bdb5")

    assert "all acceptance tests" not in notebook_source
    assert "normal-path invariant checks" in notebook_source
    assert "selected failure branches were not triggered" in notebook_source
    assert (
        'attendee_columns = ["radius", "query", "rank", "neighbor", "tanimoto"]'
        in check_cell
    )
    assert "display(attendee_atlas.head(12))" in check_cell
```

Update the existing Module 2 flow and discussion tests to require the new role
wording, normal-path heading, branch limitation, exact plural discussion
question, and `raise`/`raise` instructor explanation.

Extend the clean-kernel test to collect the HTML output for
`cell-b766b437bdb5`, normalize its `<th>` labels, and require:

```python
assert table_headers == ["", "radius", "query", "rank", "neighbor", "tanimoto"]
assert "Normal-path invariant checks passed" in stream_text
assert "Selected failure branches were not triggered" in stream_text
```

- [ ] **Step 2: Run the focused tests and witness RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nvmolkit-live-m2-mpl IPYTHONDIR=/private/tmp/nvmolkit-live-m2-ipython JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py::test_module2_roles_distinguish_hosted_selection_from_local_reference_values \
  tests/test_workshop_llm_agent.py::test_module2_validation_copy_is_limited_to_normal_path_invariants \
  tests/test_workshop_llm_agent.py::test_module2_discussion_and_answer_key_pair_the_three_semantic_items \
  tests/test_workshop_notebook_execution.py::test_module2_reference_executes_cleanly_without_key_or_hosted_client
```

Expected: failures show the old “you select,” “all acceptance tests,” missing
branch limitation, and eight-column table.

- [ ] **Step 3: Apply the minimal notebook repair**

Preserve all cell IDs and clean notebook metadata. Use this exact role model in
the Goal:

```text
Hosted mode asks Nemotron to choose the two bounded policy values for this run.
Reference mode uses fixed local reference policy values with no hosted
selection. In both modes, Python applies the matching allow-listed
implementation. You evaluate the choices afterward, run the checks, and
interpret the result.
```

In the visible receipt, branch on exact implementation labels. Print the
Nemotron selection only for `hosted_nemotron`. For `reference`, print that
fixed local reference policy values were used and that no hosted selection
occurred. Use neutral `policy receipt` wording for the shared source; do not
call it a hosted receipt.

Rename Step 4 to:

```markdown
## Step 4 — Run the bound function and its normal-path invariant checks

This cell checks the valid fixed run. It does not trigger the selected
missing-anchor or invalid-matrix failure branches.
```

Keep the full `atlas` unchanged. Replace only its display and result text:

```python
print("✓ Normal-path invariant checks passed for the fixed valid run.")
print("Selected failure branches were not triggered by this fixed valid run.")
attendee_columns = ["radius", "query", "rank", "neighbor", "tanimoto"]
attendee_atlas = atlas.loc[:, attendee_columns].round({"tanimoto": 3})
display(attendee_atlas.head(12))
```

Use this exact discussion question:

```text
Are both selected policies appropriate? If not, which values would you choose
and why?
```

State in the answer key that `raise`/`raise` is appropriate for the fixed
teaching run, while the recorded run values remain the values actually used by
that run.

- [ ] **Step 4: Run Module 2 GREEN and adjacent gates**

Run the RED command again. Expected: all selected tests pass.

Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nvmolkit-live-m2-mpl IPYTHONDIR=/private/tmp/nvmolkit-live-m2-ipython JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py \
  tests/test_workshop_notebook_inventory.py
```

Expected: all tests pass; the clean-kernel Module 2 execution uses reference
mode and makes zero hosted calls.

- [ ] **Step 5: Format, inspect, and commit Task 1**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format --check \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
git diff --check
git status --short
```

Expected: checks pass and only the five Task 1 files are modified. Notebook
changes are limited to the ten named cells.

Commit:

```bash
git add notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py \
  docs/superpowers/specs/2026-08-18-live-browser-notebook-experience-design.md \
  docs/superpowers/plans/2026-08-18-live-browser-notebook-experience.md
git commit -m "fix: clarify the neighborhood lesson"
```

### Task 2: Capture Module 3 results and separate audit status

**Files:**
- Modify: `notebooks/module3_interactive_workflow.py`
- Modify: `notebooks/03_full_agent_reframe_panel_design.ipynb` cell `m3-setup` only for the helper version lock
- Modify: `tests/test_workshop_llm_agent.py`
- Modify: `docs/superpowers/plans/2026-08-18-live-browser-notebook-experience.md` only for approved review repairs that keep callback exception redaction inside the output context and use one exact failed-analysis status

- [ ] **Step 1: Add failing live-output and status tests**

Replace the narrow callback test with a parameterized test named
`test_module3_widget_captures_callback_and_truthfully_labels_audit_state`.
Use an `ipywidgets.Output` subclass that records `__enter__`, records any
exception received by `__exit__`, and returns `True` to emulate Jupyter
exception suppression. Use a fake plan that recommends strategy 2 and a fake
agent that records `request_plan` and `run` calls. Set the radio control to
strategy 1 before approval.

The core assertions are:

```python
assert workflow.result_output in workflow.root.children
assert workflow.result_output.enter_count == 1
assert fake_agent.plan_calls == 1
assert fake_agent.run_calls == [1]
assert callback_calls == [run]
assert workflow.plan.recommended_strategy == 2
assert workflow.agent_run.approved_strategy == 1

visible_text = _widget_text(workflow.root)
assert expected_status in visible_text
assert expected_status in workflow.transcript_text
if run.audit is None:
    assert "Agent workflow complete" not in visible_text

workflow._approve_and_run(workflow.approve_button)
assert fake_agent.plan_calls == 1
assert fake_agent.run_calls == [1]
assert callback_calls == [run]
```

Parameter cases:

```python
(
    PanelAudit.model_validate(_panel_audit_payload()),
    "Analysis validated; audit complete",
),
(None, "Analysis validated; audit unavailable"),
```

Keep the existing callback-error redaction assertion as a separate case. Assert
that the raw exception never reaches the output context, transcript, or card,
while the redacted failure card appears and the successful run stays unchanged.

Add focused cases for an agent that returns `PanelAgentRun(success=False)` and
an agent whose `run` method raises. Both cases must show the exact status
`Analysis did not validate` in the final card and transcript. The old
`Agent workflow did not pass` and `Agent run stopped safely` titles must be
absent. Neither case may call the completion callback or enter a success state.

- [ ] **Step 2: Run the widget test and witness RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py -k "module3_widget or interactive_workflow_waits"
```

Expected: failure because there is no dedicated output widget and a missing
audit still renders “Agent workflow complete.”

- [ ] **Step 3: Add the minimal widget output and status behavior**

Bump:

```python
MODULE3_WORKFLOW_VERSION = "2026.08.18.4"
```

Update the exact version lock in notebook cell `m3-setup` to the same value.

In `InteractivePanelDesignWorkflow.__init__`, add:

```python
self.result_output = widgets.Output(
    layout=widgets.Layout(border="1px solid #b8b8b8", padding="8px")
)
self.root = widgets.VBox((self.start_button, self._cards, self.result_output))
```

Run the completion renderer only inside that context:

```python
if self._on_complete is not None:
    with self.result_output:
        try:
            self._on_complete(self.agent_run)
        except Exception as error:
            message = _safe_text(f"{type(error).__name__}: {error}")
            self._line(f"Completion display failed: {message}")
            self._append(
                self._html_card(
                    "Completion display failed",
                    "<p>The validated scientific result is unchanged.</p>"
                    f"<pre>{escape(message)}</pre>",
                )
            )
```

In `_progress`, append these transcript lines before the existing cards:

```python
self._line("Analysis validated; audit complete")
self._line("Analysis validated; audit unavailable")
```

Use the matching line in each audit branch. In `_show_final_result`, use:

```python
status = (
    "Analysis validated; audit complete"
    if run.audit is not None
    else "Analysis validated; audit unavailable"
)
self._line(status)
self._append(self._html_card(status, body))
```

Use one exact failed-analysis status in both failure paths:

```python
FAILED_ANALYSIS_STATUS = "Analysis did not validate"

# In the run exception path:
self._error_card(FAILED_ANALYSIS_STATUS, error)

# In the returned unsuccessful-run branch:
self._line(FAILED_ANALYSIS_STATUS)
self._append(self._html_card(FAILED_ANALYSIS_STATUS, body))
```

Keep callback error isolation. Reuse the agent module's exact
`_redact_sensitive_text` helper for every widget error path so punctuation-bearing
`nvapi-` keys and named `NVIDIA_API_KEY` values cannot reach a card or transcript.

- [ ] **Step 4: Run GREEN and the workflow-adjacent suite**

Run the RED command again. Expected: pass.

Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py
```

Expected: all tests pass, with one plan call, one analysis call, and one
completion callback in the new widget test.

- [ ] **Step 5: Format, inspect, and commit Task 2**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py
git diff --check
git status --short
```

Expected: only the approved plan review repair, helper, Module 3 notebook
version lock, and focused test are modified.

Commit:

```bash
git add docs/superpowers/plans/2026-08-18-live-browser-notebook-experience.md \
  notebooks/module3_interactive_workflow.py \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  tests/test_workshop_llm_agent.py
git commit -m "fix: capture the panel workflow result"
```

### Task 3: Add the canonical Module 3 receipt and readable gallery

**Files:**
- Modify: `notebooks/03_full_agent_reframe_panel_design.ipynb` cells `m3-render`, `m3-launch-title`, `m3-launch`, `m3-state-title`, `m3-state`, `m3-gallery`, `m3-final`
- Modify: `tests/test_workshop_notebook_execution.py`
- Modify: `tests/test_workshop_llm_agent.py` only if a shared notebook-source helper is needed

- [ ] **Step 1: Add failing receipt, replay, inconsistency, and UX tests**

Add this UX test:

```python
def test_module3_notebook_explains_replay_uses_guardrail_language_and_three_columns():
    notebook_text = MODULE3_PATH.read_text(encoding="utf-8").lower()
    source = _module3_code_source()

    assert "approve plan & run agent" in notebook_text
    assert "rerun steps 5 and 6" in notebook_text
    assert "descriptor-range coverage is a guardrail" in notebook_text
    assert "minimum tanimoto distance is the strategy-sensitive" in notebook_text
    assert "molsPerRow=3" in source
```

Add `test_module3_receipt_replays_canonical_evidence_without_new_calls`. Use a
real reference-mode run in a temporary candidate workspace, execute the
`m3-render` definitions with controlled globals, then replace
`panel_agent.request_plan` and `panel_agent.run` with functions that raise.
Call the receipt and gallery loaders twice. Assert:

```python
first = build_validated_panel_receipt(run, recommended_strategy=2)
second = build_validated_panel_receipt(run, recommended_strategy=2)
assert first == second
assert first["analysis_status"] == "validated"
assert first["audit_status"] == "reference audit complete"
assert request_count == 0
assert analysis_execution_count == 1
```

Add `test_module3_pending_cells_are_safe_without_calls`. Execute the Step 5 and
Step 6 cells with `agent_run = None` and counters that raise if plan, audit, or
analysis code is reached. Both cells must print a waiting message, make zero
calls, and return without an exception.

Add parameterized `test_module3_receipt_rejects_inconsistent_retained_evidence`
for these mutations:

```python
(
    "different workflow run object",
    "wrong fixed path",
    "missing trace",
    "wrong trace mode",
    "wrong approved strategy",
    "trace success false",
    "run and trace audit disagreement",
)
```

Each case must raise `ValueError` before printing a success receipt or drawing
the gallery. Extend the clean reference-kernel test by wrapping
`PanelDesignAgent.run`, appending the Step 5 and Step 6 code a second time, and
requiring one analysis execution and zero hosted client calls.

- [ ] **Step 2: Run the receipt tests and witness RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nvmolkit-live-m3-mpl IPYTHONDIR=/private/tmp/nvmolkit-live-m3-ipython JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_notebook_execution.py -k "module3 and (receipt or guardrail or reference)"
```

Expected: failures show the missing canonical receipt, stale global gallery,
missing replay instruction, old metric wording, and four-column grid.

- [ ] **Step 3: Implement one canonical receipt in `m3-render`**

Import `DEFAULT_MODEL` from `workshop_llm_agent` in `m3-setup`.

Make `agent_run` the one canonical run object. Require its paths to be the fixed
regular, non-symlink workspace files. Use
`_validate_panel_artifacts_snapshot()` so receipt fields come from the exact
report snapshot that passed independent validation. Add these functions:

As a final review repair, securely read and bind `reframe_candidates.csv` to
`candidate_pool[input_columns].to_csv(index=False)`, bind `analysis.py` to the
exact controller-rendered source for the retained approved strategy, and bind
`report.json`'s strategy name to that same approved strategy before returning
any receipt or gallery data.

```python
def _expected_run_paths():
    return {
        "analysis_path": AGENT_WORKDIR / "analysis.py",
        "panel_path": AGENT_WORKDIR / "panel.csv",
        "report_path": AGENT_WORKDIR / "report.json",
        "trace_path": AGENT_WORKDIR / "agent_trace.json",
    }


def _require_canonical_run(run, recommended_strategy):
    if run is None or run.success is not True:
        raise ValueError("A successful validated panel run is required.")
    if PANEL_AGENT_MODE == "hosted":
        if module3_workflow is None or module3_workflow.agent_run is not run:
            raise ValueError("The hosted receipt does not match the retained workflow run.")
    for attribute, expected in _expected_run_paths().items():
        observed = Path(getattr(run, attribute))
        if observed != expected or observed.is_symlink() or not observed.is_file():
            raise ValueError(f"{expected.name} is not the fixed regular run artifact.")

    loaded_panel, loaded_report, loaded_trace, receipt = (
        load_validated_panel_artifacts(run)
    )
    expected_model = DEFAULT_MODEL if PANEL_AGENT_MODE == "hosted" else None
    if loaded_trace.get("mode") != PANEL_AGENT_MODE:
        raise ValueError("The retained trace mode does not match this notebook run.")
    if loaded_trace.get("model") != expected_model:
        raise ValueError("The retained trace model does not match this notebook run.")
    if loaded_trace.get("approved_strategy") != run.approved_strategy:
        raise ValueError("The retained trace strategy does not match the run.")
    if loaded_trace.get("success") is not True:
        raise ValueError("The retained trace does not record validated analysis.")
    if loaded_trace["plan"]["recommended_strategy"] != recommended_strategy:
        raise ValueError("The retained recommendation does not match the plan.")

    trace_audit = loaded_trace.get("audit")
    trace_error = loaded_trace.get("audit_error")
    if run.audit is None:
        if PANEL_AGENT_MODE == "reference":
            raise ValueError("The deterministic reference audit is missing.")
        if trace_audit is not None:
            raise ValueError("The retained audit evidence is inconsistent.")
        if PANEL_AGENT_MODE == "hosted" and not trace_error:
            raise ValueError("The hosted audit status is missing.")
    elif trace_audit != run.audit.model_dump(mode="json") or trace_error:
        raise ValueError("The retained audit evidence is inconsistent.")
    return loaded_panel, loaded_report, loaded_trace, receipt
```

Build the receipt only from validated values:

```python
def build_validated_panel_receipt(run, *, recommended_strategy):
    loaded_panel, loaded_report, loaded_trace, receipt = _require_canonical_run(
        run, recommended_strategy
    )
    acceptance = loaded_report["acceptance"]
    audit_status = (
        "reference audit complete"
        if PANEL_AGENT_MODE == "reference"
        else ("complete" if run.audit is not None else "unavailable")
    )
    return {
        "mode": loaded_trace["mode"],
        "model": loaded_trace["model"],
        "recommended_strategy": recommended_strategy,
        "approved_strategy": run.approved_strategy,
        "backend": loaded_report["backend"],
        "baseline_minimum_distance": acceptance["baseline_minimum_distance"],
        "selected_minimum_distance": acceptance["selected_minimum_distance"],
        "baseline_descriptor_coverage": acceptance[
            "baseline_descriptor_coverage"
        ],
        "selected_descriptor_coverage": acceptance[
            "selected_descriptor_coverage"
        ],
        "analysis_status": "validated",
        "audit_status": audit_status,
        "acceptance_passed": receipt["acceptance_passed"],
    }
```

The existing immediate renderer calls this helper, prints the compact receipt,
and draws the current figures. It may update `panel` for immediate display, but
Steps 5 and 6 must revalidate instead of trusting that global.

- [ ] **Step 4: Make Steps 5 and 6 authoritative, replay-safe displays**

In hosted mode, obtain the recommendation from
`module3_workflow.plan.recommended_strategy`; in reference mode use
`reference_plan.recommended_strategy`.

Step 5 calls
`build_validated_panel_receipt(agent_run, recommended_strategy=recommended_strategy)`,
prints the sorted receipt, and displays the existing attempt table. It does not
call `request_plan()`, `run()`, `_request_audit()`, or subprocess.

Step 6 calls `_require_canonical_run(agent_run, recommended_strategy)` again
and draws from the returned panel. Change:

```python
molsPerRow=3
```

Both cells must retain an explicit workflow-state guard before those calls.
Only the validated-run branch may enter the complete canonical receipt or
gallery code specified above. Reserve the sponsor-pending message for hosted
`planning`, `awaiting_approval`, or `executing` states. A `plan_failed` state
directs the attendee to **Retry Plan**; reference mode without a run directs the
attendee to rerun Step 4. These states must remain validation- and
execution-free.

Use these instructions before Step 4:

```markdown
1. Review the two proposed strategies.
2. Accept the recommendation or choose the other bounded strategy.
3. Click **Approve Plan & Run Agent**.
4. After the workflow reports its analysis and audit status, rerun Steps 5 and
   6 to inspect the authoritative receipt and chemistry gallery.
```

Use this interpretation in the final synthesis:

```text
Descriptor-range coverage is a guardrail because both allow-listed strategies
seed descriptor extrema. Minimum Tanimoto distance is the strategy-sensitive
comparison in this bounded lesson.
```

- [ ] **Step 5: Run GREEN, clean-kernel replay, and adjacent tests**

Run the RED command again. Expected: pass.

Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nvmolkit-live-m3-mpl IPYTHONDIR=/private/tmp/nvmolkit-live-m3-ipython JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py \
  tests/test_workshop_notebook_inventory.py
```

Expected: all tests pass. Reference mode performs one local analysis and zero
hosted calls even when Steps 5 and 6 execute twice.

- [ ] **Step 6: Format, inspect, and commit Task 3**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format --check \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py
git diff --check
git status --short
```

Expected: only the Module 3 notebook and focused tests are modified in this
task.

Commit:

```bash
git add notebooks/03_full_agent_reframe_panel_design.ipynb \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
git commit -m "fix: add a durable panel-design receipt"
```

### Task 4: Run the release gate and exact-scope review

**Files:**
- Verify only; no production change is expected

- [ ] **Step 1: Run the full deterministic suite**

Run one heavy command:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nvmolkit-live-final-mpl IPYTHONDIR=/private/tmp/nvmolkit-live-final-ipython JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q -p no:cacheprovider
```

Expected: all local tests pass, with only the declared GPU-only skip when no
compatible local GPU is available.

- [ ] **Step 2: Run static, notebook, and scope checks**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py tests/test_workshop_notebook_inventory.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format --check \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py tests/test_workshop_notebook_inventory.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m py_compile \
  notebooks/module3_interactive_workflow.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py tests/test_workshop_notebook_inventory.py
git diff --check
git status --short
git diff 4648c259df947d0a93776a04f45282473928c8fa --name-only
```

Expected changed implementation range:

```text
docs/superpowers/specs/2026-08-18-live-browser-notebook-experience-design.md
docs/superpowers/plans/2026-08-18-live-browser-notebook-experience.md
notebooks/02_agent_assisted_reframe_neighborhoods.ipynb
notebooks/03_full_agent_reframe_panel_design.ipynb
notebooks/module3_interactive_workflow.py
tests/test_workshop_llm_agent.py
tests/test_workshop_notebook_execution.py
```

`tests/test_workshop_notebook_inventory.py` may change only if its existing
clean-notebook contract needs a focused assertion; otherwise it remains
unchanged. No model, data, setup, Module 1, demo, or Launchable file may change.

- [ ] **Step 3: Run independent reviews**

Dispatch a fresh specification reviewer against the approved design and this
plan. Resolve every Critical or Important issue, then re-run the affected
tests. Only after specification approval, dispatch a fresh code-quality
reviewer. Resolve every Critical or Important issue and repeat the relevant
review.

- [ ] **Step 4: Prepare the fresh browser acceptance gate**

Do not count a stale instance. Use the exact post-fix commit on only
`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`. Record:

```text
Module 2 submit_neighborhood_policy requests: 1
Module 3 submit_panel_plan requests: 1
Module 3 submit_panel_audit attempts: 1
Module 3 local analysis executions: 1
```

After rerunning Module 3 Steps 5 and 6, all counts must remain unchanged. The
browser must show the dedicated callback output, truthful audit status,
canonical receipt, `nvmolkit-gpu` backend, compact Module 2 table, readable
three-column gallery, and no key or raw provider response.

Do not deploy, stop, delete, or mutate a Brev environment without the user's
separate approval. Local completion is not live browser acceptance.
