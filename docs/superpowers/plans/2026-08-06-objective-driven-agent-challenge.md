# Objective-Driven Agent Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append a bounded, visible maximin-diversity challenge after the existing six-stage workflow so Nemotron can propose, measure, revise, and reach or explicitly miss a quantitative panel-selection target.

**Architecture:** Preserve the six scientific stages and E01-E06. Add a pure `objective_challenge.py` domain layer for pool construction, exact scoring, target calculation, figures, and O01; add deterministic proposal receipts; extend the existing controller with a maximum-three-attempt post-workflow loop; and extend the existing widget with one live-updating challenge card before the final conclusion.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, Matplotlib, RDKit, ipywidgets 8, OpenAI-compatible hosted Nemotron client, nvMolKit, pytest, nbformat

---

## File Structure

- Create `objective_challenge.py`: immutable objective types, eight-candidate pool, maximin score, exact benchmark, target, attempt evaluation, O01, and figures.
- Modify `chemistry_workflow.py`: expose read-only validated similarity and eligible-candidate views without changing the six stage transitions.
- Create `objective_receipts.py`: deterministic validated-proposal and Python-evaluator receipts.
- Modify `demo_agent.py`: strict objective proposal model, bounded attempt methods, turn accounting, O01-aware conclusion schema and prompt.
- Modify `interactive_workflow.py`: one final challenge card, automatic bounded attempt progression, attempt details, trajectory, ledger, structures, heatmap, and conclusion rename.
- Modify `notebooks/nvmolkit_nemotron_demo.ipynb`: audience narrative and operating instructions only.
- Modify `README.md`: document the appended objective challenge and evidence boundary.
- Create `tests/test_objective_challenge.py`: pure scientific objective tests.
- Create `tests/test_objective_receipts.py`: deterministic receipt tests.
- Modify `tests/test_demo_agent.py`: hosted objective-loop and conclusion-gate tests.
- Modify `tests/test_interactive_workflow.py`: live card, attempt history, success/failure, and retry tests.
- Modify `tests/test_notebook.py`: compact narrative and unchanged public launch surface.
- Modify `tests/test_gpu_acceptance.py`: default-data eligibility, positive benchmark gap, real attempt evaluation, and objective termination gates.

### Task 1: Implement the pure objective domain

**Files:**
- Create: `tests/test_objective_challenge.py`
- Create: `objective_challenge.py`
- Modify: `chemistry_workflow.py:208-212, 324-355`
- Modify: `tests/test_chemistry_workflow.py`

- [ ] **Step 1: Write failing tests for pool construction, scoring, benchmark, target, validation, O01, and figures**

Create synthetic optimized `WorkflowState` fixtures with eight MMFF-eligible single-member clusters and a controlled 8-by-8 similarity artifact. Assert:

```python
def test_build_objective_context_uses_eight_distinct_eligible_clusters():
    context = build_objective_context(optimized_state())
    assert len(context.candidates) == 8
    assert len({item.cluster_id for item in context.candidates}) == 8
    assert context.baseline_ids == tuple(item.molecule_id for item in context.candidates[:4])


def test_evaluate_panel_uses_minimum_pairwise_tanimoto_distance():
    context = controlled_context()
    result = evaluate_diverse_panel(context, context.baseline_ids, attempt_number=1)
    assert result.score == pytest.approx(0.35)
    assert result.limiting_pair == ("mol-0", "mol-1")
    assert result.constraints_passed is True


def test_target_is_eighty_percent_of_attainable_improvement():
    context = controlled_context(baseline_score=0.35, benchmark_score=0.60)
    assert context.target_score == pytest.approx(0.55)


@pytest.mark.parametrize("selected_ids", [
    ("mol-0", "mol-0", "mol-2", "mol-3"),
    ("mol-0", "mol-1", "mol-2"),
    ("mol-0", "mol-1", "mol-2", "outside"),
])
def test_invalid_panels_fail_before_scoring(selected_ids):
    with pytest.raises(ValueError):
        evaluate_diverse_panel(controlled_context(), selected_ids, attempt_number=1)


def test_o01_excludes_hidden_benchmark_panel_and_is_canonical_json():
    record = build_objective_evidence(completed_run())
    payload = json.loads(record.payload_json)
    assert record.key == "O01"
    assert "benchmark_panel" not in payload
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == record.payload_json


def test_objective_figures_render_trajectory_structures_and_final_heatmap():
    figures = objective_figures(completed_run(), optimized_state())
    assert len(figures) == 3
    assert figures[0].axes[0].get_title() == "Objective score trajectory"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_objective_challenge.py -v
```

Expected: collection fails because `objective_challenge` does not exist.

- [ ] **Step 3: Implement immutable objective types and pure functions**

Implement these public interfaces in `objective_challenge.py`:

```python
PANEL_SIZE = 4
CANDIDATE_COUNT = 8
MAX_ATTEMPTS = 3
TARGET_FRACTION = 0.8


@dataclass(frozen=True)
class ObjectiveCandidate:
    molecule_id: str
    molecule_index: int
    source_row: int
    cluster_id: int


@dataclass(frozen=True)
class ObjectiveContext:
    candidates: tuple[ObjectiveCandidate, ...]
    baseline_ids: tuple[str, ...]
    baseline_score: float
    benchmark_score: float
    target_score: float
    distance_matrix: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class ObjectiveAttempt:
    attempt_number: int
    selected_ids: tuple[str, ...]
    decision_basis: str
    score: float
    limiting_pair: tuple[str, str]
    constraints_passed: bool
    achieved: bool


@dataclass(frozen=True)
class ObjectiveRun:
    context: ObjectiveContext
    attempts: tuple[ObjectiveAttempt, ...]
    achieved: bool
    termination_reason: str
    final_ids: tuple[str, ...]
    final_score: float
```

Add two public, non-mutating helpers to `chemistry_workflow.py`: `validated_similarity_matrix(state) -> np.ndarray`, which performs the same finite/range/symmetry/diagonal checks as E03 construction and returns a host copy, and `eligible_representative_groups(state) -> tuple[dict[str, Any], ...]`, which returns defensive copies of `_eligibility_groups(...)`. `build_objective_context(state)` must validate `WorkflowPhase.OPTIMIZED`, use only these public helpers, take eight clusters, enumerate `itertools.combinations(candidate_ids, 4)`, and set `target_score = baseline + 0.8 * (benchmark - baseline)`. Tie-breaking uses lexicographically sorted molecule-ID pairs and panels. `evaluate_diverse_panel(...)` rejects wrong counts, duplicates, out-of-pool IDs, and repeated cluster IDs before reading scores. `build_objective_evidence(...)` emits canonical O01 JSON without the benchmark panel. `objective_figures(...)` returns a trajectory figure, RDKit 2D grid image, and four-by-four Tanimoto heatmap.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_objective_challenge.py -v
```

Expected: all objective-domain tests pass.

- [ ] **Step 5: Commit the pure domain**

```bash
git add chemistry_workflow.py objective_challenge.py tests/test_chemistry_workflow.py tests/test_objective_challenge.py
git commit -m "feat: add bounded molecular diversity objective"
```

### Task 2: Add deterministic objective receipts

**Files:**
- Create: `tests/test_objective_receipts.py`
- Create: `objective_receipts.py`

- [ ] **Step 1: Write failing receipt tests**

```python
def test_objective_receipt_displays_validated_ids_and_fixed_executor():
    proposal = ObjectiveProposal(
        selected_ids=["mol-0", "mol-2", "mol-5", "mol-7"],
        decision_basis="Replace the limiting analogue.",
    )
    receipt = objective_receipt(proposal)
    assert receipt.validated_proposal == (
        "select_diverse_panel(selected_ids=['mol-0', 'mol-2', 'mol-5', 'mol-7'])"
    )
    assert receipt.python_evaluation == (
        "result = evaluate_diverse_panel(\n"
        "    selected_ids=proposal.selected_ids,\n"
        "    candidate_pool=candidate_pool,\n"
        "    similarity_matrix=similarity_matrix,\n"
        ")"
    )
    assert "decision_basis" not in repr(receipt)
```

- [ ] **Step 2: Run the receipt test and verify RED**

Run `python3 -m pytest tests/test_objective_receipts.py -v`.

Expected: collection fails because `objective_receipts` does not exist.

- [ ] **Step 3: Implement the frozen receipt and exact-model guard**

```python
@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    python_evaluation: str


def objective_receipt(proposal: ObjectiveProposal) -> ObjectiveReceipt:
    if type(proposal) is not ObjectiveProposal:
        raise ValueError("Proposal does not match the objective schema.")
    ids = repr(proposal.selected_ids)
    return ObjectiveReceipt(
        validated_proposal=f"select_diverse_panel(selected_ids={ids})",
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )
```

- [ ] **Step 4: Run receipt tests and verify GREEN**

Run `python3 -m pytest tests/test_objective_receipts.py -v`.

Expected: all receipt tests pass.

- [ ] **Step 5: Commit receipts**

```bash
git add objective_receipts.py tests/test_objective_receipts.py
git commit -m "feat: add objective proposal receipts"
```

### Task 3: Extend the bounded hosted controller

**Files:**
- Modify: `demo_agent.py:50-55, 114-218, 243-275, 360-512, 591-783`
- Modify: `tests/test_demo_agent.py`

- [ ] **Step 1: Write failing schema and loop tests**

Add tests proving that `ObjectiveProposal` requires four unique IDs, `begin_objective_challenge()` is legal only after the six stages, every attempt is returned to the same conversation, success stops immediately, three misses terminate explicitly, and synthesis is gated on objective termination:

```python
def test_controller_runs_objective_attempts_until_success():
    value, completions = completed_controller(objective_responses=[
        objective_response(["mol-0", "mol-1", "mol-2", "mol-4"]),
        objective_response(["mol-0", "mol-2", "mol-5", "mol-7"]),
    ])
    context = value.begin_objective_challenge()
    first = value.evaluate_next_objective_attempt()
    second = value.evaluate_next_objective_attempt()
    assert first.achieved is False
    assert second.achieved is True
    assert value.objective_run.termination_reason == "target_achieved"
    assert value.session.messages[-1]["role"] == "tool"


def test_conclusion_is_blocked_before_objective_termination():
    value, _ = completed_controller(objective_responses=[])
    with pytest.raises(ToolCallError, match="objective challenge"):
        value.request_synthesis()
```

- [ ] **Step 2: Run focused controller tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_demo_agent.py -k 'objective or conclusion_is_blocked' -v
```

Expected: failures because the objective schema and controller methods are absent.

- [ ] **Step 3: Add the strict schema, prompt, controller state, and methods**

Add `ObjectiveProposal` with a `field_validator` enforcing four unique IDs, extend the request-call cap to eleven accepted turns, and add controller fields `objective_context`, `objective_attempts`, `objective_run`, and `objective_prompt_appended`.

Implement:

```python
def begin_objective_challenge(self) -> ObjectiveContext:
    scientific = self.scientific_result()
    if self.objective_context is not None:
        raise ToolCallError("The objective challenge can be initialized exactly once.")
    context = build_objective_context(self.session.state)
    self.objective_context = context
    self.session.messages.append({"role": "user", "content": objective_prompt(context)})
    self.objective_prompt_appended = True
    if context.benchmark_score == context.baseline_score:
        self.objective_run = no_improvement_run(context)
    return context


def evaluate_next_objective_attempt(self) -> ObjectiveAttempt:
    context = self.objective_context
    if context is None or self.objective_run is not None:
        raise ToolCallError("The objective challenge is not awaiting an attempt.")
    proposal = _request_call(
        self.session, self.client, "select_diverse_panel", ObjectiveProposal, DEFAULT_MODEL
    )
    try:
        attempt = evaluate_diverse_panel(
            context,
            tuple(proposal.selected_ids),
            attempt_number=len(self.objective_attempts) + 1,
            decision_basis=proposal.decision_basis,
        )
    except Exception:
        raise ToolCallError("The objective evaluator rejected the proposed panel.") from None
    self.objective_attempts.append(attempt)
    _append_tool_result(self.session, objective_attempt_payload(attempt, context))
    if attempt.achieved or len(self.objective_attempts) == MAX_ATTEMPTS:
        self.objective_run = finalize_objective_run(context, tuple(self.objective_attempts))
    return attempt
```

Extend the conclusion schema to seven unique themes including `objective_driven_selection`, allow `O01`, require O01 for that theme and for limitations, and add O01 to the serialized evidence supplied to the final hosted call. Rename user-facing synthesis prompt wording to “evidence-backed conclusion.”

Refactor `scientific_result()` so its six-stage invariant remains exact but its accepted turn-count check allows the already-completed scientific report to be read after objective attempts. `request_synthesis()` must not call a helper that requires `turn_count == 7`; it must require the exact six stage results, optimized phase, no pending stage proposal, a completed `objective_run`, and `7 <= turn_count <= 10` before appending E01-E06 plus O01 and requesting the eleventh-or-earlier conclusion call.

- [ ] **Step 4: Run controller tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_demo_agent.py -v
```

Expected: all controller tests pass.

- [ ] **Step 5: Commit controller behavior**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add bounded objective attempt loop"
```

### Task 4: Add the live objective card

**Files:**
- Modify: `interactive_workflow.py:62-474`
- Modify: `tests/test_interactive_workflow.py`

- [ ] **Step 1: Write failing widget tests**

Extend the fake controller with objective context and attempts. Assert the six existing approvals reveal one `Run Objective Challenge` button instead of requesting the conclusion; clicking it renders the objective, accepted attempts, both receipts, score trajectory, ledger, final figures, and then the conclusion. Add separate success, three-attempt miss, no-improvement, hosted retry, duplicate-click, and unexpected-failure cases.

```python
def test_six_stages_open_one_objective_challenge_before_conclusion(monkeypatch):
    workflow, controller = started()
    for _ in range(6):
        workflow.approve_button.click()
    assert workflow.status == "objective_ready"
    assert controller.calls.count("synthesis") == 0
    assert "Run Objective Challenge" in html_text(workflow.active_card)


def test_objective_card_retains_attempts_and_visible_execution_receipts(monkeypatch):
    workflow, controller = completed_stages_with_objective_attempts()
    workflow.objective_button.click()
    text = html_text(workflow.active_card)
    assert "Validated Nemotron proposal" in text
    assert "Evaluation executed by Python" in text
    assert "Limiting pair" in text
    assert "Goal achieved" in text
    assert len(workflow.objective_attempt_cards) == 2
```

- [ ] **Step 2: Run widget tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_interactive_workflow.py -k objective -v
```

Expected: failures because the objective widget state and callbacks do not exist.

- [ ] **Step 3: Implement the single appended card and bounded callbacks**

Add `objective_button`, `objective_attempt_cards`, and `objective_output` fields. Replace both six-stage completion branches with `_show_objective_challenge()`. The button callback must disable itself, call `begin_objective_challenge()`, render the baseline and target, loop while `controller.objective_run is None`, append each accepted attempt, update the same Matplotlib trajectory output, and call `_request_synthesis()` only after termination.

Render each attempt with:

```python
widgets.Accordion((widgets.HTML(
    f"<p><b>Decision summary:</b> {escape(attempt.decision_basis)}</p>"
    f"<b>Validated Nemotron proposal</b><pre>{escape(receipt.validated_proposal)}</pre>"
    f"<b>Evaluation executed by Python</b><pre>{escape(receipt.python_evaluation)}</pre>"
    f"<p><b>D_min:</b> {attempt.score:.3f} &nbsp; "
    f"<b>Limiting pair:</b> {escape(' / '.join(attempt.limiting_pair))} &nbsp; "
    f"<b>Result:</b> {'Goal achieved' if attempt.achieved else 'Revise'}</p>"
),))
```

Keep previous accordions collapsed, keep the current one expanded, and reuse `_safe_message`, `_stop`, and duplicate-click protection. Change the final card title to `Evidence-Backed Conclusion`.

- [ ] **Step 4: Run widget tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_interactive_workflow.py -v
```

Expected: all widget tests pass, including the existing six-stage tests after updating their expected conclusion gate.

- [ ] **Step 5: Commit the UI**

```bash
git add interactive_workflow.py tests/test_interactive_workflow.py
git commit -m "feat: visualize objective-driven agent attempts"
```

### Task 5: Update notebook narrative and documentation

**Files:**
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing notebook-contract tests**

Assert notebook Markdown contains `Molecular Evidence Generation`, `Objective-Driven Agent Challenge`, `Run Objective Challenge`, `minimum pairwise Tanimoto distance`, and `Evidence-Backed Conclusion`; assert the public code surface remains one `launch_interactive_workflow(...)` call and contains no direct objective executor call.

- [ ] **Step 2: Run notebook tests and verify RED**

Run `python3 -m pytest tests/test_notebook.py -v`.

Expected: narrative assertions fail against the current notebook copy.

- [ ] **Step 3: Update only notebook Markdown and README prose**

Keep the existing preflight, user-goal assignment, and `launch_interactive_workflow(...)` code cells. Explain the two-movement narrative, one added button, maximin score, three-attempt bound, visible proposal/code distinction, and claim boundary. Update README usage and acceptance descriptions with the same terms.

- [ ] **Step 4: Run notebook tests and verify GREEN**

Run `python3 -m pytest tests/test_notebook.py -v`.

Expected: all notebook contract tests pass.

- [ ] **Step 5: Commit presentation copy**

```bash
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py README.md
git commit -m "docs: present the objective-driven notebook narrative"
```

### Task 6: Add integration and GPU acceptance gates

**Files:**
- Modify: `tests/test_gpu_acceptance.py`
- Modify: `tests/test_skill_snapshot.py`

- [ ] **Step 1: Write failing integration assertions**

Extend the GPU response script with up to three strict `select_diverse_panel` calls derived from the live bounded candidate IDs. Require:

```python
assert len(context.candidates) == 8
assert context.benchmark_score > context.baseline_score
assert 1 <= len(result.objective_run.attempts) <= 3
assert result.objective_run.termination_reason in {
    "target_achieved", "attempt_limit_reached"
}
assert result.objective_evidence.key == "O01"
```

Add a skill-snapshot assertion that the vendored nvMolKit skill remains unchanged by this feature.

- [ ] **Step 2: Run non-GPU integration tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_gpu_acceptance.py -k 'not real_gpu' -v
python3 -m pytest tests/test_skill_snapshot.py -v
```

Expected: the new objective integration assertion fails before the production integration is complete; skill snapshot remains green.

- [ ] **Step 3: Complete synchronous compatibility and result plumbing**

Update `run_scientific_loop` compatibility only where required, and ensure the interactive controller's final `WorkflowResult` includes `objective_run` and `objective_evidence`. Do not auto-run the objective from the legacy scientific-only function. Add a separate full-workflow helper only if an existing test or GPU harness requires non-widget execution.

- [ ] **Step 4: Run the full CPU suite**

Run:

```bash
python3 -m pytest -q
```

Expected: all CPU tests pass with zero failures.

- [ ] **Step 5: Run structural hygiene checks**

Run:

```bash
git diff --check
python3 -m compileall -q chemistry_workflow.py objective_challenge.py objective_receipts.py demo_agent.py interactive_workflow.py command_receipts.py
```

Expected: both commands exit zero with no output.

- [ ] **Step 6: Commit integration gates**

```bash
git add tests/test_gpu_acceptance.py tests/test_skill_snapshot.py demo_agent.py
git commit -m "test: gate objective-driven challenge integration"
```

### Task 7: Verify the launchable on Brev

**Files:**
- Modify only if a verified defect is found: implementation or test files named above
- Record results in the final handoff; do not commit credentials or transient runtime output

- [ ] **Step 1: Run the repository's documented Brev/GPU acceptance command**

Use the current Launchable instance and documented acceptance path from `README.md` and `tests/test_gpu_acceptance.py`. Do not print or persist `NVIDIA_API_KEY`.

- [ ] **Step 2: Verify the fixed dataset objective gate**

Confirm from structured output that the pool has eight candidates, the baseline-to-benchmark gap is positive, one to three real hosted proposals were evaluated, and success is reported only when `final_score >= target_score`.

- [ ] **Step 3: Verify the live notebook presentation**

Open the notebook through the Launchable, approve the six existing stages, click `Run Objective Challenge`, and confirm the same card visibly retains baseline and attempts, proposal and Python receipts, score/limiting-pair feedback, trajectory, ledger, final structures, heatmap, and `Evidence-Backed Conclusion`.

- [ ] **Step 4: Run final local verification after any remote-driven fix**

Run:

```bash
python3 -m pytest -q
git diff --check
git status --short
```

Expected: tests pass, diff check exits zero, and status contains only intentional feature changes.

- [ ] **Step 5: Commit any verified remote fix separately**

```bash
git add objective_challenge.py objective_receipts.py demo_agent.py interactive_workflow.py notebooks/nvmolkit_nemotron_demo.ipynb README.md tests
git commit -m "fix: close objective challenge acceptance gaps"
```

Skip this commit when the remote run reveals no defect.
