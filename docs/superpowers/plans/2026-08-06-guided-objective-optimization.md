# Guided Objective Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Nemotron use deterministic, ranked molecular-swap evidence to improve a four-compound diversity panel and reliably reach the fixed target within three accepted attempts.

**Architecture:** Keep the six-stage chemistry workflow unchanged. Add a pure counterfactual-ranking layer to `objective_challenge.py`, make the bounded controller expose and enforce that shortlist after each miss, and render accepted decisions as the approved compact decision ladder. Invalid hosted revisions use a separate two-response correction budget and never enter the scientific ledger.

**Tech Stack:** Python 3.12, dataclasses, Pydantic v2, NumPy, RDKit, ipywidgets, pytest, nvMolKit 0.5.0, CUDA-enabled PyTorch, hosted Nemotron tool calls, Brev NVIDIA L4.

---

## File Map

- Modify `objective_challenge.py`: immutable swap record, exhaustive legal one-swap enumeration, deterministic ranking, selected-swap attempt metadata, and O01 serialization.
- Modify `demo_agent.py`: shortlist protocol payload, listed-panel enforcement, duplicate rejection, two-response correction budget, and revised hosted-turn bounds.
- Modify `objective_receipts.py`: render the exact validated intervention without model-generated code.
- Modify `interactive_workflow.py`: render the approved observe → act → measure decision ladder.
- Modify `tests/test_objective_challenge.py`: ranking, invariants, serialization, and no-hard-coded-answer coverage.
- Modify `tests/test_objective_agent_loop.py`: same-conversation guidance, duplicate/list enforcement, correction accounting, and termination coverage.
- Modify `tests/test_objective_receipts.py`: deterministic intervention receipt coverage.
- Modify `tests/test_interactive_workflow.py`: compact decision-ladder rendering and prior-attempt retention.
- Modify `tests/test_gpu_acceptance.py`: exercise controller-generated swaps on the qualified L4 dataset instead of injecting the hidden best panel.
- Do not modify `notebooks/nvmolkit_nemotron_demo.ipynb` unless a failing notebook-copy test proves explanatory text must change.

### Task 1: Add deterministic legal-swap ranking

**Files:**
- Modify: `objective_challenge.py:20-175`
- Test: `tests/test_objective_challenge.py`

- [ ] **Step 1: Write failing ranking tests**

Add `rank_legal_swaps` to the imports. Also import `Path` and `objective_challenge`, then add:

```python
def test_rank_legal_swaps_returns_only_target_reaching_moves_when_available():
    context = build_objective_context(optimized_state())
    current = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current panel.",
    )

    suggestions = rank_legal_swaps(context, current)

    assert len(suggestions) == 3
    assert all(item.predicted_score >= context.target_score for item in suggestions)
    assert all(item.score_delta > 0 for item in suggestions)
    assert suggestions == tuple(sorted(
        suggestions,
        key=lambda item: (
            -item.predicted_score,
            item.replace_id,
            item.replacement_id,
            item.resulting_ids,
        ),
    ))


def test_ranked_swaps_equal_direct_evaluation_and_preserve_constraints():
    context = build_objective_context(optimized_state())
    current = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current panel.",
    )

    for suggestion in rank_legal_swaps(context, current):
        measured = evaluate_diverse_panel(
            context,
            suggestion.resulting_ids,
            attempt_number=2,
            decision_basis="Evaluate the suggested intervention.",
        )
        assert measured.score == pytest.approx(suggestion.predicted_score)
        assert measured.limiting_pair == suggestion.limiting_pair
        assert suggestion.score_delta == pytest.approx(measured.score - current.score)
        assert suggestion.replace_id in current.selected_ids
        assert suggestion.replacement_id not in current.selected_ids
        assert len(set(suggestion.resulting_ids)) == PANEL_SIZE


def test_ranked_swaps_are_empty_after_success_and_source_has_no_answer_ids():
    context = build_objective_context(optimized_state())
    achieved = evaluate_diverse_panel(
        context,
        ("mol-0", "mol-2", "mol-4", "mol-6"),
        attempt_number=1,
        decision_basis="Use a target-reaching panel.",
    )

    assert achieved.achieved is True
    assert rank_legal_swaps(context, achieved) == ()
    source = Path(objective_challenge.__file__).read_text(encoding="utf-8")
    assert not any(f'"mol-{index}"' in source for index in range(CANDIDATE_COUNT))
    assert "CHEMBL" not in source


def test_every_below_target_fixture_panel_has_a_two_revision_path():
    context = build_objective_context(optimized_state())
    candidate_ids = tuple(item.molecule_id for item in context.candidates)
    misses = []
    for panel in itertools.combinations(candidate_ids, PANEL_SIZE):
        first = evaluate_diverse_panel(
            context,
            panel,
            attempt_number=1,
            decision_basis="Evaluate one fixture panel.",
        )
        if first.achieved:
            continue
        misses.append(first)
        first_moves = rank_legal_swaps(context, first)
        assert first_moves
        for first_move in first_moves:
            second = evaluate_diverse_panel(
                context,
                first_move.resulting_ids,
                attempt_number=2,
                decision_basis="Apply the first guided revision.",
            )
            if second.achieved:
                continue
            assert any(
                move.predicted_score + 1e-12 >= context.target_score
                for move in rank_legal_swaps(context, second)
            )
    assert misses
```

Also import `itertools` in the test module.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest \
  tests/test_objective_challenge.py::test_rank_legal_swaps_returns_only_target_reaching_moves_when_available \
  tests/test_objective_challenge.py::test_ranked_swaps_equal_direct_evaluation_and_preserve_constraints \
  tests/test_objective_challenge.py::test_ranked_swaps_are_empty_after_success_and_source_has_no_answer_ids \
  tests/test_objective_challenge.py::test_every_below_target_fixture_panel_has_a_two_revision_path \
  -q
```

Expected: collection fails because `rank_legal_swaps` is not defined.

- [ ] **Step 3: Implement the immutable swap record and pure ranker**

Place this record before `ObjectiveAttempt`:

```python
SUGGESTION_LIMIT = 3


@dataclass(frozen=True)
class ObjectiveSwap:
    replace_id: str
    replacement_id: str
    resulting_ids: tuple[str, ...]
    predicted_score: float
    score_delta: float
    limiting_pair: tuple[str, str]
```

Add after `evaluate_diverse_panel`:

```python
def rank_legal_swaps(
    context: ObjectiveContext,
    current: ObjectiveAttempt,
) -> tuple[ObjectiveSwap, ...]:
    """Return up to three deterministic, strictly improving one-ID swaps."""
    if type(context) is not ObjectiveContext or type(current) is not ObjectiveAttempt:
        raise ValueError("Swap ranking requires exact objective records.")
    if current.achieved:
        return ()
    candidates = {item.molecule_id: item for item in context.candidates}
    if any(item not in candidates for item in current.selected_ids):
        raise ValueError("Current panel is outside the objective candidate pool.")
    ranked: list[ObjectiveSwap] = []
    for replace_id in current.selected_ids:
        replace_position = current.selected_ids.index(replace_id)
        for replacement_id in sorted(candidates):
            if replacement_id in current.selected_ids:
                continue
            resulting = list(current.selected_ids)
            resulting[replace_position] = replacement_id
            resulting_ids = tuple(resulting)
            cluster_ids = {
                candidates[molecule_id].cluster_id for molecule_id in resulting_ids
            }
            if len(cluster_ids) != PANEL_SIZE:
                continue
            predicted_score, limiting_pair = _score_panel(context, resulting_ids)
            score_delta = float(predicted_score - current.score)
            if score_delta <= 1e-12:
                continue
            ranked.append(ObjectiveSwap(
                replace_id=replace_id,
                replacement_id=replacement_id,
                resulting_ids=resulting_ids,
                predicted_score=float(predicted_score),
                score_delta=score_delta,
                limiting_pair=limiting_pair,
            ))
    ranked.sort(key=lambda item: (
        -item.predicted_score,
        item.replace_id,
        item.replacement_id,
        item.resulting_ids,
    ))
    target_reaching = [
        item for item in ranked
        if item.predicted_score + 1e-12 >= context.target_score
    ]
    eligible = target_reaching if target_reaching else ranked
    return tuple(eligible[:SUGGESTION_LIMIT])
```

- [ ] **Step 4: Run the objective module and verify GREEN**

```bash
python -m pytest tests/test_objective_challenge.py -q
```

Expected: all tests in the module pass.

- [ ] **Step 5: Commit the pure ranking layer**

```bash
git add objective_challenge.py tests/test_objective_challenge.py
git commit -m "feat: rank legal objective swaps"
```

### Task 2: Bind attempts and O01 to the selected intervention

**Files:**
- Modify: `objective_challenge.py:45-175,219-255`
- Test: `tests/test_objective_challenge.py`

- [ ] **Step 1: Write failing selected-swap and evidence tests**

```python
def test_evaluate_panel_records_only_an_exact_selected_swap():
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current panel.",
    )
    selected = rank_legal_swaps(context, first)[0]

    second = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Choose the measured improving swap.",
        selected_swap=selected,
    )

    assert second.selected_swap is selected
    with pytest.raises(ValueError, match="selected swap"):
        evaluate_diverse_panel(
            context,
            context.baseline_ids,
            attempt_number=2,
            decision_basis="Claim a swap without using its panel.",
            selected_swap=selected,
        )


def test_o01_serializes_the_selected_intervention_without_hidden_answers():
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current panel.",
    )
    selected = rank_legal_swaps(context, first)[0]
    second = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Choose the improving swap.",
        selected_swap=selected,
    )
    payload = json.loads(build_objective_evidence(
        finalize_objective_run(context, (first, second))
    ).payload_json)

    intervention = payload["attempts"][1]["selected_swap"]
    assert intervention["replace_id"] == selected.replace_id
    assert intervention["replacement_id"] == selected.replacement_id
    assert intervention["score_delta"] == pytest.approx(selected.score_delta)
    assert payload["attempts"][0]["selected_swap"] is None
    assert "benchmark_panel" not in json.dumps(payload)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest \
  tests/test_objective_challenge.py::test_evaluate_panel_records_only_an_exact_selected_swap \
  tests/test_objective_challenge.py::test_o01_serializes_the_selected_intervention_without_hidden_answers \
  -q
```

Expected: both tests fail because `ObjectiveAttempt` and `evaluate_diverse_panel` do not accept `selected_swap`.

- [ ] **Step 3: Add selected-swap metadata and validation**

Append this defaulted field to `ObjectiveAttempt`:

```python
    selected_swap: ObjectiveSwap | None = None
```

Extend the evaluator signature:

```python
def evaluate_diverse_panel(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...] | list[str],
    *,
    attempt_number: int,
    decision_basis: str,
    selected_swap: ObjectiveSwap | None = None,
) -> ObjectiveAttempt:
```

After scoring and before returning, add:

```python
    if selected_swap is not None:
        if type(selected_swap) is not ObjectiveSwap:
            raise ValueError("Objective selected swap has an invalid type.")
        if set(panel) != set(selected_swap.resulting_ids):
            raise ValueError("Objective panel does not match the selected swap.")
        if abs(score - selected_swap.predicted_score) > 1e-12:
            raise ValueError("Objective score does not match the selected swap.")
```

Pass `selected_swap=selected_swap` into the returned `ObjectiveAttempt`.

- [ ] **Step 4: Serialize the intervention in O01**

Add this key to every attempt object in `build_objective_evidence`:

```python
                "selected_swap": None if attempt.selected_swap is None else {
                    "replace_id": attempt.selected_swap.replace_id,
                    "replacement_id": attempt.selected_swap.replacement_id,
                    "resulting_ids": list(attempt.selected_swap.resulting_ids),
                    "predicted_score": attempt.selected_swap.predicted_score,
                    "score_delta": attempt.selected_swap.score_delta,
                    "limiting_pair": list(attempt.selected_swap.limiting_pair),
                },
```

- [ ] **Step 5: Run the module and verify GREEN**

```bash
python -m pytest tests/test_objective_challenge.py -q
```

Expected: all objective-challenge tests pass.

- [ ] **Step 6: Commit attempt provenance**

```bash
git add objective_challenge.py tests/test_objective_challenge.py
git commit -m "feat: retain selected objective intervention"
```

### Task 3: Enforce guided revisions with a separate correction budget

**Files:**
- Modify: `demo_agent.py:35-48,700-1010`
- Modify: `interactive_workflow.py:199-225`
- Test: `tests/test_objective_agent_loop.py`
- Test: `tests/test_interactive_workflow.py`

- [ ] **Step 1: Write failing controller tests**

Import `rank_legal_swaps` in `tests/test_objective_agent_loop.py`, then add:

```python
def test_missed_attempt_returns_ranked_legal_swaps_to_same_conversation():
    controller, _ = completed_controller([
        proposal(["mol-0", "mol-1", "mol-2", "mol-3"], "Measure the panel."),
    ])
    controller.begin_objective_challenge()
    pending = controller.request_objective_attempt()
    first = controller.execute_objective_attempt(pending)
    payload = json.loads(controller.session.messages[-1]["content"])

    expected = rank_legal_swaps(controller.objective_context, first)
    assert payload["legal_improving_swaps"] == [
        {
            "replace_id": item.replace_id,
            "replacement_id": item.replacement_id,
            "resulting_ids": list(item.resulting_ids),
            "predicted_score": item.predicted_score,
            "score_delta": item.score_delta,
            "limiting_pair": list(item.limiting_pair),
        }
        for item in expected
    ]


def test_duplicate_revision_is_corrected_without_consuming_an_attempt():
    first_ids = ["mol-0", "mol-1", "mol-2", "mol-3"]
    controller, completions = completed_controller([
        proposal(first_ids, "Measure the panel."),
        proposal(first_ids, "Repeat the panel."),
        proposal(["mol-1", "mol-2", "mol-3", "mol-4"], "Use a listed swap."),
    ])
    controller.begin_objective_challenge()
    first_pending = controller.request_objective_attempt()
    controller.execute_objective_attempt(first_pending)
    expected = controller.objective_suggestions[0]
    completions.responses[-1] = proposal(
        list(expected.resulting_ids),
        "Choose a listed improving swap.",
    )

    revised = controller.request_objective_attempt()

    assert set(revised.selected_ids) == set(expected.resulting_ids)
    assert len(controller.objective_attempts) == 1
    assert controller.objective_rejection_count == 1
    rejected = json.loads(controller.session.messages[-2]["content"])
    assert rejected["accepted"] is False
    assert rejected["reason"] == "duplicate_panel"


def test_unlisted_revisions_exhaust_two_corrections_and_fail_closed():
    first_ids = ["mol-0", "mol-1", "mol-2", "mol-3"]
    controller, _ = completed_controller([
        proposal(first_ids, "Measure the panel."),
        proposal(first_ids, "Repeat once."),
        proposal(first_ids, "Repeat twice."),
    ])
    controller.begin_objective_challenge()
    first_pending = controller.request_objective_attempt()
    controller.execute_objective_attempt(first_pending)

    with pytest.raises(demo_agent.ToolCallError, match="correction limit"):
        controller.request_objective_attempt()

    assert len(controller.objective_attempts) == 1
    assert controller.objective_rejection_count == 2
    assert controller.pending_objective is None
```

- [ ] **Step 2: Run the controller tests and verify RED**

```bash
python -m pytest \
  tests/test_objective_agent_loop.py::test_missed_attempt_returns_ranked_legal_swaps_to_same_conversation \
  tests/test_objective_agent_loop.py::test_duplicate_revision_is_corrected_without_consuming_an_attempt \
  tests/test_objective_agent_loop.py::test_unlisted_revisions_exhaust_two_corrections_and_fail_closed \
  -q
```

Expected: failures show missing `objective_suggestions` and `objective_rejection_count` behavior.

- [ ] **Step 3: Add explicit state and hosted-turn limits**

Import `ObjectiveSwap` and `rank_legal_swaps` in `demo_agent.py`. Add:

```python
MAX_OBJECTIVE_CORRECTIONS = 2
MAX_OBJECTIVE_HOSTED_TURNS = 12
MAX_OBJECTIVE_SYNTHESIS_TURNS = 13
```

Add controller fields:

```python
    pending_objective_swap: ObjectiveSwap | None = None
    objective_suggestions: tuple[ObjectiveSwap, ...] = ()
    objective_rejection_count: int = 0
```

Add deterministic payload and panel helpers above the controller:

```python
def _swap_payload(item: ObjectiveSwap) -> dict[str, Any]:
    return {
        "replace_id": item.replace_id,
        "replacement_id": item.replacement_id,
        "resulting_ids": list(item.resulting_ids),
        "predicted_score": item.predicted_score,
        "score_delta": item.score_delta,
        "limiting_pair": list(item.limiting_pair),
    }


def _panel_key(selected_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(selected_ids))
```

- [ ] **Step 4: Implement bounded proposal correction**

Replace the single hosted-call block in `request_objective_attempt` with:

```python
        while True:
            proposal = _request_call(
                self.session,
                self.client,
                "select_diverse_panel",
                ObjectiveProposal,
                DEFAULT_MODEL,
                _max_turns=MAX_OBJECTIVE_HOSTED_TURNS,
            )
            proposal_key = _panel_key(proposal.selected_ids)
            prior_keys = {
                _panel_key(attempt.selected_ids) for attempt in self.objective_attempts
            }
            selected_swap = next(
                (
                    item for item in self.objective_suggestions
                    if _panel_key(item.resulting_ids) == proposal_key
                ),
                None,
            )
            candidate_ids = {
                candidate.molecule_id for candidate in self.objective_context.candidates
            }
            reason = None
            if proposal_key in prior_keys:
                reason = "duplicate_panel"
            elif any(item not in candidate_ids for item in proposal.selected_ids):
                reason = "out_of_pool_panel"
            elif self.objective_attempts and selected_swap is None:
                reason = "panel_not_in_legal_improving_swaps"
            if reason is None:
                self.pending_objective = ObjectiveProposal.model_validate(
                    proposal.model_dump()
                )
                self.pending_objective_swap = selected_swap
                return self.pending_objective
            self.objective_rejection_count += 1
            _append_tool_result(self.session, {
                "accepted": False,
                "reason": reason,
                "corrections_remaining": (
                    MAX_OBJECTIVE_CORRECTIONS - self.objective_rejection_count
                ),
                "legal_improving_swaps": [
                    _swap_payload(item) for item in self.objective_suggestions
                ],
                "candidate_ids": (
                    [] if self.objective_attempts else sorted(candidate_ids)
                ),
                "instruction": (
                    "Submit exactly one listed resulting_ids panel with a concise "
                    "limiting-pair rationale."
                    if self.objective_attempts
                    else "Submit four unique IDs from candidate_ids."
                ),
            })
            if self.objective_rejection_count >= MAX_OBJECTIVE_CORRECTIONS:
                raise ToolCallError("The objective proposal correction limit was reached.")
```

- [ ] **Step 5: Bind execution to the selected swap and return new guidance**

Pass `selected_swap=self.pending_objective_swap` into `evaluate_diverse_panel`. After successful evaluation, clear `pending_objective_swap` and calculate:

```python
        self.objective_suggestions = (
            rank_legal_swaps(context, attempt)
            if not attempt.achieved and len(self.objective_attempts) < MAX_ATTEMPTS
            else ()
        )
```

Add this to the accepted tool-result payload:

```python
                "legal_improving_swaps": [
                    _swap_payload(item) for item in self.objective_suggestions
                ],
```

If a miss has attempts remaining but produces no suggestions, raise `ToolCallError("No legal improving objective revision is available.")` before making another hosted request.

- [ ] **Step 6: Update exact turn invariants**

In `scientific_result`, allow `7 <= session.turn_count <= MAX_OBJECTIVE_HOSTED_TURNS` when an objective context exists. Use `_max_turns=MAX_OBJECTIVE_SYNTHESIS_TURNS` for objective-aware synthesis. Update `_objective_retryable` in `interactive_workflow.py` to require:

```python
                and self.controller.session.turn_count == (
                    7 + len(attempts) + self.controller.objective_rejection_count
                )
```

Successful no-rejection tests retain their current turn counts. Correction tests assert one hosted turn per rejected proposal.
Add `self.objective_rejection_count = 0`, `self.objective_suggestions = ()`, and
`self.pending_objective_swap = None` to the fake `Controller` in
`tests/test_interactive_workflow.py` so retry-state tests exercise the same public state contract.

- [ ] **Step 7: Run controller and UI retry tests**

```bash
python -m pytest \
  tests/test_objective_agent_loop.py \
  tests/test_interactive_workflow.py::test_known_objective_proposal_failure_after_measured_attempt_has_one_guarded_retry \
  tests/test_interactive_workflow.py::test_retry_objective_rechecks_pending_state_and_stops \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit guided controller behavior**

```bash
git add demo_agent.py interactive_workflow.py tests/test_objective_agent_loop.py tests/test_interactive_workflow.py
git commit -m "feat: guide bounded objective revisions"
```

### Task 4: Render the approved decision ladder

**Files:**
- Modify: `objective_receipts.py`
- Modify: `interactive_workflow.py:462-590`
- Test: `tests/test_objective_receipts.py`
- Test: `tests/test_interactive_workflow.py`

- [ ] **Step 1: Write failing receipt and ladder tests**

Import `ObjectiveSwap` in `tests/test_objective_receipts.py`, then add:

```python
def test_objective_receipt_renders_exact_selected_intervention():
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.75,
        score_delta=0.40,
        limiting_pair=("mol-0", "mol-2"),
    )

    receipt = objective_receipt(proposal(), swap)

    assert receipt.validated_intervention == (
        "replace molecule_id='mol-1' with molecule_id='mol-5'"
    )
    assert "0.75" not in receipt.validated_intervention
```

Update the fake controller's second `ObjectiveAttempt` to carry a deterministic `selected_swap`, then add to `tests/test_interactive_workflow.py`:

```python
def test_objective_attempts_render_as_observe_act_measure_decision_ladder(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow, _controller = started()

    run_objective(workflow)

    summary = workflow.objective_summary.value
    details = " ".join(html_text(card) for card in workflow.objective_attempt_cards)
    assert "Observe" in summary
    assert "Agent action" in summary
    assert "Measure" in summary
    assert "replace mol-1" in summary
    assert "with mol-4" in summary
    assert "+0.450" in summary
    assert "0.800 ≥ 0.710" in summary
    assert "Validated intervention" in details
    assert "Goal achieved" in details
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest \
  tests/test_objective_receipts.py::test_objective_receipt_renders_exact_selected_intervention \
  tests/test_interactive_workflow.py::test_objective_attempts_render_as_observe_act_measure_decision_ladder \
  -q
```

Expected: failures show the missing receipt field and decision-ladder labels.

- [ ] **Step 3: Extend the deterministic receipt**

Import `ObjectiveSwap`, then change the receipt API to:

```python
@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    validated_intervention: str | None
    python_evaluation: str


def objective_receipt(
    proposal: ObjectiveProposal,
    selected_swap: ObjectiveSwap | None = None,
) -> ObjectiveReceipt:
    if type(proposal) is not ObjectiveProposal:
        raise ValueError("Proposal does not match the objective schema.")
    if selected_swap is not None and type(selected_swap) is not ObjectiveSwap:
        raise ValueError("Intervention does not match the objective swap schema.")
    selected_ids = repr(list(proposal.selected_ids))
    intervention = None
    if selected_swap is not None:
        intervention = (
            f"replace molecule_id={selected_swap.replace_id!r} "
            f"with molecule_id={selected_swap.replacement_id!r}"
        )
    return ObjectiveReceipt(
        validated_proposal=f"select_diverse_panel(selected_ids={selected_ids})",
        validated_intervention=intervention,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )
```

Update existing expected `ObjectiveReceipt` values with `validated_intervention=None`.

- [ ] **Step 4: Render stable decision-ladder rows**

Add this static helper to `InteractiveWorkflow`:

```python
    @staticmethod
    def _objective_attempt_row(context, attempt, prior_attempt=None) -> str:
        swap = attempt.selected_swap
        if swap is None:
            action = "Initial panel"
            observation = "Initial agent proposal"
            delta = attempt.score - context.baseline_score
        else:
            if prior_attempt is None:
                raise ValueError("A guided revision requires its prior attempt.")
            action = f"replace {swap.replace_id} with {swap.replacement_id}"
            observation = (
                f"limiting pair {prior_attempt.limiting_pair[0]} / "
                f"{prior_attempt.limiting_pair[1]}"
            )
            delta = swap.score_delta
        comparison = (
            f"{attempt.score:.3f} ≥ {context.target_score:.3f}"
            if attempt.achieved
            else f"{attempt.score:.3f} < {context.target_score:.3f}"
        )
        outcome = "Goal achieved" if attempt.achieved else "Revise"
        return (
            "<div style='border-left:3px solid #76B900;padding:8px 12px;margin:6px 0'>"
            f"<b>Attempt {attempt.attempt_number}</b> · {escape(outcome)}<br>"
            f"<small><b>Observe:</b> {escape(observation)}</small><br>"
            f"<small><b>Agent action:</b> {escape(action)}</small><br>"
            f"<small><b>Measure:</b> {escape(comparison)} · Δ {delta:+.3f}</small>"
            "</div>"
        )
```

Use the helper in `_objective_summary_html` instead of the existing table rows:

```python
        decision_ladder = "".join(
            InteractiveWorkflow._objective_attempt_row(
                context,
                attempt,
                attempts[index - 1] if index else None,
            )
            for index, attempt in enumerate(attempts)
        )
```

Preserve the target definition, score strip, outcome line, accordion retention, and figure rendering. The first row says `Initial agent proposal`; every revision row observes the preceding attempt's measured limiting pair, not the counterfactual's predicted limiting pair.

In `_append_objective_attempt`, call:

```python
        receipt = objective_receipt(proposal, attempt.selected_swap)
```

When `receipt.validated_intervention` is present, add a **Validated intervention** block before **Evaluation executed by Python**.

- [ ] **Step 5: Run UI and receipt modules and verify GREEN**

```bash
python -m pytest tests/test_objective_receipts.py tests/test_interactive_workflow.py -q
```

Expected: all tests pass; prior attempt cards remain present and only the newest is expanded.

- [ ] **Step 6: Commit the decision ladder**

```bash
git add objective_receipts.py interactive_workflow.py tests/test_objective_receipts.py tests/test_interactive_workflow.py
git commit -m "feat: show objective decision ladder"
```

### Task 5: Qualify the controller-generated path on GPU

**Files:**
- Modify: `tests/test_gpu_acceptance.py:15-210`

- [ ] **Step 1: Write the failing GPU-source gate**

Add these required source tokens to `test_gpu_acceptance_source_gates_default_objective_challenge`:

```python
        "objective_suggestions",
        "selected_swap.score_delta",
        "len(set(accepted_panels)) == len(accepted_panels)",
```

Run:

```bash
python -m pytest tests/test_gpu_acceptance.py::test_gpu_acceptance_source_gates_default_objective_challenge -q
```

Expected: FAIL because the GPU test still injects a directly enumerated `best_panel`.

- [ ] **Step 2: Replace hidden-best-panel injection with controller guidance**

Remove the direct `best_panel = max(...)` block. After `begin_objective_challenge`, qualify the complete bounded search space and then build each accepted response from the controller's current shortlist.

Keep the `itertools` import for this dataset-wide reachability gate before the live path. It replaces the hidden-best-panel calculation:

```python
    all_panels = list(itertools.combinations(candidate_ids, 4))
    below_target = []
    for panel in all_panels:
        first = evaluate_diverse_panel(
            context,
            panel,
            attempt_number=1,
            decision_basis="Qualify one bounded starting panel.",
        )
        if first.achieved:
            continue
        below_target.append(first)
        first_moves = rank_legal_swaps(context, first)
        assert first_moves
        for first_move in first_moves:
            second = evaluate_diverse_panel(
                context,
                first_move.resulting_ids,
                attempt_number=2,
                decision_basis="Qualify the first guided move.",
                selected_swap=first_move,
            )
            if second.achieved:
                continue
            assert any(
                move.predicted_score + 1e-12 >= context.target_score
                for move in rank_legal_swaps(context, second)
            )
    assert len(all_panels) == 70
    assert len(below_target) == 35
```

Import `rank_legal_swaps` with `evaluate_diverse_panel`. Then script the accepted controller path:

```python
    completions.expected_names.append("select_diverse_panel")
    completions.arguments.append({
        "selected_ids": list(context.baseline_ids),
        "decision_basis": "Measure the defined baseline before revising it.",
    })
    accepted_panels = []
    accepted_scores = []
    while controller.objective_run is None:
        proposal = controller.request_objective_attempt()
        attempt = controller.execute_objective_attempt(proposal)
        accepted_panels.append(tuple(sorted(attempt.selected_ids)))
        accepted_scores.append(attempt.score)
        if controller.objective_run is None:
            assert controller.objective_suggestions
            selected_swap = controller.objective_suggestions[0]
            assert selected_swap.score_delta > 0
            completions.expected_names.append("select_diverse_panel")
            completions.arguments.append({
                "selected_ids": list(selected_swap.resulting_ids),
                "decision_basis": (
                    "Replace the measured limiting member using the highest-scoring "
                    "legal improving swap."
                ),
            })

    assert len(set(accepted_panels)) == len(accepted_panels)
    assert all(
        later > earlier + 1e-12
        for earlier, later in zip(accepted_scores, accepted_scores[1:])
    )
```

Keep the target, O01, CUDA tensor, matrix, clustering, conformer, and evidence assertions. Replace fixed turn-count assertions with:

```python
    assert controller.session.turn_count == 7 + len(accepted_panels)
    assert len(completions.calls) == controller.session.turn_count
```

- [ ] **Step 3: Run the full CPU suite before synchronization**

```bash
python -m pytest -q
git diff --check
```

Expected: all tests pass locally, with the L4 test skipped when `RUN_GPU_TESTS` is unset, and `git diff --check` exits 0.

- [ ] **Step 4: Commit the GPU acceptance change**

```bash
git add tests/test_gpu_acceptance.py
git commit -m "test: qualify guided objective path on GPU"
```

### Task 6: Run live Brev acceptance and preserve an auditable receipt

**Files:**
- No repository file changes unless verification exposes a defect.

- [ ] **Step 1: Confirm the selected environment and clean local state**

```bash
/opt/homebrew/bin/brev ls
git status --short --branch
git log -5 --oneline
```

Expected: `nvmolkit---nemotron-notebook-ea05c9` is running in the user-selected organization; the local branch contains only planned commits and no uncommitted source changes.

- [ ] **Step 2: Synchronize only the files changed by this plan**

Use `brev copy` for these exact files:

```text
objective_challenge.py
demo_agent.py
objective_receipts.py
interactive_workflow.py
tests/test_objective_challenge.py
tests/test_objective_agent_loop.py
tests/test_objective_receipts.py
tests/test_interactive_workflow.py
tests/test_gpu_acceptance.py
```

Destination: `/home/ubuntu/codex-objective-agent-challenge-20260806/` with test files placed under its `tests/` directory. Do not copy credentials, Jupyter runtime files, local visual-companion files, or notebook outputs.

- [ ] **Step 3: Run the complete GPU-enabled suite**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ea05c9 \
  "env -C /home/ubuntu/codex-objective-agent-challenge-20260806 RUN_GPU_TESTS=1 \
  /home/ubuntu/codex-objective-agent-challenge-20260806/.venv/bin/python -m pytest -q"
```

Expected: every test passes. The existing Matplotlib/PyParsing deprecation warnings are acceptable; failures are not.

- [ ] **Step 4: Execute one fresh hosted objective run without exposing credentials**

Use the registered `nvMolKit Objective (L4)` kernel or a credential-safe kernel attachment. Emit only this receipt shape:

```json
{
  "accepted_attempt_count": 3,
  "accepted_panels_unique": true,
  "scores_strictly_improve_after_miss": true,
  "target_score": 0.8312661528587342,
  "final_score": 0.8472222238779068,
  "termination_reason": "target_achieved",
  "objective_evidence_key": "O01",
  "workflow_status": "completed"
}
```

The exact attempt count may be one or two if Nemotron reaches the target earlier. Acceptance requires unique panels, strict improvement after a miss, `final_score >= target_score`, `target_achieved`, O01, and completed UI status. Never print the API key, full environment, request headers, or raw credential-bearing objects.

- [ ] **Step 5: Inspect the live decision ladder**

Confirm that every accepted row visibly contains:

```text
Observe
Agent action
Measure
Δ
Revise or Goal achieved
Validated Nemotron proposal
Validated intervention for revisions
Evaluation executed by Python
```

Also confirm the existing six stage cards, trajectory figure, four final structures, final heatmap, and Evidence-Backed Conclusion remain present.

- [ ] **Step 6: Run final repository verification**

```bash
python -m pytest -q
git diff --check
git status --short --branch
git log -6 --oneline
```

Expected: local tests pass with only the GPU test skipped when appropriate, formatting is clean, and `main` contains the planned commits with no unintended files.

## Completion Gate

Do not call the feature complete unless all of the following are true:

- the initial objective proposal remains a real Nemotron choice;
- every later accepted panel exactly matches a current legal improving swap;
- duplicate and unlisted proposals do not consume scientific attempts;
- the total correction budget is exactly two hosted responses;
- accepted scores strictly improve after every miss;
- the qualified L4 run reaches the fixed target within three accepted attempts;
- the decision ladder exposes observation, action, execution, delta, target, and outcome;
- O01 and the conclusion preserve the exact measured result;
- no molecule IDs or winning panels are hard-coded in production code; and
- the complete GPU-enabled suite has zero failures.
