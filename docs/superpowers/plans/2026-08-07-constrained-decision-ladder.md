# Constrained Decision Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free-form objective proposal and conclusion with a state-bound, evidence-optimal action-selection loop and an evidence-controlled conclusion that are impressive, scientifically truthful, and empirically reliable on the fixed Brev demo.

**Architecture:** Keep the six upstream nvMolKit/RDKit stages unchanged. Refactor the objective domain around one shared integer score key, complete co-limiting-pair records, immutable action menus, and exact reachability certification; let Nemotron select a state-bound argmax action while Python owns all calculations. Add a separate immutable finding catalog so the final model call chooses only predicate-true evidence emphasis and ordering, while persistent widgets render every factual statement and image from deterministic state.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, RDKit, Matplotlib, ipywidgets 8, OpenAI-compatible NVIDIA Nemotron API, nvMolKit 0.5.0, pytest, nbformat, Brev L4

---

## Starting State and File Map

Execute from the existing isolated worktree and branch containing approved spec commit `f2fbe08`. The interim objective-rationale commits on this branch are intentionally superseded; do not preserve `ObjectiveDecisionBasis` or objective `decision_basis` merely because they already exist.

- Modify `objective_challenge.py`: shared score semantics, co-limiting pairs, action menus, state IDs, reachability certificate, measured attempts, O01, and figures.
- Modify `demo_agent.py`: strict state-bound action schema, five counters, correction protocol, atomic execution, finding-selection schema, and conclusion fallback.
- Modify `objective_receipts.py`: validated selection, planned command, executed measurement, and evaluation-failure receipts.
- Create `objective_findings.py`: evidence parsing, measured summary, immutable predicate-true finding catalog, deterministic headlines, and validated selected emphasis.
- Modify `interactive_workflow.py`: baseline-first ladder, action menu, selection and measurement states, persistent figures, measured summary, and evidence-controlled conclusion.
- Modify `notebooks/nvmolkit_nemotron_demo.ipynb`: copy only; keep the eight-cell structure and one public launch call.
- Modify `README.md`: exact responsibility boundary and qualification commands.
- Create `scripts/run_objective_reliability.py`: reproducible 20-trial objective reliability receipt and three-run end-to-end mode.
- Modify `tests/test_objective_challenge.py`, `tests/test_objective_agent_loop.py`, `tests/test_objective_receipts.py`, `tests/test_interactive_workflow.py`, `tests/test_demo_agent.py`, `tests/test_notebook.py`, and `tests/test_gpu_acceptance.py`.
- Create `tests/test_objective_findings.py` and `tests/test_objective_reliability.py`.
- Create `tests/objective_fixtures.py`: shared synthetic objective contexts, optimized states, canonical evidence reports, terminal runs, and menus used by fresh task workers.

### Task 1: Establish shared score semantics and complete co-limiting pairs

**Files:**
- Modify: `objective_challenge.py`
- Create: `tests/objective_fixtures.py`
- Modify: `tests/test_objective_challenge.py`

- [ ] **Step 1: Write failing boundary and co-limiting-pair tests**

First move the existing `FakeTensor`, `FakeGpuResult`, `optimized_state`, and `two_revision_context` helpers from `tests/test_objective_challenge.py` into `tests/objective_fixtures.py`. Preserve their current implementations and imports. Add this general context constructor there after `measure_panel` exists in the RED-to-GREEN step:

```python
def context_from_distance(
    distance: np.ndarray,
    *,
    baseline_ids: tuple[str, str, str, str] = ("mol-0", "mol-1", "mol-2", "mol-3"),
    target_score: float = 0.75,
) -> ObjectiveContext:
    values = np.array(distance, dtype=float, copy=True)
    values.setflags(write=False)
    candidates = tuple(
        ObjectiveCandidate(f"mol-{index}", index, index, index)
        for index in range(CANDIDATE_COUNT)
    )
    provisional = ObjectiveContext(candidates, baseline_ids, 0.0, 0.0, target_score, values)
    baseline = measure_panel(provisional, baseline_ids).score
    benchmark = max(
        measure_panel(provisional, panel).score
        for panel in itertools.combinations(tuple(item.molecule_id for item in candidates), 4)
    )
    return replace(provisional, baseline_score=baseline, benchmark_score=benchmark)


def controlled_context(
    *,
    distances: dict[tuple[str, str], float],
    default_distance: float,
    target_score: float = 0.75,
) -> ObjectiveContext:
    matrix = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), default_distance, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    for (first_id, second_id), value in distances.items():
        first = int(first_id.removeprefix("mol-"))
        second = int(second_id.removeprefix("mol-"))
        matrix[first, second] = matrix[second, first] = value
    return context_from_distance(matrix, target_score=target_score)


BOUNDARY_CASES = (
    (0.5000000000004, 0.5, False),
    (0.5000000000005, 0.5, True),
    (0.5000000000006, 0.5, True),
)
```

Import these shared helpers from `tests.objective_fixtures` in every later task instead of inventing local context/report/menu factories.

Add focused tests using synthetic `ObjectiveContext` matrices:

```python
def test_score_key_uses_one_trillion_half_up_units():
    assert score_key(0.5000000000004) == 500_000_000_000
    assert score_key(0.5000000000005) == 500_000_000_001
    assert score_key(np.float32(0.5)) == 500_000_000_000
    with pytest.raises(ValueError):
        score_key(True)


def test_measure_panel_retains_every_canonical_co_limiting_pair():
    context = controlled_context(
        distances={
            ("mol-0", "mol-1"): 0.4,
            ("mol-2", "mol-3"): 0.4,
        },
        default_distance=0.8,
    )
    measurement = measure_panel(context, ("mol-3", "mol-2", "mol-1", "mol-0"))
    assert measurement.score_key == score_key(0.4)
    assert measurement.limiting_pairs == (
        ("mol-0", "mol-1"),
        ("mol-2", "mol-3"),
    )


@pytest.mark.parametrize(("candidate", "current", "expected"), BOUNDARY_CASES)
def test_improvement_uses_score_keys(candidate, current, expected):
    assert is_strict_improvement(candidate, current) is expected


@pytest.mark.parametrize(
    ("score", "target", "expected"),
    BOUNDARY_CASES,
)
def test_target_attainment_uses_the_same_score_key(score, target, expected):
    assert target_is_achieved(score, target) is expected


```

Reuse the existing `optimized_state()` fixture. Task 1 deliberately does not define action menus, objective attempts, or terminal runs; those are introduced together in Task 2 so this task can be executed and committed independently.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_challenge.py -k 'score_key or co_limiting or improvement'
```

Expected: failures because `score_key`, `measure_panel`, `PanelMeasurement`, and `limiting_pairs` do not exist.

- [ ] **Step 3: Implement the shared domain types and comparator**

Replace the singular-pair and tolerance-only representation with these interfaces:

```python
SCORE_SCALE = 10**12


def score_key(value: float | np.floating) -> int:
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        raise ValueError("Objective score must be a finite float in [0, 1].")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("Objective score must be a finite float in [0, 1].")
    return int(np.floor(normalized * SCORE_SCALE + 0.5))


def is_strict_improvement(candidate: float, current: float) -> bool:
    return score_key(candidate) > score_key(current)


def target_is_achieved(score: float, target: float) -> bool:
    return score_key(score) >= score_key(target)


@dataclass(frozen=True)
class PanelMeasurement:
    selected_ids: tuple[str, ...]
    score: float
    score_key: int
    limiting_pairs: tuple[tuple[str, str], ...]
    achieved: bool


def measure_panel(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...] | list[str],
) -> PanelMeasurement:
    panel, _ = _validated_panel(context, selected_ids)
    scored = tuple(
        (score_key(float(distance)), float(distance), tuple(sorted((first_id, second_id))))
        for first_id, second_id, distance in _panel_distances(context, panel)
    )
    minimum_key = min(item[0] for item in scored)
    limiting_pairs = tuple(sorted(item[2] for item in scored if item[0] == minimum_key))
    raw_score = min(item[1] for item in scored if item[0] == minimum_key)
    return PanelMeasurement(
        selected_ids=panel,
        score=raw_score,
        score_key=minimum_key,
        limiting_pairs=limiting_pairs,
        achieved=target_is_achieved(raw_score, context.target_score),
    )
```

Define `_panel_distances(context, panel)` in `objective_challenge.py` as the single iterator over `itertools.combinations(panel, 2)`. It resolves candidate positions, converts every NumPy scalar to a built-in float, validates finiteness/range, and yields `(first_id, second_id, distance)`.

Update the existing baseline and panel-scoring paths to use `PanelMeasurement`, score keys, and `limiting_pairs`. Do not add any hosted-agent schema or terminal-state behavior in this task.

- [ ] **Step 4: Run domain tests and verify GREEN**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_challenge.py
```

Expected: all objective-domain tests pass, including values immediately below, on, and above score-key boundaries.

- [ ] **Step 5: Commit Task 1**

```bash
git add objective_challenge.py tests/objective_fixtures.py tests/test_objective_challenge.py
git commit -m "Refactor objective score truth"
```

### Task 2: Build immutable action menus, state IDs, and reachability certification

**Files:**
- Modify: `objective_challenge.py`
- Modify: `tests/test_objective_challenge.py`
- Modify: `tests/test_gpu_acceptance.py`

- [ ] **Step 1: Write failing action-menu and reachability tests**

```python
def test_action_menu_contains_top_three_then_displays_by_stable_id():
    context = controlled_context_with_ranked_swaps()
    baseline = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, baseline, accepted_attempt_count=0)
    assert len(menu.actions) == 3
    assert tuple(action.swap_id for action in menu.actions) == tuple(
        sorted(action.swap_id for action in menu.actions)
    )
    all_legal = enumerate_legal_swaps(context, baseline)
    expected_keys = sorted(
        (action.predicted_score_key for action in all_legal), reverse=True
    )[:3]
    assert sorted(
        (action.predicted_score_key for action in menu.actions), reverse=True
    ) == expected_keys
    assert {action.predicted_score_key for action in accepted_maxima(menu)} == {
        max(action.predicted_score_key for action in menu.actions)
    }


def test_every_offered_action_preserves_all_scientific_constraints():
    context = controlled_context_with_ranked_swaps()
    source = measure_panel(context, context.baseline_ids)
    candidates = {item.molecule_id: item for item in context.candidates}
    for action in build_action_menu(context, source, accepted_attempt_count=0).actions:
        assert len(action.resulting_ids) == len(set(action.resulting_ids)) == 4
        assert set(action.resulting_ids) <= set(candidates)
        assert len({candidates[item].cluster_id for item in action.resulting_ids}) == 4
        assert len(set(source.selected_ids) - set(action.resulting_ids)) == 1
        assert len(set(action.resulting_ids) - set(source.selected_ids)) == 1
        assert action.predicted_score_key > source.score_key
        assert all(
            action.replace_id in pair for pair in source.limiting_pairs
        )


def test_state_id_changes_with_source_state_or_offered_actions():
    context = controlled_context_with_ranked_swaps()
    baseline = measure_panel(context, context.baseline_ids)
    first = build_action_menu(context, baseline, accepted_attempt_count=0)
    next_source = measure_panel(context, first.actions[0].resulting_ids)
    second = build_action_menu(context, next_source, accepted_attempt_count=1)
    assert first.state_id != second.state_id
    assert first.state_id == build_action_menu(
        context, baseline, accepted_attempt_count=0
    ).state_id


def test_reachability_branches_over_every_offered_maximum():
    assert certify_argmax_reachability(
        controlled_context_with_tied_paths(all_paths_reach=True)
    ) is True
    assert certify_argmax_reachability(
        controlled_context_with_tied_paths(all_paths_reach=False)
    ) is False


def test_empty_menu_has_no_maxima_and_finalizes_truthfully():
    context = controlled_context_without_improving_swaps()
    current = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, current, accepted_attempt_count=0)
    assert menu.actions == ()
    assert accepted_maxima(menu) == ()
    run = finalize_no_legal_swap(context, (), current, menu)
    assert run.termination_reason == "no_legal_improving_swap"


def test_baseline_optimal_short_circuits_before_action_menu():
    context = build_objective_context(optimized_state(baseline_optimal=True))
    run = baseline_terminal_run(context)
    assert run.termination_reason == "baseline_already_optimal"
    assert run.attempts == ()


@pytest.mark.parametrize(("candidate", "current", "expected"), BOUNDARY_CASES)
def test_action_inclusion_maximality_ordering_and_reachability_share_boundary(
    candidate, current, expected
):
    context, distinguished_swap_id = boundary_policy_context(candidate, current)
    source = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, accepted_attempt_count=0)
    included = {item.swap_id for item in menu.actions}
    assert (distinguished_swap_id in included) is expected
    if expected:
        assert distinguished_swap_id in {
            item.swap_id for item in accepted_maxima(menu)
        }
        ranked = sorted(
            menu.actions, key=lambda item: (-item.predicted_score_key, item.swap_id)
        )
        assert ranked[0].swap_id == distinguished_swap_id
    assert certify_argmax_reachability(context) is expected


@pytest.mark.parametrize(
    ("reason", "achieved", "attempt_count"),
    [
        ("baseline_already_optimal", True, 0),
        ("objective_correction_limit", False, 0),
        ("objective_provider_failure", False, 0),
        ("evaluation_not_completed", False, 0),
        ("attempt_limit_reached", False, 3),
        ("target_achieved", True, 2),
    ],
)
def test_terminal_run_supports_every_objective_outcome(reason, achieved, attempt_count):
    context, attempts, current = terminal_fixture(reason, attempt_count)
    run = terminal_objective_run(context, attempts, current, reason)
    assert run.termination_reason == reason
    assert run.achieved is achieved
    assert len(run.attempts) == attempt_count
    payload = json.loads(build_objective_evidence(run).payload_json)
    assert payload["termination_reason"] == reason
    assert payload["attempt_count"] == attempt_count
```

Add `controlled_context_with_ranked_swaps()`, `controlled_context_with_tied_paths(all_paths_reach: bool)`, `controlled_context_without_improving_swaps()`, `controlled_context_with_action_count(action_count)`, and `boundary_policy_context(candidate, current)` beside the existing `two_revision_context()` helper. Their explicit distance matrices must create, respectively, four distinct improving swaps with known descending score keys, two equal-key first actions whose second branches either both reach or one misses by step three, a below-target panel with no strictly improving one-swap action, exactly zero through three legal improving swaps, and one distinguished substitution at the supplied boundary against a source score of `current`. In `boundary_policy_context`, all other legal swaps stay at or below the source key. When the candidate key improves, set the target to `candidate`; when it does not, set it to `(score_key(current) + 1) / SCORE_SCALE`. Inclusion, maximality, ordering, target attainment, and certification then all have the expected shared outcome. Extend the existing `optimized_state` helper with `baseline_optimal: bool = False`. Do not mock `build_action_menu`; these tests must exercise production enumeration and comparison.

The production context builder continues to admit only candidates returned by `eligible_representative_groups`; add an assertion to the existing context-construction test that every candidate ID belongs to that MMFF94-eligible source set. The synthetic constraint test above then verifies the remaining legality and provenance invariants on the already eligible bounded pool.

Add `terminal_fixture(reason, attempt_count)` to `tests/objective_fixtures.py`. Give each requested outcome its own valid synthetic context. For measured outcomes, repeatedly take the first action from `accepted_maxima(build_action_menu(...))`, evaluate it into a real `ObjectiveAttempt`, and return exactly `attempt_count` committed attempts plus the last `PanelMeasurement`. For zero-attempt failures, return the baseline measurement. Never mutate `achieved`, target values, scores, or attempts after construction merely to satisfy the expected reason.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_challenge.py -k 'action_menu or state_id or reachability or empty_menu or baseline_optimal or terminal_run or boundary'
```

Expected: failures because the menu, state revision, and certificate interfaces are absent.

- [ ] **Step 3: Implement canonical actions and exact certification**

Use immutable interfaces:

```python
DECISION_RULE = "maximize_predicted_minimum_distance"


@dataclass(frozen=True)
class ObjectiveSwap:
    swap_id: str
    replace_id: str
    replacement_id: str
    resulting_ids: tuple[str, ...]
    predicted_score: float
    predicted_score_key: int
    score_delta: float
    limiting_pairs: tuple[tuple[str, str], ...]
    target_status: Literal["below_target", "meets_target"]


@dataclass(frozen=True)
class ObjectiveActionMenu:
    state_id: str
    source: PanelMeasurement
    accepted_attempt_count: int
    actions: tuple[ObjectiveSwap, ...]


def accepted_maxima(menu: ObjectiveActionMenu) -> tuple[ObjectiveSwap, ...]:
    if not menu.actions:
        return ()
    maximum = max(action.predicted_score_key for action in menu.actions)
    return tuple(action for action in menu.actions if action.predicted_score_key == maximum)
```

Add the persisted terminal domain in this task, after `ObjectiveAttempt` and action menus exist:

```python
TerminationReason = Literal[
    "target_achieved",
    "baseline_already_optimal",
    "attempt_limit_reached",
    "no_legal_improving_swap",
    "objective_correction_limit",
    "objective_provider_failure",
    "evaluation_not_completed",
]


@dataclass(frozen=True)
class ObjectiveRun:
    context: ObjectiveContext
    baseline: PanelMeasurement
    attempts: tuple[ObjectiveAttempt, ...]
    achieved: bool
    termination_reason: TerminationReason
    final_ids: tuple[str, ...]
    final_score: float
    final_score_key: int


def terminal_objective_run(
    context: ObjectiveContext,
    attempts: tuple[ObjectiveAttempt, ...],
    current: PanelMeasurement,
    reason: TerminationReason,
) -> ObjectiveRun:
    baseline = measure_panel(context, context.baseline_ids)
    numbers = tuple(item.attempt_number for item in attempts)
    if numbers != tuple(range(1, len(attempts) + 1)):
        raise ValueError("Objective attempts must be sequential.")
    expected_current = baseline if not attempts else attempts[-1].measurement
    if current != expected_current:
        raise ValueError("Objective terminal state must use the last measured panel.")
    if reason == "target_achieved" and (not attempts or not current.achieved):
        raise ValueError("Target success requires a measured successful attempt.")
    if reason == "baseline_already_optimal" and (
        attempts or baseline.score_key != score_key(context.benchmark_score)
    ):
        raise ValueError("Baseline-optimal termination is inconsistent.")
    if reason == "attempt_limit_reached" and (
        len(attempts) != MAX_ATTEMPTS or current.achieved
    ):
        raise ValueError("Attempt-limit termination requires three measured misses.")
    if reason in {
        "objective_correction_limit",
        "objective_provider_failure",
        "evaluation_not_completed",
    } and (len(attempts) >= MAX_ATTEMPTS or current.achieved):
        raise ValueError("Failure termination cannot follow completed success.")
    if reason == "no_legal_improving_swap" and current.achieved:
        raise ValueError("No-swap termination requires a below-target measured panel.")
    return ObjectiveRun(
        context=context,
        baseline=baseline,
        attempts=attempts,
        achieved=reason in {"target_achieved", "baseline_already_optimal"},
        termination_reason=reason,
        final_ids=current.selected_ids,
        final_score=current.score,
        final_score_key=current.score_key,
    )
```

`ObjectiveAttempt` exposes a `measurement: PanelMeasurement` property. Update `ObjectiveSwap`, `ObjectiveAttempt`, `ObjectiveRun`, finalizers, O01, and figure construction to use score keys and `limiting_pairs`; remove objective `decision_basis`. `build_objective_evidence(run)` must exist for every terminal reason and include the baseline, committed attempts, final measured state, achieved flag, exact reason, and attempt count—including zero-attempt provider/correction/evaluation failures.

Add the schema-independent domain evaluator here so Task 2's terminal fixtures and every later controller task use one implementation:

```python
def evaluate_selected_swap(
    context: ObjectiveContext,
    menu: ObjectiveActionMenu,
    action: ObjectiveSwap,
    *,
    attempt_number: int,
) -> ObjectiveAttempt:
    if action not in menu.actions:
        raise ValueError("Selected action is not from the current menu.")
    if action not in accepted_maxima(menu):
        raise ValueError("Selected action is not maximal in the current menu.")
    measurement = measure_panel(context, action.resulting_ids)
    return ObjectiveAttempt(
        attempt_number=attempt_number,
        state_id=menu.state_id,
        selected_ids=measurement.selected_ids,
        score=measurement.score,
        score_key=measurement.score_key,
        limiting_pairs=measurement.limiting_pairs,
        constraints_passed=True,
        achieved=measurement.achieved,
        selected_swap=action,
    )
```

Generate `swap_id` as `f"{replace_id}->{replacement_id}"`. Enumerate all cluster-valid one-ID replacements; retain only `predicted_score_key > source.score_key`; assert each retained action removes at least one molecule from every source co-limiting pair. Sort for inclusion by `(-predicted_score_key, swap_id)`, take three, then sort the displayed tuple by `swap_id`.

Expose that full enumeration as the pure `enumerate_legal_swaps(context, source)` function used by `build_action_menu` and the boundary-inclusion test; it returns every improving legal action before the three-action cap, ordered by `(-predicted_score_key, swap_id)`. The certificate must call `build_action_menu` rather than this uncapped helper so it certifies the exact choices the model can see.

Compute `state_id` as `"state-" + sha256(canonical_json).hexdigest()[:16]`, where canonical JSON contains source IDs, source score key, co-limiting pairs, accepted-attempt count, and displayed action IDs. `certify_argmax_reachability(context)` must call `build_action_menu` and `accepted_maxima`, branch over every accepted maximum, and require every branch to achieve within `MAX_ATTEMPTS`.

Call the certificate from `build_objective_context`; raise `RuntimeError("Objective target is not reachable under the bounded decision policy.")` before any hosted objective call when false.

Add `baseline_terminal_run(context)` and `finalize_no_legal_swap(context, attempts, current, menu)`. The latter requires `menu.source == current`, `menu.actions == ()`, and a below-target current measurement before delegating to `terminal_objective_run(..., "no_legal_improving_swap")`.

- [ ] **Step 4: Run objective and GPU source-gate tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_challenge.py tests/test_gpu_acceptance.py
```

Expected: all CPU tests pass; the real GPU test remains skipped without `RUN_GPU_TESTS=1`.

- [ ] **Step 5: Commit Task 2**

```bash
git add objective_challenge.py tests/test_objective_challenge.py tests/test_gpu_acceptance.py
git commit -m "Add certified objective action menus"
```

### Task 3: Replace free-form proposals with the bounded hosted selection state machine

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_objective_agent_loop.py`

**Fixture contract:** Keep the existing `FakeCompletions`, `response`, `optimized_state`, `full_report`, and `completed_controller` helpers in `tests/test_objective_agent_loop.py`. Change `FakeCompletions.create` so an `Exception` item is raised. Replace the old `proposal(...)` helper with:

```python
def initial_menu(*, baseline_optimal=False):
    context = build_objective_context(optimized_state(baseline_optimal=baseline_optimal))
    baseline = measure_panel(context, context.baseline_ids)
    return build_action_menu(context, baseline, accepted_attempt_count=0)


def selection(menu, index=0, **overrides):
    arguments = {
        "state_id": menu.state_id,
        "swap_id": menu.actions[index].swap_id,
        "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
        "decision_rule": "maximize_predicted_minimum_distance",
    }
    arguments.update(overrides)
    return response("select_next_panel_swap", arguments)
```

Import `httpx`, `build_action_menu`, `build_objective_context`, and `measure_panel`. Every test below must build its response from the exact menu fixture; do not hard-code a valid state hash.

- [ ] **Step 1: Write failing schema, stale-state, counter, and protocol tests**

Define test helpers that return exact current `state_id`, `swap_id`, and co-limiting pairs. Add:

```python
def test_objective_selection_schema_contains_no_free_text_or_numbers():
    fields = tuple(demo_agent.ObjectiveSelection.model_fields)
    assert fields == (
        "state_id",
        "swap_id",
        "observed_limiting_pairs",
        "decision_rule",
    )
    assert "decision_basis" not in demo_agent.ObjectiveSelection.model_json_schema()["properties"]


def test_stale_state_id_is_rejected_without_scientific_attempt():
    menu = initial_menu()
    controller, _ = completed_controller([
        selection(menu, state_id="state-0000000000000000"),
        selection(menu, state_id="state-0000000000000000"),
    ])
    controller.begin_objective_challenge()
    with pytest.raises(ObjectiveCorrectionLimitError):
        controller.request_objective_selection()
    assert controller.accepted_attempt_count == 0
    assert controller.rejected_selection_count == 2


def test_invalid_invalid_stops_without_third_provider_request():
    menu = initial_menu()
    controller, completions = completed_controller([
        selection(menu, swap_id="unavailable"),
        selection(menu, swap_id="unavailable"),
    ])
    controller.begin_objective_challenge()
    with pytest.raises(ObjectiveCorrectionLimitError):
        controller.request_objective_selection()
    assert controller.correction_prompts_sent == 1
    assert controller.selection_response_count == 2
    assert controller.provider_request_attempt_count == 2
    assert len(completions.calls) == 2


def test_retry_prompt_replays_only_the_exact_current_decision_state():
    menu = initial_menu()
    controller, _ = completed_controller([
        selection(menu, swap_id="unavailable"),
        selection(menu),
    ])
    controller.begin_objective_challenge()
    accepted = controller.request_objective_selection()
    assert accepted.swap_id in {item.swap_id for item in accepted_maxima(menu)}
    retry_messages = [
        item for item in controller.session.messages
        if item.get("role") == "user"
        and item.get("content", "").startswith('{"candidate_actions":')
    ]
    assert len(retry_messages) == 1
    payload = json.loads(retry_messages[0]["content"])
    assert tuple(payload) == (
        "candidate_actions",
        "current_limiting_pairs",
        "decision_rule",
        "remaining_rejections",
    )
    assert payload["candidate_actions"] == action_table_payload(menu)
    assert payload["current_limiting_pairs"] == [
        list(pair) for pair in menu.source.limiting_pairs
    ]
    assert payload["decision_rule"] == "maximize_predicted_minimum_distance"
    assert payload["remaining_rejections"] == 1
    stopped, _ = completed_controller([
        selection(menu, swap_id="unavailable"),
        selection(menu, swap_id="unavailable"),
    ])
    stopped.begin_objective_challenge()
    with pytest.raises(ObjectiveCorrectionLimitError):
        stopped.request_objective_selection()
    assert len([
        item for item in stopped.session.messages
        if item.get("role") == "user"
        and item.get("content", "").startswith('{"candidate_actions":')
    ]) == 1


def test_transport_retry_has_separate_request_attempt_accounting():
    menu = initial_menu()
    controller, _ = completed_controller([
        httpx.ConnectError("offline"),
        selection(menu),
    ])
    controller.begin_objective_challenge()
    with pytest.raises(ToolCallError, match="hosted Nemotron request failed"):
        controller.request_objective_selection()
    proposal = controller.request_objective_selection(is_transport_retry=True)
    assert proposal.swap_id in {item.swap_id for item in controller.pending_action_menu.actions}
    assert controller.provider_request_attempt_count == 2
    assert controller.selection_response_count == 1


def test_second_transport_failure_stops_as_objective_provider_failure():
    controller, _ = completed_controller([
        httpx.ConnectError("offline-1"),
        httpx.ConnectError("offline-2"),
    ])
    controller.begin_objective_challenge()
    with pytest.raises(ToolCallError):
        controller.request_objective_selection()
    with pytest.raises(ToolCallError):
        controller.request_objective_selection(is_transport_retry=True)
    assert controller.objective_failure_reason == "objective_provider_failure"
    assert controller.provider_request_attempt_count == 2
    assert controller.selection_response_count == 0


def test_baseline_optimal_makes_no_hosted_objective_request():
    controller, completions = completed_controller([], baseline_optimal=True)
    controller.begin_objective_challenge()
    assert controller.objective_run.termination_reason == "baseline_already_optimal"
    assert completions.calls == []


def test_unreachable_policy_uses_specific_eligibility_error(monkeypatch):
    controller, completions = completed_controller([])
    monkeypatch.setattr(demo_agent, "build_objective_context", Mock(
        side_effect=RuntimeError("Objective target is not reachable under the bounded decision policy.")
    ))
    with pytest.raises(ObjectiveEligibilityError, match="not reachable"):
        controller.begin_objective_challenge()
    assert completions.calls == []
```

Add a parameterized transition table covering exact state effects:

```python
@pytest.mark.parametrize(
    ("events", "accepted", "rejected", "prompts", "responses", "requests", "reason"),
    [
        (("valid",), 1, 0, 0, 1, 1, None),
        (("nonmax", "valid"), 1, 1, 1, 2, 2, None),
        (("wrong_pair", "valid"), 1, 1, 1, 2, 2, None),
        (("wrong_tool", "valid"), 1, 1, 1, 2, 2, None),
        (("malformed", "valid"), 1, 1, 1, 2, 2, None),
        (("valid", "invalid", "invalid"), 1, 2, 1, 3, 3, "objective_correction_limit"),
        (("invalid", "valid", "invalid"), 1, 2, 1, 3, 3, "objective_correction_limit"),
        (("invalid", "invalid"), 0, 2, 1, 2, 2, "objective_correction_limit"),
        (("transport", "valid"), 1, 0, 0, 1, 2, None),
        (("transport", "transport"), 0, 0, 0, 0, 2, "objective_provider_failure"),
    ],
)
def test_objective_transition_table(events, accepted, rejected, prompts, responses, requests, reason):
    controller = run_transition_events(events)
    assert controller.accepted_attempt_count == accepted
    assert controller.rejected_selection_count == rejected
    assert controller.correction_prompts_sent == prompts
    assert controller.selection_response_count == responses
    assert controller.provider_request_attempt_count == requests
    assert (None if controller.objective_run is None else controller.objective_run.termination_reason) == reason
    assert_assistant_tool_pairing(controller.session.messages)
```

Implement `run_transition_events` in the test file using a nonterminal three-step synthetic context and translating each event to the current menu's exact response: `nonmax` chooses the lowest offered score key, `wrong_pair` replaces the exact pair enum, `wrong_tool` changes the function name, `malformed` supplies invalid JSON, `transport` raises `httpx.ConnectError`, and `valid` selects the first `accepted_maxima`. If the sequence contains an accepted event, call both request and execute before the next event. `assert_assistant_tool_pairing` compares every assistant tool-call ID with the following tool message ID. For `valid -> invalid -> invalid`, additionally assert the one committed measurement and O01 attempt remain present after terminalization; for `invalid -> valid -> invalid`, assert the second rejection terminalized the run rather than resetting after the accepted measurement. Add separate tests that pre-set each counter to its maximum and prove the next request/transition is locally blocked without calling the provider.

- [ ] **Step 2: Run focused state-machine tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_demo_agent.py tests/test_objective_agent_loop.py -k 'objective_selection or stale_state or invalid_invalid or transport_retry'
```

Expected: failures because the old `ObjectiveProposal` and `decision_basis` protocol remain.

- [ ] **Step 3: Implement the strict selection model and dynamic schema**

Replace `ObjectiveDecisionBasis` and `ObjectiveProposal` with:

```python
DecisionRule = Literal["maximize_predicted_minimum_distance"]
MoleculeSwapId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=170, pattern=r"^[^\s\r\n`]+->[^\s\r\n`]+$"),
]


class ObjectiveSelection(_StrictModel):
    state_id: Annotated[str, StringConstraints(pattern=r"^state-[0-9a-f]{16}$")]
    swap_id: MoleculeSwapId
    observed_limiting_pairs: list[list[MoleculeId]] = Field(min_length=1, max_length=6)
    decision_rule: DecisionRule
```

Extend `_tool_definition(..., objective_menu: ObjectiveActionMenu | None = None)` so `state_id` has a one-value enum, `swap_id` enumerates only the current action IDs, `observed_limiting_pairs` has one exact array enum, and `decision_rule` has one literal. Rename the tool to `select_next_panel_swap` and disable the generic text-only recursive retries for this tool so every assistant response is counted explicitly.

Add controller fields exactly matching the specification:

```python
pending_action_menu: ObjectiveActionMenu | None = None
pending_objective_selection: ObjectiveSelection | None = None
accepted_attempt_count: int = 0
rejected_selection_count: int = 0
correction_prompts_sent: int = 0
selection_response_count: int = 0
provider_request_attempt_count: int = 0
objective_transport_retry_used: bool = False
objective_failure_reason: str | None = None
```

`begin_objective_challenge()` measures the deterministic baseline and builds the first menu. `request_objective_selection()` checks all five counters before every request, increments provider attempts before calling the client, increments response count only after an assistant response, validates the exact pending revision and argmax, pairs every rejected call, sends only one correction prompt, and stops on the second rejection. The correction content is canonical JSON with exactly the four lexicographically ordered keys asserted above; `action_table_payload(menu)` contains only the same deterministically rendered action rows already shown to the model. State binding remains in the unchanged pending tool schema. The prompt includes no separate state field, rejection rationale, new ranking cue, model prose, or numerical field absent from the current menu. Do not reset rejection counts after accepted attempts.

Map the exact reachability `RuntimeError` to `ObjectiveEligibilityError` without replacing its specific safe text. Short-circuit baseline optimal before building a menu. On correction/provider terminal states, construct `ObjectiveRun` and O01 immediately via `terminal_objective_run`; conclusion gating therefore sees an immutable terminal result even with zero measured attempts.

- [ ] **Step 4: Run controller tests and verify GREEN**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_demo_agent.py tests/test_objective_agent_loop.py
```

Expected: all controller and protocol tests pass with no objective free-form rationale fields.

- [ ] **Step 5: Commit Task 3**

```bash
git add demo_agent.py tests/test_demo_agent.py tests/test_objective_agent_loop.py
git commit -m "Constrain hosted objective action selection"
```

### Task 4: Make execution atomic and bind deterministic receipts

**Files:**
- Modify: `objective_challenge.py`
- Modify: `demo_agent.py`
- Modify: `objective_receipts.py`
- Modify: `tests/test_objective_challenge.py`
- Modify: `tests/test_objective_agent_loop.py`
- Modify: `tests/test_objective_receipts.py`

**Fixture contract:** Reuse Task 3's `completed_controller`, `initial_menu`, and `selection` helpers. Add this fail-once list to `tests/test_objective_agent_loop.py`:

```python
class FailOnceList(list):
    def __init__(self, values, *, fail_on_append=False):
        super().__init__(values)
        self.fail_on_append = fail_on_append

    def append(self, value):
        if self.fail_on_append:
            self.fail_on_append = False
            raise RuntimeError("injected append failure")
        super().append(value)
```

- [ ] **Step 1: Write failing atomicity and receipt tests**

```python
def test_evaluator_failure_pairs_error_without_measured_attempt(monkeypatch):
    controller = controller_with_pending_selection()
    monkeypatch.setattr(demo_agent, "evaluate_selected_swap", Mock(side_effect=RuntimeError()))
    with pytest.raises(ObjectiveEvaluationError):
        controller.execute_objective_selection(controller.pending_objective_selection)
    assert controller.accepted_attempt_count == 0
    assert controller.objective_attempts == []
    assert controller.objective_failure_reason == "evaluation_not_completed"
    assert controller.session.messages[-1]["role"] == "tool"
    assert json.loads(controller.session.messages[-1]["content"])["accepted"] is False


@pytest.mark.parametrize(
    ("failure_point", "target_achieving"),
    [
        ("next_menu", False),
        ("prospective_o01", True),
        ("success_serialization", False),
        ("message_append", False),
        ("commit_invariant", False),
    ],
)
def test_transition_failure_rolls_back_then_commits_terminal_error(
    monkeypatch, failure_point, target_achieving
):
    controller = controller_with_pending_selection(target_achieving=target_achieving)
    before = objective_snapshot(controller)
    inject_transition_failure(monkeypatch, controller, failure_point)
    with pytest.raises(ObjectiveEvaluationError):
        controller.execute_objective_selection(controller.pending_objective_selection)
    assert controller.accepted_attempt_count == before.accepted_attempt_count
    assert tuple(controller.objective_attempts) == before.attempts
    assert controller.objective_run.termination_reason == "evaluation_not_completed"
    assert controller.objective_evidence.key == "O01"
    assert json.loads(controller.session.messages[-1]["content"])["reason"] == "evaluation_not_completed"
    assert_assistant_tool_pairing(controller.session.messages)


def test_receipt_distinguishes_validated_planned_and_measured_states():
    context = build_objective_context(optimized_state())
    menu = build_action_menu(
        context, measure_panel(context, context.baseline_ids), accepted_attempt_count=0
    )
    action = accepted_maxima(menu)[0]
    choice = ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=action.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )
    planned = objective_receipt(choice, menu, measurement=None)
    assert planned.status == "validated_selection"
    assert "select_next_panel_swap" in planned.planned_command
    attempt = evaluate_selected_swap(context, menu, action, attempt_number=1)
    measured = objective_receipt(choice, menu, measurement=attempt)
    assert measured.status == "measured"
    assert measured.executed_measurement is not None
    assert "decision_basis" not in repr(measured)
```

Implement `controller_with_pending_selection(target_achieving=False)` with two purpose-built valid contexts. The false branch's accepted maximum remains below target and therefore must build a next menu; the true branch's accepted maximum reaches target and therefore must build prospective O01 without constructing a next menu. Create the controller with a response built from that exact context's initial menu, call `begin_objective_challenge()`, and call `request_objective_selection()`. `objective_snapshot` returns a frozen test tuple containing deep-copied plain-list message values, counters, attempts, menu, selection, run, and evidence. `inject_transition_failure` patches, respectively, `build_action_menu` only for the nonterminal branch, the first call to `build_objective_evidence` only for the terminal branch, `_serialize`, `session.messages` with `FailOnceList`, or `_validate_objective_commit`; after the injected failure, evidence construction and error-message append must be allowed to succeed. Each injection helper asserts its patched call was reached exactly once so a test cannot pass without exercising its named path.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_agent_loop.py tests/test_objective_receipts.py -k 'evaluator_failure or receipt_distinguishes or atomic'
```

Expected: failures because execution mutates the old attempt model and receipts still render `select_diverse_panel`.

- [ ] **Step 3: Implement prospective evaluation and guarded commit**

Reuse Task 2's schema-independent `evaluate_selected_swap`; it accepts a validated domain action and never a Pydantic agent model. Define `ObjectiveEvaluationError(ToolCallError)` in `demo_agent.py`. `execute_objective_selection` resolves `selection.swap_id` to the exact menu action, independently revalidates state, exact co-limiting pairs, rule, and membership in `accepted_maxima(menu)`, then passes that domain action to `evaluate_selected_swap`.

Create a frozen `_ObjectiveTransition` containing the complete prospective attempts tuple, next menu, run, O01, counters, pending fields, and serialized tool message. Create `_ObjectiveCommitSnapshot.capture(controller)` and `.restore(controller)` using `tuple(copy.deepcopy(list(controller.session.messages)))` for persisted message values and exact copies for every objective field/counter. Restore messages with `controller.session.messages = list(copy.deepcopy(snapshot.messages))`; do not preserve an injected list subclass or its failure flag. `_commit_objective_transition(controller, transition)` must capture, apply, append the tool result, call `_validate_objective_commit`, and on any exception restore the complete snapshot before re-raising. The `message_append` test must assert that the injected success append fails once and that the terminal error append on the restored ordinary list succeeds.

In `execute_objective_selection`, compute and validate the attempt, next menu, terminal run, O01, and serialized success tool result before calling the commit helper. On any evaluation, prospective-menu, evidence, serialization, message-append, or invariant failure, verify the snapshot was restored; then create `terminal_objective_run(..., "evaluation_not_completed")`, build its O01, append the paired error tool result, clear pending state, and raise `ObjectiveEvaluationError`. A render failure occurs after commit and must never call execution again.

Replace `ObjectiveReceipt` with frozen fields `status`, `validated_selection`, `planned_command`, `python_evaluation`, and optional `executed_measurement`. Validate exact menu revision, selected action, source measurement, and resulting attempt before rendering stable code templates.

- [ ] **Step 4: Run protocol and receipt suites**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_challenge.py tests/test_objective_agent_loop.py tests/test_objective_receipts.py
```

Expected: all tests pass; every hosted assistant selection has exactly one paired success, rejection, or evaluation-error tool result.

- [ ] **Step 5: Commit Task 4**

```bash
git add objective_challenge.py demo_agent.py objective_receipts.py tests/test_objective_challenge.py tests/test_objective_agent_loop.py tests/test_objective_receipts.py
git commit -m "Make objective execution atomic and auditable"
```

### Task 5: Render the persistent bounded decision ladder

**Files:**
- Modify: `interactive_workflow.py`
- Modify: `tests/test_interactive_workflow.py`

**Fixture contract:** Migrate the existing local `Controller` fake in `tests/test_interactive_workflow.py` from `ObjectiveProposal` to the Task 2/3 public types. Build one synthetic context from `tests.objective_fixtures.two_revision_context()`, set `baseline = measure_panel(context, context.baseline_ids)`, and set `pending_action_menu = build_action_menu(context, baseline, 0)`. Implement fake `request_objective_selection` by returning an `ObjectiveSelection` for `accepted_maxima(pending_action_menu)[0]`; implement fake `execute_objective_selection` by calling the real `evaluate_selected_swap` and updating the next menu/run. Add `ready_interactive_workflow()` that only instantiates and returns `InteractiveWorkflow(Controller())`; the test itself calls `_show_objective_challenge()` exactly once. Add `production_menu(action_count)` by calling `controlled_context_with_action_count(action_count)`, measuring its baseline, and invoking `build_action_menu`; assert its action count and recomputed `state_id` are valid rather than truncating a hashed menu. Add `completed_objective_workflow()` that calls `_continue_objective_challenge()`, and `combined_html(widget)` that recursively joins every `widgets.HTML.value`. Add `evaluation_failed_workflow()` by configuring that same fake so execution records `terminal_objective_run(context, (), baseline, "evaluation_not_completed")`, clears pending state, raises `ObjectiveEvaluationError`, and then driving `_continue_objective_challenge()` once.

- [ ] **Step 1: Write failing baseline, menu, tie, failure, and persistence tests**

```python
def test_objective_card_renders_baseline_before_agent_attempts():
    workflow = ready_interactive_workflow()
    workflow._show_objective_challenge()
    html = workflow.objective_summary.value
    assert "Step 0" in html
    assert "Measured baseline" in html
    assert "Attempt 1" not in html


def test_attempt_card_shows_menu_selection_command_and_measurement():
    workflow = completed_objective_workflow()
    html = combined_html(workflow.objective_card)
    for label in ("Observe", "Candidate actions", "Nemotron choice", "Execute", "Measure"):
        assert label in html
    assert "decision_basis" not in html
    assert "select_next_panel_swap" in html


@pytest.mark.parametrize("action_count", [0, 1, 2, 3])
def test_action_menu_card_renders_exact_available_count(action_count):
    html = InteractiveWorkflow._objective_action_menu_html(
        production_menu(action_count=action_count)
    )
    assert html.count("aria-label='Candidate action'") == action_count
    if action_count == 0:
        assert "No legal improving substitution" in html


@pytest.mark.parametrize(("candidate", "current", "expected"), BOUNDARY_CASES)
def test_display_uses_the_shared_boundary_table(candidate, current, expected):
    left, right, status = InteractiveWorkflow._score_comparison(candidate, current)
    if expected:
        assert status == "ordered"
        assert float(left) > float(right)
    else:
        assert status == "tied at 1e-12 decision precision"


def test_evaluation_failure_never_renders_executed_or_success():
    workflow = evaluation_failed_workflow()
    html = combined_html(workflow.objective_card)
    assert "evaluation not completed" in html.lower()
    assert "Goal achieved" not in html
```

- [ ] **Step 2: Run widget tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_interactive_workflow.py -k 'baseline_before or menu_selection or action_menu_card or display_uses or evaluation_failure'
```

Expected: failures because the current ladder begins with a free-form initial panel and uses singular limiting pairs.

- [ ] **Step 3: Implement the baseline-first persistent ladder**

Replace `_objective_attempt_row` inputs with `ObjectiveActionMenu`, `ObjectiveSelection`, and committed `ObjectiveAttempt`. Render menu rows ordered by `swap_id`, but include deterministic score, delta, resulting co-limiting pairs, and target status. Escape every molecule ID and state value.

Use `score_key` for display decisions:

```python
def _score_comparison(first: float, second: float) -> tuple[str, str, str]:
    if score_key(first) == score_key(second):
        return _display(first, 12), _display(second, 12), "tied at 1e-12 decision precision"
    precision = _precision_that_preserves_order(first, second)
    return f"{first:.{precision}f}", f"{second:.{precision}f}", "ordered"
```

Store all factual ledger and conclusion text in `widgets.HTML.value`, not transient `widgets.Output`. A validated-but-unmeasured selection gets a neutral **evaluation not completed** row. Rendering failures reconstruct from controller state and never re-execute chemistry.

- [ ] **Step 4: Run the complete widget suite**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_interactive_workflow.py
```

Expected: all UI, retry, escaping, color, and persistence tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add interactive_workflow.py tests/test_interactive_workflow.py
git commit -m "Render persistent objective decision ladder"
```

### Task 6: Build the predicate-true finding catalog and measured summary

**Files:**
- Create: `objective_findings.py`
- Create: `tests/test_objective_findings.py`
- Modify: `objective_challenge.py`
- Modify: `tests/objective_fixtures.py`

**Fixture contract:** Extend `tests/objective_fixtures.py` with `evidence_report()` returning six real `EvidenceRecord` objects whose canonical JSON payloads contain the exact production keys used by E01-E06, and `report_and_run(reason="target_achieved")` returning that report plus a valid `terminal_objective_run`. Do not use `{}` placeholder evidence. Values are: 256 raw/valid and zero invalid molecules; Morgan radius 2 and 1024 bits; fixed similarity quartiles plus one named most-similar pair; cutoff 0.4, 70 clusters, 37 singletons, and largest sizes; four representatives and 20 generated conformers; 19 converged and one unconverged conformer. Use `terminal_fixture` from Task 2 to construct each valid reason.

- [ ] **Step 1: Write failing catalog, truth, and scope tests**

```python
def test_measured_summary_labels_distance_and_similarity_as_complements():
    report, run = report_and_run()
    summary = build_measured_summary(report, run)
    assert summary.final_distance == pytest.approx(0.8374999910593033)
    assert summary.limiting_similarities == pytest.approx((0.1625000089406967,))


def test_catalog_distinguishes_candidate_and_final_cluster_coverage():
    report, run = report_and_run()
    catalog = build_finding_catalog(report, run)
    text = " ".join(item.text for item in catalog.findings)
    assert "eight-candidate pool spans 8 distinct clusters" in text
    assert "four-compound final panel spans 4 distinct clusters" in text
    assert "final panel spans 8" not in text


def test_default_catalog_has_nonvacuous_choices_in_four_themes():
    report, run = report_and_run()
    catalog = build_finding_catalog(report, run)
    alternatives = {
        theme: len(catalog.ids_for_theme(theme)) for theme in CONCLUSION_THEMES
    }
    assert alternatives["objective_driven_selection"] >= 2
    assert sum(count >= 2 for count in alternatives.values()) >= 4


def test_conformer_findings_preserve_within_molecule_energy_scope():
    report, run = report_and_run()
    catalog = build_finding_catalog(report, run)
    assert any(
        "within each molecule among converged sampled conformers" in item.text
        for item in catalog.findings
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("valid_molecule_count", 256),
        ("fingerprint_radius", 2),
        ("fingerprint_size", 1024),
        ("cluster_cutoff", 0.4),
        ("cluster_count", 70),
        ("singleton_count", 37),
        ("candidate_pool_count", 8),
        ("candidate_cluster_count", 8),
        ("final_panel_count", 4),
        ("final_cluster_count", 4),
        ("generated_conformer_count", 20),
        ("converged_conformer_count", 19),
        ("unconverged_conformer_count", 1),
    ],
)
def test_measured_summary_derives_every_required_field(field, expected):
    report, run = report_and_run()
    assert getattr(build_measured_summary(report, run), field) == expected


@pytest.mark.parametrize(
    ("reason", "headline_fragment"),
    [
        ("target_achieved", "Target achieved"),
        ("baseline_already_optimal", "Baseline already optimal"),
        ("attempt_limit_reached", "Objective not achieved within attempt limit"),
        ("no_legal_improving_swap", "No legal improving substitution"),
        ("objective_correction_limit", "Objective selection stopped after invalid responses"),
        ("objective_provider_failure", "Objective provider unavailable"),
        ("evaluation_not_completed", "Objective evaluation not completed"),
    ],
)
def test_every_terminal_reason_has_truthful_deterministic_headline(reason, headline_fragment):
    report, run = report_and_run(reason)
    assert headline_fragment in build_measured_summary(report, run).headline


def test_finding_predicate_is_rechecked_against_snapshot():
    snapshot = build_evidence_snapshot(*report_and_run())
    finding = build_finding_catalog_from_snapshot(snapshot).findings[0]
    validate_finding(finding, snapshot)
    with pytest.raises(ValueError):
        validate_finding(finding, replace(snapshot, valid_molecule_count=255))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_findings.py
```

Expected: collection fails because `objective_findings.py` does not exist.

- [ ] **Step 3: Implement immutable facts and findings**

Create:

```python
CONCLUSION_THEMES = (
    "dataset_scope",
    "molecular_representation",
    "similarity_structure",
    "clustering",
    "conformational_sampling",
    "objective_driven_selection",
    "limitations_and_next_steps",
)


@dataclass(frozen=True)
class EvidenceFinding:
    finding_id: str
    theme: str
    evidence_keys: tuple[str, ...]
    predicate_id: str
    text: str


@dataclass(frozen=True)
class MeasuredSummary:
    headline: str
    facts: tuple[str, ...]
    valid_molecule_count: int
    fingerprint_radius: int
    fingerprint_size: int
    cluster_cutoff: float
    cluster_count: int
    singleton_count: int
    candidate_pool_count: int
    candidate_cluster_count: int
    final_panel_count: int
    final_cluster_count: int
    generated_conformer_count: int
    converged_conformer_count: int
    unconverged_conformer_count: int
    final_distance: float
    target_distance: float
    target_margin: float
    limiting_pairs: tuple[tuple[str, str], ...]
    limiting_similarities: tuple[float, ...]


@dataclass(frozen=True)
class FindingCatalog:
    findings: tuple[EvidenceFinding, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.finding_id for item in self.findings)

    def ids_for_theme(self, theme: str) -> tuple[str, ...]:
        return tuple(item.finding_id for item in self.findings if item.theme == theme)
```

Add a frozen `EvidenceSnapshot` containing every parsed field used by `MeasuredSummary` or a finding. Parse E01-E06 and O01 canonical JSON into this type; reject missing keys, non-finite numbers, contradictory counts, or an O01 record that does not reconstruct the supplied `ObjectiveRun`.

Define a closed `_FINDING_PREDICATES: dict[str, Callable[[EvidenceSnapshot], bool]]`. `validate_finding(finding, snapshot)` rejects unknown predicate IDs, false predicates, wrong evidence keys, or text that differs from the deterministic renderer for that predicate. Call it during catalog construction, after hosted selection validation, and immediately before UI rendering. Generate at least the following two alternatives where their predicates hold: valid-count versus exclusion scope; representation definition versus reuse; distribution versus most-similar pair; cluster totals versus largest-cluster structure; convergence totals versus within-molecule energy rule; target result versus final-panel cluster coverage; scope limit versus next experimental validation. Text is generated only from the validated snapshot and never stored as model input prose.

- [ ] **Step 4: Run finding and objective evidence tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_objective_findings.py tests/test_objective_challenge.py
```

Expected: all evidence, complement, cardinality, and claim-scope tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add objective_findings.py objective_challenge.py tests/objective_fixtures.py tests/test_objective_findings.py tests/test_objective_challenge.py
git commit -m "Add evidence-controlled objective findings"
```

### Task 7: Integrate finding selection, deterministic conclusion, and persistent images

**Files:**
- Modify: `demo_agent.py`
- Modify: `interactive_workflow.py`
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_objective_agent_loop.py`
- Modify: `tests/test_interactive_workflow.py`

**Fixture contract:** Import `report_and_run` from `tests.objective_fixtures`. In `tests/test_objective_agent_loop.py`, add:

```python
def catalog():
    report, run = report_and_run()
    return build_finding_catalog(report, run)


def finding_selection(active_catalog, ordered_ids=None):
    ids = list(ordered_ids or active_catalog.ids)
    return response("select_evidence_findings", {"ordered_finding_ids": ids})


def completed_terminal_controller(reason, finding_responses):
    report, run = report_and_run(reason)
    controller, completions = completed_controller(finding_responses)
    controller.objective_context = run.context
    controller.objective_attempts = list(run.attempts)
    controller.objective_run = run
    controller.objective_evidence = build_objective_evidence(run)
    controller.report = report
    return controller, completions
```

In `tests/test_interactive_workflow.py`, import `embed_minimal_html` from `ipywidgets.embed`. Migrate the existing local `Controller` fake to expose `pending_action_menu`, `pending_objective_selection`, five counters, `request_objective_selection`, and `execute_objective_selection`; construct its domain objects with the Task 2 public helpers rather than `SimpleNamespace`. Add recursive `walk_widgets(widget)` yielding the widget and every `children` descendant, `combined_html(widget)` joining every `widgets.HTML.value`, and `completed_workflow(finding_selection_available=True)` that drives the existing `InteractiveWorkflow` fake through its six stages, objective selections, and either a selected or unavailable conclusion.

- [ ] **Step 1: Write failing conclusion schema, fallback, and persistence tests**

```python
def test_finding_selection_has_one_closed_ordered_field():
    assert tuple(FindingSelection.model_fields) == ("ordered_finding_ids",)
    schema = _tool_definition(
        "select_evidence_findings",
        FindingSelection,
        finding_catalog=catalog(),
    )["function"]["parameters"]
    assert schema["properties"]["ordered_finding_ids"]["items"]["enum"] == list(catalog().ids)


def test_invalid_finding_selection_falls_back_without_downgrading_success():
    controller, _ = completed_terminal_controller(
        "target_achieved", [finding_selection(catalog(), ordered_ids=["D01"] * 7)]
    )
    result = controller.request_synthesis()
    assert result.objective_run.achieved is True
    assert result.conclusion.finding_selection_status == "finding_selection_unavailable"
    assert "Target achieved" in result.conclusion.measured_summary.headline


def test_standalone_widget_state_contains_conclusion_and_png_images():
    workflow = completed_workflow()
    assert all(isinstance(child, (widgets.HTML, widgets.Image, widgets.Accordion, widgets.VBox))
               for child in workflow.objective_card.children)
    assert "Evidence-Backed Conclusion" in combined_html(workflow.root)
    assert any(isinstance(child, widgets.Image) and child.value.startswith(b"\x89PNG")
               for child in walk_widgets(workflow.root))


def test_widget_state_round_trip_preserves_text_and_images(tmp_path):
    workflow = completed_workflow()
    target = tmp_path / "objective-widget.html"
    embed_minimal_html(target, views=[workflow.root], title="objective acceptance")
    exported = target.read_text(encoding="utf-8")
    for text in ("Step 0", "Candidate actions", "Nemotron choice", "Evidence-Backed Conclusion"):
        assert text in exported
    assert "image/png" in exported or "iVBOR" in exported


def test_fallback_never_claims_agent_selected_emphasis():
    workflow = completed_workflow(finding_selection_available=False)
    html = combined_html(workflow.root).lower()
    assert "agent-selected evidence emphasis" not in html
    assert "agent-selected emphasis unavailable" in html
```

- [ ] **Step 2: Run focused conclusion tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_demo_agent.py tests/test_objective_agent_loop.py tests/test_interactive_workflow.py -k 'finding_selection or downgrading_success or standalone_widget or round_trip or fallback_never_claims'
```

Expected: failures because the current conclusion accepts free-form headline/prose and figures live in transient `Output` widgets.

- [ ] **Step 3: Implement the closed finding-selection contract**

Replace objective `ObjectiveSubmitConclusionArgs` with:

```python
class FindingSelection(_StrictModel):
    ordered_finding_ids: list[str] = Field(min_length=7, max_length=7)


@dataclass(frozen=True)
class EvidenceControlledConclusion:
    evidence_snapshot: EvidenceSnapshot
    measured_summary: MeasuredSummary
    ordered_findings: tuple[EvidenceFinding, ...]
    finding_selection_status: Literal["selected", "finding_selection_unavailable"]
```

The dynamic schema enumerates only current predicate-true IDs. Local validation requires seven unique IDs and exactly one per theme, and calls `validate_finding` for every selected item against the current `EvidenceSnapshot`. Make exactly one hosted call with no semantic retry. A malformed tool response gets one paired rejected tool result; a transport failure without a tool call sets `finding_selection_unavailable`. In both cases, return the deterministic measured summary and canonical theme-order findings. Never alter the O01-derived success/failure headline.

Update `WorkflowResult` to hold `EvidenceControlledConclusion` for objective runs while retaining the old non-objective conclusion type for backward compatibility.

- [ ] **Step 4: Persist conclusion text and figures in widget state**

Render summary and findings into `widgets.HTML`. Convert each Matplotlib figure and RDKit PIL grid to PNG bytes before widget construction:

```python
def _persistent_image_widget(value: object) -> widgets.Image:
    png = _png_bytes(value)
    return widgets.Image(value=png, format="png")
```

Use **agent-selected evidence emphasis** only when `finding_selection_status == "selected"` and the theme has multiple valid findings; label single-option themes **required measured finding**. When status is `finding_selection_unavailable`, label the whole fallback **agent-selected emphasis unavailable** and every rendered catalog item **deterministic fallback finding**, regardless of how many alternatives existed. Immediately before rendering each finding, call `validate_finding` again against the retained snapshot. If rendering fails after controller completion, retain `workflow_result`, show a safe placeholder, and permit reconstruction without hosted or chemistry calls.

- [ ] **Step 5: Run controller and widget suites**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_demo_agent.py tests/test_objective_agent_loop.py tests/test_interactive_workflow.py
```

Expected: all conclusion, fallback, retry, rendering, and standalone-state tests pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add demo_agent.py interactive_workflow.py tests/test_demo_agent.py tests/test_objective_agent_loop.py tests/test_interactive_workflow.py
git commit -m "Render evidence-controlled persistent conclusion"
```

### Task 8: Update the notebook contract and add reproducible qualification

**Files:**
- Modify: `demo_agent.py`
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `README.md`
- Create: `scripts/run_objective_reliability.py`
- Create: `tests/test_objective_reliability.py`
- Modify: `tests/test_notebook.py`
- Modify: `tests/test_gpu_acceptance.py`

**Fixture contract:** Import `optimized_state`, `evidence_report`, and `report_and_run` from `tests.objective_fixtures`. Add `prepared_snapshot()` to `tests/test_objective_reliability.py`: construct the exact six-stage `WorkflowPlan`, six ordered `StageResult` objects, optimized state, real six-record report, and **16 total messages**—one system message, one user message, and seven assistant/tool pairs with matching IDs—plus `turn_count=7`. Add `ScriptedControllerFactory(snapshot, retry_trial_index=None)`; each call uses `clone_prepared_controller`, installs a new `FakeCompletions` that chooses the current `accepted_maxima`, optionally injects one first transport failure for the selected trial, and records the controller. Add `reliability_receipt_with_failure(name)` by starting from a fully passing frozen receipt and using `dataclasses.replace` to fail only the named gate; `name=None` returns the passing receipt. Add `receipt_with_sensitive_trial_metadata()` by inserting `api_key` and `raw_model_response` sentinel keys into one per-trial dict, proving the allowlist writer removes them.

- [ ] **Step 1: Write failing notebook, harness, and GPU source-gate tests**

```python
def test_notebook_describes_bounded_evidence_optimal_selection():
    text = read_notebook().cells[5].source
    for phrase in (
        "deterministically evaluated candidate actions",
        "state-bound Nemotron choice",
        "co-limiting pairs",
        "Python owns every numerical result",
        "agent-selected evidence emphasis",
    ):
        assert phrase in text
    assert "qualitative interpretation is not automatically fact-verified" not in text


def test_reliability_receipt_separates_decision_and_transport_results(tmp_path):
    factory = ScriptedControllerFactory(prepared_snapshot(), retry_trial_index=0)
    records = run_trials(factory, trials=20)
    assert len(records) == 20
    assert sum(item["argmax_selected"] for item in records) == 20
    assert sum(item["retry_assisted"] for item in records) == 1
    assert sum(item["clean_first_request"] for item in records) == 19


def test_prepared_snapshot_clones_isolated_sessions_and_state():
    factory = ScriptedControllerFactory(prepared_snapshot())
    first, second = factory(), factory()
    first.session.messages.append({"role": "user", "content": "mutation"})
    first.session.state.records[0]["id"] = "mutated"
    assert first.session.messages != second.session.messages
    assert second.session.state.records[0]["id"] != "mutated"
    assert first.objective_attempts is not second.objective_attempts


@pytest.mark.parametrize(
    "failed_gate",
    [
        "incomplete_trial",
        "non_argmax",
        "unpaired_message",
        "unsafe_claim",
        "missing_end_to_end",
        "nonzero_temperature",
    ],
)
def test_reliability_cli_returns_nonzero_for_every_gate(monkeypatch, failed_gate):
    receipt = reliability_receipt_with_failure(failed_gate)
    assert qualification_exit_code(receipt) == 1


def test_canonical_receipt_excludes_secrets_and_raw_model_prose(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-sentinel")
    receipt = receipt_with_sensitive_trial_metadata()
    target = tmp_path / "receipt.json"
    write_reliability_receipt(target, receipt)
    rendered = target.read_text(encoding="utf-8")
    assert "nvapi-secret-sentinel" not in rendered
    assert "raw-prose-sentinel" not in rendered
    assert json.loads(rendered)["production_temperature_zero"] is True
```

Extend the GPU source gate to require `certify_argmax_reachability`, `state_id`, `accepted_maxima`, all co-limiting pairs, O01 success, and persistent conclusion construction.

- [ ] **Step 2: Run notebook and harness tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_notebook.py tests/test_objective_reliability.py tests/test_gpu_acceptance.py
```

Expected: failures because copy and the reliability harness are not implemented; the live GPU test remains skipped.

- [ ] **Step 3: Implement the deterministic reliability runner**

Create a CLI with `--trials`, `--end-to-end-runs`, and `--output`:

```python
@dataclass(frozen=True)
class ReliabilityReceipt:
    requested_trials: int
    completed_trials: int
    argmax_successes: int
    clean_first_request_trials: int
    retry_assisted_trials: int
    requested_end_to_end_runs: int
    completed_end_to_end_runs: int
    message_pairing_passes: int
    claim_safety_passes: int
    production_temperature_zero: bool
    objective_trials: tuple[dict[str, object], ...]
    end_to_end_runs: tuple[dict[str, object], ...]
    failed_trials: tuple[dict[str, object], ...]


def run_trials(
    controller_factory: Callable[[], BoundedWorkflowController],
    *,
    trials: int,
) -> tuple[dict[str, object], ...]:
    return tuple(_run_one_objective(controller_factory()) for _ in range(trials))


def run_end_to_end(
    controller_factory: Callable[[], BoundedWorkflowController],
    *,
    runs: int,
) -> tuple[dict[str, object], ...]:
    return tuple(_run_one_end_to_end(controller_factory()) for _ in range(runs))


def run_qualification(
    objective_factory: Callable[[], BoundedWorkflowController],
    end_to_end_factory: Callable[[], BoundedWorkflowController],
    *,
    trials: int,
    end_to_end_runs: int,
) -> ReliabilityReceipt:
    objective = run_trials(objective_factory, trials=trials)
    end_to_end = run_end_to_end(end_to_end_factory, runs=end_to_end_runs)
    return _summarize(
        objective,
        end_to_end,
        requested_trials=trials,
        requested_end_to_end_runs=end_to_end_runs,
    )
```

Add these public snapshot interfaces to `demo_agent.py`:

```python
@dataclass(frozen=True)
class PreparedScientificSnapshot:
    messages: tuple[dict[str, Any], ...]
    state: WorkflowState
    plan: WorkflowPlan
    stage_results: tuple[StageResult, ...]
    report: WorkflowReport
    turn_count: int


def clone_prepared_controller(
    snapshot: PreparedScientificSnapshot,
    *,
    client: Any,
    executors: dict[str, Any],
) -> BoundedWorkflowController:
    return BoundedWorkflowController(
        session=AgentSession(
            messages=copy.deepcopy(list(snapshot.messages)),
            state=copy.deepcopy(snapshot.state),
            turn_count=snapshot.turn_count,
        ),
        client=client,
        executors=executors,
        plan=copy.deepcopy(snapshot.plan),
        stage_results=list(copy.deepcopy(snapshot.stage_results)),
        report=copy.deepcopy(snapshot.report),
        objective_required=True,
    )
```

Validate the snapshot before cloning: exact seven turns, optimized state, exact six ordered stage results, exact E01-E06 report, and assistant/tool pairing. The production CLI builds one prepared snapshot by running the fixed six scientific stages with a deterministic scripted plan/parameter client, so the 20 objective trials isolate the hosted objective selector. Each clone receives the real hosted client and begins with no objective prompt, menu, counters, attempts, run, or O01. The three end-to-end runs instead use fresh states and the real hosted client for plan, stages, objective, and finding selection.

`write_reliability_receipt(path, receipt)` writes canonical JSON from an explicit allowlist: model, environment, aggregate counts, and per-trial accepted/rejected/transport counters, scores, target, termination, message-pairing result, deterministic-claim-validation result, retry status, and conclusion status. It reconstructs each output dict from those keys rather than dumping arbitrary trial metadata; therefore API keys, environment secrets, raw assistant content, and raw model responses cannot enter the artifact. Assert every hosted request uses the production model, forced strict tool, `temperature=0.0`, and `enable_thinking=False`.

Exit nonzero unless objective trials equal `--trials`, every trial selects argmax and reaches target within three measured attempts, every assistant call is paired, every measured summary/finding passes deterministic claim validation, production temperature is zero, and all requested end-to-end runs complete through the evidence-controlled conclusion. `qualification_exit_code` implements these checks directly over `ReliabilityReceipt` and has one failing test per gate. The CLI must call `run_qualification`, not `run_trials`, so a requested end-to-end count can never be silently ignored.

- [ ] **Step 4: Update notebook and README copy without changing structure**

Keep exactly eight notebook cells, at most 25 code lines, one `launch_interactive_workflow(...)` call, and no embedded objective implementation. Replace the obsolete free-form-rationale and automatically-unverified-prose language with the approved responsibility boundary. Document CPU, GPU, 20-trial, and three-run commands in `README.md`.

- [ ] **Step 5: Run all local gates**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q
git diff --check
```

Expected: the full CPU suite passes with only the existing GPU-only skip; `git diff --check` prints nothing.

- [ ] **Step 6: Commit Task 8**

```bash
git add demo_agent.py README.md notebooks/nvmolkit_nemotron_demo.ipynb scripts/run_objective_reliability.py tests/test_demo_agent.py tests/test_notebook.py tests/test_objective_reliability.py tests/test_gpu_acceptance.py
git commit -m "Add objective reliability qualification"
```

## Final Review and Live Acceptance

- [ ] **Step 1: Dispatch a fresh final spec-compliance reviewer**

Require an explicit mapping from every section of `docs/superpowers/specs/2026-08-07-constrained-decision-ladder-design.md` to implementation and tests. Any missing or extra behavior returns to the responsible task worker.

- [ ] **Step 2: Dispatch a fresh final code/scientific-quality reviewer**

Require review of numerical semantics, co-limiting-pair invariants, state/message pairing, atomicity, false scientific claims, persistent widget state, and test independence. Resolve every Critical, Important, or Minor finding and re-review.

- [ ] **Step 3: Run the complete GPU suite on the authorized Brev L4**

Run in the task-owned remote checkout:

```bash
RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q
```

Expected: every test passes; warnings are recorded separately and are not silently treated as failures or successes.

- [ ] **Step 4: Run the hosted reliability campaign**

Run:

```bash
.venv/bin/python scripts/run_objective_reliability.py \
  --trials 20 \
  --end-to-end-runs 3 \
  --output visual-qa/objective-reliability.json
```

Expected: exit code 0, `20/20` target-achieving argmax trials, `3/3` evidence-controlled end-to-end conclusions, and separately reported clean versus retry-assisted transport outcomes.

- [ ] **Step 5: Perform rendered persistence acceptance**

Open the live notebook, run the workflow, reopen or export the widget state, and verify that Step 0, action menus, state-bound Nemotron choices, commands, executed measurements, trajectory, structures, heatmap, measured summary, and selected findings remain visible. Capture the exact rendered artifact and acceptance receipt without exposing credentials.

- [ ] **Step 6: Merge only after all gates pass**

Use the finishing-development-branch workflow. Merge locally into `main` only after the full local suite, independent final reviews, L4 GPU suite, 20/20 objective campaign, 3/3 end-to-end campaign, and rendered persistence inspection all pass. Do not call the feature conference-ready or substitute a cached trajectory if any gate remains open.
