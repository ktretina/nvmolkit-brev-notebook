import itertools
import json
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pytest

from chemistry_workflow import WorkflowPhase, WorkflowState
import objective_challenge
from objective_challenge import (
    CANDIDATE_COUNT,
    MAX_ATTEMPTS,
    PANEL_SIZE,
    ObjectiveAttempt,
    ObjectiveCandidate,
    ObjectiveContext,
    ObjectiveRun,
    ObjectiveSwap,
    TerminationReason,
    accepted_maxima,
    baseline_terminal_run,
    build_action_menu,
    build_objective_context,
    build_objective_evidence,
    evaluate_diverse_panel,
    evaluate_selected_swap,
    enumerate_legal_swaps,
    finalize_no_legal_swap,
    finalize_objective_run,
    objective_figures,
    rank_legal_swaps,
    terminal_objective_run,
)
from objective_fixtures import (
    BOUNDARY_CASES,
    TARGET_BOUNDARY_CASES,
    context_from_distance,
    boundary_policy_context,
    controlled_context,
    controlled_context_with_ranked_swaps,
    controlled_context_with_tied_paths,
    controlled_context_with_three_misses,
    controlled_context_with_action_count,
    controlled_context_without_improving_swaps,
    optimized_state,
    quantized_baseline_target_context,
    terminal_fixture,
    two_revision_context,
)


def test_action_menu_is_capped_by_rank_then_displayed_by_swap_id():
    context = controlled_context_with_ranked_swaps()
    source = objective_challenge.measure_panel(context, context.baseline_ids)

    ranked = enumerate_legal_swaps(context, source)
    menu = build_action_menu(context, source, 0)

    assert len(ranked) == 4
    assert {action.swap_id for action in menu.actions} == {
        action.swap_id for action in ranked[:3]
    }
    assert tuple(action.swap_id for action in menu.actions) == tuple(
        sorted(action.swap_id for action in menu.actions)
    )
    assert all(action.predicted_score_key == objective_challenge.score_key(action.predicted_score) for action in ranked)
    assert all(action.limiting_pairs for action in ranked)
    assert accepted_maxima(menu) == tuple(
        action for action in menu.actions
        if action.predicted_score_key == max(item.predicted_score_key for item in menu.actions)
    )


def test_state_id_is_stable_and_sensitive_to_source_count_and_displayed_actions():
    context = controlled_context_with_ranked_swaps()
    source = objective_challenge.measure_panel(context, context.baseline_ids)
    first = build_action_menu(context, source, 0)

    assert first.state_id == build_action_menu(context, source, 0).state_id
    assert first.state_id.startswith("state-") and len(first.state_id) == 22
    assert first.state_id != build_action_menu(context, source, 1).state_id
    next_source = objective_challenge.measure_panel(context, first.actions[0].resulting_ids)
    assert first.state_id != build_action_menu(context, next_source, 1).state_id


def test_selected_swap_evaluation_is_menu_bound_and_measurement_backed():
    context = controlled_context_with_ranked_swaps()
    source = objective_challenge.measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)
    selected = accepted_maxima(menu)[0]

    attempt = evaluate_selected_swap(context, menu, selected, 1)

    assert attempt.state_id == menu.state_id
    assert attempt.measurement == objective_challenge.measure_panel(
        context, selected.resulting_ids
    )
    assert attempt.score_key == selected.predicted_score_key
    assert attempt.limiting_pairs == selected.limiting_pairs
    with pytest.raises(ValueError, match="accepted maximum"):
        evaluate_selected_swap(context, menu, menu.actions[-1], 1)
    with pytest.raises(ValueError, match="current menu"):
        evaluate_selected_swap(context, menu, replace(selected, swap_id="forged"), 1)


def test_baseline_terminal_run_is_exact_and_o01_has_no_model_rationale():
    context = build_objective_context(optimized_state(baseline_optimal=True))

    run = baseline_terminal_run(context)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert run.termination_reason == "baseline_already_optimal"
    assert run.attempts == () and run.achieved is True
    assert payload["attempt_count"] == 0
    assert payload["baseline"]["score_key"] == payload["final_measurement"]["score_key"]
    assert "decision_basis" not in build_objective_evidence(run).payload_json


@pytest.mark.parametrize("action_count", range(4))
def test_action_menu_supports_every_bounded_display_count(action_count):
    context = controlled_context_with_action_count(action_count)
    source = objective_challenge.measure_panel(context, context.baseline_ids)

    menu = build_action_menu(context, source, 0)

    assert len(menu.actions) == action_count


@pytest.mark.parametrize(("candidate", "current", "improving"), BOUNDARY_CASES)
def test_boundary_policy_uses_exact_key_inclusion_and_maximality(
    candidate, current, improving
):
    context = boundary_policy_context(candidate, current)
    source = objective_challenge.measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)

    assert bool(menu.actions) is improving
    if improving:
        assert accepted_maxima(menu) == menu.actions
        assert menu.actions[0].predicted_score_key == objective_challenge.score_key(candidate)


def test_empty_menu_finalizes_truthful_no_legal_terminal_and_o01():
    context = controlled_context_without_improving_swaps()
    current = objective_challenge.measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, current, 0)

    run = finalize_no_legal_swap(context, (), current, menu)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert menu.actions == ()
    assert run.termination_reason == "no_legal_improving_swap"
    assert run.achieved is False
    assert payload["attempt_count"] == 0
    assert payload["final_measurement"]["limiting_pairs"]


@pytest.mark.parametrize(
    "reason",
    (
        TerminationReason.OBJECTIVE_CORRECTION_LIMIT,
        TerminationReason.OBJECTIVE_PROVIDER_FAILURE,
        TerminationReason.EVALUATION_NOT_COMPLETED,
    ),
)
def test_zero_attempt_failure_reasons_have_complete_terminal_evidence(reason):
    context = controlled_context_with_ranked_swaps()

    run = terminal_objective_run(context, (), reason)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert run.achieved is False
    assert payload["termination_reason"] == reason.value
    assert payload["attempt_count"] == 0
    assert payload["baseline"] == payload["final_measurement"]


def test_build_context_fails_closed_when_production_policy_is_uncertified(monkeypatch):
    monkeypatch.setattr(objective_challenge, "certify_argmax_reachability", lambda context: False)

    with pytest.raises(
        RuntimeError,
        match=r"^Objective target is not reachable under the bounded decision policy\.$",
    ):
        build_objective_context(optimized_state())


@pytest.mark.parametrize("all_paths_reach", (True, False))
def test_certificate_branches_every_displayed_tied_maximum(all_paths_reach):
    context = controlled_context_with_tied_paths(all_paths_reach)
    baseline = objective_challenge.measure_panel(context, context.baseline_ids)
    maxima = accepted_maxima(build_action_menu(context, baseline, 0))

    assert len(maxima) == 2
    assert len({action.predicted_score_key for action in maxima}) == 1
    assert objective_challenge.certify_argmax_reachability(context) is all_paths_reach


@pytest.mark.parametrize(
    ("reason", "attempt_count", "achieved"),
    (
        ("target_achieved", 1, True),
        ("baseline_already_optimal", 0, True),
        ("attempt_limit_reached", 3, False),
        ("no_legal_improving_swap", 0, False),
        ("objective_correction_limit", 0, False),
        ("objective_provider_failure", 0, False),
        ("evaluation_not_completed", 0, False),
    ),
)
def test_every_terminal_reason_has_one_truthful_run_and_o01(
    reason, attempt_count, achieved
):
    run = terminal_fixture(reason, attempt_count)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert run.termination_reason == reason
    assert run.achieved is achieved
    assert payload["attempt_count"] == attempt_count
    assert payload["termination_reason"] == reason
    assert payload["final_measurement"]["score_key"] == run.final_score_key


def test_baseline_terminal_recomputes_attainable_best_instead_of_trusting_context():
    context = controlled_context_with_ranked_swaps()
    forged = replace(context, benchmark_score=context.baseline_score)

    with pytest.raises(ValueError, match="context scores"):
        baseline_terminal_run(forged)


@pytest.mark.parametrize("forgery", ("baseline", "same_key_raw"))
def test_o01_rejects_every_incoherent_stored_benchmark(forgery):
    context = controlled_context_with_ranked_swaps()
    run = terminal_objective_run(
        context, (), TerminationReason.OBJECTIVE_PROVIDER_FAILURE
    )
    benchmark = objective_challenge.attainable_benchmark(context).score
    forged_score = (
        context.baseline_score
        if forgery == "baseline"
        else float(np.nextafter(benchmark, 0.0))
    )
    assert forged_score != benchmark
    if forgery == "same_key_raw":
        assert objective_challenge.score_key(forged_score) == objective_challenge.score_key(benchmark)
    forged_context = replace(context, benchmark_score=forged_score)
    forged_run = replace(
        run,
        context=forged_context,
        baseline=objective_challenge.measure_panel(
            forged_context, forged_context.baseline_ids
        ),
    )

    with pytest.raises(ValueError, match="context scores"):
        build_objective_evidence(forged_run)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("baseline_score", lambda context: float(np.nextafter(context.baseline_score, 1.0))),
        ("target_score", lambda context: float(np.nextafter(context.target_score, 1.0))),
    ),
)
def test_authoritative_policy_rejects_same_key_raw_context_score_forgery(
    field, forged_value
):
    context = build_objective_context(optimized_state())
    value = forged_value(context)
    assert value != getattr(context, field)
    assert objective_challenge.score_key(value) == objective_challenge.score_key(
        getattr(context, field)
    )
    forged = replace(context, **{field: value})
    source = objective_challenge.measure_panel(forged, forged.baseline_ids)

    with pytest.raises(ValueError, match="context scores"):
        build_action_menu(forged, source, 0)
    with pytest.raises(ValueError, match="context scores"):
        objective_challenge.certify_argmax_reachability(forged)


def test_weaker_forged_target_is_rejected_by_every_authoritative_entry():
    context = build_objective_context(optimized_state())
    valid_run = terminal_objective_run(
        context, (), TerminationReason.OBJECTIVE_PROVIDER_FAILURE
    )
    forged = replace(context, target_score=context.baseline_score)
    forged_source = objective_challenge.measure_panel(forged, forged.baseline_ids)
    forged_run = replace(valid_run, context=forged, baseline=forged_source)
    assert forged_source.achieved is True

    with pytest.raises(ValueError, match="context scores"):
        build_action_menu(forged, forged_source, 0)
    with pytest.raises(ValueError, match="context scores"):
        terminal_objective_run(
            forged, (), TerminationReason.OBJECTIVE_PROVIDER_FAILURE
        )
    with pytest.raises(ValueError, match="context scores"):
        build_objective_evidence(forged_run)


def test_exact_production_context_scores_pass_authoritative_validation():
    context = build_objective_context(optimized_state())
    source = objective_challenge.measure_panel(context, context.baseline_ids)

    assert build_action_menu(context, source, 0).actions
    assert objective_challenge.certify_argmax_reachability(context) is True


def test_quantized_baseline_target_is_zero_substitution_target_success():
    context = quantized_baseline_target_context()
    baseline = objective_challenge.measure_panel(context, context.baseline_ids)

    assert baseline.score_key == objective_challenge.score_key(context.target_score)
    assert baseline.score_key < objective_challenge.score_key(context.benchmark_score)
    assert baseline.achieved is True
    assert objective_challenge.certify_argmax_reachability(context) is True

    run = terminal_objective_run(context, (), TerminationReason.TARGET_ACHIEVED)
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert run.attempts == ()
    assert run.final_ids == baseline.selected_ids
    assert run.final_score == baseline.score
    assert payload["attempt_count"] == 0
    assert payload["termination_reason"] == "target_achieved"


def test_zero_substitution_target_success_rejects_below_target_baseline():
    context = controlled_context_with_ranked_swaps()

    with pytest.raises(ValueError, match="measured target success"):
        terminal_objective_run(context, (), TerminationReason.TARGET_ACHIEVED)


def test_zero_substitution_target_reason_does_not_replace_exact_baseline_optimal_reason():
    context = build_objective_context(optimized_state(baseline_optimal=True))

    with pytest.raises(ValueError, match="baseline-optimal"):
        terminal_objective_run(context, (), TerminationReason.TARGET_ACHIEVED)


def test_legacy_quantized_baseline_success_normalizes_to_zero_substitution_target():
    context = quantized_baseline_target_context()
    legacy_baseline = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure baseline Step 0.",
    )

    run = finalize_objective_run(context, (legacy_baseline,))
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert legacy_baseline.achieved is True
    assert legacy_baseline.score_key < objective_challenge.score_key(
        context.benchmark_score
    )
    assert run.termination_reason == "target_achieved"
    assert run.attempts == ()
    assert payload["attempt_count"] == 0
    assert payload["termination_reason"] == "target_achieved"


def test_legacy_exact_baseline_optimal_normalizes_to_baseline_terminal_reason():
    context = build_objective_context(optimized_state(baseline_optimal=True))
    legacy_baseline = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure baseline Step 0.",
    )

    run = finalize_objective_run(context, (legacy_baseline,))

    assert run.termination_reason == "baseline_already_optimal"
    assert run.attempts == ()


def test_legacy_finalizer_rejects_arbitrary_target_panel_without_swap():
    context = controlled_context_with_ranked_swaps()
    arbitrary = evaluate_diverse_panel(
        context,
        enumerate_legal_swaps(
            context, objective_challenge.measure_panel(context, context.baseline_ids)
        )[0].resulting_ids,
        attempt_number=1,
        decision_basis="Unsafe direct target panel.",
    )
    assert arbitrary.achieved is True and arbitrary.selected_swap is None

    with pytest.raises(ValueError, match="baseline Step 0"):
        finalize_objective_run(context, (arbitrary,))

    forged_run = ObjectiveRun(
        context=context,
        attempts=(arbitrary,),
        achieved=True,
        termination_reason="target_achieved",
        final_ids=arbitrary.selected_ids,
        final_score=arbitrary.score,
    )
    with pytest.raises(ValueError, match="exact current state"):
        build_objective_evidence(forged_run)


def test_legacy_finalizer_rejects_nonmax_revision_and_normalizes_exact_maximum():
    context = controlled_context_with_ranked_swaps()
    baseline = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure baseline Step 0.",
    )
    source = objective_challenge.measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)
    nonmax = menu.actions[-1]
    unsafe = evaluate_diverse_panel(
        context,
        nonmax.resulting_ids,
        attempt_number=2,
        decision_basis="Unsafe nonmax revision.",
        selected_swap=nonmax,
    )
    with pytest.raises(ValueError, match="accepted maximum"):
        finalize_objective_run(context, (baseline, unsafe))

    maximum = accepted_maxima(menu)[0]
    safe = evaluate_diverse_panel(
        context,
        maximum.resulting_ids,
        attempt_number=2,
        decision_basis="Exact accepted maximum.",
        selected_swap=maximum,
    )
    run = finalize_objective_run(context, (baseline, safe))
    payload = json.loads(build_objective_evidence(run).payload_json)

    assert len(run.attempts) == 1
    assert run.attempts[0].state_id == menu.state_id
    assert payload["attempt_count"] == 1
    assert payload["baseline"]["selected_ids"] == list(context.baseline_ids)
    assert all(item["state_id"] for item in payload["attempts"])


def test_legacy_adapter_supports_exactly_three_accepted_substitutions_after_step_zero():
    context = controlled_context_with_three_misses()
    legacy = [
        evaluate_diverse_panel(
            context,
            context.baseline_ids,
            attempt_number=1,
            decision_basis="Measure baseline Step 0.",
        )
    ]
    current = legacy[0]
    for legacy_number in (2, 3, 4):
        selected = rank_legal_swaps(context, current)[0]
        current = evaluate_diverse_panel(
            context,
            selected.resulting_ids,
            attempt_number=legacy_number,
            decision_basis="Apply the current accepted maximum.",
            selected_swap=selected,
        )
        legacy.append(current)

    run = finalize_objective_run(context, tuple(legacy))

    assert run.termination_reason == "attempt_limit_reached"
    assert tuple(attempt.attempt_number for attempt in run.attempts) == (1, 2, 3)


def forged_nonachieved_attempt(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...],
    *,
    score: float | bool | None = None,
) -> ObjectiveAttempt:
    measurement = objective_challenge.measure_panel(context, selected_ids)
    return ObjectiveAttempt(
        attempt_number=1,
        selected_ids=selected_ids,
        decision_basis="Forged state used to verify ranker validation.",
        score=measurement.score if score is None else score,
        limiting_pair=measurement.limiting_pairs[0],
        constraints_passed=True,
        achieved=False,
    )


def successful_run() -> tuple[WorkflowState, ObjectiveRun]:
    state = optimized_state()
    context = build_objective_context(state)
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, first)[0]
    second = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Remove the closest baseline analogue.",
        selected_swap=selected,
    )
    return state, finalize_objective_run(context, (first, second))


def test_constants_keep_the_challenge_visually_bounded():
    assert (CANDIDATE_COUNT, PANEL_SIZE, MAX_ATTEMPTS) == (8, 4, 3)


def test_score_key_uses_one_trillion_half_up_units():
    assert objective_challenge.score_key(0.5000000000004) == 500_000_000_000
    assert objective_challenge.score_key(0.5000000000005) == 500_000_000_001
    assert objective_challenge.score_key(np.float32(0.5)) == 500_000_000_000
    with pytest.raises(ValueError):
        objective_challenge.score_key(True)


@pytest.mark.parametrize(
    "invalid_score",
    (0, "0.5", None, float("nan"), float("inf"), -0.1, 1.1),
)
def test_score_key_rejects_invalid_values_without_echoing_them(invalid_score):
    with pytest.raises(
        ValueError, match=r"^Objective score must be a finite float in \[0, 1\]\.$"
    ):
        objective_challenge.score_key(invalid_score)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.4999999999994, 499_999_999_999),
        (0.4999999999995, 500_000_000_000),
        (0.4999999999996, 500_000_000_000),
    ),
)
def test_score_key_is_exact_below_on_and_above_a_half_unit_boundary(value, expected):
    assert objective_challenge.score_key(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        0.9000000000005,
        float(np.nextafter(0.9000000000005, 0.0)),
        float(np.nextafter(0.9000000000005, 1.0)),
        np.float32(0.1),
        np.nextafter(np.float32(0.5), np.float32(0.0)),
        np.nextafter(np.float32(0.5), np.float32(1.0)),
    ),
)
def test_score_key_matches_exact_binary_float_half_up_reference(value):
    normalized = float(value)
    exact_reference = int(
        (
            Decimal.from_float(normalized) * Decimal(10**12)
            + Decimal("0.5")
        ).to_integral_value(rounding=ROUND_FLOOR)
    )

    assert objective_challenge.score_key(value) == exact_reference


def test_measure_panel_retains_every_canonical_co_limiting_pair():
    context = controlled_context(
        distances={
            ("mol-0", "mol-1"): 0.4,
            ("mol-2", "mol-3"): 0.4000000000001,
        },
        default_distance=0.8,
    )

    measurement = objective_challenge.measure_panel(
        context, ("mol-3", "mol-2", "mol-1", "mol-0")
    )

    assert measurement.selected_ids == ("mol-3", "mol-2", "mol-1", "mol-0")
    assert measurement.score == 0.4
    assert measurement.score_key == objective_challenge.score_key(0.4)
    assert measurement.limiting_pairs == (
        ("mol-0", "mol-1"),
        ("mol-2", "mol-3"),
    )


@pytest.mark.parametrize(("candidate", "current", "expected"), BOUNDARY_CASES)
def test_improvement_uses_score_keys(candidate, current, expected):
    assert objective_challenge.is_strict_improvement(candidate, current) is expected


@pytest.mark.parametrize(("score", "target", "expected"), TARGET_BOUNDARY_CASES)
def test_target_attainment_uses_the_same_score_key(score, target, expected):
    assert objective_challenge.target_is_achieved(score, target) is expected


def test_context_fixture_derives_scores_and_preserves_read_only_float64_matrix():
    matrix = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.8, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    matrix[0, 1] = matrix[1, 0] = 0.35

    context = context_from_distance(matrix)

    assert context.distance_matrix.dtype == np.dtype(np.float64)
    assert context.distance_matrix.flags.writeable is False
    assert context.baseline_score == pytest.approx(0.35)
    assert context.benchmark_score == pytest.approx(0.8)


def _context_with_matrix(distance_matrix) -> ObjectiveContext:
    valid = controlled_context(distances={}, default_distance=0.8)
    return replace(valid, distance_matrix=distance_matrix)


@pytest.mark.parametrize(
    ("distance_matrix", "message"),
    (
        (np.full((8, 8), "0.0", dtype=str), "numeric"),
        (np.zeros((8, 9), dtype=float), "shape"),
        (
            np.array(
                [
                    [0.0 if row == column else (0.7 if (row, column) == (0, 1) else 0.8)
                    for column in range(8)]
                    for row in range(8)
                ],
                dtype=float,
            ),
            "symmetric",
        ),
        (np.eye(8, dtype=float), "diagonal"),
        (
            np.where(
                np.indices((8, 8))[0] == np.indices((8, 8))[1],
                0.0,
                np.nan,
            ),
            "finite",
        ),
        (
            np.where(
                np.indices((8, 8))[0] == np.indices((8, 8))[1],
                0.0,
                1.1,
            ),
            r"\[0, 1\]",
        ),
    ),
)
def test_objective_context_rejects_malformed_distance_matrices(
    distance_matrix, message
):
    with pytest.raises(ValueError, match=message):
        _context_with_matrix(distance_matrix)


def test_objective_context_owns_a_read_only_float64_distance_matrix():
    source = np.full((8, 8), 0.8, dtype=np.float32)
    np.fill_diagonal(source, 0.0)
    context = _context_with_matrix(source)
    original = objective_challenge.measure_panel(
        context, context.baseline_ids
    )

    source[0, 1] = source[1, 0] = 0.1

    assert context.distance_matrix.dtype == np.dtype(np.float64)
    assert context.distance_matrix.flags.owndata is False
    assert isinstance(context.distance_matrix.base.base, bytes)
    assert context.distance_matrix.flags.writeable is False
    assert objective_challenge.measure_panel(context, context.baseline_ids) == original
    with pytest.raises(ValueError):
        context.distance_matrix[0, 1] = 0.1


def test_distance_matrix_cannot_be_made_writeable_and_menu_identity_stays_stable():
    context = controlled_context_with_ranked_swaps()
    source = objective_challenge.measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)

    with pytest.raises(ValueError):
        context.distance_matrix.setflags(write=True)
    with pytest.raises(ValueError):
        context.distance_matrix[0, 1] = 0.0

    assert build_action_menu(context, source, 0) == menu


@pytest.mark.parametrize("invalid_distance", (float("nan"), float("inf"), -0.1, 1.1))
def test_panel_measurement_rejects_invalid_pair_distances(invalid_distance):
    context = controlled_context(distances={}, default_distance=0.8)
    invalid_matrix = np.array(context.distance_matrix, dtype=np.float64, copy=True)
    invalid_matrix[0, 1] = invalid_matrix[1, 0] = invalid_distance
    invalid_matrix.setflags(write=False)

    with pytest.raises(ValueError, match="distance"):
        replace(context, distance_matrix=invalid_matrix)


def test_panel_measurement_reuses_fail_closed_panel_validation():
    context = controlled_context(distances={}, default_distance=0.8)

    with pytest.raises(ValueError, match="four unique molecule IDs"):
        objective_challenge.measure_panel(
            context, ("mol-0", "mol-0", "mol-2", "mol-3")
        )
    with pytest.raises(ValueError, match="out-of-pool"):
        objective_challenge.measure_panel(
            context, ("mol-0", "mol-1", "mol-2", "outside")
        )


def test_build_context_uses_eight_distinct_mmff_eligible_clusters():
    context = build_objective_context(optimized_state())

    assert len(context.candidates) == 8
    assert len({candidate.cluster_id for candidate in context.candidates}) == 8
    assert context.baseline_ids == ("mol-0", "mol-1", "mol-2", "mol-3")
    assert context.baseline_score == pytest.approx(0.35)
    assert context.benchmark_score == pytest.approx(0.80)
    assert context.target_score == pytest.approx(0.71)
    assert context.distance_matrix.flags.writeable is False


def test_build_context_canonicalizes_upstream_tolerated_directional_noise():
    state = optimized_state()
    state.similarity.tensor.values[0, 1] += 5e-8

    context = build_objective_context(state)

    assert np.array_equal(context.distance_matrix, context.distance_matrix.T)
    assert context.distance_matrix.flags.owndata is False
    assert isinstance(context.distance_matrix.base.base, bytes)
    assert context.distance_matrix.flags.writeable is False


def _replace_context_candidates(
    context: ObjectiveContext, candidates: tuple[ObjectiveCandidate, ...]
) -> ObjectiveContext:
    return replace(context, candidates=candidates)


def test_objective_context_requires_exactly_eight_exact_candidates():
    context = controlled_context(distances={}, default_distance=0.8)

    with pytest.raises(ValueError, match="eight exact candidates"):
        _replace_context_candidates(context, context.candidates[:-1])

    class CandidateSubclass(ObjectiveCandidate):
        pass

    subclass = CandidateSubclass(**context.candidates[0].__dict__)
    with pytest.raises(ValueError, match="eight exact candidates"):
        _replace_context_candidates(context, (subclass, *context.candidates[1:]))


def test_objective_context_rejects_duplicate_or_empty_candidate_ids():
    context = controlled_context(distances={}, default_distance=0.8)

    for molecule_id in ("", context.candidates[1].molecule_id):
        candidates = (
            replace(context.candidates[0], molecule_id=molecule_id),
            *context.candidates[1:],
        )
        with pytest.raises(ValueError, match="molecule IDs"):
            _replace_context_candidates(context, candidates)


def test_objective_context_rejects_reserved_swap_delimiter_before_identity_collision():
    context = controlled_context_with_ranked_swaps()
    colliding_ids = ("a->b", "c", "a", "b->c", "d", "e", "f", "g")
    candidates = tuple(
        replace(candidate, molecule_id=molecule_id)
        for candidate, molecule_id in zip(context.candidates, colliding_ids)
    )

    with pytest.raises(ValueError, match="reserved delimiter"):
        replace(
            context,
            candidates=candidates,
            baseline_ids=colliding_ids[:4],
        )


@pytest.mark.parametrize("field", ("molecule_index", "source_row"))
def test_objective_context_rejects_ambiguous_candidate_provenance(field):
    context = controlled_context(distances={}, default_distance=0.8)
    candidates = (
        replace(
            context.candidates[0],
            **{field: getattr(context.candidates[1], field)},
        ),
        *context.candidates[1:],
    )

    with pytest.raises(ValueError, match="provenance"):
        _replace_context_candidates(context, candidates)


def test_objective_context_rejects_boolean_or_negative_candidate_provenance():
    context = controlled_context(distances={}, default_distance=0.8)

    for field, value in (("molecule_index", True), ("source_row", -1)):
        candidates = (
            replace(context.candidates[0], **{field: value}),
            *context.candidates[1:],
        )
        with pytest.raises(ValueError, match="provenance"):
            _replace_context_candidates(context, candidates)


def test_objective_context_requires_eight_distinct_cluster_ids():
    context = controlled_context(distances={}, default_distance=0.8)
    candidates = (
        replace(
            context.candidates[0], cluster_id=context.candidates[1].cluster_id
        ),
        *context.candidates[1:],
    )

    with pytest.raises(ValueError, match="cluster IDs"):
        _replace_context_candidates(context, candidates)


@pytest.mark.parametrize(
    "baseline_ids",
    (
        ("mol-0", "mol-1", "mol-2"),
        ("mol-0", "mol-1", "mol-2", "mol-2"),
        ("mol-0", "mol-1", "mol-2", "outside"),
    ),
)
def test_objective_context_rejects_invalid_or_out_of_pool_baseline(baseline_ids):
    context = controlled_context(distances={}, default_distance=0.8)

    with pytest.raises(ValueError, match="baseline"):
        replace(context, baseline_ids=baseline_ids)


def test_build_context_requires_optimized_state_and_eight_eligible_clusters():
    state = optimized_state()
    state.phase = WorkflowPhase.CLUSTERED
    with pytest.raises(RuntimeError, match="OPTIMIZED"):
        build_objective_context(state)

    state = optimized_state()
    state.clusters.pop()
    with pytest.raises(RuntimeError, match="eight eligible distinct clusters"):
        build_objective_context(state)


def test_evaluate_panel_uses_minimum_pairwise_distance_and_stable_limiting_pair():
    context = build_objective_context(optimized_state())
    result = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Evaluate the baseline.",
    )

    assert result.score == pytest.approx(0.35)
    assert result.limiting_pair == ("mol-0", "mol-1")
    assert result.constraints_passed is True
    assert result.achieved is False


@pytest.mark.parametrize(
    "selected_ids",
    [
        ("mol-0", "mol-0", "mol-2", "mol-3"),
        ("mol-0", "mol-1", "mol-2"),
        ("mol-0", "mol-1", "mol-2", "outside"),
    ],
)
def test_invalid_panels_fail_before_scoring(selected_ids):
    context = build_objective_context(optimized_state())
    with pytest.raises(ValueError):
        evaluate_diverse_panel(
            context,
            selected_ids,
            attempt_number=1,
            decision_basis="Invalid proposal.",
        )


def test_finalize_selects_best_attempt_and_uses_explicit_termination_reasons():
    context = build_objective_context(optimized_state())
    miss = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Baseline remains redundant.",
    )
    selected = rank_legal_swaps(context, miss)[0]
    success = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Remove the limiting analogue.",
        selected_swap=selected,
    )

    achieved = finalize_objective_run(context, (miss, success))
    exhausted = terminal_fixture("attempt_limit_reached", 3)

    assert achieved.achieved is True
    assert achieved.termination_reason == "target_achieved"
    assert achieved.final_ids == success.selected_ids
    assert exhausted.achieved is False
    assert exhausted.termination_reason == "attempt_limit_reached"


def test_o01_is_canonical_and_does_not_expose_the_hidden_benchmark_panel():
    _state, run = successful_run()
    record = build_objective_evidence(run)
    payload = json.loads(record.payload_json)

    assert record.key == "O01"
    assert record.label == "Objective-driven panel selection"
    assert "benchmark_panel" not in record.payload_json
    assert payload["achieved"] is True
    assert payload["baseline"]["limiting_pairs"][0] == ["mol-0", "mol-1"]
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == record.payload_json


def test_o01_retains_every_recomputed_canonical_limiting_pair():
    context = controlled_context(
        distances={
            ("mol-0", "mol-1"): 0.4,
            ("mol-2", "mol-3"): 0.4,
        },
        default_distance=0.8,
    )
    run = terminal_objective_run(
        context, (), TerminationReason.OBJECTIVE_PROVIDER_FAILURE
    )

    payload = json.loads(build_objective_evidence(run).payload_json)

    assert payload["baseline"]["limiting_pairs"] == [
        ["mol-0", "mol-1"],
        ["mol-2", "mol-3"],
    ]


def test_evaluate_panel_records_only_an_exact_selected_swap():
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, first)[0]

    second = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Apply the selected legal swap.",
        selected_swap=selected,
    )

    assert second.selected_swap is selected
    with pytest.raises(ValueError, match="selected swap"):
        evaluate_diverse_panel(
            context,
            context.baseline_ids,
            attempt_number=2,
            decision_basis="Reject a mismatched selected legal swap.",
            selected_swap=selected,
        )


def test_evaluate_panel_rejects_a_selected_swap_with_a_malformed_panel():
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, first)[0]
    malformed = ObjectiveSwap(
        replace_id=selected.replace_id,
        replacement_id=selected.replacement_id,
        resulting_ids=selected.resulting_ids + (selected.resulting_ids[0],),
        predicted_score=selected.predicted_score,
        score_delta=selected.score_delta,
        limiting_pair=selected.limiting_pair,
    )

    with pytest.raises(ValueError, match="selected swap|Objective proposal"):
        evaluate_diverse_panel(
            context,
            selected.resulting_ids,
            attempt_number=2,
            decision_basis="Reject the malformed selected legal swap.",
            selected_swap=malformed,
        )


@pytest.mark.parametrize(
    "mutate_swap",
    (
        lambda swap: replace(swap, replace_id="outside-pool"),
        lambda swap: replace(swap, replacement_id=swap.replace_id),
        lambda swap: replace(swap, replace_id=swap.replacement_id),
        lambda swap: replace(swap, score_delta=swap.score_delta + 0.1),
        lambda swap: replace(swap, score_delta=-0.1),
        lambda swap: replace(swap, score_delta=float("nan")),
        lambda swap: replace(swap, resulting_ids=list(swap.resulting_ids)),
        lambda swap: replace(swap, limiting_pair=list(swap.limiting_pair)),
        lambda swap: replace(swap, predicted_score=np.float64(swap.predicted_score)),
        lambda swap: replace(swap, score_delta=np.float64(swap.score_delta)),
        lambda swap: replace(swap, resulting_ids=(1, *swap.resulting_ids[1:])),
        lambda swap: replace(swap, limiting_pair=(swap.limiting_pair[0], 1)),
    ),
)
def test_evaluate_panel_rejects_noncanonical_selected_swap_fields(mutate_swap):
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, first)[0]

    with pytest.raises(ValueError, match="selected swap|Objective proposal"):
        evaluate_diverse_panel(
            context,
            selected.resulting_ids,
            attempt_number=2,
            decision_basis="Reject a noncanonical selected legal swap.",
            selected_swap=mutate_swap(selected),
        )


def test_evaluate_panel_rejects_integer_selected_swap_scores():
    context = build_objective_context(optimized_state())
    current = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, current)[0]

    for forged in (
        replace(selected, predicted_score=int(selected.predicted_score)),
        replace(selected, score_delta=int(selected.score_delta)),
    ):
        with pytest.raises(ValueError, match="built-in floats"):
            evaluate_diverse_panel(
                context,
                selected.resulting_ids,
                attempt_number=2,
                decision_basis="Reject integer raw score evidence.",
                selected_swap=forged,
            )


@pytest.mark.parametrize("field", ("predicted_score", "score_delta"))
def test_evaluate_panel_rejects_different_raw_swap_values_with_the_same_score_key(
    field,
):
    context = build_objective_context(optimized_state())
    current = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, current)[0]
    original = getattr(selected, field)
    forged_raw = float(np.nextafter(original, 1.0))
    assert forged_raw != original
    assert objective_challenge.score_key(forged_raw) == objective_challenge.score_key(
        original
    )

    with pytest.raises(ValueError, match="predicted score|score delta"):
        evaluate_diverse_panel(
            context,
            selected.resulting_ids,
            attempt_number=2,
            decision_basis="Reject altered raw score evidence.",
            selected_swap=replace(selected, **{field: forged_raw}),
        )


def test_o01_serializes_the_selected_intervention_without_hidden_answers():
    context = build_objective_context(optimized_state())
    first = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )
    selected = rank_legal_swaps(context, first)[0]
    second = evaluate_diverse_panel(
        context,
        selected.resulting_ids,
        attempt_number=2,
        decision_basis="Apply the selected legal swap.",
        selected_swap=selected,
    )

    payload = json.loads(
        build_objective_evidence(finalize_objective_run(context, (first, second))).payload_json
    )

    assert payload["attempt_count"] == 1
    assert payload["attempts"][0]["selected_swap"] == {
        "swap_id": selected.swap_id,
        "replace_id": selected.replace_id,
        "replacement_id": selected.replacement_id,
        "resulting_ids": list(selected.resulting_ids),
        "predicted_score": selected.predicted_score,
        "predicted_score_key": selected.predicted_score_key,
        "score_delta": selected.score_delta,
        "limiting_pair": list(selected.limiting_pair),
        "limiting_pairs": [
            list(pair)
            for pair in objective_challenge.measure_panel(
                context, selected.resulting_ids
            ).limiting_pairs
        ],
        "target_status": selected.target_status,
    }
    assert "benchmark_panel" not in payload


def test_objective_figures_show_trajectory_final_structures_and_heatmap():
    state, run = successful_run()
    trajectory, structures, heatmap = objective_figures(run, state)

    assert trajectory.axes[0].get_title() == "Objective score trajectory"
    assert structures.size[0] > 0 and structures.size[1] > 0
    assert heatmap.axes[0].get_title() == "Final-panel Tanimoto similarity"
    assert heatmap.axes[0].images[0].get_array().shape == (4, 4)


def test_rank_legal_swaps_returns_three_deterministic_target_reaching_suggestions():
    context = build_objective_context(optimized_state())
    baseline = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )

    suggestions = rank_legal_swaps(context, baseline)

    assert len(suggestions) == 3
    assert all(suggestion.predicted_score >= context.target_score for suggestion in suggestions)
    assert all(suggestion.score_delta > 0 for suggestion in suggestions)
    assert suggestions == tuple(
        sorted(
            suggestions,
            key=lambda suggestion: (
                -suggestion.predicted_score,
                suggestion.replace_id,
                suggestion.replacement_id,
                suggestion.resulting_ids,
            ),
        )
    )


def test_rank_legal_swaps_suggestions_match_direct_evaluation():
    context = build_objective_context(optimized_state())
    current = evaluate_diverse_panel(
        context,
        context.baseline_ids,
        attempt_number=1,
        decision_basis="Measure the current policy baseline.",
    )

    for suggestion in rank_legal_swaps(context, current):
        measured = evaluate_diverse_panel(
            context,
            suggestion.resulting_ids,
            attempt_number=2,
            decision_basis="Check the suggested legal swap.",
        )

        assert measured.score == pytest.approx(suggestion.predicted_score)
        assert measured.limiting_pair == suggestion.limiting_pair
        assert suggestion.score_delta == pytest.approx(measured.score - current.score)
        assert suggestion.replace_id in current.selected_ids
        assert suggestion.replacement_id not in current.selected_ids
        assert len(suggestion.resulting_ids) == PANEL_SIZE
        assert len(set(suggestion.resulting_ids)) == PANEL_SIZE


def test_rank_legal_swaps_validates_achieved_attempt_before_returning_none():
    context = build_objective_context(optimized_state())
    achieved = evaluate_diverse_panel(
        context,
        ("mol-0", "mol-2", "mol-4", "mol-6"),
        attempt_number=1,
        decision_basis="Use a diverse four-cluster panel.",
    )
    assert achieved.achieved is True
    assert rank_legal_swaps(context, achieved) == ()
    for forged in (
        replace(
            achieved,
            selected_ids=("outside-0", "outside-1", "outside-2", "outside-3"),
        ),
        replace(achieved, score=float(np.nextafter(achieved.score, 0.0))),
        replace(achieved, limiting_pair=("mol-0", "mol-0")),
        replace(achieved, constraints_passed=False),
    ):
        with pytest.raises(ValueError):
            rank_legal_swaps(context, forged)


@pytest.mark.parametrize("forged_score", (-1.0, float("nan")))
def test_rank_legal_swaps_rejects_forged_nonachieved_attempt_scores(forged_score):
    context = build_objective_context(optimized_state())
    forged = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-2", "mol-4", "mol-6"),
        decision_basis="This forged record pretends a solved panel is pending.",
        score=forged_score,
        limiting_pair=("mol-0", "mol-2"),
        constraints_passed=True,
        achieved=False,
    )

    with pytest.raises(ValueError):
        rank_legal_swaps(context, forged)


@pytest.mark.parametrize(
    "selected_ids",
    (
        ("mol-0", "mol-0", "mol-2", "mol-3"),
        ("mol-0", "mol-1", "mol-2"),
    ),
)
def test_rank_legal_swaps_rejects_forged_malformed_nonachieved_panels(selected_ids):
    context = build_objective_context(optimized_state())

    with pytest.raises(ValueError):
        rank_legal_swaps(
            context, forged_nonachieved_attempt(context, selected_ids)
        )


def test_objective_context_rejects_cross_context_cluster_conflicts():
    context = build_objective_context(optimized_state())
    with pytest.raises(ValueError, match="cluster IDs"):
        ObjectiveContext(
            candidates=tuple(
                ObjectiveCandidate(
                    molecule_id=candidate.molecule_id,
                    molecule_index=candidate.molecule_index,
                    source_row=candidate.source_row,
                    cluster_id=0
                    if candidate.molecule_id == "mol-1"
                    else candidate.cluster_id,
                )
                for candidate in context.candidates
            ),
            baseline_ids=context.baseline_ids,
            baseline_score=context.baseline_score,
            benchmark_score=context.benchmark_score,
            target_score=context.target_score,
            distance_matrix=context.distance_matrix,
        )


def test_rank_legal_swaps_rejects_boolean_current_scores_explicitly():
    context = build_objective_context(optimized_state())

    with pytest.raises(ValueError, match="non-boolean"):
        rank_legal_swaps(
            context,
            forged_nonachieved_attempt(context, context.baseline_ids, score=True),
        )


def test_rank_legal_swaps_rejects_integer_current_score_instead_of_coercing_it():
    context = controlled_context(
        distances={("mol-0", "mol-1"): 0.0},
        default_distance=0.8,
    )
    forged = forged_nonachieved_attempt(
        context, context.baseline_ids, score=0
    )

    with pytest.raises(ValueError, match="built-in float"):
        rank_legal_swaps(context, forged)


def test_rank_legal_swaps_rejects_different_raw_score_with_the_same_score_key():
    context = build_objective_context(optimized_state())
    forged_raw = float(np.nextafter(context.baseline_score, 1.0))
    assert forged_raw != context.baseline_score
    assert objective_challenge.score_key(forged_raw) == objective_challenge.score_key(
        context.baseline_score
    )
    forged = forged_nonachieved_attempt(
        context, context.baseline_ids, score=forged_raw
    )

    with pytest.raises(ValueError, match="does not match"):
        rank_legal_swaps(context, forged)


def test_rank_legal_swaps_supports_two_revisions_and_preserves_panel_order():
    context = two_revision_context()
    current = evaluate_diverse_panel(
        context,
        ("candidate-1", "candidate-0", "candidate-2", "candidate-3"),
        attempt_number=1,
        decision_basis="Start in a deliberately noncanonical panel order.",
    )

    first_ranker = rank_legal_swaps(context, current)

    assert current.achieved is False
    assert first_ranker
    first_swap = first_ranker[0]
    assert first_swap.replace_id == "candidate-0"
    assert first_swap.replacement_id == "candidate-4"
    assert first_swap.resulting_ids == (
        "candidate-1",
        "candidate-4",
        "candidate-2",
        "candidate-3",
    )
    assert current.score < first_swap.predicted_score < context.target_score

    second = evaluate_diverse_panel(
        context,
        first_swap.resulting_ids,
        attempt_number=2,
        decision_basis="Resolve the first independent limiting pair.",
    )
    second_ranker = rank_legal_swaps(context, second)

    assert second.achieved is False
    assert second_ranker
    third = evaluate_diverse_panel(
        context,
        second_ranker[0].resulting_ids,
        attempt_number=3,
        decision_basis="Resolve the remaining independent limiting pair.",
    )
    assert second_ranker[0].predicted_score >= context.target_score
    assert third.achieved is True


def test_rank_legal_swaps_can_reach_target_within_two_attempts_for_every_miss():
    context = build_objective_context(optimized_state())
    candidate_ids = tuple(candidate.molecule_id for candidate in context.candidates)
    misses = 0

    for panel in itertools.combinations(candidate_ids, PANEL_SIZE):
        first = evaluate_diverse_panel(
            context,
            panel,
            attempt_number=1,
            decision_basis="Evaluate a candidate panel.",
        )
        if first.achieved:
            continue
        misses += 1
        first_ranker = rank_legal_swaps(context, first)
        assert first_ranker
        for first_suggestion in first_ranker:
            second = evaluate_diverse_panel(
                context,
                first_suggestion.resulting_ids,
                attempt_number=2,
                decision_basis="Apply a proposed legal swap.",
            )
            if not second.achieved:
                assert any(
                    suggestion.predicted_score >= context.target_score
                    for suggestion in rank_legal_swaps(context, second)
                )

    assert misses >= 1
