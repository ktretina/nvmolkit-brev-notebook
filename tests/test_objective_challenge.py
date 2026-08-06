import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

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
    build_objective_context,
    build_objective_evidence,
    evaluate_diverse_panel,
    finalize_objective_run,
    objective_figures,
    rank_legal_swaps,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class FakeGpuResult:
    def __init__(self, values):
        self.tensor = FakeTensor(values)

    def torch(self):
        return self.tensor


def optimized_state() -> WorkflowState:
    smiles = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCF", "C1CC1")
    distance = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.80, dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[0, 1] = distance[1, 0] = 0.35
    return WorkflowState(
        phase=WorkflowPhase.OPTIMIZED,
        records=[
            {"id": f"mol-{index}", "smiles": value, "source_row": index}
            for index, value in enumerate(smiles)
        ],
        molecules=[Chem.MolFromSmiles(value) for value in smiles],
        similarity=FakeGpuResult(1.0 - distance),
        clusters=[[index] for index in range(CANDIDATE_COUNT)],
    )


def two_revision_context() -> ObjectiveContext:
    candidates = tuple(
        ObjectiveCandidate(
            molecule_id=f"candidate-{index}",
            molecule_index=index,
            source_row=index,
            cluster_id=index,
        )
        for index in range(CANDIDATE_COUNT)
    )
    distance = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.90, dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[0, 1] = distance[1, 0] = 0.30
    distance[2, 3] = distance[3, 2] = 0.40
    distance.setflags(write=False)
    return ObjectiveContext(
        candidates=candidates,
        baseline_ids=("candidate-0", "candidate-1", "candidate-2", "candidate-3"),
        baseline_score=0.30,
        benchmark_score=0.90,
        target_score=0.75,
        distance_matrix=distance,
    )


def forged_nonachieved_attempt(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...],
    *,
    score: float | bool | None = None,
) -> ObjectiveAttempt:
    computed_score, limiting_pair = objective_challenge._score_panel(
        context, selected_ids
    )
    return ObjectiveAttempt(
        attempt_number=1,
        selected_ids=selected_ids,
        decision_basis="Forged state used to verify ranker validation.",
        score=computed_score if score is None else score,
        limiting_pair=limiting_pair,
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
    second = evaluate_diverse_panel(
        context,
        ("mol-0", "mol-2", "mol-4", "mol-6"),
        attempt_number=2,
        decision_basis="Remove the closest baseline analogue.",
    )
    return state, finalize_objective_run(context, (first, second))


def test_constants_keep_the_challenge_visually_bounded():
    assert (CANDIDATE_COUNT, PANEL_SIZE, MAX_ATTEMPTS) == (8, 4, 3)


def test_build_context_uses_eight_distinct_mmff_eligible_clusters():
    context = build_objective_context(optimized_state())

    assert len(context.candidates) == 8
    assert len({candidate.cluster_id for candidate in context.candidates}) == 8
    assert context.baseline_ids == ("mol-0", "mol-1", "mol-2", "mol-3")
    assert context.baseline_score == pytest.approx(0.35)
    assert context.benchmark_score == pytest.approx(0.80)
    assert context.target_score == pytest.approx(0.71)
    assert context.distance_matrix.flags.writeable is False


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
    success = evaluate_diverse_panel(
        context,
        ("mol-0", "mol-2", "mol-4", "mol-6"),
        attempt_number=2,
        decision_basis="Remove the limiting analogue.",
    )

    achieved = finalize_objective_run(context, (miss, success))
    misses = tuple(
        evaluate_diverse_panel(
            context,
            context.baseline_ids,
            attempt_number=number,
            decision_basis="Baseline remains redundant.",
        )
        for number in (1, 2, 3)
    )
    exhausted = finalize_objective_run(context, misses)

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
    assert payload["attempts"][0]["limiting_pair"] == ["mol-0", "mol-1"]
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == record.payload_json


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

    assert payload["attempts"][0]["selected_swap"] is None
    assert payload["attempts"][1]["selected_swap"] == {
        "replace_id": selected.replace_id,
        "replacement_id": selected.replacement_id,
        "resulting_ids": list(selected.resulting_ids),
        "predicted_score": selected.predicted_score,
        "score_delta": selected.score_delta,
        "limiting_pair": list(selected.limiting_pair),
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


def test_rank_legal_swaps_returns_none_after_success_and_stays_fixture_agnostic():
    context = build_objective_context(optimized_state())
    achieved = evaluate_diverse_panel(
        context,
        ("mol-0", "mol-2", "mol-4", "mol-6"),
        attempt_number=1,
        decision_basis="Use a diverse four-cluster panel.",
    )
    out_of_pool_achieved = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("outside-0", "outside-1", "outside-2", "outside-3"),
        decision_basis="An already completed attempt needs no replacement.",
        score=1.0,
        limiting_pair=("outside-0", "outside-1"),
        constraints_passed=False,
        achieved=True,
    )
    source = Path(objective_challenge.__file__).read_text(encoding="utf-8")

    assert achieved.achieved is True
    assert rank_legal_swaps(context, achieved) == ()
    assert rank_legal_swaps(context, out_of_pool_achieved) == ()
    assert all(f"mol-{index}" not in source for index in range(CANDIDATE_COUNT))
    assert "CHEMBL" not in source


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


def test_rank_legal_swaps_rejects_cross_context_cluster_conflicts():
    context = build_objective_context(optimized_state())
    conflicting_context = ObjectiveContext(
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

    with pytest.raises(ValueError):
        rank_legal_swaps(
            conflicting_context,
            forged_nonachieved_attempt(
                conflicting_context, conflicting_context.baseline_ids
            ),
        )


def test_rank_legal_swaps_rejects_boolean_current_scores_explicitly():
    context = build_objective_context(optimized_state())

    with pytest.raises(ValueError, match="non-boolean"):
        rank_legal_swaps(
            context,
            forged_nonachieved_attempt(context, context.baseline_ids, score=True),
        )


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
