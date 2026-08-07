import itertools
from dataclasses import replace

import numpy as np
from rdkit import Chem

from chemistry_workflow import WorkflowPhase, WorkflowState
import objective_challenge
from objective_challenge import CANDIDATE_COUNT, ObjectiveCandidate, ObjectiveContext
from objective_challenge import SCORE_SCALE, score_key
from objective_challenge import (
    TerminationReason,
    accepted_maxima,
    baseline_terminal_run,
    build_action_menu,
    evaluate_selected_swap,
    finalize_no_legal_swap,
    measure_panel,
    terminal_objective_run,
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


def optimized_state(baseline_optimal: bool = False) -> WorkflowState:
    smiles = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCF", "C1CC1")
    distance = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.80, dtype=float)
    np.fill_diagonal(distance, 0.0)
    if not baseline_optimal:
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


def context_from_distance(
    distance: np.ndarray,
    *,
    baseline_ids: tuple[str, str, str, str] = (
        "mol-0",
        "mol-1",
        "mol-2",
        "mol-3",
    ),
    target_score: float = 0.75,
) -> ObjectiveContext:
    values = np.array(distance, dtype=np.float64, copy=True)
    values.setflags(write=False)
    candidates = tuple(
        ObjectiveCandidate(f"mol-{index}", index, index, index)
        for index in range(CANDIDATE_COUNT)
    )
    provisional = ObjectiveContext(
        candidates, baseline_ids, 0.0, 0.0, target_score, values
    )
    baseline = objective_challenge.measure_panel(provisional, baseline_ids).score
    benchmark = max(
        objective_challenge.measure_panel(provisional, panel).score
        for panel in itertools.combinations(
            tuple(item.molecule_id for item in candidates), 4
        )
    )
    return replace(
        provisional, baseline_score=baseline, benchmark_score=benchmark
    )


def controlled_context(
    *,
    distances: dict[tuple[str, str], float],
    default_distance: float,
    target_score: float = 0.75,
) -> ObjectiveContext:
    matrix = np.full(
        (CANDIDATE_COUNT, CANDIDATE_COUNT), default_distance, dtype=np.float64
    )
    np.fill_diagonal(matrix, 0.0)
    for (first_id, second_id), value in distances.items():
        first = int(first_id.removeprefix("mol-"))
        second = int(second_id.removeprefix("mol-"))
        matrix[first, second] = matrix[second, first] = value
    return context_from_distance(matrix, target_score=target_score)


def controlled_context_with_ranked_swaps() -> ObjectiveContext:
    """Four isolated legal improvements with known descending score keys."""
    distances = {("mol-0", "mol-1"): 0.10}
    for replacement, blocked, quality in (
        ("mol-4", "mol-0", 0.90),
        ("mol-5", "mol-0", 0.80),
        ("mol-6", "mol-1", 0.70),
        ("mol-7", "mol-1", 0.60),
    ):
        distances[(replacement, blocked)] = 0.10
        for baseline in ("mol-0", "mol-1", "mol-2", "mol-3"):
            if baseline != blocked:
                distances[(replacement, baseline)] = quality
    return controlled_context(
        distances=distances, default_distance=0.95, target_score=0.55
    )


def controlled_context_without_improving_swaps() -> ObjectiveContext:
    return controlled_context(
        distances={}, default_distance=0.50, target_score=0.60
    )


def controlled_context_with_action_count(count: int) -> ObjectiveContext:
    if type(count) is not int or not 0 <= count <= 3:
        raise ValueError("Fixture action count must be in [0, 3].")
    distances = {("mol-0", "mol-1"): 0.10}
    specifications = (
        ("mol-4", "mol-0", 0.90),
        ("mol-5", "mol-0", 0.80),
        ("mol-6", "mol-1", 0.70),
        ("mol-7", "mol-1", 0.10),
    )
    for index, (replacement, blocked, quality) in enumerate(specifications):
        effective_quality = quality if index < count else 0.10
        distances[(replacement, blocked)] = 0.10
        for baseline in ("mol-0", "mol-1", "mol-2", "mol-3"):
            if baseline != blocked:
                distances[(replacement, baseline)] = effective_quality
    return controlled_context(
        distances=distances, default_distance=0.95, target_score=0.95
    )


def boundary_policy_context(candidate: float, current: float) -> ObjectiveContext:
    distances = {("mol-0", "mol-1"): current}
    for replacement in ("mol-4", "mol-5", "mol-6", "mol-7"):
        quality = candidate if replacement == "mol-4" else current
        distances[(replacement, "mol-0")] = current
        for baseline in ("mol-1", "mol-2", "mol-3"):
            distances[(replacement, baseline)] = quality
    target = (
        candidate
        if score_key(candidate) > score_key(current)
        else (score_key(current) + 1) / SCORE_SCALE
    )
    return controlled_context(
        distances=distances, default_distance=0.95, target_score=target
    )


def controlled_context_with_tied_paths(all_paths_reach: bool) -> ObjectiveContext:
    distances = {
        ("mol-0", "mol-1"): 0.10,
        ("mol-4", "mol-0"): 0.10,
        ("mol-4", "mol-1"): 0.40,
        ("mol-5", "mol-0"): 0.40,
        ("mol-5", "mol-1"): 0.10,
        ("mol-6", "mol-0"): 0.10,
        ("mol-6", "mol-1"): 0.10,
        ("mol-6", "mol-5"): 0.10,
        ("mol-6", "mol-7"): 0.10,
        ("mol-4", "mol-7"): 0.10,
        ("mol-7", "mol-0"): 0.10,
        ("mol-7", "mol-1"): 0.10,
        ("mol-4", "mol-5"): 0.10,
        ("mol-4", "mol-6"): 0.90,
        ("mol-5", "mol-7"): 0.90 if all_paths_reach else 0.50,
    }
    return controlled_context(
        distances=distances, default_distance=0.90, target_score=0.80
    )


def controlled_context_with_three_misses() -> ObjectiveContext:
    matrix = np.array(
        [
            [0, .2, .4, .2, .2, .1, .4, .8],
            [.2, 0, .8, .3, .1, .3, .4, .2],
            [.4, .8, 0, .5, .6, .8, .6, .2],
            [.2, .3, .5, 0, .6, .8, .7, .8],
            [.2, .1, .6, .6, 0, .6, .6, .6],
            [.1, .3, .8, .8, .6, 0, .2, .9],
            [.4, .4, .6, .7, .6, .2, 0, .2],
            [.8, .2, .2, .8, .6, .9, .2, 0],
        ],
        dtype=np.float64,
    )
    return context_from_distance(matrix, target_score=1.0)


def terminal_fixture(reason: str | TerminationReason, attempt_count: int):
    reason = TerminationReason(reason)
    if reason is TerminationReason.BASELINE_ALREADY_OPTIMAL:
        if attempt_count != 0:
            raise ValueError("Baseline terminal fixture requires zero attempts.")
        state = optimized_state(baseline_optimal=True)
        return baseline_terminal_run(objective_challenge.build_objective_context(state))
    if reason is TerminationReason.NO_LEGAL_IMPROVING_SWAP:
        if attempt_count != 0:
            raise ValueError("No-legal terminal fixture requires zero attempts.")
        context = controlled_context_without_improving_swaps()
        current = measure_panel(context, context.baseline_ids)
        menu = build_action_menu(context, current, 0)
        return finalize_no_legal_swap(context, (), current, menu)

    context = (
        controlled_context_with_ranked_swaps()
        if reason is TerminationReason.TARGET_ACHIEVED
        else controlled_context_with_three_misses()
    )
    attempts = []
    current = measure_panel(context, context.baseline_ids)
    for number in range(1, attempt_count + 1):
        menu = build_action_menu(context, current, number - 1)
        attempt = evaluate_selected_swap(context, menu, accepted_maxima(menu)[0], number)
        attempts.append(attempt)
        current = attempt.measurement
    return terminal_objective_run(context, tuple(attempts), reason)


BOUNDARY_CASES = (
    (0.5000000000004, 0.5, False),
    (0.5000000000005, 0.5, True),
    (0.5000000000006, 0.5, True),
)


TARGET_BOUNDARY_CASES = (
    (0.4999999999994, 0.5, False),
    (0.4999999999995, 0.5, True),
    (0.5, 0.5, True),
    (0.5000000000005, 0.5, True),
)
