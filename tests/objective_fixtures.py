import itertools
from dataclasses import replace

import numpy as np
from rdkit import Chem

from chemistry_workflow import WorkflowPhase, WorkflowState
import objective_challenge
from objective_challenge import CANDIDATE_COUNT, ObjectiveCandidate, ObjectiveContext


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
