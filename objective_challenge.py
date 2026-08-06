"""Pure, bounded molecular-diversity objective built from retained workflow evidence."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chemistry_workflow import (
    EvidenceRecord,
    WorkflowPhase,
    WorkflowState,
    eligible_representative_groups,
    validated_similarity_matrix,
)


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


def _score_panel(
    context: ObjectiveContext, selected_ids: tuple[str, ...]
) -> tuple[float, tuple[str, str]]:
    positions = {
        candidate.molecule_id: position
        for position, candidate in enumerate(context.candidates)
    }
    scored_pairs = []
    for first_id, second_id in itertools.combinations(selected_ids, 2):
        pair = tuple(sorted((first_id, second_id)))
        distance = float(
            context.distance_matrix[positions[first_id], positions[second_id]]
        )
        scored_pairs.append((distance, pair))
    return min(scored_pairs, key=lambda item: (item[0], item[1]))


def build_objective_context(state: WorkflowState) -> ObjectiveContext:
    """Build the fixed eight-candidate challenge and its attainable score bound."""
    if state.phase is not WorkflowPhase.OPTIMIZED:
        raise RuntimeError("Objective challenge requires an OPTIMIZED workflow.")
    similarity = validated_similarity_matrix(state)
    groups = eligible_representative_groups(state)
    if len(groups) < CANDIDATE_COUNT:
        raise RuntimeError(
            "Objective challenge requires eight eligible distinct clusters."
        )
    candidates = tuple(
        ObjectiveCandidate(
            molecule_id=str(group["members"][0]["molecule_id"]),
            molecule_index=int(group["members"][0]["molecule_index"]),
            source_row=int(group["members"][0]["source_row"]),
            cluster_id=int(group["cluster_id"]),
        )
        for group in groups[:CANDIDATE_COUNT]
    )
    indices = [candidate.molecule_index for candidate in candidates]
    distance_matrix = np.array(
        1.0 - similarity[np.ix_(indices, indices)], dtype=float, copy=True
    )
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0
    if not np.isfinite(distance_matrix).all() or np.any(
        (distance_matrix < -1e-7) | (distance_matrix > 1.0 + 1e-7)
    ):
        raise RuntimeError("Objective distance matrix invariants are invalid.")
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    distance_matrix.setflags(write=False)

    provisional = ObjectiveContext(
        candidates=candidates,
        baseline_ids=tuple(candidate.molecule_id for candidate in candidates[:PANEL_SIZE]),
        baseline_score=0.0,
        benchmark_score=0.0,
        target_score=0.0,
        distance_matrix=distance_matrix,
    )
    baseline_score, _pair = _score_panel(provisional, provisional.baseline_ids)
    panels = itertools.combinations(
        sorted(candidate.molecule_id for candidate in candidates), PANEL_SIZE
    )
    benchmark_score = max(_score_panel(provisional, panel)[0] for panel in panels)
    target_score = baseline_score + TARGET_FRACTION * (
        benchmark_score - baseline_score
    )
    return ObjectiveContext(
        candidates=candidates,
        baseline_ids=provisional.baseline_ids,
        baseline_score=float(baseline_score),
        benchmark_score=float(benchmark_score),
        target_score=float(target_score),
        distance_matrix=distance_matrix,
    )


def evaluate_diverse_panel(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...] | list[str],
    *,
    attempt_number: int,
    decision_basis: str,
) -> ObjectiveAttempt:
    """Validate and score one proposed panel without mutating workflow state."""
    panel = tuple(selected_ids)
    if len(panel) != PANEL_SIZE or len(set(panel)) != PANEL_SIZE:
        raise ValueError("Objective proposals require four unique molecule IDs.")
    candidates = {candidate.molecule_id: candidate for candidate in context.candidates}
    if any(molecule_id not in candidates for molecule_id in panel):
        raise ValueError("Objective proposal contains an out-of-pool molecule ID.")
    cluster_ids = {candidates[molecule_id].cluster_id for molecule_id in panel}
    if len(cluster_ids) != PANEL_SIZE:
        raise ValueError("Objective proposal must use four distinct clusters.")
    if type(attempt_number) is not int or not 1 <= attempt_number <= MAX_ATTEMPTS:
        raise ValueError("Objective attempt number is outside the accepted bound.")
    basis = decision_basis.strip() if isinstance(decision_basis, str) else ""
    if not basis or len(basis) > 240 or any(character in basis for character in "\r\n`"):
        raise ValueError("Objective decision basis is invalid.")
    score, limiting_pair = _score_panel(context, panel)
    return ObjectiveAttempt(
        attempt_number=attempt_number,
        selected_ids=panel,
        decision_basis=basis,
        score=float(score),
        limiting_pair=limiting_pair,
        constraints_passed=True,
        achieved=bool(score + 1e-12 >= context.target_score),
    )


def finalize_objective_run(
    context: ObjectiveContext, attempts: tuple[ObjectiveAttempt, ...]
) -> ObjectiveRun:
    """Close a successful attempt sequence or an exact three-attempt miss."""
    if not attempts or len(attempts) > MAX_ATTEMPTS:
        raise ValueError("Objective run has an invalid accepted-attempt count.")
    if tuple(attempt.attempt_number for attempt in attempts) != tuple(
        range(1, len(attempts) + 1)
    ):
        raise ValueError("Objective attempts are not sequential.")
    achieved_positions = [index for index, attempt in enumerate(attempts) if attempt.achieved]
    if achieved_positions and achieved_positions != [len(attempts) - 1]:
        raise ValueError("Objective run continued after the target was achieved.")
    achieved = bool(attempts[-1].achieved)
    if not achieved and len(attempts) != MAX_ATTEMPTS:
        raise ValueError("An unsuccessful objective run requires three attempts.")
    best = max(attempts, key=lambda attempt: (attempt.score, -attempt.attempt_number))
    return ObjectiveRun(
        context=context,
        attempts=attempts,
        achieved=achieved,
        termination_reason="target_achieved" if achieved else "attempt_limit_reached",
        final_ids=best.selected_ids,
        final_score=best.score,
    )


def no_improvement_run(context: ObjectiveContext) -> ObjectiveRun:
    """Represent the explicit case where the current policy is already optimal."""
    if not np.isclose(context.baseline_score, context.benchmark_score, rtol=0.0, atol=1e-12):
        raise ValueError("A no-improvement run requires an optimal baseline.")
    return ObjectiveRun(
        context=context,
        attempts=(),
        achieved=True,
        termination_reason="baseline_already_optimal",
        final_ids=context.baseline_ids,
        final_score=context.baseline_score,
    )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_objective_evidence(run: ObjectiveRun) -> EvidenceRecord:
    """Build immutable O01 without exposing the hidden benchmark panel."""
    context = run.context
    payload = {
        "objective": "maximize minimum pairwise Morgan/Tanimoto distance",
        "score_definition": "D_min=min(1-Tanimoto(i,j)) over selected pairs",
        "candidate_pool_rule": "first MMFF94-eligible member from each of eight largest eligible clusters",
        "candidate_ids": [candidate.molecule_id for candidate in context.candidates],
        "candidate_cluster_ids": [candidate.cluster_id for candidate in context.candidates],
        "panel_size": PANEL_SIZE,
        "attempt_limit": MAX_ATTEMPTS,
        "baseline_ids": list(context.baseline_ids),
        "baseline_score": context.baseline_score,
        "benchmark_score": context.benchmark_score,
        "target_rule": "baseline plus 80 percent of attainable improvement",
        "target_score": context.target_score,
        "attempts": [
            {
                "attempt_number": attempt.attempt_number,
                "selected_ids": list(attempt.selected_ids),
                "decision_basis": attempt.decision_basis,
                "score": attempt.score,
                "limiting_pair": list(attempt.limiting_pair),
                "constraints_passed": attempt.constraints_passed,
                "achieved": attempt.achieved,
            }
            for attempt in run.attempts
        ],
        "termination_reason": run.termination_reason,
        "final_ids": list(run.final_ids),
        "final_score": run.final_score,
        "achieved": run.achieved,
        "claim_scope": "structural diversity within the bounded candidate pool",
    }
    return EvidenceRecord(
        "O01",
        "Objective-driven panel selection",
        _canonical_json(payload),
        "deterministic Python evaluation of nvMolKit Tanimoto evidence",
    )


def objective_figures(run: ObjectiveRun, state: WorkflowState) -> tuple[Any, ...]:
    """Render the compact attempt trajectory, final structures, and heatmap."""
    from matplotlib.figure import Figure
    from rdkit.Chem import Draw

    context = run.context
    trajectory = Figure(figsize=(6.2, 2.8), layout="constrained")
    axes = trajectory.subplots()
    labels = ["Baseline", *(f"Attempt {attempt.attempt_number}" for attempt in run.attempts)]
    scores = [context.baseline_score, *(attempt.score for attempt in run.attempts)]
    axes.plot(range(len(scores)), scores, color="#444444", marker="o")
    axes.axhline(context.target_score, color="#76B900", linestyle="--", label="Target")
    axes.set_xticks(range(len(scores)), labels)
    axes.set_ylabel("Minimum pairwise distance")
    axes.set_title("Objective score trajectory")
    axes.set_ylim(0.0, min(1.0, max(scores + [context.target_score]) + 0.1))
    axes.legend(loc="best")

    candidate_by_id = {candidate.molecule_id: candidate for candidate in context.candidates}
    final_molecules = [
        state.molecules[candidate_by_id[molecule_id].molecule_index]
        for molecule_id in run.final_ids
    ]
    structures = Draw.MolsToGridImage(
        final_molecules,
        legends=list(run.final_ids),
        molsPerRow=PANEL_SIZE,
        subImgSize=(180, 150),
    )

    positions = {
        candidate.molecule_id: position
        for position, candidate in enumerate(context.candidates)
    }
    selected_positions = [positions[molecule_id] for molecule_id in run.final_ids]
    similarity = 1.0 - context.distance_matrix[np.ix_(selected_positions, selected_positions)]
    heatmap = Figure(figsize=(4.5, 3.8), layout="constrained")
    heatmap_axes = heatmap.subplots()
    image = heatmap_axes.imshow(similarity, vmin=0.0, vmax=1.0, cmap="viridis")
    heatmap_axes.set_xticks(range(PANEL_SIZE), run.final_ids, rotation=25, ha="right")
    heatmap_axes.set_yticks(range(PANEL_SIZE), run.final_ids)
    heatmap_axes.set_title("Final-panel Tanimoto similarity")
    heatmap.colorbar(image, ax=heatmap_axes, label="Tanimoto similarity")
    return trajectory, structures, heatmap
