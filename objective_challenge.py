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
SUGGESTION_LIMIT = 3
SCORE_TOLERANCE = 1e-12
SCORE_SCALE = 10**12


def score_key(value: float | np.floating) -> int:
    """Return the canonical half-up quantized unit for a valid objective score."""
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        raise ValueError("Objective score must be a finite float in [0, 1].")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("Objective score must be a finite float in [0, 1].")
    numerator, denominator = normalized.as_integer_ratio()
    return (2 * numerator * SCORE_SCALE + denominator) // (2 * denominator)


def is_strict_improvement(candidate: float, current: float) -> bool:
    """Return whether candidate is strictly better in canonical score units."""
    return score_key(candidate) > score_key(current)


def target_is_achieved(score: float, target: float) -> bool:
    """Return whether score reaches target in canonical score units."""
    return score_key(score) >= score_key(target)


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

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or (
            len(self.candidates) != CANDIDATE_COUNT
            or any(
                type(candidate) is not ObjectiveCandidate
                for candidate in self.candidates
            )
        ):
            raise ValueError("Objective context requires eight exact candidates.")
        molecule_ids = tuple(candidate.molecule_id for candidate in self.candidates)
        if (
            any(
                type(molecule_id) is not str or not molecule_id
                for molecule_id in molecule_ids
            )
            or len(set(molecule_ids)) != CANDIDATE_COUNT
        ):
            raise ValueError(
                "Objective candidate molecule IDs must be unique nonempty strings."
            )
        molecule_indices = tuple(
            candidate.molecule_index for candidate in self.candidates
        )
        source_rows = tuple(candidate.source_row for candidate in self.candidates)
        if any(
            type(value) is not int or value < 0
            for value in (*molecule_indices, *source_rows)
        ) or any(
            len(set(values)) != CANDIDATE_COUNT
            for values in (molecule_indices, source_rows)
        ):
            raise ValueError(
                "Objective candidate provenance must use unique nonnegative integers."
            )
        cluster_ids = tuple(candidate.cluster_id for candidate in self.candidates)
        if (
            any(
                type(cluster_id) is not int or cluster_id < 0
                for cluster_id in cluster_ids
            )
            or len(set(cluster_ids)) != CANDIDATE_COUNT
        ):
            raise ValueError("Objective candidates require eight distinct cluster IDs.")
        candidates_by_id = {
            candidate.molecule_id: candidate for candidate in self.candidates
        }
        if (
            type(self.baseline_ids) is not tuple
            or len(self.baseline_ids) != PANEL_SIZE
            or any(type(molecule_id) is not str for molecule_id in self.baseline_ids)
            or len(set(self.baseline_ids)) != PANEL_SIZE
            or any(
                molecule_id not in candidates_by_id
                for molecule_id in self.baseline_ids
            )
            or len(
                {
                    candidates_by_id[molecule_id].cluster_id
                    for molecule_id in self.baseline_ids
                }
            )
            != PANEL_SIZE
        ):
            raise ValueError(
                "Objective baseline requires four unique in-pool IDs from distinct clusters."
            )
        try:
            source = np.asarray(self.distance_matrix)
            if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
                source.dtype, np.complexfloating
            ):
                raise ValueError
            distance_matrix = np.array(
                source, dtype=np.float64, copy=True
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Objective distance matrix must be numeric.") from error
        if distance_matrix.shape != (CANDIDATE_COUNT, CANDIDATE_COUNT):
            raise ValueError("Objective distance matrix has an invalid shape.")
        if not np.isfinite(distance_matrix).all():
            raise ValueError("Objective distance matrix must contain finite values.")
        if np.any((distance_matrix < 0.0) | (distance_matrix > 1.0)):
            raise ValueError("Objective distance matrix values must be in [0, 1].")
        if not np.array_equal(distance_matrix, distance_matrix.T):
            raise ValueError("Objective distance matrix must be symmetric.")
        if np.any(np.diag(distance_matrix) != 0.0):
            raise ValueError("Objective distance matrix must have a zero diagonal.")
        distance_matrix.setflags(write=False)
        object.__setattr__(self, "distance_matrix", distance_matrix)


@dataclass(frozen=True)
class PanelMeasurement:
    selected_ids: tuple[str, ...]
    score: float
    score_key: int
    limiting_pairs: tuple[tuple[str, str], ...]
    achieved: bool


@dataclass(frozen=True)
class ObjectiveSwap:
    replace_id: str
    replacement_id: str
    resulting_ids: tuple[str, ...]
    predicted_score: float
    score_delta: float
    limiting_pair: tuple[str, str]


@dataclass(frozen=True)
class ObjectiveAttempt:
    attempt_number: int
    selected_ids: tuple[str, ...]
    decision_basis: str
    score: float
    limiting_pair: tuple[str, str]
    constraints_passed: bool
    achieved: bool
    selected_swap: ObjectiveSwap | None = None


@dataclass(frozen=True)
class ObjectiveRun:
    context: ObjectiveContext
    attempts: tuple[ObjectiveAttempt, ...]
    achieved: bool
    termination_reason: str
    final_ids: tuple[str, ...]
    final_score: float


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
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0
    if not np.isfinite(distance_matrix).all() or np.any(
        (distance_matrix < -1e-7) | (distance_matrix > 1.0 + 1e-7)
    ):
        raise RuntimeError("Objective distance matrix invariants are invalid.")
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0
    distance_matrix.setflags(write=False)

    provisional = ObjectiveContext(
        candidates=candidates,
        baseline_ids=tuple(candidate.molecule_id for candidate in candidates[:PANEL_SIZE]),
        baseline_score=0.0,
        benchmark_score=0.0,
        target_score=0.0,
        distance_matrix=distance_matrix,
    )
    baseline = measure_panel(provisional, provisional.baseline_ids)
    panels = itertools.combinations(
        sorted(candidate.molecule_id for candidate in candidates), PANEL_SIZE
    )
    benchmark = max(
        (measure_panel(provisional, panel) for panel in panels),
        key=lambda measurement: (
            measurement.score_key,
            measurement.score,
            measurement.selected_ids,
        ),
    )
    baseline_score = baseline.score
    benchmark_score = benchmark.score
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


def _validated_panel(
    context: ObjectiveContext, selected_ids: tuple[str, ...] | list[str]
) -> tuple[tuple[str, ...], dict[str, ObjectiveCandidate]]:
    """Return a structurally legal panel and its candidate lookup."""
    panel = tuple(selected_ids)
    if len(panel) != PANEL_SIZE or len(set(panel)) != PANEL_SIZE:
        raise ValueError("Objective proposals require four unique molecule IDs.")
    candidates = {candidate.molecule_id: candidate for candidate in context.candidates}
    if any(molecule_id not in candidates for molecule_id in panel):
        raise ValueError("Objective proposal contains an out-of-pool molecule ID.")
    if (
        len({candidates[molecule_id].cluster_id for molecule_id in panel})
        != PANEL_SIZE
    ):
        raise ValueError("Objective proposal must use four distinct clusters.")
    return panel, candidates


def _panel_distances(
    context: ObjectiveContext, panel: tuple[str, ...]
):
    """Yield the six validated pair distances for one structurally legal panel."""
    positions = {
        candidate.molecule_id: position
        for position, candidate in enumerate(context.candidates)
    }
    for first_id, second_id in itertools.combinations(panel, 2):
        try:
            distance = float(
                context.distance_matrix[
                    positions[first_id], positions[second_id]
                ]
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Objective pair distance is invalid.") from error
        if not np.isfinite(distance) or not 0.0 <= distance <= 1.0:
            raise ValueError(
                "Objective pair distance must be finite and in [0, 1]."
            )
        yield first_id, second_id, distance


def measure_panel(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...] | list[str],
) -> PanelMeasurement:
    """Validate and measure one panel with canonical quantized score semantics."""
    panel, _ = _validated_panel(context, selected_ids)
    scored = tuple(
        (
            score_key(distance),
            distance,
            tuple(sorted((first_id, second_id))),
        )
        for first_id, second_id, distance in _panel_distances(context, panel)
    )
    minimum_key = min(item[0] for item in scored)
    limiting_pairs = tuple(
        sorted(item[2] for item in scored if item[0] == minimum_key)
    )
    raw_score = min(item[1] for item in scored if item[0] == minimum_key)
    return PanelMeasurement(
        selected_ids=panel,
        score=raw_score,
        score_key=minimum_key,
        limiting_pairs=limiting_pairs,
        achieved=target_is_achieved(raw_score, context.target_score),
    )


def _score_panel(
    context: ObjectiveContext, selected_ids: tuple[str, ...]
) -> tuple[float, tuple[str, str]]:
    """Return the legacy singular-pair view of a canonical panel measurement."""
    measurement = measure_panel(context, selected_ids)
    return measurement.score, measurement.limiting_pairs[0]


def _validated_selected_swap(
    context: ObjectiveContext, selected_swap: object
) -> tuple[str, ...]:
    """Validate canonical provenance for a scored legal one-ID panel swap."""
    if type(selected_swap) is not ObjectiveSwap:
        raise ValueError("Objective selected swap must use the exact swap type.")
    candidate_ids = {candidate.molecule_id for candidate in context.candidates}
    if (
        type(selected_swap.replace_id) is not str
        or type(selected_swap.replacement_id) is not str
        or selected_swap.replace_id not in candidate_ids
        or selected_swap.replacement_id not in candidate_ids
    ):
        raise ValueError("Objective selected swap IDs must be exact in-pool strings.")
    if type(selected_swap.resulting_ids) is not tuple or any(
        type(molecule_id) is not str for molecule_id in selected_swap.resulting_ids
    ):
        raise ValueError("Objective selected swap resulting IDs must be an exact string tuple.")
    selected_swap_panel, _ = _validated_panel(context, selected_swap.resulting_ids)
    if (
        type(selected_swap.limiting_pair) is not tuple
        or len(selected_swap.limiting_pair) != 2
        or any(type(molecule_id) is not str for molecule_id in selected_swap.limiting_pair)
    ):
        raise ValueError("Objective selected swap limiting pair must be an exact string tuple.")
    if (
        type(selected_swap.predicted_score) is not float
        or not np.isfinite(selected_swap.predicted_score)
        or type(selected_swap.score_delta) is not float
        or not np.isfinite(selected_swap.score_delta)
    ):
        raise ValueError("Objective selected swap scores must be finite built-in floats.")
    if (
        selected_swap.replacement_id not in selected_swap_panel
        or selected_swap.replace_id in selected_swap_panel
    ):
        raise ValueError("Objective selected swap IDs do not describe the resulting panel.")
    replacement_position = selected_swap_panel.index(selected_swap.replacement_id)
    predecessor_panel = (
        selected_swap_panel[:replacement_position]
        + (selected_swap.replace_id,)
        + selected_swap_panel[replacement_position + 1 :]
    )
    predecessor_panel, _ = _validated_panel(context, predecessor_panel)
    predecessor = measure_panel(context, predecessor_panel)
    resulting = measure_panel(context, selected_swap_panel)
    expected_delta = resulting.score - predecessor.score
    if (
        not is_strict_improvement(resulting.score, predecessor.score)
        or selected_swap.score_delta != expected_delta
    ):
        raise ValueError("Objective selected swap score delta is invalid.")
    if selected_swap.predicted_score != resulting.score:
        raise ValueError("Objective selected swap predicted score does not match the panel.")
    if selected_swap.limiting_pair != resulting.limiting_pairs[0]:
        raise ValueError("Objective selected swap limiting pair does not match the panel.")
    return selected_swap_panel


def evaluate_diverse_panel(
    context: ObjectiveContext,
    selected_ids: tuple[str, ...] | list[str],
    *,
    attempt_number: int,
    decision_basis: str,
    selected_swap: ObjectiveSwap | None = None,
) -> ObjectiveAttempt:
    """Validate and score one proposed panel without mutating workflow state."""
    panel, _ = _validated_panel(context, selected_ids)
    if type(attempt_number) is not int or not 1 <= attempt_number <= MAX_ATTEMPTS:
        raise ValueError("Objective attempt number is outside the accepted bound.")
    basis = decision_basis.strip() if isinstance(decision_basis, str) else ""
    if not basis or len(basis) > 240 or any(character in basis for character in "\r\n`"):
        raise ValueError("Objective decision basis is invalid.")
    measurement = measure_panel(context, panel)
    if selected_swap is not None:
        selected_swap_panel = _validated_selected_swap(context, selected_swap)
        if set(panel) != set(selected_swap_panel):
            raise ValueError("Objective selected swap does not match the selected panel.")
    return ObjectiveAttempt(
        attempt_number=attempt_number,
        selected_ids=panel,
        decision_basis=basis,
        score=measurement.score,
        limiting_pair=measurement.limiting_pairs[0],
        constraints_passed=True,
        achieved=measurement.achieved,
        selected_swap=selected_swap,
    )


def rank_legal_swaps(
    context: ObjectiveContext, current: ObjectiveAttempt
) -> tuple[ObjectiveSwap, ...]:
    """Rank legal one-ID panel swaps by their predicted objective improvement."""
    if type(context) is not ObjectiveContext or type(current) is not ObjectiveAttempt:
        raise ValueError("Objective swap ranking requires exact objective types.")
    if current.achieved:
        return ()
    panel, candidates = _validated_panel(context, current.selected_ids)
    recomputed = measure_panel(context, panel)
    if type(current.score) is not float or not np.isfinite(current.score):
        raise ValueError(
            "Objective attempt score must be a finite non-boolean built-in float."
        )
    if current.score != recomputed.score:
        raise ValueError("Objective attempt score does not match its panel.")
    if current.limiting_pair != recomputed.limiting_pairs[0]:
        raise ValueError("Objective attempt limiting pair does not match its panel.")
    if current.constraints_passed is not True:
        raise ValueError("Objective attempt constraints must be passed.")
    if recomputed.achieved:
        raise ValueError("Objective attempt below-target state is inconsistent.")

    suggestions = []
    current_ids = set(panel)
    for replace_position, replace_id in enumerate(panel):
        for replacement_id, replacement in candidates.items():
            if replacement_id in current_ids:
                continue
            resulting_ids = (
                panel[:replace_position]
                + (replacement_id,)
                + panel[replace_position + 1 :]
            )
            try:
                resulting_ids, _ = _validated_panel(context, resulting_ids)
            except ValueError:
                continue
            predicted = measure_panel(context, resulting_ids)
            score_delta = predicted.score - recomputed.score
            if not is_strict_improvement(predicted.score, recomputed.score):
                continue
            suggestions.append(
                ObjectiveSwap(
                    replace_id=replace_id,
                    replacement_id=replacement.molecule_id,
                    resulting_ids=resulting_ids,
                    predicted_score=predicted.score,
                    score_delta=float(score_delta),
                    limiting_pair=predicted.limiting_pairs[0],
                )
            )

    suggestions.sort(
        key=lambda suggestion: (
            -score_key(suggestion.predicted_score),
            suggestion.replace_id,
            suggestion.replacement_id,
            suggestion.resulting_ids,
        )
    )
    target_reaching = [
        suggestion
        for suggestion in suggestions
        if target_is_achieved(suggestion.predicted_score, context.target_score)
    ]
    return tuple((target_reaching or suggestions)[:SUGGESTION_LIMIT])


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
    best = max(
        attempts,
        key=lambda attempt: (score_key(attempt.score), -attempt.attempt_number),
    )
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
    if score_key(context.baseline_score) != score_key(context.benchmark_score):
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
    attempt_payloads = []
    for attempt in run.attempts:
        measurement = measure_panel(context, attempt.selected_ids)
        selected_swap_payload = None
        if attempt.selected_swap is not None:
            selected_swap = attempt.selected_swap
            swap_measurement = measure_panel(context, selected_swap.resulting_ids)
            selected_swap_payload = {
                "replace_id": selected_swap.replace_id,
                "replacement_id": selected_swap.replacement_id,
                "resulting_ids": list(selected_swap.resulting_ids),
                "predicted_score": selected_swap.predicted_score,
                "score_delta": selected_swap.score_delta,
                "limiting_pair": list(swap_measurement.limiting_pairs[0]),
                "limiting_pairs": [
                    list(pair) for pair in swap_measurement.limiting_pairs
                ],
            }
        attempt_payloads.append(
            {
                "attempt_number": attempt.attempt_number,
                "selected_ids": list(attempt.selected_ids),
                "decision_basis": attempt.decision_basis,
                "score": attempt.score,
                "limiting_pair": list(measurement.limiting_pairs[0]),
                "limiting_pairs": [
                    list(pair) for pair in measurement.limiting_pairs
                ],
                "constraints_passed": attempt.constraints_passed,
                "achieved": attempt.achieved,
                "selected_swap": selected_swap_payload,
            }
        )
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
        "attempts": attempt_payloads,
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
