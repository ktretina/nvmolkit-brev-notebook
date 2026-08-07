"""Pure, bounded molecular-diversity objective built from retained workflow evidence."""

from __future__ import annotations

import itertools
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
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
                type(molecule_id) is not str
                or not molecule_id
                or "->" in molecule_id
                for molecule_id in molecule_ids
            )
            or len(set(molecule_ids)) != CANDIDATE_COUNT
        ):
            if any(
                type(molecule_id) is str and "->" in molecule_id
                for molecule_id in molecule_ids
            ):
                raise ValueError(
                    "Objective candidate molecule IDs contain the reserved delimiter '->'."
                )
            raise ValueError("Objective candidate molecule IDs must be unique nonempty strings.")
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
        immutable_bytes = distance_matrix.tobytes(order="C")
        distance_matrix = np.frombuffer(
            immutable_bytes, dtype=np.float64
        ).reshape((CANDIDATE_COUNT, CANDIDATE_COUNT))
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
    limiting_pair: tuple[str, str] | None = None
    swap_id: str = ""
    predicted_score_key: int | None = None
    limiting_pairs: tuple[tuple[str, str], ...] = ()
    target_status: str = "below_target"

    def __post_init__(self) -> None:
        if not self.swap_id:
            object.__setattr__(self, "swap_id", f"{self.replace_id}->{self.replacement_id}")
        if self.predicted_score_key is None:
            object.__setattr__(self, "predicted_score_key", score_key(self.predicted_score))
        if not self.limiting_pairs and self.limiting_pair is not None:
            object.__setattr__(self, "limiting_pairs", (self.limiting_pair,))
        if self.limiting_pair is None and self.limiting_pairs:
            object.__setattr__(self, "limiting_pair", self.limiting_pairs[0])


@dataclass(frozen=True)
class ObjectiveActionMenu:
    state_id: str
    source: PanelMeasurement
    accepted_attempt_count: int
    actions: tuple[ObjectiveSwap, ...]


@dataclass(frozen=True)
class ObjectiveAttempt:
    attempt_number: int
    selected_ids: tuple[str, ...]
    decision_basis: str = field(repr=False)
    score: float
    limiting_pair: tuple[str, str]
    constraints_passed: bool
    achieved: bool
    selected_swap: ObjectiveSwap | None = None
    state_id: str = ""
    score_key: int | None = None
    limiting_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.score_key is None and type(self.score) is float:
            try:
                object.__setattr__(self, "score_key", score_key(self.score))
            except ValueError:
                # Legacy callers can still construct forged records for downstream
                # rejection; authoritative evaluators never emit this state.
                pass
        if not self.limiting_pairs and self.limiting_pair:
            object.__setattr__(self, "limiting_pairs", (self.limiting_pair,))

    @property
    def measurement(self) -> PanelMeasurement:
        return PanelMeasurement(
            selected_ids=self.selected_ids,
            score=self.score,
            score_key=self.score_key,
            limiting_pairs=self.limiting_pairs,
            achieved=self.achieved,
        )


class TerminationReason(str, Enum):
    TARGET_ACHIEVED = "target_achieved"
    BASELINE_ALREADY_OPTIMAL = "baseline_already_optimal"
    ATTEMPT_LIMIT_REACHED = "attempt_limit_reached"
    NO_LEGAL_IMPROVING_SWAP = "no_legal_improving_swap"
    OBJECTIVE_CORRECTION_LIMIT = "objective_correction_limit"
    OBJECTIVE_PROVIDER_FAILURE = "objective_provider_failure"
    EVALUATION_NOT_COMPLETED = "evaluation_not_completed"


@dataclass(frozen=True)
class ObjectiveRun:
    context: ObjectiveContext
    attempts: tuple[ObjectiveAttempt, ...]
    achieved: bool
    termination_reason: TerminationReason | str
    final_ids: tuple[str, ...]
    final_score: float
    baseline: PanelMeasurement | None = None
    final_score_key: int | None = None

    def __post_init__(self) -> None:
        try:
            reason = TerminationReason(self.termination_reason)
        except ValueError as error:
            raise ValueError("Objective run has an invalid termination reason.") from error
        object.__setattr__(self, "termination_reason", reason)
        if self.baseline is None:
            object.__setattr__(self, "baseline", measure_panel(self.context, self.context.baseline_ids))
        if self.final_score_key is None:
            object.__setattr__(self, "final_score_key", score_key(self.final_score))


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
    benchmark = attainable_benchmark(provisional)
    baseline_score = baseline.score
    benchmark_score = benchmark.score
    target_score = baseline_score + TARGET_FRACTION * (
        benchmark_score - baseline_score
    )
    context = ObjectiveContext(
        candidates=candidates,
        baseline_ids=provisional.baseline_ids,
        baseline_score=float(baseline_score),
        benchmark_score=float(benchmark_score),
        target_score=float(target_score),
        distance_matrix=distance_matrix,
    )
    if not certify_argmax_reachability(context):
        raise RuntimeError(
            "Objective target is not reachable under the bounded decision policy."
        )
    return context


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


def attainable_benchmark(context: ObjectiveContext) -> PanelMeasurement:
    """Recompute the exact best attainable four-panel with canonical ties."""
    if type(context) is not ObjectiveContext:
        raise ValueError("Objective benchmark requires an exact context.")
    candidate_ids = tuple(sorted(candidate.molecule_id for candidate in context.candidates))
    return max(
        (
            measure_panel(context, panel)
            for panel in itertools.combinations(candidate_ids, PANEL_SIZE)
        ),
        key=lambda measurement: (
            measurement.score_key,
            measurement.score,
            measurement.selected_ids,
        ),
    )


def _validated_context_scores(
    context: ObjectiveContext,
) -> tuple[PanelMeasurement, PanelMeasurement, float]:
    baseline = measure_panel(context, context.baseline_ids)
    benchmark = attainable_benchmark(context)
    expected_target = float(
        baseline.score
        + TARGET_FRACTION * (benchmark.score - baseline.score)
    )
    if (
        type(context.baseline_score) is not float
        or not np.isfinite(context.baseline_score)
        or context.baseline_score != baseline.score
        or score_key(context.baseline_score) != baseline.score_key
        or type(context.benchmark_score) is not float
        or not np.isfinite(context.benchmark_score)
        or context.benchmark_score != benchmark.score
        or score_key(context.benchmark_score) != benchmark.score_key
        or type(context.target_score) is not float
        or not np.isfinite(context.target_score)
        or context.target_score != expected_target
        or score_key(context.target_score) != score_key(expected_target)
    ):
        raise ValueError(
            "Stored objective context scores do not match exact recomputation."
        )
    return baseline, benchmark, expected_target


def _validated_measurement(
    context: ObjectiveContext, source: object
) -> PanelMeasurement:
    if type(source) is not PanelMeasurement:
        raise ValueError("Objective policy source must be an exact panel measurement.")
    recomputed = measure_panel(context, source.selected_ids)
    if source != recomputed:
        raise ValueError("Objective policy source does not match the current measurement.")
    return recomputed


def enumerate_legal_swaps(
    context: ObjectiveContext, source: PanelMeasurement
) -> tuple[ObjectiveSwap, ...]:
    """Enumerate every canonical strict one-ID improvement before menu capping."""
    if type(context) is not ObjectiveContext:
        raise ValueError("Objective swap enumeration requires an exact context.")
    _validated_context_scores(context)
    current = _validated_measurement(context, source)
    if current.achieved:
        return ()
    panel, candidates = _validated_panel(context, current.selected_ids)
    current_ids = set(panel)
    actions: list[ObjectiveSwap] = []
    for replace_position, replace_id in enumerate(panel):
        for replacement_id in sorted(candidates):
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
            if predicted.score_key <= current.score_key:
                continue
            if not all(replace_id in pair for pair in current.limiting_pairs):
                raise AssertionError(
                    "A strict improvement must replace a member of every co-limiting pair."
                )
            actions.append(
                ObjectiveSwap(
                    swap_id=f"{replace_id}->{replacement_id}",
                    replace_id=replace_id,
                    replacement_id=replacement_id,
                    resulting_ids=resulting_ids,
                    predicted_score=predicted.score,
                    predicted_score_key=predicted.score_key,
                    score_delta=float(predicted.score - current.score),
                    limiting_pair=predicted.limiting_pairs[0],
                    limiting_pairs=predicted.limiting_pairs,
                    target_status=(
                        "meets_target" if predicted.achieved else "below_target"
                    ),
                )
            )
    return tuple(sorted(actions, key=lambda action: (-action.predicted_score_key, action.swap_id)))


def _menu_state_id(
    source: PanelMeasurement,
    accepted_attempt_count: int,
    swap_ids: tuple[str, ...],
) -> str:
    canonical = _canonical_json(
        {
            "selected_ids": list(source.selected_ids),
            "score_key": source.score_key,
            "limiting_pairs": [list(pair) for pair in source.limiting_pairs],
            "accepted_attempt_count": accepted_attempt_count,
            "displayed_swap_ids": list(swap_ids),
        }
    )
    return "state-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_action_menu(
    context: ObjectiveContext,
    source: PanelMeasurement,
    accepted_attempt_count: int,
) -> ObjectiveActionMenu:
    """Build one immutable capped offer set for an exact measured state revision."""
    _validated_context_scores(context)
    current = _validated_measurement(context, source)
    if (
        type(accepted_attempt_count) is not int
        or not 0 <= accepted_attempt_count <= MAX_ATTEMPTS
    ):
        raise ValueError("Accepted objective attempt count is outside the bounded policy.")
    ranked = enumerate_legal_swaps(context, current)
    displayed = tuple(sorted(ranked[:SUGGESTION_LIMIT], key=lambda action: action.swap_id))
    swap_ids = tuple(action.swap_id for action in displayed)
    return ObjectiveActionMenu(
        state_id=_menu_state_id(current, accepted_attempt_count, swap_ids),
        source=current,
        accepted_attempt_count=accepted_attempt_count,
        actions=displayed,
    )


def accepted_maxima(menu: ObjectiveActionMenu) -> tuple[ObjectiveSwap, ...]:
    if type(menu) is not ObjectiveActionMenu:
        raise ValueError("Accepted maxima require an exact objective action menu.")
    if not menu.actions:
        return ()
    maximum = max(action.predicted_score_key for action in menu.actions)
    return tuple(
        action for action in menu.actions if action.predicted_score_key == maximum
    )


def evaluate_selected_swap(
    context: ObjectiveContext,
    menu: ObjectiveActionMenu,
    action: ObjectiveSwap,
    attempt_number: int,
) -> ObjectiveAttempt:
    """Commit one displayed tied-argmax action to deterministic measured evidence."""
    if type(menu) is not ObjectiveActionMenu:
        raise ValueError("Objective action evaluation requires an exact current menu.")
    rebuilt = build_action_menu(context, menu.source, menu.accepted_attempt_count)
    if menu != rebuilt:
        raise ValueError("Objective action evaluation requires the exact current menu.")
    if type(action) is not ObjectiveSwap or action not in menu.actions:
        raise ValueError("Objective action is not in the exact current menu.")
    if action not in accepted_maxima(menu):
        raise ValueError("Objective action is not an accepted maximum.")
    if (
        type(attempt_number) is not int
        or attempt_number != menu.accepted_attempt_count + 1
        or not 1 <= attempt_number <= MAX_ATTEMPTS
    ):
        raise ValueError("Objective attempt number is not sequential for this menu.")
    measured = measure_panel(context, action.resulting_ids)
    if (
        action.swap_id != f"{action.replace_id}->{action.replacement_id}"
        or action.replace_id not in menu.source.selected_ids
        or action.replacement_id in menu.source.selected_ids
        or set(action.resulting_ids)
        != (set(menu.source.selected_ids) - {action.replace_id})
        | {action.replacement_id}
        or action.predicted_score != measured.score
        or action.predicted_score_key != measured.score_key
        or action.limiting_pairs != measured.limiting_pairs
        or action.limiting_pair != measured.limiting_pairs[0]
        or action.score_delta != measured.score - menu.source.score
        or action.target_status
        != ("meets_target" if measured.achieved else "below_target")
    ):
        raise ValueError("Objective action fields do not match deterministic measurement.")
    return ObjectiveAttempt(
        attempt_number=attempt_number,
        state_id=menu.state_id,
        selected_swap=action,
        selected_ids=measured.selected_ids,
        decision_basis="",
        score=measured.score,
        score_key=measured.score_key,
        limiting_pair=measured.limiting_pairs[0],
        limiting_pairs=measured.limiting_pairs,
        constraints_passed=True,
        achieved=measured.achieved,
    )


def certify_argmax_reachability(context: ObjectiveContext) -> bool:
    """Prove every displayed tied-maximum branch reaches target within the bound."""
    if type(context) is not ObjectiveContext:
        raise ValueError("Objective certification requires an exact context.")
    baseline, _benchmark, _target = _validated_context_scores(context)
    if baseline.achieved or baseline.score_key == score_key(context.benchmark_score):
        return True

    def reaches(source: PanelMeasurement, accepted_count: int) -> bool:
        if source.achieved:
            return True
        if accepted_count >= MAX_ATTEMPTS:
            return False
        menu = build_action_menu(context, source, accepted_count)
        maxima = accepted_maxima(menu)
        if not maxima:
            return False
        return all(
            reaches(
                evaluate_selected_swap(
                    context, menu, action, accepted_count + 1
                ).measurement,
                accepted_count + 1,
            )
            for action in maxima
        )

    return reaches(baseline, 0)


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
    legacy_bound = MAX_ATTEMPTS + (1 if selected_swap is not None else 0)
    if type(attempt_number) is not int or not 1 <= attempt_number <= legacy_bound:
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
        score_key=measurement.score_key,
        limiting_pair=measurement.limiting_pairs[0],
        limiting_pairs=measurement.limiting_pairs,
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
    panel, _ = _validated_panel(context, current.selected_ids)
    recomputed = measure_panel(context, panel)
    if type(current.score) is not float or not np.isfinite(current.score):
        raise ValueError(
            "Objective attempt score must be a finite non-boolean built-in float."
        )
    if current.score != recomputed.score:
        raise ValueError("Objective attempt score does not match its panel.")
    if current.limiting_pair != recomputed.limiting_pairs[0]:
        raise ValueError("Objective attempt limiting pair does not match its panel.")
    if (
        current.score_key != recomputed.score_key
        or current.limiting_pairs != recomputed.limiting_pairs
    ):
        raise ValueError("Objective attempt evidence does not match its panel.")
    if current.constraints_passed is not True:
        raise ValueError("Objective attempt constraints must be passed.")
    if current.achieved is not recomputed.achieved:
        raise ValueError("Objective attempt achieved state is inconsistent.")
    if current.achieved:
        return ()

    accepted_count = max(current.attempt_number - 1, 0)
    return accepted_maxima(build_action_menu(context, recomputed, accepted_count))


def _attempt_matches_policy(
    context: ObjectiveContext,
    source: PanelMeasurement,
    accepted_count: int,
    attempt: ObjectiveAttempt,
) -> PanelMeasurement:
    if type(attempt) is not ObjectiveAttempt:
        raise ValueError("Objective run requires exact measured attempts.")
    menu = build_action_menu(context, source, accepted_count)
    if attempt.state_id != menu.state_id or attempt.selected_swap is None:
        raise ValueError("Objective attempt does not identify the exact current state.")
    expected = evaluate_selected_swap(
        context, menu, attempt.selected_swap, accepted_count + 1
    )
    if attempt != expected:
        raise ValueError("Objective attempt does not match deterministic measurement.")
    return expected.measurement


def terminal_objective_run(
    context: ObjectiveContext,
    attempts: tuple[ObjectiveAttempt, ...],
    termination_reason: TerminationReason | str,
    *,
    menu: ObjectiveActionMenu | None = None,
) -> ObjectiveRun:
    """Validate and persist one terminal model for every objective outcome."""
    if type(context) is not ObjectiveContext or type(attempts) is not tuple:
        raise ValueError("Objective terminal run requires exact domain types.")
    try:
        reason = TerminationReason(termination_reason)
    except ValueError as error:
        raise ValueError("Objective run has an invalid termination reason.") from error
    baseline, actual_benchmark, _target = _validated_context_scores(context)
    current = baseline
    for accepted_count, attempt in enumerate(attempts):
        if current.achieved:
            raise ValueError("Objective run continued after measured target success.")
        current = _attempt_matches_policy(
            context, current, accepted_count, attempt
        )

    if reason is TerminationReason.BASELINE_ALREADY_OPTIMAL:
        if attempts or baseline.score_key != actual_benchmark.score_key:
            raise ValueError("Objective baseline is not actually optimal.")
    elif reason is TerminationReason.TARGET_ACHIEVED:
        if not current.achieved:
            raise ValueError("Target success requires measured target success.")
        if not attempts and baseline.score_key == actual_benchmark.score_key:
            raise ValueError(
                "Exact baseline-optimal state requires baseline-optimal termination."
            )
    elif reason is TerminationReason.ATTEMPT_LIMIT_REACHED:
        if len(attempts) != MAX_ATTEMPTS or current.achieved:
            raise ValueError("Attempt-limit termination requires exactly three measured misses.")
    elif reason is TerminationReason.NO_LEGAL_IMPROVING_SWAP:
        if current.achieved or menu is None:
            raise ValueError("No-legal termination requires a below-target empty menu.")
        rebuilt = build_action_menu(context, current, len(attempts))
        if menu != rebuilt or menu.actions:
            raise ValueError("No-legal termination requires the exact current empty menu.")
    else:
        if current.achieved or len(attempts) >= MAX_ATTEMPTS:
            raise ValueError("Pre-evaluation failure cannot follow success or max attempts.")

    achieved = reason in {
        TerminationReason.TARGET_ACHIEVED,
        TerminationReason.BASELINE_ALREADY_OPTIMAL,
    }
    return ObjectiveRun(
        context=context,
        baseline=baseline,
        attempts=attempts,
        achieved=achieved,
        termination_reason=reason,
        final_ids=current.selected_ids,
        final_score=current.score,
        final_score_key=current.score_key,
    )


def baseline_terminal_run(context: ObjectiveContext) -> ObjectiveRun:
    """Finalize the exact Step-0 baseline-optimal state without an attempt."""
    return terminal_objective_run(
        context, (), TerminationReason.BASELINE_ALREADY_OPTIMAL
    )


def finalize_no_legal_swap(
    context: ObjectiveContext,
    attempts: tuple[ObjectiveAttempt, ...],
    current: PanelMeasurement,
    menu: ObjectiveActionMenu,
) -> ObjectiveRun:
    """Finalize a measured below-target state whose exact menu is empty."""
    expected = (
        measure_panel(context, context.baseline_ids)
        if not attempts
        else attempts[-1].measurement
    )
    if type(current) is not PanelMeasurement or current != expected:
        raise ValueError("No-legal finalizer current measurement is inconsistent.")
    return terminal_objective_run(
        context,
        attempts,
        TerminationReason.NO_LEGAL_IMPROVING_SWAP,
        menu=menu,
    )


def finalize_objective_run(
    context: ObjectiveContext, attempts: tuple[ObjectiveAttempt, ...]
) -> ObjectiveRun:
    """Normalize the legacy baseline-first flow into authoritative substitutions."""
    _validated_context_scores(context)
    if not attempts or len(attempts) > MAX_ATTEMPTS + 1:
        raise ValueError("Objective run has an invalid accepted-attempt count.")
    if all(attempt.state_id for attempt in attempts):
        reason = (
            TerminationReason.TARGET_ACHIEVED
            if attempts[-1].achieved
            else TerminationReason.ATTEMPT_LIMIT_REACHED
        )
        return terminal_objective_run(context, attempts, reason)

    baseline = measure_panel(context, context.baseline_ids)
    legacy_baseline = attempts[0]
    if (
        type(legacy_baseline) is not ObjectiveAttempt
        or legacy_baseline.attempt_number != 1
        or legacy_baseline.state_id
        or legacy_baseline.selected_swap is not None
        or legacy_baseline.measurement != baseline
        or legacy_baseline.constraints_passed is not True
    ):
        raise ValueError(
            "Legacy objective finalization requires the exact measured baseline Step 0."
        )
    if baseline.achieved:
        if len(attempts) != 1:
            raise ValueError("Objective run continued after measured baseline success.")
        return baseline_terminal_run(context)

    normalized: list[ObjectiveAttempt] = []
    current = baseline
    for substitution_number, legacy in enumerate(attempts[1:], start=1):
        if (
            type(legacy) is not ObjectiveAttempt
            or legacy.attempt_number != substitution_number + 1
            or legacy.state_id
            or legacy.selected_swap is None
        ):
            raise ValueError("Legacy objective revisions are not sequential substitutions.")
        menu = build_action_menu(context, current, substitution_number - 1)
        offered = next(
            (
                action
                for action in menu.actions
                if action.swap_id == legacy.selected_swap.swap_id
            ),
            None,
        )
        if offered is None or legacy.selected_swap != offered:
            raise ValueError("Legacy objective revision is not in the exact current menu.")
        if offered not in accepted_maxima(menu):
            raise ValueError("Legacy objective revision is not an accepted maximum.")
        measured = measure_panel(context, offered.resulting_ids)
        if (
            legacy.measurement != measured
            or legacy.constraints_passed is not True
        ):
            raise ValueError("Legacy objective revision evidence is stale or mismatched.")
        authoritative = evaluate_selected_swap(
            context, menu, offered, substitution_number
        )
        normalized.append(authoritative)
        current = authoritative.measurement

    if not normalized:
        raise ValueError("A below-target legacy baseline is not a terminal run.")
    reason = (
        TerminationReason.TARGET_ACHIEVED
        if normalized[-1].achieved
        else TerminationReason.ATTEMPT_LIMIT_REACHED
    )
    return terminal_objective_run(context, tuple(normalized), reason)


def no_improvement_run(context: ObjectiveContext) -> ObjectiveRun:
    """Represent the explicit case where the current policy is already optimal."""
    return baseline_terminal_run(context)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_objective_evidence(run: ObjectiveRun) -> EvidenceRecord:
    """Build immutable terminal O01 without model prose or a hidden benchmark panel."""
    if type(run) is not ObjectiveRun:
        raise ValueError("Objective evidence requires an exact terminal run.")
    context = run.context
    baseline, benchmark, expected_target = _validated_context_scores(context)
    terminal_menu = None
    if run.termination_reason is TerminationReason.NO_LEGAL_IMPROVING_SWAP:
        current = (
            run.attempts[-1].measurement
            if run.attempts
            else measure_panel(context, context.baseline_ids)
        )
        terminal_menu = build_action_menu(context, current, len(run.attempts))
    validated_run = terminal_objective_run(
        context,
        run.attempts,
        run.termination_reason,
        menu=terminal_menu,
    )
    if (
        run.achieved != validated_run.achieved
        or run.final_ids != validated_run.final_ids
        or run.final_score != validated_run.final_score
        or run.final_score_key != validated_run.final_score_key
    ):
        raise ValueError("Objective terminal evidence is inconsistent.")
    attempt_payloads = []
    for attempt in run.attempts:
        measurement = measure_panel(context, attempt.selected_ids)
        if (
            attempt.score != measurement.score
            or attempt.score_key != measurement.score_key
            or attempt.limiting_pairs != measurement.limiting_pairs
            or attempt.achieved != measurement.achieved
        ):
            raise ValueError("Objective evidence attempt does not match measurement.")
        selected_swap_payload = None
        if attempt.selected_swap is not None:
            selected_swap = attempt.selected_swap
            selected_swap_payload = {
                "swap_id": selected_swap.swap_id,
                "replace_id": selected_swap.replace_id,
                "replacement_id": selected_swap.replacement_id,
                "resulting_ids": list(selected_swap.resulting_ids),
                "predicted_score": selected_swap.predicted_score,
                "predicted_score_key": selected_swap.predicted_score_key,
                "score_delta": selected_swap.score_delta,
                "limiting_pair": list(selected_swap.limiting_pairs[0]),
                "limiting_pairs": [
                    list(pair) for pair in selected_swap.limiting_pairs
                ],
                "target_status": selected_swap.target_status,
            }
        attempt_payloads.append(
            {
                "attempt_number": attempt.attempt_number,
                "state_id": attempt.state_id,
                "selected_ids": list(attempt.selected_ids),
                "score": attempt.score,
                "score_key": attempt.score_key,
                "limiting_pair": list(measurement.limiting_pairs[0]),
                "limiting_pairs": [
                    list(pair) for pair in measurement.limiting_pairs
                ],
                "constraints_passed": attempt.constraints_passed,
                "achieved": attempt.achieved,
                "selected_swap": selected_swap_payload,
            }
        )
    final = measure_panel(context, run.final_ids)
    if (
        run.baseline != baseline
        or run.final_score != final.score
        or run.final_score_key != final.score_key
        or run.achieved
        != (
            run.termination_reason
            in {
                TerminationReason.TARGET_ACHIEVED,
                TerminationReason.BASELINE_ALREADY_OPTIMAL,
            }
        )
    ):
        raise ValueError("Objective terminal evidence is inconsistent.")

    def measurement_payload(measurement: PanelMeasurement) -> dict[str, Any]:
        return {
            "selected_ids": list(measurement.selected_ids),
            "score": measurement.score,
            "score_key": measurement.score_key,
            "limiting_pairs": [list(pair) for pair in measurement.limiting_pairs],
            "achieved": measurement.achieved,
        }

    payload = {
        "objective": "maximize minimum pairwise Morgan/Tanimoto distance",
        "score_definition": "D_min=min(1-Tanimoto(i,j)) over selected pairs",
        "candidate_pool_rule": "first MMFF94-eligible member from each of eight largest eligible clusters",
        "candidate_ids": [candidate.molecule_id for candidate in context.candidates],
        "candidate_cluster_ids": [candidate.cluster_id for candidate in context.candidates],
        "panel_size": PANEL_SIZE,
        "attempt_limit": MAX_ATTEMPTS,
        "baseline_ids": list(context.baseline_ids),
        "baseline_score": baseline.score,
        "baseline_score_key": baseline.score_key,
        "baseline": measurement_payload(baseline),
        "benchmark_score": benchmark.score,
        "benchmark_score_key": benchmark.score_key,
        "target_rule": "baseline plus 80 percent of attainable improvement",
        "target_score": expected_target,
        "target_score_key": score_key(expected_target),
        "attempts": attempt_payloads,
        "attempt_count": len(run.attempts),
        "termination_reason": run.termination_reason.value,
        "final_ids": list(run.final_ids),
        "final_score": run.final_score,
        "final_score_key": run.final_score_key,
        "final_measurement": measurement_payload(final),
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
