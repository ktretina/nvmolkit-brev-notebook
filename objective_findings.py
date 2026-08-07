"""Deterministic, evidence-controlled findings for the objective Decision Ladder."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from chemistry_workflow import EvidenceRecord, WorkflowReport
from objective_challenge import (
    ObjectiveRun,
    build_objective_evidence,
    score_key,
    validate_objective_evidence,
)


CONCLUSION_THEMES = (
    "dataset_scope",
    "molecular_representation",
    "similarity_structure",
    "clustering",
    "conformational_sampling",
    "objective_driven_selection",
    "limitations_and_next_steps",
)

_WORKFLOW_KEYS = tuple(f"E0{index}" for index in range(1, 7))
_ALL_KEYS = (*_WORKFLOW_KEYS, "O01")
_CANONICAL_RECORD_METADATA = {
    "E01": ("Library inspection", "RDKit input validation"),
    "E02": ("Morgan fingerprints", "MorganFingerprintGenerator"),
    "E03": ("Tanimoto similarity", "crossTanimotoSimilarity"),
    "E04": ("Fused Butina clusters", "fused_butina"),
    "E05": ("Representative embedding", "EmbedMolecules"),
    "E06": ("MMFF94 optimization", "MMFFOptimizeMoleculesConfs"),
}


@dataclass(frozen=True)
class EvidenceFinding:
    finding_id: str
    theme: str
    evidence_keys: tuple[str, ...]
    predicate_id: str
    text: str


@dataclass(frozen=True)
class MeasuredSummary:
    raw_count: int
    valid_molecule_count: int
    invalid_count: int
    excluded_count: int
    fingerprint_radius: int
    fingerprint_size: int
    representation_name: str
    similarity_quartiles: tuple[float, float, float]
    similarity_p90: float
    similarity_max: float
    most_similar_pair_ids: tuple[str, str]
    most_similar_pair_similarity: float
    cluster_cutoff: float
    cluster_count: int
    singleton_count: int
    singleton_fraction: float
    largest_cluster_sizes: tuple[int, ...]
    representative_count: int
    generated_conformer_count: int
    attempted_conformer_count: int
    converged_conformer_count: int
    unconverged_conformer_count: int
    optimization_comparison_scope: str
    candidate_pool_count: int
    candidate_cluster_count: int
    final_panel_count: int
    final_cluster_count: int
    baseline_distance: float
    benchmark_distance: float
    target_distance: float
    final_distance: float
    target_margin: float
    final_max_similarity: float
    limiting_pairs: tuple[tuple[str, str], ...]
    limiting_similarities: tuple[float, ...]
    attempt_count: int
    termination_reason: str
    achieved: bool
    headline: str
    facts: tuple[str, ...]


_HEADLINE_PREFIXES = {
    "target_achieved": "Target achieved",
    "baseline_already_optimal": "Baseline already optimal",
    "attempt_limit_reached": "Objective not achieved within attempt limit",
    "no_legal_improving_swap": "No legal improving substitution",
    "objective_correction_limit": "Objective selection stopped after invalid responses",
    "objective_provider_failure": "Objective provider unavailable",
    "evaluation_not_completed": "Objective evaluation not completed",
}


def _headline(reason: str, final_distance: float, target_distance: float) -> str:
    try:
        prefix = _HEADLINE_PREFIXES[reason]
    except KeyError as error:
        raise ValueError("O01 termination reason is unknown.") from error
    return (
        f"{prefix}: final minimum distance {final_distance:.6f}; "
        f"target {target_distance:.6f}."
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonfinite(nested)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evidence values must be finite.")


def _exact_keys(key: str, payload: Mapping[str, Any], expected: set[str]) -> None:
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing:
        raise ValueError(f"{key} is missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{key} contains unknown keys: {sorted(extra)}")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer count.")
    return value


def _number(
    value: Any,
    label: str,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if not lower <= result <= upper:
        raise ValueError(f"{label} must be in [{lower}, {upper}].")
    return result


@dataclass(frozen=True)
class EvidenceSnapshot:
    records: tuple[EvidenceRecord, ...]
    summary: MeasuredSummary

    @classmethod
    def from_report(cls, report: WorkflowReport, run: ObjectiveRun) -> "EvidenceSnapshot":
        if type(report) is not WorkflowReport:
            raise ValueError("Evidence snapshot requires an exact workflow report.")
        return cls.from_records((*report.evidence, build_objective_evidence(run)), run)

    @classmethod
    def from_records(
        cls,
        records: Iterable[EvidenceRecord],
        run: ObjectiveRun,
    ) -> "EvidenceSnapshot":
        if type(run) is not ObjectiveRun:
            raise ValueError("Evidence snapshot requires an exact objective run.")
        records = tuple(records)
        if any(type(record) is not EvidenceRecord for record in records):
            raise ValueError("Evidence snapshot requires exact evidence records.")
        by_key = {record.key: record for record in records}
        if len(by_key) != len(records) or tuple(sorted(by_key)) != _ALL_KEYS:
            raise ValueError(
                "Evidence snapshot requires exactly one record for E01-E06 and O01."
            )

        payloads: dict[str, dict[str, Any]] = {}
        for key in _ALL_KEYS:
            record = by_key[key]
            if key in _CANONICAL_RECORD_METADATA:
                expected_label, expected_provenance = _CANONICAL_RECORD_METADATA[key]
                if record.label != expected_label:
                    raise ValueError(f"{key} label is not canonical.")
                if record.provenance != expected_provenance:
                    raise ValueError(f"{key} provenance is not canonical.")
            try:
                payload = json.loads(record.payload_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{key} payload is not valid JSON.") from error
            if type(payload) is not dict:
                raise ValueError(f"{key} payload must be a JSON object.")
            _reject_nonfinite(payload)
            try:
                canonical = _canonical_json(payload)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} payload is not canonical finite JSON.") from error
            if record.payload_json != canonical:
                raise ValueError(f"{key} payload must use canonical JSON.")
            payloads[key] = payload

        validate_objective_evidence(by_key["O01"], run)
        return cls(
            tuple(by_key[key] for key in _ALL_KEYS),
            _build_summary(payloads, run),
        )


def _build_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    run: ObjectiveRun,
) -> MeasuredSummary:
    e01, e02, e03 = payloads["E01"], payloads["E02"], payloads["E03"]
    e04, e05, e06, o01 = (
        payloads["E04"],
        payloads["E05"],
        payloads["E06"],
        payloads["O01"],
    )
    _exact_keys(
        "E01",
        e01,
        {
            "raw_count",
            "valid_count",
            "invalid_count",
            "invalid_ids",
            "preview_count",
            "count_unit",
        },
    )
    _exact_keys(
        "E02",
        e02,
        {
            "fingerprint_radius",
            "fingerprint_size_bits",
            "packed_shape",
            "molecule_count",
            "active_bits_min",
            "active_bits_median",
            "active_bits_max",
            "executor",
            "size_unit",
        },
    )
    _exact_keys(
        "E03",
        e03,
        {
            "matrix_shape",
            "q1",
            "median",
            "q3",
            "p90",
            "max_off_diagonal",
            "most_similar_pair",
            "similarity_unit",
        },
    )
    _exact_keys(
        "E04",
        e04,
        {
            "cutoff",
            "cluster_count",
            "singleton_count",
            "singleton_fraction",
            "largest_cluster_sizes",
            "assignment_count",
            "cutoff_unit",
        },
    )
    _exact_keys(
        "E05",
        e05,
        {
            "requested_representative_count",
            "selected_representative_count",
            "selection_shortfall",
            "representative_policy",
            "representatives",
            "requested_conformers_per_representative",
            "generated_conformer_count",
            "partial_embedding_ids",
            "zero_embedding_ids",
            "count_unit",
        },
    )
    _exact_keys(
        "E06",
        e06,
        {
            "attempted_conformer_count",
            "converged_conformer_count",
            "unconverged_conformer_count",
            "per_conformer_records",
            "selected_conformer_records",
            "energy_unit",
            "comparison_scope",
        },
    )

    raw = _integer(e01["raw_count"], "E01 raw count", minimum=2)
    valid = _integer(e01["valid_count"], "E01 valid count", minimum=2)
    invalid = _integer(e01["invalid_count"], "E01 invalid count")
    if (
        raw != valid + invalid
        or type(e01["invalid_ids"]) is not list
        or len(e01["invalid_ids"]) != invalid
    ):
        raise ValueError("E01 count fields are contradictory.")
    if e01["count_unit"] != "rows":
        raise ValueError("E01 count unit is not canonical.")

    radius = _integer(e02["fingerprint_radius"], "E02 fingerprint radius")
    size = _integer(
        e02["fingerprint_size_bits"], "E02 fingerprint size", minimum=1
    )
    molecule_count = _integer(e02["molecule_count"], "E02 molecule count")
    if molecule_count != valid or e02["packed_shape"] != [valid, size // 32]:
        raise ValueError("E02 count and packed-shape fields are contradictory.")
    if radius not in (2, 3) or size not in (1024, 2048):
        raise ValueError("E02 fingerprint parameters are unsupported.")
    if e02["executor"] != "nvMolKit GPU":
        raise ValueError("E02 executor is not canonical.")
    if e02["size_unit"] != "bits":
        raise ValueError("E02 size unit is not canonical.")
    active_bits = tuple(
        _number(e02[name], f"E02 {name}", upper=float(size))
        for name in ("active_bits_min", "active_bits_median", "active_bits_max")
    )
    if tuple(sorted(active_bits)) != active_bits:
        raise ValueError("E02 active-bit fields are contradictory.")

    similarities = tuple(
        _number(e03[name], f"E03 {name}")
        for name in ("q1", "median", "q3", "p90", "max_off_diagonal")
    )
    if tuple(sorted(similarities)) != similarities or e03["matrix_shape"] != [valid, valid]:
        raise ValueError("E03 similarity fields are contradictory.")
    if e03["similarity_unit"] != "Tanimoto coefficient":
        raise ValueError("E03 similarity unit must be the Tanimoto coefficient.")
    pair = e03["most_similar_pair"]
    _exact_keys(
        "E03 most_similar_pair",
        pair,
        {"molecule_ids", "source_rows", "similarity"},
    )
    if (
        type(pair["molecule_ids"]) is not list
        or len(pair["molecule_ids"]) != 2
        or any(type(value) is not str or not value for value in pair["molecule_ids"])
        or len(set(pair["molecule_ids"])) != 2
        or type(pair["source_rows"]) is not list
        or len(pair["source_rows"]) != 2
        or any(type(value) is not int or value < 0 for value in pair["source_rows"])
        or len(set(pair["source_rows"])) != 2
        or _number(pair["similarity"], "E03 pair similarity") != similarities[-1]
    ):
        raise ValueError("E03 most-similar pair fields are contradictory.")

    cutoff = _number(e04["cutoff"], "E04 cutoff")
    if not 0.40 <= cutoff <= 0.60:
        raise ValueError("E04 cutoff is outside the supported range.")
    if e04["cutoff_unit"] != "Tanimoto distance":
        raise ValueError("E04 cutoff unit is not canonical.")
    cluster_count = _integer(e04["cluster_count"], "E04 cluster count", minimum=1)
    singletons = _integer(e04["singleton_count"], "E04 singleton count")
    assignments = _integer(e04["assignment_count"], "E04 assignment count")
    singleton_fraction = _number(
        e04["singleton_fraction"], "E04 singleton fraction"
    )
    sizes = e04["largest_cluster_sizes"]
    non_singleton_count = cluster_count - singletons
    if (
        assignments != valid
        or cluster_count > assignments
        or singletons > cluster_count
        or not math.isclose(
            singleton_fraction,
            singletons / assignments,
            rel_tol=0,
            abs_tol=1e-15,
        )
        or type(sizes) is not list
        or not sizes
        or len(sizes) > min(15, cluster_count)
        or any(type(value) is not int or value < 1 for value in sizes)
        or sizes != sorted(sizes, reverse=True)
    ):
        raise ValueError("E04 cluster count fields are contradictory.")
    expected_largest_count = min(15, cluster_count)
    if len(sizes) != expected_largest_count:
        raise ValueError("E04 largest-cluster count is contradictory.")
    listed_non_singletons = min(non_singleton_count, expected_largest_count)
    expected_listed_singletons = expected_largest_count - listed_non_singletons
    if (
        sum(size > 1 for size in sizes) != listed_non_singletons
        or sum(size == 1 for size in sizes) != expected_listed_singletons
    ):
        raise ValueError("E04 largest-cluster sizes are infeasible.")
    remaining_non_singletons = non_singleton_count - listed_non_singletons
    remaining_singletons = singletons - expected_listed_singletons
    remaining_assignments = assignments - sum(sizes) - remaining_singletons
    if remaining_non_singletons == 0:
        feasible_largest_sizes = remaining_assignments == 0
    else:
        maximum_remaining_size = sizes[-1]
        feasible_largest_sizes = (
            2 * remaining_non_singletons
            <= remaining_assignments
            <= maximum_remaining_size * remaining_non_singletons
        )
    if not feasible_largest_sizes:
        raise ValueError("E04 largest-cluster sizes are infeasible for assignments.")

    requested_reps = _integer(
        e05["requested_representative_count"], "E05 requested representatives"
    )
    selected_reps = _integer(
        e05["selected_representative_count"], "E05 selected representatives"
    )
    shortfall = _integer(e05["selection_shortfall"], "E05 selection shortfall")
    per_rep = _integer(
        e05["requested_conformers_per_representative"],
        "E05 conformers per representative",
    )
    generated = _integer(
        e05["generated_conformer_count"], "E05 generated conformers"
    )
    if not 3 <= requested_reps <= 6:
        raise ValueError("E05 representative count must be 3 through 6 inclusive.")
    if (
        not 3 <= selected_reps <= requested_reps
        or shortfall != requested_reps - selected_reps
    ):
        raise ValueError(
            "E05 selected representative count must be 3 through the requested count."
        )
    if not 3 <= per_rep <= 8:
        raise ValueError(
            "E05 conformers per representative must be 3 through 8 inclusive."
        )
    if generated < 1:
        raise ValueError("E05 must contain at least one generated conformer.")
    representatives = e05["representatives"]
    partial_embedding_ids = e05["partial_embedding_ids"]
    zero_embedding_ids = e05["zero_embedding_ids"]
    if (
        requested_reps != selected_reps + shortfall
        or type(representatives) is not list
        or len(representatives) != selected_reps
        or generated > selected_reps * per_rep
    ):
        raise ValueError("E05 representative count fields are contradictory.")
    if e05["count_unit"] != "conformers":
        raise ValueError("E05 count unit is not canonical.")
    if e05["representative_policy"] not in {
        "largest_clusters_first",
        "include_singleton_if_available",
    }:
        raise ValueError("E05 representative policy is unsupported.")
    representative_by_id: dict[str, Mapping[str, Any]] = {}
    for representative in representatives:
        if type(representative) is not dict:
            raise ValueError("E05 representative records must be objects.")
        _exact_keys(
            "E05 representative",
            representative,
            {"molecule_id", "source_row", "cluster_id"},
        )
        molecule_id = representative["molecule_id"]
        if (
            type(molecule_id) is not str
            or not molecule_id
            or molecule_id in representative_by_id
            or type(representative["source_row"]) is not int
            or representative["source_row"] < 0
            or type(representative["cluster_id"]) is not int
            or not 0 <= representative["cluster_id"] < cluster_count
        ):
            raise ValueError("E05 representative provenance is contradictory.")
        representative_by_id[molecule_id] = representative
    if len({item["cluster_id"] for item in representatives}) != selected_reps:
        raise ValueError("E05 representative cluster provenance is contradictory.")
    representative_ids = set(representative_by_id)
    if (
        type(partial_embedding_ids) is not list
        or type(zero_embedding_ids) is not list
        or any(type(value) is not str for value in (*partial_embedding_ids, *zero_embedding_ids))
        or len(set(partial_embedding_ids)) != len(partial_embedding_ids)
        or len(set(zero_embedding_ids)) != len(zero_embedding_ids)
        or not set(partial_embedding_ids).issubset(representative_ids)
        or not set(zero_embedding_ids).issubset(representative_ids)
        or set(partial_embedding_ids) & set(zero_embedding_ids)
    ):
        raise ValueError("E05 embedding ID accounting is contradictory.")
    full_embedding_count = (
        selected_reps - len(partial_embedding_ids) - len(zero_embedding_ids)
    )
    minimum_generated = full_embedding_count * per_rep + len(partial_embedding_ids)
    maximum_generated = (
        full_embedding_count * per_rep
        + len(partial_embedding_ids) * max(0, per_rep - 1)
    )
    if not minimum_generated <= generated <= maximum_generated:
        raise ValueError("E05 embedding counts are infeasible.")

    attempted = _integer(
        e06["attempted_conformer_count"], "E06 attempted conformers"
    )
    converged = _integer(
        e06["converged_conformer_count"], "E06 converged conformers"
    )
    unconverged = _integer(
        e06["unconverged_conformer_count"], "E06 unconverged conformers"
    )
    per_conformer = e06["per_conformer_records"]
    selected = e06["selected_conformer_records"]
    if (
        attempted != generated
        or attempted != converged + unconverged
        or type(per_conformer) is not list
        or len(per_conformer) != attempted
        or sum(record.get("converged") is True for record in per_conformer)
        != converged
        or type(selected) is not list
        or len(selected) > selected_reps
        or any(record.get("converged") is not True for record in selected)
        or e06["comparison_scope"] != "within molecule only"
    ):
        raise ValueError(
            "E06 conformer count or comparison-scope fields are contradictory."
        )
    if e06["energy_unit"] != "kcal/mol":
        raise ValueError("E06 energy unit is not canonical.")
    conformer_keys = {
        "molecule_id",
        "cluster_id",
        "conformer_index",
        "energy_kcal_mol",
        "converged",
    }
    conformers_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in per_conformer:
        if type(record) is not dict:
            raise ValueError("E06 conformer records must be objects.")
        _exact_keys("E06 conformer", record, conformer_keys)
        molecule_id = record["molecule_id"]
        conformer_index = record["conformer_index"]
        key = (molecule_id, conformer_index)
        representative = representative_by_id.get(molecule_id)
        if (
            representative is None
            or type(conformer_index) is not int
            or conformer_index < 0
            or key in conformers_by_key
            or record["cluster_id"] != representative["cluster_id"]
            or isinstance(record["energy_kcal_mol"], bool)
            or not isinstance(record["energy_kcal_mol"], (int, float))
            or not math.isfinite(float(record["energy_kcal_mol"]))
            or type(record["converged"]) is not bool
        ):
            raise ValueError("E06 conformer provenance is contradictory.")
        conformers_by_key[key] = record
    selected_ids: set[str] = set()
    for record in selected:
        if type(record) is not dict:
            raise ValueError("E06 selected-conformer records must be objects.")
        _exact_keys(
            "E06 selected conformer",
            record,
            {*conformer_keys, "selected_conformer_id"},
        )
        key = (record["molecule_id"], record["conformer_index"])
        source = conformers_by_key.get(key)
        expected_id = f"{record['molecule_id']}:conf-{record['conformer_index']}"
        converged_for_molecule = [
            item
            for (molecule_id, _), item in conformers_by_key.items()
            if molecule_id == record["molecule_id"] and item["converged"]
        ]
        if (
            source is None
            or any(record[name] != source[name] for name in conformer_keys)
            or record["selected_conformer_id"] != expected_id
            or record["molecule_id"] in selected_ids
            or not converged_for_molecule
            or float(record["energy_kcal_mol"])
            != min(float(item["energy_kcal_mol"]) for item in converged_for_molecule)
        ):
            raise ValueError("E06 selected-conformer provenance is contradictory.")
        selected_ids.add(record["molecule_id"])
    conformer_counts = {
        molecule_id: sum(
            key_molecule_id == molecule_id
            for key_molecule_id, _ in conformers_by_key
        )
        for molecule_id in representative_by_id
    }
    actual_partial_ids = {
        molecule_id
        for molecule_id, count in conformer_counts.items()
        if 0 < count < per_rep
    }
    actual_zero_ids = {
        molecule_id for molecule_id, count in conformer_counts.items() if count == 0
    }
    if (
        any(count > per_rep for count in conformer_counts.values())
        or actual_partial_ids != set(partial_embedding_ids)
        or actual_zero_ids != set(zero_embedding_ids)
    ):
        raise ValueError("E05 embedding IDs do not match E06 conformer records.")
    for molecule_id, count in conformer_counts.items():
        indices = sorted(
            conformer_index
            for key_molecule_id, conformer_index in conformers_by_key
            if key_molecule_id == molecule_id
        )
        if indices != list(range(count)):
            raise ValueError("E06 conformer indices must be contiguous per molecule.")
    molecules_with_converged_samples = {
        molecule_id
        for (molecule_id, _), record in conformers_by_key.items()
        if record["converged"]
    }
    if selected_ids != molecules_with_converged_samples:
        raise ValueError(
            "E06 selected minima must cover exactly the molecules with converged samples."
        )
    selected_by_id = {record["molecule_id"]: record for record in selected}
    for molecule_id in molecules_with_converged_samples:
        canonical = min(
            (
                record
                for (key_molecule_id, _), record in conformers_by_key.items()
                if key_molecule_id == molecule_id and record["converged"]
            ),
            key=lambda record: (
                float(record["energy_kcal_mol"]),
                record["conformer_index"],
            ),
        )
        selected_record = selected_by_id[molecule_id]
        if (
            selected_record["conformer_index"] != canonical["conformer_index"]
            or float(selected_record["energy_kcal_mol"])
            != float(canonical["energy_kcal_mol"])
        ):
            raise ValueError("E06 selected conformer is not the canonical minimum.")

    candidate_ids = o01["candidate_ids"]
    candidate_clusters = o01["candidate_cluster_ids"]
    final_ids = o01["final_ids"]
    candidate_pool_count = len(candidate_ids)
    candidate_cluster_count = len(set(candidate_clusters))
    final_panel_count = len(final_ids)
    final_cluster_count = len(
        {
            candidate_clusters[candidate_ids.index(molecule_id)]
            for molecule_id in final_ids
        }
    )
    if (
        len(candidate_clusters) != candidate_pool_count
        or any(type(cluster_id) is not int for cluster_id in candidate_clusters)
        or any(not 0 <= cluster_id < cluster_count for cluster_id in candidate_clusters)
        or candidate_cluster_count > cluster_count
        or final_cluster_count > cluster_count
    ):
        raise ValueError("O01 cluster IDs are outside the E04 cluster domain.")
    baseline = _number(o01["baseline_score"], "O01 baseline score")
    benchmark = _number(o01["benchmark_score"], "O01 benchmark score")
    target = _number(o01["target_score"], "O01 target score")
    final = _number(o01["final_score"], "O01 final score")
    final_measurement = o01["final_measurement"]
    limiting_pairs = tuple(tuple(pair) for pair in final_measurement["limiting_pairs"])
    if any(
        len(pair) != 2
        or any(type(molecule_id) is not str for molecule_id in pair)
        for pair in limiting_pairs
    ):
        raise ValueError("O01 limiting pairs are malformed.")
    candidate_positions = {
        candidate.molecule_id: position
        for position, candidate in enumerate(run.context.candidates)
    }
    limiting_distances = []
    for first_id, second_id in limiting_pairs:
        if first_id not in final_ids or second_id not in final_ids:
            raise ValueError("O01 limiting pair is outside the final panel.")
        distance = float(
            run.context.distance_matrix[
                candidate_positions[first_id], candidate_positions[second_id]
            ]
        )
        if score_key(distance) != final_measurement["score_key"]:
            raise ValueError("O01 limiting pair does not have the final score key.")
        limiting_distances.append(distance)
    limiting_similarities = tuple(1.0 - distance for distance in limiting_distances)
    final_similarity = max(limiting_similarities)
    reason = o01["termination_reason"]
    achieved = o01["achieved"]
    headline = _headline(reason, final, target)
    facts = (
        f"{valid} valid molecules were retained from {raw} raw rows; {invalid} were excluded.",
        f"Morgan fingerprints used radius {radius} and size {size} bits.",
        (
            "Tanimoto similarity quartiles were "
            f"{similarities[0]:.3f}, {similarities[1]:.3f}, and {similarities[2]:.3f}."
        ),
        f"The library produced {cluster_count} clusters, including {singletons} singletons.",
        f"MMFF94 converged {converged} of {attempted} sampled conformers.",
        (
            f"The final panel contains {final_panel_count} compounds from "
            f"{final_cluster_count} distinct clusters."
        ),
        headline,
    )
    return MeasuredSummary(
        raw_count=raw,
        valid_molecule_count=valid,
        invalid_count=invalid,
        excluded_count=invalid,
        fingerprint_radius=radius,
        fingerprint_size=size,
        representation_name=(
            f"Morgan radius-{radius} {size}-bit fingerprints with Tanimoto similarity"
        ),
        similarity_quartiles=(similarities[0], similarities[1], similarities[2]),
        similarity_p90=similarities[3],
        similarity_max=similarities[4],
        most_similar_pair_ids=tuple(pair["molecule_ids"]),
        most_similar_pair_similarity=float(pair["similarity"]),
        cluster_cutoff=cutoff,
        cluster_count=cluster_count,
        singleton_count=singletons,
        singleton_fraction=singleton_fraction,
        largest_cluster_sizes=tuple(sizes),
        representative_count=selected_reps,
        generated_conformer_count=generated,
        attempted_conformer_count=attempted,
        converged_conformer_count=converged,
        unconverged_conformer_count=unconverged,
        optimization_comparison_scope=e06["comparison_scope"],
        candidate_pool_count=candidate_pool_count,
        candidate_cluster_count=candidate_cluster_count,
        final_panel_count=final_panel_count,
        final_cluster_count=final_cluster_count,
        baseline_distance=baseline,
        benchmark_distance=benchmark,
        target_distance=target,
        final_distance=final,
        target_margin=(score_key(final) - score_key(target)) / 10**12,
        final_max_similarity=final_similarity,
        limiting_pairs=limiting_pairs,
        limiting_similarities=limiting_similarities,
        attempt_count=_integer(o01["attempt_count"], "O01 attempt count"),
        termination_reason=reason,
        achieved=achieved,
        headline=headline,
        facts=facts,
    )


@dataclass(frozen=True)
class _FindingPredicate:
    finding_id: str
    theme: str
    evidence_keys: tuple[str, ...]
    holds: Callable[[MeasuredSummary], bool]
    render: Callable[[MeasuredSummary], str]


_FINDING_PREDICATES: dict[str, _FindingPredicate] = {
    "all_rows_valid": _FindingPredicate(
        "F01",
        "dataset_scope",
        ("E01",),
        lambda summary: summary.valid_molecule_count > 0,
        lambda summary: (
            f"The dataset contained {summary.valid_molecule_count} valid molecules from "
            f"{summary.raw_count} raw rows."
        ),
    ),
    "exclusion_scope": _FindingPredicate(
        "F02",
        "dataset_scope",
        ("E01",),
        lambda summary: summary.excluded_count >= 0,
        lambda summary: (
            f"Input validation excluded {summary.excluded_count} rows; conclusions "
            "apply only to the retained valid molecules."
        ),
    ),
    "has_invalid_rows": _FindingPredicate(
        "F02-invalid",
        "dataset_scope",
        ("E01",),
        lambda summary: summary.invalid_count > 0,
        lambda summary: f"Input validation excluded {summary.invalid_count} invalid rows.",
    ),
    "representation_definition": _FindingPredicate(
        "F03",
        "molecular_representation",
        ("E02",),
        lambda summary: summary.fingerprint_radius in (2, 3)
        and summary.fingerprint_size in (1024, 2048),
        lambda summary: f"Molecular structure was represented with {summary.representation_name}.",
    ),
    "representation_reuse": _FindingPredicate(
        "F04",
        "molecular_representation",
        ("E02", "E03", "E04", "O01"),
        lambda summary: summary.fingerprint_radius in (2, 3)
        and summary.fingerprint_size in (1024, 2048),
        lambda summary: (
            "The same Morgan/Tanimoto representation was reused for similarity, "
            "clustering evidence, and objective scoring."
        ),
    ),
    "similarity_distribution": _FindingPredicate(
        "F05",
        "similarity_structure",
        ("E03",),
        lambda summary: bool(summary.similarity_quartiles),
        lambda summary: (
            "Pairwise Tanimoto similarity quartiles were "
            f"{summary.similarity_quartiles[0]:.3f}, "
            f"{summary.similarity_quartiles[1]:.3f}, and "
            f"{summary.similarity_quartiles[2]:.3f}."
        ),
    ),
    "most_similar_pair": _FindingPredicate(
        "F06",
        "similarity_structure",
        ("E03",),
        lambda summary: len(summary.most_similar_pair_ids) == 2,
        lambda summary: (
            f"The most-similar named pair was {summary.most_similar_pair_ids[0]} "
            f"and {summary.most_similar_pair_ids[1]} at Tanimoto "
            f"{summary.most_similar_pair_similarity:.3f}."
        ),
    ),
    "cluster_totals": _FindingPredicate(
        "F07",
        "clustering",
        ("E04",),
        lambda summary: summary.cluster_count > 0,
        lambda summary: (
            f"A {summary.cluster_cutoff!r} Tanimoto-distance cutoff produced "
            f"{summary.cluster_count} clusters, including "
            f"{summary.singleton_count} singletons."
        ),
    ),
    "largest_cluster_structure": _FindingPredicate(
        "F08",
        "clustering",
        ("E04",),
        lambda summary: bool(summary.largest_cluster_sizes),
        lambda summary: (
            "The largest cluster sizes were "
            + ", ".join(str(value) for value in summary.largest_cluster_sizes[:5])
            + "."
        ),
    ),
    "convergence_totals": _FindingPredicate(
        "F09",
        "conformational_sampling",
        ("E05", "E06"),
        lambda summary: summary.attempted_conformer_count
        == summary.converged_conformer_count + summary.unconverged_conformer_count,
        lambda summary: (
            f"MMFF94 converged {summary.converged_conformer_count} of "
            f"{summary.attempted_conformer_count} sampled conformers; "
            f"{summary.unconverged_conformer_count} did not converge."
        ),
    ),
    "within_molecule_energy_scope": _FindingPredicate(
        "F10",
        "conformational_sampling",
        ("E06",),
        lambda summary: summary.optimization_comparison_scope == "within molecule only",
        lambda summary: (
            "Lowest-energy selection was performed within each molecule among "
            "converged sampled conformers."
        ),
    ),
    "target_result": _FindingPredicate(
        "F11",
        "objective_driven_selection",
        ("O01",),
        lambda summary: bool(summary.termination_reason),
        lambda summary: summary.headline,
    ),
    "final_panel_cluster_coverage": _FindingPredicate(
        "F12",
        "objective_driven_selection",
        ("O01",),
        lambda summary: summary.candidate_pool_count == 8
        and summary.candidate_cluster_count == 8
        and summary.final_panel_count == 4
        and summary.final_cluster_count == 4,
        lambda summary: (
            "The eight-candidate pool spans 8 distinct clusters; the "
            "four-compound final panel spans 4 distinct clusters."
        ),
    ),
    "bounded_scope_limit": _FindingPredicate(
        "F13",
        "limitations_and_next_steps",
        ("O01",),
        lambda summary: summary.candidate_pool_count == 8,
        lambda summary: (
            "The result is bounded to the eight-candidate objective pool and "
            "does not establish broader chemical-space coverage."
        ),
    ),
    "next_experimental_validation": _FindingPredicate(
        "F14",
        "limitations_and_next_steps",
        ("E06", "O01"),
        lambda summary: summary.final_panel_count == 4,
        lambda summary: (
            "Next, subject the four-compound final panel to experimental "
            "validation before drawing activity or developability conclusions."
        ),
    ),
}


def validate_finding(
    finding: EvidenceFinding,
    snapshot: EvidenceSnapshot,
) -> EvidenceFinding:
    """Revalidate a finding at later hosted-selection and UI boundaries."""
    if type(finding) is not EvidenceFinding or type(snapshot) is not EvidenceSnapshot:
        raise ValueError("Finding validation requires exact finding and snapshot values.")
    try:
        predicate = _FINDING_PREDICATES[finding.predicate_id]
    except KeyError as error:
        raise ValueError("Finding uses an unknown predicate.") from error
    if not predicate.holds(snapshot.summary):
        raise ValueError("Finding predicate is false for this evidence snapshot.")
    if finding.finding_id != predicate.finding_id:
        raise ValueError("Finding ID does not match its predicate.")
    if finding.theme != predicate.theme:
        raise ValueError("Finding theme does not match its predicate.")
    if finding.evidence_keys != predicate.evidence_keys:
        raise ValueError("Finding evidence keys do not match its predicate.")
    if finding.text != predicate.render(snapshot.summary):
        raise ValueError("Finding text is not the deterministic snapshot rendering.")
    return finding


@dataclass(frozen=True)
class FindingCatalog:
    findings: tuple[EvidenceFinding, ...]

    def __post_init__(self) -> None:
        if type(self.findings) is not tuple or any(
            type(finding) is not EvidenceFinding for finding in self.findings
        ):
            raise ValueError("Finding catalog requires an exact finding tuple.")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Finding catalog IDs must be unique.")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(finding.finding_id for finding in self.findings)

    def ids_for_theme(self, theme: str) -> tuple[str, ...]:
        if theme not in CONCLUSION_THEMES:
            raise ValueError("Unknown conclusion theme.")
        return tuple(
            finding.finding_id
            for finding in self.findings
            if finding.theme == theme
        )


def build_evidence_snapshot(
    report: WorkflowReport,
    run: ObjectiveRun,
) -> EvidenceSnapshot:
    return EvidenceSnapshot.from_report(report, run)


def build_measured_summary(
    report: WorkflowReport,
    run: ObjectiveRun,
) -> MeasuredSummary:
    return build_evidence_snapshot(report, run).summary


def build_finding_catalog_from_snapshot(
    snapshot: EvidenceSnapshot,
) -> FindingCatalog:
    if type(snapshot) is not EvidenceSnapshot:
        raise ValueError("Finding catalog requires an exact evidence snapshot.")
    findings = tuple(
        EvidenceFinding(
            finding_id=predicate.finding_id,
            theme=predicate.theme,
            evidence_keys=predicate.evidence_keys,
            predicate_id=predicate_id,
            text=predicate.render(snapshot.summary),
        )
        for predicate_id, predicate in _FINDING_PREDICATES.items()
        if predicate_id != "has_invalid_rows" and predicate.holds(snapshot.summary)
    )
    for finding in findings:
        validate_finding(finding, snapshot)
    return FindingCatalog(findings)


def build_finding_catalog(
    report: WorkflowReport,
    run: ObjectiveRun,
) -> FindingCatalog:
    return build_finding_catalog_from_snapshot(build_evidence_snapshot(report, run))
