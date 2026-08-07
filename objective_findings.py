"""Deterministic, evidence-controlled findings for the objective Decision Ladder."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from chemistry_workflow import EvidenceRecord, WorkflowReport
from objective_challenge import (
    ObjectiveRun,
    build_objective_evidence,
    validate_objective_evidence,
)


CONCLUSION_THEMES = (
    "data_quality",
    "fingerprint_configuration",
    "similarity_profile",
    "cluster_structure",
    "conformer_sampling",
    "optimization_quality",
    "objective_outcome",
)

_WORKFLOW_KEYS = tuple(f"E0{index}" for index in range(1, 7))
_ALL_KEYS = (*_WORKFLOW_KEYS, "O01")


@dataclass(frozen=True)
class EvidenceFinding:
    theme: str
    text: str
    evidence_keys: tuple[str, ...]
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class MeasuredSummary:
    raw_count: int
    valid_count: int
    invalid_count: int
    fingerprint_radius: int
    fingerprint_size_bits: int
    similarity_q1: float
    similarity_median: float
    similarity_q3: float
    similarity_p90: float
    similarity_max: float
    most_similar_pair_ids: tuple[str, str]
    most_similar_pair_similarity: float
    cluster_cutoff: float
    cluster_count: int
    singleton_count: int
    singleton_fraction: float
    largest_cluster_sizes: tuple[int, ...]
    selected_representative_count: int
    generated_conformer_count: int
    converged_conformer_count: int
    unconverged_conformer_count: int
    optimization_comparison_scope: str
    candidate_cluster_count: int
    final_cluster_count: int
    baseline_min_distance: float
    benchmark_min_distance: float
    target_min_distance: float
    final_min_distance: float
    final_max_similarity: float
    attempt_count: int
    termination_reason: str
    achieved: bool
    headline: str


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


def _number(value: Any, label: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if not lower <= result <= upper:
        raise ValueError(f"{label} must be in [{lower}, {upper}].")
    return result


def _headline(reason: str, final_distance: float, achieved: bool) -> str:
    readable = reason.replace("_", " ")
    status = "achieved" if achieved else "not achieved"
    return (
        f"{readable.capitalize()}: objective {status} at final minimum "
        f"distance {final_distance:.3f}."
    )


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
        cls, records: Iterable[EvidenceRecord], run: ObjectiveRun
    ) -> "EvidenceSnapshot":
        if type(run) is not ObjectiveRun:
            raise ValueError("Evidence snapshot requires an exact objective run.")
        records = tuple(records)
        if any(type(record) is not EvidenceRecord for record in records):
            raise ValueError("Evidence snapshot requires exact evidence records.")
        by_key = {record.key: record for record in records}
        if len(by_key) != len(records) or tuple(sorted(by_key)) != _ALL_KEYS:
            raise ValueError("Evidence snapshot requires exactly one record for E01-E06 and O01.")

        payloads: dict[str, dict[str, Any]] = {}
        for key in _ALL_KEYS:
            record = by_key[key]
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
            if canonical != record.payload_json:
                raise ValueError(f"{key} payload must use canonical JSON.")
            payloads[key] = payload

        validate_objective_evidence(by_key["O01"], run)
        summary = _build_summary(payloads)
        return cls(tuple(by_key[key] for key in _ALL_KEYS), summary)

    def payload(self, key: str) -> Mapping[str, Any]:
        for record in self.records:
            if record.key == key:
                return MappingProxyType(json.loads(record.payload_json))
        raise KeyError(key)


def _build_summary(payloads: Mapping[str, Mapping[str, Any]]) -> MeasuredSummary:
    e01, e02, e03 = payloads["E01"], payloads["E02"], payloads["E03"]
    e04, e05, e06, o01 = (
        payloads["E04"], payloads["E05"], payloads["E06"], payloads["O01"]
    )
    _exact_keys("E01", e01, {"raw_count", "valid_count", "invalid_count", "invalid_ids", "preview_count", "count_unit"})
    _exact_keys("E02", e02, {"fingerprint_radius", "fingerprint_size_bits", "packed_shape", "molecule_count", "active_bits_min", "active_bits_median", "active_bits_max", "executor", "size_unit"})
    _exact_keys("E03", e03, {"matrix_shape", "q1", "median", "q3", "p90", "max_off_diagonal", "most_similar_pair", "similarity_unit"})
    _exact_keys("E04", e04, {"cutoff", "cluster_count", "singleton_count", "singleton_fraction", "largest_cluster_sizes", "assignment_count", "cutoff_unit"})
    _exact_keys("E05", e05, {"requested_representative_count", "selected_representative_count", "selection_shortfall", "representative_policy", "representatives", "requested_conformers_per_representative", "generated_conformer_count", "partial_embedding_ids", "zero_embedding_ids", "count_unit"})
    _exact_keys("E06", e06, {"attempted_conformer_count", "converged_conformer_count", "unconverged_conformer_count", "per_conformer_records", "selected_conformer_records", "energy_unit", "comparison_scope"})

    raw = _integer(e01["raw_count"], "E01 raw count")
    valid = _integer(e01["valid_count"], "E01 valid count")
    invalid = _integer(e01["invalid_count"], "E01 invalid count")
    if raw != valid + invalid or type(e01["invalid_ids"]) is not list or len(e01["invalid_ids"]) != invalid:
        raise ValueError("E01 count fields are contradictory.")

    radius = _integer(e02["fingerprint_radius"], "E02 fingerprint radius")
    size = _integer(e02["fingerprint_size_bits"], "E02 fingerprint size", minimum=1)
    molecule_count = _integer(e02["molecule_count"], "E02 molecule count")
    if molecule_count != valid or e02["packed_shape"] != [valid, size // 32]:
        raise ValueError("E02 count and packed-shape fields are contradictory.")
    for name in ("active_bits_min", "active_bits_median", "active_bits_max"):
        _number(e02[name], f"E02 {name}", upper=float(size))
    if not e02["active_bits_min"] <= e02["active_bits_median"] <= e02["active_bits_max"]:
        raise ValueError("E02 active-bit fields are contradictory.")

    similarities = tuple(_number(e03[name], f"E03 {name}") for name in ("q1", "median", "q3", "p90", "max_off_diagonal"))
    if tuple(sorted(similarities)) != similarities or e03["matrix_shape"] != [valid, valid]:
        raise ValueError("E03 similarity fields are contradictory.")
    pair = e03["most_similar_pair"]
    _exact_keys("E03 most_similar_pair", pair, {"molecule_ids", "source_rows", "similarity"})
    if (
        type(pair["molecule_ids"]) is not list
        or len(pair["molecule_ids"]) != 2
        or any(type(value) is not str or not value for value in pair["molecule_ids"])
        or type(pair["source_rows"]) is not list
        or len(pair["source_rows"]) != 2
        or any(type(value) is not int or value < 0 for value in pair["source_rows"])
        or _number(pair["similarity"], "E03 pair similarity") != similarities[-1]
    ):
        raise ValueError("E03 most-similar pair fields are contradictory.")

    cutoff = _number(e04["cutoff"], "E04 cutoff")
    cluster_count = _integer(e04["cluster_count"], "E04 cluster count", minimum=1)
    singletons = _integer(e04["singleton_count"], "E04 singleton count")
    assignments = _integer(e04["assignment_count"], "E04 assignment count")
    singleton_fraction = _number(e04["singleton_fraction"], "E04 singleton fraction")
    sizes = e04["largest_cluster_sizes"]
    if (
        assignments != valid
        or singletons > cluster_count
        or not math.isclose(singleton_fraction, singletons / assignments, rel_tol=0, abs_tol=1e-15)
        or type(sizes) is not list
        or not sizes
        or len(sizes) > min(15, cluster_count)
        or any(type(value) is not int or value < 1 for value in sizes)
        or sizes != sorted(sizes, reverse=True)
    ):
        raise ValueError("E04 cluster count fields are contradictory.")

    requested_reps = _integer(e05["requested_representative_count"], "E05 requested representatives")
    selected_reps = _integer(e05["selected_representative_count"], "E05 selected representatives")
    shortfall = _integer(e05["selection_shortfall"], "E05 selection shortfall")
    per_rep = _integer(e05["requested_conformers_per_representative"], "E05 conformers per representative")
    generated = _integer(e05["generated_conformer_count"], "E05 generated conformers")
    representatives = e05["representatives"]
    if (
        requested_reps != selected_reps + shortfall
        or type(representatives) is not list
        or len(representatives) != selected_reps
        or generated > selected_reps * per_rep
    ):
        raise ValueError("E05 representative count fields are contradictory.")
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

    attempted = _integer(e06["attempted_conformer_count"], "E06 attempted conformers")
    converged = _integer(e06["converged_conformer_count"], "E06 converged conformers")
    unconverged = _integer(e06["unconverged_conformer_count"], "E06 unconverged conformers")
    per_conformer = e06["per_conformer_records"]
    selected = e06["selected_conformer_records"]
    if (
        attempted != generated
        or attempted != converged + unconverged
        or type(per_conformer) is not list
        or len(per_conformer) != attempted
        or sum(record.get("converged") is True for record in per_conformer) != converged
        or type(selected) is not list
        or len(selected) > selected_reps
        or any(record.get("converged") is not True for record in selected)
        or e06["comparison_scope"] != "within molecule only"
    ):
        raise ValueError("E06 conformer count or comparison-scope fields are contradictory.")
    conformers_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    conformer_keys = {
        "molecule_id", "cluster_id", "conformer_index", "energy_kcal_mol", "converged"
    }
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
        _exact_keys("E06 selected conformer", record, {*conformer_keys, "selected_conformer_id"})
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

    baseline = _number(o01["baseline_score"], "O01 baseline score")
    benchmark = _number(o01["benchmark_score"], "O01 benchmark score")
    target = _number(o01["target_score"], "O01 target score")
    final = _number(o01["final_score"], "O01 final score")
    candidate_clusters = len(o01["candidate_cluster_ids"])
    final_clusters = len({
        o01["candidate_cluster_ids"][o01["candidate_ids"].index(molecule_id)]
        for molecule_id in o01["final_ids"]
    })
    reason = o01["termination_reason"]
    achieved = o01["achieved"]
    return MeasuredSummary(
        raw, valid, invalid, radius, size,
        similarities[0], similarities[1], similarities[2], similarities[3], similarities[4],
        tuple(pair["molecule_ids"]), float(pair["similarity"]), cutoff, cluster_count,
        singletons, singleton_fraction, tuple(sizes), selected_reps, generated, converged,
        unconverged, e06["comparison_scope"], candidate_clusters, final_clusters,
        baseline, benchmark, target, final, 1.0 - final, _integer(o01["attempt_count"], "O01 attempt count"),
        reason, achieved, _headline(reason, final, achieved),
    )


@dataclass(frozen=True)
class _FindingSpec:
    theme: str
    evidence_keys: tuple[str, ...]
    holds: Callable[[MeasuredSummary], bool]
    render: Callable[[MeasuredSummary], str]


_FINDING_SPECS: Mapping[str, _FindingSpec] = MappingProxyType({
    "all_rows_valid": _FindingSpec("data_quality", ("E01",), lambda s: s.invalid_count == 0 and s.raw_count == s.valid_count, lambda s: f"All {s.raw_count} input rows were valid; 0 invalid rows were retained."),
    "has_invalid_rows": _FindingSpec("data_quality", ("E01",), lambda s: s.invalid_count > 0, lambda s: f"Input validation rejected {s.invalid_count} of {s.raw_count} rows."),
    "validation_counts_reconciled": _FindingSpec("data_quality", ("E01", "E02"), lambda s: s.raw_count == s.valid_count + s.invalid_count, lambda s: f"Input accounting reconciled {s.raw_count} raw rows to {s.valid_count} valid and {s.invalid_count} invalid rows."),
    "morgan_radius_2": _FindingSpec("fingerprint_configuration", ("E02",), lambda s: s.fingerprint_radius == 2, lambda s: f"Morgan fingerprints used radius {s.fingerprint_radius}."),
    "fingerprint_size_1024": _FindingSpec("fingerprint_configuration", ("E02",), lambda s: s.fingerprint_size_bits == 1024, lambda s: f"Each Morgan fingerprint used {s.fingerprint_size_bits} bits."),
    "similarity_quartiles_available": _FindingSpec("similarity_profile", ("E03",), lambda s: s.similarity_q1 <= s.similarity_median <= s.similarity_q3, lambda s: f"Pairwise Tanimoto similarity quartiles were {s.similarity_q1:.3f}, {s.similarity_median:.3f}, and {s.similarity_q3:.3f}."),
    "most_similar_pair_available": _FindingSpec("similarity_profile", ("E03",), lambda s: bool(s.most_similar_pair_ids), lambda s: f"The most-similar named pair was {s.most_similar_pair_ids[0]} and {s.most_similar_pair_ids[1]} at Tanimoto {s.most_similar_pair_similarity:.3f}."),
    "clusters_measured": _FindingSpec("cluster_structure", ("E04",), lambda s: s.cluster_count > 0, lambda s: f"A {s.cluster_cutoff:.1f} Tanimoto-distance cutoff assigned the library to {s.cluster_count} clusters."),
    "singletons_present": _FindingSpec("cluster_structure", ("E04",), lambda s: s.singleton_count > 0, lambda s: f"The clustering contained {s.singleton_count} singleton clusters ({s.singleton_fraction:.1%} of molecules)."),
    "representatives_complete": _FindingSpec("conformer_sampling", ("E05",), lambda s: s.selected_representative_count == 4, lambda s: f"Embedding selected all {s.selected_representative_count} requested cluster representatives."),
    "conformers_generated": _FindingSpec("conformer_sampling", ("E05",), lambda s: s.generated_conformer_count > 0, lambda s: f"The {s.selected_representative_count} representatives generated {s.generated_conformer_count} conformers."),
    "optimization_counts_reconciled": _FindingSpec("optimization_quality", ("E06",), lambda s: s.converged_conformer_count + s.unconverged_conformer_count == s.generated_conformer_count, lambda s: f"MMFF94 converged {s.converged_conformer_count} of {s.generated_conformer_count} conformers, with {s.unconverged_conformer_count} unconverged."),
    "within_molecule_energy_scope": _FindingSpec("optimization_quality", ("E06",), lambda s: s.optimization_comparison_scope == "within molecule only", lambda s: "Lowest-energy conformers were selected within each molecule only; energy comparisons remained molecule-local."),
    "bounded_candidate_to_final": _FindingSpec("objective_outcome", ("O01",), lambda s: s.candidate_cluster_count == 8 and s.final_cluster_count == 4, lambda s: f"The bounded objective compared {s.candidate_cluster_count} candidate clusters and retained {s.final_cluster_count} final clusters."),
    "distance_similarity_complement": _FindingSpec("objective_outcome", ("E03", "O01"), lambda s: math.isclose(s.final_max_similarity, 1.0 - s.final_min_distance, rel_tol=0, abs_tol=1e-15), lambda s: f"The final minimum distance was {s.final_min_distance:.3f}, equivalent to a maximum final-panel similarity of {s.final_max_similarity:.3f}."),
    "terminal_outcome": _FindingSpec("objective_outcome", ("O01",), lambda s: bool(s.termination_reason), lambda s: s.headline),
})


def validate_finding(finding: EvidenceFinding, snapshot: EvidenceSnapshot) -> EvidenceFinding:
    """Revalidate a finding at any later hosted-selection or UI boundary."""
    if type(finding) is not EvidenceFinding or type(snapshot) is not EvidenceSnapshot:
        raise ValueError("Finding validation requires exact finding and snapshot values.")
    if finding.theme not in CONCLUSION_THEMES:
        raise ValueError("Finding uses an unknown theme.")
    if type(finding.predicates) is not tuple or not finding.predicates:
        raise ValueError("Finding requires a closed predicate tuple.")
    unknown = tuple(name for name in finding.predicates if name not in _FINDING_SPECS)
    if unknown:
        raise ValueError(f"Finding uses an unknown predicate: {unknown[0]}")
    if len(finding.predicates) != 1:
        raise ValueError("Finding predicate combination is not registered.")
    spec = _FINDING_SPECS[finding.predicates[0]]
    if not spec.holds(snapshot.summary):
        raise ValueError("Finding predicate is false for this evidence snapshot.")
    if finding.theme != spec.theme:
        raise ValueError("Finding theme does not match its predicate.")
    if finding.evidence_keys != spec.evidence_keys:
        raise ValueError("Finding evidence keys do not match its predicate.")
    if finding.text != spec.render(snapshot.summary):
        raise ValueError("Finding text is not the deterministic snapshot rendering.")
    return finding


@dataclass(frozen=True)
class FindingCatalog:
    findings: tuple[EvidenceFinding, ...]
    themes: tuple[str, ...] = CONCLUSION_THEMES

    def __post_init__(self) -> None:
        if type(self.findings) is not tuple or any(type(item) is not EvidenceFinding for item in self.findings):
            raise ValueError("Finding catalog requires an exact finding tuple.")
        if (
            type(self.themes) is not tuple
            or len(set(self.themes)) != len(self.themes)
            or any(theme not in CONCLUSION_THEMES for theme in self.themes)
        ):
            raise ValueError("Finding catalog requires known unique themes.")
        if any(finding.theme not in self.themes for finding in self.findings):
            raise ValueError("Finding catalog contains an unrequested theme.")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: EvidenceSnapshot,
        themes: Iterable[str] = CONCLUSION_THEMES,
    ) -> "FindingCatalog":
        requested = tuple(themes)
        if len(set(requested)) != len(requested) or any(theme not in CONCLUSION_THEMES for theme in requested):
            raise ValueError("Finding catalog requested an unknown or duplicate theme.")
        findings = tuple(
            EvidenceFinding(spec.theme, spec.render(snapshot.summary), spec.evidence_keys, (predicate,))
            for theme in requested
            for predicate, spec in _FINDING_SPECS.items()
            if spec.theme == theme and spec.holds(snapshot.summary)
        )
        for finding in findings:
            validate_finding(finding, snapshot)
        return cls(findings, requested)

    @property
    def by_theme(self) -> Mapping[str, tuple[EvidenceFinding, ...]]:
        return MappingProxyType({theme: self.for_theme(theme) for theme in self.themes})

    def for_theme(self, theme: str) -> tuple[EvidenceFinding, ...]:
        if theme not in CONCLUSION_THEMES:
            raise ValueError("Unknown conclusion theme.")
        return tuple(finding for finding in self.findings if finding.theme == theme)
