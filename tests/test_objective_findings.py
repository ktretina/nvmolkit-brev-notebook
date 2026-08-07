import json
from dataclasses import replace

import pytest

from chemistry_workflow import EvidenceRecord, WorkflowReport
from objective_challenge import TerminationReason, build_objective_evidence
from objective_findings import (
    CONCLUSION_THEMES,
    EvidenceFinding,
    EvidenceSnapshot,
    FindingCatalog,
    validate_finding,
)
from objective_fixtures import evidence_report, report_and_run


def mutate_record(report, key, mutate):
    records = []
    for record in report.evidence:
        if record.key == key:
            payload = json.loads(record.payload_json)
            mutate(payload)
            record = replace(
                record,
                payload_json=json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=True
                ),
            )
        records.append(record)
    return WorkflowReport(tuple(records))


@pytest.mark.parametrize("reason", tuple(TerminationReason))
def test_snapshot_parses_every_terminal_reason_with_truthful_headline(reason):
    report, run = report_and_run(reason)

    snapshot = EvidenceSnapshot.from_report(report, run)

    assert snapshot.summary.termination_reason == reason.value
    assert snapshot.summary.achieved is run.achieved
    assert reason.value.replace("_", " ") in snapshot.summary.headline.lower()


def test_measured_summary_exposes_all_measured_fields_and_complement():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)

    summary = EvidenceSnapshot.from_report(report, run).summary

    assert (summary.raw_count, summary.valid_count, summary.invalid_count) == (256, 256, 0)
    assert (summary.fingerprint_radius, summary.fingerprint_size_bits) == (2, 1024)
    assert (summary.similarity_q1, summary.similarity_median, summary.similarity_q3, summary.similarity_p90) == (0.071, 0.118, 0.184, 0.291)
    assert summary.most_similar_pair_ids == ("mol-17", "mol-203")
    assert summary.most_similar_pair_similarity == 0.873
    assert (summary.cluster_cutoff, summary.cluster_count, summary.singleton_count) == (0.4, 70, 37)
    assert summary.largest_cluster_sizes[:4] == (15, 12, 11, 10)
    assert (summary.selected_representative_count, summary.generated_conformer_count) == (4, 20)
    assert (summary.converged_conformer_count, summary.unconverged_conformer_count) == (19, 1)
    assert (summary.candidate_cluster_count, summary.final_cluster_count) == (8, 4)
    assert summary.final_max_similarity == pytest.approx(1.0 - summary.final_min_distance)
    assert summary.optimization_comparison_scope == "within molecule only"


@pytest.mark.parametrize(
    ("key", "mutation", "message"),
    (
        ("E01", lambda payload: payload.pop("raw_count"), "missing"),
        ("E01", lambda payload: payload.__setitem__("valid_count", 255), "count"),
        ("E03", lambda payload: payload.__setitem__("median", float("nan")), "finite"),
        ("E04", lambda payload: payload.__setitem__("assignment_count", 255), "count"),
        ("E06", lambda payload: payload.__setitem__("converged_conformer_count", 20), "count"),
    ),
)
def test_snapshot_rejects_missing_nonfinite_and_contradictory_evidence(key, mutation, message):
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)

    with pytest.raises(ValueError, match=message):
        EvidenceSnapshot.from_report(mutate_record(report, key, mutation), run)


def test_snapshot_rejects_o01_that_does_not_reconstruct_supplied_run():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    forged_o01 = replace(
        build_objective_evidence(run),
        payload_json=build_objective_evidence(
            report_and_run(TerminationReason.OBJECTIVE_PROVIDER_FAILURE)[1]
        ).payload_json,
    )

    with pytest.raises(ValueError, match="O01"):
        EvidenceSnapshot.from_records((*report.evidence, forged_o01), run)


def test_catalog_is_validated_deterministic_and_has_nonvacuous_choices():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    snapshot = EvidenceSnapshot.from_report(report, run)

    catalog = FindingCatalog.from_snapshot(snapshot)

    assert tuple(catalog.by_theme) == CONCLUSION_THEMES
    assert all(len(catalog.for_theme(theme)) >= 2 for theme in CONCLUSION_THEMES)
    assert sum(len(catalog.for_theme(theme)) > 1 for theme in CONCLUSION_THEMES) >= 4
    assert all(
        validate_finding(finding, snapshot) is finding
        for finding in catalog.findings
    )
    assert catalog == FindingCatalog.from_snapshot(snapshot)


def test_findings_preserve_candidate_final_and_energy_scope_boundaries():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    catalog = FindingCatalog.from_snapshot(EvidenceSnapshot.from_report(report, run))
    text = " ".join(finding.text for finding in catalog.findings).lower()

    assert "8 candidate clusters" in text
    assert "4 final clusters" in text
    assert "within each molecule" in text
    assert "across molecules" not in text


def test_validate_finding_rejects_unknown_false_wrong_keys_and_mutated_text():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    snapshot = EvidenceSnapshot.from_report(report, run)
    finding = FindingCatalog.from_snapshot(snapshot).findings[0]

    for forged, message in (
        (replace(finding, predicates=("unknown_predicate",)), "unknown"),
        (replace(finding, predicates=("has_invalid_rows",)), "false"),
        (replace(finding, evidence_keys=("E06",)), "evidence"),
        (replace(finding, text=finding.text + " altered"), "deterministic"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_finding(forged, snapshot)


def test_mutated_snapshot_requires_catalog_revalidation():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    snapshot = EvidenceSnapshot.from_report(report, run)
    finding = next(
        finding
        for finding in FindingCatalog.from_snapshot(snapshot).findings
        if "37 singleton" in finding.text
    )
    mutated = mutate_record(
        report, "E04", lambda payload: payload.update(singleton_count=38, singleton_fraction=38 / 256)
    )
    mutated_snapshot = EvidenceSnapshot.from_report(mutated, run)

    with pytest.raises(ValueError, match="deterministic"):
        validate_finding(finding, mutated_snapshot)


@pytest.mark.parametrize(
    ("key", "mutation"),
    (
        ("E05", lambda payload: payload["representatives"][0].pop("cluster_id")),
        ("E06", lambda payload: payload["per_conformer_records"][0].pop("energy_kcal_mol")),
        ("E06", lambda payload: payload["selected_conformer_records"][0].update(conformer_index=1)),
    ),
)
def test_snapshot_rejects_nested_missing_or_unreconciled_conformer_records(key, mutation):
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)

    with pytest.raises(ValueError):
        EvidenceSnapshot.from_report(mutate_record(report, key, mutation), run)


def test_requested_themes_are_the_only_catalog_surfaces():
    report, run = report_and_run(TerminationReason.TARGET_ACHIEVED)
    snapshot = EvidenceSnapshot.from_report(report, run)

    catalog = FindingCatalog.from_snapshot(
        snapshot, themes=("cluster_structure", "objective_outcome")
    )

    assert tuple(catalog.by_theme) == ("cluster_structure", "objective_outcome")
    assert {finding.theme for finding in catalog.findings} == {
        "cluster_structure", "objective_outcome"
    }
