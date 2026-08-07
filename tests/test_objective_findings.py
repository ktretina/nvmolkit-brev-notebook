import json
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from chemistry_workflow import WorkflowReport
from objective_challenge import TerminationReason, build_objective_evidence
from objective_findings import (
    CONCLUSION_THEMES,
    EvidenceFinding,
    EvidenceSnapshot,
    FindingCatalog,
    _FINDING_PREDICATES,
    build_evidence_snapshot,
    build_finding_catalog,
    build_finding_catalog_from_snapshot,
    build_measured_summary,
    validate_finding,
)
from objective_fixtures import report_and_run


EXPECTED_THEMES = (
    "dataset_scope",
    "molecular_representation",
    "similarity_structure",
    "clustering",
    "conformational_sampling",
    "objective_driven_selection",
    "limitations_and_next_steps",
)

HEADLINE_FRAGMENTS = {
    TerminationReason.TARGET_ACHIEVED: "Target achieved",
    TerminationReason.BASELINE_ALREADY_OPTIMAL: "Baseline already optimal",
    TerminationReason.ATTEMPT_LIMIT_REACHED: "Objective not achieved within attempt limit",
    TerminationReason.NO_LEGAL_IMPROVING_SWAP: "No legal improving substitution",
    TerminationReason.OBJECTIVE_CORRECTION_LIMIT: "Objective selection stopped after invalid responses",
    TerminationReason.OBJECTIVE_PROVIDER_FAILURE: "Objective provider unavailable",
    TerminationReason.EVALUATION_NOT_COMPLETED: "Objective evaluation not completed",
}


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


def test_public_contract_uses_exact_themes_and_frozen_field_shapes():
    assert CONCLUSION_THEMES == EXPECTED_THEMES
    assert tuple(field.name for field in fields(EvidenceFinding)) == (
        "finding_id",
        "theme",
        "evidence_keys",
        "predicate_id",
        "text",
    )
    assert tuple(field.name for field in fields(FindingCatalog)) == ("findings",)
    finding = EvidenceFinding("F01", "dataset_scope", ("E01",), "all_rows_valid", "x")
    with pytest.raises(FrozenInstanceError):
        finding.text = "changed"


def test_public_builders_return_the_same_validated_snapshot_summary_and_catalog():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)

    assert isinstance(snapshot, EvidenceSnapshot)
    assert build_measured_summary(report, run) == snapshot.summary
    assert build_finding_catalog(report, run) == build_finding_catalog_from_snapshot(snapshot)


def test_target_fixture_preserves_exact_accepted_distance_and_complement():
    report, run = report_and_run()
    summary = build_measured_summary(report, run)

    assert summary.final_distance == 0.8374999910593033
    assert summary.final_max_similarity == 0.1625000089406967
    assert summary.target_margin == summary.final_distance - summary.target_distance
    assert summary.candidate_pool_count == 8
    assert summary.final_panel_count == 4
    assert summary.limiting_pairs == run.attempts[-1].limiting_pairs
    assert summary.limiting_similarities == tuple(
        summary.final_max_similarity for _ in summary.limiting_pairs
    )


@pytest.mark.parametrize("reason", tuple(TerminationReason))
def test_all_terminal_reasons_have_explicit_truthful_headlines(reason):
    report, run = report_and_run(reason)
    summary = build_measured_summary(report, run)

    assert summary.termination_reason == reason.value
    assert summary.achieved is run.achieved
    assert HEADLINE_FRAGMENTS[reason] in summary.headline


def test_measured_summary_exposes_workflow_counts_and_scope():
    report, run = report_and_run()
    summary = build_measured_summary(report, run)

    assert (summary.raw_count, summary.valid_molecule_count, summary.invalid_count) == (256, 256, 0)
    assert (summary.fingerprint_radius, summary.fingerprint_size) == (2, 1024)
    assert type(summary.facts) is tuple and summary.facts
    assert not hasattr(summary, "valid_count")
    assert not hasattr(summary, "fingerprint_size_bits")
    assert summary.similarity_quartiles == (0.071, 0.118, 0.184)
    assert summary.similarity_p90 == 0.291
    assert summary.most_similar_pair_ids == ("mol-17", "mol-203")
    assert summary.most_similar_pair_similarity == 0.873
    assert (summary.cluster_cutoff, summary.cluster_count, summary.singleton_count) == (0.4, 70, 37)
    assert summary.largest_cluster_sizes[:4] == (15, 12, 11, 10)
    assert (summary.representative_count, summary.generated_conformer_count) == (4, 20)
    assert (summary.converged_conformer_count, summary.unconverged_conformer_count) == (19, 1)
    assert summary.optimization_comparison_scope == "within molecule only"


def test_catalog_has_required_alternatives_ids_and_exact_scope_language():
    report, run = report_and_run()
    catalog = build_finding_catalog(report, run)

    assert catalog.ids == tuple(finding.finding_id for finding in catalog.findings)
    assert len(catalog.ids) == len(set(catalog.ids))
    assert all(len(catalog.ids_for_theme(theme)) >= 2 for theme in CONCLUSION_THEMES)
    assert sum(len(catalog.ids_for_theme(theme)) > 1 for theme in CONCLUSION_THEMES) >= 4
    text = " ".join(finding.text for finding in catalog.findings)
    assert "eight-candidate pool spans 8 distinct clusters" in text
    assert "four-compound final panel spans 4 distinct clusters" in text
    assert "within each molecule among converged sampled conformers" in text


def test_catalog_alternatives_cover_each_approved_subject_pair():
    report, run = report_and_run()
    catalog = build_finding_catalog(report, run)
    by_theme = {
        theme: " ".join(
            finding.text for finding in catalog.findings if finding.theme == theme
        ).lower()
        for theme in CONCLUSION_THEMES
    }

    assert "256 valid" in by_theme["dataset_scope"] and "excluded" in by_theme["dataset_scope"]
    assert "morgan" in by_theme["molecular_representation"] and "reused" in by_theme["molecular_representation"]
    assert "quartile" in by_theme["similarity_structure"] and "mol-17" in by_theme["similarity_structure"]
    assert "70 clusters" in by_theme["clustering"] and "largest" in by_theme["clustering"]
    assert "19" in by_theme["conformational_sampling"] and "within each molecule" in by_theme["conformational_sampling"]
    assert "target" in by_theme["objective_driven_selection"] and "final panel" in by_theme["objective_driven_selection"]
    assert "bounded" in by_theme["limitations_and_next_steps"] and "experimental validation" in by_theme["limitations_and_next_steps"]


def test_closed_predicate_registry_and_revalidation_reject_forgeries():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    finding = build_finding_catalog_from_snapshot(snapshot).findings[0]

    assert type(_FINDING_PREDICATES) is dict
    assert finding.predicate_id in _FINDING_PREDICATES
    assert validate_finding(finding, snapshot) is finding
    for forged, message in (
        (replace(finding, predicate_id="unknown"), "unknown"),
        (replace(finding, predicate_id="has_invalid_rows"), "false"),
        (replace(finding, evidence_keys=("E06",)), "evidence"),
        (replace(finding, text=finding.text + " changed"), "deterministic"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_finding(forged, snapshot)


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
    report, run = report_and_run()
    with pytest.raises(ValueError, match=message):
        build_evidence_snapshot(mutate_record(report, key, mutation), run)


def test_snapshot_rejects_o01_that_does_not_reconstruct_run():
    report, run = report_and_run()
    forged = replace(
        build_objective_evidence(run),
        payload_json=build_objective_evidence(
            report_and_run(TerminationReason.OBJECTIVE_PROVIDER_FAILURE)[1]
        ).payload_json,
    )
    with pytest.raises(ValueError, match="O01"):
        EvidenceSnapshot.from_records((*report.evidence, forged), run)


def test_mutated_snapshot_invalidates_previously_rendered_finding():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    finding = next(
        finding
        for finding in build_finding_catalog_from_snapshot(snapshot).findings
        if "37 singleton" in finding.text
    )
    mutated = mutate_record(
        report,
        "E04",
        lambda payload: payload.update(
            singleton_count=38, singleton_fraction=38 / 256
        ),
    )
    with pytest.raises(ValueError, match="deterministic"):
        validate_finding(finding, build_evidence_snapshot(mutated, run))


@pytest.mark.parametrize(
    ("key", "mutation", "message"),
    (
        (
            "E03",
            lambda payload: payload["most_similar_pair"].update(
                molecule_ids=["mol-17", "mol-17"]
            ),
            "pair",
        ),
        (
            "E04",
            lambda payload: payload.update(
                largest_cluster_sizes=[200] + payload["largest_cluster_sizes"][1:]
            ),
            "cluster",
        ),
        (
            "E05",
            lambda payload: payload.update(partial_embedding_ids=["mol-0"]),
            "embedding",
        ),
        (
            "E06",
            lambda payload: payload["selected_conformer_records"].pop(),
            "selected",
        ),
    ),
)
def test_snapshot_rejects_exact_impossible_cross_record_payloads(
    key, mutation, message
):
    report, run = report_and_run()

    with pytest.raises(ValueError, match=message):
        build_evidence_snapshot(mutate_record(report, key, mutation), run)
