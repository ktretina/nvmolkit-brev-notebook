import json
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from chemistry_workflow import WorkflowReport
from objective_challenge import (
    TerminationReason,
    accepted_maxima,
    build_action_menu,
    build_objective_evidence,
    evaluate_selected_swap,
    measure_panel,
    score_key,
    terminal_objective_run,
)
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
from objective_fixtures import context_from_distance, evidence_report, report_and_run


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


def mutate_record_metadata(report, key, **changes):
    return WorkflowReport(
        tuple(
            replace(record, **changes) if record.key == key else record
            for record in report.evidence
        )
    )


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
    assert summary.final_max_similarity == 1.0 - summary.final_distance
    assert summary.target_margin == (
        score_key(summary.final_distance) - score_key(summary.target_distance)
    ) / 10**12
    assert summary.candidate_pool_count == 8
    assert summary.final_panel_count == 4
    assert summary.limiting_pairs == run.attempts[-1].limiting_pairs
    assert summary.limiting_similarities == tuple(
        summary.final_max_similarity for _ in summary.limiting_pairs
    )


def test_raw_unequal_key_tied_target_has_zero_decision_margin():
    final = 0.5000000000003
    target = 0.5000000000003599
    benchmark = 0.5250000000004499
    matrix = np.full((8, 8), 0.4, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for first in range(4):
        for second in range(first + 1, 4):
            matrix[first, second] = matrix[second, first] = 0.95
    matrix[0, 1] = matrix[1, 0] = 0.4
    for baseline_index in (1, 2, 3):
        matrix[4, baseline_index] = matrix[baseline_index, 4] = final
    for first in range(4, 8):
        for second in range(first + 1, 8):
            matrix[first, second] = matrix[second, first] = benchmark
    context = context_from_distance(matrix)
    assert context.target_score == target
    baseline = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, baseline, 0)
    action = next(item for item in accepted_maxima(menu) if item.predicted_score == final)
    attempt = evaluate_selected_swap(context, menu, action, 1)
    run = terminal_objective_run(
        context, (attempt,), TerminationReason.TARGET_ACHIEVED
    )
    summary = build_measured_summary(evidence_report(), run)

    assert final != target
    assert score_key(final) == score_key(target)
    assert summary.achieved is True
    assert summary.target_margin == 0.0
    assert "Target achieved" in summary.headline


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


def test_snapshot_cross_checks_o01_cluster_ids_against_e04_domain():
    report, run = report_and_run()
    impossible = mutate_record(
        report,
        "E04",
        lambda payload: payload.update(
            cluster_count=4,
            singleton_count=0,
            singleton_fraction=0.0,
            largest_cluster_sizes=[100, 70, 50, 36],
        ),
    )

    with pytest.raises(ValueError, match="E04.*O01|O01.*cluster"):
        build_evidence_snapshot(impossible, run)


@pytest.mark.parametrize(
    ("radius", "size", "cutoff"),
    ((2, 1024, 0.4), (3, 2048, 0.456789), (3, 1024, 0.6)),
)
def test_supported_parameters_render_without_hardcoded_values_or_precision_loss(
    radius, size, cutoff
):
    report, run = report_and_run()
    report = mutate_record(
        report,
        "E02",
        lambda payload: payload.update(
            fingerprint_radius=radius,
            fingerprint_size_bits=size,
            packed_shape=[256, size // 32],
        ),
    )
    report = mutate_record(
        report, "E04", lambda payload: payload.update(cutoff=cutoff)
    )

    catalog = build_finding_catalog(report, run)
    representation = " ".join(
        finding.text
        for finding in catalog.findings
        if finding.theme == "molecular_representation"
    )
    clustering = " ".join(
        finding.text
        for finding in catalog.findings
        if finding.theme == "clustering"
    )

    assert len(catalog.ids_for_theme("molecular_representation")) >= 2
    assert f"radius-{radius}" in representation
    assert f"{size}-bit" in representation
    assert str(cutoff) in clustering


@pytest.mark.parametrize(
    ("key", "mutation", "message"),
    (
        (
            "E02",
            lambda payload: payload.update(fingerprint_radius=1),
            "parameters",
        ),
        (
            "E02",
            lambda payload: payload.update(
                fingerprint_size_bits=4096, packed_shape=[256, 128]
            ),
            "parameters",
        ),
        ("E04", lambda payload: payload.update(cutoff=0.600001), "cutoff"),
    ),
)
def test_snapshot_rejects_unsupported_production_parameter_ranges(
    key, mutation, message
):
    report, run = report_and_run()

    with pytest.raises(ValueError, match=message):
        build_evidence_snapshot(mutate_record(report, key, mutation), run)


def test_each_co_limiter_uses_its_actual_raw_distance_and_similarity():
    matrix = np.full((8, 8), 0.8, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    matrix[0, 1] = matrix[1, 0] = 0.5
    matrix[2, 3] = matrix[3, 2] = 0.5000000000004
    context = context_from_distance(matrix)
    run = terminal_objective_run(
        context, (), TerminationReason.OBJECTIVE_PROVIDER_FAILURE
    )

    summary = build_measured_summary(evidence_report(), run)

    assert summary.limiting_pairs == (("mol-0", "mol-1"), ("mol-2", "mol-3"))
    assert summary.limiting_similarities == (0.5, 1.0 - 0.5000000000004)
    assert len(set(summary.limiting_similarities)) == 2


@pytest.mark.parametrize(
    ("key", "field", "value", "message"),
    (
        ("E02", "executor", "CPU", "executor"),
        ("E02", "size_unit", "bytes", "unit"),
        ("E03", "similarity_unit", "Euclidean distance", "Tanimoto"),
        ("E06", "energy_unit", "joules", "energy"),
    ),
)
def test_snapshot_rejects_unknown_method_and_unit_literals(key, field, value, message):
    report, run = report_and_run()

    with pytest.raises(ValueError, match=message):
        build_evidence_snapshot(
            mutate_record(report, key, lambda payload: payload.update({field: value})),
            run,
        )


@pytest.mark.parametrize(
    ("key", "changes"),
    (
        ("E01", {"label": "Input"}),
        ("E02", {"provenance": "unknown"}),
        ("E04", {"label": "Clusters"}),
        ("E06", {"provenance": "CPU optimizer"}),
    ),
)
def test_snapshot_rejects_noncanonical_labels_and_provenance(key, changes):
    report, run = report_and_run()

    with pytest.raises(ValueError, match="label|provenance"):
        build_evidence_snapshot(mutate_record_metadata(report, key, **changes), run)


def test_snapshot_rejects_noncontiguous_conformer_indices():
    report, run = report_and_run()
    mutated = mutate_record(
        report,
        "E06",
        lambda payload: next(
            record
            for record in payload["per_conformer_records"]
            if record["molecule_id"] == "mol-3" and record["conformer_index"] == 4
        ).update(conformer_index=5),
    )

    with pytest.raises(ValueError, match="contiguous"):
        build_evidence_snapshot(mutated, run)


def test_snapshot_rejects_noncanonical_selected_conformer_on_energy_tie():
    report, run = report_and_run()

    def forge_tie(payload):
        tied = next(
            record
            for record in payload["per_conformer_records"]
            if record["molecule_id"] == "mol-0" and record["conformer_index"] == 1
        )
        tied["energy_kcal_mol"] = -10.0
        selected = next(
            record
            for record in payload["selected_conformer_records"]
            if record["molecule_id"] == "mol-0"
        )
        selected.update(
            conformer_index=1,
            energy_kcal_mol=-10.0,
            selected_conformer_id="mol-0:conf-1",
        )

    with pytest.raises(ValueError, match="canonical|selected"):
        build_evidence_snapshot(mutate_record(report, "E06", forge_tie), run)


@pytest.mark.parametrize(
    ("mutations", "message"),
    (
        (
            (
                ("E05", {"requested_representative_count": 0, "selected_representative_count": 0, "representatives": [], "generated_conformer_count": 0}),
                ("E06", {"attempted_conformer_count": 0, "converged_conformer_count": 0, "unconverged_conformer_count": 0, "per_conformer_records": [], "selected_conformer_records": []}),
            ),
            "representative|generated",
        ),
        ((("E05", {"requested_representative_count": 2}),), "representative"),
        ((("E05", {"requested_conformers_per_representative": 2}),), "conformers per representative"),
        ((("E05", {"requested_conformers_per_representative": 9}),), "conformers per representative"),
        ((("E05", {"generated_conformer_count": 0}),), "generated"),
    ),
)
def test_snapshot_rejects_out_of_policy_or_empty_embedding_cardinalities(
    mutations, message
):
    report, run = report_and_run()
    for key, updates in mutations:
        report = mutate_record(
            report, key, lambda payload, updates=updates: payload.update(updates)
        )

    with pytest.raises(ValueError, match=message):
        build_evidence_snapshot(report, run)


def test_representation_reuse_requires_clustering_evidence_and_revalidation():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    finding = next(
        finding
        for finding in build_finding_catalog_from_snapshot(snapshot).findings
        if finding.predicate_id == "representation_reuse"
    )

    assert finding.evidence_keys == ("E02", "E03", "E04", "O01")
    with pytest.raises(ValueError, match="evidence"):
        validate_finding(
            replace(finding, evidence_keys=("E02", "E03", "O01")), snapshot
        )


@pytest.mark.parametrize("selected_count", (1, 2))
def test_snapshot_rejects_fewer_than_three_consistent_selected_representatives(
    selected_count,
):
    report, run = report_and_run()
    retained_ids = {f"mol-{index}" for index in range(selected_count)}
    report = mutate_record(
        report,
        "E05",
        lambda payload: payload.update(
            selected_representative_count=selected_count,
            selection_shortfall=4 - selected_count,
            representatives=payload["representatives"][:selected_count],
            generated_conformer_count=selected_count * 5,
        ),
    )
    report = mutate_record(
        report,
        "E06",
        lambda payload: payload.update(
            attempted_conformer_count=selected_count * 5,
            converged_conformer_count=selected_count * 5,
            unconverged_conformer_count=0,
            per_conformer_records=[
                record
                for record in payload["per_conformer_records"]
                if record["molecule_id"] in retained_ids
            ],
            selected_conformer_records=[
                record
                for record in payload["selected_conformer_records"]
                if record["molecule_id"] in retained_ids
            ],
        ),
    )

    with pytest.raises(ValueError, match="selected representative"):
        build_evidence_snapshot(report, run)
