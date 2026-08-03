import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from rdkit import Chem

import chemistry_workflow

from chemistry_workflow import (
    EvidenceRecord,
    RepresentativePolicy,
    StageResult,
    WorkflowReport,
    WorkflowPhase,
    WorkflowState,
    build_workflow_report,
    discover_fused_butina_clusters,
    embed_representative_conformers,
    eligible_stage,
    generate_morgan_fingerprints,
    inspect_library,
    measure_tanimoto_similarity,
    optimize_conformers_mmff94,
    select_representatives,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _state_snapshot(state: WorkflowState) -> dict[str, object]:
    return {
        "phase": state.phase,
        "records": list(state.records),
        "molecules": list(state.molecules),
        "fingerprints": state.fingerprints,
        "similarity": state.similarity,
        "clusters": list(state.clusters),
        "representative_records": list(state.representative_records),
        "conformer_molecules": list(state.conformer_molecules),
        "optimization_result": state.optimization_result,
        "summaries": dict(state.summaries),
    }


def _clustered_state() -> WorkflowState:
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "C1CC1"]
    source_rows = [10, 2, 7, 4, 8, 1, 12]
    clusters = [[0, 1, 2], [3, 4], [5], [6]]
    candidates = []
    for cluster_id, cluster in enumerate(clusters):
        candidates.append(
            {
                "cluster_id": cluster_id,
                "candidate_ids": [f"mol-{index}" for index in cluster],
                "source_rows": [source_rows[index] for index in cluster],
                "is_singleton": len(cluster) == 1,
            }
        )
    return WorkflowState(
        phase=WorkflowPhase.CLUSTERED,
        records=[
            {"id": f"mol-{index}", "smiles": value, "source_row": source_rows[index]}
            for index, value in enumerate(smiles)
        ],
        molecules=[Chem.MolFromSmiles(value) for value in smiles],
        clusters=clusters,
        summaries={
            "inspect_library": {
                "raw_count": 8,
                "valid_count": 7,
                "invalid_count": 1,
                "invalid_ids": ["invalid"],
                "preview_count": 7,
                "executor": "RDKit input validation",
            },
            "generate_morgan_fingerprints": {
                "entry_point": "MorganFingerprintGenerator",
                "fingerprint_radius": 2,
                "fingerprint_size": 1024,
                "molecule_count": 7,
                "packed_shape": [7, 32],
                "active_bits_min": 3,
                "active_bits_median": 4.0,
                "active_bits_max": 6,
                "cuda_device": "cuda:0",
            },
            "measure_tanimoto_similarity": {
                "entry_point": "crossTanimotoSimilarity",
                "matrix_shape": [7, 7],
                "q1": 0.1,
                "median": 0.2,
                "q3": 0.3,
                "p90": 0.6,
                "max": 0.9,
                "most_similar_nonidentical_pair": {
                    "molecule_ids": ["mol-0", "mol-1"],
                    "source_rows": [10, 2],
                    "similarity": 0.9,
                },
            },
            "discover_fused_butina_clusters": {
                "entry_point": "fused_butina",
                "cluster_cutoff": 0.5,
                "molecule_count": 7,
                "cluster_count": 4,
                "singleton_count": 2,
                "singleton_fraction": 2 / 7,
                "largest_cluster_sizes": [3, 2, 1, 1],
                "assignment_count": 7,
                "representative_eligibility": {
                    "eligible_cluster_count": 4,
                    "eligible_singleton_count": 2,
                    "maximum_representative_count": 4,
                    "candidates_by_cluster": candidates,
                },
            },
        },
    )


def test_new_state_exposes_only_input_inspection():
    state = WorkflowState()

    assert state.phase is WorkflowPhase.NEW
    assert eligible_stage(state) == "inspect_library"
    assert state.records == []
    assert state.molecules == []
    assert state.fingerprints is None
    assert state.similarity is None
    assert state.clusters == []
    assert state.representative_records == []
    assert state.conformer_molecules == []
    assert state.optimization_result is None
    assert state.summaries == {}


@pytest.mark.parametrize(
    ("phase", "stage"),
    [
        (WorkflowPhase.NEW, "inspect_library"),
        (WorkflowPhase.INSPECTED, "generate_morgan_fingerprints"),
        (WorkflowPhase.FINGERPRINTED, "measure_tanimoto_similarity"),
        (WorkflowPhase.COMPARED, "discover_fused_butina_clusters"),
        (WorkflowPhase.CLUSTERED, "embed_representative_conformers"),
        (WorkflowPhase.EMBEDDED, "optimize_conformers_mmff94"),
        (WorkflowPhase.OPTIMIZED, "submit_synthesis"),
    ],
)
def test_eligible_stage_maps_every_phase_exactly(phase: WorkflowPhase, stage: str):
    assert eligible_stage(WorkflowState(phase=phase)) == stage


def test_stage_result_is_frozen():
    result = StageResult(
        stage="inspect_library",
        display_label="RDKit input validation",
        summary={},
    )

    assert result.figures == ()
    with pytest.raises(FrozenInstanceError):
        result.stage = "other"  # type: ignore[misc]


def test_inspection_reports_invalid_rows_and_preserves_source_indices(
    tmp_path: Path,
):
    sample = _write_csv(
        tmp_path / "sample.csv",
        [
            {"id": "valid-1", "smiles": "CCO"},
            {"id": "invalid-1", "smiles": "not-smiles"},
            {"id": "valid-2", "smiles": "c1ccccc1"},
        ],
    )
    state = WorkflowState()

    result = inspect_library(state, sample, expected_rows=3)

    assert isinstance(result, StageResult)
    assert result.stage == "inspect_library"
    assert result.display_label == "RDKit input validation"
    assert result.summary == {
        "raw_count": 3,
        "valid_count": 2,
        "invalid_count": 1,
        "invalid_ids": ["invalid-1"],
        "preview_count": 2,
        "executor": "RDKit input validation",
    }
    assert state.phase is WorkflowPhase.INSPECTED
    assert state.records == [
        {"id": "valid-1", "smiles": "CCO", "source_row": 0},
        {"id": "valid-2", "smiles": "c1ccccc1", "source_row": 2},
    ]
    assert len(state.molecules) == 2
    assert state.summaries == {"inspect_library": result.summary}
    assert eligible_stage(state) == "generate_morgan_fingerprints"


def test_inspection_accepts_bundled_molecule_id_and_normalizes_records():
    data_path = PROJECT_ROOT / "data" / "sample_molecules.csv"
    source = pd.read_csv(data_path)
    state = WorkflowState()

    result = inspect_library(state, data_path)

    assert state.phase is WorkflowPhase.INSPECTED
    assert result.summary["raw_count"] == 256
    assert (
        result.summary["valid_count"] + result.summary["invalid_count"]
        == result.summary["raw_count"]
    )
    assert state.records[0]["id"] == source.iloc[0]["molecule_id"]
    assert state.records[0]["source_row"] == 0


def test_inspection_summary_is_json_safe_and_excludes_rdkit_molecules(
    tmp_path: Path,
):
    sample = _write_csv(tmp_path / "sample.csv", [{"id": "valid-1", "smiles": "CCO"}])
    state = WorkflowState()

    result = inspect_library(state, sample, expected_rows=1)

    assert json.loads(json.dumps(result.summary)) == result.summary
    assert all(value not in state.molecules for value in result.summary.values())


def test_inspection_rejects_wrong_row_count_without_mutating_state(
    tmp_path: Path,
):
    sample = _write_csv(tmp_path / "sample.csv", [{"id": "one", "smiles": "CCO"}])
    state = WorkflowState(summaries={"sentinel": {"kept": True}})
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="expected 3 rows"):
        inspect_library(state, sample, expected_rows=3)

    assert _state_snapshot(state) == before


@pytest.mark.parametrize(
    "rows",
    [
        [{"smiles": "CCO"}],
        [{"id": "one"}],
    ],
)
def test_inspection_rejects_missing_required_columns_without_mutating_state(
    tmp_path: Path, rows: list[dict[str, str]]
):
    sample = _write_csv(tmp_path / "sample.csv", rows)
    state = WorkflowState(summaries={"sentinel": {"kept": True}})
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="id and smiles"):
        inspect_library(state, sample, expected_rows=1)

    assert _state_snapshot(state) == before


@pytest.mark.parametrize(
    "phase",
    [
        WorkflowPhase.INSPECTED,
        WorkflowPhase.FINGERPRINTED,
        WorkflowPhase.COMPARED,
        WorkflowPhase.CLUSTERED,
        WorkflowPhase.EMBEDDED,
        WorkflowPhase.OPTIMIZED,
    ],
)
def test_inspection_rejects_repeated_or_out_of_phase_calls_without_mutation(
    tmp_path: Path, phase: WorkflowPhase
):
    sample = _write_csv(tmp_path / "sample.csv", [{"id": "one", "smiles": "CCO"}])
    state = WorkflowState(phase=phase, summaries={"sentinel": {"kept": True}})
    before = _state_snapshot(state)

    with pytest.raises(RuntimeError, match="NEW phase"):
        inspect_library(state, sample, expected_rows=1)

    assert _state_snapshot(state) == before


def test_inspection_rejects_all_invalid_rows_without_mutating_state(
    tmp_path: Path,
):
    sample = _write_csv(
        tmp_path / "sample.csv",
        [
            {"id": "invalid-1", "smiles": "not-smiles"},
            {"id": "invalid-2", "smiles": "also-not-smiles"},
        ],
    )
    state = WorkflowState(summaries={"sentinel": {"kept": True}})
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="zero valid molecules"):
        inspect_library(state, sample, expected_rows=2)

    assert _state_snapshot(state) == before


def test_inspection_caps_preview_at_24(tmp_path: Path):
    sample = _write_csv(
        tmp_path / "sample.csv",
        [{"id": f"valid-{index}", "smiles": "CCO"} for index in range(30)],
    )

    result = inspect_library(WorkflowState(), sample, expected_rows=30)

    assert result.summary["preview_count"] == 24


class _FakeTensor:
    def __init__(self, values, *, device="cuda:fake"):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.device = device

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class _FakeGpuResult:
    def __init__(self, values, *, device="cuda:fake"):
        self.tensor = _FakeTensor(values, device=device)

    def torch(self):
        return self.tensor


@pytest.fixture
def inspected_state():
    return WorkflowState(
        phase=WorkflowPhase.INSPECTED,
        records=[
            {"id": "mol-0", "smiles": "CCO", "source_row": 2},
            {"id": "mol-1", "smiles": "CC", "source_row": 5},
            {"id": "mol-2", "smiles": "[Xe]", "source_row": 9},
        ],
        molecules=[
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("[Xe]"),
        ],
        summaries={"inspect_library": {"valid_count": 3}},
    )


@pytest.fixture
def fake_gpu(monkeypatch):
    calls = {
        "generator": [],
        "fingerprints": [],
        "similarity": [],
        "cluster": [],
        "sync": 0,
    }
    packed = np.zeros((3, 64), dtype=np.int32)
    packed[0, 0] = 0b11
    packed[1, 0] = 0b111
    packed[2, 0] = 0b1111
    fingerprint_result = _FakeGpuResult(packed)
    similarity_result = _FakeGpuResult(
        [[1.0, 0.2, 0.8], [0.2, 1.0, 0.5], [0.8, 0.5, 1.0]]
    )

    class Generator:
        def __init__(self, *, radius, fpSize):
            calls["generator"].append((radius, fpSize))

        def GetFingerprints(self, molecules):
            calls["fingerprints"].append(molecules)
            return fingerprint_result

    def similarity(fingerprints):
        calls["similarity"].append(fingerprints)
        return similarity_result

    def cluster(fingerprints, *, cutoff):
        calls["cluster"].append((fingerprints, cutoff))
        return [[0, 2], [1]], [2, 1]

    def sync():
        calls["sync"] += 1

    monkeypatch.setattr(
        chemistry_workflow, "_morgan_generator_class", lambda: Generator
    )
    monkeypatch.setattr(chemistry_workflow, "_cross_tanimoto_similarity", similarity)
    monkeypatch.setattr(chemistry_workflow, "_fused_butina", cluster)
    monkeypatch.setattr(chemistry_workflow, "_synchronize_cuda", sync)
    return calls, fingerprint_result, similarity_result


def test_similarity_chain_records_gpu_calls_summaries_figures_and_eligibility(
    inspected_state, fake_gpu
):
    calls, fingerprint_result, similarity_result = fake_gpu

    fingerprint = generate_morgan_fingerprints(
        inspected_state, fingerprint_radius=3, fingerprint_size=2048
    )
    assert fingerprint.summary == {
        "entry_point": "MorganFingerprintGenerator",
        "fingerprint_radius": 3,
        "fingerprint_size": 2048,
        "molecule_count": 3,
        "packed_shape": [3, 64],
        "active_bits_min": 2,
        "active_bits_median": 3.0,
        "active_bits_max": 4,
        "cuda_device": "cuda:fake",
    }
    assert fingerprint.display_label == "nvMolKit MorganFingerprintGenerator"
    assert len(fingerprint.figures) == 1
    fingerprint_axes = fingerprint.figures[0].axes
    assert len(fingerprint_axes) == 1
    assert fingerprint_axes[0].get_title() == "Morgan fingerprint density"
    assert fingerprint_axes[0].get_xlabel() == (
        "Active Morgan fingerprint bits per molecule"
    )
    assert sum(patch.get_height() for patch in fingerprint_axes[0].patches) == 3
    assert inspected_state.fingerprints is fingerprint_result
    assert calls["generator"] == [(3, 2048)]
    assert calls["fingerprints"] == [inspected_state.molecules]
    assert inspected_state.phase is WorkflowPhase.FINGERPRINTED

    similarity = measure_tanimoto_similarity(inspected_state)
    assert similarity.summary == {
        "entry_point": "crossTanimotoSimilarity",
        "matrix_shape": [3, 3],
        "q1": 0.35,
        "median": 0.5,
        "q3": 0.65,
        "p90": pytest.approx(0.74),
        "max": 0.8,
        "most_similar_nonidentical_pair": {
            "molecule_ids": ["mol-0", "mol-2"],
            "source_rows": [2, 9],
            "similarity": 0.8,
        },
    }
    assert similarity.display_label == "nvMolKit crossTanimotoSimilarity"
    assert len(similarity.figures) == 1
    similarity_axes = similarity.figures[0].axes
    assert len(similarity_axes) == 2
    assert similarity_axes[0].get_title() == "All-pairs Tanimoto similarity"
    assert len(similarity_axes[0].images) == 1
    assert similarity_axes[0].images[0].get_array().shape == (3, 3)
    assert similarity_axes[0].images[0].get_clim() == (0.0, 1.0)
    assert inspected_state.similarity is similarity_result
    assert calls["similarity"] == [fingerprint_result]
    assert inspected_state.phase is WorkflowPhase.COMPARED

    clusters = discover_fused_butina_clusters(inspected_state, cluster_cutoff=0.47)
    assert clusters.summary["entry_point"] == "fused_butina"
    assert clusters.summary["cluster_cutoff"] == 0.47
    assert clusters.summary["cluster_count"] == 2
    assert clusters.summary["singleton_count"] == 1
    assert clusters.summary["singleton_fraction"] == pytest.approx(1 / 3)
    assert clusters.summary["largest_cluster_sizes"] == [2, 1]
    assert clusters.summary["assignment_count"] == 3
    assert clusters.summary["representative_eligibility"] == {
        "eligible_cluster_count": 2,
        "eligible_singleton_count": 1,
        "maximum_representative_count": 2,
        "candidates_by_cluster": [
            {
                "cluster_id": 0,
                "candidate_ids": ["mol-0"],
                "source_rows": [2],
                "is_singleton": False,
            },
            {
                "cluster_id": 1,
                "candidate_ids": ["mol-1"],
                "source_rows": [5],
                "is_singleton": True,
            },
        ],
    }
    assert (
        clusters.display_label == "nvMolKit fused_butina with RDKit MMFF94 eligibility"
    )
    assert len(clusters.figures) == 1
    cluster_axes = clusters.figures[0].axes
    assert len(cluster_axes) == 1
    assert cluster_axes[0].get_title() == "Largest fused Butina clusters"
    assert [patch.get_height() for patch in cluster_axes[0].patches] == [2, 1]
    assert inspected_state.clusters == [[0, 2], [1]]
    assert calls["cluster"] == [(fingerprint_result.tensor, 0.47)]
    assert calls["sync"] == 3
    assert inspected_state.phase is WorkflowPhase.CLUSTERED
    assert all(
        json.loads(json.dumps(result.summary, allow_nan=False)) == result.summary
        for result in (fingerprint, similarity, clusters)
    )


@pytest.mark.parametrize("radius", [1, 4, True])
def test_fingerprint_radius_is_bounded_before_gpu_execution(
    inspected_state, fake_gpu, radius
):
    calls, _, _ = fake_gpu
    before = _state_snapshot(inspected_state)
    with pytest.raises(ValueError, match="radius must be 2 or 3"):
        generate_morgan_fingerprints(
            inspected_state, fingerprint_radius=radius, fingerprint_size=1024
        )
    assert calls["generator"] == []
    assert _state_snapshot(inspected_state) == before


@pytest.mark.parametrize("size", [512, 4096, True])
def test_fingerprint_size_is_bounded_before_gpu_execution(
    inspected_state, fake_gpu, size
):
    calls, _, _ = fake_gpu
    before = _state_snapshot(inspected_state)
    with pytest.raises(ValueError, match="size must be 1024 or 2048"):
        generate_morgan_fingerprints(
            inspected_state, fingerprint_radius=2, fingerprint_size=size
        )
    assert calls["generator"] == []
    assert _state_snapshot(inspected_state) == before


def test_fingerprint_rejects_bad_packed_shape_atomically(
    inspected_state, fake_gpu, monkeypatch
):
    calls, _, _ = fake_gpu
    bad_result = _FakeGpuResult(np.zeros((3, 31), dtype=np.int32))

    class Generator:
        def __init__(self, **kwargs):
            calls["generator"].append(kwargs)

        def GetFingerprints(self, molecules):
            return bad_result

    monkeypatch.setattr(
        chemistry_workflow, "_morgan_generator_class", lambda: Generator
    )
    before = _state_snapshot(inspected_state)
    with pytest.raises(RuntimeError, match="packed Morgan fingerprint shape"):
        generate_morgan_fingerprints(
            inspected_state, fingerprint_radius=2, fingerprint_size=1024
        )
    assert _state_snapshot(inspected_state) == before


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([[1.0, 0.2]], "shape"),
        ([[1.0, np.nan, 0.2], [np.nan, 1.0, 0.3], [0.2, 0.3, 1.0]], "non-finite"),
        ([[1.0, 0.2, 0.4], [0.3, 1.0, 0.5], [0.4, 0.5, 1.0]], "symmetric"),
        ([[1.0, -0.1, 0.4], [-0.1, 1.0, 1.2], [0.4, 1.2, 1.0]], "0 through 1"),
        ([[0.9, 0.2, 0.4], [0.2, 1.0, 0.5], [0.4, 0.5, 1.0]], "diagonal"),
    ],
)
def test_similarity_rejects_invalid_matrix_atomically(
    inspected_state, fake_gpu, monkeypatch, values, message
):
    _, fingerprint_result, _ = fake_gpu
    inspected_state.phase = WorkflowPhase.FINGERPRINTED
    inspected_state.fingerprints = fingerprint_result
    monkeypatch.setattr(
        chemistry_workflow,
        "_cross_tanimoto_similarity",
        lambda fingerprints: _FakeGpuResult(values),
    )
    before = _state_snapshot(inspected_state)
    with pytest.raises(RuntimeError, match=message):
        measure_tanimoto_similarity(inspected_state)
    assert _state_snapshot(inspected_state) == before


@pytest.mark.parametrize("cutoff", [0.399, 0.601, True])
def test_cluster_cutoff_is_bounded_before_gpu_execution(
    inspected_state, fake_gpu, cutoff
):
    calls, fingerprint_result, similarity_result = fake_gpu
    inspected_state.phase = WorkflowPhase.COMPARED
    inspected_state.fingerprints = fingerprint_result
    inspected_state.similarity = similarity_result
    before = _state_snapshot(inspected_state)
    with pytest.raises(ValueError, match="0.40 through 0.60"):
        discover_fused_butina_clusters(inspected_state, cluster_cutoff=cutoff)
    assert calls["cluster"] == []
    assert _state_snapshot(inspected_state) == before


@pytest.mark.parametrize("cutoff", [0.40, 0.60])
def test_cluster_cutoff_inclusive_boundaries_execute_and_are_forwarded_unchanged(
    inspected_state, fake_gpu, cutoff
):
    calls, fingerprint_result, similarity_result = fake_gpu
    inspected_state.phase = WorkflowPhase.COMPARED
    inspected_state.fingerprints = fingerprint_result
    inspected_state.similarity = similarity_result

    result = discover_fused_butina_clusters(inspected_state, cluster_cutoff=cutoff)

    assert calls["cluster"] == [(fingerprint_result.tensor, cutoff)]
    assert result.summary["cluster_cutoff"] == cutoff
    assert inspected_state.phase is WorkflowPhase.CLUSTERED


@pytest.mark.parametrize(
    "clusters",
    [
        [[0], [1]],
        [[0, 1], [1, 2]],
        [[0, 1], [2, 3]],
    ],
)
def test_cluster_rejects_incomplete_duplicate_or_out_of_range_assignment_atomically(
    inspected_state, fake_gpu, monkeypatch, clusters
):
    _, fingerprint_result, similarity_result = fake_gpu
    inspected_state.phase = WorkflowPhase.COMPARED
    inspected_state.fingerprints = fingerprint_result
    inspected_state.similarity = similarity_result
    monkeypatch.setattr(
        chemistry_workflow,
        "_fused_butina",
        lambda fingerprints, *, cutoff: (
            clusters,
            [len(cluster) for cluster in clusters],
        ),
    )
    before = _state_snapshot(inspected_state)
    with pytest.raises(RuntimeError, match="assigned exactly once"):
        discover_fused_butina_clusters(inspected_state, cluster_cutoff=0.5)
    assert _state_snapshot(inspected_state) == before


@pytest.mark.parametrize(
    ("phase", "function", "kwargs", "message"),
    [
        (
            WorkflowPhase.NEW,
            generate_morgan_fingerprints,
            {"fingerprint_radius": 2, "fingerprint_size": 1024},
            "INSPECTED",
        ),
        (
            WorkflowPhase.FINGERPRINTED,
            generate_morgan_fingerprints,
            {"fingerprint_radius": 2, "fingerprint_size": 1024},
            "INSPECTED",
        ),
        (WorkflowPhase.INSPECTED, measure_tanimoto_similarity, {}, "FINGERPRINTED"),
        (WorkflowPhase.COMPARED, measure_tanimoto_similarity, {}, "FINGERPRINTED"),
        (
            WorkflowPhase.FINGERPRINTED,
            discover_fused_butina_clusters,
            {"cluster_cutoff": 0.5},
            "COMPARED",
        ),
        (
            WorkflowPhase.CLUSTERED,
            discover_fused_butina_clusters,
            {"cluster_cutoff": 0.5},
            "COMPARED",
        ),
    ],
)
def test_similarity_chain_rejects_out_of_phase_and_repeat_before_science(
    inspected_state, fake_gpu, phase, function, kwargs, message
):
    calls, _, _ = fake_gpu
    inspected_state.phase = phase
    before = _state_snapshot(inspected_state)
    with pytest.raises(RuntimeError, match=message):
        function(inspected_state, **kwargs)
    assert calls["generator"] == []
    assert calls["similarity"] == []
    assert calls["cluster"] == []
    assert _state_snapshot(inspected_state) == before


def test_representative_types_are_exact_and_evidence_is_frozen_and_canonical():
    assert [policy.value for policy in RepresentativePolicy] == [
        "largest_clusters_first",
        "include_singleton_if_available",
    ]
    record = EvidenceRecord("E01", "Input", '{"a":1,"b":2}', "source")
    report = WorkflowReport(evidence=(record,))
    assert tuple(report.__dataclass_fields__) == ("evidence",)
    assert json.loads(record.payload_json) == {"a": 1, "b": 2}
    with pytest.raises(FrozenInstanceError):
        record.key = "E02"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.evidence = ()  # type: ignore[misc]


def test_representatives_follow_cluster_size_then_source_row_and_member_source_row():
    state = _clustered_state()
    selected, shortfall = select_representatives(
        state, 3, RepresentativePolicy.LARGEST_CLUSTERS_FIRST
    )
    assert shortfall == 0
    assert selected == [
        {"molecule_id": "mol-1", "source_row": 2, "cluster_id": 0, "molecule_index": 1},
        {"molecule_id": "mol-3", "source_row": 4, "cluster_id": 1, "molecule_index": 3},
        {"molecule_id": "mol-5", "source_row": 1, "cluster_id": 2, "molecule_index": 5},
    ]


def test_singleton_policy_reserves_one_singleton_without_duplicate_cluster():
    state = _clustered_state()
    state.clusters = [[0, 1], [2, 3], [4, 5], [6]]
    eligibility = state.summaries["discover_fused_butina_clusters"][
        "representative_eligibility"
    ]
    eligibility["candidates_by_cluster"] = [
        {
            "cluster_id": cluster_id,
            "candidate_ids": [f"mol-{index}" for index in cluster],
            "source_rows": [state.records[index]["source_row"] for index in cluster],
            "is_singleton": len(cluster) == 1,
        }
        for cluster_id, cluster in enumerate(state.clusters)
    ]
    selected, shortfall = select_representatives(
        state, 3, RepresentativePolicy.INCLUDE_SINGLETON_IF_AVAILABLE
    )
    assert shortfall == 0
    assert [item["cluster_id"] for item in selected] == [2, 0, 3]
    assert len({item["cluster_id"] for item in selected}) == 3


def test_representative_selection_reports_shortfall_and_rejects_fewer_than_three_clusters():
    state = _clustered_state()
    selected, shortfall = select_representatives(
        state, 6, RepresentativePolicy.LARGEST_CLUSTERS_FIRST
    )
    assert len(selected) == 4
    assert shortfall == 2

    eligibility = state.summaries["discover_fused_butina_clusters"][
        "representative_eligibility"
    ]
    eligibility["candidates_by_cluster"] = eligibility["candidates_by_cluster"][:2]
    with pytest.raises(RuntimeError, match="at least 3 eligible distinct clusters"):
        select_representatives(
            state, 3, RepresentativePolicy.LARGEST_CLUSTERS_FIRST
        )


class _FakeOptimizationResult:
    def __init__(self, molecules, *, pairs=None, energies=None, converged=None, coordinates=None):
        expected_pairs = [
            (mol_index, conf_index)
            for mol_index, molecule in enumerate(molecules)
            for conf_index in range(molecule.GetNumConformers())
        ]
        pairs = expected_pairs if pairs is None else pairs
        self.mol_indices = _FakeGpuResult([pair[0] for pair in pairs])
        self.conf_indices = _FakeGpuResult([pair[1] for pair in pairs])
        self.energies = _FakeGpuResult(
            list(range(1, len(pairs) + 1)) if energies is None else energies
        )
        self.converged = _FakeGpuResult(
            [1] * len(pairs) if converged is None else converged
        )
        if coordinates is None:
            coordinates = [
                [
                    np.full((molecule.GetNumAtoms(), 3), conf_index + mol_index, dtype=float)
                    for conf_index in range(molecule.GetNumConformers())
                ]
                for mol_index, molecule in enumerate(molecules)
            ]
        self._coordinates = coordinates

    def per_molecule(self):
        return self._coordinates


@pytest.fixture
def conformer_gpu(monkeypatch):
    calls = {"embed": [], "optimize": [], "sync": 0}
    generated_counts = [3, 2, 0]

    def embed(molecules, parameters, *, confsPerMolecule, maxIterations):
        calls["embed"].append(
            (molecules, parameters.randomSeed, parameters.useRandomCoords, confsPerMolecule, maxIterations)
        )
        for molecule, count in zip(molecules, generated_counts):
            molecule.RemoveAllConformers()
            for conformer_id in range(min(count, confsPerMolecule)):
                conformer = Chem.Conformer(molecule.GetNumAtoms())
                conformer.SetId(conformer_id)
                molecule.AddConformer(conformer, assignId=True)

    def optimize(molecules, *, maxIters, output):
        calls["optimize"].append((molecules, maxIters, output))
        return _FakeOptimizationResult(
            molecules,
            energies=[3.0, 1.0, 2.0, 8.0, 7.0],
            converged=[1, 1, 0, 0, 1],
        )

    monkeypatch.setattr(chemistry_workflow, "_embed_molecules", embed)
    monkeypatch.setattr(chemistry_workflow, "_optimize_mmff94", optimize)
    monkeypatch.setattr(
        chemistry_workflow, "_coordinate_output_device", lambda: "DEVICE"
    )
    monkeypatch.setattr(
        chemistry_workflow,
        "_synchronize_cuda",
        lambda: calls.__setitem__("sync", calls["sync"] + 1),
    )
    return calls


def test_embedding_accounts_for_partial_and_zero_results_atomically(conformer_gpu):
    state = _clustered_state()
    result = embed_representative_conformers(
        state,
        representative_count=3,
        representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
        conformers_per_representative=3,
    )
    assert result.summary["entry_point"] == "EmbedMolecules"
    assert result.summary["selection_executor"] == "Python/RDKit"
    assert result.summary["selected_representative_count"] == 3
    assert result.summary["selection_shortfall"] == 0
    assert result.summary["generated_conformer_count"] == 5
    assert result.summary["partial_embedding_ids"] == ["mol-3"]
    assert result.summary["zero_embedding_ids"] == ["mol-5"]
    assert state.phase is WorkflowPhase.EMBEDDED
    assert len(state.representative_records) == 3
    assert len(state.conformer_molecules) == 2
    calls = conformer_gpu
    assert calls["embed"][0][1:] == (7, True, 3, -1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"representative_count": 2}, "representative count"),
        ({"representative_count": 7}, "representative count"),
        ({"conformers_per_representative": 2}, "conformers per representative"),
        ({"conformers_per_representative": 9}, "conformers per representative"),
    ],
)
def test_embedding_validates_bounds_before_execution(conformer_gpu, kwargs, message):
    state = _clustered_state()
    before = _state_snapshot(state)
    arguments = {
        "representative_count": 3,
        "representative_policy": RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
        "conformers_per_representative": 3,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        embed_representative_conformers(state, **arguments)
    assert conformer_gpu["embed"] == []
    assert _state_snapshot(state) == before


def _embedded_state(conformer_gpu) -> WorkflowState:
    state = _clustered_state()
    state.fingerprints = _FakeGpuResult(np.zeros((7, 32), dtype=np.int32))
    state.similarity = _FakeGpuResult(np.eye(7, dtype=float))
    embed_representative_conformers(
        state,
        representative_count=3,
        representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
        conformers_per_representative=3,
    )
    return state


def test_optimization_reconciles_pairs_selects_within_molecule_and_builds_figures(conformer_gpu):
    state = _embedded_state(conformer_gpu)
    result = optimize_conformers_mmff94(state)
    assert result.summary["entry_point"] == "MMFFOptimizeMoleculesConfs"
    assert result.summary["attempted_conformer_count"] == 5
    assert result.summary["converged_conformer_count"] == 3
    assert result.summary["unconverged_conformer_count"] == 2
    assert [record["selected_conformer_id"] for record in result.summary["selected_conformer_records"]] == [
        "mol-1:conf-1",
        "mol-3:conf-1",
    ]
    assert len(result.figures) == 2
    assert result.figures[0].axes[0].get_ylabel() == "MMFF94 energy (kcal/mol)"
    assert state.phase is WorkflowPhase.OPTIMIZED
    assert conformer_gpu["optimize"][0][1:] == (500, "DEVICE")


@pytest.mark.parametrize(
    ("pairs", "energies", "converged", "message"),
    [
        ([(0, 0)] * 5, None, None, "incomplete or duplicated"),
        (None, [1.0, 2.0, float("nan"), 4.0, 5.0], None, "non-finite"),
        (None, None, [1, 1, 1, 1], "same length"),
    ],
)
def test_optimization_rejects_bad_results_atomically(
    conformer_gpu, monkeypatch, pairs, energies, converged, message
):
    state = _embedded_state(conformer_gpu)
    before = _state_snapshot(state)

    def bad_optimize(molecules, *, maxIters, output):
        return _FakeOptimizationResult(
            molecules, pairs=pairs, energies=energies, converged=converged
        )

    monkeypatch.setattr(chemistry_workflow, "_optimize_mmff94", bad_optimize)
    with pytest.raises(RuntimeError, match=message):
        optimize_conformers_mmff94(state)
    assert _state_snapshot(state) == before


def test_workflow_report_has_exact_frozen_e01_e06_schemas(conformer_gpu):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    report = build_workflow_report(state)
    assert [record.key for record in report.evidence] == [
        "E01", "E02", "E03", "E04", "E05", "E06"
    ]
    assert [record.provenance for record in report.evidence] == [
        "RDKit input validation",
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    ]
    expected_keys = [
        {"raw_count", "valid_count", "invalid_count", "invalid_ids", "preview_count", "count_unit"},
        {"fingerprint_radius", "fingerprint_size_bits", "packed_shape", "molecule_count", "active_bits_min", "active_bits_median", "active_bits_max", "executor", "size_unit"},
        {"matrix_shape", "q1", "median", "q3", "p90", "max_off_diagonal", "most_similar_pair", "similarity_unit"},
        {"cutoff", "cluster_count", "singleton_count", "singleton_fraction", "largest_cluster_sizes", "assignment_count", "cutoff_unit"},
        {"requested_representative_count", "selected_representative_count", "selection_shortfall", "representative_policy", "representatives", "requested_conformers_per_representative", "generated_conformer_count", "partial_embedding_ids", "zero_embedding_ids", "count_unit"},
        {"attempted_conformer_count", "converged_conformer_count", "unconverged_conformer_count", "per_conformer_records", "selected_conformer_records", "energy_unit", "comparison_scope"},
    ]
    for record, keys in zip(report.evidence, expected_keys):
        payload = json.loads(record.payload_json)
        assert set(payload) == keys
        assert record.payload_json == json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    assert json.loads(report.evidence[0].payload_json)["count_unit"] == "rows"
    assert json.loads(report.evidence[1].payload_json)["size_unit"] == "bits"
    assert json.loads(report.evidence[2].payload_json)["similarity_unit"] == "Tanimoto coefficient"
    assert json.loads(report.evidence[3].payload_json)["cutoff_unit"] == "Tanimoto distance"
    assert json.loads(report.evidence[4].payload_json)["count_unit"] == "conformers"
    e06 = json.loads(report.evidence[5].payload_json)
    assert e06["energy_unit"] == "kcal/mol"
    assert e06["comparison_scope"] == "within molecule only"


@pytest.mark.parametrize(
    ("stage", "mutation", "message"),
    [
        ("generate_morgan_fingerprints", lambda summary: summary.__setitem__("entry_point", "wrong"), "provenance"),
        ("discover_fused_butina_clusters", lambda summary: summary.__setitem__("assignment_count", 6), "assignment"),
        ("embed_representative_conformers", lambda summary: summary["representatives"][0].__setitem__("cluster_id", 99), "representative provenance"),
        ("optimize_conformers_mmff94", lambda summary: summary.__setitem__("converged_conformer_count", 99), "convergence totals"),
    ],
)
def test_workflow_report_rejects_unreconciled_or_unknown_provenance(
    conformer_gpu, stage, mutation, message
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    mutation(state.summaries[stage])
    with pytest.raises(RuntimeError, match=message):
        build_workflow_report(state)


@pytest.mark.parametrize("bad_count", [3.5, "3", None])
def test_embedding_rejects_non_integer_representative_counts_atomically(
    conformer_gpu, bad_count
):
    state = _clustered_state()
    before = _state_snapshot(state)
    with pytest.raises(ValueError, match="representative count"):
        embed_representative_conformers(
            state,
            representative_count=bad_count,
            representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
            conformers_per_representative=3,
        )
    assert _state_snapshot(state) == before


@pytest.mark.parametrize("corruption", ["duplicate_cluster", "unknown_molecule"])
def test_selection_rejects_duplicate_or_unknown_eligibility_provenance_atomically(
    corruption
):
    state = _clustered_state()
    candidates = state.summaries["discover_fused_butina_clusters"][
        "representative_eligibility"
    ]["candidates_by_cluster"]
    if corruption == "duplicate_cluster":
        candidates[1]["cluster_id"] = candidates[0]["cluster_id"]
    else:
        candidates[0]["candidate_ids"][0] = "unknown"
    before = _state_snapshot(state)
    with pytest.raises(RuntimeError, match="duplicate or unknown|unknown representative"):
        select_representatives(
            state, 3, RepresentativePolicy.LARGEST_CLUSTERS_FIRST
        )
    assert _state_snapshot(state) == before


def test_embedding_rejects_all_zero_results_atomically(conformer_gpu, monkeypatch):
    state = _clustered_state()
    before = _state_snapshot(state)

    def zero_embed(molecules, parameters, *, confsPerMolecule, maxIterations):
        for molecule in molecules:
            molecule.RemoveAllConformers()

    monkeypatch.setattr(chemistry_workflow, "_embed_molecules", zero_embed)
    with pytest.raises(RuntimeError, match="zero conformers"):
        embed_representative_conformers(
            state,
            representative_count=3,
            representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
            conformers_per_representative=3,
        )
    assert _state_snapshot(state) == before


def test_optimization_uses_returned_pairs_as_authoritative_coordinate_mapping(
    conformer_gpu, monkeypatch
):
    state = _embedded_state(conformer_gpu)
    pairs = [(1, 1), (0, 2), (0, 0), (1, 0), (0, 1)]
    coordinates = [
        [
            np.full((state.conformer_molecules[0].GetNumAtoms(), 3), value)
            for value in (2.0, 0.0, 1.0)
        ],
        [
            np.full((state.conformer_molecules[1].GetNumAtoms(), 3), value)
            for value in (11.0, 10.0)
        ],
    ]

    def shuffled_optimize(molecules, *, maxIters, output):
        return _FakeOptimizationResult(
            molecules,
            pairs=pairs,
            energies=[5.0, 4.0, 3.0, 2.0, 1.0],
            converged=[1] * 5,
            coordinates=coordinates,
        )

    monkeypatch.setattr(chemistry_workflow, "_optimize_mmff94", shuffled_optimize)
    optimize_conformers_mmff94(state)
    for molecule_index, molecule in enumerate(state.conformer_molecules):
        for conformer_index in range(molecule.GetNumConformers()):
            point = molecule.GetConformer(conformer_index).GetAtomPosition(0)
            assert point.x == pytest.approx(10 * molecule_index + conformer_index)


def test_optimization_rejects_unreconciled_coordinate_totals_atomically(
    conformer_gpu, monkeypatch
):
    state = _embedded_state(conformer_gpu)
    before = _state_snapshot(state)

    def bad_coordinates(molecules, *, maxIters, output):
        result = _FakeOptimizationResult(molecules)
        result._coordinates[0] = result._coordinates[0][:-1]
        return result

    monkeypatch.setattr(chemistry_workflow, "_optimize_mmff94", bad_coordinates)
    with pytest.raises(RuntimeError, match="coordinate conformer totals"):
        optimize_conformers_mmff94(state)
    assert _state_snapshot(state) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda summary: summary.__setitem__("unexpected", 1), "unexpected keys"),
        (lambda summary: summary.__setitem__("median", float("inf")), "non-finite"),
    ],
)
def test_workflow_report_rejects_unexpected_keys_and_nonfinite_values(
    conformer_gpu, mutation, message
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    mutation(state.summaries["measure_tanimoto_similarity"])
    with pytest.raises(RuntimeError, match=message):
        build_workflow_report(state)


@pytest.mark.parametrize("artifact_name", ["fingerprints", "similarity"])
def test_workflow_report_requires_retained_fingerprint_and_similarity_artifacts(
    conformer_gpu, artifact_name
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    setattr(state, artifact_name, None)
    with pytest.raises(RuntimeError, match=f"{artifact_name[:-1] if artifact_name.endswith('s') else artifact_name} artifact"):
        build_workflow_report(state)


@pytest.mark.parametrize(
    ("artifact_field", "replacement", "message"),
    [
        ("energies", [99.0, 1.0, 2.0, 8.0, 7.0], "energy artifact"),
        ("converged", [0, 1, 0, 0, 1], "convergence artifact"),
    ],
)
def test_workflow_report_reconciles_optimization_artifact_values(
    conformer_gpu, artifact_field, replacement, message
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    setattr(state.optimization_result, artifact_field, _FakeGpuResult(replacement))
    with pytest.raises(RuntimeError, match=message):
        build_workflow_report(state)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonminimum", "forged_id"])
def test_workflow_report_rejects_invalid_selected_conformer_records(
    conformer_gpu, corruption
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    selected = state.summaries["optimize_conformers_mmff94"][
        "selected_conformer_records"
    ]
    per_records = state.summaries["optimize_conformers_mmff94"][
        "per_conformer_records"
    ]
    if corruption == "missing":
        selected.pop()
    elif corruption == "duplicate":
        selected.append(selected[0].copy())
    elif corruption == "nonminimum":
        replacement = next(
            record.copy()
            for record in per_records
            if record["molecule_id"] == selected[0]["molecule_id"]
            and record["converged"]
            and record["conformer_index"] != selected[0]["conformer_index"]
        )
        replacement["selected_conformer_id"] = (
            f"{replacement['molecule_id']}:conf-{replacement['conformer_index']}"
        )
        selected[0] = replacement
    else:
        selected[0]["selected_conformer_id"] = "forged"
    with pytest.raises(RuntimeError, match="selected conformer"):
        build_workflow_report(state)


def test_workflow_report_rejects_deleted_required_payload_key(conformer_gpu):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    del state.summaries["embed_representative_conformers"][
        "generated_conformer_count"
    ]
    with pytest.raises(RuntimeError, match="missing or unexpected keys"):
        build_workflow_report(state)


@pytest.mark.parametrize("field", ["molecule_id", "cluster_id"])
def test_workflow_report_rejects_forged_unconverged_conformer_provenance(
    conformer_gpu, field
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    unconverged = next(
        record
        for record in state.summaries["optimize_conformers_mmff94"][
            "per_conformer_records"
        ]
        if not record["converged"]
    )
    unconverged[field] = "forged" if field == "molecule_id" else 99
    with pytest.raises(RuntimeError, match="conformer provenance"):
        build_workflow_report(state)


def test_workflow_report_rejects_coordinated_forgery_of_selected_labels(
    conformer_gpu,
):
    state = _embedded_state(conformer_gpu)
    optimize_conformers_mmff94(state)
    optimization = state.summaries["optimize_conformers_mmff94"]
    selected = optimization["selected_conformer_records"][0]
    underlying = next(
        record
        for record in optimization["per_conformer_records"]
        if record["optimization_molecule_index"]
        == selected["optimization_molecule_index"]
        and record["conformer_index"] == selected["conformer_index"]
    )
    for record in (underlying, selected):
        record["molecule_id"] = "forged"
        record["cluster_id"] = 99
    selected["selected_conformer_id"] = f"forged:conf-{selected['conformer_index']}"
    with pytest.raises(RuntimeError, match="conformer provenance"):
        build_workflow_report(state)
