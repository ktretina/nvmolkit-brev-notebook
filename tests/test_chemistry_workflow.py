import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from rdkit import Chem

import chemistry_workflow

from chemistry_workflow import (
    StageResult,
    WorkflowPhase,
    WorkflowState,
    discover_fused_butina_clusters,
    eligible_stage,
    generate_morgan_fingerprints,
    inspect_library,
    measure_tanimoto_similarity,
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
