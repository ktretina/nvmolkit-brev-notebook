import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from chemistry_workflow import (
    StageResult,
    WorkflowPhase,
    WorkflowState,
    eligible_stage,
    inspect_library,
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
def test_eligible_stage_maps_every_phase_exactly(
    phase: WorkflowPhase, stage: str
):
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
    sample = _write_csv(
        tmp_path / "sample.csv", [{"id": "valid-1", "smiles": "CCO"}]
    )
    state = WorkflowState()

    result = inspect_library(state, sample, expected_rows=1)

    assert json.loads(json.dumps(result.summary)) == result.summary
    assert all(value not in state.molecules for value in result.summary.values())


def test_inspection_rejects_wrong_row_count_without_mutating_state(
    tmp_path: Path,
):
    sample = _write_csv(
        tmp_path / "sample.csv", [{"id": "one", "smiles": "CCO"}]
    )
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
    sample = _write_csv(
        tmp_path / "sample.csv", [{"id": "one", "smiles": "CCO"}]
    )
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
