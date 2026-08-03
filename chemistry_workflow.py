from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


class WorkflowPhase(StrEnum):
    NEW = "new"
    INSPECTED = "inspected"
    FINGERPRINTED = "fingerprinted"
    COMPARED = "compared"
    CLUSTERED = "clustered"
    EMBEDDED = "embedded"
    OPTIMIZED = "optimized"


@dataclass(frozen=True)
class StageResult:
    stage: str
    display_label: str
    summary: dict[str, Any]
    figures: tuple[Any, ...] = ()


@dataclass
class WorkflowState:
    phase: WorkflowPhase = WorkflowPhase.NEW
    records: list[dict[str, Any]] = field(default_factory=list)
    molecules: list[Any] = field(default_factory=list)
    fingerprints: Any = None
    similarity: Any = None
    clusters: list[list[int]] = field(default_factory=list)
    representative_records: list[dict[str, Any]] = field(default_factory=list)
    conformer_molecules: list[Any] = field(default_factory=list)
    optimization_result: Any = None
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)


_NEXT_STAGE = {
    WorkflowPhase.NEW: "inspect_library",
    WorkflowPhase.INSPECTED: "generate_morgan_fingerprints",
    WorkflowPhase.FINGERPRINTED: "measure_tanimoto_similarity",
    WorkflowPhase.COMPARED: "discover_fused_butina_clusters",
    WorkflowPhase.CLUSTERED: "embed_representative_conformers",
    WorkflowPhase.EMBEDDED: "optimize_conformers_mmff94",
    WorkflowPhase.OPTIMIZED: "submit_synthesis",
}


def eligible_stage(state: WorkflowState) -> str:
    return _NEXT_STAGE[state.phase]


def inspect_library(
    state: WorkflowState, data_path: Path, expected_rows: int = 256
) -> StageResult:
    if state.phase is not WorkflowPhase.NEW:
        raise RuntimeError("inspect_library requires a state in the NEW phase")

    raw_records = pd.read_csv(data_path)
    if "molecule_id" in raw_records.columns:
        identifier_column = "molecule_id"
    elif "id" in raw_records.columns:
        identifier_column = "id"
    else:
        identifier_column = None
    if identifier_column is None or "smiles" not in raw_records.columns:
        raise ValueError("input library requires id and smiles columns")
    if len(raw_records) != expected_rows:
        raise ValueError(
            f"input library expected {expected_rows} rows; found {len(raw_records)}"
        )

    records: list[dict[str, Any]] = []
    molecules: list[Any] = []
    invalid_ids: list[str] = []
    for source_row, row in raw_records.iterrows():
        molecule_id = str(row[identifier_column])
        smiles = str(row["smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid_ids.append(molecule_id)
            continue
        records.append(
            {"id": molecule_id, "smiles": smiles, "source_row": int(source_row)}
        )
        molecules.append(molecule)

    if not molecules:
        raise ValueError("input library produced zero valid molecules")

    summary: dict[str, Any] = {
        "raw_count": int(len(raw_records)),
        "valid_count": int(len(molecules)),
        "invalid_count": int(len(invalid_ids)),
        "invalid_ids": invalid_ids,
        "preview_count": int(min(len(molecules), 24)),
        "executor": "RDKit input validation",
    }
    summaries = dict(state.summaries)
    summaries["inspect_library"] = summary

    state.records = records
    state.molecules = molecules
    state.summaries = summaries
    state.phase = WorkflowPhase.INSPECTED

    return StageResult(
        stage="inspect_library",
        display_label="RDKit input validation",
        summary=summary,
    )
