from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
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


def _morgan_generator_class():
    from nvmolkit.fingerprints import MorganFingerprintGenerator

    return MorganFingerprintGenerator


def _cross_tanimoto_similarity(fingerprints):
    from nvmolkit.similarity import crossTanimotoSimilarity

    return crossTanimotoSimilarity(fingerprints)


def _fused_butina(fingerprints, *, cutoff: float):
    from nvmolkit.clustering import fused_butina

    return fused_butina(fingerprints, cutoff=cutoff)


def _synchronize_cuda() -> None:
    import torch

    torch.cuda.synchronize()


def _host_array(tensor: Any) -> np.ndarray:
    host_value = tensor.cpu() if hasattr(tensor, "cpu") else tensor
    if hasattr(host_value, "numpy"):
        return np.asarray(host_value.numpy())
    return np.asarray(host_value)


def _fingerprint_density_figure(active_bits: np.ndarray):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(5.5, 3.25), layout="constrained")
    axes = figure.subplots()
    axes.hist(active_bits, bins=min(20, max(1, len(active_bits))))
    axes.set_xlabel("Active Morgan fingerprint bits per molecule")
    axes.set_ylabel("Molecule count")
    axes.set_title("Morgan fingerprint density")
    return figure


def _similarity_heatmap_figure(similarity_matrix: np.ndarray):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4.5, 4.0), layout="constrained")
    axes = figure.subplots()
    image = axes.imshow(similarity_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axes.set_xlabel("Molecule index")
    axes.set_ylabel("Molecule index")
    axes.set_title("All-pairs Tanimoto similarity")
    figure.colorbar(image, ax=axes, label="Tanimoto similarity")
    return figure


def _cluster_size_figure(cluster_sizes: list[int]):
    from matplotlib.figure import Figure

    largest = sorted(cluster_sizes, reverse=True)[:15]
    figure = Figure(figsize=(5.5, 3.25), layout="constrained")
    axes = figure.subplots()
    axes.bar(range(1, len(largest) + 1), largest)
    axes.set_xlabel("Cluster rank by descending size")
    axes.set_ylabel("Molecule count")
    axes.set_title("Largest fused Butina clusters")
    return figure


def generate_morgan_fingerprints(
    state: WorkflowState,
    *,
    fingerprint_radius: int,
    fingerprint_size: int,
) -> StageResult:
    """Run nvMolKit Morgan fingerprints after strict parameter validation."""
    if state.phase is not WorkflowPhase.INSPECTED:
        raise RuntimeError(
            "generate_morgan_fingerprints requires a state in the INSPECTED phase"
        )
    if isinstance(fingerprint_radius, bool) or fingerprint_radius not in (2, 3):
        raise ValueError("fingerprint radius must be 2 or 3")
    if isinstance(fingerprint_size, bool) or fingerprint_size not in (1024, 2048):
        raise ValueError("fingerprint size must be 1024 or 2048")

    generator = _morgan_generator_class()(
        radius=fingerprint_radius, fpSize=fingerprint_size
    )
    fingerprints = generator.GetFingerprints(state.molecules)
    fingerprint_tensor = fingerprints.torch()
    expected_shape = (len(state.molecules), fingerprint_size // 32)
    if tuple(fingerprint_tensor.shape) != expected_shape:
        raise RuntimeError(
            "The packed Morgan fingerprint shape did not match the molecule count and size."
        )

    _synchronize_cuda()
    packed = _host_array(fingerprint_tensor)
    if packed.shape != expected_shape:
        raise RuntimeError(
            "The packed Morgan fingerprint shape changed during host transfer."
        )
    packed_unsigned = (
        packed.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)
    ).astype(np.uint32)
    active_bits = np.unpackbits(packed_unsigned.view(np.uint8), axis=1).sum(
        axis=1, dtype=np.int64
    )

    summary: dict[str, Any] = {
        "entry_point": "MorganFingerprintGenerator",
        "fingerprint_radius": int(fingerprint_radius),
        "fingerprint_size": int(fingerprint_size),
        "molecule_count": int(len(state.molecules)),
        "packed_shape": [int(value) for value in expected_shape],
        "active_bits_min": int(active_bits.min()),
        "active_bits_median": float(np.median(active_bits)),
        "active_bits_max": int(active_bits.max()),
        "cuda_device": str(fingerprint_tensor.device),
    }
    figure = _fingerprint_density_figure(active_bits)
    summaries = dict(state.summaries)
    summaries["generate_morgan_fingerprints"] = summary

    state.fingerprints = fingerprints
    state.summaries = summaries
    state.phase = WorkflowPhase.FINGERPRINTED
    return StageResult(
        stage="generate_morgan_fingerprints",
        display_label="nvMolKit MorganFingerprintGenerator",
        summary=summary,
        figures=(figure,),
    )


def measure_tanimoto_similarity(state: WorkflowState) -> StageResult:
    """Run nvMolKit all-pairs Tanimoto and summarize the off-diagonal matrix."""
    if state.phase is not WorkflowPhase.FINGERPRINTED:
        raise RuntimeError(
            "measure_tanimoto_similarity requires a state in the FINGERPRINTED phase"
        )

    molecule_count = len(state.molecules)
    if molecule_count < 2:
        raise RuntimeError("Tanimoto comparison requires at least two molecules")
    similarity = _cross_tanimoto_similarity(state.fingerprints)
    _synchronize_cuda()
    similarity_matrix = _host_array(similarity.torch())
    expected_shape = (molecule_count, molecule_count)
    if similarity_matrix.shape != expected_shape:
        raise RuntimeError("The all-pairs Tanimoto matrix shape was unexpected.")
    if not np.isfinite(similarity_matrix).all():
        raise RuntimeError("The Tanimoto matrix contains non-finite values.")
    if not np.allclose(similarity_matrix, similarity_matrix.T, rtol=0.0, atol=1e-7):
        raise RuntimeError("The Tanimoto matrix is not symmetric.")
    if np.any((similarity_matrix < 0.0) | (similarity_matrix > 1.0)):
        raise RuntimeError("Tanimoto values must remain in the range 0 through 1.")
    if not np.allclose(
        np.diag(similarity_matrix), np.ones(molecule_count), rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("The Tanimoto matrix diagonal must be approximately 1.")

    upper_rows, upper_columns = np.triu_indices(molecule_count, k=1)
    off_diagonal = similarity_matrix[upper_rows, upper_columns]
    pair_position = int(np.argmax(off_diagonal))
    first_index = int(upper_rows[pair_position])
    second_index = int(upper_columns[pair_position])
    pair_similarity = float(off_diagonal[pair_position])
    summary: dict[str, Any] = {
        "entry_point": "crossTanimotoSimilarity",
        "matrix_shape": [int(value) for value in similarity_matrix.shape],
        "q1": float(np.quantile(off_diagonal, 0.25)),
        "median": float(np.median(off_diagonal)),
        "q3": float(np.quantile(off_diagonal, 0.75)),
        "p90": float(np.quantile(off_diagonal, 0.90)),
        "max": pair_similarity,
        "most_similar_nonidentical_pair": {
            "molecule_ids": [
                str(state.records[first_index]["id"]),
                str(state.records[second_index]["id"]),
            ],
            "source_rows": [
                int(state.records[first_index]["source_row"]),
                int(state.records[second_index]["source_row"]),
            ],
            "similarity": pair_similarity,
        },
    }
    figure = _similarity_heatmap_figure(similarity_matrix)
    summaries = dict(state.summaries)
    summaries["measure_tanimoto_similarity"] = summary

    state.similarity = similarity
    state.summaries = summaries
    state.phase = WorkflowPhase.COMPARED
    return StageResult(
        stage="measure_tanimoto_similarity",
        display_label="nvMolKit crossTanimotoSimilarity",
        summary=summary,
        figures=(figure,),
    )


def discover_fused_butina_clusters(
    state: WorkflowState,
    *,
    cluster_cutoff: float,
) -> StageResult:
    """Run nvMolKit fused Butina and validate one assignment per molecule."""
    if state.phase is not WorkflowPhase.COMPARED:
        raise RuntimeError(
            "discover_fused_butina_clusters requires a state in the COMPARED phase"
        )
    if (
        isinstance(cluster_cutoff, bool)
        or not isinstance(cluster_cutoff, (int, float))
        or not 0.40 <= float(cluster_cutoff) <= 0.60
    ):
        raise ValueError("cluster cutoff must be 0.40 through 0.60 inclusive")

    cutoff = float(cluster_cutoff)
    cluster_result = _fused_butina(state.fingerprints.torch(), cutoff=cutoff)
    _synchronize_cuda()
    clusters_raw = cluster_result[0]
    clusters = [
        [int(molecule_index) for molecule_index in cluster] for cluster in clusters_raw
    ]
    molecule_count = len(state.molecules)
    assigned_indices = [index for cluster in clusters for index in cluster]
    if len(assigned_indices) != molecule_count or sorted(assigned_indices) != list(
        range(molecule_count)
    ):
        raise RuntimeError("Every molecule must be assigned exactly once.")

    from rdkit.Chem import AllChem

    candidates_by_cluster: list[dict[str, Any]] = []
    eligible_singleton_count = 0
    for cluster_id, cluster in enumerate(clusters):
        eligible_indices = [
            molecule_index
            for molecule_index in cluster
            if bool(
                AllChem.MMFFHasAllMoleculeParams(
                    Chem.AddHs(Chem.Mol(state.molecules[molecule_index]))
                )
            )
        ]
        if not eligible_indices:
            continue
        is_singleton = len(cluster) == 1
        eligible_singleton_count += int(is_singleton)
        candidates_by_cluster.append(
            {
                "cluster_id": int(cluster_id),
                "candidate_ids": [
                    str(state.records[index]["id"]) for index in eligible_indices
                ],
                "source_rows": [
                    int(state.records[index]["source_row"])
                    for index in eligible_indices
                ],
                "is_singleton": bool(is_singleton),
            }
        )

    cluster_sizes = [len(cluster) for cluster in clusters]
    singleton_count = sum(size == 1 for size in cluster_sizes)
    eligible_cluster_count = len(candidates_by_cluster)
    summary: dict[str, Any] = {
        "entry_point": "fused_butina",
        "cluster_cutoff": cutoff,
        "molecule_count": int(molecule_count),
        "cluster_count": int(len(clusters)),
        "singleton_count": int(singleton_count),
        "singleton_fraction": float(singleton_count / molecule_count),
        "largest_cluster_sizes": [
            int(size) for size in sorted(cluster_sizes, reverse=True)[:15]
        ],
        "assignment_count": int(len(assigned_indices)),
        "representative_eligibility": {
            "eligible_cluster_count": int(eligible_cluster_count),
            "eligible_singleton_count": int(eligible_singleton_count),
            "maximum_representative_count": int(eligible_cluster_count),
            "candidates_by_cluster": candidates_by_cluster,
        },
    }
    figure = _cluster_size_figure(cluster_sizes)
    summaries = dict(state.summaries)
    summaries["discover_fused_butina_clusters"] = summary

    state.clusters = clusters
    state.summaries = summaries
    state.phase = WorkflowPhase.CLUSTERED
    return StageResult(
        stage="discover_fused_butina_clusters",
        display_label="nvMolKit fused_butina with RDKit MMFF94 eligibility",
        summary=summary,
        figures=(figure,),
    )
