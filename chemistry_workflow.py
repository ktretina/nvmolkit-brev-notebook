from __future__ import annotations

import json
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


class RepresentativePolicy(StrEnum):
    LARGEST_CLUSTERS_FIRST = "largest_clusters_first"
    INCLUDE_SINGLETON_IF_AVAILABLE = "include_singleton_if_available"


@dataclass(frozen=True)
class StageResult:
    stage: str
    display_label: str
    summary: dict[str, Any]
    figures: tuple[Any, ...] = ()


@dataclass(frozen=True)
class EvidenceRecord:
    key: str
    label: str
    payload_json: str
    provenance: str


@dataclass(frozen=True)
class WorkflowReport:
    evidence: tuple[EvidenceRecord, ...]


@dataclass
class WorkflowState:
    phase: WorkflowPhase = WorkflowPhase.NEW
    records: list[dict[str, Any]] = field(default_factory=list)
    molecules: list[Any] = field(default_factory=list)
    fingerprints: Any = None
    fingerprint_parameters: tuple[int, int] | None = None
    similarity: Any = None
    clusters: list[list[int]] = field(default_factory=list)
    cluster_cutoff: float | None = None
    representative_records: list[dict[str, Any]] = field(default_factory=list)
    conformer_molecules: list[Any] = field(default_factory=list)
    optimization_result: Any = None
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    invalid_ids: tuple[str, ...] = ()
    embedding_parameters: tuple[int, str, int] | None = None


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
    state.invalid_ids = tuple(invalid_ids)
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


def _embed_molecules(molecules, parameters, *, confsPerMolecule: int, maxIterations: int):
    from nvmolkit.embedMolecules import EmbedMolecules

    return EmbedMolecules(
        molecules,
        parameters,
        confsPerMolecule=confsPerMolecule,
        maxIterations=maxIterations,
    )


def _coordinate_output_device():
    from nvmolkit.types import CoordinateOutput

    return CoordinateOutput.DEVICE


def _optimize_mmff94(molecules, *, maxIters: int, output):
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

    return MMFFOptimizeMoleculesConfs(molecules, maxIters=maxIters, output=output)


def _synchronize_cuda() -> None:
    import torch

    torch.cuda.synchronize()


def _host_array(tensor: Any) -> np.ndarray:
    host_value = tensor.cpu() if hasattr(tensor, "cpu") else tensor
    if hasattr(host_value, "numpy"):
        return np.asarray(host_value.numpy())
    return np.asarray(host_value)


def _fingerprint_artifact_metrics(fingerprints: Any) -> dict[str, Any]:
    fingerprint_tensor = fingerprints.torch()
    packed = _host_array(fingerprint_tensor)
    if packed.ndim != 2:
        raise RuntimeError("The packed Morgan fingerprint shape was unexpected.")
    packed_unsigned = (
        packed.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)
    ).astype(np.uint32)
    active_bits = np.unpackbits(packed_unsigned.view(np.uint8), axis=1).sum(
        axis=1, dtype=np.int64
    )
    return {
        "packed_shape": [int(value) for value in packed.shape],
        "molecule_count": int(packed.shape[0]),
        "active_bits_min": int(active_bits.min()),
        "active_bits_median": float(np.median(active_bits)),
        "active_bits_max": int(active_bits.max()),
        "cuda_device": str(fingerprint_tensor.device),
        "active_bits": active_bits,
    }


def _similarity_artifact_metrics(
    similarity_matrix: np.ndarray, records: list[dict[str, Any]]
) -> dict[str, Any]:
    molecule_count = len(records)
    upper_rows, upper_columns = np.triu_indices(molecule_count, k=1)
    off_diagonal = similarity_matrix[upper_rows, upper_columns]
    pair_position = int(np.argmax(off_diagonal))
    first_index = int(upper_rows[pair_position])
    second_index = int(upper_columns[pair_position])
    pair_similarity = float(off_diagonal[pair_position])
    return {
        "matrix_shape": [int(value) for value in similarity_matrix.shape],
        "q1": float(np.quantile(off_diagonal, 0.25)),
        "median": float(np.median(off_diagonal)),
        "q3": float(np.quantile(off_diagonal, 0.75)),
        "p90": float(np.quantile(off_diagonal, 0.90)),
        "max": pair_similarity,
        "most_similar_nonidentical_pair": {
            "molecule_ids": [
                str(records[first_index]["id"]),
                str(records[second_index]["id"]),
            ],
            "source_rows": [
                int(records[first_index]["source_row"]),
                int(records[second_index]["source_row"]),
            ],
            "similarity": pair_similarity,
        },
    }


def _cluster_artifact_metrics(
    clusters: list[list[int]], molecule_count: int, cutoff: float
) -> dict[str, Any]:
    cluster_sizes = [len(cluster) for cluster in clusters]
    singleton_count = sum(size == 1 for size in cluster_sizes)
    return {
        "cluster_cutoff": float(cutoff),
        "molecule_count": int(molecule_count),
        "cluster_count": int(len(clusters)),
        "singleton_count": int(singleton_count),
        "singleton_fraction": float(singleton_count / molecule_count),
        "largest_cluster_sizes": [
            int(size) for size in sorted(cluster_sizes, reverse=True)[:15]
        ],
        "assignment_count": int(sum(cluster_sizes)),
    }


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


def _eligibility_groups(
    records: list[dict[str, Any]], molecules: list[Any], clusters: list[list[int]]
) -> list[dict[str, Any]]:
    """Derive MMFF94-eligible cluster members from retained molecule artifacts."""
    from rdkit.Chem import AllChem

    groups: list[dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters):
        members = []
        for molecule_index in cluster:
            molecule = Chem.AddHs(Chem.Mol(molecules[molecule_index]))
            if AllChem.MMFFHasAllMoleculeParams(molecule):
                record = records[molecule_index]
                members.append({
                    "molecule_id": str(record["id"]),
                    "source_row": int(record["source_row"]),
                    "cluster_id": int(cluster_id),
                    "molecule_index": int(molecule_index),
                })
        if members:
            members.sort(key=lambda record: record["source_row"])
            groups.append({
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster),
                "minimum_source_row": min(
                    int(records[index]["source_row"]) for index in cluster
                ),
                "is_singleton": len(cluster) == 1,
                "members": members,
            })
    groups.sort(key=lambda group: (-group["cluster_size"], group["minimum_source_row"]))
    return groups


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
    expected_shape = (len(state.molecules), fingerprint_size // 32)
    if tuple(fingerprints.torch().shape) != expected_shape:
        raise RuntimeError(
            "The packed Morgan fingerprint shape did not match the molecule count and size."
        )

    _synchronize_cuda()
    metrics = _fingerprint_artifact_metrics(fingerprints)
    if tuple(metrics["packed_shape"]) != expected_shape:
        raise RuntimeError(
            "The packed Morgan fingerprint shape changed during host transfer."
        )

    summary: dict[str, Any] = {
        "entry_point": "MorganFingerprintGenerator",
        "fingerprint_radius": int(fingerprint_radius),
        "fingerprint_size": int(fingerprint_size),
        **{key: metrics[key] for key in (
            "molecule_count",
            "packed_shape",
            "active_bits_min",
            "active_bits_median",
            "active_bits_max",
            "cuda_device",
        )},
    }
    figure = _fingerprint_density_figure(metrics["active_bits"])
    summaries = dict(state.summaries)
    summaries["generate_morgan_fingerprints"] = summary

    state.fingerprints = fingerprints
    state.fingerprint_parameters = (int(fingerprint_radius), int(fingerprint_size))
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

    summary: dict[str, Any] = {
        "entry_point": "crossTanimotoSimilarity",
        **_similarity_artifact_metrics(similarity_matrix, state.records),
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

    cluster_metrics = _cluster_artifact_metrics(clusters, molecule_count, cutoff)
    groups = _eligibility_groups(state.records, state.molecules, clusters)
    summary: dict[str, Any] = {
        "entry_point": "fused_butina",
        **cluster_metrics,
        "representative_eligibility": {
            "eligible_cluster_count": len(groups),
            "eligible_singleton_count": sum(group["is_singleton"] for group in groups),
            "maximum_representative_count": len(groups),
            "candidates_by_cluster": [
                {
                    "cluster_id": group["cluster_id"],
                    "candidate_ids": [member["molecule_id"] for member in group["members"]],
                    "source_rows": [member["source_row"] for member in group["members"]],
                    "is_singleton": group["is_singleton"],
                }
                for group in sorted(groups, key=lambda group: group["cluster_id"])
            ],
        },
    }
    figure = _cluster_size_figure([len(cluster) for cluster in clusters])
    summaries = dict(state.summaries)
    summaries["discover_fused_butina_clusters"] = summary

    state.clusters = clusters
    state.cluster_cutoff = cutoff
    state.summaries = summaries
    state.phase = WorkflowPhase.CLUSTERED
    return StageResult(
        stage="discover_fused_butina_clusters",
        display_label="nvMolKit fused_butina with RDKit MMFF94 eligibility",
        summary=summary,
        figures=(figure,),
    )


def select_representatives(
    state: WorkflowState,
    representative_count: int,
    representative_policy: RepresentativePolicy,
) -> tuple[list[dict[str, Any]], int]:
    """Select one MMFF94-eligible member per cluster."""
    if state.phase is not WorkflowPhase.CLUSTERED:
        raise RuntimeError("select_representatives requires a state in the CLUSTERED phase")
    if type(representative_count) is not int or not 3 <= representative_count <= 6:
        raise ValueError("representative count must be 3 through 6 inclusive")
    try:
        policy = RepresentativePolicy(representative_policy)
    except (ValueError, TypeError) as error:
        raise ValueError("representative policy is invalid") from error
    groups = _eligibility_groups(state.records, state.molecules, state.clusters)
    if len(groups) < 3:
        raise RuntimeError("at least 3 eligible distinct clusters are required")

    selected_groups: list[dict[str, Any]] = []
    if policy is RepresentativePolicy.INCLUDE_SINGLETON_IF_AVAILABLE:
        reserved = next((group for group in groups if group["is_singleton"]), None)
        fill_limit = representative_count - int(reserved is not None)
        selected_groups.extend(
            group for group in groups if group is not reserved
        )
        selected_groups = selected_groups[: max(0, fill_limit)]
        if reserved is not None and len(selected_groups) < representative_count:
            selected_groups.append(reserved)
    else:
        selected_groups = groups[:representative_count]
    selected = [dict(group["members"][0]) for group in selected_groups]
    return selected, representative_count - len(selected)


def _embedding_count_figure(records: list[dict[str, Any]], requested: int):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(5.5, 3.25), layout="constrained")
    axes = figure.subplots()
    axes.bar(
        [record["molecule_id"] for record in records],
        [record["generated_conformer_count"] for record in records],
    )
    axes.axhline(requested, color="black", linestyle="--", linewidth=1)
    axes.set_ylabel("Generated conformers")
    axes.set_title("nvMolKit conformer embedding")
    return figure


def embed_representative_conformers(
    state: WorkflowState,
    representative_count: int,
    representative_policy: RepresentativePolicy,
    conformers_per_representative: int,
) -> StageResult:
    """Select with Python/RDKit, then run nvMolKit EmbedMolecules."""
    if state.phase is not WorkflowPhase.CLUSTERED:
        raise RuntimeError(
            "embed_representative_conformers requires a state in the CLUSTERED phase"
        )
    if type(representative_count) is not int or not 3 <= representative_count <= 6:
        raise ValueError("representative count must be 3 through 6 inclusive")
    if (
        type(conformers_per_representative) is not int
        or not 3 <= conformers_per_representative <= 8
    ):
        raise ValueError("conformers per representative must be 3 through 8 inclusive")
    try:
        policy = RepresentativePolicy(representative_policy)
    except (ValueError, TypeError) as error:
        raise ValueError("representative policy is invalid") from error

    selected, selection_shortfall = select_representatives(
        state, representative_count, policy
    )
    from rdkit.Chem import AllChem

    molecules = [
        Chem.AddHs(Chem.Mol(state.molecules[record["molecule_index"]]))
        for record in selected
    ]
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 7
    parameters.useRandomCoords = True
    _embed_molecules(
        molecules,
        parameters,
        confsPerMolecule=conformers_per_representative,
        maxIterations=-1,
    )
    _synchronize_cuda()

    representative_records: list[dict[str, Any]] = []
    conformer_molecules: list[Any] = []
    partial_ids: list[str] = []
    zero_ids: list[str] = []
    for selected_record, molecule in zip(selected, molecules):
        generated_count = int(molecule.GetNumConformers())
        if generated_count < 0 or generated_count > conformers_per_representative:
            raise RuntimeError("EmbedMolecules returned an invalid conformer count.")
        record = {
            **selected_record,
            "generated_conformer_count": generated_count,
        }
        representative_records.append(record)
        if generated_count == 0:
            zero_ids.append(record["molecule_id"])
        else:
            conformer_molecules.append(molecule)
            if generated_count < conformers_per_representative:
                partial_ids.append(record["molecule_id"])
    if not conformer_molecules:
        raise RuntimeError("All selected representatives produced zero conformers.")

    summary: dict[str, Any] = {
        "entry_point": "EmbedMolecules",
        "selection_executor": "Python/RDKit",
        "requested_representative_count": int(representative_count),
        "selected_representative_count": int(len(selected)),
        "selection_shortfall": int(selection_shortfall),
        "representative_policy": policy.value,
        "representatives": [
            {
                "molecule_id": record["molecule_id"],
                "source_row": record["source_row"],
                "cluster_id": record["cluster_id"],
            }
            for record in representative_records
        ],
        "requested_conformers_per_representative": int(
            conformers_per_representative
        ),
        "generated_conformer_count": int(
            sum(record["generated_conformer_count"] for record in representative_records)
        ),
        "partial_embedding_ids": partial_ids,
        "zero_embedding_ids": zero_ids,
    }
    json.dumps(summary, allow_nan=False)
    figure = _embedding_count_figure(representative_records, conformers_per_representative)
    summaries = dict(state.summaries)
    summaries["embed_representative_conformers"] = summary
    state.representative_records = representative_records
    state.embedding_parameters = (
        int(representative_count),
        policy.value,
        int(conformers_per_representative),
    )
    state.conformer_molecules = conformer_molecules
    state.summaries = summaries
    state.phase = WorkflowPhase.EMBEDDED
    return StageResult(
        stage="embed_representative_conformers",
        display_label="nvMolKit EmbedMolecules",
        summary=summary,
        figures=(figure,),
    )


def _conformer_energy_figure(records: list[dict[str, Any]]):
    from matplotlib.figure import Figure

    molecule_ids = list(dict.fromkeys(record["molecule_id"] for record in records))
    figure = Figure(figsize=(7.0, 3.8), layout="constrained")
    axes = figure.subplots()
    for record in records:
        axes.scatter(
            molecule_ids.index(record["molecule_id"]),
            record["energy_kcal_mol"],
            marker="o" if record["converged"] else "x",
            color="#76B900" if record["converged"] else "#d62728",
        )
    axes.set_xticks(range(len(molecule_ids)), molecule_ids, rotation=20)
    axes.set_xlabel("Representative molecule (within-molecule comparison only)")
    axes.set_ylabel("MMFF94 energy (kcal/mol)")
    axes.set_title("Conformer convergence and energy")
    return figure


def _optimized_structure_figure(
    molecules: list[Any], selected_records: list[dict[str, Any]]
):
    from matplotlib.figure import Figure

    panel_count = max(1, min(6, len(selected_records)))
    figure = Figure(figsize=(5.2 * panel_count, 4.2), layout="constrained")
    axes_value = figure.subplots(
        1, panel_count, subplot_kw={"projection": "3d"}, squeeze=False
    )
    axes_list = list(axes_value.flat)
    if not selected_records:
        axes_list[0].text2D(0.1, 0.5, "No converged conformer")
        axes_list[0].set_axis_off()
        return figure
    colors = {1: "#d9d9d9", 6: "#4d4d4d", 7: "#377eb8", 8: "#e41a1c"}
    for axes, record in zip(axes_list, selected_records[:6]):
        molecule = molecules[record["optimization_molecule_index"]]
        conformer = molecule.GetConformer(record["conformer_index"])
        coordinates = np.array(
            [list(conformer.GetAtomPosition(index)) for index in range(molecule.GetNumAtoms())]
        )
        axes.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=[colors.get(atom.GetAtomicNum(), "#984ea3") for atom in molecule.GetAtoms()],
        )
        for bond in molecule.GetBonds():
            indices = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            axes.plot(*[coordinates[indices, dimension] for dimension in range(3)], color="#777777")
        axes.set_title(record["selected_conformer_id"])
        axes.set_axis_off()
    figure.suptitle("Lowest-energy converged sampled conformers")
    return figure


def _optimization_records(
    result: Any, molecules: list[Any], representatives: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    arrays = [
        _host_array(getattr(result, name).torch()).reshape(-1)
        for name in ("energies", "converged", "mol_indices", "conf_indices")
    ]
    energies, convergence, molecule_indices, conformer_indices = arrays
    if len({len(array) for array in arrays}) != 1:
        raise RuntimeError("MMFF94 result buffers must have the same length.")
    if not np.isfinite(energies).all():
        raise RuntimeError("MMFF94 energies contain non-finite values.")
    convergence_values = [int(value) for value in convergence.tolist()]
    if set(convergence_values) - {0, 1}:
        raise RuntimeError("MMFF94 convergence flags must be binary.")
    expected_pairs = {
        (molecule_index, conformer_index)
        for molecule_index, molecule in enumerate(molecules)
        for conformer_index in range(molecule.GetNumConformers())
    }
    pairs = [
        (int(molecule_index), int(conformer_index))
        for molecule_index, conformer_index in zip(
            molecule_indices.tolist(), conformer_indices.tolist()
        )
    ]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise RuntimeError("MMFF94 molecule/conformer indices are incomplete or duplicated.")
    return [
        {
            "molecule_id": str(representatives[molecule_index]["molecule_id"]),
            "cluster_id": int(representatives[molecule_index]["cluster_id"]),
            "conformer_index": conformer_index,
            "energy_kcal_mol": float(energy),
            "converged": bool(did_converge),
            "optimization_molecule_index": molecule_index,
        }
        for energy, did_converge, (molecule_index, conformer_index) in zip(
            energies.tolist(), convergence_values, pairs
        )
    ]


def optimize_conformers_mmff94(state: WorkflowState) -> StageResult:
    """Run nvMolKit MMFF94 and reconcile every molecule/conformer result pair."""
    if state.phase is not WorkflowPhase.EMBEDDED:
        raise RuntimeError(
            "optimize_conformers_mmff94 requires a state in the EMBEDDED phase"
        )
    molecules = [Chem.Mol(molecule) for molecule in state.conformer_molecules]
    successful_records = [
        record
        for record in state.representative_records
        if record["generated_conformer_count"] > 0
    ]
    if len(molecules) != len(successful_records) or not molecules:
        raise RuntimeError("Embedded molecule provenance is unreconciled.")

    result = _optimize_mmff94(
        molecules, maxIters=500, output=_coordinate_output_device()
    )
    _synchronize_cuda()
    per_conformer_records = _optimization_records(result, molecules, successful_records)
    result_pairs = [
        (record["optimization_molecule_index"], record["conformer_index"])
        for record in per_conformer_records
    ]

    per_molecule = result.per_molecule()
    if len(per_molecule) != len(molecules):
        raise RuntimeError("MMFF94 coordinate molecule totals are unreconciled.")
    from rdkit.Geometry import Point3D

    for molecule, coordinates_for_molecule in zip(molecules, per_molecule):
        if len(coordinates_for_molecule) != molecule.GetNumConformers():
            raise RuntimeError("MMFF94 coordinate conformer totals are unreconciled.")
    coordinate_offsets = [0] * len(molecules)
    coordinate_updates: list[tuple[Any, int, np.ndarray]] = []
    for molecule_index, conformer_index in result_pairs:
        offset = coordinate_offsets[molecule_index]
        coordinates_for_molecule = per_molecule[molecule_index]
        if offset >= len(coordinates_for_molecule):
            raise RuntimeError("MMFF94 coordinate pairs are unreconciled.")
        coordinates = _host_array(coordinates_for_molecule[offset])
        coordinate_offsets[molecule_index] += 1
        molecule = molecules[molecule_index]
        if coordinates.shape != (molecule.GetNumAtoms(), 3):
            raise RuntimeError("An optimized coordinate array has the wrong shape.")
        if not np.isfinite(coordinates).all():
            raise RuntimeError("Optimized coordinates contain non-finite values.")
        coordinate_updates.append((molecule, conformer_index, coordinates))
    if coordinate_offsets != [len(values) for values in per_molecule]:
        raise RuntimeError("MMFF94 coordinate pairs are unreconciled.")
    for molecule, conformer_index, coordinates in coordinate_updates:
        conformer = molecule.GetConformer(conformer_index)
        for atom_index, (x, y, z) in enumerate(coordinates):
            conformer.SetAtomPosition(atom_index, Point3D(float(x), float(y), float(z)))

    per_conformer_records.sort(
        key=lambda record: (
            record["optimization_molecule_index"], record["conformer_index"]
        )
    )
    selected_records: list[dict[str, Any]] = []
    for molecule_index, representative in enumerate(successful_records):
        converged_records = [
            record
            for record in per_conformer_records
            if record["optimization_molecule_index"] == molecule_index
            and record["converged"]
        ]
        if converged_records:
            selected = min(
                converged_records,
                key=lambda record: (
                    record["energy_kcal_mol"], record["conformer_index"]
                ),
            ).copy()
            selected["selected_conformer_id"] = (
                f"{representative['molecule_id']}:conf-{selected['conformer_index']}"
            )
            selected_records.append(selected)
    attempted = len(per_conformer_records)
    converged_count = sum(record["converged"] for record in per_conformer_records)
    summary: dict[str, Any] = {
        "entry_point": "MMFFOptimizeMoleculesConfs",
        "attempted_conformer_count": attempted,
        "converged_conformer_count": int(converged_count),
        "unconverged_conformer_count": int(attempted - converged_count),
        "per_conformer_records": per_conformer_records,
        "selected_conformer_records": selected_records,
    }
    json.dumps(summary, allow_nan=False)
    figures = (
        _conformer_energy_figure(per_conformer_records),
        _optimized_structure_figure(molecules, selected_records),
    )
    summaries = dict(state.summaries)
    summaries["optimize_conformers_mmff94"] = summary
    state.conformer_molecules = molecules
    state.optimization_result = result
    state.summaries = summaries
    state.phase = WorkflowPhase.OPTIMIZED
    return StageResult(
        stage="optimize_conformers_mmff94",
        display_label="nvMolKit MMFFOptimizeMoleculesConfs",
        summary=summary,
        figures=figures,
    )


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _reject_nonfinite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonfinite(nested)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise RuntimeError("Workflow evidence contains non-finite values.")


def _canonical_json(payload: dict[str, Any]) -> str:
    _reject_nonfinite(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _similarity_evidence(state: WorkflowState) -> dict[str, Any]:
    if state.similarity is None:
        raise RuntimeError("similarity artifact is missing.")
    matrix = _host_array(state.similarity.torch())
    molecule_count = len(state.records)
    if (
        matrix.shape != (molecule_count, molecule_count)
        or not np.isfinite(matrix).all()
        or np.any((matrix < 0) | (matrix > 1))
        or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-7)
        or not np.allclose(np.diag(matrix), 1, rtol=0, atol=1e-7)
    ):
        raise RuntimeError("Similarity artifact invariants are invalid.")
    metrics = _similarity_artifact_metrics(matrix, state.records)
    return {
        "matrix_shape": metrics["matrix_shape"],
        "q1": metrics["q1"],
        "median": metrics["median"],
        "q3": metrics["q3"],
        "p90": metrics["p90"],
        "max_off_diagonal": metrics["max"],
        "most_similar_pair": metrics["most_similar_nonidentical_pair"],
        "similarity_unit": "Tanimoto coefficient",
    }


def _representative_evidence(state: WorkflowState) -> dict[str, Any]:
    if state.embedding_parameters is None:
        raise RuntimeError("embedding run parameters are missing.")
    requested, policy, conformers_per_representative = state.embedding_parameters
    representatives: list[dict[str, Any]] = []
    seen_clusters: set[int] = set()
    for record in state.representative_records:
        molecule_index = int(record["molecule_index"])
        cluster_id = int(record["cluster_id"])
        if (
            not 0 <= molecule_index < len(state.records)
            or not 0 <= cluster_id < len(state.clusters)
            or molecule_index not in state.clusters[cluster_id]
            or cluster_id in seen_clusters
        ):
            raise RuntimeError("representative provenance is duplicate or unknown.")
        source = state.records[molecule_index]
        if (
            str(record["molecule_id"]) != str(source["id"])
            or int(record["source_row"]) != int(source["source_row"])
        ):
            raise RuntimeError("representative provenance is duplicate or unknown.")
        generated = int(record["generated_conformer_count"])
        if not 0 <= generated <= conformers_per_representative:
            raise RuntimeError("Embedding totals are unreconciled.")
        seen_clusters.add(cluster_id)
        representatives.append(
            {
                "molecule_id": str(record["molecule_id"]),
                "source_row": int(record["source_row"]),
                "cluster_id": cluster_id,
            }
        )
    generated_total = sum(
        int(record["generated_conformer_count"])
        for record in state.representative_records
    )
    return {
        "requested_representative_count": requested,
        "selected_representative_count": len(representatives),
        "selection_shortfall": requested - len(representatives),
        "representative_policy": policy,
        "representatives": representatives,
        "requested_conformers_per_representative": conformers_per_representative,
        "generated_conformer_count": generated_total,
        "partial_embedding_ids": [
            str(record["molecule_id"])
            for record in state.representative_records
            if 0 < int(record["generated_conformer_count"])
            < conformers_per_representative
        ],
        "zero_embedding_ids": [
            str(record["molecule_id"])
            for record in state.representative_records
            if int(record["generated_conformer_count"]) == 0
        ],
        "count_unit": "conformers",
    }


def _optimization_evidence(
    state: WorkflowState, generated_conformer_count: int
) -> dict[str, Any]:
    if state.optimization_result is None:
        raise RuntimeError("Optimization artifact is missing.")
    representatives = [
        record
        for record in state.representative_records
        if int(record["generated_conformer_count"]) > 0
    ]
    if len(representatives) != len(state.conformer_molecules):
        raise RuntimeError("Embedded molecule provenance is unreconciled.")
    rows = _optimization_records(
        state.optimization_result, state.conformer_molecules, representatives
    )
    if len(rows) != generated_conformer_count:
        raise RuntimeError("Conformer pairs are incomplete or duplicated.")
    rows.sort(
        key=lambda record: (
            record["optimization_molecule_index"], record["conformer_index"]
        )
    )

    selected: list[dict[str, Any]] = []
    for molecule_index in range(len(representatives)):
        converged_rows = [
            record
            for record in rows
            if record["optimization_molecule_index"] == molecule_index
            and record["converged"]
        ]
        if not converged_rows:
            continue
        minimum = min(
            converged_rows,
            key=lambda record: (
                record["energy_kcal_mol"], record["conformer_index"]
            ),
        )
        selected.append(
            {
                **{key: minimum[key] for key in (
                    "molecule_id", "cluster_id", "conformer_index",
                    "energy_kcal_mol", "converged",
                )},
                "selected_conformer_id": (
                    f"{minimum['molecule_id']}:conf-{minimum['conformer_index']}"
                ),
            }
        )
    public_rows = [
        {key: row[key] for key in (
            "molecule_id", "cluster_id", "conformer_index",
            "energy_kcal_mol", "converged",
        )}
        for row in rows
    ]
    converged_count = sum(row["converged"] for row in rows)
    return {
        "attempted_conformer_count": len(rows),
        "converged_conformer_count": int(converged_count),
        "unconverged_conformer_count": len(rows) - int(converged_count),
        "per_conformer_records": public_rows,
        "selected_conformer_records": selected,
        "energy_unit": "kcal/mol",
        "comparison_scope": "within molecule only",
    }


def build_workflow_report(state: WorkflowState) -> WorkflowReport:
    """Build immutable E01-E06 evidence directly from retained artifacts."""
    if state.phase is not WorkflowPhase.OPTIMIZED:
        raise RuntimeError("build_workflow_report requires an OPTIMIZED workflow")
    molecule_count = len(state.records)
    if molecule_count < 2 or len(state.molecules) != molecule_count:
        raise RuntimeError("Validated molecule rows are unreconciled.")
    if state.fingerprints is None:
        raise RuntimeError("fingerprint artifact is missing.")
    if state.fingerprint_parameters is None:
        raise RuntimeError("fingerprint run parameters are missing.")
    radius, fingerprint_size = state.fingerprint_parameters
    fingerprint = _fingerprint_artifact_metrics(state.fingerprints)
    expected_shape = [molecule_count, fingerprint_size // 32]
    if fingerprint["packed_shape"] != expected_shape:
        raise RuntimeError("Fingerprint dimensions are unreconciled.")

    similarity = _similarity_evidence(state)
    if state.cluster_cutoff is None:
        raise RuntimeError("cluster cutoff artifact is missing.")
    assigned = [index for cluster in state.clusters for index in cluster]
    if sorted(assigned) != list(range(molecule_count)):
        raise RuntimeError("Cluster assignment is incomplete or duplicated.")
    cluster = _cluster_artifact_metrics(
        state.clusters, molecule_count, state.cluster_cutoff
    )
    embedding = _representative_evidence(state)
    optimization = _optimization_evidence(
        state, embedding["generated_conformer_count"]
    )

    payloads = (
        ("E01", "Library inspection", {
            "raw_count": molecule_count + len(state.invalid_ids),
            "valid_count": molecule_count,
            "invalid_count": len(state.invalid_ids),
            "invalid_ids": list(state.invalid_ids),
            "preview_count": min(molecule_count, 24),
            "count_unit": "rows",
        }, "RDKit input validation"),
        ("E02", "Morgan fingerprints", {
            "fingerprint_radius": radius,
            "fingerprint_size_bits": fingerprint_size,
            "packed_shape": fingerprint["packed_shape"],
            "molecule_count": fingerprint["molecule_count"],
            "active_bits_min": fingerprint["active_bits_min"],
            "active_bits_median": fingerprint["active_bits_median"],
            "active_bits_max": fingerprint["active_bits_max"],
            "executor": "nvMolKit GPU",
            "size_unit": "bits",
        }, "MorganFingerprintGenerator"),
        ("E03", "Tanimoto similarity", similarity, "crossTanimotoSimilarity"),
        ("E04", "Fused Butina clusters", {
            "cutoff": cluster["cluster_cutoff"],
            "cluster_count": cluster["cluster_count"],
            "singleton_count": cluster["singleton_count"],
            "singleton_fraction": cluster["singleton_fraction"],
            "largest_cluster_sizes": cluster["largest_cluster_sizes"],
            "assignment_count": cluster["assignment_count"],
            "cutoff_unit": "Tanimoto distance",
        }, "fused_butina"),
        ("E05", "Representative embedding", embedding, "EmbedMolecules"),
        ("E06", "MMFF94 optimization", optimization, "MMFFOptimizeMoleculesConfs"),
    )
    return WorkflowReport(tuple(
        EvidenceRecord(key, label, _canonical_json(payload), provenance)
        for key, label, payload, provenance in payloads
    ))
