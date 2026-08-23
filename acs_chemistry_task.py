#!/usr/bin/env python3
"""Run the bounded ACS nvMolKit similarity task and package its artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 1024
TOP_PAIR_COUNT = 10

# NEMOTRON_EDIT_POINT: change this threshold to answer a different similarity question.
HIGHLIGHT_THRESHOLD = 0.70
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "sample_molecules.csv"

ARCHIVE_MEMBERS = (
    "README.md",
    "summary.json",
    "top_10_pairs.csv",
    "similarity_heatmap.png",
)


@dataclass(frozen=True)
class GpuAnalysis:
    """The small, explicit boundary between GPU work and artifact creation."""

    similarity: np.ndarray
    gpu_name: str
    device: str
    torch_version: str
    nvmolkit_version: str


GpuRunner = Callable[[list[Any], int, int], GpuAnalysis]


def run_gpu_analysis(molecules: list[Any], radius: int, size: int) -> GpuAnalysis:
    """Run real nvMolKit Morgan fingerprints and all-pairs Tanimoto on CUDA."""
    import nvmolkit
    import torch

    from chemistry_workflow import (
        WorkflowPhase,
        WorkflowState,
        generate_morgan_fingerprints,
        measure_tanimoto_similarity,
        validated_similarity_matrix,
    )

    state = WorkflowState(
        phase=WorkflowPhase.INSPECTED,
        records=[
            {"id": str(index), "source_row": index} for index in range(len(molecules))
        ],
        molecules=molecules,
    )
    generate_morgan_fingerprints(
        state,
        fingerprint_radius=radius,
        fingerprint_size=size,
    )
    measure_tanimoto_similarity(state)
    fingerprint_tensor = state.fingerprints.torch()
    return GpuAnalysis(
        similarity=validated_similarity_matrix(state),
        gpu_name=str(torch.cuda.get_device_name(fingerprint_tensor.device)),
        device=str(fingerprint_tensor.device),
        torch_version=str(torch.__version__),
        nvmolkit_version=str(nvmolkit.__version__),
    )


def _load_dataset(dataset_path: Path) -> tuple[list[dict[str, str]], list[Any]]:
    table = pd.read_csv(dataset_path, dtype=str)
    if list(table.columns) != ["molecule_id", "smiles"]:
        raise ValueError("dataset must contain only molecule_id and smiles columns")
    if len(table) < 5:
        raise ValueError("dataset must contain at least five molecules for ten pairs")
    if table["molecule_id"].duplicated().any():
        raise ValueError("molecule_id values must be unique")

    records: list[dict[str, str]] = []
    molecules: list[Any] = []
    for row in table.itertuples(index=False):
        molecule_id = str(row.molecule_id)
        smiles = str(row.smiles)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES for {molecule_id}")
        records.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
            }
        )
        molecules.append(molecule)
    return records, molecules


def _validated_matrix(value: np.ndarray, molecule_count: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = (molecule_count, molecule_count)
    if matrix.shape != expected_shape:
        raise RuntimeError(f"similarity matrix must have shape {expected_shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("similarity matrix contains non-finite values")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise RuntimeError("similarity values must be between zero and one")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-7):
        raise RuntimeError("similarity matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-7):
        raise RuntimeError("similarity matrix diagonal must equal one")
    return matrix


def _ranked_pairs(
    records: list[dict[str, str]], matrix: np.ndarray
) -> list[dict[str, object]]:
    pairs = [
        {
            "molecule_1_id": records[first]["molecule_id"],
            "molecule_2_id": records[second]["molecule_id"],
            "smiles_1": records[first]["smiles"],
            "smiles_2": records[second]["smiles"],
            "tanimoto_similarity": float(matrix[first, second]),
            "source_pair": (first, second),
        }
        for first in range(len(records))
        for second in range(first + 1, len(records))
        if records[first]["canonical_smiles"] != records[second]["canonical_smiles"]
    ]
    pairs.sort(
        key=lambda pair: (
            -float(pair["tanimoto_similarity"]),
            pair["source_pair"],
        )
    )
    return pairs


def _write_pairs(path: Path, pairs: list[dict[str, object]]) -> None:
    fields = (
        "rank",
        "molecule_1_id",
        "molecule_2_id",
        "smiles_1",
        "smiles_2",
        "tanimoto_similarity",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, pair in enumerate(pairs[:TOP_PAIR_COUNT], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **{key: pair[key] for key in fields[1:-1]},
                    "tanimoto_similarity": f"{float(pair['tanimoto_similarity']):.6f}",
                }
            )


def _write_heatmap(path: Path, matrix: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from matplotlib.figure import Figure

        figure = Figure(figsize=(6.0, 5.0), dpi=120, layout="constrained")
        axes = figure.subplots()
        image = axes.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        highlighted_rows, highlighted_columns = np.where(
            np.triu(matrix >= HIGHLIGHT_THRESHOLD, k=1)
        )
        axes.scatter(
            highlighted_columns,
            highlighted_rows,
            marker="s",
            facecolors="none",
            edgecolors="#d62728",
            linewidths=0.8,
        )
        axes.set_title("All-pairs nvMolKit Tanimoto similarity")
        axes.set_xlabel("Molecule index")
        axes.set_ylabel("Molecule index")
        figure.colorbar(image, ax=axes, label="Tanimoto similarity")
        figure.savefig(
            path,
            format="png",
            dpi=120,
            metadata={"Software": "ACS nvMolKit task"},
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_readme(path: Path, summary: dict[str, object]) -> None:
    gpu = summary["gpu"]
    results = summary["results"]
    assert isinstance(gpu, dict) and isinstance(results, dict)
    top_pair = results["top_pair"]
    assert isinstance(top_pair, dict)
    path.write_text(
        "\n".join(
            [
                "# ACS nvMolKit similarity results",
                "",
                f"nvMolKit ran Morgan fingerprints and all-pairs Tanimoto similarity on {gpu['gpu_name']}.",
                f"The heatmap marks pairs at or above {HIGHLIGHT_THRESHOLD:.2f}.",
                "",
                "## Top pair",
                "",
                f"- {top_pair['molecule_1_id']} and {top_pair['molecule_2_id']}",
                f"- Tanimoto similarity: {float(top_pair['tanimoto_similarity']):.6f}",
                "",
                "## Files",
                "",
                "- `top_10_pairs.csv`: the ten highest-scoring unique pairs.",
                "- `similarity_heatmap.png`: the complete similarity matrix.",
                "- `summary.json`: parameters, GPU identity, and summary metrics.",
                "- `results.zip`: the four files above in one deterministic archive.",
                "",
                "## Interpretation limit",
                "",
                "Structural similarity does not establish biological activity, safety, or synthetic feasibility.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_deterministic_zip(output_dir: Path) -> None:
    with zipfile.ZipFile(
        output_dir / "results.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in ARCHIVE_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (output_dir / name).read_bytes(), compresslevel=9)


def run_analysis(
    dataset_path: Path,
    output_dir: Path,
    *,
    gpu_runner: GpuRunner = run_gpu_analysis,
) -> dict[str, object]:
    """Run one bounded analysis and return the written summary."""
    records, molecules = _load_dataset(dataset_path)
    gpu = gpu_runner(molecules, FINGERPRINT_RADIUS, FINGERPRINT_SIZE)
    matrix = _validated_matrix(gpu.similarity, len(records))
    pairs = _ranked_pairs(records, matrix)
    top_pair = pairs[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_pairs(output_dir / "top_10_pairs.csv", pairs)
    _write_heatmap(output_dir / "similarity_heatmap.png", matrix)
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "filename": dataset_path.name,
            "molecule_count": len(records),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "parameters": {
            "fingerprint_radius": FINGERPRINT_RADIUS,
            "fingerprint_size": FINGERPRINT_SIZE,
            "highlight_threshold": HIGHLIGHT_THRESHOLD,
            "top_pair_count": TOP_PAIR_COUNT,
        },
        "gpu": {
            "device": gpu.device,
            "gpu_name": gpu.gpu_name,
            "nvmolkit_version": gpu.nvmolkit_version,
            "torch_version": gpu.torch_version,
        },
        "results": {
            "unique_pair_count": len(pairs),
            "pairs_at_or_above_threshold": sum(
                float(pair["tanimoto_similarity"]) >= HIGHLIGHT_THRESHOLD
                for pair in pairs
            ),
            "top_pair": {
                "molecule_1_id": top_pair["molecule_1_id"],
                "molecule_2_id": top_pair["molecule_2_id"],
                "tanimoto_similarity": round(float(top_pair["tanimoto_similarity"]), 6),
            },
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_readme(output_dir / "README.md", summary)
    _write_deterministic_zip(output_dir)
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    gpu_runner: GpuRunner = run_gpu_analysis,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory that will receive the five task artifacts.",
    )
    args = parser.parse_args(argv)
    run_analysis(DEFAULT_DATASET, args.output, gpu_runner=gpu_runner)
    print(f"Download: {args.output / 'results.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
