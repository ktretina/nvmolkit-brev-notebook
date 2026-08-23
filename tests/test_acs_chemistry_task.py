from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "acs_chemistry_task.py"


def _load_task_module() -> ModuleType:
    assert SCRIPT.is_file(), "the ACS chemistry task script is missing"
    spec = importlib.util.spec_from_file_location("acs_chemistry_task_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path) -> Path:
    pd.DataFrame(
        [
            {"molecule_id": "mol-a", "smiles": "CCO"},
            {"molecule_id": "mol-b", "smiles": "CCN"},
            {"molecule_id": "mol-c", "smiles": "CCC"},
            {"molecule_id": "mol-d", "smiles": "c1ccccc1"},
            {"molecule_id": "mol-e", "smiles": "CC(=O)O"},
        ]
    ).to_csv(path, index=False)
    return path


def test_run_analysis_writes_the_complete_bounded_artifact_set(tmp_path: Path) -> None:
    module = _load_task_module()
    dataset = _write_dataset(tmp_path / "molecules.csv")
    output = tmp_path / "results"

    def fake_gpu_analysis(molecules: list[object], radius: int, size: int):
        assert len(molecules) == 5
        assert (radius, size) == (2, 1024)
        return module.GpuAnalysis(
            similarity=np.array(
                [
                    [1.00, 0.91, 0.30, 0.20, 0.40],
                    [0.91, 1.00, 0.80, 0.10, 0.50],
                    [0.30, 0.80, 1.00, 0.60, 0.70],
                    [0.20, 0.10, 0.60, 1.00, 0.25],
                    [0.40, 0.50, 0.70, 0.25, 1.00],
                ],
                dtype=np.float32,
            ),
            gpu_name="NVIDIA L4",
            device="cuda:0",
            torch_version="2.7.1+cu128",
            nvmolkit_version="0.5.0",
        )

    returned_summary = module.run_analysis(
        dataset, output, gpu_runner=fake_gpu_analysis
    )

    expected_names = {
        "README.md",
        "results.zip",
        "similarity_heatmap.png",
        "summary.json",
        "top_10_pairs.csv",
    }
    assert {path.name for path in output.iterdir()} == expected_names

    with (output / "top_10_pairs.csv").open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 10
    assert pairs[0] == {
        "rank": "1",
        "molecule_1_id": "mol-a",
        "molecule_2_id": "mol-b",
        "smiles_1": "CCO",
        "smiles_2": "CCN",
        "tanimoto_similarity": "0.910000",
    }
    assert [float(pair["tanimoto_similarity"]) for pair in pairs] == sorted(
        (float(pair["tanimoto_similarity"]) for pair in pairs), reverse=True
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary == returned_summary
    assert summary["schema_version"] == 1
    assert summary["dataset"]["molecule_count"] == 5
    assert summary["parameters"] == {
        "fingerprint_radius": 2,
        "fingerprint_size": 1024,
        "highlight_threshold": 0.7,
        "top_pair_count": 10,
    }
    assert summary["gpu"] == {
        "device": "cuda:0",
        "gpu_name": "NVIDIA L4",
        "nvmolkit_version": "0.5.0",
        "torch_version": "2.7.1+cu128",
    }
    assert summary["results"]["pairs_at_or_above_threshold"] == 2
    assert summary["results"]["top_pair"] == {
        "molecule_1_id": "mol-a",
        "molecule_2_id": "mol-b",
        "tanimoto_similarity": 0.91,
    }

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "nvMolKit" in readme
    assert "NVIDIA L4" in readme
    assert "0.70" in readme
    assert "mol-a" in readme and "mol-b" in readme
    assert "Structural similarity does not establish biological activity" in readme

    image = (output / "similarity_heatmap.png").read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 1_000

    with zipfile.ZipFile(output / "results.zip") as archive:
        assert archive.namelist() == [
            "README.md",
            "summary.json",
            "top_10_pairs.csv",
            "similarity_heatmap.png",
        ]
        for member in archive.infolist():
            assert member.date_time == (2026, 1, 1, 0, 0, 0)
            assert (
                archive.read(member.filename) == (output / member.filename).read_bytes()
            )

    repeated_output = tmp_path / "repeated-results"
    module.run_analysis(dataset, repeated_output, gpu_runner=fake_gpu_analysis)
    assert {name: (output / name).read_bytes() for name in sorted(expected_names)} == {
        name: (repeated_output / name).read_bytes() for name in sorted(expected_names)
    }


def test_main_uses_the_bundled_dataset_and_reports_the_download(
    tmp_path: Path, capsys
) -> None:
    module = _load_task_module()
    output = tmp_path / "current"

    def fake_gpu_analysis(molecules: list[object], radius: int, size: int):
        molecule_count = len(molecules)
        matrix = np.full((molecule_count, molecule_count), 0.25, dtype=np.float32)
        np.fill_diagonal(matrix, 1.0)
        return module.GpuAnalysis(
            similarity=matrix,
            gpu_name="NVIDIA L4",
            device="cuda:0",
            torch_version="2.7.1+cu128",
            nvmolkit_version="0.5.0",
        )

    result = module.main(
        ["--output", str(output)],
        gpu_runner=fake_gpu_analysis,
    )

    assert result == 0
    assert json.loads((output / "summary.json").read_text())["dataset"] == {
        "filename": "sample_molecules.csv",
        "molecule_count": 256,
        "sha256": "7063a5d8eded837e3e648c44894fbe742d5863a0929bb5765b1c6330722fb034",
    }
    stdout = capsys.readouterr().out
    assert str(output / "results.zip") in stdout


def test_ranked_pairs_excludes_identical_canonical_smiles(tmp_path: Path) -> None:
    module = _load_task_module()
    dataset = tmp_path / "canonical-duplicates.csv"
    pd.DataFrame(
        [
            {"molecule_id": "mol-a", "smiles": "CCO"},
            {"molecule_id": "mol-b", "smiles": "OCC"},
            {"molecule_id": "mol-c", "smiles": "CCN"},
            {"molecule_id": "mol-d", "smiles": "CCC"},
            {"molecule_id": "mol-e", "smiles": "c1ccccc1"},
        ]
    ).to_csv(dataset, index=False)
    records, _ = module._load_dataset(dataset)
    matrix = np.array(
        [
            [1.00, 1.00, 0.91, 0.30, 0.20],
            [1.00, 1.00, 0.90, 0.29, 0.19],
            [0.91, 0.90, 1.00, 0.80, 0.10],
            [0.30, 0.29, 0.80, 1.00, 0.60],
            [0.20, 0.19, 0.10, 0.60, 1.00],
        ]
    )

    pairs = module._ranked_pairs(records, matrix)

    assert len(pairs) == 9
    assert {pairs[0]["molecule_1_id"], pairs[0]["molecule_2_id"]} == {
        "mol-a",
        "mol-c",
    }
    assert all(
        {pair["molecule_1_id"], pair["molecule_2_id"]} != {"mol-a", "mol-b"}
        for pair in pairs
    )


def test_ranked_pairs_keep_raw_scores_for_ordering_and_thresholds(
    tmp_path: Path,
) -> None:
    module = _load_task_module()
    dataset = _write_dataset(tmp_path / "molecules.csv")
    records, _ = module._load_dataset(dataset)
    matrix = np.full((5, 5), 0.10, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = 0.6999996
    matrix[0, 2] = matrix[2, 0] = 0.7000004

    pairs = module._ranked_pairs(records, matrix)

    assert (pairs[0]["molecule_1_id"], pairs[0]["molecule_2_id"]) == (
        "mol-a",
        "mol-c",
    )
    assert pairs[0]["tanimoto_similarity"] == 0.7000004
    assert (
        sum(
            float(pair["tanimoto_similarity"]) >= module.HIGHLIGHT_THRESHOLD
            for pair in pairs
        )
        == 1
    )
