from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image
from rdkit import Chem
from rdkit.Geometry import Point3D

import acs_workshop_runner as runner
import chemistry_workflow
from objective_fixtures import (
    controlled_context_with_three_misses,
    controlled_context_with_ranked_swaps,
    controlled_context_with_tied_paths,
    target_achieved_context,
)


MANIFEST_FILES = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
    "data/PROVENANCE.md",
    "objective_challenge.py",
)

EXPECTED_STAGE_IMAGES = {
    "inspect_library": ("library_preview.png",),
    "generate_morgan_fingerprints": ("fingerprint_density.png",),
    "measure_tanimoto_similarity": ("similarity_heatmap.png",),
    "discover_fused_butina_clusters": ("cluster_sizes.png",),
    "embed_representative_conformers": ("embedding_counts.png",),
    "optimize_conformers_mmff94": (
        "conformer_energies.png",
        "optimized_structures.png",
    ),
}

EXPECTED_STAGE_DATA = {
    "inspect_library": (),
    "generate_morgan_fingerprints": (),
    "measure_tanimoto_similarity": (
        "top_similarity_pairs.csv",
        "similarity_matrix.csv",
    ),
    "discover_fused_butina_clusters": ("cluster_assignments.csv",),
    "embed_representative_conformers": (),
    "optimize_conformers_mmff94": (
        "mmff94_energies.csv",
        "optimized_conformers.sdf",
        "workflow_evidence.json",
    ),
}

EXPECTED_STAGE_METADATA = {
    "inspect_library": (
        "What is in the fixed molecule library?",
        "RDKit input validation",
        "validation does not establish activity or suitability",
    ),
    "generate_morgan_fingerprints": (
        "What do the GPU Morgan fingerprints show?",
        "nvMolKit MorganFingerprintGenerator",
        "fingerprints are structural descriptors, not biological evidence",
    ),
    "measure_tanimoto_similarity": (
        "Which molecules are most similar in this fingerprint space?",
        "nvMolKit crossTanimotoSimilarity",
        "similarity does not establish activity, binding, efficacy, or safety",
    ),
    "discover_fused_butina_clusters": (
        "How does fused Butina partition the library?",
        "nvMolKit fused_butina with RDKit MMFF94 eligibility",
        "clusters depend on this fingerprint and cutoff",
    ),
    "embed_representative_conformers": (
        "Did ETKDGv3 generate the requested representative conformers?",
        "nvMolKit EmbedMolecules",
        "sampled conformers are not experimental structures",
    ),
    "optimize_conformers_mmff94": (
        "Which sampled conformers converged under MMFF94?",
        "nvMolKit MMFFOptimizeMoleculesConfs",
        "MMFF94 compares sampled force-field geometries within each molecule only",
    ),
}

EXPECTED_STAGE_DIRECTORIES = {
    "inspect_library": "01-inspection",
    "generate_morgan_fingerprints": "02-fingerprints",
    "measure_tanimoto_similarity": "03-similarity",
    "discover_fused_butina_clusters": "04-clusters",
    "embed_representative_conformers": "05-conformers",
    "optimize_conformers_mmff94": "06-mmff94",
}

EXPECTED_STAGE_FACTS = {
    "inspect_library": {
        "raw_count": 256,
        "valid_count": 256,
        "invalid_count": 0,
        "preview_count": 24,
        "unused_internal_detail": "DO_NOT_RENDER",
    },
    "generate_morgan_fingerprints": {
        "fingerprint_radius": 2,
        "fingerprint_size": 1024,
        "packed_shape": [256, 32],
        "active_bits_min": 4,
        "active_bits_median": 17.5,
        "active_bits_max": 41,
        "unused_internal_detail": "DO_NOT_RENDER",
    },
    "measure_tanimoto_similarity": {
        "q1": 0.125,
        "median": 0.25,
        "q3": 0.375,
        "p90": 0.625,
        "most_similar_nonidentical_pair": {
            "molecule_ids": ["CHEMBL6223", "CHEMBL6228"],
            "source_rows": [21, 22],
            "similarity": 1.0,
        },
        "unused_internal_detail": "DO_NOT_RENDER",
    },
    "discover_fused_butina_clusters": {
        "cluster_cutoff": 0.4,
        "cluster_count": 39,
        "singleton_count": 12,
        "largest_cluster_sizes": [31, 25, 18, 14, 12],
        "unused_internal_detail": "DO_NOT_RENDER",
    },
    "embed_representative_conformers": {
        "requested_representative_count": 6,
        "selected_representative_count": 6,
        "requested_conformers_per_representative": 5,
        "generated_conformer_count": 29,
        "partial_embedding_ids": ["CHEMBL300"],
        "zero_embedding_ids": [],
        "unused_internal_detail": "DO_NOT_RENDER",
    },
    "optimize_conformers_mmff94": {
        "attempted_conformer_count": 29,
        "converged_conformer_count": 27,
        "unconverged_conformer_count": 2,
        "selected_conformer_records": [
            {"molecule_id": "CHEMBL100", "energy_kcal_mol": -12.3456},
            {"molecule_id": "CHEMBL200", "energy_kcal_mol": 3.25},
        ],
        "unused_internal_detail": "DO_NOT_RENDER",
    },
}

EXPECTED_STAGE_RESULTS = {
    "inspect_library": (
        "256 raw rows; 256 valid molecules; 0 invalid molecules; "
        "24 molecules in the preview."
    ),
    "generate_morgan_fingerprints": (
        "Morgan radius 2 with 1024 bits produced packed shape 256 x 32; "
        "active bits min 4, median 17.500, max 41."
    ),
    "measure_tanimoto_similarity": (
        'top non-self pair "CHEMBL6223" and "CHEMBL6228" had Tanimoto '
        "similarity 1.000; q1 0.125, median 0.250, q3 0.375, p90 0.625."
    ),
    "discover_fused_butina_clusters": (
        "cutoff 0.40 produced 39 clusters with 12 singletons; "
        "largest cluster sizes: 31, 25, 18, 14, 12."
    ),
    "embed_representative_conformers": (
        "selected 6 of 6 representatives and generated 29 of 30 requested "
        "conformers; 1 partial ID, 0 zero IDs; ETKDGv3 seed 7."
    ),
    "optimize_conformers_mmff94": (
        "29 conformers attempted; 27 converged; 2 unconverged; "
        'within-molecule minima: "CHEMBL100"=-12.346 kcal/mol, '
        '"CHEMBL200"=3.250 kcal/mol; maximum iterations 500.'
    ),
}

FIXED_GPU = runner.GpuIdentity(
    name="NVIDIA L4",
    device="cuda:0",
    torch_version="2.7.1+cu128",
    nvmolkit_version="0.5.0",
)


def write_manifest(root: Path) -> runner.WorkshopPaths:
    paths = runner.WorkshopPaths(root)
    paths.state_root.mkdir(mode=0o700)
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in MANIFEST_FILES
    }
    payload = {"schema_version": 1, "files": files}
    paths.manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    paths.manifest_path.chmod(0o444)
    return paths


@pytest.fixture
def workshop_paths(tmp_path: Path) -> runner.WorkshopPaths:
    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "workshop"
    sources = {
        "TOOLS.md": source_root / "launchable" / "acs_workspace_tools.md",
        "acs_workshop_runner.py": source_root / "acs_workshop_runner.py",
        "chemistry_workflow.py": source_root / "chemistry_workflow.py",
        "data/sample_molecules.csv": source_root / "data" / "sample_molecules.csv",
        "data/PROVENANCE.md": source_root / "data" / "PROVENANCE.md",
        "objective_challenge.py": source_root / "objective_challenge.py",
    }
    for name, source in sources.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return write_manifest(root)


class _FakeTensor:
    def __init__(self, values: object, *, device: str = "cuda:0") -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.device = device

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values.copy()


class _FakeGpuResult:
    def __init__(self, values: object, *, device: str = "cuda:0") -> None:
        self.tensor = _FakeTensor(values, device=device)

    def torch(self) -> _FakeTensor:
        return self.tensor


class _FakeOptimizationResult:
    def __init__(self, molecules: list[Chem.Mol]) -> None:
        pairs = [
            (molecule_index, conformer_index)
            for molecule_index, molecule in enumerate(molecules)
            for conformer_index in range(molecule.GetNumConformers())
        ]
        self.mol_indices = _FakeGpuResult([pair[0] for pair in pairs])
        self.conf_indices = _FakeGpuResult([pair[1] for pair in pairs])
        self.energies = _FakeGpuResult(
            [
                float(10 * molecule_index + conformer_index) + 0.125
                for molecule_index, conformer_index in pairs
            ]
        )
        self.converged = _FakeGpuResult(
            [
                int((molecule_index, conformer_index) != (5, 4))
                for molecule_index, conformer_index in pairs
            ]
        )
        self._coordinates = [
            [
                np.asarray(
                    [
                        (
                            molecule_index + atom_index / 100.0,
                            conformer_index + atom_index / 200.0,
                            molecule_index + conformer_index + atom_index / 300.0,
                        )
                        for atom_index in range(molecule.GetNumAtoms())
                    ],
                    dtype=float,
                )
                for conformer_index in range(molecule.GetNumConformers())
            ]
            for molecule_index, molecule in enumerate(molecules)
        ]

    def per_molecule(self) -> list[list[np.ndarray]]:
        return self._coordinates


def _workshop_similarity_matrix() -> np.ndarray:
    molecule_count = 256
    matrix = np.eye(molecule_count, dtype=float)
    for first in range(molecule_count):
        for second in range(first + 1, molecule_count):
            score = ((first * molecule_count + second) % 997) / 10_000.0
            matrix[first, second] = score
            matrix[second, first] = score
    ranked_pairs = (
        (7, 9, 0.99),
        (0, 1, 0.95),
        (0, 2, 0.95),
        (2, 3, 0.94),
        (4, 5, 0.93),
        (6, 8, 0.92),
        (10, 11, 0.91),
        (12, 13, 0.90),
        (14, 15, 0.89),
        (16, 17, 0.88),
        (18, 19, 0.87),
    )
    for first, second, score in ranked_pairs:
        matrix[first, second] = score
        matrix[second, first] = score
    return matrix


@pytest.fixture
def fake_workshop_gpu(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    similarity_matrix = _workshop_similarity_matrix()
    fingerprint_result = _FakeGpuResult(np.zeros((256, 32), dtype=np.int32))
    similarity_result = _FakeGpuResult(similarity_matrix)

    class Generator:
        def __init__(self, *, radius: int, fpSize: int) -> None:
            assert (radius, fpSize) == (2, 1024)

        def GetFingerprints(self, molecules: list[Chem.Mol]) -> _FakeGpuResult:
            assert len(molecules) == 256
            return fingerprint_result

    def cluster(_fingerprints: _FakeTensor, *, cutoff: float):
        assert cutoff == 0.40
        clusters = [list(range(start, start + 32)) for start in range(0, 256, 32)]
        return clusters, [len(cluster_members) for cluster_members in clusters]

    def embed(
        molecules: list[Chem.Mol],
        parameters: object,
        *,
        confsPerMolecule: int,
        maxIterations: int,
    ) -> None:
        assert getattr(parameters, "randomSeed") == 7
        assert getattr(parameters, "useRandomCoords") is True
        assert (len(molecules), confsPerMolecule, maxIterations) == (6, 5, -1)
        for molecule in molecules:
            molecule.RemoveAllConformers()
            for conformer_index in range(confsPerMolecule):
                conformer = Chem.Conformer(molecule.GetNumAtoms())
                conformer.SetId(conformer_index)
                molecule.AddConformer(conformer, assignId=True)

    def optimize(
        molecules: list[Chem.Mol], *, maxIters: int, output: object
    ) -> _FakeOptimizationResult:
        assert len(molecules) == 6
        assert maxIters == 500
        assert output == "cuda:0"
        return _FakeOptimizationResult(molecules)

    monkeypatch.setattr(
        chemistry_workflow, "_morgan_generator_class", lambda: Generator
    )
    monkeypatch.setattr(
        chemistry_workflow,
        "_cross_tanimoto_similarity",
        lambda fingerprints: similarity_result,
    )
    monkeypatch.setattr(chemistry_workflow, "_fused_butina", cluster)
    monkeypatch.setattr(chemistry_workflow, "_embed_molecules", embed)
    monkeypatch.setattr(chemistry_workflow, "_optimize_mmff94", optimize)
    monkeypatch.setattr(
        chemistry_workflow, "_coordinate_output_device", lambda: "cuda:0"
    )
    monkeypatch.setattr(chemistry_workflow, "_synchronize_cuda", lambda: None)
    monkeypatch.setattr(runner, "_gpu_identity", lambda: FIXED_GPU)
    return similarity_matrix


@pytest.fixture
def completed_similarity(
    workshop_paths: runner.WorkshopPaths,
    fake_workshop_gpu: np.ndarray,
) -> Path:
    del fake_workshop_gpu
    runner.run_stage("measure_tanimoto_similarity", paths=workshop_paths)
    return workshop_paths.output_root / "03-similarity"


@pytest.fixture
def completed_clusters(
    workshop_paths: runner.WorkshopPaths,
    fake_workshop_gpu: np.ndarray,
) -> Path:
    del fake_workshop_gpu
    runner.run_stage("discover_fused_butina_clusters", paths=workshop_paths)
    return workshop_paths.output_root / "04-clusters"


@pytest.fixture
def completed_mmff94(
    workshop_paths: runner.WorkshopPaths,
    fake_workshop_gpu: np.ndarray,
) -> Path:
    del fake_workshop_gpu
    runner.run_stage("optimize_conformers_mmff94", paths=workshop_paths)
    return workshop_paths.output_root / "06-mmff94"


@pytest.fixture
def workflow_executions() -> dict[str, runner.WorkflowExecution]:
    records = [
        {"id": f"CHEMBL{index}", "smiles": "C", "source_row": index}
        for index in range(256)
    ]
    clusters = [list(range(cluster_id, 256, 39)) for cluster_id in range(39)]
    representative_records: list[dict[str, object]] = []
    conformer_molecules: list[Chem.Mol] = []
    generated_counts = (5, 5, 5, 5, 5, 4)
    for cluster_id, generated_count in enumerate(generated_counts):
        molecule_index = clusters[cluster_id][0]
        representative_records.append(
            {
                "molecule_id": records[molecule_index]["id"],
                "source_row": records[molecule_index]["source_row"],
                "cluster_id": cluster_id,
                "molecule_index": molecule_index,
                "generated_conformer_count": generated_count,
            }
        )
        molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
        assert molecule is not None
        for conformer_index in range(generated_count):
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            conformer.SetId(conformer_index)
            molecule.AddConformer(conformer, assignId=True)
        conformer_molecules.append(molecule)
    optimization_result = _FakeOptimizationResult(conformer_molecules)
    per_conformer_records = [
        {
            "molecule_id": representative_records[molecule_index]["molecule_id"],
            "cluster_id": molecule_index,
            "conformer_index": conformer_index,
            "energy_kcal_mol": float(10 * molecule_index + conformer_index) + 0.125,
            "converged": (molecule_index, conformer_index) != (5, 3),
            "optimization_molecule_index": molecule_index,
        }
        for molecule_index, molecule in enumerate(conformer_molecules)
        for conformer_index in range(molecule.GetNumConformers())
    ]
    state = runner.WorkflowState(
        phase=chemistry_workflow.WorkflowPhase.OPTIMIZED,
        records=records,
        molecules=[Chem.MolFromSmiles("C") for _ in range(256)],
        fingerprints=_FakeGpuResult(np.zeros((256, 32), dtype=np.int32)),
        fingerprint_parameters=(2, 1024),
        similarity=_FakeGpuResult(np.eye(256, dtype=float)),
        clusters=clusters,
        cluster_cutoff=0.40,
        representative_records=representative_records,
        conformer_molecules=conformer_molecules,
        optimization_result=optimization_result,
        summaries={
            "optimize_conformers_mmff94": {
                "per_conformer_records": per_conformer_records,
            }
        },
        embedding_parameters=(6, "largest_clusters_first", 5),
    )
    results: list[runner.StageResult] = []
    executions: dict[str, runner.WorkflowExecution] = {}
    for stage_name in runner.STAGE_ORDER:
        image_names = EXPECTED_STAGE_IMAGES[stage_name]
        if stage_name == "inspect_library":
            figures: tuple[object, ...] = (
                Image.new("RGB", (24, 16), color=(118, 185, 0)),
            )
        else:
            figures = tuple(Figure(figsize=(1.0, 1.0)) for _ in image_names)
        result = runner.StageResult(
            stage=stage_name,
            display_label=EXPECTED_STAGE_METADATA[stage_name][1],
            summary=EXPECTED_STAGE_FACTS[stage_name],
            figures=figures,
        )
        results.append(result)
        executions[stage_name] = runner.WorkflowExecution(
            state=state,
            stage_results=tuple(results),
            gpu=None if stage_name == "inspect_library" else FIXED_GPU,
        )
    return executions


@pytest.fixture(params=runner.STAGE_ORDER)
def completed_stage(
    request: pytest.FixtureRequest,
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> Path:
    stage_name = str(request.param)
    runner.run_stage(
        stage_name,
        paths=workshop_paths,
        workflow_executor=lambda selected: workflow_executions[selected],
    )
    return workshop_paths.output_root / EXPECTED_STAGE_DIRECTORIES[stage_name]


def test_similarity_csvs_match_records_and_matrix(
    completed_similarity: Path,
    fake_workshop_gpu: np.ndarray,
) -> None:
    pairs = pd.read_csv(completed_similarity / "top_similarity_pairs.csv")
    matrix = pd.read_csv(completed_similarity / "similarity_matrix.csv")
    assert list(pairs.columns) == [
        "rank",
        "molecule_1_id",
        "molecule_1_source_row",
        "molecule_2_id",
        "molecule_2_source_row",
        "tanimoto_similarity",
    ]
    assert pairs["rank"].tolist() == list(range(1, 11))
    assert matrix.shape == (256, 258)
    assert matrix.columns[:2].tolist() == ["molecule_id", "source_row"]
    assert matrix["molecule_id"].tolist() == matrix.columns[2:].tolist()
    assert matrix["source_row"].tolist() == list(range(256))
    assert np.allclose(matrix.iloc[:, 2:].to_numpy(), fake_workshop_gpu)

    expected_pairs = sorted(
        (
            (-float(fake_workshop_gpu[first, second]), first, second)
            for first in range(256)
            for second in range(first + 1, 256)
        )
    )[:10]
    for row, (negative_score, first, second) in zip(
        pairs.to_dict(orient="records"), expected_pairs, strict=True
    ):
        assert row["molecule_1_id"] == matrix.iloc[first]["molecule_id"]
        assert row["molecule_1_source_row"] == first
        assert row["molecule_2_id"] == matrix.iloc[second]["molecule_id"]
        assert row["molecule_2_source_row"] == second
        assert row["tanimoto_similarity"] == pytest.approx(-negative_score)


def test_cluster_assignments_cover_each_molecule_once(
    completed_clusters: Path,
) -> None:
    rows = pd.read_csv(completed_clusters / "cluster_assignments.csv")
    assert list(rows.columns) == [
        "molecule_index",
        "molecule_id",
        "source_row",
        "cluster_id",
        "cluster_size",
    ]
    assert rows["molecule_index"].tolist() == list(range(256))
    assert rows["molecule_id"].is_unique
    assert rows["source_row"].tolist() == list(range(256))
    assert rows.groupby("cluster_id")["molecule_index"].count().to_dict() == {
        cluster_id: 32 for cluster_id in range(8)
    }
    assert rows["cluster_size"].tolist() == [32] * 256


def test_mmff94_csv_and_sdf_have_matching_provenance(
    completed_mmff94: Path,
) -> None:
    rows = pd.read_csv(completed_mmff94 / "mmff94_energies.csv")
    assert list(rows.columns) == [
        "record_id",
        "molecule_id",
        "source_row",
        "cluster_id",
        "conformer_index",
        "energy_kcal_mol",
        "converged",
    ]
    supplier = Chem.SDMolSupplier(
        str(completed_mmff94 / "optimized_conformers.sdf"),
        removeHs=False,
    )
    molecules = [molecule for molecule in supplier if molecule is not None]
    assert len(molecules) == len(rows) == 30
    assert [molecule.GetProp("ACS_RECORD_ID") for molecule in molecules] == (
        rows["record_id"].tolist()
    )
    for molecule, row in zip(molecules, rows.to_dict(orient="records"), strict=True):
        assert molecule.GetProp("MOLECULE_ID") == row["molecule_id"]
        assert int(molecule.GetProp("SOURCE_ROW")) == row["source_row"]
        assert int(molecule.GetProp("CLUSTER_ID")) == row["cluster_id"]
        assert int(molecule.GetProp("CONFORMER_INDEX")) == row["conformer_index"]
        assert float(molecule.GetProp("MMFF94_ENERGY_KCAL_MOL")) == pytest.approx(
            row["energy_kcal_mol"]
        )
        assert molecule.GetProp("CONVERGED").lower() == str(row["converged"]).lower()


def test_workflow_evidence_has_parsed_e01_through_e06(
    completed_mmff94: Path,
) -> None:
    payload = json.loads(
        (completed_mmff94 / "workflow_evidence.json").read_text(encoding="utf-8")
    )
    assert set(payload) == {"schema_version", "evidence"}
    assert payload["schema_version"] == 1
    assert [record["key"] for record in payload["evidence"]] == [
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
    ]
    assert all(
        set(record) == {"key", "label", "payload", "provenance"}
        for record in payload["evidence"]
    )
    assert all(type(record["payload"]) is dict for record in payload["evidence"])


@pytest.mark.parametrize(
    "mutation",
    (
        "malformed_representative",
        "unknown_conformer_id",
        "boolean_cluster_id",
        "nonfinite_coordinate",
    ),
)
def test_mmff94_csv_validation_rejects_invalid_provenance_before_writing(
    workflow_executions: dict[str, runner.WorkflowExecution],
    mutation: str,
) -> None:
    state = workflow_executions["optimize_conformers_mmff94"].state
    if mutation == "malformed_representative":
        state.representative_records.append(
            {
                "molecule_index": 200,
                "molecule_id": "CHEMBL200",
                "source_row": 200,
                "cluster_id": 5,
                "generated_conformer_count": "5",
            }
        )
    elif mutation == "unknown_conformer_id":
        state.conformer_molecules[0].GetConformer(0).SetId(99)
    elif mutation == "boolean_cluster_id":
        rows = state.summaries["optimize_conformers_mmff94"]["per_conformer_records"]
        assert isinstance(rows, list)
        rows[0]["cluster_id"] = False
    elif mutation == "nonfinite_coordinate":
        state.conformer_molecules[0].GetConformer(0).SetAtomPosition(
            0, Point3D(float("nan"), 0.0, 0.0)
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=r"^Workshop chemistry export is invalid\.$"):
        runner._mmff94_exports(state)


def _manifest_payload(paths: runner.WorkshopPaths) -> dict[str, object]:
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


def _replace_manifest(paths: runner.WorkshopPaths, payload: dict[str, object]) -> None:
    paths.manifest_path.chmod(0o600)
    paths.manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    paths.manifest_path.chmod(0o444)


def _assert_manifest_rejected_before_executor(
    paths: runner.WorkshopPaths,
) -> None:
    calls: list[str] = []

    def forbidden_executor(stage_name: str) -> runner.WorkflowExecution:
        calls.append(stage_name)
        raise AssertionError("workflow executor must not run")

    with pytest.raises(RuntimeError, match="(?i)manifest") as caught:
        runner.run_stage(
            "inspect_library",
            paths=paths,
            workflow_executor=forbidden_executor,
        )
    assert calls == []
    assert len(str(caught.value)) < 160


def test_stage_summary_has_closed_finite_schema(completed_stage: Path) -> None:
    summary_path = completed_stage / "summary.json"
    summary_text = summary_path.read_text(encoding="utf-8")
    payload = json.loads(summary_text)
    assert set(payload) == {
        "schema_version",
        "stage",
        "dataset",
        "profile",
        "gpu",
        "facts",
        "artifacts",
    }
    assert payload["schema_version"] == 1
    assert set(payload["dataset"]) == {"filename", "molecule_count", "sha256"}
    assert payload["dataset"] == {
        "filename": "sample_molecules.csv",
        "molecule_count": 256,
        "sha256": runner.DATASET_SHA256,
    }
    assert payload["profile"] == runner.PROFILE
    expected_gpu = None
    if payload["stage"] != "inspect_library":
        expected_gpu = {
            "name": "NVIDIA L4",
            "device": "cuda:0",
            "torch_version": "2.7.1+cu128",
            "nvmolkit_version": "0.5.0",
        }
    assert payload["gpu"] == expected_gpu
    assert payload["facts"] == EXPECTED_STAGE_FACTS[payload["stage"]]
    expected_artifacts = sorted(
        (
            "README.md",
            "summary.json",
            *EXPECTED_STAGE_IMAGES[payload["stage"]],
            *EXPECTED_STAGE_DATA[payload["stage"]],
        )
    )
    assert payload["artifacts"] == expected_artifacts
    assert sorted(path.name for path in completed_stage.iterdir()) == expected_artifacts
    assert summary_text == (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    json.dumps(payload, allow_nan=False)


def test_matplotlib_and_pil_image_adapters_write_readable_pngs(
    completed_stage: Path,
) -> None:
    payload = json.loads((completed_stage / "summary.json").read_text())
    expected_images = EXPECTED_STAGE_IMAGES[payload["stage"]]
    assert tuple(sorted(path.name for path in completed_stage.glob("*.png"))) == tuple(
        sorted(expected_images)
    )
    for path in completed_stage.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(path) as image:
            image.verify()


def test_image_adapters_reject_the_wrong_runtime_type(tmp_path: Path) -> None:
    pil_image = Image.new("RGB", (8, 8), color="white")
    matplotlib_figure = Figure(figsize=(1.0, 1.0))
    with pytest.raises(TypeError, match=r"^Expected an exact Matplotlib Figure\.$"):
        runner._save_matplotlib_figure(pil_image, tmp_path / "wrong-mpl.png")
    with pytest.raises(TypeError, match=r"^Expected a PIL image\.$"):
        runner._save_pil_image(matplotlib_figure, tmp_path / "wrong-pil.png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stage_name,bad_figures,error_type,error_message",
    (
        (
            "inspect_library",
            (),
            RuntimeError,
            r"^Workshop stage image count is invalid\.$",
        ),
        (
            "generate_morgan_fingerprints",
            (Image.new("RGB", (8, 8), color="white"),),
            TypeError,
            r"^Expected an exact Matplotlib Figure\.$",
        ),
    ),
)
def test_image_contract_rejects_count_or_type_before_publication(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    stage_name: str,
    bad_figures: tuple[object, ...],
    error_type: type[Exception],
    error_message: str,
) -> None:
    source = workflow_executions[stage_name]
    final_result = source.stage_results[-1]
    bad_result = runner.StageResult(
        stage=final_result.stage,
        display_label=final_result.display_label,
        summary=final_result.summary,
        figures=bad_figures,
    )
    execution = runner.WorkflowExecution(
        state=source.state,
        stage_results=(*source.stage_results[:-1], bad_result),
        gpu=source.gpu,
    )
    with pytest.raises(error_type, match=error_message):
        runner.run_stage(
            stage_name,
            paths=workshop_paths,
            workflow_executor=lambda selected: execution,
        )
    stage_directory = (
        workshop_paths.output_root / EXPECTED_STAGE_DIRECTORIES[stage_name]
    )
    assert not stage_directory.exists()


def test_stage_readme_uses_fixed_metadata_and_measured_result_source(
    completed_stage: Path,
) -> None:
    payload = json.loads((completed_stage / "summary.json").read_text())
    question, method, scientific_limit = EXPECTED_STAGE_METADATA[payload["stage"]]
    readme = (completed_stage / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {question}\n")
    assert f"- Method: {method}\n" in readme
    assert "- Result source: `summary.json`" in readme
    assert f"- Result: {EXPECTED_STAGE_RESULTS[payload['stage']]}\n" in readme
    assert "unused_internal_detail" not in readme
    assert "DO_NOT_RENDER" not in readme
    assert f"- Scientific limit: {scientific_limit}\n" in readme
    assert readme.endswith("\n")


def test_stage_result_envelope_is_exact(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    stage_name = "optimize_conformers_mmff94"
    result = runner.run_stage(
        stage_name,
        paths=workshop_paths,
        workflow_executor=lambda selected: workflow_executions[selected],
    )
    stage_directory = workshop_paths.output_root / "06-mmff94"
    summary_payload = json.loads(
        (stage_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert result == {
        "schema_version": 1,
        "status": "complete",
        "stage": stage_name,
        "summary": summary_payload,
        "image_paths": [
            str((stage_directory / name).resolve())
            for name in EXPECTED_STAGE_IMAGES[stage_name]
        ],
        "artifact_directory": str(stage_directory.resolve()),
        "results_zip_path": str((workshop_paths.output_root / "results.zip").resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def test_run_lesson_parser_has_exact_fixed_choices_and_terminal_mappings() -> None:
    parser = runner.build_parser()
    subcommands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    lesson_parser = subcommands.choices["run-lesson"]
    lesson_action = next(
        action for action in lesson_parser._actions if action.dest == "lesson"
    )
    expected = {
        "data-and-representation": "generate_morgan_fingerprints",
        "relationships-and-groups": "discover_fused_butina_clusters",
        "sampled-3d-geometry": "optimize_conformers_mmff94",
    }
    assert tuple(lesson_action.choices) == tuple(expected)
    assert runner.LESSON_TERMINAL_STAGES == expected
    for lesson in expected:
        assert parser.parse_args(["run-lesson", lesson]).lesson == lesson


def test_run_lesson_executes_one_terminal_prefix_and_returns_closed_compact_items(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    calls: list[str] = []

    def execute(stage_name: str) -> runner.WorkflowExecution:
        calls.append(stage_name)
        return workflow_executions[stage_name]

    result = runner.run_lesson(
        "relationships-and-groups",
        paths=workshop_paths,
        workflow_executor=execute,
    )

    assert calls == ["discover_fused_butina_clusters"]
    assert set(result) == {
        "schema_version",
        "status",
        "lesson",
        "completed_stages",
        "results_zip_path",
        "artifact_relative_zip_path",
    }
    assert result["lesson"] == "relationships-and-groups"
    assert [item["stage"] for item in result["completed_stages"]] == [
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
    ]
    for item in result["completed_stages"]:
        assert set(item) == {
            "stage",
            "result",
            "image_paths",
            "summary_path",
            "readme_path",
            "artifact_directory",
        }
        assert item["result"] == EXPECTED_STAGE_RESULTS[item["stage"]]
        assert "facts" not in item
        assert "records" not in item
        assert "matrix" not in item
    assert not (workshop_paths.output_root / "01-inspection").exists()
    assert (workshop_paths.output_root / "03-similarity").is_dir()
    assert (workshop_paths.output_root / "04-clusters").is_dir()


def test_run_lesson_failure_does_not_publish_a_partial_fixed_stage(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = workflow_executions["generate_morgan_fingerprints"]
    invalid_inspection = runner.StageResult(
        stage="inspect_library",
        display_label="inspection",
        summary=EXPECTED_STAGE_FACTS["inspect_library"],
        figures=(),
    )
    invalid_execution = runner.WorkflowExecution(
        state=execution.state,
        stage_results=(invalid_inspection, execution.stage_results[-1]),
        gpu=execution.gpu,
    )

    with pytest.raises(
        RuntimeError, match=r"^Workshop stage image count is invalid\.$"
    ):
        runner.run_lesson(
            "data-and-representation",
            paths=workshop_paths,
            workflow_executor=lambda stage: invalid_execution,
        )
    assert not (workshop_paths.output_root / "01-inspection").exists()
    assert not (workshop_paths.output_root / "02-fingerprints").exists()


def test_run_lesson_reuses_valid_stage_bytes_and_rejects_a_symlink_target(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    def execute(stage: str) -> runner.WorkflowExecution:
        return workflow_executions[stage]

    runner.run_lesson(
        "data-and-representation", paths=workshop_paths, workflow_executor=execute
    )
    stage_directory = workshop_paths.output_root / "01-inspection"
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in stage_directory.iterdir()
    }
    runner.run_lesson(
        "data-and-representation", paths=workshop_paths, workflow_executor=execute
    )
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in stage_directory.iterdir()
    }
    assert after == before

    backing = workshop_paths.output_root / "01-inspection-real"
    os.replace(stage_directory, backing)
    os.symlink(backing.name, stage_directory)
    with pytest.raises(RuntimeError):
        runner.run_lesson(
            "data-and-representation", paths=workshop_paths, workflow_executor=execute
        )
    assert stage_directory.is_symlink()
    assert backing.is_dir()


def test_run_lesson_rebuilds_only_safe_public_zip_members_and_preserves_old_zip_on_failure(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(stage: str) -> runner.WorkflowExecution:
        return workflow_executions[stage]

    runner.run_lesson(
        "data-and-representation", paths=workshop_paths, workflow_executor=execute
    )
    archive_path = workshop_paths.output_root / "results.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
    expected = {"README.md", "data/sample_molecules.csv", "data/PROVENANCE.md"}
    for stage_name in ("inspect_library", "generate_morgan_fingerprints"):
        directory = EXPECTED_STAGE_DIRECTORIES[stage_name]
        expected.update(
            f"{directory}/{name}"
            for name in (
                "README.md",
                "summary.json",
                *EXPECTED_STAGE_IMAGES[stage_name],
                *EXPECTED_STAGE_DATA[stage_name],
            )
        )
    assert set(members) == expected
    assert len(members) == len(set(members))
    assert all(
        not name.startswith("/") and ".." not in Path(name).parts for name in members
    )
    assert all(".acs-stage-" not in name and name != "results.zip" for name in members)

    previous_bytes = archive_path.read_bytes()
    real_replace = runner.os.replace

    def fail_archive_replace(source: object, destination: object) -> None:
        if Path(destination) == archive_path:
            raise OSError("replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_archive_replace)
    with pytest.raises(OSError, match=r"^replace failed$"):
        runner.run_lesson(
            "data-and-representation", paths=workshop_paths, workflow_executor=execute
        )
    assert archive_path.read_bytes() == previous_bytes


@pytest.mark.parametrize(
    "lesson,relative_artifact,mutation",
    (
        pytest.param(
            "relationships-and-groups",
            "03-similarity/top_similarity_pairs.csv",
            "csv",
            id="csv",
        ),
        pytest.param(
            "sampled-3d-geometry",
            "06-mmff94/workflow_evidence.json",
            "json",
            id="json",
        ),
        pytest.param(
            "sampled-3d-geometry",
            "06-mmff94/optimized_conformers.sdf",
            "sdf",
            id="sdf",
        ),
        pytest.param(
            "relationships-and-groups",
            "04-clusters/cluster_sizes.png",
            "png",
            id="png",
        ),
    ),
)
def test_run_lesson_rejects_nonempty_tampered_public_artifacts_without_replacing_zip(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    lesson: str,
    relative_artifact: str,
    mutation: str,
) -> None:
    def execute(stage: str) -> runner.WorkflowExecution:
        return workflow_executions[stage]

    runner.run_lesson(lesson, paths=workshop_paths, workflow_executor=execute)
    target = workshop_paths.output_root / relative_artifact
    archive_path = workshop_paths.output_root / "results.zip"
    previous_archive = archive_path.read_bytes()
    if mutation == "png":
        Image.new("RGB", (8, 8), color="red").save(target, format="PNG")
    else:
        target.write_text(f"tampered {mutation}\n", encoding="utf-8")
    tampered_bytes = target.read_bytes()

    with pytest.raises(RuntimeError, match=r"^Workshop stage artifacts are invalid\.$"):
        runner.run_lesson(lesson, paths=workshop_paths, workflow_executor=execute)
    assert target.read_bytes() == tampered_bytes
    assert archive_path.read_bytes() == previous_archive


def test_provenance_manifest_mutation_stops_lesson_before_executor(
    workshop_paths: runner.WorkshopPaths,
) -> None:
    provenance = workshop_paths.root / "data" / "PROVENANCE.md"
    provenance.write_text("changed\n", encoding="utf-8")
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="(?i)manifest"):
        runner.run_lesson(
            "data-and-representation",
            paths=workshop_paths,
            workflow_executor=lambda stage: calls.append(stage),
        )
    assert calls == []


def test_stage_summary_does_not_mutate_stage_result_facts(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    stage_name = "measure_tanimoto_similarity"
    facts = workflow_executions[stage_name].stage_results[-1].summary
    expected = dict(facts)
    runner.run_stage(
        stage_name,
        paths=workshop_paths,
        workflow_executor=lambda selected: workflow_executions[selected],
    )
    assert facts == expected


def test_cli_contract_constants_and_workshop_paths_are_fixed(tmp_path: Path) -> None:
    paths = runner.WorkshopPaths(tmp_path)
    assert runner.SCHEMA_VERSION == 1
    assert (
        runner.DATASET_SHA256
        == "7063a5d8eded837e3e648c44894fbe742d5863a0929bb5765b1c6330722fb034"
    )
    assert runner.STAGE_ORDER == (
        "inspect_library",
        "generate_morgan_fingerprints",
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
        "embed_representative_conformers",
        "optimize_conformers_mmff94",
    )
    assert runner.PROFILE == {
        "fingerprint_radius": 2,
        "fingerprint_size_bits": 1024,
        "cluster_cutoff": 0.40,
        "representative_policy": "largest_clusters_first",
        "representative_count": 6,
        "conformers_per_representative": 5,
        "etkdg_random_seed": 7,
        "mmff94_max_iterations": 500,
    }
    assert runner.MANIFEST_FILES == MANIFEST_FILES
    assert paths.dataset_path == tmp_path / "data" / "sample_molecules.csv"
    assert paths.output_root == tmp_path / "outputs" / "workshop"
    assert paths.state_root == tmp_path / ".acs-workshop-state"
    assert paths.manifest_path == paths.state_root / "manifest.json"
    assert paths.context_path == paths.state_root / "context.json"
    assert paths.history_path == paths.state_root / "history.json"


def test_cli_exposes_only_fixed_commands_without_path_options() -> None:
    parser = runner.build_parser()
    subcommands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert tuple(subcommands.choices) == (
        "run-stage",
        "run-lesson",
        "objective-start",
        "objective-step",
    )

    run_stage_parser = subcommands.choices["run-stage"]
    stage_action = next(
        action for action in run_stage_parser._actions if action.dest == "stage_name"
    )
    assert tuple(stage_action.choices) == runner.STAGE_ORDER
    for stage_name in runner.STAGE_ORDER:
        assert parser.parse_args(["run-stage", stage_name]).stage_name == stage_name

    objective_step = parser.parse_args(
        ["objective-step", "--state-id", "state-1", "--swap-id", "A->B"]
    )
    assert objective_step.state_id == "state-1"
    assert objective_step.swap_id == "A->B"
    with pytest.raises(ValueError, match=r"^Invalid workshop arguments\.$"):
        parser.parse_args(["objective-step", "--state-id", "state-1"])
    with pytest.raises(ValueError, match=r"^Invalid workshop arguments\.$"):
        parser.parse_args(["run-stage", "not-a-stage"])
    with pytest.raises(ValueError, match=r"^Invalid workshop arguments\.$"):
        parser.parse_args(["run-stage", runner.STAGE_ORDER[0], "extra-path"])

    help_text = "\n".join(
        [parser.format_help()]
        + [
            command_parser.format_help()
            for command_parser in subcommands.choices.values()
        ]
    ).lower()
    for forbidden in ("--dataset", "--output", "--retry", "--url", "--command"):
        assert forbidden not in help_text


def test_cli_help_fails_before_usage_when_manifest_is_missing(
    workshop_paths: runner.WorkshopPaths,
) -> None:
    workshop_paths.manifest_path.unlink()
    completed = subprocess.run(
        [sys.executable, str(workshop_paths.root / "acs_workshop_runner.py"), "--help"],
        cwd=workshop_paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert len(completed.stderr.splitlines()) == 1
    assert completed.stderr.startswith("Error:")
    assert "manifest" in completed.stderr.lower()
    assert "usage:" not in completed.stderr.lower()


def test_cli_main_verifies_default_manifest_before_building_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[runner.WorkshopPaths] = []

    def fail_manifest(paths: runner.WorkshopPaths) -> None:
        seen.append(paths)
        raise RuntimeError("Workshop integrity manifest is invalid.")

    monkeypatch.setattr(runner, "verify_manifest", fail_manifest)
    monkeypatch.setattr(
        runner,
        "build_parser",
        lambda: pytest.fail("parser must not be built before manifest verification"),
    )
    assert runner.main(["--help"]) == 2
    assert seen == [runner.DEFAULT_PATHS]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Workshop integrity manifest is invalid.\n"


def test_cli_main_preserves_help_after_manifest_verification(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[runner.WorkshopPaths] = []
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: seen.append(paths))

    with pytest.raises(SystemExit) as caught:
        runner.main(["--help"])

    assert caught.value.code == 0
    assert seen == [runner.DEFAULT_PATHS]
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:")
    assert captured.err == ""


def test_manifest_accepts_the_exact_fixed_file_set(
    workshop_paths: runner.WorkshopPaths,
) -> None:
    mode = os.lstat(workshop_paths.manifest_path).st_mode
    assert stat.S_ISREG(mode)
    assert stat.S_IMODE(mode) == 0o444
    assert runner.verify_manifest(workshop_paths) is None


def test_manifest_verification_precedes_science(
    workshop_paths: runner.WorkshopPaths,
) -> None:
    workshop_paths.manifest_path.unlink()
    _assert_manifest_rejected_before_executor(workshop_paths)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_root_key",
        "missing_root_key",
        "extra_file_key",
        "missing_file_key",
        "uppercase_hash",
        "short_hash",
        "wrong_schema_version",
        "boolean_schema_version",
    ),
)
def test_manifest_rejects_noncanonical_schema_before_executor(
    workshop_paths: runner.WorkshopPaths, mutation: str
) -> None:
    payload = _manifest_payload(workshop_paths)
    files = payload["files"]
    assert isinstance(files, dict)
    if mutation == "extra_root_key":
        payload["extra"] = None
    elif mutation == "missing_root_key":
        del payload["schema_version"]
    elif mutation == "extra_file_key":
        files["extra.py"] = "0" * 64
    elif mutation == "missing_file_key":
        del files[MANIFEST_FILES[0]]
    elif mutation == "uppercase_hash":
        files[MANIFEST_FILES[0]] = str(files[MANIFEST_FILES[0]]).upper()
    elif mutation == "short_hash":
        files[MANIFEST_FILES[0]] = "0" * 63
    elif mutation == "wrong_schema_version":
        payload["schema_version"] = 2
    elif mutation == "boolean_schema_version":
        payload["schema_version"] = True
    else:
        raise AssertionError(mutation)
    _replace_manifest(workshop_paths, payload)
    _assert_manifest_rejected_before_executor(workshop_paths)


def test_manifest_rejects_changed_file_bytes_before_executor(
    workshop_paths: runner.WorkshopPaths,
) -> None:
    marker = b"do-not-expose-these-file-bytes"
    target = workshop_paths.root / "objective_challenge.py"
    target.write_bytes(target.read_bytes() + marker)
    with pytest.raises(RuntimeError, match="(?i)manifest") as caught:
        runner.run_stage(
            "inspect_library",
            paths=workshop_paths,
            workflow_executor=lambda stage: pytest.fail(stage),
        )
    assert marker.decode() not in str(caught.value)


@pytest.mark.parametrize(
    "target_kind",
    ("manifest", "fixed_file", "fixed_file_ancestor", "manifest_ancestor"),
)
def test_manifest_rejects_symlinks_before_executor(
    workshop_paths: runner.WorkshopPaths, target_kind: str
) -> None:
    if target_kind == "manifest":
        target = workshop_paths.manifest_path
        backing = target.with_name("manifest-real.json")
        os.replace(target, backing)
        os.symlink(backing.name, target)
    elif target_kind == "fixed_file":
        target = workshop_paths.root / "objective_challenge.py"
        backing = workshop_paths.root / "objective_challenge-real.py"
        os.replace(target, backing)
        os.symlink(backing.name, target)
    elif target_kind == "fixed_file_ancestor":
        target = workshop_paths.root / "data"
        backing = workshop_paths.root / "data-real"
        os.replace(target, backing)
        os.symlink(backing.name, target)
    elif target_kind == "manifest_ancestor":
        target = workshop_paths.state_root
        backing = workshop_paths.root / ".acs-workshop-state-real"
        os.replace(target, backing)
        os.symlink(backing.name, target)
    else:
        raise AssertionError(target_kind)
    _assert_manifest_rejected_before_executor(workshop_paths)


def _record_workflow_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> list[
    tuple[
        str,
        runner.WorkflowState,
        tuple[object, ...],
        dict[str, object],
        runner.StageResult,
    ]
]:
    calls: list[
        tuple[
            str,
            runner.WorkflowState,
            tuple[object, ...],
            dict[str, object],
            runner.StageResult,
        ]
    ] = []

    def replacement(stage_name: str):
        def run(
            state: runner.WorkflowState,
            *args: object,
            **kwargs: object,
        ) -> runner.StageResult:
            result = runner.StageResult(
                stage=stage_name,
                display_label=stage_name,
                summary={},
            )
            calls.append((stage_name, state, args, kwargs, result))
            return result

        return run

    for stage_name in runner.STAGE_ORDER:
        monkeypatch.setattr(runner, stage_name, replacement(stage_name))
    return calls


@pytest.mark.parametrize(
    "stage_name,prefix_length",
    tuple(
        (stage_name, index + 1) for index, stage_name in enumerate(runner.STAGE_ORDER)
    ),
)
def test_workflow_prefix_uses_exact_order_and_fixed_values(
    workshop_paths: runner.WorkshopPaths,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    prefix_length: int,
) -> None:
    calls = _record_workflow_stages(monkeypatch)
    gpu = runner.GpuIdentity(
        name="NVIDIA L4",
        device="cuda:0",
        torch_version="2.7.1+cu128",
        nvmolkit_version="0.5.0",
    )
    gpu_calls: list[None] = []

    def gpu_identity() -> runner.GpuIdentity:
        gpu_calls.append(None)
        return gpu

    monkeypatch.setattr(runner, "_gpu_identity", gpu_identity)

    execution = runner.execute_workflow_prefix(stage_name, paths=workshop_paths)

    expected_calls = (
        (
            "inspect_library",
            (workshop_paths.dataset_path,),
            {"expected_rows": 256},
        ),
        (
            "generate_morgan_fingerprints",
            (),
            {"fingerprint_radius": 2, "fingerprint_size": 1024},
        ),
        ("measure_tanimoto_similarity", (), {}),
        (
            "discover_fused_butina_clusters",
            (),
            {"cluster_cutoff": 0.40},
        ),
        (
            "embed_representative_conformers",
            (),
            {
                "representative_count": 6,
                "representative_policy": (
                    runner.RepresentativePolicy.LARGEST_CLUSTERS_FIRST
                ),
                "conformers_per_representative": 5,
            },
        ),
        ("optimize_conformers_mmff94", (), {}),
    )
    observed_calls = tuple((name, args, kwargs) for name, _, args, kwargs, _ in calls)
    assert observed_calls == expected_calls[:prefix_length]
    assert all(state is execution.state for _, state, _, _, _ in calls)
    assert execution.stage_results == tuple(call[-1] for call in calls)
    assert gpu_calls == ([] if prefix_length == 1 else [None])
    assert execution.gpu is (None if prefix_length == 1 else gpu)


def test_workflow_prefix_rejects_unsupported_stage_before_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "inspect_library",
        lambda *args, **kwargs: pytest.fail((args, kwargs)),
    )
    with pytest.raises(ValueError, match=r"^Unsupported workshop stage\.$"):
        runner.execute_workflow_prefix("not-a-stage")


def test_inspection_does_not_require_cuda(
    workshop_paths: runner.WorkshopPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_gpu_identity",
        lambda: pytest.fail("inspection must remain CPU-capable"),
    )
    execution = runner.execute_workflow_prefix(
        "inspect_library",
        paths=workshop_paths,
    )
    assert tuple(result.stage for result in execution.stage_results) == (
        "inspect_library",
    )
    assert execution.gpu is None


def make_fake_torch(
    available: bool,
    device_count: int,
    device_name: str,
    *,
    torch_version: str = "2.7.1+cu128",
) -> tuple[ModuleType, list[str]]:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("is_available")
            return available

        @staticmethod
        def device_count() -> int:
            calls.append("device_count")
            return device_count

        @staticmethod
        def get_device_name(index: int) -> str:
            calls.append(f"get_device_name:{index}")
            return device_name

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = torch_version
    fake_torch.cuda = FakeCuda()
    return fake_torch, calls


@pytest.mark.parametrize(
    "available,device_count,device_name,expected_calls",
    (
        (False, 1, "NVIDIA L4", ["is_available"]),
        (True, 0, "", ["is_available", "device_count"]),
        (True, 2, "NVIDIA L4", ["is_available", "device_count"]),
        (
            True,
            1,
            "NVIDIA A100-SXM4-80GB",
            ["is_available", "device_count", "get_device_name:0"],
        ),
    ),
)
def test_gpu_stages_require_exactly_one_nvidia_l4(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    device_count: int,
    device_name: str,
    expected_calls: list[str],
) -> None:
    fake_torch, calls = make_fake_torch(available, device_count, device_name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(
        RuntimeError,
        match=r"^GPU stages require exactly one NVIDIA L4\.$",
    ):
        runner._gpu_identity()
    assert calls == expected_calls


def test_nvidia_l4_identity_includes_fixed_device_and_package_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch, calls = make_fake_torch(True, 1, "NVIDIA L4")
    package_calls: list[str] = []

    def package_version(package_name: str) -> str:
        package_calls.append(package_name)
        return "0.5.0"

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner.importlib.metadata, "version", package_version)

    assert runner._gpu_identity() == runner.GpuIdentity(
        name="NVIDIA L4",
        device="cuda:0",
        torch_version="2.7.1+cu128",
        nvmolkit_version="0.5.0",
    )
    assert calls == ["is_available", "device_count", "get_device_name:0"]
    assert package_calls == ["nvmolkit"]


def test_manifest_precedes_real_workflow_executor_and_selected_paths_are_passed(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def verify(paths: runner.WorkshopPaths) -> None:
        calls.append(("verify", paths))

    def execute(
        stage_name: str,
        *,
        paths: runner.WorkshopPaths,
    ) -> runner.WorkflowExecution:
        calls.append(("execute", (stage_name, paths)))
        return workflow_executions[stage_name]

    monkeypatch.setattr(runner, "verify_manifest", verify)
    monkeypatch.setattr(runner, "execute_workflow_prefix", execute)

    result = runner.run_stage("inspect_library", paths=workshop_paths)

    assert result["stage"] == "inspect_library"
    assert result["status"] == "complete"
    assert calls == [
        ("verify", workshop_paths),
        ("execute", ("inspect_library", workshop_paths)),
    ]


def test_manifest_precedes_injected_one_argument_workflow_executor(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def verify(paths: runner.WorkshopPaths) -> None:
        calls.append(("verify", paths))

    def injected(stage_name: str) -> runner.WorkflowExecution:
        calls.append(("injected", stage_name))
        return workflow_executions[stage_name]

    monkeypatch.setattr(runner, "verify_manifest", verify)
    monkeypatch.setattr(
        runner,
        "execute_workflow_prefix",
        lambda *args, **kwargs: pytest.fail((args, kwargs)),
    )

    result = runner.run_stage(
        "inspect_library",
        paths=workshop_paths,
        workflow_executor=injected,
    )

    assert result["stage"] == "inspect_library"
    assert result["status"] == "complete"
    assert calls == [
        ("verify", workshop_paths),
        ("injected", "inspect_library"),
    ]


def test_cli_main_emits_one_canonical_json_object_for_run_stage(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, object]] = []

    def verify(paths: runner.WorkshopPaths) -> None:
        calls.append(("verify", paths))

    def execute(
        stage_name: str,
        *,
        paths: runner.WorkshopPaths,
    ) -> runner.WorkflowExecution:
        calls.append(("execute", (stage_name, paths)))
        return workflow_executions[stage_name]

    monkeypatch.setattr(runner, "DEFAULT_PATHS", workshop_paths)
    monkeypatch.setattr(runner, "verify_manifest", verify)
    monkeypatch.setattr(runner, "execute_workflow_prefix", execute)

    assert runner.main(["run-stage", "inspect_library"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"
    assert payload["stage"] == "inspect_library"
    assert captured.out == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert captured.err == ""
    assert calls == [
        ("verify", workshop_paths),
        ("verify", workshop_paths),
        ("execute", ("inspect_library", workshop_paths)),
    ]


def test_cli_main_emits_one_safe_error_line_for_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)
    monkeypatch.setattr(
        runner,
        "run_stage",
        lambda stage_name, *, paths: (_ for _ in ()).throw(
            ValueError("Unsupported workshop stage.")
        ),
    )

    assert runner.main(["run-stage", "inspect_library"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Unsupported workshop stage.\n"


@pytest.mark.parametrize(
    "error",
    (
        pytest.param(
            RuntimeError(
                "unsafe /private/unsafe-token\nsecond line\twith hidden-token"
            ),
            id="runtime-error",
        ),
        pytest.param(
            ValueError("unsafe /private/unsafe-token\r\n\x1b[31mhidden-token"),
            id="value-error",
        ),
    ),
)
def test_cli_main_redacts_unapproved_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: RuntimeError | ValueError,
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)

    def fail_stage(
        stage_name: str, *, paths: runner.WorkshopPaths
    ) -> dict[str, object]:
        raise error

    monkeypatch.setattr(runner, "run_stage", fail_stage)

    assert runner.main(["run-stage", "inspect_library"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Workshop execution failed.\n"
    assert "/private/unsafe-token" not in captured.err
    assert "hidden-token" not in captured.err


def test_cli_main_redacts_unexpected_operational_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)

    def fail_stage(
        stage_name: str, *, paths: runner.WorkshopPaths
    ) -> dict[str, object]:
        raise KeyError("/private/unsafe-token")

    monkeypatch.setattr(runner, "run_stage", fail_stage)

    assert runner.main(["run-stage", "inspect_library"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Workshop execution failed.\n"


@pytest.mark.parametrize(
    "interruption",
    (
        pytest.param(KeyboardInterrupt(), id="keyboard-interrupt"),
        pytest.param(SystemExit(7), id="system-exit"),
    ),
)
def test_cli_main_does_not_catch_base_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interruption: BaseException,
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)

    def interrupt_stage(
        stage_name: str, *, paths: runner.WorkshopPaths
    ) -> dict[str, object]:
        raise interruption

    monkeypatch.setattr(runner, "run_stage", interrupt_stage)

    with pytest.raises(BaseException) as caught:
        runner.main(["run-stage", "inspect_library"])
    assert caught.value is interruption
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    (
        ["run-stage", "not-a-stage"],
        ["objective-step", "--state-id", "state-1", "--swap-id"],
        ["--unknown-option"],
    ),
    ids=("invalid-stage", "missing-value", "unknown-option"),
)
def test_cli_main_emits_one_safe_error_line_for_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)

    assert runner.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error_lines = captured.err.splitlines()
    assert len(error_lines) == 1
    assert error_lines[0].startswith("Error:")
    assert error_lines[0].strip() == error_lines[0]
    assert "usage:" not in captured.err.lower()


def _objective_execution(
    workflow_executions: dict[str, runner.WorkflowExecution],
    context=None,
) -> runner.WorkflowExecution:
    execution = workflow_executions["optimize_conformers_mmff94"]
    similarity = np.eye(256, dtype=float)
    source = target_achieved_context() if context is None else context
    similarity[:8, :8] = 1.0 - source.distance_matrix
    execution.state.similarity = _FakeGpuResult(similarity)
    return execution


def test_third_lesson_initializes_private_objective_and_start_is_pending(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )

    assert sorted(path.name for path in workshop_paths.state_root.iterdir()) == [
        "context.json",
        "history.json",
        "manifest.json",
    ]
    for path in (workshop_paths.context_path, workshop_paths.history_path):
        assert stat.S_ISREG(os.lstat(path).st_mode)
        assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    context = json.loads(workshop_paths.context_path.read_text(encoding="utf-8"))
    assert set(context) == {
        "schema_version",
        "dataset_sha256",
        "profile",
        "candidates",
        "baseline_ids",
        "baseline_score",
        "benchmark_score",
        "target_score",
        "distance_matrix",
        "stage_results_zip_sha256",
    }
    assert len(context["candidates"]) == 8
    assert np.asarray(context["distance_matrix"]).shape == (8, 8)
    reconstructed = runner._context_from_payload(context)
    assert runner.certify_argmax_reachability(reconstructed)

    monkeypatch.setattr(runner, "execute_workflow_prefix", pytest.fail)
    pending = runner.objective_start(paths=workshop_paths)
    assert set(pending) == {
        "schema_version",
        "status",
        "terminal",
        "attempt_count",
        "attempt_limit",
        "state_id",
        "current",
        "target_score",
        "actions",
        "achieved",
        "termination_reason",
        "image_paths",
        "artifact_directory",
        "results_zip_path",
        "artifact_relative_zip_path",
    }
    assert pending["status"] == "pending"
    assert pending["attempt_count"] == 0
    assert 1 <= len(pending["actions"]) <= 3
    with zipfile.ZipFile(workshop_paths.output_root / "results.zip") as archive:
        assert all(".acs-workshop-state" not in name for name in archive.namelist())


def test_objective_step_accepts_maximum_publishes_and_retries_exactly(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    terminal = runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    history_bytes = workshop_paths.history_path.read_bytes()
    assert terminal["status"] == "complete"
    assert terminal["attempt_count"] == 1
    assert terminal["termination_reason"] == "target_achieved"
    assert (
        runner.objective_step(
            pending["state_id"], selected["swap_id"], paths=workshop_paths
        )
        == terminal
    )
    assert workshop_paths.history_path.read_bytes() == history_bytes

    objective_directory = workshop_paths.output_root / "07-objective"
    assert sorted(path.name for path in objective_directory.iterdir()) == [
        "README.md",
        "final_panel.png",
        "final_similarity_heatmap.png",
        "objective_evidence.json",
        "objective_summary.json",
        "score_trajectory.png",
    ]
    for image_name in terminal["image_paths"]:
        with Image.open(image_name) as image:
            image.verify()
    summary = json.loads(
        (objective_directory / "objective_summary.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (objective_directory / "objective_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["evidence"][0]["key"] == "O01"
    assert evidence["evidence"][0]["payload"] == {
        key: value for key, value in summary.items() if key != "schema_version"
    }
    with zipfile.ZipFile(workshop_paths.output_root / "results.zip") as archive:
        members = set(archive.namelist())
    assert "07-objective/objective_summary.json" in members
    assert all(".acs-workshop-state" not in member for member in members)
    terminal_state_bytes = workshop_paths.history_path.read_bytes()
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    assert workshop_paths.history_path.read_bytes() == terminal_state_bytes


def test_objective_step_rejects_nonmaximum_without_mutation(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_ranked_swaps()
    )
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    lower = min(pending["actions"], key=lambda action: action["predicted_score"])
    assert lower["predicted_score"] < max(
        action["predicted_score"] for action in pending["actions"]
    )
    before = workshop_paths.history_path.read_bytes()
    with pytest.raises(ValueError, match="accepted exact menu action"):
        runner.objective_step(
            pending["state_id"], lower["swap_id"], paths=workshop_paths
        )
    assert workshop_paths.history_path.read_bytes() == before


def test_third_lesson_preserves_progress_and_rejects_incomplete_state(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    run = lambda: runner.run_lesson(  # noqa: E731
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    run()
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    before = workshop_paths.history_path.read_bytes()
    run()
    assert workshop_paths.history_path.read_bytes() == before

    workshop_paths.context_path.unlink()
    with pytest.raises(RuntimeError, match="objective state"):
        run()
    assert workshop_paths.history_path.read_bytes() == before


def test_tied_maxima_are_accepted_and_attempt_limit_is_exact(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    result = runner.objective_start(paths=workshop_paths)
    saw_tie = False
    while not result["terminal"]:
        maximum = max(action["predicted_score"] for action in result["actions"])
        tied = [
            action
            for action in result["actions"]
            if action["predicted_score"] == maximum
        ]
        saw_tie = saw_tie or len(tied) > 1
        result = runner.objective_step(
            result["state_id"], tied[-1]["swap_id"], paths=workshop_paths
        )
    assert result["terminal"] is True
    assert result["termination_reason"] == "target_achieved"
    assert saw_tie
    before = workshop_paths.history_path.read_bytes()
    with pytest.raises(ValueError):
        runner.objective_step("state-invented", "mol-0->mol-7", paths=workshop_paths)
    assert workshop_paths.history_path.read_bytes() == before

    context = controlled_context_with_three_misses()
    attempts = []
    current = runner.measure_panel(context, context.baseline_ids)
    for number in range(1, 4):
        menu = runner.build_action_menu(context, current, number - 1)
        attempt = runner.evaluate_selected_swap(
            context, menu, runner.accepted_maxima(menu)[0], number
        )
        attempts.append(attempt)
        current = attempt.measurement
    _, menu, run = runner._derive_objective(context, tuple(attempts))
    assert menu is None
    assert run.termination_reason.value == "attempt_limit_reached"


def test_terminal_publication_interruption_recovers_without_another_attempt(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    real_publish = runner._publish_objective
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated interruption")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(runner, "_publish_objective", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        runner.objective_step(
            pending["state_id"], selected["swap_id"], paths=workshop_paths
        )
    stored = json.loads(workshop_paths.history_path.read_text(encoding="utf-8"))
    assert stored["accepted_attempt_count"] == 1
    recovered = runner.objective_start(paths=workshop_paths)
    assert recovered["status"] == "complete"
    assert recovered["attempt_count"] == 1
    assert calls == 2


def test_private_json_is_indented_and_rejects_forged_cached_result(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    context_bytes = workshop_paths.context_path.read_bytes()
    assert context_bytes.startswith(b'{\n  "')
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    state = json.loads(workshop_paths.history_path.read_text(encoding="utf-8"))
    state["last_result"]["status"] = "forged"
    workshop_paths.history_path.write_bytes(runner._private_json_bytes(state))
    with pytest.raises(RuntimeError, match="objective state"):
        runner.objective_start(paths=workshop_paths)


def test_terminal_zip_extends_bound_archive_not_mutated_stage_directory(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    archive_path = workshop_paths.output_root / "results.zip"
    member_name = "06-mmff94/conformer_energies.png"
    with zipfile.ZipFile(archive_path) as archive:
        bound_member = archive.read(member_name)
    stage_image = workshop_paths.output_root / member_name
    Image.new("RGB", (11, 7), color="red").save(stage_image, format="PNG")
    mutated_member = stage_image.read_bytes()
    assert mutated_member != bound_member

    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read(member_name) == bound_member
        assert archive.read(member_name) != mutated_member


def test_forged_terminal_zip_matching_mutated_stage_directory_fails_closed(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    tmp_path: Path,
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    stage_image = workshop_paths.output_root / "06-mmff94/conformer_energies.png"
    Image.new("RGB", (13, 9), color="blue").save(stage_image, format="PNG")
    forged = tmp_path / "forged-results.zip"
    archive_path = workshop_paths.output_root / "results.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(forged, "w") as archive:
        for name, contents in members:
            if name == "06-mmff94/conformer_energies.png":
                contents = stage_image.read_bytes()
            runner._zip_member(archive, name, contents)
    os.replace(forged, workshop_paths.output_root / "results.zip")
    before = (workshop_paths.output_root / "results.zip").read_bytes()

    with pytest.raises(RuntimeError, match="objective state"):
        runner.objective_start(paths=workshop_paths)
    assert (workshop_paths.output_root / "results.zip").read_bytes() == before


@pytest.mark.parametrize("command", ("start", "step"))
def test_objective_commands_reject_nonprivate_state_root_without_mutation(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    command: str,
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    before = workshop_paths.history_path.read_bytes()
    workshop_paths.state_root.chmod(0o755)

    with pytest.raises(RuntimeError, match="objective state"):
        if command == "start":
            runner.objective_start(paths=workshop_paths)
        else:
            runner.objective_step(
                pending["state_id"], selected["swap_id"], paths=workshop_paths
            )
    assert workshop_paths.history_path.read_bytes() == before


def test_new_objective_action_is_evaluated_exactly_once(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    real_evaluator = runner.evaluate_selected_swap
    calls = 0

    def counted_evaluator(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_evaluator(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_selected_swap", counted_evaluator)
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    assert calls == 1


def test_objective_history_rejects_skipped_attempt_number_without_mutation(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )

    context_bytes = workshop_paths.context_path.read_bytes()
    context = runner._context_from_payload(json.loads(context_bytes))
    current = runner.measure_panel(context, context.baseline_ids)
    skipped_menu = runner.build_action_menu(context, current, 1)
    skipped_attempt = runner.evaluate_selected_swap(
        context,
        skipped_menu,
        runner.accepted_maxima(skipped_menu)[0],
        2,
    )
    next_current, next_menu, next_run = runner._resolve_objective_state(
        context,
        (skipped_attempt,),
        skipped_attempt.measurement,
        validate_terminal=True,
    )
    forged_state = runner._state_payload(
        hashlib.sha256(context_bytes).hexdigest(),
        next_current,
        next_menu,
        next_run,
        (skipped_attempt,),
    )
    workshop_paths.history_path.write_bytes(runner._private_json_bytes(forged_state))
    history_before = workshop_paths.history_path.read_bytes()
    archive_before = (workshop_paths.output_root / "results.zip").read_bytes()

    with pytest.raises(RuntimeError, match="objective state"):
        runner.objective_start(paths=workshop_paths)
    assert workshop_paths.history_path.read_bytes() == history_before
    assert (workshop_paths.output_root / "results.zip").read_bytes() == archive_before


def test_third_lesson_invalid_state_preserves_prior_terminal_archive(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(workflow_executions)
    run = lambda: runner.run_lesson(  # noqa: E731
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    run()
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    archive_path = workshop_paths.output_root / "results.zip"
    archive_before = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        assert "07-objective/objective_summary.json" in archive.namelist()

    state = json.loads(workshop_paths.history_path.read_text(encoding="utf-8"))
    state["accepted_attempt_count"] = 0
    workshop_paths.history_path.write_bytes(runner._private_json_bytes(state))

    with pytest.raises(RuntimeError, match="objective state"):
        run()
    assert archive_path.read_bytes() == archive_before
    with zipfile.ZipFile(archive_path) as archive:
        assert "07-objective/objective_summary.json" in archive.namelist()


def test_pending_duplicate_retry_rejects_tampered_bound_archive_without_mutation(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    pending = runner.objective_start(paths=workshop_paths)
    selected = max(pending["actions"], key=lambda action: action["predicted_score"])
    next_pending = runner.objective_step(
        pending["state_id"], selected["swap_id"], paths=workshop_paths
    )
    assert next_pending["status"] == "pending"
    history_before = workshop_paths.history_path.read_bytes()
    archive_path = workshop_paths.output_root / "results.zip"
    archive_path.write_bytes(b"tampered archive")
    archive_before = archive_path.read_bytes()

    with pytest.raises(RuntimeError, match="objective state"):
        runner.objective_step(
            pending["state_id"], selected["swap_id"], paths=workshop_paths
        )
    assert workshop_paths.history_path.read_bytes() == history_before
    assert archive_path.read_bytes() == archive_before


@pytest.mark.parametrize("limit_kind", ("member", "aggregate"))
def test_results_archive_limits_fail_before_reading_expanded_members(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limit_kind: str,
) -> None:
    assert runner._RESULTS_ARCHIVE_MAX_MEMBER_BYTES == 8 * 1024 * 1024
    assert runner._RESULTS_ARCHIVE_MAX_EXPANDED_BYTES == 32 * 1024 * 1024
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    source_path = workshop_paths.output_root / "results.zip"
    with zipfile.ZipFile(source_path) as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]

    malicious_path = tmp_path / f"{limit_kind}.zip"
    oversized = b"x" * (runner._RESULTS_ARCHIVE_MAX_MEMBER_BYTES + 1)
    aggregate = b"y" * (7 * 1024 * 1024)
    with zipfile.ZipFile(malicious_path, "w") as archive:
        for index, (name, contents) in enumerate(members):
            if limit_kind == "member" and index == 0:
                contents = oversized
            elif limit_kind == "aggregate" and index < 5:
                contents = aggregate
            runner._zip_member(archive, name, contents)

    monkeypatch.setattr(
        runner.zipfile.ZipFile,
        "read",
        lambda *args, **kwargs: pytest.fail(
            "archive content was read before declared-size limits were checked"
        ),
    )
    with pytest.raises(RuntimeError, match="objective state"):
        runner._validated_results_archive(malicious_path)
