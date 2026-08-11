import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from matplotlib.figure import Figure
from PIL import Image

import acs_workshop_runner as runner


MANIFEST_FILES = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
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
        "objective_challenge.py": source_root / "objective_challenge.py",
    }
    for name, source in sources.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return write_manifest(root)


@pytest.fixture
def workflow_executions() -> dict[str, runner.WorkflowExecution]:
    state = runner.WorkflowState()
    state.records = [
        {"id": f"CHEMBL{index}", "smiles": "C", "source_row": index}
        for index in range(256)
    ]
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
        ("README.md", "summary.json", *EXPECTED_STAGE_IMAGES[payload["stage"]])
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


@pytest.mark.parametrize(
    "argv",
    (
        ["objective-start"],
        ["objective-step", "--state-id", "state-1", "--swap-id", "A->B"],
    ),
)
def test_cli_objective_commands_return_one_not_implemented_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)

    assert runner.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Workshop objective execution is not implemented.\n"
