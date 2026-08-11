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

import acs_workshop_runner as runner


MANIFEST_FILES = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
    "objective_challenge.py",
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
        return runner.WorkflowExecution(runner.WorkflowState(), (), None)

    monkeypatch.setattr(runner, "verify_manifest", verify)
    monkeypatch.setattr(runner, "execute_workflow_prefix", execute)

    result = runner.run_stage("inspect_library", paths=workshop_paths)

    assert result == {"stage": "inspect_library"}
    assert calls == [
        ("verify", workshop_paths),
        ("execute", ("inspect_library", workshop_paths)),
    ]


def test_manifest_precedes_injected_one_argument_workflow_executor(
    workshop_paths: runner.WorkshopPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def verify(paths: runner.WorkshopPaths) -> None:
        calls.append(("verify", paths))

    def injected(stage_name: str) -> runner.WorkflowExecution:
        calls.append(("injected", stage_name))
        return runner.WorkflowExecution(runner.WorkflowState(), (), None)

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

    assert result == {"stage": "inspect_library"}
    assert calls == [
        ("verify", workshop_paths),
        ("injected", "inspect_library"),
    ]


def test_cli_main_emits_one_canonical_json_object_for_run_stage(
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
        return runner.WorkflowExecution(runner.WorkflowState(), (), None)

    monkeypatch.setattr(runner, "verify_manifest", verify)
    monkeypatch.setattr(runner, "execute_workflow_prefix", execute)

    assert runner.main(["run-stage", "inspect_library"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"stage":"inspect_library"}\n'
    assert captured.err == ""
    assert calls == [
        ("verify", runner.DEFAULT_PATHS),
        ("verify", runner.DEFAULT_PATHS),
        ("execute", ("inspect_library", runner.DEFAULT_PATHS)),
    ]


def test_cli_main_emits_one_safe_error_line_for_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, "verify_manifest", lambda paths: None)
    monkeypatch.setattr(
        runner,
        "run_stage",
        lambda stage_name: (_ for _ in ()).throw(
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

    def fail_stage(stage_name: str) -> dict[str, object]:
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

    def fail_stage(stage_name: str) -> dict[str, object]:
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

    def interrupt_stage(stage_name: str) -> dict[str, object]:
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
