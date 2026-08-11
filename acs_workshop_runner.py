from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

from chemistry_workflow import (
    RepresentativePolicy,
    StageResult,
    WorkflowState,
    discover_fused_butina_clusters,
    embed_representative_conformers,
    generate_morgan_fingerprints,
    inspect_library,
    measure_tanimoto_similarity,
    optimize_conformers_mmff94,
)

SCHEMA_VERSION: Final = 1
DATASET_SHA256: Final = (
    "7063a5d8eded837e3e648c44894fbe742d5863a0929bb5765b1c6330722fb034"
)
STAGE_ORDER: Final = (
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
)
PROFILE: Final[dict[str, Any]] = {
    "fingerprint_radius": 2,
    "fingerprint_size_bits": 1024,
    "cluster_cutoff": 0.40,
    "representative_policy": "largest_clusters_first",
    "representative_count": 6,
    "conformers_per_representative": 5,
    "etkdg_random_seed": 7,
    "mmff94_max_iterations": 500,
}
MANIFEST_FILES: Final = (
    "TOOLS.md",
    "acs_workshop_runner.py",
    "chemistry_workflow.py",
    "data/sample_molecules.csv",
    "objective_challenge.py",
)

__all__ = (
    "DATASET_SHA256",
    "DEFAULT_PATHS",
    "GpuIdentity",
    "MANIFEST_FILES",
    "PROFILE",
    "RepresentativePolicy",
    "SCHEMA_VERSION",
    "STAGE_ORDER",
    "StageResult",
    "WorkflowExecution",
    "WorkflowExecutor",
    "WorkflowState",
    "WorkshopPaths",
    "build_parser",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "execute_workflow_prefix",
    "generate_morgan_fingerprints",
    "inspect_library",
    "main",
    "measure_tanimoto_similarity",
    "optimize_conformers_mmff94",
    "run_stage",
    "verify_manifest",
)


@dataclass(frozen=True)
class WorkshopPaths:
    root: Path

    @property
    def dataset_path(self) -> Path:
        return self.root / "data" / "sample_molecules.csv"

    @property
    def output_root(self) -> Path:
        return self.root / "outputs" / "workshop"

    @property
    def state_root(self) -> Path:
        return self.root / ".acs-workshop-state"

    @property
    def manifest_path(self) -> Path:
        return self.state_root / "manifest.json"

    @property
    def context_path(self) -> Path:
        return self.state_root / "context.json"

    @property
    def history_path(self) -> Path:
        return self.state_root / "history.json"


@dataclass(frozen=True)
class GpuIdentity:
    name: str
    device: str
    torch_version: str
    nvmolkit_version: str


@dataclass(frozen=True)
class WorkflowExecution:
    state: WorkflowState
    stage_results: tuple[StageResult, ...]
    gpu: GpuIdentity | None


WorkflowExecutor = Callable[[str], WorkflowExecution]
DEFAULT_PATHS: Final = WorkshopPaths(Path(__file__).resolve().parent)

_MANIFEST_ERROR: Final = "Workshop integrity manifest is invalid."
_GENERIC_ERROR: Final = "Workshop execution failed."
_SAFE_ERROR_MESSAGES: Final = frozenset(
    {
        _MANIFEST_ERROR,
        "GPU stages require exactly one NVIDIA L4.",
        "Invalid workshop arguments.",
        "Unsupported workshop stage.",
        "Workshop objective execution is not implemented.",
    }
)
_LOWER_HEXADECIMAL: Final = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: Any) -> bytes:
    """Return one canonical UTF-8 JSON record."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _invalid_manifest() -> RuntimeError:
    return RuntimeError(_MANIFEST_ERROR)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _fixed_root(paths: WorkshopPaths) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        root_mode = os.lstat(root).st_mode
    except OSError as error:
        raise _invalid_manifest() from error
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise _invalid_manifest()
    return root


def _regular_file_below_root(
    paths: WorkshopPaths,
    target: Path,
    fixed_root: Path,
) -> Path:
    lexical_root = Path(os.path.abspath(os.fspath(paths.root)))
    lexical_target = Path(os.path.abspath(os.fspath(target)))
    try:
        relative_target = lexical_target.relative_to(lexical_root)
    except ValueError as error:
        raise _invalid_manifest() from error
    if not relative_target.parts:
        raise _invalid_manifest()

    current = fixed_root
    try:
        for index, component in enumerate(relative_target.parts):
            current = current / component
            mode = os.lstat(current).st_mode
            is_target = index == len(relative_target.parts) - 1
            if stat.S_ISLNK(mode):
                raise _invalid_manifest()
            if is_target:
                if not stat.S_ISREG(mode):
                    raise _invalid_manifest()
            elif not stat.S_ISDIR(mode):
                raise _invalid_manifest()
        current.resolve(strict=True).relative_to(fixed_root)
    except (OSError, ValueError) as error:
        raise _invalid_manifest() from error
    return current


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise _invalid_manifest() from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _invalid_manifest()
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            return source.read()
    except OSError as error:
        raise _invalid_manifest() from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_manifest(paths: WorkshopPaths) -> None:
    """Verify the fixed workshop source files before any execution."""
    try:
        fixed_root = _fixed_root(paths)
        manifest_path = _regular_file_below_root(paths, paths.manifest_path, fixed_root)
        manifest_bytes = _read_regular_file(manifest_path)
        payload = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
        if _canonical_json_bytes(payload) != manifest_bytes:
            raise _invalid_manifest()
        if type(payload) is not dict or set(payload) != {"schema_version", "files"}:
            raise _invalid_manifest()
        if type(payload["schema_version"]) is not int:
            raise _invalid_manifest()
        if payload["schema_version"] != SCHEMA_VERSION:
            raise _invalid_manifest()

        files = payload["files"]
        if type(files) is not dict or set(files) != set(MANIFEST_FILES):
            raise _invalid_manifest()
        for name in MANIFEST_FILES:
            expected_hash = files[name]
            if (
                type(expected_hash) is not str
                or len(expected_hash) != 64
                or any(
                    character not in _LOWER_HEXADECIMAL for character in expected_hash
                )
            ):
                raise _invalid_manifest()
            target = _regular_file_below_root(paths, paths.root / name, fixed_root)
            current_hash = hashlib.sha256(_read_regular_file(target)).hexdigest()
            if current_hash != expected_hash:
                raise _invalid_manifest()
        if files["data/sample_molecules.csv"] != DATASET_SHA256:
            raise _invalid_manifest()
    except RuntimeError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise _invalid_manifest() from error


class _WorkshopArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        raise ValueError("Invalid workshop arguments.")


def _safe_error_message(error: Exception) -> str:
    message = str(error)
    return message if message in _SAFE_ERROR_MESSAGES else _GENERIC_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = _WorkshopArgumentParser(
        description="Run the fixed ACS chemistry workshop."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_parser = subparsers.add_parser("run-stage")
    stage_parser.add_argument("stage_name", choices=STAGE_ORDER)

    subparsers.add_parser("objective-start")

    objective_step_parser = subparsers.add_parser("objective-step")
    objective_step_parser.add_argument("--state-id", required=True)
    objective_step_parser.add_argument("--swap-id", required=True)
    return parser


def _gpu_identity() -> GpuIdentity:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU stages require exactly one NVIDIA L4.")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA L4":
        raise RuntimeError("GPU stages require exactly one NVIDIA L4.")
    return GpuIdentity(
        name=name,
        device="cuda:0",
        torch_version=str(torch.__version__),
        nvmolkit_version=importlib.metadata.version("nvmolkit"),
    )


def execute_workflow_prefix(
    stage_name: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
) -> WorkflowExecution:
    if stage_name not in STAGE_ORDER:
        raise ValueError("Unsupported workshop stage.")
    stage_index = STAGE_ORDER.index(stage_name)
    state = WorkflowState()
    results: list[StageResult] = [
        inspect_library(state, paths.dataset_path, expected_rows=256)
    ]
    gpu: GpuIdentity | None = None
    if stage_index >= 1:
        gpu = _gpu_identity()
        results.append(
            generate_morgan_fingerprints(
                state,
                fingerprint_radius=2,
                fingerprint_size=1024,
            )
        )
    if stage_index >= 2:
        results.append(measure_tanimoto_similarity(state))
    if stage_index >= 3:
        results.append(discover_fused_butina_clusters(state, cluster_cutoff=0.40))
    if stage_index >= 4:
        results.append(
            embed_representative_conformers(
                state,
                representative_count=6,
                representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
                conformers_per_representative=5,
            )
        )
    if stage_index >= 5:
        results.append(optimize_conformers_mmff94(state))
    return WorkflowExecution(state, tuple(results), gpu)


def run_stage(
    stage_name: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
    workflow_executor: WorkflowExecutor | None = None,
) -> dict[str, Any]:
    verify_manifest(paths)
    if workflow_executor is None:
        execute_workflow_prefix(stage_name, paths=paths)
    else:
        workflow_executor(stage_name)
    return {"stage": stage_name}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        verify_manifest(DEFAULT_PATHS)
        arguments = build_parser().parse_args(argv)
        if arguments.command == "run-stage":
            result = run_stage(arguments.stage_name)
        elif arguments.command == "objective-start":
            raise RuntimeError("Workshop objective execution is not implemented.")
        else:
            raise RuntimeError("Workshop objective execution is not implemented.")
        print(_canonical_json_bytes(result).decode("utf-8"), end="")
        return 0
    except Exception as error:
        print(f"Error: {_safe_error_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
