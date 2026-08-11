from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import pandas as pd
import numpy as np
from rdkit import Chem

from chemistry_workflow import (
    RepresentativePolicy,
    StageResult,
    WorkflowState,
    build_workflow_report,
    discover_fused_butina_clusters,
    embed_representative_conformers,
    generate_morgan_fingerprints,
    inspect_library,
    measure_tanimoto_similarity,
    optimize_conformers_mmff94,
    validated_similarity_matrix,
)
from objective_challenge import (
    MAX_ATTEMPTS,
    ObjectiveActionMenu,
    ObjectiveAttempt,
    ObjectiveCandidate,
    ObjectiveContext,
    ObjectiveRun,
    ObjectiveSwap,
    PanelMeasurement,
    TerminationReason,
    accepted_maxima,
    attainable_benchmark,
    baseline_terminal_run,
    build_action_menu,
    build_objective_context,
    build_objective_evidence,
    certify_argmax_reachability,
    evaluate_selected_swap,
    finalize_no_legal_swap,
    measure_panel,
    objective_figures,
    resolve_menu_action,
    terminal_objective_run,
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
STAGE_DIRECTORIES: Final = {
    "inspect_library": "01-inspection",
    "generate_morgan_fingerprints": "02-fingerprints",
    "measure_tanimoto_similarity": "03-similarity",
    "discover_fused_butina_clusters": "04-clusters",
    "embed_representative_conformers": "05-conformers",
    "optimize_conformers_mmff94": "06-mmff94",
}
STAGE_DATA_NAMES: Final = {
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
    "data/PROVENANCE.md",
    "objective_challenge.py",
)
LESSON_TERMINAL_STAGES: Final = {
    "data-and-representation": "generate_morgan_fingerprints",
    "relationships-and-groups": "discover_fused_butina_clusters",
    "sampled-3d-geometry": "optimize_conformers_mmff94",
}
LESSON_STAGES: Final = {
    "data-and-representation": (
        "inspect_library",
        "generate_morgan_fingerprints",
    ),
    "relationships-and-groups": (
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
    ),
    "sampled-3d-geometry": (
        "embed_representative_conformers",
        "optimize_conformers_mmff94",
    ),
}

__all__ = (
    "DATASET_SHA256",
    "DEFAULT_PATHS",
    "GpuIdentity",
    "MANIFEST_FILES",
    "LESSON_TERMINAL_STAGES",
    "PROFILE",
    "RepresentativePolicy",
    "SCHEMA_VERSION",
    "STAGE_DIRECTORIES",
    "STAGE_DATA_NAMES",
    "STAGE_ORDER",
    "STAGE_SPECS",
    "StageResult",
    "StageSpec",
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
    "objective_start",
    "objective_step",
    "optimize_conformers_mmff94",
    "run_stage",
    "run_lesson",
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
class StageSpec:
    directory: str
    question: str
    method: str
    limit: str
    image_names: tuple[str, ...]


STAGE_SPECS: Final = {
    "inspect_library": StageSpec(
        directory=STAGE_DIRECTORIES["inspect_library"],
        question="What is in the fixed molecule library?",
        method="RDKit input validation",
        limit="validation does not establish activity or suitability",
        image_names=("library_preview.png",),
    ),
    "generate_morgan_fingerprints": StageSpec(
        directory=STAGE_DIRECTORIES["generate_morgan_fingerprints"],
        question="What do the GPU Morgan fingerprints show?",
        method="nvMolKit MorganFingerprintGenerator",
        limit="fingerprints are structural descriptors, not biological evidence",
        image_names=("fingerprint_density.png",),
    ),
    "measure_tanimoto_similarity": StageSpec(
        directory=STAGE_DIRECTORIES["measure_tanimoto_similarity"],
        question="Which molecules are most similar in this fingerprint space?",
        method="nvMolKit crossTanimotoSimilarity",
        limit="similarity does not establish activity, binding, efficacy, or safety",
        image_names=("similarity_heatmap.png",),
    ),
    "discover_fused_butina_clusters": StageSpec(
        directory=STAGE_DIRECTORIES["discover_fused_butina_clusters"],
        question="How does fused Butina partition the library?",
        method="nvMolKit fused_butina with RDKit MMFF94 eligibility",
        limit="clusters depend on this fingerprint and cutoff",
        image_names=("cluster_sizes.png",),
    ),
    "embed_representative_conformers": StageSpec(
        directory=STAGE_DIRECTORIES["embed_representative_conformers"],
        question=("Did ETKDGv3 generate the requested representative conformers?"),
        method="nvMolKit EmbedMolecules",
        limit="sampled conformers are not experimental structures",
        image_names=("embedding_counts.png",),
    ),
    "optimize_conformers_mmff94": StageSpec(
        directory=STAGE_DIRECTORIES["optimize_conformers_mmff94"],
        question="Which sampled conformers converged under MMFF94?",
        method="nvMolKit MMFFOptimizeMoleculesConfs",
        limit=(
            "MMFF94 compares sampled force-field geometries within each molecule only"
        ),
        image_names=("conformer_energies.png", "optimized_structures.png"),
    ),
}


@dataclass(frozen=True)
class WorkflowExecution:
    state: WorkflowState
    stage_results: tuple[StageResult, ...]
    gpu: GpuIdentity | None


@dataclass(frozen=True)
class _SdfExportRecord:
    record_id: str
    molecule: Any
    conformer_index: int
    properties: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _StageDataExports:
    csv_files: tuple[tuple[str, Any], ...] = ()
    sdf_file: tuple[str, tuple[_SdfExportRecord, ...]] | None = None
    json_files: tuple[tuple[str, dict[str, Any]], ...] = ()


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


def _private_json_bytes(value: Any) -> bytes:
    return _formatted_json(value).encode("utf-8")


def _exact_json_equal(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if type(first) is dict:
        return set(first) == set(second) and all(
            _exact_json_equal(first[key], second[key]) for key in first
        )
    if type(first) is list:
        return len(first) == len(second) and all(
            _exact_json_equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
    return first == second


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

    lesson_parser = subparsers.add_parser("run-lesson")
    lesson_parser.add_argument("lesson", choices=tuple(LESSON_TERMINAL_STAGES))

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


def _save_matplotlib_figure(figure: Any, path: Path) -> None:
    from matplotlib.figure import Figure

    if type(figure) is not Figure:
        raise TypeError("Expected an exact Matplotlib Figure.")
    figure.savefig(
        path,
        format="png",
        dpi=120,
        metadata={"Software": "ACS workshop runner"},
    )


def _save_pil_image(image: Any, path: Path) -> None:
    from PIL.Image import Image

    if not isinstance(image, Image):
        raise TypeError("Expected a PIL image.")
    image.save(path, format="PNG", optimize=False)


def _stage_result(
    stage_name: str,
    execution: WorkflowExecution,
) -> StageResult:
    if not execution.stage_results:
        raise RuntimeError("Workshop stage result is invalid.")
    result = execution.stage_results[-1]
    if result.stage != stage_name:
        raise RuntimeError("Workshop stage result is invalid.")
    return result


def _validate_stage_images(
    stage_name: str,
    result: StageResult,
    stage_spec: StageSpec,
) -> None:
    if len(result.figures) != len(stage_spec.image_names):
        raise RuntimeError("Workshop stage image count is invalid.")
    if stage_name == "inspect_library":
        from PIL.Image import Image

        if any(not isinstance(image, Image) for image in result.figures):
            raise TypeError("Expected a PIL image.")
        return

    from matplotlib.figure import Figure

    if any(type(figure) is not Figure for figure in result.figures):
        raise TypeError("Expected an exact Matplotlib Figure.")


def _fact_int(facts: dict[str, Any], key: str) -> int:
    value = facts.get(key)
    if type(value) is not int:
        raise RuntimeError("Workshop stage facts are invalid.")
    return value


def _fact_number(facts: dict[str, Any], key: str) -> float:
    value = facts.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Workshop stage facts are invalid.")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Workshop stage facts are invalid.")
    return number


def _fact_list(facts: dict[str, Any], key: str) -> list[Any]:
    value = facts.get(key)
    if type(value) is not list:
        raise RuntimeError("Workshop stage facts are invalid.")
    return value


def _fact_dict(facts: dict[str, Any], key: str) -> dict[str, Any]:
    value = facts.get(key)
    if type(value) is not dict:
        raise RuntimeError("Workshop stage facts are invalid.")
    return value


def _fact_string(value: Any) -> str:
    if type(value) is not str:
        raise RuntimeError("Workshop stage facts are invalid.")
    return value


def _fact_int_list(facts: dict[str, Any], key: str) -> list[int]:
    values = _fact_list(facts, key)
    if any(type(value) is not int for value in values):
        raise RuntimeError("Workshop stage facts are invalid.")
    return values


def _stage_result_text(stage_name: str, facts: dict[str, Any]) -> str:
    if stage_name == "inspect_library":
        return (
            f"{_fact_int(facts, 'raw_count')} raw rows; "
            f"{_fact_int(facts, 'valid_count')} valid molecules; "
            f"{_fact_int(facts, 'invalid_count')} invalid molecules; "
            f"{_fact_int(facts, 'preview_count')} molecules in the preview."
        )
    if stage_name == "generate_morgan_fingerprints":
        shape = _fact_int_list(facts, "packed_shape")
        if len(shape) != 2:
            raise RuntimeError("Workshop stage facts are invalid.")
        return (
            f"Morgan radius {_fact_int(facts, 'fingerprint_radius')} with "
            f"{_fact_int(facts, 'fingerprint_size')} bits produced packed shape "
            f"{shape[0]} x {shape[1]}; active bits min "
            f"{_fact_int(facts, 'active_bits_min')}, median "
            f"{_fact_number(facts, 'active_bits_median'):.3f}, max "
            f"{_fact_int(facts, 'active_bits_max')}."
        )
    if stage_name == "measure_tanimoto_similarity":
        pair = _fact_dict(facts, "most_similar_nonidentical_pair")
        molecule_ids = _fact_list(pair, "molecule_ids")
        if len(molecule_ids) != 2:
            raise RuntimeError("Workshop stage facts are invalid.")
        first_id, second_id = (_fact_string(value) for value in molecule_ids)
        return (
            f"top non-self pair {json.dumps(first_id)} and "
            f"{json.dumps(second_id)} had Tanimoto similarity "
            f"{_fact_number(pair, 'similarity'):.3f}; q1 "
            f"{_fact_number(facts, 'q1'):.3f}, median "
            f"{_fact_number(facts, 'median'):.3f}, q3 "
            f"{_fact_number(facts, 'q3'):.3f}, p90 "
            f"{_fact_number(facts, 'p90'):.3f}."
        )
    if stage_name == "discover_fused_butina_clusters":
        largest_sizes = _fact_int_list(facts, "largest_cluster_sizes")
        largest_text = ", ".join(str(size) for size in largest_sizes)
        return (
            f"cutoff {_fact_number(facts, 'cluster_cutoff'):.2f} produced "
            f"{_fact_int(facts, 'cluster_count')} clusters with "
            f"{_fact_int(facts, 'singleton_count')} singletons; "
            f"largest cluster sizes: {largest_text}."
        )
    if stage_name == "embed_representative_conformers":
        selected_count = _fact_int(facts, "selected_representative_count")
        requested_per_representative = _fact_int(
            facts, "requested_conformers_per_representative"
        )
        requested_conformer_count = selected_count * requested_per_representative
        return (
            f"selected {selected_count} of "
            f"{_fact_int(facts, 'requested_representative_count')} representatives "
            f"and generated {_fact_int(facts, 'generated_conformer_count')} of "
            f"{requested_conformer_count} requested conformers; "
            f"{len(_fact_list(facts, 'partial_embedding_ids'))} partial ID, "
            f"{len(_fact_list(facts, 'zero_embedding_ids'))} zero IDs; "
            f"ETKDGv3 seed {PROFILE['etkdg_random_seed']}."
        )
    if stage_name == "optimize_conformers_mmff94":
        minimum_records = _fact_list(facts, "selected_conformer_records")
        minimum_values: list[str] = []
        for record in minimum_records:
            if type(record) is not dict:
                raise RuntimeError("Workshop stage facts are invalid.")
            molecule_id = _fact_string(record.get("molecule_id"))
            energy = _fact_number(record, "energy_kcal_mol")
            minimum_values.append(f"{json.dumps(molecule_id)}={energy:.3f} kcal/mol")
        minima_text = ", ".join(minimum_values) if minimum_values else "none"
        return (
            f"{_fact_int(facts, 'attempted_conformer_count')} conformers attempted; "
            f"{_fact_int(facts, 'converged_conformer_count')} converged; "
            f"{_fact_int(facts, 'unconverged_conformer_count')} unconverged; "
            f"within-molecule minima: {minima_text}; maximum iterations "
            f"{PROFILE['mmff94_max_iterations']}."
        )
    raise ValueError("Unsupported workshop stage.")


def _stage_readme(
    stage_name: str,
    stage_spec: StageSpec,
    facts: dict[str, Any],
) -> str:
    result_text = _stage_result_text(stage_name, facts)
    return (
        f"# {stage_spec.question}\n\n"
        f"- Method: {stage_spec.method}\n"
        "- Result source: `summary.json`\n"
        f"- Result: {result_text}\n"
        f"- Scientific limit: {stage_spec.limit}\n"
    )


def _formatted_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    )


def _invalid_chemistry_export() -> RuntimeError:
    return RuntimeError("Workshop chemistry export is invalid.")


def _validated_library_rows(state: WorkflowState) -> list[tuple[str, int]]:
    if len(state.records) != 256:
        raise _invalid_chemistry_export()
    rows: list[tuple[str, int]] = []
    for record in state.records:
        if type(record) is not dict:
            raise _invalid_chemistry_export()
        molecule_id = record.get("id")
        source_row = record.get("source_row")
        if (
            type(molecule_id) is not str
            or not molecule_id
            or type(source_row) is not int
            or source_row < 0
        ):
            raise _invalid_chemistry_export()
        rows.append((molecule_id, source_row))
    molecule_ids = [molecule_id for molecule_id, _ in rows]
    source_rows = [source_row for _, source_row in rows]
    if len(set(molecule_ids)) != len(rows) or len(set(source_rows)) != len(rows):
        raise _invalid_chemistry_export()
    return rows


def _top_similarity_rows(state: WorkflowState) -> list[dict[str, Any]]:
    library_rows = _validated_library_rows(state)
    matrix = validated_similarity_matrix(state)
    ranked: list[tuple[float, int, int]] = []
    for first in range(len(library_rows)):
        for second in range(first + 1, len(library_rows)):
            ranked.append((float(matrix[first, second]), first, second))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(ranked) < 10:
        raise _invalid_chemistry_export()
    return [
        {
            "rank": rank,
            "molecule_1_id": library_rows[first][0],
            "molecule_1_source_row": library_rows[first][1],
            "molecule_2_id": library_rows[second][0],
            "molecule_2_source_row": library_rows[second][1],
            "tanimoto_similarity": score,
        }
        for rank, (score, first, second) in enumerate(ranked[:10], start=1)
    ]


def _similarity_exports(state: WorkflowState) -> _StageDataExports:
    library_rows = _validated_library_rows(state)
    matrix = validated_similarity_matrix(state)
    molecule_ids = [molecule_id for molecule_id, _ in library_rows]
    source_rows = [source_row for _, source_row in library_rows]
    pair_columns = [
        "rank",
        "molecule_1_id",
        "molecule_1_source_row",
        "molecule_2_id",
        "molecule_2_source_row",
        "tanimoto_similarity",
    ]
    pair_frame = pd.DataFrame(_top_similarity_rows(state), columns=pair_columns)
    matrix_frame = pd.DataFrame(matrix, columns=molecule_ids)
    matrix_frame.insert(0, "source_row", source_rows)
    matrix_frame.insert(0, "molecule_id", molecule_ids)
    if matrix_frame.shape != (256, 258):
        raise _invalid_chemistry_export()
    return _StageDataExports(
        csv_files=(
            ("top_similarity_pairs.csv", pair_frame),
            ("similarity_matrix.csv", matrix_frame),
        )
    )


def _cluster_exports(state: WorkflowState) -> _StageDataExports:
    library_rows = _validated_library_rows(state)
    cluster_rows: list[dict[str, Any]] = []
    assigned: list[int] = []
    if type(state.clusters) is not list or not state.clusters:
        raise _invalid_chemistry_export()
    for cluster_id, cluster in enumerate(state.clusters):
        if type(cluster) is not list or not cluster:
            raise _invalid_chemistry_export()
        cluster_size = len(cluster)
        for molecule_index in cluster:
            if type(molecule_index) is not int or not 0 <= molecule_index < len(
                library_rows
            ):
                raise _invalid_chemistry_export()
            molecule_id, source_row = library_rows[molecule_index]
            assigned.append(molecule_index)
            cluster_rows.append(
                {
                    "molecule_index": molecule_index,
                    "molecule_id": molecule_id,
                    "source_row": source_row,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                }
            )
    if sorted(assigned) != list(range(len(library_rows))):
        raise _invalid_chemistry_export()
    cluster_rows.sort(key=lambda row: row["molecule_index"])
    columns = [
        "molecule_index",
        "molecule_id",
        "source_row",
        "cluster_id",
        "cluster_size",
    ]
    return _StageDataExports(
        csv_files=(
            ("cluster_assignments.csv", pd.DataFrame(cluster_rows, columns=columns)),
        )
    )


def _representative_source(
    state: WorkflowState,
    representative: dict[str, Any],
) -> tuple[str, int, int, int]:
    molecule_index = representative.get("molecule_index")
    molecule_id = representative.get("molecule_id")
    source_row = representative.get("source_row")
    cluster_id = representative.get("cluster_id")
    generated_count = representative.get("generated_conformer_count")
    if (
        type(molecule_index) is not int
        or not 0 <= molecule_index < len(state.records)
        or type(molecule_id) is not str
        or type(source_row) is not int
        or type(cluster_id) is not int
        or cluster_id < 0
        or type(generated_count) is not int
        or generated_count <= 0
    ):
        raise _invalid_chemistry_export()
    source = state.records[molecule_index]
    if (
        source.get("id") != molecule_id
        or source.get("source_row") != source_row
        or cluster_id >= len(state.clusters)
        or molecule_index not in state.clusters[cluster_id]
    ):
        raise _invalid_chemistry_export()
    return molecule_id, source_row, cluster_id, generated_count


def _mmff94_exports(state: WorkflowState) -> _StageDataExports:
    _validated_library_rows(state)
    optimization_summary = state.summaries.get("optimize_conformers_mmff94")
    if type(optimization_summary) is not dict:
        raise _invalid_chemistry_export()
    raw_rows = optimization_summary.get("per_conformer_records")
    if type(raw_rows) is not list or not raw_rows:
        raise _invalid_chemistry_export()

    successful_representatives: list[dict[str, Any]] = []
    for record in state.representative_records:
        if type(record) is not dict:
            raise _invalid_chemistry_export()
        generated_count = record.get("generated_conformer_count")
        if type(generated_count) is not int or generated_count < 0:
            raise _invalid_chemistry_export()
        if generated_count > 0:
            successful_representatives.append(record)
    if len(successful_representatives) != len(state.conformer_molecules):
        raise _invalid_chemistry_export()

    representative_sources: list[tuple[str, int, int, int]] = []
    expected_pairs: set[tuple[int, int]] = set()
    for optimization_molecule_index, (representative, molecule) in enumerate(
        zip(
            successful_representatives,
            state.conformer_molecules,
            strict=True,
        )
    ):
        source = _representative_source(state, representative)
        if (
            not isinstance(molecule, Chem.Mol)
            or molecule.GetNumConformers() != source[3]
        ):
            raise _invalid_chemistry_export()
        try:
            for conformer_index in range(molecule.GetNumConformers()):
                conformer = molecule.GetConformer(conformer_index)
                if conformer.GetId() != conformer_index or any(
                    not math.isfinite(float(coordinate))
                    for position in conformer.GetPositions()
                    for coordinate in position
                ):
                    raise _invalid_chemistry_export()
        except (RuntimeError, ValueError) as error:
            raise _invalid_chemistry_export() from error
        representative_sources.append(source)
        expected_pairs.update(
            (optimization_molecule_index, conformer_index)
            for conformer_index in range(molecule.GetNumConformers())
        )

    public_rows: list[dict[str, Any]] = []
    sdf_records: list[_SdfExportRecord] = []
    observed_pairs: set[tuple[int, int]] = set()
    for raw_row in raw_rows:
        if type(raw_row) is not dict:
            raise _invalid_chemistry_export()
        raw_optimization_molecule_index = raw_row.get("optimization_molecule_index")
        raw_conformer_index = raw_row.get("conformer_index")
        energy = raw_row.get("energy_kcal_mol")
        converged = raw_row.get("converged")
        raw_molecule_id = raw_row.get("molecule_id")
        raw_cluster_id = raw_row.get("cluster_id")
        if (
            type(raw_optimization_molecule_index) is not int
            or not 0 <= raw_optimization_molecule_index < len(representative_sources)
            or type(raw_conformer_index) is not int
            or raw_conformer_index < 0
            or isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
            or type(converged) is not bool
            or type(raw_molecule_id) is not str
            or type(raw_cluster_id) is not int
        ):
            raise _invalid_chemistry_export()
        molecule_id, source_row, cluster_id, _ = representative_sources[
            raw_optimization_molecule_index
        ]
        if raw_molecule_id != molecule_id or raw_cluster_id != cluster_id:
            raise _invalid_chemistry_export()
        pair = (raw_optimization_molecule_index, raw_conformer_index)
        if pair not in expected_pairs or pair in observed_pairs:
            raise _invalid_chemistry_export()
        observed_pairs.add(pair)
        record_id = f"{molecule_id}:cluster-{cluster_id}:conf-{raw_conformer_index}"
        public_row = {
            "record_id": record_id,
            "molecule_id": molecule_id,
            "source_row": source_row,
            "cluster_id": cluster_id,
            "conformer_index": raw_conformer_index,
            "energy_kcal_mol": float(energy),
            "converged": converged,
        }
        public_rows.append(public_row)
        sdf_records.append(
            _SdfExportRecord(
                record_id=record_id,
                molecule=state.conformer_molecules[raw_optimization_molecule_index],
                conformer_index=raw_conformer_index,
                properties=(
                    ("ACS_RECORD_ID", record_id),
                    ("MOLECULE_ID", molecule_id),
                    ("SOURCE_ROW", str(source_row)),
                    ("CLUSTER_ID", str(cluster_id)),
                    ("CONFORMER_INDEX", str(raw_conformer_index)),
                    ("CONVERGED", str(converged).lower()),
                    ("MMFF94_ENERGY_KCAL_MOL", repr(float(energy))),
                ),
            )
        )
    if observed_pairs != expected_pairs:
        raise _invalid_chemistry_export()

    report = build_workflow_report(state)
    expected_evidence_keys = [f"E{index:02d}" for index in range(1, 7)]
    if [record.key for record in report.evidence] != expected_evidence_keys:
        raise _invalid_chemistry_export()
    evidence_rows: list[dict[str, Any]] = []
    for record in report.evidence:
        try:
            payload = json.loads(
                record.payload_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise _invalid_chemistry_export() from error
        if type(payload) is not dict:
            raise _invalid_chemistry_export()
        evidence_rows.append(
            {
                "key": record.key,
                "label": record.label,
                "payload": payload,
                "provenance": record.provenance,
            }
        )
    evidence_payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence": evidence_rows,
    }
    columns = [
        "record_id",
        "molecule_id",
        "source_row",
        "cluster_id",
        "conformer_index",
        "energy_kcal_mol",
        "converged",
    ]
    return _StageDataExports(
        csv_files=(
            ("mmff94_energies.csv", pd.DataFrame(public_rows, columns=columns)),
        ),
        sdf_file=("optimized_conformers.sdf", tuple(sdf_records)),
        json_files=(("workflow_evidence.json", evidence_payload),),
    )


def _prepare_stage_data(
    stage_name: str,
    state: WorkflowState,
) -> _StageDataExports:
    if stage_name == "measure_tanimoto_similarity":
        exports = _similarity_exports(state)
    elif stage_name == "discover_fused_butina_clusters":
        exports = _cluster_exports(state)
    elif stage_name == "optimize_conformers_mmff94":
        exports = _mmff94_exports(state)
    else:
        exports = _StageDataExports()
    names = [name for name, _ in exports.csv_files]
    if exports.sdf_file is not None:
        names.append(exports.sdf_file[0])
    names.extend(name for name, _ in exports.json_files)
    if tuple(names) != STAGE_DATA_NAMES[stage_name]:
        raise _invalid_chemistry_export()
    return exports


def _write_sdf(
    path: Path,
    records: tuple[_SdfExportRecord, ...],
) -> None:
    with Chem.SDWriter(str(path)) as writer:
        for record in records:
            molecule = Chem.Mol(record.molecule)
            for property_name, property_value in record.properties:
                molecule.SetProp(property_name, property_value)
            writer.write(molecule, confId=record.conformer_index)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    observed = list(supplier)
    if (
        any(molecule is None for molecule in observed)
        or len(observed) != len(records)
        or [molecule.GetProp("ACS_RECORD_ID") for molecule in observed]
        != [record.record_id for record in records]
    ):
        raise _invalid_chemistry_export()


def _write_stage_data(
    stage_directory: Path,
    exports: _StageDataExports,
) -> None:
    for name, frame in exports.csv_files:
        frame.to_csv(stage_directory / name, index=False, lineterminator="\n")
    if exports.sdf_file is not None:
        name, records = exports.sdf_file
        _write_sdf(stage_directory / name, records)
    for name, payload in exports.json_files:
        (stage_directory / name).write_text(_formatted_json(payload), encoding="utf-8")


def _stage_artifact_names(stage_name: str, stage_spec: StageSpec) -> list[str]:
    return sorted(
        (
            "README.md",
            "summary.json",
            *stage_spec.image_names,
            *STAGE_DATA_NAMES[stage_name],
        )
    )


def _stage_publication(
    stage_name: str,
    execution: WorkflowExecution,
    *,
    paths: WorkshopPaths,
) -> tuple[dict[str, Any], str, StageSpec, _StageDataExports]:
    try:
        stage_spec = STAGE_SPECS[stage_name]
    except KeyError as error:
        raise ValueError("Unsupported workshop stage.") from error
    result = _stage_result(stage_name, execution)
    _validate_stage_images(stage_name, result, stage_spec)
    data_exports = _prepare_stage_data(stage_name, execution.state)

    if stage_name == "inspect_library":
        if execution.gpu is not None:
            raise RuntimeError("Workshop stage GPU identity is invalid.")
        gpu_payload: dict[str, str] | None = None
    else:
        if type(execution.gpu) is not GpuIdentity:
            raise RuntimeError("Workshop stage GPU identity is invalid.")
        gpu_payload = asdict(execution.gpu)

    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage_name,
        "dataset": {
            "filename": paths.dataset_path.name,
            "molecule_count": len(execution.state.records),
            "sha256": DATASET_SHA256,
        },
        "profile": dict(PROFILE),
        "gpu": gpu_payload,
        "facts": result.summary,
        "artifacts": _stage_artifact_names(stage_name, stage_spec),
    }
    return (
        summary_payload,
        _stage_readme(stage_name, stage_spec, result.summary),
        stage_spec,
        data_exports,
    )


def _safe_output_root(paths: WorkshopPaths) -> Path:
    fixed_root = _fixed_root(paths)
    try:
        relative = paths.output_root.relative_to(paths.root)
    except ValueError as error:
        raise RuntimeError("Workshop output root is invalid.") from error
    current = fixed_root
    for component in relative.parts:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise RuntimeError("Workshop output root is invalid.") from error
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RuntimeError("Workshop output root is invalid.")
    return current


def _regular_stage_file(directory: Path, name: str) -> Path:
    path = directory / name
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise RuntimeError("Workshop stage artifacts are invalid.") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise RuntimeError("Workshop stage artifacts are invalid.")
    return path


def _validate_stage_directory(
    stage_name: str,
    directory: Path,
    *,
    expected_summary: dict[str, Any] | None = None,
    expected_readme: str | None = None,
) -> None:
    stage_spec = STAGE_SPECS[stage_name]
    try:
        mode = os.lstat(directory).st_mode
        names = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise RuntimeError("Workshop stage artifacts are invalid.") from error
    expected_names = set(_stage_artifact_names(stage_name, stage_spec))
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or names != expected_names:
        raise RuntimeError("Workshop stage artifacts are invalid.")
    files = {name: _regular_stage_file(directory, name) for name in expected_names}
    try:
        summary_text = files["summary.json"].read_text(encoding="utf-8")
        summary = json.loads(summary_text)
        readme = files["README.md"].read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Workshop stage artifacts are invalid.") from error
    if (
        type(summary) is not dict
        or set(summary)
        != {
            "schema_version",
            "stage",
            "dataset",
            "profile",
            "gpu",
            "facts",
            "artifacts",
        }
        or summary["stage"] != stage_name
        or summary["schema_version"] != SCHEMA_VERSION
        or summary["dataset"]
        != {
            "filename": "sample_molecules.csv",
            "molecule_count": 256,
            "sha256": DATASET_SHA256,
        }
        or summary["profile"] != PROFILE
        or type(summary["facts"]) is not dict
        or summary["artifacts"] != _stage_artifact_names(stage_name, stage_spec)
        or summary_text != _formatted_json(summary)
        or readme != _stage_readme(stage_name, stage_spec, summary["facts"])
    ):
        raise RuntimeError("Workshop stage artifacts are invalid.")
    if expected_summary is not None and summary != expected_summary:
        raise RuntimeError("Workshop stage artifacts are invalid.")
    if expected_readme is not None and readme != expected_readme:
        raise RuntimeError("Workshop stage artifacts are invalid.")
    for image_name in stage_spec.image_names:
        try:
            from PIL import Image

            with Image.open(files[image_name]) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise RuntimeError("Workshop stage artifacts are invalid.") from error
    for data_name in STAGE_DATA_NAMES[stage_name]:
        try:
            if files[data_name].stat().st_size == 0:
                raise RuntimeError("Workshop stage artifacts are invalid.")
        except OSError as error:
            raise RuntimeError("Workshop stage artifacts are invalid.") from error


def _remove_task_owned_stage_directory(path: Path, output_root: Path) -> None:
    if path.parent == output_root and path.name.startswith(".acs-stage-"):
        shutil.rmtree(path, ignore_errors=True)


def _write_stage_directory(
    stage_name: str,
    execution: WorkflowExecution,
    stage_spec: StageSpec,
    summary_payload: dict[str, Any],
    readme_text: str,
    data_exports: _StageDataExports,
    directory: Path,
) -> None:
    result = _stage_result(stage_name, execution)
    for figure, image_name in zip(result.figures, stage_spec.image_names, strict=True):
        image_path = directory / image_name
        if stage_name == "inspect_library":
            _save_pil_image(figure, image_path)
        else:
            _save_matplotlib_figure(figure, image_path)
    (directory / "README.md").write_text(readme_text, encoding="utf-8")
    (directory / "summary.json").write_text(
        _formatted_json(summary_payload), encoding="utf-8"
    )
    _write_stage_data(directory, data_exports)


def _stage_directories_match(
    stage_name: str,
    expected_directory: Path,
    fixed_directory: Path,
) -> None:
    names = _stage_artifact_names(stage_name, STAGE_SPECS[stage_name])
    for name in names:
        if _read_regular_file(expected_directory / name) != _read_regular_file(
            fixed_directory / name
        ):
            raise RuntimeError("Workshop stage artifacts are invalid.")


def _publish_stage(
    stage_name: str,
    execution: WorkflowExecution,
    *,
    paths: WorkshopPaths,
) -> tuple[dict[str, Any], Path, StageSpec]:
    summary_payload, readme_text, stage_spec, data_exports = _stage_publication(
        stage_name, execution, paths=paths
    )
    output_root = _safe_output_root(paths)
    stage_directory = output_root / stage_spec.directory
    temporary_directory: Path | None = Path(
        tempfile.mkdtemp(prefix=".acs-stage-", dir=output_root)
    )
    try:
        _write_stage_directory(
            stage_name,
            execution,
            stage_spec,
            summary_payload,
            readme_text,
            data_exports,
            temporary_directory,
        )
        _validate_stage_directory(
            stage_name,
            temporary_directory,
            expected_summary=summary_payload,
            expected_readme=readme_text,
        )
        try:
            os.lstat(stage_directory)
        except FileNotFoundError:
            os.replace(temporary_directory, stage_directory)
            temporary_directory = None
        else:
            _validate_stage_directory(
                stage_name,
                stage_directory,
                expected_summary=summary_payload,
                expected_readme=readme_text,
            )
            _stage_directories_match(stage_name, temporary_directory, stage_directory)
    except OSError as error:
        raise RuntimeError("Workshop stage artifacts are invalid.") from error
    finally:
        if temporary_directory is not None:
            _remove_task_owned_stage_directory(temporary_directory, output_root)
    return summary_payload, stage_directory, stage_spec


def run_stage(
    stage_name: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
    workflow_executor: WorkflowExecutor | None = None,
) -> dict[str, Any]:
    verify_manifest(paths)
    if stage_name not in STAGE_SPECS:
        raise ValueError("Unsupported workshop stage.")
    if workflow_executor is None:
        execution = execute_workflow_prefix(stage_name, paths=paths)
    else:
        execution = workflow_executor(stage_name)
    stage_summary_payload, stage_directory, stage_spec = _publish_stage(
        stage_name,
        execution,
        paths=paths,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stage": stage_name,
        "summary": stage_summary_payload,
        "image_paths": [
            str((stage_directory / name).resolve()) for name in stage_spec.image_names
        ],
        "artifact_directory": str(stage_directory.resolve()),
        "results_zip_path": str((paths.output_root / "results.zip").resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def _lesson_execution_for_stage(
    stage_name: str,
    execution: WorkflowExecution,
    lesson_stages: tuple[str, str],
) -> WorkflowExecution:
    retained = execution.stage_results[-2:]
    if tuple(result.stage for result in retained) != lesson_stages:
        raise RuntimeError("Workshop lesson result is invalid.")
    return _prefix_execution_for_stage(stage_name, execution)


def _prefix_execution_for_stage(
    stage_name: str,
    execution: WorkflowExecution,
) -> WorkflowExecution:
    try:
        result = next(
            result for result in execution.stage_results if result.stage == stage_name
        )
    except StopIteration as error:
        raise RuntimeError("Workshop lesson result is invalid.") from error
    return WorkflowExecution(
        state=execution.state,
        stage_results=(result,),
        gpu=None if stage_name == "inspect_library" else execution.gpu,
    )


def _compact_stage_item(
    stage_name: str,
    summary: dict[str, Any],
    stage_directory: Path,
    stage_spec: StageSpec,
) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "result": _stage_result_text(stage_name, _fact_dict(summary, "facts")),
        "image_paths": [
            str((stage_directory / image_name).resolve())
            for image_name in stage_spec.image_names
        ],
        "summary_path": str((stage_directory / "summary.json").resolve()),
        "readme_path": str((stage_directory / "README.md").resolve()),
        "artifact_directory": str(stage_directory.resolve()),
    }


_OBJECTIVE_CONTEXT_KEYS: Final = {
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
_OBJECTIVE_STATE_KEYS: Final = {
    "schema_version",
    "dataset_sha256",
    "context_sha256",
    "current",
    "menu",
    "accepted_attempt_count",
    "terminal",
    "termination_reason",
    "attempts",
    "last_request",
    "last_result",
}
_OBJECTIVE_FILES: Final = (
    "README.md",
    "objective_summary.json",
    "objective_evidence.json",
    "score_trajectory.png",
    "final_panel.png",
    "final_similarity_heatmap.png",
)
# The fixed workshop bundle contains only small text, table, structure, and image
# artifacts. Bound expansion before reading any member so a damaged or hostile
# archive cannot turn a retry into an unbounded memory allocation.
_RESULTS_ARCHIVE_MAX_MEMBER_BYTES: Final = 8 * 1024 * 1024
_RESULTS_ARCHIVE_MAX_EXPANDED_BYTES: Final = 32 * 1024 * 1024


def _objective_error() -> RuntimeError:
    return RuntimeError("Workshop objective state is invalid.")


def _validated_private_root(paths: WorkshopPaths) -> Path:
    try:
        mode = os.lstat(paths.state_root).st_mode
    except OSError as error:
        raise _objective_error() from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or stat.S_IMODE(mode) != 0o700:
        raise _objective_error()
    return paths.state_root


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_regular_file(path)).hexdigest()


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(_private_json_bytes(payload))
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _read_private_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        mode = os.lstat(path).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode) or stat.S_IMODE(mode) != 0o600:
            raise _objective_error()
        payload_bytes = _read_regular_file(path)
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _objective_error() from error
    if type(payload) is not dict or _private_json_bytes(payload) != payload_bytes:
        raise _objective_error()
    return payload, payload_bytes


def _measurement_payload(
    measurement: PanelMeasurement, *, public: bool = False
) -> dict[str, Any]:
    payload = {
        "selected_ids": list(measurement.selected_ids),
        "score": measurement.score,
        "limiting_pairs": [list(pair) for pair in measurement.limiting_pairs],
        "achieved": measurement.achieved,
    }
    if not public:
        payload["score_key"] = measurement.score_key
    return payload


def _action_payload(action: ObjectiveSwap, *, public: bool = False) -> dict[str, Any]:
    payload = {
        "swap_id": action.swap_id,
        "replace_id": action.replace_id,
        "replacement_id": action.replacement_id,
        "resulting_ids": list(action.resulting_ids),
        "predicted_score": action.predicted_score,
        "score_delta": action.score_delta,
        "limiting_pairs": [list(pair) for pair in action.limiting_pairs],
        "target_status": action.target_status,
    }
    if not public:
        payload["predicted_score_key"] = action.predicted_score_key
        payload["limiting_pair"] = list(action.limiting_pair or ())
    return payload


def _attempt_payload(attempt: ObjectiveAttempt) -> dict[str, Any]:
    return {
        "attempt_number": attempt.attempt_number,
        "state_id": attempt.state_id,
        "selected_ids": list(attempt.selected_ids),
        "score": attempt.score,
        "score_key": attempt.score_key,
        "limiting_pair": list(attempt.limiting_pair),
        "limiting_pairs": [list(pair) for pair in attempt.limiting_pairs],
        "constraints_passed": attempt.constraints_passed,
        "achieved": attempt.achieved,
        "selected_swap": _action_payload(attempt.selected_swap),
    }


def _menu_payload(menu: ObjectiveActionMenu) -> dict[str, Any]:
    return {
        "state_id": menu.state_id,
        "source": _measurement_payload(menu.source),
        "accepted_attempt_count": menu.accepted_attempt_count,
        "actions": [_action_payload(action) for action in menu.actions],
    }


def _context_payload(
    context: ObjectiveContext, stage_zip_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "profile": dict(PROFILE),
        "candidates": [asdict(candidate) for candidate in context.candidates],
        "baseline_ids": list(context.baseline_ids),
        "baseline_score": context.baseline_score,
        "benchmark_score": context.benchmark_score,
        "target_score": context.target_score,
        "distance_matrix": context.distance_matrix.tolist(),
        "stage_results_zip_sha256": stage_zip_sha256,
    }


def _context_from_payload(payload: dict[str, Any]) -> ObjectiveContext:
    try:
        if (
            set(payload) != _OBJECTIVE_CONTEXT_KEYS
            or payload["schema_version"] != SCHEMA_VERSION
            or type(payload["schema_version"]) is not int
            or payload["dataset_sha256"] != DATASET_SHA256
            or not _exact_json_equal(payload["profile"], PROFILE)
            or type(payload["candidates"]) is not list
            or len(payload["candidates"]) != 8
            or type(payload["stage_results_zip_sha256"]) is not str
            or len(payload["stage_results_zip_sha256"]) != 64
            or any(
                character not in _LOWER_HEXADECIMAL
                for character in payload["stage_results_zip_sha256"]
            )
        ):
            raise ValueError
        candidates = tuple(
            ObjectiveCandidate(**candidate) for candidate in payload["candidates"]
        )
        context = ObjectiveContext(
            candidates=candidates,
            baseline_ids=tuple(payload["baseline_ids"]),
            baseline_score=payload["baseline_score"],
            benchmark_score=payload["benchmark_score"],
            target_score=payload["target_score"],
            distance_matrix=np.asarray(payload["distance_matrix"], dtype=np.float64),
        )
        baseline = measure_panel(context, context.baseline_ids)
        benchmark = attainable_benchmark(context)
        if (
            baseline.score != context.baseline_score
            or benchmark.score != context.benchmark_score
            or not certify_argmax_reachability(context)
            or not _exact_json_equal(
                _context_payload(context, payload["stage_results_zip_sha256"]),
                payload,
            )
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise _objective_error() from error
    return context


def _swap_from_payload(payload: object) -> ObjectiveSwap:
    if type(payload) is not dict:
        raise _objective_error()
    try:
        swap = ObjectiveSwap(
            swap_id=payload["swap_id"],
            replace_id=payload["replace_id"],
            replacement_id=payload["replacement_id"],
            resulting_ids=tuple(payload["resulting_ids"]),
            predicted_score=payload["predicted_score"],
            predicted_score_key=payload["predicted_score_key"],
            score_delta=payload["score_delta"],
            limiting_pair=tuple(payload["limiting_pair"]),
            limiting_pairs=tuple(tuple(pair) for pair in payload["limiting_pairs"]),
            target_status=payload["target_status"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _objective_error() from error
    if not _exact_json_equal(_action_payload(swap), payload):
        raise _objective_error()
    return swap


def _derive_objective(
    context: ObjectiveContext, attempts: tuple[ObjectiveAttempt, ...]
) -> tuple[PanelMeasurement, ObjectiveActionMenu | None, ObjectiveRun | None]:
    current = measure_panel(context, context.baseline_ids)
    for position, attempt in enumerate(attempts, start=1):
        if attempt.attempt_number != position:
            raise _objective_error()
        menu = build_action_menu(context, current, position - 1)
        expected = evaluate_selected_swap(
            context, menu, attempt.selected_swap, position
        )
        if not _exact_json_equal(_attempt_payload(expected), _attempt_payload(attempt)):
            raise _objective_error()
        current = expected.measurement
    return _resolve_objective_state(context, attempts, current, validate_terminal=True)


def _current_objective_run(
    context: ObjectiveContext,
    attempts: tuple[ObjectiveAttempt, ...],
    current: PanelMeasurement,
    reason: TerminationReason,
) -> ObjectiveRun:
    return ObjectiveRun(
        context=context,
        baseline=measure_panel(context, context.baseline_ids),
        attempts=attempts,
        achieved=reason
        in {
            TerminationReason.TARGET_ACHIEVED,
            TerminationReason.BASELINE_ALREADY_OPTIMAL,
        },
        termination_reason=reason,
        final_ids=current.selected_ids,
        final_score=current.score,
        final_score_key=current.score_key,
    )


def _resolve_objective_state(
    context: ObjectiveContext,
    attempts: tuple[ObjectiveAttempt, ...],
    current: PanelMeasurement,
    *,
    validate_terminal: bool = False,
) -> tuple[PanelMeasurement, ObjectiveActionMenu | None, ObjectiveRun | None]:
    baseline = measure_panel(context, context.baseline_ids)
    benchmark = attainable_benchmark(context)
    if baseline.score_key == benchmark.score_key:
        return current, None, baseline_terminal_run(context)
    if current.achieved:
        if not validate_terminal:
            return (
                current,
                None,
                _current_objective_run(
                    context, attempts, current, TerminationReason.TARGET_ACHIEVED
                ),
            )
        return (
            current,
            None,
            terminal_objective_run(
                context, attempts, TerminationReason.TARGET_ACHIEVED
            ),
        )
    if len(attempts) == MAX_ATTEMPTS:
        if not validate_terminal:
            return (
                current,
                None,
                _current_objective_run(
                    context,
                    attempts,
                    current,
                    TerminationReason.ATTEMPT_LIMIT_REACHED,
                ),
            )
        return (
            current,
            None,
            terminal_objective_run(
                context, attempts, TerminationReason.ATTEMPT_LIMIT_REACHED
            ),
        )
    menu = build_action_menu(context, current, len(attempts))
    if not menu.actions:
        if not validate_terminal:
            return (
                current,
                None,
                _current_objective_run(
                    context,
                    attempts,
                    current,
                    TerminationReason.NO_LEGAL_IMPROVING_SWAP,
                ),
            )
        return current, None, finalize_no_legal_swap(context, attempts, current, menu)
    return current, menu, None


def _state_payload(
    context_sha256: str,
    current: PanelMeasurement,
    menu: ObjectiveActionMenu | None,
    run: ObjectiveRun | None,
    attempts: tuple[ObjectiveAttempt, ...],
    *,
    last_request: dict[str, str] | None = None,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "context_sha256": context_sha256,
        "current": _measurement_payload(current),
        "menu": None if menu is None else _menu_payload(menu),
        "accepted_attempt_count": len(attempts),
        "terminal": run is not None,
        "termination_reason": (None if run is None else run.termination_reason.value),
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
        "last_request": last_request,
        "last_result": last_result,
    }


def _load_objective_state(
    paths: WorkshopPaths,
) -> tuple[
    ObjectiveContext,
    tuple[ObjectiveAttempt, ...],
    PanelMeasurement,
    ObjectiveActionMenu | None,
    ObjectiveRun | None,
    dict[str, Any],
]:
    _validated_private_root(paths)
    context_payload, context_bytes = _read_private_json(paths.context_path)
    state_payload, _ = _read_private_json(paths.history_path)
    context = _context_from_payload(context_payload)
    try:
        if (
            set(state_payload) != _OBJECTIVE_STATE_KEYS
            or state_payload["schema_version"] != SCHEMA_VERSION
            or type(state_payload["schema_version"]) is not int
            or state_payload["dataset_sha256"] != DATASET_SHA256
            or state_payload["context_sha256"]
            != hashlib.sha256(context_bytes).hexdigest()
            or type(state_payload["attempts"]) is not list
            or not 0 <= len(state_payload["attempts"]) <= MAX_ATTEMPTS
            or type(state_payload["accepted_attempt_count"]) is not int
            or type(state_payload["terminal"]) is not bool
        ):
            raise ValueError
        attempts = tuple(
            ObjectiveAttempt(
                attempt_number=item["attempt_number"],
                state_id=item["state_id"],
                selected_ids=tuple(item["selected_ids"]),
                score=item["score"],
                score_key=item["score_key"],
                limiting_pair=tuple(item["limiting_pair"]),
                limiting_pairs=tuple(tuple(pair) for pair in item["limiting_pairs"]),
                constraints_passed=item["constraints_passed"],
                achieved=item["achieved"],
                selected_swap=_swap_from_payload(item["selected_swap"]),
            )
            for item in state_payload["attempts"]
        )
        if not _exact_json_equal(
            [_attempt_payload(attempt) for attempt in attempts],
            state_payload["attempts"],
        ):
            raise ValueError
        current, menu, run = _derive_objective(context, attempts)
        expected = _state_payload(
            state_payload["context_sha256"],
            current,
            menu,
            run,
            attempts,
            last_request=state_payload["last_request"],
            last_result=state_payload["last_result"],
        )
        if not _exact_json_equal(expected, state_payload):
            raise ValueError
        last_request = state_payload["last_request"]
        last_result = state_payload["last_result"]
        if (last_request is None) != (last_result is None):
            raise ValueError
        if last_request is not None and (
            type(last_request) is not dict
            or set(last_request) != {"state_id", "swap_id"}
            or any(type(value) is not str for value in last_request.values())
            or type(last_result) is not dict
            or not attempts
            or last_request
            != {
                "state_id": attempts[-1].state_id,
                "swap_id": attempts[-1].selected_swap.swap_id,
            }
        ):
            raise ValueError
        if last_request is not None:
            expected_result = (
                _terminal_envelope(paths, run)
                if run is not None
                else _pending_envelope(paths, context, current, menu)
            )
            if not _exact_json_equal(last_result, expected_result):
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise _objective_error() from error
    return context, attempts, current, menu, run, state_payload


def _initialize_objective_state(
    paths: WorkshopPaths, execution: WorkflowExecution, stage_archive: Path
) -> None:
    _validated_private_root(paths)
    context = build_objective_context(execution.state)
    expected_context = _context_payload(context, _sha256_file(stage_archive))
    context_exists = paths.context_path.exists() or paths.context_path.is_symlink()
    state_exists = paths.history_path.exists() or paths.history_path.is_symlink()
    if context_exists != state_exists:
        raise _objective_error()
    if not context_exists:
        context_sha256 = hashlib.sha256(
            _private_json_bytes(expected_context)
        ).hexdigest()
        current, menu, run = _derive_objective(context, ())
        initial = _state_payload(context_sha256, current, menu, run, ())
        _atomic_private_json(paths.context_path, expected_context)
        _atomic_private_json(paths.history_path, initial)
        return
    stored_context, _ = _read_private_json(paths.context_path)
    if not _exact_json_equal(stored_context, expected_context):
        raise _objective_error()
    _load_objective_state(paths)


def _pending_envelope(
    paths: WorkshopPaths,
    context: ObjectiveContext,
    current: PanelMeasurement,
    menu: ObjectiveActionMenu,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "terminal": False,
        "attempt_count": menu.accepted_attempt_count,
        "attempt_limit": MAX_ATTEMPTS,
        "state_id": menu.state_id,
        "current": _measurement_payload(current, public=True),
        "target_score": context.target_score,
        "actions": [_action_payload(action, public=True) for action in menu.actions],
        "achieved": None,
        "termination_reason": None,
        "image_paths": [],
        "artifact_directory": str((paths.output_root / "07-objective").resolve()),
        "results_zip_path": str((paths.output_root / "results.zip").resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def _terminal_attempt_payload(attempt: ObjectiveAttempt) -> dict[str, Any]:
    measurement = attempt.measurement
    return {
        "attempt_number": attempt.attempt_number,
        "state_id": attempt.state_id,
        "selected_ids": list(attempt.selected_ids),
        "score": attempt.score,
        "limiting_pairs": [list(pair) for pair in measurement.limiting_pairs],
        "achieved": attempt.achieved,
        "selected_swap": _action_payload(attempt.selected_swap, public=True),
    }


def _terminal_envelope(paths: WorkshopPaths, run: ObjectiveRun) -> dict[str, Any]:
    objective_directory = paths.output_root / "07-objective"
    final = measure_panel(run.context, run.final_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "terminal": True,
        "attempt_count": len(run.attempts),
        "attempt_limit": MAX_ATTEMPTS,
        "baseline": _measurement_payload(run.baseline, public=True),
        "target_score": run.context.target_score,
        "final": _measurement_payload(final, public=True),
        "attempts": [_terminal_attempt_payload(attempt) for attempt in run.attempts],
        "achieved": run.achieved,
        "termination_reason": run.termination_reason.value,
        "image_paths": [
            str((objective_directory / name).resolve())
            for name in (
                "score_trajectory.png",
                "final_panel.png",
                "final_similarity_heatmap.png",
            )
        ],
        "artifact_directory": str(objective_directory.resolve()),
        "results_zip_path": str((paths.output_root / "results.zip").resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def _bundle_readme(present_stages: set[str]) -> str:
    questions = (
        (
            "What is in the fixed molecule library and its fingerprint representation?",
            ("01-inspection", "02-fingerprints"),
        ),
        (
            "Which molecules are similar and how are they grouped?",
            ("03-similarity", "04-clusters"),
        ),
        (
            "What sampled 3D geometries and MMFF94 results are available?",
            ("05-conformers", "06-mmff94"),
        ),
        (
            "What objective challenge output is available?",
            ("07-objective",),
        ),
    )
    lines = ["# ACS workshop results", ""]
    for question, directories in questions:
        available = all(
            (directory == "07-objective" and "objective" in present_stages)
            or any(
                STAGE_DIRECTORIES.get(stage) == directory for stage in present_stages
            )
            for directory in directories
        )
        location = ", ".join(f"`{directory}/`" for directory in directories)
        state = "available" if available else "pending"
        lines.append(f"- {question} {location} ({state}).")
    return "\n".join(lines) + "\n"


def _zip_member(archive: zipfile.ZipFile, name: str, contents: bytes) -> None:
    member = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_DEFLATED
    member.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(member, contents)


def _build_results_zip_candidate(
    paths: WorkshopPaths, execution: WorkflowExecution
) -> Path:
    output_root = _safe_output_root(paths)
    present: list[tuple[str, Path]] = []
    for stage_name in STAGE_ORDER:
        directory = output_root / STAGE_DIRECTORIES[stage_name]
        try:
            os.lstat(directory)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError("Workshop stage artifacts are invalid.") from error
        try:
            stage_execution = _prefix_execution_for_stage(stage_name, execution)
        except RuntimeError:
            continue
        _, validated_directory, _ = _publish_stage(
            stage_name, stage_execution, paths=paths
        )
        present.append((stage_name, validated_directory))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".acs-results-", suffix=".zip", dir=output_root
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            strict_timestamps=True,
        ) as archive:
            _zip_member(
                archive,
                "README.md",
                _bundle_readme({name for name, _ in present}).encode("utf-8"),
            )
            for source_name in ("data/sample_molecules.csv", "data/PROVENANCE.md"):
                _zip_member(
                    archive, source_name, (paths.root / source_name).read_bytes()
                )
            for stage_name, directory in present:
                for artifact_name in _stage_artifact_names(
                    stage_name, STAGE_SPECS[stage_name]
                ):
                    _zip_member(
                        archive,
                        f"{STAGE_DIRECTORIES[stage_name]}/{artifact_name}",
                        _read_regular_file(directory / artifact_name),
                    )
    except Exception:
        try:
            if temporary_path.exists() and temporary_path.parent == output_root:
                temporary_path.unlink()
        except OSError:
            pass
        raise
    return temporary_path


def _rebuild_results_zip(paths: WorkshopPaths, execution: WorkflowExecution) -> Path:
    output_root = _safe_output_root(paths)
    temporary_path = _build_results_zip_candidate(paths, execution)
    archive_path = output_root / "results.zip"
    try:
        os.replace(temporary_path, archive_path)
    except Exception:
        try:
            if temporary_path.exists() and temporary_path.parent == output_root:
                temporary_path.unlink()
        except OSError:
            pass
        raise
    return archive_path


def _objective_readme(run: ObjectiveRun) -> str:
    return (
        "# Bounded molecular-diversity objective\n\n"
        "- Objective: maximize the minimum pairwise Morgan/Tanimoto distance.\n"
        f"- Result: {run.termination_reason.value}.\n"
        f"- Accepted actions: {len(run.attempts)} of {MAX_ATTEMPTS}.\n"
        "- Scope: structural diversity in the fixed eight-molecule candidate pool.\n"
    )


def _objective_payloads(run: ObjectiveRun) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = build_objective_evidence(run)
    objective_payload = json.loads(evidence.payload_json)
    summary = {"schema_version": SCHEMA_VERSION, **objective_payload}
    evidence_payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence": [
            {
                "key": evidence.key,
                "label": evidence.label,
                "payload": objective_payload,
                "provenance": evidence.provenance,
            }
        ],
    }
    return summary, evidence_payload


def _objective_render_state(paths: WorkshopPaths) -> WorkflowState:
    state = WorkflowState()
    inspect_library(state, paths.dataset_path)
    return state


def _write_objective_directory(
    directory: Path, run: ObjectiveRun, paths: WorkshopPaths
) -> None:
    summary, evidence = _objective_payloads(run)
    (directory / "README.md").write_text(_objective_readme(run), encoding="utf-8")
    (directory / "objective_summary.json").write_text(
        _formatted_json(summary), encoding="utf-8"
    )
    (directory / "objective_evidence.json").write_text(
        _formatted_json(evidence), encoding="utf-8"
    )
    trajectory, panel, heatmap = objective_figures(run, _objective_render_state(paths))
    _save_matplotlib_figure(trajectory, directory / "score_trajectory.png")
    _save_pil_image(panel, directory / "final_panel.png")
    _save_matplotlib_figure(heatmap, directory / "final_similarity_heatmap.png")


def _validate_objective_directory(directory: Path, run: ObjectiveRun) -> None:
    try:
        mode = os.lstat(directory).st_mode
        names = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise _objective_error() from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or names != set(_OBJECTIVE_FILES):
        raise _objective_error()
    summary, evidence = _objective_payloads(run)
    expected_text = {
        "README.md": _objective_readme(run),
        "objective_summary.json": _formatted_json(summary),
        "objective_evidence.json": _formatted_json(evidence),
    }
    for name, text in expected_text.items():
        path = directory / name
        if _read_regular_file(path) != text.encode("utf-8"):
            raise _objective_error()
    for name in _OBJECTIVE_FILES[3:]:
        path = directory / name
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise _objective_error() from error


def _objective_directories_match(first: Path, second: Path) -> None:
    for name in _OBJECTIVE_FILES:
        if _read_regular_file(first / name) != _read_regular_file(second / name):
            raise _objective_error()


def _publish_objective_directory(paths: WorkshopPaths, run: ObjectiveRun) -> Path:
    output_root = _safe_output_root(paths)
    fixed = output_root / "07-objective"
    temporary = Path(tempfile.mkdtemp(prefix=".acs-objective-", dir=output_root))
    try:
        _write_objective_directory(temporary, run, paths)
        _validate_objective_directory(temporary, run)
        try:
            os.lstat(fixed)
        except FileNotFoundError:
            os.replace(temporary, fixed)
            temporary = Path()
        else:
            _validate_objective_directory(fixed, run)
            _objective_directories_match(temporary, fixed)
    finally:
        if (
            temporary.name.startswith(".acs-objective-")
            and temporary.parent == output_root
        ):
            shutil.rmtree(temporary, ignore_errors=True)
    return fixed


def _results_archive_bytes(
    present_stages: set[str],
    stage_members: dict[str, bytes],
    objective_members: dict[str, bytes] | None = None,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        strict_timestamps=True,
    ) as archive:
        available = set(present_stages)
        if objective_members is not None:
            available.add("objective")
        _zip_member(archive, "README.md", _bundle_readme(available).encode("utf-8"))
        for name in ("data/sample_molecules.csv", "data/PROVENANCE.md"):
            _zip_member(archive, name, stage_members[name])
        for stage_name in STAGE_ORDER:
            if stage_name not in present_stages:
                continue
            for artifact_name in _stage_artifact_names(
                stage_name, STAGE_SPECS[stage_name]
            ):
                member_name = f"{STAGE_DIRECTORIES[stage_name]}/{artifact_name}"
                _zip_member(archive, member_name, stage_members[member_name])
        if objective_members is not None:
            for name in _OBJECTIVE_FILES:
                member_name = f"07-objective/{name}"
                _zip_member(archive, member_name, objective_members[member_name])
    return stream.getvalue()


def _validated_results_archive(
    archive_path: Path,
) -> tuple[bytes, set[str], dict[str, bytes], dict[str, bytes] | None]:
    raw = _read_regular_file(archive_path)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise _objective_error()
            objective_present = any(name.startswith("07-objective/") for name in names)
            present_stages: set[str] = set()
            expected_names = {
                "README.md",
                "data/sample_molecules.csv",
                "data/PROVENANCE.md",
            }
            observed_names = set(names)
            for stage_name in STAGE_ORDER:
                stage_names = {
                    f"{STAGE_DIRECTORIES[stage_name]}/{artifact_name}"
                    for artifact_name in _stage_artifact_names(
                        stage_name, STAGE_SPECS[stage_name]
                    )
                }
                overlap = stage_names & observed_names
                if overlap and overlap != stage_names:
                    raise _objective_error()
                if overlap:
                    present_stages.add(stage_name)
                    expected_names.update(stage_names)
            if not {
                "embed_representative_conformers",
                "optimize_conformers_mmff94",
            }.issubset(present_stages):
                raise _objective_error()
            if objective_present:
                expected_names.update(
                    f"07-objective/{name}" for name in _OBJECTIVE_FILES
                )
            if observed_names != expected_names:
                raise _objective_error()
            expected_attributes = (stat.S_IFREG | 0o644) << 16
            declared_expanded_bytes = 0
            for info in infos:
                if (
                    info.is_dir()
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.external_attr != expected_attributes
                    or info.filename.startswith("/")
                    or "\\" in info.filename
                    or any(part in {"", ".", ".."} for part in info.filename.split("/"))
                    or info.file_size > _RESULTS_ARCHIVE_MAX_MEMBER_BYTES
                ):
                    raise _objective_error()
                declared_expanded_bytes += info.file_size
                if declared_expanded_bytes > _RESULTS_ARCHIVE_MAX_EXPANDED_BYTES:
                    raise _objective_error()
            members: dict[str, bytes] = {}
            expanded_bytes = 0
            for info in infos:
                contents = archive.read(info)
                expanded_bytes += len(contents)
                if (
                    len(contents) != info.file_size
                    or expanded_bytes > _RESULTS_ARCHIVE_MAX_EXPANDED_BYTES
                ):
                    raise _objective_error()
                members[info.filename] = contents
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        if isinstance(error, RuntimeError):
            raise
        raise _objective_error() from error
    expected_readme = _bundle_readme(
        present_stages | ({"objective"} if objective_present else set())
    ).encode("utf-8")
    if members["README.md"] != expected_readme:
        raise _objective_error()
    stage_members = {
        name: contents
        for name, contents in members.items()
        if name != "README.md" and not name.startswith("07-objective/")
    }
    objective_members = None
    if objective_present:
        objective_members = {
            name: members[name] for name in members if name.startswith("07-objective/")
        }
    return raw, present_stages, stage_members, objective_members


def _publish_objective(paths: WorkshopPaths, run: ObjectiveRun) -> Path:
    objective_directory = _publish_objective_directory(paths, run)
    output_root = _safe_output_root(paths)
    archive_path = output_root / "results.zip"
    context_payload, _ = _read_private_json(paths.context_path)
    binding = context_payload["stage_results_zip_sha256"]
    raw, present_stages, stage_members, current_objective = _validated_results_archive(
        archive_path
    )
    canonical_stage = _results_archive_bytes(present_stages, stage_members)
    if hashlib.sha256(canonical_stage).hexdigest() != binding:
        raise _objective_error()
    objective_members = {
        f"07-objective/{name}": _read_regular_file(objective_directory / name)
        for name in _OBJECTIVE_FILES
    }
    expected = _results_archive_bytes(present_stages, stage_members, objective_members)
    if current_objective is not None:
        if current_objective != objective_members or raw != expected:
            raise _objective_error()
        return archive_path
    if hashlib.sha256(raw).hexdigest() != binding:
        raise _objective_error()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".acs-objective-results-", suffix=".zip", dir=output_root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(expected)
        os.replace(temporary, archive_path)
        temporary = Path()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if (
            temporary.name.startswith(".acs-objective-results-")
            and temporary.parent == output_root
        ):
            try:
                temporary.unlink()
            except OSError:
                pass
    return archive_path


def _validate_bound_stage_archive(paths: WorkshopPaths) -> None:
    context_payload, _ = _read_private_json(paths.context_path)
    if (
        _sha256_file(paths.output_root / "results.zip")
        != context_payload["stage_results_zip_sha256"]
    ):
        raise _objective_error()


def objective_start(*, paths: WorkshopPaths = DEFAULT_PATHS) -> dict[str, Any]:
    verify_manifest(paths)
    context, attempts, current, menu, run, state_payload = _load_objective_state(paths)
    del attempts
    if run is None:
        if menu is None:
            raise _objective_error()
        _validate_bound_stage_archive(paths)
        return _pending_envelope(paths, context, current, menu)
    result = _terminal_envelope(paths, run)
    last_result = state_payload["last_result"]
    if last_result is not None and last_result != result:
        raise _objective_error()
    _publish_objective(paths, run)
    return result


def objective_step(
    state_id: str,
    swap_id: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    verify_manifest(paths)
    context, attempts, current, menu, run, state_payload = _load_objective_state(paths)
    request = {"state_id": state_id, "swap_id": swap_id}
    if request == state_payload["last_request"]:
        result = state_payload["last_result"]
        if run is None:
            _validate_bound_stage_archive(paths)
        else:
            _publish_objective(paths, run)
        return result
    if run is not None or menu is None:
        raise ValueError("Objective selection does not match the exact menu revision.")
    _validate_bound_stage_archive(paths)
    action = resolve_menu_action(
        context,
        menu,
        state_id=state_id,
        swap_id=swap_id,
        observed_limiting_pairs=current.limiting_pairs,
        decision_rule="maximize_predicted_minimum_distance",
    )
    if action not in accepted_maxima(menu):
        raise ValueError("Objective selection is not an accepted exact menu action.")
    attempt = evaluate_selected_swap(context, menu, action, len(attempts) + 1)
    updated_attempts = (*attempts, attempt)
    next_current, next_menu, next_run = _resolve_objective_state(
        context, updated_attempts, attempt.measurement
    )
    result = (
        _pending_envelope(paths, context, next_current, next_menu)
        if next_run is None and next_menu is not None
        else _terminal_envelope(paths, next_run)
    )
    updated_state = _state_payload(
        state_payload["context_sha256"],
        next_current,
        next_menu,
        next_run,
        updated_attempts,
        last_request=request,
        last_result=result,
    )
    _atomic_private_json(paths.history_path, updated_state)
    if next_run is not None:
        _publish_objective(paths, next_run)
    return result


def run_lesson(
    lesson: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
    workflow_executor: WorkflowExecutor | None = None,
) -> dict[str, Any]:
    verify_manifest(paths)
    try:
        terminal_stage = LESSON_TERMINAL_STAGES[lesson]
        lesson_stages = LESSON_STAGES[lesson]
    except KeyError as error:
        raise ValueError("Invalid workshop arguments.") from error
    if workflow_executor is None:
        execution = execute_workflow_prefix(terminal_stage, paths=paths)
    else:
        execution = workflow_executor(terminal_stage)
    completed_stages: list[dict[str, Any]] = []
    for stage_name in lesson_stages:
        stage_execution = _lesson_execution_for_stage(
            stage_name, execution, lesson_stages
        )
        summary, directory, stage_spec = _publish_stage(
            stage_name, stage_execution, paths=paths
        )
        completed_stages.append(
            _compact_stage_item(stage_name, summary, directory, stage_spec)
        )
    if lesson == "sampled-3d-geometry":
        output_root = _safe_output_root(paths)
        temporary_archive = _build_results_zip_candidate(paths, execution)
        try:
            _initialize_objective_state(paths, execution, temporary_archive)
            _, _, _, _, objective_run, _ = _load_objective_state(paths)
            archive_path = output_root / "results.zip"
            os.replace(temporary_archive, archive_path)
            temporary_archive = Path()
        finally:
            if (
                temporary_archive.name.startswith(".acs-results-")
                and temporary_archive.parent == output_root
            ):
                try:
                    temporary_archive.unlink()
                except OSError:
                    pass
        if objective_run is not None:
            _publish_objective(paths, objective_run)
    else:
        archive_path = _rebuild_results_zip(paths, execution)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "lesson": lesson,
        "completed_stages": completed_stages,
        "results_zip_path": str(archive_path.resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        verify_manifest(DEFAULT_PATHS)
        arguments = build_parser().parse_args(argv)
        if arguments.command == "run-stage":
            result = run_stage(arguments.stage_name, paths=DEFAULT_PATHS)
        elif arguments.command == "run-lesson":
            result = run_lesson(arguments.lesson, paths=DEFAULT_PATHS)
        elif arguments.command == "objective-start":
            result = objective_start(paths=DEFAULT_PATHS)
        else:
            result = objective_step(
                arguments.state_id,
                arguments.swap_id,
                paths=DEFAULT_PATHS,
            )
        print(_canonical_json_bytes(result).decode("utf-8"), end="")
        return 0
    except Exception as error:
        print(f"Error: {_safe_error_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
