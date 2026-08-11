from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import pandas as pd
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


def _publish_stage(
    stage_name: str,
    execution: WorkflowExecution,
    *,
    paths: WorkshopPaths,
) -> tuple[dict[str, Any], Path, StageSpec]:
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

    artifact_names = sorted(
        (
            "README.md",
            "summary.json",
            *stage_spec.image_names,
            *STAGE_DATA_NAMES[stage_name],
        )
    )
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
        "artifacts": artifact_names,
    }
    summary_text = _formatted_json(summary_payload)
    readme_text = _stage_readme(stage_name, stage_spec, result.summary)

    stage_directory = paths.output_root / stage_spec.directory
    stage_directory.mkdir(parents=True, exist_ok=True)
    for figure, image_name in zip(result.figures, stage_spec.image_names, strict=True):
        image_path = stage_directory / image_name
        if stage_name == "inspect_library":
            _save_pil_image(figure, image_path)
        else:
            _save_matplotlib_figure(figure, image_path)
    (stage_directory / "README.md").write_text(readme_text, encoding="utf-8")
    (stage_directory / "summary.json").write_text(summary_text, encoding="utf-8")
    _write_stage_data(stage_directory, data_exports)
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


def main(argv: Sequence[str] | None = None) -> int:
    try:
        verify_manifest(DEFAULT_PATHS)
        arguments = build_parser().parse_args(argv)
        if arguments.command == "run-stage":
            result = run_stage(arguments.stage_name, paths=DEFAULT_PATHS)
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
