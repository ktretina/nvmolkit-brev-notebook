# ACS Fall 2026 Attendee Workshop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Extend the ACS OpenClaw Launchable with six fixed chemistry stages and one bounded objective challenge, qualify the complete experience on a fresh Brev L4, and then create the concise attendee reference sheet.

**Architecture:** One new Python runner reuses chemistry_workflow.py and objective_challenge.py. It owns fixed stage execution, validated chemistry artifacts, deterministic bundles, and schema-checked objective state; OpenClaw and hosted Nemotron remain the only conversational agent layer. The existing setup script uploads and verifies the runner, while one canonical prompt file drives both live acceptance and the final Markdown page.

**Tech Stack:** Python 3.13 in the NemoClaw sandbox, Python 3.12 for local tests, RDKit, nvMolKit 0.5.0, PyTorch 2.7.1+cu128, NumPy, pandas, Matplotlib, Pillow, Bash, pytest, Ruff, mypy, Node.js, Gitleaks, Brev, NemoClaw, OpenClaw, and hosted NVIDIA Nemotron.

---

## Scope and file map

Create during local implementation:

- **acs_workshop_runner.py** — fixed CLI, manifest verification, workflow execution, artifacts, ZIP, objective checkpoint, replay, and terminal figures.
- **tests/test_acs_workshop_runner.py** — runner, schema, artifact, transaction, and objective tests.
- **launchable/acs_workshop_prompts.md** — seven canonical attendee prompts and live-acceptance source.
- **tests/test_acs_workshop_prompts.py** — deterministic prompt parser and contract tests.
- **scripts/verify_acs_workshop_acceptance.py** — facilitator-only
  workspace, archive, transcript, and state-log receipts.
- **tests/test_verify_acs_workshop_acceptance.py** — receipt, bound, path,
  transcript, and secret-pattern tests.

Modify during local implementation:

- **launchable/acs_nemoclaw_launchable_setup.sh** — upload two new files, create the fixed-file manifest, run smoke checks, and preserve the current seed acceptance.
- **launchable/acs_workspace_tools.md** — replace generic code-generation guidance with the fixed runner contract.
- **tests/test_acs_nemoclaw_launchable_setup.py** — setup, cleanup, upload, manifest, permission, and smoke-test contracts.
- **tests/test_nemoclaw_phase_zero_setup.py** — preserve the existing seed prompt and media expectations while allowing the new fixed workshop assets.

Create only after fresh live acceptance:

- **docs/acs-fall-2026-workshop.md** — final attendee reference sheet.
- **tests/test_acs_fall_2026_workshop_page.py** — page links, hardware, costs, prompt identity, and secret-safety tests.

Do not modify unless a failing approved test proves a real defect:

- chemistry_workflow.py
- objective_challenge.py
- demo_agent.py
- interactive_workflow.py
- objective_findings.py
- launchable/acs_console_bootstrap.sh.in
- launchable/nemoclaw_phase_zero.sh
- launchable/start_artifact_server.sh
- launchable/openclaw_secure_link_proxy.mjs
- launchable/acs_task_prompt.txt
- acs_chemistry_task.py

Use this local test environment for every Python task:

~~~bash
env PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest
~~~

### Task 1: Lock the runner CLI, fixed paths, integrity manifest, and workflow prefix

**Files:**

- Create: tests/test_acs_workshop_runner.py
- Create: acs_workshop_runner.py

- [ ] **Step 1: Write the failing CLI and manifest tests**

Create tests/test_acs_workshop_runner.py with imports for hashlib, json, os, stat, subprocess, sys, Path, pytest, and acs_workshop_runner as runner. Add a fixture that copies the five fixed files into a temporary project root and writes this exact manifest shape:

~~~python
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


def test_cli_exposes_only_fixed_commands_without_path_options() -> None:
    parser = runner.build_parser()
    help_text = parser.format_help()
    assert "run-stage" in help_text
    assert "objective-start" in help_text
    assert "objective-step" in help_text
    assert "--dataset" not in help_text
    assert "--output" not in help_text
    assert "--retry" not in help_text


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
    assert "manifest" in completed.stderr
    assert "usage:" not in completed.stdout


def test_manifest_verification_precedes_science(
    workshop_paths: runner.WorkshopPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_executor(stage_name: str) -> runner.WorkflowExecution:
        nonlocal called
        called = True
        raise AssertionError(stage_name)

    workshop_paths.manifest_path.unlink()
    with pytest.raises(RuntimeError, match="manifest"):
        runner.run_stage(
            "inspect_library",
            paths=workshop_paths,
            workflow_executor=forbidden_executor,
        )
    assert called is False
~~~

Add parameterized mutations for extra root keys, extra file keys, missing keys, uppercase or short hashes, a symlinked manifest, a symlinked fixed file, and one changed byte. Each mutation must fail before the executor runs.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "cli or manifest"
~~~

Expected: collection fails with ModuleNotFoundError for acs_workshop_runner.

- [ ] **Step 3: Add the fixed runner types, constants, parser, and manifest verifier**

Create acs_workshop_runner.py. Define these public types and constants exactly:

~~~python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
~~~

Implement a strict canonical JSON helper and verify:

- manifest is a regular non-symlink file;
- every existing path component from the fixed resolved root to the manifest and
  each fixed file is a real directory, not a symlink;
- top-level keys are exactly schema_version and files;
- files keys are exactly MANIFEST_FILES;
- every hash is 64 lowercase hexadecimal characters;
- each target is a regular non-symlink file below the fixed root;
- each current file hash equals the manifest hash; and
- the dataset hash equals DATASET_SHA256.

The verifier must raise RuntimeError with a short message that never includes file contents.

Create build_parser() with argparse subcommands. run-stage uses choices=STAGE_ORDER. objective-step requires --state-id and --swap-id. No parser option accepts a dataset, output, parameter, retry, URL, or command.

- [ ] **Step 4: Add the exact workflow-prefix executor and L4 check**

Implement the fixed executor with this exact call order:

~~~python
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
    state = WorkflowState()
    results: list[StageResult] = [
        inspect_library(state, paths.dataset_path, expected_rows=256)
    ]
    gpu = None
    if stage_name != STAGE_ORDER[0]:
        gpu = _gpu_identity()
        results.append(
            generate_morgan_fingerprints(
                state,
                fingerprint_radius=2,
                fingerprint_size=1024,
            )
        )
    if STAGE_ORDER.index(stage_name) >= 2:
        results.append(measure_tanimoto_similarity(state))
    if STAGE_ORDER.index(stage_name) >= 3:
        results.append(
            discover_fused_butina_clusters(state, cluster_cutoff=0.40)
        )
    if STAGE_ORDER.index(stage_name) >= 4:
        results.append(
            embed_representative_conformers(
                state,
                representative_count=6,
                representative_policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
                conformers_per_representative=5,
            )
        )
    if STAGE_ORDER.index(stage_name) >= 5:
        results.append(optimize_conformers_mmff94(state))
    return WorkflowExecution(state, tuple(results), gpu)
~~~

Use this exact dispatch boundary:

~~~python
def run_stage(
    stage_name: str,
    *,
    paths: WorkshopPaths = DEFAULT_PATHS,
    workflow_executor: WorkflowExecutor | None = None,
) -> dict[str, Any]:
    verify_manifest(paths)
    execution = (
        execute_workflow_prefix(stage_name, paths=paths)
        if workflow_executor is None
        else workflow_executor(stage_name)
    )
    # Publish and return the closed envelope.
~~~

This keeps injected test executors one-argument callables while the real
executor always receives the selected paths. run_stage() must call
verify_manifest() before resolving or calling the executor.
main() must call verify_manifest(DEFAULT_PATHS) before parse_args(), including
for --help. Tests may call build_parser().format_help() directly without a
manifest. main() prints exactly one JSON object on success, prints one Error
line to stderr on failure, and returns 0 or 2.

- [ ] **Step 5: Add exact-order and GPU-boundary tests**

Use monkeypatch wrappers around the six real imported functions to record names and keyword arguments. Assert every canonical name calls exactly the expected prefix and fixed values. Patch _gpu_identity only at the GPU boundary. Add:

~~~python
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
    assert execution.gpu is None


@pytest.mark.parametrize(
    "device_count,device_name",
    ((0, ""), (2, "NVIDIA L4"), (1, "NVIDIA A100-SXM4-80GB")),
)
def test_gpu_stages_require_exactly_one_nvidia_l4(
    monkeypatch: pytest.MonkeyPatch,
    device_count: int,
    device_name: str,
) -> None:
    fake_torch = make_fake_torch(device_count, device_name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="exactly one NVIDIA L4"):
        runner._gpu_identity()
~~~

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "cli or manifest or workflow_prefix or nvidia_l4 or inspection"
~~~

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add fixed ACS workshop runner"
~~~

Expected: the staged secret scan passes and one local commit is created.

### Task 2: Add closed stage summaries, README files, and strict image adapters

**Files:**

- Modify: acs_workshop_runner.py
- Modify: tests/test_acs_workshop_runner.py

- [ ] **Step 1: Write failing stage-summary and image tests**

Create a WorkflowExecution fixture with a real WorkflowState, fixed StageResult summaries, one PIL image for inspection, and Matplotlib Figure objects for later stages. Add these assertions:

~~~python
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


def test_stage_summary_has_closed_finite_schema(
    completed_stage: Path,
) -> None:
    payload = json.loads((completed_stage / "summary.json").read_text())
    assert set(payload) == {
        "schema_version",
        "stage",
        "dataset",
        "profile",
        "gpu",
        "facts",
        "artifacts",
    }
    assert set(payload["dataset"]) == {"filename", "molecule_count", "sha256"}
    assert payload["dataset"]["sha256"] == runner.DATASET_SHA256
    assert payload["profile"] == runner.PROFILE
    json.dumps(payload, allow_nan=False)


def test_matplotlib_and_pil_adapters_write_readable_pngs(
    completed_stage: Path,
) -> None:
    for path in completed_stage.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(path) as image:
            image.verify()
~~~

Assert each README contains the stage question, method, result source, and its fixed scientific limit. Assert stages 2 through 6 have the exact GpuIdentity object serialized, while inspection has gpu set to null.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "stage_summary or image or readme"
~~~

Expected: tests fail because stage publication is not implemented.

- [ ] **Step 3: Add fixed stage metadata and save adapters**

Add a frozen StageSpec with directory, question, method, limit, and image names. Define all six entries exactly from the approved specification.

Use this exact stage-to-directory mapping:

~~~python
STAGE_DIRECTORIES = {
    "inspect_library": "01-inspection",
    "generate_morgan_fingerprints": "02-fingerprints",
    "measure_tanimoto_similarity": "03-similarity",
    "discover_fused_butina_clusters": "04-clusters",
    "embed_representative_conformers": "05-conformers",
    "optimize_conformers_mmff94": "06-mmff94",
}
~~~

Add strict adapters:

~~~python
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
~~~

Only inspection library_preview.png uses the PIL adapter. Every stage-2 through
stage-6 figure, including optimized_structures.png, uses the Matplotlib adapter
because chemistry_workflow._optimized_structure_figure returns an exact
Matplotlib Figure. Reject the wrong number or type of figures before publishing
anything.

Write JSON with sort_keys=True, indent=2, allow_nan=False, UTF-8, and one final newline. The stage summary top-level keys must match the test exactly. facts is the existing StageResult.summary without mutation. artifacts is the sorted exact filename list for that directory, including README.md and summary.json.

- [ ] **Step 4: Add stage README generation and a JSON result envelope**

Generate a short deterministic README from StageSpec and measured facts. Do not add model prose.

run_stage() returns and main() prints this exact envelope:

~~~python
{
    "schema_version": 1,
    "status": "complete",
    "stage": stage_name,
    "summary": stage_summary_payload,
    "image_paths": [
        str((stage_directory / name).resolve())
        for name in stage_spec.image_names
    ],
    "artifact_directory": str(stage_directory.resolve()),
    "results_zip_path": str((paths.output_root / "results.zip").resolve()),
    "artifact_relative_zip_path": "workshop/results.zip",
}
~~~

stage_summary_payload is the same closed payload written to summary.json,
including profile, gpu, facts, and artifacts. This lets the agent report the
fixed seed, maximum iterations, and GPU without another tool call. Do not emit
a MEDIA line from Python. The OpenClaw prompt owns media display.

- [ ] **Step 5: Run Task 2 tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "stage_summary or image or readme or result_envelope"
~~~

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add ACS workshop stage artifacts"
~~~

### Task 3: Add reusable similarity, cluster, MMFF94, SDF, and workflow-evidence files

**Files:**

- Modify: acs_workshop_runner.py
- Modify: tests/test_acs_workshop_runner.py

- [ ] **Step 1: Write failing chemistry-data tests**

Use the existing fake_gpu and conformer_gpu fixtures from tests/test_chemistry_workflow.py or reproduce their narrow tensor/result boundaries in the runner test file. Run the real chemistry workflow through the requested stage.

Add exact CSV and SDF checks:

~~~python
def test_similarity_csvs_match_records_and_matrix(completed_similarity: Path) -> None:
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


def test_mmff94_csv_and_sdf_have_matching_provenance(
    completed_mmff94: Path,
) -> None:
    rows = pd.read_csv(completed_mmff94 / "mmff94_energies.csv")
    supplier = Chem.SDMolSupplier(
        str(completed_mmff94 / "optimized_conformers.sdf"),
        removeHs=False,
    )
    molecules = [molecule for molecule in supplier if molecule is not None]
    assert len(molecules) == len(rows)
    assert [molecule.GetProp("ACS_RECORD_ID") for molecule in molecules] == (
        rows["record_id"].tolist()
    )
~~~

Assert workflow_evidence.json has schema_version and an evidence array with exact keys E01 through E06 in order. Each payload must be parsed JSON, not a nested JSON string.

- [ ] **Step 2: Run the chemistry-data tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "similarity_csv or cluster_assignments or mmff94_csv or workflow_evidence"
~~~

Expected: tests fail because the data writers do not exist.

- [ ] **Step 3: Implement the similarity and cluster exports**

Extend the chemistry_workflow imports with validated_similarity_matrix and
build_workflow_report. Add pandas and RDKit Chem imports at module scope. For
similarity, obtain the validated host matrix from
validated_similarity_matrix(state). Enumerate upper-triangle pairs, sort by
negative raw similarity and then the two matrix indices, and write the first
ten.

~~~python
def _top_similarity_rows(state: WorkflowState) -> list[dict[str, Any]]:
    matrix = validated_similarity_matrix(state)
    rows: list[dict[str, Any]] = []
    for first in range(len(state.records)):
        for second in range(first + 1, len(state.records)):
            rows.append(
                {
                    "molecule_1_id": state.records[first]["id"],
                    "molecule_1_source_row": state.records[first]["source_row"],
                    "molecule_2_id": state.records[second]["id"],
                    "molecule_2_source_row": state.records[second]["source_row"],
                    "tanimoto_similarity": float(matrix[first, second]),
                    "_first": first,
                    "_second": second,
                }
            )
    rows.sort(
        key=lambda row: (
            -row["tanimoto_similarity"],
            row["_first"],
            row["_second"],
        )
    )
    public = []
    for rank, row in enumerate(rows[:10], start=1):
        public.append(
            {
                "rank": rank,
                **{
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                },
            }
        )
    return public
~~~

Write similarity_matrix.csv with molecule_id, source_row, and the 256 molecule IDs as ordered value columns.

For clusters, flatten state.clusters into one row per molecule, include cluster size, sort by molecule_index, and verify exact coverage before writing.

- [ ] **Step 4: Implement MMFF94 CSV and SDF exports**

Use state.summaries["optimize_conformers_mmff94"]["per_conformer_records"],
state.representative_records, and state.conformer_molecules. The optimization
rows do not contain source_row, so first build the successful representative
list in the exact same order as optimize_conformers_mmff94:

~~~python
successful_representatives = [
    record
    for record in state.representative_records
    if record["generated_conformer_count"] > 0
]
~~~

For each optimization row, validate optimization_molecule_index and join its
molecule_id, cluster_id, and source_row to that exact representative. Define:

~~~python
record_id = (
    f"{row['molecule_id']}:cluster-{row['cluster_id']}:"
    f"conf-{row['conformer_index']}"
)
~~~

Write mmff94_energies.csv with:

~~~text
record_id,molecule_id,source_row,cluster_id,conformer_index,energy_kcal_mol,converged
~~~

For each row, obtain the matching optimized RDKit molecule from
state.conformer_molecules[row["optimization_molecule_index"]], copy it, set
these string properties, and write only
confId=row["conformer_index"]:

~~~text
ACS_RECORD_ID
MOLECULE_ID
SOURCE_ROW
CLUSTER_ID
CONFORMER_INDEX
CONVERGED
MMFF94_ENERGY_KCAL_MOL
~~~

Use Chem.SDWriter in a context that always closes it. After writing, reopen the SDF and verify record count and ACS_RECORD_ID ordering before stage publication.

- [ ] **Step 5: Implement E01 through E06 serialization**

Call build_workflow_report(state). Write:

~~~python
{
    "schema_version": 1,
    "evidence": [
        {
            "key": record.key,
            "label": record.label,
            "payload": json.loads(record.payload_json),
            "provenance": record.provenance,
        }
        for record in report.evidence
    ],
}
~~~

Require the keys to be exactly E01 through E06 before writing.

- [ ] **Step 6: Run Task 3 tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "similarity_csv or cluster_assignments or mmff94_csv or workflow_evidence"
~~~

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add downloadable workshop chemistry data"
~~~

### Task 4: Make stage publication rollback-safe and build the deterministic complete ZIP

**Files:**

- Modify: acs_workshop_runner.py
- Modify: tests/test_acs_workshop_runner.py

- [ ] **Step 1: Write failing transaction, symlink, and ZIP tests**

Add tests that seed a prior complete stage directory and ZIP, inject a failure
into the second os.replace call, and assert every prior byte is restored. Add
directory-shaped and symlink-shaped managed targets and require failure before
deletion. Add symlinked-ancestor cases for workspace/outputs,
workspace/outputs/workshop, and one stage directory.

Add deterministic ZIP assertions:

~~~python
def test_results_zip_is_safe_complete_and_deterministic(
    completed_workshop: runner.WorkshopPaths,
) -> None:
    archive = completed_workshop.output_root / "results.zip"
    first = archive.read_bytes()
    runner.run_stage(
        "measure_tanimoto_similarity",
        paths=completed_workshop,
        workflow_executor=fixed_similarity_execution,
    )
    assert archive.read_bytes() == first
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert names == sorted(names)
        assert "results.zip" not in names
        assert all(not name.startswith("/") and ".." not in PurePosixPath(name).parts for name in names)
        assert all(info.date_time == (2026, 1, 1, 0, 0, 0) for info in bundle.infolist())
        assert all((info.external_attr >> 16) == 0o100644 for info in bundle.infolist())
        assert not any(".acs-workshop-state" in name for name in names)
~~~

Require artifact_manifest.json to list every stage file with path, size, and SHA-256, but not itself or results.zip.

- [ ] **Step 2: Run the transaction tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "results_zip or publication or symlink or artifact_manifest"
~~~

Expected: tests fail because the current writer does not publish as one transaction.

- [ ] **Step 3: Add fixed-target validation and temporary publication paths**

Allow publication only to the StageSpec directory under paths.output_root,
paths.output_root/artifact_manifest.json, and paths.output_root/results.zip.
Use the same path-component walker as fixed-input verification: every existing
component from the resolved fixed workspace root to each managed target must
be a real directory or the expected regular final file, never a symlink. Reject
workspace/outputs and output_root ancestor symlinks before creating, deleting,
renaming, or backing up anything. Create temporary paths inside the same
validated parent with mode 0700 and a random suffix.

Write the new stage completely, validate its exact file set, build artifact_manifest.json from existing completed stage directories plus the temporary replacement stage, and then build the ZIP from the same virtual file set.

The manifest schema is:

~~~python
{
    "schema_version": 1,
    "files": [
        {
            "path": relative_path,
            "size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        for relative_path, source in sorted(public_files.items())
    ],
}
~~~

- [ ] **Step 4: Add deterministic ZIP creation and validation**

For every sorted member, construct ZipInfo with:

~~~python
info.date_time = (2026, 1, 1, 0, 0, 0)
info.compress_type = zipfile.ZIP_DEFLATED
info.create_system = 3
info.external_attr = 0o100644 << 16
~~~

Reject absolute names, dot-dot components, symlinks, backups, temporary paths, private state, the current results.zip, and files outside completed stage directories. Include artifact_manifest.json in the archive. Run ZipFile.testzip() before publication.

- [ ] **Step 5: Commit all public paths with rollback backups**

Implement one transaction for the stage directory, artifact_manifest.json, and results.zip:

1. rename existing targets to same-parent backup names;
2. rename the three validated temporary targets into place;
3. fsync their parent directories;
4. remove backups only after all replacements succeed; and
5. on any error, remove newly published targets and restore every backup in reverse order.

Never use shutil.rmtree on a symlink. Cleanup may remove only generated temporary or backup paths whose validated parent is paths.output_root.

- [ ] **Step 6: Run Task 4 tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "results_zip or publication or symlink or artifact_manifest"
~~~

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Make workshop artifact publication transactional"
~~~

### Task 5: Add the private objective checkpoint and exact history replay

**Files:**

- Modify: acs_workshop_runner.py
- Modify: tests/test_acs_workshop_runner.py

- [ ] **Step 1: Write failing checkpoint-schema and lifecycle tests**

Use optimized_state(), target_achieved_context(), controlled_context_with_tied_paths(), and other real fixtures from tests/objective_fixtures.py.

Add tests for these exact rules:

- stage 6 creates context.json and history.json only when both are absent;
- both files are regular non-symlink files with mode 0600;
- a matching stage-6 rerun preserves accepted history byte-for-byte;
- one missing file, a profile or dataset mismatch, a context hash mismatch, an extra key, a non-finite value, a wrong matrix dimension, or a changed candidate fails without public replacement;
- changing one non-limiting distance-matrix value and recomputing every stored
  hash still fails a stage-6 rerun;
- context.json and history.json never appear in artifact_manifest.json or results.zip; and
- objective-start reads but never writes history;
- objective-start and objective-step with no stage-6 checkpoint fail with the
  exact safe sentence Complete prompt 6 before starting the objective.; and
- a one-file, malformed, or conflicting checkpoint uses the generic sentence
  Objective checkpoint is invalid. and never prints stored values.

Use this exact expected history:

~~~python
assert history == {
    "schema_version": 1,
    "dataset_sha256": runner.DATASET_SHA256,
    "context_sha256": runner._canonical_sha256(context),
    "accepted_steps": [],
}
~~~

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "checkpoint or objective_start or private_state"
~~~

Expected: tests fail because no objective state exists.

- [ ] **Step 3: Add exact context and history serializers**

Import the real objective types and functions:

~~~python
from objective_challenge import (
    MAX_ATTEMPTS,
    ObjectiveAttempt,
    ObjectiveCandidate,
    ObjectiveContext,
    PanelMeasurement,
    TerminationReason,
    attainable_benchmark,
    baseline_terminal_run,
    build_action_menu,
    build_objective_context,
    certify_argmax_reachability,
    evaluate_selected_swap,
    finalize_no_legal_swap,
    measure_panel,
    resolve_menu_action,
    score_key,
    terminal_objective_run,
)
~~~

Serialize context.json with exactly:

~~~python
{
    "schema_version": 1,
    "dataset_sha256": DATASET_SHA256,
    "profile": PROFILE,
    "candidates": [
        {
            "molecule_id": candidate.molecule_id,
            "molecule_index": candidate.molecule_index,
            "source_row": candidate.source_row,
            "cluster_id": candidate.cluster_id,
        }
        for candidate in context.candidates
    ],
    "baseline_ids": list(context.baseline_ids),
    "baseline_score": context.baseline_score,
    "benchmark_score": context.benchmark_score,
    "target_score": context.target_score,
    "distance_matrix": context.distance_matrix.tolist(),
}
~~~

Serialize history.json with exactly schema_version, dataset_sha256, context_sha256, and accepted_steps. Each accepted step has exactly state_id and swap_id.

The loader must enforce closed keys, exact types, finite values, eight candidates, a four-ID baseline, an 8-by-8 float64 matrix, dataset hash, exact PROFILE, lowercase SHA-256, and maximum three accepted steps. Reconstruct ObjectiveContext and call measure_panel(), attainable_benchmark(), build_action_menu(), and certify_argmax_reachability() so the existing domain validators independently check stored scores and the bounded tied-maximum policy remains reachable.

- [ ] **Step 4: Add mode-0600 atomic private writers**

Create the state directory as a regular non-symlink directory with mode 0700.
Reject every symlinked path component below the fixed workspace root. For each
JSON file:

1. create a same-directory temporary file with os.open flags O_WRONLY, O_CREAT, O_EXCL, and O_NOFOLLOW when available;
2. set mode 0600;
3. write canonical UTF-8 JSON plus one final newline;
4. flush and fsync;
5. reject an existing symlink or non-regular target;
6. replace with os.replace; and
7. fsync the state directory.

Do not rewrite a valid matching checkpoint on stage-6 replay. When creating the
initial pair, prepare and validate both temporary files first. Add both private
files to the stage-6 replacement transaction, with same-parent backups and
reverse-order rollback, so a failure cannot leave only context.json or only
history.json.

- [ ] **Step 5: Extend the stage-6 transaction**

After optimize_conformers_mmff94 and before public publication:

~~~python
context = build_objective_context(execution.state)
context_payload = _objective_context_payload(context)
history_payload = _empty_history_payload(context_payload)
~~~

If both private files are absent, prepare them as part of the same rollback
scope as the stage-6 public directory, artifact manifest, and ZIP. If both
exist, load and reconstruct them, compare the newly generated canonical
context-payload bytes with the stored canonical context-payload bytes, replay
all accepted history, and preserve the original bytes. Do not use
ObjectiveContext dataclass equality because distance_matrix has compare=False.
If only one file exists or any context byte differs, fail before any public
path changes.

- [ ] **Step 6: Add exact history replay and objective-start**

Replay every stored step only through the real domain functions:

~~~python
def _replay_history(
    context: ObjectiveContext,
    history: dict[str, Any],
) -> tuple[tuple[ObjectiveAttempt, ...], PanelMeasurement]:
    current = measure_panel(context, context.baseline_ids)
    attempts: list[ObjectiveAttempt] = []
    for accepted_count, stored in enumerate(history["accepted_steps"]):
        menu = build_action_menu(context, current, accepted_count)
        action = resolve_menu_action(
            context,
            menu,
            state_id=stored["state_id"],
            swap_id=stored["swap_id"],
            observed_limiting_pairs=menu.source.limiting_pairs,
            decision_rule="maximize_predicted_minimum_distance",
        )
        attempt = evaluate_selected_swap(
            context,
            menu,
            action,
            accepted_count + 1,
        )
        attempts.append(attempt)
        current = attempt.measurement
    return tuple(attempts), current
~~~

objective_start() verifies the fixed-file manifest, loads both private files, replays history, and returns the current terminal result or menu. It never creates or rewrites private state.

- [ ] **Step 7: Run Task 5 tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "checkpoint or objective_start or private_state or replay_history"
~~~

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 5**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add bounded workshop objective state"
~~~

### Task 6: Add objective selection, terminal evidence, and native figures

**Files:**

- Modify: acs_workshop_runner.py
- Modify: tests/test_acs_workshop_runner.py

- [ ] **Step 1: Write failing action and terminal tests**

Add tests that prove:

- objective-step accepts one exact current displayed tied maximum;
- it evaluates the new action exactly once before appending;
- stale state IDs, invented swaps, displayed non-maximum swaps, duplicate submissions, and fourth attempts do not change history;
- baseline optimal, zero-attempt target success, target achieved, no legal swap, and attempt-limit outcomes use the real terminal functions;
- terminal success writes exactly six objective files;
- objective_summary.json agrees with O01;
- score trajectory and heatmap are valid Matplotlib PNGs;
- final_panel.png is a valid PIL PNG made from fixed-dataset molecule indices; and
- a terminal rerun is idempotent.

Use this core acceptance assertion:

~~~python
assert set(path.name for path in objective_directory.iterdir()) == {
    "README.md",
    "objective_summary.json",
    "objective_evidence.json",
    "score_trajectory.png",
    "final_panel.png",
    "final_similarity_heatmap.png",
}
~~~

- [ ] **Step 2: Run objective tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  -k "objective_step or terminal_objective or objective_figure"
~~~

Expected: tests fail because action persistence and terminal output are missing.

- [ ] **Step 3: Implement the closed menu and envelope serializer**

Serialize nonterminal output with these exact top-level keys:

~~~python
{
    "schema_version": 1,
    "status": "pending",
    "terminal": False,
    "attempt_count": accepted_count,
    "attempt_limit": MAX_ATTEMPTS,
    "state_id": menu.state_id,
    "source": _measurement_payload(menu.source),
    "actions": [_action_payload(action) for action in menu.actions],
    "achieved": None,
    "termination_reason": None,
    "image_paths": [],
    "artifact_directory": str(objective_directory.resolve()),
    "results_zip_path": str(results_zip.resolve()),
    "artifact_relative_zip_path": "workshop/results.zip",
}
~~~

Measurement keys are selected_ids, score, score_key, limiting_pairs, and achieved. Action keys are swap_id, replace_id, replacement_id, resulting_ids, predicted_score, predicted_score_key, score_delta, limiting_pairs, and target_status. Require finite JSON and no hidden benchmark panel.

Recompute and normalize the two union-typed domain fields before serialization:

~~~python
baseline = measure_panel(run.context, run.context.baseline_ids)
reason = TerminationReason(run.termination_reason)
~~~

Serialize terminal output with these exact top-level keys:

~~~python
{
    "schema_version": 1,
    "status": "complete",
    "terminal": True,
    "attempt_count": len(run.attempts),
    "attempt_limit": MAX_ATTEMPTS,
    "baseline": _measurement_payload(baseline),
    "target_score": run.context.target_score,
    "target_score_key": score_key(run.context.target_score),
    "final": _measurement_payload(
        measure_panel(run.context, run.final_ids)
    ),
    "attempts": [_attempt_payload(attempt) for attempt in run.attempts],
    "achieved": run.achieved,
    "termination_reason": reason.value,
    "image_paths": [
        str((objective_directory / name).resolve())
        for name in (
            "score_trajectory.png",
            "final_panel.png",
            "final_similarity_heatmap.png",
        )
    ],
    "artifact_directory": str(objective_directory.resolve()),
    "results_zip_path": str(results_zip.resolve()),
    "artifact_relative_zip_path": "workshop/results.zip",
}
~~~

Attempt keys are attempt_number, state_id, selected_ids, score, score_key,
limiting_pairs, achieved, and selected_swap. selected_swap is the same closed
action payload returned by its menu. Add closed-key tests for a
baseline-terminal objective-start and a post-step terminal objective-step.

- [ ] **Step 4: Implement objective-step and atomic append**

After manifest, checkpoint, and history replay:

~~~python
menu = build_action_menu(context, current, len(attempts))
action = resolve_menu_action(
    context,
    menu,
    state_id=state_id,
    swap_id=swap_id,
    observed_limiting_pairs=menu.source.limiting_pairs,
    decision_rule="maximize_predicted_minimum_distance",
)
attempt = evaluate_selected_swap(
    context,
    menu,
    action,
    len(attempts) + 1,
)
~~~

Append only {"state_id": state_id, "swap_id": swap_id} through the mode-0600 atomic writer. Reject a terminal state and len(attempts) >= MAX_ATTEMPTS before evaluation. If the history write fails, do not publish terminal artifacts.

- [ ] **Step 5: Implement exact terminal resolution**

Resolve in this order:

1. baseline score key equals benchmark score key: baseline_terminal_run(context);
2. current achieved: terminal_objective_run(context, attempts, TerminationReason.TARGET_ACHIEVED);
3. attempt count equals MAX_ATTEMPTS: terminal_objective_run(context, attempts, TerminationReason.ATTEMPT_LIMIT_REACHED);
4. current menu has no actions: finalize_no_legal_swap(context, attempts, current, menu);
5. otherwise return the pending menu.

Do not manufacture provider-failure or correction-limit states in this local runner.
Use one terminal-publication helper from both objective-start and
objective-step. This makes a baseline-optimal objective publish immediately,
and it lets objective-start reconstruct and republish a valid terminal result
if history was committed but a prior terminal artifact publication failed.

- [ ] **Step 6: Write O01, reconstruct RDKit molecules, and save figures**

Extend the objective_challenge imports with build_objective_evidence and
objective_figures. Call build_objective_evidence(run) and parse its
payload_json.

Write objective_summary.json as:

~~~python
{
    "schema_version": 1,
    **json.loads(record.payload_json),
}
~~~

Write objective_evidence.json as:

~~~python
{
    "schema_version": 1,
    "evidence": {
        "key": record.key,
        "label": record.label,
        "payload": json.loads(record.payload_json),
        "provenance": record.provenance,
    },
}
~~~

Rebuild only the fixed library:

~~~python
render_state = WorkflowState()
inspect_library(render_state, paths.dataset_path, expected_rows=256)
trajectory, structures, heatmap = objective_figures(run, render_state)
~~~

Save trajectory and heatmap with the Matplotlib adapter and structures with the PIL adapter. Publish the exact objective directory and updated complete ZIP through the existing rollback transaction.

- [ ] **Step 7: Keep facilitator verification outside the production runner**

Do not add archive, transcript, state-log, live-acceptance, or facilitator
receipt functions to acs_workshop_runner.py. Its only responsibilities remain
the three in the approved design: execute a fixed stage prefix, serialize fixed
artifacts, and expose the bounded objective CLI. Task 9 implements acceptance
verification in a separate facilitator-only script.

- [ ] **Step 8: Run Task 6 tests and the real objective regression suite**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_runner.py \
  tests/test_objective_challenge.py
~~~

Expected: all tests pass.

- [ ] **Step 9: Commit Task 6**

Run:

~~~bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add workshop objective challenge artifacts"
~~~

### Task 7: Wire the runner and trusted manifest into the NemoClaw setup

**Files:**

- Modify: launchable/acs_nemoclaw_launchable_setup.sh
- Modify: launchable/acs_workspace_tools.md
- Modify: tests/test_acs_nemoclaw_launchable_setup.py
- Modify: tests/test_nemoclaw_phase_zero_setup.py

- [ ] **Step 1: Write failing setup source-contract tests**

Extend required_assets, cleanup, upload, hash, and post-seed assertions for:

~~~text
acs_workshop_runner.py
objective_challenge.py
~~~

Require exactly ten sandbox uploads. Require cleanup of the two fixed files, .acs-workshop-state, and outputs/workshop before upload on a full setup rerun.

Add ordering assertions:

~~~python
assert source.index('expected_runner_sha="$(host_sha256 "$workshop_runner")"') < agent_turn
assert source.index('expected_objective_sha="$(host_sha256 "$objective_challenge")"') < agent_turn
assert source.index('"acs_workshop_runner.py": os.environ["ACS_EXPECTED_RUNNER_SHA"]') > agent_turn
assert source.index("manifest.json") > agent_turn
assert source.index("chmod 0444") < source.index("ACS chemistry workspace is ready")
assert source.index("acs_workshop_runner.py --help") < source.index("printf 'ready\\n'")
~~~

Require the setup to keep the current threshold-0.80 seed prompt, exact artifact set, proxy ports, and threshold ZIP probe unchanged. Require no workshop ZIP probe before an attendee runs the stages.

- [ ] **Step 2: Run setup tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py
~~~

Expected: the new runner and manifest assertions fail.

- [ ] **Step 3: Add setup variables, required assets, cleanup, hashes, and uploads**

Add:

~~~bash
readonly workshop_runner="$project_dir/acs_workshop_runner.py"
readonly objective_challenge="$project_dir/objective_challenge.py"
~~~

Add both to required_assets. Add the two workspace files plus .acs-workshop-state and outputs/workshop to the exact setup-owned cleanup list.

Compute hashes with host_sha256 before upload:

~~~bash
expected_runner_sha="$(host_sha256 "$workshop_runner")"
expected_objective_sha="$(host_sha256 "$objective_challenge")"
readonly expected_runner_sha expected_objective_sha
~~~

Upload both to the existing workspace parent. Add both files to the post-seed workspace_hashes validation. Keep acs_chemistry_task.py writable for the existing one-line seed edit.

- [ ] **Step 4: Create the canonical fixed-file manifest from host-derived hashes**

After post-seed validation, create .acs-workshop-state with mode 0700. Pass the five expected hashes as environment values to a bounded in-sandbox Python command. It writes:

~~~json
{
  "schema_version": 1,
  "files": {
    "TOOLS.md": "host-derived hash",
    "acs_workshop_runner.py": "host-derived hash",
    "chemistry_workflow.py": "host-derived hash",
    "data/sample_molecules.csv": "host-derived hash",
    "objective_challenge.py": "host-derived hash"
  }
}
~~~

The Python command must use a same-directory O_EXCL temporary file, mode 0600, flush, fsync, os.replace, and final chmod 0444. It must reject a pre-existing symlink or non-regular manifest and must write canonical JSON plus one newline. Do not derive authoritative values from sandbox bytes.

- [ ] **Step 5: Set fixed inputs read-only and run smoke checks**

Run chmod 0444 on:

~~~text
acs_workshop_runner.py
objective_challenge.py
chemistry_workflow.py
data/sample_molecules.csv
TOOLS.md
.acs-workshop-state/manifest.json
~~~

Then run, with the fixed PYTHONPATH:

~~~bash
python3 -c 'import acs_workshop_runner, objective_challenge'
python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py --help
~~~

Both checks occur before the ready marker. The help path verifies the manifest before printing help.

- [ ] **Step 6: Replace the generic TOOLS.md task rules**

Replace the instruction to create Python source and one ZIP per task. State:

- only acs_workshop_runner.py creates workshop files and workshop/results.zip;
- the six exact run-stage commands;
- objective-start and the quoted objective-step form;
- the fixed dataset, output, and Download Results paths;
- no edits to runner, workflow, objective, dataset, TOOLS.md, or manifest;
- no installs, network use, arbitrary paths, repairs, or alternate commands;
- fingerprints, similarity, clusters, sampled conformers, and force-field energy have the approved scientific limits; and
- only the setup-owned acs_task_prompt.txt may make its exact seed edit to acs_chemistry_task.py.

- [ ] **Step 7: Run setup and protected regression tests**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py \
  tests/test_acs_chemistry_task.py \
  tests/test_openclaw_secure_link_proxy.py

bash -n \
  launchable/acs_nemoclaw_launchable_setup.sh \
  launchable/acs_console_bootstrap.sh.in \
  launchable/start_artifact_server.sh
~~~

Expected: all Python tests pass and Bash syntax exits zero.

- [ ] **Step 8: Commit Task 7**

Run:

~~~bash
git add \
  launchable/acs_nemoclaw_launchable_setup.sh \
  launchable/acs_workspace_tools.md \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py
gitleaks git --staged --no-banner --redact .
git commit -m "Wire workshop runner into NemoClaw setup"
~~~

### Task 8: Create the seven canonical, self-contained OpenClaw prompts

**Files:**

- Create: launchable/acs_workshop_prompts.md
- Create: tests/test_acs_workshop_prompts.py

- [ ] **Step 1: Write the failing prompt parser and contract tests**

Parse these marker IDs in this exact order:

~~~python
PROMPT_IDS = (
    "01-inspection",
    "02-fingerprints",
    "03-similarity",
    "04-clusters",
    "05-conformers",
    "06-mmff94",
    "07-objective",
)
~~~

Use markers:

~~~text
<!-- ACS_PROMPT:01-inspection:BEGIN -->
<!-- ACS_PROMPT:01-inspection:END -->
~~~

The parser requires exactly one tilde-fenced text block between each pair. Add tests that prompts 1 through 6 contain their exact run-stage command once, skill path, prohibitions, stop-on-error rule, workshop/results.zip, and exact MEDIA image.

For prompt 7 require objective-start once, the objective-step template once, STATE_ID_FROM_MENU, SWAP_ID_FROM_MENU, quoted values, no more than three steps, terminal stop, and the three evidence files.

Reject 18789, /org/ environment URLs, gateway-token, BuildDoneVideo, any secret-shaped nvapi value, package installation, curl, wget, pip, conda, and arbitrary output or dataset options.

- [ ] **Step 2: Run prompt tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_prompts.py
~~~

Expected: collection or file-existence failure because the prompt source does not exist.

- [ ] **Step 3: Create prompts 1 through 6 with the exact fixed form**

For each prompt, use the marker pair and one text fence. Use this exact sentence
order:

1. Question followed by the exact Question cell below.
2. Work only in /sandbox/.openclaw/workspace.
3. Read /sandbox/.openclaw/skills/nvmolkit-usage/SKILL.md before chemistry work.
4. Do not install packages, use the network, edit setup-verified fixed files,
   or run another command.
5. Run this command exactly once, followed by the fixed PYTHONPATH, runner path,
   run-stage, and exact Stage cell below.
6. If it fails, report the exact error and stop. Do not repair or retry.
7. On success, report the exact Facts cell and Scientific limit cell below.
8. Say that the complete bundle is available through Download Results at
   workshop/results.zip.
9. End with one MEDIA line using the exact Image path cell below.

Use every table cell verbatim:

| ID | Question | Stage | Facts | Scientific limit | Image path |
| --- | --- | --- | --- | --- | --- |
| 01-inspection | What is in the fixed molecule library? | inspect_library | raw, valid, invalid, and preview counts | validation does not establish activity or suitability | outputs/workshop/01-inspection/library_preview.png |
| 02-fingerprints | What do the GPU Morgan fingerprints show? | generate_morgan_fingerprints | radius, size, packed shape, active-bit minimum, median, maximum, and GPU | fingerprints are structural descriptors, not biological evidence | outputs/workshop/02-fingerprints/fingerprint_density.png |
| 03-similarity | Which molecules are most similar in this fingerprint space? | measure_tanimoto_similarity | top non-self pair, similarity, quartiles, p90, and GPU | similarity does not establish activity, binding, efficacy, or safety | outputs/workshop/03-similarity/similarity_heatmap.png |
| 04-clusters | How does fused Butina partition the library? | discover_fused_butina_clusters | cutoff, cluster count, singleton count, largest cluster sizes, and GPU | clusters depend on this fingerprint and cutoff | outputs/workshop/04-clusters/cluster_sizes.png |
| 05-conformers | Did ETKDGv3 generate the requested representative conformers? | embed_representative_conformers | selected representatives, requested and generated counts, partial or zero IDs, seed, and GPU | sampled conformers are not experimental structures | outputs/workshop/05-conformers/embedding_counts.png |
| 06-mmff94 | Which sampled conformers converged under MMFF94? | optimize_conformers_mmff94 | attempted, converged, unconverged, within-molecule minimum energies, maximum iterations, and GPU | MMFF94 compares sampled force-field geometries within each molecule only | outputs/workshop/06-mmff94/optimized_structures.png |

- [ ] **Step 4: Create the complete bounded objective prompt**

The seventh prompt must:

1. ask whether the bounded agent can improve the four-compound panel minimum pairwise Morgan/Tanimoto distance;
2. use the same workspace, skill, no-install, no-network, no-edit, and stop-on-error rules;
3. run objective-start exactly once;
4. at each pending menu, report current score and all co-limiting pairs;
5. select only one displayed action with the highest predicted score;
6. substitute exact returned values into:

~~~text
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-step --state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'
~~~

7. state that the swap ID can contain -> and must remain quoted;
8. run at most three objective-step commands and stop immediately on terminal status;
9. ground the final answer only in:

~~~text
outputs/workshop/06-mmff94/workflow_evidence.json
outputs/workshop/07-objective/objective_summary.json
outputs/workshop/07-objective/objective_evidence.json
~~~

10. report baseline, target, final score, accepted swaps, terminal reason, final panel, and the structural-diversity-only limit;
11. name Download Results at workshop/results.zip; and
12. end with:

~~~text
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png
~~~

- [ ] **Step 5: Run prompt tests and verify GREEN**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_prompts.py
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit Task 8**

Run:

~~~bash
git add launchable/acs_workshop_prompts.md tests/test_acs_workshop_prompts.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add tested ACS workshop prompts"
~~~

### Task 9: Add the bounded facilitator acceptance verifier

**Files:**

- Create: scripts/verify_acs_workshop_acceptance.py
- Create: tests/test_verify_acs_workshop_acceptance.py
- Read: acs_workshop_runner.py

- [ ] **Step 1: Write failing CLI, transcript, state-log, and receipt tests**

Require exactly four commands:

~~~text
workspace
archive PATH
transcript
state-logs
~~~

No command accepts a shell command, URL, session key, state root, output root,
model, retry, or network option.

Create bounded fake sessions.json and JSONL fixtures for the exact session key:

~~~text
agent:main:explicit:acs-workshop-live
~~~

Add tests that prove:

- workspace runs the facilitator script's own verify_completed_workshop() and
  emits only its closed receipt;
- archive runs the facilitator script's own verify_downloaded_archive() for one
  regular non-symlink file and emits only its closed receipt;
- workspace and archive return the same closed safe receipt for the same
  completed public bytes;
- either mode rejects a changed GPU name, summary value, artifact-manifest
  hash, CSV identifier, SDF record ID, objective value, unsafe ZIP member,
  private-state member, or secret-shaped value;
- verify_completed_workshop() calls acs_workshop_runner.verify_manifest(paths)
  before reading any output or private checkpoint; mutating any one
  setup-verified file after artifact creation must fail;
- transcript resolves one sessionFile below the fixed main-agent sessions root,
  loads one regular read-only canonical prompt file from PROMPTS_PATH, requires
  the exact seven prompt bytes and hashes in order, requires at least seven
  assistant text messages, and verifies the bounded tool protocol below;
- state-logs scans only the two fixed ACS state directories and emits the exact
  closed receipt below;
- transcript and state-logs still pass when importing acs_workshop_runner is
  deliberately blocked, proving that host-only modes do not load sandbox
  chemistry packages;
- missing, duplicate, malformed, oversized, symlinked, or out-of-root inputs
  fail with one generic Error line and no file content;
- each secret class fails when the test constructs it from harmless string
  fragments: an nvapi value, a populated gateway-token field, a URL query or
  fragment token, a tokenized dashboard URL, or a private-key block; and
- ordinary chemistry IDs, MEDIA paths, public Launchable URLs, and the words
  API key do not create false positives.

Use a sanitized fixture with the pinned OpenClaw 2026.7.1 message shape:
message rows contain message.role plus string or list message.content; tool
calls are content blocks with type toolCall, name, id, and arguments. Also test
the exact NemoClaw compact-catalog wrapper, tool_call, with closed name and args
fields. Reject every other tool-call spelling, open field, or malformed
argument instead of guessing.

Normalize only Read and Exec. The accepted transcript must contain, in order:

1. the six exact `run-stage` Exec command strings, once each;
2. the exact `objective-start` Exec command once; and
3. zero to three `objective-step` Exec commands whose single-quoted state ID
   and swap ID equal the corresponding validated history records.

Each of the seven turns must read the exact installed nvMolKit skill path once.
Additional Read calls are allowed only for that turn's applicable path in the
exact OPTIONAL_PUBLIC_READS tuple defined below. Reject private-state reads, source reads,
another tool, another Exec command, compound shell syntax, installs, network
commands, retries, or calls after terminal objective status. Mutations in the
test suite must prove that a changed prompt, reordered stage, extra Exec,
unquoted objective value, non-history swap, network command, or extra tool
fails closed.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_verify_acs_workshop_acceptance.py
~~~

Expected: collection fails because the verifier script does not exist.

- [ ] **Step 3: Implement the closed verifier CLI**

Use only the standard library at module import time. Workspace and archive may
lazy-load acs_workshop_runner and the scientific packages already pinned by the
workshop; transcript and state-logs may not. Define:

~~~python
SESSION_KEY = "agent:main:explicit:acs-workshop-live"
SESSIONS_ROOT = Path("/sandbox/.openclaw/agents/main/sessions")
PROMPTS_PATH = Path("/tmp/acs-workshop-acceptance/acs_workshop_prompts.md")
HISTORY_PATH = Path(
    "/sandbox/.openclaw/workspace/.acs-workshop-state/history.json"
)
STATE_ROOTS = (
    Path.home() / ".local/state/acs-phase-zero",
    Path.home() / ".local/state/acs-nemoclaw-launchable",
)
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024
MAX_TRANSCRIPT_LINES = 4096
MAX_STATE_FILES = 256
MAX_STATE_FILE_BYTES = 4 * 1024 * 1024
MAX_STATE_TOTAL_BYTES = 16 * 1024 * 1024
~~~

Import acs_workshop_runner lazily inside workspace and archive only. The module
top level and the transcript and state-logs call paths must not import RDKit,
pandas, nvMolKit, or acs_workshop_runner.

Define the exact Exec prefix and commands:

~~~python
RUNNER_PREFIX = (
    "env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 "
    "/sandbox/.openclaw/workspace/acs_workshop_runner.py"
)
STAGE_COMMANDS = tuple(
    f"{RUNNER_PREFIX} run-stage {stage}"
    for stage in (
        "inspect_library",
        "generate_morgan_fingerprints",
        "measure_tanimoto_similarity",
        "discover_fused_butina_clusters",
        "embed_representative_conformers",
        "optimize_conformers_mmff94",
    )
)
OBJECTIVE_START_COMMAND = f"{RUNNER_PREFIX} objective-start"
SKILL_PATH = "/sandbox/.openclaw/skills/nvmolkit-usage/SKILL.md"
OPTIONAL_PUBLIC_READS = (
    *(f"/sandbox/.openclaw/workspace/outputs/workshop/{index:02d}-{slug}/summary.json"
      for index, slug in (
          (1, "inspection"),
          (2, "fingerprints"),
          (3, "similarity"),
          (4, "clusters"),
          (5, "conformers"),
          (6, "mmff94"),
      )),
    "/sandbox/.openclaw/workspace/outputs/workshop/06-mmff94/workflow_evidence.json",
    "/sandbox/.openclaw/workspace/outputs/workshop/07-objective/objective_summary.json",
    "/sandbox/.openclaw/workspace/outputs/workshop/07-objective/objective_evidence.json",
)
~~~

The objective-step parser must accept only this anchored byte form, with one
space between tokens and no newline or shell metacharacter:

~~~text
<RUNNER_PREFIX> objective-step --state-id '<64 lowercase hex>' --swap-id '<validated history swap ID>'
~~~

Transcript mode validates HISTORY_PATH with a small standard-library closed
schema parser. Require only schema_version, dataset_sha256, context_sha256, and
accepted_steps; each accepted step has exactly state_id and swap_id. Workspace
mode remains responsible for full scientific replay. This keeps transcript
mode independent of sandbox chemistry imports while binding every observed
objective-step command to the accepted history.

Parse the seven prompt blocks with the same exact marker/fence rules as Task 8.
Hash each extracted UTF-8 block, then require each role=user message to equal
the corresponding block byte-for-byte. Do not accept normalized whitespace or
a count-only match.

Expose one non-CLI helper for the Task 11 orchestrator:

~~~python
def extract_prompt_block(
    source_path: str,
    prompt_id: str,
    expected_sha256: str,
) -> str:
    ...
~~~

It accepts only one ID from the fixed seven-ID tuple, applies the exact
marker/fence parser, requires the expected digest to be 64 lowercase hex, and
returns the block only when its SHA-256 matches. Add duplicate-marker,
unknown-ID, wrong-hash, symlink, oversized-file, and success tests. It does not
print, normalize, or cache the prompt.

Define this facilitator-only immutable result in the verifier script, not in
acs_workshop_runner.py:

~~~python
@dataclass(frozen=True)
class WorkshopVerification:
    archive_sha256: str
    archive_size: int
    stage_count: int
    objective_attempt_count: int
    objective_termination_reason: str
    gpu_name: str
~~~

verify_completed_workshop(paths) must first lazily import the runner and call
runner.verify_manifest(paths). Only after that succeeds may it read the six
exact stage directories, 07-objective, artifact_manifest.json, results.zip,
and the two private checkpoint files. Apply every closed schema and cross-file
invariant from the runner tests, verify manifest bytes and ZIP bytes, require
NVIDIA L4 for stages 2 through 6, verify history replay and terminal O01
evidence, and return WorkshopVerification without printing content.

verify_downloaded_archive(path) lazily imports the runner only after validating
one regular non-symlink ZIP path. Open without extracting. Enforce a maximum
archive size of 64 MiB, at most 128 members, at most 64 MiB total uncompressed
data, a maximum 16 MiB per member, exact deterministic metadata, safe names,
the complete public file set, artifact-manifest hashes, closed summaries, PNGs
through Pillow, CSV and SDF provenance through pandas and RDKit
ForwardSDMolSupplier over BytesIO, and objective evidence agreement. Reject
private state and secret-shaped values.

Both functions reject symlinked path components, duplicate ZIP names,
encryption, comments, extra fields, unsupported compression, trailing bytes,
and any extra file. They never execute or extract archive contents.

For transcript:

1. require sessions.json and its selected sessionFile to be regular
   non-symlink files;
2. require the selected sessionFile to resolve below SESSIONS_ROOT;
3. enforce byte and line bounds before parsing;
4. parse each line as one JSON object;
5. recursively collect strings only from message rows;
6. load PROMPTS_PATH only after a component-walk rejects symlinks, require one
   regular file with no group/other/write bit, parse seven canonical blocks,
   and require the seven role=user message strings to match them byte-for-byte
   in order;
7. normalize the two exact pinned tool-call envelopes, require one SKILL_PATH
   Read per turn, allow only OPTIONAL_PUBLIC_READS, and enforce STAGE_COMMANDS,
   OBJECTIVE_START_COMMAND, and the history-bound objective-step commands in
   their exact order;
8. count role=assistant rows that contain text and reject a tool error, timeout,
   nonzero Exec result, or tool call after terminal objective state;
9. scan all collected text in memory for the fixed secret patterns;
10. discard prompt, answer, tool-argument, and tool-result strings after
    hashing; and
11. return this exact receipt:

~~~python
{
    "schema_version": 1,
    "status": "pass",
    "mode": "transcript",
    "session_key_sha256": hashlib.sha256(SESSION_KEY.encode()).hexdigest(),
    "transcript_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    "canonical_prompts_sha256": canonical_prompts_sha256,
    "user_message_count": 7,
    "assistant_text_message_count": assistant_count,
    "exec_call_count": 7 + objective_step_count,
    "objective_step_count": objective_step_count,
    "tool_policy": "pass",
    "secret_scan": "pass",
}
~~~

For state-logs, walk only STATE_ROOTS without following links. Reject special
files, enforce the fixed file and byte bounds, scan bytes in memory, sort by
root label and relative path, and hash each relative name, size, and content
hash into one aggregate digest. Never return a path or content. Return exactly:

~~~python
{
    "schema_version": 1,
    "status": "pass",
    "mode": "state-logs",
    "file_count": file_count,
    "total_bytes": total_bytes,
    "aggregate_sha256": aggregate_sha256,
    "secret_scan": "pass",
}
~~~

Require the exact seven keys, bool-excluding integer types for both counts,
nonnegative bounded values, and a 64-character lowercase SHA-256. Add
unexpected-key, missing-key, wrong-type, negative, and digest-mutation tests.

For workspace and archive, convert the WorkshopVerification dataclass to this
closed receipt:

~~~python
{
    "schema_version": 1,
    "status": "pass",
    "mode": mode,
    "archive_sha256": receipt.archive_sha256,
    "archive_size": receipt.archive_size,
    "stage_count": receipt.stage_count,
    "objective_attempt_count": receipt.objective_attempt_count,
    "objective_termination_reason": receipt.objective_termination_reason,
    "gpu_name": receipt.gpu_name,
    "secret_scan": "pass",
}
~~~

main() prints one canonical JSON receipt and returns 0, or prints one generic
Error line without an exception repr or stored value and returns 2.

- [ ] **Step 4: Run mutation and GREEN tests**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_verify_acs_workshop_acceptance.py \
  tests/test_acs_workshop_runner.py \
  -k "acceptance or verify_completed_workshop or verify_downloaded_archive"
~~~

Expected: all selected tests pass.

Run:

~~~bash
/private/tmp/nvmolkit-ui-venv312/bin/ruff format --check \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_verify_acs_workshop_acceptance.py
~~~

~~~bash
/private/tmp/nvmolkit-ui-venv312/bin/ruff check \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_verify_acs_workshop_acceptance.py
~~~

~~~bash
env PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/mypy \
  --strict \
  --ignore-missing-imports \
  scripts/verify_acs_workshop_acceptance.py
~~~

Expected: every static command exits zero.

- [ ] **Step 5: Commit Task 9**

Run:

~~~bash
git add \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_verify_acs_workshop_acceptance.py
gitleaks git --staged --no-banner --redact .
git commit -m "Add bounded ACS workshop acceptance verifier"
~~~

### Task 10: Run the complete local verification and independent reviews

**Files:**

- Verify: every file changed in Tasks 1 through 9
- Modify: tests/test_demo_agent.py
- Compare: docs/superpowers/specs/2026-08-11-acs-fall-2026-attendee-workshop-design.md

- [ ] **Step 1: Isolate the two notebook key-preflight tests from host CUDA packages**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_demo_agent.py::test_notebook_preflight_fails_closed_when_launch_key_is_missing \
  tests/test_demo_agent.py::test_notebook_preflight_rejects_launch_key_with_unsafe_permissions
~~~

Expected before repair: both tests fail because this local test environment has
no torch package, so they do not reach the key behavior they claim to test.

Add one local helper in tests/test_demo_agent.py that monkeypatches imports for
a CUDA-available torch object and the five exact nvMolKit entry points already
used by test_notebook_preflight_checks_cuda_and_exact_nvmolkit_capabilities().
Call it at the start of only the two key-preflight tests. Do not change
demo_agent.py and do not weaken the CUDA capability test.

Rerun the exact command. Expected: two tests pass. Then commit:

~~~bash
git add tests/test_demo_agent.py
git commit -m "Isolate notebook key preflight tests"
~~~

- [ ] **Step 2: Run the protected focused regression suite**

Run one low-memory Python test process:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_chemistry_task.py \
  tests/test_chemistry_workflow.py \
  tests/test_objective_challenge.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py \
  tests/test_openclaw_secure_link_proxy.py \
  tests/test_acs_workshop_runner.py \
  tests/test_acs_workshop_prompts.py \
  tests/test_verify_acs_workshop_acceptance.py \
  tests/test_demo_agent.py::test_notebook_preflight_fails_closed_when_launch_key_is_missing \
  tests/test_demo_agent.py::test_notebook_preflight_rejects_launch_key_with_unsafe_permissions
~~~

Expected: every selected test passes. A failure in an existing seed-task, timeout, proxy, upload, cleanup, or objective test blocks deployment.

- [ ] **Step 3: Run the complete repository suite**

Run:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q
~~~

Expected: every collected test passes or has an existing explicit skip. Record
the exact pass and skip counts. Mocked local tests prove control behavior only;
they do not prove live CUDA, L4 hardware, hosted inference, or browser UX.

- [ ] **Step 4: Run Node, Python, Bash, format, lint, and type gates**

Run each command separately:

~~~bash
node --test tests/openclaw_secure_link_proxy.test.mjs
~~~

~~~bash
node --check launchable/openclaw_secure_link_proxy.mjs
~~~

~~~bash
env PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/python -m py_compile \
  acs_workshop_runner.py \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_acs_workshop_runner.py \
  tests/test_acs_workshop_prompts.py \
  tests/test_verify_acs_workshop_acceptance.py
~~~

~~~bash
bash -n \
  launchable/acs_nemoclaw_launchable_setup.sh \
  launchable/acs_console_bootstrap.sh.in \
  launchable/nemoclaw_phase_zero.sh \
  launchable/start_artifact_server.sh
~~~

~~~bash
/private/tmp/nvmolkit-ui-venv312/bin/ruff format --check \
  acs_workshop_runner.py \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_acs_workshop_runner.py \
  tests/test_acs_workshop_prompts.py \
  tests/test_verify_acs_workshop_acceptance.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_demo_agent.py
~~~

~~~bash
/private/tmp/nvmolkit-ui-venv312/bin/ruff check \
  acs_workshop_runner.py \
  scripts/verify_acs_workshop_acceptance.py \
  tests/test_acs_workshop_runner.py \
  tests/test_acs_workshop_prompts.py \
  tests/test_verify_acs_workshop_acceptance.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_demo_agent.py
~~~

~~~bash
env PYTHONPATH=. /private/tmp/nvmolkit-ui-venv312/bin/mypy \
  --strict \
  --ignore-missing-imports \
  acs_workshop_runner.py \
  scripts/verify_acs_workshop_acceptance.py
~~~

Expected: every command exits zero.

- [ ] **Step 5: Run repository and secret gates**

Run:

~~~bash
git diff --check
~~~

~~~bash
gitleaks git \
  --log-opts="origin/acs-fall-2026-launchable..HEAD" \
  --no-banner \
  --redact \
  .
~~~

~~~bash
git status --short --branch
~~~

Expected: diff check and Gitleaks pass. Status shows only intentional, committed workshop changes and no generated outputs, secrets, private state, or test cache.

- [ ] **Step 6: Perform a read-only specification review**

Give an independent reviewer the approved specification, this plan, and the exact implementation commit. Require findings first and only Critical or Important contract gaps. The reviewer must inspect:

- the fixed CLI and manifest boundary;
- every stage output and scientific limit;
- private objective state and replay;
- rollback and ZIP safety;
- setup upload, permissions, ports, token, and timeout behavior;
- prompt identity and bounded commands; and
- the explicit public-push and Brev approval gates.

If a finding is valid, add a failing focused test before the smallest repair,
rerun the affected task gate, and then rerun Steps 2 through 5.

- [ ] **Step 7: Perform a read-only quality and security review**

Use a different reviewer. Require checks for:

- path traversal, symlink following, unsafe deletion, and rollback loss;
- open JSON schemas, non-finite numbers, mutable trusted files, and checkpoint reset;
- arbitrary command, dataset, output, network, or package-install paths;
- secret, gateway-token, raw-provider-response, and tokenized-URL exposure;
- a fourth objective action or acceptance of a non-maximum action; and
- claims that local tests prove live GPU, model, browser, or download behavior.

Repair valid findings with the same RED to GREEN process and rerun Steps 2
through 5.

- [ ] **Step 8: Record the clean local implementation commit**

Run:

~~~bash
git log -1 --format='%H %s'
~~~

~~~bash
git status --short --branch
~~~

Expected: record one exact reviewed 40-character implementation commit and a clean worktree. Stop here and obtain explicit approval before a public push, Launchable edit, deployment, instance access, or lifecycle action.

### Task 11: Publish the reviewed implementation and qualify one fresh Brev L4

**Files:**

- Read: launchable/acs_console_bootstrap.sh.in
- Read: launchable/ACS_LAUNCHABLE_FIELDS.md
- Read: launchable/acs_workshop_prompts.md
- Download for inspection: one fresh outputs/workshop/results.zip

**External contract:**

- Organization: the exact private organization that owns the existing
  Launchable, recorded only in the local acceptance note
- Launchable: env-3Hlp4pHBlTTlfDxfH41KkGhTeCV
- One fresh instance owned only by this acceptance run
- Hardware: one NVIDIA L4, x86-64, 4 CPUs, 16 GiB RAM, 128 GiB disk
- Secure Links: Open Chemistry Agent on 18788 and Download Results on 8765
- Raw 18789 stays private
- Lifecycle authority: stop and delete only the exact fresh acceptance instance after evidence is preserved
- Organization switching: prohibited

ACS_PUBLIC_COMMIT, ACS_ORG_ID, and ACS_INSTANCE_ID below are plan notation.
Codex shell calls use fresh processes, so execution must render the exact
validated literal into every independent command. Never rely on a shell
variable from an earlier tool call. Recheck the organization and instance
literal immediately before stop or delete.

After the fresh instance is pinned in Step 7, Steps 8 through 13 run under one
finally-style lifecycle guard. Success, test failure, hosted-model timeout,
user interruption, or the time/cost ceiling all enter Step 14. Before cleanup,
preserve only already-approved bounded receipts and non-secret diagnostics; do
not keep a failing VM alive for investigation. Then stop and delete only the
exact pinned instance under the approved authority. If cleanup itself fails,
stop all other work and report the still-billable exact instance immediately.

- [ ] **Step 1: Obtain the public-push and fresh-instance approval**

Show the user:

- the exact reviewed local commit;
- the branch and public repository target;
- the Launchable ID;
- the exact organization;
- one-instance L4 hardware and current displayed hourly price;
- a hard stop at the earlier of six elapsed hours or a projected total compute
  charge of 6 USD;
- the two Secure Links;
- the planned remote read/write scope; and
- the exact stop and delete cleanup authority.

Proceed only after explicit approval covers the public push, Launchable update, one fresh billable deployment, remote acceptance, and exact-instance stop and delete. Approval does not permit an organization switch or access to another instance.

- [ ] **Step 2: Recheck the installed Brev control surface**

Run each command separately and record the output:

~~~bash
/opt/homebrew/bin/brev --version
~~~

~~~bash
/opt/homebrew/bin/brev exec --help
~~~

~~~bash
/opt/homebrew/bin/brev ls --help
~~~

~~~bash
/opt/homebrew/bin/brev copy --help
~~~

~~~bash
/opt/homebrew/bin/brev stop --help
~~~

~~~bash
/opt/homebrew/bin/brev delete --help
~~~

Expected for the currently qualified client: version v0.6.332; exec accepts one named instance and one command string; ls supports --org; exec, copy, stop, and delete do not support --org. If the installed help differs, stop and revise these commands. Do not guess and do not upgrade the CLI.

- [ ] **Step 3: Verify the active organization without changing it**

Run:

~~~bash
/opt/homebrew/bin/brev ls orgs --json
~~~

From that read-only result, set ACS_ORG_ID to the exact organization that owns
the Launchable and confirm it is already active. Record the resolved value only
in the mode-0600 acceptance note. Then run:

~~~bash
/opt/homebrew/bin/brev ls \
  --org "$ACS_ORG_ID" \
  --json
~~~

Expected: the exact pinned organization exists and is already the active organization used by commands that lack --org. If it is not active, stop and ask the user to serialize the Brev context. Do not run brev set, brev org set, login, logout, reset, or refresh.

- [ ] **Step 4: Push the exact reviewed branch and verify the remote commit**

Bind ACS_PUBLIC_COMMIT to the exact reviewed commit before pushing:

~~~bash
ACS_PUBLIC_COMMIT="$(git rev-parse HEAD)"
readonly ACS_PUBLIC_COMMIT
~~~

Verify it equals the 40-character commit recorded in Task 10. Then run:

~~~bash
git push origin acs-fall-2026-launchable
~~~

~~~bash
git ls-remote origin refs/heads/acs-fall-2026-launchable
~~~

Expected: the remote branch hash equals the exact reviewed local implementation commit. If it differs, stop before any Launchable edit.

- [ ] **Step 5: Render and verify the exact-commit Console bootstrap**

First prove that the reviewed template has one replacement site:

~~~bash
python3 -c 'from pathlib import Path; p=Path("launchable/acs_console_bootstrap.sh.in"); text=p.read_text(encoding="utf-8"); assert text.count("@REVIEWED_PUBLIC_COMMIT_SHA@") == 1; assert "fa9154175e4783264d8fd8b07610d344b618576b" not in text; print("BOOTSTRAP_TEMPLATE_OK")'
~~~

Copy launchable/acs_console_bootstrap.sh.in to
/private/tmp/acs-console-bootstrap.sh. Then use apply_patch for exactly this
one-line replacement, rendering the reviewed 40-character commit literally:

~~~diff
-readonly repo_commit="@REVIEWED_PUBLIC_COMMIT_SHA@"
+readonly repo_commit="REVIEWED_40_HEX_PUBLIC_COMMIT_LITERAL"
~~~

Do not change the repository URL or any other template byte. Assert the
rendered literal is the recorded Task 10 commit, then verify:

~~~bash
bash -n /private/tmp/acs-console-bootstrap.sh
~~~

~~~bash
LC_ALL=C wc -c < /private/tmp/acs-console-bootstrap.sh
~~~

~~~bash
python3 -c 'from pathlib import Path; import sys; text = Path(sys.argv[1]).read_text(encoding="utf-8"); commit = sys.argv[2]; assert "@REVIEWED_PUBLIC_COMMIT_SHA@" not in text; assert "fa9154175e4783264d8fd8b07610d344b618576b" not in text; assert text.count(commit) == 1; print("BOOTSTRAP_COMMIT_OK")' \
  /private/tmp/acs-console-bootstrap.sh \
  "$ACS_PUBLIC_COMMIT"
~~~

Expected: Bash syntax passes; the payload is no more than 16384 bytes; neither the template token nor the prior commit remains; the reviewed commit occurs exactly once; and no key value is present.

- [ ] **Step 6: Update the existing Launchable in the Brev Console**

No supported, callable Launchable-authoring API is available in this task. Give the user the rendered bootstrap and ask them to update only Launchable env-3Hlp4pHBlTTlfDxfH41KkGhTeCV in the signed-in Console.

They must preserve:

- VM mode;
- Source set to I do not have any code files;
- the required NVIDIA_INFERENCE_API_KEY text parameter with no default;
- the accepted one-L4 default resource row;
- Secure Link 18788 named Open Chemistry Agent;
- Secure Link 8765 named Download Results;
- no Secure Link for 18789; and
- the current staged access policy.

Ask the user to save, then deploy one fresh instance from the exact Launchable URL. Do not use private Console endpoints. The user supplies the exact new instance name or ID.

- [ ] **Step 7: Pin and verify the exact fresh instance**

Set the local shell variable ACS_INSTANCE_ID to the exact fresh value supplied
by the user. Record the resolved value in the private acceptance note before
any remote command. Do not use a sample value, name prefix, glob, or command
substitution to select the instance. Then run:

~~~bash
/opt/homebrew/bin/brev ls \
  --org "$ACS_ORG_ID" \
  --json
~~~

Expected: exactly one entry matches ACS_INSTANCE_ID, it belongs to the approved fresh deployment, and its hardware, state, ownership, and creation time match the contract. Do not infer ownership from a name prefix.

Record the deployment creation time and displayed hourly price. Compute the
cleanup deadline as the earlier of creation time plus six hours or the time at
which displayed price multiplied by elapsed hours reaches 6 USD. Before every
later remote or browser step, recheck the current time against that deadline.
At or beyond the deadline, skip the remaining acceptance work and enter the
Step 14 cleanup path.

- [ ] **Step 8: Verify setup, hardware, model, timeout, ports, and manifest**

Run these remote read-only checks only against ACS_INSTANCE_ID:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "test -f ~/.local/state/acs-nemoclaw-launchable/ready && cat ~/.local/state/acs-nemoclaw-launchable/ready"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "nvidia-smi --query-gpu=name --format=csv,noheader"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "nemoclaw acs-chemistry-agent config get --key agents.defaults.model.primary --format json"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "nemoclaw acs-chemistry-agent config get --key models.providers.inference.timeoutSeconds --format json"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "ss -H -ltn | awk '{print \$4}' | sort"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "curl -fsS -o /dev/null http://127.0.0.1:18788/ && curl -fsS -o /dev/null http://127.0.0.1:8765/"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent agent --help"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec --help"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/openshell sandbox upload --help"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py --help"
~~~

Expected: ready; exactly one NVIDIA L4; primary model
inference/nvidia/nemotron-3-super-120b-a12b; provider timeout 300; 18789
listens only on loopback; host services for 18788 and 8765 are live; the three
command surfaces exist; agent help explicitly lists -m, --session-id, --json,
and --timeout; exec help supports the exact `-- COMMAND` form used below;
OpenShell help supports `sandbox upload SANDBOX SOURCE DESTINATION`; and runner
help succeeds only with the fixed manifest. Use only flags shown by this live help. Do not print a gateway token,
dashboard URL, API key, environment dump, or config file.

- [ ] **Step 9: Run the seven canonical prompts in one new session**

Parse the seven marked blocks from launchable/acs_workshop_prompts.md and compare their hashes with the locally tested prompt blocks. Use one new session ID:

~~~text
acs-workshop-live
~~~

Before the first turn, compute the seven expected SHA-256 values locally with
the tested Task 8 parser. Record only ID and digest in the mode-0600 acceptance
note. For each independent Brev call, render exact literals for
ACS_INSTANCE_ID, ACS_PUBLIC_COMMIT, PROMPT_ID, and PROMPT_SHA256. Then run this
one remote shell command; do not depend on values from another shell:

~~~bash
/opt/homebrew/bin/brev exec "ACS_INSTANCE_ID_LITERAL" \
  'set -Eeuo pipefail
   readonly checkout="$HOME/.local/share/acs-nemoclaw-launchable/source-ACS_PUBLIC_COMMIT_LITERAL"
   readonly verifier="$checkout/scripts/verify_acs_workshop_acceptance.py"
   readonly source="$checkout/launchable/acs_workshop_prompts.md"
   prompt="$(python3 -c '\''import runpy,sys; ns=runpy.run_path(sys.argv[1]); text=ns["extract_prompt_block"](sys.argv[2],sys.argv[3],sys.argv[4]); sys.stdout.write(text)'\'' "$verifier" "$source" "PROMPT_ID_LITERAL" "PROMPT_SHA256_LITERAL")"
   "$HOME/.local/bin/nemoclaw" acs-chemistry-agent agent \
     --session-id acs-workshop-live \
     --json \
     --timeout 600 \
     -m "$prompt" >/dev/null 2>&1
   unset prompt'
~~~

The executor must replace each `*_LITERAL` token before the call, assert that
no token remains, and record the fully rendered command hash rather than the
command text. extract_prompt_block() performs the one-marker, one-fence,
nonempty, allowed-ID, and exact-hash checks. Hold the extracted prompt only in
the one remote shell variable shown above.

Execute one invocation at a time on the exact Brev instance through brev exec.
Suppress the CLI transcript because the same turn is preserved in the
Dashboard session; emit only the prompt ID, exit status, start time, and end
time to the local acceptance record. The one remote shell exits after each
turn, so the prompt variable does not persist. Stop on the first nonzero exit
or timeout. Do not retry automatically. A hosted-model idle timeout permits one
user-approved retry of the same unchanged prompt only.

Expected: all seven prompts complete in order in the same session. The six stages run their exact commands once. The objective runs objective-start once and accepts zero to three state-bound displayed tied-maximum swaps before a truthful terminal result.

- [ ] **Step 10: Run the bounded workspace, transcript, and state-log verifier**

First verify the remote host checkout is the accepted public commit:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "test \"\$(git -C \$HOME/.local/share/acs-nemoclaw-launchable/source-$ACS_PUBLIC_COMMIT rev-parse HEAD)\" = \"$ACS_PUBLIC_COMMIT\""
~~~

Create one exact acceptance-only sandbox directory:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- rm -rf -- /tmp/acs-workshop-acceptance"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- mkdir -m 700 -- /tmp/acs-workshop-acceptance"
~~~

Run this local command from the exact reviewed worktree:

~~~bash
shasum -a 256 \
  scripts/verify_acs_workshop_acceptance.py \
  launchable/acs_workshop_prompts.md
~~~

Compute the verifier and canonical-prompt SHA-256 values from that exact public
checkout and render those 64-character lowercase values as
VERIFIER_SHA256_LITERAL and PROMPTS_SHA256_LITERAL in the later sandbox check.
Upload both reviewed files through the same qualified OpenShell surface used
by setup:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/openshell sandbox upload acs-chemistry-agent \$HOME/.local/share/acs-nemoclaw-launchable/source-$ACS_PUBLIC_COMMIT/scripts/verify_acs_workshop_acceptance.py /tmp/acs-workshop-acceptance"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/openshell sandbox upload acs-chemistry-agent \$HOME/.local/share/acs-nemoclaw-launchable/source-$ACS_PUBLIC_COMMIT/launchable/acs_workshop_prompts.md /tmp/acs-workshop-acceptance"
~~~

Fail closed unless both targets are regular non-symlink files with the exact
reviewed bytes. Then make them read-only:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- sh -ceu 'test -f /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py; test ! -L /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py; test -f /tmp/acs-workshop-acceptance/acs_workshop_prompts.md; test ! -L /tmp/acs-workshop-acceptance/acs_workshop_prompts.md; test \"\$(sha256sum /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py | cut -d \" \" -f 1)\" = VERIFIER_SHA256_LITERAL; test \"\$(sha256sum /tmp/acs-workshop-acceptance/acs_workshop_prompts.md | cut -d \" \" -f 1)\" = PROMPTS_SHA256_LITERAL; chmod 0444 /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py /tmp/acs-workshop-acceptance/acs_workshop_prompts.md'"
~~~

Assert both literal tokens were rendered before this call. Now run the fixed
workspace and transcript modes:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- env PYTHONPATH=/sandbox/.openclaw/workspace:/tmp/.local/lib/python3.13/site-packages python3 /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py workspace"
~~~

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- env PYTHONPATH=/sandbox/.openclaw/workspace:/tmp/.local/lib/python3.13/site-packages python3 /tmp/acs-workshop-acceptance/verify_acs_workshop_acceptance.py transcript"
~~~

Run state-logs on the host checkout. This mode lazily avoids importing the
sandbox-only chemistry dependencies:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "python3 \$HOME/.local/share/acs-nemoclaw-launchable/source-$ACS_PUBLIC_COMMIT/scripts/verify_acs_workshop_acceptance.py state-logs"
~~~

Expected: three canonical JSON receipts with status pass. The workspace receipt
reports six stages, NVIDIA L4, a terminal objective, and zero to three attempts.
The transcript receipt reports exactly seven byte-matched canonical user
messages, at least seven assistant text messages, seven fixed base Exec calls,
zero to three history-bound objective-step calls, tool_policy pass, and
secret_scan pass. The state-log receipt has only its seven closed keys and
reports secret_scan pass. No receipt contains prompts, answers, paths,
environment variables, config, tokens, or file contents.

Remove only the exact acceptance-only sandbox directory:

~~~bash
/opt/homebrew/bin/brev exec "$ACS_INSTANCE_ID" \
  "\$HOME/.local/bin/nemoclaw acs-chemistry-agent exec -- rm -rf -- /tmp/acs-workshop-acceptance"
~~~

- [ ] **Step 11: Perform the browser acceptance with the user**

Ask the user to open the authenticated Open Chemistry Agent Secure Link and select the acs-workshop-live session. They verify:

- all seven assistant answers are visible in order;
- every required image is shown as a native image, not a raw path;
- no tool error or timeout is present;
- the visible setup output and all seven answers contain no API-key value,
  populated gateway token, tokenized URL, or private-key block;
- the reported values match the downloaded summaries; and
- the raw 18789 service is not exposed as a Secure Link.

This user-visible check is required because CLI health and PNG validity do not prove browser rendering.

- [ ] **Step 12: Download and inspect the authenticated artifact bundle**

Ask the user to open Download Results, enter workshop, and download results.zip. Use the downloaded local path they provide. Inspect it without executing archive contents:

~~~bash
unzip -t /absolute/path/to/results.zip
~~~

Run a read-only verifier for member paths, fixed timestamps and modes, hashes, manifest coverage, PNG integrity, SDF and CSV agreement, objective evidence agreement, and absence of .acs-workshop-state, keys, gateway tokens, tokenized URLs, and raw provider responses.

Run the tested local archive mode with the exact downloaded path:

~~~bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /private/tmp/nvmolkit-ui-venv312/bin/python \
  scripts/verify_acs_workshop_acceptance.py \
  archive /absolute/path/to/results.zip
~~~

Expected: unzip and the canonical archive receipt pass; the receipt hash and
size equal the workspace receipt. Record its byte size and SHA-256. Compare
displayed answers with the downloaded summary files. This is the download
gate; remote file existence is not a substitute.

- [ ] **Step 13: Record measured timing and acceptance evidence**

Create
/private/tmp/acs-workshop-acceptance-$ACS_PUBLIC_COMMIT.md with apply_patch,
set its mode to 0600, and keep it outside the repository. It contains only:

- reviewed implementation commit;
- Launchable ID and exact fresh instance ID;
- hardware facts and Brev displayed price;
- setup start, ready, and elapsed time;
- each prompt start, end, elapsed time, and status;
- browser image-render result;
- ZIP size, SHA-256, and verification result;
- secret-scan result; and
- cleanup status.

Do not record the API key, gateway token, tokenized dashboard URL, raw model response, private environment URL, or private config. These measured times are the only duration values allowed in the final attendee page.

- [ ] **Step 14: Stop and delete only the exact acceptance instance**

Immediately before lifecycle mutation, rerun the exact pinned organization listing and confirm ACS_INSTANCE_ID, ownership, and state. Then use the already-approved lifecycle authority:

~~~bash
/opt/homebrew/bin/brev stop "$ACS_INSTANCE_ID"
~~~

Use separate read-only calls to the exact organization listing, no more often
than every 30 seconds, until only ACS_INSTANCE_ID reports stopped. Do not use a
blocking wait longer than 60 seconds. Revalidate the exact ID and ownership,
then run:

~~~bash
/opt/homebrew/bin/brev delete "$ACS_INSTANCE_ID"
~~~

Confirm only this instance is absent from:

~~~bash
/opt/homebrew/bin/brev ls \
  --org "$ACS_ORG_ID" \
  --json
~~~

Expected: the exact acceptance instance is deleted. Do not use --all, a pipeline, a name pattern, or another instance. Report that deletion is irreversible and that the downloaded artifacts remain local.

### Task 12: Create the final attendee reference sheet from accepted evidence

**Files:**

- Create: tests/test_acs_fall_2026_workshop_page.py
- Create: docs/acs-fall-2026-workshop.md
- Read: launchable/acs_workshop_prompts.md
- Read: the local acceptance note from Task 11

- [ ] **Step 1: Write the failing reference-sheet tests**

Use the prompt parser from tests/test_acs_workshop_prompts.py. Require the page to contain the same seven marker pairs and byte-identical prompt blocks in the same order.

Add exact link constants:

~~~python
REPOSITORY_URL = "https://github.com/ktretina/nvmolkit-brev-notebook"
LAB_1_URL = (
    "https://brev.nvidia.com/launchable/deploy/now?"
    "launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW"
)
LAB_2_URL = (
    "https://brev.nvidia.com/launchable/deploy/now?"
    "launchableID=env-3Hlp4pHBlTTlfDxfH41KkGhTeCV"
)
BREV_URL = "https://brev.nvidia.com/"
NVIDIA_ACCOUNT_URL = "https://developer.nvidia.com/login"
NEMOTRON_KEY_URL = (
    "https://build.nvidia.com/nvidia/"
    "nemotron-3-super-120b-a12b?nim=hosted"
)
OFFICIAL_DOC_URLS = (
    "https://docs.nvidia.com/brev/latest/getting-started/quickstart",
    "https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/deployment/brev-web-ui",
    "https://docs.api.nvidia.com/nim/docs/product",
)
~~~

Require:

- the exact title ACS Fall 2026: GPU Molecular Analysis with nvMolKit and Nemotron;
- one stable anchor for each approved content block and strictly increasing
  byte offsets for all 16 anchors;
- the repository, both Launchables, both account/key pages, and all three
  official documentation URLs above;
- exactly the two approved launchableID values and no other Launchable ID;
- NVIDIA_API_KEY for Lab 1 and NVIDIA_INFERENCE_API_KEY for Lab 2;
- wording that the same private nvapi-... value is used in both fields;
- one NVIDIA L4, x86-64, 4 CPUs, 16 GiB RAM, and 128 GiB disk;
- Download Results and workshop/results.zip;
- separate hosted-endpoint rate-limit and Brev compute cost warnings;
- one same-prompt retry only after a hosted-model timeout;
- exact attendee actions to wait until setup reports ready, open the protected
  Secure Link named Open Chemistry Agent, and start one new session before
  pasting prompt 1;
- scientific limits and guidance to stop or delete both workshop environments,
  and every additional workshop environment the attendee created; and
- only duration values that exactly match the accepted evidence.

Reject:

~~~python
FORBIDDEN_TEXT = (
    "BuildDoneVideo",
    "18789",
    "gateway-token",
    "Gateway Token",
    "token=",
    "/org/",
    "draft",
)
~~~

Also reject a complete secret-shaped nvapi key, a private apps.run.brev.nvidia.com instance host, a raw /sandbox path presented as a browser URL, and claims that either the VM or hosted endpoint has unlimited free use.
Reject common unfinished-work markers through a case-insensitive regular
expression without storing an unfinished marker in the final page.

- [ ] **Step 2: Run the page tests and verify RED**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_fall_2026_workshop_page.py
~~~

Expected: failure because docs/acs-fall-2026-workshop.md does not exist.

- [ ] **Step 3: Recheck every public workshop link before writing the page**

Use read-only web access and follow redirects for exactly these nine public
destinations: the repository, both Launchable deployment URLs, Brev sign-up,
NVIDIA login, the hosted Nemotron key page, and the three OFFICIAL_DOC_URLS.
Require a successful final HTTP response and check a stable content signal:

- the repository resolves under ktretina/nvmolkit-brev-notebook;
- each Launchable URL retains its exact launchableID after the Brev sign-in or
  deployment-page redirect;
- Brev exposes its sign-in/create-account entry;
- NVIDIA login exposes login or sign-up;
- the hosted Nemotron page identifies Nemotron 3 Super and an API-key action;
  and
- each documentation URL resolves under its expected docs.nvidia.com or
  docs.api.nvidia.com path.

If a public request stops at an expected sign-in wall, do not treat the shell
page as proof. Ask the signed-in user to open that same URL read-only and
confirm the named deployment form or API-key control. Record that check as
user-visible signed-in evidence, separate from the public HTTP result.

Record only URL, UTC check time, final public URL, status, and pass/fail. Do not
authenticate, deploy, submit a form, record cookies, or infer private
Launchable configuration from a public page. Any failed, redirected-to-wrong-
ID, or content-mismatched link blocks the attendee-ready page until the link is
corrected and its static test is updated.

- [ ] **Step 4: Write the concise page in the approved order**

Create docs/acs-fall-2026-workshop.md with these exact ordered content blocks
from the approved specification:

1. workshop title and one-sentence purpose;
2. before-the-workshop checklist;
3. Brev account steps and official link;
4. NVIDIA account and hosted Nemotron API-key steps;
5. private-key warning;
6. separate hosted-endpoint rate-limit and Brev-compute cost note;
7. the two-lab key-entry table;
8. Lab 1 description and guided-notebook Launchable;
9. Lab 2 description and OpenClaw Launchable;
10. default-hardware instructions and the visible Lab 2 resource check;
11. setup-readiness, protected-chat, and one-new-session instructions;
12. the seven copy-paste prompts;
13. native-image and Download Results instructions;
14. concise troubleshooting, ending with the instruction to stop or delete
    both lab environments and every additional workshop environment;
15. scientific-use limits; and
16. official source links.

Headings may group adjacent blocks, but no block may move before or after
another. Do not add a separate section between scientific limits and official
sources.

Use:

- Brev account at https://brev.nvidia.com/;
- NVIDIA login or sign-up at https://developer.nvidia.com/login;
- Generate API Key at the accepted hosted Nemotron model page;
- no key in chat, screenshots, notebooks, or downloaded files;
- API access for prototyping can be free and rate-limited;
- Brev VM compute uses credits or incurs charges;
- Lab 1 uses NVIDIA_API_KEY and the guided notebook;
- Lab 2 uses NVIDIA_INFERENCE_API_KEY and Nemotron 3 Super in OpenClaw;
- the same private key value for both fields;
- the visible Lab 2 resource row, without requiring the text g6.xlarge;
- wait until Launchable setup reports ready before opening a lab;
- open the protected Secure Link named Open Chemistry Agent for Lab 2;
- start one new OpenClaw session and use it for all seven prompts;
- native images in chat;
- the authenticated Download Results page for workshop/results.zip;
- one unchanged retry after a hosted-model timeout;
- stop if the runner, manifest, GPU, stage-6 checkpoint, or objective validator fails;
- no install, model, command, dataset, or output-path repair;
- no claim that fingerprints, similarity, clusters, sampled conformers, or MMFF94 imply activity, efficacy, safety, synthesis, or experimental structure; and
- stop or delete each lab environment after that lab, and finish with no
  workshop environment still running.

Include these official resources:

~~~text
https://docs.nvidia.com/brev/latest/getting-started/quickstart
https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/deployment/brev-web-ui
https://docs.api.nvidia.com/nim/docs/product
~~~

Do not claim Lab 1 uses Nemotron 3 Super. Its notebook uses its own configured NVIDIA model.

- [ ] **Step 5: Copy the canonical prompts without editing them**

Copy the seven complete marker regions from launchable/acs_workshop_prompts.md into the Copy-paste prompts section byte-for-byte. Do not rewrap lines, change punctuation, change commands, add hidden instructions, or substitute accepted output values.

- [ ] **Step 6: Add only accepted timing guidance**

If Task 11 recorded all setup and prompt timings, add a small expectations note using the measured values and label them as one accepted L4 run, not a service guarantee. If any timing is missing or ambiguous, omit timing guidance entirely. Never estimate or round an unverified duration into the page.

- [ ] **Step 7: Run the post-acceptance page gates**

Run:

~~~bash
env PYTHONPATH=. \
  /private/tmp/nvmolkit-ui-venv312/bin/pytest -q \
  tests/test_acs_workshop_prompts.py \
  tests/test_acs_fall_2026_workshop_page.py
~~~

Expected: prompt-source tests and page tests pass.

Run:

~~~bash
gitleaks git --staged --no-banner --redact .
~~~

~~~bash
git diff --check
~~~

Expected: no secret and no whitespace error.

- [ ] **Step 8: Review the page as an ACS attendee**

Use a read-only reviewer who has only the page and the accepted evidence. Ask whether a chemist who is not a Brev, NemoClaw, or command-line expert can:

- create both accounts and one private API key;
- put the key in the correct field for each lab;
- select the visible default L4 resource row;
- wait for readiness, open the protected Open Chemistry Agent link, and start
  one new session;
- open both labs;
- run the seven prompts without inventing commands;
- view images and download the correct ZIP;
- understand one-retry and stop conditions;
- interpret the science conservatively; and
- stop or delete both lab environments so billing ends.

Repair valid clarity gaps without changing canonical prompt bytes, links, scientific facts, or accepted timing evidence. Rerun Step 7 after any repair.

- [ ] **Step 9: Commit the final attendee guide locally**

Run:

~~~bash
git add \
  docs/acs-fall-2026-workshop.md \
  tests/test_acs_fall_2026_workshop_page.py
~~~

~~~bash
gitleaks git --staged --no-banner --redact .
~~~

~~~bash
git commit -m "Add ACS Fall 2026 attendee guide"
~~~

Expected: one local guide commit. Do not push this commit unless the user gives a separate explicit publication request.

### Task 13: Final verification and handoff

**Files:**

- Verify: all implementation and attendee-guide files
- Deliver: docs/acs-fall-2026-workshop.md

- [ ] **Step 1: Rerun the complete local regression and static gates**

Repeat Task 10 Steps 2 through 5 with
tests/test_acs_fall_2026_workshop_page.py added to the focused suite. Record
the exact commands, exit codes, test counts, secret-scan result, and final
commit.

- [ ] **Step 2: Verify branch and public-state boundaries**

Run:

~~~bash
git status --short --branch
~~~

~~~bash
git log --oneline --decorate -12
~~~

~~~bash
git ls-remote origin refs/heads/acs-fall-2026-launchable
~~~

Expected: the worktree is clean; the locally committed guide may be ahead of the public implementation commit; the exact difference is recorded. Do not silently push, publish a GitHub Page, change Launchable access, or deploy another instance.

- [ ] **Step 3: Deliver the attendee-ready file and evidence boundary**

Give the user:

- a clickable local link to docs/acs-fall-2026-workshop.md;
- the local guide commit;
- the public implementation commit used by the qualified Launchable;
- the exact two Launchable URLs;
- the fresh-L4 acceptance result and measured timing scope;
- the downloaded ZIP hash and validation result;
- confirmation that the exact acceptance instance was stopped and deleted; and
- any remaining evidence limit.

State clearly that the Markdown file is ready for the user to place in GitHub. Publication, GitHub Pages setup, another Launchable deployment, broader access, and conference-scale capacity testing remain separate actions unless the user asks for them.
