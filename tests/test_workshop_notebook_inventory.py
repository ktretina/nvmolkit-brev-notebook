import ast
import json
import re
import subprocess
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
MODULE1_NOTEBOOK_PATH = NOTEBOOK_DIR / "01_direct_nvmolkit_reframe.ipynb"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
README_PATH = REPO_ROOT / "README.md"
LAUNCHABLE_FIELDS_PATH = REPO_ROOT / "launchable" / "fields.md"
FIXED_MODEL = "nvidia/nvidia/nemotron-3-nano-30b-a3b"
ORDERED_NOTEBOOK_NAMES = (
    "01_direct_nvmolkit_reframe.ipynb",
    "02_agent_assisted_reframe_neighborhoods.ipynb",
    "03_full_agent_reframe_panel_design.ipynb",
)
PRIMARY_NOTEBOOK_NAMES = {
    *ORDERED_NOTEBOOK_NAMES,
}
GENERATED_DIRECTORY_NAMES = {
    ".ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
}
GENERATED_FILE_NAMES = {".DS_Store"}
GENERATED_FILE_SUFFIXES = {".log", ".pyc", ".pyo"}
MODULE1_CELL_IDS = (
    "cell-51c91e7ea419",
    "cell-f38b80a4df5a",
    "cell-a5ae8306d03b",
    "cell-f3f4dc28ff90",
    "cell-e2551a466884",
    "cell-e51239292d22",
    "cell-9f1999dd251d",
    "cell-6bad8e7aa6e9",
    "cell-9293c88b2875",
    "cell-d7a78c3d5f16",
    "cell-c6cdbb049a79",
    "cell-7dabc4a334bf",
    "cell-c5e8c0433cbd",
    "cell-70eb96263ae6",
    "cell-00a63d6c51e3",
    "cell-14aa9d3c7dec",
    "cell-advanced-large-run",
    "cell-advanced-large-run-code",
    "cell-02f0731c9452",
    "cell-e416d0944996",
    "cell-7d83af67d461",
    "cell-5071adec6f11",
    "f80ade1e",
    "26bc80a6",
)
MODULE1_TAGS = {
    "cell-9f1999dd251d": ["exercise"],
    "cell-c6cdbb049a79": ["exercise"],
    "cell-c5e8c0433cbd": ["exercise"],
    "cell-14aa9d3c7dec": ["exercise"],
    "cell-advanced-large-run-code": ["advanced"],
    "f80ade1e": ["answer-key", "instructor-only", "solution"],
    "26bc80a6": ["answer-key", "instructor-only"],
}
RUNTIME_FORMATTER_ARGUMENTS = (
    ("rdkit_fp_seconds", "nvmolkit_fp_seconds"),
    ("rdkit_similarity_seconds", "nvmolkit_similarity_seconds"),
    ("rdkit_clustering_seconds", "nvmolkit_clustering_seconds"),
)


def test_primary_notebook_inventory_is_exact():
    assert {
        path.name for path in NOTEBOOK_DIR.glob("*.ipynb")
    } == PRIMARY_NOTEBOOK_NAMES


def test_release_docs_publish_the_three_notebook_path_and_launch_contract():
    readme = README_PATH.read_text(encoding="utf-8")
    fields = LAUNCHABLE_FIELDS_PATH.read_text(encoding="utf-8")
    readme_opening = readme.split("## Three-notebook workshop path", 1)[0]

    assert "three-notebook workshop" in readme_opening.lower()
    assert "direct" in readme_opening and "nvMolKit" in readme_opening
    assert "companion demo" not in readme.lower()
    assert "nvmolkit_nemotron_demo.ipynb" not in readme

    for document in (readme, fields):
        notebook_positions = [document.index(name) for name in ORDERED_NOTEBOOK_NAMES]
        assert notebook_positions == sorted(notebook_positions)
        assert "Module 1" in document
        assert "recommended" in document.lower()
        assert "hosted mode" in document.lower()
        assert "reference mode" in document.lower()
        assert FIXED_MODEL in document
        assert "NVIDIA_INFERENCE_API_KEY" in document
        assert "https://inference-api.nvidia.com/v1" in document
        assert "8888" in document
        assert "75 GiB" in document
        assert "Only my organization" in document


def test_module2_explains_the_organizer_supplied_inference_hub_key():
    module2 = nbformat.read(
        NOTEBOOK_DIR / "02_agent_assisted_reframe_neighborhoods.ipynb", as_version=4
    )
    markdown = "\n".join(
        cell.source for cell in module2.cells if cell.cell_type == "markdown"
    )
    assert "organizer-supplied" in markdown
    assert "Inference Hub" in markdown
    assert "`sk-`" in markdown
    assert "`nvapi-`" not in markdown


def test_primary_notebooks_are_clean_python_312_notebooks():
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["nbformat"] == 4, path

        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

        assert notebook.metadata.kernelspec.name == "python3", path
        assert notebook.metadata.kernelspec.language == "python", path
        assert notebook.metadata.language_info.name == "python", path
        assert notebook.metadata.language_info.version.split(".")[:2] == ["3", "12"], (
            path
        )

        cell_ids = [cell.id for cell in notebook.cells]
        assert all(cell_ids), path
        assert len(cell_ids) == len(set(cell_ids)), path
        assert not notebook.metadata.get("widgets"), path

        for cell in notebook.cells:
            assert cell.get("execution_count") is None, path
            assert not cell.get("outputs"), path
            assert not cell.get("attachments"), path


def test_release_inventory_excludes_generated_files():
    release_paths = {
        Path(path)
        for path in subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    }
    generated_paths = {
        path
        for path in release_paths
        if GENERATED_DIRECTORY_NAMES.intersection(path.parts)
        or path.name in GENERATED_FILE_NAMES
        or path.suffix in GENERATED_FILE_SUFFIXES
    }
    assert generated_paths == set()


def _module1_code_source():
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    return "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def _module1_cell_source(cell_id):
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    return next(cell.source for cell in notebook.cells if cell.id == cell_id)


def _module1_function_source(function_name):
    source = _module1_code_source()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return ast.get_source_segment(source, function)


def _assert_module1_runtime_formatter_calls(source):
    formatter_calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "format_runtime_ratio"
    ]
    assert len(formatter_calls) == len(RUNTIME_FORMATTER_ARGUMENTS)

    observed_arguments = []
    for call in formatter_calls:
        assert len(call.args) == 2
        assert not call.keywords
        assert all(isinstance(argument, ast.Name) for argument in call.args)
        observed_arguments.append(tuple(argument.id for argument in call.args))

    assert tuple(observed_arguments) == RUNTIME_FORMATTER_ARGUMENTS
    reverse_arguments = {
        (gpu_name, cpu_name) for cpu_name, gpu_name in RUNTIME_FORMATTER_ARGUMENTS
    }
    assert reverse_arguments.isdisjoint(observed_arguments)


def test_nvmolkit_dependency_is_pinned_to_exact_supported_version():
    requirement_lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()

    assert [line for line in requirement_lines if line.startswith("nvmolkit")] == [
        "nvmolkit==0.6.0"
    ]


def test_module1_notebook_json_structure_and_clean_state_are_preserved():
    stored = json.loads(MODULE1_NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert tuple(cell["id"] for cell in stored["cells"]) == MODULE1_CELL_IDS
    assert stored["metadata"] == {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
        "workshop": {
            "conference": "ACS Fall 2026",
            "generated": "2026-08-13",
            "topic": "nvMolKit and agentic cheminformatics",
        },
    }
    for cell in stored["cells"]:
        assert cell["metadata"] == {"tags": MODULE1_TAGS.get(cell["id"], [])}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
        else:
            assert "execution_count" not in cell
            assert "outputs" not in cell


def test_module1_setup_guidance_uses_the_exact_nvmolkit_release():
    setup = _module1_cell_source("cell-f38b80a4df5a")
    command_blocks = "\n".join(
        re.findall(r"```(?:bash)?\n(.*?)```", setup, flags=re.DOTALL)
    )

    assert "`nvmolkit==0.6.0`" in setup
    assert "nvMolKit 0.5" not in setup
    assert "python -m pip install -r requirements.txt" in command_blocks
    assert not re.search(r"\bnvmolkit(?=\s|$)", command_blocks, flags=re.IGNORECASE)
    assert "`nvmolkit_compat.py`" in setup
    assert "`NVMOLKIT_READY == True`" not in setup
    assert "nvMolKit version and CUDA device count" in setup
    assert "CPU teaching fallback and its reason" in setup


def test_module1_default_and_advanced_controls_are_bounded():
    source = _module1_code_source()

    assert 'DATA_SOURCE = "snapshot"' in source
    assert "SAMPLE_SIZE = 96" in source
    assert "FP_BITS = 1024" in source
    assert "ADVANCED_LARGE_RUN = False" in source
    assert "ADVANCED_SAMPLE_SIZE = 10_000" in source
    assert 'source="live"' in source


def test_module1_computation_summary_does_not_claim_unrun_3d_methods():
    summary = _module1_cell_source("cell-e2551a466884")

    assert "ETKDG" not in summary
    assert "MMFF" not in summary


def test_module1_checks_memory_immediately_before_condensed_distance_allocation():
    source = _module1_code_source()

    assert source.count("require_bounded_condensed_distances(n_items)") == 1
    assert re.search(
        r"require_bounded_condensed_distances\(n_items\)\n"
        r"\s+distances = np\.empty\(n_items \* \(n_items - 1\) // 2, dtype=np\.float64\)",
        source,
    )


def test_module1_advanced_10k_path_is_gpu_only_and_reports_provenance():
    source = _module1_cell_source("cell-advanced-large-run-code")

    assert "rdkit_butina_from_fingerprints" not in source
    assert "if not NVMOLKIT_READY:" in source
    assert "requires a compatible NVIDIA GPU" in source
    assert "MorganFingerprintGenerator" in source
    assert "nvmolkit_butina_from_fingerprints" in source
    assert "torch.cuda.synchronize" in source
    assert 'advanced_reframe.attrs["source"]' in source
    assert "len(advanced_reframe)" in source
    assert "square_matrix_bytes" not in source
    assert "require_memory_within_limit" not in source
    assert "advanced_square_bytes" not in source
    assert "float32 square matrix estimate" not in source
    assert "limit_mib=512" not in source


def test_module1_advanced_memory_copy_matches_fused_butina_allocation_model():
    source = _module1_cell_source("cell-advanced-large-run")
    setup_copy = _module1_cell_source("cell-e51239292d22")

    assert "fused Butina avoids an `N x N` pairwise matrix" in source
    assert "uses packed fingerprints plus linear working buffers" in source
    assert "full notebook" in source
    assert "CUDA allocator" in source
    assert "still exhaust GPU memory" in source
    assert "512 MiB" not in source
    assert "square-matrix estimate" not in source
    assert "memory estimate" not in setup_copy


def test_module1_uses_the_version_compatible_fused_butina_adapter():
    setup = _module1_cell_source("cell-a5ae8306d03b")
    converter = _module1_function_source("nvmolkit_butina_from_fingerprints")

    assert "from nvmolkit_compat import normalize_fused_butina_result" in setup
    assert "return_centroids=True" in converter
    assert re.search(
        r"normalize_fused_butina_result\(\n"
        r"\s+raw_result, molecule_count=len\(fingerprint_matrix\)\n"
        r"\s+\)",
        converter,
    )
    assert "return labels, centroids" in converter
    assert "for cluster_id, members in enumerate" not in converter


def test_module1_runtime_copy_is_neutral_and_clustering_rows_are_specific():
    source = _module1_code_source()
    runtime_label = (
        "Observed runtime ratio (RDKit CPU / nvMolKit GPU; >1 favors nvMolKit):"
    )
    clustering = _module1_function_source("nvmolkit_butina_from_fingerprints")
    clustering_cell = _module1_cell_source("cell-14aa9d3c7dec")

    assert source.count(runtime_label) == 1
    assert "speedup" not in source.lower()
    assert '"backend": "RDKit CPU: condensed distance + Butina"' in clustering_cell
    assert '"backend": "nvMolKit GPU: fused fingerprint clustering"' in clustering_cell
    assert "return labels, centroids" in clustering
    assert '"backend": "RDKit CPU"' in _module1_cell_source("cell-c6cdbb049a79")
    assert '"backend": "nvMolKit GPU"' in _module1_cell_source("cell-c6cdbb049a79")


def test_module1_runtime_formatter_is_neutral_in_a_clean_kernel():
    source = _module1_code_source()
    assert "def format_runtime_ratio(cpu_seconds, gpu_seconds):" in source

    formatter_source = _module1_function_source("format_runtime_ratio")
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                formatter_source, id="test-runtime-formatter-definition"
            ),
            nbformat.v4.new_code_cell(
                """\
label = "Observed runtime ratio (RDKit CPU / nvMolKit GPU; >1 favors nvMolKit):"
assert format_runtime_ratio(1.0, 2.0) == f"{label} 0.50×"
assert format_runtime_ratio(1.0, 1.0) == f"{label} 1.00×"
assert format_runtime_ratio(2.0, 1.0) == f"{label} 2.00×"
""",
                id="test-runtime-formatter-assertions",
            ),
        ]
    )
    executor = ExecutePreprocessor(timeout=60, kernel_name="python3")

    executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    assert all(not cell.outputs for cell in notebook.cells)


def test_module1_all_three_gpu_runtime_branches_use_the_formatter():
    _assert_module1_runtime_formatter_calls(_module1_code_source())


@pytest.mark.parametrize(("cpu_name", "gpu_name"), RUNTIME_FORMATTER_ARGUMENTS)
def test_module1_runtime_formatter_call_gate_rejects_reversed_arguments(
    cpu_name, gpu_name
):
    source = _module1_code_source()
    forward_call = f"format_runtime_ratio({cpu_name}, {gpu_name})"
    reverse_call = f"format_runtime_ratio({gpu_name}, {cpu_name})"
    mutated_source = source.replace(forward_call, reverse_call, 1)

    assert mutated_source != source
    with pytest.raises(AssertionError):
        _assert_module1_runtime_formatter_calls(mutated_source)


def test_module1_report_contract_is_present():
    source = _module1_code_source()

    for field in ("source", "rows", "backend", "fingerprint_bits", "elapsed_seconds"):
        assert f'"{field}"' in source
    assert "MODULE1_REPORT_JSON=" in source
    assert "RDKit CPU fallback (not GPU evidence)" in source


def test_module1_default_executes_without_network_and_emits_report(
    monkeypatch, tmp_path
):
    notebook_path = NOTEBOOK_DIR / "01_direct_nvmolkit_reframe.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    blocker = nbformat.v4.new_code_cell(
        """\
_original_read_csv = pd.read_csv
_blocked_network_attempts = []


def _local_only_read_csv(source, *args, **kwargs):
    if str(source).startswith(("http://", "https://")):
        _blocked_network_attempts.append(str(source))
        raise AssertionError("Notebook default attempted network access")
    return _original_read_csv(source, *args, **kwargs)


pd.read_csv = _local_only_read_csv
""",
        id="test-network-blocker",
    )
    notebook.cells.insert(3, blocker)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert _blocked_network_attempts == []", id="test-network-assertion"
        )
    )

    matplotlib_dir = tmp_path / "matplotlib"
    matplotlib_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_dir))
    monkeypatch.setenv(
        "REFRAME_CSV", "https://hostile.invalid/reframe.csv?token=do-not-disclose"
    )
    executor = ExecutePreprocessor(timeout=300, kernel_name="python3")
    executor.preprocess(
        notebook,
        {"metadata": {"path": str(NOTEBOOK_DIR)}},
    )

    output_path = tmp_path / "module1-executed.ipynb"
    nbformat.write(notebook, output_path)
    assert output_path.is_file()

    stream_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    match = re.search(r"^MODULE1_REPORT_JSON=(\{.*\})$", stream_text, re.MULTILINE)
    assert match is not None
    report = json.loads(match.group(1))
    assert report["source"] == "bundled_snapshot"
    assert report["rows"] == 96
    assert report["fingerprint_bits"] == 1024
    assert report["backend"] in {
        "nvMolKit GPU",
        "RDKit CPU fallback (not GPU evidence)",
    }
    if report["backend"].startswith("RDKit"):
        assert "not GPU evidence" in report["backend"]
    assert report["elapsed_seconds"] > 0
