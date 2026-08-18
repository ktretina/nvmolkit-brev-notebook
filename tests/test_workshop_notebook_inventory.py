import json
import re
import subprocess
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
PRIMARY_NOTEBOOK_NAMES = {
    "01_direct_nvmolkit_reframe.ipynb",
    "02_agent_assisted_reframe_neighborhoods.ipynb",
    "03_full_agent_reframe_panel_design.ipynb",
    "nvmolkit_nemotron_demo.ipynb",
}
GENERATED_DIRECTORY_NAMES = {
    ".ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
}
GENERATED_FILE_NAMES = {".DS_Store"}
GENERATED_FILE_SUFFIXES = {".log", ".pyc", ".pyo"}


def test_primary_notebook_inventory_is_exact():
    assert {
        path.name for path in NOTEBOOK_DIR.glob("*.ipynb")
    } == PRIMARY_NOTEBOOK_NAMES


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
    notebook = nbformat.read(
        NOTEBOOK_DIR / "01_direct_nvmolkit_reframe.ipynb", as_version=4
    )
    return "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def _module1_cell_source(cell_id):
    notebook = nbformat.read(
        NOTEBOOK_DIR / "01_direct_nvmolkit_reframe.ipynb", as_version=4
    )
    return next(cell.source for cell in notebook.cells if cell.id == cell_id)


def test_module1_default_and_advanced_controls_are_bounded():
    source = _module1_code_source()

    assert 'DATA_SOURCE = "snapshot"' in source
    assert "SAMPLE_SIZE = 96" in source
    assert "FP_BITS = 1024" in source
    assert "ADVANCED_LARGE_RUN = False" in source
    assert "ADVANCED_SAMPLE_SIZE = 10_000" in source
    assert 'source="live"' in source
    assert "limit_mib=512" in source


def test_module1_checks_memory_immediately_before_condensed_distance_allocation():
    source = _module1_code_source()

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
    estimate_print = source.index("10k float32 square matrix estimate")
    memory_guard = source.index(
        "require_memory_within_limit(advanced_square_bytes, limit_mib=512)"
    )
    live_load = source.index('source="live"')
    assert estimate_print < memory_guard < live_load


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
