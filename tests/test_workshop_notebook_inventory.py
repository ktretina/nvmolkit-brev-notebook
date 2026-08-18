import json
import subprocess
from pathlib import Path

import nbformat


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
    assert {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")} == PRIMARY_NOTEBOOK_NAMES


def test_primary_notebooks_are_clean_python_312_notebooks():
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["nbformat"] == 4, path

        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

        assert notebook.metadata.kernelspec.name == "python3", path
        assert notebook.metadata.kernelspec.language == "python", path
        assert notebook.metadata.language_info.name == "python", path
        assert notebook.metadata.language_info.version.split(".")[:2] == ["3", "12"], path

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
