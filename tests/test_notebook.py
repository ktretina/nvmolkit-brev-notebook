from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nvmolkit_nemotron_demo.ipynb"
)

STORY_HEADINGS = [
    "# nvMolKit + Nemotron",
    "## 1. Preflight",
    "## 2. Molecular sample",
    "## 3. Nemotron plan",
    "## 4. Fingerprints, similarity, and clusters",
    "## 5. Conformers and MMFF94",
    "## 6. What the results mean",
]

REQUIRED_SOURCE_TERMS = [
    "MorganFingerprintGenerator",
    "crossTanimotoSimilarity",
    "fused_butina",
    "EmbedMolecules",
    "MMFFOptimizeMoleculesConfs",
    "torch.cuda.synchronize",
    "MolsToGridImage",
    "sns.heatmap",
    "py3Dmol.view",
]


def read_notebook():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def test_notebook_is_valid_and_story_headings_are_in_order():
    notebook = read_notebook()
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    positions = [markdown.index(heading) for heading in STORY_HEADINGS]
    assert positions == sorted(positions)


def test_notebook_contains_required_workflow_and_visuals():
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert all(term in source for term in REQUIRED_SOURCE_TERMS)


def test_notebook_contains_no_api_key_and_no_saved_execution_state():
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "nvapi-" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
