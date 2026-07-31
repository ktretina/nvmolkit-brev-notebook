import ast
import json
import subprocess
from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nvmolkit_nemotron_demo.ipynb"
)
REPO_ROOT = NOTEBOOK_PATH.parents[1]

STORY_HEADINGS = [
    "# nvMolKit + Nemotron",
    "## 1. Preflight",
    "## 2. Molecular sample",
    "## 3. Nemotron plan",
    "## 4. Fingerprints, similarity, and clusters",
    "## 5. Conformers and MMFF94",
    "## 6. What the results mean",
]

REQUIRED_CALLS = {
    "MorganFingerprintGenerator",
    "crossTanimotoSimilarity",
    "fused_butina",
    "EmbedMolecules",
    "MMFFOptimizeMoleculesConfs",
    "Point3D",
    "torch.cuda.synchronize",
    "Draw.MolsToGridImage",
    "sns.heatmap",
    "py3Dmol.view",
}


def read_notebook():
    stored = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert stored["nbformat"] == 4

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=nbformat.NO_CONVERT)
    nbformat.validate(notebook)
    return notebook


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_notebook_has_exact_v4_story_structure_and_is_only_source_notebook():
    notebook = read_notebook()
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]
    tracked_notebooks = subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_notebooks = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts and "executed" not in path.parts
    )
    intro = notebook.cells[0].source
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert headings == STORY_HEADINGS
    assert tracked_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    assert source_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    assert all(
        phrase in intro
        for phrase in ("API entry-point map", "runtime requirements", "recipes", "boundaries")
    )
    assert all(
        requirement in setup
        for requirement in (
            "sys.implementation.name",
            "(3, 12)",
            '"Linux"',
            '"x86_64"',
        )
    )
    assert "Runtime: Linux x86-64 with CPython 3.12" in fields
    assert "Linux x86-64 with CPython 3.12" in readme
    assert "qualified for a fresh launch" not in fields.lower()
    assert "qualified for a fresh launch" not in readme.lower()
    assert "not yet live-qualified" in fields
    assert "not yet live-qualified" in readme
    assert all(
        acceptance in fields.lower() and acceptance in readme.lower()
        for acceptance in ("gpu", "hosted inference", "rendered visuals", "secure link")
    )
    assert ".jupyter.pid" in gitignore


def test_notebook_code_calls_required_workflow_and_visuals():
    notebook = read_notebook()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    tree = ast.parse(code)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {dotted_name(node.func) for node in calls}
    fused_butina_call = next(
        call for call in calls if dotted_name(call.func) == "fused_butina"
    )
    mmff_call = next(
        call for call in calls if dotted_name(call.func) == "MMFFOptimizeMoleculesConfs"
    )
    mmff_keywords = {keyword.arg: keyword.value for keyword in mmff_call.keywords}
    referenced_names = {
        dotted_name(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert REQUIRED_CALLS <= called_names
    assert len(fused_butina_call.args) == 1
    assert isinstance(fused_butina_call.args[0], ast.Call)
    assert dotted_name(fused_butina_call.args[0].func) == "fingerprints.torch"
    assert dotted_name(mmff_keywords["output"]) == "CoordinateOutput.DEVICE"
    assert {
        "optimization_result.energies.numpy",
        "optimization_result.converged.numpy",
        "optimization_result.mol_indices.numpy",
        "optimization_result.conf_indices.numpy",
        "optimization_result.per_molecule",
    } <= called_names
    assert {
        "optimization_result.energies",
        "optimization_result.converged",
        "optimization_result.mol_indices",
        "optimization_result.conf_indices",
        "optimization_result.per_molecule",
    } <= referenced_names
    assert "converged_conformers" in string_literals
    assert {
        "requested_conformers",
        "generated_conformers",
        "mmff_attempted_conformers",
    } <= string_literals
    assert "optimized_conformers" not in string_literals

    requested_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "requested_conformers"
            for target in node.targets
        )
    )
    embed_call = next(call for call in calls if dotted_name(call.func) == "EmbedMolecules")
    assert requested_assignment.lineno < embed_call.lineno

    explanation_call = next(
        call for call in calls if dotted_name(call.func) == "request_explanation"
    )
    explanation_line = explanation_call.lineno
    lines = code.splitlines()
    before_explanation = "\n".join(lines[: explanation_line - 1])
    after_explanation = "\n".join(lines[explanation_line:])
    boundary_terms = (
        "binding",
        "activity",
        "ADMET",
        "efficacy",
        "safety",
        "synthesizability",
        "clinical relevance",
        "experimentally validated conformations",
    )
    assert all(term in before_explanation for term in boundary_terms)
    assert "Agent-generated interpretation; verify independently" in before_explanation
    assert all(term in after_explanation for term in boundary_terms)


def test_notebook_contains_no_api_key_and_no_saved_execution_state():
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "nvapi-" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
