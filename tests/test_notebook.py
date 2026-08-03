import ast
import http.server
import json
import os
import re
import subprocess
import sys
import threading
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
    "## 2. Nemotron learns the nvMolKit skill",
    "## 3. Molecular sample",
    "## 4. Mapping molecular similarity",
    "### 4.1 Morgan fingerprints",
    "### 4.2 All-pairs Tanimoto similarity",
    "### 4.3 Fused Butina clusters",
    "## 5. Conformers and MMFF94",
    "## 6. What the results mean",
]

INTRODUCTION = """# nvMolKit + Nemotron

This notebook demonstrates a guided chemistry agent using NVIDIA Nemotron to call a small set of allow-listed scientific tools backed by nvMolKit on an NVIDIA GPU. Nemotron first reads the BioNeMo Agent Toolkit skill for nvMolKit, learning the library's supported operations, API boundaries, and GPU requirements. It then works through a molecular library one analysis at a time: validating the sample, generating Morgan fingerprints, measuring all-pairs Tanimoto similarity, identifying structural clusters, and generating and minimizing representative conformers.

Each stage follows the same transparent pattern. The notebook defines a bounded scientific function; Nemotron requests that function through a structured tool call; the notebook validates and executes it; the result is visualized immediately; and Nemotron provides a short interpretation. A final synthesis combines the numerical results from every stage into a detailed scientific discussion.

Brev supplies the GPU environment, nvMolKit performs the batched GPU chemistry operations, RDKit handles molecule parsing and display preparation, and the notebook enforces the execution and scientific-safety boundaries. Nemotron chooses validated tool parameters and explains results, but it does not execute arbitrary Python.

This is a cheminformatics demonstration, not a benchmark or validated scientific study. Fingerprints, similarities, clusters, force-field energies, and candidate geometries are computational outputs. They do not establish binding, biological activity, ADMET properties, efficacy, safety, synthesizability, or clinical relevance."""

TOOL_STAGES = (
    (
        "## 2. Nemotron learns the nvMolKit skill",
        "read_nvmolkit_skill",
        "ReadSkillArgs",
        "skill_artifact",
    ),
    (
        "## 3. Molecular sample",
        "prepare_molecular_sample",
        "PrepareSampleArgs",
        "sample_artifact",
    ),
    (
        "### 4.1 Morgan fingerprints",
        "compute_morgan_fingerprints",
        "FingerprintArgs",
        "fingerprint_artifact",
    ),
    (
        "### 4.2 All-pairs Tanimoto similarity",
        "compute_tanimoto_similarity",
        "SimilarityArgs",
        "similarity_artifact",
    ),
    (
        "### 4.3 Fused Butina clusters",
        "cluster_with_fused_butina",
        "ClusterArgs",
        "cluster_artifact",
    ),
)

EXPECTED_CAPABILITY_ROWS = [
    {
        "Capability": "Morgan fingerprints",
        "Entry point": "MorganFingerprintGenerator",
        "Role": "Molecular representation",
    },
    {
        "Capability": "Tanimoto similarity",
        "Entry point": "crossTanimotoSimilarity",
        "Role": "Pairwise structural similarity",
    },
    {
        "Capability": "Butina clustering",
        "Entry point": "fused_butina",
        "Role": "Structural grouping",
    },
    {
        "Capability": "ETKDG embedding",
        "Entry point": "EmbedMolecules",
        "Role": "Candidate 3D conformers",
    },
    {
        "Capability": "MMFF94 optimization",
        "Entry point": "MMFFOptimizeMoleculesConfs",
        "Role": "Force-field minimization",
    },
]


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


def artifact_subscript_path(node):
    if not isinstance(node, ast.Subscript):
        return None
    keys = []
    current = node
    while isinstance(current, ast.Subscript):
        if not isinstance(current.slice, ast.Constant) or not isinstance(
            current.slice.value, str
        ):
            return None
        keys.append(current.slice.value)
        current = current.value
    if not isinstance(current, ast.Name) or not current.id.endswith("_artifact"):
        return None
    return current.id, tuple(reversed(keys))


def artifact_accesses(node):
    access = artifact_subscript_path(node)
    if access is not None:
        yield access
        return
    if isinstance(node, ast.Name) and node.id.endswith("_artifact"):
        yield node.id, ()
        return
    for child in ast.iter_child_nodes(node):
        yield from artifact_accesses(child)


def notebook_code(notebook):
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def heading_cell_index(notebook, heading):
    return next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and heading in cell.source.splitlines()
    )


def test_notebook_has_exact_v4_story_intro_and_is_only_source_notebook():
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
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert headings == STORY_HEADINGS
    assert notebook.cells[0].source == INTRODUCTION
    assert tracked_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    assert source_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
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


def test_preflight_uses_fixed_local_artifacts_gpu_probe_and_strict_agent_contracts():
    notebook = read_notebook()
    code = notebook_code(notebook)
    tree = ast.parse(code)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "demo_agent"
        for alias in node.names
    }

    assert {
        "ReadSkillArgs",
        "PrepareSampleArgs",
        "FingerprintArgs",
        "SimilarityArgs",
        "ClusterArgs",
        "request_and_execute_step",
        "request_brief_interpretation",
    } <= imports
    assert 'DATA_PATH = PROJECT_ROOT / "data" / "sample_molecules.csv"' in code
    assert 'SKILL_PATH = PROJECT_ROOT / "skills" / "nvmolkit" / "SKILL.md"' in code
    assert "Path.cwd().parents" in code
    assert "torch.cuda.is_available()" in code
    assert "torch.cuda.get_device_capability(0)" in code
    assert "nvmolkit.__version__" in code
    preflight_tree = ast.parse(notebook.cells[2].source)
    probe_call = next(
        node
        for node in ast.walk(preflight_tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "MorganFingerprintGenerator"
    )
    probe_keywords = {keyword.arg: keyword.value for keyword in probe_call.keywords}
    assert probe_keywords["radius"].value == 2
    assert probe_keywords["fpSize"].value == 1024
    assert "GetFingerprints(probe_molecules)" in notebook.cells[2].source
    assert "torch.cuda.synchronize()" in code
    assert 'model = "nvidia/nemotron-3-nano-30b-a3b"' in code
    assert "from nvmolkit.embedMolecules import EmbedMolecules" in code
    assert "from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs" in code


def test_each_guided_tool_has_task_function_forced_call_result_and_brief_in_order():
    notebook = read_notebook()
    code = notebook_code(notebook)
    tree = ast.parse(code)
    request_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "request_and_execute_step"
    ]
    brief_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "request_brief_interpretation"
    ]

    assert len(request_calls) == len(brief_calls) == 5
    for heading, function_name, annotation_name, artifact_name in TOOL_STAGES:
        heading_index = heading_cell_index(notebook, heading)
        task_cell, function_cell, request_cell, result_cell, brief_cell = notebook.cells[
            heading_index + 1 : heading_index + 6
        ]
        assert [
            task_cell.cell_type,
            function_cell.cell_type,
            request_cell.cell_type,
            result_cell.cell_type,
            brief_cell.cell_type,
        ] == ["markdown", "code", "code", "code", "code"]
        assert task_cell.source.startswith("**Task.**")

        function_tree = ast.parse(function_cell.source)
        function = next(
            node for node in function_tree.body if isinstance(node, ast.FunctionDef)
        )
        assert function.name == function_name
        assert [argument.arg for argument in function.args.args] == ["args"]
        assert dotted_name(function.args.args[0].annotation) == annotation_name

        assert "# Validation completes before the executor runs." in request_cell.source
        request_tree = ast.parse(request_cell.source)
        stage_request = next(
            node
            for node in ast.walk(request_tree)
            if isinstance(node, ast.Call)
            and dotted_name(node.func) == "request_and_execute_step"
        )
        keywords = {keyword.arg: keyword.value for keyword in stage_request.keywords}
        assert keywords["tool_name"].value == function_name
        assert dotted_name(keywords["executor"]) == function_name
        assert f'**Requested tool:**' in request_cell.source
        assert f'**Validated arguments:**' in request_cell.source
        assert artifact_name in request_cell.source

        assert "json.dumps" in result_cell.source
        assert artifact_name in result_cell.source
        assert "request_brief_interpretation" in brief_cell.source
        assert '"Interpretation unavailable"' in brief_cell.source
        assert "except Exception" in brief_cell.source

    tool_name_literals = [
        keyword.value.value
        for call in request_calls
        for keyword in call.keywords
        if keyword.arg == "tool_name"
    ]
    assert tool_name_literals == [stage[1] for stage in TOOL_STAGES]
    assert "analyze_molecule_library" not in code
    assert "request_explanation" not in code
    assert "request_tool_call" not in code
    assert "default" not in code.lower()
    assert "eval(" not in code
    assert "exec(" not in code
    assert "importlib" not in code


def test_skill_stage_reads_pinned_text_once_and_bounds_later_grounding():
    notebook = read_notebook()
    code = notebook_code(notebook)
    skill_function_cell = notebook.cells[
        heading_cell_index(notebook, "## 2. Nemotron learns the nvMolKit skill") + 2
    ].source
    skill_result_cell = notebook.cells[
        heading_cell_index(notebook, "## 2. Nemotron learns the nvMolKit skill") + 4
    ].source
    later_request_cells = [
        notebook.cells[heading_cell_index(notebook, heading) + 3].source
        for heading, *_ in TOOL_STAGES[1:]
    ]
    skill_function_tree = ast.parse(skill_function_cell)
    capabilities_assignment = next(
        node
        for node in skill_function_tree.body[0].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "capabilities"
            for target in node.targets
        )
    )
    summary_assignment = next(
        node
        for node in skill_function_tree.body[0].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "summary"
            for target in node.targets
        )
    )
    summary_items = {
        key.value: value
        for key, value in zip(
            summary_assignment.value.keys, summary_assignment.value.values
        )
    }
    figure_context = ast.literal_eval(summary_items["figure_context"])
    skill_result_tree = ast.parse(skill_result_cell)
    capability_table_call = next(
        node
        for node in ast.walk(skill_result_tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "pd.DataFrame"
    )

    assert skill_function_cell.count("SKILL_PATH.read_text") == 1
    assert "ce151c15470991c8cb9a0efdd531a124c346ca5b" in skill_function_cell
    assert "https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/" in skill_function_cell
    assert "There is no CPU fallback" in skill_function_cell
    assert ast.literal_eval(capabilities_assignment.value) == EXPECTED_CAPABILITY_ROWS
    assert dotted_name(summary_items["capabilities"]) == "capabilities"
    assert figure_context == {
        "visual": "capability table",
        "rows": 5,
        "columns": ["Capability", "Entry point", "Role"],
    }
    assert ast.literal_eval(capability_table_call.args[0]) == EXPECTED_CAPABILITY_ROWS
    assert all(
        list(row) == ["Capability", "Entry point", "Role"]
        for row in ast.literal_eval(capability_table_call.args[0])
    )
    assert code.count('skill_artifact["skill_text"]') == 1
    assert all('"skill_grounding": skill_grounding' in cell for cell in later_request_cells)
    assert all("skill_text" not in cell for cell in later_request_cells)


def test_hosted_payloads_contain_only_json_safe_summaries_and_one_skill_text():
    notebook = read_notebook()
    hosted_calls = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if not isinstance(node, ast.Call):
                continue
            call_name = dotted_name(node.func)
            if call_name == "request_and_execute_step":
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                hosted_calls.append((call_name, keywords["context"], node))
            elif call_name == "request_brief_interpretation":
                hosted_calls.append((call_name, node.args[2], node))
                hosted_calls.append((call_name, node.args[3], node))

    assert len(hosted_calls) == 15
    forbidden_local_keys = {
        "molecules",
        "frame",
        "fingerprints",
        "tensor",
        "active_bits",
        "result",
        "matrix",
        "assignments",
        "clusters",
        "reported_cluster_sizes",
    }
    skill_text_accesses = []
    for call_name, payload, call in hosted_calls:
        for artifact_name, path in artifact_accesses(payload):
            assert path, f"Whole local artifact exposed: {artifact_name}"
            assert path[0] not in forbidden_local_keys
            if path[0] == "summary":
                continue
            assert (
                call_name == "request_brief_interpretation"
                and dotted_name(call.args[1]) == "skill_decision"
                and artifact_name == "skill_artifact"
                and path == ("skill_text",)
            )
            skill_text_accesses.append((artifact_name, path))

    assert skill_text_accesses == [("skill_artifact", ("skill_text",))]


def test_brief_contexts_state_each_stage_specific_scientific_boundary():
    notebook = read_notebook()
    expected_boundaries = {
        "## 2. Nemotron learns the nvMolKit skill": (
            "Explain the documented capabilities and GPU/API limitations."
        ),
        "## 3. Molecular sample": (
            "The 24-molecule preview cannot establish whole-library chemistry."
        ),
        "### 4.1 Morgan fingerprints": (
            "Interpret representation and density only; do not infer biological activity."
        ),
        "### 4.2 All-pairs Tanimoto similarity": (
            "Explain the off-diagonal distribution and most-similar pair without "
            "inferring shared biological activity."
        ),
        "### 4.3 Fused Butina clusters": (
            "Discuss fragmentation, diversity, singletons, and cutoff sensitivity "
            "without biological claims."
        ),
    }

    for heading, boundary in expected_boundaries.items():
        brief_cell = notebook.cells[heading_cell_index(notebook, heading) + 5]
        string_literals = {
            node.value
            for node in ast.walk(ast.parse(brief_cell.source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert boundary in string_literals


def test_sample_stage_keeps_fixed_raw_shape_filters_invalid_smiles_and_previews_24():
    notebook = read_notebook()
    heading_index = heading_cell_index(notebook, "## 3. Molecular sample")
    function_cell = notebook.cells[heading_index + 2].source
    result_cell = notebook.cells[heading_index + 4].source

    assert "pd.read_csv(DATA_PATH)" in function_cell
    assert "len(raw_sample) != 256" in function_cell
    assert '["molecule_id", "smiles"]' in function_cell
    assert "Chem.MolFromSmiles" in function_cell
    assert "molecule is not None" in function_cell
    assert "excluded_identifiers" in function_cell
    assert "zero valid molecules" in function_cell.lower()
    assert "Expected 256 valid" not in function_cell
    assert "raw_rows" in function_cell
    assert "valid_molecules" in function_cell
    assert "invalid_molecules" in function_cell
    assert "preview_count" in function_cell
    assert "figure_context" in function_cell
    assert "Invalid SMILES" in result_cell
    assert "excluded identifiers" in result_cell
    assert 'else "none"' in result_cell
    assert 'sample_artifact["molecules"][:24]' in result_cell
    assert "molsPerRow=6" in result_cell
    assert "Draw.MolsToGridImage" in result_cell


def test_similarity_chain_uses_gpu_artifacts_bounded_statistics_and_static_visuals():
    notebook = read_notebook()
    code = notebook_code(notebook)
    tree = ast.parse(code)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {dotted_name(node.func) for node in calls}
    fingerprint_function = notebook.cells[
        heading_cell_index(notebook, "### 4.1 Morgan fingerprints") + 2
    ].source
    fingerprint_result = notebook.cells[
        heading_cell_index(notebook, "### 4.1 Morgan fingerprints") + 4
    ].source
    similarity_function = notebook.cells[
        heading_cell_index(notebook, "### 4.2 All-pairs Tanimoto similarity") + 2
    ].source
    similarity_result = notebook.cells[
        heading_cell_index(notebook, "### 4.2 All-pairs Tanimoto similarity") + 4
    ].source
    cluster_function = notebook.cells[
        heading_cell_index(notebook, "### 4.3 Fused Butina clusters") + 2
    ].source
    cluster_result = notebook.cells[
        heading_cell_index(notebook, "### 4.3 Fused Butina clusters") + 4
    ].source

    assert {
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "torch.cuda.synchronize",
        "Draw.MolsToGridImage",
        "sns.heatmap",
        "plt.hist",
        "plt.bar",
    } <= called_names
    assert "fingerprint_radius" in fingerprint_function
    assert "fingerprint_size" in fingerprint_function
    assert "GetFingerprints" in fingerprint_function
    assert "GPU-resident" in fingerprint_function
    assert "packed" in fingerprint_function
    assert "active hashed bits" in fingerprint_function
    assert "torch.cuda.synchronize()" in fingerprint_function
    assert all(
        statistic in fingerprint_function
        for statistic in ("min_active_bits", "median_active_bits", "mean_active_bits", "max_active_bits")
    )
    assert "plt.hist" in fingerprint_result
    assert "Active Morgan fingerprint bits per molecule" in fingerprint_result

    similarity_tree = ast.parse(similarity_function)
    tanimoto_call = next(
        node
        for node in ast.walk(similarity_tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "crossTanimotoSimilarity"
    )
    assert len(tanimoto_call.args) == 1
    assert isinstance(tanimoto_call.args[0], ast.Subscript)
    assert dotted_name(tanimoto_call.args[0].value) == "fingerprint_artifact"
    assert tanimoto_call.args[0].slice.value == "fingerprints"
    assert "np.isfinite" in similarity_function
    assert "np.allclose" in similarity_function
    assert "Self-similarity on the diagonal" in similarity_function
    assert "np.triu_indices" in similarity_function
    assert all(
        key in similarity_function
        for key in ("q1", "median", "q3", "p90", "max_off_diagonal", "most_similar_nonidentical_pair_ids")
    )
    assert "sns.heatmap" in similarity_result
    assert "vmin=0" in similarity_result and "vmax=1" in similarity_result
    assert "Unordered" in similarity_result and "input order" in similarity_result
    assert "cluster_order" not in similarity_result

    fused_call = next(call for call in calls if dotted_name(call.func) == "fused_butina")
    assert isinstance(fused_call.args[0], ast.Call)
    assert dotted_name(fused_call.args[0].func) == "fingerprints.torch"
    cluster_task = notebook.cells[
        heading_cell_index(notebook, "### 4.3 Fused Butina clusters") + 1
    ].source
    assert "0.50 Tanimoto-distance cutoff" in cluster_task
    assert "Tanimoto-distance threshold" in cluster_function
    assert "similarity > 1 - cutoff" in cluster_function
    assert "Lowering it" in cluster_function
    cutoff_comment_index = cluster_function.splitlines().index(
        "    # The cutoff is a Tanimoto-distance threshold: similarity > 1 - cutoff."
    )
    assert cluster_function.splitlines()[cutoff_comment_index + 1] == (
        "    # Lowering it requires greater similarity and can create more singletons."
    )
    assert "\\n" not in cluster_function
    assert "singletons" in cluster_function.lower()
    assert "sorted(assigned_indices) != list(range(molecule_count))" in cluster_function
    assert "assignments" in cluster_function and '"clusters"' in cluster_function
    assert all(
        key in cluster_function
        for key in ("cluster_count", "singleton_count", "singleton_fraction", "largest_cluster_sizes")
    )
    assert "[:15]" in cluster_result
    assert "plt.bar" in cluster_result
    assert "singletons" in cluster_result.lower()
    assert "sns.heatmap" not in cluster_result
    assert code.count("json.dumps") >= 5


def test_notebook_contains_no_api_key_and_no_saved_execution_state():
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert source.count("nvapi-") == 1
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None


def test_sections_five_and_six_are_markdown_only_transitions_for_task_three():
    notebook = read_notebook()
    section_five_index = heading_cell_index(
        notebook, "## 5. Conformers and MMFF94"
    )
    task_three_tail = notebook.cells[section_five_index:]

    assert all(cell.cell_type == "markdown" for cell in task_three_tail)
    assert [cell.source.splitlines()[0] for cell in task_three_tail] == [
        "## 5. Conformers and MMFF94",
        "## 6. What the results mean",
    ]
    assert all(len(cell.source.split("\n\n", 1)) == 2 for cell in task_three_tail)


def test_setup_uses_brev_managed_python_and_leaves_jupyter_to_brev():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert setup.splitlines()[0] == "#!/bin/bash"
    assert '${HOME}/.venv/bin/python3' in setup
    assert "command -v python3.12" not in setup
    assert '"${PYTHON}" -m pip --version' in setup
    assert '"${PYTHON}" -m ensurepip --upgrade' in setup
    assert '"${PYTHON}" -m pip install --upgrade pip' in setup
    assert '"${PYTHON}" -m pip install -r requirements.txt' in setup
    assert '"${PYTHON}" - <<\'PY\'' in setup
    assert 'url = "http://127.0.0.1:8888/api"' in setup
    assert "/api/status" not in setup
    assert "while time.monotonic() < deadline" in setup
    assert "urllib.request.urlopen" in setup
    assert "response.geturl()" in setup
    assert "json.load(response)" in setup
    assert all(
        forbidden not in setup
        for forbidden in ("jupyter lab", "nohup", "PID_FILE", "kill ", "-m venv")
    )
    assert "jupyterlab==" not in requirements


def test_setup_runs_only_managed_runtime_and_probes_existing_jupyter(tmp_path):
    fake_home = tmp_path / "home"
    managed_bin = fake_home / ".venv" / "bin"
    fake_bin = tmp_path / "bin"
    managed_bin.mkdir(parents=True)
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations.log"

    managed_python = managed_bin / "python3"
    managed_python.write_text(
        """#!/bin/bash
set -u
case "${1:-}" in
  -c)
    printf 'VERSION_CHECK %s\n' "${2:-}" >>"${INVOCATION_LOG}"
    ;;
  --version)
    printf 'Python 3.12.13\n'
    ;;
  -m)
    printf 'MODULE %s %s %s\n' "${2:-}" "${3:-}" "${4:-}" >>"${INVOCATION_LOG}"
    if [[ "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
      exit 1
    fi
    ;;
  -)
    payload="$(</dev/stdin)"
    if [[ "${payload}" == *"torch.cuda.is_available"* ]]; then
      printf 'SMOKE\n' >>"${INVOCATION_LOG}"
    elif [[ "${payload}" == *"urllib.request.urlopen"* ]]; then
      printf 'HEALTH\n' >>"${INVOCATION_LOG}"
    else
      printf 'UNEXPECTED_STDIN\n' >>"${INVOCATION_LOG}"
      exit 91
    fi
    ;;
  *)
    printf 'UNEXPECTED %s\n' "$*" >>"${INVOCATION_LOG}"
    exit 92
    ;;
esac
""",
        encoding="utf-8",
    )
    managed_python.chmod(0o755)

    (fake_bin / "python3.12").write_text(
        """#!/bin/bash
printf 'FALLBACK %s\n' "$*" >>"${INVOCATION_LOG}"
exit 99
""",
        encoding="utf-8",
    )
    (fake_bin / "python3.12").chmod(0o755)
    (fake_bin / "uname").write_text(
        """#!/bin/bash
case "${1:-}" in
  -s) printf 'Linux\n' ;;
  -m) printf 'x86_64\n' ;;
  *) exit 93 ;;
esac
""",
        encoding="utf-8",
    )
    (fake_bin / "uname").chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "INVOCATION_LOG": str(invocation_log),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "launchable" / "setup.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert any("sys.implementation.name" in line for line in invocations)
    assert "MODULE pip --version " in invocations
    ensurepip_index = invocations.index("MODULE ensurepip --upgrade ")
    pip_upgrade_index = invocations.index("MODULE pip install --upgrade")
    requirements_index = invocations.index("MODULE pip install -r")
    smoke_index = invocations.index("SMOKE")
    assert "HEALTH" in invocations
    health_index = invocations.index("HEALTH")
    assert ensurepip_index < pip_upgrade_index < requirements_index < smoke_index < health_index
    assert all(not line.startswith("FALLBACK") for line in invocations)


def health_probe_source():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    start = setup.index("import json\nimport time\nimport urllib.error")
    return setup[start : setup.index("\nPY", start)]


def run_health_probe(server_mode):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if server_mode == "redirect":
                if self.path == "/api":
                    self.send_response(302)
                    self.send_header("Location", "/login")
                    self.end_headers()
                else:
                    body = b"<html>login</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return

            if self.path != "/api":
                self.send_error(404)
                return
            body = b'{"version": "4.4.5"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe = re.sub(
            r'url = "http://127\.0\.0\.1:8888/[^"]+"',
            f'url = "http://127.0.0.1:{server.server_port}/api"',
            health_probe_source(),
        )
        probe = probe.replace(
            "deadline = time.monotonic() + 60",
            "deadline = time.monotonic() + 0.2",
        ).replace("time.sleep(1)", "time.sleep(0.01)")
        return subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_health_probe_rejects_redirect_to_login_html():
    result = run_health_probe("redirect")

    assert result.returncode != 0
    assert "did not become healthy" in result.stderr


def test_health_probe_accepts_versioned_api_json():
    result = run_health_probe("valid")

    assert result.returncode == 0, result.stderr
    assert "health probe passed" in result.stdout


def test_notebook_prompts_for_missing_api_key_without_exposing_it():
    notebook = read_notebook()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    tree = ast.parse(code)
    getpass_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "getpass"
    ]
    assert len(getpass_calls) == 1
    getpass_call = getpass_calls[0]
    getpass_if = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and getpass_call in list(ast.walk(node))
    )

    assert 'os.environ.get("NVIDIA_API_KEY"' in code
    assert 'os.environ["NVIDIA_API_KEY"]' not in code
    assert "NEMOTRON_MODEL" not in code
    assert 'model = "nvidia/nemotron-3-nano-30b-a3b"' in code
    assert getpass_call.args[0].value == (
        "Hosted NVIDIA Developer API key from the Nemotron build.nvidia.com "
        "model page (starts with nvapi-; bare key only; input hidden): "
    )
    assert any(
        isinstance(node, ast.Raise) for node in ast.walk(getpass_if)
    ), "The missing-key fallback must reject empty input."
    assert "api_key" not in {
        dotted_name(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "print"
        for arg in node.args
    }


def test_launch_instructions_use_brev_managed_jupyter_and_hidden_key_prompt():
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for instructions in (fields, readme):
        assert "Brev-managed Jupyter" in instructions
        assert "Enable Jupyter" in instructions
        assert "hidden" in instructions.lower()
        assert "not rely on setup-variable persistence" in instructions
        assert "NEMOTRON_MODEL" not in instructions

    assert "hidden notebook prompt" in read_notebook().cells[1].source


def test_docs_assign_tool_contract_and_execution_to_the_correct_components():
    intro = read_notebook().cells[0].source.lower()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "agent toolkit skill" in intro
    assert "structured tool call" in intro
    assert "notebook validates and executes it" in intro
    assert "tool contract" in readme
    assert "notebook" in readme and "executes" in readme
    for documentation in (intro, readme):
        assert "model executes python" not in documentation
        assert "dynamically loaded" not in documentation
