import ast
import http.server
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest


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

ALL_TOOL_NAMES = tuple(stage[1] for stage in TOOL_STAGES) + (
    "generate_and_optimize_conformers",
)

ANALYSIS_SUMMARY_STAGES = {
    "skill": "skill_artifact",
    "sample": "sample_artifact",
    "fingerprints": "fingerprint_artifact",
    "similarity": "similarity_artifact",
    "clusters": "cluster_artifact",
    "conformers_mmff94": "conformer_artifact",
}

SCIENTIFIC_BOUNDARY = (
    "These computational results do not establish binding, biological activity, "
    "ADMET, efficacy, safety, synthesizability, clinical relevance, or "
    "experimentally validated conformations. Sampled force-field minima are not "
    "global or experimental conformations."
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


class _FakeSeries(list):
    def astype(self, _type):
        return _FakeSeries(map(str, self))

    def tolist(self):
        return list(self)


class _Matrix(list):
    def __init__(self, rows):
        super().__init__(map(list, rows))
        self.shape = (len(self), len(self[0]) if self else 0)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return super().__getitem__(key)
        rows, column = key
        selected = super().__getitem__(rows) if isinstance(rows, slice) else [super().__getitem__(index) for index in rows]
        return [row[column] for row in selected]


class _FakeNumpy:
    @staticmethod
    def isfinite(value):
        if isinstance(value, _Matrix):
            return SimpleNamespace(
                all=lambda: all(math.isfinite(item) for row in value for item in row)
            )
        return math.isfinite(float(value))

    array = staticmethod(lambda rows, dtype=None: _Matrix(rows))
    ceil = staticmethod(math.ceil)
    maximum = staticmethod(lambda values, floor: [max(value, floor) for value in values])

    @staticmethod
    def ptp(matrix, axis):
        assert axis == 0
        return [
            max(row[column] for row in matrix) - min(row[column] for row in matrix)
            for column in range(matrix.shape[1])
        ]


class _FakePoint3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = map(float, (x, y, z))

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class _FakeConformer:
    def __init__(self, atom_count):
        self.positions = [_FakePoint3D(0, 0, 0) for _ in range(atom_count)]

    def SetAtomPosition(self, index, point):
        self.positions[index] = point

    def GetAtomPosition(self, index):
        return self.positions[index]


class _FakeMolecule:
    def __init__(
        self,
        *,
        heavy_atoms,
        cluster_id,
        eligible=True,
        embed_count=3,
        energies=(),
        convergence=(),
        atom_count=3,
        bonds=((0, 1), (1, 2)),
    ):
        self.heavy_atoms, self.cluster_id = heavy_atoms, cluster_id
        self.eligible, self.embed_count = eligible, embed_count
        self.energies, self.convergence = list(energies), list(convergence)
        self.atom_count, self.bonds, self.conformers = atom_count, list(bonds), []

    def copy(self):
        return _FakeMolecule(
            heavy_atoms=self.heavy_atoms,
            cluster_id=self.cluster_id,
            eligible=self.eligible,
            embed_count=self.embed_count,
            energies=self.energies,
            convergence=self.convergence,
            atom_count=self.atom_count,
            bonds=self.bonds,
        )

    def embed(self, requested):
        self.conformers = [
            _FakeConformer(self.atom_count)
            for _ in range(min(self.embed_count, requested))
        ]

    def GetNumHeavyAtoms(self):
        return self.heavy_atoms

    def GetNumConformers(self):
        return len(self.conformers)

    def GetNumAtoms(self):
        return self.atom_count

    def GetConformer(self, index):
        return self.conformers[index]

    def GetAtoms(self):
        return [SimpleNamespace(GetAtomicNum=lambda: 6) for _ in range(self.atom_count)]

    def GetBonds(self):
        return [
            SimpleNamespace(
                GetBeginAtomIdx=lambda begin=begin: begin,
                GetEndAtomIdx=lambda end=end: end,
            )
            for begin, end in self.bonds
        ]


class _Buffer:
    def __init__(self, values):
        self.values = list(values)

    def torch(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def numpy(self):
        return _Matrix(self.values)

    def __len__(self):
        return len(self.values)


class _RecordingAxis:
    def __init__(self):
        self.scatter_calls, self.plot_calls, self.text_calls = [], [], []
        self.annotations = []
        self.xlabel = self.ylabel = self.title = None
        self.legend_called = False

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))

    def annotate(self, *args, **kwargs):
        self.annotations.append((args, kwargs))

    def text(self, *args, **kwargs):
        self.text_calls.append((args, kwargs))

    def set_xlabel(self, value):
        self.xlabel = value

    def set_ylabel(self, value):
        self.ylabel = value

    def set_title(self, value):
        self.title = value

    def legend(self, *args, **kwargs):
        self.legend_called = True

    def set_xticks(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def axis(self, *args, **kwargs):
        pass

    def set_axis_off(self):
        pass

    def set_box_aspect(self, value):
        pass


class _RecordingFigure:
    def __init__(self):
        self.axes, self.closed = [], False

    def add_subplot(self, *args, **kwargs):
        axis = _RecordingAxis()
        self.axes.append(axis)
        return axis

    def tight_layout(self):
        pass

    def suptitle(self, value):
        pass


class _HeadlessPlot:
    def subplots(self, *args, **kwargs):
        figure, axis = _RecordingFigure(), _RecordingAxis()
        figure.axes.append(axis)
        return figure, axis

    def figure(self, *args, **kwargs):
        return _RecordingFigure()

    def show(self):
        pass

    def close(self, figure):
        figure.closed = True


def _section_five_runtime(molecules, *, duplicate_pair=False, bad_shape=False):
    notebook = read_notebook()
    section_five_index = heading_cell_index(notebook, "## 5. Conformers and MMFF94")
    definition_source = next(
        cell.source
        for cell in notebook.cells[section_five_index:]
        if cell.cell_type == "code"
        and "def generate_and_optimize_conformers" in cell.source
    )

    class FakeChem:
        @staticmethod
        def Mol(molecule):
            return molecule.copy()

        @staticmethod
        def AddHs(molecule):
            return molecule

    class FakeAllChem:
        @staticmethod
        def MMFFHasAllMoleculeParams(molecule):
            return molecule.eligible

        @staticmethod
        def ETKDGv3():
            return SimpleNamespace(useRandomCoords=False, randomSeed=0)

    def fake_embed_molecules(mols, params, *, confsPerMolecule, maxIterations):
        assert params.useRandomCoords is True and params.randomSeed == 7
        assert maxIterations == -1
        for molecule in mols:
            molecule.embed(confsPerMolecule)

    def fake_mmff(mols, *, maxIters, output):
        assert maxIters == 500 and output is coordinate_output.DEVICE
        flat = []
        for mol_index, mol in enumerate(mols):
            conformer_order = list(range(mol.GetNumConformers()))
            if len(conformer_order) >= 3:
                conformer_order[:3] = [2, 0, 1]
            flat.extend(
                (mol_index, conf_index, mol)
                for conf_index in conformer_order
            )
        mol_indices = [entry[0] for entry in flat]
        conf_indices = [entry[1] for entry in flat]
        if duplicate_pair and len(flat) > 1:
            mol_indices[-1], conf_indices[-1] = mol_indices[0], conf_indices[0]
        per_molecule = [
            [
                _Buffer(
                    [[10.0 * mol_index + conf_index + atom, atom, -atom] for atom in range(mol.GetNumAtoms())]
                )
                for result_mol_index, conf_index, _result_mol in flat
                if result_mol_index == mol_index
            ]
            for mol_index, mol in enumerate(mols)
        ]
        if bad_shape:
            per_molecule[0][0] = _Buffer([[0.0, 0.0, 0.0]])
        return SimpleNamespace(
            energies=_Buffer([mol.energies[conf] for _, conf, mol in flat]),
            converged=_Buffer([mol.convergence[conf] for _, conf, mol in flat]),
            mol_indices=_Buffer(mol_indices),
            conf_indices=_Buffer(conf_indices),
            per_molecule=lambda: per_molecule,
        )

    plot = _HeadlessPlot()
    coordinate_output = SimpleNamespace(DEVICE=object())
    namespace = {
        "AllChem": FakeAllChem,
        "Chem": FakeChem,
        "ConformerArgs": object,
        "CoordinateOutput": coordinate_output,
        "EmbedMolecules": fake_embed_molecules,
        "MMFFOptimizeMoleculesConfs": fake_mmff,
        "Point3D": _FakePoint3D,
        "cluster_artifact": {
            "assignments": [molecule.cluster_id for molecule in molecules]
        },
        "json": json,
        "np": _FakeNumpy,
        "plt": plot,
        "sample_artifact": {
            "frame": {"molecule_id": _FakeSeries(f"mol-{i}" for i in range(len(molecules)))},
            "molecules": molecules,
        },
        "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    }
    exec(compile(definition_source, "<section-five>", "exec"), namespace)
    return namespace, plot


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

    assert len(request_calls) == len(brief_calls) == 6
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
    assert tool_name_literals == list(ALL_TOOL_NAMES)
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

    assert len(hosted_calls) == 18
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
        "optimized_molecules",
        "coordinates",
        "per_molecule",
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
        "## 5. Conformers and MMFF94": (
            "Discuss convergence, sampling, and within-molecule energy ranking only."
        ),
    }

    for heading, boundary in expected_boundaries.items():
        heading_index = heading_cell_index(notebook, heading)
        next_heading_index = next(
            (
                index
                for index in range(heading_index + 1, len(notebook.cells))
                if notebook.cells[index].cell_type == "markdown"
                and notebook.cells[index].source.startswith("#")
            ),
            len(notebook.cells),
        )
        brief_cell = next(
            cell
            for cell in notebook.cells[heading_index + 1 : next_heading_index]
            if cell.cell_type == "code"
            and "request_brief_interpretation" in cell.source
        )
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


def test_section_five_presents_tool_six_static_results_optional_view_and_brief_in_order():
    notebook = read_notebook()
    section_five_index = heading_cell_index(notebook, "## 5. Conformers and MMFF94")
    section_six_index = heading_cell_index(notebook, "## 6. What the results mean")
    section_cells = notebook.cells[section_five_index:section_six_index]

    assert section_cells[0].cell_type == "markdown"
    assert section_cells[0].source == "## 5. Conformers and MMFF94"
    assert section_cells[1].cell_type == "markdown"
    assert section_cells[1].source.startswith("**Task.**")
    assert all(
        phrase in section_cells[1].source
        for phrase in (
            "MMFF94-eligible representatives",
            "distinct clusters",
            "4 representatives",
            "4 conformers",
            "ETKDGv3",
            "nvMolKit MMFF94",
            "only within each molecule",
        )
    )

    function_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code"
        and "def generate_and_optimize_conformers" in cell.source
    )
    request_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code"
        and "request_and_execute_step" in cell.source
    )
    result_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code"
        and "plot_conformer_energies(" in cell.source
        and "plot_lowest_energy_conformers(" in cell.source
        and "def plot_conformer_energies" not in cell.source
    )
    optional_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code" and "import py3Dmol" in cell.source
    )
    brief_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code"
        and "request_brief_interpretation" in cell.source
    )

    assert 1 < function_index < request_index < result_index < optional_index < brief_index
    function_tree = ast.parse(section_cells[function_index].source)
    function = next(
        node
        for node in function_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_and_optimize_conformers"
    )
    assert [argument.arg for argument in function.args.args] == ["args"]
    assert dotted_name(function.args.args[0].annotation) == "ConformerArgs"

    request_cell = section_cells[request_index]
    request_tree = ast.parse(request_cell.source)
    request = next(
        node
        for node in ast.walk(request_tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "request_and_execute_step"
    )
    keywords = {keyword.arg: keyword.value for keyword in request.keywords}
    assert keywords["tool_name"].value == "generate_and_optimize_conformers"
    assert dotted_name(keywords["executor"]) == "generate_and_optimize_conformers"
    assert "# Validation completes before the executor runs." in request_cell.source
    assert "representative_count" in request_cell.source and "4" in request_cell.source
    assert "conformers_per_representative" in request_cell.source
    assert "**Requested tool:**" in request_cell.source
    assert "**Validated arguments:**" in request_cell.source
    for artifact_name, path in artifact_accesses(keywords["context"]):
        assert path == ("summary",), f"Unbounded Tool 6 context from {artifact_name}: {path}"

    result_cell = section_cells[result_index].source
    assert "pd.DataFrame" in result_cell
    assert "per_conformer_records" in result_cell
    assert "plot_conformer_energies(conformer_artifact)" in result_cell
    assert "plot_lowest_energy_conformers(conformer_artifact)" in result_cell

    optional_cell = section_cells[optional_index].source
    assert "selected_conformer_records" in optional_cell
    assert "except Exception" in optional_cell
    assert 'display("Optional interactive 3D view unavailable.")' in optional_cell
    assert "repr(" not in optional_cell and "str(error" not in optional_cell

    brief_cell = section_cells[brief_index].source
    assert 'conformer_artifact["summary"]' in brief_cell
    assert 'conformer_artifact["summary"]["figure_context"]' in brief_cell
    assert "request_brief_interpretation" in brief_cell
    assert 'display("Interpretation unavailable")' in brief_cell
    assert "except Exception" in brief_cell
    assert "repr(" not in brief_cell and "str(error" not in brief_cell


def test_conformer_function_enforces_selection_embedding_device_mmff_and_json_safe_accounting():
    notebook = read_notebook()
    section_five_index = heading_cell_index(notebook, "## 5. Conformers and MMFF94")
    section_six_index = heading_cell_index(notebook, "## 6. What the results mean")
    function_cell = next(
        cell
        for cell in notebook.cells[section_five_index:section_six_index]
        if cell.cell_type == "code"
        and "def generate_and_optimize_conformers" in cell.source
    )
    source = function_cell.source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {dotted_name(call.func) for call in calls}

    assert "# Representatives are chosen from distinct clusters by lower heavy-atom count, then stable molecule index." in source
    assert "AllChem.MMFFHasAllMoleculeParams" in called_names
    assert "Chem.AddHs" in called_names
    assert "GetNumHeavyAtoms" in source
    assert "sorted(" in source
    assert "eligible_distinct_cluster_count" in source
    assert "fewer distinct eligible representatives" in source.lower()
    assert "at least one" in source.lower()

    assert "AllChem.ETKDGv3" in called_names
    assert "useRandomCoords = True" in source
    assert "randomSeed = 7" in source
    embed_call = next(call for call in calls if dotted_name(call.func) == "EmbedMolecules")
    embed_keywords = {keyword.arg: keyword.value for keyword in embed_call.keywords}
    assert "maxIterations" in embed_keywords
    assert isinstance(embed_keywords["maxIterations"], ast.UnaryOp)
    assert isinstance(embed_keywords["maxIterations"].op, ast.USub)
    assert embed_keywords["maxIterations"].operand.value == 1
    assert "conformers_per_representative" in source
    assert "# Zero embeddings are recorded and excluded; partial embeddings continue to MMFF94." in source
    assert "zero_embedding_representatives" in source
    assert "partial_embedding_representatives" in source
    assert "generated_conformer_count" in source
    assert "all representatives produced zero conformers" in source.lower()

    mmff_call = next(
        call
        for call in calls
        if dotted_name(call.func) == "MMFFOptimizeMoleculesConfs"
    )
    mmff_keywords = {keyword.arg: keyword.value for keyword in mmff_call.keywords}
    assert mmff_keywords["maxIters"].value == 500
    assert dotted_name(mmff_keywords["output"]) == "CoordinateOutput.DEVICE"
    assert "torch.cuda.synchronize" in called_names
    assert all(name in source for name in ("energies", "convergence", "mol_indices", "conf_indices"))
    assert "math.isfinite" in called_names or "np.isfinite" in called_names
    assert "# Optimized device coordinates are copied back into their matching RDKit conformers for reliable static rendering." in source
    assert "Point3D" in called_names
    assert "SetAtomPosition" in source

    assert "# Raw MMFF energies are comparable only within one molecule; molecules are never ranked against each other." in source
    for required_key in (
        "requested_representative_count",
        "requested_conformers_per_representative",
        "selected_representative_count",
        "generated_conformer_count",
        "attempted_conformer_count",
        "converged_conformer_count",
        "unconverged_conformer_count",
        "representative_identifiers",
        "per_conformer_records",
        "selected_conformer_records",
        "selected_conformer_ids",
        "figure_context",
    ):
        assert required_key in source
    for per_conformer_key in (
        "representative_id",
        "molecule_index",
        "cluster_id",
        "conformer_index",
        "energy_kcal_mol",
        "converged",
    ):
        assert per_conformer_key in source
    assert "None" in source
    assert "optimized_molecules" in source
    assert "per_molecule" in source
    summary_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "summary"
            for target in node.targets
        )
    )
    assert isinstance(summary_assignment.value, ast.Dict)
    summary_keys = {
        key.value
        for key in summary_assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "molecules",
        "optimized_molecules",
        "coordinates",
        "per_molecule",
        "rdkit_molecules",
    }.isdisjoint(summary_keys)


def test_static_conformer_plots_are_matplotlib_complete_and_precede_py3dmol():
    notebook = read_notebook()
    section_five_index = heading_cell_index(notebook, "## 5. Conformers and MMFF94")
    section_six_index = heading_cell_index(notebook, "## 6. What the results mean")
    section_cells = notebook.cells[section_five_index:section_six_index]
    definition_cell = next(
        cell
        for cell in section_cells
        if cell.cell_type == "code" and "def plot_conformer_energies" in cell.source
    )
    tree = ast.parse(definition_cell.source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert {"plot_conformer_energies", "plot_lowest_energy_conformers"} <= set(functions)
    energy_source = ast.get_source_segment(
        definition_cell.source, functions["plot_conformer_energies"]
    )
    conformer_source = ast.get_source_segment(
        definition_cell.source, functions["plot_lowest_energy_conformers"]
    )

    assert "per_conformer_records" in energy_source
    assert "converged" in energy_source and "unconverged" in energy_source.lower()
    assert "missing" in energy_source.lower() and "None" in energy_source
    assert "kcal/mol" in energy_source
    assert "within molecule" in energy_source.lower()
    assert "scatter" in energy_source
    assert "legend" in energy_source
    assert "title" in energy_source

    assert 'projection="3d"' in conformer_source
    assert "selected_conformer_records" in conformer_source
    assert "GetConformer" in conformer_source
    assert "GetAtomPosition" in conformer_source
    assert "GetBonds" in conformer_source
    assert "GetBeginAtomIdx" in conformer_source and "GetEndAtomIdx" in conformer_source
    assert ".scatter(" in conformer_source and ".plot(" in conformer_source
    assert "up to 6" in conformer_source.lower() or "[:6]" in conformer_source
    assert "no conformer converged" in conformer_source.lower()
    assert "py3Dmol" not in conformer_source

    static_call_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code"
        and "plot_conformer_energies(conformer_artifact)" in cell.source
        and "plot_lowest_energy_conformers(conformer_artifact)" in cell.source
        and "def plot_conformer_energies" not in cell.source
    )
    optional_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code" and "import py3Dmol" in cell.source
    )
    assert static_call_index < optional_index
    assert "# Static Matplotlib figures remain authoritative and render before the optional py3Dmol enhancement." in section_cells[static_call_index].source


def test_conformer_runtime_accounts_original_request_and_selects_within_molecules():
    molecules = [
        _FakeMolecule(
            heavy_atoms=4,
            cluster_id=0,
            embed_count=3,
            energies=[70.0, 60.0, 50.0],
            convergence=[1, 1, 1],
        ),
        _FakeMolecule(
            heavy_atoms=2,
            cluster_id=0,
            embed_count=3,
            energies=[9.0, float("nan"), 2.0],
            convergence=[1, 0, 1],
        ),
        _FakeMolecule(
            heavy_atoms=2,
            cluster_id=1,
            embed_count=0,
            energies=[],
            convergence=[],
        ),
        _FakeMolecule(
            heavy_atoms=2,
            cluster_id=2,
            embed_count=2,
            energies=[100.0, 90.0],
            convergence=[1, 1],
        ),
        _FakeMolecule(
            heavy_atoms=2,
            cluster_id=2,
            embed_count=3,
            energies=[30.0, 20.0, 10.0],
            convergence=[1, 1, 1],
        ),
        _FakeMolecule(
            heavy_atoms=1,
            cluster_id=3,
            eligible=False,
            embed_count=3,
            energies=[3.0, 2.0, 1.0],
            convergence=[1, 1, 1],
        ),
    ]
    namespace, _plot = _section_five_runtime(molecules)

    artifact = namespace["generate_and_optimize_conformers"](
        SimpleNamespace(
            representative_count=4,
            conformers_per_representative=3,
        )
    )

    summary = artifact["summary"]
    assert summary["requested_representative_count"] == 4
    assert summary["requested_conformers_per_representative"] == 3
    assert summary["requested_conformer_count"] == 12
    assert summary["selected_representative_count"] == 3
    assert summary["generated_conformer_count"] == 5
    assert summary["attempted_conformer_count"] == 5
    assert summary["converged_conformer_count"] == 4
    assert summary["unconverged_conformer_count"] == 1
    assert summary["selection_notice"] == (
        "Selected fewer distinct eligible representatives than requested: 3 of 4."
    )
    assert [
        representative["molecule_index"]
        for representative in summary["representative_identifiers"]
    ] == [1, 2, 3]
    assert summary["zero_embedding_representatives"] == [
        {
            "representative_id": "mol-2",
            "molecule_index": 2,
            "cluster_id": 1,
            "generated_conformer_count": 0,
            "requested_conformer_count": 3,
        }
    ]
    assert summary["partial_embedding_representatives"] == [
        {
            "representative_id": "mol-3",
            "molecule_index": 3,
            "cluster_id": 2,
            "generated_conformer_count": 2,
            "requested_conformer_count": 3,
        }
    ]
    assert len(summary["per_conformer_records"]) == 5
    assert all(
        set(record)
        >= {
            "representative_id",
            "molecule_index",
            "cluster_id",
            "conformer_index",
            "energy_kcal_mol",
            "converged",
        }
        for record in summary["per_conformer_records"]
    )
    assert summary["per_conformer_records"][1]["energy_kcal_mol"] is None
    assert [
        (
            record["representative_id"],
            record["conformer_index"],
            record["energy_kcal_mol"],
        )
        for record in summary["selected_conformer_records"]
    ] == [
        ("mol-1", 2, 2.0),
        ("mol-3", 1, 90.0),
    ]
    assert summary["selected_conformer_ids"] == [
        "mol-1:conf-2",
        "mol-3:conf-1",
    ]
    first_optimized_molecule = artifact["optimized_molecules"][0]
    assert [
        first_optimized_molecule.GetConformer(conformer_index).positions[2].x
        for conformer_index in range(3)
    ] == [2.0, 3.0, 4.0]
    first_selected = summary["selected_conformer_records"][0]
    selected_point = first_optimized_molecule.GetConformer(
        first_selected["conformer_index"]
    ).positions[2]
    assert first_selected["conformer_index"] == 2
    assert first_selected["energy_kcal_mol"] == 2.0
    assert selected_point.x == 4.0
    copied_point = artifact["optimized_molecules"][1].GetConformer(1).positions[2]
    assert (copied_point.x, copied_point.y, copied_point.z) == (13.0, 2.0, -2.0)


def test_conformer_runtime_excludes_zero_embeddings_and_rejects_all_zero():
    molecules = [
        _FakeMolecule(
            heavy_atoms=index + 1,
            cluster_id=index,
            embed_count=0,
            energies=[],
            convergence=[],
        )
        for index in range(3)
    ]
    namespace, _plot = _section_five_runtime(molecules)

    with pytest.raises(RuntimeError, match="All representatives produced zero"):
        namespace["generate_and_optimize_conformers"](
            SimpleNamespace(
                representative_count=3,
                conformers_per_representative=3,
            )
        )


@pytest.mark.parametrize(
    ("runtime_options", "error_message"),
    [
        ({"duplicate_pair": True}, "indices are incomplete or duplicated"),
        ({"bad_shape": True}, "coordinate array has the wrong shape"),
    ],
)
def test_conformer_runtime_validates_flat_pairs_and_nested_coordinate_shapes(
    runtime_options, error_message
):
    molecules = [
        _FakeMolecule(
            heavy_atoms=index + 1,
            cluster_id=index,
            embed_count=1,
            energies=[float(index + 1)],
            convergence=[1],
        )
        for index in range(3)
    ]
    namespace, _plot = _section_five_runtime(molecules, **runtime_options)

    with pytest.raises(RuntimeError, match=error_message):
        namespace["generate_and_optimize_conformers"](
            SimpleNamespace(
                representative_count=3,
                conformers_per_representative=3,
            )
        )


def test_conformer_energy_plot_draws_every_attempt_and_all_marker_classes():
    namespace, plot = _section_five_runtime([])
    artifact = {
        "summary": {
            "per_conformer_records": [
                {
                    "representative_id": "mol-a",
                    "conformer_index": 0,
                    "energy_kcal_mol": 1.0,
                    "converged": True,
                },
                {
                    "representative_id": "mol-a",
                    "conformer_index": 1,
                    "energy_kcal_mol": 2.0,
                    "converged": False,
                },
                {
                    "representative_id": "mol-b",
                    "conformer_index": 0,
                    "energy_kcal_mol": None,
                    "converged": False,
                },
            ]
        }
    }

    figure = namespace["plot_conformer_energies"](artifact)
    axis = figure.axes[0]

    assert len(axis.scatter_calls) == 3
    assert {
        call[1]["label"] for call in axis.scatter_calls
    } == {"Converged", "Unconverged", "Missing/non-finite energy"}
    assert len(axis.annotations) == 1
    assert axis.ylabel == "MMFF94 energy (kcal/mol)"
    assert "within molecule" in axis.title
    assert axis.legend_called is True
    plot.close(figure)
    assert figure.closed is True


def test_static_conformer_plot_draws_atoms_bonds_and_empty_explanation():
    namespace, plot = _section_five_runtime([])
    molecule = _FakeMolecule(
        heavy_atoms=3,
        cluster_id=0,
        embed_count=1,
        energies=[1.25],
        convergence=[1],
        bonds=((0, 1), (1, 2)),
    )
    molecule.embed(1)
    selected_artifact = {
        "optimized_molecules": [molecule],
        "summary": {
            "selected_conformer_records": [
                {
                    "representative_id": "mol-a",
                    "optimization_molecule_index": 0,
                    "conformer_index": 0,
                    "energy_kcal_mol": 1.25,
                }
            ]
        },
    }

    selected_figure = namespace["plot_lowest_energy_conformers"](
        selected_artifact
    )
    selected_axis = selected_figure.axes[0]
    assert len(selected_axis.scatter_calls) == 1
    assert len(selected_axis.plot_calls) == len(molecule.GetBonds()) == 2
    assert "within-molecule MMFF94" in selected_axis.title
    plot.close(selected_figure)

    empty_figure = namespace["plot_lowest_energy_conformers"](
        {
            "optimized_molecules": [],
            "summary": {"selected_conformer_records": []},
        }
    )
    empty_axis = empty_figure.axes[0]
    assert len(empty_axis.text_calls) == 1
    assert "No conformer converged" in empty_axis.text_calls[0][0][2]
    plot.close(empty_figure)
    assert selected_figure.closed is True and empty_figure.closed is True


def test_section_six_serializes_exact_summaries_tables_results_and_bounds_synthesis():
    notebook = read_notebook()
    section_six_index = heading_cell_index(notebook, "## 6. What the results mean")
    section_cells = notebook.cells[section_six_index:]

    assert section_cells[0].cell_type == "markdown"
    assert section_cells[0].source == "## 6. What the results mean"
    assert section_cells[1].cell_type == "markdown"
    intro = section_cells[1].source
    assert "detailed, PhD-level but presentation-readable interpretation" in intro
    assert "text-only" in intro
    assert "figure descriptions, not pixels" in intro

    code_cells = [cell for cell in section_cells if cell.cell_type == "code"]
    combined = "\n".join(cell.source for cell in code_cells)
    tree = ast.parse(combined)
    analysis_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "analysis_summary"
            for target in node.targets
        )
    )
    assert isinstance(analysis_assignment.value, ast.Dict)
    analysis_items = {
        key.value: value
        for key, value in zip(
            analysis_assignment.value.keys, analysis_assignment.value.values
        )
    }
    assert set(analysis_items) == set(ANALYSIS_SUMMARY_STAGES)
    for stage, artifact_name in ANALYSIS_SUMMARY_STAGES.items():
        assert artifact_subscript_path(analysis_items[stage]) == (
            artifact_name,
            ("summary",),
        )

    serialization_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "json.dumps"
        and any(
            isinstance(argument, ast.Name) and argument.id == "analysis_summary"
            for argument in node.args
        )
    )
    serialization_keywords = {
        keyword.arg: keyword.value for keyword in serialization_call.keywords
    }
    assert serialization_keywords["allow_nan"].value is False
    assert "pd.DataFrame" in combined
    assert '"stage"' in combined
    assert '"key quantitative result"' in combined
    assert "display(analysis_summary)" not in combined

    boundary_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "scientific_boundary"
            for target in node.targets
        )
    )
    assert ast.literal_eval(boundary_assignment.value) == SCIENTIFIC_BOUNDARY
    boundary_display_cells = [
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code" and "display(scientific_boundary)" in cell.source
    ]
    synthesis_cell_index = next(
        index
        for index, cell in enumerate(section_cells)
        if cell.cell_type == "code" and "request_final_synthesis" in cell.source
    )
    assert len(boundary_display_cells) == 2
    assert boundary_display_cells[0] < synthesis_cell_index < boundary_display_cells[1]

    final_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "request_final_synthesis"
    ]
    assert len(final_calls) == 1
    final_call = final_calls[0]
    assert [dotted_name(argument) for argument in final_call.args[:2]] == [
        "api_key",
        "analysis_summary",
    ]
    assert any(
        keyword.arg == "model" and dotted_name(keyword.value) == "model"
        for keyword in final_call.keywords
    )
    synthesis_cell = section_cells[synthesis_cell_index].source
    assert 'display("Final synthesis unavailable")' in synthesis_cell
    assert "except Exception" in synthesis_cell
    assert "repr(" not in synthesis_cell and "str(error" not in synthesis_cell


def test_readme_describes_six_guided_calls_visual_cadence_and_acceptance_boundaries():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "six forced" in lowered
    assert "vendored" in lowered and "skill" in lowered and "runtime" in lowered
    assert all(tool_name in readme for tool_name in ALL_TOOL_NAMES)
    assert all(
        visual in lowered
        for visual in (
            "molecule grid",
            "fingerprint histogram",
            "tanimoto heatmap",
            "cluster",
            "energy",
            "static 3d",
        )
    )
    assert "six brief interpretations" in lowered
    assert "detailed" in lowered and "synthesis" in lowered
    assert "text-only" in lowered and "figure_context" in readme
    assert all(role in lowered for role in ("brev", "nemotron", "notebook", "nvmolkit", "rdkit"))
    assert "hidden" in lowered and "nvidia_api_key" in lowered
    assert all(
        boundary in lowered
        for boundary in (
            "local deterministic",
            "gpu acceptance",
            "hosted inference acceptance",
        )
    )
    assert "not yet live-qualified" in lowered
    assert "model executes python" not in lowered


def test_task_four_does_not_change_frozen_runtime_launch_or_data_files():
    frozen_paths = (
        "requirements.txt",
        "launchable/setup.sh",
        "launchable/fields.md",
        "data/sample_molecules.csv",
    )
    for relative_path in frozen_paths:
        committed = subprocess.run(
            [
                "git",
                "show",
                f"31c8567b6ed743e56e87ee3475b4c143a7614c9b:{relative_path}",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert (REPO_ROOT / relative_path).read_bytes() == committed


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
