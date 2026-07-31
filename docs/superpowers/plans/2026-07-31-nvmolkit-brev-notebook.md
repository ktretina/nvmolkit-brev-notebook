# Minimal nvMolKit Brev Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Brev Launchable packet whose single guided notebook uses hosted Nemotron planning and local GPU-backed nvMolKit to fingerprint, compare, cluster, embed, optimize, and visualize a small molecular library.

**Architecture:** A strict Pydantic contract converts one hosted Nemotron response into a bounded workflow plan; arbitrary generated code is never executed. The notebook owns the linear teaching flow and nvMolKit calls, while a single helper module owns API interaction and plan validation. Brev runs one setup script that creates the Python environment and starts JupyterLab.

**Tech Stack:** Python 3.12, JupyterLab, nvMolKit 0.5.0, CUDA-enabled PyTorch, RDKit, OpenAI Python client against NVIDIA's OpenAI-compatible endpoint, Pydantic, pandas, seaborn/matplotlib, py3Dmol, pytest, nbformat.

---

## File Map

- `demo_agent.py` — strict plan schema, Nemotron prompt, hosted API calls, fallback decision, and bounded final explanation.
- `notebooks/nvmolkit_nemotron_demo.ipynb` — the only end-user interface and all chemistry/visualization cells.
- `data/sample_molecules.csv` — exactly 256 attributed ChEMBL records with `molecule_id` and `smiles` columns.
- `data/PROVENANCE.md` — exact ChEMBL query date, API URL, selection rule, and license/source links.
- `requirements.txt` — the only dependency manifest.
- `launchable/setup.sh` — idempotent environment setup and JupyterLab startup.
- `launchable/fields.md` — paste-ready Brev Console configuration.
- `tests/test_demo_agent.py` — plan-contract and mocked-client tests.
- `tests/test_notebook.py` — notebook structure, secret, and required-story checks.
- `tests/test_gpu_acceptance.py` — opt-in GPU smoke test for the exact nvMolKit operations.
- `.gitignore` — Python, notebook checkpoint, environment, log, and executed-notebook exclusions.
- `README.md` — concise use, architecture, claim boundaries, and acceptance instructions.

Do not add a web application, container, service, benchmark harness, database, or code-generation executor.

### Task 1: Scaffold the repository and strict agent-plan contract

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `demo_agent.py`
- Create: `tests/test_demo_agent.py`

- [ ] **Step 1: Add the minimal dependency manifest and ignores**

Create `requirements.txt` with one package per line:

```text
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
nvmolkit==0.5.0
jupyterlab==4.4.5
openai==1.97.1
pydantic==2.11.7
pandas==2.3.1
matplotlib==3.10.3
seaborn==0.13.2
py3Dmol==2.5.2
pytest==8.4.1
nbformat==5.10.4
```

Before implementation, confirm the pinned PyTorch CUDA wheel remains available and compatible with the Brev driver. If it is not, change only the PyTorch CUDA 12.x pin and record the verified version in the commit.

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
*.pyc
*.log
notebooks/executed/
```

- [ ] **Step 2: Write failing tests for the bounded plan schema**

Create `tests/test_demo_agent.py`:

```python
import json

import pytest
from pydantic import ValidationError

from demo_agent import DEFAULT_PLAN, WorkflowPlan, parse_plan


def test_valid_plan_is_accepted() -> None:
    plan = parse_plan(json.dumps(DEFAULT_PLAN))
    assert plan.fingerprint_radius == 2
    assert plan.fingerprint_size == 1024
    assert plan.cluster_cutoff == 0.5
    assert plan.representative_count == 4
    assert plan.conformers_per_representative == 4


@pytest.mark.parametrize(
    "payload",
    [
        {**DEFAULT_PLAN, "fingerprint_radius": 4},
        {**DEFAULT_PLAN, "fingerprint_size": 4096},
        {**DEFAULT_PLAN, "cluster_cutoff": 0.0},
        {**DEFAULT_PLAN, "representative_count": 7},
        {**DEFAULT_PLAN, "conformers_per_representative": 9},
        {**DEFAULT_PLAN, "execute_python": "import os"},
    ],
)
def test_invalid_or_extra_fields_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_plan(json.dumps(payload))


def test_prose_wrapped_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_plan(f"Here is the plan: {json.dumps(DEFAULT_PLAN)}")
```

- [ ] **Step 3: Run the contract tests and confirm the intended failure**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pydantic==2.11.7 pytest==8.4.1
.venv/bin/pytest tests/test_demo_agent.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'demo_agent'`.

- [ ] **Step 4: Implement the smallest strict plan model**

Create `demo_agent.py` with:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PLAN: dict[str, int | float] = {
    "fingerprint_radius": 2,
    "fingerprint_size": 1024,
    "cluster_cutoff": 0.5,
    "representative_count": 4,
    "conformers_per_representative": 4,
}


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fingerprint_radius: Literal[2, 3] = 2
    fingerprint_size: Literal[1024, 2048] = 1024
    cluster_cutoff: float = Field(default=0.5, ge=0.2, le=0.8)
    representative_count: int = Field(default=4, ge=1, le=6)
    conformers_per_representative: int = Field(default=4, ge=1, le=8)


def parse_plan(raw: str) -> WorkflowPlan:
    return WorkflowPlan.model_validate_json(raw)
```

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/pytest tests/test_demo_agent.py -v`

Expected: `8 passed`.

- [ ] **Step 6: Commit the contract scaffold**

```bash
git add .gitignore requirements.txt demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add bounded Nemotron plan contract"
```

### Task 2: Add the hosted Nemotron planner and explanation calls

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`

- [ ] **Step 1: Add failing mocked-client tests**

Append to `tests/test_demo_agent.py`:

```python
from types import SimpleNamespace

from demo_agent import PlanDecision, request_explanation, request_plan


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = next(self.contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def test_request_plan_accepts_valid_model_json() -> None:
    client = FakeClient([json.dumps(DEFAULT_PLAN)])
    decision = request_plan("secret", client=client)
    assert decision.source == "nemotron"
    assert decision.error is None
    assert decision.plan == WorkflowPlan(**DEFAULT_PLAN)


def test_request_plan_uses_labeled_default_after_invalid_json() -> None:
    client = FakeClient(["not json"])
    decision = request_plan("secret", client=client)
    assert decision.source == "default_after_error"
    assert decision.plan == WorkflowPlan(**DEFAULT_PLAN)
    assert "validation" in decision.error.lower()


def test_request_plan_uses_labeled_default_after_api_error() -> None:
    class BrokenCompletions:
        def create(self, **kwargs: object) -> None:
            raise RuntimeError("offline")

    client = SimpleNamespace(chat=SimpleNamespace(completions=BrokenCompletions()))
    decision = request_plan("secret", client=client)
    assert decision.source == "default_after_error"
    assert "offline" in decision.error


def test_request_plan_requires_api_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        request_plan("")


def test_explanation_prompt_preserves_claim_boundaries() -> None:
    client = FakeClient(["These are computed descriptors, not biological evidence."])
    text = request_explanation(
        "secret",
        {"molecules": 256, "clusters": 12, "optimized_conformers": 16},
        client=client,
    )
    call = client.chat.completions.calls[0]
    prompt = str(call["messages"])
    assert "not evidence of binding" in prompt
    assert "computed descriptors" in text
```

- [ ] **Step 2: Run the new tests and confirm failure**

Run: `.venv/bin/pytest tests/test_demo_agent.py -v`

Expected: import fails because `PlanDecision`, `request_plan`, and `request_explanation` are not defined.

- [ ] **Step 3: Implement the hosted client boundary**

Append the following imports and definitions to `demo_agent.py`:

```python
import json
from dataclasses import dataclass
from typing import Any

from openai import APIError, OpenAI
from pydantic import ValidationError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

PLAN_SYSTEM_PROMPT = """You plan one bounded nvMolKit demonstration.
nvMolKit is a GPU-accelerated, batched cheminformatics library. The executor will
always compute Morgan fingerprints, pairwise Tanimoto similarity, Butina clusters,
ETKDGv3 conformers, and MMFF94 geometry optimization. Choose only parameter values.
Return exactly one JSON object with these keys and no prose:
fingerprint_radius, fingerprint_size, cluster_cutoff, representative_count,
conformers_per_representative. Do not request code execution or make biological,
binding, activity, ADMET, efficacy, safety, or clinical claims.
"""


@dataclass(frozen=True)
class PlanDecision:
    plan: WorkflowPlan
    source: Literal["nemotron", "default_after_error"]
    error: str | None
    raw: str | None


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def request_plan(
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> PlanDecision:
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is required for Nemotron planning")
    active_client = client or _client(api_key)
    raw: str | None = None
    try:
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": "Plan the introductory workflow."},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = response.choices[0].message.content or ""
        return PlanDecision(parse_plan(raw), "nemotron", None, raw)
    except (APIError, ValidationError, RuntimeError, ValueError, IndexError, AttributeError) as exc:
        return PlanDecision(
            WorkflowPlan(**DEFAULT_PLAN),
            "default_after_error",
            f"Plan validation or API error: {exc}",
            raw,
        )


def request_explanation(
    api_key: str,
    summary: dict[str, int | float | str],
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> str:
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is required for Nemotron explanation")
    active_client = client or _client(api_key)
    prompt = (
        "Explain this nvMolKit demo result in at most 120 words: "
        f"{json.dumps(summary, sort_keys=True)}. State that these are computed "
        "descriptors and force-field geometries, not evidence of binding, activity, "
        "ADMET, efficacy, safety, or clinical relevance."
    )
    response = active_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=220,
    )
    return response.choices[0].message.content or ""
```

Catch only anticipated response/validation failures. Do not add retries or a general exception handler.

- [ ] **Step 4: Install the mocked-test dependency and run tests**

Run:

```bash
.venv/bin/python -m pip install openai==1.97.1
.venv/bin/pytest tests/test_demo_agent.py -v
```

Expected: `13 passed`.

- [ ] **Step 5: Commit the hosted-agent boundary**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add hosted Nemotron planner"
```

### Task 3: Add the 256-molecule attributed sample

**Files:**
- Create: `data/sample_molecules.csv`
- Create: `data/PROVENANCE.md`

- [ ] **Step 1: Retrieve a deterministic ChEMBL sample**

Use the public ChEMBL API endpoint below on the implementation date, select records in returned order, retain the first 256 records with a non-empty canonical SMILES and molecule identifier, and canonicalize each SMILES with RDKit before writing the file:

```text
https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0
```

Run this mechanical retrieval command:

```bash
.venv/bin/python - <<'PY'
import csv
import json
import urllib.request
from pathlib import Path

from rdkit import Chem

url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0"
with urllib.request.urlopen(url) as response:
    payload = json.load(response)

rows = []
for record in payload["molecules"]:
    molecule_id = record.get("molecule_chembl_id")
    structures = record.get("molecule_structures") or {}
    mol = Chem.MolFromSmiles(structures.get("canonical_smiles", ""))
    if molecule_id and mol is not None:
        rows.append((molecule_id, Chem.MolToSmiles(mol, canonical=True)))
    if len(rows) == 256:
        break

assert len(rows) == 256
destination = Path("data/sample_molecules.csv")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["molecule_id", "smiles"])
    writer.writerows(rows)
PY
```

The resulting CSV must have this exact header and exactly 256 data rows:

```csv
molecule_id,smiles
```

Do not keep a dataset-builder framework in the repository. Inspect the resulting diff before staging it.

- [ ] **Step 2: Add provenance and license context**

Create `data/PROVENANCE.md`:

```markdown
# Sample provenance

`sample_molecules.csv` contains the first 256 valid molecule records returned by
the public ChEMBL molecule API query below on 2026-07-31, after RDKit canonical
SMILES parsing and removal of records without a molecule identifier or structure.

Source query:
<https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0>

ChEMBL documentation and licensing:

- <https://chembl.gitbook.io/chembl-interface-documentation>
- <https://chembl.gitbook.io/chembl-interface-documentation/downloads>

The file is included solely as a small demonstration input. Check the current
ChEMBL terms and the licenses of every input before production or redistribution.
```

If retrieval occurs after 2026-07-31, replace the date with the actual UTC retrieval date in both the file and README; do not preserve an inaccurate date.

- [ ] **Step 3: Validate count, uniqueness, and parsability**

Run this exact validation from the repository root:

```bash
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
from rdkit import Chem

rows = list(csv.DictReader(Path("data/sample_molecules.csv").open()))
assert len(rows) == 256
assert len({row["molecule_id"] for row in rows}) == 256
assert all(Chem.MolFromSmiles(row["smiles"]) is not None for row in rows)
print("validated 256 unique molecules")
PY
```

Expected: `validated 256 unique molecules`.

- [ ] **Step 4: Commit the sample and provenance**

```bash
git add data/sample_molecules.csv data/PROVENANCE.md
git commit -m "data: add attributed ChEMBL demo sample"
```

### Task 4: Build the single guided notebook and its static checks

**Files:**
- Create: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Create: `tests/test_notebook.py`

- [ ] **Step 1: Write failing notebook-structure tests**

Create `tests/test_notebook.py`:

```python
from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/nvmolkit_nemotron_demo.ipynb")


def notebook_text() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(str(cell.source) for cell in notebook.cells)


def test_notebook_is_valid_and_has_linear_story() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    headings = [
        "# nvMolKit + Nemotron",
        "## 1. Preflight",
        "## 2. Molecular sample",
        "## 3. Nemotron plan",
        "## 4. Fingerprints, similarity, and clusters",
        "## 5. Conformers and MMFF94",
        "## 6. What the results mean",
    ]
    text = notebook_text()
    assert all(heading in text for heading in headings)


def test_notebook_contains_required_gpu_calls_and_visuals() -> None:
    text = notebook_text()
    required = [
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
    assert all(item in text for item in required)


def test_notebook_has_no_embedded_secret_or_saved_output() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = notebook_text()
    assert "nvapi-" not in text
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
```

- [ ] **Step 2: Run the static tests and confirm failure**

Run:

```bash
.venv/bin/python -m pip install nbformat==5.10.4
.venv/bin/pytest tests/test_notebook.py -v
```

Expected: all three tests fail because the notebook does not exist.

- [ ] **Step 3: Create the notebook with exactly seven teaching sections**

Create a valid nbformat v4 notebook with no saved outputs. Use the headings asserted above and the following code, split into short cells at the indicated conceptual boundaries.

Preflight cell:

```python
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import py3Dmol
import seaborn as sns
import torch
from IPython.display import Markdown, display
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDistGeom import ETKDGv3

sys.path.insert(0, str(Path("..").resolve()))
from demo_agent import request_explanation, request_plan

import nvmolkit
from nvmolkit.clustering import fused_butina
from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
from nvmolkit.similarity import crossTanimotoSimilarity

assert torch.cuda.is_available(), "A CUDA-capable NVIDIA GPU is required."
assert os.environ.get("NVIDIA_API_KEY"), "Set the masked NVIDIA_API_KEY Launchable parameter."
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch/CUDA:", torch.__version__, torch.version.cuda)
print("nvMolKit:", nvmolkit.__version__)

probe = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN", "c1ccccc1"]]
probe_result = MorganFingerprintGenerator(radius=2, fpSize=1024).GetFingerprints(probe)
torch.cuda.synchronize()
assert tuple(probe_result.torch().shape) == (3, 32)
```

Dataset and grid cells:

```python
sample = pd.read_csv("../data/sample_molecules.csv")
mols = [Chem.MolFromSmiles(smiles) for smiles in sample["smiles"]]
valid = [(row, mol) for (_, row), mol in zip(sample.iterrows(), mols) if mol is not None]
assert len(valid) == 256, f"Expected 256 valid molecules, found {len(valid)}"
sample = pd.DataFrame([row for row, _ in valid]).reset_index(drop=True)
mols = [mol for _, mol in valid]
display(Draw.MolsToGridImage(mols[:24], molsPerRow=6, subImgSize=(180, 150)))
```

Agent-plan cell:

```python
decision = request_plan(
    os.environ["NVIDIA_API_KEY"],
    model=os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
)
display(Markdown(f"**Plan source:** `{decision.source}`"))
if decision.error:
    display(Markdown(f"**Fallback used:** {decision.error}"))
display(decision.plan.model_dump())
plan = decision.plan
```

Fingerprint, similarity, and clustering cells:

```python
fpgen = MorganFingerprintGenerator(
    radius=plan.fingerprint_radius,
    fpSize=plan.fingerprint_size,
)
fingerprints = fpgen.GetFingerprints(mols)
similarity_result = crossTanimotoSimilarity(fingerprints)
clusters, _cluster_sizes = fused_butina(fingerprints, cutoff=plan.cluster_cutoff)
torch.cuda.synchronize()

similarity = similarity_result.torch().cpu().numpy()
cluster_ids = [-1] * len(mols)
for cluster_id, members in enumerate(clusters):
    for molecule_index in members:
        cluster_ids[molecule_index] = cluster_id
assert similarity.shape == (256, 256)
assert len(cluster_ids) == 256 and min(cluster_ids) >= 0
assert bool(torch.isfinite(similarity_result.torch()).all())
sample["cluster"] = cluster_ids
print("Clusters:", sample["cluster"].nunique())
```

```python
order = sample.sort_values("cluster").index.to_numpy()
plt.figure(figsize=(9, 7))
sns.heatmap(similarity[order][:, order], cmap="viridis", vmin=0, vmax=1)
plt.title("Morgan fingerprint Tanimoto similarity, ordered by Butina cluster")
plt.xlabel("Molecules")
plt.ylabel("Molecules")
plt.show()
```

Representative selection, embedding, and optimization cells:

```python
representative_indices = (
    sample.groupby("cluster", sort=False).head(1).index[: plan.representative_count].tolist()
)
representative_ids = sample.loc[representative_indices, "molecule_id"].tolist()
representatives = [Chem.AddHs(Chem.Mol(mols[index])) for index in representative_indices]

params = ETKDGv3()
params.useRandomCoords = True
params.randomSeed = 7
EmbedMolecules(
    representatives,
    params,
    confsPerMolecule=plan.conformers_per_representative,
    maxIterations=-1,
)
energies = MMFFOptimizeMoleculesConfs(representatives, maxIters=500)

assert all(mol.GetNumConformers() > 0 for mol in representatives)
assert len(energies) == len(representatives)
assert all(len(values) == mol.GetNumConformers() for mol, values in zip(representatives, energies))
```

```python
views = []
for molecule_id, mol, molecule_energies in zip(representative_ids, representatives, energies):
    best_conf = min(range(len(molecule_energies)), key=molecule_energies.__getitem__)
    block = Chem.MolToMolBlock(mol, confId=best_conf)
    view = py3Dmol.view(width=350, height=280)
    view.addModel(block, "mol")
    view.setStyle({"stick": {}})
    view.zoomTo()
    display(Markdown(f"**{molecule_id}** — lowest computed MMFF94 energy: {molecule_energies[best_conf]:.2f}"))
    view.show()
    views.append(view)
```

Result explanation cell:

```python
summary = {
    "molecules": len(mols),
    "clusters": int(sample["cluster"].nunique()),
    "representatives": len(representatives),
    "optimized_conformers": sum(mol.GetNumConformers() for mol in representatives),
}
explanation = request_explanation(
    os.environ["NVIDIA_API_KEY"],
    summary,
    model=os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
)
display(Markdown(explanation))
display(Markdown(
    "**Boundary:** These are computed fingerprints, similarity relationships, clusters, "
    "and force-field geometries—not evidence of binding, biological activity, ADMET, "
    "efficacy, safety, synthesizability, or clinical relevance."
))
```

In the introductory Markdown, link directly to the nvMolKit repository, public documentation, and Agent Toolkit skill. Explain that the hosted Nemotron model plans/explains while nvMolKit performs the chemistry computations on the Brev GPU.

- [ ] **Step 4: Run notebook static checks**

Run: `.venv/bin/pytest tests/test_notebook.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit the notebook**

```bash
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py
git commit -m "feat: add guided nvMolKit Nemotron notebook"
```

### Task 5: Add the Brev setup script and Console packet

**Files:**
- Create: `launchable/setup.sh`
- Create: `launchable/fields.md`
- Create: `README.md`

- [ ] **Step 1: Create the setup script**

Create executable `launchable/setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PORT="${JUPYTER_PORT:-8888}"

cd "${PROJECT_DIR}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import nvmolkit

assert torch.cuda.is_available(), "CUDA-enabled PyTorch is not available"
print("GPU:", torch.cuda.get_device_name(0))
print("nvMolKit:", nvmolkit.__version__)
PY

nohup "${VENV_DIR}/bin/jupyter" lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.root_dir="${PROJECT_DIR}" \
  > "${PROJECT_DIR}/jupyter.log" 2>&1 &

echo "JupyterLab started on port ${PORT}. Use only through the access-controlled Brev Secure Link."
```

The `pkill` target is restricted to the configured Jupyter port inside the task-owned VM. Do not add whole-instance cleanup or unrelated process termination.

- [ ] **Step 2: Verify shell syntax and size**

Run:

```bash
chmod +x launchable/setup.sh
bash -n launchable/setup.sh
LC_ALL=C wc -c < launchable/setup.sh
```

Expected: `bash -n` exits 0 and the byte count is below `16384`.

- [ ] **Step 3: Write the paste-ready Console fields**

Create `launchable/fields.md` with:

```markdown
# Brev Console fields

- Name: `nvMolKit + Nemotron Notebook`
- Description: `A guided GPU notebook for nvMolKit fingerprints, similarity, clustering, conformers, and MMFF94 optimization with bounded hosted Nemotron planning.`
- Source: the user-approved repository at the accepted release commit
- Mode: VM
- Hardware: one NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer
- Disk: 50 GiB
- Setup script: `launchable/setup.sh`
- Secure Link port: `8888`
- Secure Link access: organization-only; do not expose port 8888 as an unrestricted public TCP port
- Launch parameter: `NVIDIA_API_KEY`, required, masked, empty default
- Optional launch parameter: `NEMOTRON_MODEL`, unmasked, default `nvidia/nemotron-3-nano-30b-a3b`
- Optional launch parameter: `JUPYTER_PORT`, unmasked, default `8888`

Launchables are created and published in the Brev web Console. Do not paste an API key into this file, Git, notebook cells, shell history, or chat.
```

- [ ] **Step 4: Write the concise user README**

Create `README.md` with these sections and no additional deployment framework:

```markdown
# nvMolKit + Nemotron Brev notebook

One guided Jupyter notebook that uses hosted Nemotron to select bounded parameters
and nvMolKit on a Brev GPU to compute fingerprints, similarity, Butina clusters,
ETKDGv3 conformers, and MMFF94 geometry optimization.

## What runs where

- Brev provides the GPU VM and Secure Link.
- Nemotron plans parameters and explains the returned summary through NVIDIA's hosted API.
- nvMolKit executes the batched chemistry operations on the Brev GPU.
- RDKit parses molecules and renders the molecule grid.

## Launch

Use the exact fields in `launchable/fields.md`, provide `NVIDIA_API_KEY` as a masked
Launchable parameter, and open port 8888 through the access-controlled Secure Link.
Run `notebooks/nvmolkit_nemotron_demo.ipynb` from top to bottom.

## Verify

Run `pytest -q` for local contract/static checks. On the launched GPU VM, run
`RUN_GPU_TESTS=1 pytest tests/test_gpu_acceptance.py -v`, then execute the notebook
and confirm all three visuals render without saved credentials.

## Boundaries

The outputs are computational fingerprints, similarity relationships, clusters,
and force-field geometries. They do not establish binding, activity, ADMET,
efficacy, safety, synthesizability, or clinical relevance. This demo makes no
performance claim.

## Sources

- https://github.com/NVIDIA-BioNeMo/nvMolKit
- https://nvidia-bionemo.github.io/nvMolKit/
- https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/main/library-skills/nvMolKit/SKILL.md
```

- [ ] **Step 5: Commit the Launchable packet**

```bash
git add launchable/setup.sh launchable/fields.md README.md
git commit -m "feat: add minimal Brev Launchable packet"
```

### Task 6: Add GPU acceptance and perform the completion audit

**Files:**
- Create: `tests/test_gpu_acceptance.py`
- Modify: `README.md` only if the verified command or dependency pin differs

- [ ] **Step 1: Write the opt-in GPU acceptance test**

Create `tests/test_gpu_acceptance.py`:

```python
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 on the task-owned Brev GPU",
)


def test_nvmolkit_gpu_workflow() -> None:
    import torch
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3
    from nvmolkit.clustering import fused_butina
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.similarity import crossTanimotoSimilarity

    assert torch.cuda.is_available()
    smiles = ["CCO", "CCN", "CCCO", "c1ccccc1", "c1ccncc1", "CC(=O)O"] * 32
    mols = [Chem.MolFromSmiles(smiles_value) for smiles_value in smiles]

    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=1024).GetFingerprints(mols)
    similarity = crossTanimotoSimilarity(fingerprints)
    clusters, _cluster_sizes = fused_butina(fingerprints, cutoff=0.5)
    torch.cuda.synchronize()
    assert similarity.torch().shape == (192, 192)
    assert sum(len(members) for members in clusters) == 192

    conformer_mols = [Chem.AddHs(Chem.Mol(mol)) for mol in mols[:4]]
    params = ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 7
    EmbedMolecules(conformer_mols, params, confsPerMolecule=2, maxIterations=-1)
    energies = MMFFOptimizeMoleculesConfs(conformer_mols, maxIters=200)
    assert all(mol.GetNumConformers() == 2 for mol in conformer_mols)
    assert all(len(values) == 2 for values in energies)
```

- [ ] **Step 2: Run all non-GPU tests locally**

Run: `.venv/bin/pytest -q`

Expected: agent and notebook tests pass; `test_nvmolkit_gpu_workflow` is skipped unless `RUN_GPU_TESTS=1`.

- [ ] **Step 3: Run repository hygiene checks**

Run:

```bash
git diff --check
rg -n "nvapi-[A-Za-z0-9_-]+|NVIDIA_API_KEY=" . --glob '!docs/superpowers/**'
test "$(wc -l < data/sample_molecules.csv | tr -d ' ')" -eq 257
bash -n launchable/setup.sh
```

Expected: `git diff --check`, CSV count, and shell syntax succeed; secret scan returns no credential value. Literal documentation references to the variable name are allowed, but no assignment or key-shaped value is allowed.

- [ ] **Step 4: Commit the acceptance test**

```bash
git add tests/test_gpu_acceptance.py README.md
git commit -m "test: add nvMolKit GPU acceptance"
```

- [ ] **Step 5: Prepare Brev deployment without provisioning**

Record the installed CLI boundary before using it:

```bash
/opt/homebrew/bin/brev --version
/opt/homebrew/bin/brev create --help
```

Do not create, start, stop, reset, or delete an instance until the user has provided the exact Brev organization, confirmed unique task ownership, reviewed the GPU type and hourly price, and explicitly approved cost. Create/publish the Launchable definition in the Brev Console using `launchable/fields.md`. If Brev requires a public source, stop and obtain explicit approval before publishing a sanitized public repository.

- [ ] **Step 6: Qualify one clean user-approved deployment**

After the user provides the Launchable or instance identifier and cost approval, verify the exact task-owned instance, then run:

```bash
nvidia-smi
RUN_GPU_TESTS=1 .venv/bin/pytest tests/test_gpu_acceptance.py -v
.venv/bin/pytest -q
```

Execute `notebooks/nvmolkit_nemotron_demo.ipynb` from top to bottom in JupyterLab and confirm:

- the accepted plan source is `nemotron` for the live-agent path;
- a deliberately invalid mocked plan produces `default_after_error`;
- fingerprint, similarity, cluster, conformer, and MMFF94 assertions pass;
- the molecule grid, heatmap, and 3D conformer views render;
- the final explanation retains the explicit claim boundary;
- no API key appears in tracked files, notebook outputs, logs, or acceptance evidence.

If any item fails, preserve diagnostics and report the exact blocker. Do not label the Launchable demo-ready until the complete list passes.
