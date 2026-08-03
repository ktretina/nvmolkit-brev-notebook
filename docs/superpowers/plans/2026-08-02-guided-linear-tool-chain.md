# Guided Linear nvMolKit Tool Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the standalone Brev notebook into a presentation-ready guided chain in which Nemotron calls six bounded nvMolKit-related tools one at a time, each result is visualized immediately, and the final model response synthesizes the actual results.

**Architecture:** Keep the notebook as a transparent, sequential tool runtime and generalize `demo_agent.py` only enough to issue and validate one forced named function call at a time. Vendor the public nvMolKit Agent Toolkit skill so the first tool can ground Nemotron without a runtime network dependency, retain scientific artifacts locally between sections, and pass only JSON-safe summaries and figure descriptions to the text-only model. Preserve the existing Launchable, dependencies, dataset, and GPU acceptance boundary.

**Tech Stack:** Python 3.12, Jupyter/nbformat, OpenAI-compatible NVIDIA hosted inference, Pydantic, nvMolKit 0.5.0, RDKit, PyTorch/CUDA, Matplotlib, Seaborn, pytest.

---

## File map

- `skills/nvmolkit/SKILL.md`: exact vendored public skill text read by Tool 1.
- `skills/nvmolkit/PROVENANCE.md`: immutable source URL, upstream commit, retrieval date, license, byte count, and SHA-256.
- `demo_agent.py`: six strict argument models, forced-call validation, brief interpretation, and final synthesis clients.
- `notebooks/nvmolkit_nemotron_demo.ipynb`: guided narrative, allow-listed scientific functions, local artifacts, figures, and result handoffs.
- `tests/test_skill_snapshot.py`: snapshot/provenance integrity.
- `tests/test_demo_agent.py`: hosted-call schema, validation, auth, and explanation contracts.
- `tests/test_notebook.py`: notebook structure, ordering, implementation boundaries, visuals, comments, and cleared-state checks.
- `tests/test_gpu_acceptance.py`: existing live nvMolKit GPU gate; change only if split stages expose a missing acceptance assertion.
- `README.md`: concise architecture, run instructions, expected presentation flow, and acceptance boundary.

### Task 1: Pin the nvMolKit Agent Toolkit skill

**Files:**
- Create: `skills/nvmolkit/SKILL.md`
- Create: `skills/nvmolkit/PROVENANCE.md`
- Create: `tests/test_skill_snapshot.py`

- [ ] **Step 1: Resolve and record one public upstream revision**

Read the public `NVIDIA-BioNeMo/bionemo-agent-toolkit` default branch, resolve its exact 40-character commit, inspect the repository license, and retrieve only `library-skills/nvMolKit/SKILL.md`. The completed provenance file must contain the following six fields with values obtained from that revision:

~~~markdown
# nvMolKit skill provenance

- Source: a GitHub blob permalink containing the resolved 40-character commit
- Upstream commit: the same resolved 40-character commit
- Retrieved: 2026-08-02
- License: the upstream repository license identifier
- Byte count: the decimal byte length of the vendored file
- SHA-256: the lowercase hexadecimal digest of the vendored file
~~~

- [ ] **Step 2: Write the failing integrity test**

Create `tests/test_skill_snapshot.py` with these executable assertions:

~~~python
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "nvmolkit" / "SKILL.md"
PROVENANCE = ROOT / "skills" / "nvmolkit" / "PROVENANCE.md"


def test_vendored_skill_matches_recorded_provenance():
    payload = SKILL.read_bytes()
    provenance = PROVENANCE.read_text(encoding="utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    assert f"- Byte count: `{len(payload)}`" in provenance
    assert f"- SHA-256: `{digest}`" in provenance
    assert re.search(r"- Upstream commit: `[0-9a-f]{40}`", provenance)
    assert "/blob/" in provenance and "/library-skills/nvMolKit/SKILL.md" in provenance
    assert "- License:" in provenance


def test_skill_names_the_demo_entry_points_and_gpu_boundary():
    text = SKILL.read_text(encoding="utf-8")
    for name in (
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    ):
        assert name in text
    assert "GPU" in text
~~~

- [ ] **Step 3: Run the test and verify RED**

Run:

~~~bash
python -m pytest tests/test_skill_snapshot.py -q
~~~

Expected: failure because the two vendored files do not exist yet.

- [ ] **Step 4: Vendor the exact bytes and complete provenance**

Write the fetched bytes unchanged to `skills/nvmolkit/SKILL.md`. Calculate byte count and SHA-256 from that file, enter the resolved values in `PROVENANCE.md`, and do not copy any other upstream content.

- [ ] **Step 5: Run the integrity test and commit**

Run:

~~~bash
python -m pytest tests/test_skill_snapshot.py -q
git diff --check
git add skills/nvmolkit/SKILL.md skills/nvmolkit/PROVENANCE.md tests/test_skill_snapshot.py
git commit -m "docs: pin nvMolKit agent skill"
~~~

Expected: both tests pass and only the three named files enter the commit.

### Task 2: Generalize one forced, validated Nemotron step

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`

- [ ] **Step 1: Replace the composite-plan tests with six strict schemas**

Write tests that import these exact public contracts:

~~~python
TOOL_ARGUMENT_MODELS = {
    "read_nvmolkit_skill": ReadSkillArgs,
    "prepare_molecular_sample": PrepareSampleArgs,
    "compute_morgan_fingerprints": FingerprintArgs,
    "compute_tanimoto_similarity": SimilarityArgs,
    "cluster_with_fused_butina": ClusterArgs,
    "generate_and_optimize_conformers": ConformerArgs,
}
~~~

Require the models to validate these values and reject extra, missing, non-strict, or out-of-range values:

~~~python
VALID_ARGUMENTS = {
    "read_nvmolkit_skill": {},
    "prepare_molecular_sample": {"preview_count": 24},
    "compute_morgan_fingerprints": {"fingerprint_radius": 2, "fingerprint_size": 1024},
    "compute_tanimoto_similarity": {},
    "cluster_with_fused_butina": {"cluster_cutoff": 0.5},
    "generate_and_optimize_conformers": {
        "representative_count": 4,
        "conformers_per_representative": 4,
    },
}
~~~

The range assertions must cover `cluster_cutoff` below 0.40 and above 0.60, representatives outside 3 through 6, and conformers outside 3 through 8.

- [ ] **Step 2: Write failing forced-call and no-execution tests**

Use the existing fake OpenAI client pattern and a recording executor:

~~~python
executed = []

def executor(arguments):
    executed.append(arguments)
    return {"ok": True}

with pytest.raises(ToolCallError):
    request_and_execute_step(
        "nvapi-",
        tool_name="prepare_molecular_sample",
        task_prompt="Prepare the fixed sample.",
        context={},
        executor=executor,
        client=fake_client(FakeCompletions(tool_calls=[])),
    )
assert executed == []
~~~

Parameterize missing, multiple, wrong-name, wrong-type, empty-ID, malformed JSON, missing-field, extra-field, and invalid-range responses. Require exactly one executor call only for a valid response. Assert `tool_choice` forces the named function, the request exposes only that schema, `chat_template_kwargs.enable_thinking` is `False`, and neither the decision nor exceptions contain the API key.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

~~~bash
python -m pytest tests/test_demo_agent.py -q
~~~

Expected: failures because the existing implementation exposes only `analyze_molecule_library` and permits deterministic fallback execution after malformed calls.

- [ ] **Step 4: Implement the minimal generic contract**

Define strict Pydantic models with these signatures:

~~~python
class ReadSkillArgs(StrictArgs):
    pass

class PrepareSampleArgs(StrictArgs):
    preview_count: Literal[24]

class FingerprintArgs(StrictArgs):
    fingerprint_radius: Literal[2, 3]
    fingerprint_size: Literal[1024, 2048]

class SimilarityArgs(StrictArgs):
    pass

class ClusterArgs(StrictArgs):
    cluster_cutoff: float = Field(ge=0.40, le=0.60)

class ConformerArgs(StrictArgs):
    representative_count: int = Field(ge=3, le=6)
    conformers_per_representative: int = Field(ge=3, le=8)
~~~

Implement `request_tool_call(...)` and `request_and_execute_step(...)` so validation completes before the passed executor can run. Return a frozen `ToolDecision` containing only validated arguments, `source="nemotron"`, expected tool name, non-empty call ID, and raw argument JSON. Remove `DEFAULT_PLAN`, composite fallback execution, and `analyze_molecule_library`; API, malformed-response, and validation failures must raise secret-safe `ToolCallError` or the existing hosted-key guidance.

- [ ] **Step 5: Add brief and final explanation tests**

Require:

~~~python
brief = request_brief_interpretation(
    "nvapi-", decision, tool_result, figure_context, client=fake_client(completions)
)
final = request_final_synthesis(
    "nvapi-", analysis_summary, client=fake_client(final_completions)
)
~~~

Assert the brief system prompt explicitly requests 2 to 4 sentences, states that the model receives a text figure description rather than pixels, and includes the validated assistant tool call followed by its `role="tool"` result. Assert the final prompt requests 450 to 650 words, includes all six section keys in serialized JSON, names the six required themes, and forbids binding, activity, ADMET, efficacy, safety, synthesizability, clinical, and experimentally validated conformation claims.

- [ ] **Step 6: Implement explanation requests and verify GREEN**

Implement `request_brief_interpretation` as one non-streaming continuation of the validated tool exchange and `request_final_synthesis` as one bounded text request over `json.dumps(analysis_summary)`. Translate 401/403 to `AUTH_GUIDANCE`; allow other explanation errors to propagate so the notebook can display its non-fatal notice.

Run:

~~~bash
python -m pytest tests/test_demo_agent.py -q
git diff --check
git add demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add guided Nemotron tool steps"
~~~

Expected: focused tests pass with no deterministic scientific execution after an invalid hosted response.

### Task 3: Build the skill, sample, fingerprint, similarity, and cluster narrative

**Files:**
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_notebook.py`

- [ ] **Step 1: Write failing structural and ordering tests**

Require this exact heading sequence:

~~~python
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
~~~

Parse notebook cells in order and assert that each of the first five tools has a task markdown cell, its named function definition, `request_and_execute_step`, a static result display, and `request_brief_interpretation` before the next subsection. Assert the introduction contains the four approved paragraphs verbatim from the design specification.

- [ ] **Step 2: Write failing scientific-boundary tests**

Use AST and string assertions to require these exact functions and entry points:

~~~text
read_nvmolkit_skill
prepare_molecular_sample
compute_morgan_fingerprints
compute_tanimoto_similarity
cluster_with_fused_butina
~~~

Require the fixed `DATA_PATH`, `preview_count == 24`, 256 raw-row shape validation, visible invalid count and excluded IDs, `Draw.MolsToGridImage(..., molsPerRow=6)`, nvMolKit fingerprint/similarity/clustering calls, `torch.cuda.synchronize()`, an active-bit histogram, a 0-to-1 similarity heatmap, diagonal-excluded summary statistics, and a top-15 cluster-size bar chart with singleton count.

- [ ] **Step 3: Run notebook tests and verify RED**

Run:

~~~bash
python -m pytest tests/test_notebook.py -q
~~~

Expected: failures because the notebook still has the old composite tool and old heading order.

- [ ] **Step 4: Rebuild Sections 1 through 4 as a guided chain**

Generate a valid nbformat 4 notebook whose committed outputs are empty and execution counts are null. Keep preflight and secret handling concise. Define each bounded scientific function in its visible section and retain large local artifacts in explicit variables such as `sample_artifact`, `fingerprint_artifact`, `similarity_artifact`, and `cluster_artifact`; pass only JSON-safe `summary` plus `figure_context` to Nemotron.

The five request prompts must recommend these deterministic presentation values where applicable:

~~~python
{"preview_count": 24}
{"fingerprint_radius": 2, "fingerprint_size": 1024}
{}
{"cluster_cutoff": 0.50}
~~~

Every executor remains a directly named local function; do not use `eval`, dynamic imports, model paths, arbitrary names, or an automatic loop.

- [ ] **Step 5: Add explanatory comments and immediate figures**

Place concise `#` comments next to the relevant code for fixed-path and invalid-SMILES handling, validation before execution, GPU-resident results and synchronization, diagonal exclusion, and Butina cutoff/singleton interpretation. Display the 24-molecule grid, active-bit histogram, unordered heatmap, and top-15 cluster chart immediately in their owning subsections, followed by a 2-to-4-sentence Nemotron interpretation call with a non-fatal `Interpretation unavailable` exception path.

- [ ] **Step 6: Verify Sections 1 through 4 and commit**

Run:

~~~bash
python -m pytest tests/test_notebook.py -q
python -m pytest tests/test_demo_agent.py tests/test_skill_snapshot.py -q
git diff --check
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py
git commit -m "feat: add guided molecular similarity chain"
~~~

Expected: structural tests pass; the notebook remains cleared, parseable, and limited to fixed local executors.

### Task 4: Add conformer visuals, detailed synthesis, and presentation documentation

**Files:**
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`
- Modify only if an assertion is missing: `tests/test_gpu_acceptance.py`

- [ ] **Step 1: Write failing conformer and synthesis tests**

Require `generate_and_optimize_conformers` to consume the validated cluster artifact and accept only `ConformerArgs`. Through AST and source assertions require distinct-cluster deterministic representative selection, lower-heavy-atom preference, `ETKDGv3` with fixed seed, `EmbedMolecules`, zero/partial generation accounting, `MMFFOptimizeMoleculesConfs(..., maxIters=500, output=CoordinateOutput.DEVICE)`, synchronization, optimized device coordinates copied into RDKit conformers, and within-molecule-only energy ranking.

Require two independent static plotting functions:

~~~python
plot_conformer_energies(conformer_summary)
plot_lowest_energy_conformers(representative_molecules, conformer_summary)
~~~

Require converged/unconverged attempted conformers in the energy plot, molecular bonds in every Matplotlib 3D panel, and the lowest-energy converged conformer for each successful representative. Assert these functions execute before the optional guarded `py3Dmol.view` block.

- [ ] **Step 2: Write failing final-summary tests**

Require `analysis_summary` to contain exactly these stage keys:

~~~python
{
    "skill",
    "sample",
    "fingerprints",
    "similarity",
    "clusters",
    "conformers_mmff94",
}
~~~

Assert `json.dumps(analysis_summary)` is present as a serialization gate, no RDKit molecule, tensor, matrix, coordinate array, key, or display object enters it, a compact table replaces the raw dictionary, and `request_final_synthesis` receives it. Require the scientific-boundary notice before and after the final response.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

~~~bash
python -m pytest tests/test_notebook.py -q
~~~

Expected: failures because the static conformer plots and detailed result synthesis are not yet implemented.

- [ ] **Step 4: Implement the conformer/MMFF94 stage**

Recommend four representatives and four conformers in the Nemotron prompt. Preserve partial embedding and MMFF failures in the JSON-safe result with requested, generated, attempted, converged, and unconverged counts; representative IDs; per-conformer energy and convergence records; selected conformer IDs; and figure descriptions. Plot every attempted result, mark missing or non-finite energy explicitly, and compare energies only among conformers of the same molecule.

Render the lowest-energy converged optimized structure per representative with Matplotlib 3D atom coordinates and bond segments. Only after both static figures, attempt py3Dmol inside a guarded block; a JavaScript/view failure prints a short optional-view notice and does not interrupt the workflow.

- [ ] **Step 5: Implement the detailed Section 6 synthesis**

Build the six-key JSON-safe summary, verify it with `json.dumps`, and render a small presentation table of the main measurements. Call `request_final_synthesis` with the real summaries and request 450 to 650 words covering dataset scope, representation, pairwise similarity, clustering/diversity, conformational sampling/MMFF94, and limitations/next analyses. Keep the final text quantitative and prohibit experimental or biological overclaims.

- [ ] **Step 6: Update README and run all local acceptance checks**

Update README to describe six forced calls, immediate static visuals, short per-stage interpretations, the detailed final synthesis, the vendored skill snapshot, and the text-only figure-context boundary. Keep the existing Brev launch and hosted-key instructions.

Run:

~~~bash
python -m pytest -q
git diff --check
git diff -- requirements.txt launchable/setup.sh launchable/fields.md data/sample_molecules.csv
~~~

Expected: the full suite passes with the GPU test skipped locally; the final diff command is empty.

- [ ] **Step 7: Commit the presentation workflow**

Run:

~~~bash
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py README.md
git add tests/test_gpu_acceptance.py
git commit -m "feat: complete guided nvMolKit presentation"
~~~

If `tests/test_gpu_acceptance.py` did not need a change, omit its `git add`. Never stage `.DS_Store`.

### Task 5: Final review, merge, publish, and live Brev acceptance

**Files:**
- Verify all tracked project files
- Preserve live user notebook as: `/home/ubuntu/nvmolkit_nemotron_demo.pre-guided-chain-20260802.ipynb`

- [ ] **Step 1: Run final local review and acceptance**

Dispatch a fresh final reviewer after all per-task specification and quality reviews are resolved. Then run one low-memory verification command at a time:

~~~bash
python -m pytest -q
git diff --check
git status --short --branch
~~~

Expected: all deterministic tests pass, the GPU test is skipped locally, and `.DS_Store` is the only unrelated untracked file.

- [ ] **Step 2: Merge the reviewed branch into standalone `main`**

Use the finishing-a-development-branch workflow, merge without touching the surrounding workbench, rerun the full suite on `main`, and confirm the design and implementation plan commits are included.

- [ ] **Step 3: Publish the standalone public repository**

Push only standalone `main` to `https://github.com/ktretina/nvmolkit-brev-notebook`. Verify the public remote branch resolves to the local `main` commit and contains no surrounding-workbench paths or secrets.

- [ ] **Step 4: Preserve the live notebook and update the Brev checkout**

Using the exact task-owned instance `nvmolkit---nemotron-notebook-ec6247` in organization `agents-in-ls`, first verify that the backup path does not exist. Copy the current live notebook to the explicit backup path, restore the tracked notebook in the remote checkout if necessary, fast-forward the checkout to published `main`, and do not stop or delete the billable instance.

- [ ] **Step 5: Run deterministic and GPU live acceptance**

On the Brev instance run:

~~~bash
/home/ubuntu/.venv/bin/python3 -m pytest -q
RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python3 -m pytest tests/test_gpu_acceptance.py -v
~~~

Expected: the full deterministic suite passes and the existing L4 GPU acceptance test passes.

- [ ] **Step 6: Report the remaining hosted acceptance boundary**

Tell the user to reload JupyterLab, restart the notebook kernel, run all cells, and provide a hosted Developer API key only through the hidden prompt. Expected presentation evidence is six valid forced Nemotron calls, the molecule grid, active-bit histogram, similarity heatmap, cluster-size chart, two static MMFF94/conformer figures, six brief interpretations, and one 450-to-650-word synthesis. Do not claim this hosted/rendered acceptance until the user completes that private-key run.
