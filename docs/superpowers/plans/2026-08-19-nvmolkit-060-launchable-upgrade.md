# nvMolKit 0.6.0 Notebook Launchable Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the four-notebook `nvMolKit + Nemotron Notebook` release to exact nvMolKit 0.6.0, integrate the valid intent of PRs #1 and #2, and qualify only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

**Architecture:** One self-contained adapter in `notebooks/nvmolkit_compat.py` normalizes the nvMolKit 0.5 member-list result and the nvMolKit 0.6 cluster-ID result into labels, member lists, and centroids. Module 1, Module 3 controller-rendered analysis, and the companion workflow all use that adapter contract. The release keeps exact dependency pins, neutral runtime-ratio language, accurate fused-memory boundaries, and the existing bounded agent and Brev security contracts.

**Tech Stack:** Python 3.12, NumPy, RDKit, nvMolKit 0.6.0, PyTorch/CUDA, Jupyter notebooks, pytest, Ruff, Bash, GitHub CLI, Brev CLI and web Console.

---

## File map

- Create `notebooks/nvmolkit_compat.py`: one self-contained result normalizer.
- Create `tests/test_nvmolkit_compat.py`: GPU-free adapter contract tests.
- Modify `requirements.txt`: exact nvMolKit 0.6.0 pin.
- Modify `notebooks/01_direct_nvmolkit_reframe.ipynb`: adapter use, truthful performance and memory text.
- Modify `tests/test_workshop_notebook_inventory.py`: Module 1 source and clean-kernel contracts.
- Modify `tests/test_gpu_acceptance.py`: exact runtime version and live cluster invariants.
- Modify `chemistry_workflow.py`: companion workflow adapter use.
- Modify `command_receipts.py`: exact nvMolKit 0.6 clustering receipt.
- Modify `tests/test_chemistry_workflow.py`: old/new result-shape and atomic rejection tests.
- Modify `tests/test_command_receipts.py`: exact receipt source.
- Modify `notebooks/workshop_llm_agent.py`: embed and use the shared normalizer in controller-owned Module 3 analysis.
- Modify `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`: helper-version lock only.
- Modify `notebooks/03_full_agent_reframe_panel_design.ipynb`: helper-version lock only.
- Modify `tests/test_workshop_llm_agent.py`: rendered-source and cluster-partition tests.
- Modify `tests/test_workshop_notebook_execution.py`: clean reference execution and lock checks.
- Modify `README.md` and `launchable/fields.md` only if an existing statement names nvMolKit 0.5.0 or the removed memory estimate; do not add unrelated copy.
- Create `docs/superpowers/reports/2026-08-19-nvmolkit-060-qualification.md`: local, GPU, hosted, browser, PR, and Launchable evidence ledger.

### Task 1: Add the shared fused-Butina result adapter

**Files:**
- Create: `notebooks/nvmolkit_compat.py`
- Create: `tests/test_nvmolkit_compat.py`

- [ ] **Step 1: Write GPU-free failing tests for both public result shapes**

Add tests with a fake asynchronous result:

```python
class FakeAsyncResult:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return np.asarray(self.value)


def test_normalizes_v05_member_lists_and_centroids():
    labels, clusters, centroids = normalize_fused_butina_result(
        ([(0, 2), (1,)], [2, 3], [0, 1]), molecule_count=3
    )
    assert labels.tolist() == [0, 1, 0]
    assert clusters == ((0, 2), (1,))
    assert centroids.tolist() == [0, 1]


def test_normalizes_v06_async_cluster_ids_and_centroids():
    labels, clusters, centroids = normalize_fused_butina_result(
        (FakeAsyncResult([0, 1, 0]), FakeAsyncResult([0, 1])),
        molecule_count=3,
    )
    assert labels.tolist() == [0, 1, 0]
    assert clusters == ((0, 2), (1,))
    assert centroids.tolist() == [0, 1]
```

Add parametrized failures for a wrong tuple length, noninteger labels, wrong
label length, negative or noncontiguous labels, duplicate/missing/out-of-range
v0.5 members, wrong centroid count, and a centroid outside its cluster. Assert a
generic `ValueError` without including raw provider or molecule data.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider tests/test_nvmolkit_compat.py
```

Expected: collection fails because `nvmolkit_compat` does not exist.

- [ ] **Step 3: Implement the self-contained normalizer**

Implement this public signature:

```python
def normalize_fused_butina_result(raw_result, *, molecule_count):
    """Return validated `(labels, clusters, centroids)` host data."""
```

The complete function must keep its array conversion helper inside the public
function so `inspect.getsource(normalize_fused_butina_result)` is safe to embed
in Module 3. It may use only the global name `np`. For a two-item result, read
cluster IDs and centroids; for a three-item result, build labels from exact
member lists and ignore only the historical cumulative-size item. Require
labels `0..cluster_count-1`, an exact molecule partition, and centroid
membership before returning:

```python
return labels.astype(int, copy=False), tuple(clusters), centroids.astype(int, copy=False)
```

- [ ] **Step 4: Run focused GREEN and static checks**

Run the Step 2 command, then:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  notebooks/nvmolkit_compat.py tests/test_nvmolkit_compat.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format --check \
  notebooks/nvmolkit_compat.py tests/test_nvmolkit_compat.py
python3 -m py_compile notebooks/nvmolkit_compat.py tests/test_nvmolkit_compat.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the adapter**

```bash
git add notebooks/nvmolkit_compat.py tests/test_nvmolkit_compat.py
git commit -m "feat: normalize nvMolKit clustering results"
```

Request a fresh specification review, then a fresh quality review. Fix every
Critical or Important finding and repeat the corresponding review before Task
2.

### Task 2: Upgrade Module 1 and the exact runtime contract

**Files:**
- Modify: `requirements.txt`
- Modify: `notebooks/01_direct_nvmolkit_reframe.ipynb`
- Modify: `tests/test_workshop_notebook_inventory.py`
- Modify: `tests/test_gpu_acceptance.py`

- [ ] **Step 1: Write failing dependency, adapter, wording, and memory tests**

Require:

```python
assert (REPO_ROOT / "requirements.txt").read_text().splitlines()[2] == "nvmolkit==0.6.0"
assert 'nvmolkit.__version__ == "0.6.0"' in gpu_acceptance_source
assert "from nvmolkit_compat import normalize_fused_butina_result" in module1_source
assert "Observed runtime ratio (RDKit CPU / nvMolKit GPU; >1 favors nvMolKit)" in module1_source
assert "speedup" not in module1_source.lower()
assert "RDKit CPU: condensed distance + Butina" in module1_source
assert "nvMolKit GPU: fused fingerprint clustering" in module1_source
assert "10k float32 square matrix estimate" not in module1_source
assert "require_memory_within_limit" not in module1_source
assert "square_matrix_bytes" not in module1_source
assert "avoids an N x N pairwise matrix" in advanced_markdown
assert "can still exhaust GPU memory" in advanced_markdown
```

Keep the existing assertion that `require_bounded_condensed_distances` appears
immediately before RDKit's `np.empty` condensed-vector allocation. Extend the
clean-kernel Module 1 test to require a neutral ratio line when GPU is active,
without requiring a value above one.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider \
  tests/test_workshop_notebook_inventory.py tests/test_gpu_acceptance.py \
  -k 'module1 or nvmolkit_version or gpu_acceptance'
```

Expected: failures show the 0.5.0 pin, local legacy converter, old matrix guard,
and old ratio labels.

- [ ] **Step 3: Make the minimum Module 1 and runtime changes**

Change only `nvmolkit==0.5.0` to `nvmolkit==0.6.0`. In Module 1:

- import `normalize_fused_butina_result` from `nvmolkit_compat`;
- replace the local legacy converter body with a call using
  `return_centroids=True` and `molecule_count=len(fingerprint_matrix)`;
- remove `require_memory_within_limit` and `square_matrix_bytes` imports;
- remove the advanced square estimate and its 512 MiB call;
- retain `ADVANCED_LARGE_RUN = False`, exact 10,000-row bound, GPU check, source
  receipt, and CUDA synchronization;
- use the exact neutral ratio and backend strings from Step 1;
- preserve every cell ID, notebook metadata, and empty output/execution state.

Update the GPU version assertion to exact `0.6.0`. Add live assertions that
every molecule has one cluster, each centroid belongs to its cluster, and
cluster labels are contiguous without assuming a fixed label permutation.

- [ ] **Step 4: Run focused and adjacent GREEN**

Run the Step 2 command, all of `tests/test_workshop_notebook_inventory.py`,
`tests/test_nvmolkit_compat.py`, and the clean Module 1 execution test. Compile
every Module 1 code cell with `compile(cell.source, cell.id, "exec")` and assert
no saved outputs, execution counts, or widget state.

- [ ] **Step 5: Commit Module 1 and runtime changes**

```bash
git add requirements.txt notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_notebook_inventory.py tests/test_gpu_acceptance.py
git commit -m "fix: upgrade the direct nvMolKit lesson" \
  -m "Co-authored-by: Kevin Boyd <kboyd@nvidia.com>"
```

Complete fresh specification and quality reviews before Task 3.

### Task 3: Upgrade the companion workflow and receipts

**Files:**
- Modify: `chemistry_workflow.py`
- Modify: `command_receipts.py`
- Modify: `tests/test_chemistry_workflow.py`
- Modify: `tests/test_command_receipts.py`

- [ ] **Step 1: Write failing old/new-shape and receipt tests**

Extend the companion clustering fixture to run twice: once with
`([(0, 2), (1,)], [2, 3], [0, 1])`, and once with fake asynchronous
`([0, 1, 0], [0, 1])`. Both must produce `[[0, 2], [1]]`, the same summary,
and exact assignment. Add failures showing malformed labels or centroids leave
the workflow state unchanged.

Require the receipt to show:

```text
cluster_ids, centroids = fused_butina(fingerprints.torch(), cutoff=<value>, return_centroids=True)
```

and a repository-owned normalization step, without the old `[0]` expression.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider \
  tests/test_chemistry_workflow.py tests/test_command_receipts.py \
  -k 'cluster or fused_butina or receipt'
```

Expected: the v0.6 fake cannot satisfy the legacy `[0]` member-list path and
the receipt still records the old call.

- [ ] **Step 3: Route the companion through the adapter**

Import:

```python
from notebooks.nvmolkit_compat import normalize_fused_butina_result
```

Call fused Butina with `return_centroids=True`, normalize with the known
molecule count, and assign the returned `clusters` only after all adapter and
workflow validations pass. Preserve existing summary keys and stage labels.
Update the receipt to state the exact 0.6 call and normalization without
claiming host or GPU timing.

- [ ] **Step 4: Run focused and adjacent GREEN**

Run all of `tests/test_chemistry_workflow.py`,
`tests/test_command_receipts.py`, `tests/test_demo_agent.py`, and
`tests/test_notebook.py` serially with `MPLBACKEND=Agg`. Run Ruff check/format,
`py_compile`, and `git diff --check` on the four changed files.

- [ ] **Step 5: Commit the companion upgrade**

```bash
git add chemistry_workflow.py command_receipts.py \
  tests/test_chemistry_workflow.py tests/test_command_receipts.py
git commit -m "fix: adapt the companion clustering workflow"
```

Complete fresh specification and quality reviews before Task 4.

### Task 4: Upgrade Module 3 controller-owned analysis

**Files:**
- Modify: `notebooks/workshop_llm_agent.py`
- Modify: `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
- Modify: `notebooks/03_full_agent_reframe_panel_design.ipynb`
- Modify: `tests/test_workshop_llm_agent.py`
- Modify: `tests/test_workshop_notebook_execution.py`

- [ ] **Step 1: Write failing rendered-source and reference tests**

Require `_render_panel_analysis(1, 24)` to contain the exact dedented source of
`normalize_fused_butina_result`, call v0.6 fused Butina with
`return_centroids=True`, and use normalized labels, clusters, and centroids.
Require the source validator to keep accepting only controller-owned code.

Add GPU-free execution tests that replace the generated fused call with the
old three-item and new two-item fake results. Require the same label-invariant
partition and selected-panel invariants. Require a malformed result to fail
before `panel.csv`, `report.json`, or a passing trace is retained.

Require the shared helper version and both notebook locks to equal
`2026.08.19.1`.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py \
  -k 'panel or module3 or workshop_agent_version'
```

Expected: failures show old triple-unpack source and old helper locks.

- [ ] **Step 3: Embed the exact adapter source without weakening isolation**

At controller render time:

```python
normalizer_source = textwrap.dedent(
    inspect.getsource(normalize_fused_butina_result)
)
```

Insert `normalizer_source` into the fixed generated program after imports. Do
not alter `[executable, "-I", "-"]`, child environment allowlists, source AST
validation, exact source artifacts, timeout, or secret redaction. Replace the
old triple-unpack loop with normalized labels, clusters, and centroids. Keep
the CPU reference branch and all independent artifact validation.

Bump `WORKSHOP_AGENT_VERSION` and only the corresponding Module 2 and Module 3
locks to `2026.08.19.1`. Preserve notebook cell IDs, metadata, and clean state.

- [ ] **Step 4: Run focused and adjacent GREEN**

Run the Step 2 command, then all of:

```bash
tests/test_workshop_llm_agent.py
tests/test_workshop_notebook_execution.py
tests/test_workshop_notebook_inventory.py
```

Run Module 2 and Module 3 clean reference kernels with hosted-client and network
blockers. Confirm the Module 3 replay tests still add zero hosted calls and zero
analysis executions. Run Ruff, notebook compilation, schema, clean-state, and
diff checks.

- [ ] **Step 5: Commit the Module 3 upgrade**

```bash
git add notebooks/workshop_llm_agent.py \
  notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
git commit -m "fix: adapt the bounded panel lesson to nvMolKit 0.6"
```

Complete fresh specification and quality reviews before Task 5.

### Task 5: Run whole-release verification and create the evidence ledger

**Files:**
- Create: `docs/superpowers/reports/2026-08-19-nvmolkit-060-qualification.md`
- Modify: `README.md` and `launchable/fields.md` only if the exact-version or
  removed-memory statements require correction.

- [ ] **Step 1: Run the complete serial local suite**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/private/tmp/nvmolkit-v06-final-mpl \
  IPYTHONDIR=/private/tmp/nvmolkit-v06-final-ipython \
  JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider
```

Expected: zero failures; the GPU test may skip only because the Mac has no L4.

- [ ] **Step 2: Run release static and security gates**

Run changed-file Ruff check/format, compile all tracked Python plus every code
cell, `bash -n launchable/setup.sh`, notebook schema/clean-state checks,
`git diff --check`, and Gitleaks across the design base through `HEAD`. Require
a clean worktree after committing the report.

- [ ] **Step 3: Write the local evidence ledger**

Record exact commands, commit, interpreter, pass/skip counts, changed files,
and unresolved live gates under distinct headings:

```markdown
## Local deterministic evidence
## L4 GPU evidence
## Hosted-model evidence
## Browser evidence
## GitHub PR closeout
## Brev Launchable definition
## Residual risks and rollback
```

Use `NOT RUN` rather than an inferred result. Do not place keys, private URLs,
transcripts, or raw model output in the ledger.

- [ ] **Step 4: Commit the verified release record**

```bash
git add docs/superpowers/reports/2026-08-19-nvmolkit-060-qualification.md
git commit -m "test: record nvMolKit 0.6 qualification"
```

Dispatch one final specification reviewer and one final quality reviewer for
the complete range from `25781fdb` through `HEAD`. Fix and re-review all
Critical or Important findings.

### Task 6: Publish, resolve both PRs, and update the target Launchable

**Files:**
- Modify: `docs/superpowers/reports/2026-08-19-nvmolkit-060-qualification.md`
  only for new external evidence.

- [ ] **Step 1: Push the independently accepted source**

Verify `origin/main` still equals the recorded base or perform a read-only
fetch and re-review any divergence. Push the accepted branch to `main` without
force. Record the public commit URL and verify GitHub's `main` resolves to the
same SHA.

- [ ] **Step 2: Resolve PR #1 and PR #2**

Comment on PR #1 that the final commit integrates the v0.6 result conversion
across Module 1, Module 3, and the companion workflow, adds shared validation,
pins the exact runtime, and credits Kevin Boyd. Comment on PR #2 that the final
commit removes the false square-matrix guard while retaining real memory
boundaries, hardware provenance, and neutral ratios. Link the final commit and
test evidence, then close both PRs as superseded.

- [ ] **Step 3: Inspect the stored Launchable through supported Brev surfaces**

First read `brev --version`, `brev create --help`, and the exact dry-run help.
Then run:

```bash
brev create --launchable env-3HJtJW3qHg4Dw1I3xt75BfpBmZW \
  --dry-run --no-check-latest
```

Do not change the selected organization, refresh credentials, or use private
Console endpoints. Compare source, fields, setup body, model, key parameter,
port, and access policy with `launchable/fields.md`.

- [ ] **Step 4: Apply any required supported Console definition edit**

If the dry run proves the saved definition already reads current `main` and
the setup body is exact, record `NO DEFINITION EDIT REQUIRED`. Otherwise use
the Brev web Console to update only the repository source/commit or saved setup
body required by the diff. Keep every other field unchanged. If this Console
surface is not callable, provide the user one exact change and stop rather than
using an undocumented API.

- [ ] **Step 5: Create one fresh L4 environment and run live gates**

Pin exact organization, Launchable ID, instance ID, provider, region, L4 SKU,
and source commit before running commands. Execute the live qualification order
from the design. Store only safe counts, hashes, modes, exit status, and concise
scientific receipts. Do not print the protected API key or raw hosted output.

- [ ] **Step 6: Finish browser acceptance and handoff**

Use the organization-only port-8888 Secure Link. Confirm all four notebook
files are visible, Module 3 widgets render and accept one human approval, and
the required tables, receipt, and gallery render in the browser. Report the
exact notebook URL only if it contains no token, code, or credential query.
Keep or stop the environment according to the user's explicit handoff choice;
report its lifecycle and cost state.

- [ ] **Step 7: Amend the evidence ledger and commit**

Replace only the corresponding `NOT RUN` values with fresh external evidence.
Run the full local verification gate again if the ledger or any source changed.
Commit the ledger, push without force, and verify the final public SHA.
