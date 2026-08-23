# Four-Notebook nvMolKit Launchable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the three supplied workshop modules onto the current secure nvMolKit source, make the resulting four-notebook Launchable executable, and validate it locally and on one fresh L4.

**Architecture:** Current `main` remains the source of truth. Import only reviewed new notebook assets, give the three new modules deterministic acceptance modes, and preserve the existing interactive demo. Publish one exact revision, then update and validate only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

**Tech Stack:** Python 3.12, Jupyter/nbformat/nbclient, RDKit, nvMolKit 0.5.0, CUDA PyTorch 2.7.1, OpenAI-compatible NVIDIA hosted inference, pytest, Brev VM/Jupyter.

---

## File map

- `notebooks/01_direct_nvmolkit_reframe.ipynb`: direct deterministic nvMolKit lesson.
- `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`: bounded hosted-policy lesson with reference mode.
- `notebooks/03_full_agent_reframe_panel_design.ipynb`: bounded panel-design lesson with interactive and reference modes.
- `notebooks/nvmolkit_nemotron_demo.ipynb`: existing compact interactive demo; retain current bytes.
- `notebooks/workshop_common.py`: explicit snapshot/live data loading and bounded workload helpers.
- `notebooks/workshop_llm_agent.py`: only the bounded Module 2 policy and Module 3 plan/audit paths.
- `notebooks/module3_interactive_workflow.py`: widget presentation and completion callback.
- `notebooks/data/reframe_teaching_snapshot.csv`: fixed 96-compound acceptance dataset.
- `tests/test_workshop_notebook_inventory.py`: exact release inventory and notebook hygiene.
- `tests/test_workshop_common.py`: deterministic data and workload bounds.
- `tests/test_workshop_llm_agent.py`: bounded hosted contracts, renderers, artifacts, and secret safety.
- `tests/test_workshop_notebook_execution.py`: clean-kernel execution of the three new reference modes.
- `tests/test_demo_agent.py`: isolate two key-contract tests from unavailable Mac CUDA.
- `README.md`, `launchable/fields.md`: exact four-notebook attendee inventory; no setup-policy changes.

### Task 1: Import only the reviewed workshop assets

**Files:**
- Create: `tests/test_workshop_notebook_inventory.py`
- Create: the three new notebook files and their required helper/data files listed above
- Preserve: `notebooks/nvmolkit_nemotron_demo.ipynb`

- [ ] **Step 1: Write the failing inventory test**

```python
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXPECTED = {
    "01_direct_nvmolkit_reframe.ipynb",
    "02_agent_assisted_reframe_neighborhoods.ipynb",
    "03_full_agent_reframe_panel_design.ipynb",
    "nvmolkit_nemotron_demo.ipynb",
}


def test_release_has_exact_attendee_notebook_inventory():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == EXPECTED


def test_attendee_notebooks_are_clean_python_312_documents():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata.kernelspec.name == "python3"
        assert notebook.metadata.language_info.version.startswith("3.12")
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        for cell in notebook.cells:
            assert cell.get("execution_count") is None
            assert not cell.get("outputs", [])
            assert not cell.get("attachments", {})


def test_release_excludes_generated_notebook_files():
    forbidden = {".DS_Store", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
    assert not [path for path in ROOT.rglob("*") if path.name in forbidden]
```

- [ ] **Step 2: Run the focused test and witness RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 -m pytest -q -p no:cacheprovider tests/test_workshop_notebook_inventory.py
```

Expected: inventory failure because only the existing demo is present.

- [ ] **Step 3: Import the exact reviewed files**

Copy only the approved paths from `/Users/ktretina/Downloads/nvmolkit-brev-notebook 3`. Do not copy existing shared source, setup files, documentation, checkpoints, caches, or generated files. Confirm the existing demo SHA-256 is unchanged before and after the import.

- [ ] **Step 4: Run the focused test and inspect the staged tree**

Expected: schema/inventory tests pass. `git status --short` lists only the approved files.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_direct_nvmolkit_reframe.ipynb \
  notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  notebooks/workshop_common.py notebooks/workshop_llm_agent.py \
  notebooks/module3_interactive_workflow.py \
  notebooks/data/reframe_teaching_snapshot.csv \
  tests/test_workshop_notebook_inventory.py
git commit -m "feat: add reviewed nvMolKit workshop modules"
```

### Task 2: Make Module 1 deterministic and bounded

**Files:**
- Modify: `notebooks/workshop_common.py`
- Modify: `notebooks/01_direct_nvmolkit_reframe.ipynb`
- Create: `tests/test_workshop_common.py`
- Modify: `tests/test_workshop_notebook_inventory.py`

- [ ] **Step 1: Write failing snapshot and bound tests**

```python
def test_snapshot_is_default_and_does_not_use_network(monkeypatch):
    monkeypatch.setattr(
        workshop_common.pd,
        "read_csv",
        lambda path, *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected network")
        ) if str(path).startswith("http") else real_read_csv(path, *args, **kwargs),
    )
    first = workshop_common.load_reframe(sample_size=96)
    second = workshop_common.load_reframe(sample_size=96)
    assert first.attrs["source"] == "bundled_snapshot"
    assert first["canonical_ikey"].tolist() == second["canonical_ikey"].tolist()


def test_snapshot_rejects_more_rows_than_it_contains():
    with pytest.raises(ValueError, match="snapshot contains 96"):
        workshop_common.load_reframe(sample_size=97)


def test_advanced_square_matrix_size_is_reported_before_allocation():
    assert workshop_common.square_matrix_bytes(10_000) == 400_000_000
```

Add notebook-source assertions for `DATA_SOURCE = "snapshot"`, `SAMPLE_SIZE = 96`, `FP_BITS = 1024`, and an explicit advanced live-data warning.

- [ ] **Step 2: Run tests and witness RED**

Expected: current loader attempts the live URL and notebook defaults are 10,000 rows and 128 bits.

- [ ] **Step 3: Implement the explicit data API**

```python
def load_reframe(
    sample_size: int = 96,
    anchor_terms: tuple[str, ...] = (),
    random_state: int = 2026,
    *,
    source: str = "snapshot",
):
    if source not in {"snapshot", "live"}:
        raise ValueError("source must be 'snapshot' or 'live'.")
    location = SNAPSHOT_PATH if source == "snapshot" else REFRAME_URL
    frame = pd.read_csv(location)
    if source == "snapshot" and sample_size > len(frame):
        raise ValueError(f"snapshot contains {len(frame)} rows; requested {sample_size}.")
    # Keep the existing parse, anchor, deterministic-sampling, and attrs logic.


def square_matrix_bytes(row_count: int, *, item_size: int = 4) -> int:
    if type(row_count) is not int or row_count < 1:
        raise ValueError("row_count must be a positive integer.")
    return row_count * row_count * item_size


def condensed_distance_bytes(row_count: int, *, item_size: int = 8) -> int:
    if type(row_count) is not int or row_count < 1:
        raise ValueError("row_count must be a positive integer.")
    return row_count * (row_count - 1) // 2 * item_size


def require_bounded_condensed_distances(
    row_count: int, *, maximum_bytes: int = 128 * 1024 * 1024
) -> int:
    required = condensed_distance_bytes(row_count)
    if required > maximum_bytes:
        raise ValueError(
            f"CPU condensed distances require {required} bytes; "
            f"the workshop limit is {maximum_bytes}."
        )
    return required
```

Update Module 1 to use the snapshot/96/1,024 defaults and keep the live 10,000-row path in a separate advanced cell.
Call `require_bounded_condensed_distances()` immediately before the CPU
condensed-distance allocation. Add one captured-output test that requires the
notebook to report the exact data source, row count, backend, fingerprint size,
and elapsed-time label. Do not accept a warning without an enforced bound.
The default path uses the 128 MiB limit. The advanced cell must require the
attendee to set `ADVANCED_LARGE_RUN = True`, print the estimated bytes first,
and then pass an explicit 512 MiB limit. This admits the roughly 400 MB
10,000-row condensed float64 array while keeping it impossible to enter by only
changing `SAMPLE_SIZE`.

- [ ] **Step 4: Verify GREEN and run Module 1 in a clean local kernel**

Use `NVMOLKIT_WORKSHOP_MODE=reference` and a temporary output notebook. Never write outputs back to the tracked notebook.

- [ ] **Step 5: Commit**

```bash
git add notebooks/workshop_common.py notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_common.py tests/test_workshop_notebook_inventory.py
git commit -m "fix: bound the direct nvMolKit lesson"
```

### Task 3: Make Module 2 coherent and reduce the agent surface

**Files:**
- Modify: `notebooks/workshop_llm_agent.py`
- Modify: `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
- Create: `tests/test_workshop_llm_agent.py`

- [ ] **Step 1: Write failing bounded-policy tests**

Test these exact behaviors:

- model stays `nvidia/nemotron-3-nano-30b-a3b`;
- hosted response contains only `MISSING_ANCHOR`, `INVALID_MATRIX`, and two explanations;
- Python renders the function from validated policies;
- no hosted text becomes executable source;
- `reference` mode uses a labeled fixed plan and makes zero client calls;
- `interactive` mode requires the protected key and uses the validated rendered function;
- tracked source contains neither `/nvmolkit-brev-notebook` nor `/.venv`;
- key-shaped values are redacted from all exceptions;
- obsolete code-fence ingestion and generic generated-code repair helpers are absent.

- [ ] **Step 2: Run the focused test and witness RED**

Expected: false absolute paths, reference/generated mismatch, and unused normalizer paths fail.

- [ ] **Step 3: Implement one explicit mode boundary**

```python
WORKSHOP_MODE_ENV = "NVMOLKIT_WORKSHOP_MODE"


def workshop_mode() -> str:
    mode = os.environ.get(WORKSHOP_MODE_ENV, "interactive").strip().lower()
    if mode not in {"interactive", "reference"}:
        raise ValueError(f"{WORKSHOP_MODE_ENV} must be interactive or reference.")
    return mode
```

Keep model output limited to the policy schema. Validate and execute only the Python-owned renderer. In reference mode, use one fixed valid policy and label it `reference`; in interactive mode, label it `hosted_nemotron`.

Remove paths that accept arbitrary model source or repair malformed source.
Do not retain them behind an unreachable branch. Keep narrow validation of the
exact controller-rendered function.

- [ ] **Step 4: Update the notebook flow**

The notebook must derive paths from `Path(workshop_llm_agent.__file__).resolve()`, choose mode once, use the selected implementation without a manual flag contradiction, and print the exact mode.

- [ ] **Step 5: Verify GREEN and execute reference mode in a clean kernel**

Expected: no network or key access; output states `reference`; neighborhood results and acceptance tests pass.

- [ ] **Step 6: Commit**

```bash
git add notebooks/workshop_llm_agent.py \
  notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  tests/test_workshop_llm_agent.py
git commit -m "fix: bound the agent-assisted neighborhood lesson"
```

### Task 4: Make Module 3 meaningful and executable

**Files:**
- Modify: `notebooks/workshop_llm_agent.py`
- Modify: `notebooks/module3_interactive_workflow.py`
- Modify: `notebooks/03_full_agent_reframe_panel_design.ipynb`
- Modify: `tests/test_workshop_llm_agent.py`
- Create: `tests/test_workshop_notebook_execution.py`

- [ ] **Step 1: Write failing panel and interaction tests**

Require:

```python
assert candidate_count == 96
assert panel_count == 24
assert panel_keys < candidate_keys
assert selected_min_distance >= baseline_min_distance
assert selected_descriptor_coverage >= baseline_descriptor_coverage
assert (
    selected_min_distance > baseline_min_distance
    or selected_descriptor_coverage > baseline_descriptor_coverage
)
```

Define minimum distance as the minimum upper-triangle value of
`1 - Tanimoto similarity`. Define descriptor coverage as the mean, across
`MolWt`, `cLogP`, and `TPSA`, of
`(panel_max - panel_min) / (candidate_max - candidate_min)`, with a zero
candidate range contributing `1.0`. The baseline is the first 24 candidates in
stable source order.

Also test that reference mode completes with no hosted calls, interactive launch returns without raising, no later cell raises because approval is pending, a completion callback renders the same validated artifacts, the child process lacks `NVIDIA_API_KEY`, and stale or malformed artifacts fail closed.

- [ ] **Step 2: Run tests and witness RED**

Expected: current panel is 96 from 96, the notebook raises before widget approval, and no reference execution exists.

- [ ] **Step 3: Add the key-free reference API and completion callback**

Add `mode: Literal["hosted", "reference"] = "hosted"` to
`PanelDesignAgent.__init__`. In hosted mode, keep the current protected-key and
client behavior. In reference mode, require `api_key is None` and
`client is None`; never construct an OpenAI client. `request_plan()` returns a
Python-owned `reference_panel_plan()`, and `_request_audit()` returns a
Python-owned `reference_panel_audit()` built from the already validated report.
Both return the existing strict `PanelPlan` and `PanelAudit` schemas, and the
trace records `mode="reference"` or `mode="hosted"`.

Add an optional `on_complete: Callable[[PanelAgentRun], None]` to both
`InteractivePanelDesignWorkflow.__init__` and
`launch_interactive_panel_design()`. The wrapper must forward it. Call it only
after a successful validated run and redact callback failures from the
scientific result.

- [ ] **Step 4: Reorder the notebook**

Define artifact loading, validation, and rendering before the launch cell. Use `PANEL_SIZE = 24`, the 96-row snapshot, and relative paths. In reference mode, run the reference plan and call the renderer. In interactive mode, display the widget and let its completion callback call the same renderer. `Run All` must finish without a cell error while waiting for approval.

- [ ] **Step 5: Verify GREEN and execute Module 3 reference mode**

Require validated `panel.csv`, `report.json`, `agent_trace.json`, readable plots, 24 unique selections, strict-subset status, and passing baseline comparisons.

- [ ] **Step 6: Commit**

```bash
git add notebooks/workshop_llm_agent.py notebooks/module3_interactive_workflow.py \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  tests/test_workshop_llm_agent.py tests/test_workshop_notebook_execution.py
git commit -m "fix: complete the bounded panel-design lesson"
```

### Task 5: Integrate the release and remove baseline ambiguity

**Files:**
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_workshop_notebook_inventory.py`
- Modify: `tests/test_workshop_notebook_execution.py`
- Modify: `README.md`
- Modify: `launchable/fields.md`
- Modify: `requirements.txt` only if clean-kernel execution proves a missing direct dependency

- [ ] **Step 1: Write failing documentation and gate tests**

Require README and fields to list all four notebooks, Module 1 first, interactive/reference mode boundaries, the unchanged fixed model, unchanged key handling, port 8888, 75 GiB disk, and organization-only access.

Add a reusable fake supported-runtime import fixture to the two key-validation tests so they test missing/unsafe key behavior rather than fail first on Mac CUDA.

- [ ] **Step 2: Run the complete local gate and record RED**

Run the full suite with `PYTHONDONTWRITEBYTECODE=1` and cache disabled. Expected before fixes: two key tests plus new documentation assertions fail.

- [ ] **Step 3: Make only the required integration changes**

Do not alter `launchable/setup.sh`, the existing demo notebook, key storage, hardware, access, or ports unless a new failing integration test proves it is necessary.

- [ ] **Step 4: Run fresh verification**

Run, one command at a time:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 -m pytest -q -p no:cacheprovider
python3.12 -m ruff format --check notebooks/*.py tests/*.py
python3.12 -m ruff check notebooks/*.py tests/*.py
bash -n launchable/setup.sh
git diff --check
```

Require zero unexpected test failures. Execute Modules 1–3 in clean reference-mode kernels and keep outputs only in a temporary directory.

- [ ] **Step 5: Secret and release-tree scan**

Reject real `nvapi-` values, private key headers, generated workspaces, notebook outputs, checkpoints, caches, and the unrelated ACS files. Confirm the existing demo SHA-256 still matches the starting revision.

- [ ] **Step 6: Commit**

```bash
git add README.md launchable/fields.md tests/test_demo_agent.py \
  tests/test_workshop_notebook_inventory.py \
  tests/test_workshop_notebook_execution.py requirements.txt
git commit -m "docs: publish the four-notebook workshop path"
```

Omit `requirements.txt` from the commit if it did not change.

### Task 6: Independent review and exact-source publication

**Files:** all changed files in the branch

- [ ] **Step 1: Run independent specification review**

Compare every changed file to the approved design. Repair all Critical and Important findings with a witnessed RED/GREEN test.

- [ ] **Step 2: Run independent code-quality review**

Review scientific claims, resource bounds, subprocess isolation, secret handling, notebook behavior, and YAGNI. Repair every Critical and Important finding.

- [ ] **Step 3: Repeat full local verification**

Require zero unexpected failures and a clean worktree. Record exact commit and test counts.

- [ ] **Step 4: Verify remote-main precondition**

`git ls-remote origin refs/heads/main` must still equal the approved starting commit. If it moved, stop and reconcile instead of force-pushing.

- [ ] **Step 5: Push the reviewed branch to `main`**

```bash
git push origin HEAD:main
git ls-remote origin refs/heads/main
```

Never push or modify `acs-fall-2026-launchable` or another Launchable branch.

### Task 7: Update and validate the one Brev Launchable

**Scope:** only `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`; at most one new environment

- [ ] **Step 1: Re-establish Brev identity without changing the active organization**

Use Brev CLI v0.6.332 or a freshly verified later version. Inspect exact help
first. If login is required, use the supported device/browser flow; never paste
credentials into chat. List organizations read-only. Do not run `brev set`.

Read the current active organization with the supported `brev org` command and
require it to equal the approved organization before any `exec`, `start`,
`stop`, or create action. If it differs, stop and ask the user to switch it;
the agent must not change the active organization.

Version 0.6.332 has `--org` on `brev ls`, but not on `brev exec`, `start`, or
`stop`. Before each such command, run `brev ls --org <exact-org> --json`, require
one matching immutable instance ID/name pair, then use that exact instance name
for the command. Abort if the name is absent, duplicated, or belongs to another
organization. Do not imply that unsupported per-command organization flags
exist.

- [ ] **Step 2: Verify the Console definition**

Confirm source repository/branch, saved setup body, required `NVIDIA_API_KEY`, VM mode, one L4, 75 GiB disk, Jupyter enabled, port 8888, and organization-only access. Change only fields that differ from the reviewed contract.

- [ ] **Step 3: Create exactly one fresh environment**

Use the unchanged Launchable deployment URL. Supply the key only through the Launchable secret parameter. Record instance ID, name, organization, listed hourly price, and reviewed commit without printing the key.

- [ ] **Step 4: Verify setup and source**

Require clean exact source commit, CPython 3.12, one NVIDIA L4, CUDA, nvMolKit 0.5.0, protected key mode 0600, and Jupyter `/api` health.

- [ ] **Step 5: Run L4 deterministic gates**

Run the full test suite and GPU acceptance. Execute Modules 1–3 in fresh reference-mode kernels. Require zero unexpected failures and expected artifacts.

- [ ] **Step 6: Run hosted/browser gates**

In JupyterLab:

- Module 2: one real hosted policy response, validated Python-owned function, and review.
- Module 3: Start Agent, approve one plan, complete 24-compound panel analysis, and inspect the audit and figures.
- Existing demo: complete plan, six approvals, evidence, objective, conclusion, and all figures.

Never include the key in outputs, files, logs, screenshots, or receipts.

- [ ] **Step 7: Verify restart recovery**

Stop/start only the exact approved environment if the Brev instance type supports it. Recheck Jupyter, kernel, key file permissions, and one bounded notebook path. If stop/start is unsupported, report persistence as unverified rather than creating another instance.

- [ ] **Step 8: Hand off the running environment**

Leave the one environment running. Provide:

- Launchable URL;
- organization-only JupyterLab URL;
- direct paths for all four notebooks;
- reviewed commit and acceptance summary;
- current hourly price and a clear warning that billing continues;
- exact stop action.

Do not stop, delete, or replace the environment without later user approval.
