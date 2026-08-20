# nvMolKit 0.6.0 qualification record

## Local deterministic evidence

**Status:** PASS for the deterministic local release gate.

**Tested source:** `0cc05cc7506a89df2fd64c2cfab3055829d59adb`

**Comparison base:** `25781fdbd50ffa894b6f94da8fd2284fa518b9c7`

**Environment:** Python 3.12.3 on Darwin 25.6.0 arm64. The run was serial,
used the non-interactive Matplotlib backend, used fresh writable Matplotlib and
IPython directories under `/private/tmp`, and disabled the pytest cache.

**Complete suite:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/private/tmp/nvmolkit-v06-final-mpl \
  IPYTHONDIR=/private/tmp/nvmolkit-v06-final-ipython \
  JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider
```

Result: exit 0; `937 passed, 1 skipped in 113.04s`. The only skip was
`tests/test_gpu_acceptance.py::test_nvmolkit_gpu_workflow`, whose declared
reason is `set RUN_GPU_TESTS=1 on the task-owned Brev GPU`.

**Changed-file lint and format checks:**

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff check \
  chemistry_workflow.py command_receipts.py notebooks/nvmolkit_compat.py \
  notebooks/workshop_llm_agent.py tests/test_chemistry_workflow.py \
  tests/test_command_receipts.py tests/test_gpu_acceptance.py \
  tests/test_nvmolkit_compat.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py \
  tests/test_workshop_notebook_inventory.py
```

Result: exit 0; all 11 changed Python files passed Ruff lint.

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m ruff format \
  --check command_receipts.py notebooks/nvmolkit_compat.py \
  notebooks/workshop_llm_agent.py tests/test_gpu_acceptance.py \
  tests/test_nvmolkit_compat.py tests/test_workshop_llm_agent.py \
  tests/test_workshop_notebook_execution.py \
  tests/test_workshop_notebook_inventory.py
```

Result: exit 0; all eight files were already formatted.

The whole-file Ruff formatter also reported existing formatting debt in
`chemistry_workflow.py`, `tests/test_chemistry_workflow.py`, and
`tests/test_command_receipts.py`. The same three base versions fail the
whole-file formatter. No broad reformat was made. Every changed hunk in those
three files passed Ruff 0.15.0 range-format checks at these exact ranges:

```text
chemistry_workflow.py: 13-14, 175-182, 524-537
tests/test_chemistry_workflow.py: 390-397, 446-448, 560, 691-706, 714-786
tests/test_command_receipts.py: 54-60, 201-205, 479-526
```

Each range was checked with this exact command form and exited 0:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m ruff format --check --range=<start>-<end> <file>
```

**Compilation, notebook, shell, whitespace, and secret gates:**

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c \
  'import pathlib, subprocess; paths=[pathlib.Path(p.decode()) for p in subprocess.check_output(["git","ls-files","-z","*.py"]).split(b"\0") if p]; [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in paths]; print(f"compiled {len(paths)} tracked Python files")'
```

Result: exit 0; 30 tracked Python files compiled.

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c \
  'import pathlib, nbformat; paths=sorted(pathlib.Path("notebooks").glob("*.ipynb")); cells=[(path, cell) for path in paths for cell in nbformat.read(path, as_version=nbformat.NO_CONVERT).cells if cell.cell_type=="code"]; [compile(cell.source, f"{path}:{cell.id}", "exec") for path,cell in cells]; print(f"compiled {len(cells)} code cells in {len(paths)} notebooks")'
```

Result: exit 0; 31 code cells compiled across all four notebooks.

```bash
bash -n launchable/setup.sh
```

Result: exit 0.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/private/tmp/nvmolkit-v06-final-mpl \
  IPYTHONDIR=/private/tmp/nvmolkit-v06-final-ipython \
  JUPYTER_PLATFORM_DIRS=1 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q -p no:cacheprovider \
  tests/test_workshop_notebook_inventory.py::test_primary_notebooks_are_clean_python_312_notebooks
```

Result: exit 0; `1 passed in 0.43s`. This validated nbformat v4 schema,
Python 3.12 metadata, non-empty unique cell IDs, clean outputs and execution
counts, absent attachments, and absent saved widget state for all four
notebooks.

```bash
git diff --check 25781fdbd50ffa894b6f94da8fd2284fa518b9c7..HEAD
```

Result: exit 0.

```bash
/opt/homebrew/bin/gitleaks git --redact --no-banner \
  --log-opts=25781fdbd50ffa894b6f94da8fd2284fa518b9c7..HEAD .
```

Result: exit 0 with Gitleaks 8.30.1; six commits and about 78.11 KB scanned;
no leaks found.

**Tested changed-file scope:** 17 files, 1,980 insertions, and 112 deletions
from the comparison base through the tested source:

```text
chemistry_workflow.py
command_receipts.py
docs/superpowers/plans/2026-08-19-nvmolkit-060-launchable-upgrade.md
docs/superpowers/specs/2026-08-19-nvmolkit-060-launchable-upgrade-design.md
notebooks/01_direct_nvmolkit_reframe.ipynb
notebooks/02_agent_assisted_reframe_neighborhoods.ipynb
notebooks/03_full_agent_reframe_panel_design.ipynb
notebooks/nvmolkit_compat.py
notebooks/workshop_llm_agent.py
requirements.txt
tests/test_chemistry_workflow.py
tests/test_command_receipts.py
tests/test_gpu_acceptance.py
tests/test_nvmolkit_compat.py
tests/test_workshop_llm_agent.py
tests/test_workshop_notebook_execution.py
tests/test_workshop_notebook_inventory.py
```

`README.md` and `launchable/fields.md` were inspected for an incorrect exact
nvMolKit version or the removed square-matrix memory statement. Neither file
contained such a statement, so neither file was changed.

## L4 GPU evidence

**Status:** PASS for the fresh L4 runtime and complete GPU-enabled test suite.

The user created a fresh deployment from Launchable
`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`. Qualification used only the resulting
task-owned instance:

```text
name: nvmolkit---nemotron-notebook-436a34
instance ID: 28zcii3yz
machine: g6.xlarge
GPU: one NVIDIA L4
```

The preflight verified a clean checkout at published commit
`fa8fefb49e7e288c3ff7823f1005244db1732667`, the expected public origin,
setup-script SHA-256
`daf428d54e3bdb3ad289ff069f5e5aff143edffed2a5497f8d687ffa77ec3153`,
Python 3.12.14, nvMolKit 0.6.0, one L4 with CUDA capability 8.9, and a healthy
Jupyter Server 2.20.0 API. The protected key was checked only for regular-file,
owner, mode-0600, and bounded-size properties. Its value was not printed or
recorded.

The full suite then ran serially with `RUN_GPU_TESTS=1`, the ambient
`NVIDIA_API_KEY` variable strictly absent, and the pytest child launched with
that variable removed. Result: exit 0; exactly `938 passed`, with zero skips,
failures, or errors. A separate JUnit receipt verifier confirmed all eight
required GPU, reference-kernel, inventory, replay, and provenance gates.

Evidence hashes:

```text
pytest log: 412bae3b6ef1f9726b5b02a6897ac07907af9bd5de2705305f2f555ea6e0170f
JUnit XML:  06d218bf050d4fd2864d0ba635441a2fdaade51e285ec49d1f241828b48be96f
```

An optional text scan in the first wrapper used a malformed grep expression
and was discarded. It is not part of this PASS decision. The independent
JUnit verifier, exact test counts, required-gate statuses, clean-tree checks,
and process-environment controls are the accepted evidence.

This gate establishes fresh L4/CUDA execution with nvMolKit 0.6.0. It does not
establish throughput, latency, cost, or device-memory performance.

## Hosted-model evidence

**Status:** PASS for the hosted-kernel smoke; browser-human acceptance was not
part of this gate.

A reviewed, temporary-only harness executed in-memory copies of Modules 2 and
3 with the exact managed Python 3.12 kernel, one L4, nvMolKit 0.6.0, and model
`nvidia/nemotron-3-nano-30b-a3b`. Provider retries were disabled. It completed
in 23.644 seconds and made exactly three hosted requests:

- Module 2 made one `submit_neighborhood_policy` request, returned mode
  `hosted_nemotron`, and produced a 60-row GPU-backed atlas.
- Module 3 made one `submit_panel_plan` request and one
  `submit_panel_audit` request. The sponsor path approved strategy 2 after the
  model recommended strategy 1. Exactly one isolated `nvmolkit-gpu` analysis
  ran, selected 24 of 96 candidates, passed independent artifact validation,
  and completed the audit.
- Replaying the Module 3 receipt and gallery produced zero new hosted requests
  and zero new analysis executions. Both the original and replayed gallery
  cells emitted SVG output.

The harness saved no executed source notebooks, left the published checkout
clean, and passed its exact-key and `nvapi-` output/artifact scans. Receipt
hashes:

```text
Module 2 in-memory notebook: eaf48f91e689ca989a776d37c561a0cd77ebe62092b9783471aa74bf1b0a0ff6
Module 3 in-memory notebook: c294e84f987022a6df8d137482a64f199b2e764e6c772316146579dd1c62f41f
Module 3 validated report:   b44db12ade7a95863c6bbb0f7808c3b0634c5b40b1084266489c3720c6e97159
```

The recorded elapsed time is one diagnostic observation, not a performance
benchmark. This result establishes one successful hosted-kernel workflow. It
does not establish endpoint reliability, a latency distribution, JupyterLab
frontend rendering, or the complete human attendee experience.

## Browser evidence

**Status:** NOT RUN.

No human JupyterLab session or Secure Link flow was exercised. The
hosted-kernel smoke verified widget-driven workflow state and SVG MIME output,
but it did not verify browser rendering, manual widget interaction, autosave
behavior, or the live attendee flow.

## GitHub PR closeout

**Status:** COMPLETE.

The published source used by the fresh deployment was verified at
`fa8fefb49e7e288c3ff7823f1005244db1732667`. Runtime source and notebook
qualification remains attributed to tested source
`0cc05cc7506a89df2fd64c2cfab3055829d59adb`. Commit `c8a2411` is a
qualification/report-only documentation commit relative to that tested source:
it added this report and corrected two factual examples in the implementation
plan. Commit `fa8fefb` added publication and Brev preflight evidence. Neither
commit changed runtime source or notebooks.

- [PR #1](https://github.com/ktretina/nvmolkit-brev-notebook/pull/1)
  received an
  [evidence-linked supersession comment](https://github.com/ktretina/nvmolkit-brev-notebook/pull/1#issuecomment-5349527453)
  and was verified `CLOSED`.
- [PR #2](https://github.com/ktretina/nvmolkit-brev-notebook/pull/2)
  received an
  [evidence-linked supersession comment](https://github.com/ktretina/nvmolkit-brev-notebook/pull/2#issuecomment-5349529207)
  and was verified `CLOSED`.

The accepted implementation was integrated directly; neither superseded pull
request was merged. Kevin Boyd's contribution credit is retained in the
accepted history.

## Brev Launchable definition

**Status:** PREFLIGHT PASS; fresh deployment runtime passed through the
hosted-kernel smoke; browser-human acceptance was not run; the saved definition
was not directly inspected or edited.

The installed Brev CLI was version 0.6.332. The active organization was
`agents-in-ls`, ID `org-3FVWXFV8irpOznkgYhF29j6Osqx`.

An exact CLI dry run for Launchable
`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` reported:

- name `nvMolKit + Nemotron Notebook`;
- instance type `g6.xlarge`;
- 75 GiB storage;
- VM mode; and
- exactly one required parameter, `NVIDIA_API_KEY`.

A second dry run supplied a clearly dummy, non-secret value for that parameter.
It succeeded and created no instance. These dry runs establish only that the
installed CLI could resolve and preview deployment of the named Launchable.
They do not establish the saved repository source, setup body, branch, notebook
order, access policy, or live runtime behavior.

NVIDIA's
[Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
documents Launchable creation and editing in the Brev Console. The
[CLI instance-management documentation](https://docs.nvidia.com/brev/cli/instance-management)
covers instance deployment and management. No supported CLI definition-editing
surface was identified, and no Launchable definition edit was made through the
CLI in this qualification run.

The user completed that supported deployment action. The resulting fresh
instance built the published source at `fa8fefb`, passed the setup, L4,
GPU-suite, and hosted-kernel gates recorded above, and remained clean. No old
instance was patched or used as nvMolKit 0.6.0 evidence.

This runtime evidence proves what the Launchable produced for this deployment.
It does not directly inspect every saved Console field or prove the
organization-only Secure Link and JupyterLab browser experience. Those remain
separate Console/browser checks.

## Residual risks and rollback

- Browser-human acceptance is still `NOT RUN`. The L4 suite and hosted-kernel
  smoke do not prove JupyterLab widget rendering, a human click path, or the
  organization-only Secure Link.
- The fresh deployment proves the resulting runtime, but the Brev CLI did not
  directly inspect every saved Launchable Console field.
- The task-owned instance was still running at the end of qualification and may
  continue to incur charges until the user stops it.
- Three changed Python files retain whole-file Ruff formatting debt that was
  already present at the comparison base. The changed hunks are range-format
  clean; removing the unrelated debt is outside this upgrade.
- Commits `c8a24115f6a75efe4dc8fb9bcaec3e88b8fe7f25`,
  `fa8fefb49e7e288c3ff7823f1005244db1732667`, and this report update are
  documentation changes made after tested source
  `0cc05cc7506a89df2fd64c2cfab3055829d59adb`. They do not change runtime code or
  notebooks.
- Use non-destructive Git reverts to return to the accepted comparison base if
  a later review finds a release-blocking issue. After any live Launchable
  update, restore the last accepted source only through the supported Brev
  surface and repeat every affected external gate.
