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

**Status:** NOT RUN.

No fresh L4 environment running nvMolKit 0.6.0 was created or used. The
GPU-only test remains intentionally skipped until it runs on a fresh,
task-owned Brev L4. Local deterministic tests do not establish CUDA runtime
compatibility, numerical execution, performance, or device-memory behavior.

A read-only check of the user-provided existing instance found an old checkout:
the origin was the intended GitHub repository, the branch was `main`, and HEAD
was `25781fdbd50ffa894b6f94da8fd2284fa518b9c7`. That instance had Python
3.12.14 and nvMolKit 0.5.0, not nvMolKit 0.6.0. Its worktree was not clean.
Therefore, it was neither patched nor reused as upgrade evidence.

## Hosted-model evidence

**Status:** NOT RUN.

No hosted-model request was made. Local reference-mode, mock-client, protocol,
and replay tests do not establish hosted endpoint availability, latency, or
response behavior.

## Browser evidence

**Status:** NOT RUN.

No JupyterLab browser session or Secure Link was opened. Local notebook schema,
clean-state, compile, and execution tests do not establish browser rendering,
widget interaction, or the live attendee flow.

## GitHub PR closeout

**Status:** COMPLETE.

GitHub `main` was verified at
`c8a24115f6a75efe4dc8fb9bcaec3e88b8fe7f25`. Runtime source and notebook
qualification remains attributed to tested source
`0cc05cc7506a89df2fd64c2cfab3055829d59adb`. Commit `c8a2411` is a
qualification/report-only documentation commit relative to that tested source:
it added this report and corrected two factual examples in the implementation
plan; it did not change runtime source or notebooks.

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

**Status:** PREFLIGHT PASS; DEFINITION UPDATE NOT DONE.

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
surface was identified, and no Launchable definition edit was made.

The old instance credential file was checked only for metadata. Its mode was
600, owner UID was 1000, and size was 70 bytes. Its contents and value were not
read, copied, or recorded.

**Next required action:** make a fresh deployment from the
[supported Launchable deployment page](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW),
with the user entering their `NVIDIA_API_KEY` there. This is blocked on that
user action. The old, dirty nvMolKit 0.5.0 instance must not be patched or used
as nvMolKit 0.6.0 evidence.

Until a fresh deployment completes, L4 GPU, hosted-model, and browser evidence
for nvMolKit 0.6.0 remain `NOT RUN`. This record does not claim that the
Launchable definition or live attendee experience is updated or ready.

## Residual risks and rollback

- L4 GPU, hosted-model, and browser gates for nvMolKit 0.6.0 are still
  `NOT RUN`. They must remain separate from local evidence, GitHub publication,
  CLI deployment preflight, and inspection of the old nvMolKit 0.5.0 instance.
- The Brev CLI dry run did not inspect or edit the saved Launchable definition.
  A fresh deployment is required, and it is blocked on the user entering their
  key through the supported deployment page.
- Three changed Python files retain whole-file Ruff formatting debt that was
  already present at the comparison base. The changed hunks are range-format
  clean; removing the unrelated debt is outside this upgrade.
- Commit `c8a24115f6a75efe4dc8fb9bcaec3e88b8fe7f25` and this report update are
  documentation changes made after tested source
  `0cc05cc7506a89df2fd64c2cfab3055829d59adb`. They do not change runtime code or
  notebooks.
- Before publication, use non-destructive Git reverts to return to the accepted
  comparison base if a final review finds a release-blocking issue. After any
  live Launchable update, restore the last accepted source only through the
  supported Brev surface and repeat every affected external gate.
