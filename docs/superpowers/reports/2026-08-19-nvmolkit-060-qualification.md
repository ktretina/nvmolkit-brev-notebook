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

No L4 environment was created or used during this local gate. The GPU-only
test remains intentionally skipped until it runs on the task-owned Brev L4.
Local deterministic tests do not establish CUDA runtime compatibility,
numerical execution, performance, or device-memory behavior.

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

**Status:** NOT RUN.

No branch was pushed, no pull request was commented on, and no pull request was
closed during this task.

## Brev Launchable definition

**Status:** NOT RUN.

No Brev command, Console operation, definition inspection, definition edit, or
deployment was performed during this task. The saved Launchable source, setup
body, fields, access policy, and current environment state are not qualified by
this local record.

## Residual risks and rollback

- L4 GPU, hosted-model, browser, GitHub closeout, and Brev Launchable gates are
  still `NOT RUN`. They must remain separate from this deterministic local
  evidence.
- Three changed Python files retain whole-file Ruff formatting debt that was
  already present at the comparison base. The changed hunks are range-format
  clean; removing the unrelated debt is outside this upgrade.
- The report and the two factual plan-example corrections are documentation
  changes made after tested source `0cc05cc7506a89df2fd64c2cfab3055829d59adb`.
  They do not change runtime code or notebooks.
- Before publication, use non-destructive Git reverts to return to the accepted
  comparison base if a final review finds a release-blocking issue. After any
  live Launchable update, restore the last accepted source only through the
  supported Brev surface and repeat every affected external gate.
