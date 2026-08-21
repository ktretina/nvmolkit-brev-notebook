# Zero-Input Workshop Key Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every attendee-facing API-key input from the nvMolKit + Nemotron Notebook Launchable while keeping the shared workshop credential out of the repository.

**Architecture:** `launchable/setup.sh` becomes a redacted operator template with one replaceable sentinel. The private Brev Console copy contains the workshop key and persists it to the existing protected file; repository tests render only fake keys. Launchable and notebook copy state that deployment has no Setup values and attendees do not create or enter an NVIDIA API key.

**Tech Stack:** Bash, Python 3.12, pytest, Jupyter notebook JSON, Brev Launchable documentation.

---

### Task 1: Convert setup input handling to a redacted template

**Files:**
- Modify: `tests/test_notebook.py`
- Modify: `launchable/setup.sh`

- [ ] **Step 1: Add the sentinel and render-only test harness**

Add this constant below `REPO_ROOT` in `tests/test_notebook.py`:

```python
SETUP_KEY_SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
```

Change `_run_setup` so tests render only a fake key into the temporary copied
script. Keep all existing fake-home, fake-Python, invocation-log, and execution
logic. Apply these exact changes:

```diff
-def _run_setup(tmp_path, setup_values):
+def _run_setup(tmp_path, rendered_key=None, setup_values=None):
```

```diff
-    } | setup_values
+    } | (setup_values or {})
     copied_setup = tmp_path / "brev-generated-setup.sh"
-    copied_setup.write_bytes((REPO_ROOT / "launchable" / "setup.sh").read_bytes())
+    setup_source = (REPO_ROOT / "launchable" / "setup.sh").read_text(
+        encoding="utf-8"
+    )
+    if rendered_key is not None:
+        assert setup_source.count(SETUP_KEY_SENTINEL) == 1
+        setup_source = setup_source.replace(SETUP_KEY_SENTINEL, rendered_key)
+    copied_setup.write_text(setup_source, encoding="utf-8")
```

In `test_setup_uses_brev_managed_python_and_leaves_jupyter_to_brev`, replace
the environment-input cleanup assertions with this template contract:

```python
    assert setup.count(SETUP_KEY_SENTINEL) == 1
    assert "required in Brev Setup values" not in setup
    assert 'launch_api_key="${NVIDIA_INFERENCE_API_KEY}"' not in setup
    assert 'launch_api_key="${NVIDIA_API_KEY}"' not in setup
    assert setup.index("unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY") < setup.index(
        'install -d -m 700 "${api_key_directory}"'
    )
```

- [ ] **Step 2: Replace environment-input tests with zero-input tests**

Delete these obsolete tests:

- `test_setup_prefers_inference_key_and_runs_only_managed_runtime`
- `test_setup_accepts_legacy_variable_name_only_for_sk_key`
- `test_setup_harness_does_not_inherit_ambient_credentials`
- `test_setup_rejects_invalid_primary_without_falling_back`
- `test_setup_rejects_empty_primary_without_falling_back`
- `test_setup_rejects_legacy_build_key`

Add these exact tests:

```python
def test_unrendered_setup_template_fails_before_installation(tmp_path):
    result, fake_home, log = _run_setup(tmp_path)

    assert result.returncode != 0
    assert "private Brev Console copy" in result.stderr
    assert SETUP_KEY_SENTINEL not in result.stdout + result.stderr
    assert not log.exists()
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_rendered_setup_persists_workshop_key_and_ignores_environment_keys(tmp_path):
    rendered_key = "sk-rendered-workshop-test-key-must-not-leak"
    ambient_primary = "sk-ambient-primary-test-key-must-not-leak"
    ambient_legacy = "sk-ambient-legacy-test-key-must-not-leak"
    result, fake_home, log = _run_setup(
        tmp_path,
        rendered_key=rendered_key,
        setup_values={
            "NVIDIA_INFERENCE_API_KEY": ambient_primary,
            "NVIDIA_API_KEY": ambient_legacy,
        },
    )

    assert result.returncode == 0, result.stderr
    key_directory = fake_home / ".config" / "nvmolkit"
    key_file = key_directory / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == rendered_key
    assert key_directory.stat().st_mode & 0o777 == 0o700
    assert key_file.stat().st_mode & 0o777 == 0o600
    combined_output = result.stdout + result.stderr
    assert rendered_key not in combined_output
    assert ambient_primary not in combined_output
    assert ambient_legacy not in combined_output
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert invocations.count("ENV_CLEAN") == 2
    assert invocations.index("MODULE ensurepip --upgrade ") < invocations.index(
        "MODULE pip install --upgrade"
    )
    assert invocations.index("MODULE pip install --upgrade") < invocations.index(
        "MODULE pip install -r"
    )
    assert invocations.index("MODULE pip install -r") < invocations.index(
        "SMOKE"
    ) < invocations.index("HEALTH")


def test_rendered_setup_rejects_non_inference_key_without_leaking(tmp_path):
    invalid_key = "nvapi-rendered-build-key-must-not-leak"
    result, fake_home, log = _run_setup(tmp_path, rendered_key=invalid_key)

    assert result.returncode != 0
    assert "Inference Hub key beginning with sk-" in result.stderr
    assert invalid_key not in result.stdout + result.stderr
    assert not log.exists()
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()
```

Retain the existing widget-setting assertions by adding them to
`test_rendered_setup_persists_workshop_key_and_ignores_environment_keys` after
the permission checks:

```python
    widget_settings = (
        fake_home
        / ".jupyter"
        / "lab"
        / "user-settings"
        / "@jupyter-widgets"
        / "jupyterlab-manager"
        / "plugin.jupyterlab-settings"
    )
    assert json.loads(widget_settings.read_text(encoding="utf-8")) == {
        "saveState": True
    }
```

- [ ] **Step 3: Run the setup tests and confirm the intended RED state**

Run:

```bash
/Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q tests/test_notebook.py -k "setup"
```

Expected: failures show that `launchable/setup.sh` still reads Brev Setup-value
environment variables and does not contain the template sentinel.

- [ ] **Step 4: Implement the minimal setup template**

Replace the current credential-selection block at the top of
`launchable/setup.sh` with this exact block, before project discovery:

```bash
launch_api_key='__NVIDIA_INFERENCE_API_KEY__'
unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY

if [[ "${launch_api_key}" == __NVIDIA_* ]]; then
  unset launch_api_key
  echo "Error: render a private Brev Console copy by replacing the credential placeholder." >&2
  exit 1
fi

if [[ "${launch_api_key}" != sk-* ]]; then
  unset launch_api_key
  echo "Error: provide an NVIDIA Inference Hub key beginning with sk-." >&2
  exit 1
fi
```

Remove the old environment-variable selection and its duplicate `sk-`
validation. Preserve the existing atomic write, modes, `unset launch_api_key`,
dependency installation, GPU smoke test, and Jupyter health probe unchanged.

- [ ] **Step 5: Run the focused setup tests to GREEN**

Run:

```bash
/Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q tests/test_notebook.py -k "setup or health_probe"
bash -n launchable/setup.sh
test "$(wc -c < launchable/setup.sh)" -le 16384
```

Expected: all selected tests pass, Bash syntax passes, and the size check exits
zero.

- [ ] **Step 6: Commit the setup template change**

```bash
git add launchable/setup.sh tests/test_notebook.py
git commit -m "Remove attendee key input from setup"
```

### Task 2: Align Launchable and notebook instructions with zero-input deployment

**Files:**
- Modify: `tests/test_notebook.py`
- Modify: `tests/test_workshop_notebook_inventory.py`
- Modify: `README.md`
- Modify: `launchable/fields.md`
- Modify: `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`

- [ ] **Step 1: Write failing documentation-contract tests**

Replace
`test_launchable_contract_fixes_storage_model_port_and_one_setup_value` with:

```python
def test_launchable_contract_fixes_storage_model_port_and_zero_setup_values():
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    assert "75 GiB" in fields
    assert "50 GiB" not in fields
    assert "No Launch parameters or Setup values" in fields
    assert "required Text parameter" not in fields
    assert "no default" not in fields.lower()
    assert "`nvidia/nvidia/nemotron-3-nano-30b-a3b`" in fields
    assert "`https://inference-api.nvidia.com/v1`" in fields
    assert "port `8888`" in fields
    assert "Remove `NVIDIA_INFERENCE_API_KEY`, `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT`" in fields
    assert "redacted operator template" in fields
    assert "private working copy" in fields
```

In `test_readme_preserves_launch_and_separate_acceptance_gates`, add:

```python
    assert "No Launch parameters or Setup values" in readme
    assert "Attendees enter no API key" in readme
    assert "required Text parameter" not in readme
```

In `test_release_docs_publish_the_three_notebook_path_and_launch_contract`, add
these assertions inside the document loop:

```python
        assert "No Launch parameters or Setup values" in document
        assert "required Text parameter" not in document
        assert "entered once in Brev Setup values" not in document
```

Replace `test_module2_explains_the_organizer_supplied_inference_hub_key` with:

```python
def test_module2_explains_the_preprovisioned_inference_hub_key():
    module2 = nbformat.read(
        NOTEBOOK_DIR / "02_agent_assisted_reframe_neighborhoods.ipynb", as_version=4
    )
    markdown = "\n".join(
        cell.source for cell in module2.cells if cell.cell_type == "markdown"
    )
    assert "organizer-supplied" in markdown
    assert "Inference Hub" in markdown
    assert "preprovisions" in markdown
    assert "do not create or enter" in markdown
    assert "entered once in Brev Setup values" not in markdown
    assert "`nvapi-`" not in markdown
```

- [ ] **Step 2: Run the documentation tests and confirm the intended RED state**

Run:

```bash
PYTHONPATH=/private/tmp/nvmolkit-test-deps /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q tests/test_notebook.py::test_readme_preserves_launch_and_separate_acceptance_gates tests/test_notebook.py::test_launchable_contract_fixes_storage_model_port_and_zero_setup_values tests/test_workshop_notebook_inventory.py::test_release_docs_publish_the_three_notebook_path_and_launch_contract tests/test_workshop_notebook_inventory.py::test_module2_explains_the_preprovisioned_inference_hub_key
```

Expected: failures point only to the stale one-input Launchable and Module 2
copy.

- [ ] **Step 3: Update the README launch steps**

Replace the numbered list under `## Launch` with this exact content:

```markdown
1. Create or edit the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md). Set the default disk storage to **75 GiB**. `launchable/setup.sh` is a redacted operator template, not a deployable script. Make a private working copy, replace its credential placeholder with the approved workshop-only Inference Hub key, and paste only that rendered copy into the Software configuration setup-script field. Never commit, upload, or share the rendered copy. Updating the repository does not replace the script body already saved in a Launchable.
2. Configure **No Launch parameters or Setup values**. Remove `NVIDIA_INFERENCE_API_KEY`, `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT`. The saved setup script provisions the organizer-supplied credential. Attendees enter no API key and do not need to create an NVIDIA API account or key.
3. Enable Jupyter and keep access set to **Only my organization** with a Secure Link on the fixed port `8888`; do not expose unrestricted public TCP. The hosted model is fixed to `nvidia/nvidia/nemotron-3-nano-30b-a3b` at `https://inference-api.nvidia.com/v1`.
4. The rendered setup script stores the workshop key outside the repository in `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY` with file mode `0600`; notebook preflight loads it automatically without a prompt. A person who controls a deployed VM can recover this shared key. Use a workshop-only key, monitor it, and rotate or revoke it after the event.
5. Open JupyterLab through the Secure Link and start with `notebooks/01_direct_nvmolkit_reframe.ipynb`.
```

Keep the rest of the README unchanged.

- [ ] **Step 4: Update the Brev Console field contract**

Replace the Launch-parameter bullet in `launchable/fields.md` with:

```markdown
- **Launch parameters:** **No Launch parameters or Setup values.** Remove `NVIDIA_INFERENCE_API_KEY`, `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT`.
```

Replace the two credential paragraphs after the notebook list with:

```markdown
The organizer preprovisions an approved workshop-only Inference Hub key. Attendees enter no API key and do not need to create an NVIDIA API account or key.

Author this Launchable in the Brev web Console only. `launchable/setup.sh` is a redacted operator template and must not be pasted as-is. Make a private working copy, replace `__NVIDIA_INFERENCE_API_KEY__` with the approved `sk-` workshop key, and paste only the rendered copy into the saved setup-script field. Never commit, upload, attach, or share that rendered copy. A repository update does not replace the saved Console body.

The rendered setup script stores the key at `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY`, outside the repository, with directory mode `0700` and file mode `0600`. The notebooks load it automatically and do not request it. A person who controls a deployed VM can recover the key. Use a workshop-only key, monitor it during the event, and rotate or revoke it afterward. Deleting a VM removes that VM's file but does not replace key revocation.
```

- [ ] **Step 5: Update the Module 2 attendee copy only**

In `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`, replace the stale
sentence in the introductory markdown cell with:

```text
Interactive mode requires network access. The Launchable preprovisions the organizer-supplied NVIDIA Inference Hub key; attendees do not create or enter an NVIDIA API key. For recovery, set `NVMOLKIT_WORKSHOP_MODE=reference`, restart, and rerun the notebook; reference mode makes zero client calls. Keep `workshop_llm_agent.py`, `workshop_common.py`, and `data/reframe_teaching_snapshot.csv` beside this notebook.
```

Do not modify notebook code, outputs, metadata, model locks, or any other
notebook.

- [ ] **Step 6: Run the documentation tests to GREEN**

Run:

```bash
PYTHONPATH=/private/tmp/nvmolkit-test-deps /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q tests/test_notebook.py tests/test_workshop_notebook_inventory.py -k "readme or launchable or release_docs or module2_explains"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the instruction changes**

```bash
git add README.md launchable/fields.md notebooks/02_agent_assisted_reframe_neighborhoods.ipynb tests/test_notebook.py tests/test_workshop_notebook_inventory.py
git commit -m "Document zero-input workshop deployment"
```

### Task 3: Verify the complete repository change

**Files:**
- Verify only; no production file changes expected.

- [ ] **Step 1: Run focused setup, documentation, inventory, and helper tests**

Run:

```bash
PYTHONPATH=/private/tmp/nvmolkit-test-deps JUPYTER_PATH=/private/tmp/nvmolkit-kernel-20260821/share/jupyter MPLBACKEND=Agg /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q tests/test_notebook.py tests/test_workshop_notebook_inventory.py tests/test_workshop_llm_agent.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run static setup and stale-copy checks**

Run:

```bash
bash -n launchable/setup.sh
test "$(wc -c < launchable/setup.sh)" -le 16384
! rg -n 'sk-[A-Za-z0-9_-]{20,}' README.md launchable notebooks
! rg -n 'required Text parameter `NVIDIA_INFERENCE_API_KEY`|entered once in Brev Setup values|Keep exactly one parameter' README.md launchable notebooks/02_agent_assisted_reframe_neighborhoods.ipynb
git diff --check origin/main...HEAD
```

Expected: every command exits zero and neither `rg` command prints a match.

- [ ] **Step 3: Run the complete deterministic suite**

Run:

```bash
PYTHONPATH=/private/tmp/nvmolkit-test-deps JUPYTER_PATH=/private/tmp/nvmolkit-kernel-20260821/share/jupyter MPLBACKEND=Agg /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/objective-rationale-quality/.venv/bin/python -m pytest -q
```

Expected: at least `947 passed, 1 skipped`, with no failures. The lower bound
accounts for replacing six obsolete environment-input cases with three
zero-input template cases.

- [ ] **Step 4: Confirm the exact scope**

Run:

```bash
git status --short --branch
git diff --name-only origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected changed production surfaces are only `launchable/setup.sh`,
`launchable/fields.md`, `README.md`, and the introductory markdown in Module 2.
Tests, this plan, and the approved design may also differ. No other notebook,
Launchable, or remote environment changes are allowed.

The user's saved Brev Console update is configuration evidence only. Do not
claim future-instance acceptance until the user deploys a fresh instance and
confirms hosted notebook and browser tests.
