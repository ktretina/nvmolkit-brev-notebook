# Launchable Fixed Setup Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Brev deployment request only an NVIDIA API key, carry that key securely into later Jupyter kernels, and fix storage, model, and Jupyter port values.

**Architecture:** The VM setup script will copy the launch-time key into a private file outside the repository with directory mode `0700` and file mode `0600`. Notebook preflight will use the process environment first, then securely read that file without a prompt; the existing Python call path will pass the returned key directly to the OpenAI client. The model and Jupyter port remain source constants, while the Brev Console authoring guide will specify 75 GiB storage and only one launch parameter.

**Tech Stack:** Bash, Python 3.12, pytest, Brev VM Launchables, JupyterLab

---

### Task 1: Define the protected credential contract

**Files:**
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_notebook.py`

- [ ] **Step 1: Write failing preflight tests**

Add tests that prove preflight reads `~/.config/nvmolkit/NVIDIA_API_KEY`, rejects missing credentials without calling `getpass`, rejects unsafe file permissions, and does not print the secret.

- [ ] **Step 2: Write failing setup tests**

Extend the setup harness with a launch-time sentinel key and assert that setup creates the private directory and file with modes `0700` and `0600`, without exposing the key in stdout or stderr.

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_demo_agent.py tests/test_notebook.py
```

Expected: the new tests fail because setup does not persist the key and preflight still prompts.

### Task 2: Implement the key handoff

**Files:**
- Modify: `launchable/setup.sh`
- Modify: `demo_agent.py`

- [ ] **Step 1: Persist the setup value safely**

Require `NVIDIA_API_KEY`, create `~/.config/nvmolkit` with mode `0700`, write through a same-directory private temporary file, set mode `0600`, atomically rename it, and unset the setup process variable. Never print the value.

- [ ] **Step 2: Load the protected value without a prompt**

Use `NVIDIA_API_KEY` from the current process when present. Otherwise, open the fixed credential path without following symlinks, verify the current user owns a regular file with mode `0600`, read a bounded UTF-8 value, and fail with redeployment guidance when unavailable. Remove `getpass` from preflight.

- [ ] **Step 3: Run focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_demo_agent.py tests/test_notebook.py
```

Expected: all focused tests pass.

### Task 3: Fix the Launchable-facing values

**Files:**
- Modify: `launchable/fields.md`
- Modify: `README.md`

- [ ] **Step 1: Update the authoring contract**

Specify 75 GiB default storage, a single required text parameter named `NVIDIA_API_KEY` with no default, the fixed model `nvidia/nemotron-3-nano-30b-a3b`, and the fixed Jupyter Secure Link port `8888`. Explicitly remove `NEMOTRON_MODEL` and `JUPYTER_PORT` from Launch parameters.

- [ ] **Step 2: Update operator guidance**

Explain that the key is copied to a private VM file and loaded automatically by the notebook. Remove the hidden-prompt instructions and describe deletion of the environment as the credential-removal boundary.

- [ ] **Step 3: Add source assertions**

Assert the exact disk, parameter, model, port, permission, and no-prompt contract in repository tests.

### Task 4: Verify and publish

**Files:**
- Verify: all repository files

- [ ] **Step 1: Run the full suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: no failures; the live GPU test may remain skipped when no compatible GPU gate is enabled.

- [ ] **Step 2: Run static checks**

Run:

```bash
.venv/bin/python -m py_compile demo_agent.py
git diff --check
git status --short
```

Expected: compilation and diff checks succeed, and status contains only intended files.

- [ ] **Step 3: Review the Brev boundary**

Confirm that repository code cannot edit an existing Launchable definition. Record the exact Brev Console changes: 75 GiB disk, remove `NEMOTRON_MODEL`, remove `JUPYTER_PORT`, retain required `NVIDIA_API_KEY`, and use the updated setup script commit.
