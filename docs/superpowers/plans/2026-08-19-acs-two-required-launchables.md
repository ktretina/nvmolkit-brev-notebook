# ACS Two Required Launchables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the ACS attendee guide so that the Notebook and Conversational Launchables are both required, Modules 1–3 form the required notebook path, and the integrated companion demo remains optional.

**Architecture:** Keep the change within the existing Markdown page and its focused contract test. Replace only the page content outside the four marked conversational prompt blocks, and preserve those blocks through their existing SHA-256 locks. Use section-scoped tests so that required status, setup fields, model roles, notebook actions, and scientific boundaries cannot be satisfied by unrelated text elsewhere on the page.

**Tech Stack:** Markdown, Python 3.12, pytest, Ruff, Git

---

### Task 1: Lock the two-required-Launchables contract in focused tests

**Files:**
- Modify: `tests/test_acs_fall_2026_workshop_page.py`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Add exact Launchable and notebook constants**

Add these constants after `PAGE`:

```python
NOTEBOOK_LAUNCHABLE_ID = "env-3HJtJW3qHg4Dw1I3xt75BfpBmZW"
CONVERSATIONAL_LAUNCHABLE_ID = "env-3Hlp4pHBlTTlfDxfH41KkGhTeCV"
NOTEBOOK_LAUNCHABLE_URL = (
    "https://brev.nvidia.com/launchable/deploy/now?launchableID="
    f"{NOTEBOOK_LAUNCHABLE_ID}"
)
CONVERSATIONAL_LAUNCHABLE_URL = (
    "https://brev.nvidia.com/launchable/deploy/now?launchableID="
    f"{CONVERSATIONAL_LAUNCHABLE_ID}"
)
NOTEBOOK_PATHS = (
    "notebooks/01_direct_nvmolkit_reframe.ipynb",
    "notebooks/02_agent_assisted_reframe_neighborhoods.ipynb",
    "notebooks/03_full_agent_reframe_panel_design.ipynb",
    "notebooks/nvmolkit_nemotron_demo.ipynb",
)
```

- [ ] **Step 2: Add a section-scoped normalization helper**

Add this helper after `_source()`:

```python
def _normalized_section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index + len(start))
    return " ".join(source[start_index:end_index].split())
```

- [ ] **Step 3: Replace the old page-order and lab-role assertions**

Update `test_page_has_the_short_attendee_order_and_current_links` so that its
ordered headings are:

```python
sections = (
    "# ACS Fall 2026 GPU chemistry workshop",
    "## Before the workshop",
    "## Complete both required labs",
    "### Required Lab 1 — nvMolKit + Nemotron Notebook",
    "### Required Lab 2 — Conversational OpenClaw",
    "## Four prompts",
    "## Download your results",
    "## Finish and remove your environments",
    "## Scientific limits",
    "## Official links",
)
```

Use `NOTEBOOK_LAUNCHABLE_URL` and `CONVERSATIONAL_LAUNCHABLE_URL` in the URL
loop instead of repeating their string literals.

Replace `test_lab_roles_hardware_and_signed_in_boundary_are_explicit` with:

```python
def test_both_launchables_are_required_in_the_fixed_order() -> None:
    source = _source()
    notebook_heading = "### Required Lab 1 — nvMolKit + Nemotron Notebook"
    conversational_heading = "### Required Lab 2 — Conversational OpenClaw"
    prework = _normalized_section(
        source,
        "## Before the workshop",
        "## Complete both required labs",
    )

    assert source.index(notebook_heading) < source.index(conversational_heading)
    assert NOTEBOOK_LAUNCHABLE_URL in source
    assert CONVERSATIONAL_LAUNCHABLE_URL in source
    assert f"`{NOTEBOOK_LAUNCHABLE_ID}`" in prework
    assert f"`{CONVERSATIONAL_LAUNCHABLE_ID}`" in prework
    assert "Optional notebook" not in source
    assert "Optional instructor-led companion" not in source
    assert "not required for the hands-on workshop" not in source
    assert "Choose your lab" not in source
    assert "Optional integrated companion demo" in source
    assert "Stop both workshop environments" in source


def test_required_labs_keep_separate_credentials_models_and_readiness() -> None:
    source = _source()
    notebook = _normalized_section(
        source,
        "### Required Lab 1 — nvMolKit + Nemotron Notebook",
        "### Required Lab 2 — Conversational OpenClaw",
    )
    conversational = _normalized_section(
        source,
        "### Required Lab 2 — Conversational OpenClaw",
        "## Four prompts",
    )

    assert NOTEBOOK_LAUNCHABLE_URL in notebook
    assert "`NVIDIA_API_KEY`" in notebook
    assert "port 8888 Secure Link" in notebook
    assert "installed nvMolKit version" in notebook
    assert "one CUDA device" in notebook
    assert "If it reports CPU fallback, stop and ask the facilitator." in notebook
    assert "Complete Modules 1–3 in order." in notebook
    assert "required notebook path uses hosted mode" in notebook
    assert "zero hosted model calls" in notebook
    assert "`nvidia/nemotron-3-nano-30b-a3b`" in notebook

    assert CONVERSATIONAL_LAUNCHABLE_URL in conversational
    assert "`NVIDIA_INFERENCE_API_KEY`" in conversational
    assert "Nemotron 3 Super 120B-A12B" in conversational
    assert "one NVIDIA L4" in conversational
    assert "x86-64" in conversational
    assert "4 CPUs" in conversational
    assert "16 GiB RAM" in conversational
    assert "128 GiB disk" in conversational
    assert "Wait until setup is ready" in conversational
    assert "Open Chemistry Agent" in conversational
    assert "create one new session" in conversational
```

- [ ] **Step 4: Add exact notebook-content and boundary tests**

Add these tests before `test_marked_prompt_blocks_are_byte_locked`:

```python
def test_required_notebook_path_matches_the_current_four_notebooks() -> None:
    source = _source()
    notebook = _normalized_section(
        source,
        "### Required Lab 1 — nvMolKit + Nemotron Notebook",
        "### Required Lab 2 — Conversational OpenClaw",
    )

    positions = [notebook.index(path) for path in NOTEBOOK_PATHS]
    assert positions == sorted(positions)
    assert "Module 1 uses no LLM" in notebook
    assert "GPU Morgan fingerprints" in notebook
    assert "fused Butina clustering" in notebook
    assert "60-row neighborhood atlas" in notebook
    assert "two bounded failure policies" in notebook
    assert "Python renders, validates, binds, and executes" in notebook
    assert "24 compounds from the fixed 96-row ReFRAME snapshot" in notebook
    assert "Review both allow-listed strategies" in notebook
    assert "Approve Plan & Run Agent" in notebook
    assert "rerun Steps 5 and 6" in notebook
    assert "six approved stages" in notebook
    assert "objective challenge" in notebook


def test_optional_activities_do_not_make_either_launchable_optional() -> None:
    source = _source()
    notebook = _normalized_section(
        source,
        "### Required Lab 1 — nvMolKit + Nemotron Notebook",
        "### Required Lab 2 — Conversational OpenClaw",
    )
    conversational = _normalized_section(
        source,
        "### Required Lab 2 — Conversational OpenClaw",
        "## Four prompts",
    )

    assert "Optional advanced run" in notebook
    assert "10,000-row" in notebook
    assert "Optional integrated companion demo" in notebook
    assert "optional exploration" in conversational
    assert "Required Lab 1" in source
    assert "Required Lab 2" in source


def test_notebook_roles_limits_and_measurements_are_section_bound() -> None:
    source = _source()
    intro = " ".join(source[: source.index("## Before the workshop")].split())
    limits = _normalized_section(source, "## Scientific limits", "## Official links")

    assert (
        "Nemotron plans and selects within validated choices. Python validates "
        "and executes deterministic chemistry. The [nvMolKit library](https://"
        "github.com/NVIDIA-BioNeMo/nvMolKit) performs the configured GPU molecular "
        "operations. RDKit supports input handling, descriptors, CPU reference "
        "work, and visualization."
    ) in intro
    assert "deterministic 96-row ReFRAME snapshot" in limits
    assert "24-compound panel" in limits
    assert "run-specific timing and throughput" in limits
    assert "not a general acceleration or speedup claim" in limits
    for unsupported in (
        "binding",
        "activity",
        "ADMET",
        "efficacy",
        "safety",
        "synthesizability",
        "clinical value",
        "experimental structure",
    ):
        assert unsupported in limits
```

- [ ] **Step 5: Replace the existing agentic-role test with section-scoped assertions**

Replace `test_agentic_ai_sandbox_and_chemistry_roles_are_clear` with:

```python
def test_agentic_ai_sandbox_and_chemistry_roles_are_clear() -> None:
    source = _source()
    intro = " ".join(source[: source.index("## Before the workshop")].split())
    conversational = _normalized_section(
        source,
        "### Required Lab 2 — Conversational OpenClaw",
        "## Four prompts",
    )
    official_links = source[source.index("## Official links") :]

    assert "bounded agentic AI workflow for chemistry" in intro
    assert "follows the workflow pattern presented by the" in intro
    assert (
        "[NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/"
        "bionemo-agent-toolkit)" in intro
    )
    assert (
        "Nemotron plans and selects within validated choices. Python validates "
        "and executes deterministic chemistry. The [nvMolKit library](https://"
        "github.com/NVIDIA-BioNeMo/nvMolKit) performs the configured GPU molecular "
        "operations. RDKit supports input handling, descriptors, CPU reference "
        "work, and visualization."
    ) in intro
    assert "components from" not in intro
    assert "It is not an unrestricted or fully autonomous AI scientist." in intro

    assert "sandboxed conversational workspace" in conversational
    assert "configured agentic chemistry analyses" in conversational
    assert "four preset prompts below are tested starting points" in conversational
    assert (
        "change the questions and the requested interpretation about these analyses"
        in conversational
    )
    assert "approved tools, fixed data, and configured" in conversational
    assert "https://github.com/NVIDIA-BioNeMo/nvMolKit" in conversational
    assert "directions that interest you" not in conversational

    assert "https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit" in official_links
    assert "https://github.com/NVIDIA-BioNeMo/nvMolKit" in official_links
```

In `test_download_and_cleanup_are_attendee_actions`, replace:

```python
assert "Stop every workshop environment you started" in source
```

with:

```python
assert "Stop both workshop environments" in source
```

- [ ] **Step 6: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_acs_fall_2026_workshop_page.py
```

Expected: FAIL because the page still calls the notebook optional, has the old
section headings, omits the notebook modules and readiness checks, and lacks the
notebook-specific limits. The prompt-hash test must still pass.

### Task 2: Rewrite only the attendee-guide content outside the locked prompts

**Files:**
- Modify: `docs/acs-fall-2026-workshop.md`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Replace the introduction through the pre-prompt instructions**

Replace the content after the title and before `## Four prompts` with this
Markdown. Do not alter the title or any marked prompt block.

```markdown
This workshop uses two required, complementary agentic chemistry environments
to demonstrate a bounded agentic AI workflow for chemistry. It follows the
workflow pattern presented by the
[NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit):
a model plans and selects within validated choices, and approved code executes
the chemistry. It is not an unrestricted or fully autonomous AI scientist.

Nemotron plans and selects within validated choices. Python validates and
executes deterministic chemistry. The
[nvMolKit library](https://github.com/NVIDIA-BioNeMo/nvMolKit) performs the
configured GPU molecular operations. RDKit supports input handling,
descriptors, CPU reference work, and visualization. The Notebook Launchable
teaches this pattern through three guided modules. The Conversational
Launchable provides a separate sandboxed OpenClaw experience with four tested
prompts.

## Before the workshop

Complete these steps before you arrive:

1. Create [one NVIDIA account](https://account.nvidia.com/), verify your email,
   and complete an NVIDIA Cloud Account if prompted.
2. Sign in to [NVIDIA Brev](https://brev.nvidia.com/). Complete its onboarding
   and make sure your organization has Brev credits or a payment method.
3. Open the [NVIDIA API-key page](https://build.nvidia.com/settings/api-keys).
   Generate and copy one API key. Complete phone verification if requested.

Hosted prototype access can be rate-limited. Brev GPU compute is separate and
billable. Never paste the key into chat, a screenshot, or a file. Use the same
private API-key value in both required setup fields:

| Required lab | Launchable ID | Setup field | Model role |
| --- | --- | --- | --- |
| Lab 1 — nvMolKit + Nemotron Notebook | `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` | `NVIDIA_API_KEY` | Module 1 uses no LLM. In hosted mode, Modules 2–3 and the companion use `nvidia/nemotron-3-nano-30b-a3b`. |
| Lab 2 — Conversational OpenClaw | `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV` | `NVIDIA_INFERENCE_API_KEY` | The four-prompt path uses [NVIDIA Nemotron 3 Super 120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted). |

## Complete both required labs

Complete Lab 1 first and Lab 2 second. A saved Launchable or an older successful
deployment does not prove that a new environment is ready.

### Required Lab 1 — nvMolKit + Nemotron Notebook

[Deploy nvMolKit + Nemotron Notebook](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW).

1. Enter the API key in `NVIDIA_API_KEY`, then deploy.
2. Wait until setup finishes. Open JupyterLab through the port 8888 Secure Link.
3. Open Module 1 and run its initialization cell. Confirm that the output shows
   the installed nvMolKit version and one CUDA device. If it reports CPU
   fallback, stop and ask the facilitator.
4. Complete Modules 1–3 in order.

The required notebook path uses hosted mode for Modules 2–3. Reference mode is
an instructor-directed recovery path. It makes zero hosted model calls and is
not evidence that Nemotron ran. In hosted mode, Modules 2–3 use
`nvidia/nemotron-3-nano-30b-a3b`.

#### Module 1 — Direct nvMolKit

Open `notebooks/01_direct_nvmolkit_reframe.ipynb`. Module 1 uses no LLM. Run
direct GPU Morgan fingerprints, Tanimoto similarity, and fused Butina
clustering. Compare the bounded settings and clearly labeled RDKit CPU
reference work.

**Optional advanced run:** Keep the 10,000-row path off unless the instructor
asks you to run it. Timing and throughput are observations from the exact
hardware, input, and parameters in your run, not general speedup claims.

#### Module 2 — Agent-assisted neighborhoods

Open `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`. Nemotron
selects two bounded failure policies; it does not write executable code. Python
renders, validates, binds, and executes the allow-listed implementation. Review
the 60-row neighborhood atlas and the representation-sensitivity results.

#### Module 3 — Agent-guided panel design

Open `notebooks/03_full_agent_reframe_panel_design.ipynb`. Nemotron proposes a
bounded plan and later audits the result. Review both allow-listed strategies,
select one, and click **Approve Plan & Run Agent**. The validated analysis
selects 24 compounds from the fixed 96-row ReFRAME snapshot. When it finishes,
rerun Steps 5 and 6 to display the receipt and three-column gallery.

#### Optional integrated companion demo

Open `notebooks/nvmolkit_nemotron_demo.ipynb` only if time permits or the
instructor requests it. It combines six approved stages, a bounded objective
challenge, and an evidence-backed conclusion. It is not part of required
completion.

### Required Lab 2 — Conversational OpenClaw

[Deploy the conversational OpenClaw Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3Hlp4pHBlTTlfDxfH41KkGhTeCV).

This required lab uses Nemotron 3 Super 120B-A12B.

Use this sandboxed conversational workspace to explore the configured agentic
chemistry analyses. The four preset prompts below are tested starting points.
You can change the questions and the requested interpretation about these
analyses, while the sandbox keeps execution within the approved tools, fixed
data, and configured
[nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) capabilities.

1. Use the default hardware. Confirm that the visible row shows one NVIDIA L4,
   x86-64, 4 CPUs, 16 GiB RAM, and 128 GiB disk. The row does not need to show
   `g6.xlarge`.
2. Enter the API key in `NVIDIA_INFERENCE_API_KEY`, then deploy.
3. Wait until setup is ready. Open **Open Chemistry Agent** and create one new
   session.
4. Paste the four prompts below unchanged and in order into that same session.
   Wait for each answer before sending the next prompt.

The installed `nvmolkit-usage` skill remains available for optional exploration
after the tested exercise. It describes the supported
[nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) functions in this
environment. The four tested prompts do not read it.

If an LLM request times out, start a new session and retry the whole prompt
once. Do not retry individual commands. After a second timeout, ask the
facilitator.
```

- [ ] **Step 2: Update cleanup for exactly two required environments**

Replace the first sentence under `## Finish and remove your environments` with:

```markdown
Stop both workshop environments as soon as the exercise ends.
```

Keep the existing delete, storage-cost, permanence, and download-first copy.

- [ ] **Step 3: Add notebook-specific scientific limits**

Insert these bullets at the start of `## Scientific limits`, before the
existing 256-record ChEMBL limits:

```markdown
- The notebook modules use a deterministic 96-row ReFRAME snapshot, not representative chemical space.
- The 24-compound panel optimizes a bounded structural-fingerprint and descriptor-coverage objective; it is not a globally optimal or experimentally validated panel.
- The neighborhood atlas depends on its selected Morgan radii and hashed fingerprint length.
- Module 1 reports run-specific timing and throughput for the visible hardware, inputs, and parameters; this is not a general acceleration or speedup claim.
```

Replace the one paragraph that starts with `These computations describe` with:

```markdown
These computations describe molecular structures, structural similarity,
diversity, clustering, and sampled force-field geometries. They do not prove
identity, binding, activity, ADMET, efficacy, safety, synthesizability,
clinical value, or experimental structure.
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_acs_fall_2026_workshop_page.py
```

Expected: all tests pass, including all four unchanged prompt hashes.

### Task 3: Verify scope and create the implementation commit

**Files:**
- Verify: `docs/acs-fall-2026-workshop.md`
- Verify: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Run focused formatting and static checks**

Run:

```bash
python3 -m ruff format --check tests/test_acs_fall_2026_workshop_page.py
python3 -m ruff check tests/test_acs_fall_2026_workshop_page.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Prove that the four prompt blocks are byte-identical**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_acs_fall_2026_workshop_page.py::test_marked_prompt_blocks_are_byte_locked
```

Expected: `1 passed`.

- [ ] **Step 3: Confirm the exact implementation scope**

Run:

```bash
git status --short
git diff --name-only HEAD
```

Expected: only these two implementation files are modified:

```text
docs/acs-fall-2026-workshop.md
tests/test_acs_fall_2026_workshop_page.py
```

- [ ] **Step 4: Review the final diff against the specification**

Run:

```bash
git diff -- docs/acs-fall-2026-workshop.md tests/test_acs_fall_2026_workshop_page.py
```

Confirm all of the following before staging:

```text
Both Launchables are required and ordered Notebook first.
Modules 1-3 are required; only the companion and advanced activities are optional.
Nano and Super model roles and API-key fields are distinct.
The notebook readiness check uses visible Module 1 output.
The four conversational prompt blocks have no diff.
Notebook and conversational scientific limits remain bounded.
No Launchable, notebook, runtime, setup, chemistry, or Brev file changed.
```

- [ ] **Step 5: Commit only the two implementation files**

```bash
git add docs/acs-fall-2026-workshop.md tests/test_acs_fall_2026_workshop_page.py
git commit -m "docs: require both ACS workshop launchables"
```

- [ ] **Step 6: Run post-commit verification**

Run:

```bash
git status --short
git show --stat --oneline HEAD
git diff HEAD^ HEAD --check
```

Expected: the worktree is clean; the implementation commit contains exactly the
two approved files; the range diff check exits zero.

### Task 4: Independent review gates

**Files:**
- Review: `docs/acs-fall-2026-workshop.md`
- Review: `tests/test_acs_fall_2026_workshop_page.py`
- Reference: `docs/superpowers/specs/2026-08-19-acs-two-required-launchables-design.md`

- [ ] **Step 1: Request an independent specification review**

Ask a fresh reviewer to compare the implementation commit with the design
specification. Require the reviewer to confirm the two required labs, fixed
order, notebook content, model and key separation, readiness checks, optional
companion boundary, scientific limits, exact two-file scope, and prompt hashes.
The reviewer must report Critical and Important findings only.

- [ ] **Step 2: Repair any confirmed specification finding with RED/GREEN evidence**

For each confirmed finding, add or strengthen one focused regression test,
witness its failure, make the smallest page or test repair, and rerun the full
focused test file. Amend only the implementation commit.

- [ ] **Step 3: Request an independent quality review**

Ask a different fresh reviewer to check misleading wording, unsupported product
or scientific claims, section-local test strength, prompt integrity, stale
deployment claims, readability, and scope. The reviewer must report Critical
and Important findings only.

- [ ] **Step 4: Repair confirmed quality findings and run final gates**

Use the same focused RED/GREEN rule for each confirmed finding. Then rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_acs_fall_2026_workshop_page.py
python3 -m ruff format --check tests/test_acs_fall_2026_workshop_page.py
python3 -m ruff check tests/test_acs_fall_2026_workshop_page.py
git diff HEAD^ HEAD --check
git status --short
```

Expected: all tests and static checks pass, both independent reviewers return
READY, and the worktree is clean. Do not push, merge, deploy, or change either
Launchable without a separate user approval.
