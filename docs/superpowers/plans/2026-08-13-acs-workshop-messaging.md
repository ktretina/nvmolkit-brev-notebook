# ACS Workshop Messaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the attendee guide as a concise, accurate demonstration of agentic AI and the BioNeMo Agent Toolkit pattern for chemistry, while clearly explaining sandboxed exploration and nvMolKit's role.

**Architecture:** Change only the canonical Markdown guide and its focused contract tests. Keep the four live-accepted prompt blocks byte-identical, add exact message and link assertions first, then apply the minimum prose changes outside the prompt markers.

**Tech Stack:** Markdown, Python 3.12, pytest, Ruff, SHA-256 prompt locks

---

## File map

- Modify `docs/acs-fall-2026-workshop.md`: attendee-facing introduction,
  conversational Launchable description, tested-path clarification, nvMolKit
  skill explanation, and official resources.
- Modify `tests/test_acs_fall_2026_workshop_page.py`: message, link, role, and
  byte-identical prompt checks.

### Task 1: Lock the approved message and prompt boundaries

**Files:**
- Modify: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Add the failing message and resource assertions**

Add `import hashlib` beside the existing imports. Add both repository URLs to
the URL tuple in `test_page_has_the_short_attendee_order_and_current_links`:

```python
"https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit",
"https://github.com/NVIDIA-BioNeMo/nvMolKit",
```

Add this focused test:

```python
def test_agentic_ai_sandbox_and_chemistry_roles_are_clear() -> None:
    source = _source()
    intro = " ".join(source[: source.index("## Before the workshop")].split())
    required_lab = " ".join(
        source[
            source.index("**Required hands-on lab:**") : source.index(
                "As of August 11, 2026"
            )
        ].split()
    )
    official_links = source[source.index("## Official links") :]

    assert "bounded agentic AI workflow for chemistry" in intro
    assert "follows the pattern presented by the" in intro
    assert (
        "[NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/"
        "bionemo-agent-toolkit)" in intro
    )
    assert (
        "a model plans and reasons, approved tools execute the work, and validation "
        "keeps the results grounded" in intro
    )
    assert (
        "hosted NVIDIA Nemotron reasons about chemistry questions, and OpenClaw "
        "coordinates approved tools inside an OpenShell sandbox" in intro
    )
    assert "components from" not in intro

    for role in (
        "GPU Morgan fingerprints",
        "Tanimoto similarity",
        "ETKDG conformer generation",
        "MMFF94 optimization",
        "CPU Butina clustering",
    ):
        assert role in intro

    assert "sandboxed conversational workspace" in required_lab
    assert "configured agentic chemistry analyses" in required_lab
    assert "four preset prompts below are tested starting points" in required_lab
    assert (
        "change the questions and the requested interpretation about these analyses"
        in required_lab
    )
    assert "approved tools, fixed data, and configured" in required_lab
    assert "https://github.com/NVIDIA-BioNeMo/nvMolKit" in required_lab
    assert "directions that interest you" not in required_lab

    assert "https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit" in official_links
    assert "https://github.com/NVIDIA-BioNeMo/nvMolKit" in official_links
```

- [ ] **Step 2: Lock the four accepted prompt bytes**

Add the accepted prompt digests:

```python
PROMPT_SHA256 = {
    "01-data-and-representation": (
        "ccea479eb0762db9adb25f7fcc3e4a60758400f55646714ec8489ad2d474e482"
    ),
    "02-relationships-and-groups": (
        "048c34ac064ee30dce7df1be1ec37a9e6ebc002d552cf21bef67401325e40ee4"
    ),
    "03-sampled-3d-geometry": (
        "357706bafd1eb73e852bc72da419c37dd0f1a5f6d234edf12e24df104ad2e724"
    ),
    "04-objective": (
        "00b83de39a40a93344749b1a379537285f7b502fb38050cb7824cac774727f75"
    ),
}
```

Add this test after `_prompt_blocks`:

```python
def test_live_accepted_prompt_bytes_do_not_change() -> None:
    blocks = _prompt_blocks(_source())

    assert {
        prompt_id: hashlib.sha256(block.encode("utf-8")).hexdigest()
        for prompt_id, block in blocks.items()
    } == PROMPT_SHA256
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
python3 -m pytest -q tests/test_acs_fall_2026_workshop_page.py
```

Expected: the new messaging test and the two new required URL assertions fail;
the prompt digest test passes.

### Task 2: Apply the minimum attendee-guide copy update

**Files:**
- Modify: `docs/acs-fall-2026-workshop.md`
- Test: `tests/test_acs_fall_2026_workshop_page.py`

- [ ] **Step 1: Replace only the introduction before `## Before the workshop`**

Use this exact copy:

```markdown
This workshop demonstrates a bounded agentic AI workflow for chemistry. It
follows the pattern presented by the
[NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit):
a model plans and reasons, approved tools execute the work, and validation keeps
the results grounded. Here, hosted NVIDIA Nemotron reasons about chemistry
questions, and OpenClaw coordinates approved tools inside an OpenShell sandbox.

The [nvMolKit library](https://github.com/NVIDIA-BioNeMo/nvMolKit) runs GPU
Morgan fingerprints, Tanimoto similarity, ETKDG conformer generation, and
MMFF94 optimization on one NVIDIA L4. RDKit supports input handling,
visualization, and CPU Butina clustering. No local coding setup is required.
```

- [ ] **Step 2: Replace only the required hands-on lab description**

Keep the existing Launchable link and use this exact copy after it:

```markdown
Use this sandboxed conversational workspace to explore the configured agentic
chemistry analyses. The four preset prompts below are tested starting points.
You can change the questions and the requested interpretation about these
analyses, while the sandbox keeps execution within the approved tools, fixed
data, and configured
[nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) capabilities.
```

- [ ] **Step 3: Clarify the tested path and optional skill**

Change launch step 4 to:

```markdown
4. For the tested workshop path, paste the four prompts below unchanged and in
   order into that same session. Wait for each answer before sending the next
   prompt.
```

Replace the current skill note with:

```markdown
The installed `nvmolkit-usage` skill remains available for optional exploration
after the tested exercise. It describes the supported
[nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) functions in this
environment. The four tested prompts do not read it.
```

- [ ] **Step 4: Add the two official resources**

Add these entries immediately after the workshop repository entry:

```markdown
- [NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit)
- [NVIDIA nvMolKit library](https://github.com/NVIDIA-BioNeMo/nvMolKit)
```

- [ ] **Step 5: Run the focused page tests and confirm GREEN**

Run:

```bash
python3 -m pytest -q tests/test_acs_fall_2026_workshop_page.py
```

Expected: all attendee-page tests pass, including the four prompt digests.

- [ ] **Step 6: Run the final narrow gates**

Run:

```bash
ruff check tests/test_acs_fall_2026_workshop_page.py
ruff format --check tests/test_acs_fall_2026_workshop_page.py
git diff --check
git status --short
```

Expected: Ruff and diff checks pass. Status shows only the guide, its focused
test, and this implementation plan as local work relative to the prior design
commit.

- [ ] **Step 7: Review and commit the implementation**

Confirm that the diff does not touch any text between an
`ACS_PROMPT:*:BEGIN` marker and its matching `ACS_PROMPT:*:END` marker. Then
commit only the plan, guide, and focused test:

```bash
git add \
  docs/superpowers/plans/2026-08-13-acs-workshop-messaging.md \
  docs/acs-fall-2026-workshop.md \
  tests/test_acs_fall_2026_workshop_page.py
git diff --cached --check
git commit -m "Update ACS workshop agentic AI messaging"
```

Do not push, redeploy, or change the Launchable.
