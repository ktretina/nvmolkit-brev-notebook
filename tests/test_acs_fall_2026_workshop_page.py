from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "acs-fall-2026-workshop.md"

PROMPT_IDS = (
    "01-data-and-representation",
    "02-relationships-and-groups",
    "03-sampled-3d-geometry",
    "04-objective",
)
RUNNER = (
    "env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 "
    "/sandbox/.openclaw/workspace/acs_workshop_runner.py"
)
LESSON_COMMANDS = {
    "01-data-and-representation": f"{RUNNER} run-lesson data-and-representation",
    "02-relationships-and-groups": f"{RUNNER} run-lesson relationships-and-groups",
    "03-sampled-3d-geometry": f"{RUNNER} run-lesson sampled-3d-geometry",
}
OBJECTIVE_START = f"{RUNNER} objective-start"
OBJECTIVE_STEP = (
    f"{RUNNER} objective-step --state-id 'STATE_ID_FROM_MENU' "
    "--swap-id 'SWAP_ID_FROM_MENU'"
)
MEDIA_LINES = {
    "01-data-and-representation": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "01-inspection/library_preview.png"
    ),
    "02-relationships-and-groups": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "04-clusters/cluster_sizes.png"
    ),
    "03-sampled-3d-geometry": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "06-mmff94/optimized_structures.png"
    ),
    "04-objective": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "07-objective/final_panel.png"
    ),
}
RESPONSE_HEADINGS = (
    "Question",
    "What ran",
    "Measured result",
    "Meaning",
    "Scientific limit",
    "Image and download location",
)

REQUIRED_SECTIONS = (
    "## Before the workshop",
    "## Complete both required labs",
    "## Required Lab 1 — nvMolKit + Nemotron Notebook",
    "## Required Lab 2 — Conversational OpenClaw",
    "## Four prompts",
    "## Download",
    "## Finish",
    "## Scientific limits",
    "## Official links",
)

APPROVED_MODULE_1_TIMING_SENTENCE = (
    "Any Module 1 timing or throughput is an observation from the exact Notebook "
    "hardware, inputs, parameters, and the attendee run. It is not a general "
    "acceleration or speedup claim."
)
MODULE_1_START_HEADING = "### Module 1 — Direct nvMolKit ReFRAME analysis"
MODULE_2_START_HEADING = "### Module 2 — Agent-assisted ReFRAME neighborhoods"
MODULE_1_SUBSECTION_SHA256 = (
    "a12d68a8e4f9be649922c1246db8e1902bee9673312e094fe512b89d953c7cd7"
)
PERFORMANCE_VOCABULARY = re.compile(
    r"\b(?:accelerat\w*|benchmark\w*|fast\w*|latenc(?:y|ies)|outperform\w*|"
    r"performance\w*|speed\w*|throughput\w*|timing\w*)\b",
    flags=re.IGNORECASE,
)
NUMERIC_COMPARATIVE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\b|times?\b|-?\s*fold\b)",
    flags=re.IGNORECASE,
)


def _source() -> str:
    assert PAGE.is_file(), "the canonical ACS attendee page is missing"
    return PAGE.read_text(encoding="utf-8")


def _section(source: str, heading: str) -> str:
    marker = f"## {heading}"
    assert marker in source, f"missing section: {marker}"
    start = source.index(marker)
    next_heading = source.find("\n## ", start + len(marker))
    end = len(source) if next_heading == -1 else next_heading
    return source[start:end]


def _subsection(section: str, heading: str, *, level: int = 3) -> str:
    prefix = "#" * level
    marker = f"{prefix} {heading}"
    assert marker in section, f"missing subsection: {marker}"
    start = section.index(marker)
    next_heading = section.find(f"\n{prefix} ", start + len(marker))
    end = len(section) if next_heading == -1 else next_heading
    return section[start:end]


def _between_markers(source: str, start_marker: str, end_marker: str) -> str:
    assert start_marker in source, f"missing start marker: {start_marker}"
    assert end_marker in source, f"missing end marker: {end_marker}"
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _raw_module_1_subsection(source: str) -> str:
    return _between_markers(
        source,
        MODULE_1_START_HEADING,
        MODULE_2_START_HEADING,
    )


def _assert_approved_module_1_subsection(module_1: str) -> None:
    actual_hash = hashlib.sha256(module_1.encode("utf-8")).hexdigest()
    assert actual_hash == MODULE_1_SUBSECTION_SHA256


def _assert_only_approved_module_1_performance_language(module_1: str) -> None:
    assert module_1.count(APPROVED_MODULE_1_TIMING_SENTENCE) == 1
    remainder = module_1.replace(APPROVED_MODULE_1_TIMING_SENTENCE, "", 1)
    assert PERFORMANCE_VOCABULARY.search(remainder) is None
    assert NUMERIC_COMPARATIVE.search(remainder) is None


def _prompt_blocks(source: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    last_end = -1
    for prompt_id in PROMPT_IDS:
        begin = f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->"
        end = f"<!-- ACS_PROMPT:{prompt_id}:END -->"
        assert source.count(begin) == 1
        assert source.count(end) == 1
        begin_index = source.index(begin)
        end_index = source.index(end)
        assert last_end < begin_index < end_index
        marked = source[begin_index + len(begin) : end_index]
        fences = re.findall(r"~~~text\n(.*?)\n~~~", marked, flags=re.DOTALL)
        assert len(fences) == 1
        blocks[prompt_id] = fences[0]
        last_end = end_index
    assert source.count("<!-- ACS_PROMPT:") == 8
    return blocks


def _full_marked_prompt_blocks(source: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    last_end = -1
    for prompt_id in PROMPT_IDS:
        begin = f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->"
        end = f"<!-- ACS_PROMPT:{prompt_id}:END -->"
        assert source.count(begin) == 1
        assert source.count(end) == 1
        begin_index = source.index(begin)
        end_index = source.index(end, begin_index)
        assert last_end < begin_index < end_index
        block_end = end_index + len(end)
        blocks[prompt_id] = source[begin_index:block_end]
        last_end = block_end
    assert source.count("<!-- ACS_PROMPT:") == 8
    return blocks


def test_both_launchables_are_required_and_the_notebook_is_first() -> None:
    source = _source()

    for removed_wording in (
        "Optional notebook",
        "Optional instructor-led companion",
        "not required for the hands-on workshop",
        "Choose your lab",
        "As of August 11, 2026",
        "public HTTP checks",
    ):
        assert removed_wording not in source

    headings = tuple(re.findall(r"^## .+$", source, flags=re.MULTILINE))
    assert headings == REQUIRED_SECTIONS

    lab_1_url = (
        "https://brev.nvidia.com/launchable/deploy/now?launchableID="
        "env-3HJtJW3qHg4Dw1I3xt75BfpBmZW"
    )
    lab_2_url = (
        "https://brev.nvidia.com/launchable/deploy/now?launchableID="
        "env-3Hlp4pHBlTTlfDxfH41KkGhTeCV"
    )
    lab_1 = _section(source, "Required Lab 1 — nvMolKit + Nemotron Notebook")
    lab_2 = _section(source, "Required Lab 2 — Conversational OpenClaw")

    assert lab_1_url in lab_1
    assert "env-3HJtJW3qHg4Dw1I3xt75BfpBmZW" in lab_1
    assert lab_2_url in lab_2
    assert "env-3Hlp4pHBlTTlfDxfH41KkGhTeCV" in lab_2
    assert source.index(lab_1_url) < source.index(lab_2_url)


def test_required_labs_overview_has_the_numbered_six_step_workshop_path() -> None:
    source = _source()
    overview = _between_markers(
        source,
        "## Complete both required labs",
        "## Required Lab 1 — nvMolKit + Nemotron Notebook",
    )
    compact_overview = " ".join(overview.split())
    actions = (
        "Deploy and open **nvMolKit + Nemotron Notebook**.",
        "Complete numbered notebook Modules 1–3 in order.",
        "Optionally run the integrated companion demo in the same Notebook Launchable.",
        "Deploy and open the **Conversational OpenClaw Launchable**.",
        "Complete the four existing conversational prompts in order.",
        "Stop both Brev environments after the workshop.",
    )

    numbered_actions = tuple(
        compact_overview.index(f"{number}. {action}")
        for number, action in enumerate(actions, start=1)
    )
    assert numbered_actions == tuple(sorted(numbered_actions))
    assert tuple(re.findall(r"(?m)^(\d+)\. ", overview)) == tuple(
        str(number) for number in range(1, 7)
    )


def test_page_has_current_account_repository_and_official_links() -> None:
    source = _source()

    for url in (
        "https://brev.nvidia.com/",
        "https://account.nvidia.com/",
        "https://build.nvidia.com/settings/api-keys",
        "https://github.com/ktretina/nvmolkit-brev-notebook",
        "https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit",
        "https://github.com/NVIDIA-BioNeMo/nvMolKit",
        "https://docs.nvidia.com/brev/getting-started/quickstart",
        "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/"
        "integrations/nvidia-integrations.html",
        "https://docs.nvidia.com/brev/concepts/gpu-instances",
    ):
        assert url in source


def test_agentic_ai_sandbox_and_chemistry_roles_are_clear() -> None:
    source = _source()
    intro = " ".join(source[: source.index("## Before the workshop")].split())
    required_lab = " ".join(
        _section(source, "Required Lab 2 — Conversational OpenClaw").split()
    )
    official_links = source[source.index("## Official links") :]

    assert "bounded BioNeMo Agent Toolkit workflow pattern" in intro
    assert (
        "[NVIDIA BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/"
        "bionemo-agent-toolkit)" in intro
    )
    assert "not an unrestricted or fully autonomous AI scientist" in intro
    for required_statement in (
        "There are two required, complementary agentic chemistry environments.",
        "The nvMolKit + Nemotron Notebook teaches direct nvMolKit and bounded "
        "Nemotron through three guided modules.",
        "Conversational OpenClaw provides a separate sandboxed OpenClaw experience "
        "with four tested prompts.",
    ):
        assert required_statement in intro
    approved_role_statement = (
        "Nemotron plans and selects within validated choices. Python validates and "
        "executes deterministic chemistry. nvMolKit performs the configured GPU "
        "molecular operations. RDKit supports input handling, descriptors, CPU "
        "reference work, and visualization."
    )
    assert approved_role_statement in intro
    assert (
        "[NVIDIA nvMolKit library](https://github.com/NVIDIA-BioNeMo/nvMolKit)" in intro
    )

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


def test_prework_keeps_the_notebook_key_and_makes_openclaw_zero_input() -> None:
    source = _source()
    prework = _section(source, "Before the workshop")
    compact_prework = " ".join(prework.split())

    for required in (
        "one NVIDIA account",
        "verify your email",
        "phone verification",
        "NVIDIA Cloud Account",
        "Brev credits or a payment method",
        "rate-limited",
        "Brev GPU compute is separate and billable",
        "Never paste the key into chat, a screenshot, or a file",
        "`NVIDIA_API_KEY`",
        "This key is only for the nvMolKit + Nemotron Notebook",
        "No API key or Setup values",
    ):
        assert required in compact_prework

    expected_table = (
        "| Required environment | Launchable ID | Attendee setup | Model use |\n"
        "| --- | --- | --- | --- |\n"
        "| nvMolKit + Nemotron Notebook | "
        "`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` | Enter the API key in "
        "`NVIDIA_API_KEY` | "
        "Module 1: no LLM; hosted Modules 2–3 and companion: "
        "`nvidia/nemotron-3-nano-30b-a3b` |\n"
        "| Conversational OpenClaw | `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV` | "
        "No API key or Setup values | "
        "NVIDIA Nemotron 3 Super 120B-A12B |"
    )
    assert expected_table in prework
    assert "`NVIDIA_INFERENCE_API_KEY`" not in prework
    assert "Use the same private API-key value in both fields." not in prework


def test_models_are_separate_by_required_lab_and_notebook_module() -> None:
    source = _source()
    prework = _section(source, "Before the workshop")
    lab_1 = _section(source, "Required Lab 1 — nvMolKit + Nemotron Notebook")
    lab_2 = _section(source, "Required Lab 2 — Conversational OpenClaw")
    notebook_model = "`nvidia/nemotron-3-nano-30b-a3b`"
    conversational_model = "Nemotron 3 Super 120B-A12B"
    public_build_model_url = (
        "https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted"
    )

    assert "Module 1 uses no LLM" in lab_1
    assert notebook_model in lab_1
    assert "Hosted Modules 2–3" in lab_1
    assert "optional companion" in lab_1
    assert conversational_model not in lab_1
    assert conversational_model in prework
    assert conversational_model in lab_2
    assert public_build_model_url not in prework
    assert public_build_model_url not in lab_2
    assert "build.nvidia.com" not in lab_2
    assert "inference-api.nvidia.com" not in lab_2
    assert notebook_model not in lab_2


def test_notebook_readiness_is_visible_and_fails_closed_on_cpu_fallback() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")

    readiness_steps = (
        "Enter the API key in `NVIDIA_API_KEY`",
        "Wait until setup is ready",
        "Open JupyterLab through the port 8888 Secure Link",
        "Run the Module 1 initialization",
        "installed nvMolKit version",
        "one CUDA device",
    )
    positions = [lab_1.index(step) for step in readiness_steps]
    assert positions == sorted(positions)
    assert "If it reports CPU fallback, stop and ask the facilitator." in lab_1


def test_notebook_paths_and_required_module_order_are_exact() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    paths = (
        "`notebooks/01_direct_nvmolkit_reframe.ipynb`",
        "`notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`",
        "`notebooks/03_full_agent_reframe_panel_design.ipynb`",
        "`notebooks/nvmolkit_nemotron_demo.ipynb`",
    )
    positions = [lab_1.index(path) for path in paths]

    assert positions == sorted(positions)
    assert "Complete Modules 1–3 in order." in lab_1
    assert "Hosted mode is required for Modules 2–3" in lab_1
    assert "Reference mode is instructor-directed recovery" in lab_1
    assert "makes zero hosted model calls" in lab_1
    assert "is not evidence that Nemotron ran" in lab_1


def test_notebook_module_1_actions_and_timing_are_bounded() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    module_1 = _subsection(lab_1, "Module 1 — Direct nvMolKit ReFRAME analysis")
    optional_marker = "**Optional advanced run:**"

    for required in (
        "Module 1 uses no LLM",
        "Direct GPU Morgan fingerprints",
        "Tanimoto similarity",
        "fused Butina clustering",
        "labeled RDKit CPU reference work",
        APPROVED_MODULE_1_TIMING_SENTENCE,
        optional_marker,
        "10,000-row",
        "only when instructed",
        "Compare bounded fingerprint radius, bit length, sample size, and Butina "
        "cutoff.",
        "Compare the GPU results with clearly labeled RDKit CPU reference work.",
    ):
        assert required in module_1

    assert module_1.index(APPROVED_MODULE_1_TIMING_SENTENCE) < module_1.index(
        optional_marker
    )


def test_notebook_module_1_uses_only_approved_performance_language() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    module_1 = _subsection(lab_1, "Module 1 — Direct nvMolKit ReFRAME analysis")
    _assert_only_approved_module_1_performance_language(module_1)


def test_notebook_module_1_subsection_is_byte_locked() -> None:
    module_1 = _raw_module_1_subsection(_source())

    assert module_1.startswith(MODULE_1_START_HEADING)
    assert MODULE_2_START_HEADING not in module_1
    _assert_approved_module_1_subsection(module_1)


def test_notebook_module_1_hash_lock_rejects_arbitrary_performance_prose() -> None:
    module_1 = _raw_module_1_subsection(_source())
    claim = "nvMolKit completes this work in half the time RDKit needs."
    anchor = "\n\n**Optional advanced run:**"
    assert module_1.count(anchor) == 1
    mutated = module_1.replace(anchor, f"\n\n{claim}{anchor}", 1)
    assert mutated != module_1
    assert (
        hashlib.sha256(mutated.encode("utf-8")).hexdigest()
        != MODULE_1_SUBSECTION_SHA256
    )

    with pytest.raises(AssertionError):
        _assert_approved_module_1_subsection(mutated)


@pytest.mark.parametrize(
    "claim",
    (
        "nvMolKit is 100 times faster than RDKit.",
        "universal 100-fold acceleration",
        "consistently outperforms RDKit",
        "lower latency",
        "higher throughput",
        "a 100x improvement",
        "a 100 times gain",
        "a 100-fold gain",
        "a 100 fold gain",
        "a 2.5x improvement",
        "a 2.5 times gain",
    ),
)
def test_notebook_module_1_rejects_unapproved_performance_language(
    claim: str,
) -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    module_1 = _subsection(lab_1, "Module 1 — Direct nvMolKit ReFRAME analysis")

    with pytest.raises(AssertionError):
        _assert_only_approved_module_1_performance_language(f"{module_1}\n{claim}")


def test_optional_companion_is_bounded_within_the_required_notebook() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    companion = _subsection(lab_1, "Optional integrated companion demo", level=4)

    assert "only within the required notebook environment" in companion
    assert "six approved stages" in companion
    assert "bounded objective challenge" in companion
    assert "evidence-backed conclusion" in companion
    assert "It is not required completion." in companion


def test_notebook_module_2_and_3_attendee_actions_are_explicit() -> None:
    lab_1 = _section(_source(), "Required Lab 1 — nvMolKit + Nemotron Notebook")
    module_3 = _between_markers(
        lab_1,
        "#### Module 3 — Agent-guided panel design",
        "#### Optional integrated companion demo",
    )
    assert (
        re.search(
            r"\battendee\b[^.]*\bvalidat(?:e|es|ed|ing)\b",
            module_3,
            flags=re.IGNORECASE,
        )
        is None
    )

    for module_2_requirement in (
        "Nemotron selects two bounded failure policies",
        "does not write executable code",
        "Python renders, validates, binds, and executes the allow-listed "
        "implementation",
        "Review the 60-row atlas and representation sensitivity",
    ):
        assert module_2_requirement in lab_1

    for module_3_requirement in (
        "Nemotron gives a bounded plan and audit",
        "The attendee reviews both bounded, allow-listed strategies",
        "selects one",
        "clicks **Approve Plan & Run Agent** to approve execution",
        "Python executes the selected strategy and independently validates the "
        "resulting 24-of-96 panel and its artifacts",
        "fixed 96-row ReFRAME snapshot",
        "The attendee then reruns Steps 5 and 6 for the receipt and gallery",
    ):
        assert module_3_requirement in module_3


def test_conversational_lab_keeps_hardware_session_and_recovery_contract() -> None:
    lab_2 = _section(_source(), "Required Lab 2 — Conversational OpenClaw")
    compact_lab_2 = " ".join(lab_2.split())

    for required in (
        "sandboxed conversational workspace",
        "one NVIDIA L4",
        "x86-64",
        "4 CPUs",
        "16 GiB RAM",
        "128 GiB disk",
        "does not need to show `g6.xlarge`",
        "There are no Setup values to enter. Select **Deploy**",
        "Wait until setup is ready",
        "Open **Open Chemistry Agent**",
        "Access is automatic",
        "Create one new session",
        "paste the four prompts below unchanged and in order",
        "optional exploration",
        "If an LLM request times out",
        "retry the whole prompt once",
        "Do not retry individual commands",
        "After a second timeout, ask the facilitator",
    ):
        assert required in compact_lab_2

    attendee_actions = (
        "Use the default hardware",
        "There are no Setup values to enter. Select **Deploy**",
        "Wait until setup is ready",
        "Open **Open Chemistry Agent**",
        "Access is automatic",
        "Create one new session",
        "paste the four prompts below unchanged and in order",
    )
    positions = [compact_lab_2.index(action) for action in attendee_actions]
    assert positions == sorted(positions)
    for forbidden in (
        "NVIDIA_INFERENCE_API_KEY",
        "Enter the API key",
        "token",
        "password",
        "administrator",
        "setup-script",
    ):
        assert forbidden not in compact_lab_2


def test_deployment_checks_are_evergreen_without_a_live_readiness_claim() -> None:
    both_labs = _section(_source(), "Complete both required labs")

    assert "signed-in Launchable page" in both_labs
    assert "For the Notebook deployment" in both_labs
    assert "required `NVIDIA_API_KEY` Setup field" in both_labs
    assert "For the conversational OpenClaw deployment" in both_labs
    assert "no Setup values" in both_labs
    assert "setup completes" in both_labs
    assert "expected app opens" in both_labs
    assert "This guide makes no live-readiness claim." in both_labs


def test_marked_prompt_blocks_are_byte_locked() -> None:
    expected_hashes = {
        "01-data-and-representation": "39ca26c1b494dbe01bcbaabf27d72d755b444915e9ff26c874e629f09610bf22",
        "02-relationships-and-groups": "5d556991910812a24bb09b23cd250fd4a7157986948082fb8cc05cb3d52c1f5e",
        "03-sampled-3d-geometry": "6779b1bfbe141a72c795d5e648ad33a5e7ddd55a8bc953b0c1ae116f757be34a",
        "04-objective": "ec93fcfa236b6000980178626b322aeb0786a52a53a0132338784221c24550ea",
    }

    blocks = _prompt_blocks(_source())
    assert {
        prompt_id: hashlib.sha256(block.encode("utf-8")).hexdigest()
        for prompt_id, block in blocks.items()
    } == expected_hashes


def test_full_marked_prompt_blocks_are_byte_locked() -> None:
    expected_hashes = {
        "01-data-and-representation": "5f4d12063bfba5be83abf2b8b3fa06d3d9b234d927e0ff4335d1049bad735f6d",
        "02-relationships-and-groups": "7de44e0efcd1c3f9023c4a4ee3c8a4fa0b4758872cad1c0cfa5fee3685baae19",
        "03-sampled-3d-geometry": "f7e14b338b3c3b104c56d89baa2e8db0e20378a8a53d0f345524cdffa6035d4e",
        "04-objective": "8356ebf74d0034cbff6a361f5060d2008c2664f633f072290a9944fbf93cd726",
    }

    blocks = _full_marked_prompt_blocks(_source())
    assert {
        prompt_id: hashlib.sha256(block.encode("utf-8")).hexdigest()
        for prompt_id, block in blocks.items()
    } == expected_hashes


def test_timeout_recovery_retries_one_whole_prompt_outside_prompt_blocks() -> None:
    source = _source()
    lab_2 = " ".join(
        _section(source, "Required Lab 2 — Conversational OpenClaw").split()
    )
    recovery = (
        "If an LLM request times out, start a new session and retry the whole prompt "
        "once. Do not retry individual commands. After a second timeout, ask the "
        "facilitator."
    )

    assert lab_2.count(recovery) == 1
    for block in _prompt_blocks(source).values():
        assert "retry the whole prompt" not in block


def test_exactly_four_self_contained_prompts_use_only_the_fixed_runner() -> None:
    source = _source()
    blocks = _prompt_blocks(source)

    for prompt_id, command in LESSON_COMMANDS.items():
        assert blocks[prompt_id].count(command) == 1
    assert blocks["04-objective"].count(OBJECTIVE_START) == 1
    assert blocks["04-objective"].count(OBJECTIVE_STEP) == 1
    assert source.count(f"{RUNNER} run-lesson") == 3
    assert source.count(OBJECTIVE_START) == 1
    assert source.count(f"{RUNNER} objective-step") == 1
    assert "run-stage" not in source
    assert "--dataset" not in source
    assert "--output" not in source

    skill_path = "/sandbox/.openclaw/skills/nvmolkit-usage/SKILL.md"
    for prompt_id in PROMPT_IDS:
        assert skill_path not in blocks[prompt_id]
    assert "The installed `nvmolkit-usage` skill remains available" in source

    for prompt_id in PROMPT_IDS[:3]:
        assert "Run only this exact command, once:" in blocks[prompt_id]
    assert "Run only the exact commands below." in blocks["04-objective"]

    for prompt_id, block in blocks.items():
        assert "Work only in `/sandbox/.openclaw/workspace`." in block
        assert "Do not read or edit files." in block
        assert "Do not install software or use the network." in block
        assert "Do not run an alternate command." in block
        assert "report the error and stop" in block
        assert "Do not repair or retry" in block
        heading_positions = [block.index(heading) for heading in RESPONSE_HEADINGS]
        assert heading_positions == sorted(heading_positions)
        assert "Download Results" in block
        assert "`workshop/results.zip`" in block
        assert block.rstrip().endswith(MEDIA_LINES[prompt_id])

    for prompt_id in PROMPT_IDS[:3]:
        assert "Use at most three measured-result bullets." in blocks[prompt_id]
    assert (
        "Use at most three measured facts: baseline `D_min`, final `D_min`, "
        "and their change." in blocks["04-objective"]
    )


def test_prompt_blocks_lead_with_scientific_objectives_before_execution() -> None:
    for block in _prompt_blocks(_source()).values():
        positions = (
            block.index("Question:"),
            block.index("Scientific objective:"),
            block.index("Execution contract:"),
            block.index("Answer contract:"),
        )
        assert positions == tuple(sorted(positions))


def test_lesson_prompts_stop_after_the_first_complete_result() -> None:
    clauses = (
        "This command has a one-call budget.",
        "The budget is consumed when the command is submitted.",
        "top-level `status: complete`",
        "stop all tool use",
        "first completed result as authoritative",
        "Do not emit an empty response or run the command again.",
        "return its decoded `answer_markdown` string exactly",
    )
    for prompt_id, command in LESSON_COMMANDS.items():
        block = _prompt_blocks(_source())[prompt_id]
        assert block.count(command) == 1
        for clause in clauses:
            assert clause in block
        assert block.index(command) < block.index("top-level `status: complete`")
        assert block.index("top-level `status: complete`") < block.index(
            "Answer contract:"
        )


def test_prompt_science_boundaries_match_the_runner_contract() -> None:
    blocks = _prompt_blocks(_source())
    first = blocks["01-data-and-representation"]
    second = blocks["02-relationships-and-groups"]
    third = blocks["03-sampled-3d-geometry"]
    objective = blocks["04-objective"]

    assert "Do not use the words `accelerated` or `acceleration`" in first
    assert "cutoff `0.40` is Tanimoto distance, not Tanimoto similarity" in second
    assert "similarity `1.0` does not prove molecular identity" in second
    assert (
        "nvMolKit computed fingerprints and Tanimoto similarities on GPU; "
        "RDKit performed Butina clustering on CPU." in second
    )
    assert "returns both conformer stages" in third
    assert "Do not run it once per stage." in third
    assert "`D_min` is the minimum pairwise Tanimoto distance" in objective
    assert "`D_min = min(1 - Tanimoto similarity)`" in objective
    assert "Do not call `D_min` a similarity score." in objective
    assert "higher `D_min` means greater separation" in objective
    assert (
        "Do not report intermediate, predicted, target, or per-step scores "
        "anywhere in the answer." in objective
    )


def test_relationships_question_attributes_gpu_similarity_before_distance() -> None:
    block = _prompt_blocks(_source())["02-relationships-and-groups"]
    question = (
        "Question: Which molecules are similar, and how does Butina group them "
        "from distances derived from GPU-computed Tanimoto similarities?"
    )

    assert block.startswith(question)
    assert "GPU-computed Tanimoto distances" not in block


def test_prompt_answer_limits_match_canonical_bullet_contract() -> None:
    blocks = _prompt_blocks(_source())
    for prompt_id in PROMPT_IDS[:3]:
        assert "Use at most three measured-result bullets." in blocks[prompt_id]
    assert (
        "Use at most three measured facts: baseline `D_min`, final `D_min`, "
        "and their change." in blocks["04-objective"]
    )


def test_objective_prompt_uses_the_actual_menu_contract() -> None:
    block = _prompt_blocks(_source())["04-objective"]

    assert "Run `objective-start` exactly once" in block
    assert "zero objective-step commands" in block
    assert "at most three objective-step commands" in block
    assert "maximum numeric `predicted_score`" in block
    assert "predicted `D_min`" in block
    assert "exact returned `state_id` and `swap_id`" in block
    assert "Keep both substituted values single-quoted" in block
    assert "stop immediately when `terminal` is `true`" in block


def test_scientific_limits_cover_both_required_labs_without_overclaiming() -> None:
    limits = _section(_source(), "Scientific limits")

    for notebook_limit in (
        "deterministic 96-row ReFRAME snapshot",
        "bounded 24-compound panel",
        "fingerprint-dependent atlas",
        "run-specific timings",
    ):
        assert notebook_limit in limits

    for conversational_limit in (
        "deterministic 256-record ChEMBL convenience sample, not representative "
        "chemical space",
        "cutoff `0.40` is Tanimoto distance",
        "not centroids, medoids, or globally optimal representatives",
        "radius-2, 1024-bit hashed fingerprint",
        "similarity `1.0` does not prove molecular identity",
        "MMFF94 energies compare sampled conformers within one molecule only",
        "`D_min` is the weakest-link diversity score within eight fixed candidates",
        "real GPU execution, not acceleration or speedup",
    ):
        assert conversational_limit in limits

    assert (
        "The results do not prove identity, binding, activity, ADMET, efficacy, "
        "safety, synthesizability, clinical value, or experimental structure." in limits
    )


def test_download_and_cleanup_are_attendee_actions() -> None:
    source = _source()
    download = _section(source, "Download")
    finish = _section(source, "Finish")

    assert "Open **Download Results**" in download
    assert "open `workshop/`" in download
    assert "download `results.zip`" in download
    assert "not a browser URL" in download
    assert "Stop both workshop environments" in finish
    assert "Delete each environment when you are finished" in finish
    assert "Stopped storage can still incur a charge" in finish
    assert "Deletion is permanent" in finish
    assert "download your results first" in finish


def test_page_contains_no_unsafe_or_expansive_instructions() -> None:
    source = _source()
    lower = source.lower()

    for forbidden in (
        "builddonevideo",
        "18789",
        "gateway-token",
        "dashboard-url",
        "?token=",
        "/org/",
        "pip install",
        "conda install",
        "curl ",
        "wget ",
    ):
        assert forbidden not in lower
    assert re.search(r"nvapi-[A-Za-z0-9_-]{10,}", source) is None
