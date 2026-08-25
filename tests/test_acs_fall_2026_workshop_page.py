from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "acs-fall-2026-workshop.md"
PRESENTATION = ROOT / "docs" / "NVIDIA-ACS-Fall-2026-Workshop.pdf"

PROMPT_IDS = (
    "01-data-and-representation",
    "02-relationships-and-groups",
    "03-sampled-3d-geometry",
    "04-objective",
)
REQUIRED_SECTIONS = (
    "## Before you begin",
    "## Choose either journey",
    "## Notebook journey",
    "## OpenClaw journey",
    "## Four OpenClaw prompts",
    "## Results and downloads",
    "## Finish",
    "## Scientific limits",
    "## Official links",
)
FORBIDDEN_ATTENDEE_TERMS = (
    "/sandbox/",
    "PYTHONPATH=",
    "acs_workshop_runner.py",
    "MEDIA:",
    "answer_markdown",
    "state_id",
    "swap_id",
    "one-call budget",
    "Run only this exact command",
)


def _source() -> str:
    assert PAGE.is_file(), "the canonical ACS attendee page is missing"
    return PAGE.read_text(encoding="utf-8")


def _section(source: str, heading: str) -> str:
    marker = f"## {heading}"
    assert marker in source, f"missing section: {marker}"
    start = source.index(marker)
    next_heading = source.find("\n## ", start + len(marker))
    return source[start:] if next_heading == -1 else source[start:next_heading]


def _prompt_blocks(source: str) -> dict[str, str]:
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
        marked = source[begin_index + len(begin) : end_index]
        fences = re.findall(r"~~~text\n(.*?)\n~~~", marked, flags=re.DOTALL)
        assert len(fences) == 1
        blocks[prompt_id] = fences[0]
        last_end = end_index
    assert source.count("<!-- ACS_PROMPT:") == 8
    return blocks


def test_page_has_two_optional_independent_journeys() -> None:
    source = _source()
    assert tuple(re.findall(r"^## .+$", source, flags=re.MULTILINE)) == REQUIRED_SECTIONS
    choice = _section(source, "Choose either journey")
    compact_choice = " ".join(choice.split())
    assert "Both journeys are optional and independent" in choice
    assert "You can do either one, both, or neither" in compact_choice

    lowered = source.lower()
    for stale in (
        "required lab",
        "complete both",
        "companion",
    ):
        assert stale not in lowered


def test_notebook_journey_uses_current_three_modules() -> None:
    section = _section(_source(), "Notebook journey")
    for notebook in (
        "`01_direct_nvmolkit_reframe.ipynb`",
        "`02_agent_assisted_reframe_neighborhoods.ipynb`",
        "`03_full_agent_reframe_panel_design.ipynb`",
    ):
        assert notebook in section
    assert "Module 1" in section and "direct nvMolKit" in section
    assert "**Approve Plan & Run Agent**" in section
    assert "`nvmolkit_nemotron_demo.ipynb`" not in section


def test_openclaw_journey_is_zero_input_and_attendee_directed() -> None:
    section = _section(_source(), "OpenClaw journey")
    assert "Leave **Setup** empty" in section
    assert "**Open Chemistry Agent**" in section
    assert "new chat" in section
    assert "Wait for the answer and image" in section
    assert "**Download Results**" in section


def test_attendee_prompts_hide_internal_execution_contract() -> None:
    blocks = _prompt_blocks(_source())
    for prompt in blocks.values():
        for forbidden in FORBIDDEN_ATTENDEE_TERMS:
            assert forbidden not in prompt


def test_prompts_preserve_science_and_repeat_safety() -> None:
    blocks = _prompt_blocks(_source())
    first, second, third, fourth = (blocks[prompt_id] for prompt_id in PROMPT_IDS)

    assert "RDKit" in first and "CPU" in first
    assert "nvMolKit" in first and "GPU" in first
    assert "represented for comparison" in first
    assert "scientific limits" in first

    assert "GPU fingerprint and similarity" in second
    assert "CPU clustering" in second
    assert "Butina" in second and "distance cutoff" in second
    assert "do not prove" in second

    assert "GPU" in third
    assert "convergence" in third and "MMFF94 energy" in third
    assert "not experimental structures" in third

    assert "minimum pairwise Tanimoto distance" in fourth
    assert "`D_min`" in fourth
    assert "bounded descriptor exercise" in fourth
    assert "not autonomous molecular design" in fourth

    for prompt in blocks.values():
        assert "image" in prompt.lower()
        assert "download" in prompt.lower()
        assert "existing result" in prompt
        assert "running it again" in prompt


def test_recovery_is_safe_and_requires_no_repair() -> None:
    section = _section(_source(), "OpenClaw journey")
    assert "Send each prompt only once" in section
    assert "If OpenClaw is still working, wait" in section
    assert "accidentally send the same prompt twice" in section
    assert "do not try to repair anything" in section
    assert "take a screenshot and continue with another optional activity" in section


def test_results_are_named_without_internal_paths() -> None:
    section = _section(_source(), "Results and downloads")
    for expected in (
        "library preview",
        "cluster-size",
        "optimized-structure",
        "final-panel",
        "`results.zip`",
    ):
        assert expected in section
    for forbidden in FORBIDDEN_ATTENDEE_TERMS:
        assert forbidden not in section


def test_marked_prompt_blocks_are_byte_locked() -> None:
    expected_hashes = {
        "01-data-and-representation": "ba3a7a11c86d5ec781537c23cc5e153e2f42f74d9300b91835d0d3361760642f",
        "02-relationships-and-groups": "9287a0a5114149712210770b649fdda4157bef9165a514a350fbbae41a426aa0",
        "03-sampled-3d-geometry": "46aa528617a52838a93fc5a37da159fcb33d76bb9d90a71181a20e8039a0f8e2",
        "04-objective": "905bf47c129bbd01da7f630b09951194e042330fcf4f7a7f6f806975b0ea8c4c",
    }
    blocks = _prompt_blocks(_source())
    assert {
        prompt_id: hashlib.sha256(block.encode("utf-8")).hexdigest()
        for prompt_id, block in blocks.items()
    } == expected_hashes


def test_no_credential_value_or_attendee_admin_action_is_present() -> None:
    source = _source()
    lowered = source.lower()
    assert re.search(r"nvapi-[A-Za-z0-9_-]{8,}", source) is None
    assert re.search(r"(?i)(?:api[_ -]?key|token|password)\s*[=:]\s*\S+", source) is None
    for action in ("open a terminal", "run this command", "edit the file", "ask an administrator"):
        assert action not in lowered
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
        assert forbidden not in lowered


def test_scientific_roles_and_limits_remain_explicit() -> None:
    source = _source()
    notebook = _section(source, "Notebook journey")
    limits = _section(source, "Scientific limits")
    compact_limits = " ".join(limits.split())
    assert "RDKit on the CPU" in notebook
    assert "nvMolKit" in notebook and "on the GPU" in notebook
    assert "Python validates every bounded agent action" in notebook
    for boundary in (
        "radius-2, 1024-bit",
        "Tanimoto distance rule",
        "not experimental structures",
        "only within the same molecule",
        "does not demonstrate autonomous molecular design",
    ):
        assert boundary in compact_limits


def test_current_launchable_and_official_links_are_present() -> None:
    source = _source()
    for value in (
        "env-3HJtJW3qHg4Dw1I3xt75BfpBmZW",
        "env-3Hlp4pHBlTTlfDxfH41KkGhTeCV",
        "[Workshop presentation](NVIDIA-ACS-Fall-2026-Workshop.pdf)",
        "https://github.com/NVIDIA-BioNeMo/nvMolKit",
        "https://docs.nvidia.com/brev/latest/launchables/index.html",
        "https://docs.nvidia.com/brev/latest/concepts/gpu-instances.html",
    ):
        assert value in source
    assert PRESENTATION.is_file()


def test_page_has_no_obsolete_api_key_or_nvidia_inference_references() -> None:
    lowered = _source().lower()
    for value in (
        "nvidia_api_key",
        "nvidia inference",
        "inference.nvidia.com",
        "api key",
        "inference key",
        "hosted inference",
    ):
        assert value not in lowered
