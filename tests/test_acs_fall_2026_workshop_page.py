from __future__ import annotations

import re
from pathlib import Path


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


def _source() -> str:
    assert PAGE.is_file(), "the canonical ACS attendee page is missing"
    return PAGE.read_text(encoding="utf-8")


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


def test_page_has_the_short_attendee_order_and_current_links() -> None:
    source = _source()
    sections = (
        "# ACS Fall 2026 GPU chemistry workshop",
        "## Before the workshop",
        "## Choose your lab",
        "## Launch the required lab",
        "## Four prompts",
        "## Download your results",
        "## Finish and remove your environments",
        "## Scientific limits",
        "## Official links",
    )
    positions = [source.index(section) for section in sections]
    assert positions == sorted(positions)

    for url in (
        "https://brev.nvidia.com/",
        "https://account.nvidia.com/",
        "https://build.nvidia.com/settings/api-keys",
        "https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted",
        "https://github.com/ktretina/nvmolkit-brev-notebook",
        "https://brev.nvidia.com/launchable/deploy/now?launchableID="
        "env-3HJtJW3qHg4Dw1I3xt75BfpBmZW",
        "https://brev.nvidia.com/launchable/deploy/now?launchableID="
        "env-3Hlp4pHBlTTlfDxfH41KkGhTeCV",
        "https://docs.nvidia.com/brev/getting-started/quickstart",
        "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/"
        "integrations/nvidia-integrations.html",
        "https://docs.nvidia.com/brev/concepts/gpu-instances",
    ):
        assert url in source


def test_prework_is_one_account_secret_safe_and_cost_aware() -> None:
    source = _source()

    assert "one NVIDIA account" in source
    assert "verify your email" in source
    assert "phone verification" in source
    assert "NVIDIA Cloud Account" in source
    assert "Brev credits or a payment method" in source
    assert "rate-limited" in source
    assert "Brev GPU compute is separate and billable" in source
    assert "Never paste the key into chat, a screenshot, or a file" in source
    assert "`NVIDIA_API_KEY`" in source
    assert "`NVIDIA_INFERENCE_API_KEY`" in source


def test_lab_roles_hardware_and_signed_in_boundary_are_explicit() -> None:
    source = _source()

    assert "Optional instructor-led companion" in source
    assert "not required for the hands-on workshop" in source
    assert "Required hands-on lab" in source
    assert "one NVIDIA L4" in source
    assert "x86-64" in source
    assert "4 CPUs" in source
    assert "16 GiB RAM" in source
    assert "128 GiB disk" in source
    assert "does not need to show `g6.xlarge`" in source
    assert "Wait until setup is ready" in source
    assert "Open Chemistry Agent" in source
    assert "create one new session" in source
    assert "public HTTP checks" in source
    assert "do not prove signed-in deployability" in source


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
    assert blocks["01-data-and-representation"].count(skill_path) == 1
    for prompt_id in PROMPT_IDS[1:]:
        assert skill_path not in blocks[prompt_id]

    for prompt_id, block in blocks.items():
        assert "Work only in `/sandbox/.openclaw/workspace`." in block
        assert "Do not install software or use the network." in block
        assert "Do not edit any fixed file or run an alternate command." in block
        assert "report the error and stop" in block
        assert "Do not repair or retry" in block
        heading_positions = [block.index(heading) for heading in RESPONSE_HEADINGS]
        assert heading_positions == sorted(heading_positions)
        assert "at most three measured facts" in block
        assert "Download Results" in block
        assert "`workshop/results.zip`" in block
        assert block.rstrip().endswith(MEDIA_LINES[prompt_id])


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


def test_page_and_prompts_state_all_seven_scientific_limits() -> None:
    source = _source()

    for statement in (
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
        assert statement in source
    assert "does not demonstrate unrestricted autonomous design" in source


def test_download_and_cleanup_are_attendee_actions() -> None:
    source = _source()

    assert "Open **Download Results**" in source
    assert "open `workshop/`" in source
    assert "download `results.zip`" in source
    assert "not a browser URL" in source
    assert "Stop every workshop environment you started" in source
    assert "Delete each environment when you are finished" in source
    assert "Stopped storage can still incur a charge" in source
    assert "Deletion is permanent" in source


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
