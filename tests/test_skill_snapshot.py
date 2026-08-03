"""Integrity checks for the vendored public nvMolKit agent skill."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = PROJECT_ROOT / "skills" / "nvmolkit" / "SKILL.md"
PROVENANCE_PATH = PROJECT_ROOT / "skills" / "nvmolkit" / "PROVENANCE.md"
PINNED_BYTE_COUNT = 17699
PINNED_SHA256 = "1e7aa4102c100a7dfd06f1b093c68159fc74146ca6a9bfc1683f85236a059af2"


def _provenance_value(provenance: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", provenance, re.MULTILINE)
    assert match, f"Missing provenance field: {label}"
    return match.group(1).strip()


def test_nvmolkit_skill_is_a_pinned_public_snapshot() -> None:
    """The local copy remains byte-identical to its declared public source."""
    snapshot = SKILL_PATH.read_bytes()
    provenance = PROVENANCE_PATH.read_text(encoding="utf-8")

    source_url = _provenance_value(provenance, "Source blob")
    source_commit = _provenance_value(provenance, "Upstream commit")
    retrieval_date = _provenance_value(provenance, "Retrieval date")
    expected_byte_count = _provenance_value(provenance, "Byte count")
    expected_digest = _provenance_value(provenance, "SHA-256")
    license_name = _provenance_value(provenance, "Upstream license")

    assert re.fullmatch(r"[0-9a-f]{40}", source_commit)
    assert source_commit == "ce151c15470991c8cb9a0efdd531a124c346ca5b"
    assert retrieval_date == "2026-08-02"
    assert source_url == (
        "https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/"
        f"{source_commit}/library-skills/nvMolKit/SKILL.md"
    )
    assert license_name == (
        "Apache-2.0 OR CC-BY-4.0 "
        "(Apache License 2.0 or Creative Commons Attribution 4.0 International)"
    )
    actual_digest = hashlib.sha256(snapshot).hexdigest()
    assert len(snapshot) == PINNED_BYTE_COUNT
    assert actual_digest == PINNED_SHA256
    assert int(expected_byte_count) == PINNED_BYTE_COUNT
    assert expected_digest == PINNED_SHA256

    skill = snapshot.decode("utf-8")
    for required_text in (
        "MorganFingerprintGenerator",
        "crossTanimotoSimilarity",
        "fused_butina",
        "EmbedMolecules",
        "MMFFOptimizeMoleculesConfs",
    ):
        assert required_text in skill
    assert "There is no CPU fallback" in skill
    assert "An NVIDIA GPU with compute capability 7.0 (V100) or higher" in skill
