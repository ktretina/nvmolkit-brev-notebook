"""Shared ReFRAME loading and descriptor helpers for the ACS nvMolKit workshop."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


REFRAME_URL = "https://reframedb.org/assets/csv/reframe_smiles_list.csv"
SNAPSHOT_PATH = Path(__file__).with_name("data") / "reframe_teaching_snapshot.csv"
MODULE1_SNAPSHOT_PATH = (
    Path(__file__).with_name("data") / "reframe_module1_snapshot_10k.csv"
)
MODULE1_PROVENANCE_PATH = MODULE1_SNAPSHOT_PATH.with_suffix(".provenance.json")
MODULE1_MIN_SAMPLE_SIZE = 96
MODULE1_MAX_SAMPLE_SIZE = 10_000
MODULE1_INTEGRITY_ERROR = (
    "Bundled Module 1 ReFRAME snapshot failed its integrity check."
)
DEFAULT_MEMORY_LIMIT_MIB = 128
REQUIRED_COLUMNS = {
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
}


def _positive_integer(value, *, name):
    """Return a positive built-in integer, rejecting implicit coercions."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def square_matrix_bytes(item_count):
    """Estimate bytes for an item_count-square float32 similarity matrix."""
    count = _positive_integer(item_count, name="item_count")
    return count * count * 4


def condensed_distance_bytes(item_count):
    """Estimate bytes for RDKit Butina's condensed float64 distance vector."""
    count = _positive_integer(item_count, name="item_count")
    return count * (count - 1) // 2 * 8


def require_bounded_condensed_distances(row_count, *, maximum_bytes=128 * 1024 * 1024):
    """Return the condensed-vector bytes after enforcing a strict byte bound."""
    count = _positive_integer(row_count, name="row_count")
    maximum = _positive_integer(maximum_bytes, name="maximum_bytes")
    required_bytes = condensed_distance_bytes(count)
    if required_bytes > maximum:
        raise MemoryError(
            f"Condensed distance allocation requires {required_bytes:,} bytes, "
            f"above maximum {maximum:,} bytes."
        )
    return required_bytes


def require_memory_within_limit(required_bytes, *, limit_mib=DEFAULT_MEMORY_LIMIT_MIB):
    """Fail before an estimated allocation exceeds an explicit memory limit."""
    if isinstance(required_bytes, bool) or not isinstance(required_bytes, int):
        raise TypeError("required_bytes must be a non-negative integer.")
    if required_bytes < 0:
        raise ValueError("required_bytes must be a non-negative integer.")
    limit = _positive_integer(limit_mib, name="limit_mib")
    limit_bytes = limit * 1024 * 1024
    if required_bytes > limit_bytes:
        raise MemoryError(
            f"Estimated allocation requires {required_bytes:,} bytes, "
            f"above the {limit} MiB limit."
        )
    return required_bytes


def _load_module1_snapshot(requested_size):
    try:
        observed_hash = hashlib.sha256(MODULE1_SNAPSHOT_PATH.read_bytes()).hexdigest()
        provenance = json.loads(
            MODULE1_PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        if observed_hash != provenance["final_sha256"]:
            raise ValueError(MODULE1_INTEGRITY_ERROR)
        frame = pd.read_csv(
            MODULE1_SNAPSHOT_PATH, keep_default_na=False, dtype=str
        )
    except Exception:
        raise ValueError(MODULE1_INTEGRITY_ERROR) from None

    missing = REQUIRED_COLUMNS - set(frame.columns)
    valid_smiles = frame["smile"].str.strip().ne("") if "smile" in frame else None
    valid_keys = (
        frame["canonical_ikey"].str.strip().ne("")
        if "canonical_ikey" in frame
        else None
    )
    valid_urls = (
        frame["reframedb_url"].str.startswith("https://")
        if "reframedb_url" in frame
        else None
    )
    if (
        missing
        or len(frame) != MODULE1_MAX_SAMPLE_SIZE
        or valid_smiles is None
        or not valid_smiles.all()
        or valid_keys is None
        or not valid_keys.all()
        or valid_urls is None
        or not valid_urls.all()
        or frame["canonical_ikey"].duplicated().any()
    ):
        raise ValueError(MODULE1_INTEGRITY_ERROR)

    molecules = [Chem.MolFromSmiles(smiles) for smiles in frame["smile"]]
    if any(molecule is None for molecule in molecules):
        raise ValueError(MODULE1_INTEGRITY_ERROR)
    result = frame.iloc[:requested_size].copy().reset_index(drop=True)
    result["name"] = result["name"].replace("", "unnamed compound")
    result["_mol"] = molecules[:requested_size]
    result.attrs.update(source="bundled_snapshot_10k", invalid_count=0)
    return result


def load_reframe(
    sample_size=96,
    anchor_terms=(),
    random_state=2026,
    *,
    source="snapshot",
    use_configured_csv=False,
):
    """Load, deduplicate, deterministically sample, and parse ReFRAME compounds."""
    if not isinstance(source, str) or source not in {
        "snapshot",
        "snapshot_10k",
        "live",
    }:
        raise ValueError("source must be 'snapshot', 'snapshot_10k', or 'live'.")
    if type(use_configured_csv) is not bool:
        raise TypeError("use_configured_csv must be a bool.")
    if source == "live" and use_configured_csv:
        raise ValueError(
            "source='live' cannot be combined with use_configured_csv=True."
        )
    if source == "snapshot_10k" and use_configured_csv:
        raise ValueError(
            "source='snapshot_10k' cannot be combined with use_configured_csv=True."
        )
    requested_size = _positive_integer(sample_size, name="sample_size")
    if source == "snapshot_10k":
        if not MODULE1_MIN_SAMPLE_SIZE <= requested_size <= MODULE1_MAX_SAMPLE_SIZE:
            raise ValueError("source='snapshot_10k' supports 96 through 10,000 rows.")
        return _load_module1_snapshot(requested_size)
    if use_configured_csv:
        configured_csv = os.environ.get("REFRAME_CSV", "")
        if not configured_csv:
            raise ValueError("Could not load the configured ReFRAME CSV.")
        try:
            frame = pd.read_csv(configured_csv)
        except Exception:
            raise ValueError("Could not load the configured ReFRAME CSV.") from None
        source_label = "configured_csv"
    elif source == "snapshot":
        frame = pd.read_csv(SNAPSHOT_PATH)
        source_label = "bundled_snapshot"
    else:
        try:
            frame = pd.read_csv(REFRAME_URL)
        except Exception:
            raise ValueError("Could not load the live ReFRAME export.") from None
        source_label = "live"

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        if source_label == "configured_csv":
            raise ValueError("Could not load the configured ReFRAME CSV.") from None
        raise ValueError(f"ReFRAME schema changed; missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["name"] = frame["name"].fillna("unnamed compound")
    frame = frame.dropna(subset=["smile", "canonical_ikey"])
    frame = frame.drop_duplicates("canonical_ikey", keep="first").reset_index(drop=True)
    if source_label == "bundled_snapshot" and requested_size > len(frame):
        raise ValueError(
            f"Bundled ReFRAME snapshot contains {len(frame)} rows; "
            f"requested {requested_size}."
        )

    anchors = []
    for term in anchor_terms:
        hit = frame[frame["name"].str.contains(term, case=False, regex=False)]
        if not hit.empty:
            anchors.append(hit.iloc[[0]])
    anchor_frame = (
        pd.concat(anchors, ignore_index=True).drop_duplicates("canonical_ikey")
        if anchors
        else frame.iloc[0:0]
    )

    target_n = min(requested_size, len(frame))
    anchor_frame = anchor_frame.iloc[:target_n]
    remaining = frame[~frame["canonical_ikey"].isin(anchor_frame["canonical_ikey"])]
    random_n = max(0, target_n - len(anchor_frame))
    sampled = remaining.sample(
        n=min(random_n, len(remaining)), random_state=random_state
    )
    frame = pd.concat([anchor_frame, sampled], ignore_index=True).drop_duplicates(
        "canonical_ikey"
    )

    molecules = [Chem.MolFromSmiles(smiles) for smiles in frame["smile"]]
    valid = np.asarray([molecule is not None for molecule in molecules])
    frame = frame.loc[valid].reset_index(drop=True)
    frame["_mol"] = [molecule for molecule in molecules if molecule is not None]
    frame.attrs.update(source=source_label, invalid_count=int((~valid).sum()))
    return frame


def add_descriptors(frame):
    """Add a compact set of interpretable physicochemical descriptors."""
    result = frame.copy()
    molecules = result["_mol"].tolist()
    result["MolWt"] = [Descriptors.MolWt(molecule) for molecule in molecules]
    result["cLogP"] = [Crippen.MolLogP(molecule) for molecule in molecules]
    result["TPSA"] = [rdMolDescriptors.CalcTPSA(molecule) for molecule in molecules]
    result["HBD"] = [Lipinski.NumHDonors(molecule) for molecule in molecules]
    result["HBA"] = [Lipinski.NumHAcceptors(molecule) for molecule in molecules]
    result["RotB"] = [Lipinski.NumRotatableBonds(molecule) for molecule in molecules]
    return result
