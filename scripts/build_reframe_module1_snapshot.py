#!/usr/bin/env python3
"""Build the fixed Module 1 ReFRAME snapshot from a local raw export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase


SOURCE_URL = "https://reframedb.org/assets/csv/reframe_smiles_list.csv"
RETRIEVED_DATE = "2026-08-24"
LEGACY_SHA256 = "d7ca4a132bfb18a9098e1e662c6e0d20fc522b4485f5f7366089cee287c18840"
RAW_SHA256 = "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
CORE_KEY_ORDER_SHA256 = "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
ANCHOR_TERMS = ("imatinib", "linezolid", "ritonavir")
CORE_ANCHORS = (
    "KTUFNOKKBVMGRW",
    "TYZROVQLWOKYKF",
    "NCDNCNXCDXHOMX",
)
REQUIRED_COLUMNS = (
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
)
RAW_REQUIRED_COLUMNS = (
    "smile",
    "ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
)
CORE_ROWS = 96
TARGET_ROWS = 10_000
ADDITION_ROWS = TARGET_ROWS - CORE_ROWS
HYPERLINK_RE = re.compile(r'=HYPERLINK\("(https://[^\"]+)"\)')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def _normalize_url(value: str) -> str:
    value = value.strip()
    if not value.startswith("="):
        return value
    match = HYPERLINK_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported ReFRAME URL formula: {value}")
    return match.group(1)


def _reconstruct_core(legacy_path: Path) -> pd.DataFrame:
    if _sha256(legacy_path) != LEGACY_SHA256:
        raise ValueError("The legacy ReFRAME teaching snapshot SHA-256 does not match.")

    legacy = pd.read_csv(legacy_path, keep_default_na=False, dtype=str)
    _require_columns(legacy, REQUIRED_COLUMNS)
    if len(legacy) != CORE_ROWS:
        raise ValueError(f"The legacy snapshot must contain exactly {CORE_ROWS} rows.")

    anchors = []
    for term in ANCHOR_TERMS:
        match = legacy[legacy["name"].str.contains(term, case=False, regex=False)]
        if match.empty:
            raise ValueError(f"The legacy snapshot has no anchor for {term!r}.")
        anchors.append(match.iloc[[0]])
    anchor_frame = pd.concat(anchors, ignore_index=True).drop_duplicates(
        "canonical_ikey"
    )
    remaining = legacy[~legacy["canonical_ikey"].isin(anchor_frame["canonical_ikey"])]
    core = pd.concat(
        [
            anchor_frame,
            remaining.sample(n=CORE_ROWS - len(anchor_frame), random_state=2026),
        ],
        ignore_index=True,
    )
    keys = core["canonical_ikey"].tolist()
    if tuple(keys[:3]) != CORE_ANCHORS:
        raise ValueError("The reconstructed core anchors do not match.")
    if _sequence_sha256(keys) != CORE_KEY_ORDER_SHA256:
        raise ValueError("The reconstructed core key order SHA-256 does not match.")
    return core.loc[:, list(REQUIRED_COLUMNS)].copy()


def build_snapshot(
    raw_path: Path,
    legacy_path: Path,
    output_path: Path,
    provenance_path: Path,
) -> None:
    """Build output_path and provenance_path from verified local inputs."""
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Module 1 output.")
    if _sha256(raw_path) != RAW_SHA256:
        raise ValueError("The raw ReFRAME export SHA-256 does not match.")

    raw = pd.read_csv(raw_path, keep_default_na=False, dtype=str)
    _require_columns(raw, RAW_REQUIRED_COLUMNS)
    if "canonical_ikey" in raw.columns:
        raise ValueError("The raw export must use 'ikey', not 'canonical_ikey'.")
    core = _reconstruct_core(legacy_path)
    seen_keys = set(core["canonical_ikey"])

    additions = []
    skipped_rows = 0
    RDLogger.DisableLog("rdApp.error")
    try:
        for row in raw.to_dict(orient="records"):
            key = row["ikey"].strip()
            smile = row["smile"].strip()
            url = _normalize_url(row["reframedb_url"])
            if (
                not key
                or not smile
                or not url.startswith("https://")
                or key in seen_keys
                or Chem.MolFromSmiles(smile) is None
            ):
                skipped_rows += 1
                continue
            seen_keys.add(key)
            additions.append(
                {
                    "smile": smile,
                    "canonical_ikey": key,
                    "name": row["name"].strip() or "unnamed compound",
                    "source": row["source"].strip(),
                    "source_id": row["source_id"].strip(),
                    "status": row["status"].strip(),
                    "reframedb_url": url,
                }
            )
    finally:
        RDLogger.EnableLog("rdApp.error")

    additions = pd.DataFrame(additions, columns=REQUIRED_COLUMNS).sort_values(
        "canonical_ikey", kind="mergesort"
    )
    if len(additions) < ADDITION_ROWS:
        raise ValueError(
            f"Only {len(additions)} eligible additions are available; "
            f"{ADDITION_ROWS} are required."
        )
    snapshot = pd.concat(
        [core, additions.iloc[:ADDITION_ROWS]], ignore_index=True
    ).loc[:, list(REQUIRED_COLUMNS)]
    if len(snapshot) != TARGET_ROWS or not snapshot["canonical_ikey"].is_unique:
        raise ValueError("The final snapshot does not have exactly 10,000 unique rows.")

    snapshot.to_csv(output_path, index=False, lineterminator="\n")
    provenance = {
        "asset": output_path.name,
        "authorization": (
            "Written redistribution permission confirmed by the workshop organizer "
            "on 2026-08-24; the permission record is retained by the organizer."
        ),
        "core_order_sha256": CORE_KEY_ORDER_SHA256,
        "curation_rules": [
            "preserve the anchored 96-row Module 1 sequence",
            "exclude the 96 core identifiers",
            "retain the first RDKit-valid row per identifier",
            "sort additions by canonical_ikey",
            "normalize ReFRAME HYPERLINK page links to HTTPS URLs",
        ],
        "final_row_count": TARGET_ROWS,
        "final_sha256": _sha256(output_path),
        "legacy_core_row_count": CORE_ROWS,
        "raw_row_count": len(raw),
        "raw_sha256": RAW_SHA256,
        "rdkit_version": rdBase.rdkitVersion,
        "retrieved_date": RETRIEVED_DATE,
        "skipped_raw_row_count": skipped_rows,
        "source_url": SOURCE_URL,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fixed Module 1 ReFRAME snapshot from a local raw export."
    )
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("legacy_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("provenance_path", type=Path)
    args = parser.parse_args()
    build_snapshot(
        args.raw_path,
        args.legacy_path,
        args.output_path,
        args.provenance_path,
    )


if __name__ == "__main__":
    main()
