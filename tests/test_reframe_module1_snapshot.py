import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SNAPSHOT_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_teaching_snapshot.csv"
SNAPSHOT_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_module1_snapshot_10k.csv"
PROVENANCE_PATH = SNAPSHOT_PATH.with_suffix(".provenance.json")

LEGACY_SHA256 = "d7ca4a132bfb18a9098e1e662c6e0d20fc522b4485f5f7366089cee287c18840"
RAW_EXPORT_SHA256 = "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
CORE_KEY_ORDER_SHA256 = "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
FINAL_CSV_SHA256 = "4e7855a56ff07522a302be65fd6b60bbf077a2568bddd6fe31b46347b4c531ac"
RAW_ROW_COUNT = 13_831
LEGACY_CORE_ROW_COUNT = 96
RDKIT_VERSION = "2026.03.5"
SKIPPED_RAW_ROW_COUNT = 814
CORE_ANCHORS = (
    "KTUFNOKKBVMGRW",
    "TYZROVQLWOKYKF",
    "NCDNCNXCDXHOMX",
)
SOURCE_URL = "https://reframedb.org/assets/csv/reframe_smiles_list.csv"
RETRIEVED_DATE = "2026-08-24"
CURATION_RULES = [
    "preserve the anchored 96-row Module 1 sequence",
    "exclude the 96 core identifiers",
    "retain the first RDKit-valid row per identifier",
    "sort additions by canonical_ikey",
    "normalize ReFRAME HYPERLINK page links to HTTPS URLs",
]
AUTHORIZATION = (
    "Written redistribution permission confirmed by the workshop organizer on "
    "2026-08-24; the permission record is retained by the organizer."
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def test_module1_reframe_snapshot_is_a_verified_10k_local_asset():
    assert _sha256(LEGACY_SNAPSHOT_PATH) == LEGACY_SHA256

    snapshot = pd.read_csv(SNAPSHOT_PATH, keep_default_na=False, dtype=str)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert len(snapshot) == 10_000
    assert tuple(snapshot.columns) == REQUIRED_COLUMNS
    assert snapshot["canonical_ikey"].notna().all()
    assert snapshot["canonical_ikey"].str.strip().ne("").all()
    assert snapshot["canonical_ikey"].is_unique
    assert snapshot["smile"].notna().all()
    assert snapshot["smile"].str.strip().ne("").all()
    assert snapshot["reframedb_url"].str.startswith("https://").all()
    assert not snapshot["reframedb_url"].str.startswith("=").any()
    assert all(
        Chem.MolFromSmiles(smile) is not None for smile in snapshot["smile"]
    )

    keys = snapshot["canonical_ikey"].tolist()
    assert tuple(keys[:3]) == CORE_ANCHORS
    assert _sequence_sha256(keys[:96]) == CORE_KEY_ORDER_SHA256
    assert keys[96:] == sorted(keys[96:])

    assert provenance["raw_sha256"] == RAW_EXPORT_SHA256
    assert provenance["raw_row_count"] == RAW_ROW_COUNT
    assert provenance["legacy_core_row_count"] == LEGACY_CORE_ROW_COUNT
    assert provenance["rdkit_version"] == RDKIT_VERSION
    assert provenance["final_row_count"] == 10_000
    assert provenance["final_sha256"] == FINAL_CSV_SHA256
    assert _sha256(SNAPSHOT_PATH) == FINAL_CSV_SHA256
    assert provenance["source_url"] == SOURCE_URL
    assert provenance["retrieved_date"] == RETRIEVED_DATE
    assert provenance["core_order_sha256"] == CORE_KEY_ORDER_SHA256
    assert len(provenance["curation_rules"]) == 5
    assert provenance["curation_rules"] == CURATION_RULES
    assert provenance["asset"] == SNAPSHOT_PATH.name
    assert provenance["authorization"] == AUTHORIZATION
    assert provenance["skipped_raw_row_count"] == SKIPPED_RAW_ROW_COUNT
