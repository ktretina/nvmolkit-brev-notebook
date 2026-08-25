# Module 1 10,000-Molecule Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an attendee change only `SAMPLE_SIZE` in Module 1 and execute Steps 1 through 3 successfully for every integer from 96 through 10,000.

**Architecture:** Add a deterministic, Module-1-only 10,000-row ReFRAME snapshot and an explicit `snapshot_10k` loader source. Keep the existing 96-row snapshot and Modules 2 and 3 unchanged. Step 3 continues to compute only three query-to-library similarity rows; Step 4 retains its existing quadratic-memory guard and requires resetting the sample to 96.

**Tech Stack:** Python 3.12, pandas 2.3.1, RDKit supplied through nvMolKit 0.6.0, nbformat/nbconvert, pytest 8.4.1, nvMolKit GPU operations, Git, GitHub, NVIDIA Brev.

---

## File map

- Create `scripts/build_reframe_module1_snapshot.py`: deterministic offline curation of the approved public ReFRAME export.
- Create `tests/test_build_reframe_module1_snapshot.py`: focused builder behavior and failure tests.
- Create `notebooks/data/reframe_module1_snapshot_10k.csv`: generated 10,000-row Module 1 asset.
- Create `notebooks/data/reframe_module1_snapshot_10k.provenance.json`: public provenance, hashes, rules, and authorization statement.
- Create `tests/test_reframe_module1_snapshot.py`: real-asset integrity and compatibility acceptance.
- Modify `notebooks/workshop_common.py`: add the explicit `snapshot_10k` source without changing legacy `snapshot` behavior.
- Modify `tests/test_workshop_common.py`: drive range, prefix, integrity, and legacy-regression behavior.
- Modify `notebooks/01_direct_nvmolkit_reframe.ipynb`: update only existing Module 1 instructions and its Step 1 source constant.
- Modify `tests/test_workshop_notebook_inventory.py`: protect notebook wording, execution, receipts, and allocation boundaries.
- Do not modify `notebooks/data/reframe_teaching_snapshot.csv`, Modules 2 or 3, API-key handling, the setup renderer, or another Launchable.

## Pre-implementation source gate

This is a user-operated, read-only Console gate because no supported authenticated Launchable-authoring interface is available to this task. Before Task 1, the user opens only `nvMolKit + Nemotron Notebook` (`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`) in the Brev Console and reports its repository URL, branch, and fixed-commit setting without saving. The current implementation branch is based on `origin/main`; proceed only if the saved notebook Launchable also follows `main`. If it does not, stop and rebuild this feature branch from the observed source branch before changing code. Do not inspect another Launchable.

### Task 1: Build and verify the deterministic Module 1 snapshot

**Files:**
- Create: `scripts/build_reframe_module1_snapshot.py`
- Create: `tests/test_build_reframe_module1_snapshot.py`
- Create: `tests/test_reframe_module1_snapshot.py`
- Generate: `notebooks/data/reframe_module1_snapshot_10k.csv`
- Generate: `notebooks/data/reframe_module1_snapshot_10k.provenance.json`
- Read only: `notebooks/data/reframe_teaching_snapshot.csv`

- [ ] **Step 0: Pin the Brev task contract and obtain the Python environment**

Complete this gate before running any test in this task. Do not install the chemistry stack on the memory-constrained Mac. Record this contract:

```text
Task label: module1-10k-snapshot
Brev organization: agents-in-ls
Instance: nvmolkit---nemotron-notebook-80f439 (abnc4jxm4), only if the user reconfirms exclusive ownership
Remote directory: /home/ubuntu/workspace/codex/module1-10k-snapshot
Repository branch: fix/module1-10k-snapshot
Processes/containers/tmux/ports: none
GPU policy: one target L4; no sharing
Allowed lifecycle actions: read/copy/exec; start and stop require separate approval
Cost boundary: no new or resumed billing without explicit approval
```

The installed Brev CLI was `v0.6.332` when this plan was written. Re-run the exact version and help gates:

```bash
/opt/homebrew/bin/brev --version
/opt/homebrew/bin/brev ls --help
/opt/homebrew/bin/brev start --help
/opt/homebrew/bin/brev stop --help
/opt/homebrew/bin/brev exec --help
/opt/homebrew/bin/brev copy --help
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

Do not run `brev set` or `brev refresh`. Because `exec` and `copy` do not expose an organization flag in `v0.6.332`, do not use them until the user confirms Brev work is serialized, the active organization is already `agents-in-ls`, and this instance is exclusively assigned to this task. Request approval before starting the retained instance. If it does not exist, request separate cost approval for one fresh deployment of only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

After approval, immediately re-list the exact target. Start it only if it is the confirmed stopped instance, then create the isolated verification tree:

```bash
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev start nvmolkit---nemotron-notebook-80f439
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 "id -un"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 "test -d /home/ubuntu/workspace"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "mkdir -p /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/scripts /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/notebooks/data"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "/home/ubuntu/.venv/bin/python --version"
```

Expected: exact instance `abnc4jxm4`, one L4, and CPython 3.12. If the instance is already running, omit `brev start`; never restart it. Stop on any identity, ownership, organization, hardware, or Python mismatch.

- [ ] **Step 1: Write failing unit tests for the offline builder**

Create `tests/test_build_reframe_module1_snapshot.py`:

```python
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_reframe_module1_snapshot import build_snapshot


COLUMNS = (
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
)


def _write_csv(path: Path, rows, *, raw=False):
    columns = tuple("ikey" if raw and name == "canonical_ikey" else name for name in COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if raw:
                payload["ikey"] = payload.pop("canonical_ikey")
            writer.writerow(payload)


def _row(key, name, smiles="CCO", *, raw=False):
    url = f"https://reframedb.org/compound_data/{key}"
    if raw:
        url = f'=HYPERLINK("{url}")'
    return {
        "smile": smiles,
        "canonical_ikey": key,
        "name": name,
        "source": "test",
        "source_id": key,
        "status": "available",
        "reframedb_url": url,
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_snapshot_preserves_anchored_core_and_selects_valid_unique_additions(
    tmp_path,
):
    legacy_path = tmp_path / "legacy.csv"
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "snapshot.csv"
    provenance_path = tmp_path / "snapshot.provenance.json"

    legacy_rows = [
        _row("KEY-IMATINIB", "imatinib"),
        _row("KEY-LINEZOLID", "linezolid"),
        _row("KEY-RITONAVIR", "ritonavir"),
        _row("KEY-CORE", "core compound"),
    ]
    raw_rows = [
        *(_row(row["canonical_ikey"], row["name"], raw=True) for row in legacy_rows),
        _row("KEY-NEW", "invalid first duplicate", "not-a-smiles", raw=True),
        _row("KEY-NEW", "valid second duplicate", "CCN", raw=True),
    ]
    _write_csv(legacy_path, legacy_rows)
    _write_csv(raw_path, raw_rows, raw=True)

    build_snapshot(
        raw_path=raw_path,
        legacy_path=legacy_path,
        output_path=output_path,
        provenance_path=provenance_path,
        expected_raw_sha256=_sha256(raw_path),
        core_rows=4,
        target_rows=5,
    )

    result = pd.read_csv(output_path, keep_default_na=False)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert result["canonical_ikey"].tolist()[:3] == [
        "KEY-IMATINIB",
        "KEY-LINEZOLID",
        "KEY-RITONAVIR",
    ]
    assert result["canonical_ikey"].tolist()[-1] == "KEY-NEW"
    assert result.iloc[-1]["name"] == "valid second duplicate"
    assert result.iloc[-1]["reframedb_url"].startswith("https://")
    assert not result["reframedb_url"].str.startswith("=HYPERLINK").any()
    assert provenance["raw_sha256"] == _sha256(raw_path)
    assert provenance["final_sha256"] == _sha256(output_path)
    assert provenance["final_row_count"] == 5


def test_build_snapshot_rejects_unapproved_raw_bytes(tmp_path):
    legacy_path = tmp_path / "legacy.csv"
    raw_path = tmp_path / "raw.csv"
    _write_csv(legacy_path, [_row("KEY-IMATINIB", "imatinib")])
    _write_csv(raw_path, [_row("KEY-NEW", "new", raw=True)], raw=True)

    with pytest.raises(ValueError, match="raw ReFRAME SHA-256 does not match"):
        build_snapshot(
            raw_path=raw_path,
            legacy_path=legacy_path,
            output_path=tmp_path / "snapshot.csv",
            provenance_path=tmp_path / "snapshot.json",
            expected_raw_sha256="0" * 64,
            core_rows=1,
            target_rows=2,
        )


@pytest.mark.parametrize(
    ("field", "unsafe_value", "expected_error"),
    (
        (
            "reframedb_url",
            '=WEBSERVICE("https://example.invalid")',
            "unsupported ReFRAME URL formula",
        ),
        ("name", "+unsafe-spreadsheet-value", "contains a spreadsheet formula"),
    ),
)
def test_build_snapshot_rejects_spreadsheet_formulas(
    tmp_path, field, unsafe_value, expected_error
):
    legacy_path = tmp_path / "legacy.csv"
    raw_path = tmp_path / "raw.csv"
    legacy_rows = [
        _row("KEY-IMATINIB", "imatinib"),
        _row("KEY-LINEZOLID", "linezolid"),
        _row("KEY-RITONAVIR", "ritonavir"),
        _row("KEY-CORE", "core compound"),
    ]
    _write_csv(legacy_path, legacy_rows)
    malformed = _row("KEY-NEW", "new", raw=True)
    malformed[field] = unsafe_value
    _write_csv(raw_path, [malformed], raw=True)

    with pytest.raises(ValueError, match=expected_error):
        build_snapshot(
            raw_path=raw_path,
            legacy_path=legacy_path,
            output_path=tmp_path / "snapshot.csv",
            provenance_path=tmp_path / "snapshot.json",
            expected_raw_sha256=_sha256(raw_path),
            core_rows=4,
            target_rows=5,
        )
```

- [ ] **Step 2: Run the builder tests and verify the intended red state**

Copy only the new test into the isolated verification tree and run it with the approved Python 3.12 environment:

```bash
/opt/homebrew/bin/brev copy tests/test_build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/verification /home/ubuntu/.venv/bin/python -m pytest /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/test_build_reframe_module1_snapshot.py -v"
```

Expected: collection fails because `scripts.build_reframe_module1_snapshot` does not exist.

- [ ] **Step 3: Implement the deterministic offline builder**

Create `scripts/build_reframe_module1_snapshot.py`:

```python
#!/usr/bin/env python3
"""Build the fixed Module 1 ReFRAME snapshot from an approved local export."""

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
APPROVED_RAW_SHA256 = (
    "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
)
CORE_ORDER_SHA256 = (
    "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
)
ANCHOR_TERMS = ("imatinib", "linezolid", "ritonavir")
ANCHOR_KEYS = ("KTUFNOKKBVMGRW", "TYZROVQLWOKYKF", "NCDNCNXCDXHOMX")
OUTPUT_COLUMNS = (
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
)
RAW_COLUMNS = set(OUTPUT_COLUMNS) - {"canonical_ikey"} | {"ikey"}
HYPERLINK = re.compile(r'^=HYPERLINK\("(?P<url>https://[^"\r\n]+)"\)$')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(values) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _plain_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("unsupported ReFRAME URL formula")
    value = value.strip()
    match = HYPERLINK.fullmatch(value)
    if match:
        return match.group("url")
    if value.startswith("https://"):
        return value
    raise ValueError("unsupported ReFRAME URL formula")


def _ordered_core(legacy_path: Path, core_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(legacy_path, keep_default_na=False)
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing or len(frame) != core_rows:
        raise ValueError("legacy teaching snapshot does not match its contract")
    frame = frame.drop_duplicates("canonical_ikey", keep="first").reset_index(drop=True)
    anchors = []
    for term in ANCHOR_TERMS:
        hit = frame[frame["name"].str.contains(term, case=False, regex=False)]
        if hit.empty:
            raise ValueError(f"legacy teaching snapshot is missing anchor {term}")
        anchors.append(hit.iloc[[0]])
    anchor_frame = pd.concat(anchors, ignore_index=True).drop_duplicates(
        "canonical_ikey"
    )
    remaining = frame[
        ~frame["canonical_ikey"].isin(anchor_frame["canonical_ikey"])
    ]
    sampled = remaining.sample(n=len(remaining), random_state=2026)
    ordered = pd.concat([anchor_frame, sampled], ignore_index=True)
    molecules = [Chem.MolFromSmiles(value) for value in ordered["smile"]]
    if any(molecule is None for molecule in molecules):
        raise ValueError("legacy teaching snapshot contains an invalid SMILES")
    return ordered.loc[:, OUTPUT_COLUMNS]


def _candidate_rows(raw_path: Path, excluded_keys: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, keep_default_na=False)
    missing = RAW_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"raw ReFRAME export is missing columns: {sorted(missing)}")
    raw = raw.rename(columns={"ikey": "canonical_ikey"})
    accepted = []
    accepted_keys = set()
    for record in raw.to_dict(orient="records"):
        key = str(record["canonical_ikey"]).strip()
        smiles = str(record["smile"]).strip()
        if not key or not smiles or key in excluded_keys or key in accepted_keys:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        record["canonical_ikey"] = key
        record["smile"] = smiles
        record["name"] = str(record["name"]).strip() or "unnamed compound"
        record["reframedb_url"] = _plain_url(record["reframedb_url"])
        accepted.append({column: record[column] for column in OUTPUT_COLUMNS})
        accepted_keys.add(key)
    return pd.DataFrame(accepted, columns=OUTPUT_COLUMNS).sort_values(
        "canonical_ikey", kind="stable", ignore_index=True
    )


def build_snapshot(
    *,
    raw_path: Path,
    legacy_path: Path,
    output_path: Path,
    provenance_path: Path,
    expected_raw_sha256: str = APPROVED_RAW_SHA256,
    core_rows: int = 96,
    target_rows: int = 10_000,
) -> None:
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("output paths must not already exist")
    observed_raw_sha256 = sha256_file(raw_path)
    if observed_raw_sha256 != expected_raw_sha256:
        raise ValueError("raw ReFRAME SHA-256 does not match the approved export")

    core = _ordered_core(legacy_path, core_rows)
    if core_rows == 96:
        if tuple(core["canonical_ikey"].iloc[:3]) != ANCHOR_KEYS:
            raise ValueError("legacy Module 1 anchors changed")
        if _sequence_sha256(core["canonical_ikey"]) != CORE_ORDER_SHA256:
            raise ValueError("legacy Module 1 row order changed")

    additions = _candidate_rows(raw_path, set(core["canonical_ikey"]))
    required_additions = target_rows - core_rows
    if required_additions < 0 or len(additions) < required_additions:
        raise ValueError("approved ReFRAME export has too few valid unique additions")
    result = pd.concat(
        [core, additions.iloc[:required_additions]], ignore_index=True
    ).loc[:, OUTPUT_COLUMNS]
    if len(result) != target_rows or not result["canonical_ikey"].is_unique:
        raise ValueError("generated Module 1 snapshot failed row-count or uniqueness checks")
    if not result[["smile", "canonical_ikey"]].apply(
        lambda column: column.map(lambda value: isinstance(value, str) and bool(value.strip()))
    ).all(axis=None):
        raise ValueError("generated Module 1 snapshot contains blank required values")
    if not result["reframedb_url"].str.startswith("https://").all():
        raise ValueError("generated Module 1 snapshot contains a non-HTTPS URL")
    if result.astype(str).apply(
        lambda column: column.str.startswith(("=", "+", "-", "@")).any()
    ).any():
        raise ValueError("generated Module 1 snapshot contains a spreadsheet formula")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, lineterminator="\n")
    with raw_path.open(encoding="utf-8") as raw_handle:
        raw_row_count = sum(1 for _ in raw_handle) - 1
    provenance = {
        "asset": output_path.name,
        "authorization": (
            "Written redistribution permission confirmed by the workshop organizer "
            "on 2026-08-24; the permission record is retained by the organizer."
        ),
        "core_order_sha256": _sequence_sha256(core["canonical_ikey"]),
        "curation_rules": [
            "preserve the current anchored 96-row Module 1 sequence",
            "exclude all 96 core identifiers",
            "retain the first RDKit-valid raw row per identifier in export order",
            "sort additions by canonical_ikey",
            "normalize exact HYPERLINK formulas to plain HTTPS URLs",
        ],
        "final_row_count": len(result),
        "final_sha256": sha256_file(output_path),
        "raw_row_count": raw_row_count,
        "raw_sha256": observed_raw_sha256,
        "rdkit_version": rdBase.rdkitVersion,
        "retrieved_date": RETRIEVED_DATE,
        "source_url": SOURCE_URL,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("legacy_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("provenance_path", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.error")
    build_snapshot(**vars(args))
    print(args.output_path)
    print(args.provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the builder unit tests and verify green**

```bash
/opt/homebrew/bin/brev copy scripts/build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/scripts/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/verification /home/ubuntu/.venv/bin/python -m pytest /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/test_build_reframe_module1_snapshot.py -v"
```

Expected: all builder tests pass.

- [ ] **Step 5: Write the real-asset acceptance test before generating the asset**

Create `tests/test_reframe_module1_snapshot.py`:

```python
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_teaching_snapshot.csv"
SNAPSHOT_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_module1_snapshot_10k.csv"
PROVENANCE_PATH = SNAPSHOT_PATH.with_suffix(".provenance.json")
LEGACY_BYTES_SHA256 = "d7ca4a132bfb18a9098e1e662c6e0d20fc522b4485f5f7366089cee287c18840"
CORE_ORDER_SHA256 = "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
RAW_SHA256 = "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
ANCHOR_KEYS = ("KTUFNOKKBVMGRW", "TYZROVQLWOKYKF", "NCDNCNXCDXHOMX")
REQUIRED_COLUMNS = {
    "smile",
    "canonical_ikey",
    "name",
    "source",
    "source_id",
    "status",
    "reframedb_url",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def test_module1_snapshot_asset_is_valid_reproducible_and_authorized():
    frame = pd.read_csv(SNAPSHOT_PATH, keep_default_na=False)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert _sha256(LEGACY_PATH) == LEGACY_BYTES_SHA256
    assert len(frame) == 10_000
    assert REQUIRED_COLUMNS <= set(frame.columns)
    assert frame["canonical_ikey"].is_unique
    assert frame["smile"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    assert frame["canonical_ikey"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all()
    assert frame["reframedb_url"].str.startswith("https://").all()
    assert not frame.astype(str).apply(
        lambda column: column.str.startswith(("=", "+", "-", "@")).any()
    ).any()
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in frame["smile"])
    assert tuple(frame["canonical_ikey"].iloc[:3]) == ANCHOR_KEYS
    assert _sequence_sha256(frame["canonical_ikey"].iloc[:96]) == CORE_ORDER_SHA256
    assert provenance["raw_sha256"] == RAW_SHA256
    assert provenance["final_row_count"] == 10_000
    assert provenance["final_sha256"] == _sha256(SNAPSHOT_PATH)
    assert "Written redistribution permission" in provenance["authorization"]
```

- [ ] **Step 6: Verify the asset test fails because the generated files do not exist**

```bash
/opt/homebrew/bin/brev copy tests/test_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/verification /home/ubuntu/.venv/bin/python -m pytest /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/test_reframe_module1_snapshot.py -v"
```

Expected: failure identifying the missing Module 1 snapshot.

- [ ] **Step 7: Reconfirm the exact remote target before curation**

The approved Step 0 contract still applies. Re-list before copying the public export or generating an asset:

```bash
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

Expected: the authorized instance identity and target hardware are visible. Do not touch another instance.

- [ ] **Step 8: Download and verify the fixed public export locally**

```bash
curl -fsSL --max-time 60 \
  https://reframedb.org/assets/csv/reframe_smiles_list.csv \
  -o /tmp/reframe_smiles_list.2026-08-24.csv
shasum -a 256 /tmp/reframe_smiles_list.2026-08-24.csv
```

Expected SHA-256:

```text
5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522
```

- [ ] **Step 9: Generate the asset twice in the approved environment**

After the contract and access gate pass, verify the remote user and persistent workspace, create only the task directory, and copy the three inputs:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "id -un"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "pwd"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "test -d /home/ubuntu/workspace"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "mkdir -p /home/ubuntu/workspace/codex/module1-10k-snapshot/input /home/ubuntu/workspace/codex/module1-10k-snapshot/build-a /home/ubuntu/workspace/codex/module1-10k-snapshot/build-b"
/opt/homebrew/bin/brev copy \
  scripts/build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/input/build_reframe_module1_snapshot.py
/opt/homebrew/bin/brev copy \
  notebooks/data/reframe_teaching_snapshot.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_teaching_snapshot.csv
/opt/homebrew/bin/brev copy \
  /tmp/reframe_smiles_list.2026-08-24.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_smiles_list.2026-08-24.csv
```

Run the builder twice with `/home/ubuntu/.venv/bin/python` and distinct output directories:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "/home/ubuntu/.venv/bin/python /home/ubuntu/workspace/codex/module1-10k-snapshot/input/build_reframe_module1_snapshot.py /home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_smiles_list.2026-08-24.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_teaching_snapshot.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/build-a/reframe_module1_snapshot_10k.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/build-a/reframe_module1_snapshot_10k.provenance.json"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "/home/ubuntu/.venv/bin/python /home/ubuntu/workspace/codex/module1-10k-snapshot/input/build_reframe_module1_snapshot.py /home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_smiles_list.2026-08-24.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/input/reframe_teaching_snapshot.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/build-b/reframe_module1_snapshot_10k.csv /home/ubuntu/workspace/codex/module1-10k-snapshot/build-b/reframe_module1_snapshot_10k.provenance.json"
mkdir -p /tmp/module1-build-a /tmp/module1-build-b
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/build-a/reframe_module1_snapshot_10k.csv \
  /tmp/module1-build-a/reframe_module1_snapshot_10k.csv
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/build-a/reframe_module1_snapshot_10k.provenance.json \
  /tmp/module1-build-a/reframe_module1_snapshot_10k.provenance.json
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/build-b/reframe_module1_snapshot_10k.csv \
  /tmp/module1-build-b/reframe_module1_snapshot_10k.csv
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/build-b/reframe_module1_snapshot_10k.provenance.json \
  /tmp/module1-build-b/reframe_module1_snapshot_10k.provenance.json
```

Run locally after the copies return:

```bash
cmp /tmp/module1-build-a/reframe_module1_snapshot_10k.csv \
    /tmp/module1-build-b/reframe_module1_snapshot_10k.csv
cmp /tmp/module1-build-a/reframe_module1_snapshot_10k.provenance.json \
    /tmp/module1-build-b/reframe_module1_snapshot_10k.provenance.json
mv /tmp/module1-build-a/reframe_module1_snapshot_10k.csv \
   notebooks/data/reframe_module1_snapshot_10k.csv
mv /tmp/module1-build-a/reframe_module1_snapshot_10k.provenance.json \
   notebooks/data/reframe_module1_snapshot_10k.provenance.json
```

Expected: both comparisons exit 0, and only the first verified output pair moves into `notebooks/data/`. Do not copy the raw export into the repository.

- [ ] **Step 10: Run the asset and builder tests in the approved environment**

Create an isolated verification tree and copy only the files needed by these two tests:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "mkdir -p /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/scripts /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/notebooks/data"
/opt/homebrew/bin/brev copy scripts/build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/scripts/
/opt/homebrew/bin/brev copy tests/test_build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/
/opt/homebrew/bin/brev copy tests/test_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/
/opt/homebrew/bin/brev copy notebooks/data/reframe_teaching_snapshot.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/notebooks/data/
/opt/homebrew/bin/brev copy notebooks/data/reframe_module1_snapshot_10k.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/notebooks/data/
/opt/homebrew/bin/brev copy notebooks/data/reframe_module1_snapshot_10k.provenance.json \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/verification/notebooks/data/
```

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/verification /home/ubuntu/.venv/bin/python -m pytest /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/test_build_reframe_module1_snapshot.py /home/ubuntu/workspace/codex/module1-10k-snapshot/verification/tests/test_reframe_module1_snapshot.py -v"
```

Expected: all tests pass.

- [ ] **Step 11: Commit the builder and immutable data asset**

```bash
git add -- \
  scripts/build_reframe_module1_snapshot.py \
  tests/test_build_reframe_module1_snapshot.py \
  tests/test_reframe_module1_snapshot.py \
  notebooks/data/reframe_module1_snapshot_10k.csv \
  notebooks/data/reframe_module1_snapshot_10k.provenance.json
git commit -m "feat: add deterministic Module 1 ReFRAME snapshot"
```

### Task 2: Add the isolated `snapshot_10k` loader source

**Files:**
- Modify: `notebooks/workshop_common.py:14-158`
- Modify: `tests/test_workshop_common.py:1-225`

- [ ] **Step 0: Sync one complete task-scoped checkout**

Run from the local repository root after Task 1 is committed. Require a clean local worktree, an absent remote destination, and then copy this checkout only into the task directory:

```bash
git status --short
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "test ! -e /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source"
/opt/homebrew/bin/brev copy ./ \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/
```

Expected: `git status --short` is empty, and the copy creates an isolated full checkout. Do not copy into `/home/ubuntu/nvmolkit-brev-notebook`.

- [ ] **Step 1: Pin legacy behavior and write failing source/range tests**

Add `import hashlib` to `tests/test_workshop_common.py`, then add:

```python
def _sequence_sha256(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def test_legacy_teaching_snapshot_remains_byte_for_byte_unchanged():
    assert hashlib.sha256(workshop_common.SNAPSHOT_PATH.read_bytes()).hexdigest() == (
        "d7ca4a132bfb18a9098e1e662c6e0d20fc522b4485f5f7366089cee287c18840"
    )


def test_legacy_module1_reference_order_remains_unchanged():
    frame = workshop_common.load_reframe(
        96,
        anchor_terms=("imatinib", "linezolid", "ritonavir"),
        random_state=2026,
        source="snapshot",
    )
    assert _sequence_sha256(frame["canonical_ikey"]) == (
        "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
    )


@pytest.mark.parametrize("source", [None, "", "auto", "Snapshot", True, 1])
def test_loader_accepts_only_explicit_supported_sources(source):
    with pytest.raises(
        ValueError,
        match=r"^source must be 'snapshot', 'snapshot_10k', or 'live'\.$",
    ):
        workshop_common.load_reframe(sample_size=1, source=source)


@pytest.mark.parametrize(
    "sample_size",
    [None, True, 96.0, "96", workshop_common.np.int64(96)],
)
def test_snapshot_10k_requires_a_builtin_integer_before_read(monkeypatch, sample_size):
    reads = []
    monkeypatch.setattr(workshop_common.pd, "read_csv", reads.append)
    with pytest.raises(
        TypeError,
        match=(
            r"^sample_size for source='snapshot_10k' "
            r"must be a built-in integer\.$"
        ),
    ):
        workshop_common.load_reframe(sample_size=sample_size, source="snapshot_10k")
    assert reads == []


@pytest.mark.parametrize("sample_size", [95, 10_001])
def test_snapshot_10k_rejects_sizes_outside_the_exercise_range(
    monkeypatch, sample_size
):
    reads = []
    monkeypatch.setattr(workshop_common.pd, "read_csv", reads.append)
    with pytest.raises(
        ValueError,
        match=(
            r"^source='snapshot_10k' supports sample_size values "
            r"from 96 through 10,000\.$"
        ),
    ):
        workshop_common.load_reframe(sample_size=sample_size, source="snapshot_10k")
    assert reads == []


def test_snapshot_10k_rejects_configured_csv_override_before_read(monkeypatch):
    reads = []
    monkeypatch.setattr(workshop_common.pd, "read_csv", reads.append)
    with pytest.raises(
        ValueError,
        match=(
            r"^source='snapshot_10k' cannot be combined with "
            r"use_configured_csv=True\.$"
        ),
    ):
        workshop_common.load_reframe(
            sample_size=96, source="snapshot_10k", use_configured_csv=True
        )
    assert reads == []
```

Replace the old invalid-source test rather than leaving two tests with conflicting messages.

- [ ] **Step 2: Run the focused tests and verify red**

```bash
/opt/homebrew/bin/brev copy tests/test_workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_common.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_common.py -k 'legacy or supported_sources or snapshot_10k_requires or outside_the_exercise_range or configured_csv_override' -v"
```

Expected: legacy pins pass; new-source tests fail because `snapshot_10k` is unsupported.

- [ ] **Step 3: Add source constants and exact size validation**

Add beside `SNAPSHOT_PATH` in `notebooks/workshop_common.py`:

```python
MODULE1_SNAPSHOT_PATH = (
    Path(__file__).with_name("data") / "reframe_module1_snapshot_10k.csv"
)
MODULE1_MIN_SAMPLE_SIZE = 96
MODULE1_MAX_SAMPLE_SIZE = 10_000
MODULE1_SNAPSHOT_INTEGRITY_ERROR = (
    "Bundled Module 1 ReFRAME snapshot failed its integrity check."
)
```

Add before `load_reframe`:

```python
def _module1_sample_size(value):
    if type(value) is not int:
        raise TypeError(
            "sample_size for source='snapshot_10k' must be a built-in integer."
        )
    if not MODULE1_MIN_SAMPLE_SIZE <= value <= MODULE1_MAX_SAMPLE_SIZE:
        raise ValueError(
            "source='snapshot_10k' supports sample_size values "
            "from 96 through 10,000."
        )
    return value
```

Change the source allow-list and add an explicit configured-source rejection:

```python
if not isinstance(source, str) or source not in {
    "snapshot",
    "snapshot_10k",
    "live",
}:
    raise ValueError("source must be 'snapshot', 'snapshot_10k', or 'live'.")
if type(use_configured_csv) is not bool:
    raise TypeError("use_configured_csv must be a bool.")
if source == "snapshot_10k" and use_configured_csv:
    raise ValueError(
        "source='snapshot_10k' cannot be combined with use_configured_csv=True."
    )
```

- [ ] **Step 4: Write failing deterministic-prefix and integrity tests**

Add to `tests/test_workshop_common.py`:

```python
def _module1_frame(row_count=10_000):
    return workshop_common.pd.DataFrame(
        {
            "smile": ["CCO"] * row_count,
            "canonical_ikey": [f"TESTKEY-{index:05d}" for index in range(row_count)],
            "name": [f"compound-{index}" for index in range(row_count)],
            "source": ["test"] * row_count,
            "source_id": [f"test-{index}" for index in range(row_count)],
            "status": ["available"] * row_count,
            "reframedb_url": [
                f"https://example.invalid/{index}" for index in range(row_count)
            ],
        }
    )


@pytest.mark.parametrize("sample_size", [96, 97, 512, 2_048, 9_999, 10_000])
def test_snapshot_10k_returns_the_requested_committed_prefix(
    monkeypatch, sample_size
):
    source_frame = _module1_frame()
    reads = []
    molecule = object()

    def fake_read_csv(source):
        reads.append(source)
        return source_frame.copy()

    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(workshop_common.Chem, "MolFromSmiles", lambda _value: molecule)
    monkeypatch.setenv("REFRAME_CSV", "https://hostile.invalid/ignored.csv")

    result = workshop_common.load_reframe(
        sample_size=sample_size,
        anchor_terms=("compound-9999",),
        random_state=7,
        source="snapshot_10k",
    )

    assert reads == [workshop_common.MODULE1_SNAPSHOT_PATH]
    assert len(result) == sample_size
    assert result["canonical_ikey"].tolist() == (
        source_frame["canonical_ikey"].iloc[:sample_size].tolist()
    )
    assert result["_mol"].tolist() == [molecule] * sample_size
    assert result.attrs == {"source": "bundled_snapshot_10k", "invalid_count": 0}


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_column",
        "wrong_row_count",
        "blank_smiles",
        "duplicate_key",
        "invalid_smiles",
        "formula_url",
        "formula_name",
        "read_failure",
    ),
)
def test_snapshot_10k_fails_closed_on_asset_corruption(monkeypatch, mutation):
    frame = _module1_frame()
    if mutation == "missing_column":
        frame = frame.drop(columns=["status"])
    elif mutation == "wrong_row_count":
        frame = frame.iloc[:-1].copy()
    elif mutation == "blank_smiles":
        frame.loc[0, "smile"] = ""
    elif mutation == "duplicate_key":
        frame.loc[1, "canonical_ikey"] = frame.loc[0, "canonical_ikey"]
    elif mutation == "formula_url":
        frame.loc[0, "reframedb_url"] = '=HYPERLINK("https://example.invalid")'
    elif mutation == "formula_name":
        frame.loc[0, "name"] = "+unsafe-spreadsheet-value"

    def fake_read_csv(_source):
        if mutation == "read_failure":
            raise OSError("private filesystem detail")
        return frame

    def fake_molecule(smiles):
        if mutation == "invalid_smiles" and smiles == "CCO":
            return None
        return object()

    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(workshop_common.Chem, "MolFromSmiles", fake_molecule)
    with pytest.raises(
        ValueError,
        match=r"^Bundled Module 1 ReFRAME snapshot failed its integrity check\.$",
    ):
        workshop_common.load_reframe(sample_size=96, source="snapshot_10k")
```

- [ ] **Step 5: Run prefix/integrity tests and verify red**

```bash
/opt/homebrew/bin/brev copy tests/test_workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_common.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_common.py -k 'snapshot_10k_returns or snapshot_10k_fails_closed' -v"
```

Expected: failures because the isolated loader does not exist.

- [ ] **Step 6: Implement the minimal isolated snapshot loader**

Add before `load_reframe`:

```python
def _load_module1_snapshot(requested_size):
    try:
        frame = pd.read_csv(MODULE1_SNAPSHOT_PATH)
    except Exception:
        raise ValueError(MODULE1_SNAPSHOT_INTEGRITY_ERROR) from None

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing or len(frame) != MODULE1_MAX_SAMPLE_SIZE:
        raise ValueError(MODULE1_SNAPSHOT_INTEGRITY_ERROR)
    valid_smiles = frame["smile"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    valid_keys = frame["canonical_ikey"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    valid_urls = frame["reframedb_url"].map(
        lambda value: isinstance(value, str) and value.startswith("https://")
    )
    has_formula_prefix = frame.astype(str).apply(
        lambda column: column.str.startswith(("=", "+", "-", "@")).any()
    ).any()
    if (
        not valid_smiles.all()
        or not valid_keys.all()
        or not valid_urls.all()
        or has_formula_prefix
        or frame["canonical_ikey"].duplicated().any()
    ):
        raise ValueError(MODULE1_SNAPSHOT_INTEGRITY_ERROR)

    try:
        molecules = [Chem.MolFromSmiles(smiles) for smiles in frame["smile"]]
    except Exception:
        raise ValueError(MODULE1_SNAPSHOT_INTEGRITY_ERROR) from None
    if any(molecule is None for molecule in molecules):
        raise ValueError(MODULE1_SNAPSHOT_INTEGRITY_ERROR)

    frame = frame.copy()
    frame["name"] = frame["name"].fillna("unnamed compound")
    result = frame.iloc[:requested_size].copy().reset_index(drop=True)
    result["_mol"] = molecules[:requested_size]
    result.attrs.update(source="bundled_snapshot_10k", invalid_count=0)
    return result
```

In `load_reframe`, after source/boolean/combination validation and before the existing `_positive_integer` call, add:

```python
if source == "snapshot_10k":
    return _load_module1_snapshot(_module1_sample_size(sample_size))
```

Do not change the remaining legacy snapshot, configured CSV, or live-source code.

- [ ] **Step 7: Run loader tests and verify green**

```bash
/opt/homebrew/bin/brev copy notebooks/workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/workshop_common.py
/opt/homebrew/bin/brev copy tests/test_workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_common.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_common.py -v"
```

Expected: all loader and memory tests pass.

- [ ] **Step 8: Commit the isolated loader**

```bash
git add -- notebooks/workshop_common.py tests/test_workshop_common.py
git commit -m "feat: load nested Module 1 samples through 10k"
```

### Task 3: Update only the Module 1 attendee exercise

**Files:**
- Modify: `notebooks/01_direct_nvmolkit_reframe.ipynb` existing cells only
- Modify: `tests/test_workshop_notebook_inventory.py:278-336,426-489`

- [ ] **Step 1: Write failing notebook-source tests**

Update `test_module1_default_and_advanced_controls_are_bounded` to require:

```python
assert 'DATA_SOURCE = "snapshot_10k"' in source
assert "SAMPLE_SIZE = 96" in source
assert "FP_BITS = 1024" in source
assert "ADVANCED_LARGE_RUN = False" in source
assert "ADVANCED_SAMPLE_SIZE = 10_000" in source
assert 'source="live"' in source
```

Add:

```python
def test_module1_sample_size_exercise_has_one_supported_rerun_boundary():
    step1 = _module1_cell_source("cell-e51239292d22")
    step3 = _module1_cell_source("cell-7dabc4a334bf")
    step4 = _module1_cell_source("cell-00a63d6c51e3")

    exercise = step1 + step3
    for value in ("512", "2_048", "10_000"):
        assert value in exercise
    assert "Change only <code>SAMPLE_SIZE</code>" in step3
    assert "rerun Steps 1 through 3" in step3
    assert "restore `SAMPLE_SIZE = 96`" in step4
    assert "Change <code>SAMPLE_SIZE</code> and observe" not in step4


def test_module1_steps_1_through_3_keep_a_three_by_n_workload():
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    end = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.id == "cell-c5e8c0433cbd"
    )
    source = "\n".join(
        cell.source for cell in notebook.cells[: end + 1] if cell.cell_type == "code"
    )
    assert "comparison_count = len(anchor_molecules) * len(molecules)" in source
    assert "crossTanimotoSimilarity(anchor_fps, fingerprints)" in source
    assert not re.search(
        r"(?m)^\s*require_bounded_condensed_distances\(", source
    )
    assert "Butina.ClusterData" not in source
    assert "np.empty(n_items * (n_items - 1) // 2" not in source
```

- [ ] **Step 2: Run the source tests and verify red**

```bash
/opt/homebrew/bin/brev copy tests/test_workshop_notebook_inventory.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_notebook_inventory.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_notebook_inventory.py -k 'default_and_advanced_controls or sample_size_exercise or three_by_n' -v"
```

Expected: source-control and wording tests fail; the allocation-boundary test passes.

- [ ] **Step 3: Patch only the approved Module 1 cells**

Use `apply_patch` against the notebook JSON. Keep all cell IDs, tags, metadata, null execution counts, and empty outputs.

In `cell-f38b80a4df5a`, replace only its final paragraph with:

```text
This notebook imports shared ReFRAME loading and descriptor helpers from `workshop_common.py` and fused Butina compatibility helpers from `nvmolkit_compat.py`; keep both helper files and both fixed data files beside the workshop notebooks. `data/reframe_teaching_snapshot.csv` preserves the 96-row inputs for Modules 2 and 3. `data/reframe_module1_snapshot_10k.csv` supplies the nested local Module 1 scaling exercise.
```

Replace `cell-e51239292d22` with:

```text
## Step 1 — Load and quality-check ReFRAME

The reference run uses the first 96 rows of a fixed, bundled 10,000-compound Module 1 snapshot. The source is local and makes no network request. Larger samples are nested, so each requested sample contains the smaller one.

<div class="task"><b>Your edit:</b> Run the 96-row reference first. Then try <code>SAMPLE_SIZE = 512</code>, <code>2_048</code>, and <code>10_000</code>. Change only <code>SAMPLE_SIZE</code> and rerun Steps 1 through 3.</div>
```

In `cell-9f1999dd251d`, change only:

```python
DATA_SOURCE = "snapshot_10k"
```

Keep `SAMPLE_SIZE = 96`.

In `cell-7dabc4a334bf`, replace only list item 2 with the following text. Preserve list items 1 and 3:

```text
<li>Change only <code>SAMPLE_SIZE</code> in Step 1 to <code>512</code>, <code>2_048</code>, or <code>10_000</code>, then rerun Steps 1 through 3. How does the speed ratio change?</li>
```

In `cell-00a63d6c51e3`, add before the existing explanation:

```text
Before continuing, restore `SAMPLE_SIZE = 96` in Step 1 and rerun Steps 1 through 3. The RDKit reference below constructs a quadratic condensed-distance vector and intentionally remains bounded.
```

Replace its third task with:

```text
<li>Keep <code>SAMPLE_SIZE = 96</code> for this bounded CPU/GPU clustering comparison.</li>
```

In `cell-7d83af67d461`, add this bullet before the existing scientific takeaways. Do not change the advanced code cells:

```text
- The fixed 10,000-compound Module 1 snapshot supports the reproducible scaling exercise. The later live-export cell is optional, GPU-only, and can change when ReFRAME changes.
```

- [ ] **Step 4: Run notebook structure and source tests**

```bash
/opt/homebrew/bin/brev copy notebooks/01_direct_nvmolkit_reframe.ipynb \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/01_direct_nvmolkit_reframe.ipynb
/opt/homebrew/bin/brev copy tests/test_workshop_notebook_inventory.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_notebook_inventory.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_notebook_inventory.py -k 'json_structure or default_and_advanced_controls or sample_size_exercise or three_by_n or advanced' -v"
```

Expected: all selected tests pass and the stored notebook remains clean.

- [ ] **Step 5: Commit the attendee-facing notebook change**

```bash
git add -- notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_notebook_inventory.py
git commit -m "fix: support Module 1 Step 3 scaling through 10k"
```

### Task 4: Execute Steps 1 through 3 at representative sizes

**Files:**
- Modify: `tests/test_workshop_notebook_inventory.py:1-489`

- [ ] **Step 1: Add the clean-kernel prefix execution test**

Add `import os` at the top, then add:

```python
@pytest.mark.parametrize("sample_size", [96, 512, 2_048, 10_000])
def test_module1_steps_1_through_3_execute_at_supported_sizes(
    monkeypatch, tmp_path, sample_size
):
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    end = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.id == "cell-c5e8c0433cbd"
    )
    notebook.cells = notebook.cells[: end + 1]
    sample_cell = next(cell for cell in notebook.cells if cell.id == "cell-9f1999dd251d")
    assert sample_cell.source.count("SAMPLE_SIZE = 96") == 1
    sample_cell.source = sample_cell.source.replace(
        "SAMPLE_SIZE = 96", f"SAMPLE_SIZE = {sample_size}", 1
    )

    setup_index = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.id == "cell-a5ae8306d03b"
    )
    notebook.cells.insert(
        setup_index + 1,
        nbformat.v4.new_code_cell(
            """\
_original_read_csv = pd.read_csv
_blocked_network_attempts = []
def _local_only_read_csv(source, *args, **kwargs):
    if str(source).startswith(("http://", "https://")):
        _blocked_network_attempts.append(str(source))
        raise AssertionError("Module 1 scaling attempted network access")
    return _original_read_csv(source, *args, **kwargs)
pd.read_csv = _local_only_read_csv
if NVMOLKIT_READY:
    torch.cuda.reset_peak_memory_stats()
""",
            id="test-module1-scale-preflight",
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """\
import json
import resource
import sys
assert DATA_SOURCE == "snapshot_10k"
assert reframe.attrs["source"] == "bundled_snapshot_10k"
assert len(reframe) == SAMPLE_SIZE
assert len(characterized) == SAMPLE_SIZE
assert FINGERPRINT_BATCH_SIZE == SAMPLE_SIZE
assert len(anchors) == 3
assert comparison_count == 3 * SAMPLE_SIZE
assert rdkit_similarity.shape == (3, SAMPLE_SIZE)
assert similarity.shape == (3, SAMPLE_SIZE)
assert fingerprint_runtime["seconds"].gt(0).all()
assert similarity_runtime["seconds"].gt(0).all()
assert _blocked_network_attempts == []
expected_backends = {"RDKit CPU"}
if NVMOLKIT_READY:
    expected_backends.add("nvMolKit GPU")
assert set(fingerprint_runtime["backend"]) == expected_backends
assert set(similarity_runtime["backend"]) == expected_backends
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
receipt = {
    "sample_size": SAMPLE_SIZE,
    "comparison_count": comparison_count,
    "similarity_shape": list(similarity.shape),
    "fingerprint_backends": fingerprint_runtime["backend"].tolist(),
    "similarity_backends": similarity_runtime["backend"].tolist(),
    "elapsed_seconds": time.perf_counter() - MODULE1_STARTED,
    "peak_rss_bytes": peak_rss_bytes,
    "cuda_peak_allocated_bytes": (
        torch.cuda.max_memory_allocated() if NVMOLKIT_READY else None
    ),
    "cuda_peak_reserved_bytes": (
        torch.cuda.max_memory_reserved() if NVMOLKIT_READY else None
    ),
}
print("MODULE1_SCALE_RECEIPT_JSON=" + json.dumps(receipt, sort_keys=True))
""",
            id="test-module1-scale-receipt",
        )
    )

    matplotlib_dir = tmp_path / f"matplotlib-{sample_size}"
    matplotlib_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_dir))
    monkeypatch.setenv(
        "REFRAME_CSV", "https://hostile.invalid/reframe.csv?token=do-not-disclose"
    )
    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})

    stream_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    match = re.search(
        r"^MODULE1_SCALE_RECEIPT_JSON=(\{.*\})$", stream_text, re.MULTILINE
    )
    assert match is not None
    receipt = json.loads(match.group(1))
    assert receipt["sample_size"] == sample_size
    assert receipt["similarity_shape"] == [3, sample_size]
    if os.environ.get("RUN_GPU_TESTS") == "1":
        assert receipt["fingerprint_backends"] == ["RDKit CPU", "nvMolKit GPU"]
        assert receipt["similarity_backends"] == ["RDKit CPU", "nvMolKit GPU"]
        assert receipt["cuda_peak_allocated_bytes"] > 0
        assert receipt["cuda_peak_reserved_bytes"] > 0
```

- [ ] **Step 2: Update the default full-notebook receipt expectation**

In `test_module1_default_executes_without_network_and_emits_report`, change only:

```python
assert report["source"] == "bundled_snapshot_10k"
```

Keep the expected default row count at 96.

- [ ] **Step 3: Run representative prefix execution on the approved instance**

```bash
/opt/homebrew/bin/brev copy tests/test_workshop_notebook_inventory.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_notebook_inventory.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_notebook_inventory.py -k 'steps_1_through_3_execute_at_supported_sizes' -vv"
```

Expected: four passes, exact row counts, and no network attempts. This run checks behavior; Step 4 adds the strict GPU-evidence assertions.

- [ ] **Step 4: Run the GPU-gated form on the target L4**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_notebook_inventory.py -k 'steps_1_through_3_execute_at_supported_sizes' -vv -s"
```

Expected: four passes; every receipt includes both `RDKit CPU` and `nvMolKit GPU`. Retain the 10,000-row receipt containing elapsed time, peak RSS, and CUDA peak allocated/reserved bytes.

- [ ] **Step 5: Commit the repeatable scale acceptance**

```bash
git add -- tests/test_workshop_notebook_inventory.py
git commit -m "test: execute Module 1 scaling through 10k"
```

### Task 5: Run regression gates and independent reviews

**Files:**
- Verify all changed files
- Do not modify protected Modules 2 and 3 or the legacy snapshot

- [ ] **Step 1: Prove protected files are byte-for-byte unchanged**

```bash
git diff --exit-code 513232a -- \
  notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  notebooks/data/reframe_teaching_snapshot.csv \
  launchable
git diff --exit-code 513232a -- \
  . \
  ':(exclude)docs/superpowers/plans/2026-08-24-module1-10k-snapshot.md' \
  ':(exclude)scripts/build_reframe_module1_snapshot.py' \
  ':(exclude)tests/test_build_reframe_module1_snapshot.py' \
  ':(exclude)tests/test_reframe_module1_snapshot.py' \
  ':(exclude)notebooks/data/reframe_module1_snapshot_10k.csv' \
  ':(exclude)notebooks/data/reframe_module1_snapshot_10k.provenance.json' \
  ':(exclude)notebooks/workshop_common.py' \
  ':(exclude)tests/test_workshop_common.py' \
  ':(exclude)notebooks/01_direct_nvmolkit_reframe.ipynb' \
  ':(exclude)tests/test_workshop_notebook_inventory.py'
git diff --name-only 513232a
```

Expected: both `--exit-code` checks produce no output. The final command lists only the ten exact allowed paths above. This proves that Modules 2 and 3, the old snapshot, the setup/key system, all Launchable files, and every unrelated path are unchanged.

- [ ] **Step 2: Run focused deterministic tests**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_build_reframe_module1_snapshot.py tests/test_reframe_module1_snapshot.py tests/test_workshop_common.py tests/test_workshop_notebook_inventory.py -v"
```

Expected: all tests pass. A non-GPU run is not GPU evidence.

- [ ] **Step 3: Run Module 2 and 3 regressions**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_notebook_execution.py tests/test_workshop_llm_agent.py -v"
```

Expected: all tests pass and both notebooks retain their fixed 96-row contracts.

- [ ] **Step 4: Run the complete deterministic suite**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest -q"
git diff --check
git status --short
```

Expected: the suite passes, `git diff --check` reports no problems, and the worktree contains no uncommitted implementation files.

- [ ] **Step 5: Run independent specification review**

Dispatch a fresh reviewer with the approved spec and implementation diff. Require it to check every requirement, especially the 96–10,000 range, nested prefixes, no network, exact three-anchor Step 3 workload, and target-only scope. Fix any blocking finding and rerun focused tests.

- [ ] **Step 6: Run independent code-quality and scientific-UX review**

Dispatch a different reviewer. Require it to check data provenance, duplicate resolution, formula neutralization, error disclosure, notebook clarity for an ACS chemist, performance-test claim boundaries, and absence of unrelated changes. Record lower-severity findings as residual risks; fix blocking findings only.

- [ ] **Step 7: Commit review fixes, if any**

If a blocking review fix changes an approved source or test file, copy the exact approved implementation set to `full-source` and rerun the complete remote suite before committing:

```bash
/opt/homebrew/bin/brev copy scripts/build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/scripts/build_reframe_module1_snapshot.py
/opt/homebrew/bin/brev copy tests/test_build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_build_reframe_module1_snapshot.py
/opt/homebrew/bin/brev copy tests/test_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_reframe_module1_snapshot.py
/opt/homebrew/bin/brev copy notebooks/data/reframe_module1_snapshot_10k.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/data/reframe_module1_snapshot_10k.csv
/opt/homebrew/bin/brev copy notebooks/data/reframe_module1_snapshot_10k.provenance.json \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/data/reframe_module1_snapshot_10k.provenance.json
/opt/homebrew/bin/brev copy notebooks/workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/workshop_common.py
/opt/homebrew/bin/brev copy tests/test_workshop_common.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_common.py
/opt/homebrew/bin/brev copy notebooks/01_direct_nvmolkit_reframe.ipynb \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/notebooks/01_direct_nvmolkit_reframe.ipynb
/opt/homebrew/bin/brev copy tests/test_workshop_notebook_inventory.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source/tests/test_workshop_notebook_inventory.py
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-snapshot/full-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-snapshot/full-source /home/ubuntu/.venv/bin/python -m pytest -q"
```

Do not add a cache, key-management abstraction, renderer feature, or unrelated cleanup in this cycle. Record lower-severity findings, including repeated parsing of the 10,000-row file, as residual risks unless target acceptance shows an attendee-blocking delay.

```bash
git add -- \
  docs/superpowers/plans/2026-08-24-module1-10k-snapshot.md \
  scripts/build_reframe_module1_snapshot.py \
  tests/test_build_reframe_module1_snapshot.py \
  tests/test_reframe_module1_snapshot.py \
  notebooks/data/reframe_module1_snapshot_10k.csv \
  notebooks/data/reframe_module1_snapshot_10k.provenance.json \
  notebooks/workshop_common.py \
  tests/test_workshop_common.py \
  notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_notebook_inventory.py
git commit -m "fix: address Module 1 10k review findings"
```

If no files changed, do not create an empty commit.

### Task 6: Publish and validate only the target Launchable

**Files and external scope:**
- Publish only the reviewed repository branch used by Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`
- Update no setup script, credential, port, compute setting, or other Launchable

- [ ] **Step 1: Reconfirm the user-operated saved-source gate**

The user reopens only `nvMolKit + Nemotron Notebook` with Launchable ID `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`. Record the configured repository URL, source branch, and fixed-commit setting. Exit without saving.

Expected: repository `https://github.com/ktretina/nvmolkit-brev-notebook.git`, branch `main`, and no fixed old commit. If any value differs, stop and amend this plan; do not infer, save, or inspect another Launchable.

- [ ] **Step 2: Obtain explicit publication approval**

Present the reviewed commit list, tests, independent reviews, and confirmed target source branch. Ask before pushing or opening a draft pull request. Opening the draft does not authorize merge or a Console edit.

- [ ] **Step 3: Publish without force after approval**

Push the feature branch:

```bash
git push -u origin fix/module1-10k-snapshot
gh pr list --repo ktretina/nvmolkit-brev-notebook \
  --state open --base main --head fix/module1-10k-snapshot
```

If the list is empty, open one draft pull request only:

```bash
gh pr create --repo ktretina/nvmolkit-brev-notebook \
  --base main --head fix/module1-10k-snapshot --draft \
  --title "Support Module 1 scaling through 10k" \
  --body "Adds one deterministic Module-1-only ReFRAME snapshot, an isolated snapshot_10k loader path, bounded attendee instructions, and clean-kernel acceptance at 96, 512, 2,048, and 10,000. Modules 2 and 3, the legacy snapshot, setup/key handling, and all other Launchables are unchanged."
```

Do not create a duplicate PR. Require green checks and review. Obtain separate approval before marking it ready or merging. Merge without force and without deleting the feature branch:

```bash
gh pr ready --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot
gh pr checks --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot
gh pr merge --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot --merge
```

- [ ] **Step 4: Confirm future target deployments resolve the reviewed commit**

If the confirmed target follows `main`, do not edit its Console definition. Verify `origin/main` equals the reviewed merge commit. If Step 1 instead found a pin, this plan is blocked and must be amended before any Console save; no source-reference edit is authorized by this plan.

- [ ] **Step 5: Obtain explicit cost approval for one fresh target deployment**

Use only organization `agents-in-ls` and only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`. Do not start or create a paid instance before approval. After approval, the user opens exactly:

```text
https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW
```

This is user-operated because the installed CLI exposes no verified Launchable-deployment command and this task has no supported authenticated Console control. The user sets the instance name to `nvmolkit-module1-10k-20260824`, confirms one NVIDIA L4 and 75 GiB storage, changes no Setup value or other field, and selects **Deploy Launchable** once. The user then provides the resulting Access URL. Do not retry an ambiguous deployment.

- [ ] **Step 6: Verify the fresh instance identity and source**

After deployment, record the exact instance name and ID. Verify:

```bash
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook rev-parse HEAD"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "/home/ubuntu/.venv/bin/python --version"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "curl -fsS http://127.0.0.1:8888/api"
```

Expected: HEAD equals the reviewed merge commit, the checkout is clean, and Python is 3.12. Also verify the committed snapshot/provenance hashes, NVIDIA L4 identity, nvMolKit 0.6.0, CUDA availability, and Jupyter `/api` health.

- [ ] **Step 7: Run the complete target GPU suite**

```bash
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "env -C /home/ubuntu/nvmolkit-brev-notebook PYTHONPATH=/home/ubuntu/nvmolkit-brev-notebook RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python -m pytest -q"
```

Expected: the suite passes. Record the exact count, duration, warnings, GPU, CUDA, nvMolKit version, and Module 1 10,000-row scale receipt. Do not reuse the earlier `971 passed, 1 skipped` result.

- [ ] **Step 8: Perform the user-operated fresh browser acceptance**

Open the target instance's organization-only Jupyter Secure Link and navigate to:

```text
lab/tree/nvmolkit-brev-notebook/notebooks/01_direct_nvmolkit_reframe.ipynb
```

The user restarts the kernel and runs Steps 1 through 3 at 96. The user changes only `SAMPLE_SIZE` to `512`, `2_048`, and then `10_000`, rerunning Steps 1 through 3 each time. At each size require the matching `Valid unique structures` and `Fingerprint batch size`, three anchors, both RDKit and nvMolKit runtime rows, one observed runtime ratio, and no network or loader error. At 10,000 specifically require:

- `Valid unique structures: 10,000`;
- `Fingerprint batch size: 10,000`;
- three anchors;
- both RDKit and nvMolKit runtime rows;
- one observed runtime ratio;
- no network or loader error.

Restore `SAMPLE_SIZE = 96` before Step 4. Record the browser result separately from automated GPU evidence.

- [ ] **Step 9: Confirm the saved target update and provide the notebook URL**

Only after the browser pass, report that future deployments of the target Launchable resolve the reviewed source. Give the user the exact Jupyter Secure Link URL for the notebook. Do not claim that any other Launchable changed.

- [ ] **Step 10: Request lifecycle approval for the exact test instance**

Ask before stopping the new paid instance. If approved, stop only the recorded instance and verify its final state. Do not delete it or act on another environment.

```bash
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev stop nvmolkit-module1-10k-20260824
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

---

## Completion evidence

The work is complete only when all of the following are true:

- the public repository contains the approved 10,000-row asset and provenance;
- the old 96-row asset and Modules 2 and 3 are unchanged;
- loader and notebook tests pass for the specified boundaries;
- the four representative Step 1–3 runs pass on the target L4;
- independent specification and quality reviews have no blocking findings;
- the reviewed commit is the source for future deployments of only the target Launchable;
- one fresh target browser run passes at `SAMPLE_SIZE = 96`, `512`, `2_048`, and `10_000`;
- the user receives the direct notebook test URL.
