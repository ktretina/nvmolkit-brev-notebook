# Module 1 10,000-Molecule Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an attendee change only `SAMPLE_SIZE` in Module 1 and execute Steps 1 through 3 successfully for every integer from 96 through 10,000.

**Architecture:** Commit one deterministic 10,000-row ReFRAME asset for Module 1 and add one explicit `snapshot_10k` loader branch that returns nested prefixes. Keep the legacy 96-row asset and Modules 2 and 3 unchanged. Validate once on an existing approved L4, review, then run one fresh target-Launchable acceptance.

**Tech Stack:** Python 3.12, pandas 2.3.1, RDKit through nvMolKit 0.6.0, nbformat/nbconvert, pytest 8.4.1, Git, GitHub, NVIDIA Brev.

---

## Scope and file map

- Create `scripts/build_reframe_module1_snapshot.py`: one-purpose offline builder for the approved ReFRAME export.
- Create `notebooks/data/reframe_module1_snapshot_10k.csv`: fixed 10,000-row Module 1 asset.
- Create `notebooks/data/reframe_module1_snapshot_10k.provenance.json`: source, hashes, curation rules, and redistribution basis.
- Create `tests/test_reframe_module1_snapshot.py`: acceptance test for the committed asset.
- Modify `notebooks/workshop_common.py`: add only `source="snapshot_10k"`.
- Modify `tests/test_workshop_common.py`: test range, nested prefixes, local-only loading, and legacy behavior.
- Modify `notebooks/01_direct_nvmolkit_reframe.ipynb`: change only existing Module 1 cells.
- Modify `tests/test_workshop_notebook_inventory.py`: test instructions and clean-kernel execution at four sizes.

Do not modify the legacy snapshot, Modules 2 or 3, setup or key handling, the renderer, the hosted model, Step 4's memory guard, or another Launchable.

## Required gates

1. The user confirms in the Brev Console that only Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` uses repository `https://github.com/ktretina/nvmolkit-brev-notebook.git`, branch `main`, with no fixed older commit. Do not save. Stop if this differs.
2. The user confirms this is the only active Brev CLI task for the session. Before each Brev command batch, compare `brev ls --json` with `brev ls --org agents-in-ls --json`; both must resolve the same intended instance. If the active organization differs, stop and ask the user. Do not run `brev set` or `brev refresh`.
3. Use retained instance `nvmolkit---nemotron-notebook-80f439` (`abnc4jxm4`) only after the user reconfirms exclusive use and approves resumed billing if it is stopped. Work only in `/home/ubuntu/workspace/codex/module1-10k-lean`.
4. Pushing, opening or merging a PR, deploying, and stopping an instance each require the stated approval. Console and browser steps are user-operated.

### Task 1: Build and commit the fixed asset

**Files:**
- Create: `scripts/build_reframe_module1_snapshot.py`
- Create: `notebooks/data/reframe_module1_snapshot_10k.csv`
- Create: `notebooks/data/reframe_module1_snapshot_10k.provenance.json`
- Create: `tests/test_reframe_module1_snapshot.py`
- Read only: `notebooks/data/reframe_teaching_snapshot.csv`

- [ ] **Step 1: Write the failing asset acceptance test**

Create `tests/test_reframe_module1_snapshot.py`:

```python
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_teaching_snapshot.csv"
SNAPSHOT_PATH = (
    REPO_ROOT / "notebooks" / "data" / "reframe_module1_snapshot_10k.csv"
)
PROVENANCE_PATH = SNAPSHOT_PATH.with_suffix(".provenance.json")
LEGACY_SHA256 = "d7ca4a132bfb18a9098e1e662c6e0d20fc522b4485f5f7366089cee287c18840"
RAW_SHA256 = "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
CORE_ORDER_SHA256 = (
    "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
)
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


def test_module1_snapshot_is_valid_fixed_and_authorized():
    frame = pd.read_csv(SNAPSHOT_PATH, keep_default_na=False, dtype=str)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert _sha256(LEGACY_PATH) == LEGACY_SHA256
    assert len(frame) == 10_000
    assert REQUIRED_COLUMNS <= set(frame.columns)
    assert frame["canonical_ikey"].is_unique
    assert frame["smile"].str.strip().ne("").all()
    assert frame["canonical_ikey"].str.strip().ne("").all()
    assert frame["reframedb_url"].str.startswith("https://").all()
    assert not frame["reframedb_url"].str.startswith("=").any()
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in frame["smile"])
    assert tuple(frame["canonical_ikey"].iloc[:3]) == ANCHOR_KEYS
    assert _sequence_sha256(frame["canonical_ikey"].iloc[:96]) == CORE_ORDER_SHA256
    assert provenance["raw_sha256"] == RAW_SHA256
    assert provenance["final_row_count"] == 10_000
    assert provenance["final_sha256"] == _sha256(SNAPSHOT_PATH)
    assert "Written redistribution permission" in provenance["authorization"]
```

- [ ] **Step 2: Verify the test is red in the approved Python environment**

First run only the read-only checks:

```bash
/opt/homebrew/bin/brev --version
/opt/homebrew/bin/brev ls --help
/opt/homebrew/bin/brev start --help
/opt/homebrew/bin/brev stop --help
/opt/homebrew/bin/brev exec --help
/opt/homebrew/bin/brev copy --help
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

Confirm that both lists resolve retained instance ID `abnc4jxm4` in
`agents-in-ls`. Stop on any difference. After the Required gates pass and any
required start is approved, run:

```bash
/opt/homebrew/bin/brev start nvmolkit---nemotron-notebook-80f439
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "mkdir -p /home/ubuntu/workspace/codex/module1-10k-lean/data-source/tests /home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data /home/ubuntu/workspace/codex/module1-10k-lean/data-source/scripts"
/opt/homebrew/bin/brev copy tests/test_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/tests/
/opt/homebrew/bin/brev copy notebooks/data/reframe_teaching_snapshot.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-lean/data-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-lean/data-source /home/ubuntu/.venv/bin/python -m pytest tests/test_reframe_module1_snapshot.py -v"
```

If the instance is already running, omit `brev start`. Expected: failure because `reframe_module1_snapshot_10k.csv` is absent. Stop on any organization, instance ID, L4, ownership, or Python 3.12 mismatch.

- [ ] **Step 3: Implement the one-purpose builder**

Create `scripts/build_reframe_module1_snapshot.py`:

```python
#!/usr/bin/env python3
"""Build the fixed Module 1 ReFRAME snapshot from one approved export."""

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
RAW_SHA256 = "5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522"
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _plain_url(value):
    value = str(value).strip()
    match = HYPERLINK.fullmatch(value)
    if match:
        return match.group("url")
    if value.startswith("https://"):
        return value
    raise ValueError("unsupported ReFRAME page-link formula")


def _ordered_core(legacy_path):
    frame = pd.read_csv(legacy_path, keep_default_na=False, dtype=str)
    if len(frame) != 96 or set(OUTPUT_COLUMNS) - set(frame.columns):
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
    ordered = pd.concat(
        [anchor_frame, remaining.sample(n=len(remaining), random_state=2026)],
        ignore_index=True,
    ).loc[:, OUTPUT_COLUMNS]
    if tuple(ordered["canonical_ikey"].iloc[:3]) != ANCHOR_KEYS:
        raise ValueError("legacy Module 1 anchors changed")
    if _sequence_sha256(ordered["canonical_ikey"]) != CORE_ORDER_SHA256:
        raise ValueError("legacy Module 1 row order changed")
    return ordered


def build_snapshot(raw_path, legacy_path, output_path, provenance_path):
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("output paths must not already exist")
    if _sha256(raw_path) != RAW_SHA256:
        raise ValueError("raw ReFRAME SHA-256 does not match the approved export")

    core = _ordered_core(legacy_path)
    raw = pd.read_csv(raw_path, keep_default_na=False, dtype=str)
    missing = RAW_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"raw ReFRAME export is missing columns: {sorted(missing)}")
    raw = raw.rename(columns={"ikey": "canonical_ikey"})

    excluded = set(core["canonical_ikey"])
    accepted_keys = set()
    additions = []
    for row in raw.to_dict(orient="records"):
        key = row["canonical_ikey"].strip()
        smiles = row["smile"].strip()
        if not key or not smiles or key in excluded or key in accepted_keys:
            continue
        if Chem.MolFromSmiles(smiles) is None:
            continue
        record = {column: str(row[column]).strip() for column in OUTPUT_COLUMNS}
        record["canonical_ikey"] = key
        record["smile"] = smiles
        record["name"] = record["name"] or "unnamed compound"
        record["reframedb_url"] = _plain_url(record["reframedb_url"])
        additions.append(record)
        accepted_keys.add(key)

    additions_frame = pd.DataFrame(additions, columns=OUTPUT_COLUMNS).sort_values(
        "canonical_ikey", kind="stable", ignore_index=True
    )
    if len(additions_frame) < 9_904:
        raise ValueError("approved export has too few valid unique additions")
    result = pd.concat([core, additions_frame.iloc[:9_904]], ignore_index=True)
    if len(result) != 10_000 or not result["canonical_ikey"].is_unique:
        raise ValueError("generated snapshot failed row-count or uniqueness checks")
    if not result["reframedb_url"].str.startswith("https://").all():
        raise ValueError("generated snapshot contains a non-HTTPS page link")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, lineterminator="\n")
    provenance = {
        "asset": output_path.name,
        "authorization": (
            "Written redistribution permission confirmed by the workshop organizer "
            "on 2026-08-24; the permission record is retained by the organizer."
        ),
        "core_order_sha256": _sequence_sha256(core["canonical_ikey"]),
        "curation_rules": [
            "preserve the anchored 96-row Module 1 sequence",
            "exclude the 96 core identifiers",
            "retain the first RDKit-valid row per identifier",
            "sort additions by canonical_ikey",
            "normalize ReFRAME HYPERLINK page links to HTTPS URLs",
        ],
        "final_row_count": len(result),
        "final_sha256": _sha256(output_path),
        "raw_row_count": len(raw),
        "raw_sha256": RAW_SHA256,
        "rdkit_version": rdBase.rdkitVersion,
        "retrieved_date": RETRIEVED_DATE,
        "source_url": SOURCE_URL,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("legacy_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("provenance_path", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.error")
    build_snapshot(**vars(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate once and verify the asset**

Run from the local repository root:

```bash
curl -fsSL --max-time 60 \
  https://reframedb.org/assets/csv/reframe_smiles_list.csv \
  -o /tmp/reframe_smiles_list.2026-08-24.csv
shasum -a 256 /tmp/reframe_smiles_list.2026-08-24.csv
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev copy scripts/build_reframe_module1_snapshot.py \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/scripts/
/opt/homebrew/bin/brev copy /tmp/reframe_smiles_list.2026-08-24.csv \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/reframe_smiles_list.2026-08-24.csv
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "/home/ubuntu/.venv/bin/python /home/ubuntu/workspace/codex/module1-10k-lean/data-source/scripts/build_reframe_module1_snapshot.py /home/ubuntu/workspace/codex/module1-10k-lean/data-source/reframe_smiles_list.2026-08-24.csv /home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/reframe_teaching_snapshot.csv /home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/reframe_module1_snapshot_10k.csv /home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/reframe_module1_snapshot_10k.provenance.json"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-lean/data-source PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-lean/data-source /home/ubuntu/.venv/bin/python -m pytest tests/test_reframe_module1_snapshot.py -v"
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/reframe_module1_snapshot_10k.csv \
  notebooks/data/reframe_module1_snapshot_10k.csv
/opt/homebrew/bin/brev copy \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/data-source/notebooks/data/reframe_module1_snapshot_10k.provenance.json \
  notebooks/data/reframe_module1_snapshot_10k.provenance.json
```

Expected raw SHA-256: `5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522`. Expected test result: pass. Do not commit the raw export.

- [ ] **Step 5: Commit the asset**

```bash
git add -- \
  scripts/build_reframe_module1_snapshot.py \
  tests/test_reframe_module1_snapshot.py \
  notebooks/data/reframe_module1_snapshot_10k.csv \
  notebooks/data/reframe_module1_snapshot_10k.provenance.json
git commit -m "feat: add fixed Module 1 ReFRAME snapshot"
```

### Task 2: Add the loader and update Module 1

**Files:**
- Modify: `notebooks/workshop_common.py`
- Modify: `tests/test_workshop_common.py`
- Modify: `notebooks/01_direct_nvmolkit_reframe.ipynb`
- Modify: `tests/test_workshop_notebook_inventory.py`

- [ ] **Step 1: Add the failing loader and notebook tests**

In `tests/test_workshop_common.py`, add `import hashlib` with the imports,
replace the invalid-source expectation, and add:

```python
LEGACY_MODULE1_ORDER_SHA256 = (
    "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
)


def test_legacy_module1_snapshot_order_is_unchanged():
    frame = workshop_common.load_reframe(
        sample_size=96,
        anchor_terms=("imatinib", "linezolid", "ritonavir"),
        source="snapshot",
    )
    observed = hashlib.sha256(
        "\n".join(frame["canonical_ikey"]).encode("utf-8")
    ).hexdigest()

    assert len(frame) == 96
    assert observed == LEGACY_MODULE1_ORDER_SHA256


@pytest.mark.parametrize("source", [None, "", "auto", "Snapshot", True, 1])
def test_loader_accepts_only_explicit_supported_sources(source):
    with pytest.raises(
        ValueError,
        match=r"^source must be 'snapshot', 'snapshot_10k', or 'live'\.$",
    ):
        workshop_common.load_reframe(sample_size=96, source=source)


@pytest.mark.parametrize("sample_size", [95, 10_001])
def test_snapshot_10k_rejects_sizes_outside_the_exercise(sample_size):
    with pytest.raises(
        ValueError,
        match=r"^source='snapshot_10k' supports 96 through 10,000 rows\.$",
    ):
        workshop_common.load_reframe(sample_size=sample_size, source="snapshot_10k")


@pytest.mark.parametrize(
    "sample_size", [None, True, 96.0, "96", workshop_common.np.int64(96)]
)
def test_snapshot_10k_rejects_non_builtin_integers(sample_size):
    with pytest.raises(TypeError, match=r"^sample_size must be a positive integer\.$"):
        workshop_common.load_reframe(sample_size=sample_size, source="snapshot_10k")


def test_snapshot_10k_returns_nested_local_prefixes(monkeypatch):
    real_read_csv = workshop_common.pd.read_csv
    reads = []

    def local_read_csv(source, *args, **kwargs):
        reads.append(source)
        if str(source).startswith(("http://", "https://")):
            raise AssertionError("snapshot_10k attempted network access")
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(workshop_common.pd, "read_csv", local_read_csv)
    sizes = (96, 97, 512, 2_048, 9_999, 10_000)
    samples = {
        size: workshop_common.load_reframe(
            sample_size=size,
            anchor_terms=("imatinib", "linezolid", "ritonavir"),
            source="snapshot_10k",
        )
        for size in sizes
    }

    assert reads == [workshop_common.MODULE1_SNAPSHOT_PATH] * len(sizes)
    for size in sizes:
        assert len(samples[size]) == size
    for smaller, larger in zip(sizes, sizes[1:]):
        assert samples[smaller]["canonical_ikey"].tolist() == samples[larger][
            "canonical_ikey"
        ].iloc[:smaller].tolist()
    assert samples[96].attrs == {
        "source": "bundled_snapshot_10k",
        "invalid_count": 0,
    }


def test_snapshot_10k_missing_asset_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        workshop_common, "MODULE1_SNAPSHOT_PATH", tmp_path / "missing.csv"
    )
    monkeypatch.setattr(
        workshop_common,
        "MODULE1_PROVENANCE_PATH",
        tmp_path / "missing.provenance.json",
    )
    with pytest.raises(
        ValueError,
        match=r"^Bundled Module 1 ReFRAME snapshot failed its integrity check\.$",
    ):
        workshop_common.load_reframe(sample_size=96, source="snapshot_10k")


def test_snapshot_10k_rejects_configured_csv_before_read(monkeypatch):
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
            sample_size=96,
            source="snapshot_10k",
            use_configured_csv=True,
        )
    assert reads == []
```

In `tests/test_workshop_notebook_inventory.py`, update the default-source assertion and add:

```python
def test_module1_sample_size_exercise_is_bounded_to_steps_1_through_3():
    source = _module1_code_source()
    step1 = _module1_cell_source("cell-e51239292d22")
    step3 = _module1_cell_source("cell-7dabc4a334bf")
    step4 = _module1_cell_source("cell-00a63d6c51e3")

    assert 'DATA_SOURCE = "snapshot_10k"' in source
    assert "SAMPLE_SIZE = 96" in source
    for value in ("512", "2_048", "10_000"):
        assert value in step1 + step3
    assert "rerun Steps 1 through 3" in step3
    assert "restore `SAMPLE_SIZE = 96`" in step4
    assert "Change <code>SAMPLE_SIZE</code> and observe" not in step4


def test_module1_steps_1_through_3_do_not_allocate_an_n_by_n_result():
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    end = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.id == "cell-c5e8c0433cbd"
    )
    source = "\n\n".join(
        cell.source
        for cell in notebook.cells[: end + 1]
        if cell.cell_type == "code"
    )
    forbidden = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in {"empty", "zeros", "ones", "full", "pdist", "squareform"}:
            forbidden.append(name)

    assert forbidden == []
    assert source.count(
        "crossTanimotoSimilarity(anchor_fps, fingerprints)"
    ) == 1


@pytest.mark.parametrize("sample_size", [96, 512, 2_048, 10_000])
def test_module1_steps_1_through_3_execute_at_supported_sizes(
    monkeypatch, tmp_path, sample_size
):
    notebook = nbformat.read(MODULE1_NOTEBOOK_PATH, as_version=4)
    end = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.id == "cell-c5e8c0433cbd"
    )
    notebook.cells = notebook.cells[: end + 1]
    sample_cell = next(
        cell for cell in notebook.cells if cell.id == "cell-9f1999dd251d"
    )
    sample_cell.source = sample_cell.source.replace(
        "SAMPLE_SIZE = 96", f"SAMPLE_SIZE = {sample_size}", 1
    )
    setup_index = next(
        index
        for index, cell in enumerate(notebook.cells)
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
        raise AssertionError("Module 1 attempted network access")
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
import os
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
assert _blocked_network_attempts == []
expected_backends = {"RDKit CPU"}
if NVMOLKIT_READY:
    expected_backends.add("nvMolKit GPU")
assert set(fingerprint_runtime["backend"]) == expected_backends
assert set(similarity_runtime["backend"]) == expected_backends
if os.environ.get("RUN_GPU_TESTS") == "1":
    assert NVMOLKIT_READY
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
receipt = {
    "sample_size": SAMPLE_SIZE,
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
    monkeypatch.setenv("REFRAME_CSV", "https://hostile.invalid/ignored.csv")
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
    print(match.group(0))
```

Also change the existing default report assertion to:

```python
assert report["source"] == "bundled_snapshot_10k"
```

- [ ] **Step 2: Run one red test pass**

Copy the current worktree once and run only the new source/range tests:

```bash
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "test ! -e /home/ubuntu/workspace/codex/module1-10k-lean/source-red"
/opt/homebrew/bin/brev copy ./ \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/source-red/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-lean/source-red PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-lean/source-red /home/ubuntu/.venv/bin/python -m pytest tests/test_workshop_common.py tests/test_workshop_notebook_inventory.py -k 'snapshot_10k or sample_size_exercise' -v"
```

Expected: failures because `snapshot_10k` and the new notebook source are not implemented.

- [ ] **Step 3: Implement the minimal loader branch**

The committed-asset test validates all 10,000 molecules before publication.
At runtime, the loader also parses the full fixed asset before returning the
requested prefix, so invalid data outside a small request cannot pass silently.

Add imports and constants in `notebooks/workshop_common.py`:

```python
import hashlib
import json

MODULE1_SNAPSHOT_PATH = (
    Path(__file__).with_name("data") / "reframe_module1_snapshot_10k.csv"
)
MODULE1_PROVENANCE_PATH = MODULE1_SNAPSHOT_PATH.with_suffix(".provenance.json")
MODULE1_MIN_SAMPLE_SIZE = 96
MODULE1_MAX_SAMPLE_SIZE = 10_000
MODULE1_INTEGRITY_ERROR = (
    "Bundled Module 1 ReFRAME snapshot failed its integrity check."
)
```

Add before `load_reframe`:

```python
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
```

Replace only the validation block at the start of `load_reframe` with:

```python
if not isinstance(source, str) or source not in {
    "snapshot",
    "snapshot_10k",
    "live",
}:
    raise ValueError("source must be 'snapshot', 'snapshot_10k', or 'live'.")
if type(use_configured_csv) is not bool:
    raise TypeError("use_configured_csv must be a bool.")
if source == "live" and use_configured_csv:
    raise ValueError("source='live' cannot be combined with use_configured_csv=True.")
if source == "snapshot_10k" and use_configured_csv:
    raise ValueError(
        "source='snapshot_10k' cannot be combined with use_configured_csv=True."
    )
requested_size = _positive_integer(sample_size, name="sample_size")
if source == "snapshot_10k":
    if not MODULE1_MIN_SAMPLE_SIZE <= requested_size <= MODULE1_MAX_SAMPLE_SIZE:
        raise ValueError("source='snapshot_10k' supports 96 through 10,000 rows.")
    return _load_module1_snapshot(requested_size)
```

Leave the existing configured CSV, legacy snapshot, live source, descriptor helpers, and memory helpers unchanged.

- [ ] **Step 4: Patch only the approved Module 1 cells**

Use `apply_patch` against `notebooks/01_direct_nvmolkit_reframe.ipynb`. Preserve every cell ID, tag, metadata field, null execution count, and empty output.

- In `cell-f38b80a4df5a`, retain the setup text and add `data/reframe_module1_snapshot_10k.csv` as the fixed Module 1 scaling asset.
- Replace `cell-e51239292d22` with:

```text
## Step 1 — Load and quality-check ReFRAME

The reference run uses the first 96 rows of a fixed, bundled 10,000-compound Module 1 snapshot. The source is local and makes no network request. Larger samples are nested, so each requested sample contains the smaller one.

<div class="task"><b>Your edit:</b> Run the 96-row reference first. Then try <code>SAMPLE_SIZE = 512</code>, <code>2_048</code>, and <code>10_000</code>. Change only <code>SAMPLE_SIZE</code> and rerun Steps 1 through 3.</div>
```

- In `cell-9f1999dd251d`, change only:

```python
DATA_SOURCE = "snapshot_10k"
```

- In `cell-7dabc4a334bf`, replace only list item 2 with:

```html
<li>Change only <code>SAMPLE_SIZE</code> in Step 1 to <code>512</code>, <code>2_048</code>, or <code>10_000</code>, then rerun Steps 1 through 3. How does the speed ratio change?</li>
```

- At the beginning of `cell-00a63d6c51e3`, add:

```text
Before continuing, restore `SAMPLE_SIZE = 96` in Step 1 and rerun Steps 1 through 3. The RDKit reference below constructs a quadratic condensed-distance vector and intentionally remains bounded.
```

- In the same cell, replace only list item 3 with:

```html
<li>Keep <code>SAMPLE_SIZE = 96</code> for this bounded CPU/GPU clustering comparison.</li>
```

- In `cell-7d83af67d461`, add one takeaway: the fixed snapshot is the reproducible exercise source; the later live-export cell is optional and may drift. Do not change the advanced code cells.

- [ ] **Step 5: Run one complete green test pass on the L4**

Copy the implemented worktree once to a new task directory and run the complete suite with the GPU assertions enabled:

```bash
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "test ! -e /home/ubuntu/workspace/codex/module1-10k-lean/source-green"
/opt/homebrew/bin/brev copy ./ \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/source-green/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-lean/source-green PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-lean/source-green RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python -m pytest -q -s"
```

Expected: the full suite passes. Retain the exact test count and the four `MODULE1_SCALE_RECEIPT_JSON` lines. The 10,000-row receipt must show a `(3, 10000)` similarity result, both runtime backends, elapsed time, peak host memory, and positive CUDA peak allocation/reservation.

- [ ] **Step 6: Prove scope and commit**

```bash
git diff --exit-code 513232a -- \
  notebooks/02_agent_assisted_reframe_neighborhoods.ipynb \
  notebooks/03_full_agent_reframe_panel_design.ipynb \
  notebooks/data/reframe_teaching_snapshot.csv \
  launchable
git diff --check
git status --short
git add -- \
  notebooks/workshop_common.py \
  tests/test_workshop_common.py \
  notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_notebook_inventory.py
git commit -m "fix: support Module 1 scaling through 10k"
```

Expected: no protected diff, no whitespace error, and no unrelated path staged.

### Task 3: Review, publish, and validate only the target Launchable

**Scope:** Reviewed commits only; repository `ktretina/nvmolkit-brev-notebook`; base `main`; Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`; at most one fresh environment.

- [ ] **Step 1: Run the two independent reviews**

Dispatch one specification reviewer and one code-quality/scientific-UX reviewer against the approved design and `513232a..HEAD`. Both must verify:

- every sample size from 96 through 10,000 is accepted;
- tested samples are nested and local-only;
- Step 3 remains exactly a three-anchor-by-library workload;
- Step 4 still requires 96 and retains its memory guard;
- Modules 2 and 3, the legacy asset, setup/key handling, renderer, and other Launchables are unchanged;
- provenance and performance wording do not overstate scientific evidence.

Fix blocking findings only. Record lower-severity findings as residual risks.

- [ ] **Step 2: If review changes code, run one replacement test pass**

Skip this step when reviews require no change. Otherwise copy the reviewed worktree once and rerun the same complete suite:

```bash
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "test ! -e /home/ubuntu/workspace/codex/module1-10k-lean/source-reviewed"
/opt/homebrew/bin/brev copy ./ \
  nvmolkit---nemotron-notebook-80f439:/home/ubuntu/workspace/codex/module1-10k-lean/source-reviewed/
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-80f439 \
  "env -C /home/ubuntu/workspace/codex/module1-10k-lean/source-reviewed PYTHONPATH=/home/ubuntu/workspace/codex/module1-10k-lean/source-reviewed RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python -m pytest -q -s"
git add -- \
  scripts/build_reframe_module1_snapshot.py \
  tests/test_reframe_module1_snapshot.py \
  notebooks/data/reframe_module1_snapshot_10k.csv \
  notebooks/data/reframe_module1_snapshot_10k.provenance.json \
  notebooks/workshop_common.py \
  tests/test_workshop_common.py \
  notebooks/01_direct_nvmolkit_reframe.ipynb \
  tests/test_workshop_notebook_inventory.py
git commit -m "fix: address Module 1 10k review findings"
```

- [ ] **Step 3: Obtain publication approval and open one draft PR**

Present the commits, full-suite result, four scale receipts, protected-path diff, reviews, and residual risks. After approval:

```bash
git push -u origin fix/module1-10k-snapshot
gh pr list --repo ktretina/nvmolkit-brev-notebook \
  --state open --base main --head fix/module1-10k-snapshot
```

If the list is empty, create one draft:

```bash
gh pr create --repo ktretina/nvmolkit-brev-notebook \
  --base main --head fix/module1-10k-snapshot --draft \
  --title "Support Module 1 scaling through 10k" \
  --body "Adds one fixed Module-1-only ReFRAME asset, one isolated snapshot_10k loader branch, bounded instructions, and acceptance at 96, 512, 2,048, and 10,000. Modules 2 and 3, the legacy snapshot, setup/key handling, and all other Launchables are unchanged."
```

Do not create a duplicate. Check the draft PR without merging:

```bash
gh pr checks --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot
```

Expected: the draft PR checks pass. Keep the PR unmerged so the target
Launchable's future source remains unchanged during fresh acceptance.

- [ ] **Step 4: Obtain cost approval and run one fresh target deployment**

After approval, the user opens only:

```text
https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW
```

The user keeps one NVIDIA L4 and 75 GiB storage, changes no setup value, sets instance name `nvmolkit-module1-10k-20260824`, deploys once, and provides the Access URL. Do not retry an ambiguous deployment.

The fresh environment initially deploys the current `main`. Check out the
exact reviewed feature commit inside this one test environment before running
acceptance. This does not change the saved Launchable or any future instance.

Verify the exact fresh instance:

```bash
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
REVIEWED_SHA=$(git rev-parse HEAD)
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook fetch origin fix/module1-10k-snapshot"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook checkout --detach $REVIEWED_SHA"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook rev-parse HEAD"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "curl -fsS http://127.0.0.1:8888/api"
/opt/homebrew/bin/brev exec nvmolkit-module1-10k-20260824 \
  "env -C /home/ubuntu/nvmolkit-brev-notebook PYTHONPATH=/home/ubuntu/nvmolkit-brev-notebook RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python -m pytest -q -s"
```

The two Brev lists must resolve the same exact fresh instance in `agents-in-ls`;
stop if they differ. The initial repository status must be clean. Expected:
`rev-parse` equals `REVIEWED_SHA`, the detached tree remains clean, one L4,
Jupyter health, complete suite pass, and fresh scale receipts. Do not reuse
evidence from the retained instance.

- [ ] **Step 5: Run the user-operated browser acceptance and hand off**

The user opens the organization-only Jupyter Secure Link and navigates to:

```text
lab/tree/nvmolkit-brev-notebook/notebooks/01_direct_nvmolkit_reframe.ipynb
```

Restart the kernel. Run Steps 1 through 3 at 96, then change only `SAMPLE_SIZE` to 512, 2,048, and 10,000, rerunning Steps 1 through 3 each time. At every size require the matching valid-structure and fingerprint-batch counts, three anchors, both runtime rows, one observed ratio, and no network or loader error. Restore 96 before Step 4.

If the browser pass fails, keep the PR unmerged, preserve evidence, and request
approval before any corrective write.

- [ ] **Step 6: Verify checks, then obtain separate merge approval**

After browser acceptance, present the fresh evidence and request approval to
make the PR ready. After approval:

```bash
gh pr ready --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot
gh pr checks --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot
```

Stop unless every required check is green. Present that output and request a
separate merge approval. Only after that approval, run:

```bash
gh pr merge --repo ktretina/nvmolkit-brev-notebook \
  fix/module1-10k-snapshot --merge
git ls-remote origin refs/heads/main
```

Expected: `main` resolves to the reviewed merge. Because the confirmed target
Launchable follows `main`, future target instances now receive the accepted
tree without a Console edit.

- [ ] **Step 7: Hand off and stop only with approval**

Give the user the exact direct notebook URL and state that only the target
Launchable was qualified. Ask before stopping the fresh instance. If approved:

```bash
/opt/homebrew/bin/brev ls --json
/opt/homebrew/bin/brev ls --org agents-in-ls --json
/opt/homebrew/bin/brev stop nvmolkit-module1-10k-20260824
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

If the fresh suite, browser, checks, or merge fails, do not call the update
complete or expose it to attendees. Preserve evidence and request approval
before any revert.

## Completion evidence

- Apart from this plan document, exactly the eight implementation paths in the
  file map changed.
- The asset contains 10,000 valid unique rows with approved provenance.
- Modules 2 and 3, the legacy snapshot, setup/key handling, renderer, and other Launchables are unchanged.
- One complete L4 suite and both independent reviews pass on the reviewed tree.
- One fresh target deployment and browser sequence pass at 96, 512, 2,048, and 10,000.
- Future deployments of only `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` resolve the reviewed `main` commit.
- The user receives the direct notebook URL.
