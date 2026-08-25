import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "notebooks" / "workshop_common.py"


def _load_workshop_common():
    spec = importlib.util.spec_from_file_location(
        "workshop_common_for_tests", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workshop_common = _load_workshop_common()

LEGACY_MODULE1_ORDER_SHA256 = (
    "eec75f8f1dad7076211de4a25ad40ac051e9984a1e26ea2ca14c4726ff4efbba"
)
LEGACY_MODULE3_ORDER_SHA256 = (
    "cf4c96272124749412be6bdb2ad1c46e38998d7915c4d7c0271b5ef378beb6b1"
)
LEGACY_MODULE3_FIRST_24_ORDER_SHA256 = (
    "444a91d7d6f7cf5faa8ba9166839f929326df0d5b74c6ca9677531c4f1776e7f"
)
MODULE1_INTEGRITY_ERROR = (
    r"^Bundled Module 1 ReFRAME snapshot failed its integrity check\.$"
)


def _teaching_frame():
    return workshop_common.pd.DataFrame(
        [
            {
                "smile": "CCO",
                "canonical_ikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "name": "ethanol",
                "source": "test",
                "source_id": "test-1",
                "status": "approved",
                "reframedb_url": "https://example.invalid/ethanol",
            }
        ]
    )


def _write_corrupt_module1_snapshot(tmp_path, monkeypatch, corruption):
    snapshot_path = tmp_path / "reframe_module1_snapshot_10k.csv"
    provenance_path = tmp_path / "reframe_module1_snapshot_10k.provenance.json"
    frame = workshop_common.pd.DataFrame(
        [
            {
                "smile": "CC",
                "canonical_ikey": f"TEST-KEY-{index}",
                "name": f"test compound {index}",
                "source": "test",
                "source_id": f"test-{index}",
                "status": "approved",
                "reframedb_url": f"https://example.invalid/{index}",
            }
            for index in range(100)
        ]
    )

    if corruption == "wrong_row_count":
        frame = frame.iloc[:-1]
    elif corruption == "incomplete_schema":
        frame = frame.drop(columns="status")
    elif corruption == "duplicate_canonical_ikey":
        frame.loc[frame.index[-1], "canonical_ikey"] = frame.loc[
            frame.index[0], "canonical_ikey"
        ]
    elif corruption == "invalid_smiles":
        frame.loc[frame.index[-1], "smile"] = "not-a-smiles"

    frame.to_csv(snapshot_path, index=False, lineterminator="\n")
    observed_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if corruption == "malformed_provenance_json":
        provenance_path.write_text("{", encoding="utf-8")
    else:
        final_hash = "0" * 64 if corruption == "hash_mismatch" else observed_hash
        provenance_path.write_text(
            json.dumps({"final_sha256": final_hash}) + "\n", encoding="utf-8"
        )

    monkeypatch.setattr(workshop_common, "MODULE1_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(
        workshop_common, "MODULE1_PROVENANCE_PATH", provenance_path
    )
    monkeypatch.setattr(workshop_common, "MODULE1_MIN_SAMPLE_SIZE", 96)
    monkeypatch.setattr(workshop_common, "MODULE1_MAX_SAMPLE_SIZE", 100)


def test_configured_csv_opt_in_requires_environment_value(monkeypatch):
    monkeypatch.delenv("REFRAME_CSV", raising=False)

    with pytest.raises(
        ValueError, match=r"^Could not load the configured ReFRAME CSV\.$"
    ):
        workshop_common.load_reframe(sample_size=1, use_configured_csv=True)


def test_explicit_missing_reframe_csv_fails_closed(monkeypatch):
    configured_source = "/private/secret/missing-reframe.csv"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        raise FileNotFoundError(f"missing {configured_source}")

    monkeypatch.setenv("REFRAME_CSV", configured_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with pytest.raises(
        ValueError, match=r"^Could not load the configured ReFRAME CSV\.$"
    ) as captured:
        workshop_common.load_reframe(sample_size=1, use_configured_csv=True)

    assert configured_source not in str(captured.value)
    assert reads == [configured_source]


def test_malformed_explicit_reframe_csv_does_not_disclose_source(monkeypatch):
    configured_source = "https://example.invalid/reframe.csv?token=do-not-disclose"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        raise workshop_common.pd.errors.ParserError(
            f"malformed source {configured_source}"
        )

    monkeypatch.setenv("REFRAME_CSV", configured_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(
            ValueError, match=r"^Could not load the configured ReFRAME CSV\.$"
        ) as captured:
            workshop_common.load_reframe(sample_size=1, use_configured_csv=True)

    assert configured_source not in str(captured.value)
    assert all(configured_source not in str(item.message) for item in caught)
    assert reads == [configured_source]


def test_configured_csv_opt_in_loads_only_configured_source(monkeypatch):
    configured_source = "/data/approved-reframe.csv"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == configured_source:
            return _teaching_frame()
        raise AssertionError(f"Unexpected source: {source!r}")

    monkeypatch.setenv("REFRAME_CSV", configured_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    frame = workshop_common.load_reframe(
        sample_size=1, source="snapshot", use_configured_csv=True
    )

    assert reads == [configured_source]
    assert frame.attrs["source"] == "configured_csv"


def test_live_source_rejects_configured_csv_opt_in_before_read(monkeypatch):
    monkeypatch.setenv("REFRAME_CSV", "/private/secret/reframe.csv")
    reads = []
    monkeypatch.setattr(workshop_common.pd, "read_csv", reads.append)

    with pytest.raises(
        ValueError,
        match=(
            r"^source='live' cannot be combined with "
            r"use_configured_csv=True\.$"
        ),
    ):
        workshop_common.load_reframe(
            sample_size=1, source="live", use_configured_csv=True
        )

    assert reads == []


@pytest.mark.parametrize("use_configured_csv", [None, 0, 1, "false", [], ()])
def test_configured_csv_opt_in_requires_an_exact_bool(use_configured_csv):
    with pytest.raises(TypeError, match=r"^use_configured_csv must be a bool\.$"):
        workshop_common.load_reframe(
            sample_size=1, use_configured_csv=use_configured_csv
        )


def test_snapshot_source_ignores_hostile_reframe_csv(monkeypatch):
    hostile_source = "https://hostile.invalid/reframe.csv?token=do-not-disclose"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == workshop_common.SNAPSHOT_PATH:
            return _teaching_frame()
        raise AssertionError(f"Unexpected source: {source!r}")

    monkeypatch.setenv("REFRAME_CSV", hostile_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    frame = workshop_common.load_reframe(sample_size=1, source="snapshot")

    assert reads == [workshop_common.SNAPSHOT_PATH]
    assert frame.attrs["source"] == "bundled_snapshot"


def test_default_snapshot_is_local_deterministic_and_complete(monkeypatch):
    real_read_csv = workshop_common.pd.read_csv
    reads = []

    def local_read_csv(source, *args, **kwargs):
        reads.append(source)
        if str(source).startswith(("http://", "https://")):
            raise AssertionError("The default loader must not attempt network access.")
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setenv(
        "REFRAME_CSV", "https://hostile.invalid/reframe.csv?token=do-not-disclose"
    )
    monkeypatch.setattr(workshop_common.pd, "read_csv", local_read_csv)

    first = workshop_common.load_reframe()
    second = workshop_common.load_reframe()

    assert reads == [workshop_common.SNAPSHOT_PATH, workshop_common.SNAPSHOT_PATH]
    assert len(first) == len(second) == 96
    assert first["canonical_ikey"].is_unique
    assert first["canonical_ikey"].tolist() == second["canonical_ikey"].tolist()
    assert first.attrs["source"] == second.attrs["source"] == "bundled_snapshot"


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


def test_legacy_module3_snapshot_orders_are_unchanged():
    frame = workshop_common.load_reframe(sample_size=96, source="snapshot")
    identifiers = frame["canonical_ikey"].tolist()

    assert hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest() == (
        LEGACY_MODULE3_ORDER_SHA256
    )
    assert hashlib.sha256("\n".join(identifiers[:24]).encode("utf-8")).hexdigest() == (
        LEGACY_MODULE3_FIRST_24_ORDER_SHA256
    )


def test_snapshot_rejects_a_request_larger_than_its_inventory(monkeypatch):
    monkeypatch.delenv("REFRAME_CSV", raising=False)

    with pytest.raises(
        ValueError,
        match=r"^Bundled ReFRAME snapshot contains 96 rows; requested 97\.$",
    ):
        workshop_common.load_reframe(sample_size=97, source="snapshot")


@pytest.mark.parametrize("source", [None, "", "auto", "Snapshot", True, 1])
def test_loader_accepts_only_explicit_supported_sources(source):
    with pytest.raises(
        ValueError,
        match=r"^source must be 'snapshot', 'snapshot_10k', or 'live'\.$",
    ):
        workshop_common.load_reframe(sample_size=1, source=source)


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
    with pytest.raises(ValueError, match=MODULE1_INTEGRITY_ERROR):
        workshop_common.load_reframe(sample_size=96, source="snapshot_10k")


def test_snapshot_10k_malformed_provenance_fails_closed(monkeypatch, tmp_path):
    _write_corrupt_module1_snapshot(
        tmp_path, monkeypatch, "malformed_provenance_json"
    )

    with pytest.raises(ValueError, match=MODULE1_INTEGRITY_ERROR):
        workshop_common.load_reframe(sample_size=96, source="snapshot_10k")


def test_snapshot_10k_hash_mismatch_fails_closed(monkeypatch, tmp_path):
    _write_corrupt_module1_snapshot(tmp_path, monkeypatch, "hash_mismatch")

    with pytest.raises(ValueError, match=MODULE1_INTEGRITY_ERROR):
        workshop_common.load_reframe(sample_size=96, source="snapshot_10k")


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_row_count",
        "incomplete_schema",
        "duplicate_canonical_ikey",
        "invalid_smiles",
    ],
)
def test_snapshot_10k_semantic_corruption_fails_closed(
    monkeypatch, tmp_path, corruption
):
    _write_corrupt_module1_snapshot(tmp_path, monkeypatch, corruption)

    with pytest.raises(ValueError, match=MODULE1_INTEGRITY_ERROR):
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


def test_explicit_live_read_failure_is_generic_and_does_not_fallback(monkeypatch):
    sensitive_failure = "network failed with token=do-not-disclose"
    hostile_source = "https://hostile.invalid/reframe.csv?token=ambient-secret"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == workshop_common.REFRAME_URL:
            raise OSError(sensitive_failure)
        raise AssertionError(f"Unexpected source: {source!r}")

    monkeypatch.setenv("REFRAME_CSV", hostile_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(
            ValueError, match=r"^Could not load the live ReFRAME export\.$"
        ) as captured:
            workshop_common.load_reframe(sample_size=1, source="live")

    assert sensitive_failure not in str(captured.value)
    assert all(sensitive_failure not in str(item.message) for item in caught)
    assert reads == [workshop_common.REFRAME_URL]


def test_memory_estimators_match_the_allocations_used_by_module_1():
    assert workshop_common.square_matrix_bytes(10_000) == 400_000_000
    assert workshop_common.condensed_distance_bytes(10_000) == 399_960_000


@pytest.mark.parametrize("count", [None, True, 0, -1, 1.5, "10000"])
def test_memory_estimators_reject_invalid_counts_and_types(count):
    for estimator in (
        workshop_common.square_matrix_bytes,
        workshop_common.condensed_distance_bytes,
    ):
        with pytest.raises((TypeError, ValueError)):
            estimator(count)


def test_default_memory_limit_rejects_10k_but_explicit_512_mib_accepts():
    required_bytes = workshop_common.square_matrix_bytes(10_000)

    with pytest.raises(MemoryError, match=r"400,000,000 bytes.*128 MiB"):
        workshop_common.require_memory_within_limit(required_bytes)

    assert (
        workshop_common.require_memory_within_limit(required_bytes, limit_mib=512)
        == required_bytes
    )


def test_bounded_condensed_distances_rejects_10k_by_default_and_accepts_512_mib():
    with pytest.raises(MemoryError, match=r"399,960,000 bytes.*134,217,728 bytes"):
        workshop_common.require_bounded_condensed_distances(10_000)

    assert (
        workshop_common.require_bounded_condensed_distances(
            10_000, maximum_bytes=512 * 1024 * 1024
        )
        == 399_960_000
    )


@pytest.mark.parametrize("row_count", [None, True, 0, -1, 1.5, "10000"])
def test_bounded_condensed_distances_rejects_invalid_row_counts(row_count):
    with pytest.raises((TypeError, ValueError)):
        workshop_common.require_bounded_condensed_distances(row_count)


@pytest.mark.parametrize("maximum_bytes", [None, True, 0, -1, 1.5, "134217728"])
def test_bounded_condensed_distances_rejects_invalid_maximum_bytes(maximum_bytes):
    with pytest.raises((TypeError, ValueError)):
        workshop_common.require_bounded_condensed_distances(
            96, maximum_bytes=maximum_bytes
        )
