import importlib.util
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


def test_snapshot_rejects_a_request_larger_than_its_inventory(monkeypatch):
    monkeypatch.delenv("REFRAME_CSV", raising=False)

    with pytest.raises(
        ValueError,
        match=r"^Bundled ReFRAME snapshot contains 96 rows; requested 97\.$",
    ):
        workshop_common.load_reframe(sample_size=97, source="snapshot")


@pytest.mark.parametrize("source", [None, "", "auto", "Snapshot", True, 1])
def test_loader_accepts_only_explicit_snapshot_or_live_sources(source):
    with pytest.raises(ValueError, match=r"^source must be 'snapshot' or 'live'\.$"):
        workshop_common.load_reframe(sample_size=1, source=source)


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
