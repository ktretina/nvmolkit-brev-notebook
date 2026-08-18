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


def test_explicit_missing_reframe_csv_fails_closed(monkeypatch):
    configured_source = "/private/secret/missing-reframe.csv"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == configured_source:
            raise FileNotFoundError(f"missing {configured_source}")
        return _teaching_frame()

    monkeypatch.setenv("REFRAME_CSV", configured_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with pytest.raises(
        ValueError, match=r"^Could not load the configured ReFRAME CSV\.$"
    ) as captured:
        workshop_common.load_reframe(sample_size=1)

    assert configured_source not in str(captured.value)
    assert reads == [configured_source]


def test_malformed_explicit_reframe_csv_does_not_disclose_source(monkeypatch):
    configured_source = "https://example.invalid/reframe.csv?token=do-not-disclose"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == configured_source:
            raise workshop_common.pd.errors.ParserError(
                f"malformed source {configured_source}"
            )
        return _teaching_frame()

    monkeypatch.setenv("REFRAME_CSV", configured_source)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(
            ValueError, match=r"^Could not load the configured ReFRAME CSV\.$"
        ) as captured:
            workshop_common.load_reframe(sample_size=1)

    assert configured_source not in str(captured.value)
    assert all(configured_source not in str(item.message) for item in caught)
    assert reads == [configured_source]


def test_live_read_failure_uses_snapshot_with_generic_warning(monkeypatch):
    sensitive_failure = "network failed with token=do-not-disclose"
    reads = []

    def fake_read_csv(source):
        reads.append(source)
        if source == workshop_common.REFRAME_URL:
            raise OSError(sensitive_failure)
        if source == workshop_common.SNAPSHOT_PATH:
            return _teaching_frame()
        raise AssertionError(f"Unexpected source: {source!r}")

    monkeypatch.delenv("REFRAME_CSV", raising=False)
    monkeypatch.setattr(workshop_common.pd, "read_csv", fake_read_csv)

    with pytest.warns(
        UserWarning,
        match=(
            r"^Could not load the live ReFRAME export; "
            r"using the shared teaching snapshot\.$"
        ),
    ) as caught:
        frame = workshop_common.load_reframe(sample_size=1)

    assert sensitive_failure not in str(caught[0].message)
    assert reads == [workshop_common.REFRAME_URL, workshop_common.SNAPSHOT_PATH]
    assert frame.attrs["source"] == "shared 1-row teaching snapshot"
