"""Fault-injected wrapper lifecycle checks; no native execution is required."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1":
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import EngineConfig, V1UnsupportedError  # noqa: E402
import gafime.v1_adapter as adapter  # noqa: E402


class _StaleReport:
    def __init__(self):
        self.export_calls = 0

    def __arrow_c_array__(self, requested_schema=None):
        self.export_calls += 1
        return "stale report must never be exported"


class _BrokenName:
    def __str__(self):
        raise RuntimeError("name conversion failed")


class _CommittedHandle:
    def __init__(self, fault, cleanup_error):
        self.fault = fault
        self.cleanup_error = cleanup_error
        self.closed = False
        self.commits = 0
        self.close_calls = 0
        self.analyze_calls = 0
        self.metadata_reads = 0
        self.wrapper = None

    def update_target_buffer(self, target):
        self.commits += 1

    def reseed(self, seed):
        self.commits += 1

    def analyze(self):
        self.analyze_calls += 1
        raise AssertionError("analysis must not follow failed identity refresh")

    def _assert_invalidated_before_read(self):
        self.metadata_reads += 1
        assert self.commits == 1
        assert self.wrapper._native_report is None
        assert self.wrapper._last_report is None
        assert self.wrapper._scenario_plan is None
        assert self.wrapper._generated_feature_start is None
        assert self.wrapper._graph_replayed is False

    @property
    def feature_names(self):
        self._assert_invalidated_before_read()
        if self.fault == "missing_names":
            raise AttributeError("feature_names")
        if self.fault == "names_getter":
            raise RuntimeError("feature names getter failed")
        if self.fault == "interrupt":
            raise KeyboardInterrupt("metadata interrupted")
        if self.fault == "name_conversion":
            return [_BrokenName()]
        return ["new_base", "new_generated"]

    @property
    def generated_feature_start(self):
        self._assert_invalidated_before_read()
        raise RuntimeError("generated feature start getter failed")

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.cleanup_error:
            raise SystemExit("injected cleanup failure")


@pytest.mark.parametrize("operation", ["update_target", "reseed"])
@pytest.mark.parametrize("family", ["time_series", "decision_path"])
@pytest.mark.parametrize("cleanup_error", [False, True])
@pytest.mark.parametrize(
    "fault,error,match",
    [
        ("missing_names", V1UnsupportedError, "rediscovered feature identities"),
        ("names_getter", RuntimeError, "feature names getter failed"),
        ("name_conversion", RuntimeError, "name conversion failed"),
        ("start_getter", RuntimeError, "generated feature start getter failed"),
        ("interrupt", KeyboardInterrupt, "metadata interrupted"),
    ],
)
def test_committed_generated_metadata_failure_retires_artifact(
    monkeypatch, operation, family, cleanup_error, fault, error, match
):
    config = EngineConfig(
        backend="core",
        random_seed=None if operation == "reseed" else 7,
        enable_time_series_functions=family == "time_series",
        enable_decision_path_functions=family == "decision_path",
        permutation_tests=0,
        num_repeats=1,
    )
    handle = _CommittedHandle(fault, cleanup_error)
    artifact = adapter.NativeCompiledGafime(
        config=config,
        feature_names=["old_base", "old_generated"],
        native_handle=handle,
        boundary_name="fault-injected",
        export=True,
    )
    handle.wrapper = artifact
    stale = _StaleReport()
    artifact._native_report = stale
    artifact._last_report = object()
    artifact._scenario_plan = object()
    artifact._generated_feature_start = 1
    artifact._graph_replayed = True
    monkeypatch.setattr(adapter, "_fresh_random_seed", lambda: 23)

    with pytest.raises(error, match=match):
        if operation == "update_target":
            artifact.update_target([1.0, 2.0])
        else:
            artifact.analyze()

    assert handle.commits == 1
    assert handle.metadata_reads > 0
    assert handle.closed
    assert handle.close_calls == 1
    assert handle.analyze_calls == 0
    assert artifact._closed
    assert artifact._native_report is None
    assert artifact._last_report is None
    assert artifact._scenario_plan is None
    assert artifact._generated_feature_start is None
    assert artifact._graph_replayed is False
    # New names are committed only after all metadata reads/conversions succeed.
    assert artifact.feature_names == ["old_base", "old_generated"]
    with pytest.raises(RuntimeError, match="closed"):
        artifact.export_arrow()
    with pytest.raises(RuntimeError, match="closed"):
        artifact.analyze()
    with pytest.raises(RuntimeError, match="closed"):
        artifact.update_target([2.0, 1.0])
    assert stale.export_calls == 0
    artifact.close()
    assert handle.close_calls == 1
