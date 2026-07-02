"""time_series family wiring test (no native wheel needed — fake boundary).

Verifies that enabling time_series routes analyze() to the native
analyze_time_series expand+mine path and surfaces the expanded feature names.
"""
import sys
from pathlib import Path

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime.v1_adapter as adapter
from gafime import EngineConfig, GafimeEngine


class _FakeReport:
    def __len__(self):
        return 0

    def records(self):
        return []


class _FakeBoundary:
    BOUNDARY_NAME = "fake"

    def __init__(self):
        self.calls = []

    def compile_continuous(self, *a, **k):
        raise AssertionError("time_series must use analyze_time_series, not compile_continuous")

    def analyze_time_series(self, payload, flat, target, rows, cols, names, lags, windows, velocity):
        self.calls.append(
            {"rows": rows, "cols": cols, "names": list(names),
             "lags": list(lags), "windows": list(windows), "velocity": velocity,
             "ts_disabled": not payload.get("enable_time_series_functions", False)}
        )
        expanded = list(names) + [f"{n}_lag1" for n in names]
        return _FakeReport(), expanded


def test_time_series_routes_to_native_expand(monkeypatch):
    fake = _FakeBoundary()
    monkeypatch.setattr(adapter, "_load_boundary", lambda: fake)
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
    )
    rep = GafimeEngine(cfg).analyze([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], feature_names=["a", "b"])

    assert fake.calls, "analyze_time_series was not called"
    call = fake.calls[0]
    assert call["lags"] == [1] and call["cols"] == 2
    assert call["ts_disabled"], "inner continuous mining must run with TS flag cleared"
    assert rep.feature_names == ["a", "b", "a_lag1", "b_lag1"]
    assert "time_series expanded" in rep.warnings[0]


def test_continuous_path_when_time_series_disabled(monkeypatch):
    fake = _FakeBoundary()
    # without the flag, analyze must NOT hit analyze_time_series
    monkeypatch.setattr(adapter, "_load_boundary", lambda: fake)
    cfg = EngineConfig(enable_time_series_functions=False, metric_names=("pearson",))
    try:
        GafimeEngine(cfg).analyze([[1.0, 2.0]], [0.0], feature_names=["a", "b"])
    except AssertionError:
        pass  # FakeBoundary.compile_continuous asserts -> proves we took the continuous path
    assert not fake.calls, "time_series path must not run when the flag is off"
