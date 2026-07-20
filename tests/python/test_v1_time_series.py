"""time_series family wiring test (no native wheel needed — fake boundary).

Verifies that enabling time_series routes analyze() to the native
analyze_time_series expand+mine path and surfaces the expanded feature names.
"""

import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime.v1_adapter as adapter  # noqa: E402
from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


class _FakeReport:
    def __len__(self):
        return 0

    def records(self):
        return []


class _FakeArtifact:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self, report):
        self._report = report
        self.closed = False
        self.rows = 2

    def analyze(self):
        return self._report

    def close(self):
        self.closed = True


class _FakeBoundary:
    BOUNDARY_NAME = "fake"

    def __init__(self):
        self.calls = []

    def compile_continuous(self, *a, **k):
        raise AssertionError(
            "time_series must use compile_time_series, not compile_continuous"
        )

    def analyze_time_series(
        self, payload, flat, target, rows, cols, names, lags, windows, velocity
    ):
        self.calls.append(
            {
                "path": "analyze",
                "rows": rows,
                "cols": cols,
                "names": list(names),
                "lags": list(lags),
                "windows": list(windows),
                "velocity": velocity,
                "ts_disabled": not payload.get("enable_time_series_functions", False),
            }
        )
        expanded = list(names) + [f"{n}_lag1" for n in names]
        return _FakeReport(), expanded

    def compile_time_series(
        self, payload, flat, target, rows, cols, names, lags, windows, velocity
    ):
        self.calls.append(
            {
                "path": "compile",
                "rows": rows,
                "cols": cols,
                "names": list(names),
                "lags": list(lags),
                "windows": list(windows),
                "velocity": velocity,
                "ts_disabled": not payload.get("enable_time_series_functions", False),
            }
        )
        expanded = list(names) + [f"{n}_lag1" for n in names]
        return _FakeArtifact(_FakeReport()), expanded


def test_time_series_routes_to_native_expand(monkeypatch):
    fake = _FakeBoundary()
    monkeypatch.setattr(adapter, "_load_boundary", lambda: fake)
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
    )
    rep = GafimeEngine(cfg).analyze(
        [[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], feature_names=["a", "b"]
    )

    assert fake.calls, "analyze_time_series was not called"
    call = fake.calls[0]
    assert call["path"] == "analyze"
    assert call["lags"] == [1] and call["cols"] == 2
    assert call["ts_disabled"], "inner continuous mining must run with TS flag cleared"
    assert rep.feature_names == ["a", "b", "a_lag1", "b_lag1"]
    assert "time_series expanded" in rep.warnings[0]


def test_time_series_compile_returns_expanded_resident_artifact(monkeypatch):
    fake = _FakeBoundary()
    fake.BOUNDARY_NAME = "gafime-py"
    monkeypatch.setattr(adapter, "_load_boundary", lambda: fake)
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
    )
    artifact = GafimeEngine(cfg).compile(
        [[1.0, 2.0], [3.0, 4.0]],
        [0.0, 1.0],
        feature_names=["a", "b"],
    )
    try:
        assert artifact.feature_names == ["a", "b", "a_lag1", "b_lag1"]
        plan = artifact.scenario_plan
        assert plan.n_features == 2
        assert plan.feature_candidate_count == 2
        assert plan.planned_count == 11
        assert plan.time_series is not None
        assert plan.time_series.feature_stop == 2
        report = artifact.analyze()
        assert report.feature_names == artifact.feature_names
        assert "time_series expanded" in report.warnings[0]
    finally:
        artifact.close()


@pytest.mark.parametrize(
    ("top_k_features", "max_candidates", "expected_generated"),
    [(0, 8, 0), (1, 0, 0), (1, 2, 2)],
)
def test_time_series_generation_caps_bound_executed_candidates(
    top_k_features, max_candidates, expected_generated
):
    pytest.importorskip("gafime.gafime_py")
    rows = 8
    X = [[float(row), float(row * 10), float(row * 100)] for row in range(rows)]
    y = [float(row) for row in range(rows)]
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=64,
            top_k_features_for_time_series=top_k_features,
            max_time_series_candidates=max_candidates,
        ),
    )

    report = GafimeEngine(cfg).analyze(X, y, feature_names=["a", "b", "c"])
    generated = report.feature_names[3:]
    assert len(generated) == expected_generated
    assert len(report.interactions) == 3 + expected_generated
    if expected_generated == 2:
        assert generated == ["a_lag1", "a_delta1"]


def test_time_series_source_selection_uses_unary_strength_for_eager_and_compiled():
    pytest.importorskip("gafime.gafime_py")
    rows = 10
    X = [[0.0, 1.0, float(row)] for row in range(rows)]
    y = [float(row) for row in range(rows)]
    names = ["prefix_zero", "prefix_one", "signal"]
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=64,
            top_k_features_for_time_series=1,
            max_time_series_candidates=1,
        ),
    )

    eager = GafimeEngine(cfg).analyze(X, y, feature_names=names)
    artifact = GafimeEngine(cfg).compile(X, y, feature_names=names)
    try:
        compiled = artifact.analyze()
        assert eager.feature_names == [*names, "signal_lag1"]
        assert artifact.feature_names == eager.feature_names
        assert compiled.feature_names == eager.feature_names
        assert [item.combo for item in compiled.interactions] == [
            item.combo for item in eager.interactions
        ]
    finally:
        artifact.close()


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


def test_time_series_carries_significance_when_requested():
    pytest.importorskip("gafime.gafime_py")
    rows = 80
    X = [[float(i)] for i in range(rows)]
    y = [0.0] + [float(i - 1) for i in range(1, rows)]
    cfg = EngineConfig(
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
        permutation_tests=50,
        num_repeats=5,
        permutation_p_threshold=0.05,
        stability_std_threshold=0.10,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=16),
    )
    report = GafimeEngine(cfg).analyze(X, y, feature_names=["x"])
    assert len(report.permutations) > 0
    assert len(report.stability) > 0
    assert report.decision.signal_detected is True
    assert any(name == "x_lag1" for name in report.feature_names)
