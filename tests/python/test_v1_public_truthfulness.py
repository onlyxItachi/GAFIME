from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
_INSTALLED_PACKAGE_MODE = os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") == "1"
if not _INSTALLED_PACKAGE_MODE and str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
from gafime.errors import V1UnsupportedError  # noqa: E402
from gafime.v1_adapter import (  # noqa: E402
    _significance_from_native,
    analyze_arrow_with_v1_boundary,
)
from gafime import v1_adapter  # noqa: E402


class _Series:
    def __init__(self, values):
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def null_count(self):
        return sum(value is None for value in self._values)


class _Frame:
    def __init__(self, rows, columns):
        self._rows = [tuple(row) for row in rows]
        self.columns = list(columns)
        self.width = len(self.columns)

    def iter_rows(self):
        return iter(self._rows)

    def get_columns(self):
        return [_Series(row[index] for row in self._rows) for index in range(self.width)]


class _ArrowSeries:
    def __init__(self, values):
        self._values = list(values)
        self.null_count = sum(value is None for value in self._values)


class _ArrowTargetFrame:
    def __init__(self, values):
        self.num_columns = 1
        self._column = _ArrowSeries(values)

    def column(self, index):
        assert index == 0
        return self._column


class _NativeReport:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(
        self,
        *,
        metric_count=1,
        significance=False,
        pvalue=0.01,
        std=0.01,
        device="cpu",
        graph_replayed=False,
    ):
        self.metric_count = metric_count
        self.significance = significance
        self.pvalue = pvalue
        self.std = std
        self.device = device
        self.is_gpu = device != "cpu"
        self.backend_name = f"v1-{device}-cabi" if self.is_gpu else "v1-rust-cpu"
        self.graph_replayed = graph_replayed
        self.pvalue_calls = 0
        self.mean_calls = 0
        self.std_calls = 0

    def __len__(self):
        return 1

    def combo(self, index):
        assert index == 0
        return [0]

    def metric_values(self, index):
        assert index == 0
        return [1.0] * self.metric_count

    def candidate_id(self, index):
        assert index == 0
        return 7

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
        return [0]

    def has_significance(self):
        return self.significance

    def significance_rows(self):
        return [0]

    def significance_pvalues(self):
        self.pvalue_calls += 1
        return [[self.pvalue] * self.metric_count]

    def significance_means(self):
        self.mean_calls += 1
        return [[1.0] * self.metric_count]

    def significance_stds(self):
        self.std_calls += 1
        return [[self.std] * self.metric_count]


class _NativeArtifact:
    def __init__(self, config, *, honor_graph=True):
        self.config = config
        self.device = str(config["backend"]).replace("core", "cpu")
        self.is_gpu = self.device in {"cuda", "rocm", "hip", "metal"}
        self.backend_name = f"v1-{self.device}-cabi" if self.is_gpu else "v1-rust-cpu"
        requested = bool(config.get("compile_flags", {}).get("graph", False))
        self.graph_requested = requested and honor_graph
        self.closed = False
        self.updated_targets = []

    def analyze(self):
        return _NativeReport(
            metric_count=len(self.config["metric_names"]),
            significance=(
                int(self.config["permutation_tests"]) > 0
                or int(self.config["num_repeats"]) > 1
            ),
            device=self.device,
            graph_replayed=self.graph_requested,
        )

    def close(self):
        self.closed = True

    def update_target(self, target):
        self.updated_targets.append(list(target))


def _boundary(*, honor_graph=True):
    configured_calls = []
    raw_arrow_calls = []
    artifacts = []

    def compile_continuous(config, features, target, *, rows, cols):
        configured_calls.append(
            {
                "config": config,
                "features": list(features),
                "target": list(target),
                "rows": rows,
                "cols": cols,
            }
        )
        artifact = _NativeArtifact(config, honor_graph=honor_graph)
        artifacts.append(artifact)
        return artifact

    def analyze_continuous_arrow(
        feature_frame,
        target_frame,
        *,
        max_arity,
        max_combinations_per_k,
        metric_ids,
    ):
        raw_arrow_calls.append(
            {
                "max_arity": max_arity,
                "max_combinations_per_k": max_combinations_per_k,
                "metric_ids": metric_ids,
            }
        )
        return _NativeReport(metric_count=len(metric_ids))

    return SimpleNamespace(
        BOUNDARY_NAME="fake-gafime-py",
        compile_continuous=compile_continuous,
        analyze_continuous_arrow=analyze_continuous_arrow,
        configured_calls=configured_calls,
        raw_arrow_calls=raw_arrow_calls,
        artifacts=artifacts,
    )


def _frames(*, target=(1.0, 2.0, 3.0)):
    features = _Frame([(1.0,), (2.0,), (3.0,)], ["x"])
    target_frame = _Frame([(value,) for value in target], ["y"])
    return features, target_frame


def test_arrow_routes_full_config_through_normal_boundary(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    features, target = _frames()
    config = EngineConfig(
        backend="cuda",
        metric_names=("pearson",),
        permutation_tests=7,
        num_repeats=1,
        mi_bins=24,
        budget=ComputeBudget(max_comb_size=1, keep_in_vram=False),
    )

    report = analyze_arrow_with_v1_boundary(config, features, target, ["x"])

    assert boundary.raw_arrow_calls == []
    assert len(boundary.configured_calls) == 1
    payload = boundary.configured_calls[0]["config"]
    assert payload["backend"] == "cuda"
    assert payload["permutation_tests"] == 7
    assert payload["mi_bins"] == 24
    assert len(report.permutations) == 1
    assert report.stability == []
    assert report.backend.device == "cuda"


def test_arrow_cpu_shortcut_is_used_only_for_compatible_config(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    features, target = _frames()
    config = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
    )

    report = analyze_arrow_with_v1_boundary(config, features, target, ["x"])

    assert len(boundary.raw_arrow_calls) == 1
    assert boundary.configured_calls == []
    assert report.backend.device == "cpu"


@pytest.mark.parametrize(
    "config",
    [
        EngineConfig(backend="CPU", metric_names=("pearson",), permutation_tests=0, num_repeats=1),
        EngineConfig(backend="cpu", metric_names=("pearson",), permutation_tests=-1, num_repeats=1),
        EngineConfig(backend="cpu", metric_names=("pearson",), permutation_tests=0, num_repeats=0),
    ],
)
def test_arrow_shortcut_does_not_normalize_config_semantics(monkeypatch, config):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    features, target = _frames()

    analyze_arrow_with_v1_boundary(config, features, target, ["x"])

    assert boundary.raw_arrow_calls == []
    assert len(boundary.configured_calls) == 1
    payload = boundary.configured_calls[0]["config"]
    assert payload["backend"] == config.backend
    assert payload["permutation_tests"] == config.permutation_tests
    assert payload["num_repeats"] == config.num_repeats


def test_arrow_target_validation_supports_property_protocol(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    features, _ = _frames()
    target = _ArrowTargetFrame([1.0, 2.0, 3.0])
    config = EngineConfig(backend="cpu", permutation_tests=0, num_repeats=1)

    report = analyze_arrow_with_v1_boundary(config, features, target, ["x"])

    assert len(boundary.raw_arrow_calls) == 1
    assert report.backend.device == "cpu"


@pytest.mark.parametrize(
    "target_frame, message",
    [
        (_Frame([(1.0, 2.0)], ["y", "extra"]), "exactly one column"),
        (_Frame([(1.0,), (None,)], ["y"]), "must not contain null"),
    ],
)
def test_arrow_rejects_invalid_target_before_native_call(monkeypatch, target_frame, message):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    features = _Frame([(1.0,), (2.0,)], ["x"])
    config = EngineConfig(backend="cpu", permutation_tests=0, num_repeats=1)

    with pytest.raises(ValueError, match=message):
        analyze_arrow_with_v1_boundary(config, features, target_frame, ["x"])

    assert boundary.raw_arrow_calls == []
    assert boundary.configured_calls == []


@pytest.mark.parametrize(
    "target_data, message",
    [
        ({"y": [1.0, 2.0], "extra": [3.0, 4.0]}, "exactly one column"),
        ({"y": [1.0, None]}, "null target values"),
    ],
)
def test_native_arrow_boundary_rejects_invalid_target(target_data, message):
    if _INSTALLED_PACKAGE_MODE:
        import polars as pl
        import gafime.gafime_py as boundary
    else:
        pl = pytest.importorskip("polars")
        boundary = pytest.importorskip("gafime.gafime_py")
    features = pl.DataFrame({"x": [1.0, 2.0]}).cast(pl.Float32).rechunk()
    target = pl.DataFrame(target_data).cast(pl.Float32).rechunk()

    with pytest.raises(ValueError, match=message):
        boundary.analyze_continuous_arrow(
            features,
            target,
            max_arity=1,
            max_combinations_per_k=2,
            metric_ids=[1],
        )


def test_compile_plan_flag_controls_scenario_plan(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    config = EngineConfig(backend="cpu", permutation_tests=0, num_repeats=1)

    hidden = GafimeEngine(config).compile(
        [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(plan=False)
    )
    exposed = GafimeEngine(config).compile(
        [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(plan=True)
    )

    assert hidden.scenario_plan is None
    assert exposed.scenario_plan is boundary.artifacts[1]
    assert boundary.configured_calls[0]["config"]["compile_flags"]["plan"] is False
    assert boundary.configured_calls[1]["config"]["compile_flags"]["plan"] is True


@pytest.mark.parametrize("backend_name", ["cpu", "metal"])
def test_graph_request_rejects_unsupported_backend(monkeypatch, backend_name):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    config = EngineConfig(backend=backend_name, permutation_tests=0, num_repeats=1)

    with pytest.raises(V1UnsupportedError, match="graph=True.*unsupported"):
        GafimeEngine(config).compile(
            [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(graph=True)
        )

    assert boundary.configured_calls == []


def test_graph_request_requires_native_activation_and_replay(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    config = EngineConfig(backend="cuda", permutation_tests=0, num_repeats=1)

    artifact = GafimeEngine(config).compile(
        [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(graph=True)
    )
    assert artifact.graph_requested is True
    assert artifact.graph_replayed is False
    artifact.analyze()
    assert artifact.graph_replayed is True
    assert boundary.configured_calls[0]["config"]["compile_flags"]["graph"] is True

    no_graph_boundary = _boundary(honor_graph=False)
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: no_graph_boundary)
    with pytest.raises(V1UnsupportedError, match="does not expose activated graph"):
        GafimeEngine(config).compile(
            [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(graph=True)
        )
    assert no_graph_boundary.artifacts[0].closed is True


def test_graph_replay_state_resets_before_each_execution(monkeypatch):
    boundary = _boundary()
    monkeypatch.setattr(v1_adapter, "_load_boundary", lambda: boundary)
    config = EngineConfig(backend="cuda", permutation_tests=0, num_repeats=1)
    artifact = GafimeEngine(config).compile(
        [[1.0], [2.0]], [1.0, 2.0], ["x"], flags=CompileFlags(graph=True)
    )

    artifact.analyze()
    assert artifact.graph_replayed is True
    artifact.update_target([2.0, 1.0])
    assert artifact.graph_replayed is False

    artifact.native_handle.analyze = lambda: _NativeReport(
        metric_count=1,
        device="cuda",
        graph_replayed=False,
    )
    with pytest.raises(V1UnsupportedError, match="did not confirm graph replay"):
        artifact.analyze()
    assert artifact.graph_replayed is False


def test_permutation_only_reports_and_gates_only_permutation():
    native = _NativeReport(significance=True, pvalue=0.01, std=9.0)
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=9,
        num_repeats=1,
    )

    stability, permutations, decision = _significance_from_native(native, ["x"], config)

    assert stability == []
    assert len(permutations) == 1
    assert decision.signal_detected is True
    assert "permutation" in decision.message
    assert "stability" not in decision.message
    assert native.pvalue_calls == 1
    assert native.mean_calls == native.std_calls == 0


def test_stability_only_reports_and_gates_only_stability():
    native = _NativeReport(significance=True, pvalue=9.0, std=0.01)
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=3,
    )

    stability, permutations, decision = _significance_from_native(native, ["x"], config)

    assert len(stability) == 1
    assert permutations == []
    assert decision.signal_detected is True
    assert "stability" in decision.message
    assert "permutation" not in decision.message
    assert native.pvalue_calls == 0
    assert native.mean_calls == native.std_calls == 1


def test_significance_identity_matches_generated_interaction_family():
    native = _NativeReport(significance=True, pvalue=0.01, std=0.01)
    native.combo = lambda index: [1]
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=2,
    )

    stability, permutations, _decision = _significance_from_native(
        native,
        ["base", "base_lag1"],
        config,
    )

    assert stability[0].family == permutations[0].family == "time_series"
    assert stability[0].candidate_id == permutations[0].candidate_id == "time_series:7"
    assert stability[0].expression == permutations[0].expression == "base_lag1"


def test_requested_significance_cannot_fall_back_to_interaction_success():
    native = _NativeReport(significance=False)
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=1,
    )

    with pytest.raises(V1UnsupportedError, match="requested permutation"):
        _significance_from_native(native, ["x"], config)


def test_requested_significance_requires_exact_row_shape():
    native = _NativeReport(significance=True)
    native.significance_pvalues = lambda: []
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=1,
    )

    with pytest.raises(V1UnsupportedError, match="row count 0.*row count 1"):
        _significance_from_native(native, ["x"], config)


@pytest.mark.parametrize("pvalue", [float("nan"), -0.1, 1.1])
def test_permutation_reporting_rejects_invalid_pvalues(pvalue):
    native = _NativeReport(significance=True, pvalue=pvalue)
    config = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=1,
    )

    with pytest.raises(V1UnsupportedError, match=r"p-values in \[0, 1\]"):
        _significance_from_native(native, ["x"], config)
