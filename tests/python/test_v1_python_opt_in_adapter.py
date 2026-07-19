from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
from gafime.errors import V1UnsupportedError  # noqa: E402


class _FakeRecord:
    def __init__(self, combo, metrics, candidate_id):
        self.combo = combo
        self.metrics = metrics
        self.candidate_id = candidate_id


class _FakeReport:
    rows = 4
    cols = 2
    max_arity = 1
    metric_ids = [1, 4]

    def __init__(self):
        self._records = [
            _FakeRecord([0], [1.0, 1.0], 0),
            _FakeRecord([1], [-1.0, 1.0], 1),
        ]
        self.combo_calls = 0
        self.metric_value_calls = 0
        self.candidate_id_calls = 0
        self.rank_calls = 0

    def __len__(self):
        return len(self._records)

    def record(self, index):
        raise AssertionError("normal v1 report access must not materialize native records")

    def records(self):
        raise AssertionError("normal v1 report access must not request native record lists")

    def combo(self, index):
        self.combo_calls += 1
        return self._records[index].combo

    def metric_values(self, index):
        self.metric_value_calls += 1
        return self._records[index].metrics

    def candidate_id(self, index):
        self.candidate_id_calls += 1
        return self._records[index].candidate_id

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
        self.rank_calls += 1
        indices = [0, 1]
        if limit is not None:
            indices = indices[:limit]
        return indices


class _FakeArtifact:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self):
        self.closed = False
        self.updated_targets = []

    def analyze(self):
        return _FakeReport()

    def update_target(self, target):
        self.updated_targets.append(list(target))

    def close(self):
        self.closed = True


class _FakeComponentReport(_FakeReport):
    def __init__(self):
        super().__init__()
        self.component_calls = 0
        self.batch_calls = 0

    def interaction_components(self, index):
        self.component_calls += 1
        record = self._records[index]
        return record.combo, record.metrics, record.candidate_id

    def interaction_components_batch(self, start, limit):
        self.batch_calls += 1
        return [
            (record.combo, record.metrics, record.candidate_id)
            for record in self._records[start : start + limit]
        ]


def _install_fake_boundary(name: str):
    module = types.ModuleType(name)
    calls = []
    analyze_calls = []
    artifacts = []

    def compile_continuous(config, features, target, *, rows, cols):
        calls.append(
            {
                "config": config,
                "features": features,
                "target": target,
                "rows": rows,
                "cols": cols,
            }
        )
        artifact = _FakeArtifact()
        artifacts.append(artifact)
        return artifact

    def analyze_continuous(config, features, target, *, rows, cols):
        analyze_calls.append(
            {
                "config": config,
                "features": features,
                "target": target,
                "rows": rows,
                "cols": cols,
            }
        )
        return _FakeReport()

    module.compile_continuous = compile_continuous
    module.analyze_continuous = analyze_continuous
    module.calls = calls
    module.analyze_calls = analyze_calls
    module.artifacts = artifacts
    module.BOUNDARY_NAME = "fake-gafime-py"
    sys.modules[name] = module
    return module


def _set_env(key: str, value: str | None):
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return old


def _restore_env(key: str, value: str | None):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def test_v1_engine_uses_boundary_by_default():
    module_name = "_fake_gafime_v1_boundary"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b"],
        )
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)

    assert fake.calls
    assert fake.calls[0]["config"]["metric_names"] == ["pearson", "r2"]
    assert fake.calls[0]["config"]["mi_approximate"] is False
    assert fake.calls[0]["config"]["budget"]["max_comb_size"] == 1
    assert fake.calls[0]["features"] == [1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]
    assert fake.calls[0]["rows"] == 4
    assert fake.calls[0]["cols"] == 2
    assert report.backend.name == "v1-rust-cpu"
    assert report.interactions.is_native_backed
    assert [item.combo for item in report.interactions] == [(0,), (1,)]
    assert report.interactions[0].metrics == {"pearson": 1.0, "r2": 1.0}
    assert report.warnings == []


def test_mi_approximate_reaches_native_boundary():
    module_name = "_fake_gafime_v1_boundary_mi_approx"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("mutual_info",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
            mi_bins=24,
            mi_approximate=True,
        )
        GafimeEngine(cfg).analyze(
            [[0.0], [1.0], [2.0], [3.0]],
            [0.0, 1.0, 1.0, 0.0],
            ["a"],
        )
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)

    assert fake.calls
    assert fake.calls[0]["config"]["mi_bins"] == 24
    assert fake.calls[0]["config"]["mi_approximate"] is True


def test_legacy_env_no_longer_overrides_v1_boundary():
    module_name = "_fake_gafime_v1_boundary"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    old_legacy = _set_env("GAFIME_USE_LEGACY_ENGINE", "1")
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.0, 3.0, 4.0], ["a"])
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        _restore_env("GAFIME_USE_LEGACY_ENGINE", old_legacy)
        sys.modules.pop(module_name, None)

    assert fake.calls
    assert report.backend.name == "v1-rust-cpu"


def test_public_analyze_reuses_resident_cache_and_invalidates_on_feature_change():
    from gafime import v1_adapter

    module_name = "_fake_gafime_v1_boundary_resident_analyze_cache"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    old_cache_size = _set_env("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        X = [[1.0], [2.0], [3.0], [4.0]]
        y1 = [1.0, 2.0, 3.0, 4.0]
        y2 = [4.0, 3.0, 2.0, 1.0]

        GafimeEngine(cfg).analyze(X, y1, ["a"])
        GafimeEngine(cfg).analyze(X, y1, ["a"])
        assert len(fake.calls) == 1
        assert fake.artifacts[0].updated_targets == []

        GafimeEngine(cfg).analyze(X, y2, ["a"])
        assert len(fake.calls) == 1
        assert fake.artifacts[0].updated_targets == [y2]

        X[0][0] = 10.0
        GafimeEngine(cfg).analyze(X, y2, ["a"])
        assert len(fake.calls) == 2
        assert fake.calls[1]["features"][0] == 10.0
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        _restore_env("GAFIME_V1_ANALYZE_CACHE_SIZE", old_cache_size)
        sys.modules.pop(module_name, None)


def test_public_analyze_cache_respects_keep_in_vram_false():
    from gafime import v1_adapter

    module_name = "_fake_gafime_v1_boundary_resident_analyze_cache_disabled"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    old_cache_size = _set_env("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8, keep_in_vram=False),
            permutation_tests=0,
            num_repeats=1,
        )
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        GafimeEngine(cfg).analyze(X, y, ["a"])
        GafimeEngine(cfg).analyze(X, y, ["a"])
        assert fake.calls == []
        assert len(fake.analyze_calls) == 2
        assert fake.artifacts == []
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        _restore_env("GAFIME_V1_ANALYZE_CACHE_SIZE", old_cache_size)
        sys.modules.pop(module_name, None)


def test_cache_disabled_custom_boundary_without_one_shot_keeps_compile_compatibility():
    from gafime import v1_adapter

    module_name = "_fake_gafime_v1_boundary_compile_only"
    fake = _install_fake_boundary(module_name)
    del fake.analyze_continuous
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    old_cache_size = _set_env("GAFIME_V1_ANALYZE_CACHE_SIZE", "0")
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        GafimeEngine(cfg).analyze(
            [[1.0], [2.0], [3.0], [4.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a"],
        )
        assert len(fake.calls) == 1
        assert fake.artifacts[0].closed
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        _restore_env("GAFIME_V1_ANALYZE_CACHE_SIZE", old_cache_size)
        sys.modules.pop(module_name, None)


def test_shipped_style_nested_boundary_avoids_python_flattening():
    from gafime import v1_adapter

    module_name = "_fake_gafime_v1_boundary_nested_rows"
    fake = _install_fake_boundary(module_name)
    row_analyze_calls = []
    row_compile_calls = []

    def analyze_continuous_rows(config, features, target):
        row_analyze_calls.append((config, features, target))
        return _FakeReport()

    def compile_continuous_rows(config, features, target):
        row_compile_calls.append((config, features, target))
        artifact = _FakeArtifact()
        fake.artifacts.append(artifact)
        return artifact

    fake.analyze_continuous_rows = analyze_continuous_rows
    fake.compile_continuous_rows = compile_continuous_rows
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    old_cache_size = _set_env("GAFIME_V1_ANALYZE_CACHE_SIZE", "0")
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        GafimeEngine(cfg).analyze(X, y, ["a"])
        artifact = GafimeEngine(cfg).compile(X, y, ["a"])
        artifact.close()
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        _restore_env("GAFIME_V1_ANALYZE_CACHE_SIZE", old_cache_size)
        sys.modules.pop(module_name, None)

    assert len(row_analyze_calls) == 1
    assert row_analyze_calls[0][1] is X
    assert row_analyze_calls[0][2] is y
    assert len(row_compile_calls) == 1
    assert row_compile_calls[0][1] is X
    assert row_compile_calls[0][2] is y
    assert fake.calls == []
    assert fake.analyze_calls == []


def test_native_report_view_ranks_lazily_and_materializes_only_for_export():
    module_name = "_fake_gafime_v1_boundary_report_view"
    fake = _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b"],
        )
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)

    assert fake.calls
    native = report.interactions.native_handle
    assert report.interactions.is_native_backed
    assert native.combo_calls == 0
    assert native.metric_value_calls == 0
    assert native.candidate_id_calls == 0
    top = report.interactions.top_k(1, metric_name="pearson")
    assert top.native_handle is native
    assert top.native_indices == (0,)
    assert native.rank_calls == 1
    assert native.combo_calls == 0

    first = top[0]
    assert first.combo == (0,)
    assert first.metrics == {"pearson": 1.0, "r2": 1.0}
    assert native.combo_calls == 1
    assert native.metric_value_calls == 1
    assert native.candidate_id_calls == 1

    with pytest.warns(DeprecationWarning, match="materializes the native report"):
        exported = report.to_dict()
    assert exported["interactions"][0]["combo"] == [0]
    assert exported["interactions"][1]["combo"] == [1]


def test_native_report_view_uses_one_component_call_when_available():
    from gafime.reporting import NativeContinuousInteractions

    native = _FakeComponentReport()
    interactions = NativeContinuousInteractions(native, ["a", "b"], ["pearson", "r2"])

    first = interactions[0]

    assert first.combo == (0,)
    assert first.metrics == {"pearson": 1.0, "r2": 1.0}
    assert native.component_calls == 1
    assert native.combo_calls == 0
    assert native.metric_value_calls == 0
    assert native.candidate_id_calls == 0

    materialized = list(interactions)
    assert [item.combo for item in materialized] == [(0,), (1,)]
    assert native.batch_calls == 1
    assert native.component_calls == 1


def test_v1_compile_export_flag_gates_result_export():
    module_name = "_fake_gafime_v1_boundary"
    _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    try:
        from gafime.compile import CompileFlags

        cfg = EngineConfig(metric_names=("pearson",), budget=ComputeBudget(max_comb_size=1))
        # Compiling WITH export=True now succeeds — export handles are wired.
        artifact = GafimeEngine(cfg).compile(
            [[1.0], [2.0], [3.0]],
            [1.0, 2.0, 3.0],
            ["a"],
            flags=CompileFlags(export=True),
        )
        assert artifact.export is True

        # Without the flag, requesting an export handle raises an explicit v1 error.
        plain = GafimeEngine(cfg).compile([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], ["a"])
        with pytest.raises(V1UnsupportedError):
            plain.__arrow_c_array__()
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)
