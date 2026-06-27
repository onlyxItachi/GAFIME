from __future__ import annotations

import os
import sys
import types

import pytest

from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.errors import V1UnsupportedError


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

    def __len__(self):
        return len(self._records)

    def record(self, index):
        return self._records[index]

    def combo(self, index):
        return self._records[index].combo

    def metric_values(self, index):
        return self._records[index].metrics

    def candidate_id(self, index):
        return self._records[index].candidate_id

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
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

    def analyze(self):
        return _FakeReport()

    def close(self):
        self.closed = True


def _install_fake_boundary(name: str):
    module = types.ModuleType(name)
    calls = []

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
        return _FakeArtifact()

    module.compile_continuous = compile_continuous
    module.calls = calls
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
    assert fake.calls[0]["config"]["budget"]["max_comb_size"] == 1
    assert fake.calls[0]["features"] == [1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]
    assert fake.calls[0]["rows"] == 4
    assert fake.calls[0]["cols"] == 2
    assert report.backend.name == "v1-rust-cpu"
    assert report.interactions.is_native_backed
    assert [item.combo for item in report.interactions] == [(0,), (1,)]
    assert report.interactions[0].metrics == {"pearson": 1.0, "r2": 1.0}
    assert report.warnings == []


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


def test_v1_compile_rejects_export_handles_until_native_export_is_wired():
    module_name = "_fake_gafime_v1_boundary"
    _install_fake_boundary(module_name)
    old_module = _set_env("GAFIME_V1_BOUNDARY_MODULE", module_name)
    try:
        from gafime.compile import CompileFlags

        cfg = EngineConfig(metric_names=("pearson",), budget=ComputeBudget(max_comb_size=1))
        with pytest.raises(V1UnsupportedError):
            GafimeEngine(cfg).compile(
                [[1.0], [2.0], [3.0]],
                [1.0, 2.0, 3.0],
                ["a"],
                flags=CompileFlags(export=True),
            )
    finally:
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)
