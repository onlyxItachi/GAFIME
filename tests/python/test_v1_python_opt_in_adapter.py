from __future__ import annotations

import os
import sys
import types

from gafime import ComputeBudget, EngineConfig, GafimeEngine


class _FakeRecord:
    def __init__(self, combo, metrics):
        self.combo = combo
        self.metrics = metrics
        self.candidate_id = 0


class _FakeReport:
    rows = 4
    cols = 2
    max_arity = 1
    metric_ids = [1, 4]

    def records(self):
        return [
            _FakeRecord([0], [1.0, 1.0]),
            _FakeRecord([1], [-1.0, 1.0]),
        ]


def _install_fake_boundary(name: str):
    module = types.ModuleType(name)
    calls = []

    def analyze_continuous_cpu(features, target, *, max_arity, max_combinations_per_k, metric_ids):
        calls.append(
            {
                "features": features,
                "target": target,
                "max_arity": max_arity,
                "max_combinations_per_k": max_combinations_per_k,
                "metric_ids": metric_ids,
            }
        )
        return _FakeReport()

    module.analyze_continuous_cpu = analyze_continuous_cpu
    module.calls = calls
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


def test_v1_engine_env_uses_python_adapter_boundary():
    module_name = "_fake_gafime_v1_boundary"
    fake = _install_fake_boundary(module_name)
    old_v1 = _set_env("GAFIME_V1_ENGINE", "1")
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
        _restore_env("GAFIME_V1_ENGINE", old_v1)
        _restore_env("GAFIME_V1_BOUNDARY_MODULE", old_module)
        sys.modules.pop(module_name, None)

    assert fake.calls
    assert fake.calls[0]["metric_ids"] == [1, 4]
    assert fake.calls[0]["max_arity"] == 1
    assert report.backend.name == "v1-rust-cpu"
    assert [item.combo for item in report.interactions] == [(0,), (1,)]
    assert report.interactions[0].metrics == {"pearson": 1.0, "r2": 1.0}
    assert report.warnings == ["GAFIME_V1_ENGINE=1 used experimental Rust v1 continuous boundary."]


def test_legacy_env_takes_priority_over_v1_opt_in():
    old_v1 = _set_env("GAFIME_V1_ENGINE", "1")
    old_legacy = _set_env("GAFIME_USE_LEGACY_ENGINE", "1")
    try:
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze(
            [[1.0], [2.0], [3.0], [4.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a"],
        )
    finally:
        _restore_env("GAFIME_V1_ENGINE", old_v1)
        _restore_env("GAFIME_USE_LEGACY_ENGINE", old_legacy)

    assert report.backend.name == "core"
