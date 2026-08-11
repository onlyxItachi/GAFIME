#!/usr/bin/env python3
from __future__ import annotations

import math
import os

import gafime


ALL_METRICS = ("pearson", "spearman", "mutual_info", "r2")
PAYLOAD_ENVS = {
    "cuda": "GAFIME_CUDA_V1_LIB",
    "rocm": "GAFIME_ROCM_V1_LIB",
    "metal": "GAFIME_METAL_V1_LIB",
}


def configured_backends() -> list[str]:
    backends = ["core", "auto"]
    for backend, env_name in PAYLOAD_ENVS.items():
        if os.environ.get(env_name):
            backends.append(backend)
    return backends


def report_map(report) -> dict[tuple[int, ...], dict[str, float]]:
    out = {}
    for item in report.interactions:
        metrics = {str(name): float(value) for name, value in item.metrics.items()}
        if set(metrics) != set(ALL_METRICS):
            raise AssertionError(
                f"{item.combo} metric keys {sorted(metrics)} != {sorted(ALL_METRICS)}"
            )
        for metric, value in metrics.items():
            if not math.isfinite(value):
                raise AssertionError(f"{item.combo} {metric} is not finite: {value}")
        out[tuple(int(value) for value in item.combo)] = metrics
    return out


def assert_compiled_matches_eager(
    cfg: gafime.EngineConfig, X, y, names: list[str], label: str
) -> object:
    engine = gafime.GafimeEngine(cfg)
    eager = engine.analyze(X, y, names)
    compiled = engine.compile(X, y, names)
    try:
        from_compiled = compiled.analyze()
    finally:
        compiled.close()
    eager_map = report_map(eager)
    compiled_map = report_map(from_compiled)
    if eager_map != compiled_map:
        raise AssertionError(f"{label}: compiled output differs from eager output")
    if not eager_map:
        raise AssertionError(f"{label}: no interactions were produced")
    return eager


def continuous_case(backend: str) -> None:
    X = [
        [0.0, 8.0, 1.0, 3.0],
        [1.0, 7.0, 1.5, 2.8],
        [2.0, 6.0, 2.0, 2.5],
        [3.0, 5.0, 2.8, 2.1],
        [4.0, 4.0, 3.1, 1.7],
        [5.0, 3.0, 3.8, 1.2],
        [6.0, 2.0, 4.4, 0.9],
        [7.0, 1.0, 5.1, 0.4],
    ]
    y = [0.0, 1.0, 2.0, 3.5, 4.2, 5.0, 6.8, 8.1]
    cfg = gafime.EngineConfig(
        backend=backend,
        precision="fp32" if backend == "metal" else "mixed",
        metric_names=ALL_METRICS,
        budget=gafime.ComputeBudget(max_comb_size=2, max_combinations_per_k=32),
        permutation_tests=0,
        num_repeats=1,
        mi_bins=16,
    )
    report = assert_compiled_matches_eager(
        cfg, X, y, ["trend", "inverse", "curved", "decay"], f"{backend}/continuous"
    )
    if not any(item.family == "interaction" for item in report.interactions):
        raise AssertionError(
            f"{backend}/continuous did not report interaction-family rows"
        )


def time_series_case(backend: str) -> None:
    X = [
        [1.0, 10.0],
        [2.0, 9.0],
        [4.0, 7.0],
        [7.0, 4.0],
        [11.0, 2.0],
        [16.0, 1.0],
        [22.0, 0.5],
        [29.0, 0.25],
        [37.0, 0.125],
        [46.0, 0.0625],
    ]
    y = [0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0]
    cfg = gafime.EngineConfig(
        backend=backend,
        precision="fp32" if backend == "metal" else "mixed",
        enable_time_series_functions=True,
        time_series_lags=(1, 2),
        time_series_windows=(3,),
        metric_names=ALL_METRICS,
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=64),
        permutation_tests=0,
        num_repeats=1,
        mi_bins=16,
    )
    report = assert_compiled_matches_eager(
        cfg, X, y, ["signal", "cost"], f"{backend}/time_series"
    )
    if not any(item.family == "time_series" for item in report.interactions):
        raise AssertionError(
            f"{backend}/time_series did not report generated time-series rows"
        )
    required = (
        "_lag1",
        "_delta1",
        "_velocity1",
        "_acceleration1",
        "_rollmean3",
        "_rollstd3",
        "_rollsum3",
    )
    missing = [
        suffix
        for suffix in required
        if not any(name.endswith(suffix) for name in report.feature_names)
    ]
    if missing:
        raise AssertionError(
            f"{backend}/time_series missing generated feature types {missing}: {report.feature_names}"
        )


def decision_path_case(backend: str) -> None:
    X: list[list[float]] = []
    y: list[float] = []
    for q0 in (0, 1):
        for q1 in (0, 1):
            for k in range(8):
                f0 = (0.15 if q0 == 0 else 0.85) + 0.002 * k
                f1 = (0.20 if q1 == 0 else 0.80) + 0.002 * k
                X.append([f0, f1, f0 * f1])
                y.append(6.0 if q0 == 1 and q1 == 1 else 0.25 * q0)
    cfg = gafime.EngineConfig(
        backend=backend,
        precision="fp32" if backend == "metal" else "mixed",
        enable_decision_path_functions=True,
        decision_path_max_depth=2,
        decision_path_rounds=1,
        decision_path_max_paths=8,
        decision_path_min_leaf=4,
        metric_names=ALL_METRICS,
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=128),
        permutation_tests=0,
        num_repeats=1,
        mi_bins=16,
    )
    report = assert_compiled_matches_eager(
        cfg, X, y, ["f0", "f1", "product"], f"{backend}/decision_path"
    )
    if not any(item.family == "decision_path" for item in report.interactions):
        raise AssertionError(
            f"{backend}/decision_path did not report generated path rows"
        )
    if not any(name.startswith("path[") for name in report.feature_names):
        raise AssertionError(f"{backend}/decision_path generated no path feature names")


def run_backend(backend: str) -> None:
    continuous_case(backend)
    time_series_case(backend)
    decision_path_case(backend)
    print(
        f"{backend}: continuous, time_series, decision_path, and {','.join(ALL_METRICS)} verified"
    )


def main() -> None:
    for backend in configured_backends():
        run_backend(backend)
    skipped = [
        backend
        for backend, env_name in PAYLOAD_ENVS.items()
        if not os.environ.get(env_name)
    ]
    if skipped:
        print(f"skipped unconfigured optional payloads: {','.join(skipped)}")
    print("all configured family/metric/backend surface checks passed")


if __name__ == "__main__":
    main()
