"""Hardcore installed-wheel benchmark for the native Metal backend.

This script is intentionally stdlib-only so GitHub macOS runners can execute it
against a freshly built wheel without adding benchmark dependencies. It stresses
the public engine path and the lower-level Metal backend APIs directly, while
writing machine-readable JSON for later performance analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


def make_dataset(
    n_samples: int,
    n_features: int,
    *,
    seed: int,
) -> tuple[list[list[float]], list[float], list[str]]:
    if n_samples < 512:
        raise ValueError("Metal hardcore benchmark expects at least 512 samples.")
    if n_features < 12:
        raise ValueError("Metal hardcore benchmark expects at least 12 features.")

    rng = random.Random(seed)
    rows: list[list[float]] = []
    for i in range(n_samples):
        row: list[float] = []
        for j in range(n_features):
            base = (
                math.sin(0.0017 * (i + 1) * (j + 1))
                + 0.45 * math.cos(0.0041 * (i + 3) * (j + 2))
                + 0.025 * (((i * (j + 5)) % 37) - 18)
                + rng.uniform(-0.004, 0.004)
            )
            row.append(float(base))

        # Edge-case columns that still remain finite and useful.
        row[n_features - 1] = 1.0
        row[n_features - 2] = -1.0 if (i % 2) else 1.0
        row[n_features - 3] = float(((i // 17) % 9) - 4) / 4.0
        if i % 4096 == 0:
            row[n_features - 4] += 7.5
        rows.append(row)

    targets: list[float] = []
    for i, row in enumerate(rows):
        lag = rows[max(i - 8, 0)][8]
        rolling_start = max(i - 16, 0)
        rolling = sum(rows[k][9] for k in range(rolling_start, i + 1)) / float(i - rolling_start + 1)
        gate = sigmoid(14.0 * (row[2] - 0.20))
        rectangle = (
            sigmoid(11.0 * (row[6] + 0.40))
            * sigmoid(11.0 * (0.80 - row[6]))
            * sigmoid(11.0 * (row[7] + 0.25))
            * sigmoid(11.0 * (0.90 - row[7]))
        )
        high_order = row[0] * row[1] * row[3] * row[4] * row[5]
        y = (
            0.34 * row[0] * row[1]
            + 0.21 * row[3] * row[4] * row[5]
            + 0.08 * high_order
            + 1.55 * gate
            + 1.10 * rectangle
            + 0.28 * lag
            + 0.20 * rolling
            + 0.03 * row[n_features - 2]
            + rng.uniform(-0.010, 0.010)
        )
        targets.append(float(y))
    return rows, targets, [f"f{i}" for i in range(n_features)]


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_continuous_combos(
    n_features: int,
    *,
    max_arity: int,
    max_candidates: int,
) -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = []
    for arity in range(1, max_arity + 1):
        combos.extend(itertools.combinations(range(n_features), arity))
    if max_candidates > 0 and len(combos) > max_candidates:
        head = combos[: max_candidates // 3]
        tail = combos[-max_candidates // 3 :]
        stride_count = max_candidates - len(head) - len(tail)
        stride = max(1, len(combos) // max(1, stride_count))
        middle = combos[::stride][:stride_count]
        combos = list(dict.fromkeys(head + middle + tail))
    return combos


def build_time_series_candidates(n_features: int):
    from gafime.time_series import TimeSeriesCandidate

    candidates: list[TimeSeriesCandidate] = []
    feature_count = min(n_features, 16)
    for feature in range(feature_count):
        for lag in (1, 2, 4, 8, 16, 32):
            for kind in (
                "time_series_lag",
                "time_series_delta",
                "time_series_velocity",
                "time_series_acceleration",
            ):
                candidates.append(
                    TimeSeriesCandidate(
                        kind=kind,
                        feature_index=feature,
                        lag=lag,
                        candidate_id=f"{kind}:{feature}:{lag}",
                    )
                )
        for window in (4, 8, 16, 32, 64):
            for kind in (
                "time_series_rolling_mean",
                "time_series_rolling_std",
                "time_series_rolling_sum",
            ):
                candidates.append(
                    TimeSeriesCandidate(
                        kind=kind,
                        feature_index=feature,
                        window=window,
                        candidate_id=f"{kind}:{feature}:{window}",
                    )
                )
    return candidates


def time_call(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return label, result, elapsed


def finite_scores(scores: dict[Any, dict[str, float]]) -> bool:
    for values in scores.values():
        for value in values.values():
            if not math.isfinite(float(value)):
                return False
    return True


def max_metric_error(
    left: dict[Any, dict[str, float]],
    right: dict[Any, dict[str, float]],
) -> float:
    error = 0.0
    for key, values in right.items():
        for metric, expected in values.items():
            error = max(error, abs(float(left[key][metric]) - float(expected)))
    return error


def score_summary(scores: dict[Any, dict[str, float]], metric: str, limit: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(
        scores.items(),
        key=lambda item: abs(float(item[1].get(metric, 0.0))),
        reverse=True,
    )[:limit]
    return [
        {
            "candidate": serializable_key(candidate),
            "metrics": {name: float(value) for name, value in values.items()},
        }
        for candidate, values in ranked
    ]


def serializable_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "candidate_id"):
        return {
            "candidate_id": getattr(value, "candidate_id", ""),
            "kind": getattr(value, "kind", type(value).__name__),
            "combo": list(getattr(value, "combo", ())),
            "params": value.params() if hasattr(value, "params") else {},
        }
    return repr(value)


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def run_command(command: Sequence[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - CI diagnostics
        return {"error_type": type(exc).__name__, "error": str(exc)}


def collect_runtime_specs() -> dict[str, Any]:
    specs: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "commands": {
            "uname": run_command(["uname", "-a"]),
            "sw_vers": run_command(["sw_vers"]),
            "xcodebuild": run_command(["xcodebuild", "-version"]),
            "metal_tool": run_command(["xcrun", "--find", "metal"]),
            "metallib_tool": run_command(["xcrun", "--find", "metallib"]),
            "sysctl_cpu": run_command(["sysctl", "machdep.cpu.brand_string", "hw.machine", "hw.model", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize"]),
        },
    }
    return specs


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import gafime
    from gafime import ComputeBudget, EngineConfig, GafimeEngine
    from gafime.backends import resolve_backend
    from gafime.backends.core_backend import CoreBackend
    from gafime.metrics import MetricSuite
    from gafime.time_series import score_time_series_candidates as score_time_series_candidates_cpu
    from gafime.utils.arrays import coerce_inputs

    rows, target, feature_names = make_dataset(args.n_samples, args.n_features, seed=args.seed)
    X, y, _ = coerce_inputs(rows, target, feature_names)
    backend, backend_warnings = resolve_backend(EngineConfig(backend="metal"), X, y)
    backend_info = backend.info()
    if backend_info.name != "metal-native" or not backend_info.is_gpu:
        raise RuntimeError(f"Expected metal-native GPU backend, got {backend_info!r}")

    metric_suite = MetricSuite(("pearson", "r2"), mi_bins=args.mi_bins)
    continuous_combos = build_continuous_combos(
        args.n_features,
        max_arity=args.max_arity,
        max_candidates=args.max_continuous_candidates,
    )
    time_series_candidates = build_time_series_candidates(args.n_features)

    _, continuous_scores, continuous_seconds = time_call(
        "continuous",
        lambda: backend.score_combos(X, y, continuous_combos, metric_suite),
    )
    if len(continuous_scores) != len(continuous_combos) or not finite_scores(continuous_scores):
        raise AssertionError("Metal continuous scores are incomplete or non-finite.")

    core = CoreBackend()
    check_combos = list(dict.fromkeys(
        continuous_combos[:40]
        + continuous_combos[len(continuous_combos) // 3 : len(continuous_combos) // 3 + 40]
        + continuous_combos[-40:]
        + [(0, 1), (3, 4, 5), (0, 1, 2, 3, 4)]
    ))
    core_continuous = core.score_combos(X, y, check_combos, metric_suite)
    continuous_error = max_metric_error(continuous_scores, core_continuous)
    if continuous_error > args.max_parity_error:
        raise AssertionError(f"Metal continuous/Core parity drift too high: {continuous_error}")

    _, ts_scores, ts_seconds = time_call(
        "time_series",
        lambda: backend.score_time_series_candidates(X, y, time_series_candidates, metric_suite),
    )
    if len(ts_scores) != len(time_series_candidates) or not finite_scores(ts_scores):
        raise AssertionError("Metal time-series scores are incomplete or non-finite.")
    cpu_ts = score_time_series_candidates_cpu(X, y, time_series_candidates[:96], metric_suite)
    ts_error = max_metric_error(ts_scores, cpu_ts)
    if ts_error > args.max_parity_error:
        raise AssertionError(f"Metal time-series/CPU parity drift too high: {ts_error}")

    engine_budget = ComputeBudget(
        max_comb_size=min(args.engine_max_arity, args.max_arity),
        max_combinations_per_k=args.engine_max_combinations_per_k,
        top_features_for_higher_k=min(args.n_features, 96),
        max_time_series_candidates=args.engine_max_time_series_candidates,
        top_k_features_for_time_series=min(args.n_features, 32),
        vram_budget_mb=32768,
    )
    engine_config = EngineConfig(
        backend="metal",
        metric_names=("pearson", "r2"),
        mi_bins=args.mi_bins,
        num_repeats=1,
        permutation_tests=0,
        random_seed=args.seed,
        enable_time_series_functions=True,
        time_series_lags=(1, 2, 4, 8, 16, 32),
        time_series_windows=(4, 8, 16, 32, 64),
        budget=engine_budget,
    )
    _, report, engine_seconds = time_call(
        "engine",
        lambda: GafimeEngine(engine_config).analyze(rows, target, feature_names=feature_names),
    )
    if report.backend is None or report.backend.name != "metal-native":
        raise AssertionError(f"Engine did not report metal-native backend: {report.backend!r}")

    families: dict[str, int] = {}
    for item in report.interactions:
        families[item.family] = families.get(item.family, 0) + 1
    if not {"interaction", "time_series_function"}.issubset(families):
        raise AssertionError(f"Engine did not keep all expected families: {families}")

    total_direct_candidate_rows = (
        len(continuous_combos) + len(time_series_candidates)
    ) * args.n_samples
    total_direct_seconds = continuous_seconds + ts_seconds

    return {
        "status": "ok",
        "gafime_version": getattr(gafime, "__version__", "unknown"),
        "runtime_specs": collect_runtime_specs(),
        "backend": to_jsonable(backend_info),
        "backend_warnings": backend_warnings,
        "dataset": {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "seed": args.seed,
            "signals": [
                "continuous pair f0*f1",
                "continuous high-order f0*f1*f3*f4*f5",
                "lag/rolling time-series f8/f9",
                "constant and low-cardinality finite edge columns",
            ],
        },
        "direct_backend": {
            "continuous": {
                "candidates": len(continuous_combos),
                "candidate_rows": len(continuous_combos) * args.n_samples,
                "seconds": continuous_seconds,
                "candidate_rows_per_second": (len(continuous_combos) * args.n_samples) / continuous_seconds,
                "max_core_error": continuous_error,
                "top": score_summary(continuous_scores, "pearson"),
            },
            "time_series": {
                "candidates": len(time_series_candidates),
                "candidate_rows": len(time_series_candidates) * args.n_samples,
                "seconds": ts_seconds,
                "candidate_rows_per_second": (len(time_series_candidates) * args.n_samples) / ts_seconds,
                "max_cpu_error": ts_error,
                "top": score_summary(ts_scores, "pearson"),
            },
            "total": {
                "candidate_rows": total_direct_candidate_rows,
                "seconds": total_direct_seconds,
                "candidate_rows_per_second": total_direct_candidate_rows / total_direct_seconds,
            },
        },
        "engine": {
            "seconds": engine_seconds,
            "backend": to_jsonable(report.backend),
            "families": families,
            "interactions": len(report.interactions),
            "warnings": report.warnings,
            "decision": to_jsonable(report.decision),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "beast", "absurd"), default="beast")
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=0)
    parser.add_argument("--max-arity", type=int, default=5)
    parser.add_argument("--max-continuous-candidates", type=int, default=0)
    parser.add_argument("--mi-bins", type=int, default=96)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-parity-error", type=float, default=0.01)
    parser.add_argument("--engine-max-arity", type=int, default=4)
    parser.add_argument("--engine-max-combinations-per-k", type=int, default=50000)
    parser.add_argument("--engine-max-time-series-candidates", type=int, default=2000)
    parser.add_argument("--output", default=os.environ.get("GAFIME_METAL_BENCHMARK_OUTPUT", "metal-hardcore-benchmark.json"))
    return parser


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.n_samples <= 0:
        args.n_samples = {"smoke": 4096, "beast": 32768, "absurd": 65536}[args.preset]
    if args.n_features <= 0:
        args.n_features = {"smoke": 16, "beast": 24, "absurd": 28}[args.preset]
    if args.preset == "smoke":
        args.engine_max_combinations_per_k = min(args.engine_max_combinations_per_k, 5000)
        args.engine_max_time_series_candidates = min(args.engine_max_time_series_candidates, 600)
    return args


def main() -> None:
    args = apply_preset(build_parser().parse_args())
    started = time.perf_counter()
    try:
        result = benchmark(args)
    except Exception as exc:  # pragma: no cover - CI diagnostics
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_specs": collect_runtime_specs(),
        }
        raise
    finally:
        elapsed = time.perf_counter() - started
        if "result" in locals():
            result["total_script_seconds"] = elapsed
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
