"""Release benchmark for the v0.4.5 native spine.

This script intentionally uses public GAFIME APIs so the same workload can be
run against older source tags such as v0.4.1 and against the current tree.
It does not publish artifacts and does not depend on a pip-published GAFIME
wheel.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Iterable


def make_data(n_samples: int, n_features: int) -> tuple[list[list[float]], list[float], list[str]]:
    if n_features < 5:
        raise ValueError("benchmark requires at least 5 features")
    X: list[list[float]] = []
    y: list[float] = []
    for i in range(n_samples):
        row = [
            math.sin((i + 1) * (j + 1) * 0.0031)
            + (((i * (j + 7)) % 53) - 26) / 31.0
            + 0.01 * math.cos((i + 3) * (j + 2) * 0.017)
            for j in range(n_features)
        ]
        signal = (
            0.45 * row[0] * row[1] * row[2] * row[3] * row[4]
            + 0.20 * row[min(5, n_features - 1)]
            - 0.10 * row[min(9, n_features - 1)]
        )
        y.append(float(signal))
        X.append([float(value) for value in row])
    return X, y, [f"f{i}" for i in range(n_features)]


def benchmark_engine(
    backend_name: str,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str],
    *,
    max_comb_size: int,
    max_combinations_per_k: int,
) -> dict[str, object]:
    from gafime import ComputeBudget, EngineConfig, GafimeEngine

    config = EngineConfig(
        backend=backend_name,
        metric_names=("pearson", "r2"),
        budget=ComputeBudget(
            max_comb_size=max_comb_size,
            max_combinations_per_k=max_combinations_per_k,
            top_features_for_higher_k=64,
        ),
        num_repeats=1,
        permutation_tests=0,
        random_seed=7,
    )
    start = time.perf_counter()
    report = GafimeEngine(config).analyze(X, y, feature_names=feature_names)
    elapsed = time.perf_counter() - start
    top = max(
        report.interactions,
        key=lambda result: abs(float(result.metrics.get("pearson", 0.0))),
    )
    return {
        "requested_backend": backend_name,
        "resolved_backend": report.backend.name,
        "elapsed_s": elapsed,
        "interactions": len(report.interactions),
        "max_arity": max(len(result.combo) for result in report.interactions),
        "families": sorted({getattr(result, "family", "interaction") for result in report.interactions}),
        "top_combo": list(top.combo),
        "top_family": getattr(top, "family", "interaction"),
        "top_pearson": float(top.metrics.get("pearson", 0.0)),
        "warnings": list(report.warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--max-comb-size", type=int, default=5)
    parser.add_argument("--max-combinations-per-k", type=int, default=100000)
    parser.add_argument("--backends", default="cpu,cuda")
    args = parser.parse_args()

    import gafime

    X, y, feature_names = make_data(args.n_samples, args.n_features)
    results: dict[str, object] = {
        "gafime_version": getattr(gafime, "__version__", "unknown"),
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "max_comb_size": args.max_comb_size,
        "max_combinations_per_k": args.max_combinations_per_k,
        "results": [],
    }

    for backend_name in [name.strip() for name in args.backends.split(",") if name.strip()]:
        try:
            result = benchmark_engine(
                backend_name,
                X,
                y,
                feature_names,
                max_comb_size=args.max_comb_size,
                max_combinations_per_k=args.max_combinations_per_k,
            )
            result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - release diagnostics
            result = {
                "requested_backend": backend_name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results["results"].append(result)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
