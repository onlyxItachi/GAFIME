#!/usr/bin/env python3
"""Measure resident Core Pearson/R2 throughput through the public API."""

from __future__ import annotations

import argparse
import gc
import json
import math
from statistics import median
from time import perf_counter_ns

import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine


def dataset(rows: int, nonfinite_position: str) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(rows, dtype=np.float32)
    feature = np.sin(index * np.float32(0.0017)).astype(np.float32)
    target = (
        np.float32(0.73) * feature
        + np.float32(0.19) * np.cos(index * np.float32(0.0009))
    ).astype(np.float32)
    if nonfinite_position != "none":
        position = {
            "first": 0,
            "middle": rows // 2,
            "last": rows - 1,
        }[nonfinite_position]
        feature[position] = np.float32(np.nan)
    return feature.reshape(rows, 1), target


def timed_samples(function, warmups: int, repetitions: int) -> list[int]:
    for _ in range(warmups):
        function()
    gc.collect()
    samples = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        function()
        samples.append(perf_counter_ns() - start)
    return samples


def summarize(samples: list[int], rows: int) -> dict[str, float | int]:
    median_ns = median(samples)
    return {
        "samples": len(samples),
        "median_ms": median_ns / 1e6,
        "min_ms": min(samples) / 1e6,
        "max_ms": max(samples) / 1e6,
        "median_melements_per_second": rows / median_ns * 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--metrics", default="pearson")
    parser.add_argument(
        "--nonfinite-position",
        default="none",
        choices=("none", "first", "middle", "last"),
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=51)
    args = parser.parse_args()
    if args.rows < 2:
        raise ValueError("rows must be >= 2")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")
    metrics = tuple(name.strip() for name in args.metrics.split(",") if name.strip())
    if not metrics or any(name not in {"pearson", "r2"} for name in metrics):
        raise ValueError("metrics must be a comma-separated subset of pearson,r2")

    matrix, target = dataset(args.rows, args.nonfinite_position)
    config = EngineConfig(
        backend="core",
        metric_names=metrics,
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=1,
            top_features_for_higher_k=1,
            keep_in_vram=True,
        ),
    )
    artifact = GafimeEngine(config).compile(
        matrix,
        target,
        feature_names=["x"],
    )
    try:
        samples = timed_samples(artifact.analyze, args.warmups, args.repetitions)
        report = artifact.analyze()
    finally:
        artifact.close()

    if len(report.interactions) != 1:
        raise AssertionError("covariance benchmark expected one unary candidate")
    result = report.interactions[0]
    for name in metrics:
        if not math.isfinite(result.metrics[name]):
            raise AssertionError(f"{name} result is not finite")
    if result.source_nonfinite != (args.nonfinite_position != "none"):
        raise AssertionError("source-nonfinite diagnostic does not match the workload")
    if "pearson" in metrics and "r2" in metrics:
        expected_r2 = result.metrics["pearson"] ** 2
        if not math.isclose(result.metrics["r2"], expected_r2, abs_tol=1.0e-6):
            raise AssertionError("R2 does not equal the squared Pearson result")

    print(
        json.dumps(
            {
                "rows": args.rows,
                "metrics": metrics,
                "nonfinite_position": args.nonfinite_position,
                "pearson": result.metrics.get("pearson"),
                "r2": result.metrics.get("r2"),
                "timing": summarize(samples, args.rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
