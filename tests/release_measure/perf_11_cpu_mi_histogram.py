#!/usr/bin/env python3
"""Measure resident Core fixed-bin MI throughput through the public API."""

from __future__ import annotations

import argparse
import gc
import json
import math
from statistics import median
from time import perf_counter_ns

import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine


def dataset(rows: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(rows, dtype=np.float32)
    feature = (
        np.sin(index * np.float32(0.0017))
        + np.float32(0.31) * np.cos(index * np.float32(0.00031))
    ).astype(np.float32)
    target = (
        np.float32(0.61) * feature
        + np.float32(0.23) * np.sin(index * np.float32(0.00073))
    ).astype(np.float32)
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
    parser.add_argument("--bins", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=51)
    args = parser.parse_args()
    if args.rows < 2:
        raise ValueError("rows must be >= 2")
    if args.bins not in {2, 4, 8, 12, 16, 24, 32, 48, 64, 96}:
        raise ValueError("bins must be in the compiled adaptive MI ladder")
    if args.rows < 8 * args.bins * args.bins:
        raise ValueError(
            "rows must satisfy the 8*bins^2 density guard so the requested "
            "specialization remains effective"
        )
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")

    matrix, target = dataset(args.rows)
    config = EngineConfig(
        backend="core",
        metric_names=("mutual_info",),
        mi_approximate=True,
        mi_bins=args.bins,
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
        raise AssertionError("MI benchmark expected one unary candidate")
    mutual_info = report.interactions[0].metrics["mutual_info"]
    if not math.isfinite(mutual_info):
        raise AssertionError("MI benchmark produced a nonfinite metric")

    print(
        json.dumps(
            {
                "rows": args.rows,
                "bins": args.bins,
                "mutual_info": mutual_info,
                "timing": summarize(samples, args.rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
