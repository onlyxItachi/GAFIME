#!/usr/bin/env python3
"""Measure safe-path interaction-diagnostic overhead through the public API.

Run the same command against a base install and the candidate install. This
harness reports timing distributions; it does not impose a universal
performance threshold because device load and toolchain state are external.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from statistics import median
from time import perf_counter_ns

from gafime import ComputeBudget, EngineConfig, GafimeEngine


def safe_dataset(rows: int, features: int):
    matrix = [
        [
            math.sin((row + 1) * (column + 1) * 0.001)
            + 0.25 * math.cos((row + 3) * (column + 2) * 0.0007)
            for column in range(features)
        ]
        for row in range(rows)
    ]
    target = [
        0.7 * matrix[row][0] - 0.4 * matrix[row][1] + 0.1 * matrix[row][2]
        for row in range(rows)
    ]
    names = [f"x{column}" for column in range(features)]
    return matrix, target, names


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


def summarize(samples: list[int]) -> dict[str, float | int]:
    return {
        "samples": len(samples),
        "median_ms": median(samples) / 1e6,
        "min_ms": min(samples) / 1e6,
        "max_ms": max(samples) / 1e6,
    }


def validate_report(report, expected: str) -> tuple[int, bool]:
    interactions = list(report.interactions)
    if not interactions:
        raise AssertionError("diagnostic overhead workload produced no interactions")
    available = bool(
        getattr(report.backend, "interaction_diagnostics_available", False)
    )
    if expected == "yes" and not available:
        raise AssertionError("candidate install did not expose interaction diagnostics")
    if expected == "no" and available:
        raise AssertionError(
            "base install unexpectedly exposed interaction diagnostics"
        )
    for interaction in interactions:
        if int(getattr(interaction, "interaction_overflow_rows", 0)) != 0:
            raise AssertionError("safe workload reported interaction overflow")
        if bool(getattr(interaction, "source_nonfinite", False)):
            raise AssertionError("safe workload reported a non-finite source")
    return len(interactions), available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="core", choices=("core", "cuda", "rocm"))
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument(
        "--expect-diagnostics",
        default="any",
        choices=("yes", "no", "any"),
    )
    args = parser.parse_args()
    if args.rows < 4 or args.features < 5:
        raise ValueError("rows must be >= 4 and features must be >= 5")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")

    matrix, target, names = safe_dataset(args.rows, args.features)
    budget = ComputeBudget(
        max_comb_size=5,
        max_combinations_per_k=4096,
        top_features_for_higher_k=args.features,
        keep_in_vram=False,
    )
    config = EngineConfig(
        backend=args.backend,
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=budget,
    )
    one_shot_engine = GafimeEngine(config)
    one_shot_samples = timed_samples(
        lambda: one_shot_engine.analyze(matrix, target, feature_names=names),
        args.warmups,
        args.repetitions,
    )
    one_shot_report = one_shot_engine.analyze(matrix, target, feature_names=names)

    artifact = GafimeEngine(config).compile(matrix, target, feature_names=names)
    try:
        resident_first_start = perf_counter_ns()
        resident_first_report = artifact.analyze()
        resident_first_ns = perf_counter_ns() - resident_first_start
        resident_samples = timed_samples(
            artifact.analyze,
            args.warmups,
            args.repetitions,
        )
        resident_report = artifact.analyze()
    finally:
        artifact.close()

    one_shot_count, one_shot_available = validate_report(
        one_shot_report,
        args.expect_diagnostics,
    )
    resident_count, resident_available = validate_report(
        resident_report,
        args.expect_diagnostics,
    )
    resident_first_count, resident_first_available = validate_report(
        resident_first_report,
        args.expect_diagnostics,
    )
    if one_shot_count != resident_count or one_shot_count != resident_first_count:
        raise AssertionError("one-shot and resident candidate counts differ")
    if (
        one_shot_available != resident_available
        or one_shot_available != resident_first_available
    ):
        raise AssertionError("one-shot and resident diagnostic availability differs")

    print(
        json.dumps(
            {
                "backend": args.backend,
                "rows": args.rows,
                "features": args.features,
                "candidate_count": one_shot_count,
                "diagnostics_available": one_shot_available,
                "one_shot": summarize(one_shot_samples),
                "resident_first_ms": resident_first_ns / 1e6,
                "resident": summarize(resident_samples),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
