#!/usr/bin/env python3
"""Measure resident GPU MI specialization throughput by candidate scale.

This benchmark times the public compile artifact after upload/planning. Work is
reported as candidates and candidate-sample pairs; it is not a matrix benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig


SUPPORTED_BINS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96)
SAMPLES_PER_JOINT_BIN = 8


def parse_bins(value: str) -> tuple[int, ...]:
    bins = tuple(int(item) for item in value.split(",") if item)
    if not bins or any(item not in SUPPORTED_BINS for item in bins):
        raise argparse.ArgumentTypeError(
            f"bins must be a comma-separated subset of {SUPPORTED_BINS}"
        )
    return bins


def candidate_count(feature_count: int, max_arity: int) -> int:
    return sum(math.comb(feature_count, arity) for arity in range(1, max_arity + 1))


def adaptive_template_bins(rows: int, maximum: int) -> int:
    selected = 2
    for bins in SUPPORTED_BINS:
        if bins > maximum or rows < SAMPLES_PER_JOINT_BIN * bins * bins:
            break
        selected = bins
    return selected


def make_data(
    rows: int, features: int, seed: int
) -> tuple[list[list[float]], list[float], list[str]]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((rows, features), dtype=np.float32)
    noise = rng.standard_normal(rows, dtype=np.float32) * np.float32(0.1)
    target = matrix[:, 0] * np.float32(0.7) - matrix[:, 1] * np.float32(0.4) + noise
    names = [f"x{index}" for index in range(features)]
    return matrix.tolist(), target.tolist(), names


def measure(
    backend: str,
    bins: int,
    matrix: list[list[float]],
    target: list[float],
    names: list[str],
    max_arity: int,
    warmups: int,
    repeats: int,
) -> dict[str, int | float | str]:
    features = len(names)
    rows = len(target)
    expected_candidates = candidate_count(features, max_arity)
    per_arity_limit = max(
        math.comb(features, arity) for arity in range(1, max_arity + 1)
    )
    config = EngineConfig(
        backend=backend,
        metric_names=("mutual_info",),
        num_repeats=1,
        permutation_tests=0,
        mi_bins=bins,
        mi_approximate=True,
        budget=ComputeBudget(
            max_comb_size=max_arity,
            max_combinations_per_k=per_arity_limit,
            vram_budget_mb=6144,
        ),
    )
    artifact = gafime.compile(
        matrix,
        target,
        names,
        config=config,
        flags=CompileFlags(plan=True, graph=False),
    )
    try:
        for _ in range(warmups):
            report = artifact.analyze()
            if len(report.interactions) != expected_candidates:
                raise AssertionError(
                    f"{backend} bins={bins}: expected {expected_candidates} candidates, "
                    f"received {len(report.interactions)}"
                )

        samples_ns: list[int] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            report = artifact.analyze()
            elapsed = time.perf_counter_ns() - start
            if len(report.interactions) != expected_candidates:
                raise AssertionError("candidate count changed during timed execution")
            samples_ns.append(elapsed)
    finally:
        artifact.close()

    median_ns = int(statistics.median(samples_ns))
    candidate_pairs = expected_candidates * rows
    return {
        "backend": backend,
        "bins": bins,
        "rows": rows,
        "features": features,
        "max_arity": max_arity,
        "candidates": expected_candidates,
        "median_ns": median_ns,
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "candidates_per_second": expected_candidates * 1e9 / median_ns,
        "candidate_sample_gevals_per_second": candidate_pairs / median_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", choices=("cuda", "rocm"))
    parser.add_argument("--bins", type=parse_bins, default=parse_bins("32,64,96"))
    parser.add_argument("--rows", type=int, default=73728)
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--max-arity", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--json-out", type=Path)
    arguments = parser.parse_args()

    backends = arguments.backend or ["cuda", "rocm"]
    if arguments.rows < 2 or arguments.features < 2:
        parser.error("--rows and --features must both be at least 2")
    if arguments.max_arity < 1 or arguments.max_arity > arguments.features:
        parser.error("--max-arity must be between 1 and --features")
    if arguments.warmups < 0 or arguments.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    downshifts = [
        (maximum, adaptive_template_bins(arguments.rows, maximum))
        for maximum in arguments.bins
        if adaptive_template_bins(arguments.rows, maximum) != maximum
    ]
    if downshifts:
        parser.error(
            "requested MI maxima do not reach their exact adaptive templates at "
            f"--rows={arguments.rows}: {downshifts}"
        )

    matrix, target, names = make_data(
        arguments.rows, arguments.features, arguments.seed
    )
    results = []
    for backend in backends:
        for bins in arguments.bins:
            result = measure(
                backend,
                bins,
                matrix,
                target,
                names,
                arguments.max_arity,
                arguments.warmups,
                arguments.repeats,
            )
            results.append(result)
            print(
                f"[{backend}] bins={bins:>2} candidates={result['candidates']:,} "
                f"median={result['median_ns'] / 1e6:.3f}ms "
                f"candidate_rate={result['candidates_per_second'] / 1e6:.3f}M/s "
                f"candidate_sample_rate={result['candidate_sample_gevals_per_second']:.3f}GEval/s"
            )

    payload = {
        "schema": "gafime.gpu-mi-specialization-perf.v1",
        "results": results,
    }
    if arguments.json_out is not None:
        arguments.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
