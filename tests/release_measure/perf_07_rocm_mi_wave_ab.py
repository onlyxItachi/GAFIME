#!/usr/bin/env python3
"""Interleaved same-process A/B for the ROCm high-bin MI wave path."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig

from perf_06_gpu_mi_specializations import (
    SUPPORTED_BINS,
    adaptive_template_bins,
    candidate_count,
    make_data,
    parse_bins,
)


@contextmanager
def rocm_payload(path: Path):
    key = "GAFIME_ROCM_V1_LIB"
    previous = os.environ.get(key)
    os.environ[key] = str(path.resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def compile_artifact(
    library: Path,
    bins: int,
    matrix: list[list[float]],
    target: list[float],
    names: list[str],
    max_arity: int,
):
    features = len(names)
    per_arity_limit = max(
        math.comb(features, arity) for arity in range(1, max_arity + 1)
    )
    config = EngineConfig(
        backend="rocm",
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
    with rocm_payload(library):
        return gafime.compile(
            matrix,
            target,
            names,
            config=config,
            flags=CompileFlags(plan=True, graph=False),
        )


def timed_analyze(artifact, expected_candidates: int) -> int:
    start = time.perf_counter_ns()
    report = artifact.analyze()
    elapsed = time.perf_counter_ns() - start
    if len(report.interactions) != expected_candidates:
        raise AssertionError(
            f"expected {expected_candidates} candidates, received {len(report.interactions)}"
        )
    return elapsed


def report_rows(report) -> dict[tuple[str, tuple[str, ...]], float]:
    rows = {}
    for interaction in report.interactions:
        key = (str(interaction.candidate_id), tuple(interaction.combo))
        if key in rows:
            raise AssertionError(f"duplicate candidate row: {key}")
        value = float(interaction.metrics["mutual_info"])
        if not math.isfinite(value):
            raise AssertionError(
                f"non-finite MI value for candidate row {key}: {value}"
            )
        rows[key] = value
    return rows


def assert_report_parity(baseline, optimized, tolerance: float) -> float:
    baseline_rows = report_rows(baseline.analyze())
    optimized_rows = report_rows(optimized.analyze())
    if baseline_rows.keys() != optimized_rows.keys():
        missing = baseline_rows.keys() - optimized_rows.keys()
        extra = optimized_rows.keys() - baseline_rows.keys()
        raise AssertionError(
            f"A/B candidate identity mismatch: missing={len(missing)} extra={len(extra)}"
        )
    max_delta = max(
        (abs(baseline_rows[key] - optimized_rows[key]) for key in baseline_rows),
        default=0.0,
    )
    if max_delta > tolerance:
        raise AssertionError(
            f"A/B MI parity delta {max_delta:.9g} exceeds tolerance {tolerance:.9g}"
        )
    return max_delta


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rocm_build_metadata(path: Path) -> dict[str, str | int]:
    match = re.search(
        rb"GAFIME_ROCM_BUILD_INFO:arch=([^;\x00]+);wave_mi_mask=([0-3])"
        rb"(?:;mi_accumulation_fp64=([01]))?",
        path.read_bytes(),
    )
    if match is None:
        raise ValueError(f"ROCm build provenance is missing from {path}")
    wave_mi_mask = int(match.group(2))
    mi_accumulation_fp64 = int(match.group(3) or 0)
    return {
        "target_arch": match.group(1).decode("ascii"),
        "wave_mi_mask": wave_mi_mask,
        "wave_mi_mode": {0: "off", 1: "64", 2: "96", 3: "64-96"}[wave_mi_mask],
        "mi_accumulation_mode": "fp64" if mi_accumulation_fp64 else "fast",
    }


def measure_pair(
    baseline_library: Path,
    optimized_library: Path,
    bins: int,
    matrix: list[list[float]],
    target: list[float],
    names: list[str],
    max_arity: int,
    warmups: int,
    repeats: int,
    parity_tolerance: float,
) -> tuple[float, float, float]:
    expected_candidates = candidate_count(len(names), max_arity)
    baseline = compile_artifact(
        baseline_library, bins, matrix, target, names, max_arity
    )
    optimized = compile_artifact(
        optimized_library, bins, matrix, target, names, max_arity
    )
    try:
        max_parity_delta = assert_report_parity(baseline, optimized, parity_tolerance)
        for iteration in range(warmups):
            order = (
                (baseline, optimized) if iteration % 2 == 0 else (optimized, baseline)
            )
            for artifact in order:
                timed_analyze(artifact, expected_candidates)

        baseline_samples: list[int] = []
        optimized_samples: list[int] = []
        for iteration in range(repeats):
            order = (
                ((baseline, baseline_samples), (optimized, optimized_samples))
                if iteration % 2 == 0
                else ((optimized, optimized_samples), (baseline, baseline_samples))
            )
            for artifact, samples in order:
                samples.append(timed_analyze(artifact, expected_candidates))
    finally:
        optimized.close()
        baseline.close()

    return (
        statistics.median(baseline_samples),
        statistics.median(optimized_samples),
        max_parity_delta,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-lib", type=Path, required=True)
    parser.add_argument("--optimized-lib", type=Path, required=True)
    parser.add_argument("--bins", type=parse_bins, default=parse_bins("32,64,96"))
    parser.add_argument("--control-bins", type=int, choices=SUPPORTED_BINS, default=32)
    parser.add_argument("--rows", type=int, default=73728)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--max-arity", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--json-out", type=Path)
    arguments = parser.parse_args()

    for option, path in (
        ("--baseline-lib", arguments.baseline_lib),
        ("--optimized-lib", arguments.optimized_lib),
    ):
        if not path.is_file():
            parser.error(f"{option} does not name a file: {path}")
    if arguments.rows < 2 or arguments.features < 2:
        parser.error("--rows and --features must both be at least 2")
    if arguments.max_arity < 1 or arguments.max_arity > arguments.features:
        parser.error("--max-arity must be between 1 and --features")
    if arguments.warmups < 0 or arguments.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    if (
        not math.isfinite(arguments.parity_tolerance)
        or arguments.parity_tolerance < 0.0
    ):
        parser.error("--parity-tolerance must be finite and non-negative")
    if arguments.control_bins not in arguments.bins:
        parser.error("--control-bins must also be present in --bins")
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

    baseline_path = arguments.baseline_lib.resolve()
    optimized_path = arguments.optimized_lib.resolve()
    if baseline_path == optimized_path or baseline_path.samefile(optimized_path):
        parser.error("baseline and optimized libraries resolve to the same file")
    baseline_sha256 = sha256_file(baseline_path)
    optimized_sha256 = sha256_file(optimized_path)
    if baseline_sha256 == optimized_sha256:
        parser.error("baseline and optimized libraries have identical SHA-256 digests")
    try:
        baseline_build = rocm_build_metadata(baseline_path)
        optimized_build = rocm_build_metadata(optimized_path)
    except ValueError as error:
        parser.error(str(error))
    if baseline_build["target_arch"] != optimized_build["target_arch"]:
        parser.error(
            f"A/B target architectures differ: {baseline_build} vs {optimized_build}"
        )
    if baseline_build["wave_mi_mask"] != 0:
        parser.error(f"baseline must disable wave MI: {baseline_build}")
    if optimized_build["wave_mi_mask"] not in (1, 2, 3):
        parser.error(f"optimized wave MI mode must not be off: {optimized_build}")

    matrix, target, names = make_data(
        arguments.rows, arguments.features, arguments.seed
    )
    candidates = candidate_count(arguments.features, arguments.max_arity)
    pairs = candidates * arguments.rows
    results: list[dict[str, int | float]] = []
    for bins in arguments.bins:
        baseline_ns, optimized_ns, max_parity_delta = measure_pair(
            arguments.baseline_lib,
            arguments.optimized_lib,
            bins,
            matrix,
            target,
            names,
            arguments.max_arity,
            arguments.warmups,
            arguments.repeats,
            arguments.parity_tolerance,
        )
        if baseline_ns <= 0 or optimized_ns <= 0:
            raise AssertionError(
                f"invalid timing sample for bins={bins}: "
                f"baseline={baseline_ns} optimized={optimized_ns}"
            )
        speedup = baseline_ns / optimized_ns
        results.append(
            {
                "bins": bins,
                "baseline_median_ns": int(baseline_ns),
                "optimized_median_ns": int(optimized_ns),
                "speedup": speedup,
                "max_mi_parity_delta": max_parity_delta,
                "baseline_candidate_sample_gevals_per_second": pairs / baseline_ns,
                "optimized_candidate_sample_gevals_per_second": pairs / optimized_ns,
            }
        )

    control_speedup = next(
        float(result["speedup"])
        for result in results
        if result["bins"] == arguments.control_bins
    )
    for result in results:
        normalized = float(result["speedup"]) / control_speedup
        result["control_normalized_speedup"] = normalized
        print(
            f"bins={int(result['bins']):>2} "
            f"baseline={float(result['baseline_median_ns']) / 1e6:.3f}ms "
            f"optimized={float(result['optimized_median_ns']) / 1e6:.3f}ms "
            f"speedup={float(result['speedup']):.4f}x "
            f"control_normalized={normalized:.4f}x "
            f"parity_delta={float(result['max_mi_parity_delta']):.3g} "
            f"baseline_rate={float(result['baseline_candidate_sample_gevals_per_second']):.3f}GEval/s "
            f"optimized_rate={float(result['optimized_candidate_sample_gevals_per_second']):.3f}GEval/s"
        )

    payload = {
        "schema": "gafime.rocm-mi-wave-ab.v2",
        "baseline_library": str(baseline_path),
        "baseline_library_bytes": arguments.baseline_lib.stat().st_size,
        "baseline_library_sha256": baseline_sha256,
        "baseline_build": baseline_build,
        "optimized_library": str(optimized_path),
        "optimized_library_bytes": arguments.optimized_lib.stat().st_size,
        "optimized_library_sha256": optimized_sha256,
        "optimized_build": optimized_build,
        "rows": arguments.rows,
        "features": arguments.features,
        "max_arity": arguments.max_arity,
        "candidates": candidates,
        "warmups": arguments.warmups,
        "repeats": arguments.repeats,
        "control_bins": arguments.control_bins,
        "parity_tolerance": arguments.parity_tolerance,
        "results": results,
    }
    if arguments.json_out is not None:
        arguments.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
