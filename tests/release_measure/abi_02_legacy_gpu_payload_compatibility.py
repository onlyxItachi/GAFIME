#!/usr/bin/env python3
"""Execute the current host boundary against an older same-ABI GPU payload.

This is a correctness/ABI gate, not a performance measure. It proves that an
ABI-1.0 payload which predates newer optional launch hints remains executable.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


PAYLOAD_ENV = {
    "cuda": "GAFIME_CUDA_V1_LIB",
    "rocm": "GAFIME_ROCM_V1_LIB",
    "metal": "GAFIME_METAL_V1_LIB",
}


def snapshot(report: object) -> dict[tuple[int, ...], dict[str, float]]:
    rows: dict[tuple[int, ...], dict[str, float]] = {}
    for item in report.interactions:
        combo = tuple(int(value) for value in item.combo)
        metrics = {str(name): float(value) for name, value in item.metrics.items()}
        if combo in rows:
            raise AssertionError(f"duplicate candidate combo {combo}")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise AssertionError(f"nonfinite metrics for {combo}: {metrics}")
        rows[combo] = metrics
    if not rows:
        raise AssertionError("legacy payload returned no candidates")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(PAYLOAD_ENV), required=True)
    parser.add_argument("--legacy-payload", type=Path, required=True)
    parser.add_argument("--legacy-metallib", type=Path)
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    payload = args.legacy_payload.resolve(strict=True)
    os.environ[PAYLOAD_ENV[args.backend]] = str(payload)
    if args.legacy_metallib is not None:
        if args.backend != "metal":
            raise ValueError("--legacy-metallib is valid only for the Metal backend")
        os.environ["GAFIME_METAL_V1_METALLIB"] = str(
            args.legacy_metallib.resolve(strict=True)
        )
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = "0"

    import gafime

    matrix = [
        [1.0, 0.0, 4.0, -2.0],
        [2.0, 1.0, 3.0, -1.0],
        [3.0, 1.0, 2.0, 0.0],
        [4.0, 2.0, 1.0, 1.0],
        [5.0, 3.0, 0.0, 2.0],
        [6.0, 5.0, -1.0, 3.0],
        [7.0, 8.0, -2.0, 4.0],
        [8.0, 13.0, -3.0, 5.0],
    ]
    target = [1.0, 1.5, 2.5, 4.0, 6.5, 10.5, 17.0, 27.5]
    names = ["a", "b", "c", "d"]
    budget = gafime.ComputeBudget(
        max_comb_size=2,
        max_combinations_per_k=64,
        top_features_for_higher_k=4,
        keep_in_vram=False,
    )
    common = dict(
        metric_names=("pearson", "r2"),
        budget=budget,
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
    )
    reference = gafime.GafimeEngine(gafime.EngineConfig(backend="core", **common)).analyze(
        matrix, target, names
    )
    actual = gafime.GafimeEngine(
        gafime.EngineConfig(backend=args.backend, **common)
    ).analyze(matrix, target, names)

    selected = getattr(actual.backend, "selected_backend", None)
    if selected != args.backend:
        raise AssertionError(
            f"requested legacy {args.backend} payload resolved as {selected!r}"
        )
    expected_rows = snapshot(reference)
    actual_rows = snapshot(actual)
    if actual_rows.keys() != expected_rows.keys():
        raise AssertionError(
            f"candidate identity mismatch: expected={sorted(expected_rows)} "
            f"actual={sorted(actual_rows)}"
        )

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 0.002 if args.backend == "metal" else 1.0e-5
    max_delta = 0.0
    for combo in expected_rows:
        for metric, expected in expected_rows[combo].items():
            actual_value = actual_rows[combo][metric]
            delta = abs(actual_value - expected)
            max_delta = max(max_delta, delta)
            if delta > tolerance:
                raise AssertionError(
                    f"{combo} {metric} delta {delta:.6g} exceeds {tolerance:.6g}"
                )

    print(
        f"LEGACY-PAYLOAD-ABI: PASS backend={args.backend} payload={payload} "
        f"candidates={len(actual_rows)} max|delta|={max_delta:.3g}"
    )


if __name__ == "__main__":
    main()
