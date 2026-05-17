from __future__ import annotations

import argparse
import time

import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.backends.native_cuda_backend import NativeCudaBackend
from gafime.discrete import DiscreteFunctionCandidate
from gafime.metrics import MetricSuite


def make_data(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    soft0 = 1.0 / (1.0 + np.exp(-12.0 * (X[:, 0] - 0.15)))
    soft1 = 1.0 / (1.0 + np.exp(-12.0 * (X[:, 1] + 0.30)))
    rect = soft0 * (1.0 / (1.0 + np.exp(-12.0 * (0.75 - X[:, 2]))))
    y = 0.9 * soft0 + 0.4 * X[:, 3] * soft1 + 0.6 * rect
    y += 0.05 * rng.normal(size=n_samples)
    return X, y.astype(np.float32)


def make_candidates(n_features: int, n_candidates: int):
    candidates = []
    quantiles = (-1.25, -0.75, -0.25, 0.0, 0.25, 0.65, 1.25)
    intervals = ((-1.25, -0.25), (-0.75, 0.25), (-0.25, 0.65), (0.0, 1.25))
    i = 0
    while len(candidates) < n_candidates:
        f0 = i % n_features
        f1 = (i * 7 + 1) % n_features
        fv = (i * 11 + 3) % n_features
        threshold = quantiles[i % len(quantiles)]
        interval0 = intervals[i % len(intervals)]
        interval1 = intervals[(i // len(intervals)) % len(intervals)]
        direction = "le" if i & 1 else "ge"
        sharpness = 12.0 + float(i % 3)
        scale0 = 0.7 + 0.1 * float((i + 1) % 5)
        scale1 = 0.8 + 0.1 * float((i + 3) % 5)
        kind_slot = i % 5
        if kind_slot == 0:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_soft_threshold",
                    feature_indices=(f0,),
                    thresholds=(threshold,),
                    direction=direction,
                    scales=(scale0,),
                    sharpness=sharpness,
                    candidate_id=f"threshold-{i}",
                )
            )
        elif kind_slot == 1:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_soft_interval",
                    feature_indices=(f0,),
                    intervals=(interval0,),
                    scales=(scale0,),
                    sharpness=sharpness,
                    candidate_id=f"interval-{i}",
                )
            )
        elif kind_slot == 2:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_value_gated_threshold",
                    feature_indices=(f0,),
                    thresholds=(threshold,),
                    direction=direction,
                    value_feature=fv,
                    scales=(scale0,),
                    sharpness=sharpness,
                    candidate_id=f"value-threshold-{i}",
                )
            )
        elif kind_slot == 3:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_soft_rectangle",
                    feature_indices=(f0, f1),
                    intervals=(interval0, interval1),
                    scales=(scale0, scale1),
                    sharpness=sharpness,
                    candidate_id=f"rectangle-{i}",
                )
            )
        else:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_value_in_soft_rectangle",
                    feature_indices=(f0, f1),
                    intervals=(interval0, interval1),
                    value_feature=fv,
                    scales=(scale0, scale1),
                    sharpness=sharpness,
                    candidate_id=f"value-rectangle-{i}",
                )
            )
        i += 1
    return candidates


def run_engine_warmup(X, y, n_candidates: int):
    budget = ComputeBudget(
        max_comb_size=2,
        max_combinations_per_k=128,
        top_features_for_higher_k=16,
        max_discrete_candidates=n_candidates,
        max_thresholds_per_feature=9,
        max_intervals_per_feature=12,
        max_feature_pairs_for_rectangles=256,
        top_k_features_for_discrete=32,
    )
    config = EngineConfig(
        backend="cuda",
        metric_names=("pearson",),
        enable_discrete_functions=True,
        num_repeats=1,
        permutation_tests=0,
        budget=budget,
    )
    t0 = time.perf_counter()
    report = GafimeEngine(config).analyze(
        X,
        y,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
    )
    elapsed = time.perf_counter() - t0
    discrete_count = sum(1 for item in report.interactions if item.family == "discrete_function")
    print(f"engine_elapsed_s={elapsed:.6f}")
    print(f"engine_backend={report.backend.name}")
    print(f"engine_discrete_count={discrete_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=180_000)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine-warmup", action="store_true")
    args = parser.parse_args()

    X, y = make_data(args.samples, args.features, args.seed)
    if args.engine_warmup:
        run_engine_warmup(X, y, min(args.candidates, 4096))

    backend = NativeCudaBackend()
    candidates = make_candidates(args.features, args.candidates)
    suite = MetricSuite(("pearson",))

    # Prime CUDA context and the native symbol before the measured loop.
    backend.score_discrete_candidates(X[:4096], y[:4096], candidates[:1024], suite)

    t0 = time.perf_counter()
    checksum = 0.0
    for _ in range(args.repeats):
        scores = backend.score_discrete_candidates(X, y, candidates, suite)
        checksum += sum(v["pearson"] for v in scores.values())
    elapsed = time.perf_counter() - t0

    print(f"backend_elapsed_s={elapsed:.6f}")
    print(f"samples={args.samples}")
    print(f"features={args.features}")
    print(f"candidates={args.candidates}")
    print(f"repeats={args.repeats}")
    print(f"checksum={checksum:.12f}")


if __name__ == "__main__":
    main()
