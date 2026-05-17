"""
Nsight Compute target for the CUDA discrete selection-score kernel.

This script intentionally calls NativeCudaBackend directly so the profiled path
is the GPU selection kernel, not sklearn model fitting or CPU fallback scoring.
"""

from __future__ import annotations

import argparse

import numpy as np

from gafime.backends.native_cuda_backend import NativeCudaBackend
from gafime.discrete import DiscreteFunctionCandidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.samples, args.features)).astype(np.float32)
    y = (
        0.8 * (X[:, 0] > 0.25).astype(np.float32)
        + 0.5 * ((X[:, 1] > -0.4) & (X[:, 2] < 0.7)).astype(np.float32)
        + 0.1 * rng.normal(size=args.samples).astype(np.float32)
    )
    baseline_pred = np.full_like(y, float(np.mean(y)))
    candidates = make_candidates(X, args.candidates)

    backend = NativeCudaBackend()
    if not getattr(backend, "_has_discrete_selection_api", False):
        raise RuntimeError("CUDA discrete selection native API is not available.")

    scores = backend.score_discrete_selection_candidates(
        X,
        y,
        candidates,
        baseline_pred=baseline_pred,
        mi_bins=16,
    )
    best = max(item["variance_reduction"] for item in scores.values())
    print(
        f"selection candidates={len(candidates)} samples={args.samples} "
        f"backend={backend.info().name} best_variance_reduction={best:.6f}"
    )


def make_candidates(X: np.ndarray, count: int) -> list[DiscreteFunctionCandidate]:
    n_features = X.shape[1]
    quantiles = np.quantile(X, [0.05, 0.10, 0.25, 0.40, 0.50, 0.65, 0.75, 0.90, 0.95], axis=0)
    scales = np.std(X, axis=0)
    candidates: list[DiscreteFunctionCandidate] = []
    i = 0
    while len(candidates) < count:
        feature_a = i % n_features
        feature_b = (i * 7 + 3) % n_features
        value_feature = (i * 5 + 1) % n_features
        q0 = i % quantiles.shape[0]
        q1 = min(q0 + 2, quantiles.shape[0] - 1)
        threshold = float(quantiles[q0, feature_a])
        low_a = float(quantiles[q0, feature_a])
        high_a = float(quantiles[q1, feature_a])
        low_b = float(quantiles[q0, feature_b])
        high_b = float(quantiles[q1, feature_b])
        scale_a = float(max(scales[feature_a], 1e-6))
        scale_b = float(max(scales[feature_b], 1e-6))

        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_threshold",
                feature_indices=(feature_a,),
                thresholds=(threshold,),
                direction="ge" if i % 2 == 0 else "le",
                scales=(scale_a,),
            )
        )
        if len(candidates) >= count:
            break
        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_interval",
                feature_indices=(feature_a,),
                intervals=((low_a, high_a),),
                scales=(scale_a,),
            )
        )
        if len(candidates) >= count:
            break
        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_value_gated_threshold",
                feature_indices=(feature_a,),
                thresholds=(threshold,),
                direction="ge",
                value_feature=value_feature,
                scales=(scale_a,),
            )
        )
        if len(candidates) >= count:
            break
        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_rectangle",
                feature_indices=(feature_a, feature_b),
                intervals=((low_a, high_a), (low_b, high_b)),
                scales=(scale_a, scale_b),
            )
        )
        if len(candidates) >= count:
            break
        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_value_in_soft_rectangle",
                feature_indices=(feature_a, feature_b),
                intervals=((low_a, high_a), (low_b, high_b)),
                value_feature=value_feature,
                scales=(scale_a, scale_b),
            )
        )
        i += 1
    return candidates


if __name__ == "__main__":
    main()

