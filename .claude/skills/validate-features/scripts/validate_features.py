#!/usr/bin/env python3
"""
GAFIME Feature Validation Script

Applies a bounded descriptive holdout heuristic to supplied feature interactions
with bootstrap intervals and an explicitly non-conclusive status.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def pearson_r(x: np.ndarray, y: np.ndarray, precision: str = "mixed") -> float:
    """Compute paired-finite Pearson correlation in the profile reduction dtype."""

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < 3:
        return 0.0
    reduction_dtype = np.float32 if precision == "fp32" else np.float64
    x = x.astype(reduction_dtype, copy=False)
    y = y.astype(reduction_dtype, copy=False)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx**2).sum() * (dy**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((dx * dy).sum() / denom)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    precision: str = "mixed",
) -> tuple:
    """Compute bootstrap confidence interval for Pearson r."""
    rng = np.random.default_rng(42)
    n = len(x)
    rs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rs[i] = pearson_r(x[idx], y[idx], precision=precision)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(rs, alpha * 100))
    hi = float(np.percentile(rs, (1 - alpha) * 100))
    return lo, hi


def interaction_vector(
    matrix: np.ndarray, feature_i: int, feature_j: int, operator: str, precision: str = "mixed"
) -> np.ndarray:
    left = matrix[:, feature_i]
    right = matrix[:, feature_j]
    if operator == "multiply":
        return left * right
    if operator == "add":
        return left + right
    if operator == "subtract":
        return left - right
    if operator == "divide":
        pointwise_dtype = np.float64 if precision == "fp64" else np.float32
        epsilon = np.asarray(1e-8, dtype=pointwise_dtype).item()
        denominator = np.where(np.abs(right) > epsilon, right, epsilon)
        return left / denominator
    raise ValueError(f"unsupported operator: {operator}")


def validate_interactions(
    X: np.ndarray,
    y: np.ndarray,
    interactions: list,
    operator: str = "multiply",
    test_size: float = 0.2,
    n_random_baselines: int = 50,
    precision: str = "mixed",
) -> dict:
    """Validate feature interactions on held-out data."""

    n = X.shape[0]
    n_features = X.shape[1]
    if precision not in {"fp32", "mixed", "fp64"}:
        raise ValueError("precision must be one of: fp32, mixed, fp64")
    if n_features < 2:
        raise ValueError("at least two numeric feature columns are required")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(42)

    # Train/test split (deterministic)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1 - test_size))
    if split < 3 or n - split < 3:
        raise ValueError(
            "train and holdout partitions must each contain at least 3 rows"
        )
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Compute random baseline
    random_rs = []
    for _ in range(n_random_baselines):
        i, j = rng.integers(0, n_features, size=2)
        if i == j:
            j = (j + 1) % n_features
        vec = interaction_vector(X_test, int(i), int(j), operator, precision=precision)
        random_rs.append(abs(pearson_r(vec, y_test, precision=precision)))

    baseline_mean = float(np.mean(random_rs))
    baseline_std = float(np.std(random_rs))
    baseline_p95 = float(np.percentile(random_rs, 95))

    # Validate each interaction
    results = []
    for pair in interactions:
        feat_i, feat_j = pair

        if not (0 <= feat_i < n_features and 0 <= feat_j < n_features):
            raise ValueError(
                f"interaction {(feat_i, feat_j)} is outside numeric feature range "
                f"0..{n_features - 1}"
            )
        if feat_i == feat_j:
            raise ValueError(
                "continuous pair interactions require two distinct features"
            )

        # Compute interaction
        train_vec = interaction_vector(
            X_train, feat_i, feat_j, operator, precision=precision
        )
        test_vec = interaction_vector(
            X_test, feat_i, feat_j, operator, precision=precision
        )

        r_train = pearson_r(train_vec, y_train, precision=precision)
        r_test = pearson_r(test_vec, y_test, precision=precision)
        ci_lo, ci_hi = bootstrap_ci(test_vec, y_test, precision=precision)

        # Verdict
        passes_heuristic = (
            abs(r_test) > baseline_p95  # Stronger than 95% of random pairs
            and abs(r_test) > 0.05  # Not negligible
            and ci_lo * ci_hi > 0  # CI doesn't cross zero (consistent sign)
        )

        degradation = abs(r_train) - abs(r_test)
        overfitting_risk = degradation > 0.2 * abs(r_train) if abs(r_train) > 0.05 else False

        results.append(
            {
                "features": [int(feat_i), int(feat_j)],
                "finite_train_rows": int(
                    np.count_nonzero(np.isfinite(train_vec) & np.isfinite(y_train))
                ),
                "finite_holdout_rows": int(
                    np.count_nonzero(np.isfinite(test_vec) & np.isfinite(y_test))
                ),
                "r_train": round(r_train, 4),
                "r_test": round(r_test, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "degradation": round(degradation, 4),
                "overfitting_risk": overfitting_risk,
                "status": (
                    "HEURISTIC_PASS" if passes_heuristic else "HEURISTIC_INCONCLUSIVE"
                ),
            }
        )

    pass_count = sum(1 for row in results if row["status"] == "HEURISTIC_PASS")
    inconclusive_count = len(results) - pass_count

    report = {
        "n_interactions_tested": len(interactions),
        "heuristic_pass_count": pass_count,
        "heuristic_inconclusive_count": inconclusive_count,
        "evidence_boundary": (
            "Descriptive holdout heuristic only; not proof of causality, generalization, "
            "or GAFIME backend parity."
        ),
        "precision": precision,
        "baseline": {
            "random_mean_r": round(baseline_mean, 4),
            "random_std_r": round(baseline_std, 4),
            "random_p95_r": round(baseline_p95, 4),
        },
        "split_info": {
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
        },
        "interactions": results,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="GAFIME Feature Validation")
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument("--interactions", "-i", required=True,
                        help="Semicolon-separated feature index pairs, e.g. '0,1;2,3;0,4'")
    parser.add_argument("--operator", default="multiply", choices=["multiply", "add", "subtract", "divide"])
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument(
        "--precision", default="mixed", choices=["fp32", "mixed", "fp64"]
    )
    args = parser.parse_args()

    # Parse interactions
    interactions = []
    for pair_str in args.interactions.split(";"):
        parts = pair_str.strip().split(",")
        if len(parts) == 2:
            interactions.append((int(parts[0]), int(parts[1])))

    if not interactions:
        print(json.dumps({"error": "No valid interactions provided"}))
        return 1

    # Load data
    path = Path(args.data)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {args.data}"}))
        return 1

    try:
        import polars as pl
        if path.suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path, infer_schema_length=10000)

        if args.target not in df.columns:
            raise ValueError(f"Target column not found: {args.target}")
        if not df.schema[args.target].is_numeric():
            raise TypeError("GAFIME requires a numeric target; encode the target first")
        feature_cols = [
            name
            for name, dtype in df.schema.items()
            if name != args.target and dtype.is_numeric()
        ]
        if not feature_cols:
            raise ValueError("No numeric feature columns remain after excluding the target")
        dtype = np.float64 if args.precision == "fp64" else np.float32
        X = df.select(feature_cols).to_numpy().astype(dtype)
        y = df[args.target].to_numpy().astype(dtype)
    except ImportError:
        import csv
        # Fallback to numpy-only loading
        print(json.dumps({"error": "Polars is required for data loading"}))
        return 1

    report = validate_interactions(
        X,
        y,
        interactions,
        operator=args.operator,
        test_size=args.test_size,
        precision=args.precision,
    )
    report["feature_names"] = feature_cols
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
