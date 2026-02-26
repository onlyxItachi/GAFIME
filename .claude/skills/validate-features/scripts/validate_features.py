#!/usr/bin/env python3
"""
GAFIME Feature Validation Script

Validates whether discovered feature interactions are genuinely predictive
by testing on held-out data with bootstrap confidence intervals.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx**2).sum() * (dy**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((dx * dy).sum() / denom)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for Pearson r."""
    rng = np.random.default_rng(42)
    n = len(x)
    rs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rs[i] = pearson_r(x[idx], y[idx])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(rs, alpha * 100))
    hi = float(np.percentile(rs, (1 - alpha) * 100))
    return lo, hi


def validate_interactions(
    X: np.ndarray,
    y: np.ndarray,
    interactions: list,
    operator: str = "multiply",
    test_size: float = 0.2,
    n_random_baselines: int = 50,
) -> dict:
    """Validate feature interactions on held-out data."""

    n = X.shape[0]
    n_features = X.shape[1]
    rng = np.random.default_rng(42)

    # Train/test split (deterministic)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Compute random baseline
    random_rs = []
    for _ in range(n_random_baselines):
        i, j = rng.integers(0, n_features, size=2)
        if i == j:
            j = (j + 1) % n_features
        if operator == "multiply":
            vec = X_test[:, i] * X_test[:, j]
        elif operator == "add":
            vec = X_test[:, i] + X_test[:, j]
        else:
            vec = X_test[:, i] * X_test[:, j]
        random_rs.append(abs(pearson_r(vec, y_test)))

    baseline_mean = float(np.mean(random_rs))
    baseline_std = float(np.std(random_rs))
    baseline_p95 = float(np.percentile(random_rs, 95))

    # Validate each interaction
    results = []
    for pair in interactions:
        feat_i, feat_j = pair

        # Compute interaction
        if operator == "multiply":
            train_vec = X_train[:, feat_i] * X_train[:, feat_j]
            test_vec = X_test[:, feat_i] * X_test[:, feat_j]
        elif operator == "add":
            train_vec = X_train[:, feat_i] + X_train[:, feat_j]
            test_vec = X_test[:, feat_i] + X_test[:, feat_j]
        elif operator == "subtract":
            train_vec = X_train[:, feat_i] - X_train[:, feat_j]
            test_vec = X_test[:, feat_i] - X_test[:, feat_j]
        elif operator == "divide":
            train_vec = X_train[:, feat_i] / (X_train[:, feat_j] + 1e-8)
            test_vec = X_test[:, feat_i] / (X_test[:, feat_j] + 1e-8)
        else:
            train_vec = X_train[:, feat_i] * X_train[:, feat_j]
            test_vec = X_test[:, feat_i] * X_test[:, feat_j]

        r_train = pearson_r(train_vec, y_train)
        r_test = pearson_r(test_vec, y_test)
        ci_lo, ci_hi = bootstrap_ci(test_vec, y_test)

        # Verdict
        is_genuine = (
            abs(r_test) > baseline_p95  # Stronger than 95% of random pairs
            and abs(r_test) > 0.05  # Not negligible
            and ci_lo * ci_hi > 0  # CI doesn't cross zero (consistent sign)
        )

        degradation = abs(r_train) - abs(r_test)
        overfitting_risk = degradation > 0.2 * abs(r_train) if abs(r_train) > 0.05 else False

        results.append({
            "features": [int(feat_i), int(feat_j)],
            "r_train": round(r_train, 4),
            "r_test": round(r_test, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "degradation": round(degradation, 4),
            "overfitting_risk": overfitting_risk,
            "verdict": "GENUINE" if is_genuine else "NOISE",
        })

    genuine_count = sum(1 for r in results if r["verdict"] == "GENUINE")
    noise_count = sum(1 for r in results if r["verdict"] == "NOISE")

    report = {
        "n_interactions_tested": len(interactions),
        "genuine_count": genuine_count,
        "noise_count": noise_count,
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

        feature_cols = [c for c in df.columns if c != args.target]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df[args.target].to_numpy().astype(np.float32)
    except ImportError:
        import csv
        # Fallback to numpy-only loading
        print(json.dumps({"error": "Polars is required for data loading"}))
        return 1

    report = validate_interactions(X, y, interactions, operator=args.operator, test_size=args.test_size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
