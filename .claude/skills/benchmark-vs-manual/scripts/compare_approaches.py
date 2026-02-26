#!/usr/bin/env python3
"""
GAFIME vs Manual Features Comparison

Runs a controlled comparison between GAFIME-discovered features
and manually crafted features using cross-validation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    manual_feature_indices: list = None,
    task: str = "classification",
    k: int = 10,
    operator: str = "multiply",
    n_folds: int = 5,
    feature_names: list = None,
) -> dict:
    """Run three-way comparison: baseline, manual, GAFIME."""

    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)
        scoring = "roc_auc"
        cv_factory = lambda: StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        from sklearn.linear_model import Ridge
        model_factory = lambda: Ridge(alpha=1.0)
        scoring = "r2"
        cv_factory = lambda: KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}

    # Experiment 1: Baseline (original features only)
    pipe_baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_factory()),
    ])
    scores_baseline = cross_val_score(pipe_baseline, X, y, cv=cv_factory(), scoring=scoring)
    results["baseline"] = {
        "mean": round(float(scores_baseline.mean()), 4),
        "std": round(float(scores_baseline.std()), 4),
        "scores": [round(float(s), 4) for s in scores_baseline],
        "n_features": X.shape[1],
    }

    # Experiment 2: Manual features (if provided)
    if manual_feature_indices:
        manual_features = []
        for pair in manual_feature_indices:
            i, j = pair
            if operator == "multiply":
                manual_features.append(X[:, i] * X[:, j])
            elif operator == "add":
                manual_features.append(X[:, i] + X[:, j])
            elif operator == "subtract":
                manual_features.append(X[:, i] - X[:, j])
            elif operator == "divide":
                manual_features.append(X[:, i] / (X[:, j] + 1e-8))

        X_manual = np.column_stack([X] + manual_features)

        pipe_manual = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_factory()),
        ])
        scores_manual = cross_val_score(pipe_manual, X_manual, y, cv=cv_factory(), scoring=scoring)
        results["manual"] = {
            "mean": round(float(scores_manual.mean()), 4),
            "std": round(float(scores_manual.std()), 4),
            "scores": [round(float(s), 4) for s in scores_manual],
            "n_features": X_manual.shape[1],
            "n_manual_features": len(manual_features),
        }

    # Experiment 3: GAFIME features
    try:
        from gafime.sklearn import GafimeSelector

        pipe_gafime = Pipeline([
            ("gafime", GafimeSelector(k=k, backend="auto", metric="pearson", operator=operator)),
            ("scaler", StandardScaler()),
            ("model", model_factory()),
        ])
        scores_gafime = cross_val_score(pipe_gafime, X, y, cv=cv_factory(), scoring=scoring)

        # Get the discovered interactions from a full fit
        gafime_selector = GafimeSelector(k=k, backend="auto", metric="pearson", operator=operator)
        gafime_selector.fit(X, y)

        discovered = []
        for feat_i, feat_j in gafime_selector.top_interactions_:
            name_i = feature_names[feat_i] if feature_names and feat_i < len(feature_names) else f"f{feat_i}"
            name_j = feature_names[feat_j] if feature_names and feat_j < len(feature_names) else f"f{feat_j}"
            discovered.append({
                "indices": [int(feat_i), int(feat_j)],
                "names": [name_i, name_j],
            })

        results["gafime"] = {
            "mean": round(float(scores_gafime.mean()), 4),
            "std": round(float(scores_gafime.std()), 4),
            "scores": [round(float(s), 4) for s in scores_gafime],
            "n_features": X.shape[1] + k,
            "discovered_interactions": discovered,
        }
    except ImportError:
        results["gafime"] = {"error": "scikit-learn or gafime.sklearn not available. Install with: pip install gafime[sklearn]"}

    # Experiment 4: Combined (manual + GAFIME) if both available
    if "manual" in results and "gafime" in results and "error" not in results["gafime"]:
        try:
            pipe_combined = Pipeline([
                ("gafime", GafimeSelector(k=k, backend="auto", metric="pearson", operator=operator)),
                ("scaler", StandardScaler()),
                ("model", model_factory()),
            ])
            X_combined_input = X_manual  # manual features already appended
            scores_combined = cross_val_score(pipe_combined, X_combined_input, y, cv=cv_factory(), scoring=scoring)
            results["combined"] = {
                "mean": round(float(scores_combined.mean()), 4),
                "std": round(float(scores_combined.std()), 4),
                "scores": [round(float(s), 4) for s in scores_combined],
                "n_features": X_combined_input.shape[1] + k,
            }
        except Exception as e:
            results["combined"] = {"error": str(e)}

    # Summary
    approaches = ["baseline", "manual", "gafime", "combined"]
    ranking = []
    for name in approaches:
        if name in results and "mean" in results[name]:
            ranking.append((name, results[name]["mean"]))
    ranking.sort(key=lambda x: x[1], reverse=True)

    results["ranking"] = [{"approach": name, "score": score} for name, score in ranking]

    if len(ranking) >= 2:
        best = ranking[0]
        baseline_score = results["baseline"]["mean"]
        results["best_approach"] = best[0]
        results["improvement_over_baseline"] = round(best[1] - baseline_score, 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="GAFIME vs Manual Feature Comparison")
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument("--manual-features", "-m", default=None,
                        help="Semicolon-separated index pairs for manual features, e.g. '0,1;2,3'")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--k", type=int, default=10, help="Number of GAFIME interactions")
    parser.add_argument("--operator", default="multiply", choices=["multiply", "add", "subtract", "divide"])
    args = parser.parse_args()

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
        feature_names = feature_cols
    except ImportError:
        print(json.dumps({"error": "Polars is required. Install with: pip install polars"}))
        return 1

    # Parse manual features
    manual_pairs = None
    if args.manual_features:
        manual_pairs = []
        for pair_str in args.manual_features.split(";"):
            parts = pair_str.strip().split(",")
            if len(parts) == 2:
                manual_pairs.append((int(parts[0]), int(parts[1])))

    report = run_comparison(
        X, y,
        manual_feature_indices=manual_pairs,
        task=args.task,
        k=args.k,
        operator=args.operator,
        feature_names=feature_names,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
