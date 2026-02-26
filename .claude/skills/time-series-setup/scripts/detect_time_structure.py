#!/usr/bin/env python3
"""
GAFIME Time-Series Structure Detector

Analyzes a dataset to detect time columns, group columns, and temporal patterns.
Recommends TimeSeriesPreprocessor configuration.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def detect_time_structure(file_path: str) -> dict:
    """Detect temporal structure in a dataset."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    if not HAS_POLARS:
        return {"error": "Polars is required. Install with: pip install polars"}

    # Load data
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=10000)
    else:
        return {"error": f"Unsupported format: {path.suffix}"}

    n_rows, n_cols = df.shape

    # Detect time columns
    time_columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype = col.dtype

        is_temporal = False
        time_info = {"name": col_name, "dtype": str(dtype)}

        # Check for datetime/date types
        if dtype == pl.Date or dtype == pl.Datetime or str(dtype).startswith("Datetime"):
            is_temporal = True
            try:
                sorted_col = col.drop_nulls().sort()
                if len(sorted_col) >= 2:
                    first = sorted_col[0]
                    last = sorted_col[-1]
                    time_info["min"] = str(first)
                    time_info["max"] = str(last)
                    span = last - first
                    time_info["span_days"] = span.days if hasattr(span, 'days') else None

                    # Detect granularity from median diff
                    if len(sorted_col) >= 3:
                        diffs = sorted_col.diff().drop_nulls()
                        if len(diffs) > 0:
                            median_diff = diffs.sort()[len(diffs) // 2]
                            if hasattr(median_diff, 'days'):
                                days = median_diff.days
                                if days < 1:
                                    time_info["granularity"] = "hourly"
                                elif days <= 1:
                                    time_info["granularity"] = "daily"
                                elif days <= 7:
                                    time_info["granularity"] = "weekly"
                                else:
                                    time_info["granularity"] = "monthly"
            except Exception:
                pass

        # Check string columns that might be dates
        if not is_temporal and (dtype == pl.Utf8 or dtype == pl.String):
            try:
                sample = col.drop_nulls().head(10)
                parsed = sample.str.to_datetime(strict=False)
                if parsed.null_count() < len(sample) * 0.5:
                    is_temporal = True
                    time_info["note"] = "String column that can be parsed as datetime"
            except Exception:
                pass

        if is_temporal:
            time_columns.append(time_info)

    # Detect potential group columns (low-to-medium cardinality non-numeric)
    group_candidates = []
    for col_name in df.columns:
        col = df[col_name]
        n_unique = col.n_unique()
        ratio = n_unique / n_rows if n_rows > 0 else 1

        is_candidate = False
        if col.dtype in (pl.Utf8, pl.String, pl.Categorical):
            if 2 <= n_unique <= n_rows * 0.5:
                is_candidate = True
        elif col.dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            if 2 <= n_unique <= n_rows * 0.5 and ratio < 0.9:
                is_candidate = True

        if is_candidate:
            # Check if it looks like an ID column
            name_lower = col_name.lower()
            score = 0
            if any(kw in name_lower for kw in ["id", "user", "customer", "account", "client", "entity"]):
                score += 3
            if any(kw in name_lower for kw in ["group", "category", "segment", "type"]):
                score += 2

            rows_per_group = n_rows / n_unique if n_unique > 0 else 0
            group_candidates.append({
                "name": col_name,
                "n_unique": n_unique,
                "rows_per_group": round(rows_per_group, 1),
                "relevance_score": score,
            })

    group_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Detect potential target columns
    target_candidates = []
    for col_name in df.columns:
        col = df[col_name]
        if not col.dtype.is_numeric():
            continue

        n_unique = col.drop_nulls().n_unique()
        name_lower = col_name.lower()
        score = 0

        if any(kw in name_lower for kw in ["target", "label", "churn", "fraud", "default", "class", "outcome"]):
            score += 5
        if n_unique == 2:
            score += 3  # Binary target
        elif n_unique <= 10:
            score += 1  # Multi-class

        if score > 0:
            target_candidates.append({
                "name": col_name,
                "n_unique": n_unique,
                "is_binary": n_unique == 2,
                "relevance_score": score,
            })

    target_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Recommend windows based on detected granularity
    granularity = None
    if time_columns:
        granularity = time_columns[0].get("granularity")

    window_recommendations = {
        "hourly": [6, 12, 24, 48, 168],
        "daily": [7, 14, 30, 60, 90, 180, 360],
        "weekly": [4, 8, 13, 26, 52],
        "monthly": [3, 6, 12, 24],
    }

    recommended_windows = window_recommendations.get(granularity, [7, 14, 30, 60, 90, 180, 360])

    # Estimate feature explosion
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    n_numeric = len(numeric_cols)
    n_windows = len(recommended_windows)
    basic_features = n_numeric * n_windows * 5  # lags, diff, rolling_mean, rolling_std, rolling_sum
    calculus_features = n_numeric * n_windows * 4  # velocity, acceleration, volatility, trend

    report = {
        "file": str(path.absolute()),
        "rows": n_rows,
        "columns": n_cols,
        "time_columns": time_columns,
        "group_candidates": group_candidates[:5],
        "target_candidates": target_candidates[:3],
        "detected_granularity": granularity,
        "recommended_windows": recommended_windows,
        "feature_estimates": {
            "numeric_input_features": n_numeric,
            "basic_features": basic_features,
            "calculus_features": calculus_features,
            "total_estimated": basic_features + calculus_features,
        },
        "recommended_config": {
            "group_col": group_candidates[0]["name"] if group_candidates else None,
            "time_col": time_columns[0]["name"] if time_columns else None,
            "windows": recommended_windows,
            "enable_calculus": True,
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="GAFIME Time-Series Structure Detector")
    parser.add_argument("file", help="Path to CSV or Parquet file")
    args = parser.parse_args()

    report = detect_time_structure(args.file)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
