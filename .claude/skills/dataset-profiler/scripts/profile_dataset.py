#!/usr/bin/env python3
"""
GAFIME Dataset Profiler

Analyzes a CSV or Parquet file and reports data quality, memory estimates,
and GAFIME compatibility information.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def profile_dataset(file_path: str, target_col: str = None, vram_gb: float = 6.0) -> dict:
    """Profile a dataset file and return analysis report."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Read the data
    if not HAS_POLARS:
        return {"error": "Polars is required. Install with: pip install polars"}

    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, infer_schema_length=10000)
    else:
        return {"error": f"Unsupported format: {path.suffix}. Use .csv or .parquet"}

    n_rows, n_cols = df.shape

    # Column analysis
    columns = []
    numeric_cols = []
    problematic_cols = []
    warnings = []

    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        null_count = col.null_count()
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0

        col_info = {
            "name": col_name,
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
            "is_numeric": col.dtype.is_numeric(),
            "is_target": col_name == target_col,
        }

        if col.dtype.is_numeric():
            numeric_cols.append(col_name)
            try:
                std_val = col.drop_nulls().cast(pl.Float64).std()
                col_info["std"] = round(float(std_val), 6) if std_val is not None else 0.0
                col_info["zero_variance"] = (std_val is not None and float(std_val) < 1e-10)
            except Exception:
                col_info["std"] = None
                col_info["zero_variance"] = False
        else:
            n_unique = col.n_unique()
            col_info["n_unique"] = n_unique
            col_info["high_cardinality"] = n_unique > 100

        # Flag problems
        if null_pct > 50:
            problematic_cols.append(col_name)
            warnings.append(f"Column '{col_name}' is {null_pct:.0f}% null")
        if col_info.get("zero_variance"):
            problematic_cols.append(col_name)
            warnings.append(f"Column '{col_name}' has zero variance (constant)")
        if not col.dtype.is_numeric() and col_name != target_col:
            warnings.append(f"Column '{col_name}' is non-numeric ({dtype_str}), needs encoding for GAFIME")

        columns.append(col_info)

    # Feature count (numeric, non-target)
    feature_cols = [c for c in numeric_cols if c != target_col]
    n_features = len(feature_cols)

    # Memory estimates
    bytes_per_element = 4  # float32
    raw_feature_bytes = n_rows * n_features * bytes_per_element
    raw_feature_mb = raw_feature_bytes / (1024 * 1024)
    target_bytes = n_rows * bytes_per_element
    mask_bytes = n_rows  # uint8
    total_gpu_mb = (raw_feature_bytes + target_bytes + mask_bytes) / (1024 * 1024)

    # VRAM analysis
    usable_vram_mb = vram_gb * 1024 * 0.75  # 25% headroom
    fits_in_vram = total_gpu_mb <= usable_vram_mb

    # Optimal batch size if streaming needed
    if not fits_in_vram and n_features > 0:
        bytes_per_row = n_features * bytes_per_element + bytes_per_element + 1  # features + target + mask
        max_rows = int(usable_vram_mb * 1024 * 1024 / bytes_per_row)
        optimal_batch = (max_rows // 1024) * 1024  # GPU-friendly alignment
        optimal_batch = max(1024, optimal_batch)
    else:
        optimal_batch = n_rows

    # Target column analysis
    target_info = None
    if target_col and target_col in df.columns:
        t_col = df[target_col]
        if t_col.dtype.is_numeric():
            n_unique = t_col.drop_nulls().n_unique()
            target_info = {
                "name": target_col,
                "dtype": str(t_col.dtype),
                "n_unique": n_unique,
                "is_binary": n_unique == 2,
                "task_type": "classification" if n_unique <= 20 else "regression",
                "null_count": t_col.null_count(),
            }

    report = {
        "file": str(path.absolute()),
        "file_size_mb": round(file_size_mb, 2),
        "rows": n_rows,
        "total_columns": n_cols,
        "numeric_columns": len(numeric_cols),
        "feature_columns": n_features,
        "target": target_info,
        "memory": {
            "raw_features_mb": round(raw_feature_mb, 2),
            "total_gpu_mb": round(total_gpu_mb, 2),
            "available_vram_mb": round(usable_vram_mb, 2),
            "fits_in_vram": fits_in_vram,
            "optimal_batch_size": optimal_batch,
            "needs_streaming": not fits_in_vram,
        },
        "data_quality": {
            "zero_variance_columns": [c["name"] for c in columns if c.get("zero_variance")],
            "high_null_columns": [c["name"] for c in columns if c.get("null_pct", 0) > 50],
            "non_numeric_columns": [c["name"] for c in columns if not c.get("is_numeric") and not c.get("is_target")],
            "problematic_column_count": len(set(problematic_cols)),
        },
        "warnings": warnings,
        "columns": columns,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="GAFIME Dataset Profiler")
    parser.add_argument("file", help="Path to CSV or Parquet file")
    parser.add_argument("--target", "-t", help="Target column name", default=None)
    parser.add_argument("--vram", "-v", type=float, default=6.0, help="Available VRAM in GB (default: 6.0)")
    args = parser.parse_args()

    report = profile_dataset(args.file, target_col=args.target, vram_gb=args.vram)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
