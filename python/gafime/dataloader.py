"""Polars-backed data loading for GAFIME's top-level API.

``gafime.dataload(path, target)`` reads a parquet/CSV/Arrow file with Polars,
converts to the selected profile's resident dtype, and runs the engine. Polars is
the *external* loader; GAFIME still owns all compute memory internally. The
Polars import is lazy so importing this module never requires Polars.

The adapter uses the Arrow-native CPU shortcut only when that entrypoint can
honor the complete ``EngineConfig``. Other configurations use the configured
native boundary, which copies rows into profile-keyed GAFIME-owned storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

from .config import EngineConfig
from .reporting import DiagnosticReport

_PARQUET = {".parquet", ".pq"}
_CSV = {".csv", ".tsv", ".txt"}
_ARROW = {".ipc", ".arrow", ".feather"}


def _resolve_feature_columns(
    columns: Sequence[str],
    target: str,
    features: Sequence[str] | None,
) -> List[str]:
    """Pick the feature columns from a frame's schema (pure, Polars-free)."""
    if not isinstance(target, str):
        raise ValueError("target must name exactly one column")
    columns = list(columns)
    target_count = columns.count(target)
    if target_count == 0:
        raise ValueError(f"target column {target!r} not found in {columns}")
    if target_count != 1:
        raise ValueError(f"target column {target!r} must appear exactly once")
    if features is None:
        selected = [name for name in columns if name != target]
    else:
        missing = [name for name in features if name not in columns]
        if missing:
            raise ValueError(f"feature columns not found: {missing}")
        if target in features:
            raise ValueError(f"target {target!r} cannot also be a feature column")
        selected = list(features)
    if not selected:
        raise ValueError("no feature columns resolved (need at least one feature)")
    return selected


def _read_frame(path: Path, **read_kwargs: Any):
    """Read a file into a Polars DataFrame, dispatched by suffix."""
    import polars as pl

    suffix = path.suffix.lower()
    if suffix in _PARQUET:
        return pl.read_parquet(path, **read_kwargs)
    if suffix in _CSV:
        return pl.read_csv(path, **read_kwargs)
    if suffix in _ARROW:
        return pl.read_ipc(path, **read_kwargs)
    raise ValueError(
        f"unsupported file type {suffix!r}; use parquet, csv, or arrow/ipc"
    )


def dataload(
    path: str | Path,
    target: str,
    features: Sequence[str] | None = None,
    *,
    config: EngineConfig | None = None,
    **read_kwargs: Any,
) -> DiagnosticReport:
    """Load a dataset with Polars and run GAFIME on it.

    Parameters
    ----------
    path: parquet / CSV / Arrow-IPC file.
    target: name of the target column.
    features: feature column names (default: every column except the target).
    config: engine configuration (default: ``EngineConfig()``).
    read_kwargs: forwarded to the Polars reader.
    """
    from .v1_adapter import _validate_precision_config, analyze_arrow_with_v1_boundary

    effective_config = config or EngineConfig()
    # Validate the complete request before importing Polars, reading a file, or
    # coercing any values. In particular, explicit Metal mixed/fp64 requests
    # fail closed without an fp32 intermediate.
    precision = _validate_precision_config(effective_config)

    import polars as pl

    frame = _read_frame(Path(path), **read_kwargs)
    feature_cols = _resolve_feature_columns(frame.columns, target, features)

    # fp32/mixed intentionally own fp32 resident values; fp64 preserves f64
    # from this first conversion onward. Rechunk so each frame arrives as one
    # Arrow record batch.
    resident_dtype = pl.Float64 if precision == "fp64" else pl.Float32
    feature_frame = frame.select(feature_cols).cast(resident_dtype).rechunk()
    target_frame = frame.select(target).cast(resident_dtype).rechunk()

    # The adapter retains the Arrow-native shortcut only when it can honor every
    # relevant setting. Other configurations use the normal configured boundary
    # rather than silently becoming a CPU/no-significance run.
    return analyze_arrow_with_v1_boundary(
        effective_config, feature_frame, target_frame, feature_cols
    )
