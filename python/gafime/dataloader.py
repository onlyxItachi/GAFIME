"""Polars-backed data loading for GAFIME's top-level API.

``gafime.dataload(path, target)`` reads a parquet/CSV/Arrow file with Polars,
quantizes to fp32 (GAFIME's execution dtype), and runs the engine. Polars is
the *external* loader; GAFIME still owns all compute memory internally. The
Polars import is lazy so importing this module never requires Polars.

The handoff uses Arrow-native ingest: Polars handles external format and
framework compatibility, while GAFIME validates and owns compute memory after
the boundary.
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
    columns = list(columns)
    if target not in columns:
        raise ValueError(f"target column {target!r} not found in {columns}")
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
    import polars as pl

    from .v1_adapter import analyze_arrow_with_v1_boundary

    frame = _read_frame(Path(path), **read_kwargs)
    feature_cols = _resolve_feature_columns(frame.columns, target, features)

    # Quantize to fp32 in Polars (GAFIME's execution dtype); rechunk so each frame
    # arrives as a single Arrow record batch.
    feature_frame = frame.select(feature_cols).cast(pl.Float32).rechunk()
    target_frame = frame.select(target).cast(pl.Float32).rechunk()

    # Zero-copy Arrow ingest: data crosses as Arrow buffers, no .rows()/.tolist().
    return analyze_arrow_with_v1_boundary(
        config or EngineConfig(), feature_frame, target_frame, feature_cols
    )
