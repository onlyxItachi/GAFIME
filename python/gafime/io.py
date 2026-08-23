from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Union

try:
    import polars as pl
except ImportError:  # pragma: no cover - package metadata requires Polars
    pl = None


class GafimeStreamer:
    """Stream numeric CSV or Parquet batches through Polars.

    ``target_cols`` selects feature columns (the historical name is retained);
    otherwise every column except ``y_col`` is used.  ``precision`` controls
    the resident batch dtype: fp64 for ``fp64`` and fp32 for ``fp32``/``mixed``.
    The streamer returns Python lists and does not itself execute GAFIME or
    retain a backend session.  Missing files, unsupported suffixes, or missing
    Polars fail immediately.
    """

    DEFAULT_VRAM_GB = 6.0
    VRAM_HEADROOM = 0.20
    BYTES_PER_FLOAT32 = 4
    BYTES_PER_FLOAT64 = 8

    def __init__(
        self,
        file_path: Union[str, Path],
        target_cols: Optional[List[str]] = None,
        y_col: Optional[str] = None,
        *,
        precision: str = "mixed",
    ) -> None:
        if pl is None:
            raise ImportError("Polars is required for GafimeStreamer.")
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        self.target_cols = target_cols
        self.y_col = y_col
        from ._precision import normalize_precision

        self.precision = normalize_precision(precision)
        self._lazy_df = self._create_lazy_reader()
        self._schema = self._lazy_df.collect_schema()
        self._all_columns = list(self._schema.names())
        self._feature_cols = (
            list(target_cols)
            if target_cols is not None
            else [column for column in self._all_columns if column != y_col]
        )
        self.n_features = len(self._feature_cols)
        self._total_rows: Optional[int] = None

    def _create_lazy_reader(self):
        suffix = self.file_path.suffix.lower()
        if suffix == ".parquet":
            return pl.scan_parquet(self.file_path)
        if suffix == ".csv":
            return pl.scan_csv(self.file_path)
        raise ValueError("Unsupported file format. Use .csv or .parquet.")

    @property
    def total_rows(self) -> int:
        """Return the lazily computed total input row count."""

        if self._total_rows is None:
            self._total_rows = int(self._lazy_df.select(pl.len()).collect().item())
        return self._total_rows

    def estimate_optimal_batch_size(
        self,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
        include_output: bool = True,
        n_combos: int = 256,
    ) -> int:
        """Estimate a row batch size from a simple memory budget model.

        ``include_output`` adds ``n_combos`` result values per row in the
        selected public-result width.  The estimate is a convenience heuristic,
        not a device allocation guarantee, and is rounded to a minimum/multiple
        of 1024 rows.
        """

        usable_bytes = vram_budget_gb * (1024**3) * (1.0 - self.VRAM_HEADROOM)
        bytes_per_value = (
            self.BYTES_PER_FLOAT64
            if self.precision == "fp64"
            else self.BYTES_PER_FLOAT32
        )
        bytes_per_row = self.n_features * bytes_per_value
        if include_output:
            result_bytes = (
                self.BYTES_PER_FLOAT32
                if self.precision == "fp32"
                else self.BYTES_PER_FLOAT64
            )
            bytes_per_row += int(n_combos) * result_bytes
        return max(1024, int(usable_bytes / max(bytes_per_row, 1)) // 1024 * 1024)

    def stream(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[List[List[float]], None, None]:
        """Yield feature-only Python row batches in source order.

        An explicit ``batch_size`` must be a positive integer.  The beta.2
        compatibility streamer does not yet reject a non-positive value; such
        a value does not advance the reader, so callers must validate it.
        """

        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        reader = self._lazy_df.select(self._feature_cols)
        current_row = 0
        total = self.total_rows
        while current_row < total:
            this_batch = min(batch_size, total - current_row)
            frame = self._cast_resident(reader.slice(current_row, this_batch).collect())
            yield self._frame_to_rows(frame, self._feature_cols)
            current_row += this_batch

    def stream_with_target(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[tuple[List[List[float]], List[float]], None, None]:
        """Yield ``(features, target)`` row batches in source order.

        ``y_col`` must have been supplied at construction time.  An explicit
        ``batch_size`` must be positive; beta.2 does not yet reject a
        non-positive value, which prevents the reader from advancing.
        """

        if self.y_col is None:
            raise ValueError("y_col must be specified for stream_with_target().")
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        reader = self._lazy_df.select(self._feature_cols + [self.y_col])
        current_row = 0
        total = self.total_rows
        while current_row < total:
            this_batch = min(batch_size, total - current_row)
            frame = self._cast_resident(reader.slice(current_row, this_batch).collect())
            yield (
                self._frame_to_rows(frame, self._feature_cols),
                [float(value) for value in frame[self.y_col].to_list()],
            )
            current_row += this_batch

    @staticmethod
    def _frame_to_rows(frame, columns: List[str]) -> List[List[float]]:
        return [[float(row[column]) for column in columns] for row in frame.to_dicts()]

    def _cast_resident(self, frame):
        dtype = pl.Float64 if self.precision == "fp64" else pl.Float32
        return frame.cast(dtype).rechunk()


def create_streamer(*args, **kwargs) -> GafimeStreamer:
    """Compatibility factory forwarding all arguments to ``GafimeStreamer``."""

    return GafimeStreamer(*args, **kwargs)


def benchmark_streaming(file_path, batch_size=None, n_batches=5):
    """Read at most ``n_batches`` and return simple batch/row counts.

    This compatibility helper measures iteration shape only; it does not run a
    backend or make a performance claim.  ``n_batches`` must be positive.
    Beta.2 does not yet reject a non-positive value and still reads one batch,
    so callers must validate this diagnostic-only argument.
    """

    streamer = GafimeStreamer(file_path)
    count = 0
    rows = 0
    for batch in streamer.stream(batch_size=batch_size):
        rows += len(batch)
        count += 1
        if count >= n_batches:
            break
    return {"batches": count, "rows": rows}


__all__ = ["GafimeStreamer", "benchmark_streaming", "create_streamer"]
