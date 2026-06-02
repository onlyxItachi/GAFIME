from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Union

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


class GafimeStreamer:
    DEFAULT_VRAM_GB = 6.0
    VRAM_HEADROOM = 0.20
    BYTES_PER_FLOAT32 = 4

    def __init__(
        self,
        file_path: Union[str, Path],
        target_cols: Optional[List[str]] = None,
        y_col: Optional[str] = None,
    ) -> None:
        if pl is None:
            raise ImportError("Polars is required for GafimeStreamer.")
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        self.target_cols = target_cols
        self.y_col = y_col
        self._lazy_df = self._create_lazy_reader()
        self._schema = self._lazy_df.collect_schema()
        self._all_columns = list(self._schema.names())
        self._feature_cols = list(target_cols) if target_cols is not None else [
            col for col in self._all_columns if col != y_col
        ]
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
        if self._total_rows is None:
            self._total_rows = int(self._lazy_df.select(pl.len()).collect().item())
        return self._total_rows

    def estimate_optimal_batch_size(
        self,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
        include_output: bool = True,
        n_combos: int = 256,
    ) -> int:
        usable_bytes = vram_budget_gb * (1024**3) * (1.0 - self.VRAM_HEADROOM)
        bytes_per_row = self.n_features * self.BYTES_PER_FLOAT32
        if include_output:
            bytes_per_row += int(n_combos) * self.BYTES_PER_FLOAT32
        return max(1024, int(usable_bytes / max(bytes_per_row, 1)) // 1024 * 1024)

    def stream(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[List[List[float]], None, None]:
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        reader = self._lazy_df.select(self._feature_cols)
        current_row = 0
        total = self.total_rows
        while current_row < total:
            this_batch = min(batch_size, total - current_row)
            yield self._frame_to_rows(reader.slice(current_row, this_batch).collect(), self._feature_cols)
            current_row += this_batch

    def stream_with_target(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[tuple[List[List[float]], List[float]], None, None]:
        if self.y_col is None:
            raise ValueError("y_col must be specified for stream_with_target().")
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        reader = self._lazy_df.select(self._feature_cols + [self.y_col])
        current_row = 0
        total = self.total_rows
        while current_row < total:
            this_batch = min(batch_size, total - current_row)
            frame = reader.slice(current_row, this_batch).collect()
            yield self._frame_to_rows(frame, self._feature_cols), [
                float(value) for value in frame[self.y_col].to_list()
            ]
            current_row += this_batch

    @staticmethod
    def _frame_to_rows(frame, columns: List[str]) -> List[List[float]]:
        return [[float(row[col]) for col in columns] for row in frame.to_dicts()]


def create_streamer(*args, **kwargs) -> GafimeStreamer:
    return GafimeStreamer(*args, **kwargs)


def benchmark_streaming(file_path, batch_size=None, n_batches=5):
    streamer = GafimeStreamer(file_path)
    count = 0
    rows = 0
    for batch in streamer.stream(batch_size=batch_size):
        rows += len(batch)
        count += 1
        if count >= n_batches:
            break
    return {"batches": count, "rows": rows}
