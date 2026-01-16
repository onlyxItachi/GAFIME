"""
GAFIME Data I/O - VRAM-Aware Streaming Module

Implements the "Chunking" and "Data(SSD)" blocks from the architecture:
    Data(SSD) -> Chunking -> VRAM Offload -> Custom Kernel

Design Philosophy: "Static VRAM Bucket"
- Pre-calculate optimal batch size based on available VRAM
- Stream exact-sized chunks that fit in pre-allocated CUDA buffers
- Avoid dynamic cudaMalloc fragmentation

Target Hardware: RTX 4060 (8GB VRAM), 16GB System RAM
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

logger = logging.getLogger(__name__)


class GafimeStreamer:
    """
    VRAM-aware data streamer for GAFIME feature interaction mining.
    
    Uses Polars lazy API to stream data from disk in precise chunks
    that fit exactly into pre-allocated CUDA buffers.
    
    Example:
        streamer = GafimeStreamer("data.parquet")
        for X_chunk in streamer.stream():
            # X_chunk is contiguous float32, ready for cudaMemcpy
            result = cuda_kernel(X_chunk)
    """
    
    # RTX 4060 specs
    DEFAULT_VRAM_GB = 6.0  # Conservative default (leaving headroom from 8GB)
    VRAM_HEADROOM = 0.20   # 20% safety margin
    BYTES_PER_FLOAT32 = 4
    
    def __init__(
        self,
        file_path: Union[str, Path],
        target_cols: Optional[List[str]] = None,
        y_col: Optional[str] = None,
    ) -> None:
        """
        Initialize the streamer with lazy schema detection.
        
        Args:
            file_path: Path to .csv or .parquet file
            target_cols: Optional list of feature columns (None = all except y_col)
            y_col: Optional target column name (excluded from features)
        """
        if not HAS_POLARS:
            raise ImportError(
                "Polars is required for GafimeStreamer. "
                "Install with: pip install polars"
            )
        
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        self.target_cols = target_cols
        self.y_col = y_col
        
        # Lazy schema detection (no data loaded yet)
        self._lazy_df = self._create_lazy_reader()
        self._schema = self._lazy_df.collect_schema()
        self._all_columns = list(self._schema.names())
        
        # Determine feature columns
        if target_cols is not None:
            self._feature_cols = list(target_cols)
        else:
            self._feature_cols = [c for c in self._all_columns if c != y_col]
        
        self.n_features = len(self._feature_cols)
        
        # Cache total rows (lazy count)
        self._total_rows: Optional[int] = None
        
        logger.debug(
            f"GafimeStreamer initialized: {self.file_path.name}, "
            f"{self.n_features} features, y_col={y_col}"
        )
    
    def _create_lazy_reader(self) -> "pl.LazyFrame":
        """Create Polars lazy reader based on file extension."""
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".parquet":
            return pl.scan_parquet(self.file_path)
        elif suffix == ".csv":
            return pl.scan_csv(self.file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Use .csv or .parquet"
            )
    
    @property
    def total_rows(self) -> int:
        """Get total row count (cached after first access)."""
        if self._total_rows is None:
            self._total_rows = self._lazy_df.select(pl.len()).collect().item()
        return self._total_rows
    
    def estimate_optimal_batch_size(
        self,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
        include_output: bool = True,
        n_combos: int = 256,
    ) -> int:
        """
        Calculate maximum rows that fit in the VRAM bucket.
        
        Formula:
            usable_vram = vram_budget * (1 - headroom)
            bytes_per_row = n_features * 4  (for X)
            bytes_per_row += n_combos * 4   (for output, if included)
            max_rows = usable_vram / bytes_per_row
        
        Args:
            vram_budget_gb: Available VRAM in GB (default: 6.0 for RTX 4060)
            include_output: Account for output matrix in VRAM
            n_combos: Expected number of feature combinations
        
        Returns:
            Optimal batch size (number of rows)
        """
        usable_bytes = vram_budget_gb * (1024 ** 3) * (1 - self.VRAM_HEADROOM)
        
        # Input X: n_samples x n_features x float32
        bytes_per_row = self.n_features * self.BYTES_PER_FLOAT32
        
        # Output: n_samples x n_combos x float32
        if include_output:
            bytes_per_row += n_combos * self.BYTES_PER_FLOAT32
        
        # Means vector (small, but include for accuracy)
        means_bytes = self.n_features * self.BYTES_PER_FLOAT32
        usable_bytes -= means_bytes
        
        max_rows = int(usable_bytes / bytes_per_row)
        
        # Round down to nice power-of-2 aligned size for GPU efficiency
        # But ensure at least 1024 rows
        aligned_rows = max(1024, (max_rows // 1024) * 1024)
        
        logger.info(
            f"📊 VRAM Budget: {vram_budget_gb:.1f}GB -> "
            f"Optimal batch: {aligned_rows:,} rows "
            f"({aligned_rows * bytes_per_row / (1024**2):.1f}MB)"
        )
        
        return aligned_rows
    
    def stream(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield contiguous float32 chunks ready for cudaMemcpy.
        
        This is the "precision pump" that fills the static VRAM bucket
        with exactly-sized batches.
        
        Args:
            batch_size: Fixed batch size (None = auto-calculate)
            vram_budget_gb: VRAM budget for auto-calculation
        
        Yields:
            np.ndarray: Contiguous float32 array [batch_size x n_features]
        """
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        
        total = self.total_rows
        logger.info(
            f"🚀 Streaming {total:,} rows in batches of {batch_size:,} "
            f"({(total + batch_size - 1) // batch_size} batches)"
        )
        
        # Select only feature columns (lazy)
        reader = self._lazy_df.select(self._feature_cols)
        
        current_row = 0
        batch_num = 0
        
        while current_row < total:
            # 1. Slice (Lazy) - No memory used yet
            remaining = total - current_row
            this_batch = min(batch_size, remaining)
            slice_lazy = reader.slice(current_row, this_batch)
            
            # 2. Materialize (Disk -> RAM)
            # This is the ONLY point where RAM is consumed
            df_chunk = slice_lazy.collect()
            
            # 3. Sanitize (JIT Converter for CUDA)
            np_chunk = self._sanitize_chunk(df_chunk)
            
            batch_num += 1
            logger.debug(
                f"Batch {batch_num}: rows {current_row:,}-{current_row + this_batch:,}, "
                f"shape={np_chunk.shape}, dtype={np_chunk.dtype}"
            )
            
            yield np_chunk
            
            current_row += this_batch
    
    def stream_with_target(
        self,
        batch_size: Optional[int] = None,
        vram_budget_gb: float = DEFAULT_VRAM_GB,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (X, y) tuples for supervised learning.
        
        Args:
            batch_size: Fixed batch size (None = auto-calculate)
            vram_budget_gb: VRAM budget for auto-calculation
        
        Yields:
            Tuple of (X, y) where X is [batch x features] and y is [batch]
        """
        if self.y_col is None:
            raise ValueError("y_col must be set to use stream_with_target()")
        
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(vram_budget_gb)
        
        total = self.total_rows
        logger.info(
            f"🚀 Streaming {total:,} rows with target in batches of {batch_size:,}"
        )
        
        # Select feature columns + target
        all_cols = self._feature_cols + [self.y_col]
        reader = self._lazy_df.select(all_cols)
        
        current_row = 0
        
        while current_row < total:
            remaining = total - current_row
            this_batch = min(batch_size, remaining)
            slice_lazy = reader.slice(current_row, this_batch)
            
            df_chunk = slice_lazy.collect()
            
            # Split into X and y
            X_df = df_chunk.select(self._feature_cols)
            y_series = df_chunk.get_column(self.y_col)
            
            X_chunk = self._sanitize_chunk(X_df)
            y_chunk = self._sanitize_vector(y_series)
            
            yield X_chunk, y_chunk
            
            current_row += this_batch
    
    def _sanitize_chunk(self, df: "pl.DataFrame") -> np.ndarray:
        """
        Convert Polars DataFrame to CUDA-ready NumPy array.
        
        The "Sanitization Pipeline":
        1. Polars -> NumPy (zero-copy if possible)
        2. Typecast to float32 (MANDATORY for RTX 4060)
        3. Ensure C-contiguous layout (linear memory for cudaMemcpy)
        """
        # Convert to NumPy (Polars may use zero-copy for some dtypes)
        np_array = df.to_numpy()
        
        # MANDATORY: Cast to float32 (RTX 4060 hates FP64)
        if np_array.dtype != np.float32:
            # Try copy=False first, but numpy may need to copy for dtype change
            np_array = np_array.astype(np.float32, copy=False)
        
        # CRITICAL: Ensure C-contiguous layout
        # Strided arrays will copy garbage data to CUDA
        if not np_array.flags['C_CONTIGUOUS']:
            np_array = np.ascontiguousarray(np_array)
        
        return np_array
    
    def _sanitize_vector(self, series: "pl.Series") -> np.ndarray:
        """Convert Polars Series to contiguous float32 vector."""
        np_vec = series.to_numpy()
        
        if np_vec.dtype != np.float32:
            np_vec = np_vec.astype(np.float32, copy=False)
        
        if not np_vec.flags['C_CONTIGUOUS']:
            np_vec = np.ascontiguousarray(np_vec)
        
        return np_vec


def benchmark_streaming(
    file_path: str,
    batch_size: Optional[int] = None,
    n_batches: int = 5,
) -> dict:
    """
    Benchmark streaming performance and compare with full load.
    
    Args:
        file_path: Path to data file
        batch_size: Batch size (None = auto)
        n_batches: Number of batches to benchmark
    
    Returns:
        Dict with timing results
    """
    import time
    
    streamer = GafimeStreamer(file_path)
    
    if batch_size is None:
        batch_size = streamer.estimate_optimal_batch_size()
    
    results = {
        "file": str(file_path),
        "total_rows": streamer.total_rows,
        "n_features": streamer.n_features,
        "batch_size": batch_size,
        "batch_times_ms": [],
    }
    
    print(f"\n📊 Benchmarking streaming: {Path(file_path).name}")
    print(f"   Total rows: {streamer.total_rows:,}")
    print(f"   Features: {streamer.n_features}")
    print(f"   Batch size: {batch_size:,}")
    print()
    
    batch_times = []
    for i, chunk in enumerate(streamer.stream(batch_size)):
        if i >= n_batches:
            break
        
        start = time.perf_counter()
        
        # Simulate minimal processing (just memory access)
        _ = chunk.sum()
        
        elapsed = (time.perf_counter() - start) * 1000
        batch_times.append(elapsed)
        
        print(f"   Batch {i+1}: {chunk.shape} -> {elapsed:.2f}ms")
    
    results["batch_times_ms"] = batch_times
    results["avg_batch_ms"] = sum(batch_times) / len(batch_times) if batch_times else 0
    
    print(f"\n   Average: {results['avg_batch_ms']:.2f}ms per batch")
    
    return results


# Convenience function
def create_streamer(
    file_path: Union[str, Path],
    target_cols: Optional[List[str]] = None,
    y_col: Optional[str] = None,
) -> GafimeStreamer:
    """Create a VRAM-aware data streamer."""
    return GafimeStreamer(file_path, target_cols, y_col)
