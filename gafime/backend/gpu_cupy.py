"""GPU backend using CuPy."""
from __future__ import annotations

from dataclasses import dataclass

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class UnaryMetricResult:
    feature_index: int
    pearson: float
    mutual_information: float


def _safe_std(values: "cp.ndarray") -> float:
    std = float(cp.std(values))
    return std


def pearson_correlation(x: "cp.ndarray", y: "cp.ndarray") -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if _safe_std(x) == 0.0 or _safe_std(y) == 0.0:
        return 0.0
    corr = cp.corrcoef(x, y)[0, 1]
    corr_value = float(corr)
    if corr_value != corr_value:
        return 0.0
    return corr_value


def mutual_information_binned(x: "cp.ndarray", y: "cp.ndarray", bins: int = 16) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if _safe_std(x) == 0.0 or _safe_std(y) == 0.0:
        return 0.0

    hist_xy, _, _ = cp.histogram2d(x, y, bins=bins)
    total = float(cp.sum(hist_xy))
    if total == 0.0:
        return 0.0
    p_xy = hist_xy / total
    p_x = cp.sum(p_xy, axis=1)
    p_y = cp.sum(p_xy, axis=0)

    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            p_ij = float(p_xy[i, j])
            if p_ij <= 0.0:
                continue
            denom = float(p_x[i]) * float(p_y[j])
            if denom <= 0.0:
                continue
            mi += p_ij * float(cp.log(p_ij / denom))
    if mi != mi:
        return 0.0
    return float(mi)


class GpuBackend:
    """CuPy-based backend for unary metrics."""

    def __init__(self, mi_bins: int = 16) -> None:
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available")
        self.mi_bins = mi_bins

    def unary_metrics(self, x, y) -> list[UnaryMetricResult]:
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        if x_gpu.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y_gpu.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x_gpu.shape[0] != y_gpu.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        results: list[UnaryMetricResult] = []
        for idx in range(x_gpu.shape[1]):
            feature = x_gpu[:, idx]
            results.append(
                UnaryMetricResult(
                    feature_index=idx,
                    pearson=pearson_correlation(feature, y_gpu),
                    mutual_information=mutual_information_binned(
                        feature, y_gpu, bins=self.mi_bins
                    ),
                )
            )
        return results

    @staticmethod
    def sort_by_signal(results):
        return sorted(
            results,
            key=lambda item: (abs(item.pearson) + item.mutual_information),
            reverse=True,
        )
