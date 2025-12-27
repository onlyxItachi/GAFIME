"""CuPy-based GPU backend."""

from __future__ import annotations

from typing import Any

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def is_available() -> bool:
    return cp is not None


def to_device(array: Any) -> Any:
    if cp is None:
        raise RuntimeError("CuPy is not available")
    return cp.asarray(array)


def pearson_correlation(x: Any, y: Any) -> float:
    if cp is None:
        raise RuntimeError("CuPy is not available")
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    x_std = cp.std(x)
    y_std = cp.std(y)
    if float(x_std) == 0.0 or float(y_std) == 0.0:
        return 0.0
    corr = cp.corrcoef(x, y)[0, 1]
    corr_val = float(corr)
    if corr_val != corr_val:
        return 0.0
    return corr_val


def mutual_information_binned(x: Any, y: Any, bins: int = 16, eps: float = 1e-12) -> float:
    if cp is None:
        raise RuntimeError("CuPy is not available")
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if float(cp.std(x)) == 0.0 or float(cp.std(y)) == 0.0:
        return 0.0

    bins = max(2, int(bins))
    x_hist, x_edges = cp.histogram(x, bins=bins)
    y_hist, y_edges = cp.histogram(y, bins=bins)
    joint_hist, _, _ = cp.histogram2d(x, y, bins=(x_edges, y_edges))

    joint_prob = joint_hist / cp.sum(joint_hist)
    x_prob = x_hist / cp.sum(x_hist)
    y_prob = y_hist / cp.sum(y_hist)

    mi = cp.float32(0.0)
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            p_xy = joint_prob[i, j]
            if float(p_xy) <= 0:
                continue
            p_x = x_prob[i]
            p_y = y_prob[j]
            denom = cp.maximum(p_x * p_y, eps)
            mi += p_xy * cp.log(p_xy / denom)
    return float(mi)
