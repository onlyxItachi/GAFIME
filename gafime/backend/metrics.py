"""Relationship metrics for GAFIME (CPU)."""
from __future__ import annotations

import numpy as np


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation defensively.

    Returns 0.0 when variance is zero or inputs are invalid.
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if _safe_std(x) == 0.0 or _safe_std(y) == 0.0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return float(corr)


def mutual_information_binned(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 16,
) -> float:
    """Compute binned mutual information between x and y.

    Uses simple histogram binning. Returns 0.0 for degenerate inputs.
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if _safe_std(x) == 0.0 or _safe_std(y) == 0.0:
        return 0.0

    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    if np.sum(hist_xy) == 0:
        return 0.0
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            p_ij = p_xy[i, j]
            if p_ij <= 0.0:
                continue
            denom = p_x[i] * p_y[j]
            if denom <= 0.0:
                continue
            mi += p_ij * np.log(p_ij / denom)
    if np.isnan(mi) or np.isinf(mi):
        return 0.0
    return float(mi)
