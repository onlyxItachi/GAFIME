"""Relationship metrics for GAFIME."""

from __future__ import annotations

import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with defensive checks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def mutual_information_binned(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 16,
    eps: float = 1e-12,
) -> float:
    """Compute mutual information using fixed-width binning."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0

    bins = max(2, int(bins))
    x_hist, x_edges = np.histogram(x, bins=bins)
    y_hist, y_edges = np.histogram(y, bins=bins)

    joint_hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))

    joint_prob = joint_hist / np.sum(joint_hist)
    x_prob = x_hist / np.sum(x_hist)
    y_prob = y_hist / np.sum(y_hist)

    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            p_xy = joint_prob[i, j]
            if p_xy <= 0:
                continue
            p_x = x_prob[i]
            p_y = y_prob[j]
            denom = max(p_x * p_y, eps)
            mi += p_xy * np.log(p_xy / denom)
    return float(mi)
