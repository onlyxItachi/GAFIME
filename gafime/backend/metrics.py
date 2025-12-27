"""Relationship metrics (Pearson correlation and mutual information)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    value: float
    is_valid: bool
    reason: str | None = None


def _to_1d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


def pearson_corr(x: np.ndarray, y: np.ndarray) -> MetricResult:
    """Compute Pearson correlation defensively."""
    x_arr = _to_1d(x)
    y_arr = _to_1d(y)
    if x_arr.size != y_arr.size:
        return MetricResult(value=float("nan"), is_valid=False, reason="length_mismatch")
    if x_arr.size < 2:
        return MetricResult(value=float("nan"), is_valid=False, reason="insufficient_samples")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return MetricResult(value=0.0, is_valid=False, reason="constant_input")
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    if np.isnan(corr):
        return MetricResult(value=float("nan"), is_valid=False, reason="nan_result")
    return MetricResult(value=float(corr), is_valid=True)


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if np.isclose(min_val, max_val):
        return np.array([min_val, max_val], dtype=float)
    return np.linspace(min_val, max_val, bins + 1)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 10,
) -> MetricResult:
    """Compute binned mutual information defensively."""
    x_arr = _to_1d(x)
    y_arr = _to_1d(y)
    if x_arr.size != y_arr.size:
        return MetricResult(value=float("nan"), is_valid=False, reason="length_mismatch")
    if x_arr.size < 2:
        return MetricResult(value=float("nan"), is_valid=False, reason="insufficient_samples")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return MetricResult(value=0.0, is_valid=False, reason="constant_input")

    bins = max(2, int(bins))
    x_edges = _bin_edges(x_arr, bins)
    y_edges = _bin_edges(y_arr, bins)
    if x_edges.size < 2 or y_edges.size < 2:
        return MetricResult(value=0.0, is_valid=False, reason="degenerate_bins")

    hist, _, _ = np.histogram2d(x_arr, y_arr, bins=(x_edges, y_edges))
    total = np.sum(hist)
    if total <= 0:
        return MetricResult(value=0.0, is_valid=False, reason="empty_histogram")

    joint = hist / total
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px @ py)
        log_term = np.where(joint > 0, np.log(ratio), 0.0)
    mi = float(np.sum(joint * log_term))
    if np.isnan(mi) or np.isinf(mi):
        return MetricResult(value=float("nan"), is_valid=False, reason="nan_result")
    return MetricResult(value=mi, is_valid=True)


def summarize_metrics(pearson: MetricResult, mi: MetricResult) -> Tuple[float, float]:
    """Return numeric values (invalid values become 0.0)."""
    pearson_value = pearson.value if pearson.is_valid else 0.0
    mi_value = mi.value if mi.is_valid else 0.0
    return pearson_value, mi_value
