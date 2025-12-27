"""Relationship metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    pearson: float
    mutual_information: float


def _safe_std(x: np.ndarray) -> float:
    std = float(np.std(x))
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    return std


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape != y.shape:
        raise ValueError("pearson_corr requires x and y with identical shapes")
    std_x = _safe_std(x)
    std_y = _safe_std(y)
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    cov = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    return float(cov / (std_x * std_y))


def mutual_information_binned(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 16,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape != y.shape:
        raise ValueError("mutual_information_binned requires x and y with identical shapes")
    if bins <= 1:
        raise ValueError("bins must be > 1")

    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    if hist_2d.sum() == 0:
        return 0.0

    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        px_py = px[:, None] * py[None, :]
        nonzero = pxy > 0
        mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    return float(mi)


def score_features(
    features: Iterable[np.ndarray],
    target: np.ndarray,
    bins: int = 16,
) -> list[MetricResult]:
    results: list[MetricResult] = []
    for feature in features:
        pearson = pearson_corr(feature, target)
        mi = mutual_information_binned(feature, target, bins=bins)
        results.append(MetricResult(pearson=pearson, mutual_information=mi))
    return results
