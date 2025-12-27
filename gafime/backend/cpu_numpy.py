"""CPU computation backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import mutual_information_binned, pearson_correlation


@dataclass
class UnaryMetricResult:
    feature_index: int
    pearson: float
    mutual_information: float


class CpuBackend:
    """Numpy-based backend for unary metrics."""

    def __init__(self, mi_bins: int = 16) -> None:
        self.mi_bins = mi_bins

    def unary_metrics(self, x: np.ndarray, y: np.ndarray) -> list[UnaryMetricResult]:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        results: list[UnaryMetricResult] = []
        for idx in range(x.shape[1]):
            feature = x[:, idx]
            results.append(
                UnaryMetricResult(
                    feature_index=idx,
                    pearson=pearson_correlation(feature, y),
                    mutual_information=mutual_information_binned(
                        feature, y, bins=self.mi_bins
                    ),
                )
            )
        return results

    @staticmethod
    def sort_by_signal(results: Iterable[UnaryMetricResult]) -> list[UnaryMetricResult]:
        return sorted(
            results,
            key=lambda item: (abs(item.pearson) + item.mutual_information),
            reverse=True,
        )
