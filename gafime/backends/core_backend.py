from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ..metrics import MetricSuite
from .base import Backend


class CoreBackend(Backend):
    name = "core"
    device_label = "cpu"
    is_gpu = False

    def __init__(self) -> None:
        import gafime_core

        super().__init__()
        self.core = gafime_core

    def score_combos(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        combos_list = list(combos)
        if not combos_list:
            return {}

        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        indices, offsets = self.core.pack_combos(combos_list)
        metrics = self.core.score_combos(
            X_arr,
            y_arr,
            indices,
            offsets,
            metric_suite.metric_names,
            metric_suite.mi_bins,
        )
        metric_names = metric_suite.metric_names
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for combo, row in zip(combos_list, metrics):
            scores[combo] = {name: float(row[i]) for i, name in enumerate(metric_names)}
        return scores
