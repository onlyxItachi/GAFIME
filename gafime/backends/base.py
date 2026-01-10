from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import ComputeBudget, EngineConfig
from ..metrics import MetricSuite
from ..metrics import cpu_metrics
from ..utils import arrays


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: str
    is_gpu: bool
    memory_total_mb: Optional[int]
    memory_free_mb: Optional[int]


class Backend:
    name = "numpy"
    device_label = "cpu"
    is_gpu = False

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self.xp = np
        self.metrics_ops = cpu_metrics

    def metric_suite(self, config: EngineConfig) -> MetricSuite:
        return MetricSuite(
            config.metric_names,
            mi_bins=config.mi_bins,
            xp=self.xp,
            ops=self.metrics_ops,
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            device=self.device_label,
            is_gpu=self.is_gpu,
            memory_total_mb=None,
            memory_free_mb=None,
        )

    def check_budget(
        self,
        X: np.ndarray,
        y: np.ndarray,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        return True, []

    def to_device(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array)

    def to_host(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array)

    def build_interaction_vector(self, X: np.ndarray, combo: Tuple[int, ...]):
        return arrays.build_interaction_vector(X, combo, xp=self.xp)

    def score_combos(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for combo in combos:
            vector = self.build_interaction_vector(X, combo)
            scores[combo] = metric_suite.score(vector, y)
        return scores

    def sample_indices(self, n_samples: int, rng: np.random.Generator):
        return rng.integers(0, n_samples, size=n_samples)

    def permute(self, y, rng: np.random.Generator):
        indices = rng.permutation(y.shape[0])
        return y[indices]

    @staticmethod
    def estimate_bytes(X: np.ndarray, y: np.ndarray) -> int:
        return int((X.nbytes + y.nbytes) * 1.2)
