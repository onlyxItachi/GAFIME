from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import ComputeBudget, EngineConfig
from ..metrics import MetricSuite
from ..metrics import cpu_metrics
from ..native_data import NativeMatrix, NativeVector, bootstrap_indices


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: str
    is_gpu: bool
    memory_total_mb: Optional[int]
    memory_free_mb: Optional[int]


class Backend:
    name = "native"
    device_label = "cpu"
    is_gpu = False

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self.metrics_ops = cpu_metrics

    def metric_suite(self, config: EngineConfig) -> MetricSuite:
        return MetricSuite(
            config.metric_names,
            mi_bins=config.mi_bins,
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
        X: NativeMatrix,
        y: NativeVector,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        return True, []

    def to_device(self, data):
        return data

    def to_host(self, data):
        return data

    def build_interaction_vector(self, X: NativeMatrix, combo: Tuple[int, ...]):
        raise NotImplementedError("Backend must implement build_interaction_vector.")

    def score_combos(
        self,
        X: NativeMatrix,
        y: NativeVector,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        raise NotImplementedError("Backend must implement native score_combos.")

    def compile_session(
        self,
        X: NativeMatrix,
        y: NativeVector,
        scenario_plan: Any,
        metric_suite: MetricSuite,
        flags: Any,
    ):
        from ..compile.sessions import BackendSession

        return BackendSession(self, X, y, scenario_plan, metric_suite, flags)

    def score_discrete_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        metric_suite: MetricSuite,
    ) -> Dict[object, Dict[str, float]]:
        from ..discrete import score_discrete_candidates

        return score_discrete_candidates(X, y, candidates, metric_suite)

    def score_discrete_selection_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        *,
        baseline_pred=None,
        mi_bins: int = 96,
    ) -> Dict[object, Dict[str, float]]:
        from ..discrete import score_discrete_selection_candidates

        return score_discrete_selection_candidates(
            X,
            y,
            candidates,
            baseline_pred=baseline_pred,
            mi_bins=mi_bins,
        )

    def find_decision_path_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        *,
        feature_ids: Iterable[int] | None,
        max_depth: int,
        max_paths: int,
        max_bins_per_feature: int,
        min_leaf: int,
        rounds: int,
        learning_rate: float,
    ) -> List[object]:
        from .core_backend import CoreBackend

        return CoreBackend().find_decision_path_candidates(
            X,
            y,
            feature_ids=feature_ids,
            max_depth=max_depth,
            max_paths=max_paths,
            max_bins_per_feature=max_bins_per_feature,
            min_leaf=min_leaf,
            rounds=rounds,
            learning_rate=learning_rate,
        )

    def score_decision_path_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        metric_suite: MetricSuite,
    ) -> Dict[object, Dict[str, float]]:
        from ..decision_path import score_decision_path_candidates

        return score_decision_path_candidates(X, y, candidates, metric_suite)

    def score_time_series_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        metric_suite: MetricSuite,
    ) -> Dict[object, Dict[str, float]]:
        from ..time_series import score_time_series_candidates

        return score_time_series_candidates(X, y, candidates, metric_suite)

    def sample_indices(self, n_samples: int, rng: random.Random) -> List[int]:
        return bootstrap_indices(n_samples, rng)

    def permute(self, y: NativeVector, rng: random.Random) -> NativeVector:
        return y.shuffled(rng)

    @staticmethod
    def estimate_bytes(X: NativeMatrix, y: NativeVector) -> int:
        return int((X.nbytes + y.nbytes) * 1.2)
