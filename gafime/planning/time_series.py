from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import EngineConfig
from ..time_series import TimeSeriesCandidate
from .combinations import select_top_features


def plan_time_series_candidates(
    n_features: int,
    feature_scores: Dict[int, float],
    config: EngineConfig,
) -> Tuple[List[TimeSeriesCandidate], List[str]]:
    warnings: List[str] = []
    if not config.enable_time_series_functions:
        return [], warnings
    budget = config.budget
    if budget.max_time_series_candidates < 1:
        warnings.append("max_time_series_candidates < 1; time-series functions disabled.")
        return [], warnings
    top_k = max(0, min(budget.top_k_features_for_time_series, n_features))
    selected = select_top_features(feature_scores, top_k) if feature_scores else list(range(top_k))
    candidates: List[TimeSeriesCandidate] = []

    def add(kind: str, feature: int, lag: int = 1, window: int = 1) -> bool:
        if len(candidates) >= budget.max_time_series_candidates:
            return False
        candidate_id = f"{kind}|feature={feature}|lag={lag}|window={window}"
        candidates.append(
            TimeSeriesCandidate(
                kind=kind,
                feature_index=int(feature),
                lag=int(lag),
                window=int(window),
                candidate_id=candidate_id,
            )
        )
        return True

    for feature in selected:
        for lag in config.time_series_lags:
            for kind in (
                "time_series_lag",
                "time_series_delta",
                "time_series_velocity",
                "time_series_acceleration",
            ):
                if not add(kind, feature, lag=max(1, int(lag))):
                    warnings.append("Time-series candidates capped by max_time_series_candidates.")
                    return candidates, warnings
        for window in config.time_series_windows:
            for kind in (
                "time_series_rolling_mean",
                "time_series_rolling_std",
                "time_series_rolling_sum",
            ):
                if not add(kind, feature, window=max(1, int(window))):
                    warnings.append("Time-series candidates capped by max_time_series_candidates.")
                    return candidates, warnings
    return candidates, warnings
