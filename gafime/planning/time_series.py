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
    rust_result = _rust_time_series_candidates(selected, config)
    if rust_result is not None:
        return rust_result

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


def _rust_time_series_candidates(
    selected_features: List[int],
    config: EngineConfig,
) -> Tuple[List[TimeSeriesCandidate], List[str]] | None:
    try:
        from .. import subfunctions
    except ImportError:
        return None
    builder_type = getattr(subfunctions, "CompilePlanBuilder", None)
    if builder_type is None:
        return None
    rows, warnings = builder_type().time_series_candidates(
        [int(feature) for feature in selected_features],
        [int(lag) for lag in config.time_series_lags],
        [int(window) for window in config.time_series_windows],
        int(config.budget.max_time_series_candidates),
    )
    candidates = [
        TimeSeriesCandidate(
            kind=str(row["kind"]),
            feature_index=int(row["feature_index"]),
            lag=int(row["lag"]),
            window=int(row["window"]),
            candidate_id=str(row["candidate_id"]),
        )
        for row in rows
    ]
    return candidates, [str(item) for item in warnings]
