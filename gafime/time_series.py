from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .metrics import MetricSuite
from .native_data import NativeMatrix, NativeVector


TIME_SERIES_KIND_CODES = {
    "time_series_lag": 1,
    "time_series_delta": 2,
    "time_series_velocity": 3,
    "time_series_acceleration": 4,
    "time_series_rolling_mean": 5,
    "time_series_rolling_std": 6,
    "time_series_rolling_sum": 7,
}


@dataclass(frozen=True)
class TimeSeriesCandidate:
    kind: str
    feature_index: int
    lag: int = 1
    window: int = 1
    candidate_id: str = ""

    @property
    def combo(self) -> Tuple[int, ...]:
        return (int(self.feature_index),)

    def params(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "feature_index": self.feature_index,
            "lag": self.lag,
            "window": self.window,
            "candidate_id": self.candidate_id,
        }


def evaluate_time_series_candidate(X: NativeMatrix, candidate: TimeSeriesCandidate) -> List[float]:
    values = X.column(candidate.feature_index)
    lag = max(1, int(candidate.lag))
    window = max(1, int(candidate.window))
    out: List[float] = []
    for idx, value in enumerate(values):
        lag_idx = max(idx - lag, 0)
        if candidate.kind == "time_series_lag":
            out.append(values[lag_idx])
        elif candidate.kind == "time_series_delta":
            out.append(value - values[lag_idx])
        elif candidate.kind == "time_series_velocity":
            out.append((value - values[lag_idx]) / float(lag))
        elif candidate.kind == "time_series_acceleration":
            lag2_idx = max(idx - 2 * lag, 0)
            out.append((value - 2.0 * values[lag_idx] + values[lag2_idx]) / float(lag * lag))
        elif candidate.kind in (
            "time_series_rolling_mean",
            "time_series_rolling_std",
            "time_series_rolling_sum",
        ):
            start = max(0, idx - window + 1)
            local = values[start : idx + 1]
            total = math.fsum(local)
            if candidate.kind == "time_series_rolling_sum":
                out.append(total)
            elif candidate.kind == "time_series_rolling_mean":
                out.append(total / float(len(local)))
            else:
                local_mean = total / float(len(local))
                variance = math.fsum((x - local_mean) * (x - local_mean) for x in local) / float(len(local))
                out.append(math.sqrt(max(variance, 0.0)))
        else:
            raise ValueError(f"Unsupported time-series candidate kind: {candidate.kind}")
    return out


def score_time_series_candidates(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Iterable[TimeSeriesCandidate],
    metric_suite: MetricSuite,
) -> Dict[TimeSeriesCandidate, Dict[str, float]]:
    scores: Dict[TimeSeriesCandidate, Dict[str, float]] = {}
    for candidate in candidates:
        scores[candidate] = metric_suite.score(evaluate_time_series_candidate(X, candidate), y)
    return scores


def describe_time_series_candidate(candidate: TimeSeriesCandidate, feature_names: List[str]) -> str:
    name = feature_names[candidate.feature_index]
    if candidate.kind in ("time_series_lag", "time_series_delta", "time_series_velocity", "time_series_acceleration"):
        return f"{candidate.kind}({name}, lag={candidate.lag})"
    return f"{candidate.kind}({name}, window={candidate.window})"
