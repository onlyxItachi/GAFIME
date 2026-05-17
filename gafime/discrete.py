from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics.cpu_metrics import pearson_corr
from .metrics import MetricSuite
from .utils.cache_ordering import order_items_cache_aware


GPU_HARD_MODE_ERROR = "GPU feature engineering with discrete hard mode is not supported!"

DISCRETE_FUNCTION_KIND_CODES = {
    "discrete_function_soft_threshold": 0,
    "discrete_function_soft_interval": 1,
    "discrete_function_value_gated_threshold": 2,
    "discrete_function_soft_rectangle": 3,
    "discrete_function_value_in_soft_rectangle": 4,
}


@dataclass(frozen=True)
class DiscreteFunctionCandidate:
    kind: str
    feature_indices: Tuple[int, ...]
    thresholds: Tuple[float, ...] = ()
    intervals: Tuple[Tuple[float, float], ...] = ()
    direction: str = "ge"
    value_feature: Optional[int] = None
    scales: Tuple[float, ...] = ()
    sharpness: float = 12.0
    mode: str = "soft"
    candidate_id: str = ""

    @property
    def combo(self) -> Tuple[int, ...]:
        seen: List[int] = []
        if self.value_feature is not None:
            seen.append(int(self.value_feature))
        for idx in self.feature_indices:
            idx_int = int(idx)
            if idx_int not in seen:
                seen.append(idx_int)
        return tuple(seen)

    def params(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "feature_indices": self.feature_indices,
            "thresholds": self.thresholds,
            "intervals": self.intervals,
            "direction": self.direction,
            "value_feature": self.value_feature,
            "scales": self.scales,
            "sharpness": self.sharpness,
            "mode": self.mode,
            "candidate_id": self.candidate_id,
        }


def discrete_function_soft_threshold(
    x,
    threshold: float,
    sharpness: float = 12.0,
    direction: str = "ge",
    scale: float | None = None,
    xp=np,
):
    scale_value = _scale_or_one(scale)
    z = sharpness * (x - threshold) / scale_value
    if direction == "le":
        z = -z
    return _sigmoid(z, xp=xp)


def discrete_function_soft_interval(
    x,
    low: float,
    high: float,
    sharpness: float = 12.0,
    scale: float | None = None,
    xp=np,
):
    scale_value = _scale_or_one(scale)
    left = _sigmoid(sharpness * (x - low) / scale_value, xp=xp)
    right = _sigmoid(sharpness * (high - x) / scale_value, xp=xp)
    return left * right


def discrete_function_value_gated_threshold(
    value,
    gate,
    threshold: float,
    sharpness: float = 12.0,
    direction: str = "ge",
    scale: float | None = None,
    xp=np,
):
    return value * discrete_function_soft_threshold(
        gate,
        threshold=threshold,
        sharpness=sharpness,
        direction=direction,
        scale=scale,
        xp=xp,
    )


def discrete_function_soft_rectangle(
    x0,
    x1,
    low0: float,
    high0: float,
    low1: float,
    high1: float,
    sharpness: float = 12.0,
    scale0: float | None = None,
    scale1: float | None = None,
    xp=np,
):
    mask0 = discrete_function_soft_interval(
        x0,
        low=low0,
        high=high0,
        sharpness=sharpness,
        scale=scale0,
        xp=xp,
    )
    mask1 = discrete_function_soft_interval(
        x1,
        low=low1,
        high=high1,
        sharpness=sharpness,
        scale=scale1,
        xp=xp,
    )
    return mask0 * mask1


def discrete_function_value_in_soft_rectangle(
    value,
    x0,
    x1,
    low0: float,
    high0: float,
    low1: float,
    high1: float,
    sharpness: float = 12.0,
    scale0: float | None = None,
    scale1: float | None = None,
    xp=np,
):
    return value * discrete_function_soft_rectangle(
        x0,
        x1,
        low0=low0,
        high0=high0,
        low1=low1,
        high1=high1,
        sharpness=sharpness,
        scale0=scale0,
        scale1=scale1,
        xp=xp,
    )


def evaluate_discrete_candidate(X, candidate: DiscreteFunctionCandidate, xp=np):
    if candidate.mode == "hard":
        return _evaluate_hard_candidate(X, candidate, xp=xp)
    if candidate.mode != "soft":
        raise ValueError("discrete_mode must be 'soft' or 'hard'.")

    if candidate.kind == "discrete_function_soft_threshold":
        feature = candidate.feature_indices[0]
        return discrete_function_soft_threshold(
            X[:, feature],
            threshold=candidate.thresholds[0],
            sharpness=candidate.sharpness,
            direction=candidate.direction,
            scale=_scale_at(candidate, 0),
            xp=xp,
        )
    if candidate.kind == "discrete_function_soft_interval":
        feature = candidate.feature_indices[0]
        low, high = candidate.intervals[0]
        return discrete_function_soft_interval(
            X[:, feature],
            low=low,
            high=high,
            sharpness=candidate.sharpness,
            scale=_scale_at(candidate, 0),
            xp=xp,
        )
    if candidate.kind == "discrete_function_value_gated_threshold":
        value_feature = _value_feature(candidate)
        gate_feature = candidate.feature_indices[0]
        return discrete_function_value_gated_threshold(
            X[:, value_feature],
            X[:, gate_feature],
            threshold=candidate.thresholds[0],
            sharpness=candidate.sharpness,
            direction=candidate.direction,
            scale=_scale_at(candidate, 0),
            xp=xp,
        )
    if candidate.kind == "discrete_function_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return discrete_function_soft_rectangle(
            X[:, feature_a],
            X[:, feature_b],
            low0=low_a,
            high0=high_a,
            low1=low_b,
            high1=high_b,
            sharpness=candidate.sharpness,
            scale0=_scale_at(candidate, 0),
            scale1=_scale_at(candidate, 1),
            xp=xp,
        )
    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        value_feature = _value_feature(candidate)
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return discrete_function_value_in_soft_rectangle(
            X[:, value_feature],
            X[:, feature_a],
            X[:, feature_b],
            low0=low_a,
            high0=high_a,
            low1=low_b,
            high1=high_b,
            sharpness=candidate.sharpness,
            scale0=_scale_at(candidate, 0),
            scale1=_scale_at(candidate, 1),
            xp=xp,
        )
    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def evaluate_discrete_mask(X, candidate: DiscreteFunctionCandidate, xp=np):
    """Return the split/membership mask used by a discrete candidate.

    Value-gated candidates are ranked by the gate they introduce, not by the
    raw value-multiplied output alone. This keeps split ranking aligned with
    impurity and residual-gain objectives.
    """
    if candidate.kind in (
        "discrete_function_soft_threshold",
        "discrete_function_soft_interval",
        "discrete_function_soft_rectangle",
    ):
        return evaluate_discrete_candidate(X, candidate, xp=xp)

    if candidate.kind == "discrete_function_value_gated_threshold":
        gate_feature = candidate.feature_indices[0]
        gate_candidate = DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(gate_feature,),
            thresholds=candidate.thresholds,
            direction=candidate.direction,
            scales=candidate.scales[:1],
            sharpness=candidate.sharpness,
            mode=candidate.mode,
        )
        return evaluate_discrete_candidate(X, gate_candidate, xp=xp)

    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        gate_candidate = DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(feature_a, feature_b),
            intervals=candidate.intervals[:2],
            scales=candidate.scales[:2],
            sharpness=candidate.sharpness,
            mode=candidate.mode,
        )
        return evaluate_discrete_candidate(X, gate_candidate, xp=xp)

    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def score_discrete_candidates(
    X,
    y,
    candidates: Iterable[DiscreteFunctionCandidate],
    metric_suite: MetricSuite,
) -> Dict[DiscreteFunctionCandidate, Dict[str, float]]:
    scores: Dict[DiscreteFunctionCandidate, Dict[str, float]] = {}
    for candidate in candidates:
        vector = evaluate_discrete_candidate(X, candidate, xp=metric_suite.xp)
        scores[candidate] = metric_suite.score(vector, y)
    return scores


def score_discrete_selection_candidate(
    X,
    y,
    candidate: DiscreteFunctionCandidate,
    *,
    baseline_pred=None,
    mi_bins: int = 16,
) -> Dict[str, float]:
    """Score a discrete candidate for feature selection.

    These are internal ranking diagnostics, independent from
    ``EngineConfig.metric_names``. User-selected metrics still control the
    public report values; this selector is deliberately not Pearson-only.
    """
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    feature = np.asarray(evaluate_discrete_candidate(X_np, candidate, xp=np), dtype=np.float64)
    mask = np.asarray(evaluate_discrete_mask(X_np, candidate, xp=np), dtype=np.float64)
    residual = _residual(y_np, baseline_pred)

    residual_corr = abs(float(pearson_corr(feature, residual, xp=np)))
    residual_r2 = residual_corr * residual_corr
    return {
        "mutual_info": _mask_mutual_info(mask, y_np, bins=mi_bins),
        "variance_reduction": _soft_variance_reduction(y_np, mask),
        "residual_abs_corr": residual_corr,
        "residual_r2_gain": residual_r2,
    }


def score_discrete_selection_candidates(
    X,
    y,
    candidates: Iterable[DiscreteFunctionCandidate],
    *,
    baseline_pred=None,
    mi_bins: int = 16,
) -> Dict[DiscreteFunctionCandidate, Dict[str, float]]:
    return {
        candidate: score_discrete_selection_candidate(
            X,
            y,
            candidate,
            baseline_pred=baseline_pred,
            mi_bins=mi_bins,
        )
        for candidate in candidates
    }


def rank_discrete_selection_scores(
    scores: Dict[DiscreteFunctionCandidate, Dict[str, float]],
    *,
    weights: Dict[str, float] | None = None,
) -> Dict[DiscreteFunctionCandidate, float]:
    """Combine split-aware selector components without using Pearson."""
    if not scores:
        return {}
    weights = weights or {
        "mutual_info": 0.30,
        "variance_reduction": 0.35,
        "residual_r2_gain": 0.35,
    }
    maxima: Dict[str, float] = {}
    for name in weights:
        maxima[name] = max(abs(values.get(name, 0.0)) for values in scores.values())

    ranked: Dict[DiscreteFunctionCandidate, float] = {}
    for candidate, values in scores.items():
        total = 0.0
        for name, weight in weights.items():
            denom = maxima.get(name, 0.0)
            if denom > 0:
                total += weight * max(values.get(name, 0.0), 0.0) / denom
        ranked[candidate] = float(total)
    return ranked


def order_discrete_candidates_cache_aware(
    candidates: Iterable[DiscreteFunctionCandidate],
    max_blocks: int = 1024,
) -> List[DiscreteFunctionCandidate]:
    candidates_list = list(candidates)
    if not candidates_list:
        return []
    feature_sets = [candidate.combo for candidate in candidates_list]
    template_ids = [
        DISCRETE_FUNCTION_KIND_CODES.get(candidate.kind, 0)
        for candidate in candidates_list
    ]
    return order_items_cache_aware(
        candidates_list,
        feature_sets,
        template_ids=template_ids,
        max_blocks=max_blocks,
    )


def discrete_candidate_from_result(result) -> DiscreteFunctionCandidate:
    params = getattr(result, "params", {}) or {}
    if getattr(result, "family", "") != "discrete_function":
        raise ValueError("InteractionResult is not a discrete_function candidate.")
    return DiscreteFunctionCandidate(
        kind=str(params["kind"]),
        feature_indices=tuple(int(idx) for idx in params["feature_indices"]),
        thresholds=tuple(float(value) for value in params.get("thresholds", ())),
        intervals=tuple(
            (float(low), float(high))
            for low, high in params.get("intervals", ())
        ),
        direction=str(params.get("direction", "ge")),
        value_feature=(
            None
            if params.get("value_feature") is None
            else int(params.get("value_feature"))
        ),
        scales=tuple(float(value) for value in params.get("scales", ())),
        sharpness=float(params.get("sharpness", 12.0)),
        mode=str(params.get("mode", "soft")),
        candidate_id=str(params.get("candidate_id", getattr(result, "candidate_id", ""))),
    )


def describe_discrete_candidate(
    candidate: DiscreteFunctionCandidate,
    feature_names: List[str],
) -> str:
    def name(idx: int) -> str:
        return feature_names[int(idx)]

    if candidate.kind == "discrete_function_soft_threshold":
        op = ">=" if candidate.direction == "ge" else "<="
        return f"{candidate.kind}({name(candidate.feature_indices[0])} {op} {_fmt(candidate.thresholds[0])})"
    if candidate.kind == "discrete_function_soft_interval":
        low, high = candidate.intervals[0]
        return f"{candidate.kind}({name(candidate.feature_indices[0])} in [{_fmt(low)}, {_fmt(high)}])"
    if candidate.kind == "discrete_function_value_gated_threshold":
        op = ">=" if candidate.direction == "ge" else "<="
        value_feature = _value_feature(candidate)
        gate_feature = candidate.feature_indices[0]
        return (
            f"{candidate.kind}({name(value_feature)} * "
            f"mask({name(gate_feature)} {op} {_fmt(candidate.thresholds[0])}))"
        )
    if candidate.kind == "discrete_function_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return (
            f"{candidate.kind}({name(feature_a)} in [{_fmt(low_a)}, {_fmt(high_a)}], "
            f"{name(feature_b)} in [{_fmt(low_b)}, {_fmt(high_b)}])"
        )
    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        value_feature = _value_feature(candidate)
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return (
            f"{candidate.kind}({name(value_feature)} * rect("
            f"{name(feature_a)} in [{_fmt(low_a)}, {_fmt(high_a)}], "
            f"{name(feature_b)} in [{_fmt(low_b)}, {_fmt(high_b)}]))"
        )
    return candidate.kind


def discrete_feature_names(
    candidate: DiscreteFunctionCandidate,
    feature_names: List[str],
) -> Tuple[str, ...]:
    return tuple(feature_names[idx] for idx in candidate.combo)


def _evaluate_hard_candidate(X, candidate: DiscreteFunctionCandidate, xp=np):
    if candidate.kind == "discrete_function_soft_threshold":
        feature = candidate.feature_indices[0]
        return _hard_threshold(
            X[:, feature],
            threshold=candidate.thresholds[0],
            direction=candidate.direction,
            xp=xp,
        )
    if candidate.kind == "discrete_function_soft_interval":
        feature = candidate.feature_indices[0]
        low, high = candidate.intervals[0]
        return _hard_interval(X[:, feature], low=low, high=high, xp=xp)
    if candidate.kind == "discrete_function_value_gated_threshold":
        value_feature = _value_feature(candidate)
        gate_feature = candidate.feature_indices[0]
        return X[:, value_feature] * _hard_threshold(
            X[:, gate_feature],
            threshold=candidate.thresholds[0],
            direction=candidate.direction,
            xp=xp,
        )
    if candidate.kind == "discrete_function_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return _hard_interval(X[:, feature_a], low_a, high_a, xp=xp) * _hard_interval(
            X[:, feature_b], low_b, high_b, xp=xp
        )
    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        value_feature = _value_feature(candidate)
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        mask = _hard_interval(X[:, feature_a], low_a, high_a, xp=xp) * _hard_interval(
            X[:, feature_b], low_b, high_b, xp=xp
        )
        return X[:, value_feature] * mask
    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def _hard_threshold(x, threshold: float, direction: str, xp=np):
    if direction == "le":
        return xp.asarray(x <= threshold, dtype=float)
    return xp.asarray(x >= threshold, dtype=float)


def _hard_interval(x, low: float, high: float, xp=np):
    return xp.asarray((x >= low) & (x <= high), dtype=float)


def _soft_variance_reduction(y: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.clip(np.asarray(mask, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if y.shape[0] == 0 or w.shape[0] != y.shape[0]:
        return 0.0

    total_sse = _weighted_sse(y, np.ones_like(w))
    if total_sse <= 1e-12:
        return 0.0

    left_weight = float(np.sum(w))
    right = 1.0 - w
    right_weight = float(np.sum(right))
    if left_weight <= 1e-9 or right_weight <= 1e-9:
        return 0.0

    split_sse = _weighted_sse(y, w) + _weighted_sse(y, right)
    gain = (total_sse - split_sse) / total_sse
    return float(max(gain, 0.0))


def _mask_mutual_info(mask: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    bins = int(max(2, min(bins, 16)))
    mask = np.clip(np.asarray(mask, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if mask.shape[0] == 0 or y.shape[0] != mask.shape[0]:
        return 0.0
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max <= y_min:
        return 0.0

    hist, _, _ = np.histogram2d(
        mask,
        y,
        bins=[
            np.linspace(0.0, 1.0, bins + 1),
            np.linspace(y_min, y_max, bins + 1),
        ],
    )
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0

    pxy = hist / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    denom = px @ py
    valid = (pxy > 0.0) & (denom > 0.0)
    return float(np.sum(pxy[valid] * np.log(pxy[valid] / denom[valid])))


def _weighted_sse(y: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-12:
        return 0.0
    mean = float(np.sum(weights * y) / total_weight)
    centered = y - mean
    return float(np.sum(weights * centered * centered))


def _residual(y: np.ndarray, baseline_pred) -> np.ndarray:
    if baseline_pred is None:
        return y - float(np.mean(y))
    pred = np.asarray(baseline_pred, dtype=np.float64).reshape(-1)
    if pred.shape[0] != y.shape[0]:
        raise ValueError("baseline_pred must have the same length as y.")
    return y - pred


def _sigmoid(z, xp=np):
    z_clipped = xp.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + xp.exp(-z_clipped))


def _scale_or_one(scale: float | None) -> float:
    if scale is None or scale <= 0:
        return 1.0
    return float(scale)


def _scale_at(candidate: DiscreteFunctionCandidate, index: int) -> float:
    if index >= len(candidate.scales):
        return 1.0
    return _scale_or_one(candidate.scales[index])


def _value_feature(candidate: DiscreteFunctionCandidate) -> int:
    if candidate.value_feature is None:
        return candidate.feature_indices[0]
    return int(candidate.value_feature)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"
