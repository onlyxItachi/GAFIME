from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics.cpu_metrics import (
    adaptive_min_effective_support,
    pearson_corr,
    soft_binary_mutual_info,
    soft_mask_support,
)
from .metrics import MetricSuite
from .native_data import NativeMatrix, NativeVector, mean
from .utils.cache_ordering import batch_items_by_template_cache_aware, order_items_cache_aware


GPU_HARD_MODE_ERROR = "GPU feature engineering with discrete hard mode is not supported!"

DISCRETE_FUNCTION_KIND_CODES = {
    "discrete_function_soft_threshold": 0,
    "discrete_function_soft_interval": 1,
    "discrete_function_value_gated_threshold": 2,
    "discrete_function_soft_rectangle": 3,
    "discrete_function_value_in_soft_rectangle": 4,
}

DISCRETE_SELECTION_TEMPLATE_FACTOR = 1_000


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
):
    scale_value = _scale_or_one(scale)
    out: List[float] = []
    for value in _to_list(x):
        z = sharpness * (value - threshold) / scale_value
        if direction == "le":
            z = -z
        out.append(_sigmoid_scalar(z))
    return out


def discrete_function_soft_interval(
    x,
    low: float,
    high: float,
    sharpness: float = 12.0,
    scale: float | None = None,
):
    scale_value = _scale_or_one(scale)
    out: List[float] = []
    for value in _to_list(x):
        left = _sigmoid_scalar(sharpness * (value - low) / scale_value)
        right = _sigmoid_scalar(sharpness * (high - value) / scale_value)
        out.append(left * right)
    return out


def discrete_function_value_gated_threshold(
    value,
    gate,
    threshold: float,
    sharpness: float = 12.0,
    direction: str = "ge",
    scale: float | None = None,
):
    mask = discrete_function_soft_threshold(
        gate,
        threshold=threshold,
        sharpness=sharpness,
        direction=direction,
        scale=scale,
    )
    return [float(v) * m for v, m in zip(_to_list(value), mask)]


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
):
    mask0 = discrete_function_soft_interval(
        x0,
        low=low0,
        high=high0,
        sharpness=sharpness,
        scale=scale0,
    )
    mask1 = discrete_function_soft_interval(
        x1,
        low=low1,
        high=high1,
        sharpness=sharpness,
        scale=scale1,
    )
    return [a * b for a, b in zip(mask0, mask1)]


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
):
    mask = discrete_function_soft_rectangle(
        x0,
        x1,
        low0=low0,
        high0=high0,
        low1=low1,
        high1=high1,
        sharpness=sharpness,
        scale0=scale0,
        scale1=scale1,
    )
    return [float(v) * m for v, m in zip(_to_list(value), mask)]


def evaluate_discrete_candidate(X: NativeMatrix, candidate: DiscreteFunctionCandidate):
    if candidate.mode == "hard":
        return _evaluate_hard_candidate(X, candidate)
    if candidate.mode != "soft":
        raise ValueError("discrete_mode must be 'soft' or 'hard'.")

    if candidate.kind == "discrete_function_soft_threshold":
        feature = candidate.feature_indices[0]
        return discrete_function_soft_threshold(
            X.column(feature),
            threshold=candidate.thresholds[0],
            sharpness=candidate.sharpness,
            direction=candidate.direction,
            scale=_scale_at(candidate, 0),
        )
    if candidate.kind == "discrete_function_soft_interval":
        feature = candidate.feature_indices[0]
        low, high = candidate.intervals[0]
        return discrete_function_soft_interval(
            X.column(feature),
            low=low,
            high=high,
            sharpness=candidate.sharpness,
            scale=_scale_at(candidate, 0),
        )
    if candidate.kind == "discrete_function_value_gated_threshold":
        value_feature = _value_feature(candidate)
        gate_feature = candidate.feature_indices[0]
        return discrete_function_value_gated_threshold(
            X.column(value_feature),
            X.column(gate_feature),
            threshold=candidate.thresholds[0],
            sharpness=candidate.sharpness,
            direction=candidate.direction,
            scale=_scale_at(candidate, 0),
        )
    if candidate.kind == "discrete_function_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return discrete_function_soft_rectangle(
            X.column(feature_a),
            X.column(feature_b),
            low0=low_a,
            high0=high_a,
            low1=low_b,
            high1=high_b,
            sharpness=candidate.sharpness,
            scale0=_scale_at(candidate, 0),
            scale1=_scale_at(candidate, 1),
        )
    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        value_feature = _value_feature(candidate)
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        return discrete_function_value_in_soft_rectangle(
            X.column(value_feature),
            X.column(feature_a),
            X.column(feature_b),
            low0=low_a,
            high0=high_a,
            low1=low_b,
            high1=high_b,
            sharpness=candidate.sharpness,
            scale0=_scale_at(candidate, 0),
            scale1=_scale_at(candidate, 1),
        )
    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def evaluate_discrete_mask(X: NativeMatrix, candidate: DiscreteFunctionCandidate):
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
        return evaluate_discrete_candidate(X, candidate)

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
        return evaluate_discrete_candidate(X, gate_candidate)

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
        return evaluate_discrete_candidate(X, gate_candidate)

    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def score_discrete_candidates(
    X,
    y,
    candidates: Iterable[DiscreteFunctionCandidate],
    metric_suite: MetricSuite,
) -> Dict[DiscreteFunctionCandidate, Dict[str, float]]:
    scores: Dict[DiscreteFunctionCandidate, Dict[str, float]] = {}
    for candidate in candidates:
        vector = evaluate_discrete_candidate(X, candidate)
        scores[candidate] = metric_suite.score(vector, y)
    return scores


def score_discrete_selection_candidate(
    X,
    y,
    candidate: DiscreteFunctionCandidate,
    *,
    baseline_pred=None,
    mi_bins: int = 96,
) -> Dict[str, float]:
    """Score a discrete candidate for feature selection.

    These are internal ranking diagnostics, independent from
    ``EngineConfig.metric_names``. User-selected metrics still control the
    public report values; this selector is deliberately not Pearson-only.
    """
    y_values = y.to_list() if isinstance(y, NativeVector) else _to_list(y)
    feature = evaluate_discrete_candidate(X, candidate)
    mask = evaluate_discrete_mask(X, candidate)
    residual = _residual(y_values, baseline_pred)

    residual_corr = abs(float(pearson_corr(feature, residual)))
    if not math.isfinite(residual_corr):
        residual_corr = 0.0
    residual_corr = float(min(max(residual_corr, 0.0), 1.0))
    residual_r2 = residual_corr * residual_corr
    return {
        "mutual_info": _mask_mutual_info(mask, y_values, bins=mi_bins),
        "variance_reduction": _soft_variance_reduction(y_values, mask),
        "residual_abs_corr": residual_corr,
        "residual_r2_gain": residual_r2,
    }


def score_discrete_selection_candidates(
    X,
    y,
    candidates: Iterable[DiscreteFunctionCandidate],
    *,
    baseline_pred=None,
    mi_bins: int = 96,
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


def discrete_selection_template_id(
    candidate: DiscreteFunctionCandidate,
    mi_bin_template: int,
) -> int:
    return (
        int(mi_bin_template) * DISCRETE_SELECTION_TEMPLATE_FACTOR
        + DISCRETE_FUNCTION_KIND_CODES.get(candidate.kind, 0)
    )


def mi_bin_template_from_discrete_selection_template(template_id: int) -> int:
    return int(template_id) // DISCRETE_SELECTION_TEMPLATE_FACTOR


def batch_discrete_selection_candidates_cache_aware(
    candidates: Iterable[DiscreteFunctionCandidate],
    *,
    mi_bin_template: int,
    max_blocks: int = 1024,
) -> List[Tuple[int, List[DiscreteFunctionCandidate]]]:
    candidates_list = list(candidates)
    if not candidates_list:
        return []
    feature_sets = [candidate.combo for candidate in candidates_list]
    template_ids = [
        discrete_selection_template_id(candidate, mi_bin_template)
        for candidate in candidates_list
    ]
    return batch_items_by_template_cache_aware(
        candidates_list,
        feature_sets,
        template_ids,
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


def _evaluate_hard_candidate(X: NativeMatrix, candidate: DiscreteFunctionCandidate):
    if candidate.kind == "discrete_function_soft_threshold":
        feature = candidate.feature_indices[0]
        return _hard_threshold(
            X.column(feature),
            threshold=candidate.thresholds[0],
            direction=candidate.direction,
        )
    if candidate.kind == "discrete_function_soft_interval":
        feature = candidate.feature_indices[0]
        low, high = candidate.intervals[0]
        return _hard_interval(X.column(feature), low=low, high=high)
    if candidate.kind == "discrete_function_value_gated_threshold":
        value_feature = _value_feature(candidate)
        gate_feature = candidate.feature_indices[0]
        mask = _hard_threshold(
            X.column(gate_feature),
            threshold=candidate.thresholds[0],
            direction=candidate.direction,
        )
        return [value * weight for value, weight in zip(X.column(value_feature), mask)]
    if candidate.kind == "discrete_function_soft_rectangle":
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        left = _hard_interval(X.column(feature_a), low_a, high_a)
        right = _hard_interval(X.column(feature_b), low_b, high_b)
        return [a * b for a, b in zip(left, right)]
    if candidate.kind == "discrete_function_value_in_soft_rectangle":
        value_feature = _value_feature(candidate)
        feature_a, feature_b = candidate.feature_indices[:2]
        (low_a, high_a), (low_b, high_b) = candidate.intervals[:2]
        left = _hard_interval(X.column(feature_a), low_a, high_a)
        right = _hard_interval(X.column(feature_b), low_b, high_b)
        mask = [a * b for a, b in zip(left, right)]
        return [value * weight for value, weight in zip(X.column(value_feature), mask)]
    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def _hard_threshold(x, threshold: float, direction: str):
    if direction == "le":
        return [1.0 if value <= threshold else 0.0 for value in _to_list(x)]
    return [1.0 if value >= threshold else 0.0 for value in _to_list(x)]


def _hard_interval(x, low: float, high: float):
    return [1.0 if low <= value <= high else 0.0 for value in _to_list(x)]


def _soft_variance_reduction(y: Sequence[float], mask: Sequence[float]) -> float:
    y_values = _to_list(y)
    weights = [min(max(float(value), 0.0), 1.0) for value in mask]
    if not y_values or len(weights) != len(y_values):
        return 0.0
    support_floor = adaptive_min_effective_support(len(y_values))
    support = soft_mask_support(weights)
    if (
        support["effective_in"] < support_floor
        or support["effective_out"] < support_floor
    ):
        return 0.0

    total_sse = _weighted_sse(y_values, [1.0] * len(weights))
    if total_sse <= 1e-12:
        return 0.0

    left_weight = float(math.fsum(weights))
    right = [1.0 - value for value in weights]
    right_weight = float(math.fsum(right))
    if left_weight <= 1e-9 or right_weight <= 1e-9:
        return 0.0

    split_sse = _weighted_sse(y_values, weights) + _weighted_sse(y_values, right)
    gain = (total_sse - split_sse) / total_sse
    return float(max(gain, 0.0))


def _mask_mutual_info(mask: Sequence[float], y: Sequence[float], bins: int = 96) -> float:
    return soft_binary_mutual_info(
        mask,
        y,
        max_bins=bins,
        min_samples_per_target_bin=25,
    )


def _weighted_sse(y: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = float(math.fsum(weights))
    if total_weight <= 1e-12:
        return 0.0
    local_mean = float(math.fsum(weight * value for weight, value in zip(weights, y)) / total_weight)
    return float(math.fsum(weight * (value - local_mean) * (value - local_mean) for weight, value in zip(weights, y)))


def _residual(y: Sequence[float], baseline_pred) -> List[float]:
    if baseline_pred is None:
        y_mean = mean(y)
        return [value - y_mean for value in y]
    pred = _to_list(baseline_pred)
    if len(pred) != len(y):
        raise ValueError("baseline_pred must have the same length as y.")
    return [value - pred_value for value, pred_value in zip(y, pred)]


def _sigmoid_scalar(z: float) -> float:
    z_clipped = min(max(float(z), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-z_clipped))


def _to_list(values) -> List[float]:
    if isinstance(values, NativeVector):
        return values.to_list()
    return [float(value) for value in values]


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
