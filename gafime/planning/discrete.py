from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

from ..config import EngineConfig
from ..discrete import DiscreteFunctionCandidate
from ..native_data import NativeMatrix, column_std, quantiles as native_quantiles
from .combinations import select_top_features


def plan_discrete_candidates(
    X: NativeMatrix,
    feature_scores: Dict[int, float],
    config: EngineConfig,
) -> Tuple[List[DiscreteFunctionCandidate], List[str]]:
    warnings: List[str] = []
    if not config.enable_discrete_functions:
        return [], warnings
    if config.discrete_threshold_source != "quantile":
        raise ValueError("discrete_threshold_source must be 'quantile' in the current Engine release.")
    if config.discrete_mode not in ("soft", "hard"):
        raise ValueError("discrete_mode must be 'soft' or 'hard'.")
    if config.discrete_gate_sharpness <= 0:
        raise ValueError("discrete_gate_sharpness must be > 0.")

    budget = config.budget
    if budget.max_discrete_candidates < 1:
        warnings.append("max_discrete_candidates < 1; discrete functions disabled.")
        return [], warnings
    if budget.max_thresholds_per_feature < 1:
        warnings.append("max_thresholds_per_feature < 1; discrete functions disabled.")
        return [], warnings
    if budget.top_k_features_for_discrete < 1:
        warnings.append("top_k_features_for_discrete < 1; discrete functions disabled.")
        return [], warnings

    n_features = X.shape[1]
    selected_features = _select_discrete_features(
        n_features=n_features,
        feature_scores=feature_scores,
        top_k=budget.top_k_features_for_discrete,
    )
    thresholds_by_feature, scales_by_feature = _thresholds_by_feature(
        X=X,
        feature_indices=selected_features,
        quantiles=config.discrete_quantiles,
        max_thresholds=budget.max_thresholds_per_feature,
    )
    intervals_by_feature = {
        idx: _intervals_from_thresholds(thresholds, budget.max_intervals_per_feature)
        for idx, thresholds in thresholds_by_feature.items()
    }
    feature_pairs = _rank_feature_pairs(
        selected_features,
        feature_scores,
        budget.max_feature_pairs_for_rectangles,
    )

    rust_result = _rust_discrete_candidates(
        selected_features=selected_features,
        thresholds_by_feature=thresholds_by_feature,
        intervals_by_feature=intervals_by_feature,
        scales_by_feature=scales_by_feature,
        feature_pairs=feature_pairs,
        config=config,
    )
    if rust_result is not None:
        return rust_result

    candidates: List[DiscreteFunctionCandidate] = []

    def add(candidate: DiscreteFunctionCandidate) -> bool:
        if len(candidates) >= budget.max_discrete_candidates:
            return False
        candidates.append(candidate)
        return True

    for feature in selected_features:
        thresholds = thresholds_by_feature.get(feature, ())
        scale = scales_by_feature.get(feature, 1.0)
        for threshold in thresholds:
            for direction in ("ge", "le"):
                if not add(
                    _candidate(
                        kind="discrete_function_soft_threshold",
                        feature_indices=(feature,),
                        thresholds=(threshold,),
                        direction=direction,
                        scales=(scale,),
                        config=config,
                    )
                ):
                    warnings.append("Discrete candidates capped by max_discrete_candidates.")
                    return candidates, warnings
                if not add(
                    _candidate(
                        kind="discrete_function_value_gated_threshold",
                        feature_indices=(feature,),
                        thresholds=(threshold,),
                        direction=direction,
                        value_feature=feature,
                        scales=(scale,),
                        config=config,
                    )
                ):
                    warnings.append("Discrete candidates capped by max_discrete_candidates.")
                    return candidates, warnings

        for interval in intervals_by_feature.get(feature, ()):
            if not add(
                _candidate(
                    kind="discrete_function_soft_interval",
                    feature_indices=(feature,),
                    intervals=(interval,),
                    scales=(scale,),
                    config=config,
                )
            ):
                warnings.append("Discrete candidates capped by max_discrete_candidates.")
                return candidates, warnings

    for feature_a, feature_b in feature_pairs:
        intervals_a = intervals_by_feature.get(feature_a, ())
        intervals_b = intervals_by_feature.get(feature_b, ())
        if not intervals_a or not intervals_b:
            continue
        scale_a = scales_by_feature.get(feature_a, 1.0)
        scale_b = scales_by_feature.get(feature_b, 1.0)
        for interval_a, interval_b in itertools.product(intervals_a, intervals_b):
            for kind, value_feature in (
                ("discrete_function_soft_rectangle", None),
                ("discrete_function_value_in_soft_rectangle", feature_a),
                ("discrete_function_value_in_soft_rectangle", feature_b),
            ):
                if not add(
                    _candidate(
                        kind=kind,
                        feature_indices=(feature_a, feature_b),
                        intervals=(interval_a, interval_b),
                        value_feature=value_feature,
                        scales=(scale_a, scale_b),
                        config=config,
                    )
                ):
                    warnings.append("Discrete candidates capped by max_discrete_candidates.")
                    return candidates, warnings

    return candidates, warnings


def _rust_discrete_candidates(
    *,
    selected_features: Sequence[int],
    thresholds_by_feature: Dict[int, Tuple[float, ...]],
    intervals_by_feature: Dict[int, Tuple[Tuple[float, float], ...]],
    scales_by_feature: Dict[int, float],
    feature_pairs: Sequence[Tuple[int, int]],
    config: EngineConfig,
) -> Tuple[List[DiscreteFunctionCandidate], List[str]] | None:
    try:
        from .. import subfunctions
    except ImportError:
        return None
    builder_type = getattr(subfunctions, "CompilePlanBuilder", None)
    if builder_type is None:
        return None

    selected = [int(feature) for feature in selected_features]
    builder = builder_type()
    discrete_candidates = getattr(builder, "discrete_candidates", None)
    if discrete_candidates is None:
        return None

    rows, warnings = discrete_candidates(
        selected,
        [
            [float(threshold) for threshold in thresholds_by_feature.get(feature, ())]
            for feature in selected
        ],
        [
            [
                (float(low), float(high))
                for low, high in intervals_by_feature.get(feature, ())
            ]
            for feature in selected
        ],
        [float(scales_by_feature.get(feature, 1.0)) for feature in selected],
        [(int(feature_a), int(feature_b)) for feature_a, feature_b in feature_pairs],
        int(config.budget.max_discrete_candidates),
    )
    candidates = [
        _candidate(
            kind=str(row["kind"]),
            feature_indices=_parse_int_tuple(row.get("feature_indices", "")),
            thresholds=_parse_float_tuple(row.get("thresholds", "")),
            intervals=_parse_intervals(row.get("intervals", "")),
            direction=str(row.get("direction", "ge")),
            value_feature=_parse_optional_int(row.get("value_feature", "")),
            scales=_parse_float_tuple(row.get("scales", "")),
            config=config,
        )
        for row in rows
    ]
    return candidates, [str(item) for item in warnings]


def _select_discrete_features(
    n_features: int,
    feature_scores: Dict[int, float],
    top_k: int,
) -> List[int]:
    if feature_scores:
        return select_top_features(feature_scores, min(top_k, n_features))
    return list(range(min(top_k, n_features)))


def _thresholds_by_feature(
    X: NativeMatrix,
    feature_indices: Sequence[int],
    quantiles: Sequence[float],
    max_thresholds: int,
) -> Tuple[Dict[int, Tuple[float, ...]], Dict[int, float]]:
    thresholds: Dict[int, Tuple[float, ...]] = {}
    scales: Dict[int, float] = {}
    q_values = [float(q) for q in quantiles if 0.0 < float(q) < 1.0]
    if max_thresholds > 0:
        q_values = q_values[:max_thresholds]

    for feature in feature_indices:
        col = X.column(feature)
        raw = native_quantiles(col, q_values) if q_values else []
        unique = _unique_sorted(raw)
        thresholds[feature] = tuple(unique)
        scale = float(column_std(X, feature))
        scales[feature] = scale if scale > 1e-12 else 1.0
    return thresholds, scales


def _intervals_from_thresholds(
    thresholds: Sequence[float],
    max_intervals: int,
) -> Tuple[Tuple[float, float], ...]:
    if max_intervals < 1:
        return ()
    values = [float(v) for v in thresholds]
    intervals: List[Tuple[float, float]] = []
    for width in range(1, len(values)):
        for start in range(0, len(values) - width):
            low = values[start]
            high = values[start + width]
            if high > low:
                intervals.append((low, high))
            if len(intervals) >= max_intervals:
                return tuple(intervals)
    return tuple(intervals)


def _rank_feature_pairs(
    feature_indices: Sequence[int],
    feature_scores: Dict[int, float],
    max_pairs: int,
) -> List[Tuple[int, int]]:
    if max_pairs < 1:
        return []
    pairs = list(itertools.combinations(feature_indices, 2))
    pairs.sort(
        key=lambda pair: (
            feature_scores.get(pair[0], 0.0) + feature_scores.get(pair[1], 0.0),
            feature_scores.get(pair[0], 0.0),
            feature_scores.get(pair[1], 0.0),
        ),
        reverse=True,
    )
    return pairs[:max_pairs]


def _candidate(
    *,
    kind: str,
    feature_indices: Tuple[int, ...],
    config: EngineConfig,
    thresholds: Tuple[float, ...] = (),
    intervals: Tuple[Tuple[float, float], ...] = (),
    direction: str = "ge",
    value_feature: int | None = None,
    scales: Tuple[float, ...] = (),
) -> DiscreteFunctionCandidate:
    candidate_id = _candidate_id(
        kind=kind,
        feature_indices=feature_indices,
        thresholds=thresholds,
        intervals=intervals,
        direction=direction,
        value_feature=value_feature,
        mode=config.discrete_mode,
    )
    return DiscreteFunctionCandidate(
        kind=kind,
        feature_indices=feature_indices,
        thresholds=thresholds,
        intervals=intervals,
        direction=direction,
        value_feature=value_feature,
        scales=scales,
        sharpness=config.discrete_gate_sharpness,
        mode=config.discrete_mode,
        candidate_id=candidate_id,
    )


def _candidate_id(
    *,
    kind: str,
    feature_indices: Tuple[int, ...],
    thresholds: Tuple[float, ...],
    intervals: Tuple[Tuple[float, float], ...],
    direction: str,
    value_feature: int | None,
    mode: str,
) -> str:
    pieces = [
        kind,
        f"mode={mode}",
        "features=" + ",".join(str(idx) for idx in feature_indices),
    ]
    if value_feature is not None:
        pieces.append(f"value={value_feature}")
    if thresholds:
        pieces.append("thresholds=" + ",".join(_fmt(value) for value in thresholds))
        pieces.append(f"direction={direction}")
    if intervals:
        interval_text = ";".join(f"{_fmt(low)}:{_fmt(high)}" for low, high in intervals)
        pieces.append(f"intervals={interval_text}")
    return "|".join(pieces)


def _unique_sorted(values: Sequence[float]) -> List[float]:
    unique: List[float] = []
    for value in sorted(float(v) for v in values):
        if not unique or abs(value - unique[-1]) > 1e-12:
            unique.append(value)
    return unique


def _parse_int_tuple(value: object) -> Tuple[int, ...]:
    text = str(value)
    if not text:
        return ()
    return tuple(int(item) for item in text.split(",") if item)


def _parse_float_tuple(value: object) -> Tuple[float, ...]:
    text = str(value)
    if not text:
        return ()
    return tuple(float(item) for item in text.split(",") if item)


def _parse_intervals(value: object) -> Tuple[Tuple[float, float], ...]:
    text = str(value)
    if not text:
        return ()
    intervals: List[Tuple[float, float]] = []
    for item in text.split(";"):
        if not item:
            continue
        low_text, high_text = item.split(":", 1)
        intervals.append((float(low_text), float(high_text)))
    return tuple(intervals)


def _parse_optional_int(value: object) -> int | None:
    text = str(value)
    return None if not text else int(text)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"
