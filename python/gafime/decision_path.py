from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class DecisionPathCandidate:
    """Portable description of the v0.5 decision-path candidate surface."""

    features: Tuple[int, ...]
    thresholds: Tuple[float, ...]
    signs: Tuple[int, ...]
    gain: float = 0.0
    support: float = 0.0
    round_id: int = 0
    native_candidate_id: int = 0
    candidate_id: str = ""

    @property
    def combo(self) -> Tuple[int, ...]:
        seen: List[int] = []
        for feature in self.features:
            feature_int = int(feature)
            if feature_int not in seen:
                seen.append(feature_int)
        return tuple(sorted(seen))

    def params(self) -> Dict[str, object]:
        return {
            "kind": "decision_path",
            "features": self.features,
            "thresholds": self.thresholds,
            "signs": self.signs,
            "gain": float(self.gain),
            "support": float(self.support),
            "round_id": int(self.round_id),
            "native_candidate_id": int(self.native_candidate_id),
            "candidate_id": self.candidate_id,
        }


def describe_decision_path_candidate(
    candidate: DecisionPathCandidate, feature_names: Sequence[str]
) -> str:
    parts: List[str] = []
    for feature, threshold, sign in zip(
        candidate.features, candidate.thresholds, candidate.signs
    ):
        operator = "<=" if int(sign) < 0 else ">"
        parts.append(f"{feature_names[int(feature)]} {operator} {float(threshold):.6g}")
    return "decision_path(" + " AND ".join(parts) + ")"


def decision_path_candidate_from_record(record: object) -> DecisionPathCandidate:
    native_id = int(getattr(record, "candidate_id"))
    return DecisionPathCandidate(
        features=tuple(int(value) for value in getattr(record, "features")),
        thresholds=tuple(float(value) for value in getattr(record, "thresholds")),
        signs=tuple(int(value) for value in getattr(record, "signs")),
        gain=float(getattr(record, "gain")),
        support=float(getattr(record, "support")),
        round_id=int(getattr(record, "round_id")),
        native_candidate_id=native_id,
        candidate_id=f"decision_path:{native_id}",
    )


def decision_path_candidate_from_result(result: object) -> DecisionPathCandidate:
    params = getattr(result, "params", {}) or {}
    if getattr(result, "family", "") != "decision_path":
        raise ValueError("InteractionResult is not a decision_path candidate.")

    required_params = ("features", "thresholds", "signs")
    supplied_params = tuple(name for name in required_params if name in params)
    if supplied_params and len(supplied_params) != len(required_params):
        missing = ", ".join(name for name in required_params if name not in params)
        raise ValueError(f"Incomplete decision_path params; missing {missing}.")

    if supplied_params:
        combo_names = tuple(str(value) for value in getattr(result, "feature_names", ()))
        if any(name.startswith("path[") for name in combo_names) and len(combo_names) != 1:
            raise ValueError(
                "A decision_path interaction that combines a generated path with "
                "another feature is not a standalone DecisionPathCandidate."
            )
        features = tuple(int(value) for value in params["features"])
        thresholds = tuple(float(value) for value in params["thresholds"])
        signs = tuple(int(value) for value in params["signs"])
    else:
        features = _native_membership_features(result)
        if not features:
            raise ValueError(
                "A decision_path result without params must contain a feature combo."
            )
        # Native v1 reports expose each discovered path as a binary feature. The
        # equivalent portable predicate is therefore membership > 0.5.
        thresholds = (0.5,) * len(features)
        signs = (1,) * len(features)

    candidate_id = str(params.get("candidate_id", getattr(result, "candidate_id", "")))
    return DecisionPathCandidate(
        features=features,
        thresholds=thresholds,
        signs=signs,
        gain=float(params.get("gain", 0.0)),
        support=float(params.get("support", 0.0)),
        round_id=int(params.get("round_id", 0)),
        native_candidate_id=int(
            params.get("native_candidate_id", _candidate_id_suffix(candidate_id))
        ),
        candidate_id=candidate_id,
    )


def evaluate_decision_path_candidate(
    X: object, candidate: DecisionPathCandidate
) -> List[float]:
    _validate_candidate(candidate)
    n_samples = int(getattr(X, "n_samples", len(X) if hasattr(X, "__len__") else 0))
    out: List[float] = []
    for row in range(n_samples):
        active = True
        undetermined = False
        for feature, threshold, sign in zip(
            candidate.features, candidate.thresholds, candidate.signs
        ):
            value = (
                X.value(row, int(feature))
                if hasattr(X, "value")
                else X[row][int(feature)]
            )
            if math.isnan(float(value)):
                undetermined = True
                continue
            active = value <= float(threshold) if int(sign) < 0 else value > float(threshold)
            if not active:
                break
        out.append(float("nan") if active and undetermined else (1.0 if active else 0.0))
    return out


def score_decision_path_candidates(
    X: object,
    y: object,
    candidates: Iterable[DecisionPathCandidate],
    metric_suite: Any,
) -> Dict[DecisionPathCandidate, Dict[str, float]]:
    return {
        candidate: metric_suite.score(evaluate_decision_path_candidate(X, candidate), y)
        for candidate in candidates
    }


def decision_path_feature_names(
    candidate: DecisionPathCandidate, feature_names: Sequence[str]
) -> Tuple[str, ...]:
    return tuple(feature_names[index] for index in candidate.combo)


def _native_membership_features(result: object) -> Tuple[int, ...]:
    combo = tuple(int(value) for value in getattr(result, "combo", ()))
    names = tuple(str(value) for value in getattr(result, "feature_names", ()))
    if len(combo) != 1 or len(names) != 1 or any(
        not name.startswith("path[") for name in names
    ):
        raise ValueError(
            "A params-free decision_path result must contain exactly one generated "
            "path-membership feature."
        )
    return combo


def _candidate_id_suffix(candidate_id: str) -> int:
    try:
        return int(candidate_id.rsplit(":", 1)[-1])
    except ValueError:
        return 0


def _validate_candidate(candidate: DecisionPathCandidate) -> None:
    if not candidate.features:
        raise ValueError("DecisionPathCandidate must contain at least one feature.")
    if not (
        len(candidate.features) == len(candidate.thresholds) == len(candidate.signs)
    ):
        raise ValueError(
            "DecisionPathCandidate features, thresholds, and signs must have equal length."
        )


__all__ = [
    "DecisionPathCandidate",
    "decision_path_candidate_from_record",
    "decision_path_candidate_from_result",
    "decision_path_feature_names",
    "describe_decision_path_candidate",
    "evaluate_decision_path_candidate",
    "score_decision_path_candidates",
]
