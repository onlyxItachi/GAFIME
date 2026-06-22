from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .metrics import MetricSuite
from .native_data import NativeMatrix, NativeVector


@dataclass(frozen=True)
class DecisionPathCandidate:
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


def decision_path_candidate_from_record(record) -> DecisionPathCandidate:
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


def evaluate_decision_path_candidate(
    X: NativeMatrix,
    candidate: DecisionPathCandidate,
) -> List[float]:
    _validate_candidate(candidate)
    out: List[float] = []
    for row in range(X.n_samples):
        active = True
        for feature, threshold, sign in zip(candidate.features, candidate.thresholds, candidate.signs):
            value = X.value(row, int(feature))
            if int(sign) < 0:
                active = value <= float(threshold)
            else:
                active = value > float(threshold)
            if not active:
                break
        out.append(1.0 if active else 0.0)
    return out


def score_decision_path_candidates(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Iterable[DecisionPathCandidate],
    metric_suite: MetricSuite,
) -> Dict[DecisionPathCandidate, Dict[str, float]]:
    scores: Dict[DecisionPathCandidate, Dict[str, float]] = {}
    for candidate in candidates:
        scores[candidate] = metric_suite.score(
            evaluate_decision_path_candidate(X, candidate),
            y,
        )
    return scores


def describe_decision_path_candidate(
    candidate: DecisionPathCandidate,
    feature_names: Sequence[str],
) -> str:
    parts: List[str] = []
    for feature, threshold, sign in zip(candidate.features, candidate.thresholds, candidate.signs):
        op = "<=" if int(sign) < 0 else ">"
        parts.append(f"{feature_names[int(feature)]} {op} {_fmt(threshold)}")
    return "decision_path(" + " AND ".join(parts) + ")"


def decision_path_feature_names(
    candidate: DecisionPathCandidate,
    feature_names: Sequence[str],
) -> Tuple[str, ...]:
    return tuple(feature_names[idx] for idx in candidate.combo)


def decision_path_candidate_from_result(result) -> DecisionPathCandidate:
    params = getattr(result, "params", {}) or {}
    if getattr(result, "family", "") != "decision_path":
        raise ValueError("InteractionResult is not a decision_path candidate.")
    return DecisionPathCandidate(
        features=tuple(int(value) for value in params["features"]),
        thresholds=tuple(float(value) for value in params["thresholds"]),
        signs=tuple(int(value) for value in params["signs"]),
        gain=float(params.get("gain", 0.0)),
        support=float(params.get("support", 0.0)),
        round_id=int(params.get("round_id", 0)),
        native_candidate_id=int(params.get("native_candidate_id", 0)),
        candidate_id=str(params.get("candidate_id", getattr(result, "candidate_id", ""))),
    )


def _validate_candidate(candidate: DecisionPathCandidate) -> None:
    if not candidate.features:
        raise ValueError("DecisionPathCandidate must contain at least one feature.")
    if not (
        len(candidate.features)
        == len(candidate.thresholds)
        == len(candidate.signs)
    ):
        raise ValueError("DecisionPathCandidate features, thresholds, and signs must have equal length.")


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"
