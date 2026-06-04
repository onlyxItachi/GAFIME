from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Tuple

from ..backends.base import BackendInfo
from ..config import EngineConfig


@dataclass(frozen=True)
class InteractionResult:
    combo: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    metrics: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class StabilityResult:
    combo: Tuple[int, ...]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class PermutationResult:
    combo: Tuple[int, ...]
    p_values: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class Decision:
    signal_detected: bool
    message: str


@dataclass
class DiagnosticReport:
    config: EngineConfig
    feature_names: List[str]
    interactions: List[InteractionResult] = field(default_factory=list)
    stability: List[StabilityResult] = field(default_factory=list)
    permutations: List[PermutationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    decision: Decision | None = None
    backend: BackendInfo | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": _jsonable(self.config),
            "feature_names": list(self.feature_names),
            "interactions": [_jsonable(item) for item in self.interactions],
            "stability": [_jsonable(item) for item in self.stability],
            "permutations": [_jsonable(item) for item in self.permutations],
            "warnings": list(self.warnings),
            "decision": _jsonable(self.decision),
            "backend": _jsonable(self.backend),
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
