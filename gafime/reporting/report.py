from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..backends.base import BackendInfo
from ..config import EngineConfig


@dataclass(frozen=True)
class InteractionResult:
    combo: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    metrics: Dict[str, float]


@dataclass(frozen=True)
class StabilityResult:
    combo: Tuple[int, ...]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]


@dataclass(frozen=True)
class PermutationResult:
    combo: Tuple[int, ...]
    p_values: Dict[str, float]


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
            "config": self.config,
            "feature_names": self.feature_names,
            "interactions": [item.__dict__ for item in self.interactions],
            "stability": [item.__dict__ for item in self.stability],
            "permutations": [item.__dict__ for item in self.permutations],
            "warnings": list(self.warnings),
            "decision": None if self.decision is None else self.decision.__dict__,
            "backend": None if self.backend is None else self.backend.__dict__,
        }
