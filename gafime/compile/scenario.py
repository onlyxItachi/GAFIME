from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ScenarioPlan:
    """Compact compile-time scenario metadata.

    The first checkpoint keeps this intentionally small. Later checkpoints add
    native descriptors and chunk offsets while retaining this public shape.
    """

    n_samples: int
    n_features: int
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def empty(cls, n_samples: int, n_features: int) -> "ScenarioPlan":
        return cls(n_samples=int(n_samples), n_features=int(n_features))
