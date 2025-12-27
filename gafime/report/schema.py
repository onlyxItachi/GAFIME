"""Report schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class FeatureScore:
    name: str
    pearson: float
    mutual_information: float


@dataclass(frozen=True)
class InteractionScore:
    features: List[str]
    pearson: float
    mutual_information: float


@dataclass(frozen=True)
class Summary:
    diagnosis: str
    max_abs_pearson: float
    max_mutual_information: float
    permutation_pvalue: float | None = None


@dataclass(frozen=True)
class Report:
    summary: Summary
    feature_scores: List[FeatureScore]
    interaction_scores: List[InteractionScore] = field(default_factory=list)
    config: dict = field(default_factory=dict)
