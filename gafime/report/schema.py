"""Output schema definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FeatureScore:
    feature_indices: List[int]
    pearson: float
    mutual_information: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_indices": self.feature_indices,
            "pearson": self.pearson,
            "mutual_information": self.mutual_information,
        }


@dataclass
class Report:
    status: str
    signal_detected: bool
    message: str
    unary_metrics: List[FeatureScore]
    interaction_metrics: List[FeatureScore]
    parameters: Dict[str, Any]
    validation: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "signal_detected": self.signal_detected,
            "message": self.message,
            "unary_metrics": [item.to_dict() for item in self.unary_metrics],
            "interaction_metrics": [item.to_dict() for item in self.interaction_metrics],
            "parameters": self.parameters,
            "validation": self.validation,
        }
