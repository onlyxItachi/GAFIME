"""Optimizer compatibility namespace for GAFIME v0.4.5.

The old standalone optimizer modules were removed from the native-only spine.
Feature planning now lives in ``GafimeEngine`` and native backend scheduling.
"""

from .adaptive import AdaptiveOptimizer
from .ensemble_search import CandidateGenerator, EnsembleSearch, FeatureRecipe
from .orchestrator import Orchestrator

__all__ = ["AdaptiveOptimizer", "CandidateGenerator", "EnsembleSearch", "FeatureRecipe", "Orchestrator"]
