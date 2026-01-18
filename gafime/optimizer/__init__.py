"""
GAFIME Optimizer Module

Contains the EnsembleSearchEngine for fast feature interaction discovery.
"""

from .ensemble_search import (
    EnsembleSearchEngine,
    SearchConfig,
    FeatureCandidate,
    quick_search,
)

__all__ = [
    "EnsembleSearchEngine",
    "SearchConfig", 
    "FeatureCandidate",
    "quick_search",
]
