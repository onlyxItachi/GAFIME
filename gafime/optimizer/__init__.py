"""
GAFIME Optimizer Module

Contains the EnsembleSearchEngine for fast feature interaction discovery.
"""

from .optimizer import (
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
