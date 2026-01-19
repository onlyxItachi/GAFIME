"""
GAFIME Optimizer Module

Contains the EnsembleSearchEngine and TimeAdaptiveOptimizer
for fast, time-budgeted feature interaction discovery.
"""

from .ensemble_search import (
    EnsembleSearchEngine,
    SearchConfig,
    FeatureCandidate,
    FeatureRecipe,
    quick_search,
)

from .adaptive import (
    TimeAdaptiveOptimizer,
    StrategyConfig,
    SearchMode,
    auto_plan,
)

from .orchestrator import (
    GafimeOrchestrator,
    gafime_search,
)

__all__ = [
    # Ensemble Search
    "EnsembleSearchEngine",
    "SearchConfig", 
    "FeatureCandidate",
    "quick_search",
    # Adaptive
    "TimeAdaptiveOptimizer",
    "StrategyConfig",
    "SearchMode",
    "auto_plan",
    # Orchestrator
    "GafimeOrchestrator",
    "gafime_search",
]
