"""
GAFIME Orchestrator

Main entry point that integrates TimeAdaptiveOptimizer with EnsembleSearchEngine.
Provides a single unified API for time-budgeted feature search.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from .adaptive import TimeAdaptiveOptimizer, StrategyConfig, SearchMode
from .ensemble_search import (
    EnsembleSearchEngine,
    SearchConfig,
    FeatureCandidate,
    CandidateGenerator,
    Scout,
    VotingSystem,
    Verifier,
)

logger = logging.getLogger(__name__)


class GafimeOrchestrator:
    """
    Main orchestrator for GAFIME feature search.
    
    Automatically selects and executes the optimal search strategy
    based on the provided time budget.
    
    Usage:
        orchestrator = GafimeOrchestrator(features, target)
        results = orchestrator.run(target_time=120.0)
        
        for candidate in results[:10]:
            print(f"{candidate.signature}: {candidate.val_r:.4f}")
    """
    
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        n_operators: int = 5,
        n_interactions: int = 2,
        max_arity: int = 2
    ):
        """
        Initialize the orchestrator.
        
        Args:
            features: Feature matrix (n_samples, n_features) float32
            target: Target vector (n_samples,) float32
            n_operators: Number of unary operators to try
            n_interactions: Number of interaction types
            max_arity: Maximum feature combination arity
        """
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.target = np.ascontiguousarray(target, dtype=np.float32)
        self.n_samples, self.n_features = features.shape
        
        self.n_operators = n_operators
        self.n_interactions = n_interactions
        self.max_arity = max_arity
        
        # Initialize adaptive optimizer
        self.adaptive = TimeAdaptiveOptimizer()
        self._calibrated = False
        
        logger.info(f"GafimeOrchestrator: {self.n_samples} samples, {self.n_features} features")
    
    def calibrate(self) -> float:
        """
        Run GPU calibration to measure throughput.
        
        Returns:
            Measured throughput in iterations/second
        """
        throughput = self.adaptive.calibrate(self.features, self.target)
        self._calibrated = True
        return throughput
    
    def plan(self, target_time: float) -> StrategyConfig:
        """
        Plan the optimal strategy for the given time budget.
        
        Args:
            target_time: Maximum desired execution time in seconds
        
        Returns:
            StrategyConfig with mode, subsample_ratio, n_scouts
        """
        if not self._calibrated:
            self.calibrate()
        
        return self.adaptive.plan_strategy(
            dataset_shape=(self.n_samples, self.n_features),
            target_time=target_time,
            n_operators=self.n_operators,
            n_interactions=self.n_interactions,
            max_arity=self.max_arity
        )
    
    def run(
        self,
        target_time: float = 120.0,
        min_correlation: float = 0.05,
        overfit_threshold: float = 0.05
    ) -> List[FeatureCandidate]:
        """
        Execute the complete search pipeline with automatic strategy selection.
        
        Args:
            target_time: Maximum desired execution time in seconds
            min_correlation: Minimum correlation to keep candidate
            overfit_threshold: Maximum |train_r - val_r| for verification
        
        Returns:
            List of verified candidates, sorted by validation correlation
        """
        total_start = time.perf_counter()
        
        # ================================================================
        # STEP 1: Calibrate and Plan
        # ================================================================
        logger.info("=" * 70)
        logger.info("GAFIME ORCHESTRATOR")
        logger.info("=" * 70)
        
        strategy = self.plan(target_time)
        logger.info(f"\n{strategy}")
        
        # ================================================================
        # STEP 2: Execute Based on Strategy Mode
        # ================================================================
        
        if strategy.mode == SearchMode.FULL_BRUTE_FORCE:
            results = self._run_full_brute_force(
                min_correlation=min_correlation,
                overfit_threshold=overfit_threshold
            )
        
        elif strategy.mode == SearchMode.AGGRESSIVE_SAMPLING:
            results = self._run_aggressive_sampling(
                subsample_ratio=strategy.subsample_ratio,
                min_correlation=min_correlation,
                overfit_threshold=overfit_threshold
            )
        
        elif strategy.mode == SearchMode.ENSEMBLE_SCOUTS:
            results = self._run_ensemble_scouts(
                n_scouts=strategy.n_scouts,
                subsample_ratio=strategy.subsample_ratio,
                min_correlation=min_correlation,
                overfit_threshold=overfit_threshold
            )
        
        else:
            raise ValueError(f"Unknown search mode: {strategy.mode}")
        
        # ================================================================
        # STEP 3: Report
        # ================================================================
        total_elapsed = time.perf_counter() - total_start
        logger.info("=" * 70)
        logger.info(f"COMPLETE: {len(results)} features found in {total_elapsed:.2f}s")
        logger.info(f"Target was {target_time:.0f}s, actual was {total_elapsed:.1f}s "
                   f"({100*total_elapsed/target_time:.0f}%)")
        logger.info("=" * 70)
        
        return results
    
    def _run_full_brute_force(
        self,
        min_correlation: float,
        overfit_threshold: float
    ) -> List[FeatureCandidate]:
        """Execute full brute force on complete data."""
        logger.info("-" * 70)
        logger.info("MODE: FULL_BRUTE_FORCE")
        logger.info("-" * 70)
        
        config = SearchConfig(
            n_scouts=1,
            subsample_ratio=1.0,
            min_samples=self.n_samples,
            top_k_per_scout=10000,  # Keep all
            min_correlation=min_correlation,
            min_votes=1,
            overfit_threshold=overfit_threshold,
            max_arity=self.max_arity
        )
        
        engine = EnsembleSearchEngine(self.features, self.target, config)
        return engine.search()
    
    def _run_aggressive_sampling(
        self,
        subsample_ratio: float,
        min_correlation: float,
        overfit_threshold: float
    ) -> List[FeatureCandidate]:
        """Execute single scout with aggressive subsampling."""
        logger.info("-" * 70)
        logger.info(f"MODE: AGGRESSIVE_SAMPLING ({subsample_ratio:.1%})")
        logger.info("-" * 70)
        
        config = SearchConfig(
            n_scouts=1,
            subsample_ratio=subsample_ratio,
            min_samples=max(1000, int(self.n_samples * 0.01)),
            top_k_per_scout=1000,
            min_correlation=min_correlation,
            min_votes=1,
            overfit_threshold=overfit_threshold,
            max_arity=self.max_arity
        )
        
        engine = EnsembleSearchEngine(self.features, self.target, config)
        return engine.search()
    
    def _run_ensemble_scouts(
        self,
        n_scouts: int,
        subsample_ratio: float,
        min_correlation: float,
        overfit_threshold: float
    ) -> List[FeatureCandidate]:
        """Execute ensemble of scouts with voting."""
        logger.info("-" * 70)
        logger.info(f"MODE: ENSEMBLE_SCOUTS ({n_scouts} scouts at {subsample_ratio:.1%})")
        logger.info("-" * 70)
        
        # Require majority vote for ensemble
        min_votes = max(2, (n_scouts + 1) // 2)
        
        config = SearchConfig(
            n_scouts=n_scouts,
            subsample_ratio=subsample_ratio,
            min_samples=max(1000, int(self.n_samples * 0.01)),
            top_k_per_scout=500,
            min_correlation=min_correlation,
            min_votes=min_votes,
            overfit_threshold=overfit_threshold,
            max_arity=self.max_arity
        )
        
        engine = EnsembleSearchEngine(self.features, self.target, config)
        return engine.search()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def gafime_search(
    features: np.ndarray,
    target: np.ndarray,
    target_time: float = 120.0,
    top_k: int = 100
) -> List[FeatureCandidate]:
    """
    One-liner feature interaction search.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        target: Target vector (n_samples,)
        target_time: Maximum execution time in seconds
        top_k: Return top K results
    
    Returns:
        Top K verified feature interactions
    
    Example:
        results = gafime_search(X, y, target_time=60.0)
        print(results[0].signature)  # Best interaction
    """
    orchestrator = GafimeOrchestrator(features, target)
    results = orchestrator.run(target_time=target_time)
    return results[:top_k]
