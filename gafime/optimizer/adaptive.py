"""
GAFIME Time-Adaptive Optimizer

Automatically selects the optimal search strategy based on a time budget.
Uses GPU calibration to estimate execution time and adapts accordingly.

Strategy Zones:
    Zone 1 (L ≤ 1.0): FULL_BRUTE_FORCE - Full data, no sampling
    Zone 2 (1.0 < L ≤ 2.0): AGGRESSIVE_SAMPLING - Single scout, reduced data
    Zone 3 (L > 2.0): ENSEMBLE_SCOUTS - Multiple scouts for signal guarantee
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from gafime.backends.fused_kernel import (
    StaticBucket,
    UnaryOp,
    InteractionType,
    create_fold_mask,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_TARGET_TIME = 30.0  # Floor: minimum 30 seconds
FAST_JOB_THRESHOLD = 60.0  # "Peanuts" rule: if <60s, force full brute force
CALIBRATION_SAMPLES = 1000  # Samples used for throughput calibration
CALIBRATION_ITERATIONS = 100  # Number of compute calls during calibration


class SearchMode(Enum):
    """Search execution modes."""
    FULL_BRUTE_FORCE = "full_brute_force"
    AGGRESSIVE_SAMPLING = "aggressive_sampling"
    ENSEMBLE_SCOUTS = "ensemble_scouts"


@dataclass
class StrategyConfig:
    """Configuration returned by the adaptive optimizer."""
    mode: SearchMode
    subsample_ratio: float
    n_scouts: int
    estimated_time: float
    load_factor: float
    
    def __str__(self) -> str:
        return (f"Strategy: {self.mode.value}\n"
                f"  Subsample ratio: {self.subsample_ratio:.2%}\n"
                f"  Scouts: {self.n_scouts}\n"
                f"  Estimated time: {self.estimated_time:.1f}s\n"
                f"  Load factor: {self.load_factor:.2f}")


# ============================================================================
# TIME-ADAPTIVE OPTIMIZER
# ============================================================================

class TimeAdaptiveOptimizer:
    """
    Automatically determines optimal search strategy based on time budget.
    
    Usage:
        optimizer = TimeAdaptiveOptimizer()
        optimizer.calibrate(sample_features, sample_target)
        
        strategy = optimizer.plan_strategy(
            dataset_shape=(1_000_000, 100),
            target_time=120.0
        )
        
        print(strategy.mode)  # SearchMode.ENSEMBLE_SCOUTS
    """
    
    def __init__(self):
        """Initialize the optimizer (not calibrated yet)."""
        self.gpu_throughput: Optional[float] = None  # items/sec
        self._calibrated = False
    
    def calibrate(
        self,
        features: np.ndarray,
        target: np.ndarray,
        n_samples: int = CALIBRATION_SAMPLES,
        n_iterations: int = CALIBRATION_ITERATIONS
    ) -> float:
        """
        Measure GPU throughput by running a benchmark.
        
        Args:
            features: Sample features for calibration
            target: Sample target for calibration
            n_samples: Number of samples to use (default 1000)
            n_iterations: Number of compute iterations (default 100)
        
        Returns:
            Measured throughput in iterations/second
        """
        logger.info(f"Calibrating GPU throughput ({n_iterations} iterations)...")
        
        # Use subset if features are large
        actual_samples = min(n_samples, len(target))
        if len(target) > n_samples:
            indices = np.random.choice(len(target), n_samples, replace=False)
            cal_features = features[indices]
            cal_target = target[indices]
        else:
            cal_features = features
            cal_target = target
        
        n_features = min(cal_features.shape[1], 2)
        
        # Create bucket
        bucket = StaticBucket(actual_samples, n_features)
        for i in range(n_features):
            bucket.upload_feature(i, cal_features[:, i].astype(np.float32))
        bucket.upload_target(cal_target.astype(np.float32))
        mask = create_fold_mask(actual_samples, n_folds=5)
        bucket.upload_mask(mask)
        
        # Warm up
        for _ in range(10):
            bucket.compute([0, 1] if n_features >= 2 else [0, 0],
                          [UnaryOp.LOG, UnaryOp.SQRT],
                          InteractionType.MULT, 0)
        
        # Benchmark
        start = time.perf_counter()
        for i in range(n_iterations):
            bucket.compute([0, 1] if n_features >= 2 else [0, 0],
                          [UnaryOp.LOG, UnaryOp.SQRT],
                          InteractionType.MULT, i % 5)
        elapsed = time.perf_counter() - start
        
        del bucket
        
        self.gpu_throughput = n_iterations / elapsed
        self._calibrated = True
        
        logger.info(f"Calibration complete: {self.gpu_throughput:.0f} iterations/sec")
        return self.gpu_throughput
    
    def estimate_full_time(
        self,
        n_samples: int,
        n_features: int,
        n_operators: int = 5,
        n_interactions: int = 2,
        max_arity: int = 2
    ) -> float:
        """
        Estimate time for full brute force search.
        
        Args:
            n_samples: Number of data samples
            n_features: Number of features
            n_operators: Number of unary operators to try
            n_interactions: Number of interaction types
            max_arity: Maximum feature combination arity
        
        Returns:
            Estimated time in seconds
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before estimating time")
        
        # Count total evaluations
        total_evals = 0
        for arity in range(2, max_arity + 1):
            n_combos = self._comb(n_features, arity)
            n_op_combos = n_operators ** arity
            total_evals += n_combos * n_op_combos * n_interactions
        
        # Estimate time
        # Factor in data size (larger data = slightly slower per iter)
        size_factor = 1.0 + (n_samples / 100_000) * 0.1
        
        estimated_time = (total_evals / self.gpu_throughput) * size_factor
        
        return estimated_time
    
    def plan_strategy(
        self,
        dataset_shape: Tuple[int, int],
        target_time: float,
        n_operators: int = 5,
        n_interactions: int = 2,
        max_arity: int = 2
    ) -> StrategyConfig:
        """
        Determine optimal strategy based on time budget.
        
        Args:
            dataset_shape: (n_samples, n_features)
            target_time: Desired maximum execution time in seconds
            n_operators: Number of unary operators
            n_interactions: Number of interaction types
            max_arity: Maximum arity
        
        Returns:
            StrategyConfig with mode, subsample_ratio, n_scouts
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before planning strategy")
        
        n_samples, n_features = dataset_shape
        
        # ================================================================
        # SANITY CHECK 1: The Floor (minimum 30s)
        # ================================================================
        if target_time < MIN_TARGET_TIME:
            logger.warning(f"Target time {target_time:.1f}s too low. "
                          f"Adjusted to minimum {MIN_TARGET_TIME:.0f}s for stability.")
            target_time = MIN_TARGET_TIME
        
        # ================================================================
        # Estimate full brute force time
        # ================================================================
        estimated_full = self.estimate_full_time(
            n_samples, n_features, n_operators, n_interactions, max_arity
        )
        
        logger.info(f"Estimated full brute force time: {estimated_full:.1f}s")
        
        # ================================================================
        # SANITY CHECK 2: The "Peanuts" Rule (fast job bypass)
        # ================================================================
        if estimated_full < FAST_JOB_THRESHOLD:
            logger.info(f"Job is fast enough (<{FAST_JOB_THRESHOLD:.0f}s). "
                       "Forcing FULL_BRUTE_FORCE for max quality.")
            return StrategyConfig(
                mode=SearchMode.FULL_BRUTE_FORCE,
                subsample_ratio=1.0,
                n_scouts=1,
                estimated_time=estimated_full,
                load_factor=estimated_full / target_time
            )
        
        # ================================================================
        # Calculate Load Factor: L = Estimated / Target
        # ================================================================
        load_factor = estimated_full / target_time
        
        logger.info(f"Load factor L = {estimated_full:.1f}s / {target_time:.1f}s = {load_factor:.2f}")
        
        # ================================================================
        # ZONE 1: Safe (L ≤ 1.0) - Full brute force fits in budget
        # ================================================================
        if load_factor <= 1.0:
            logger.info("Zone 1 (Safe): Full brute force fits in time budget")
            return StrategyConfig(
                mode=SearchMode.FULL_BRUTE_FORCE,
                subsample_ratio=1.0,
                n_scouts=1,
                estimated_time=estimated_full,
                load_factor=load_factor
            )
        
        # ================================================================
        # ZONE 2: Mild Load (1.0 < L ≤ 2.0) - Aggressive sampling
        # ================================================================
        if load_factor <= 2.0:
            # Logarithmic scaling for smooth transition (no cliff effect!)
            # Formula: 1 / (1 + log(L)) gives smoother reduction than 1/L
            subsample_ratio = 1.0 / (1.0 + math.log(load_factor))
            subsample_ratio = max(0.1, min(1.0, subsample_ratio))  # Clamp to [10%, 100%]
            
            estimated_time = estimated_full * subsample_ratio
            
            logger.info(f"Zone 2 (Mild): Aggressive sampling at {subsample_ratio:.1%} (log scale)")
            return StrategyConfig(
                mode=SearchMode.AGGRESSIVE_SAMPLING,
                subsample_ratio=subsample_ratio,
                n_scouts=1,
                estimated_time=estimated_time,
                load_factor=load_factor
            )
        
        # ================================================================
        # ZONE 3: Heavy Load (L > 2.0) - Ensemble scouts needed
        # ================================================================
        # Number of scouts: k = max(3, min(5, log2(L)))
        n_scouts = max(3, min(5, int(math.log2(load_factor))))
        
        # Logarithmic subsample scaling: 1 / (1 + log(L))
        # This gives much smoother reduction:
        #   L=2   → 59%    (vs 50% linear)
        #   L=10  → 30%    (vs 10% linear)
        #   L=30  → 23%    (vs 3.3% linear)
        #   L=100 → 18%    (vs 1% linear)
        subsample_ratio = 1.0 / (1.0 + math.log(load_factor))
        subsample_ratio = max(0.01, min(1.0, subsample_ratio))  # Clamp to [1%, 100%]
        
        # Each scout runs on a subsample, estimate total time
        estimated_time = estimated_full * subsample_ratio * n_scouts * 0.9  # 0.9 = overlap factor
        
        logger.info(f"Zone 3 (Heavy): Ensemble with {n_scouts} scouts at {subsample_ratio:.1%} each (log scale)")
        return StrategyConfig(
            mode=SearchMode.ENSEMBLE_SCOUTS,
            subsample_ratio=subsample_ratio,
            n_scouts=n_scouts,
            estimated_time=estimated_time,
            load_factor=load_factor
        )
    
    @staticmethod
    def _comb(n: int, k: int) -> int:
        """Compute C(n, k) = n! / (k! * (n-k)!)."""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def auto_plan(
    features: np.ndarray,
    target: np.ndarray,
    target_time: float = 120.0
) -> StrategyConfig:
    """
    Quick utility to plan strategy for a dataset.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        target: Target vector (n_samples,)
        target_time: Desired max execution time
    
    Returns:
        StrategyConfig with recommended settings
    """
    optimizer = TimeAdaptiveOptimizer()
    optimizer.calibrate(features, target)
    
    return optimizer.plan_strategy(
        dataset_shape=features.shape,
        target_time=target_time
    )
