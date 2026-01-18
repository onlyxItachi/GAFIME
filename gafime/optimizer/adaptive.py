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
        
        Logic Hierarchy:
        1. THE 5-MINUTE LAW: If job < 300s, force full brute force
        2. LINEAR SCALING: raw_ratio = target / estimated, with 0.9 safety margin
        3. EXECUTION MODES: >=10% data = single scout, <10% = ensemble
        
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
        
        # Estimate full brute force time
        estimated_full = self.estimate_full_time(
            n_samples, n_features, n_operators, n_interactions, max_arity
        )
        
        logger.info(f"Estimated full brute force time: {estimated_full:.1f}s")
        
        # ================================================================
        # THE 5-MINUTE LAW (Quality Assurance)
        # If job < 300s, 5 minutes is negligible. Force full data.
        # ================================================================
        FIVE_MINUTE_THRESHOLD = 300.0
        
        if estimated_full < FIVE_MINUTE_THRESHOLD:
            logger.info(f"Job estimated at {estimated_full:.1f}s (<5min). "
                       "Enforcing Full Data for maximum quality.")
            return StrategyConfig(
                mode=SearchMode.FULL_BRUTE_FORCE,
                subsample_ratio=1.0,
                n_scouts=1,
                estimated_time=estimated_full,
                load_factor=estimated_full / max(target_time, 1.0)
            )
        
        # ================================================================
        # LINEAR SCALING (The Budget Zone)
        # raw_ratio = target / estimated
        # safe_ratio = raw_ratio * 0.9 (10% safety margin)
        # ================================================================
        raw_ratio = target_time / estimated_full
        safe_ratio = raw_ratio * 0.9  # 10% safety margin for overhead
        
        # Clamp to valid range [0.001, 1.0]
        MIN_RATIO = 0.001  # Never below 0.1%
        safe_ratio = max(MIN_RATIO, min(1.0, safe_ratio))
        
        load_factor = estimated_full / target_time
        
        logger.info(f"Linear scaling: raw={raw_ratio:.2%}, safe={safe_ratio:.2%}, L={load_factor:.2f}")
        
        # ================================================================
        # EXECUTION MODES
        # ================================================================
        
        # If safe_ratio >= 0.1 (10%+ data): AGGRESSIVE_SAMPLING with 1 scout
        if safe_ratio >= 0.1:
            estimated_time = estimated_full * safe_ratio
            
            logger.info(f"Mode: AGGRESSIVE_SAMPLING at {safe_ratio:.1%} (single scout)")
            return StrategyConfig(
                mode=SearchMode.AGGRESSIVE_SAMPLING,
                subsample_ratio=safe_ratio,
                n_scouts=1,
                estimated_time=estimated_time,
                load_factor=load_factor
            )
        
        # If safe_ratio < 0.1 (very low data): ENSEMBLE_SCOUTS with 3 scouts
        # Cannot rely on single <10% sample
        n_scouts = 3
        estimated_time = estimated_full * safe_ratio * n_scouts * 0.9
        
        logger.info(f"Mode: ENSEMBLE_SCOUTS with {n_scouts} scouts at {safe_ratio:.1%} each")
        return StrategyConfig(
            mode=SearchMode.ENSEMBLE_SCOUTS,
            subsample_ratio=safe_ratio,
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
