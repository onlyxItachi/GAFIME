"""
GAFIME Ensemble Search Engine

Implements the "Subsample & Vote" strategy for fast feature interaction discovery.

Architecture:
    Phase 1: SCOUTING PARTY (Multiple subsampled brute force)
        - Multiple scouts explore 1% random subsamples
        - Each scout finds top-K candidates
        - Voting promotes candidates appearing in multiple scouts
        
    Phase 2: VERIFICATION (Full data validation)
        - Test elected candidates on full data
        - Check for overfitting (|train_r - val_r| < threshold)
        - Return final ranked list

Law of Large Numbers: A real signal in 10M rows WILL appear in 100K subsample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from gafime.backends.fused_kernel import (
    StaticBucket,
    UnaryOp,
    InteractionType,
    compute_pearson_from_stats,
    create_fold_mask,
)

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FeatureCandidate:
    """Represents a candidate feature interaction."""
    feature_indices: Tuple[int, ...]
    operators: Tuple[int, ...]
    interaction_type: int
    
    # Scores (populated during evaluation)
    train_r: float = 0.0
    val_r: float = 0.0
    scout_votes: int = 0
    avg_correlation: float = 0.0
    
    @property
    def signature(self) -> str:
        """Unique identifier for this candidate."""
        return f"{self.feature_indices}|{self.operators}|{self.interaction_type}"
    
    @property
    def overfit_gap(self) -> float:
        """Gap between train and validation (high = likely overfit)."""
        return abs(self.train_r - self.val_r)
    
    def __hash__(self):
        return hash(self.signature)
    
    def __eq__(self, other):
        return self.signature == other.signature


@dataclass 
class ScoutResult:
    """Results from a single scout run."""
    scout_id: int
    subsample_indices: np.ndarray
    top_candidates: List[FeatureCandidate]
    time_seconds: float


@dataclass
class SearchConfig:
    """Configuration for EnsembleSearchEngine."""
    # Scouting parameters
    n_scouts: int = 3
    subsample_ratio: float = 0.01
    min_samples: int = 1000  # Never go below this
    
    # Candidate selection
    top_k_per_scout: int = 1000
    min_correlation: float = 0.05
    
    # Voting
    min_votes: int = 2  # Appear in at least this many scouts
    
    # Verification
    overfit_threshold: float = 0.05
    
    # Search space
    max_arity: int = 2
    operators: List[int] = field(default_factory=lambda: [
        UnaryOp.IDENTITY, UnaryOp.LOG, UnaryOp.SQRT, 
        UnaryOp.SQUARE, UnaryOp.TANH
    ])
    interactions: List[int] = field(default_factory=lambda: [
        InteractionType.MULT, InteractionType.ADD
    ])


# ============================================================================
# CANDIDATE GENERATOR
# ============================================================================

class CandidateGenerator:
    """
    Generates all possible feature interaction candidates.
    Does NOT evaluate - just yields index/operator combinations.
    """
    
    def __init__(self, n_features: int, config: SearchConfig):
        self.n_features = n_features
        self.config = config
    
    def generate(self) -> List[FeatureCandidate]:
        """Generate all candidates for the configured search space."""
        candidates = []
        
        for arity in range(2, self.config.max_arity + 1):
            # All feature combinations of this arity
            for feature_combo in combinations(range(self.n_features), arity):
                # All operator combinations
                for op_combo in product(self.config.operators, repeat=arity):
                    # All interaction types
                    for interact in self.config.interactions:
                        candidates.append(FeatureCandidate(
                            feature_indices=feature_combo,
                            operators=op_combo,
                            interaction_type=interact
                        ))
        
        return candidates
    
    def count(self) -> int:
        """Count total candidates without generating."""
        total = 0
        for arity in range(2, self.config.max_arity + 1):
            n_feature_combos = self._comb(self.n_features, arity)
            n_op_combos = len(self.config.operators) ** arity
            n_interactions = len(self.config.interactions)
            total += n_feature_combos * n_op_combos * n_interactions
        return total
    
    @staticmethod
    def _comb(n: int, k: int) -> int:
        """Compute C(n, k)."""
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
# SCOUT (Subsample Evaluator)
# ============================================================================

class Scout:
    """
    A single scout that evaluates candidates on a random subsample.
    Uses GPU for fast evaluation.
    """
    
    def __init__(
        self,
        scout_id: int,
        features: np.ndarray,
        target: np.ndarray,
        config: SearchConfig,
        seed: int = None
    ):
        self.scout_id = scout_id
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Create subsample
        n_samples = len(target)
        subsample_size = max(
            self.config.min_samples,
            int(n_samples * self.config.subsample_ratio)
        )
        self.subsample_indices = self.rng.choice(
            n_samples, size=subsample_size, replace=False
        )
        
        # Subsample data
        self.features_sub = features[self.subsample_indices]
        self.target_sub = target[self.subsample_indices].astype(np.float32)
        self.n_samples_sub = len(self.subsample_indices)
        self.n_features = features.shape[1]
        
        logger.info(f"Scout {scout_id}: Created with {self.n_samples_sub} samples "
                   f"({100*subsample_size/n_samples:.1f}%)")
    
    def run(self, candidates: List[FeatureCandidate]) -> ScoutResult:
        """
        Evaluate all candidates on the subsample.
        Returns top-K by correlation.
        
        Batches candidates by feature pair to work within MAX_FEATURES=5 limit.
        """
        import time
        from collections import defaultdict
        
        start = time.perf_counter()
        
        # Group candidates by unique feature pairs
        # Each pair gets its own bucket (2 features fits within limit)
        pair_to_candidates: Dict[Tuple[int, ...], List[FeatureCandidate]] = defaultdict(list)
        
        for candidate in candidates:
            if len(candidate.feature_indices) == 2:
                pair_to_candidates[candidate.feature_indices].append(candidate)
        
        # Prepare mask once
        mask = create_fold_mask(self.n_samples_sub, n_folds=5, seed=self.scout_id)
        target_f32 = self.target_sub.astype(np.float32)
        
        scored_candidates = []
        
        # Evaluate each feature pair
        for (f0, f1), pair_candidates in pair_to_candidates.items():
            try:
                # Create small bucket with just 2 features
                bucket = StaticBucket(self.n_samples_sub, 2)
                
                # Upload the two features (mapped to indices 0, 1 in bucket)
                bucket.upload_feature(0, self.features_sub[:, f0].astype(np.float32))
                bucket.upload_feature(1, self.features_sub[:, f1].astype(np.float32))
                bucket.upload_target(target_f32)
                bucket.upload_mask(mask)
                
                # Evaluate all op/interaction combos for this pair
                for candidate in pair_candidates:
                    try:
                        stats = bucket.compute(
                            feature_indices=[0, 1],  # Always 0,1 in this bucket
                            ops=list(candidate.operators),
                            interaction=candidate.interaction_type,
                            val_fold=0
                        )
                        train_r, val_r = compute_pearson_from_stats(stats)
                        
                        if abs(train_r) >= self.config.min_correlation:
                            # Create copy with scores
                            scored = FeatureCandidate(
                                feature_indices=candidate.feature_indices,
                                operators=candidate.operators,
                                interaction_type=candidate.interaction_type,
                                train_r=float(train_r),
                                val_r=float(val_r)
                            )
                            scored_candidates.append(scored)
                            
                    except Exception as e:
                        logger.debug(f"Scout {self.scout_id}: Error on {candidate.signature}: {e}")
                
                del bucket
                
            except Exception as e:
                logger.warning(f"Scout {self.scout_id}: Bucket error for pair {(f0, f1)}: {e}")
        
        # Sort by absolute train correlation, keep top K
        scored_candidates.sort(key=lambda c: abs(c.train_r), reverse=True)
        top_candidates = scored_candidates[:self.config.top_k_per_scout]
        
        elapsed = time.perf_counter() - start
        logger.info(f"Scout {self.scout_id}: Evaluated {len(candidates)} candidates "
                   f"({len(pair_to_candidates)} pairs) in {elapsed:.2f}s, kept {len(top_candidates)}")
        
        return ScoutResult(
            scout_id=self.scout_id,
            subsample_indices=self.subsample_indices,
            top_candidates=top_candidates,
            time_seconds=elapsed
        )


# ============================================================================
# VOTING SYSTEM
# ============================================================================

class VotingSystem:
    """
    Aggregates results from multiple scouts using voting.
    Promotes candidates that appear in multiple scouts' top-K lists.
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    def vote(self, scout_results: List[ScoutResult]) -> List[FeatureCandidate]:
        """
        Aggregate scout results and return elected candidates.
        
        Rules:
        1. Count how many scouts promoted each candidate
        2. Keep candidates with >= min_votes
        3. Average their correlations across scouts
        """
        # Track votes and scores per candidate signature
        vote_counts: Dict[str, int] = {}
        score_sums: Dict[str, float] = {}
        candidate_map: Dict[str, FeatureCandidate] = {}
        
        for result in scout_results:
            for candidate in result.top_candidates:
                sig = candidate.signature
                vote_counts[sig] = vote_counts.get(sig, 0) + 1
                score_sums[sig] = score_sums.get(sig, 0) + abs(candidate.train_r)
                
                if sig not in candidate_map:
                    candidate_map[sig] = FeatureCandidate(
                        feature_indices=candidate.feature_indices,
                        operators=candidate.operators,
                        interaction_type=candidate.interaction_type
                    )
        
        # Select candidates meeting vote threshold
        elected = []
        for sig, votes in vote_counts.items():
            if votes >= self.config.min_votes:
                candidate = candidate_map[sig]
                candidate.scout_votes = votes
                candidate.avg_correlation = score_sums[sig] / votes
                elected.append(candidate)
        
        # Sort by average correlation
        elected.sort(key=lambda c: c.avg_correlation, reverse=True)
        
        logger.info(f"Voting: {len(candidate_map)} unique candidates, "
                   f"{len(elected)} elected (>= {self.config.min_votes} votes)")
        
        return elected


# ============================================================================
# VERIFIER (Full Data Validation)
# ============================================================================

class Verifier:
    """
    Validates elected candidates on full data.
    Checks for overfitting and returns final rankings.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        config: SearchConfig
    ):
        self.features = features
        self.target = target.astype(np.float32)
        self.config = config
        self.n_samples = len(target)
        self.n_features = features.shape[1]
    
    def verify(self, candidates: List[FeatureCandidate]) -> List[FeatureCandidate]:
        """
        Re-evaluate candidates on full data with cross-validation.
        Returns candidates passing overfit check.
        
        Batches candidates by feature pair to work within MAX_FEATURES=5 limit.
        """
        import time
        from collections import defaultdict
        
        start = time.perf_counter()
        
        # Group candidates by unique feature pairs
        pair_to_candidates: Dict[Tuple[int, ...], List[FeatureCandidate]] = defaultdict(list)
        
        for candidate in candidates:
            if len(candidate.feature_indices) == 2:
                pair_to_candidates[candidate.feature_indices].append(candidate)
        
        mask = create_fold_mask(self.n_samples, n_folds=5, seed=42)
        target_f32 = self.target.astype(np.float32)
        
        verified = []
        
        for (f0, f1), pair_candidates in pair_to_candidates.items():
            try:
                bucket = StaticBucket(self.n_samples, 2)
                bucket.upload_feature(0, self.features[:, f0].astype(np.float32))
                bucket.upload_feature(1, self.features[:, f1].astype(np.float32))
                bucket.upload_target(target_f32)
                bucket.upload_mask(mask)
                
                for candidate in pair_candidates:
                    try:
                        # Evaluate on all 5 folds, average
                        train_rs, val_rs = [], []
                        for fold in range(5):
                            stats = bucket.compute(
                                feature_indices=[0, 1],
                                ops=list(candidate.operators),
                                interaction=candidate.interaction_type,
                                val_fold=fold
                            )
                            tr, vr = compute_pearson_from_stats(stats)
                            train_rs.append(abs(tr))
                            val_rs.append(abs(vr))
                        
                        verified_candidate = FeatureCandidate(
                            feature_indices=candidate.feature_indices,
                            operators=candidate.operators,
                            interaction_type=candidate.interaction_type,
                            train_r=float(np.mean(train_rs)),
                            val_r=float(np.mean(val_rs)),
                            scout_votes=candidate.scout_votes,
                            avg_correlation=candidate.avg_correlation
                        )
                        
                        # Check overfitting
                        if verified_candidate.overfit_gap <= self.config.overfit_threshold:
                            verified.append(verified_candidate)
                        else:
                            logger.debug(f"Rejected {candidate.signature}: gap {verified_candidate.overfit_gap:.3f}")
                            
                    except Exception as e:
                        logger.debug(f"Verifier error on {candidate.signature}: {e}")
                
                del bucket
                
            except Exception as e:
                logger.warning(f"Verifier: Bucket error for pair {(f0, f1)}: {e}")
        
        # Sort by validation correlation
        verified.sort(key=lambda c: c.val_r, reverse=True)
        
        elapsed = time.perf_counter() - start
        logger.info(f"Verifier: {len(candidates)} candidates → {len(verified)} verified "
                   f"in {elapsed:.2f}s")
        
        return verified


# ============================================================================
# MAIN ENGINE
# ============================================================================

class EnsembleSearchEngine:
    """
    Main engine implementing the "Subsample & Vote" strategy.
    
    Usage:
        engine = EnsembleSearchEngine(features, target)
        results = engine.search()
        
        for candidate in results[:10]:
            print(f"{candidate.signature}: val_r={candidate.val_r:.4f}")
    """
    
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        config: SearchConfig = None
    ):
        """
        Initialize the engine.
        
        Args:
            features: 2D array (n_samples, n_features) float32
            target: 1D array (n_samples,) float32
            config: Search configuration
        """
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.target = np.ascontiguousarray(target, dtype=np.float32)
        self.config = config or SearchConfig()
        
        self.n_samples, self.n_features = features.shape
        
        logger.info(f"EnsembleSearchEngine: {self.n_samples} samples, "
                   f"{self.n_features} features")
    
    def search(self) -> List[FeatureCandidate]:
        """
        Execute the full search pipeline.
        
        Returns:
            List of verified candidates, sorted by validation correlation.
        """
        import time
        total_start = time.perf_counter()
        
        # Generate candidates
        generator = CandidateGenerator(self.n_features, self.config)
        candidates = generator.generate()
        logger.info(f"Generated {len(candidates)} candidates")
        
        # ========== PHASE 1: SCOUTING PARTY ==========
        logger.info("=" * 60)
        logger.info("PHASE 1: SCOUTING PARTY")
        logger.info("=" * 60)
        
        scout_results = []
        for i in range(self.config.n_scouts):
            scout = Scout(
                scout_id=i,
                features=self.features,
                target=self.target,
                config=self.config,
                seed=i * 42
            )
            result = scout.run(candidates)
            scout_results.append(result)
        
        # ========== VOTING ==========
        logger.info("=" * 60)
        logger.info("VOTING")
        logger.info("=" * 60)
        
        voting = VotingSystem(self.config)
        elected = voting.vote(scout_results)
        
        if not elected:
            logger.warning("No candidates elected! Try adjusting min_votes or min_correlation.")
            return []
        
        # ========== PHASE 2: VERIFICATION ==========
        logger.info("=" * 60)
        logger.info("PHASE 2: VERIFICATION")
        logger.info("=" * 60)
        
        verifier = Verifier(self.features, self.target, self.config)
        verified = verifier.verify(elected)
        
        total_elapsed = time.perf_counter() - total_start
        logger.info("=" * 60)
        logger.info(f"COMPLETE: {len(verified)} verified features in {total_elapsed:.2f}s")
        logger.info("=" * 60)
        
        return verified


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def quick_search(
    features: np.ndarray,
    target: np.ndarray,
    n_scouts: int = 3,
    subsample_ratio: float = 0.01,
    top_k: int = 100
) -> List[FeatureCandidate]:
    """
    Quick wrapper for EnsembleSearchEngine.
    
    Args:
        features: 2D array (n_samples, n_features)
        target: 1D array (n_samples,)
        n_scouts: Number of scouts for voting
        subsample_ratio: Fraction of data per scout
        top_k: Return top K results
    
    Returns:
        Top K verified candidates
    """
    config = SearchConfig(
        n_scouts=n_scouts,
        subsample_ratio=subsample_ratio
    )
    engine = EnsembleSearchEngine(features, target, config)
    results = engine.search()
    return results[:top_k]
