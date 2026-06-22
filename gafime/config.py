from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


DEFAULT_METRICS: Tuple[str, ...] = ("pearson", "spearman", "mutual_info", "r2")
DEFAULT_DISCRETE_QUANTILES: Tuple[float, ...] = (
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.65,
    0.75,
    0.90,
    0.95,
)


@dataclass(frozen=True)
class ComputeBudget:
    max_comb_size: int = 2
    max_combinations_per_k: int = 5000
    top_features_for_higher_k: int = 50
    max_generated_features: int = 0
    keep_in_vram: bool = True  # Enable CUDA by default when available
    vram_budget_mb: int = 6144  # RTX 4060 has 8GB, leave headroom
    max_discrete_candidates: int = 100_000
    max_thresholds_per_feature: int = 9
    max_intervals_per_feature: int = 12
    max_feature_pairs_for_rectangles: int = 500
    top_k_features_for_discrete: int = 50
    max_time_series_candidates: int = 100_000
    top_k_features_for_time_series: int = 50
    max_feature_candidate: Optional[int] = None


@dataclass(frozen=True)
class EngineConfig:
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    metric_names: Tuple[str, ...] = DEFAULT_METRICS
    num_repeats: int = 3
    permutation_tests: int = 25
    random_seed: Optional[int] = 7
    stability_std_threshold: float = 0.10
    permutation_p_threshold: float = 0.05
    mi_bins: int = 96
    backend: str = "auto"
    device_id: int = 0
    enable_discrete_functions: bool = False
    discrete_mode: str = "soft"
    discrete_ranking: str = "split_aware"
    discrete_threshold_source: str = "quantile"
    discrete_gate_sharpness: float = 12.0
    discrete_quantiles: Tuple[float, ...] = DEFAULT_DISCRETE_QUANTILES
    enable_time_series_functions: bool = False
    time_series_lags: Tuple[int, ...] = (1, 2, 4, 8, 16)
    time_series_windows: Tuple[int, ...] = (4, 8, 16, 32)
    enable_decision_path_functions: bool = False
    decision_path_max_depth: int = 2
    decision_path_rounds: int = 1
    decision_path_max_paths: int = 32
    decision_path_max_bins: int = 0
    decision_path_min_leaf: int = 8
    decision_path_learning_rate: float = 1.0
    decision_path_top_k_features: int = 50
