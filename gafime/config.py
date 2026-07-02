from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


DEFAULT_METRICS: Tuple[str, ...] = ("pearson", "spearman", "mutual_info", "r2")
@dataclass(frozen=True)
class ComputeBudget:
    max_comb_size: int = 2
    max_combinations_per_k: int = 5000
    top_features_for_higher_k: int = 50
    max_generated_features: int = 0
    keep_in_vram: bool = True  # Enable CUDA by default when available
    vram_budget_mb: int = 6144  # RTX 4060 has 8GB, leave headroom
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
    mi_approximate: bool = False
    backend: str = "auto"
    device_id: int = 0
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
