from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Optional, Tuple


DEFAULT_METRICS: Tuple[str, ...] = ("pearson", "spearman", "mutual_info", "r2")


@dataclass(frozen=True)
class ComputeBudget:
    """Bound candidate generation, residency, and device admission.

    Parameters
    ----------
    max_comb_size:
        Maximum continuous interaction arity.  Must be greater than zero and
        is capped by the available feature count.
    max_combinations_per_k:
        Positive per-arity candidate cap, including unary candidates.
    top_features_for_higher_k:
        Unary-screened feature count eligible for arity two and above.  Zero
        produces no higher-order candidates.
    max_generated_features:
        Reserved v0.x compatibility field.  Current v1 execution records it but
        does not use it to limit time-series or decision-path generation.
    keep_in_vram:
        Permit the bounded eager resident cache.  It is not a promise that a
        GPU is selected or that every allocation remains device-resident.
    vram_budget_mb:
        GPU admission budget in MiB; zero disables that admission ceiling.
    max_time_series_candidates:
        Upper bound for generated time-series descriptors.
    top_k_features_for_time_series:
        Unary-screened source-feature count used by time-series generation.
    max_feature_candidate:
        ``None`` uses every feature, a non-negative value caps the base feature
        prefix (zero selects none), and ``-1`` requests power-user mode. With
        otherwise default candidate limits, power-user mode retains a practical
        1024-feature guard; changing an explicit candidate limit removes that
        implicit guard. Values below ``-1`` fail.

    The first six fields are positional in the canonical v1 constructor. Later
    fields are keyword-only; historical positional layouts are not translated.
    """

    max_comb_size: int = 2
    max_combinations_per_k: int = 5000
    top_features_for_higher_k: int = 50
    max_generated_features: int = 0
    keep_in_vram: bool = True
    vram_budget_mb: int = 6144
    _: KW_ONLY
    max_time_series_candidates: int = 100_000
    top_k_features_for_time_series: int = 50
    max_feature_candidate: Optional[int] = None


@dataclass(frozen=True)
class EngineConfig:
    """Immutable declaration of one GAFIME analysis contract.

    ``metric_names`` accepts ``pearson``, ``spearman``, ``mutual_info``, and
    ``r2``.  ``num_repeats`` controls selected-candidate bootstrap stability;
    values above one request repeats.  ``permutation_tests`` controls
    family-wise maxT permutations, while ``significance_top_n`` independently
    bounds the surfaced significance rows and must be positive.
    ``random_seed=None`` requests fresh entropy for every analysis; an integer
    makes current-v1 planning and significance reproducible.

    ``mi_bins`` is an adaptive template ceiling (minimum two). Core uses
    adaptive quantile MI by default and ``mi_approximate=True`` selects its
    fixed equal-width estimator; GPU scoring already uses fixed equal-width
    histograms. ``backend`` accepts ``auto``, Core aliases
    ``core``/``cpu``/``rust``/``v1-rust-cpu``, ``cuda``, ROCm aliases
    ``rocm``/``hip``, or ``metal``.  Explicit backends never fall back.
    ``device_id`` is a non-negative device index.

    ``precision`` is keyword-only and accepts ``fp32``, ``mixed`` (default), or
    ``fp64``.  Core, CUDA, and ROCm support all three profiles; current Metal
    supports only ``fp32`` and rejects other explicit requests before input
    coercion or payload discovery.

    Time-series and decision-path switches are mutually exclusive.  Lags and
    windows consume caller row order.  Decision-path depth, rounds, path count,
    and minimum leaf must be positive; ``decision_path_max_bins=0`` requests
    exhaustive splits; learning rate must be positive; and the discovery
    shortlist must be non-negative.  Threshold fields affect the report
    decision but do not replace holdout or out-of-fold validation. Both must
    be finite, non-negative, and representable in the public result precision.

    The first ten fields through ``device_id`` may be positional; later fields
    are keyword-only. Removed legacy precision/discrete keywords and obsolete
    positional layouts raise :class:`TypeError` instead of being translated.
    """

    budget: ComputeBudget = field(default_factory=ComputeBudget)
    metric_names: Tuple[str, ...] = DEFAULT_METRICS
    num_repeats: int = 3
    permutation_tests: int = 25
    random_seed: Optional[int] = 7
    stability_std_threshold: float = 0.10
    permutation_p_threshold: float = 0.05
    # Adaptive maximum; the planner selects a sample-size-safe template.
    mi_bins: int = 96
    backend: str = "auto"
    device_id: int = 0
    _: KW_ONLY
    precision: str = "mixed"
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
    significance_top_n: int = 50
    mi_approximate: bool = False

    def __post_init__(self) -> None:
        from ._precision import normalize_precision

        # A boolean in this slot was formerly positional mi_approximate, not a
        # backend. Reject it instead of silently accepting the obsolete layout.
        if isinstance(self.backend, bool):
            raise TypeError("backend must be a string, not a boolean.")
        object.__setattr__(self, "precision", normalize_precision(self.precision))
