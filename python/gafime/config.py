from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from functools import wraps
from typing import Optional, Tuple
import warnings


DEFAULT_METRICS: Tuple[str, ...] = ("pearson", "spearman", "mutual_info", "r2")
_AMBIGUOUS_COMPUTE_BUDGET_POSITIONAL_MESSAGE = (
    "ComputeBudget accepts only its first six fields positionally. Positional "
    "argument 7 and later are ambiguous across v0.4.7 and v0.5: v0.4.7 used "
    "max_discrete_candidates in slot 7, while v0.5 uses "
    "max_time_series_candidates. Pass all later fields by keyword."
)
_REMOVED_DISCRETE_CONFIG_MESSAGE = (
    "EngineConfig positional argument 11 was enable_discrete_functions in v0.4.7, "
    "but the discrete family is not part of the v1 runtime. It is never mapped to "
    "enable_time_series_functions. Remove the legacy discrete option or pass "
    "enable_time_series_functions=... explicitly by keyword."
)
_DISABLED_DISCRETE_CONFIG_WARNING = (
    "enable_discrete_functions=False is a deprecated v0.4.7 option and is "
    "ignored by the v1 runtime. Remove it from EngineConfig."
)
_MISSING = object()


@dataclass(frozen=True)
class ComputeBudget:
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


_generated_compute_budget_init = ComputeBudget.__init__


@wraps(_generated_compute_budget_init)
def _compatible_compute_budget_init(self, *args, **kwargs) -> None:
    if len(args) > 6:
        raise TypeError(_AMBIGUOUS_COMPUTE_BUDGET_POSITIONAL_MESSAGE)
    _generated_compute_budget_init(self, *args, **kwargs)


ComputeBudget.__init__ = _compatible_compute_budget_init


@dataclass(frozen=True)
class EngineConfig:
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
    storage_dtype: str = "float32"
    compute_policy: str = "stable"
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


_generated_engine_config_init = EngineConfig.__init__


@wraps(_generated_engine_config_init)
def _compatible_engine_config_init(self, *args, **kwargs) -> None:
    positional_mi_approximate = _MISSING
    origin_main_layout = len(args) >= 9 and isinstance(args[8], bool)
    if origin_main_layout:
        if len(args) > 11:
            raise TypeError(
                "EngineConfig family switches after device_id are keyword-only."
            )
        positional_mi_approximate = args[8]
        args = (*args[:8], *args[9:])
    elif len(args) > 11:
        raise TypeError(_REMOVED_DISCRETE_CONFIG_MESSAGE)

    if positional_mi_approximate is not _MISSING:
        if "mi_approximate" in kwargs:
            raise TypeError("mi_approximate was provided both positionally and by keyword.")
        kwargs["mi_approximate"] = positional_mi_approximate

    legacy_discrete = _MISSING
    if not origin_main_layout and len(args) == 11:
        legacy_discrete = args[-1]
        args = args[:10]
    if "enable_discrete_functions" in kwargs:
        if legacy_discrete is not _MISSING:
            raise TypeError(
                "enable_discrete_functions was provided both positionally and by keyword."
            )
        legacy_discrete = kwargs.pop("enable_discrete_functions")
    if legacy_discrete is not _MISSING:
        if legacy_discrete is not False:
            raise TypeError(_REMOVED_DISCRETE_CONFIG_MESSAGE)
        warnings.warn(
            _DISABLED_DISCRETE_CONFIG_WARNING,
            DeprecationWarning,
            stacklevel=2,
        )
    _generated_engine_config_init(self, *args, **kwargs)


EngineConfig.__init__ = _compatible_engine_config_init
