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
_LEGACY_PRECISION_WARNING = (
    "storage_dtype and compute_policy are deprecated; use the single keyword-only "
    "precision profile instead."
)
_MISSING = object()


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

    The first six fields retain their historical positional order.  Later
    fields are keyword-only because their v0.4.7/v0.5 positional meanings were
    ambiguous.
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


_generated_compute_budget_init = ComputeBudget.__init__


@wraps(_generated_compute_budget_init)
def _compatible_compute_budget_init(self, *args, **kwargs) -> None:
    if len(args) > 6:
        raise TypeError(_AMBIGUOUS_COMPUTE_BUDGET_POSITIONAL_MESSAGE)
    _generated_compute_budget_init(self, *args, **kwargs)


ComputeBudget.__init__ = _compatible_compute_budget_init


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
    decision but do not replace holdout or out-of-fold validation.

    Deprecated ``storage_dtype``/``compute_policy`` keyword pairs are accepted
    only when they map unambiguously to one precision profile and emit a
    :class:`DeprecationWarning`.  The removed discrete-family switch cannot
    enable a v1 family.
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


_generated_engine_config_init = EngineConfig.__init__


@wraps(_generated_engine_config_init)
def _compatible_engine_config_init(self, *args, **kwargs) -> None:
    legacy_storage = kwargs.pop("storage_dtype", _MISSING)
    legacy_policy = kwargs.pop("compute_policy", _MISSING)
    if legacy_storage is not _MISSING or legacy_policy is not _MISSING:
        if legacy_storage is _MISSING or legacy_policy is _MISSING:
            raise TypeError(
                "deprecated storage_dtype and compute_policy must be supplied together."
            )
        if "precision" in kwargs:
            raise TypeError(
                "precision cannot be combined with deprecated storage_dtype/compute_policy."
            )
        from ._precision import precision_from_legacy_pair

        kwargs["precision"] = precision_from_legacy_pair(legacy_storage, legacy_policy)
        warnings.warn(_LEGACY_PRECISION_WARNING, DeprecationWarning, stacklevel=2)

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
            raise TypeError(
                "mi_approximate was provided both positionally and by keyword."
            )
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
    from ._precision import normalize_precision

    object.__setattr__(self, "precision", normalize_precision(self.precision))


EngineConfig.__init__ = _compatible_engine_config_init
