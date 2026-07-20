"""Bounded compatibility metadata for the v0.5 compile-plan surface.

This module never drives native execution and never materializes candidates. The
authoritative execution plans remain Rust-owned; this projection emits at most
one compact descriptor per configured arity for API compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

from ..config import ComputeBudget, EngineConfig
from .flags import CompileFlags


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
UINT128_MAX = (1 << 128) - 1
DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP = 1_024
DEFAULT_CHUNK_SIZE = 1_024
_METRIC_IDS = {"pearson": 1, "spearman": 2, "mutual_info": 3, "r2": 4}


@dataclass(frozen=True)
class ChunkRange:
    first_chunk_id: int
    chunk_count: int
    chunk_size: int

    @property
    def last_chunk_id(self) -> int:
        return self.first_chunk_id + self.chunk_count - 1 if self.chunk_count else self.first_chunk_id - 1


@dataclass(frozen=True)
class ContinuousArityDescriptor:
    arity: int
    feature_start: int
    feature_stop: int
    universe_count: int
    planned_count: int
    offset: int
    chunk_range: ChunkRange
    saturated: bool = False

    @property
    def offset_end(self) -> int:
        return min(UINT64_MAX, self.offset + self.planned_count)


@dataclass(frozen=True)
class TimeSeriesDescriptor:
    feature_start: int
    feature_stop: int
    lag_count: int
    window_count: int
    template_count: int
    universe_count: int
    planned_count: int
    offset: int
    saturated: bool = False

    @property
    def offset_end(self) -> int:
        return min(UINT64_MAX, self.offset + self.planned_count)


@dataclass(frozen=True)
class ScenarioPlan:
    n_samples: int
    n_features: int
    feature_candidate_count: int
    continuous: Tuple[ContinuousArityDescriptor, ...] = field(default_factory=tuple)
    time_series: TimeSeriesDescriptor | None = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metric_ids: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def rows(self) -> int:
        return self.n_samples

    @property
    def cols(self) -> int:
        return self.n_features

    @property
    def max_arity(self) -> int:
        return max((item.arity for item in self.continuous), default=0)

    @property
    def continuous_count(self) -> int:
        return sum(item.planned_count for item in self.continuous)

    @property
    def planned_count(self) -> int:
        count = self.continuous_count
        if self.time_series is not None:
            count = min(UINT128_MAX, count + self.time_series.planned_count)
        return count

    @classmethod
    def empty(cls, n_samples: int, n_features: int) -> "ScenarioPlan":
        return cls(int(n_samples), int(n_features), int(n_features))


def build_scenario_plan(
    X: Any,
    config: EngineConfig,
    flags: CompileFlags | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ScenarioPlan:
    shape = getattr(X, "shape", None)
    n_samples = int(getattr(X, "n_samples", shape[0] if shape else len(X)))
    n_features = int(
        getattr(
            X,
            "n_features",
            shape[1] if shape else (len(X[0]) if n_samples else 0),
        )
    )
    return build_scenario_plan_from_shape(
        n_samples=n_samples,
        n_features=n_features,
        config=config,
        flags=flags,
        chunk_size=chunk_size,
    )


def build_scenario_plan_from_shape(
    *,
    n_samples: int,
    n_features: int,
    config: EngineConfig,
    flags: CompileFlags | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ScenarioPlan:
    compile_flags = flags or CompileFlags()
    if not compile_flags.plan:
        return ScenarioPlan.empty(n_samples, n_features)

    warnings: list[str] = []
    feature_count = _feature_candidate_count(n_features, config.budget, warnings)
    continuous = _continuous_descriptors(feature_count, config.budget, chunk_size, warnings)
    offset = min(UINT64_MAX, sum(item.planned_count for item in continuous))
    time_series = _time_series_descriptor(feature_count, config, offset)
    return ScenarioPlan(
        n_samples=int(n_samples),
        n_features=int(n_features),
        feature_candidate_count=feature_count,
        continuous=continuous,
        time_series=time_series,
        warnings=tuple(warnings),
        metric_ids=tuple(_METRIC_IDS[str(name)] for name in config.metric_names),
    )


def _feature_candidate_count(
    n_features: int, budget: ComputeBudget, warnings: list[str]
) -> int:
    value = budget.max_feature_candidate
    if value is None:
        return n_features
    value = int(value)
    if value < -1:
        raise ValueError("max_feature_candidate must be >= 0 or -1 for power-user mode.")
    if value >= 0:
        return min(n_features, value)
    defaults = ComputeBudget()
    explicit_limits = any(
        getattr(budget, name) != getattr(defaults, name)
        for name in (
            "max_comb_size",
            "max_combinations_per_k",
            "top_features_for_higher_k",
            "max_time_series_candidates",
            "top_k_features_for_time_series",
        )
    )
    if explicit_limits:
        return n_features
    capped = min(n_features, DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP)
    if capped < n_features:
        warnings.append(
            "max_feature_candidate=-1 without explicit limits; applying practical "
            f"safety cap of {DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP} features."
        )
    return capped


def _continuous_descriptors(
    n_features: int,
    budget: ComputeBudget,
    chunk_size: int,
    warnings: list[str],
) -> Tuple[ContinuousArityDescriptor, ...]:
    descriptors = []
    offset = 0
    chunk_id = 0
    max_arity = max(1, min(int(budget.max_comb_size), n_features))
    for arity in range(1, max_arity + 1):
        feature_stop = n_features if arity == 1 else min(
            n_features, max(0, int(budget.top_features_for_higher_k))
        )
        universe = math.comb(feature_stop, arity) if arity <= feature_stop else 0
        saturated = universe > UINT128_MAX
        universe = min(UINT128_MAX, universe)
        cap = int(budget.max_combinations_per_k)
        planned = universe if cap < 0 else min(universe, cap)
        chunks = 0 if planned == 0 else (planned + max(1, chunk_size) - 1) // max(1, chunk_size)
        if chunks > UINT32_MAX:
            warnings.append(
                f"arity={arity} chunk count exceeds uint32; native launch ids will be split by checkpoint."
            )
        descriptors.append(
            ContinuousArityDescriptor(
                arity=arity,
                feature_start=0,
                feature_stop=feature_stop,
                universe_count=universe,
                planned_count=planned,
                offset=offset,
                chunk_range=ChunkRange(chunk_id, min(chunks, UINT32_MAX), chunk_size),
                saturated=saturated,
            )
        )
        offset = min(UINT64_MAX, offset + planned)
        chunk_id = min(UINT32_MAX, chunk_id + min(chunks, UINT32_MAX))
    return tuple(descriptors)


def _time_series_descriptor(
    n_features: int, config: EngineConfig, offset: int
) -> TimeSeriesDescriptor | None:
    if not config.enable_time_series_functions:
        return None
    feature_count = min(
        n_features, max(0, int(config.budget.top_k_features_for_time_series))
    )
    lag_count = len(tuple(config.time_series_lags))
    window_count = len(tuple(config.time_series_windows))
    template_count = lag_count * 4 + window_count * 3
    universe = min(UINT128_MAX, feature_count * template_count)
    planned = min(universe, max(0, int(config.budget.max_time_series_candidates)))
    return TimeSeriesDescriptor(
        0,
        feature_count,
        lag_count,
        window_count,
        template_count,
        universe,
        planned,
        offset,
        universe == UINT128_MAX,
    )


__all__ = [
    "ChunkRange",
    "ContinuousArityDescriptor",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP",
    "ScenarioPlan",
    "TimeSeriesDescriptor",
    "UINT32_MAX",
    "UINT64_MAX",
    "UINT128_MAX",
    "build_scenario_plan",
    "build_scenario_plan_from_shape",
]
