from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

from ..config import ComputeBudget, EngineConfig
from .flags import CompileFlags


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
UINT128_MAX = (1 << 128) - 1
DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP = 1024
DEFAULT_CHUNK_SIZE = 1024


@dataclass(frozen=True)
class ChunkRange:
    first_chunk_id: int
    chunk_count: int
    chunk_size: int

    @property
    def last_chunk_id(self) -> int:
        if self.chunk_count <= 0:
            return self.first_chunk_id - 1
        return self.first_chunk_id + self.chunk_count - 1


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
        return _saturating_u64_add(self.offset, self.planned_count)


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
        return _saturating_u64_add(self.offset, self.planned_count)


@dataclass(frozen=True)
class ScenarioPlan:
    """Compact compile-time scenario metadata."""

    n_samples: int
    n_features: int
    feature_candidate_count: int
    continuous: Tuple[ContinuousArityDescriptor, ...] = field(default_factory=tuple)
    time_series: TimeSeriesDescriptor | None = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def continuous_count(self) -> int:
        return sum(item.planned_count for item in self.continuous)

    @property
    def planned_count(self) -> int:
        count = self.continuous_count
        if self.time_series is not None:
            count = _saturating_u128_add(count, self.time_series.planned_count)
        return count

    @classmethod
    def empty(cls, n_samples: int, n_features: int) -> "ScenarioPlan":
        n_features = int(n_features)
        return cls(
            n_samples=int(n_samples),
            n_features=n_features,
            feature_candidate_count=n_features,
        )


def build_scenario_plan(
    X: Any,
    config: EngineConfig,
    flags: CompileFlags | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ScenarioPlan:
    compile_flags = flags or CompileFlags()
    n_samples = int(getattr(X, "n_samples", 0))
    n_features = int(getattr(X, "n_features", getattr(X, "shape", (0, 0))[1]))
    native_plan = _build_scenario_plan_rust(
        n_samples=n_samples,
        n_features=n_features,
        config=config,
        flags=compile_flags,
        chunk_size=chunk_size,
    )
    if native_plan is not None:
        return native_plan
    return _build_scenario_plan_python(
        n_samples=n_samples,
        n_features=n_features,
        config=config,
        flags=compile_flags,
        chunk_size=chunk_size,
    )


def _build_scenario_plan_rust(
    *,
    n_samples: int,
    n_features: int,
    config: EngineConfig,
    flags: CompileFlags,
    chunk_size: int,
) -> ScenarioPlan | None:
    try:
        from .. import subfunctions
    except ImportError:
        return None
    builder_type = getattr(subfunctions, "CompilePlanBuilder", None)
    if builder_type is None:
        return None

    budget = config.budget
    try:
        native = builder_type().build(
            int(n_samples),
            int(n_features),
            bool(flags.plan),
            int(budget.max_comb_size),
            int(budget.max_combinations_per_k),
            int(budget.top_features_for_higher_k),
            int(budget.max_time_series_candidates),
            int(budget.top_k_features_for_time_series),
            -2 if budget.max_feature_candidate is None else int(budget.max_feature_candidate),
            bool(config.enable_time_series_functions),
            [int(value) for value in config.time_series_lags],
            [int(value) for value in config.time_series_windows],
            int(chunk_size),
        )
    except TypeError as exc:
        if "required positional argument" in str(exc):
            return None
        raise
    return _scenario_plan_from_rust(native)


def _scenario_plan_from_rust(native: Any) -> ScenarioPlan:
    continuous = tuple(
        ContinuousArityDescriptor(
            arity=_int(row["arity"]),
            feature_start=_int(row["feature_start"]),
            feature_stop=_int(row["feature_stop"]),
            universe_count=_int(row["universe_count"]),
            planned_count=_int(row["planned_count"]),
            offset=_int(row["offset"]),
            chunk_range=ChunkRange(
                first_chunk_id=_int(row["first_chunk_id"]),
                chunk_count=_int(row["chunk_count"]),
                chunk_size=_int(row["chunk_size"]),
            ),
            saturated=_bool(row["saturated"]),
        )
        for row in native.continuous_descriptors()
    )
    time_series_row = native.time_series_descriptor()
    return ScenarioPlan(
        n_samples=int(native.n_samples),
        n_features=int(native.n_features),
        feature_candidate_count=int(native.feature_candidate_count),
        continuous=continuous,
        time_series=_time_series_from_rust(time_series_row) if time_series_row is not None else None,
        warnings=tuple(str(item) for item in native.warnings),
    )


def _time_series_from_rust(row: dict[str, str]) -> TimeSeriesDescriptor:
    return TimeSeriesDescriptor(
        feature_start=_int(row["feature_start"]),
        feature_stop=_int(row["feature_stop"]),
        lag_count=_int(row["lag_count"]),
        window_count=_int(row["window_count"]),
        template_count=_int(row["template_count"]),
        universe_count=_int(row["universe_count"]),
        planned_count=_int(row["planned_count"]),
        offset=_int(row["offset"]),
        saturated=_bool(row["saturated"]),
    )


def _int(value: object) -> int:
    return int(str(value))


def _bool(value: object) -> bool:
    return str(value).lower() == "true"


def _build_scenario_plan_python(
    *,
    n_samples: int,
    n_features: int,
    config: EngineConfig,
    flags: CompileFlags,
    chunk_size: int,
) -> ScenarioPlan:
    compile_flags = flags
    if not compile_flags.plan:
        return ScenarioPlan.empty(n_samples, n_features)

    warnings: list[str] = []
    feature_count = _feature_candidate_count(n_features, config.budget, warnings)
    continuous, next_offset, next_chunk_id = _continuous_descriptors(
        n_features=feature_count,
        budget=config.budget,
        chunk_size=chunk_size,
        warnings=warnings,
    )
    time_series, next_offset = _time_series_descriptor(
        n_features=feature_count,
        config=config,
        offset=next_offset,
    )
    del next_offset, next_chunk_id
    return ScenarioPlan(
        n_samples=n_samples,
        n_features=n_features,
        feature_candidate_count=feature_count,
        continuous=continuous,
        time_series=time_series,
        warnings=tuple(warnings),
    )


def _feature_candidate_count(n_features: int, budget: ComputeBudget, warnings: list[str]) -> int:
    max_feature_candidate = budget.max_feature_candidate
    if max_feature_candidate is None:
        return n_features
    if max_feature_candidate >= 0:
        return min(n_features, int(max_feature_candidate))

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
    if n_features > capped:
        warnings.append(
            "max_feature_candidate=-1 without explicit limits; "
            f"applying practical safety cap of {DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP} features."
        )
    return capped


def _continuous_descriptors(
    *,
    n_features: int,
    budget: ComputeBudget,
    chunk_size: int,
    warnings: list[str],
) -> tuple[Tuple[ContinuousArityDescriptor, ...], int, int]:
    descriptors: list[ContinuousArityDescriptor] = []
    offset = 0
    chunk_id = 0
    max_arity = max(1, min(int(budget.max_comb_size), n_features))
    for arity in range(1, max_arity + 1):
        feature_stop = n_features
        if arity > 1:
            feature_stop = min(n_features, max(0, int(budget.top_features_for_higher_k)))
        universe_count, saturated = _saturating_comb(feature_stop, arity)
        planned_count = _apply_count_cap(universe_count, budget.max_combinations_per_k)
        chunk_count = _chunk_count(planned_count, chunk_size)
        if chunk_count > UINT32_MAX:
            warnings.append(
                f"arity={arity} chunk count exceeds uint32; native launch ids will be split by checkpoint."
            )
        descriptors.append(
            ContinuousArityDescriptor(
                arity=arity,
                feature_start=0,
                feature_stop=feature_stop,
                universe_count=universe_count,
                planned_count=planned_count,
                offset=offset,
                chunk_range=ChunkRange(
                    first_chunk_id=chunk_id,
                    chunk_count=min(chunk_count, UINT32_MAX),
                    chunk_size=chunk_size,
                ),
                saturated=saturated,
            )
        )
        offset = _saturating_u64_add(offset, planned_count)
        chunk_id = min(UINT32_MAX, chunk_id + min(chunk_count, UINT32_MAX))
    return tuple(descriptors), offset, chunk_id


def _time_series_descriptor(
    *,
    n_features: int,
    config: EngineConfig,
    offset: int,
) -> tuple[TimeSeriesDescriptor | None, int]:
    if not config.enable_time_series_functions:
        return None, offset
    budget = config.budget
    feature_count = min(n_features, max(0, int(budget.top_k_features_for_time_series)))
    lag_count = len(tuple(config.time_series_lags))
    window_count = len(tuple(config.time_series_windows))
    template_count = lag_count * 4 + window_count * 3
    universe = _saturating_u128_mul(feature_count, template_count)
    planned = min(universe, max(0, int(budget.max_time_series_candidates)))
    descriptor = TimeSeriesDescriptor(
        feature_start=0,
        feature_stop=feature_count,
        lag_count=lag_count,
        window_count=window_count,
        template_count=template_count,
        universe_count=universe,
        planned_count=planned,
        offset=offset,
        saturated=universe >= UINT128_MAX,
    )
    return descriptor, _saturating_u64_add(offset, planned)


def _apply_count_cap(count: int, cap: int) -> int:
    if cap < 0:
        return count
    return min(count, int(cap))


def _chunk_count(count: int, chunk_size: int) -> int:
    if count <= 0:
        return 0
    return (count + max(1, int(chunk_size)) - 1) // max(1, int(chunk_size))


def _saturating_comb(n: int, k: int) -> tuple[int, bool]:
    if n < 0 or k < 0 or k > n:
        return 0, False
    value = math.comb(n, k)
    if value > UINT128_MAX:
        return UINT128_MAX, True
    return value, False


def _saturating_u128_add(left: int, right: int) -> int:
    return min(UINT128_MAX, int(left) + int(right))


def _saturating_u128_mul(left: int, right: int) -> int:
    return min(UINT128_MAX, int(left) * int(right))


def _saturating_u64_add(left: int, right: int) -> int:
    return min(UINT64_MAX, int(left) + int(right))
