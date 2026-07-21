from __future__ import annotations

import atexit
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import math
import os
import sys
import threading
from types import ModuleType
from typing import Any, Iterable, List, Sequence
import warnings

from ._payloads import discover_payloads
from .config import ComputeBudget, EngineConfig
from .errors import V1UnsupportedError
from .reporting import (
    BackendInfo,
    Decision,
    DiagnosticReport,
    NativeContinuousInteractions,
    PermutationResult,
    StabilityResult,
)


_BOUNDARY_MODULE_ENV = "GAFIME_V1_BOUNDARY_MODULE"
_BOUNDARY_MODULES = ("gafime.gafime_py", "gafime_py")
_ANALYZE_CACHE_ENV = "GAFIME_V1_ANALYZE_CACHE_SIZE"
_DEFAULT_ANALYZE_CACHE_SIZE = 2
_F32_MAX = float.fromhex("0x1.fffffep+127")
_PAYLOAD_SELECTION_ENV_VARS = (
    "GAFIME_CUDA_V1_LIB",
    "GAFIME_ROCM_V1_LIB",
    "GAFIME_METAL_V1_LIB",
    "GAFIME_METAL_V1_METALLIB",
)


@dataclass
class _AnalyzeCacheEntry:
    artifact: Any
    target_digest: bytes
    lock: threading.RLock = field(default_factory=threading.RLock)
    owner_thread: threading.Thread = field(default_factory=threading.current_thread)
    owner_thread_id: int = field(default_factory=threading.get_ident)


class _AnalyzeCacheState:
    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[Any, ...], _AnalyzeCacheEntry] = OrderedDict()

    def close(self, *, owner_thread_teardown: bool = False) -> None:
        entries = list(self.entries.values())
        self.entries.clear()
        _close_analyze_cache_entries(
            entries,
            suppress=True,
            owner_thread_teardown=owner_thread_teardown,
        )

    def __del__(self) -> None:
        try:
            # CPython clears threading.local values on the owning OS thread
            # after replacing its Thread object with a _DummyThread.
            self.close(owner_thread_teardown=True)
        except BaseException:
            pass


class _AnalyzeCacheLocal(threading.local):
    def __init__(self) -> None:
        self.state = _AnalyzeCacheState()


@dataclass
class _CachedCoercedInput:
    features: Any
    target: Any
    rows: int
    cols: int
    feature_names: List[str]
    feature_digest: bytes | None
    target_digest: bytes | None

    def feature_list(self) -> List[float]:
        return _f32_storage_to_list(self.features)

    def target_list(self) -> List[float]:
        return _f32_storage_to_list(self.target)

    def feature_bytes(self) -> bytes:
        return _f32_storage_to_le_bytes(self.features)

    def target_bytes(self) -> bytes:
        return _f32_storage_to_le_bytes(self.target)


_ANALYZE_CACHE_LOCAL = _AnalyzeCacheLocal()


def _current_analyze_cache() -> OrderedDict[tuple[Any, ...], _AnalyzeCacheEntry]:
    return _ANALYZE_CACHE_LOCAL.state.entries


def analyze_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    if _continuous_analyze_cache_enabled(config):
        return _analyze_continuous_with_resident_cache(config, X, y, feature_names)
    if (
        not config.enable_time_series_functions
        and not config.enable_decision_path_functions
    ):
        report = _analyze_continuous_one_shot(config, X, y, feature_names)
        if report is not None:
            return report
    artifact = compile_with_v1_boundary(config, X, y, feature_names)
    try:
        return artifact.analyze()
    finally:
        artifact.close()


def _analyze_continuous_one_shot(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
) -> DiagnosticReport | None:
    """Execute stateless eager analysis without constructing a Python artifact.

    Older custom boundary modules may expose only ``compile_continuous``; they
    retain the compatibility path above. The shipped Rust boundary exposes this
    dedicated one-shot entrypoint.
    """
    boundary = _load_boundary_for_backend(config.backend)
    analyze_buffers = getattr(boundary, "analyze_continuous_buffers", None)
    if callable(analyze_buffers):
        coerced = _coerce_row_major_f32_for_cache(X, y, feature_names)
        native_report = analyze_buffers(
            _config_payload(config),
            coerced.feature_bytes(),
            coerced.target_bytes(),
            rows=coerced.rows,
            cols=coerced.cols,
        )
        return _diagnostic_from_native_report(
            config,
            native_report,
            coerced.feature_names,
            _continuous_cap_warnings(config, coerced.cols),
        )
    analyze = getattr(boundary, "analyze_continuous", None)
    if not callable(analyze):
        return None
    nested_shape = _native_nested_shape(X, y, feature_names)
    analyze_rows = getattr(boundary, "analyze_continuous_rows", None)
    if nested_shape is not None and callable(analyze_rows):
        _, cols, names = nested_shape
        try:
            native_report = analyze_rows(_config_payload(config), X, y)
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                "X and y must contain values representable as fp32."
            ) from exc
        return _diagnostic_from_native_report(
            config,
            native_report,
            names,
            _continuous_cap_warnings(config, cols),
        )
    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    native_report = analyze(
        _config_payload(config),
        features,
        target,
        rows=rows,
        cols=cols,
    )
    return _diagnostic_from_native_report(
        config,
        native_report,
        names,
        _continuous_cap_warnings(config, cols),
    )


def _continuous_analyze_cache_enabled(config: EngineConfig) -> bool:
    capacity = _analyze_cache_capacity()
    if capacity <= 0:
        _evict_analyze_cache_if_disabled()
        return False
    if config.enable_time_series_functions or config.enable_decision_path_functions:
        return False
    if not bool(config.budget.keep_in_vram):
        return False
    return True


def _analyze_continuous_with_resident_cache(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
) -> DiagnosticReport:
    boundary = _load_boundary_for_backend(config.backend)
    coerced = _coerce_row_major_f32_for_cache(X, y, feature_names, include_digests=True)
    assert coerced.feature_digest is not None
    assert coerced.target_digest is not None
    payload = _config_payload(config)
    cache_payload = payload
    if config.random_seed is None:
        cache_payload = {**payload, "random_seed": None}
    cache_key = (
        id(boundary),
        str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
        _payload_selection_identity(),
        _freeze_cache_value(cache_payload),
        tuple(coerced.feature_names),
        int(coerced.rows),
        int(coerced.cols),
        coerced.feature_digest,
    )
    cache = _current_analyze_cache()

    entry = cache.get(cache_key)
    if entry is not None:
        cache.move_to_end(cache_key)
    _close_analyze_cache_entries(_prune_analyze_cache(cache))

    if entry is not None:
        try:
            with entry.lock:
                if entry.owner_thread is not threading.current_thread():
                    raise RuntimeError(
                        "resident GAFIME artifacts are thread-affine and cannot "
                        "be reused from another thread."
                    )
                if getattr(entry.artifact, "_closed", False):
                    entry = None
                else:
                    if entry.target_digest != coerced.target_digest:
                        entry.artifact.update_target(coerced.target_list())
                        entry.target_digest = coerced.target_digest
                    return entry.artifact.analyze()
        except BaseException:
            if cache.get(cache_key) is entry:
                cache.pop(cache_key)
            if entry is not None:
                _close_analyze_cache_entries([entry], suppress=True)
            raise

        if entry is None:
            stale = cache.get(cache_key)
            if stale is not None and getattr(stale.artifact, "_closed", False):
                cache.pop(cache_key)

    compile_buffers = getattr(boundary, "compile_continuous_buffers", None)
    if callable(compile_buffers):
        handle = compile_buffers(
            payload,
            coerced.feature_bytes(),
            coerced.target_bytes(),
            rows=coerced.rows,
            cols=coerced.cols,
        )
    else:
        handle = boundary.compile_continuous(
            payload,
            coerced.feature_list(),
            coerced.target_list(),
            rows=coerced.rows,
            cols=coerced.cols,
        )
    artifact = NativeCompiledGafime(
        config=config,
        feature_names=coerced.feature_names,
        native_handle=handle,
        boundary_name=str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
        export=False,
        warnings=_continuous_cap_warnings(config, coerced.cols),
        _scenario_feature_count=coerced.cols,
    )
    try:
        report = artifact.analyze()
    except BaseException:
        try:
            artifact.close()
        except BaseException:
            pass
        raise

    new_entry = _AnalyzeCacheEntry(
        artifact=artifact,
        target_digest=coerced.target_digest,
    )
    previous = cache.pop(cache_key, None)
    cache[cache_key] = new_entry
    evicted = _prune_analyze_cache(cache)
    if previous is not None:
        _close_analyze_cache_entry(previous)
    _close_analyze_cache_entries(evicted)
    return report


def _close_analyze_cache_entry(
    entry: _AnalyzeCacheEntry, *, owner_thread_teardown: bool = False
) -> None:
    if owner_thread_teardown:
        if entry.owner_thread_id != threading.get_ident():
            raise RuntimeError(
                "resident GAFIME artifacts must be closed by their owning thread."
            )
    elif entry.owner_thread is not threading.current_thread():
        raise RuntimeError(
            "resident GAFIME artifacts must be closed by their owning thread."
        )
    with entry.lock:
        if owner_thread_teardown:
            close = getattr(entry.artifact, "_close_from_owner_thread_teardown", None)
            if callable(close):
                close()
                return
        entry.artifact.close()


def _close_analyze_cache_entries(
    entries: Iterable[_AnalyzeCacheEntry],
    *,
    suppress: bool = False,
    owner_thread_teardown: bool = False,
) -> None:
    first_error: BaseException | None = None
    for entry in entries:
        try:
            _close_analyze_cache_entry(
                entry, owner_thread_teardown=owner_thread_teardown
            )
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress:
        raise first_error


def _analyze_cache_capacity() -> int:
    raw = os.environ.get(_ANALYZE_CACHE_ENV)
    if raw is None:
        return _DEFAULT_ANALYZE_CACHE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_ANALYZE_CACHE_SIZE


def _prune_analyze_cache(
    cache: OrderedDict[tuple[Any, ...], _AnalyzeCacheEntry],
) -> List[_AnalyzeCacheEntry]:
    evicted: List[_AnalyzeCacheEntry] = []
    capacity = _analyze_cache_capacity()
    while len(cache) > capacity:
        _, entry = cache.popitem(last=False)
        evicted.append(entry)
    return evicted


def _evict_analyze_cache_if_disabled() -> None:
    if _analyze_cache_capacity() > 0:
        return
    cache = _current_analyze_cache()
    entries = list(cache.values())
    cache.clear()
    _close_analyze_cache_entries(entries)


def _f32_array_digest(values: array) -> bytes:
    if sys.byteorder == "little":
        return _f32_buffer_digest(len(values), memoryview(values).cast("B"))
    packed = array("f", values)
    packed.byteswap()
    return _f32_buffer_digest(len(packed), memoryview(packed).cast("B"))


def _f32_buffer_digest(count: int, data: bytes | memoryview) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(int(count).to_bytes(8, "little", signed=False))
    digest.update(data)
    return digest.digest()


def _append_f32(values: array, value: object, label: str) -> None:
    out = _finite_f32(value, label)
    try:
        values.append(out)
    except OverflowError as exc:
        raise ValueError(f"{label} contains a value outside fp32 range.") from exc
    if math.isfinite(out) and not math.isfinite(values[-1]):
        values.pop()
        raise ValueError(
            f"{label} contains a value that is non-finite after fp32 conversion."
        )


def _continuous_cap_warnings(config: EngineConfig, cols: int) -> List[str]:
    budget = config.budget
    candidate_cols = _feature_candidate_count(cols, budget)
    max_per_arity = int(budget.max_combinations_per_k)
    warnings: List[str] = []
    if (
        budget.max_feature_candidate == -1
        and candidate_cols < cols
        and candidate_cols == 1_024
    ):
        warnings.append(
            "max_feature_candidate=-1 without explicit limits; "
            "applying practical safety cap of 1024 features."
        )
    max_arity = int(budget.max_comb_size)
    top_features = int(budget.top_features_for_higher_k)
    if top_features < 1 and max_arity > 1:
        warnings.append(
            "top_features_for_higher_k < 1; higher-order combos will be empty."
        )
    if max_arity > candidate_cols:
        warnings.append("max_comb_size exceeds feature count; will cap to n_features.")
    if candidate_cols > max_per_arity:
        warnings.append("Unary combinations capped by max_combinations_per_k.")

    top_features = max(0, top_features)
    if max_arity < 2 or max_per_arity < 1 or top_features < 2:
        return warnings

    screened_features = min(candidate_cols, max_per_arity, top_features)
    for arity in range(2, min(max_arity, screened_features) + 1):
        if math.comb(screened_features, arity) >= max_per_arity:
            warnings.append(f"k={arity} combinations capped by max_combinations_per_k.")
    return warnings


def _feature_candidate_count(cols: int, budget: ComputeBudget) -> int:
    value = budget.max_feature_candidate
    if value is None:
        return cols
    value = int(value)
    if value < -1:
        raise ValueError(
            "max_feature_candidate must be >= 0 or -1 for power-user mode."
        )
    if value >= 0:
        return min(cols, value)
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
    return cols if explicit_limits else min(cols, 1_024)


def _payload_selection_identity() -> tuple[tuple[str, str | None], ...]:
    return tuple((name, os.environ.get(name)) for name in _PAYLOAD_SELECTION_ENV_VARS)


def _freeze_cache_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_cache_value(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze_cache_value(item) for item in value)
    return value


def _clear_analyze_cache_for_tests() -> None:
    _ANALYZE_CACHE_LOCAL.state.close()


atexit.register(_clear_analyze_cache_for_tests)


def analyze_time_series_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    """Analyze the time_series family through the native expand+mine path."""
    boundary = _load_boundary_for_backend(config.backend)
    if not hasattr(boundary, "analyze_time_series"):
        raise V1UnsupportedError("native boundary lacks analyze_time_series")
    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    payload = _config_payload(replace(config, enable_time_series_functions=False))
    native_report, all_names = boundary.analyze_time_series(
        payload,
        features,
        target,
        rows,
        cols,
        names,
        [int(lag) for lag in config.time_series_lags],
        [int(window) for window in config.time_series_windows],
        True,
    )
    return _diagnostic_from_native_report(
        config,
        native_report,
        list(all_names),
        [f"time_series expanded {cols} base features to {len(all_names)}."],
        generated_feature_start=_generated_feature_start(
            config, cols, len(all_names)
        ),
    )


def analyze_decision_path_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    """Analyze the decision_path family through native discovery and scoring."""
    _validate_decision_path_config(config)
    _require_decision_path_permutation_support(config)
    boundary = _load_boundary_for_backend(config.backend)
    if not hasattr(boundary, "analyze_decision_path"):
        raise V1UnsupportedError("native boundary lacks analyze_decision_path")
    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    payload = _config_payload(replace(config, enable_decision_path_functions=False))
    native_report, all_names = boundary.analyze_decision_path(
        payload,
        features,
        target,
        rows,
        cols,
        names,
        int(config.decision_path_max_depth),
        int(config.decision_path_rounds),
        int(config.decision_path_max_paths),
        int(config.decision_path_max_bins),
        int(config.decision_path_min_leaf),
        float(config.decision_path_learning_rate),
    )
    return _diagnostic_from_native_report(
        config,
        native_report,
        list(all_names),
        [
            f"decision_path discovered {len(all_names) - cols} conjunction path(s) from {cols} features."
        ],
        generated_feature_start=_generated_feature_start(
            config, cols, len(all_names)
        ),
    )


def compile_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
    *,
    flags=None,
) -> "NativeCompiledGafime":
    plan, graph, export = _compile_flag_values(flags)
    _validate_graph_request(config, graph)
    if config.enable_decision_path_functions:
        _validate_decision_path_config(config)
        _require_decision_path_permutation_support(config)
    compile_flags = {
        "plan": plan,
        "graph": graph,
        "export": export,
    }

    boundary = _load_boundary_for_backend(config.backend)
    if (
        not config.enable_time_series_functions
        and not config.enable_decision_path_functions
    ):
        compile_buffers = getattr(boundary, "compile_continuous_buffers", None)
        if callable(compile_buffers):
            coerced = _coerce_row_major_f32_for_cache(X, y, feature_names)
            payload = _config_payload(config)
            payload["compile_flags"] = compile_flags
            handle = compile_buffers(
                payload,
                coerced.feature_bytes(),
                coerced.target_bytes(),
                rows=coerced.rows,
                cols=coerced.cols,
            )
            if graph:
                _require_native_graph_activation(handle)
            return NativeCompiledGafime(
                config=config,
                feature_names=coerced.feature_names,
                native_handle=handle,
                boundary_name=str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
                plan_enabled=plan,
                graph_requested=graph,
                export=export,
                warnings=_continuous_cap_warnings(config, coerced.cols),
                _scenario_feature_count=coerced.cols,
            )
        nested_shape = _native_nested_shape(X, y, feature_names)
        compile_rows = getattr(boundary, "compile_continuous_rows", None)
        if nested_shape is not None and callable(compile_rows):
            _, cols, names = nested_shape
            payload = _config_payload(config)
            payload["compile_flags"] = compile_flags
            try:
                handle = compile_rows(payload, X, y)
            except (TypeError, OverflowError) as exc:
                raise ValueError(
                    "X and y must contain values representable as fp32."
                ) from exc
            if graph:
                _require_native_graph_activation(handle)
            return NativeCompiledGafime(
                config=config,
                feature_names=names,
                native_handle=handle,
                boundary_name=str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
                plan_enabled=plan,
                graph_requested=graph,
                export=export,
                warnings=_continuous_cap_warnings(config, cols),
                _scenario_feature_count=cols,
            )

    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    generated_feature_start = None
    if config.enable_time_series_functions:
        if not hasattr(boundary, "compile_time_series"):
            raise V1UnsupportedError("native boundary lacks compile_time_series")
        payload = _config_payload(replace(config, enable_time_series_functions=False))
        payload["compile_flags"] = compile_flags
        handle, all_names = boundary.compile_time_series(
            payload,
            features,
            target,
            rows,
            cols,
            names,
            [int(lag) for lag in config.time_series_lags],
            [int(window) for window in config.time_series_windows],
            True,
        )
        warnings = [f"time_series expanded {cols} base features to {len(all_names)}."]
        names = list(all_names)
        generated_feature_start = _generated_feature_start(config, cols, len(names))
    elif config.enable_decision_path_functions:
        if not hasattr(boundary, "compile_decision_path"):
            raise V1UnsupportedError("native boundary lacks compile_decision_path")
        payload = _config_payload(replace(config, enable_decision_path_functions=False))
        payload["compile_flags"] = compile_flags
        handle, all_names = boundary.compile_decision_path(
            payload,
            features,
            target,
            rows,
            cols,
            names,
            int(config.decision_path_max_depth),
            int(config.decision_path_rounds),
            int(config.decision_path_max_paths),
            int(config.decision_path_max_bins),
            int(config.decision_path_min_leaf),
            float(config.decision_path_learning_rate),
        )
        warnings = [
            f"decision_path discovered {len(all_names) - cols} conjunction path(s) from {cols} features."
        ]
        names = list(all_names)
        generated_feature_start = _generated_feature_start(config, cols, len(names))
    else:
        payload = _config_payload(config)
        payload["compile_flags"] = compile_flags
        handle = boundary.compile_continuous(
            payload,
            features,
            target,
            rows=rows,
            cols=cols,
        )
        warnings = _continuous_cap_warnings(config, cols)
    if graph:
        _require_native_graph_activation(handle)
    return NativeCompiledGafime(
        config=config,
        feature_names=names,
        native_handle=handle,
        boundary_name=str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
        plan_enabled=plan,
        graph_requested=graph,
        export=export,
        warnings=warnings,
        _scenario_feature_count=cols,
        _generated_feature_start=generated_feature_start,
    )


_METRIC_IDS = {"pearson": 1, "spearman": 2, "mutual_info": 3, "r2": 4}


def _require_decision_path_permutation_support(config: EngineConfig) -> None:
    if int(config.permutation_tests) <= 0:
        return
    raise V1UnsupportedError(
        "decision-path permutation significance requires path rediscovery for "
        "every permuted target and is not supported by this boundary."
    )


def _validate_decision_path_config(config: EngineConfig) -> None:
    if config.decision_path_max_depth < 1:
        raise ValueError("decision_path_max_depth must be >= 1.")
    if config.decision_path_rounds < 1:
        raise ValueError("decision_path_rounds must be >= 1.")
    if config.decision_path_max_paths < 1:
        raise ValueError("decision_path_max_paths must be >= 1.")
    if config.decision_path_max_bins < 0:
        raise ValueError(
            "decision_path_max_bins must be >= 0; use 0 for exhaustive splits."
        )
    if config.decision_path_min_leaf < 1:
        raise ValueError("decision_path_min_leaf must be >= 1.")
    if config.decision_path_learning_rate <= 0:
        raise ValueError("decision_path_learning_rate must be > 0.")
    if config.decision_path_top_k_features < 0:
        raise ValueError("decision_path_top_k_features must be >= 0.")


def _decision_path_params_for_combo(
    native_report: object,
    combo: Sequence[int],
    native_candidate_id: int,
    candidate_id: str,
) -> dict[str, object]:
    if len(combo) != 1:
        return {}
    getter = getattr(native_report, "decision_path_params", None)
    if not callable(getter):
        return {}
    native_params = getter(int(combo[0]))
    if native_params is None:
        return {}
    params = dict(native_params)
    params["native_candidate_id"] = int(native_candidate_id)
    params["candidate_id"] = str(candidate_id)
    return params


class _NativeDecisionPathInteractions(NativeContinuousInteractions):
    def _result_from_components(
        self, combo_values, metric_values, native_candidate_id, *, coerce: bool
    ):
        result = super()._result_from_components(
            combo_values,
            metric_values,
            native_candidate_id,
            coerce=coerce,
        )
        if result.family != "decision_path":
            return result
        params = _decision_path_params_for_combo(
            self._native_report,
            result.combo,
            int(native_candidate_id),
            result.candidate_id,
        )
        return replace(result, params=params)


def _native_interactions(
    config: EngineConfig,
    native_report: object,
    feature_names: Sequence[str],
    *,
    generated_feature_start: int | None = None,
) -> NativeContinuousInteractions:
    interaction_type = (
        _NativeDecisionPathInteractions
        if config.enable_decision_path_functions
        else NativeContinuousInteractions
    )
    return interaction_type(
        native_report,
        feature_names,
        config.metric_names,
        generated_feature_start=generated_feature_start,
        generated_family=_generated_family(config)
        if generated_feature_start is not None
        else None,
    )


def _generated_family(config: EngineConfig) -> str | None:
    if config.enable_decision_path_functions:
        return "decision_path"
    if config.enable_time_series_functions:
        return "time_series"
    return None


def _generated_feature_start(
    config: EngineConfig,
    source_feature_count: int,
    expanded_feature_count: int,
) -> int | None:
    if _generated_family(config) is None:
        return None
    base_count = _feature_candidate_count(source_feature_count, config.budget)
    if base_count == 0 or expanded_feature_count <= base_count:
        return None
    return base_count


def _family_for_combo(
    config: EngineConfig,
    combo: Sequence[int],
    generated_feature_start: int | None,
) -> str:
    generated_family = _generated_family(config)
    if generated_family is not None and generated_feature_start is not None:
        if any(index >= generated_feature_start for index in combo):
            return generated_family
    return "interaction"


def _diagnostic_from_native_report(
    config: EngineConfig,
    native_report: Any,
    feature_names: Sequence[str],
    warnings: Sequence[str],
    *,
    generated_feature_start: int | None = None,
) -> DiagnosticReport:
    names = list(feature_names)
    interactions = _native_interactions(
        config,
        native_report,
        names,
        generated_feature_start=generated_feature_start,
    )
    stability, permutations, decision = _significance_from_native(
        native_report,
        names,
        config,
        generated_feature_start=generated_feature_start,
    )
    return DiagnosticReport(
        config=config,
        feature_names=names,
        interactions=interactions,
        stability=stability,
        permutations=permutations,
        warnings=list(warnings),
        decision=decision,
        backend=_backend_info(native_report),
    )


def analyze_arrow_with_v1_boundary(
    config: EngineConfig,
    feature_frame: object,
    target_frame: object,
    feature_names: Sequence[str],
) -> DiagnosticReport:
    """Analyze Arrow-backed frames without changing requested engine semantics.

    The low-level Arrow entrypoint is a CPU/no-significance convenience API. Use
    it only when that is exactly what the configuration requests. All other
    configurations route through the normal configured boundary using the
    frame's row iterator; this may materialize GAFIME's owned fp32 input buffer,
    but it cannot silently discard backend, family, MI, or significance options.
    """
    target = _validate_arrow_target_frame(target_frame)
    boundary = _load_boundary_for_backend(config.backend)
    if _raw_arrow_config_supported(config) and hasattr(
        boundary, "analyze_continuous_arrow"
    ):
        try:
            metric_ids = [_METRIC_IDS[str(name)] for name in config.metric_names]
        except KeyError as exc:
            raise ValueError(f"unsupported metric for Arrow ingest: {exc}") from exc
        native_report = boundary.analyze_continuous_arrow(
            feature_frame,
            target_frame,
            max_arity=int(config.budget.max_comb_size),
            max_combinations_per_k=int(config.budget.max_combinations_per_k),
            metric_ids=metric_ids,
        )
        report = _diagnostic_from_native_report(
            config, native_report, feature_names, []
        )
        report.decision = Decision(
            bool(report.interactions), "v1 continuous Arrow ingest path executed."
        )
        return report

    iter_rows = getattr(feature_frame, "iter_rows", None)
    if not callable(iter_rows):
        raise V1UnsupportedError(
            "configured Arrow analysis requires a frame exposing iter_rows(); "
            "the raw Arrow shortcut cannot honor this EngineConfig."
        )
    rows = iter_rows()
    if config.enable_time_series_functions:
        return analyze_time_series_with_v1_boundary(config, rows, target, feature_names)
    if config.enable_decision_path_functions:
        return analyze_decision_path_with_v1_boundary(
            config, rows, target, feature_names
        )
    return analyze_with_v1_boundary(config, rows, target, feature_names)


@dataclass
class NativeCompiledGafime:
    config: EngineConfig
    feature_names: List[str]
    native_handle: object
    boundary_name: str
    plan_enabled: bool = True
    graph_requested: bool = False
    export: bool = False
    warnings: List[str] = field(default_factory=list)
    _scenario_feature_count: int | None = field(default=None, repr=False)
    _generated_feature_start: int | None = field(default=None, repr=False)
    _closed: bool = False
    _last_report: DiagnosticReport | None = None
    _native_report: Any = None
    _graph_replayed: bool = False
    _scenario_plan: Any = field(default=None, init=False, repr=False)
    _owner_thread: threading.Thread = field(
        default_factory=threading.current_thread, init=False, repr=False
    )
    _owner_thread_id: int = field(
        default_factory=threading.get_ident, init=False, repr=False
    )

    @property
    def flags(self):
        from .compile.flags import CompileFlags

        return CompileFlags(
            plan=self.plan_enabled,
            graph=self.graph_requested,
            export=self.export,
        )

    @property
    def backend(self) -> BackendInfo:
        self._ensure_open()
        return _backend_info(self.native_handle)

    @property
    def scenario_plan(self) -> object | None:
        self._ensure_open()
        if not self.plan_enabled:
            return None
        if self.boundary_name != "gafime-py":
            # Custom v1 boundaries historically exposed their own compiled
            # artifact as the plan object. Preserve that identity contract.
            return self.native_handle
        if self._scenario_plan is None:
            from .compile.scenario import build_scenario_plan_from_shape

            feature_count = (
                len(self.feature_names)
                if self._scenario_feature_count is None
                else self._scenario_feature_count
            )
            self._scenario_plan = build_scenario_plan_from_shape(
                n_samples=int(getattr(self.native_handle, "rows", 0)),
                n_features=feature_count,
                config=self.config,
                flags=self.flags,
            )
        return self._scenario_plan

    @classmethod
    def from_engine(
        cls,
        engine,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
        *,
        flags=None,
    ) -> "NativeCompiledGafime":
        return engine.compile(X, y, feature_names=feature_names, flags=flags)

    @property
    def graph_replayed(self) -> bool:
        self._ensure_open()
        return self._graph_replayed

    @property
    def continuous_metric_cache_hits(self) -> int:
        self._ensure_open()
        return int(getattr(self.native_handle, "continuous_metric_cache_hits", 0))

    @property
    def continuous_metric_cache_builds(self) -> int:
        self._ensure_open()
        return int(getattr(self.native_handle, "continuous_metric_cache_builds", 0))

    @property
    def candidate_table_cache_hits(self) -> int:
        self._ensure_open()
        return int(getattr(self.native_handle, "candidate_table_cache_hits", 0))

    @property
    def exports(self):
        """Compatibility view of the v0.5 compiled export handles.

        The v1 boundary exposes the owning compiled artifact and compact native
        result report, but no independent candidate-table handle.
        """
        self._ensure_open()
        warnings.warn(
            "NativeCompiledGafime.exports is deprecated; use export_arrow() for "
            "the compact result table.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.export:
            raise V1UnsupportedError(
                "Compiled export handles are not available unless "
                "CompileFlags(export=True) is set."
            )
        from .compile.exports import ExportHandles

        backend = self.backend
        return ExportHandles(
            backend_name=str(backend.selected_backend or backend.name),
            feature_matrix_handle=self.native_handle,
            result_table_handle=self._native_report,
            candidate_table_handle=None,
        )

    def analyze(self) -> DiagnosticReport:
        self._ensure_open()
        self._graph_replayed = False
        if self.config.random_seed is None:
            reseed = getattr(self.native_handle, "reseed", None)
            if not callable(reseed):
                raise V1UnsupportedError(
                    "random_seed=None requires native compiled artifact reseed(seed) support."
                )
            try:
                reseed(_fresh_random_seed())
            except BaseException:
                self._close_after_native_failure()
                raise
        try:
            native_report = self.native_handle.analyze()
        except BaseException:
            self._close_after_native_failure()
            raise
        if self.graph_requested:
            replayed = _native_graph_replayed(native_report, self.native_handle)
            if replayed is not True:
                raise V1UnsupportedError(
                    "CompileFlags(graph=True) was requested, but the native report "
                    "did not confirm graph replay."
                )
            self._graph_replayed = True
        self._native_report = native_report
        interactions = _native_interactions(
            self.config,
            native_report,
            self.feature_names,
            generated_feature_start=self._generated_feature_start,
        )
        stability, permutations, decision = _significance_from_native(
            native_report,
            self.feature_names,
            self.config,
            generated_feature_start=self._generated_feature_start,
        )
        report = DiagnosticReport(
            config=self.config,
            feature_names=list(self.feature_names),
            interactions=interactions,
            stability=stability,
            permutations=permutations,
            warnings=list(self.warnings),
            decision=decision,
            backend=_backend_info(self.native_handle),
        )
        self._last_report = report
        return report

    def update_target(self, y: Iterable[float]) -> "NativeCompiledGafime":
        """Resident-session reuse: replace the target and re-use the resident
        matrix on the next analyze() — the features stay uploaded (on GPU) or held
        (on CPU), so only y crosses the boundary. Returns self for chaining."""
        self._ensure_open()
        if self.config.enable_decision_path_functions:
            raise V1UnsupportedError(
                "update_target is unsupported for compiled target-derived "
                "decision-path features; recompile with the new target."
            )
        target = _coerce_target_f32_storage(y)
        update_buffer = getattr(self.native_handle, "update_target_buffer", None)
        try:
            if callable(update_buffer):
                update_buffer(_f32_storage_to_le_bytes(target))
            else:
                self.native_handle.update_target(_f32_storage_to_list(target))
        except BaseException:
            self._close_after_native_failure()
            raise
        self._native_report = None
        self._last_report = None
        self._graph_replayed = False
        return self

    def __arrow_c_array__(self, requested_schema=None):
        """Zero-copy Arrow C Data Interface export of the compact result table.
        Requires the artifact to have been compiled with CompileFlags(export=True).
        Consumers such as Polars >= 1.3 and pyarrow read the (schema, array)
        capsule pair with no copy; arrow-rs owns the FFI release callbacks."""
        if not self.export:
            raise V1UnsupportedError(
                "result export requires compiling with CompileFlags(export=True)."
            )
        self._ensure_open()
        if self._native_report is None:
            self.analyze()
        if self._native_report is None:  # analyze() always sets it; guard for typing
            raise RuntimeError("native report is unavailable for export.")
        return self._native_report.__arrow_c_array__(requested_schema)

    def export_arrow(self, requested_schema=None):
        """Explicit alias for the Arrow C Data Interface export (see
        ``__arrow_c_array__``)."""
        return self.__arrow_c_array__(requested_schema)

    def close(self) -> None:
        self._ensure_owner_thread()
        self._close_native()

    def _close_from_owner_thread_teardown(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "NativeCompiledGafime must be closed by its owning thread."
            )
        self._close_native()

    def _close_native(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._native_report = None
        self._last_report = None
        self._graph_replayed = False
        self._scenario_plan = None
        close = getattr(self.native_handle, "close", None)
        if close is not None:
            close()

    def _close_after_native_failure(self) -> None:
        if not getattr(self.native_handle, "closed", False):
            return
        try:
            self._close_native()
        except BaseException:
            pass

    def _ensure_open(self) -> None:
        self._ensure_owner_thread()
        if self._closed:
            raise RuntimeError("NativeCompiledGafime is closed.")

    def _ensure_owner_thread(self) -> None:
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError(
                "NativeCompiledGafime is thread-affine; use and close it on the "
                "thread where it was compiled."
            )


def _load_boundary_for_backend(backend: str | None) -> ModuleType:
    discover_payloads(backend)
    return _load_boundary()


def _load_boundary() -> ModuleType:
    explicit = os.environ.get(_BOUNDARY_MODULE_ENV)
    names = (explicit,) if explicit else _BOUNDARY_MODULES
    failures: List[str] = []
    for name in names:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            failures.append(f"{name}: {exc}")
            continue
        if not hasattr(module, "compile_continuous"):
            failures.append(f"{name}: missing compile_continuous")
            continue
        return module
    detail = "; ".join(failures) if failures else "no module names configured"
    raise RuntimeError(f"GAFIME v1 Python boundary is unavailable: {detail}")


def _coerce_row_major_f32(
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
) -> tuple[List[float], List[float], int, int, List[str]]:
    rows = _matrix_rows(X)
    if not rows:
        raise ValueError("X must contain at least one sample.")
    cols = len(rows[0])
    if cols == 0:
        raise ValueError("X must contain at least one feature.")

    flat: List[float] = []
    for row_idx, row in enumerate(rows):
        if len(row) != cols:
            raise ValueError(f"X row {row_idx} has length {len(row)}; expected {cols}.")
        flat.extend(_finite_f32(value, f"X[{row_idx}]") for value in row)

    target = [_finite_f32(value, "y") for value in _sequence(y, "y")]
    if len(target) != len(rows):
        raise ValueError("X and y must have the same number of samples.")

    if feature_names is None:
        names = [f"f{i}" for i in range(cols)]
    else:
        names = [str(name) for name in feature_names]
        if len(names) != cols:
            raise ValueError("feature_names length must match X's feature count.")

    return flat, target, len(rows), cols, names


def _native_nested_shape(
    X: object,
    y: object,
    feature_names: Iterable[str] | None,
) -> tuple[int, int, List[str]] | None:
    """Validate nested sequences while preserving them for direct PyO3 ingest."""
    if not isinstance(X, (list, tuple)) or not isinstance(y, (list, tuple)):
        return None
    if not X:
        raise ValueError("X must contain at least one sample.")
    first_row = X[0]
    if not isinstance(first_row, (list, tuple)):
        return None
    cols = len(first_row)
    if cols == 0:
        raise ValueError("X must contain at least one feature.")
    for row_idx, row in enumerate(X):
        if not isinstance(row, (list, tuple)):
            return None
        if len(row) != cols:
            raise ValueError(f"X row {row_idx} has length {len(row)}; expected {cols}.")
        for value in row:
            _finite_f32(value, f"X[{row_idx}]")
    if len(y) != len(X):
        raise ValueError("X and y must have the same number of samples.")
    for value in y:
        _finite_f32(value, "y")
    return len(X), cols, _coerce_feature_names(feature_names, cols)


def _coerce_row_major_f32_for_cache(
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
    *,
    include_digests: bool = False,
) -> _CachedCoercedInput:
    numpy_coerced = _try_coerce_numpy_row_major_f32_for_cache(
        X, y, feature_names, include_digests=include_digests
    )
    if numpy_coerced is not None:
        return numpy_coerced

    if hasattr(X, "to_dicts") and hasattr(X, "columns"):
        columns = [str(col) for col in X.columns]
        row_iterable = ([row[col] for col in columns] for row in X.to_dicts())
    else:
        row_iterable = _sequence(X, "X")

    features = array("f")
    rows = 0
    cols: int | None = None
    for row_idx, row in enumerate(row_iterable):
        row_values = _sequence(row, f"X[{row_idx}]")
        if cols is None:
            cols = len(row_values)
            if cols == 0:
                raise ValueError("X must contain at least one feature.")
        elif len(row_values) != cols:
            raise ValueError(
                f"X row {row_idx} has length {len(row_values)}; expected {cols}."
            )
        for value in row_values:
            _append_f32(features, value, f"X[{row_idx}]")
        rows += 1

    if rows == 0:
        raise ValueError("X must contain at least one sample.")
    if cols is None:
        raise ValueError("X must contain at least one feature.")

    target = array("f")
    for value in _sequence(y, "y"):
        _append_f32(target, value, "y")
    if len(target) != rows:
        raise ValueError("X and y must have the same number of samples.")

    names = _coerce_feature_names(feature_names, cols)
    return _CachedCoercedInput(
        features=features,
        target=target,
        rows=rows,
        cols=cols,
        feature_names=names,
        feature_digest=_f32_array_digest(features) if include_digests else None,
        target_digest=_f32_array_digest(target) if include_digests else None,
    )


def _try_coerce_numpy_row_major_f32_for_cache(
    X: object,
    y: object,
    feature_names: Iterable[str] | None,
    *,
    include_digests: bool,
) -> _CachedCoercedInput | None:
    if hasattr(X, "to_dicts") and hasattr(X, "columns"):
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    try:
        source_features = np.asarray(X)
        source_target = np.asarray(y)
    except (TypeError, ValueError, OverflowError):
        return None
    if source_features.ndim != 2 or source_target.ndim != 1:
        return None
    if (
        np.issubdtype(source_features.dtype, np.complexfloating)
        or np.issubdtype(source_target.dtype, np.complexfloating)
        or not (
            np.issubdtype(source_features.dtype, np.number)
            or np.issubdtype(source_features.dtype, np.bool_)
        )
        or not (
            np.issubdtype(source_target.dtype, np.number)
            or np.issubdtype(source_target.dtype, np.bool_)
        )
    ):
        return None

    rows, cols = int(source_features.shape[0]), int(source_features.shape[1])
    if rows == 0:
        raise ValueError("X must contain at least one sample.")
    if cols == 0:
        raise ValueError("X must contain at least one feature.")
    if int(source_target.shape[0]) != rows:
        raise ValueError("X and y must have the same number of samples.")

    source_features_finite = np.isfinite(source_features)
    source_target_finite = np.isfinite(source_target)
    if bool(
        np.any(
            source_features_finite
            & ((source_features > _F32_MAX) | (source_features < -_F32_MAX))
        )
    ):
        raise ValueError("X contains a value outside fp32 range.")
    if bool(
        np.any(
            source_target_finite
            & ((source_target > _F32_MAX) | (source_target < -_F32_MAX))
        )
    ):
        raise ValueError("y contains a value outside fp32 range.")

    with np.errstate(over="ignore", invalid="ignore"):
        features = np.asarray(source_features, dtype=np.float32)
        target = np.asarray(source_target, dtype=np.float32)
    if bool(np.any(source_features_finite & ~np.isfinite(features))):
        raise ValueError("X contains a value outside fp32 range.")
    if bool(np.any(source_target_finite & ~np.isfinite(target))):
        raise ValueError("y contains a value outside fp32 range.")
    features = np.ascontiguousarray(features, dtype="<f4")
    target = np.ascontiguousarray(target, dtype="<f4")
    names = _coerce_feature_names(feature_names, cols)
    return _CachedCoercedInput(
        features=features,
        target=target,
        rows=rows,
        cols=cols,
        feature_names=names,
        feature_digest=(
            _f32_buffer_digest(rows * cols, memoryview(features).cast("B"))
            if include_digests
            else None
        ),
        target_digest=(
            _f32_buffer_digest(rows, memoryview(target).cast("B"))
            if include_digests
            else None
        ),
    )


def _coerce_target_f32_storage(values: Iterable[float]) -> object:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    if np is not None:
        try:
            source = np.asarray(values)
        except (TypeError, ValueError, OverflowError):
            source = None
        if (
            source is not None
            and source.ndim == 1
            and (
                np.issubdtype(source.dtype, np.number)
                or np.issubdtype(source.dtype, np.bool_)
            )
            and not np.issubdtype(source.dtype, np.complexfloating)
        ):
            finite = np.isfinite(source)
            if bool(np.any(finite & ((source > _F32_MAX) | (source < -_F32_MAX)))):
                raise ValueError("y contains a value outside fp32 range.")
            with np.errstate(over="ignore", invalid="ignore"):
                target = np.ascontiguousarray(source, dtype="<f4")
            if bool(np.any(finite & ~np.isfinite(target))):
                raise ValueError("y contains a value outside fp32 range.")
            return target

    target = array("f")
    for value in _sequence(values, "y"):
        _append_f32(target, value, "y")
    return target


def _f32_storage_to_list(values: object) -> List[float]:
    ravel = getattr(values, "ravel", None)
    if ravel is not None:
        return ravel(order="C").tolist()
    tolist = getattr(values, "tolist", None)
    if tolist is not None:
        return tolist()
    return list(values)  # type: ignore[arg-type]


def _f32_storage_to_le_bytes(values: object) -> bytes:
    if isinstance(values, array):
        packed = array("f", values)
        if sys.byteorder != "little":
            packed.byteswap()
        return packed.tobytes()
    return memoryview(values).cast("B").tobytes()


def _coerce_feature_names(feature_names: Iterable[str] | None, cols: int) -> List[str]:
    if feature_names is None:
        return [f"f{i}" for i in range(cols)]
    names = [str(name) for name in feature_names]
    if len(names) != cols:
        raise ValueError("feature_names length must match X's feature count.")
    return names


def _matrix_rows(X: object) -> List[List[float]]:
    if hasattr(X, "to_dicts") and hasattr(X, "columns"):
        columns = [str(col) for col in X.columns]
        return [[row[col] for col in columns] for row in X.to_dicts()]
    return [list(row) for row in _sequence(X, "X")]


def _sequence(values: object, label: str) -> Sequence:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be numeric, not a string.")
    if hasattr(values, "__len__") and hasattr(values, "__getitem__"):
        return values  # type: ignore[return-value]
    try:
        return list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable numeric sequence.") from exc


def _finite_f32(value: object, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-numeric value.") from exc
    except OverflowError as exc:
        raise ValueError(f"{label} contains a value outside fp32 range.") from exc
    if math.isfinite(out) and abs(out) > _F32_MAX:
        raise ValueError(f"{label} contains a value outside fp32 range.")
    return out


def _fresh_random_seed() -> int:
    return int.from_bytes(os.urandom(32), "little")


def _config_payload(config: EngineConfig) -> dict[str, object]:
    budget = config.budget
    if (
        budget.max_feature_candidate is not None
        and int(budget.max_feature_candidate) < -1
    ):
        raise ValueError(
            "max_feature_candidate must be >= 0 or -1 for power-user mode."
        )
    random_seed = config.random_seed
    if random_seed is None:
        # Legacy `random.Random(None)` consumed fresh OS entropy for every
        # analysis. A fresh integer also prevents resident-cache reuse from
        # silently turning that request into deterministic seed zero.
        random_seed = _fresh_random_seed()
    return {
        "backend": str(config.backend),
        "device_id": int(config.device_id),
        "metric_names": [str(name) for name in config.metric_names],
        "num_repeats": int(config.num_repeats),
        "permutation_tests": int(config.permutation_tests),
        "random_seed": random_seed,
        "mi_bins": int(config.mi_bins),
        "mi_approximate": bool(config.mi_approximate),
        "stability_std_threshold": float(config.stability_std_threshold),
        "permutation_p_threshold": float(config.permutation_p_threshold),
        "significance_top_n": int(config.significance_top_n),
        "enable_time_series_functions": bool(config.enable_time_series_functions),
        "enable_decision_path_functions": bool(config.enable_decision_path_functions),
        "time_series_lags": [int(value) for value in config.time_series_lags],
        "time_series_windows": [int(value) for value in config.time_series_windows],
        "decision_path_max_depth": int(config.decision_path_max_depth),
        "decision_path_rounds": int(config.decision_path_rounds),
        "decision_path_max_paths": int(config.decision_path_max_paths),
        "decision_path_max_bins": int(config.decision_path_max_bins),
        "decision_path_min_leaf": int(config.decision_path_min_leaf),
        "decision_path_learning_rate": float(config.decision_path_learning_rate),
        "decision_path_top_k_features": int(config.decision_path_top_k_features),
        "budget": {
            "max_comb_size": int(budget.max_comb_size),
            "max_combinations_per_k": int(budget.max_combinations_per_k),
            "top_features_for_higher_k": int(budget.top_features_for_higher_k),
            "max_generated_features": int(budget.max_generated_features),
            "keep_in_vram": bool(budget.keep_in_vram),
            "vram_budget_mb": int(budget.vram_budget_mb),
            "max_time_series_candidates": int(budget.max_time_series_candidates),
            "top_k_features_for_time_series": int(
                budget.top_k_features_for_time_series
            ),
            "max_feature_candidate": budget.max_feature_candidate,
        },
    }


def _compile_flag_values(flags: object | None) -> tuple[bool, bool, bool]:
    if flags is None:
        return True, False, False
    values = tuple(
        getattr(flags, name, default)
        for name, default in (
            ("plan", True),
            ("graph", False),
            ("export", False),
        )
    )
    if not all(isinstance(value, bool) for value in values):
        raise TypeError("compile flags plan, graph, and export must be bool values")
    return values  # type: ignore[return-value]


def _validate_graph_request(config: EngineConfig, graph: bool) -> None:
    if not graph:
        return
    if config.enable_time_series_functions or config.enable_decision_path_functions:
        raise V1UnsupportedError(
            "CompileFlags(graph=True) is unavailable for generated-family compilation."
        )
    backend = str(config.backend).lower()
    if backend in {"cpu", "core", "rust", "v1-rust-cpu", "metal"}:
        raise V1UnsupportedError(
            f"CompileFlags(graph=True) is unsupported on backend {config.backend!r}."
        )


def _require_native_graph_activation(native_handle: object) -> None:
    device = str(getattr(native_handle, "device", "unknown")).lower()
    if device not in {"cuda", "rocm", "hip"}:
        close = getattr(native_handle, "close", None)
        if callable(close):
            close()
        raise V1UnsupportedError(
            f"CompileFlags(graph=True) resolved to unsupported backend {device!r}."
        )
    requested = getattr(native_handle, "graph_requested", None)
    if callable(requested):
        requested = requested()
    if requested is not True:
        close = getattr(native_handle, "close", None)
        if callable(close):
            close()
        raise V1UnsupportedError(
            "the native boundary does not expose activated graph execution for "
            "CompileFlags(graph=True)."
        )


def _native_graph_replayed(native_report: object, native_handle: object) -> bool | None:
    for source in (native_report, native_handle):
        replayed = getattr(source, "graph_replayed", None)
        if replayed is not None:
            value = replayed() if callable(replayed) else replayed
            return value if isinstance(value, bool) else None
        replays = getattr(source, "graph_replays", None)
        if replays is not None:
            value = replays() if callable(replays) else replays
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value > 0
    return None


def _raw_arrow_config_supported(config: EngineConfig) -> bool:
    if str(config.backend) not in {"cpu", "core", "rust", "v1-rust-cpu"}:
        return False
    if config.permutation_tests != 0 or config.num_repeats != 1:
        return False
    if config.enable_time_series_functions or config.enable_decision_path_functions:
        return False
    if "mutual_info" in config.metric_names and (
        config.mi_bins != 96 or config.mi_approximate
    ):
        return False
    default_budget = ComputeBudget()
    supported_budget = replace(
        default_budget,
        max_comb_size=config.budget.max_comb_size,
        max_combinations_per_k=config.budget.max_combinations_per_k,
    )
    return config.budget == supported_budget and config.device_id == 0


def _validate_arrow_target_frame(target_frame: object) -> object:
    width = _arrow_frame_width(target_frame)
    if width != 1:
        raise ValueError("Arrow target input must contain exactly one column.")

    get_columns = getattr(target_frame, "get_columns", None)
    if callable(get_columns):
        columns = list(get_columns())
        if len(columns) != 1:
            raise ValueError("Arrow target input must contain exactly one column.")
        target = columns[0]
    else:
        column = getattr(target_frame, "column", None)
        if not callable(column):
            raise V1UnsupportedError(
                "Arrow target validation requires get_columns() or column(0)."
            )
        target = column(0)

    null_count = getattr(target, "null_count", None)
    if callable(null_count):
        null_count = null_count()
    if null_count is None:
        raise V1UnsupportedError(
            "Arrow target validation requires a null_count surface."
        )
    if int(null_count) != 0:
        raise ValueError("Arrow target column must not contain null values.")
    return target


def _arrow_frame_width(frame: object) -> int | None:
    for name in ("width", "num_columns"):
        value = getattr(frame, name, None)
        if value is not None:
            value = value() if callable(value) else value
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    for name in ("column_names", "columns"):
        value = getattr(frame, name, None)
        if value is not None:
            try:
                return len(value)
            except TypeError:
                return None
    return None


def _significance_from_native(
    native_report: Any,
    feature_names: Sequence[str],
    config: EngineConfig,
    *,
    generated_feature_start: int | None = None,
) -> tuple[List[StabilityResult], List[PermutationResult], Decision]:
    """Build only the significance dimensions requested by ``config``.

    Permutation-only and stability-only runs gate their decisions on that single
    enabled dimension. When both are enabled, both thresholds must pass for the
    same candidate/metric. Requested but absent native significance is an explicit
    contract error rather than an interactions-only success.
    """
    permutation_enabled = int(config.permutation_tests) > 0
    stability_enabled = int(config.num_repeats) > 1
    significance_requested = permutation_enabled or stability_enabled
    if not significance_requested:
        detected = len(native_report) > 0
        return (
            [],
            [],
            Decision(
                detected,
                "v1 continuous native path executed (no significance computed).",
            ),
        )

    has_significance = getattr(native_report, "has_significance", None)
    available = bool(
        has_significance() if callable(has_significance) else has_significance
    )
    if not available:
        if len(native_report) > 0:
            requested = []
            if permutation_enabled:
                requested.append("permutation")
            if stability_enabled:
                requested.append("stability")
            raise V1UnsupportedError(
                "native report did not provide requested "
                + " and ".join(requested)
                + " results."
            )
        return (
            [],
            [],
            Decision(
                False,
                "no candidates were available for requested significance evaluation.",
            ),
        )

    metric_names = tuple(str(name) for name in config.metric_names)
    names = tuple(str(name) for name in feature_names)
    rows = _native_significance_rows(native_report)
    pvalues = (
        _native_significance_matrix(
            native_report, "significance_pvalues", "permutation", len(rows)
        )
        if permutation_enabled
        else None
    )
    means = (
        _native_significance_matrix(
            native_report, "significance_means", "stability mean", len(rows)
        )
        if stability_enabled
        else None
    )
    stds = (
        _native_significance_matrix(
            native_report, "significance_stds", "stability std", len(rows)
        )
        if stability_enabled
        else None
    )

    p_threshold = float(config.permutation_p_threshold)
    std_threshold = float(config.stability_std_threshold)
    stability: List[StabilityResult] = []
    permutations: List[PermutationResult] = []
    signal = False

    for position, row in enumerate(rows):
        combo = tuple(int(value) for value in native_report.combo(row))
        candidate_feature_names = tuple(names[idx] for idx in combo if idx < len(names))
        family = _family_for_combo(config, combo, generated_feature_start)
        expression = "*".join(candidate_feature_names)
        native_candidate_id = int(native_report.candidate_id(row))
        candidate_id = f"{family}:{native_candidate_id}"
        params = _decision_path_params_for_combo(
            native_report,
            combo,
            native_candidate_id,
            candidate_id,
        )
        p_values: dict[str, float] = {}
        observed_metrics: dict[str, float] = {}
        metrics_mean: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        if permutation_enabled:
            if pvalues is None:
                raise V1UnsupportedError("native permutation results are unavailable.")
            p_values = _named_significance_values(
                metric_names, pvalues[position], "permutation"
            )
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in p_values.values()
            ):
                raise V1UnsupportedError(
                    "native permutation results must contain finite p-values in [0, 1]."
                )
            permutations.append(
                PermutationResult(
                    combo=combo,
                    p_values=p_values,
                    family=family,
                    expression=expression,
                    params=params,
                    candidate_id=candidate_id,
                )
            )
        if stability_enabled:
            if means is None or stds is None:
                raise V1UnsupportedError("native stability results are unavailable.")
            observed_metrics = _named_significance_values(
                metric_names,
                native_report.metric_values(row),
                "observed",
            )
            metrics_mean = _named_significance_values(
                metric_names, means[position], "stability mean"
            )
            metrics_std = _named_significance_values(
                metric_names, stds[position], "stability std"
            )
            if not all(
                math.isfinite(value) for value in metrics_mean.values()
            ) or not all(
                math.isfinite(value) and value >= 0.0 for value in metrics_std.values()
            ):
                raise V1UnsupportedError(
                    "native stability results contain non-finite means or invalid standard deviations."
                )
            stability.append(
                StabilityResult(
                    combo=combo,
                    metrics_mean=metrics_mean,
                    metrics_std=metrics_std,
                    family=family,
                    expression=expression,
                    params=params,
                    candidate_id=candidate_id,
                )
            )
        for name in metric_names:
            passes = []
            if permutation_enabled:
                passes.append(p_values[name] <= p_threshold)
            if stability_enabled:
                passes.append(
                    math.isfinite(observed_metrics[name])
                    and observed_metrics[name] != 0.0
                    and metrics_mean[name] != 0.0
                    and metrics_std[name] <= std_threshold
                )
            if passes and all(passes):
                signal = True

    if permutation_enabled and stability_enabled:
        if signal:
            message = (
                f"signal detected: candidate(s) passed permutation p<={p_threshold:g} "
                f"and stability std<={std_threshold:g}."
            )
        else:
            message = (
                "no significant interaction: none passed the permutation p-value and "
                "stability thresholds."
            )
    elif permutation_enabled:
        if signal:
            message = (
                f"signal detected: candidate(s) passed permutation p<={p_threshold:g}."
            )
        else:
            message = "no significant interaction: none passed the permutation p-value threshold."
    elif stability_enabled:
        if signal:
            message = (
                "stable signal detected: candidate(s) had non-zero effects and "
                f"passed stability std<={std_threshold:g}."
            )
        else:
            message = (
                "no stable interaction: none had non-zero observed/mean effects "
                "within the stability threshold."
            )
    return stability, permutations, Decision(signal, message)


def _native_significance_rows(native_report: object) -> List[int]:
    getter = getattr(native_report, "significance_rows", None)
    if not callable(getter):
        raise V1UnsupportedError("native report lacks significance_rows().")
    rows = [int(row) for row in getter()]
    if len(set(rows)) != len(rows):
        raise V1UnsupportedError("native significance rows contain duplicates.")
    if any(row < 0 or row >= len(native_report) for row in rows):
        raise V1UnsupportedError("native significance row is outside the report.")
    return rows


def _native_significance_matrix(
    native_report: object,
    method_name: str,
    label: str,
    expected_rows: int,
) -> List[Sequence[float]]:
    getter = getattr(native_report, method_name, None)
    if not callable(getter):
        raise V1UnsupportedError(f"native report lacks {method_name}().")
    values = list(getter())
    if len(values) != expected_rows:
        raise V1UnsupportedError(
            f"native {label} row count {len(values)} does not match significance "
            f"row count {expected_rows}."
        )
    return values


def _named_significance_values(
    metric_names: Sequence[str],
    values: Sequence[float],
    label: str,
) -> dict[str, float]:
    values = list(values)
    if len(values) != len(metric_names):
        raise V1UnsupportedError(
            f"native {label} metric count {len(values)} does not match requested "
            f"metric count {len(metric_names)}."
        )
    return {name: float(value) for name, value in zip(metric_names, values)}


def _backend_info(native_handle: object) -> BackendInfo:
    name = str(getattr(native_handle, "backend_name", "v1-rust-cpu"))
    device = str(getattr(native_handle, "device", "cpu"))
    is_gpu = bool(getattr(native_handle, "is_gpu", False))
    selected_backend = getattr(native_handle, "selected_backend", None)
    execution_placement = getattr(native_handle, "execution_placement", None)
    return BackendInfo(
        name=name,
        device=device,
        is_gpu=is_gpu,
        memory_total_mb=None,
        memory_free_mb=None,
        selected_backend=(
            str(selected_backend) if selected_backend is not None else None
        ),
        execution_placement=(
            str(execution_placement) if execution_placement is not None else None
        ),
    )
