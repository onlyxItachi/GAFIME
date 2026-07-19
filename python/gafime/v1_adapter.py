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
from .reporting.report import _family_for_feature_names


_BOUNDARY_MODULE_ENV = "GAFIME_V1_BOUNDARY_MODULE"
_BOUNDARY_MODULES = ("gafime.gafime_py", "gafime_py")
_ANALYZE_CACHE_ENV = "GAFIME_V1_ANALYZE_CACHE_SIZE"
_DEFAULT_ANALYZE_CACHE_SIZE = 2


@dataclass
class _AnalyzeCacheEntry:
    artifact: Any
    target_digest: bytes


@dataclass
class _CachedCoercedInput:
    features: Any
    target: Any
    rows: int
    cols: int
    feature_names: List[str]
    feature_digest: bytes
    target_digest: bytes

    def feature_list(self) -> List[float]:
        return _f32_storage_to_list(self.features)

    def target_list(self) -> List[float]:
        return _f32_storage_to_list(self.target)


_ANALYZE_CACHE_LOCK = threading.RLock()
_ANALYZE_CACHE: OrderedDict[tuple[Any, ...], _AnalyzeCacheEntry] = OrderedDict()


def analyze_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    if _continuous_analyze_cache_enabled(config):
        return _analyze_continuous_with_resident_cache(config, X, y, feature_names)
    if not config.enable_time_series_functions and not config.enable_decision_path_functions:
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
    analyze = getattr(boundary, "analyze_continuous", None)
    if not callable(analyze):
        return None
    nested_shape = _native_nested_shape(X, y, feature_names)
    analyze_rows = getattr(boundary, "analyze_continuous_rows", None)
    if nested_shape is not None and callable(analyze_rows):
        _, _, names = nested_shape
        try:
            native_report = analyze_rows(_config_payload(config), X, y)
        except (TypeError, OverflowError) as exc:
            raise ValueError("X and y must contain values representable as finite fp32.") from exc
        return _diagnostic_from_native_report(config, native_report, names, [])
    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    native_report = analyze(
        _config_payload(config),
        features,
        target,
        rows=rows,
        cols=cols,
    )
    return _diagnostic_from_native_report(config, native_report, names, [])


def _continuous_analyze_cache_enabled(config: EngineConfig) -> bool:
    if config.enable_time_series_functions or config.enable_decision_path_functions:
        return False
    if not bool(config.budget.keep_in_vram):
        return False
    return _analyze_cache_capacity() > 0


def _analyze_continuous_with_resident_cache(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
) -> DiagnosticReport:
    boundary = _load_boundary_for_backend(config.backend)
    coerced = _coerce_row_major_f32_for_cache(X, y, feature_names)
    payload = _config_payload(config)
    cache_key = (
        id(boundary),
        str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
        _freeze_cache_value(payload),
        tuple(coerced.feature_names),
        int(coerced.rows),
        int(coerced.cols),
        coerced.feature_digest,
    )

    with _ANALYZE_CACHE_LOCK:
        entry = _ANALYZE_CACHE.pop(cache_key, None)
        if entry is not None:
            if getattr(entry.artifact, "_closed", False):
                entry = None
            else:
                try:
                    if entry.target_digest != coerced.target_digest:
                        entry.artifact.update_target(coerced.target_list())
                        entry.target_digest = coerced.target_digest
                    report = entry.artifact.analyze()
                except Exception:
                    entry.artifact.close()
                    raise
                _ANALYZE_CACHE[cache_key] = entry
                return report

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
        warnings=[],
    )
    try:
        report = artifact.analyze()
    except Exception:
        artifact.close()
        raise

    with _ANALYZE_CACHE_LOCK:
        previous = _ANALYZE_CACHE.pop(cache_key, None)
        if previous is not None:
            previous.artifact.close()
        _ANALYZE_CACHE[cache_key] = _AnalyzeCacheEntry(
            artifact=artifact,
            target_digest=coerced.target_digest,
        )
        _prune_analyze_cache_locked()
    return report


def _analyze_cache_capacity() -> int:
    raw = os.environ.get(_ANALYZE_CACHE_ENV)
    if raw is None:
        return _DEFAULT_ANALYZE_CACHE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_ANALYZE_CACHE_SIZE


def _prune_analyze_cache_locked() -> None:
    capacity = _analyze_cache_capacity()
    while len(_ANALYZE_CACHE) > capacity:
        _, entry = _ANALYZE_CACHE.popitem(last=False)
        entry.artifact.close()


def _f32_array_digest(values: array) -> bytes:
    packed = array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return _f32_buffer_digest(len(packed), packed.tobytes())


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
    if not math.isfinite(values[-1]):
        values.pop()
        raise ValueError(f"{label} contains a value that is non-finite after fp32 conversion.")


def _freeze_cache_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple((str(key), _freeze_cache_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_cache_value(item) for item in value)
    return value


def _clear_analyze_cache_for_tests() -> None:
    with _ANALYZE_CACHE_LOCK:
        while _ANALYZE_CACHE:
            _, entry = _ANALYZE_CACHE.popitem()
            entry.artifact.close()


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
    )


def analyze_decision_path_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    """Analyze the decision_path family through the native expand+mine path."""
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
        int(config.decision_path_min_leaf),
        float(config.decision_path_learning_rate),
    )
    return _diagnostic_from_native_report(
        config,
        native_report,
        list(all_names),
        [f"decision_path discovered {len(all_names) - cols} conjunction path(s) from {cols} features."],
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
    compile_flags = {
        "plan": plan,
        "graph": graph,
        "export": export,
    }

    boundary = _load_boundary_for_backend(config.backend)
    if not config.enable_time_series_functions and not config.enable_decision_path_functions:
        nested_shape = _native_nested_shape(X, y, feature_names)
        compile_rows = getattr(boundary, "compile_continuous_rows", None)
        if nested_shape is not None and callable(compile_rows):
            _, _, names = nested_shape
            payload = _config_payload(config)
            payload["compile_flags"] = compile_flags
            try:
                handle = compile_rows(payload, X, y)
            except (TypeError, OverflowError) as exc:
                raise ValueError(
                    "X and y must contain values representable as finite fp32."
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
                warnings=[],
            )

    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
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
            int(config.decision_path_min_leaf),
            float(config.decision_path_learning_rate),
        )
        warnings = [f"decision_path discovered {len(all_names) - cols} conjunction path(s) from {cols} features."]
        names = list(all_names)
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
        warnings = []
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
    )


_METRIC_IDS = {"pearson": 1, "spearman": 2, "mutual_info": 3, "r2": 4}


def _diagnostic_from_native_report(
    config: EngineConfig,
    native_report: Any,
    feature_names: Sequence[str],
    warnings: Sequence[str],
) -> DiagnosticReport:
    names = list(feature_names)
    interactions = NativeContinuousInteractions(native_report, names, config.metric_names)
    stability, permutations, decision = _significance_from_native(
        native_report,
        names,
        config,
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
    if _raw_arrow_config_supported(config) and hasattr(boundary, "analyze_continuous_arrow"):
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
        report = _diagnostic_from_native_report(config, native_report, feature_names, [])
        report.decision = Decision(bool(report.interactions), "v1 continuous Arrow ingest path executed.")
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
        return analyze_decision_path_with_v1_boundary(config, rows, target, feature_names)
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
    _closed: bool = False
    _last_report: DiagnosticReport | None = None
    _native_report: Any = None
    _graph_replayed: bool = False

    @property
    def backend(self) -> BackendInfo:
        self._ensure_open()
        return _backend_info(self.native_handle)

    @property
    def scenario_plan(self) -> object | None:
        self._ensure_open()
        return self.native_handle if self.plan_enabled else None

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

    def analyze(self) -> DiagnosticReport:
        self._ensure_open()
        self._graph_replayed = False
        native_report = self.native_handle.analyze()
        if self.graph_requested:
            replayed = _native_graph_replayed(native_report, self.native_handle)
            if replayed is not True:
                raise V1UnsupportedError(
                    "CompileFlags(graph=True) was requested, but the native report "
                    "did not confirm graph replay."
                )
            self._graph_replayed = True
        self._native_report = native_report
        interactions = NativeContinuousInteractions(
            native_report,
            self.feature_names,
            self.config.metric_names,
        )
        stability, permutations, decision = _significance_from_native(
            native_report,
            self.feature_names,
            self.config,
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
        target = [_finite_f32(value, "y") for value in _sequence(y, "y")]
        self.native_handle.update_target(target)
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
        if self._closed:
            return
        close = getattr(self.native_handle, "close", None)
        if close is not None:
            close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("NativeCompiledGafime is closed.")


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
    """Return cheap shape metadata when PyO3 can ingest nested rows directly."""
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
    return len(X), cols, _coerce_feature_names(feature_names, cols)


def _coerce_row_major_f32_for_cache(
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None,
) -> _CachedCoercedInput:
    numpy_coerced = _try_coerce_numpy_row_major_f32_for_cache(X, y, feature_names)
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
            raise ValueError(f"X row {row_idx} has length {len(row_values)}; expected {cols}.")
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
        feature_digest=_f32_array_digest(features),
        target_digest=_f32_array_digest(target),
    )


def _try_coerce_numpy_row_major_f32_for_cache(
    X: object,
    y: object,
    feature_names: Iterable[str] | None,
) -> _CachedCoercedInput | None:
    if hasattr(X, "to_dicts") and hasattr(X, "columns"):
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    try:
        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return None
    if features.ndim != 2 or target.ndim != 1:
        return None
    rows, cols = int(features.shape[0]), int(features.shape[1])
    if rows == 0:
        raise ValueError("X must contain at least one sample.")
    if cols == 0:
        raise ValueError("X must contain at least one feature.")
    if int(target.shape[0]) != rows:
        raise ValueError("X and y must have the same number of samples.")
    if not bool(np.isfinite(features).all()):
        raise ValueError("X contains a non-finite value.")
    if not bool(np.isfinite(target).all()):
        raise ValueError("y contains a non-finite value.")
    features = np.ascontiguousarray(features, dtype="<f4")
    target = np.ascontiguousarray(target, dtype="<f4")
    names = _coerce_feature_names(feature_names, cols)
    return _CachedCoercedInput(
        features=features,
        target=target,
        rows=rows,
        cols=cols,
        feature_names=names,
        feature_digest=_f32_buffer_digest(rows * cols, memoryview(features).cast("B")),
        target_digest=_f32_buffer_digest(rows, memoryview(target).cast("B")),
    )


def _f32_storage_to_list(values: object) -> List[float]:
    ravel = getattr(values, "ravel", None)
    if ravel is not None:
        return ravel(order="C").tolist()
    tolist = getattr(values, "tolist", None)
    if tolist is not None:
        return tolist()
    return list(values)  # type: ignore[arg-type]


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
    if not math.isfinite(out):
        raise ValueError(f"{label} contains a non-finite value.")
    return out


def _config_payload(config: EngineConfig) -> dict[str, object]:
    budget = config.budget
    return {
        "backend": str(config.backend),
        "device_id": int(config.device_id),
        "metric_names": [str(name) for name in config.metric_names],
        "num_repeats": int(config.num_repeats),
        "permutation_tests": int(config.permutation_tests),
        "random_seed": config.random_seed,
        "mi_bins": int(config.mi_bins),
        "mi_approximate": bool(config.mi_approximate),
        "stability_std_threshold": float(config.stability_std_threshold),
        "permutation_p_threshold": float(config.permutation_p_threshold),
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
            "top_k_features_for_time_series": int(budget.top_k_features_for_time_series),
            "max_feature_candidate": budget.max_feature_candidate,
        },
    }


def _compile_flag_values(flags: object | None) -> tuple[bool, bool, bool]:
    if flags is None:
        return True, False, False
    values = tuple(getattr(flags, name, default) for name, default in (
        ("plan", True),
        ("graph", False),
        ("export", False),
    ))
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
            Decision(detected, "v1 continuous native path executed (no significance computed)."),
        )

    has_significance = getattr(native_report, "has_significance", None)
    available = bool(has_significance() if callable(has_significance) else has_significance)
    if not available:
        if len(native_report) > 0:
            requested = []
            if permutation_enabled:
                requested.append("permutation")
            if stability_enabled:
                requested.append("stability")
            raise V1UnsupportedError(
                "native report did not provide requested " + " and ".join(requested) + " results."
            )
        return (
            [],
            [],
            Decision(False, "no candidates were available for requested significance evaluation."),
        )

    metric_names = tuple(str(name) for name in config.metric_names)
    names = tuple(str(name) for name in feature_names)
    rows = _native_significance_rows(native_report)
    pvalues = (
        _native_significance_matrix(native_report, "significance_pvalues", "permutation", len(rows))
        if permutation_enabled
        else None
    )
    means = (
        _native_significance_matrix(native_report, "significance_means", "stability mean", len(rows))
        if stability_enabled
        else None
    )
    stds = (
        _native_significance_matrix(native_report, "significance_stds", "stability std", len(rows))
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
        family = _family_for_feature_names(candidate_feature_names)
        expression = "*".join(candidate_feature_names)
        candidate_id = f"{family}:{native_report.candidate_id(row)}"
        p_values: dict[str, float] = {}
        metrics_mean: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        if permutation_enabled:
            if pvalues is None:
                raise V1UnsupportedError("native permutation results are unavailable.")
            p_values = _named_significance_values(metric_names, pvalues[position], "permutation")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in p_values.values()):
                raise V1UnsupportedError(
                    "native permutation results must contain finite p-values in [0, 1]."
                )
            permutations.append(
                PermutationResult(
                    combo=combo,
                    p_values=p_values,
                    family=family,
                    expression=expression,
                    candidate_id=candidate_id,
                )
            )
        if stability_enabled:
            if means is None or stds is None:
                raise V1UnsupportedError("native stability results are unavailable.")
            metrics_mean = _named_significance_values(metric_names, means[position], "stability mean")
            metrics_std = _named_significance_values(metric_names, stds[position], "stability std")
            if not all(math.isfinite(value) for value in metrics_mean.values()) or not all(
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
                    candidate_id=candidate_id,
                )
            )
        for name in metric_names:
            passes = []
            if permutation_enabled:
                passes.append(p_values[name] <= p_threshold)
            if stability_enabled:
                passes.append(metrics_std[name] <= std_threshold)
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
            message = f"signal detected: candidate(s) passed permutation p<={p_threshold:g}."
        else:
            message = "no significant interaction: none passed the permutation p-value threshold."
    elif stability_enabled:
        if signal:
            message = f"stable signal detected: candidate(s) passed stability std<={std_threshold:g}."
        else:
            message = "no stable interaction: none passed the stability threshold."
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
        selected_backend=(str(selected_backend) if selected_backend is not None else None),
        execution_placement=(
            str(execution_placement) if execution_placement is not None else None
        ),
    )
