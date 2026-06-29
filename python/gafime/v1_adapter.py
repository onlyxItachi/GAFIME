from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from types import ModuleType
from typing import Iterable, List, Sequence

from .config import EngineConfig
from .errors import V1UnsupportedError
from .reporting import BackendInfo, Decision, DiagnosticReport, NativeContinuousInteractions


_BOUNDARY_MODULE_ENV = "GAFIME_V1_BOUNDARY_MODULE"
_BOUNDARY_MODULES = ("gafime.gafime_py", "gafime_py")


def analyze_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    artifact = compile_with_v1_boundary(config, X, y, feature_names)
    try:
        return artifact.analyze()
    finally:
        artifact.close()


def compile_with_v1_boundary(
    config: EngineConfig,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
    *,
    flags=None,
) -> "NativeCompiledGafime":
    if flags is not None and getattr(flags, "export", False):
        raise V1UnsupportedError("v1 native compile artifacts do not expose export handles yet.")

    boundary = _load_boundary()
    features, target, rows, cols, names = _coerce_row_major_f32(X, y, feature_names)
    payload = _config_payload(config)
    handle = boundary.compile_continuous(
        payload,
        features,
        target,
        rows=rows,
        cols=cols,
    )
    return NativeCompiledGafime(
        config=config,
        feature_names=names,
        native_handle=handle,
        boundary_name=str(getattr(boundary, "BOUNDARY_NAME", "gafime-py")),
    )


_METRIC_IDS = {"pearson": 1, "spearman": 2, "mutual_info": 3, "r2": 4}


def analyze_arrow_with_v1_boundary(
    config: EngineConfig,
    feature_frame: object,
    target_frame: object,
    feature_names: Sequence[str],
) -> DiagnosticReport:
    """Zero-copy Arrow ingest. feature_frame/target_frame expose the Arrow C
    stream interface (e.g. Polars DataFrames); data crosses as Arrow buffers with
    no Python-row materialization (no .rows()/.tolist())."""
    boundary = _load_boundary()
    if not hasattr(boundary, "analyze_continuous_arrow"):
        raise V1UnsupportedError("native boundary lacks analyze_continuous_arrow")
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
    names = list(feature_names)
    interactions = NativeContinuousInteractions(native_report, names, config.metric_names)
    return DiagnosticReport(
        config=config,
        feature_names=names,
        interactions=interactions,
        stability=[],
        permutations=[],
        warnings=[],
        decision=Decision(bool(interactions), "v1 continuous Arrow ingest path executed."),
        backend=BackendInfo(
            name="v1-rust-cpu",
            device="cpu",
            is_gpu=False,
            memory_total_mb=None,
            memory_free_mb=None,
        ),
    )


@dataclass
class NativeCompiledGafime:
    config: EngineConfig
    feature_names: List[str]
    native_handle: object
    boundary_name: str
    _closed: bool = False
    _last_report: DiagnosticReport | None = None

    @property
    def backend(self) -> BackendInfo:
        self._ensure_open()
        return _backend_info(self.native_handle)

    @property
    def scenario_plan(self) -> object:
        self._ensure_open()
        return self.native_handle

    def analyze(self) -> DiagnosticReport:
        self._ensure_open()
        native_report = self.native_handle.analyze()
        interactions = NativeContinuousInteractions(
            native_report,
            self.feature_names,
            self.config.metric_names,
        )
        report = DiagnosticReport(
            config=self.config,
            feature_names=list(self.feature_names),
            interactions=interactions,
            stability=[],
            permutations=[],
            warnings=[],
            decision=Decision(bool(interactions), "v1 continuous native path executed."),
            backend=_backend_info(self.native_handle),
        )
        self._last_report = report
        return report

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


def _backend_info(native_handle: object) -> BackendInfo:
    name = str(getattr(native_handle, "backend_name", "v1-rust-cpu"))
    device = str(getattr(native_handle, "device", "cpu"))
    is_gpu = bool(getattr(native_handle, "is_gpu", False))
    return BackendInfo(
        name=name,
        device=device,
        is_gpu=is_gpu,
        memory_total_mb=None,
        memory_free_mb=None,
    )
