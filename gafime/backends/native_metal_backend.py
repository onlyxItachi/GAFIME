from __future__ import annotations

import ctypes
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import ComputeBudget
from ..discrete import (
    DISCRETE_FUNCTION_KIND_CODES,
    DiscreteFunctionCandidate,
    GPU_HARD_MODE_ERROR,
)
from ..metrics import MetricSuite
from ..metrics.cpu_metrics import (
    adaptive_bin_indices,
    mi_bin_template_capacity,
    select_adaptive_mi_bins,
)
from ..native_data import NativeMatrix, NativeVector, column_means, mean
from ..time_series import TIME_SERIES_KIND_CODES, TimeSeriesCandidate
from .base import Backend, BackendInfo

logger = logging.getLogger(__name__)

GAFIME_SUCCESS = 0
GAFIME_MAX_BATCH_SIZE = 1024
GAFIME_MAX_BATCH_ARITY = 5
GAFIME_CANDIDATE_CONTINUOUS = 0
GAFIME_OP_IDENTITY = 0
GAFIME_INTERACT_MULT = 0

METAL_STATS_METRICS = ("pearson", "r2")
METAL_REPORT_METRICS = ("pearson", "spearman", "mutual_info", "r2")
DISCRETE_SELECTION_METRICS = (
    "mutual_info",
    "variance_reduction",
    "residual_abs_corr",
    "residual_r2_gain",
)


class NativeMetalBackend(Backend):
    name = "metal-native"
    device_label = "metal"
    is_gpu = True

    def __init__(self, device_id: int = 0) -> None:
        super().__init__(device_id=device_id)
        self.lib = self._load_library()
        if self.lib is None:
            raise ImportError("Native Metal library not found")
        self._setup_functions()
        self._set_shader_library_path()
        if not self._metal_available():
            raise RuntimeError("Metal not available on this system")
        self._cache_device_info()

    def _load_library(self) -> Optional[ctypes.CDLL]:
        package_dir = Path(__file__).parent.parent
        repo_dir = package_dir.parent
        search_dirs = (
            package_dir,
            repo_dir,
            repo_dir / "build",
            repo_dir / "build" / "Release",
        )
        for search_dir in search_dirs:
            for name in ("gafime_metal.dylib", "libgafime_metal.dylib"):
                lib_path = search_dir / name
                if lib_path.exists():
                    return ctypes.CDLL(str(lib_path.absolute()))
        return None

    def _setup_functions(self) -> None:
        if hasattr(self.lib, "gafime_metal_set_library_path"):
            self.lib.gafime_metal_set_library_path.restype = ctypes.c_int
            self.lib.gafime_metal_set_library_path.argtypes = [ctypes.c_char_p]

        self.lib.gafime_metal_available.restype = ctypes.c_int
        self.lib.gafime_metal_available.argtypes = []

        self.lib.gafime_metal_get_device_info.restype = ctypes.c_int
        self.lib.gafime_metal_get_device_info.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.gafime_metal_matrix_alloc.restype = ctypes.c_int
        self.lib.gafime_metal_matrix_alloc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.gafime_metal_matrix_upload.restype = ctypes.c_int
        self.lib.gafime_metal_matrix_upload.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_metal_matrix_compute_batch.restype = ctypes.c_int
        self.lib.gafime_metal_matrix_compute_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_metal_matrix_free.restype = ctypes.c_int
        self.lib.gafime_metal_matrix_free.argtypes = [ctypes.c_void_p]

        self.lib.gafime_metal_discrete_soft_batch.restype = ctypes.c_int
        self.lib.gafime_metal_discrete_soft_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]

        self.lib.gafime_metal_discrete_selection_adaptive.restype = ctypes.c_int
        self.lib.gafime_metal_discrete_selection_adaptive.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]

        self.lib.gafime_metal_time_series_batch.restype = ctypes.c_int
        self.lib.gafime_metal_time_series_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]

    def _set_shader_library_path(self) -> None:
        if not hasattr(self.lib, "gafime_metal_set_library_path"):
            return
        package_dir = Path(__file__).parent.parent
        metallib_path = package_dir / "gafime_kernels.metallib"
        if metallib_path.exists():
            rc = self.lib.gafime_metal_set_library_path(
                str(metallib_path.absolute()).encode("utf-8")
            )
            if rc != GAFIME_SUCCESS:
                raise RuntimeError(f"gafime_metal_set_library_path failed with code {rc}")

    def _metal_available(self) -> bool:
        return bool(self.lib.gafime_metal_available())

    def _cache_device_info(self) -> None:
        name_buf = ctypes.create_string_buffer(256)
        memory_mb = ctypes.c_int()
        gpu_family = ctypes.c_int()
        rc = self.lib.gafime_metal_get_device_info(
            name_buf,
            ctypes.byref(memory_mb),
            ctypes.byref(gpu_family),
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError("Unable to query Metal device info.")
        self._device_name = name_buf.value.decode("utf-8", errors="ignore")
        self._memory_total_mb = int(memory_mb.value)
        self._gpu_family = int(gpu_family.value)

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            device=f"{self._device_name} Apple{self._gpu_family}",
            is_gpu=True,
            memory_total_mb=self._memory_total_mb,
            memory_free_mb=None,
        )

    def check_budget(
        self,
        X: NativeMatrix,
        y: NativeVector,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        estimated_mb = self.estimate_bytes(X, y) / (1024 * 1024)
        if estimated_mb > budget.vram_budget_mb:
            warnings.append(
                f"Input estimate {estimated_mb:.1f} MB exceeds vram_budget_mb={budget.vram_budget_mb}; "
                "Metal uses Apple unified memory, so this is advisory."
            )
        return True, warnings

    def score_combos(
        self,
        X: NativeMatrix,
        y: NativeVector,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        combos_list = [tuple(int(idx) for idx in combo) for combo in combos]
        if not combos_list:
            return {}
        invalid = [
            combo for combo in combos_list
            if len(combo) < 1 or len(combo) > GAFIME_MAX_BATCH_ARITY
        ]
        if invalid:
            raise ValueError("Metal batch spine supports continuous combo arity 1 through 5.")

        scores = self._score_combos_stats_metrics(X, y, combos_list, metric_suite.metric_names)
        _complete_continuous_report_metrics(X, y, combos_list, metric_suite, scores)
        return scores

    def _score_combos_stats_metrics(
        self,
        X: NativeMatrix,
        y: NativeVector,
        combos: Sequence[Tuple[int, ...]],
        metric_names: Sequence[str],
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        stats_metric_names = _metal_stats_metric_names(metric_names)
        out: Dict[Tuple[int, ...], Dict[str, float]] = {combo: {} for combo in combos}
        if not stats_metric_names:
            return out

        matrix = ctypes.c_void_p()
        means = column_means(X)
        feature_major = X.feature_major_buffer()
        rc = self.lib.gafime_metal_matrix_alloc(
            ctypes.c_int(X.n_samples),
            ctypes.c_int(X.n_features),
            ctypes.c_int(GAFIME_MAX_BATCH_SIZE),
            ctypes.byref(matrix),
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_metal_matrix_alloc failed with code {rc}")
        try:
            rc = self.lib.gafime_metal_matrix_upload(
                matrix,
                _float_buffer(feature_major),
                _float_buffer(y.buffer),
                _uint8_array([0] * X.n_samples),
                _float_array(means),
            )
            if rc != GAFIME_SUCCESS:
                raise RuntimeError(f"gafime_metal_matrix_upload failed with code {rc}")

            for batch in _continuous_scheduler_batches(combos):
                _kinds, indices, _ops, _interact, _ts_params, arity, batch_size = batch
                if batch_size <= 0:
                    continue
                stats = self._launch_global_continuous_batch(
                    matrix,
                    indices,
                    int(arity),
                    int(batch_size),
                )
                for row_idx, row in enumerate(stats):
                    combo = tuple(
                        int(indices[row_idx * int(arity) + col])
                        for col in range(int(arity))
                    )
                    out[combo] = _stats_to_metrics(row, stats_metric_names)
        finally:
            self.lib.gafime_metal_matrix_free(matrix)
        return out

    def _launch_global_continuous_batch(
        self,
        matrix: ctypes.c_void_p,
        indices: Sequence[int],
        arity: int,
        batch_size: int,
    ) -> List[List[float]]:
        stats_out = (ctypes.c_float * (batch_size * 12))()
        rc = self.lib.gafime_metal_matrix_compute_batch(
            matrix,
            _int_array(indices),
            ctypes.c_int(arity),
            ctypes.c_int(batch_size),
            ctypes.c_int(255),
            stats_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_metal_matrix_compute_batch failed with code {rc}")
        return [
            [float(stats_out[row * 12 + col]) for col in range(12)]
            for row in range(batch_size)
        ]

    def score_discrete_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        metric_suite: MetricSuite,
    ) -> Dict[object, Dict[str, float]]:
        candidates_list = [candidate for candidate in candidates if isinstance(candidate, DiscreteFunctionCandidate)]
        if not candidates_list:
            return {}
        if any(candidate.mode == "hard" for candidate in candidates_list):
            raise ValueError(GPU_HARD_MODE_ERROR)

        stats_metric_names = _metal_stats_metric_names(metric_suite.metric_names)
        if not stats_metric_names:
            return _score_discrete_report_completion(X, y, candidates_list, metric_suite)

        arrays = _discrete_arrays(candidates_list)
        n_candidates = len(candidates_list)
        stats_out = (ctypes.c_float * (n_candidates * 12))()
        feature_major = X.feature_major_buffer()
        rc = self.lib.gafime_metal_discrete_soft_batch(
            _float_buffer(feature_major),
            _float_buffer(y.buffer),
            _int_array(arrays["kinds"]),
            _int_array(arrays["feature_a"]),
            _int_array(arrays["feature_b"]),
            _int_array(arrays["value_feature"]),
            _int_array(arrays["directions"]),
            _float_array(arrays["params"]),
            _float_array(arrays["scales"]),
            _float_array(arrays["sharpness"]),
            ctypes.c_int(X.n_samples),
            ctypes.c_int(X.n_features),
            ctypes.c_int(n_candidates),
            stats_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_metal_discrete_soft_batch failed with code {rc}")

        scores = {
            candidate: _stats_to_metrics(
                [float(stats_out[row * 12 + col]) for col in range(12)],
                stats_metric_names,
            )
            for row, candidate in enumerate(candidates_list)
        }
        _complete_discrete_report_metrics(X, y, candidates_list, metric_suite, scores)
        return scores

    def score_discrete_selection_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        *,
        baseline_pred=None,
        mi_bins: int = 96,
    ) -> Dict[object, Dict[str, float]]:
        candidates_list = [candidate for candidate in candidates if isinstance(candidate, DiscreteFunctionCandidate)]
        if not candidates_list:
            return {}
        if any(candidate.mode == "hard" for candidate in candidates_list):
            raise ValueError(GPU_HARD_MODE_ERROR)

        arrays = _discrete_arrays(candidates_list)
        y_values = y.to_list()
        residual = _residual_values(y_values, baseline_pred)
        target_bins = select_adaptive_mi_bins(
            X.n_samples,
            max_bins=mi_bins,
            samples_per_bin=25,
            dimensions=1,
        )
        y_bins, y_bin_count = adaptive_bin_indices(y_values, target_bins, exact_low_cardinality=True)
        target_template = mi_bin_template_capacity(y_bin_count)
        n_candidates = len(candidates_list)
        scores_out = (ctypes.c_float * (n_candidates * len(DISCRETE_SELECTION_METRICS)))()
        feature_major = X.feature_major_buffer()
        rc = self.lib.gafime_metal_discrete_selection_adaptive(
            _float_buffer(feature_major),
            _float_buffer(y.buffer),
            _float_array(residual),
            _int_array(y_bins),
            _int_array(arrays["kinds"]),
            _int_array(arrays["feature_a"]),
            _int_array(arrays["feature_b"]),
            _int_array(arrays["value_feature"]),
            _int_array(arrays["directions"]),
            _float_array(arrays["params"]),
            _float_array(arrays["scales"]),
            _float_array(arrays["sharpness"]),
            ctypes.c_int(X.n_samples),
            ctypes.c_int(X.n_features),
            ctypes.c_int(n_candidates),
            ctypes.c_int(target_template),
            ctypes.c_float(float(sum(y_values))),
            ctypes.c_float(float(sum(value * value for value in y_values))),
            scores_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_metal_discrete_selection_adaptive failed with code {rc}")

        width = len(DISCRETE_SELECTION_METRICS)
        return {
            candidate: {
                name: float(scores_out[row * width + col])
                for col, name in enumerate(DISCRETE_SELECTION_METRICS)
            }
            for row, candidate in enumerate(candidates_list)
        }

    def score_time_series_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        candidates: Iterable[object],
        metric_suite: MetricSuite,
    ) -> Dict[object, Dict[str, float]]:
        candidates_list = [candidate for candidate in candidates if isinstance(candidate, TimeSeriesCandidate)]
        if not candidates_list:
            return {}
        stats_metric_names = _metal_stats_metric_names(metric_suite.metric_names)
        if not stats_metric_names:
            return _score_time_series_report_completion(X, y, candidates_list, metric_suite)

        n_candidates = len(candidates_list)
        stats_out = (ctypes.c_float * (n_candidates * 12))()
        feature_major = X.feature_major_buffer()
        rc = self.lib.gafime_metal_time_series_batch(
            _float_buffer(feature_major),
            _float_buffer(y.buffer),
            _int_array([TIME_SERIES_KIND_CODES[candidate.kind] for candidate in candidates_list]),
            _int_array([int(candidate.feature_index) for candidate in candidates_list]),
            _int_array([max(1, int(candidate.lag)) for candidate in candidates_list]),
            _int_array([max(1, int(candidate.window)) for candidate in candidates_list]),
            ctypes.c_int(X.n_samples),
            ctypes.c_int(X.n_features),
            ctypes.c_int(n_candidates),
            stats_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_metal_time_series_batch failed with code {rc}")

        scores = {
            candidate: _stats_to_metrics(
                [float(stats_out[row * 12 + col]) for col in range(12)],
                stats_metric_names,
            )
            for row, candidate in enumerate(candidates_list)
        }
        _complete_time_series_report_metrics(X, y, candidates_list, metric_suite, scores)
        return scores


def _continuous_scheduler_batches(combos: Sequence[Tuple[int, ...]]):
    try:
        from .. import subfunctions
    except ImportError as exc:
        raise RuntimeError(
            "Metal continuous batch scoring requires the Rust subfunctions scheduler. "
            "Rebuild the local native extensions."
        ) from exc

    scheduler = subfunctions.BatchScheduler(max_blocks=GAFIME_MAX_BATCH_SIZE)
    candidate_kinds = [GAFIME_CANDIDATE_CONTINUOUS] * len(combos)
    feature_sets = [[int(feature) for feature in combo] for combo in combos]
    op_sets = [[GAFIME_OP_IDENTITY] * len(combo) for combo in combos]
    interaction_sets = [
        [GAFIME_INTERACT_MULT] * max(len(combo) - 1, 1)
        for combo in combos
    ]
    return scheduler.create_batches(
        candidate_kinds,
        feature_sets,
        op_sets,
        interaction_sets,
    )


def _metal_stats_metric_names(metric_names: Sequence[str]) -> Tuple[str, ...]:
    unsupported = [name for name in metric_names if name not in METAL_REPORT_METRICS]
    if unsupported:
        raise ValueError(
            f"Metal report metrics are {METAL_REPORT_METRICS}; "
            f"unsupported metrics requested: {tuple(unsupported)}."
        )
    return tuple(name for name in metric_names if name in METAL_STATS_METRICS)


def _missing_report_metric_names(metric_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(name for name in metric_names if name not in METAL_STATS_METRICS)


def _complete_continuous_report_metrics(
    X: NativeMatrix,
    y: NativeVector,
    combos: Sequence[Tuple[int, ...]],
    metric_suite: MetricSuite,
    scores: Dict[Tuple[int, ...], Dict[str, float]],
) -> None:
    missing = _missing_report_metric_names(metric_suite.metric_names)
    if not missing:
        return
    from .core_backend import CoreBackend

    completion_suite = MetricSuite(missing, mi_bins=metric_suite.mi_bins)
    completion = CoreBackend().score_combos(X, y, combos, completion_suite)
    for combo, values in completion.items():
        scores.setdefault(combo, {}).update(values)


def _complete_discrete_report_metrics(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[DiscreteFunctionCandidate],
    metric_suite: MetricSuite,
    scores: Dict[object, Dict[str, float]],
) -> None:
    missing = _missing_report_metric_names(metric_suite.metric_names)
    if not missing:
        return
    completion = _score_discrete_report_completion(
        X,
        y,
        candidates,
        MetricSuite(missing, mi_bins=metric_suite.mi_bins),
    )
    for candidate, values in completion.items():
        scores.setdefault(candidate, {}).update(values)


def _complete_time_series_report_metrics(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[TimeSeriesCandidate],
    metric_suite: MetricSuite,
    scores: Dict[object, Dict[str, float]],
) -> None:
    missing = _missing_report_metric_names(metric_suite.metric_names)
    if not missing:
        return
    completion = _score_time_series_report_completion(
        X,
        y,
        candidates,
        MetricSuite(missing, mi_bins=metric_suite.mi_bins),
    )
    for candidate, values in completion.items():
        scores.setdefault(candidate, {}).update(values)


def _score_discrete_report_completion(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[DiscreteFunctionCandidate],
    metric_suite: MetricSuite,
) -> Dict[object, Dict[str, float]]:
    from ..discrete import score_discrete_candidates

    return dict(score_discrete_candidates(X, y, candidates, metric_suite))


def _score_time_series_report_completion(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[TimeSeriesCandidate],
    metric_suite: MetricSuite,
) -> Dict[object, Dict[str, float]]:
    from ..time_series import score_time_series_candidates

    return dict(score_time_series_candidates(X, y, candidates, metric_suite))


def _stats_to_metrics(stats: Sequence[float], metric_names: Sequence[str]) -> Dict[str, float]:
    corr = _pearson_from_stats(stats)
    out: Dict[str, float] = {}
    for name in metric_names:
        if name == "pearson":
            out[name] = corr
        elif name == "r2":
            out[name] = corr * corr
    return out


def _pearson_from_stats(stats: Sequence[float]) -> float:
    n, sx, sy, sxx, syy, sxy = [float(value) for value in stats[:6]]
    if n <= 0.0:
        return 0.0
    numerator = n * sxy - sx * sy
    denom_x = n * sxx - sx * sx
    denom_y = n * syy - sy * sy
    denom = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
    if denom <= 0.0:
        return 0.0
    return numerator / denom


def _discrete_arrays(candidates: Sequence[DiscreteFunctionCandidate]) -> Dict[str, List[float] | List[int]]:
    kinds: List[int] = []
    feature_a: List[int] = []
    feature_b: List[int] = []
    value_feature: List[int] = []
    directions: List[int] = []
    params: List[float] = []
    scales: List[float] = []
    sharpness: List[float] = []

    for candidate in candidates:
        fa = int(candidate.feature_indices[0])
        fb = int(candidate.feature_indices[1]) if len(candidate.feature_indices) > 1 else 0
        vf = int(candidate.value_feature) if candidate.value_feature is not None else fa
        kinds.append(DISCRETE_FUNCTION_KIND_CODES[candidate.kind])
        feature_a.append(fa)
        feature_b.append(fb)
        value_feature.append(vf)
        directions.append(1 if candidate.direction == "le" else 0)
        params.extend(_discrete_params(candidate))
        scales.extend([
            _positive(candidate.scales[0] if len(candidate.scales) > 0 else 1.0),
            _positive(candidate.scales[1] if len(candidate.scales) > 1 else 1.0),
        ])
        sharpness.append(float(candidate.sharpness))
    return {
        "kinds": kinds,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "value_feature": value_feature,
        "directions": directions,
        "params": params,
        "scales": scales,
        "sharpness": sharpness,
    }


def _discrete_params(candidate: DiscreteFunctionCandidate) -> List[float]:
    if candidate.kind in (
        "discrete_function_soft_threshold",
        "discrete_function_value_gated_threshold",
    ):
        return [float(candidate.thresholds[0]), 0.0, 0.0, 0.0]
    if candidate.kind == "discrete_function_soft_interval":
        low, high = candidate.intervals[0]
        return [float(low), float(high), 0.0, 0.0]
    if candidate.kind in (
        "discrete_function_soft_rectangle",
        "discrete_function_value_in_soft_rectangle",
    ):
        low_a, high_a = candidate.intervals[0]
        low_b, high_b = candidate.intervals[1]
        return [float(low_a), float(high_a), float(low_b), float(high_b)]
    raise ValueError(f"Unsupported discrete candidate kind: {candidate.kind}")


def _residual_values(y_values: Sequence[float], baseline_pred) -> List[float]:
    if baseline_pred is None:
        y_mean = mean(y_values)
        return [float(value) - y_mean for value in y_values]
    pred = [float(value) for value in baseline_pred]
    if len(pred) != len(y_values):
        raise ValueError("baseline_pred must have the same length as y.")
    return [float(value) - pred_value for value, pred_value in zip(y_values, pred)]


def _positive(value: float) -> float:
    value = float(value)
    return value if value > 0.0 else 1.0


def _float_array(values: Sequence[float]):
    return (ctypes.c_float * len(values))(*(float(value) for value in values))


def _float_buffer(values):
    view = memoryview(values)
    if view.format not in ("f", "<f", "=f"):
        view = view.cast("f")
    return (ctypes.c_float * len(view)).from_buffer(view)


def _int_array(values: Sequence[int]):
    return (ctypes.c_int * len(values))(*(int(value) for value in values))


def _uint8_array(values: Sequence[int]):
    return (ctypes.c_uint8 * len(values))(*(int(value) & 0xFF for value in values))
