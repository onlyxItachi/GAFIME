from __future__ import annotations

import ctypes
import importlib
import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import ComputeBudget
from ..metrics import MetricSuite
from ..discrete import (
    DISCRETE_FUNCTION_KIND_CODES,
    DiscreteFunctionCandidate,
    GPU_HARD_MODE_ERROR,
)
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
GAFIME_MAX_BUCKET_FEATURES = 5
GAFIME_MIN_BATCH_ARITY = 1
GAFIME_CANDIDATE_CONTINUOUS = 0
GAFIME_OP_IDENTITY = 0
GAFIME_INTERACT_MULT = 0
GAFIME_ROCM_MEMORY_DEVICE_COPY = 0
GAFIME_ROCM_MEMORY_UMA_HOST_MAPPED = 1
ROCM_STATS_METRICS = ("pearson", "r2")
ROCM_REPORT_METRICS = ("pearson", "spearman", "mutual_info", "r2")
DISCRETE_SELECTION_METRICS = (
    "mutual_info",
    "variance_reduction",
    "residual_abs_corr",
    "residual_r2_gain",
)


@dataclass(frozen=True)
class RocmPlatformInfo:
    device_kind: str
    runtime_arch_name: str
    memory_policy: str
    integrated: bool
    managed_memory: bool
    concurrent_managed_access: bool
    unified_addressing: bool
    pageable_memory_access: bool
    pageable_host_tables: bool
    direct_managed_host_access: bool
    can_map_host_memory: bool
    memory_bus_width_bits: int
    memory_clock_khz: int
    async_engine_count: int
    max_threads_per_multiprocessor: int
    is_large_bar: bool
    asic_revision: int
    memory_pools_supported: bool
    host_register_supported: bool
    gpu_direct_rdma_supported: bool
    multiprocessor_count: int
    l2_cache_size: int
    warp_size: int

    @property
    def label(self) -> str:
        return f"{self.device_kind}/{self.memory_policy}"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class NativeRocmBackend(Backend):
    name = "rocm-native"
    device_label = "rocm"
    is_gpu = True

    def __init__(self, device_id: int = 0) -> None:
        super().__init__(device_id=device_id)
        self.lib = self._load_library()
        if self.lib is None:
            raise ImportError("Native ROCm/HIP library not found")
        self._setup_functions()
        if not self._rocm_available():
            raise RuntimeError("ROCm/HIP not available on this system")
        self.device_id = int(device_id)
        self._cache_device_info()

    def _load_library(self) -> Optional[ctypes.CDLL]:
        package_dir = Path(__file__).parent.parent
        repo_dir = package_dir.parent
        for search_dir, names in self._library_search_paths(package_dir, repo_dir):
            if os.name == "nt":
                try:
                    os.add_dll_directory(str(search_dir))
                except (OSError, AttributeError):
                    pass
            for name in names:
                lib_path = search_dir / name
                if lib_path.exists():
                    return ctypes.CDLL(str(lib_path.absolute()))
        return None

    @staticmethod
    def _library_search_paths(package_dir: Path, repo_dir: Path) -> List[Tuple[Path, Tuple[str, ...]]]:
        payload_names = ("gafime_rocm.dll", "libgafime_rocm.so", "gafime_rocm.so", "gafime_rocm.pyd")
        search_paths: List[Tuple[Path, Tuple[str, ...]]] = []
        try:
            payload = importlib.import_module("gafime_rocm")
            package_dir_fn = getattr(payload, "package_dir", None)
            library_candidates_fn = getattr(payload, "library_candidates", None)
            if callable(package_dir_fn) and callable(library_candidates_fn):
                payload_dir = Path(package_dir_fn())
                names = tuple(Path(path).name for path in library_candidates_fn())
                search_paths.append((payload_dir, names or payload_names))
        except Exception:
            pass
        for search_dir in (package_dir, repo_dir, repo_dir / "build", repo_dir / "build" / "Release"):
            search_paths.append((search_dir, payload_names))
        return search_paths

    def _setup_functions(self) -> None:
        self.lib.gafime_rocm_available.restype = ctypes.c_int
        self.lib.gafime_rocm_available.argtypes = []

        self.lib.gafime_rocm_get_device_info.restype = ctypes.c_int
        self.lib.gafime_rocm_get_device_info.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._rocm_platform_info_fn = getattr(self.lib, "gafime_rocm_get_platform_info", None)
        if self._rocm_platform_info_fn is not None:
            self._rocm_platform_info_fn.restype = ctypes.c_int
            self._rocm_platform_info_fn.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]

        self.lib.gafime_rocm_bucket_alloc.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_alloc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._rocm_bucket_alloc_with_memory_mode_fn = getattr(
            self.lib,
            "gafime_rocm_bucket_alloc_with_memory_mode",
            None,
        )
        if self._rocm_bucket_alloc_with_memory_mode_fn is not None:
            self._rocm_bucket_alloc_with_memory_mode_fn.restype = ctypes.c_int
            self._rocm_bucket_alloc_with_memory_mode_fn.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            ]
        self._rocm_bucket_uses_host_mapped_inputs_fn = getattr(
            self.lib,
            "gafime_rocm_bucket_uses_host_mapped_inputs",
            None,
        )
        if self._rocm_bucket_uses_host_mapped_inputs_fn is not None:
            self._rocm_bucket_uses_host_mapped_inputs_fn.restype = ctypes.c_int
            self._rocm_bucket_uses_host_mapped_inputs_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
        self.lib.gafime_rocm_bucket_upload_feature.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_upload_feature.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_rocm_bucket_upload_target.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_upload_target.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_rocm_bucket_upload_mask.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_upload_mask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.lib.gafime_rocm_bucket_compute_batch.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_compute_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_rocm_bucket_free.restype = ctypes.c_int
        self.lib.gafime_rocm_bucket_free.argtypes = [ctypes.c_void_p]

        self.lib.gafime_rocm_matrix_alloc.restype = ctypes.c_int
        self.lib.gafime_rocm_matrix_alloc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._rocm_matrix_alloc_with_memory_mode_fn = getattr(
            self.lib,
            "gafime_rocm_matrix_alloc_with_memory_mode",
            None,
        )
        if self._rocm_matrix_alloc_with_memory_mode_fn is not None:
            self._rocm_matrix_alloc_with_memory_mode_fn.restype = ctypes.c_int
            self._rocm_matrix_alloc_with_memory_mode_fn.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            ]
        self._rocm_matrix_uses_host_mapped_inputs_fn = getattr(
            self.lib,
            "gafime_rocm_matrix_uses_host_mapped_inputs",
            None,
        )
        if self._rocm_matrix_uses_host_mapped_inputs_fn is not None:
            self._rocm_matrix_uses_host_mapped_inputs_fn.restype = ctypes.c_int
            self._rocm_matrix_uses_host_mapped_inputs_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
        self.lib.gafime_rocm_matrix_upload.restype = ctypes.c_int
        self.lib.gafime_rocm_matrix_upload.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_rocm_matrix_compute_batch.restype = ctypes.c_int
        self.lib.gafime_rocm_matrix_compute_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gafime_rocm_matrix_free.restype = ctypes.c_int
        self.lib.gafime_rocm_matrix_free.argtypes = [ctypes.c_void_p]

        self.lib.gafime_discrete_soft_batch_rocm.restype = ctypes.c_int
        self.lib.gafime_discrete_soft_batch_rocm.argtypes = [
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
        self.lib.gafime_discrete_selection_adaptive_rocm.restype = ctypes.c_int
        self.lib.gafime_discrete_selection_adaptive_rocm.argtypes = [
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

    def _rocm_available(self) -> bool:
        return bool(self.lib.gafime_rocm_available())

    def _cache_device_info(self) -> None:
        name_buf = ctypes.create_string_buffer(256)
        memory_mb = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        rc = self.lib.gafime_rocm_get_device_info(
            self.device_id,
            name_buf,
            ctypes.byref(memory_mb),
            ctypes.byref(major),
            ctypes.byref(minor),
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError("Unable to query ROCm/HIP device info.")
        self._device_name = name_buf.value.decode("utf-8", errors="ignore")
        self._memory_total_mb = int(memory_mb.value)
        self._compute_capability = (int(major.value), int(minor.value))
        self._platform_info = self._read_platform_info()
        self._last_input_memory_mode = "uninitialized"

    def _read_platform_info(self) -> RocmPlatformInfo:
        values = [ctypes.c_int() for _ in range(20)]
        arch_buf = ctypes.create_string_buffer(256)
        if self._rocm_platform_info_fn is None:
            return _rocm_platform_info_from_caps(
                self._device_name,
                self._compute_capability[0],
                self._compute_capability[1],
            )
        rc = self._rocm_platform_info_fn(
            ctypes.c_int(self.device_id),
            arch_buf,
            *[ctypes.byref(value) for value in values],
        )
        if rc != GAFIME_SUCCESS:
            return _rocm_platform_info_from_caps(
                self._device_name,
                self._compute_capability[0],
                self._compute_capability[1],
            )
        return _rocm_platform_info_from_caps(
            self._device_name,
            self._compute_capability[0],
            self._compute_capability[1],
            runtime_arch_name=arch_buf.value.decode("utf-8", errors="ignore"),
            integrated=values[0].value,
            managed_memory=values[1].value,
            concurrent_managed_access=values[2].value,
            unified_addressing=values[3].value,
            pageable_memory_access=values[4].value,
            pageable_host_tables=values[5].value,
            direct_managed_host_access=values[6].value,
            can_map_host_memory=values[7].value,
            memory_bus_width_bits=values[8].value,
            memory_clock_khz=values[9].value,
            async_engine_count=values[10].value,
            max_threads_per_multiprocessor=values[11].value,
            is_large_bar=values[12].value,
            asic_revision=values[13].value,
            memory_pools_supported=values[14].value,
            host_register_supported=values[15].value,
            gpu_direct_rdma_supported=values[16].value,
            multiprocessor_count=values[17].value,
            l2_cache_size=values[18].value,
            warp_size=values[19].value,
        )

    @property
    def platform_info(self) -> RocmPlatformInfo:
        return self._platform_info

    @property
    def memory_mode(self) -> int:
        return _rocm_memory_mode_from_platform(self._platform_info)

    @property
    def last_input_memory_mode(self) -> str:
        return self._last_input_memory_mode

    def info(self) -> BackendInfo:
        major, minor = self._compute_capability
        target = self._platform_info.runtime_arch_name or f"hip {major}.{minor}"
        return BackendInfo(
            name=self.name,
            device=f"{self._device_name} hip {major}.{minor} target={target} [{self._platform_info.label}]",
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
                f"Input estimate {estimated_mb:.1f} MB exceeds vram_budget_mb={budget.vram_budget_mb}."
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

        invalid = [combo for combo in combos_list if len(combo) < GAFIME_MIN_BATCH_ARITY or len(combo) > GAFIME_MAX_BUCKET_FEATURES]
        if invalid:
            raise ValueError("ROCm/HIP batch spine supports continuous combo arity 1 through 5.")

        scores = self._score_combos_stats_metrics(X, y, combos_list, metric_suite.metric_names)
        _complete_continuous_report_metrics(X, y, combos_list, metric_suite, scores)
        return scores

    def compile_session(self, X, y, scenario_plan, metric_suite, flags):
        from ..compile.sessions import ResidentContinuousMatrixSession

        def allocate_matrix(X_native: NativeMatrix, y_native: NativeVector):
            retained = []
            means = column_means(X_native)
            feature_major = X_native.feature_major_buffer()
            mask = _uint8_array([0] * X_native.n_samples)
            means_ptr = _float_array(means)
            retained.extend([feature_major, y_native.buffer, mask, means_ptr])
            matrix = ctypes.c_void_p()
            rc = self._alloc_matrix(
                X_native.n_samples,
                X_native.n_features,
                GAFIME_MAX_BATCH_SIZE,
                ctypes.byref(matrix),
            )
            if rc != GAFIME_SUCCESS:
                raise RuntimeError(f"gafime_rocm_matrix_alloc failed with code {rc}")
            try:
                rc = self.lib.gafime_rocm_matrix_upload(
                    matrix,
                    _float_buffer(feature_major),
                    _float_buffer(y_native.buffer),
                    mask,
                    means_ptr,
                )
                if rc != GAFIME_SUCCESS:
                    raise RuntimeError(f"gafime_rocm_matrix_upload failed with code {rc}")
                self._last_input_memory_mode = self._input_memory_mode_label(
                    self._matrix_uses_host_mapped_inputs(matrix)
                )
            except Exception:
                self.lib.gafime_rocm_matrix_free(matrix)
                raise
            return matrix, retained

        session = ResidentContinuousMatrixSession(
            self,
            X,
            y,
            scenario_plan,
            metric_suite,
            flags,
            allocate_matrix=allocate_matrix,
            free_matrix=self.lib.gafime_rocm_matrix_free,
            launch_global_batch=self._launch_global_continuous_batch,
            scheduler_batches=_continuous_scheduler_batches,
            stats_metric_names=_stats_metric_names,
            stats_to_metrics=_stats_to_metrics,
            complete_report_metrics=_complete_continuous_report_metrics,
            max_arity=GAFIME_MAX_BUCKET_FEATURES,
            graph_backend="hip",
            graph_capture_supported=hasattr(self.lib, "gafime_rocm_graph_launch"),
        )
        if (
            getattr(flags, "graph", False)
            and "host_mapped" in str(getattr(self, "_last_input_memory_mode", ""))
        ):
            session.warnings.append(
                "HIP graph capture requested with ROCm UMA host-mapped inputs; using normal launches."
            )
        return session

    def _score_combos_stats_metrics(
        self,
        X: NativeMatrix,
        y: NativeVector,
        combos: Sequence[Tuple[int, ...]],
        metric_names: Sequence[str],
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        stats_metric_names = _stats_metric_names(metric_names)
        out: Dict[Tuple[int, ...], Dict[str, float]] = {combo: {} for combo in combos}
        if not stats_metric_names:
            return out
        means = column_means(X)
        feature_major = X.feature_major_buffer()
        y_ptr = _float_buffer(y.buffer)
        mask_ptr = _uint8_array([0] * X.n_samples)
        means_ptr = _float_array(means)
        matrix = ctypes.c_void_p()
        rc = self._alloc_matrix(
            X.n_samples,
            X.n_features,
            GAFIME_MAX_BATCH_SIZE,
            ctypes.byref(matrix),
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_rocm_matrix_alloc failed with code {rc}")
        try:
            rc = self.lib.gafime_rocm_matrix_upload(
                matrix,
                _float_buffer(feature_major),
                y_ptr,
                mask_ptr,
                means_ptr,
            )
            if rc != GAFIME_SUCCESS:
                raise RuntimeError(f"gafime_rocm_matrix_upload failed with code {rc}")
            self._last_input_memory_mode = self._input_memory_mode_label(
                self._matrix_uses_host_mapped_inputs(matrix)
            )

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
            self.lib.gafime_rocm_matrix_free(matrix)
        return out

    def _launch_global_continuous_batch(
        self,
        matrix: ctypes.c_void_p,
        indices: Sequence[int],
        arity: int,
        batch_size: int,
    ) -> List[List[float]]:
        stats_out = (ctypes.c_float * (batch_size * 12))()
        rc = self.lib.gafime_rocm_matrix_compute_batch(
            matrix,
            _int_array(indices),
            ctypes.c_int(arity),
            ctypes.c_int(batch_size),
            ctypes.c_int(255),
            stats_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_rocm_matrix_compute_batch failed with code {rc}")
        return [
            [float(stats_out[row * 12 + col]) for col in range(12)]
            for row in range(batch_size)
        ]

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
        stats_metric_names = _stats_metric_names(metric_suite.metric_names)
        if not stats_metric_names:
            return _score_time_series_report_completion(X, y, candidates_list, metric_suite)

        scores: Dict[object, Dict[str, float]] = {}
        y_ptr = _float_buffer(y.buffer)
        mask_ptr = _uint8_array([0] * X.n_samples)
        for feature_group, group_candidates in _group_time_series_for_bucket(candidates_list):
            local_map = {feature: idx for idx, feature in enumerate(feature_group)}
            bucket = ctypes.c_void_p()
            rc = self._alloc_bucket(
                X.n_samples,
                len(feature_group),
                ctypes.byref(bucket),
            )
            if rc != GAFIME_SUCCESS:
                raise RuntimeError(f"gafime_rocm_bucket_alloc failed with code {rc}")
            try:
                upload_buffers = []
                for local_idx, feature in enumerate(feature_group):
                    feature_buffer = X.column_buffer(feature)
                    upload_buffers.append(feature_buffer)
                    rc = self.lib.gafime_rocm_bucket_upload_feature(
                        bucket,
                        ctypes.c_int(local_idx),
                        _float_buffer(feature_buffer),
                    )
                    if rc != GAFIME_SUCCESS:
                        raise RuntimeError(f"gafime_rocm_bucket_upload_feature failed with code {rc}")
                rc = self.lib.gafime_rocm_bucket_upload_target(bucket, y_ptr)
                if rc != GAFIME_SUCCESS:
                    raise RuntimeError(f"gafime_rocm_bucket_upload_target failed with code {rc}")
                rc = self.lib.gafime_rocm_bucket_upload_mask(bucket, mask_ptr)
                if rc != GAFIME_SUCCESS:
                    raise RuntimeError(f"gafime_rocm_bucket_upload_mask failed with code {rc}")
                self._last_input_memory_mode = self._input_memory_mode_label(
                    self._bucket_uses_host_mapped_inputs(bucket)
                )

                for batch in _chunks_objects(group_candidates, GAFIME_MAX_BATCH_SIZE):
                    stats = self._launch_time_series_batch(bucket, batch, local_map)
                    for candidate, row in zip(batch, stats):
                        scores[candidate] = _stats_to_metrics(row, stats_metric_names)
            finally:
                self.lib.gafime_rocm_bucket_free(bucket)
        _complete_time_series_report_metrics(X, y, candidates_list, metric_suite, scores)
        return scores

    def _alloc_matrix(
        self,
        n_samples: int,
        n_features: int,
        max_batch_size: int,
        matrix_out,
    ) -> int:
        if self._rocm_matrix_alloc_with_memory_mode_fn is not None:
            return self._rocm_matrix_alloc_with_memory_mode_fn(
                ctypes.c_int(n_samples),
                ctypes.c_int(n_features),
                ctypes.c_int(max_batch_size),
                ctypes.c_int(self.memory_mode),
                matrix_out,
            )
        return self.lib.gafime_rocm_matrix_alloc(
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            ctypes.c_int(max_batch_size),
            matrix_out,
        )

    def _alloc_bucket(self, n_samples: int, n_features: int, bucket_out) -> int:
        if self._rocm_bucket_alloc_with_memory_mode_fn is not None:
            return self._rocm_bucket_alloc_with_memory_mode_fn(
                ctypes.c_int(n_samples),
                ctypes.c_int(n_features),
                ctypes.c_int(self.memory_mode),
                bucket_out,
            )
        return self.lib.gafime_rocm_bucket_alloc(
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            bucket_out,
        )

    def _matrix_uses_host_mapped_inputs(self, matrix: ctypes.c_void_p) -> bool:
        if self._rocm_matrix_uses_host_mapped_inputs_fn is None:
            return False
        out = ctypes.c_int()
        rc = self._rocm_matrix_uses_host_mapped_inputs_fn(matrix, ctypes.byref(out))
        return rc == GAFIME_SUCCESS and bool(out.value)

    def _bucket_uses_host_mapped_inputs(self, bucket: ctypes.c_void_p) -> bool:
        if self._rocm_bucket_uses_host_mapped_inputs_fn is None:
            return False
        out = ctypes.c_int()
        rc = self._rocm_bucket_uses_host_mapped_inputs_fn(bucket, ctypes.byref(out))
        return rc == GAFIME_SUCCESS and bool(out.value)

    def _input_memory_mode_label(self, uses_host_mapped_inputs: bool) -> str:
        if uses_host_mapped_inputs:
            return "uma_host_mapped"
        if self.memory_mode == GAFIME_ROCM_MEMORY_UMA_HOST_MAPPED:
            return "device_copy_fallback"
        return "device_copy"

    def _launch_time_series_batch(
        self,
        bucket: ctypes.c_void_p,
        batch: Sequence[TimeSeriesCandidate],
        local_map: Dict[int, int],
    ) -> List[List[float]]:
        batch_size = len(batch)
        arity = 1
        kinds = [TIME_SERIES_KIND_CODES[candidate.kind] for candidate in batch]
        indices = [local_map[candidate.feature_index] for candidate in batch]
        ops = [GAFIME_OP_IDENTITY] * batch_size
        interactions = [GAFIME_INTERACT_MULT] * batch_size
        ts_params: List[int] = []
        for candidate in batch:
            ts_params.extend([int(candidate.lag), int(candidate.window), 0, 0])
        stats_out = (ctypes.c_float * (batch_size * 12))()
        rc = self.lib.gafime_rocm_bucket_compute_batch(
            bucket,
            _int_array(kinds),
            _int_array(indices),
            _int_array(ops),
            _int_array(interactions),
            _int_array(ts_params),
            ctypes.c_int(arity),
            ctypes.c_int(batch_size),
            ctypes.c_int(255),
            stats_out,
        )
        if rc != GAFIME_SUCCESS:
            raise RuntimeError(f"gafime_rocm_bucket_compute_batch failed with code {rc}")
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
        stats_metric_names = _stats_metric_names(metric_suite.metric_names)
        if not stats_metric_names:
            return _score_discrete_report_completion(X, y, candidates_list, metric_suite)

        arrays = _discrete_arrays(candidates_list)
        n_candidates = len(candidates_list)
        stats_out = (ctypes.c_float * (n_candidates * 12))()
        feature_major = X.feature_major_buffer()
        rc = self.lib.gafime_discrete_soft_batch_rocm(
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
            raise RuntimeError(f"gafime_discrete_soft_batch_rocm failed with code {rc}")
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
        rc = self.lib.gafime_discrete_selection_adaptive_rocm(
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
            raise RuntimeError(f"gafime_discrete_selection_adaptive_rocm failed with code {rc}")
        out: Dict[object, Dict[str, float]] = {}
        width = len(DISCRETE_SELECTION_METRICS)
        for row, candidate in enumerate(candidates_list):
            out[candidate] = {
                name: float(scores_out[row * width + col])
                for col, name in enumerate(DISCRETE_SELECTION_METRICS)
            }
        return out


def _bool_flag(value: int | bool | None) -> bool:
    return bool(value) if value is not None else False


def _rocm_platform_info_from_caps(
    device_name: str,
    compute_major: int,
    compute_minor: int,
    *,
    runtime_arch_name: str | None = None,
    integrated: int | bool | None = None,
    managed_memory: int | bool | None = None,
    concurrent_managed_access: int | bool | None = None,
    unified_addressing: int | bool | None = None,
    pageable_memory_access: int | bool | None = None,
    pageable_host_tables: int | bool | None = None,
    direct_managed_host_access: int | bool | None = None,
    can_map_host_memory: int | bool | None = None,
    memory_bus_width_bits: int = 0,
    memory_clock_khz: int = 0,
    async_engine_count: int = 0,
    max_threads_per_multiprocessor: int = 0,
    is_large_bar: int | bool | None = None,
    asic_revision: int = 0,
    memory_pools_supported: int | bool | None = None,
    host_register_supported: int | bool | None = None,
    gpu_direct_rdma_supported: int | bool | None = None,
    multiprocessor_count: int = 0,
    l2_cache_size: int = 0,
    warp_size: int = 0,
) -> RocmPlatformInfo:
    is_integrated = _bool_flag(integrated)
    has_unified = _bool_flag(unified_addressing)
    has_pageable = _bool_flag(pageable_memory_access)
    has_managed = _bool_flag(managed_memory)
    has_concurrent_managed = _bool_flag(concurrent_managed_access)
    has_host_tables = _bool_flag(pageable_host_tables)
    has_direct_host = _bool_flag(direct_managed_host_access)
    can_map_host = _bool_flag(can_map_host_memory)
    large_bar = _bool_flag(is_large_bar)
    has_memory_pools = _bool_flag(memory_pools_supported)
    has_host_register = _bool_flag(host_register_supported)
    has_gpu_direct_rdma = _bool_flag(gpu_direct_rdma_supported)

    del device_name, compute_major, compute_minor

    device_kind = "integrated_gpu" if is_integrated else "discrete_gpu"
    if is_integrated and (has_unified or has_pageable or has_managed or has_host_tables):
        memory_policy = "shared_system_memory"
    elif is_integrated:
        memory_policy = "integrated_device_memory"
    else:
        memory_policy = "device_memory"

    return RocmPlatformInfo(
        device_kind=device_kind,
        runtime_arch_name=str(runtime_arch_name or ""),
        memory_policy=memory_policy,
        integrated=is_integrated,
        managed_memory=has_managed,
        concurrent_managed_access=has_concurrent_managed,
        unified_addressing=has_unified,
        pageable_memory_access=has_pageable,
        pageable_host_tables=has_host_tables,
        direct_managed_host_access=has_direct_host,
        can_map_host_memory=can_map_host,
        memory_bus_width_bits=int(memory_bus_width_bits),
        memory_clock_khz=int(memory_clock_khz),
        async_engine_count=int(async_engine_count),
        max_threads_per_multiprocessor=int(max_threads_per_multiprocessor),
        is_large_bar=large_bar,
        asic_revision=int(asic_revision),
        memory_pools_supported=has_memory_pools,
        host_register_supported=has_host_register,
        gpu_direct_rdma_supported=has_gpu_direct_rdma,
        multiprocessor_count=int(multiprocessor_count),
        l2_cache_size=int(l2_cache_size),
        warp_size=int(warp_size),
    )


def _rocm_memory_mode_from_platform(platform: RocmPlatformInfo) -> int:
    if platform.memory_policy == "shared_system_memory" and (
        platform.can_map_host_memory
        or platform.host_register_supported
        or platform.pageable_memory_access
        or platform.pageable_host_tables
    ):
        return GAFIME_ROCM_MEMORY_UMA_HOST_MAPPED
    return GAFIME_ROCM_MEMORY_DEVICE_COPY


def _group_time_series_for_bucket(
    candidates: Sequence[TimeSeriesCandidate],
) -> List[Tuple[Tuple[int, ...], List[TimeSeriesCandidate]]]:
    groups: Dict[Tuple[int, ...], List[TimeSeriesCandidate]] = {}
    for candidate in candidates:
        key = (int(candidate.feature_index),)
        groups.setdefault(key, []).append(candidate)
    return sorted(groups.items(), key=lambda item: item[0])


def _continuous_scheduler_batches(combos: Sequence[Tuple[int, ...]]):
    try:
        from .. import subfunctions
    except ImportError as exc:
        raise RuntimeError(
            "ROCm/HIP continuous batch scoring requires the Rust subfunctions scheduler. "
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


def _chunks_objects(values: Sequence[object], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stats_to_metrics(stats: Sequence[float], metric_names: Sequence[str]) -> Dict[str, float]:
    corr = _pearson_from_stats(stats)
    out: Dict[str, float] = {}
    for name in metric_names:
        if name == "pearson":
            out[name] = corr
        elif name == "r2":
            out[name] = corr * corr
    return out


def _stats_metric_names(metric_names: Sequence[str]) -> Tuple[str, ...]:
    unsupported = [name for name in metric_names if name not in ROCM_REPORT_METRICS]
    if unsupported:
        raise ValueError(
            f"ROCm/HIP report metrics are {ROCM_REPORT_METRICS}; "
            f"unsupported metrics requested: {tuple(unsupported)}."
        )
    return tuple(name for name in metric_names if name in ROCM_STATS_METRICS)


def _missing_report_metric_names(metric_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(name for name in metric_names if name not in ROCM_STATS_METRICS)


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


def _score_discrete_report_completion(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[DiscreteFunctionCandidate],
    metric_suite: MetricSuite,
) -> Dict[object, Dict[str, float]]:
    from ..discrete import score_discrete_candidates

    return dict(score_discrete_candidates(X, y, candidates, metric_suite))


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


def _score_time_series_report_completion(
    X: NativeMatrix,
    y: NativeVector,
    candidates: Sequence[TimeSeriesCandidate],
    metric_suite: MetricSuite,
) -> Dict[object, Dict[str, float]]:
    from ..time_series import score_time_series_candidates

    return dict(score_time_series_candidates(X, y, candidates, metric_suite))


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


def _column_major_values(X: NativeMatrix) -> List[float]:
    values: List[float] = []
    for feature in range(X.n_features):
        values.extend(X.column(feature))
    return values


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
