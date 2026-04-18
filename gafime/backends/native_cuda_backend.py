"""
GAFIME Native CUDA Backend - Python Wrapper

This module provides a Python interface to the native CUDA kernels
via ctypes, with automatic fallback to CPU backend.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import ComputeBudget, EngineConfig
from ..metrics import MetricSuite
from .base import Backend, BackendInfo

logger = logging.getLogger(__name__)


class NativeCudaBackend(Backend):
    """Native CUDA backend using ctypes to load compiled kernels."""
    
    name = "cuda-native"
    device_label = "cuda"
    is_gpu = True
    
    def __init__(self, device_id: int = 0) -> None:
        super().__init__(device_id=device_id)
        
        self.lib = self._load_library()
        if self.lib is None:
            raise ImportError("Native CUDA library not found")
        
        # Check CUDA availability
        self._setup_functions()
        if not self._cuda_available():
            raise RuntimeError("CUDA not available on this system")
        
        self.device_id = device_id
        self._cache_device_info()

        # Persistent device-bucket cache: avoids re-uploading X across
        # repeated score_combos calls on the same data (e.g. permutation tests).
        # Tuple shape: (id_X, n_samples, n_features, h_data_buffer, h_mask, bucket_handle)
        self._bucket_cache: Optional[Tuple[int, int, int, np.ndarray, np.ndarray, ctypes.c_void_p]] = None
    
    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Find and load the native CUDA library."""
        import os
        
        lib_dir = Path(__file__).parent.parent.parent
        package_dir = Path(__file__).parent.parent
        
        # On Windows, add CUDA bin to DLL search path BEFORE loading
        if os.name == 'nt':
            cuda_paths = [
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
            ]
            for cuda_bin in cuda_paths:
                if os.path.exists(cuda_bin):
                    try:
                        os.add_dll_directory(cuda_bin)
                        logger.debug(f"Added CUDA DLL directory: {cuda_bin}")
                    except (OSError, AttributeError):
                        pass
                    break
        
        lib_names = [
            "gafime_cuda.dll", "libgafime_cuda.so", "gafime_cuda.so",
        ]
        
        search_paths = [
            package_dir,
            lib_dir,
            lib_dir / "build",
            lib_dir / "build" / "Release",
        ]
        
        for search_dir in search_paths:
            for name in lib_names:
                lib_path = search_dir / name
                if lib_path.exists():
                    try:
                        logger.debug(f"Loading native library: {lib_path}")
                        return ctypes.CDLL(str(lib_path.absolute()))
                    except OSError as e:
                        logger.warning(f"Failed to load {lib_path}: {e}")
        
        return None
    
    def _setup_functions(self) -> None:
        """Setup ctypes function signatures."""
        # gafime_cuda_available
        self.lib.gafime_cuda_available.restype = ctypes.c_int
        self.lib.gafime_cuda_available.argtypes = []
        
        # gafime_get_device_info
        self.lib.gafime_get_device_info.restype = ctypes.c_int
        self.lib.gafime_get_device_info.argtypes = [
            ctypes.c_int,                           # device_id
            ctypes.c_char_p,                        # name_out
            ctypes.POINTER(ctypes.c_int),           # memory_mb_out
            ctypes.POINTER(ctypes.c_int),           # compute_cap_major_out
            ctypes.POINTER(ctypes.c_int),           # compute_cap_minor_out
        ]
        
        # gafime_feature_interaction_cuda
        self.lib.gafime_feature_interaction_cuda.restype = ctypes.c_int
        self.lib.gafime_feature_interaction_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_float),         # X
            ctypes.POINTER(ctypes.c_float),         # means
            ctypes.POINTER(ctypes.c_float),         # output
            ctypes.POINTER(ctypes.c_int32),         # combo_indices
            ctypes.POINTER(ctypes.c_int32),         # combo_offsets
            ctypes.c_int32,                         # n_samples
            ctypes.c_int32,                         # n_features
            ctypes.c_int32,                         # n_combos
        ]

        # gafime_contiguous_bucket_* (batched fast path for pair-wise Pearson)
        # Bucket holds X + y on device; per-pair compute returns 12-float stats.
        try:
            self.lib.gafime_contiguous_bucket_alloc.restype = ctypes.c_int
            self.lib.gafime_contiguous_bucket_alloc.argtypes = [
                ctypes.c_int,                           # n_samples
                ctypes.c_int,                           # n_features
                ctypes.POINTER(ctypes.c_void_p),        # bucket_out
            ]
            self.lib.gafime_contiguous_bucket_upload.restype = ctypes.c_int
            self.lib.gafime_contiguous_bucket_upload.argtypes = [
                ctypes.c_void_p,                        # bucket
                ctypes.POINTER(ctypes.c_float),         # h_data
                ctypes.POINTER(ctypes.c_uint8),         # h_mask
            ]
            self.lib.gafime_contiguous_bucket_compute.restype = ctypes.c_int
            self.lib.gafime_contiguous_bucket_compute.argtypes = [
                ctypes.c_void_p,                        # bucket
                ctypes.c_int,                           # feature_a_idx
                ctypes.c_int,                           # feature_b_idx
                ctypes.c_int,                           # op_a
                ctypes.c_int,                           # op_b
                ctypes.c_int,                           # interact_type
                ctypes.c_int,                           # val_fold_id
                ctypes.POINTER(ctypes.c_float),         # h_stats_out (12 floats)
            ]
            self.lib.gafime_contiguous_bucket_free.restype = ctypes.c_int
            self.lib.gafime_contiguous_bucket_free.argtypes = [ctypes.c_void_p]
            self._has_bucket_api = True
        except AttributeError:
            # Older library without bucket API; fast path disabled.
            self._has_bucket_api = False
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.lib.gafime_cuda_available() == 1
    
    def _cache_device_info(self) -> None:
        """Cache GPU device information."""
        name_buf = ctypes.create_string_buffer(256)
        memory_mb = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        
        result = self.lib.gafime_get_device_info(
            self.device_id,
            name_buf,
            ctypes.byref(memory_mb),
            ctypes.byref(major),
            ctypes.byref(minor),
        )
        
        if result == 0:
            self._device_name = name_buf.value.decode('utf-8')
            self._memory_mb = memory_mb.value
            self._compute_cap = (major.value, minor.value)
            self.device_label = f"cuda:{self.device_id}"
        else:
            self._device_name = "Unknown"
            self._memory_mb = 0
            self._compute_cap = (0, 0)
    
    def info(self) -> BackendInfo:
        """Return backend information."""
        return BackendInfo(
            name=self.name,
            device=f"{self._device_name} (SM{self._compute_cap[0]}.{self._compute_cap[1]})",
            is_gpu=True,
            memory_total_mb=self._memory_mb,
            memory_free_mb=None,  # Not tracked per-call
        )
    
    def check_budget(
        self,
        X: np.ndarray,
        y: np.ndarray,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        """Check if data fits in VRAM budget."""
        warnings: List[str] = []
        
        if not budget.keep_in_vram:
            warnings.append("keep_in_vram is False; native CUDA backend disabled.")
            return False, warnings
        
        required_mb = self.estimate_bytes(X, y) / (1024 * 1024)
        
        if budget.vram_budget_mb > 0:
            if required_mb > budget.vram_budget_mb:
                warnings.append(
                    f"VRAM budget exceeded ({required_mb:.1f}MB > {budget.vram_budget_mb}MB); "
                    "falling back to CPU."
                )
                return False, warnings
        
        if required_mb > self._memory_mb * 0.8:  # Leave 20% headroom
            warnings.append(
                f"Data may exceed GPU memory ({required_mb:.1f}MB / {self._memory_mb}MB); "
                "consider using CPU backend."
            )
        
        return True, warnings
    
    def build_interaction_vector(self, X: np.ndarray, combo: Tuple[int, ...]) -> np.ndarray:
        """Build interaction vector using native CUDA kernel."""
        n_samples, n_features = X.shape
        
        # Prepare data
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        means = np.mean(X_f32, axis=0).astype(np.float32)
        
        # Single combo
        combo_indices = np.array(list(combo), dtype=np.int32)
        combo_offsets = np.array([0, len(combo)], dtype=np.int32)
        output = np.zeros((n_samples, 1), dtype=np.float32)
        
        result = self.lib.gafime_feature_interaction_cuda(
            X_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            combo_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            combo_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_samples,
            n_features,
            1,  # n_combos
        )
        
        if result != 0:
            logger.warning(f"CUDA kernel failed with code {result}, falling back to CPU")
            return super().build_interaction_vector(X, combo)
        
        return output.flatten()
    
    def score_combos(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Score multiple combos using native CUDA for interaction vectors."""
        combos_list = list(combos)
        if not combos_list:
            return {}

        # Fast path: persistent device bucket for Pearson on size-1/2 combos.
        # Pairs go through GPU bucket; unaries computed in NumPy from cached
        # pre-centered X (negligible cost vs GPU launch overhead).
        if (
            self._has_bucket_api
            and metric_suite.metric_names == ("pearson",)
            and all(len(c) in (1, 2) for c in combos_list)
        ):
            try:
                pairs = [c for c in combos_list if len(c) == 2]
                unaries = [c for c in combos_list if len(c) == 1]
                scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
                if pairs:
                    scores.update(self._score_pairs_with_bucket(X, y, pairs))
                elif unaries:
                    # Need the cache primed for the unary helper to reuse centered X.
                    self._score_pairs_with_bucket(X, y, [])
                if unaries:
                    scores.update(self._score_unaries_pearson(X, y, unaries))
                return scores
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"Bucket fast path failed ({exc}); falling back to legacy CUDA path."
                )

        return self._score_combos_legacy(X, y, combos_list, metric_suite)

    def _score_pairs_with_bucket(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pair_combos: List[Tuple[int, ...]],
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Pair-wise Pearson via persistent contiguous bucket.

        Layout uploaded to device (column-major float32):
            [F0_centered | F1_centered | ... | F_{F-1}_centered | y]
        plus a dummy mask of zeros. We use val_fold_id=255 so all rows count
        as 'train'. Pearson is computed from the 6-tuple of train sufficient
        stats (n, sx, sy, sxx, syy, sxy).

        We pre-center X so the kernel's `xa * xb` matches the host-side
        `(x0 - mean(x0)) * (x1 - mean(x1))` interaction.
        """
        n_samples, n_features = X.shape
        cache = self._bucket_cache

        if (
            cache is not None
            and cache[0] == id(X)
            and cache[1] == n_samples
            and cache[2] == n_features
        ):
            _, _, _, h_data, h_mask, bucket = cache
            # Same X — only y changes between permutations. Overwrite y region in place.
            np.copyto(
                h_data[n_features * n_samples:],
                np.ascontiguousarray(y, dtype=np.float32),
            )
        else:
            # New X (or shape change) — release old bucket and rebuild host buffers.
            self._free_bucket_cache()

            X_f32 = np.ascontiguousarray(X, dtype=np.float32)
            means = X_f32.mean(axis=0, dtype=np.float32)
            X_centered = X_f32 - means

            y_f32 = np.ascontiguousarray(y, dtype=np.float32)
            h_data = np.empty(n_samples * (n_features + 1), dtype=np.float32)
            for f in range(n_features):
                h_data[f * n_samples:(f + 1) * n_samples] = X_centered[:, f]
            h_data[n_features * n_samples:] = y_f32

            h_mask = np.zeros(n_samples, dtype=np.uint8)

            bucket = ctypes.c_void_p(0)
            rc = self.lib.gafime_contiguous_bucket_alloc(
                n_samples, n_features, ctypes.byref(bucket)
            )
            if rc != 0 or not bucket.value:
                raise RuntimeError(f"bucket_alloc failed (code {rc})")

            self._bucket_cache = (id(X), n_samples, n_features, h_data, h_mask, bucket)

        rc = self.lib.gafime_contiguous_bucket_upload(
            bucket,
            h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"bucket_upload failed (code {rc})")

        stats = np.zeros(12, dtype=np.float32)
        stats_ptr = stats.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}

        # GAFIME_OP_IDENTITY=0, GAFIME_INTERACT_MULT=0, val_fold_id=255 → all train
        for combo in pair_combos:
            a, b = int(combo[0]), int(combo[1])
            rc = self.lib.gafime_contiguous_bucket_compute(
                bucket, a, b, 0, 0, 0, 255, stats_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"bucket_compute failed (code {rc})")

            n = float(stats[0])
            sx = float(stats[1]); sy = float(stats[2])
            sxx = float(stats[3]); syy = float(stats[4]); sxy = float(stats[5])
            scores[combo] = {"pearson": _pearson_from_stats(n, sx, sy, sxx, syy, sxy)}

        return scores

    def _score_unaries_pearson(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unary_combos: List[Tuple[int, ...]],
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Vectorized Pearson(x_i, y) for unary combos via NumPy — much faster
        than a per-feature CUDA launch for the typical F<=1000 case."""
        if not unary_combos:
            return {}
        idxs = np.fromiter((c[0] for c in unary_combos), dtype=np.intp,
                           count=len(unary_combos))
        Xs = np.ascontiguousarray(X[:, idxs], dtype=np.float64)
        y64 = np.ascontiguousarray(y, dtype=np.float64)
        n = float(Xs.shape[0])
        x_mean = Xs.mean(axis=0); y_mean = y64.mean()
        xc = Xs - x_mean; yc = y64 - y_mean
        cov = (xc * yc[:, None]).sum(axis=0)
        denom = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(denom > 0, cov / denom, 0.0)
        return {c: {"pearson": float(r[i])} for i, c in enumerate(unary_combos)}

    def _free_bucket_cache(self) -> None:
        if self._bucket_cache is not None:
            try:
                self.lib.gafime_contiguous_bucket_free(self._bucket_cache[5])
            except Exception:
                pass
            self._bucket_cache = None

    def __del__(self) -> None:
        try:
            self._free_bucket_cache()
        except Exception:
            pass

    def _score_combos_legacy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos_list: List[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Legacy path: build all interaction vectors via per-call kernel,
        then score in Python. Required for combos of size != 2 or for metrics
        beyond Pearson."""
        n_samples, n_features = X.shape
        n_combos = len(combos_list)

        # Prepare data
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        means = np.mean(X_f32, axis=0).astype(np.float32)

        # Pack combos
        combo_indices: List[int] = []
        combo_offsets = [0]
        for combo in combos_list:
            combo_indices.extend(combo)
            combo_offsets.append(len(combo_indices))

        combo_indices_arr = np.array(combo_indices, dtype=np.int32)
        combo_offsets_arr = np.array(combo_offsets, dtype=np.int32)
        output = np.zeros((n_samples, n_combos), dtype=np.float32)

        result = self.lib.gafime_feature_interaction_cuda(
            X_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            combo_indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            combo_offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_samples,
            n_features,
            n_combos,
        )

        if result != 0:
            logger.warning(f"CUDA kernel failed with code {result}, falling back to CPU")
            return super().score_combos(X, y, combos_list, metric_suite)

        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for i, combo in enumerate(combos_list):
            vector = output[:, i].astype(np.float64)
            scores[combo] = metric_suite.score(vector, y)

        return scores


def _pearson_from_stats(n: float, sx: float, sy: float,
                        sxx: float, syy: float, sxy: float) -> float:
    """Pearson r computed from sufficient statistics over n samples.

    Matches semantics of cpu_metrics._safe_pearson: returns 0.0 when the
    denominator is zero (any variable has zero variance).
    """
    if n <= 1.0:
        return 0.0
    var_x = sxx - sx * sx / n
    var_y = syy - sy * sy / n
    cov = sxy - sx * sy / n
    denom_sq = var_x * var_y
    if denom_sq <= 0.0:
        return 0.0
    return float(cov / (denom_sq ** 0.5))
