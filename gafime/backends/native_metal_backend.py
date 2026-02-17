"""
GAFIME Native Metal Backend - Python Wrapper

Provides Python interface to native Metal compute kernels on Apple Silicon
via ctypes. Leverages Apple's Unified Memory Architecture (UMA) for
zero-copy data sharing between CPU and GPU.

Requires: macOS with Apple Silicon (M1/M2/M3/M4)
"""

from __future__ import annotations

import ctypes
import logging
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import ComputeBudget, EngineConfig
from ..metrics import MetricSuite
from .base import Backend, BackendInfo
from .fused_kernel import (
    GAFIME_SUCCESS,
    InteractionType,
    UnaryOp,
    compute_pearson_from_stats,
)

logger = logging.getLogger(__name__)

# ============================================================================
# LIBRARY LOADER
# ============================================================================

_METAL_LIB_CACHE: Optional[ctypes.CDLL] = None
_METAL_LIB_SETUP_DONE: bool = False


def _get_metal_library() -> Optional[ctypes.CDLL]:
    """Load the Metal backend shared library (dylib)."""
    global _METAL_LIB_CACHE, _METAL_LIB_SETUP_DONE
    
    if _METAL_LIB_SETUP_DONE:
        return _METAL_LIB_CACHE
    _METAL_LIB_SETUP_DONE = True
    
    # Only attempt on macOS + Apple Silicon
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    
    search_paths = [
        Path(__file__).parent.parent.parent / "gafime_metal.dylib",
        Path(__file__).parent.parent.parent / "src" / "metal" / "gafime_metal.dylib",
        Path(__file__).parent / "gafime_metal.dylib",
    ]
    
    for path in search_paths:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                _setup_metal_functions(lib)
                _METAL_LIB_CACHE = lib
                logger.info(f"Loaded Metal backend from {path}")
                return lib
            except OSError as e:
                logger.debug(f"Failed to load Metal backend from {path}: {e}")
    
    logger.debug("Metal backend library not found")
    return None


def _setup_metal_functions(lib: ctypes.CDLL) -> None:
    """Setup ctypes function signatures for Metal backend."""
    
    # gafime_metal_available
    lib.gafime_metal_available.restype = ctypes.c_int
    lib.gafime_metal_available.argtypes = []
    
    # gafime_metal_get_device_info
    lib.gafime_metal_get_device_info.restype = ctypes.c_int
    lib.gafime_metal_get_device_info.argtypes = [
        ctypes.c_char_p,          # name_out
        ctypes.POINTER(ctypes.c_int),  # memory_mb_out
        ctypes.POINTER(ctypes.c_int),  # gpu_family_out
    ]
    
    # gafime_metal_bucket_alloc
    lib.gafime_metal_bucket_alloc.restype = ctypes.c_int
    lib.gafime_metal_bucket_alloc.argtypes = [
        ctypes.c_int,             # n_samples
        ctypes.c_int,             # n_features
        ctypes.POINTER(ctypes.c_void_p),  # bucket_out
    ]
    
    # gafime_metal_bucket_upload_feature
    lib.gafime_metal_bucket_upload_feature.restype = ctypes.c_int
    lib.gafime_metal_bucket_upload_feature.argtypes = [
        ctypes.c_void_p,          # bucket
        ctypes.c_int,             # feature_index
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,             # n_samples
    ]
    
    # gafime_metal_bucket_upload_target
    lib.gafime_metal_bucket_upload_target.restype = ctypes.c_int
    lib.gafime_metal_bucket_upload_target.argtypes = [
        ctypes.c_void_p,          # bucket
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,             # n_samples
    ]
    
    # gafime_metal_bucket_upload_mask
    lib.gafime_metal_bucket_upload_mask.restype = ctypes.c_int
    lib.gafime_metal_bucket_upload_mask.argtypes = [
        ctypes.c_void_p,          # bucket
        ctypes.POINTER(ctypes.c_uint8),  # data
        ctypes.c_int,             # n_samples
    ]
    
    # gafime_metal_bucket_compute
    lib.gafime_metal_bucket_compute.restype = ctypes.c_int
    lib.gafime_metal_bucket_compute.argtypes = [
        ctypes.c_void_p,          # bucket
        ctypes.POINTER(ctypes.c_int),    # ops
        ctypes.c_int,             # arity
        ctypes.POINTER(ctypes.c_int),    # interaction_types
        ctypes.c_int,             # val_fold_id
        ctypes.POINTER(ctypes.c_float),  # stats_out
    ]
    
    # gafime_metal_bucket_free
    lib.gafime_metal_bucket_free.restype = ctypes.c_int
    lib.gafime_metal_bucket_free.argtypes = [ctypes.c_void_p]
    
    # gafime_metal_fused_interaction
    lib.gafime_metal_fused_interaction.restype = ctypes.c_int
    lib.gafime_metal_fused_interaction.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # h_inputs
        ctypes.POINTER(ctypes.c_float),  # h_target
        ctypes.POINTER(ctypes.c_uint8),  # h_mask
        ctypes.POINTER(ctypes.c_int),    # h_ops
        ctypes.c_int,                    # arity
        ctypes.c_int,                    # interaction_type
        ctypes.c_int,                    # val_fold_id
        ctypes.c_int,                    # n_samples
        ctypes.POINTER(ctypes.c_float),  # h_stats
    ]


# ============================================================================
# METAL BACKEND CLASS
# ============================================================================


class NativeMetalBackend(Backend):
    """Native Metal backend for Apple Silicon GPUs.
    
    Uses Apple's Unified Memory Architecture (UMA) for zero-copy
    data sharing between CPU and GPU. No PCIe transfer overhead.
    """
    
    name = "metal-native"
    device_label = "metal"
    is_gpu = True
    
    def __init__(self) -> None:
        super().__init__()
        self.lib = _get_metal_library()
        if self.lib is None:
            raise ImportError("Metal backend not available")
        
        if not self.lib.gafime_metal_available():
            raise RuntimeError("Metal GPU not available (requires Apple Silicon)")
        
        self._cache_device_info()
    
    def _cache_device_info(self) -> None:
        """Cache Metal GPU device information."""
        name_buf = ctypes.create_string_buffer(256)
        memory_mb = ctypes.c_int(0)
        gpu_family = ctypes.c_int(0)
        
        ret = self.lib.gafime_metal_get_device_info(
            name_buf,
            ctypes.byref(memory_mb),
            ctypes.byref(gpu_family),
        )
        
        if ret == GAFIME_SUCCESS:
            self._gpu_name = name_buf.value.decode("utf-8", errors="replace")
            self._memory_mb = memory_mb.value
            self._gpu_family = gpu_family.value
            
            family_names = {7: "M1", 8: "M2", 9: "M3/M4"}
            family_label = family_names.get(self._gpu_family, f"Apple{self._gpu_family}")
            logger.info(
                f"Metal GPU: {self._gpu_name} ({family_label}), "
                f"Unified Memory: {self._memory_mb} MB"
            )
        else:
            self._gpu_name = "Unknown Metal GPU"
            self._memory_mb = 0
            self._gpu_family = 0
    
    def info(self) -> BackendInfo:
        """Return backend information."""
        return BackendInfo(
            name=self.name,
            device=f"metal:{self._gpu_name}",
            is_gpu=True,
            memory_total_mb=self._memory_mb,
            memory_free_mb=self._memory_mb,  # UMA: all memory is shared
        )
    
    def check_budget(
        self,
        X: np.ndarray,
        y: np.ndarray,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        """Check if data fits in unified memory budget."""
        warnings: List[str] = []
        est_bytes = self.estimate_bytes(X, y)
        est_mb = est_bytes / (1024 * 1024)
        
        # On UMA, we can use most of system RAM
        limit_mb = budget.max_vram_mb if budget.max_vram_mb else self._memory_mb
        
        if est_mb > limit_mb * 0.8:
            warnings.append(
                f"Data ({est_mb:.0f} MB) approaches unified memory limit "
                f"({limit_mb:.0f} MB). Performance may degrade."
            )
        
        if est_mb > limit_mb:
            warnings.append(
                f"Data ({est_mb:.0f} MB) exceeds unified memory limit. "
                f"Falling back to CPU."
            )
            return False, warnings
        
        return True, warnings
    
    def build_interaction_vector(
        self, X: np.ndarray, combo: Tuple[int, ...]
    ) -> np.ndarray:
        """Build interaction vector — falls back to NumPy for simple combos."""
        from ..utils import arrays
        return arrays.build_interaction_vector(X, combo, xp=np)
    
    def score_combos(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Score combos using Metal GPU compute."""
        combos_list = list(combos)
        if not combos_list:
            return {}
        
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        y_f32 = np.ascontiguousarray(y, dtype=np.float32)
        n_samples = X_f32.shape[0]
        
        # Use bucket API for batch efficiency
        max_arity = max(len(c) for c in combos_list)
        if max_arity < 2:
            max_arity = 2
        
        n_features = X_f32.shape[1]
        bucket = ctypes.c_void_p()
        
        ret = self.lib.gafime_metal_bucket_alloc(
            n_samples, n_features, ctypes.byref(bucket)
        )
        if ret != GAFIME_SUCCESS:
            logger.warning("Metal bucket alloc failed, falling back to NumPy")
            return super().score_combos(X, y, combos_list, metric_suite)
        
        try:
            # Upload all features
            for i in range(n_features):
                col = np.ascontiguousarray(X_f32[:, i])
                self.lib.gafime_metal_bucket_upload_feature(
                    bucket, i,
                    col.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_samples,
                )
            
            # Upload target
            self.lib.gafime_metal_bucket_upload_target(
                bucket,
                y_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_samples,
            )
            
            # Create fold mask (all zeros = single fold, all training)
            mask = np.zeros(n_samples, dtype=np.uint8)
            self.lib.gafime_metal_bucket_upload_mask(
                bucket,
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                n_samples,
            )
            
            scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
            
            for combo in combos_list:
                arity = len(combo)
                if arity < 2:
                    vector = self.build_interaction_vector(X, combo)
                    scores[combo] = metric_suite.score(vector, y)
                    continue
                
                ops = (ctypes.c_int * arity)(*([UnaryOp.IDENTITY] * arity))
                interact_types = (ctypes.c_int * (arity - 1))(
                    *([InteractionType.MULT] * (arity - 1))
                )
                stats = (ctypes.c_float * 12)()
                
                ret = self.lib.gafime_metal_bucket_compute(
                    bucket, ops, arity, interact_types, 255, stats
                )
                
                if ret == GAFIME_SUCCESS:
                    stats_np = np.array(stats[:], dtype=np.float32)
                    train_r, val_r = compute_pearson_from_stats(stats_np)
                    scores[combo] = {"pearson": train_r}
                else:
                    vector = self.build_interaction_vector(X, combo)
                    scores[combo] = metric_suite.score(vector, y)
            
            return scores
            
        finally:
            self.lib.gafime_metal_bucket_free(bucket)
