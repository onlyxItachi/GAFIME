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
    
    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Find and load the native CUDA library."""
        lib_dir = Path(__file__).parent.parent.parent
        
        lib_names = [
            "gafime_cuda.dll", "libgafime_cuda.so", "gafime_cuda.so",
        ]
        
        search_paths = [
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
                        return ctypes.CDLL(str(lib_path))
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
        
        n_samples, n_features = X.shape
        n_combos = len(combos_list)
        
        # Prepare data
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        means = np.mean(X_f32, axis=0).astype(np.float32)
        
        # Pack combos
        combo_indices = []
        combo_offsets = [0]
        for combo in combos_list:
            combo_indices.extend(combo)
            combo_offsets.append(len(combo_indices))
        
        combo_indices = np.array(combo_indices, dtype=np.int32)
        combo_offsets = np.array(combo_offsets, dtype=np.int32)
        output = np.zeros((n_samples, n_combos), dtype=np.float32)
        
        # Call native kernel
        result = self.lib.gafime_feature_interaction_cuda(
            X_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            combo_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            combo_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_samples,
            n_features,
            n_combos,
        )
        
        if result != 0:
            logger.warning(f"CUDA kernel failed with code {result}, falling back to CPU")
            return super().score_combos(X, y, combos_list, metric_suite)
        
        # Score each combo's interaction vector
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for i, combo in enumerate(combos_list):
            vector = output[:, i].astype(np.float64)  # MetricSuite expects float64
            scores[combo] = metric_suite.score(vector, y)
        
        return scores
