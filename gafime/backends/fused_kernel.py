"""
GAFIME Fused Kernel Interface - Python Wrapper

Provides Python interface to the new operator-fused map-reduce CUDA kernel.
Implements stats-to-Pearson computation for train/val scoring.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLETON LIBRARY LOADER (Avoid reloading DLL per instance)
# ============================================================================

_GAFIME_LIB_CACHE: Optional[ctypes.CDLL] = None
_GAFIME_LIB_SETUP_DONE: bool = False

def _get_library() -> ctypes.CDLL:
    """Get singleton CUDA library (loaded once, shared by all instances)."""
    global _GAFIME_LIB_CACHE
    
    if _GAFIME_LIB_CACHE is not None:
        return _GAFIME_LIB_CACHE
    
    # Setup DLL search paths on Windows
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
                except (OSError, AttributeError):
                    pass
                break
    
    lib_dir = Path(__file__).parent.parent.parent
    lib_names = ["gafime_cuda.dll", "libgafime_cuda.so", "gafime_cuda.so"]
    
    for name in lib_names:
        lib_path = lib_dir / name
        if lib_path.exists():
            try:
                _GAFIME_LIB_CACHE = ctypes.CDLL(str(lib_path.absolute()))
                logger.info(f"Loaded GAFIME library: {lib_path.name}")
                return _GAFIME_LIB_CACHE
            except OSError as e:
                logger.warning(f"Failed to load {lib_path}: {e}")
    
    raise ImportError("Native CUDA library not found")

# ============================================================================
# GPU AUTO-DETECTION (Query hardware specs from CUDA)
# ============================================================================

from dataclasses import dataclass

@dataclass
class GpuConfig:
    """GPU configuration detected at runtime."""
    gpu_name: str
    block_size: int
    max_blocks: int
    sm_count: int
    compute_major: int
    compute_minor: int
    l2_cache_mb: float
    
    def __str__(self):
        return (f"GPU: {self.gpu_name}\n"
                f"  Compute: {self.compute_major}.{self.compute_minor}\n"
                f"  SMs: {self.sm_count}, Block size: {self.block_size}\n"
                f"  L2 Cache: {self.l2_cache_mb:.1f} MB")

_GPU_CONFIG_CACHE: Optional[GpuConfig] = None

def get_gpu_config() -> GpuConfig:
    """
    Query GPU configuration from CUDA.
    
    Returns auto-tuned parameters based on detected hardware:
    - block_size: Optimal threads per block
    - max_blocks: Max blocks for grid  
    - sm_count: Number of streaming multiprocessors
    - compute capability, L2 cache size, etc.
    
    Example:
        config = get_gpu_config()
        print(config)
        # GPU: NVIDIA GeForce RTX 4060
        #   Compute: 8.9
        #   SMs: 24, Block size: 256
        #   L2 Cache: 32.0 MB
    """
    global _GPU_CONFIG_CACHE
    
    if _GPU_CONFIG_CACHE is not None:
        return _GPU_CONFIG_CACHE
    
    lib = _get_library()
    
    # Setup function signature
    lib.gafime_get_gpu_config.restype = ctypes.c_int
    lib.gafime_get_gpu_config.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # block_size
        ctypes.POINTER(ctypes.c_int),  # max_blocks
        ctypes.POINTER(ctypes.c_int),  # sm_count
        ctypes.POINTER(ctypes.c_int),  # compute_major
        ctypes.POINTER(ctypes.c_int),  # compute_minor
        ctypes.POINTER(ctypes.c_int),  # l2_cache_bytes
        ctypes.c_char_p,               # gpu_name
    ]
    
    block_size = ctypes.c_int()
    max_blocks = ctypes.c_int()
    sm_count = ctypes.c_int()
    compute_major = ctypes.c_int()
    compute_minor = ctypes.c_int()
    l2_cache_bytes = ctypes.c_int()
    gpu_name = ctypes.create_string_buffer(256)
    
    lib.gafime_get_gpu_config(
        ctypes.byref(block_size),
        ctypes.byref(max_blocks),
        ctypes.byref(sm_count),
        ctypes.byref(compute_major),
        ctypes.byref(compute_minor),
        ctypes.byref(l2_cache_bytes),
        gpu_name
    )
    
    _GPU_CONFIG_CACHE = GpuConfig(
        gpu_name=gpu_name.value.decode('utf-8'),
        block_size=block_size.value,
        max_blocks=max_blocks.value,
        sm_count=sm_count.value,
        compute_major=compute_major.value,
        compute_minor=compute_minor.value,
        l2_cache_mb=l2_cache_bytes.value / (1024 * 1024)
    )
    
    return _GPU_CONFIG_CACHE

# ============================================================================
# CONSTANTS (mirror interfaces.h)
# ============================================================================

# Status codes
GAFIME_SUCCESS = 0
GAFIME_ERROR_INVALID_ARGS = -1
GAFIME_ERROR_CUDA_NOT_AVAILABLE = -2
GAFIME_ERROR_OUT_OF_MEMORY = -3
GAFIME_ERROR_KERNEL_FAILED = -4

# Unary operators
class UnaryOp:
    # SFU-Heavy (Special Function Unit)
    IDENTITY = 0
    LOG = 1
    EXP = 2
    SQRT = 3
    TANH = 4
    SIGMOID = 5
    
    # ALU-Heavy (CUDA Core)
    SQUARE = 6
    NEGATE = 7
    ABS = 8
    INVERSE = 9
    CUBE = 10
    
    # Time-Series (Memory + ALU)
    ROLLING_MEAN = 11
    ROLLING_STD = 12
    
    _names = {
        0: "identity", 1: "log", 2: "exp", 3: "sqrt", 4: "tanh",
        5: "sigmoid", 6: "square", 7: "negate", 8: "abs", 9: "inverse", 10: "cube",
        11: "rolling_mean", 12: "rolling_std"
    }
    
    @classmethod
    def from_name(cls, name: str) -> int:
        """Get operator ID from name."""
        reverse = {v: k for k, v in cls._names.items()}
        return reverse.get(name.lower(), cls.IDENTITY)

# Interaction types
class InteractionType:
    MULT = 0
    ADD = 1
    SUB = 2
    DIV = 3
    MAX = 4
    MIN = 5
    
    _names = {0: "mult", 1: "add", 2: "sub", 3: "div", 4: "max", 5: "min"}
    
    @classmethod
    def from_name(cls, name: str) -> int:
        """Get interaction type from name."""
        reverse = {v: k for k, v in cls._names.items()}
        return reverse.get(name.lower(), cls.MULT)

# Stats array indices
STAT_TRAIN_N = 0
STAT_TRAIN_SX = 1
STAT_TRAIN_SY = 2
STAT_TRAIN_SXX = 3
STAT_TRAIN_SYY = 4
STAT_TRAIN_SXY = 5
STAT_VAL_N = 6
STAT_VAL_SX = 7
STAT_VAL_SY = 8
STAT_VAL_SXX = 9
STAT_VAL_SYY = 10
STAT_VAL_SXY = 11


# ============================================================================
# PEARSON FROM STATS
# ============================================================================

def pearson_from_stats(n: float, sx: float, sy: float, 
                       sxx: float, syy: float, sxy: float) -> float:
    """
    Compute Pearson correlation from accumulated statistics.
    
    Formula: r = (NΣxy - ΣxΣy) / sqrt((NΣx² - (Σx)²)(NΣy² - (Σy)²))
    
    Args:
        n: Count of samples
        sx: Sum of X
        sy: Sum of Y
        sxx: Sum of X²
        syy: Sum of Y²
        sxy: Sum of X*Y
    
    Returns:
        Pearson correlation coefficient
    """
    if n < 2:
        return 0.0
    
    cov = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    
    if var_x <= 0 or var_y <= 0:
        return 0.0
    
    return float(cov / np.sqrt(var_x * var_y))


def unpack_stats(stats: np.ndarray) -> Tuple[dict, dict]:
    """
    Unpack 12-float stats array into train and val dictionaries.
    
    Returns:
        (train_stats, val_stats) dictionaries with n, sx, sy, sxx, syy, sxy
    """
    train = {
        "n": stats[STAT_TRAIN_N],
        "sx": stats[STAT_TRAIN_SX],
        "sy": stats[STAT_TRAIN_SY],
        "sxx": stats[STAT_TRAIN_SXX],
        "syy": stats[STAT_TRAIN_SYY],
        "sxy": stats[STAT_TRAIN_SXY],
    }
    val = {
        "n": stats[STAT_VAL_N],
        "sx": stats[STAT_VAL_SX],
        "sy": stats[STAT_VAL_SY],
        "sxx": stats[STAT_VAL_SXX],
        "syy": stats[STAT_VAL_SYY],
        "sxy": stats[STAT_VAL_SXY],
    }
    return train, val


def compute_pearson_from_stats(stats: np.ndarray) -> Tuple[float, float]:
    """
    Compute train and validation Pearson from 12-float stats array.
    
    Returns:
        (train_pearson, val_pearson)
    """
    train, val = unpack_stats(stats)
    
    train_r = pearson_from_stats(
        train["n"], train["sx"], train["sy"],
        train["sxx"], train["syy"], train["sxy"]
    )
    val_r = pearson_from_stats(
        val["n"], val["sx"], val["sy"],
        val["sxx"], val["syy"], val["sxy"]
    )
    
    return train_r, val_r


# ============================================================================
# FUSED KERNEL WRAPPER
# ============================================================================

class FusedKernelWrapper:
    """
    Python wrapper for the operator-fused map-reduce CUDA kernel.
    
    Example:
        wrapper = FusedKernelWrapper()
        stats = wrapper.compute(
            features=[X[:, 0], X[:, 1]],
            target=y,
            mask=fold_mask,
            ops=[UnaryOp.LOG, UnaryOp.SQRT],
            interaction=InteractionType.MULT,
            val_fold=0
        )
        train_r, val_r = compute_pearson_from_stats(stats)
    """
    
    def __init__(self):
        self.lib = self._load_library()
        self._setup_functions()
        
        # GPU memory cache (static VRAM bucket)
        self._d_features: List[Optional[int]] = []
        self._d_target: Optional[int] = None
        self._d_mask: Optional[int] = None
        self._cached_n_samples = 0
    
    def _load_library(self) -> ctypes.CDLL:
        """Find and load the native CUDA library."""
        # Add CUDA bin to DLL search path on Windows
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
                    except (OSError, AttributeError):
                        pass
                    break
        
        lib_dir = Path(__file__).parent.parent.parent  # gafime/backends -> gafime -> project root
        lib_names = ["gafime_cuda.dll", "libgafime_cuda.so", "gafime_cuda.so"]
        
        for name in lib_names:
            lib_path = lib_dir / name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path.absolute()))
                except OSError as e:
                    logger.warning(f"Failed to load {lib_path}: {e}")
        
        raise ImportError("Native CUDA library not found")
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # gafime_cuda_available
        self.lib.gafime_cuda_available.restype = ctypes.c_int
        self.lib.gafime_cuda_available.argtypes = []
        
        # gafime_fused_interaction
        self.lib.gafime_fused_interaction.restype = ctypes.c_int
        self.lib.gafime_fused_interaction.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # h_inputs
            ctypes.POINTER(ctypes.c_float),                   # d_target
            ctypes.POINTER(ctypes.c_uint8),                   # d_mask
            ctypes.POINTER(ctypes.c_int),                     # h_ops
            ctypes.c_int,                                     # arity
            ctypes.c_int,                                     # interaction_type
            ctypes.c_int,                                     # val_fold_id
            ctypes.c_int,                                     # n_samples
            ctypes.POINTER(ctypes.c_float),                   # h_stats
        ]
    
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.lib.gafime_cuda_available() == 1
    
    def compute(
        self,
        features: List[np.ndarray],
        target: np.ndarray,
        mask: np.ndarray,
        ops: List[int],
        interaction: int = InteractionType.MULT,
        val_fold: int = 0,
    ) -> np.ndarray:
        """
        Compute fused feature interaction and return stats.
        
        Args:
            features: List of feature arrays [n_samples] each
            target: Target array [n_samples]
            mask: Fold mask array [n_samples], uint8
            ops: Unary operator IDs for each feature
            interaction: Interaction type (MULT, ADD, etc.)
            val_fold: Validation fold ID
        
        Returns:
            np.ndarray of 12 floats (train/val stats)
        """
        arity = len(features)
        if arity < 2 or arity > 5:
            raise ValueError(f"Arity must be 2-5, got {arity}")
        
        n_samples = len(target)
        
        # Ensure contiguous float32
        features_f32 = [np.ascontiguousarray(f, dtype=np.float32) for f in features]
        target_f32 = np.ascontiguousarray(target, dtype=np.float32)
        mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
        ops_arr = np.array(ops, dtype=np.int32)
        
        # Create array of pointers to feature data
        PointerArray = ctypes.POINTER(ctypes.c_float) * arity
        feature_ptrs = PointerArray()
        for i, f in enumerate(features_f32):
            feature_ptrs[i] = f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Output stats
        stats = np.zeros(12, dtype=np.float32)
        
        result = self.lib.gafime_fused_interaction(
            ctypes.cast(feature_ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            target_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ops_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            arity,
            interaction,
            val_fold,
            n_samples,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"CUDA kernel failed with code {result}")
        
        return stats


# ============================================================================
# STATIC VRAM BUCKET (Zero-malloc iteration loops)
# ============================================================================

class StaticBucket:
    """
    Static VRAM bucket for high-frequency iteration loops.
    
    Follows the "Load Data -> Use It Constantly" pattern:
    - Allocate ONCE at initialization
    - Upload data ONCE (or when batch changes)
    - Compute millions of times with NO cudaMalloc/cudaFree
    - Free ONCE at destruction
    
    Example:
        bucket = StaticBucket(n_samples=10000, n_features=5)
        bucket.upload_all(features=[f0, f1, f2, f3, f4], target=y, mask=fold_mask)
        
        # Run millions of iterations - NO GPU memory allocation!
        for combo, ops in itertools.product(combos, op_configs):
            stats = bucket.compute(
                feature_indices=[0, 1],
                ops=[UnaryOp.LOG, UnaryOp.SQRT],
                interaction=InteractionType.MULT,
                val_fold=0
            )
            train_r, val_r = compute_pearson_from_stats(stats)
        
        del bucket  # Free GPU memory
    """
    
    def __init__(self, n_samples: int, n_features: int):
        """
        Allocate static VRAM bucket.
        
        Args:
            n_samples: Number of samples (rows)
            n_features: Number of feature columns to allocate (max 5)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Use singleton library loader (no DLL reload per instance)
        self.lib = _get_library()
        self._setup_functions()
        
        # Allocate bucket on GPU
        self._bucket = ctypes.c_void_p()
        result = self.lib.gafime_bucket_alloc(
            n_samples, n_features, ctypes.byref(self._bucket)
        )
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Failed to allocate VRAM bucket: error {result}")
        
        # =====================================================================
        # PRE-ALLOCATED BUFFERS (zero allocation in hot loop)
        # =====================================================================
        # These buffers are reused across millions of compute() calls
        self._indices_buf = np.zeros(5, dtype=np.int32)      # Max 5 features
        self._ops_buf = np.zeros(5, dtype=np.int32)          # Max 5 ops
        self._interact_buf = np.zeros(4, dtype=np.int32)     # Max 4 interactions
        self._stats_buf = np.zeros(12, dtype=np.float32)     # Always 12 stats
        
        # Pre-compute ctypes pointers (avoid per-call overhead)
        self._indices_ptr = self._indices_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._ops_ptr = self._ops_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._interact_ptr = self._interact_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._stats_ptr = self._stats_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        logger.info(f"Allocated VRAM bucket: {n_samples} samples x {n_features} features")
    
    def _setup_functions(self):
        """Setup ctypes function signatures for bucket API."""
        # gafime_bucket_alloc
        self.lib.gafime_bucket_alloc.restype = ctypes.c_int
        self.lib.gafime_bucket_alloc.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
        ]
        
        # gafime_bucket_upload_feature
        self.lib.gafime_bucket_upload_feature.restype = ctypes.c_int
        self.lib.gafime_bucket_upload_feature.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]
        
        # gafime_bucket_upload_target
        self.lib.gafime_bucket_upload_target.restype = ctypes.c_int
        self.lib.gafime_bucket_upload_target.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
        ]
        
        # gafime_bucket_upload_mask
        self.lib.gafime_bucket_upload_mask.restype = ctypes.c_int
        self.lib.gafime_bucket_upload_mask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)
        ]
        
        # gafime_bucket_compute
        self.lib.gafime_bucket_compute.restype = ctypes.c_int
        self.lib.gafime_bucket_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),  # feature_indices
            ctypes.POINTER(ctypes.c_int),  # ops
            ctypes.c_int,                   # arity
            ctypes.POINTER(ctypes.c_int),   # interaction_types (array of arity-1)
            ctypes.c_int,                   # val_fold_id
            ctypes.POINTER(ctypes.c_float), # h_stats
        ]
        
        # gafime_bucket_free
        self.lib.gafime_bucket_free.restype = ctypes.c_int
        self.lib.gafime_bucket_free.argtypes = [ctypes.c_void_p]
        
        # gafime_interleaved_compute (dual-slot SFU+ALU parallelism)
        self.lib.gafime_interleaved_compute.restype = ctypes.c_int
        self.lib.gafime_interleaved_compute.argtypes = [
            ctypes.c_void_p,                  # bucket
            ctypes.POINTER(ctypes.c_int),     # feature_indices_A
            ctypes.POINTER(ctypes.c_int),     # ops_A
            ctypes.c_int,                     # arity_A
            ctypes.c_int,                     # interact_A
            ctypes.POINTER(ctypes.c_int),     # feature_indices_B
            ctypes.POINTER(ctypes.c_int),     # ops_B
            ctypes.c_int,                     # arity_B
            ctypes.c_int,                     # interact_B
            ctypes.c_int,                     # window_size
            ctypes.c_int,                     # val_fold_id
            ctypes.POINTER(ctypes.c_float),   # h_stats_A
            ctypes.POINTER(ctypes.c_float),   # h_stats_B
        ]
        
        # gafime_bucket_compute_batch (N interactions in ONE kernel launch)
        self.lib.gafime_bucket_compute_batch.restype = ctypes.c_int
        self.lib.gafime_bucket_compute_batch.argtypes = [
            ctypes.c_void_p,                  # bucket
            ctypes.POINTER(ctypes.c_int),     # batch_indices [N*2]
            ctypes.POINTER(ctypes.c_int),     # batch_ops [N*2]
            ctypes.POINTER(ctypes.c_int),     # batch_interact [N]
            ctypes.c_int,                     # batch_size
            ctypes.c_int,                     # val_fold_id
            ctypes.POINTER(ctypes.c_float),   # h_stats_batch [N*12]
        ]
    
    def upload_feature(self, feature_idx: int, data: np.ndarray):
        """Upload a single feature column to the bucket."""
        if feature_idx < 0 or feature_idx >= self.n_features:
            raise ValueError(f"Feature index {feature_idx} out of range [0, {self.n_features})")
        
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        result = self.lib.gafime_bucket_upload_feature(
            self._bucket, feature_idx,
            data_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Failed to upload feature {feature_idx}")
    
    def upload_target(self, target: np.ndarray):
        """Upload target vector to the bucket."""
        target_f32 = np.ascontiguousarray(target, dtype=np.float32)
        result = self.lib.gafime_bucket_upload_target(
            self._bucket,
            target_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if result != GAFIME_SUCCESS:
            raise RuntimeError("Failed to upload target")
    
    def upload_mask(self, mask: np.ndarray):
        """Upload fold mask to the bucket."""
        mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
        result = self.lib.gafime_bucket_upload_mask(
            self._bucket,
            mask_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        )
        if result != GAFIME_SUCCESS:
            raise RuntimeError("Failed to upload mask")
    
    def upload_all(
        self,
        features: List[np.ndarray],
        target: np.ndarray,
        mask: np.ndarray
    ):
        """Upload all data to the bucket at once."""
        if len(features) > self.n_features:
            raise ValueError(f"Too many features: {len(features)} > {self.n_features}")
        
        for i, f in enumerate(features):
            self.upload_feature(i, f)
        self.upload_target(target)
        self.upload_mask(mask)
    
    def compute(
        self,
        feature_indices: List[int],
        ops: List[int],
        interaction_types: List[int] = None,
        val_fold: int = 0,
    ) -> np.ndarray:
        """
        Compute fused interaction on pre-uploaded data.
        
        ZERO ALLOCATION! Uses pre-allocated buffers for millions of iterations.
        
        Args:
            feature_indices: Which features to use [arity] (0 to n_features-1)
            ops: Unary operator IDs for each feature
            interaction_types: Per-pair interaction types (arity-1 elements)
                               e.g., for A*B+C: [MULT, ADD]
                               If single int or None, defaults to MULT for all pairs
            val_fold: Validation fold ID
        
        Returns:
            np.ndarray of 12 floats (train/val stats)
        """
        arity = len(feature_indices)
        if arity < 2 or arity > 5:
            raise ValueError(f"Arity must be 2-5, got {arity}")
        if len(ops) != arity:
            raise ValueError(f"ops length must match feature_indices length")
        
        # Handle interaction_types: default to MULT for all if not specified
        if interaction_types is None:
            interaction_types = [InteractionType.MULT] * (arity - 1)
        elif isinstance(interaction_types, int):
            # Backward compatibility: single int means use same for all pairs
            interaction_types = [interaction_types] * (arity - 1)
        
        if len(interaction_types) != arity - 1:
            raise ValueError(f"interaction_types must have {arity-1} elements for arity {arity}")
        
        # =====================================================================
        # ZERO-ALLOCATION: Copy into pre-allocated buffers
        # =====================================================================
        self._indices_buf[:arity] = feature_indices
        self._ops_buf[:arity] = ops
        self._interact_buf[:arity-1] = interaction_types
        self._stats_buf[:] = 0  # Zero stats buffer
        
        result = self.lib.gafime_bucket_compute(
            self._bucket,
            self._indices_ptr,
            self._ops_ptr,
            arity,
            self._interact_ptr,
            val_fold,
            self._stats_ptr,
        )
        
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Bucket compute failed with code {result}")
        
        # Return a COPY to prevent caller from modifying our buffer
        return self._stats_buf.copy()
    
    def interleaved_compute(
        self,
        feature_indices_A: List[int],
        ops_A: List[int],
        interaction_A: int = InteractionType.MULT,
        feature_indices_B: List[int] = None,
        ops_B: List[int] = None,
        interaction_B: int = InteractionType.MULT,
        window_size: int = 10,
        val_fold: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TWO feature interactions in parallel using SFU+ALU interleaving.
        
        Slot A: Use SFU-heavy ops (LOG, EXP, TANH, SIGMOID)
        Slot B: Use ALU-heavy ops (SQUARE, CUBE, ROLLING_MEAN, ROLLING_STD)
        
        While Slot A stalls on SFU, Slot B executes on CUDA cores.
        Result: ~2x throughput vs single-slot compute.
        
        Args:
            feature_indices_A: Slot A feature indices (must be 2)
            ops_A: Slot A unary operators
            interaction_A: Slot A interaction type
            feature_indices_B: Slot B feature indices (must be 2)
            ops_B: Slot B unary operators  
            interaction_B: Slot B interaction type
            window_size: Window size for rolling operators
            val_fold: Validation fold ID
        
        Returns:
            (stats_A, stats_B) - two 12-float arrays
        """
        if len(feature_indices_A) != 2 or len(feature_indices_B) != 2:
            raise ValueError("Interleaved compute currently requires arity=2 for both slots")
        
        indices_A = np.array(feature_indices_A, dtype=np.int32)
        ops_A_arr = np.array(ops_A, dtype=np.int32)
        indices_B = np.array(feature_indices_B, dtype=np.int32)
        ops_B_arr = np.array(ops_B, dtype=np.int32)
        
        stats_A = np.zeros(12, dtype=np.float32)
        stats_B = np.zeros(12, dtype=np.float32)
        
        result = self.lib.gafime_interleaved_compute(
            self._bucket,
            indices_A.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ops_A_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            2,  # arity_A
            interaction_A,
            indices_B.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ops_B_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            2,  # arity_B
            interaction_B,
            window_size,
            val_fold,
            stats_A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            stats_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Interleaved compute failed with code {result}")
        
        return stats_A, stats_B
    
    def compute_batch(
        self,
        feature_pairs: List[Tuple[int, int]],
        op_pairs: List[Tuple[int, int]],
        interactions: List[int],
        val_fold: int = 0,
    ) -> np.ndarray:
        """
        Compute N feature interactions in ONE kernel launch.
        
        Eliminates per-iteration kernel launch overhead by processing
        multiple interactions in parallel on the GPU.
        
        Args:
            feature_pairs: List of (f0, f1) feature index tuples
            op_pairs: List of (op0, op1) operator tuples
            interactions: List of interaction types (one per pair)
            val_fold: Validation fold ID
        
        Returns:
            np.ndarray of shape [N, 12] containing stats for each interaction
        
        Example:
            # Process 100 interactions in one kernel call
            stats = bucket.compute_batch(
                feature_pairs=[(0, 1), (0, 2), (1, 2), ...],
                op_pairs=[(LOG, SQRT), (IDENTITY, SQUARE), ...],
                interactions=[MULT, ADD, MULT, ...],
                val_fold=0
            )
            # stats.shape == (100, 12)
        """
        batch_size = len(feature_pairs)
        if batch_size <= 0 or batch_size > 1024:
            raise ValueError(f"Batch size must be 1-1024, got {batch_size}")
        if len(op_pairs) != batch_size or len(interactions) != batch_size:
            raise ValueError("feature_pairs, op_pairs, and interactions must have same length")
        
        # Flatten to contiguous arrays
        indices_flat = np.array([(p[0], p[1]) for p in feature_pairs], dtype=np.int32).ravel()
        ops_flat = np.array([(o[0], o[1]) for o in op_pairs], dtype=np.int32).ravel()
        interact_arr = np.array(interactions, dtype=np.int32)
        stats_batch = np.zeros(batch_size * 12, dtype=np.float32)
        
        result = self.lib.gafime_bucket_compute_batch(
            self._bucket,
            indices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ops_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            interact_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            batch_size,
            val_fold,
            stats_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Batch compute failed with code {result}")
        
        return stats_batch.reshape(batch_size, 12)
    
    def __del__(self):
        """Free VRAM bucket on destruction."""
        if hasattr(self, '_bucket') and self._bucket:
            self.lib.gafime_bucket_free(self._bucket)
            logger.debug("Freed VRAM bucket")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_fold_mask(n_samples: int, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """
    Create a fold mask for cross-validation.
    
    Args:
        n_samples: Number of samples
        n_folds: Number of CV folds
        seed: Random seed
    
    Returns:
        uint8 array with fold assignments (0 to n_folds-1)
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n_samples, dtype=np.uint8)


def numpy_reference(
    features: List[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    ops: List[int],
    interaction: int,
    val_fold: int,
) -> np.ndarray:
    """
    NumPy reference implementation for testing.
    """
    n = len(target)
    
    # Apply unary ops
    transformed = []
    for f, op in zip(features, ops):
        if op == UnaryOp.LOG:
            x = np.log(np.abs(f) + 1e-8)
        elif op == UnaryOp.EXP:
            x = np.exp(np.clip(f, -20, 20))
        elif op == UnaryOp.SQRT:
            x = np.sqrt(np.abs(f))
        elif op == UnaryOp.TANH:
            x = np.tanh(f)
        elif op == UnaryOp.SIGMOID:
            x = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
        elif op == UnaryOp.SQUARE:
            x = f * f
        elif op == UnaryOp.NEGATE:
            x = -f
        elif op == UnaryOp.ABS:
            x = np.abs(f)
        elif op == UnaryOp.INVERSE:
            x = 1.0 / np.where(np.abs(f) < 1e-8, np.sign(f) * 1e-8, f)
        elif op == UnaryOp.CUBE:
            x = f ** 3
        else:
            x = f.copy()
        transformed.append(x.astype(np.float32))
    
    # Combine
    X = transformed[0]
    for x in transformed[1:]:
        if interaction == InteractionType.MULT:
            X = X * x
        elif interaction == InteractionType.ADD:
            X = X + x
        elif interaction == InteractionType.SUB:
            X = X - x
        elif interaction == InteractionType.DIV:
            X = X / np.where(np.abs(x) < 1e-8, np.sign(x) * 1e-8, x)
        elif interaction == InteractionType.MAX:
            X = np.maximum(X, x)
        elif interaction == InteractionType.MIN:
            X = np.minimum(X, x)
    
    # Split and accumulate
    train_mask = mask != val_fold
    val_mask = mask == val_fold
    
    stats = np.zeros(12, dtype=np.float32)
    
    # Train
    X_train, Y_train = X[train_mask], target[train_mask]
    stats[0] = len(X_train)
    stats[1] = np.sum(X_train)
    stats[2] = np.sum(Y_train)
    stats[3] = np.sum(X_train ** 2)
    stats[4] = np.sum(Y_train ** 2)
    stats[5] = np.sum(X_train * Y_train)
    
    # Val
    X_val, Y_val = X[val_mask], target[val_mask]
    stats[6] = len(X_val)
    stats[7] = np.sum(X_val)
    stats[8] = np.sum(Y_val)
    stats[9] = np.sum(X_val ** 2)
    stats[10] = np.sum(Y_val ** 2)
    stats[11] = np.sum(X_val * Y_val)
    
    return stats
