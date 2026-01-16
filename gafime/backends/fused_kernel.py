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
    IDENTITY = 0
    LOG = 1
    EXP = 2
    SQRT = 3
    TANH = 4
    SIGMOID = 5
    SQUARE = 6
    NEGATE = 7
    ABS = 8
    INVERSE = 9
    CUBE = 10
    
    _names = {
        0: "identity", 1: "log", 2: "exp", 3: "sqrt", 4: "tanh",
        5: "sigmoid", 6: "square", 7: "negate", 8: "abs", 9: "inverse", 10: "cube"
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
        self.lib = self._load_library()
        self._setup_functions()
        
        # Allocate bucket
        self._bucket = ctypes.c_void_p()
        result = self.lib.gafime_bucket_alloc(
            n_samples, n_features, ctypes.byref(self._bucket)
        )
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Failed to allocate VRAM bucket: error {result}")
        
        logger.info(f"Allocated VRAM bucket: {n_samples} samples x {n_features} features")
    
    def _load_library(self) -> ctypes.CDLL:
        """Load native library (same as FusedKernelWrapper)."""
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
                    return ctypes.CDLL(str(lib_path.absolute()))
                except OSError as e:
                    logger.warning(f"Failed to load {lib_path}: {e}")
        
        raise ImportError("Native CUDA library not found")
    
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
            ctypes.c_int,                   # interaction_type
            ctypes.c_int,                   # val_fold_id
            ctypes.POINTER(ctypes.c_float), # h_stats
        ]
        
        # gafime_bucket_free
        self.lib.gafime_bucket_free.restype = ctypes.c_int
        self.lib.gafime_bucket_free.argtypes = [ctypes.c_void_p]
    
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
        interaction: int = InteractionType.MULT,
        val_fold: int = 0,
    ) -> np.ndarray:
        """
        Compute fused interaction on pre-uploaded data.
        
        NO cudaMalloc/cudaFree! Safe for millions of iterations.
        
        Args:
            feature_indices: Which features to use [arity] (0 to n_features-1)
            ops: Unary operator IDs for each feature
            interaction: Interaction type
            val_fold: Validation fold ID
        
        Returns:
            np.ndarray of 12 floats (train/val stats)
        """
        arity = len(feature_indices)
        if arity < 2 or arity > 5:
            raise ValueError(f"Arity must be 2-5, got {arity}")
        if len(ops) != arity:
            raise ValueError(f"ops length must match feature_indices length")
        
        indices_arr = np.array(feature_indices, dtype=np.int32)
        ops_arr = np.array(ops, dtype=np.int32)
        stats = np.zeros(12, dtype=np.float32)
        
        result = self.lib.gafime_bucket_compute(
            self._bucket,
            indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ops_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            arity,
            interaction,
            val_fold,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        
        if result != GAFIME_SUCCESS:
            raise RuntimeError(f"Bucket compute failed with code {result}")
        
        return stats
    
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
