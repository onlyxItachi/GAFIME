"""
Test script for per-pair interaction types fix.

This tests that we can now do mixed operations like A * B + C,
where different interaction types are used for each pair.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from gafime.config import EngineConfig
from gafime.backends import resolve_backend
from gafime.backends.fused_kernel import UnaryOp, InteractionType
from gafime.metrics import MetricSuite


def test_mixed_interactions():
    """Test that A * B + C works correctly with per-pair interaction types."""
    print("=" * 60)
    print("Testing Per-Pair Interaction Types Fix")
    print("=" * 60)
    
    config = EngineConfig(backend="auto")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Create test data
    A = np.random.randn(n_samples).astype(np.float32)
    B = np.random.randn(n_samples).astype(np.float32)
    C = np.random.randn(n_samples).astype(np.float32)
    
    # Target: we'll create a target that correlates with A*B+C
    target = (A * B + C + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    X = np.column_stack([A, B, C])
    
    # Resolve the active backend on this platform (Metal, CUDA, or CPU)
    backend, warnings = resolve_backend(config, X, target)
    print(f"[OK] Resolved backend: {backend.name} ({backend.device_label})")
    
    if not backend.is_gpu:
        pytest.skip(f"No GPU backend available for testing mixed interactions. Emitting: {warnings}", allow_module_level=True)

    metric_suite = backend.metric_suite(config)
    
    # We will test using score_combos which is the generic API for NativeCudaBackend / NativeMetalBackend
    combo = (0, 1, 2)
    
    # Unfortunately, the standard `score_combos` API does not currently expose deep manual 
    # interaction_types injections for testing backwards-compatibility inside the test.
    # To truly test the kernel, we have to invoke the low-level bucket directly if it's the CUDA backend
    
    if backend.name == "cuda-native":
        from gafime.backends.fused_kernel import StaticBucket, compute_pearson_from_stats, create_fold_mask
        
        mask = create_fold_mask(n_samples, n_folds=5, seed=42)
        bucket = StaticBucket(n_samples=n_samples, n_features=3)
        bucket.upload_all(features=[A, B, C], target=target, mask=mask)
        
        # Test 1: A * B + C (correct mixed interaction)
        stats_correct = bucket.compute(
            feature_indices=[0, 1, 2],
            ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
            interaction_types=[InteractionType.MULT, InteractionType.ADD],  # A*B then +C
            val_fold=0
        )
        train_r_correct, val_r_correct = compute_pearson_from_stats(stats_correct)
        
        # Test 2: A * B * C (all MULT)
        stats_mult = bucket.compute(
            feature_indices=[0, 1, 2],
            ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
            interaction_types=[InteractionType.MULT, InteractionType.MULT],  # A*B*C
            val_fold=0
        )
        train_r_mult, val_r_mult = compute_pearson_from_stats(stats_mult)
        
        assert train_r_correct > train_r_mult + 0.1, "A*B+C should correlate better than A*B*C"
        print(f"[SUCCESS] CUDA Per-pair interaction types working (Correct: {train_r_correct:.4f} > Mult: {train_r_mult:.4f})")
        
    elif backend.name == "metal-native":
        # The Metal backend has a similar bucket compute pattern we can extract
        from gafime.backends.fused_kernel import compute_pearson_from_stats, GAFIME_SUCCESS
        import ctypes
        
        bucket_ptr = ctypes.c_void_p()
        ret = backend.lib.gafime_metal_bucket_alloc(n_samples, 3, ctypes.byref(bucket_ptr))
        assert ret == GAFIME_SUCCESS
        
        try:
            for i, col in enumerate([A, B, C]):
                col_c = np.ascontiguousarray(col)
                backend.lib.gafime_metal_bucket_upload_feature(
                    bucket_ptr, i, col_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_samples
                )
            backend.lib.gafime_metal_bucket_upload_target(
                bucket_ptr, target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_samples
            )
            mask = np.zeros(n_samples, dtype=np.uint8)
            backend.lib.gafime_metal_bucket_upload_mask(
                bucket_ptr, mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n_samples
            )
            
            # Test 1: A * B + C
            ops = (ctypes.c_int * 3)(UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY)
            interact_correct = (ctypes.c_int * 2)(InteractionType.MULT, InteractionType.ADD)
            stats = (ctypes.c_float * 12)()
            
            backend.lib.gafime_metal_bucket_compute(bucket_ptr, ops, 3, interact_correct, 255, stats)
            train_r_correct, _ = compute_pearson_from_stats(np.array(stats[:], dtype=np.float32))
            
            # Test 2: A * B * C
            interact_mult = (ctypes.c_int * 2)(InteractionType.MULT, InteractionType.MULT)
            backend.lib.gafime_metal_bucket_compute(bucket_ptr, ops, 3, interact_mult, 255, stats)
            train_r_mult, _ = compute_pearson_from_stats(np.array(stats[:], dtype=np.float32))
            
            assert train_r_correct > train_r_mult + 0.1, "A*B+C should correlate better than A*B*C"
            print(f"[SUCCESS] Metal Per-pair interaction types working (Correct: {train_r_correct:.4f} > Mult: {train_r_mult:.4f})")
            
        finally:
            backend.lib.gafime_metal_bucket_free(bucket_ptr)

if __name__ == "__main__":
    test_mixed_interactions()
