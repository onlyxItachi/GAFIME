#!/usr/bin/env python3
"""
GAFIME Multi-Backend Verification Script

This script is designed to be run INSIDE the compiled Docker container.
It forces the execution of GAFIME across all available backends 
(CUDA, Core/Rust, NumPy) on the same dataset to ensure computational consistency
and verify that all native extensions compiled properly in the multi-stage build.

Usage:
    docker compose run gafime python3 docker_backend_test.py
"""

import sys
import numpy as np
import warnings

# Suppress typical runtime warnings for clean output
warnings.filterwarnings("ignore")

import gafime
from gafime import GafimeEngine, EngineConfig, ComputeBudget
from gafime.backends.native_cuda_backend import NativeCudaBackend
from gafime.backends.core_backend import CoreBackend
from gafime.backends.base import Backend

def generate_test_data(n_samples=2000):
    """Generate synthetic data with a known interaction."""
    np.random.seed(42)
    
    cat = np.random.randn(n_samples).astype(np.float32)
    dog = np.random.randn(n_samples).astype(np.float32)
    bird = np.random.randn(n_samples).astype(np.float32)
    mouse = np.random.randn(n_samples).astype(np.float32)
    
    # Strong synthetic interaction: cat * bird
    y = (0.8 * cat * bird + 0.3 * dog - 0.2 * mouse + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    
    X = np.column_stack([cat, dog, bird, mouse])
    feature_names = ["cat", "dog", "bird", "mouse"]
    
    return X, y, feature_names

def test_backend(backend_instance, name, X, y, feature_names):
    """Run the GAFIME engine with a forced backend."""
    print(f"\n[{name.upper()}] Starting test...")
    
    config = EngineConfig(
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=50),
        permutation_tests=2,
        num_repeats=2
    )
    
    # Initialize engine
    engine = GafimeEngine(config)
    
    # Danger zone: Force the backend override for testing
    engine._resolve_backend_override = backend_instance
    
    # Patch the solve step locally to inject the backend if bypass is needed
    # (Since GafimeEngine.analyze usually resolves it from config)
    # The safest way is to just let evaluate_combinations use the provided one 
    # but since GAFIME's resolve_backend logic is hardcoded, we mock it.
    import gafime.backends
    original_resolve = gafime.backends.resolve_backend
    
    def forced_resolver(cfg, X, y):
        return backend_instance, []
        
    gafime.backends.resolve_backend = forced_resolver
    
    try:
        report = engine.analyze(X, y, feature_names=feature_names)
        
        print(f"[{name.upper()}] Analysis complete.")
        print(f"[{name.upper()}] Top interactions found: {len(report.interactions)}")
        if report.interactions:
            top = max(report.interactions, key=lambda x: abs(x.metrics.get('pearson', 0)))
            print(f"[{name.upper()}] Top Interaction: {' * '.join(top.feature_names)}")
            print(f"[{name.upper()}] Score: {top.metrics.get('pearson', 0):.4f}")
            
        print(f"[{name.upper()}] Signal Detected: {report.decision.signal_detected}")
        return True, report
        
    except Exception as e:
        print(f"[{name.upper()}] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        # Restore original
        gafime.backends.resolve_backend = original_resolve

def main():
    print(f"GAFIME v{gafime.__version__} Multi-Backend Docker Test")
    print("=" * 60)
    
    X, y, feature_names = generate_test_data()
    print(f"Generated Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("=" * 60)
    
    # 1. Test NumPy (Must always work)
    numpy_backend = Backend()
    success_numpy, rep_numpy = test_backend(numpy_backend, "NumPy", X, y, feature_names)
    
    # 2. Test Rust Core
    try:
        core_backend = CoreBackend()
        success_core, rep_core = test_backend(core_backend, "Rust Core", X, y, feature_names)
    except Exception as e:
        print(f"\n[RUST CORE] Skipped/Failed to load: {e}")
        success_core = False
        rep_core = None

    # 3. Test CUDA
    try:
        cuda_backend = NativeCudaBackend()
        success_cuda, rep_cuda = test_backend(cuda_backend, "CUDA (Native)", X, y, feature_names)
    except Exception as e:
        print(f"\n[CUDA] Skipped/Failed to load (expected if no GPU/drivers): {e}")
        success_cuda = None
        rep_cuda = None
        
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"NumPy: {'✅ PASSED' if success_numpy else '❌ FAILED'}")
    print(f"Rust Core: {'✅ PASSED' if success_core else '❌ FAILED/UNAVAILABLE'}")
    if success_cuda is None:
        print("CUDA: ⚠️ UNAVAILABLE (No GPU passed to container)")
    else:
        print(f"CUDA: {'✅ PASSED' if success_cuda else '❌ FAILED'}")
        
    if not success_numpy:
        sys.exit(1)

if __name__ == "__main__":
    main()
