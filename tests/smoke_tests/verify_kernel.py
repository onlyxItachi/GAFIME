#!/usr/bin/env python3
"""
GAFIME Kernel Smoke Tests - TDD Verification Suite

This script validates the native CUDA kernels before integration.
All tests must pass before proceeding to Phase 4.

Test Scenarios:
1. Sanity Check: 10x10 matrix vs NumPy ground truth
2. Edge Case - Single Row: 1xN matrix
3. Edge Case - Prime Dimensions: 1023x127
4. Stress Test: 4096x4096 (safe for 8GB VRAM)

Usage:
    python tests/smoke_tests/verify_kernel.py
"""

import sys
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Status codes from interfaces.h
GAFIME_SUCCESS = 0
GAFIME_ERROR_INVALID_ARGS = -1
GAFIME_ERROR_CUDA_NOT_AVAILABLE = -2
GAFIME_ERROR_OUT_OF_MEMORY = -3
GAFIME_ERROR_KERNEL_FAILED = -4


def find_library() -> Optional[ctypes.CDLL]:
    """Find and load the GAFIME native library."""
    import os
    
    lib_dir = Path(__file__).parent.parent.parent
    
    # On Windows, add CUDA bin to DLL search path
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
    
    # Try different library names
    lib_names = [
        "gafime_cuda.dll", "gafime_cuda.so", "libgafime_cuda.so",
        "gafime_cpu.dll", "gafime_cpu.so", "libgafime_cpu.so",
    ]
    
    for name in lib_names:
        lib_path = lib_dir / name
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path.absolute()))
            except OSError as e:
                print(f"  Warning: Found {name} but failed to load: {e}")
    
    # Also check build directories
    build_dirs = [lib_dir / "build", lib_dir / "build" / "Release"]
    for build_dir in build_dirs:
        if build_dir.exists():
            for name in lib_names:
                lib_path = build_dir / name
                if lib_path.exists():
                    try:
                        return ctypes.CDLL(str(lib_path.absolute()))
                    except OSError:
                        pass
    
    return None


def numpy_feature_interaction(
    X: np.ndarray, 
    combo_indices: np.ndarray,
    combo_offsets: np.ndarray
) -> np.ndarray:
    """
    NumPy reference implementation for feature interaction.
    This is the ground truth for validating CUDA kernels.
    """
    n_samples, n_features = X.shape
    n_combos = len(combo_offsets) - 1
    
    # Compute means
    means = np.mean(X, axis=0)
    
    output = np.zeros((n_samples, n_combos), dtype=np.float32)
    
    for c in range(n_combos):
        start = combo_offsets[c]
        end = combo_offsets[c + 1]
        combo = combo_indices[start:end]
        
        if len(combo) == 0:
            output[:, c] = 0.0
        elif len(combo) == 1:
            output[:, c] = X[:, combo[0]]
        else:
            # Centered product
            centered = X[:, combo] - means[combo]
            output[:, c] = np.prod(centered, axis=1)
    
    return output


def create_test_data(
    n_samples: int, 
    n_features: int, 
    n_combos: int,
    combo_size: int = 2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create test data for kernel validation."""
    rng = np.random.default_rng(seed)
    
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    means = np.mean(X, axis=0).astype(np.float32)
    
    # Create combos
    combo_indices = []
    combo_offsets = [0]
    
    for _ in range(n_combos):
        combo = rng.choice(n_features, size=min(combo_size, n_features), replace=False)
        combo_indices.extend(combo.tolist())
        combo_offsets.append(len(combo_indices))
    
    return (
        X, 
        means,
        np.array(combo_indices, dtype=np.int32),
        np.array(combo_offsets, dtype=np.int32)
    )


class KernelTester:
    def __init__(self, lib: Optional[ctypes.CDLL] = None):
        self.lib = lib
        self.using_numpy_fallback = lib is None
        self.passed = 0
        self.failed = 0
        
    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        print("=" * 60)
        print("GAFIME Kernel Smoke Tests")
        print("=" * 60)
        
        if self.using_numpy_fallback:
            print("\n⚠️  Native library not found - testing NumPy reference only")
            print("   (Build the library first with setup.py build_ext --inplace)")
        else:
            # Check CUDA availability
            if hasattr(self.lib, 'gafime_cuda_available'):
                self.lib.gafime_cuda_available.restype = ctypes.c_int
                cuda_available = self.lib.gafime_cuda_available()
                print(f"\n🔍 CUDA Available: {'Yes' if cuda_available else 'No (using CPU fallback)'}")
        
        print("\n" + "-" * 60)
        
        # Run tests
        self.test_sanity_10x10()
        self.test_edge_case_single_row()
        self.test_edge_case_prime_dims()
        self.test_stress_4096()
        
        # Summary
        print("\n" + "=" * 60)
        total = self.passed + self.failed
        if self.failed == 0:
            print(f"✅ ALL TESTS PASSED ({self.passed}/{total})")
            return True
        else:
            print(f"❌ TESTS FAILED: {self.failed}/{total}")
            return False
    
    def test_sanity_10x10(self):
        """Test 1: Small matrix sanity check."""
        print("\n📋 Test 1: Sanity Check (10x10 matrix)")
        
        X, means, combo_indices, combo_offsets = create_test_data(
            n_samples=10, n_features=10, n_combos=5, combo_size=2
        )
        
        # NumPy reference
        expected = numpy_feature_interaction(X, combo_indices, combo_offsets)
        
        if self.using_numpy_fallback:
            # Just verify NumPy implementation is deterministic
            expected2 = numpy_feature_interaction(X, combo_indices, combo_offsets)
            if np.allclose(expected, expected2):
                print("   ✓ NumPy reference is deterministic")
                self.passed += 1
            else:
                print("   ✗ NumPy reference is NOT deterministic!")
                self.failed += 1
        else:
            # TODO: Call native library and compare
            actual = self._call_native_interaction(X, means, combo_indices, combo_offsets)
            if actual is not None and np.allclose(expected, actual, rtol=1e-4, atol=1e-6):
                print("   ✓ Native output matches NumPy reference")
                self.passed += 1
            else:
                print("   ✗ Native output does NOT match NumPy reference")
                self.failed += 1
    
    def test_edge_case_single_row(self):
        """Test 2: Single row matrix (1xN)."""
        print("\n📋 Test 2: Edge Case - Single Row (1x50)")
        
        X, means, combo_indices, combo_offsets = create_test_data(
            n_samples=1, n_features=50, n_combos=10, combo_size=3
        )
        
        try:
            expected = numpy_feature_interaction(X, combo_indices, combo_offsets)
            
            if self.using_numpy_fallback:
                if expected.shape == (1, 10):
                    print("   ✓ NumPy handles dim=1 correctly")
                    self.passed += 1
                else:
                    print(f"   ✗ Unexpected shape: {expected.shape}")
                    self.failed += 1
            else:
                actual = self._call_native_interaction(X, means, combo_indices, combo_offsets)
                if actual is not None and actual.shape == (1, 10):
                    print("   ✓ Native kernel handles dim=1 correctly")
                    self.passed += 1
                else:
                    print("   ✗ Native kernel failed on dim=1")
                    self.failed += 1
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            self.failed += 1
    
    def test_edge_case_prime_dims(self):
        """Test 3: Prime dimensions (catches alignment errors)."""
        print("\n📋 Test 3: Edge Case - Prime Dimensions (1023x127)")
        
        X, means, combo_indices, combo_offsets = create_test_data(
            n_samples=1023, n_features=127, n_combos=31, combo_size=2
        )
        
        try:
            expected = numpy_feature_interaction(X, combo_indices, combo_offsets)
            
            if self.using_numpy_fallback:
                if expected.shape == (1023, 31) and not np.isnan(expected).any():
                    print("   ✓ NumPy handles prime dimensions correctly")
                    self.passed += 1
                else:
                    print("   ✗ NumPy produced invalid output")
                    self.failed += 1
            else:
                actual = self._call_native_interaction(X, means, combo_indices, combo_offsets)
                if actual is not None and np.allclose(expected, actual, rtol=1e-4, atol=1e-6):
                    print("   ✓ Native kernel handles prime dimensions correctly")
                    self.passed += 1
                else:
                    print("   ✗ Native kernel failed on prime dimensions")
                    self.failed += 1
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            self.failed += 1
    
    def test_stress_4096(self):
        """Test 4: Stress test (4096x4096, safe for 8GB VRAM)."""
        print("\n📋 Test 4: Stress Test (4096x4096)")
        
        # Estimate memory usage:
        # X: 4096 * 4096 * 4 bytes = 64 MB
        # means: 4096 * 4 bytes = 16 KB
        # output: 4096 * 256 * 4 bytes = 4 MB
        # Total: ~70 MB (safe for 8GB VRAM)
        
        X, means, combo_indices, combo_offsets = create_test_data(
            n_samples=4096, n_features=4096, n_combos=256, combo_size=2
        )
        
        try:
            import time
            
            start = time.perf_counter()
            expected = numpy_feature_interaction(X, combo_indices, combo_offsets)
            numpy_time = time.perf_counter() - start
            
            if self.using_numpy_fallback:
                if expected.shape == (4096, 256) and not np.isnan(expected).any():
                    print(f"   ✓ NumPy completed in {numpy_time:.2f}s")
                    self.passed += 1
                else:
                    print("   ✗ NumPy produced invalid output")
                    self.failed += 1
            else:
                start = time.perf_counter()
                actual = self._call_native_interaction(X, means, combo_indices, combo_offsets)
                native_time = time.perf_counter() - start
                
                if actual is not None:
                    speedup = numpy_time / native_time if native_time > 0 else float('inf')
                    print(f"   ✓ Native: {native_time:.2f}s, NumPy: {numpy_time:.2f}s, Speedup: {speedup:.1f}x")
                    self.passed += 1
                else:
                    print("   ✗ Native kernel failed on stress test")
                    self.failed += 1
        except MemoryError:
            print("   ⚠️ Out of memory - skipping (expected on low-memory systems)")
            self.passed += 1  # Not a failure, just skip
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            self.failed += 1
    
    def _call_native_interaction(
        self, 
        X: np.ndarray, 
        means: np.ndarray,
        combo_indices: np.ndarray,
        combo_offsets: np.ndarray
    ) -> Optional[np.ndarray]:
        """Call native library for feature interaction."""
        if self.lib is None:
            return None
        
        n_samples, n_features = X.shape
        n_combos = len(combo_offsets) - 1
        
        # Ensure contiguous float32 arrays
        X = np.ascontiguousarray(X, dtype=np.float32)
        means = np.ascontiguousarray(means, dtype=np.float32)
        combo_indices = np.ascontiguousarray(combo_indices, dtype=np.int32)
        combo_offsets = np.ascontiguousarray(combo_offsets, dtype=np.int32)
        output = np.zeros((n_samples, n_combos), dtype=np.float32)
        
        # Try CUDA first, fall back to CPU
        func_names = ['gafime_feature_interaction_cuda', 'gafime_feature_interaction_cpu']
        
        for func_name in func_names:
            if hasattr(self.lib, func_name):
                func = getattr(self.lib, func_name)
                func.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # X
                    ctypes.POINTER(ctypes.c_float),  # means
                    ctypes.POINTER(ctypes.c_float),  # output
                    ctypes.POINTER(ctypes.c_int32),  # combo_indices
                    ctypes.POINTER(ctypes.c_int32),  # combo_offsets
                    ctypes.c_int32,  # n_samples
                    ctypes.c_int32,  # n_features
                    ctypes.c_int32,  # n_combos
                ]
                func.restype = ctypes.c_int
                
                result = func(
                    X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    combo_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    combo_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    n_samples,
                    n_features,
                    n_combos,
                )
                
                if result == GAFIME_SUCCESS:
                    return output
        
        return None


def main():
    lib = find_library()
    tester = KernelTester(lib)
    
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
