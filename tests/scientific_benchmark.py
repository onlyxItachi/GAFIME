#!/usr/bin/env python3
"""
GAFIME Scientific Performance Benchmark Suite

Comprehensive validation of all GAFIME components:
1. Feature creation and interaction mining
2. Signal detection accuracy
3. GPU vs CPU backend performance
4. Short-term vs long-term data analysis
5. CPU fallback verification
6. Reporting system validation

Output: Structured results for PERFORMANCE_REPORT.md
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BenchmarkResults:
    """Collect and format benchmark results."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "tests": {}
        }
    
    def _get_system_info(self) -> Dict:
        """Collect system information."""
        info = {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": sys.platform,
        }
        
        # Try to get GPU info
        try:
            from gafime.backends.fused_kernel import FusedKernelWrapper
            wrapper = FusedKernelWrapper()
            if wrapper.cuda_available():
                import ctypes
                name_buf = ctypes.create_string_buffer(256)
                mem = ctypes.c_int()
                major = ctypes.c_int()
                minor = ctypes.c_int()
                wrapper.lib.gafime_get_device_info(0, name_buf, ctypes.byref(mem), 
                                                    ctypes.byref(major), ctypes.byref(minor))
                info["gpu"] = {
                    "name": name_buf.value.decode('utf-8'),
                    "memory_mb": mem.value,
                    "compute_capability": f"{major.value}.{minor.value}"
                }
        except Exception as e:
            info["gpu"] = {"error": str(e)}
        
        return info
    
    def add_test(self, name: str, result: Dict):
        """Add a test result."""
        self.results["tests"][name] = result
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.results, indent=2)


def test_feature_creation() -> Dict:
    """Test 1: Is feature creation working fine?"""
    print("\n" + "=" * 60)
    print("TEST 1: Feature Creation Validation")
    print("=" * 60)
    
    from gafime.backends.fused_kernel import (
        StaticBucket, UnaryOp, InteractionType,
        compute_pearson_from_stats, create_fold_mask, numpy_reference
    )
    
    n_samples = 5000
    n_features = 4
    
    np.random.seed(42)
    features = [np.random.randn(n_samples).astype(np.float32) for _ in range(n_features)]
    target = np.random.randn(n_samples).astype(np.float32)
    mask = create_fold_mask(n_samples, n_folds=5)
    
    # Test all unary operators
    ops_to_test = [
        ("IDENTITY", UnaryOp.IDENTITY),
        ("LOG", UnaryOp.LOG),
        ("EXP", UnaryOp.EXP),
        ("SQRT", UnaryOp.SQRT),
        ("TANH", UnaryOp.TANH),
        ("SIGMOID", UnaryOp.SIGMOID),
        ("SQUARE", UnaryOp.SQUARE),
    ]
    
    # Test all interaction types
    interactions_to_test = [
        ("MULT", InteractionType.MULT),
        ("ADD", InteractionType.ADD),
        ("SUB", InteractionType.SUB),
        ("DIV", InteractionType.DIV),
    ]
    
    bucket = StaticBucket(n_samples, n_features)
    bucket.upload_all(features, target, mask)
    
    results = {"operators": {}, "interactions": {}, "passed": 0, "failed": 0}
    
    # Test unary operators
    for op_name, op_id in ops_to_test:
        try:
            cuda_stats = bucket.compute([0, 1], [op_id, UnaryOp.IDENTITY], InteractionType.MULT, 0)
            np_stats = numpy_reference([features[0], features[1]], target, mask, 
                                        [op_id, UnaryOp.IDENTITY], InteractionType.MULT, 0)
            match = np.allclose(cuda_stats, np_stats, rtol=1e-2, atol=1e-1)
            results["operators"][op_name] = {"passed": match, "cuda_r": float(compute_pearson_from_stats(cuda_stats)[0])}
            results["passed" if match else "failed"] += 1
            print(f"  {op_name}: {'PASS' if match else 'FAIL'}")
        except Exception as e:
            results["operators"][op_name] = {"passed": False, "error": str(e)}
            results["failed"] += 1
            print(f"  {op_name}: FAIL ({e})")
    
    # Test interaction types
    for int_name, int_id in interactions_to_test:
        try:
            cuda_stats = bucket.compute([0, 1], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], int_id, 0)
            np_stats = numpy_reference([features[0], features[1]], target, mask,
                                        [UnaryOp.IDENTITY, UnaryOp.IDENTITY], int_id, 0)
            match = np.allclose(cuda_stats, np_stats, rtol=1e-2, atol=1e-1)
            results["interactions"][int_name] = {"passed": match}
            results["passed" if match else "failed"] += 1
            print(f"  {int_name}: {'PASS' if match else 'FAIL'}")
        except Exception as e:
            results["interactions"][int_name] = {"passed": False, "error": str(e)}
            results["failed"] += 1
    
    del bucket
    
    results["overall"] = "PASS" if results["failed"] == 0 else "FAIL"
    print(f"\nFeature Creation: {results['overall']} ({results['passed']}/{results['passed']+results['failed']})")
    
    return results


def test_signal_detection() -> Dict:
    """Test 2: Is created features carrying a signal to the target?"""
    print("\n" + "=" * 60)
    print("TEST 2: Signal Detection Accuracy")
    print("=" * 60)
    
    from gafime.backends.fused_kernel import (
        StaticBucket, UnaryOp, InteractionType,
        compute_pearson_from_stats, create_fold_mask
    )
    
    n_samples = 10000
    np.random.seed(42)
    
    # Create features with KNOWN signal relationships
    f0 = np.random.randn(n_samples).astype(np.float32)
    f1 = np.random.randn(n_samples).astype(np.float32)
    f_noise = np.random.randn(n_samples).astype(np.float32)
    
    # Target = f0 * f1 + noise (KNOWN signal in f0*f1)
    target = (f0 * f1 + np.random.randn(n_samples) * 0.1).astype(np.float32)
    mask = create_fold_mask(n_samples, n_folds=5)
    
    bucket = StaticBucket(n_samples, 3)
    bucket.upload_all([f0, f1, f_noise], target, mask)
    
    results = {"signal_tests": []}
    
    # Test 1: f0 * f1 should have HIGH correlation (signal exists)
    stats_signal = bucket.compute([0, 1], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], InteractionType.MULT, 0)
    r_signal, _ = compute_pearson_from_stats(stats_signal)
    
    # Test 2: f0 * f_noise should have LOW correlation (no signal)
    stats_noise = bucket.compute([0, 2], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], InteractionType.MULT, 0)
    r_noise, _ = compute_pearson_from_stats(stats_noise)
    
    # Test 3: f1 * f_noise should have LOW correlation (no signal)
    stats_noise2 = bucket.compute([1, 2], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], InteractionType.MULT, 0)
    r_noise2, _ = compute_pearson_from_stats(stats_noise2)
    
    results["signal_tests"].append({
        "combo": "f0 * f1 (signal)",
        "pearson_r": float(r_signal),
        "expected": "high (>0.9)",
        "detected": abs(r_signal) > 0.9
    })
    results["signal_tests"].append({
        "combo": "f0 * f_noise (no signal)",
        "pearson_r": float(r_noise),
        "expected": "low (<0.1)",
        "detected": abs(r_noise) < 0.1
    })
    results["signal_tests"].append({
        "combo": "f1 * f_noise (no signal)",
        "pearson_r": float(r_noise2),
        "expected": "low (<0.1)",
        "detected": abs(r_noise2) < 0.1
    })
    
    del bucket
    
    passed = all(t["detected"] for t in results["signal_tests"])
    results["overall"] = "PASS" if passed else "FAIL"
    
    print(f"  f0 * f1 (signal):     r = {r_signal:.4f} (expected >0.9)")
    print(f"  f0 * noise (no sig):  r = {r_noise:.4f} (expected <0.1)")
    print(f"  f1 * noise (no sig):  r = {r_noise2:.4f} (expected <0.1)")
    print(f"\nSignal Detection: {results['overall']}")
    
    return results


def test_gpu_performance() -> Dict:
    """Test 3: GPU backend performance (short vs long term data)."""
    print("\n" + "=" * 60)
    print("TEST 3: GPU Backend Performance")
    print("=" * 60)
    
    from gafime.backends.fused_kernel import (
        StaticBucket, FusedKernelWrapper, UnaryOp, InteractionType,
        create_fold_mask
    )
    
    results = {"scenarios": []}
    
    # Scenario configs: (name, n_samples, n_iterations)
    scenarios = [
        ("Short-term (1K samples, 100 iters)", 1000, 100),
        ("Medium-term (10K samples, 1000 iters)", 10000, 1000),
        ("Long-term (100K samples, 100 iters)", 100000, 100),
        ("Stress test (10K samples, 10K iters)", 10000, 10000),
    ]
    
    for name, n_samples, n_iters in scenarios:
        print(f"\n  {name}...")
        np.random.seed(42)
        features = [np.random.randn(n_samples).astype(np.float32) for _ in range(4)]
        target = np.random.randn(n_samples).astype(np.float32)
        mask = create_fold_mask(n_samples, n_folds=5)
        
        try:
            bucket = StaticBucket(n_samples, 4)
            
            # Measure upload time
            t0 = time.perf_counter()
            bucket.upload_all(features, target, mask)
            upload_time = time.perf_counter() - t0
            
            # Measure compute time
            t0 = time.perf_counter()
            for i in range(n_iters):
                _ = bucket.compute([0, 1], [UnaryOp.LOG, UnaryOp.SQRT], InteractionType.MULT, i % 5)
            compute_time = time.perf_counter() - t0
            
            del bucket
            
            scenario_result = {
                "name": name,
                "n_samples": n_samples,
                "n_iterations": n_iters,
                "upload_time_ms": upload_time * 1000,
                "compute_time_ms": compute_time * 1000,
                "per_iter_ms": (compute_time / n_iters) * 1000,
                "iters_per_sec": n_iters / compute_time,
                "status": "PASS"
            }
            
            print(f"    Upload: {upload_time*1000:.2f}ms")
            print(f"    Compute: {compute_time*1000:.1f}ms ({n_iters/compute_time:.0f} iter/s)")
            
        except Exception as e:
            scenario_result = {"name": name, "status": "FAIL", "error": str(e)}
            print(f"    ERROR: {e}")
        
        results["scenarios"].append(scenario_result)
    
    results["overall"] = "PASS" if all(s.get("status") == "PASS" for s in results["scenarios"]) else "FAIL"
    print(f"\nGPU Performance: {results['overall']}")
    
    return results


def test_cpu_fallback() -> Dict:
    """Test 4: CPU fallback verification."""
    print("\n" + "=" * 60)
    print("TEST 4: CPU Fallback Verification")
    print("=" * 60)
    
    from gafime.backends.fused_kernel import (
        FusedKernelWrapper, UnaryOp, InteractionType,
        compute_pearson_from_stats, create_fold_mask, numpy_reference
    )
    
    n_samples = 5000
    np.random.seed(42)
    features = [np.random.randn(n_samples).astype(np.float32) for _ in range(2)]
    target = np.random.randn(n_samples).astype(np.float32)
    mask = create_fold_mask(n_samples, n_folds=5)
    
    results = {}
    
    # Test NumPy reference implementation (pure CPU)
    print("  Testing NumPy reference (CPU)...")
    try:
        t0 = time.perf_counter()
        np_stats = numpy_reference(features, target, mask, 
                                    [UnaryOp.IDENTITY, UnaryOp.IDENTITY],
                                    InteractionType.MULT, 0)
        cpu_time = time.perf_counter() - t0
        
        r_cpu, _ = compute_pearson_from_stats(np_stats)
        results["numpy_reference"] = {
            "status": "PASS",
            "time_ms": cpu_time * 1000,
            "pearson_r": float(r_cpu)
        }
        print(f"    PASS: {cpu_time*1000:.2f}ms, r={r_cpu:.4f}")
    except Exception as e:
        results["numpy_reference"] = {"status": "FAIL", "error": str(e)}
        print(f"    FAIL: {e}")
    
    # Test GPU with same data
    print("  Testing GPU kernel...")
    try:
        wrapper = FusedKernelWrapper()
        t0 = time.perf_counter()
        gpu_stats = wrapper.compute(features, target, mask,
                                     [UnaryOp.IDENTITY, UnaryOp.IDENTITY],
                                     InteractionType.MULT, 0)
        gpu_time = time.perf_counter() - t0
        
        r_gpu, _ = compute_pearson_from_stats(gpu_stats)
        results["gpu_kernel"] = {
            "status": "PASS",
            "time_ms": gpu_time * 1000,
            "pearson_r": float(r_gpu)
        }
        print(f"    PASS: {gpu_time*1000:.2f}ms, r={r_gpu:.4f}")
        
        # Check consistency
        match = np.allclose(np_stats, gpu_stats, rtol=1e-2, atol=1e-1)
        results["consistency"] = {
            "gpu_vs_cpu_match": match,
            "max_diff": float(np.max(np.abs(np_stats - gpu_stats)))
        }
        print(f"  GPU/CPU consistency: {'PASS' if match else 'FAIL'}")
        
    except Exception as e:
        results["gpu_kernel"] = {"status": "FAIL", "error": str(e)}
        print(f"    FAIL: {e}")
    
    results["overall"] = "PASS" if (
        results.get("numpy_reference", {}).get("status") == "PASS" and
        results.get("gpu_kernel", {}).get("status") == "PASS" and
        results.get("consistency", {}).get("gpu_vs_cpu_match", False)
    ) else "FAIL"
    
    print(f"\nCPU Fallback: {results['overall']}")
    return results


def test_reporting_system() -> Dict:
    """Test 5: Reporting system validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Reporting System Validation")
    print("=" * 60)
    
    from gafime.backends.fused_kernel import (
        StaticBucket, UnaryOp, InteractionType,
        compute_pearson_from_stats, unpack_stats, create_fold_mask
    )
    
    n_samples = 5000
    np.random.seed(42)
    
    # Create data with known correlation
    f0 = np.random.randn(n_samples).astype(np.float32)
    f1 = np.random.randn(n_samples).astype(np.float32)
    target = (f0 * f1 + np.random.randn(n_samples) * 0.05).astype(np.float32)
    mask = create_fold_mask(n_samples, n_folds=5)
    
    bucket = StaticBucket(n_samples, 2)
    bucket.upload_all([f0, f1], target, mask)
    
    results = {"cross_validation_folds": []}
    
    # Test all 5 folds
    print("  Testing cross-validation folds...")
    for fold in range(5):
        stats = bucket.compute([0, 1], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], 
                               InteractionType.MULT, fold)
        train_r, val_r = compute_pearson_from_stats(stats)
        train_stats, val_stats = unpack_stats(stats)
        
        fold_result = {
            "fold": fold,
            "train_n": int(train_stats["n"]),
            "val_n": int(val_stats["n"]),
            "train_r": float(train_r),
            "val_r": float(val_r),
            "train_val_diff": abs(float(train_r) - float(val_r))
        }
        results["cross_validation_folds"].append(fold_result)
        print(f"    Fold {fold}: train_r={train_r:.4f}, val_r={val_r:.4f}, "
              f"train_n={int(train_stats['n'])}, val_n={int(val_stats['n'])}")
    
    del bucket
    
    # Validate: all folds should have similar performance (no overfitting)
    train_rs = [f["train_r"] for f in results["cross_validation_folds"]]
    val_rs = [f["val_r"] for f in results["cross_validation_folds"]]
    
    results["statistics"] = {
        "mean_train_r": float(np.mean(train_rs)),
        "std_train_r": float(np.std(train_rs)),
        "mean_val_r": float(np.mean(val_rs)),
        "std_val_r": float(np.std(val_rs)),
        "train_val_gap": abs(float(np.mean(train_rs)) - float(np.mean(val_rs)))
    }
    
    # Pass if std is low (consistent across folds) and gap is small (no overfitting)
    passed = (results["statistics"]["std_val_r"] < 0.05 and 
              results["statistics"]["train_val_gap"] < 0.1)
    results["overall"] = "PASS" if passed else "FAIL"
    
    print(f"\n  Mean train_r: {results['statistics']['mean_train_r']:.4f} ± {results['statistics']['std_train_r']:.4f}")
    print(f"  Mean val_r:   {results['statistics']['mean_val_r']:.4f} ± {results['statistics']['std_val_r']:.4f}")
    print(f"\nReporting System: {results['overall']}")
    
    return results


def run_all_benchmarks() -> BenchmarkResults:
    """Run all benchmarks and collect results."""
    print("\n" + "=" * 60)
    print("GAFIME SCIENTIFIC PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = BenchmarkResults()
    
    # Run all tests
    results.add_test("1_feature_creation", test_feature_creation())
    results.add_test("2_signal_detection", test_signal_detection())
    results.add_test("3_gpu_performance", test_gpu_performance())
    results.add_test("4_cpu_fallback", test_cpu_fallback())
    results.add_test("5_reporting_system", test_reporting_system())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, test in results.results["tests"].items():
        status = test.get("overall", "UNKNOWN")
        print(f"  {name}: {status}")
        if status != "PASS":
            all_passed = False
    
    results.results["overall"] = "PASS" if all_passed else "FAIL"
    print(f"\nOVERALL: {results.results['overall']}")
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
    
    # Save JSON results
    json_path = PROJECT_ROOT / "tests" / "benchmark_results.json"
    with open(json_path, "w") as f:
        f.write(results.to_json())
    print(f"\nResults saved to: {json_path}")
