#!/usr/bin/env python3
"""
GAFIME Benchmark Suite - Native CUDA vs NumPy Comparison

Compares streaming + CUDA kernel performance against baseline NumPy.

Metrics:
- Data loading time (streamer vs full load)
- Kernel execution time (CUDA vs NumPy)
- End-to-end throughput (rows/second)
"""

import os
import sys
import time
import ctypes
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# Add CUDA DLL directory on Windows
if os.name == 'nt':
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
    ]
    for cuda_bin in cuda_paths:
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)
            break


def generate_test_parquet(
    n_samples: int = 100_000,
    n_features: int = 100,
    output_path: Optional[str] = None,
) -> str:
    """Generate synthetic test data in Parquet format."""
    try:
        import polars as pl
    except ImportError:
        raise ImportError("Polars required: pip install polars")
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".parquet")
    
    print(f"📦 Generating test data: {n_samples:,} x {n_features}")
    
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    
    # Create DataFrame with named columns
    columns = {f"f{i}": data[:, i] for i in range(n_features)}
    columns["target"] = rng.standard_normal(n_samples).astype(np.float32)
    
    df = pl.DataFrame(columns)
    df.write_parquet(output_path)
    
    print(f"   Saved to: {output_path}")
    return output_path


def load_native_library():
    """Load the native CUDA/CPU library."""
    lib_dir = Path(__file__).parent.parent
    
    lib_names = ["gafime_cuda.dll", "gafime_cpu.dll"]
    
    for name in lib_names:
        lib_path = lib_dir / name
        if lib_path.exists():
            try:
                lib = ctypes.CDLL(str(lib_path.absolute()))
                
                # Check if CUDA available
                if hasattr(lib, 'gafime_cuda_available'):
                    lib.gafime_cuda_available.restype = ctypes.c_int
                    if lib.gafime_cuda_available():
                        print(f"✓ Loaded CUDA backend: {name}")
                        return lib, "cuda"
                
                print(f"✓ Loaded CPU backend: {name}")
                return lib, "cpu"
            except OSError as e:
                print(f"  Warning: Failed to load {name}: {e}")
    
    return None, "numpy"


def numpy_feature_interaction(
    X: np.ndarray,
    combo_indices: np.ndarray,
    combo_offsets: np.ndarray,
) -> np.ndarray:
    """NumPy reference implementation for feature interaction."""
    n_samples, n_features = X.shape
    n_combos = len(combo_offsets) - 1
    
    means = np.mean(X, axis=0)
    output = np.zeros((n_samples, n_combos), dtype=np.float32)
    
    for c in range(n_combos):
        start = combo_offsets[c]
        end = combo_offsets[c + 1]
        combo = combo_indices[start:end]
        
        if len(combo) == 1:
            output[:, c] = X[:, combo[0]]
        else:
            centered = X[:, combo] - means[combo]
            output[:, c] = np.prod(centered, axis=1)
    
    return output


def native_feature_interaction(
    lib,
    X: np.ndarray,
    combo_indices: np.ndarray,
    combo_offsets: np.ndarray,
) -> np.ndarray:
    """Call native CUDA/CPU kernel."""
    n_samples, n_features = X.shape
    n_combos = len(combo_offsets) - 1
    
    # Prepare data
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    means = np.mean(X_f32, axis=0).astype(np.float32)
    combo_indices = np.ascontiguousarray(combo_indices, dtype=np.int32)
    combo_offsets = np.ascontiguousarray(combo_offsets, dtype=np.int32)
    output = np.zeros((n_samples, n_combos), dtype=np.float32)
    
    # Setup function
    func_names = ['gafime_feature_interaction_cuda', 'gafime_feature_interaction_cpu']
    
    for func_name in func_names:
        if hasattr(lib, func_name):
            func = getattr(lib, func_name)
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ]
            func.restype = ctypes.c_int
            
            result = func(
                X_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                combo_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                combo_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                n_samples, n_features, n_combos,
            )
            
            if result == 0:
                return output
    
    raise RuntimeError("Native kernel call failed")


def create_combos(n_features: int, n_combos: int, combo_size: int = 2, seed: int = 42):
    """Create test feature combinations."""
    rng = np.random.default_rng(seed)
    
    combo_indices = []
    combo_offsets = [0]
    
    for _ in range(n_combos):
        combo = rng.choice(n_features, size=min(combo_size, n_features), replace=False)
        combo_indices.extend(combo.tolist())
        combo_offsets.append(len(combo_indices))
    
    return np.array(combo_indices, dtype=np.int32), np.array(combo_offsets, dtype=np.int32)


def run_benchmark(
    n_samples: int = 100_000,
    n_features: int = 100,
    n_combos: int = 256,
    n_iterations: int = 5,
    use_streaming: bool = True,
):
    """
    Run comprehensive benchmark comparing Native CUDA vs NumPy.
    
    Measures:
    1. Data loading (streaming vs full load)
    2. Kernel execution (CUDA vs NumPy)
    3. End-to-end throughput
    """
    print("\n" + "=" * 70)
    print("GAFIME Benchmark: Native CUDA vs NumPy")
    print("=" * 70)
    print(f"Configuration: {n_samples:,} samples x {n_features} features x {n_combos} combos")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Load native library
    lib, backend = load_native_library()
    has_native = lib is not None
    
    # Generate test data
    if use_streaming:
        parquet_path = generate_test_parquet(n_samples, n_features)
    
    # Generate NumPy data for direct comparison
    print("\n📊 Generating in-memory test data...")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    combos_idx, combos_off = create_combos(n_features, n_combos)
    
    print(f"   X shape: {X.shape}, dtype: {X.dtype}")
    print(f"   Memory: {X.nbytes / (1024**2):.1f} MB")
    print()
    
    results = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_combos": n_combos,
        "backend": backend,
    }
    
    # =========================================================================
    # BENCHMARK 1: NumPy Baseline
    # =========================================================================
    print("─" * 70)
    print("📈 BENCHMARK 1: NumPy Baseline")
    print("─" * 70)
    
    numpy_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        output_np = numpy_feature_interaction(X, combos_idx, combos_off)
        elapsed = (time.perf_counter() - start) * 1000
        numpy_times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed:.2f}ms")
    
    numpy_avg = sum(numpy_times) / len(numpy_times)
    print(f"\n   NumPy Average: {numpy_avg:.2f}ms")
    results["numpy_avg_ms"] = numpy_avg
    
    # =========================================================================
    # BENCHMARK 2: Native CUDA/CPU
    # =========================================================================
    if has_native:
        print("\n" + "─" * 70)
        print(f"⚡ BENCHMARK 2: Native {backend.upper()} Kernel")
        print("─" * 70)
        
        native_times = []
        for i in range(n_iterations):
            start = time.perf_counter()
            output_native = native_feature_interaction(lib, X, combos_idx, combos_off)
            elapsed = (time.perf_counter() - start) * 1000
            native_times.append(elapsed)
            print(f"   Iteration {i+1}: {elapsed:.2f}ms")
        
        native_avg = sum(native_times) / len(native_times)
        speedup = numpy_avg / native_avg
        print(f"\n   Native Average: {native_avg:.2f}ms")
        print(f"   🚀 Speedup vs NumPy: {speedup:.1f}x")
        
        # Verify correctness
        if np.allclose(output_np, output_native, rtol=1e-4, atol=1e-5):
            print("   ✓ Output matches NumPy reference")
        else:
            print("   ⚠ Output differs from NumPy reference!")
        
        results["native_avg_ms"] = native_avg
        results["speedup"] = speedup
    
    # =========================================================================
    # BENCHMARK 3: Streaming (if Polars available)
    # =========================================================================
    if use_streaming:
        print("\n" + "─" * 70)
        print("🌊 BENCHMARK 3: Polars Streaming + Native Kernel")
        print("─" * 70)
        
        try:
            from gafime.io import GafimeStreamer
            
            streamer = GafimeStreamer(parquet_path)
            batch_size = streamer.estimate_optimal_batch_size()
            
            stream_times = []
            total_rows_processed = 0
            
            for i in range(min(3, n_iterations)):  # Fewer iterations for streaming
                start = time.perf_counter()
                
                for chunk in streamer.stream(batch_size):
                    if has_native:
                        _ = native_feature_interaction(lib, chunk, combos_idx, combos_off)
                    else:
                        _ = numpy_feature_interaction(chunk, combos_idx, combos_off)
                    total_rows_processed += chunk.shape[0]
                
                elapsed = (time.perf_counter() - start) * 1000
                stream_times.append(elapsed)
                print(f"   Iteration {i+1}: {elapsed:.2f}ms (streamed)")
            
            stream_avg = sum(stream_times) / len(stream_times)
            throughput = (n_samples / stream_avg) * 1000  # rows/second
            
            print(f"\n   Streaming Average: {stream_avg:.2f}ms")
            print(f"   📊 Throughput: {throughput:,.0f} rows/sec")
            
            results["streaming_avg_ms"] = stream_avg
            results["throughput_rows_sec"] = throughput
            
            # Cleanup temp file
            os.unlink(parquet_path)
            
        except ImportError as e:
            print(f"   Skipped (Polars not available): {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"   NumPy Baseline:       {results.get('numpy_avg_ms', 'N/A'):.2f}ms")
    if has_native:
        print(f"   Native {backend.upper():5}:         {results.get('native_avg_ms', 'N/A'):.2f}ms")
        print(f"   Speedup:              {results.get('speedup', 'N/A'):.1f}x")
    if "streaming_avg_ms" in results:
        print(f"   Streaming E2E:        {results.get('streaming_avg_ms', 'N/A'):.2f}ms")
        print(f"   Throughput:           {results.get('throughput_rows_sec', 0):,.0f} rows/sec")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Parse command line args
    import argparse
    
    parser = argparse.ArgumentParser(description="GAFIME Benchmark Suite")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of samples")
    parser.add_argument("--features", type=int, default=100, help="Number of features")
    parser.add_argument("--combos", type=int, default=256, help="Number of combinations")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--no-streaming", action="store_true", help="Skip streaming benchmark")
    
    args = parser.parse_args()
    
    run_benchmark(
        n_samples=args.samples,
        n_features=args.features,
        n_combos=args.combos,
        n_iterations=args.iterations,
        use_streaming=not args.no_streaming,
    )
