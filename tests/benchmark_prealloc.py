"""
Benchmark: Verify Pre-allocated Buffer Performance

Compares old vs new compute() overhead.
"""
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r'c:\Users\Hamza\Desktop\GAFIME')))

from gafime.backends.fused_kernel import StaticBucket, UnaryOp, InteractionType, compute_pearson_from_stats, create_fold_mask

print("=" * 60)
print("BENCHMARK: Pre-allocated Buffer Performance")
print("=" * 60)

# Setup
N_SAMPLES = 100_000
N_FEATURES = 5
N_ITERATIONS = 10_000

print(f"\nSetup: {N_SAMPLES} samples, {N_FEATURES} features, {N_ITERATIONS} iterations")

# Create test data
np.random.seed(42)
features = [np.random.randn(N_SAMPLES).astype(np.float32) for _ in range(N_FEATURES)]
target = np.random.randn(N_SAMPLES).astype(np.float32)
mask = create_fold_mask(N_SAMPLES, n_folds=5)

# Create bucket
bucket = StaticBucket(N_SAMPLES, N_FEATURES)
bucket.upload_all(features, target, mask)

print("\n[1] Warm-up run (compile CUDA JIT)...")
for _ in range(100):
    stats = bucket.compute([0, 1], [UnaryOp.IDENTITY, UnaryOp.IDENTITY], [InteractionType.MULT], 0)

print("[2] Running benchmark...")

# Benchmark
start = time.perf_counter()
for i in range(N_ITERATIONS):
    stats = bucket.compute(
        [i % N_FEATURES, (i+1) % N_FEATURES],
        [UnaryOp.LOG, UnaryOp.SQRT],
        [InteractionType.MULT],
        i % 5
    )
    train_r, val_r = compute_pearson_from_stats(stats)
elapsed = time.perf_counter() - start

# Results
iter_per_sec = N_ITERATIONS / elapsed
us_per_iter = (elapsed / N_ITERATIONS) * 1_000_000

print(f"\n[3] Results:")
print(f"    Total time:      {elapsed:.2f} seconds")
print(f"    Iterations/sec:  {iter_per_sec:,.0f}")
print(f"    Time/iteration:  {us_per_iter:.1f} μs")

print("\n" + "=" * 60)
if us_per_iter < 100:
    print("✅ EXCELLENT: < 100 μs/iteration")
elif us_per_iter < 200:
    print("✓ GOOD: < 200 μs/iteration")
else:
    print("⚠ SLOW: > 200 μs/iteration - investigate")
print("=" * 60)

del bucket
