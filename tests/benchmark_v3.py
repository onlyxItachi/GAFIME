"""
GAFIME Performance Benchmark V3 - Batched Compute API
"""
import time
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
from gafime.backends.fused_kernel import StaticBucket, UnaryOp, InteractionType, compute_pearson_from_stats, create_fold_mask

print('=' * 70)
print('GAFIME PERFORMANCE BENCHMARK V3')
print('=' * 70)

N_SAMPLES = 100000
BATCH_SIZES = [10, 50, 100, 200, 500]
N_ITERATIONS = 1000

features = [np.random.randn(N_SAMPLES).astype(np.float32) for _ in range(5)]
target = np.random.randn(N_SAMPLES).astype(np.float32)
mask = create_fold_mask(N_SAMPLES, n_folds=5)

bucket = StaticBucket(N_SAMPLES, 5)
bucket.upload_all(features, target, mask)

# Warmup
for _ in range(100):
    bucket.compute([0,1], [0,0], [0], 0)

print(f'\nDataset: {N_SAMPLES:,} samples, 5 features')
print()

# 1. Single compute baseline
print('[1] SINGLE COMPUTE (Pre-allocated buffers):')
start = time.perf_counter()
for i in range(N_ITERATIONS):
    stats = bucket.compute([i%5,(i+1)%5], [1,3], [0], 0)
elapsed = time.perf_counter() - start
single_ips = N_ITERATIONS / elapsed
print(f'    {N_ITERATIONS:,} iterations in {elapsed:.3f}s')
print(f'    {single_ips:,.0f} iter/s')
print(f'    {elapsed/N_ITERATIONS*1e6:.2f} us/iter')

# 2. Batched compute at various sizes
print()
print('[2] BATCHED COMPUTE:')
print(f'    {"Batch":>6} | {"Total":>8} | {"Time":>8} | {"Iter/s":>10} | {"us/iter":>8} | {"Speedup":>8}')
print('    ' + '-' * 65)

for batch_size in BATCH_SIZES:
    n_batches = N_ITERATIONS // batch_size
    feature_pairs = [(i%5,(i+1)%5) for i in range(batch_size)]
    op_pairs = [(1,3) for _ in range(batch_size)]
    interactions = [0] * batch_size
    
    # Warmup
    bucket.compute_batch(feature_pairs, op_pairs, interactions, 0)
    
    start = time.perf_counter()
    for _ in range(n_batches):
        stats = bucket.compute_batch(feature_pairs, op_pairs, interactions, 0)
    elapsed = time.perf_counter() - start
    
    total = n_batches * batch_size
    ips = total / elapsed
    us_per = elapsed / total * 1e6
    speedup = ips / single_ips
    
    print(f'    {batch_size:>6} | {total:>8} | {elapsed:>7.3f}s | {ips:>10,.0f} | {us_per:>7.2f} | {speedup:>7.1f}x')

print()
print('=' * 70)
del bucket
