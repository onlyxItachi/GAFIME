"""
GAFIME V4 BENCHMARK - All Optimizations
"""
import time
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
from gafime.backends.fused_kernel import StaticBucket, create_fold_mask

print('='*60)
print('GAFIME V4 BENCHMARK - All Optimizations')
print('='*60)

N = 100000
f = [np.random.randn(N).astype(np.float32) for _ in range(5)]
t = np.random.randn(N).astype(np.float32)
m = create_fold_mask(N)
b = StaticBucket(N, 5)
b.upload_all(f, t, m)

# Warmup
for _ in range(100): b.compute([0,1],[0,0],[0],0)

print(f'Dataset: {N:,} samples, 5 features')
print()

results = []

# Single compute
start = time.perf_counter()
for i in range(1000): b.compute([i%5,(i+1)%5],[1,3],[0],0)
t1 = time.perf_counter() - start
ips1 = 1000/t1
us1 = t1/1000*1e6
print(f'[1] SINGLE COMPUTE:')
print(f'    {ips1:,.0f} iter/s, {us1:.2f} us/iter')
results.append(('Single', ips1, us1, 1.0))

# Batch sizes
for batch_size in [50, 100, 200, 500]:
    fp = [(i%5,(i+1)%5) for i in range(batch_size)]
    op, ia = [(1,3)]*batch_size, [0]*batch_size
    
    n_calls = max(1, 1000 // batch_size)
    total = n_calls * batch_size
    
    # Warmup
    b.compute_batch(fp, op, ia, 0)
    
    start = time.perf_counter()
    for _ in range(n_calls): 
        b.compute_batch(fp, op, ia, 0)
    elapsed = time.perf_counter() - start
    
    ips = total / elapsed
    us = elapsed / total * 1e6
    speedup = ips / ips1
    
    print(f'[{len(results)+1}] BATCH {batch_size}:')
    print(f'    {ips:,.0f} iter/s, {us:.2f} us/iter, {speedup:.1f}x speedup')
    results.append((f'Batch {batch_size}', ips, us, speedup))

print()
print('='*60)
print('SUMMARY TABLE:')
print('='*60)
print(f'{"Mode":<12} | {"Iter/s":>12} | {"us/iter":>10} | {"Speedup":>8}')
print('-'*50)
for name, ips, us, speedup in results:
    print(f'{name:<12} | {ips:>12,.0f} | {us:>10.2f} | {speedup:>7.1f}x')
print('='*60)

del b
