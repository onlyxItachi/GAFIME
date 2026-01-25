import time
import numpy as np
import sys
import random
sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
try:
    import gafime_cpu
except ImportError:
    print("Could not import gafime_cpu")
    sys.exit(1)

from gafime.backends.fused_kernel import create_fold_mask

def run_benchmark():
    print('='*70)
    print('FULL SYSTEM BENCHMARK: Smart Scheduler (Pinned) vs Random Scheduler')
    print('='*70)

    # Configuration
    N_FEATURES = 100
    N_SAMPLES = 500_000
    N_OPS = 12
    N_INTERACT_TYPES = 6
    BATCH_SIZE = 4000 # Compute batch size

    print(f"Dataset: {N_FEATURES} features x {N_SAMPLES} samples")
    print(f"Total Memory: {N_FEATURES * N_SAMPLES * 4 / 1024 / 1024:.1f} MB")
    print(f"GPU L2 Cache: ~32 MB (RTX 4060)")
    
    # 1. Prepare Data
    print("Preparing GPU Memory...")
    layout = gafime_cpu.ContiguousLayout(N_SAMPLES, N_FEATURES)
    # Use random data
    for _ in range(N_FEATURES):
        layout.add_feature(np.random.randn(N_SAMPLES).astype(np.float32).tolist())
    layout.set_target(np.random.randn(N_SAMPLES).astype(np.float32).tolist())
    layout.set_mask((create_fold_mask(N_SAMPLES) * 1).astype(np.uint8).tolist())
    
    bucket = gafime_cpu.ContiguousBucket(layout)
    print("Upload Complete.")

    # 2. Generate Workload
    print("\nGenerating Workloads...")
    
    # SMART Workload (Pinned)
    smart_sched = gafime_cpu.SmartScheduler(N_FEATURES, N_OPS, N_INTERACT_TYPES)
    # Generate 50 batches of work
    smart_batches = []
    TOTAL_INTERACTIONS = 50_000 # Measure 50k interactions
    
    while len(smart_batches) * BATCH_SIZE < TOTAL_INTERACTIONS:
        batch = smart_sched.generate_batch(BATCH_SIZE)
        if len(batch[0]) == 0: break
        smart_batches.append(batch)
        
    print(f"Smart Workload: {len(smart_batches)} batches ({len(smart_batches)*BATCH_SIZE} interactions)")
    
    # RANDOM Workload
    # Randomize the order of interactions in the smart batch
    # Flatten everything
    flat_fa, flat_fb, flat_oa, flat_ob, flat_int = [], [], [], [], []
    for b in smart_batches:
        flat_fa.extend(b[0])
        flat_fb.extend(b[1])
        flat_oa.extend(b[2])
        flat_ob.extend(b[3])
        flat_int.extend(b[4])
        
    # Zip, Shuffle, Unzip
    combined = list(zip(flat_fa, flat_fb, flat_oa, flat_ob, flat_int))
    random.shuffle(combined)
    
    # Re-batch
    random_batches = []
    chunk_size = BATCH_SIZE
    for i in range(0, len(combined), chunk_size):
        chunk = combined[i:i+chunk_size]
        # Unzip
        # Transpose: [(a,b,c), (d,e,f)] -> [(a,d), (b,e), (c,f)]
        unzipped = list(zip(*chunk))
        # Convert tuples to lists
        random_batches.append(tuple(list(x) for x in unzipped))
        
    print(f"Random Workload: {len(random_batches)} batches (Shuffled)")

    # 3. Benchmark SMART
    print("\n[1] Running SMART Scheduler (Pinned Execution)...")
    # Warmup
    bucket.compute(0, 1, 0, 0, 0, 0)
    
    start = time.perf_counter()
    for batch in smart_batches:
        bucket.compute_batch(batch[0], batch[1], batch[2], batch[3], batch[4], 0)
        
    smart_time = time.perf_counter() - start
    print(f"   Time: {smart_time:.4f} s")
    # Throughput in SAMPLES/sec per interaction = N_SAMPLES
    # Total Samples Processed = Interactions * N_SAMPLES
    total_samples = len(smart_batches) * BATCH_SIZE * N_SAMPLES
    print(f"   Throughput: {len(smart_batches)*BATCH_SIZE / smart_time / 1000000:.2f} M interactions/sec")
    print(f"   Effective Sample Throughput: {total_samples / smart_time / 1e9:.2f} G samples/sec")

    # 4. Benchmark RANDOM
    print("\n[2] Running RANDOM Scheduler (Scattered Execution)...")
    # Warmup
    bucket.compute(0, 1, 0, 0, 0, 0)
    
    start = time.perf_counter()
    for batch in random_batches:
        bucket.compute_batch(batch[0], batch[1], batch[2], batch[3], batch[4], 0)
        
    random_time = time.perf_counter() - start
    print(f"   Time: {random_time:.4f} s")
    print(f"   Throughput: {len(random_batches)*BATCH_SIZE / random_time / 1000000:.2f} M interactions/sec")
    print(f"   Effective Sample Throughput: {total_samples / random_time / 1e9:.2f} G samples/sec")

    print('\n' + '='*70)
    speedup = random_time / smart_time
    print(f"L2 CACHE SPEEDUP: {speedup:.2f}x")
    print('='*70)

if __name__ == "__main__":
    run_benchmark()
