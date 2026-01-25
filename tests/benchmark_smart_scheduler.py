import time
import sys
import ctypes
sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
try:
    import gafime_cpu
except ImportError:
    print("Could not import gafime_cpu")
    sys.exit(1)

def run_benchmark():
    print('='*70)
    print('Smart Scheduler Benchmark (CPU Overhead & Deduplication)')
    print('='*70)

    # Configuration: 100 features, 12 unary ops, 6 interaction types
    N_FEATURES = 100
    N_OPS = 12
    N_INTERACT_TYPES = 6
    BATCH_SIZE = 10000
    
    # Calculation of theoretical iterations
    # Loop structure:
    #   Interact (6) * OpB (12) * OpA (12) * Pairs (N*(N-1)/2)
    pairs = (N_FEATURES * (N_FEATURES - 1)) // 2
    total_theoretical = N_INTERACT_TYPES * N_OPS * N_OPS * pairs
    
    print(f"Configuration:")
    print(f"  Features: {N_FEATURES}")
    print(f"  Operators: {N_OPS}")
    print(f"  Interaction Types: {N_INTERACT_TYPES}")
    print(f"  Theoretical Combinations: {total_theoretical:,}")
    print('-'*70)

    scheduler = gafime_cpu.SmartScheduler(N_FEATURES, N_OPS, N_INTERACT_TYPES)

    start_time = time.perf_counter()
    total_generated = 0
    batches = 0
    
    while True:
        # Generate batch
        # Returns (f_a, f_b, op_a, op_b, int_type)
        f_a, f_b, op_a, op_b, int_t = scheduler.generate_batch(BATCH_SIZE)
        
        count = len(f_a)
        if count == 0:
            break
            
        total_generated += count
        batches += 1

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    unique_count = scheduler.count_seen()
    
    print(f"Results:")
    print(f"  Total Generated: {total_generated:,}")
    print(f"  Theoretical Max: {total_theoretical:,}")
    print(f"  Skipped (Dedup): {total_theoretical - total_generated:,} ({(total_theoretical - total_generated)/total_theoretical:.1%})")
    print(f"  Time Elapsed:    {elapsed:.4f} s")
    print(f"  Throughput:      {total_generated / elapsed / 1_000_000:.2f} M/sec")
    print('='*70)

if __name__ == "__main__":
    run_benchmark()
