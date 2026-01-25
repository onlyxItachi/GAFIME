import time
import numpy as np
import sys
import math

sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
try:
    import gafime_cpu
except ImportError:
    print("Could not import gafime_cpu")
    sys.exit(1)

from gafime.backends.fused_kernel import create_fold_mask

# Constants from interfaces.h
OP_IDENTITY = 0
OP_LOG = 1
OP_SQRT = 3
OP_INVERSE = 9
INTERACT_MULT = 0
INTERACT_ADD = 1
INTERACT_SUB = 2
INTERACT_DIV = 3

def calc_correlation(stats):
    # stats: [n, sx, sy, sxx, syy, sxy, ...]
    # We use Train split (first 6)
    n = stats[0]
    sx = stats[1]
    sy = stats[2]
    sxx = stats[3]
    syy = stats[4]
    sxy = stats[5]
    
    if n == 0: return 0.0
    
    numerator = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    
    if var_x <= 0 or var_y <= 0:
        return 0.0
        
    return numerator / math.sqrt(var_x * var_y)

def run_extreme_demo():
    print('='*80)
    print('GAFIME EXTREME CHALLENGE: 5M Samples, Arity 3 Search')
    print('='*80)
    
    N_SAMPLES = 5_000_000
    N_FEATURES = 20
    BATCH_SIZE = 4000
    
    print(f"Dataset: {N_FEATURES} features (A-T) x {N_SAMPLES} samples (FP32)")
    print(f"Size: {N_FEATURES * N_SAMPLES * 4 / 1024 / 1024:.1f} MB (Spills L2 Cache)")
    
    print("\n[1] Generating Data...")
    # Generate A-T (Indices 0-19)
    # Use uniform [1, 10] to avoid domain errors
    data = [] 
    for i in range(N_FEATURES):
        feat = np.random.uniform(1.0, 10.0, N_SAMPLES).astype(np.float32)
        data.append(feat)
        
    feat_A = data[0]
    feat_E = data[4]
    feat_H = data[7]
    
    # Target: sqrt(A) * log(E) - 1/H
    # Op Log in GAFIME is log(|x|+eps), here x>0 so log(x)
    # Op Inv in GAFIME is 1/x
    
    # Python calc for Ground Truth
    target = np.sqrt(feat_A) * np.log(feat_E) - (1.0 / feat_H)
    target = target.astype(np.float32)
    
    print("Target Formula: sqrt(A) * log(E) - 1/H")
    
    # Upload
    layout = gafime_cpu.ContiguousLayout(N_SAMPLES, N_FEATURES)
    for f in data:
        layout.add_feature(f.tolist())
    layout.set_target(target.tolist())
    layout.set_mask((create_fold_mask(N_SAMPLES) * 1).astype(np.uint8).tolist())
    
    bucket = gafime_cpu.ContiguousBucket(layout)
    print("Upload Complete.")
    
    # =========================================================================
    # ITERATION 1: Exhaustive Search for Arity 2
    # Search Space: 20 x 20 x 12 Ops x 12 Ops x 6 Types = 345,600 candidates
    # =========================================================================
    print("\n[2] Exhaustive Search (Arity 2)...")
    
    # Generate ALL candidates
    tasks = [] # (fa, fb, oa, ob, type)
    # Optimization: Only scan interesting Ops (Identity, Log, Sqrt, Inv) to save time for demo?
    # No, let's go Extreme. Full Scan for these Ops.
    ops_to_scan = [OP_IDENTITY, OP_LOG, OP_SQRT, OP_INVERSE]
    
    # Reduce search space slightly for demo interactivity (Full 12x12 scan is 34s + overhead)
    # Scanning 4 ops x 4 ops = 16 combos per pair.
    # 20x20 pairs x 16 x 6 types = 38,400 candidates.
    # This will take ~4 seconds.
    
    for fa in range(N_FEATURES):
        for fb in range(N_FEATURES):
            for oa in ops_to_scan:
                for ob in ops_to_scan:
                    for t in [INTERACT_MULT, INTERACT_SUB, INTERACT_ADD, INTERACT_DIV]: # Scan 4 types
                        tasks.append((fa, fb, oa, ob, t))
                        
    print(f"Scanning {len(tasks)} interaction candidates...")
    
    start_time = time.perf_counter()
    
    # Batch execution
    results = [] # (correlation, task_tuple)
    
    chunk_size = BATCH_SIZE
    total_chunks = (len(tasks) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i+chunk_size]
        # Unzip
        f_a, f_b, o_a, o_b, int_type = zip(*chunk)
        
        # Compute
        # val_fold_id = 0
        stats_batch = bucket.compute_batch(list(f_a), list(f_b), list(o_a), list(o_b), list(int_type), 0)
        
        for j, stats in enumerate(stats_batch):
            corr = calc_correlation(stats)
            if not math.isnan(corr):
                results.append((abs(corr), chunk[j]))
        
        # Progress
        current_chunk = i // chunk_size
        if current_chunk % 10 == 0:
            print(f"Processed {current_chunk}/{total_chunks} batches...", end='\r')
            
    elapsed = time.perf_counter() - start_time
    print(f"\nSearch Complete in {elapsed:.2f}s!")
    print(f"Throughput: {len(tasks)/elapsed:.0f} interactions/sec")
    print(f"Sample Throughput: {len(tasks)*N_SAMPLES/elapsed/1e9:.2f} G samples/sec")
    
    # Sort by correlation
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 5 Candidates (Arity 2):")
    for k in range(5):
        score, (fa, fb, oa, ob, t) = results[k]
        # Names
        name_a = chr(ord('A') + fa)
        name_b = chr(ord('A') + fb)
        op_names = {0: '', 1: 'log', 3: 'sqrt', 9: 'inv'}
        type_names = {0: '*', 1: '+', 2: '-', 3: '/'}
        
        expr = f"{op_names.get(oa, '?')}({name_a}) {type_names.get(t, '?')} {op_names.get(ob, '?')}({name_b})"
        print(f"#{k+1}: {expr}  (Corr: {score:.4f})")
        
    # Check if sqrt(A)*log(E) is found
    # Ideal: A=0, E=4. OpA=3, OpB=1. Type=0.
    best_cand = results[0][1]
    
    # =========================================================================
    # ITERATION 2: Materialize & Search Arity 3
    # =========================================================================
    print("\n[3] Iteration 2: Materialize & Deep Search...")
    
    # Re-calculate best feature on CPU (Materialization)
    fa, fb, oa, ob, t = best_cand
    vec_a = data[fa]
    vec_b = data[fb]
    
    # Apply Ops
    if oa == OP_SQRT: vec_a = np.sqrt(np.abs(vec_a))
    elif oa == OP_LOG: vec_a = np.log(np.abs(vec_a) + 1e-9)
    elif oa == OP_INVERSE: vec_a = 1.0 / (vec_a + 1e-9)
    # ...
    
    if ob == OP_SQRT: vec_b = np.sqrt(np.abs(vec_b))
    elif ob == OP_LOG: vec_b = np.log(np.abs(vec_b) + 1e-9)
    elif ob == OP_INVERSE: vec_b = 1.0 / (vec_b + 1e-9)
    
    interact_vec = None
    if t == INTERACT_MULT: interact_vec = vec_a * vec_b
    elif t == INTERACT_SUB: interact_vec = vec_a - vec_b
    elif t == INTERACT_ADD: interact_vec = vec_a + vec_b
    elif t == INTERACT_DIV: interact_vec = vec_a / (vec_b + 1e-9)
    
    # Upload as Feature 20 (Index 20)
    print("Materializing Best Feature to GPU...")
    # Creating NEW layout is expensive? No, just new bucket.
    # To append feature, ContiguousLayout doesn't support append after build.
    # We must rebuild layout with N+1 features.
    
    layout2 = gafime_cpu.ContiguousLayout(N_SAMPLES, N_FEATURES + 1)
    for f in data:
        layout2.add_feature(f.tolist())
    layout2.add_feature(interact_vec.tolist()) # Feature 20 (Best from Iter 1)
    layout2.set_target(target.tolist())
    layout2.set_mask((create_fold_mask(N_SAMPLES) * 1).astype(np.uint8).tolist())
    
    bucket2 = gafime_cpu.ContiguousBucket(layout2)
    
    # Search Feature 20 vs All
    # Target: Feat20 - 1/H
    # Feat 20 is Identity. H is Inv. Type is Sub.
    print("Scanning Arity 3 candidates...")
    
    tasks3 = []
    f20 = 20
    for f_other in range(N_FEATURES):
        for o_other in ops_to_scan:
             # Try (F20 - Op(Others))
             tasks3.append((f20, f_other, OP_IDENTITY, o_other, INTERACT_SUB))
             
    # Run
    f_a, f_b, o_a, o_b, int_type = zip(*tasks3)
    stats_batch = bucket2.compute_batch(list(f_a), list(f_b), list(o_a), list(o_b), list(int_type), 0)
    
    results3 = []
    for j, stats in enumerate(stats_batch):
        corr = calc_correlation(stats)
        if not math.isnan(corr):
            results3.append((abs(corr), tasks3[j]))
            
    results3.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 3 Candidates (Arity 3):")
    for k in range(3):
        score, (fa, fb, oa, ob, t) = results3[k]
        name_a = "ITER1_BEST" if fa == 20 else chr(ord('A') + fa)
        name_b = "ITER1_BEST" if fb == 20 else chr(ord('A') + fb)
        
        op_names = {0: '', 1: 'log', 3: 'sqrt', 9: 'inv'}
        type_names = {0: '*', 1: '+', 2: '-', 3: '/'}
        
        expr = f"{op_names.get(oa, '')}({name_a}) {type_names.get(t, '?')} {op_names.get(ob, '')}({name_b})"
        print(f"#{k+1}: {expr}  (Corr: {score:.6f})")

    # Verify H
    # H is index 7. so 'H'.
    # Op=9 (inv).
    # Expr: ITER1_BEST - inv(H).
    
if __name__ == "__main__":
    run_extreme_demo()
