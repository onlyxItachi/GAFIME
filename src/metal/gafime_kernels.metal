/**
 * GAFIME Metal Compute Shaders - Operator-Fused Map-Reduce Architecture
 * 
 * Metal Shading Language (MSL) implementation for Apple Silicon (M1/M2/M3/M4).
 * 
 * Architecture advantages over CUDA on Apple Silicon:
 * 1. Unified Memory (UMA): Zero-copy data sharing between CPU and GPU
 * 2. SIMD groups (32 threads): Same as CUDA warps, with simpler sync model
 * 3. No explicit sync mask: simd_shuffle_down is implicitly synchronized
 * 4. Integrated GPU: Lower latency than discrete GPUs over PCIe
 * 
 * Statistics accumulated: N, ΣX, ΣY, ΣX², ΣY², ΣXY
 * Pearson formula: r = (NΣxy - ΣxΣy) / sqrt((NΣx² - (Σx)²)(NΣy² - (Σy)²))
 */

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// CONSTANTS (mirror interfaces.h)
// ============================================================================

constant int GAFIME_OP_IDENTITY     = 0;
constant int GAFIME_OP_LOG          = 1;
constant int GAFIME_OP_EXP          = 2;
constant int GAFIME_OP_SQRT         = 3;
constant int GAFIME_OP_TANH         = 4;
constant int GAFIME_OP_SIGMOID      = 5;
constant int GAFIME_OP_SQUARE       = 6;
constant int GAFIME_OP_NEGATE       = 7;
constant int GAFIME_OP_ABS          = 8;
constant int GAFIME_OP_INVERSE      = 9;
constant int GAFIME_OP_CUBE         = 10;

constant int GAFIME_INTERACT_MULT   = 0;
constant int GAFIME_INTERACT_ADD    = 1;
constant int GAFIME_INTERACT_SUB    = 2;
constant int GAFIME_INTERACT_DIV    = 3;
constant int GAFIME_INTERACT_MAX    = 4;
constant int GAFIME_INTERACT_MIN    = 5;

constant int GAFIME_TS_LAG = 1;
constant int GAFIME_TS_DELTA = 2;
constant int GAFIME_TS_VELOCITY = 3;
constant int GAFIME_TS_ACCELERATION = 4;
constant int GAFIME_TS_ROLLING_MEAN = 5;
constant int GAFIME_TS_ROLLING_STD = 6;
constant int GAFIME_TS_ROLLING_SUM = 7;

constant int SIMD_SIZE              = 32;constant int GAFIME_CONTINUOUS_ARITY [[function_constant(0)]];

// ============================================================================
// KERNEL PARAMETERS (passed via buffer)
// ============================================================================

struct FusedParams {
    int ops[5];                 // Unary operator IDs per feature
    int interaction_types[4];   // Per-pair interaction types
    int arity;                  // Number of features (2-5)
    int val_fold_id;            // Validation fold ID
    int n_samples;              // Total samples
    int padding;                // Alignment padding
};

struct BatchParams {
    int batch_size;
    int val_fold_id;
    int n_samples;
    int padding;
};

struct MatrixBatchParams {
    int batch_size;
    int val_fold_id;
    int n_samples;
    int n_features;
};

struct TimeSeriesBatchParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int padding;
};

// ============================================================================
// UNARY OPERATORS (matching CUDA/CPU apply_op exactly)
// ============================================================================

inline float apply_op(float x, int op) {
    switch (op) {
        case 1:  // LOG
            return log(abs(x) + 1e-8f);
            
        case 2:  // EXP
            return exp(clamp(x, -20.0f, 20.0f));
            
        case 3:  // SQRT
            return sqrt(abs(x));
            
        case 4: { // TANH
            float exp2x = exp(2.0f * clamp(x, -10.0f, 10.0f));
            return (exp2x - 1.0f) / (exp2x + 1.0f);
        }
            
        case 5: { // SIGMOID
            float ex = exp(-clamp(x, -20.0f, 20.0f));
            return 1.0f / (1.0f + ex);
        }
            
        case 6:  // SQUARE
            return x * x;
            
        case 7:  // NEGATE
            return -x;
            
        case 8:  // ABS
            return abs(x);
            
        case 9:  // INVERSE
            return 1.0f / (abs(x) < 1e-8f ? copysign(1e-8f, x) : x);
            
        case 10: // CUBE
            return x * x * x;
            
        case 0:  // IDENTITY
        default:
            return x;
    }
}

// ============================================================================
// INTERACTION COMBINERS (matching CUDA/CPU combine exactly)
// ============================================================================

inline float combine(float a, float b, int interact_type) {
    switch (interact_type) {
        case 1:  // ADD
            return a + b;
        case 2:  // SUB
            return a - b;
        case 3:  // DIV
            return a / (abs(b) < 1e-8f ? copysign(1e-8f, b) : b);
        case 4:  // MAX
            return max(a, b);
        case 5:  // MIN
            return min(a, b);
        case 0:  // MULT
        default:
            return a * b;
    }
}

inline float time_series_eval(
    device const float* X,
    int n_samples,
    uint row,
    int kind,
    int feature_idx,
    int lag,
    int window
) {
    device const float* col = X + feature_idx * n_samples;
    int idx = int(row);
    int safe_lag = max(lag, 1);
    int safe_window = max(window, 1);
    int lag_idx = max(idx - safe_lag, 0);

    if (kind == GAFIME_TS_LAG) {
        return col[lag_idx];
    }
    if (kind == GAFIME_TS_DELTA) {
        return col[idx] - col[lag_idx];
    }
    if (kind == GAFIME_TS_VELOCITY) {
        return (col[idx] - col[lag_idx]) / float(safe_lag);
    }
    if (kind == GAFIME_TS_ACCELERATION) {
        int lag2_idx = max(idx - 2 * safe_lag, 0);
        return (col[idx] - 2.0f * col[lag_idx] + col[lag2_idx])
            / float(safe_lag * safe_lag);
    }
    if (kind == GAFIME_TS_ROLLING_SUM ||
        kind == GAFIME_TS_ROLLING_MEAN ||
        kind == GAFIME_TS_ROLLING_STD) {
        int start = max(0, idx - safe_window + 1);
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;
        for (int i = start; i <= idx && i < n_samples; ++i) {
            float value = col[i];
            sum += value;
            sum_sq += value * value;
            count += 1;
        }
        if (kind == GAFIME_TS_ROLLING_SUM) {
            return sum;
        }
        float local_mean = sum / float(max(count, 1));
        if (kind == GAFIME_TS_ROLLING_MEAN) {
            return local_mean;
        }
        float variance = max(sum_sq / float(max(count, 1)) - local_mean * local_mean, 0.0f);
        return sqrt(variance);
    }
    return NAN;
}

inline void threadgroup_atomic_float_store(
    threadgroup atomic_uint* addr,
    float value
) {
    atomic_store_explicit(addr, as_type<uint>(value), memory_order_relaxed);
}

inline float threadgroup_atomic_float_load(
    threadgroup atomic_uint* addr
) {
    return as_type<float>(atomic_load_explicit(addr, memory_order_relaxed));
}

inline void threadgroup_atomic_float_add(
    threadgroup atomic_uint* addr,
    float value
) {
    uint old_bits = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float old_value = as_type<float>(old_bits);
        uint new_bits = as_type<uint>(old_value + value);
        uint expected = old_bits;
        if (atomic_compare_exchange_weak_explicit(
                addr,
                &expected,
                new_bits,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
        old_bits = expected;
    }
}

// ============================================================================
// SIMD GROUP REDUCTION (equivalent to CUDA warp_reduce_6)
// ============================================================================

/**
 * Reduce 6 accumulators across a SIMD group (32 threads).
 * Uses simd_shuffle_down — no explicit sync mask needed (Metal handles it).
 */
inline void simd_reduce_6(
    thread float& n, thread float& sx, thread float& sy,
    thread float& sxx, thread float& syy, thread float& sxy
) {
    for (ushort offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        n   += simd_shuffle_down(n, offset);
        sx  += simd_shuffle_down(sx, offset);
        sy  += simd_shuffle_down(sy, offset);
        sxx += simd_shuffle_down(sxx, offset);
        syy += simd_shuffle_down(syy, offset);
        sxy += simd_shuffle_down(sxy, offset);
    }
}

// ============================================================================
// GLOBAL MATRIX BATCH KERNEL
// ============================================================================

kernel void gafime_global_continuous_kernel(
    device const float*   X_colmajor      [[buffer(0)]],
    device const float*   target          [[buffer(1)]],
    device const uchar*   mask            [[buffer(2)]],
    device const float*   means           [[buffer(3)]],
    device const int*     batch_indices   [[buffer(4)]],
    device const MatrixBatchParams& params [[buffer(5)]],
    device atomic_float*  stats_batch     [[buffer(6)]],
    uint2 gid                             [[thread_position_in_grid]],
    uint2 grid_dim                        [[threads_per_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_group_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg               [[simdgroups_per_threadgroup]]
) {
    int batch_id = int(gid.y);
    if (batch_id >= params.batch_size) return;

    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;
    float val_n = 0.0f, val_sx = 0.0f, val_sy = 0.0f;
    float val_sxx = 0.0f, val_syy = 0.0f, val_sxy = 0.0f;

    int n_samples = params.n_samples;
    int n_features = params.n_features;
    int val_fold = params.val_fold_id;
    uint threads_x = grid_dim.x;

    for (uint row = gid.x; row < uint(n_samples); row += threads_x) {
        int f0 = batch_indices[batch_id * GAFIME_CONTINUOUS_ARITY + 0];
        if (f0 < 0 || f0 >= n_features) continue;

        float x = X_colmajor[uint(f0) * uint(n_samples) + row];
        if (GAFIME_CONTINUOUS_ARITY > 1) {
            x -= means[f0];
        }

        for (int slot = 1; slot < GAFIME_CONTINUOUS_ARITY; ++slot) {
            int feature_idx = batch_indices[batch_id * GAFIME_CONTINUOUS_ARITY + slot];
            if (feature_idx < 0 || feature_idx >= n_features) {
                x = NAN;
                break;
            }
            float value = X_colmajor[uint(feature_idx) * uint(n_samples) + row] - means[feature_idx];
            x *= value;
        }

        float y = target[row];
        if (isnan(x) || isnan(y)) continue;

        uchar fold = mask[row];
        if (fold == uchar(val_fold)) {
            val_n += 1.0f;
            val_sx += x;
            val_sy += y;
            val_sxx += x * x;
            val_syy += y * y;
            val_sxy += x * y;
        } else {
            train_n += 1.0f;
            train_sx += x;
            train_sy += y;
            train_sxx += x * x;
            train_syy += y * y;
            train_sxy += x * y;
        }
    }

    simd_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    simd_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);

    threadgroup float shared_train[6 * 32];
    threadgroup float shared_val[6 * 32];

    if (simd_lane == 0) {
        shared_train[simd_group_id * 6 + 0] = train_n;
        shared_train[simd_group_id * 6 + 1] = train_sx;
        shared_train[simd_group_id * 6 + 2] = train_sy;
        shared_train[simd_group_id * 6 + 3] = train_sxx;
        shared_train[simd_group_id * 6 + 4] = train_syy;
        shared_train[simd_group_id * 6 + 5] = train_sxy;
        shared_val[simd_group_id * 6 + 0] = val_n;
        shared_val[simd_group_id * 6 + 1] = val_sx;
        shared_val[simd_group_id * 6 + 2] = val_sy;
        shared_val[simd_group_id * 6 + 3] = val_sxx;
        shared_val[simd_group_id * 6 + 4] = val_syy;
        shared_val[simd_group_id * 6 + 5] = val_sxy;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float ft[6] = {0};
        float fv[6] = {0};

        for (uint w = simd_lane; w < simd_groups_per_tg; w += SIMD_SIZE) {
            for (int j = 0; j < 6; j++) {
                ft[j] += shared_train[w * 6 + j];
                fv[j] += shared_val[w * 6 + j];
            }
        }

        for (int j = 0; j < 6; j++) {
            for (ushort offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
                ft[j] += simd_shuffle_down(ft[j], offset);
                fv[j] += simd_shuffle_down(fv[j], offset);
            }
        }

        if (simd_lane == 0) {
            device atomic_float* out = &stats_batch[batch_id * 12];
            for (int j = 0; j < 6; j++) {
                atomic_fetch_add_explicit(&out[j], ft[j], memory_order_relaxed);
                atomic_fetch_add_explicit(&out[j + 6], fv[j], memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// FUSED MAP-REDUCE KERNEL (Main compute kernel)
// ============================================================================

/**
 * Fused kernel: Transform features, combine, reduce to 12-float stats.
 * 
 * Apple Silicon optimizations:
 * - UMA: input buffers are shared memory — no device copy needed
 * - SIMD group reduction: same as warp shuffle, but implicitly synchronized
 * - Threadgroup memory: same as CUDA shared memory
 * 
 * Supports arity 2-5 with per-pair interaction types.
 */
kernel void gafime_fused_kernel(
    device const float*   input0          [[buffer(0)]],
    device const float*   input1          [[buffer(1)]],
    device const float*   input2          [[buffer(2)]],
    device const float*   input3          [[buffer(3)]],
    device const float*   input4          [[buffer(4)]],
    device const float*   target          [[buffer(5)]],
    device const uchar*   mask            [[buffer(6)]],
    device const FusedParams& params      [[buffer(7)]],
    device atomic_float*  stats_out       [[buffer(8)]],
    uint tid                              [[thread_position_in_grid]],
    uint grid_size                        [[threads_per_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_group_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg               [[simdgroups_per_threadgroup]],
    uint tg_size                          [[threads_per_threadgroup]]
) {
    // Per-thread accumulators (registers)
    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;
    float val_n = 0.0f, val_sx = 0.0f, val_sy = 0.0f;
    float val_sxx = 0.0f, val_syy = 0.0f, val_sxy = 0.0f;
    
    int n_samples = params.n_samples;
    int arity = params.arity;
    int val_fold = params.val_fold_id;
    
    // Grid-stride loop for processing samples
    for (uint i = tid; i < uint(n_samples); i += grid_size) {
        // Apply unary operators to each feature
        float v0 = apply_op(input0[i], params.ops[0]);
        float v1 = apply_op(input1[i], params.ops[1]);
        
        // Combine with per-pair interaction types
        float x = combine(v0, v1, params.interaction_types[0]);
        
        if (arity >= 3) {
            float v2 = apply_op(input2[i], params.ops[2]);
            x = combine(x, v2, params.interaction_types[1]);
        }
        if (arity >= 4) {
            float v3 = apply_op(input3[i], params.ops[3]);
            x = combine(x, v3, params.interaction_types[2]);
        }
        if (arity >= 5) {
            float v4 = apply_op(input4[i], params.ops[4]);
            x = combine(x, v4, params.interaction_types[3]);
        }
        
        float y = target[i];
        
        // NaN guard
        if (isnan(x) || isnan(y)) continue;
        
        uchar fold = mask[i];
        
        // Accumulate into train or val
        if (fold == uchar(val_fold)) {
            val_n += 1.0f;
            val_sx += x;
            val_sy += y;
            val_sxx += x * x;
            val_syy += y * y;
            val_sxy += x * y;
        } else {
            train_n += 1.0f;
            train_sx += x;
            train_sy += y;
            train_sxx += x * x;
            train_syy += y * y;
            train_sxy += x * y;
        }
    }
    
    // ========================================================================
    // SIMD group reduction (equivalent to CUDA warp reduction)
    // ========================================================================
    simd_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    simd_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);
    
    // ========================================================================
    // Threadgroup reduction via threadgroup memory
    // ========================================================================
    threadgroup float shared_train[6 * 32]; // max 32 SIMD groups per threadgroup
    threadgroup float shared_val[6 * 32];
    
    if (simd_lane == 0) {
        shared_train[simd_group_id * 6 + 0] = train_n;
        shared_train[simd_group_id * 6 + 1] = train_sx;
        shared_train[simd_group_id * 6 + 2] = train_sy;
        shared_train[simd_group_id * 6 + 3] = train_sxx;
        shared_train[simd_group_id * 6 + 4] = train_syy;
        shared_train[simd_group_id * 6 + 5] = train_sxy;
        
        shared_val[simd_group_id * 6 + 0] = val_n;
        shared_val[simd_group_id * 6 + 1] = val_sx;
        shared_val[simd_group_id * 6 + 2] = val_sy;
        shared_val[simd_group_id * 6 + 3] = val_sxx;
        shared_val[simd_group_id * 6 + 4] = val_syy;
        shared_val[simd_group_id * 6 + 5] = val_sxy;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First SIMD group does final reduction
    if (simd_group_id == 0) {
        float final_train[6] = {0};
        float final_val[6] = {0};
        
        for (uint w = simd_lane; w < simd_groups_per_tg; w += SIMD_SIZE) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += shared_train[w * 6 + j];
                final_val[j] += shared_val[w * 6 + j];
            }
        }
        
        // Final SIMD reduction
        for (int j = 0; j < 6; j++) {
            for (ushort offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
                final_train[j] += simd_shuffle_down(final_train[j], offset);
                final_val[j] += simd_shuffle_down(final_val[j], offset);
            }
        }
        
        // Lane 0 writes to global output with atomics
        if (simd_lane == 0) {
            atomic_fetch_add_explicit(&stats_out[0],  final_train[0], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[1],  final_train[1], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[2],  final_train[2], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[3],  final_train[3], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[4],  final_train[4], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[5],  final_train[5], memory_order_relaxed);
            
            atomic_fetch_add_explicit(&stats_out[6],  final_val[0], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[7],  final_val[1], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[8],  final_val[2], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[9],  final_val[3], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[10], final_val[4], memory_order_relaxed);
            atomic_fetch_add_explicit(&stats_out[11], final_val[5], memory_order_relaxed);
        }
    }
}

// ============================================================================
// BATCHED COMPUTE KERNEL (N interactions in one dispatch)
// ============================================================================

/**
 * Batched kernel: compute N feature interactions in one dispatch.
 * Each threadgroup handles one interaction from the batch.
 */
kernel void gafime_batched_kernel(
    device const float*   features_0      [[buffer(0)]],
    device const float*   features_1      [[buffer(1)]],
    device const float*   features_2      [[buffer(2)]],
    device const float*   features_3      [[buffer(3)]],
    device const float*   features_4      [[buffer(4)]],
    device const float*   target          [[buffer(5)]],
    device const uchar*   mask            [[buffer(6)]],
    device const int*     batch_indices   [[buffer(7)]],   // [N * 2]
    device const int*     batch_ops       [[buffer(8)]],   // [N * 2]
    device const int*     batch_interact  [[buffer(9)]],   // [N]
    device const BatchParams& params      [[buffer(10)]],
    device atomic_float*  stats_batch     [[buffer(11)]],  // [N * 12]
    uint2 gid                             [[thread_position_in_grid]],
    uint2 grid_dim                        [[threads_per_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_group_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg               [[simdgroups_per_threadgroup]],
    uint2 tg_size                         [[threads_per_threadgroup]]
) {
    int batch_id = gid.y;
    if (batch_id >= params.batch_size) return;
    
    // Feature pointer array
    device const float* features[5] = {features_0, features_1, features_2, features_3, features_4};
    
    // Load interaction parameters
    int f0_idx = batch_indices[batch_id * 2 + 0];
    int f1_idx = batch_indices[batch_id * 2 + 1];
    int op0 = batch_ops[batch_id * 2 + 0];
    int op1 = batch_ops[batch_id * 2 + 1];
    int interact = batch_interact[batch_id];
    
    device const float* f0 = features[f0_idx];
    device const float* f1 = features[f1_idx];
    
    // Per-thread accumulators
    float train_n = 0, train_sx = 0, train_sy = 0;
    float train_sxx = 0, train_syy = 0, train_sxy = 0;
    float val_n = 0, val_sx = 0, val_sy = 0;
    float val_sxx = 0, val_syy = 0, val_sxy = 0;
    
    int n_samples = params.n_samples;
    int val_fold = params.val_fold_id;
    uint threads_x = grid_dim.x;
    
    // Grid-stride loop within this interaction
    for (uint i = gid.x; i < uint(n_samples); i += threads_x) {
        float x0 = apply_op(f0[i], op0);
        float x1 = apply_op(f1[i], op1);
        float X = combine(x0, x1, interact);
        float Y = target[i];
        
        if (isnan(X) || isnan(Y)) continue;
        
        uchar fold = mask[i];
        if (fold == uchar(val_fold)) {
            val_n += 1.0f; val_sx += X; val_sy += Y;
            val_sxx += X*X; val_syy += Y*Y; val_sxy += X*Y;
        } else {
            train_n += 1.0f; train_sx += X; train_sy += Y;
            train_sxx += X*X; train_syy += Y*Y; train_sxy += X*Y;
        }
    }
    
    // SIMD group reduction
    simd_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    simd_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);
    
    // Threadgroup reduction via shared memory
    threadgroup float shared_train[6 * 32];
    threadgroup float shared_val[6 * 32];
    
    if (simd_lane == 0) {
        shared_train[simd_group_id * 6 + 0] = train_n;
        shared_train[simd_group_id * 6 + 1] = train_sx;
        shared_train[simd_group_id * 6 + 2] = train_sy;
        shared_train[simd_group_id * 6 + 3] = train_sxx;
        shared_train[simd_group_id * 6 + 4] = train_syy;
        shared_train[simd_group_id * 6 + 5] = train_sxy;
        
        shared_val[simd_group_id * 6 + 0] = val_n;
        shared_val[simd_group_id * 6 + 1] = val_sx;
        shared_val[simd_group_id * 6 + 2] = val_sy;
        shared_val[simd_group_id * 6 + 3] = val_sxx;
        shared_val[simd_group_id * 6 + 4] = val_syy;
        shared_val[simd_group_id * 6 + 5] = val_sxy;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        float ft[6] = {0};
        float fv[6] = {0};
        
        for (uint w = simd_lane; w < simd_groups_per_tg; w += SIMD_SIZE) {
            for (int j = 0; j < 6; j++) {
                ft[j] += shared_train[w * 6 + j];
                fv[j] += shared_val[w * 6 + j];
            }
        }
        
        for (int j = 0; j < 6; j++) {
            for (ushort offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
                ft[j] += simd_shuffle_down(ft[j], offset);
                fv[j] += simd_shuffle_down(fv[j], offset);
            }
        }
        
        if (simd_lane == 0) {
            device atomic_float* out = &stats_batch[batch_id * 12];
            for (int j = 0; j < 6; j++) {
                atomic_fetch_add_explicit(&out[j],     ft[j], memory_order_relaxed);
                atomic_fetch_add_explicit(&out[j + 6], fv[j], memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// TIME-SERIES BATCH KERNEL
// ============================================================================

kernel void gafime_time_series_batch_kernel(
    device const float*   X              [[buffer(0)]],
    device const float*   y              [[buffer(1)]],
    device const int*     kinds          [[buffer(2)]],
    device const int*     feature_index  [[buffer(3)]],
    device const int*     lags           [[buffer(4)]],
    device const int*     windows        [[buffer(5)]],
    device const TimeSeriesBatchParams& params [[buffer(6)]],
    device atomic_float*  stats_batch    [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]],
    uint2 grid_dim                       [[threads_per_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_group_id                   [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg              [[simdgroups_per_threadgroup]]
) {
    int candidate_id = int(gid.y);
    if (candidate_id >= params.n_candidates) return;

    int kind = kinds[candidate_id];
    int feature = feature_index[candidate_id];
    int lag = lags[candidate_id];
    int window = windows[candidate_id];
    if (feature < 0 || feature >= params.n_features) return;

    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;

    for (uint row = gid.x; row < uint(params.n_samples); row += grid_dim.x) {
        float x = time_series_eval(
            X, params.n_samples, row, kind, feature, lag, window
        );
        float target = y[row];
        if (isnan(x) || isnan(target)) continue;

        train_n += 1.0f;
        train_sx += x;
        train_sy += target;
        train_sxx += x * x;
        train_syy += target * target;
        train_sxy += x * target;
    }

    simd_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);

    threadgroup float shared_train[6 * 32];
    if (simd_lane == 0) {
        shared_train[simd_group_id * 6 + 0] = train_n;
        shared_train[simd_group_id * 6 + 1] = train_sx;
        shared_train[simd_group_id * 6 + 2] = train_sy;
        shared_train[simd_group_id * 6 + 3] = train_sxx;
        shared_train[simd_group_id * 6 + 4] = train_syy;
        shared_train[simd_group_id * 6 + 5] = train_sxy;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float final_train[6] = {0};
        for (uint w = simd_lane; w < simd_groups_per_tg; w += SIMD_SIZE) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += shared_train[w * 6 + j];
            }
        }

        for (int j = 0; j < 6; j++) {
            for (ushort offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
                final_train[j] += simd_shuffle_down(final_train[j], offset);
            }
        }

        if (simd_lane == 0) {
            device atomic_float* out = &stats_batch[candidate_id * 12];
            for (int j = 0; j < 6; j++) {
                atomic_fetch_add_explicit(&out[j], final_train[j], memory_order_relaxed);
            }
        }
    }
}
