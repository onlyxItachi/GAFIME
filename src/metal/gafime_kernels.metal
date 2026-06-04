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

constant int GAFIME_DISCRETE_SOFT_THRESHOLD          = 0;
constant int GAFIME_DISCRETE_SOFT_INTERVAL           = 1;
constant int GAFIME_DISCRETE_VALUE_GATED_THRESHOLD   = 2;
constant int GAFIME_DISCRETE_SOFT_RECTANGLE          = 3;
constant int GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE = 4;
constant int GAFIME_DISCRETE_DIRECTION_LE            = 1;

constant int GAFIME_TS_LAG = 1;
constant int GAFIME_TS_DELTA = 2;
constant int GAFIME_TS_VELOCITY = 3;
constant int GAFIME_TS_ACCELERATION = 4;
constant int GAFIME_TS_ROLLING_MEAN = 5;
constant int GAFIME_TS_ROLLING_STD = 6;
constant int GAFIME_TS_ROLLING_SUM = 7;

constant int SIMD_SIZE              = 32;
constant int GAFIME_SELECTION_SCORE_SIZE = 4;
constant int GAFIME_SELECTION_MUTUAL_INFO = 0;
constant int GAFIME_SELECTION_VARIANCE_REDUCTION = 1;
constant int GAFIME_SELECTION_RESIDUAL_ABS_CORR = 2;
constant int GAFIME_SELECTION_RESIDUAL_R2_GAIN = 3;
constant int GAFIME_CONTINUOUS_ARITY [[function_constant(0)]];

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

struct DiscreteBatchParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int padding;
};

struct MatrixBatchParams {
    int batch_size;
    int val_fold_id;
    int n_samples;
    int n_features;
};

struct DiscreteSelectionParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int target_bins;
    float y_sum;
    float y_sq_sum;
    int padding0;
    int padding1;
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

// ============================================================================
// DISCRETE SOFT FUNCTION HELPERS
// ============================================================================

inline float discrete_sigmoid(float z) {
    float z_clamped = clamp(z, -60.0f, 60.0f);
    return 1.0f / (1.0f + exp(-z_clamped));
}

inline float discrete_scale(float scale) {
    return scale > 1e-12f ? scale : 1.0f;
}

inline float discrete_threshold_gate(
    float x,
    float threshold,
    int direction,
    float scale,
    float sharpness
) {
    float sign = direction == GAFIME_DISCRETE_DIRECTION_LE ? -1.0f : 1.0f;
    float z = sharpness * sign * (x - threshold) / discrete_scale(scale);
    return discrete_sigmoid(z);
}

inline float discrete_interval_gate(
    float x,
    float low,
    float high,
    float scale,
    float sharpness
) {
    float safe_scale = discrete_scale(scale);
    float left = discrete_sigmoid(sharpness * (x - low) / safe_scale);
    float right = discrete_sigmoid(sharpness * (high - x) / safe_scale);
    return left * right;
}

inline float discrete_eval_soft(
    device const float* X,
    int n_samples,
    uint row,
    int kind,
    int feature_a,
    int feature_b,
    int value_feature,
    int direction,
    device const float* params,
    device const float* scales,
    float sharpness
) {
    device const float* feature0 = X + feature_a * n_samples;
    float a = feature0[row];

    switch (kind) {
        case 0:
            return discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );
        case 1:
            return discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
        case 2: {
            device const float* value_col = X + value_feature * n_samples;
            float gate = discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );
            return value_col[row] * gate;
        }
        case 3: {
            device const float* feature1 = X + feature_b * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[row], params[2], params[3], scales[1], sharpness
            );
            return mask0 * mask1;
        }
        case 4: {
            device const float* feature1 = X + feature_b * n_samples;
            device const float* value_col = X + value_feature * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[row], params[2], params[3], scales[1], sharpness
            );
            return value_col[row] * mask0 * mask1;
        }
        default:
            return NAN;
    }
}

inline float discrete_eval_mask_soft(
    device const float* X,
    int n_samples,
    uint row,
    int kind,
    int feature_a,
    int feature_b,
    int direction,
    device const float* params,
    device const float* scales,
    float sharpness
) {
    device const float* feature0 = X + feature_a * n_samples;
    float a = feature0[row];

    switch (kind) {
        case 0:
        case 2:
            return discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );
        case 1:
            return discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
        case 3:
        case 4: {
            device const float* feature1 = X + feature_b * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[row], params[2], params[3], scales[1], sharpness
            );
            return mask0 * mask1;
        }
        default:
            return NAN;
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
// DISCRETE SOFT FUNCTION BATCH KERNEL
// ============================================================================

kernel void gafime_discrete_soft_batch_kernel(
    device const float*   X              [[buffer(0)]],
    device const float*   y              [[buffer(1)]],
    device const int*     kinds          [[buffer(2)]],
    device const int*     feature_a      [[buffer(3)]],
    device const int*     feature_b      [[buffer(4)]],
    device const int*     value_feature  [[buffer(5)]],
    device const int*     directions     [[buffer(6)]],
    device const float*   candidate_params [[buffer(7)]],
    device const float*   scales         [[buffer(8)]],
    device const float*   sharpness      [[buffer(9)]],
    device const DiscreteBatchParams& params [[buffer(10)]],
    device atomic_float*  stats_batch    [[buffer(11)]],
    uint2 gid                            [[thread_position_in_grid]],
    uint2 grid_dim                       [[threads_per_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_group_id                   [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg              [[simdgroups_per_threadgroup]]
) {
    int candidate_id = int(gid.y);
    if (candidate_id >= params.n_candidates) return;

    int kind = kinds[candidate_id];
    int fa = feature_a[candidate_id];
    int fb = feature_b[candidate_id];
    int vf = value_feature[candidate_id];
    int direction = directions[candidate_id];
    device const float* p = candidate_params + candidate_id * 4;
    device const float* s = scales + candidate_id * 2;
    float k = sharpness[candidate_id];

    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;

    for (uint row = gid.x; row < uint(params.n_samples); row += grid_dim.x) {
        float x = discrete_eval_soft(
            X, params.n_samples, row, kind, fa, fb, vf, direction, p, s, k
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

kernel void gafime_discrete_selection_adaptive_kernel(
    device const float*   X              [[buffer(0)]],
    device const float*   y              [[buffer(1)]],
    device const float*   residual       [[buffer(2)]],
    device const int*     y_bins         [[buffer(3)]],
    device const int*     kinds          [[buffer(4)]],
    device const int*     feature_a      [[buffer(5)]],
    device const int*     feature_b      [[buffer(6)]],
    device const int*     value_feature  [[buffer(7)]],
    device const int*     directions     [[buffer(8)]],
    device const float*   candidate_params [[buffer(9)]],
    device const float*   scales         [[buffer(10)]],
    device const float*   sharpness      [[buffer(11)]],
    device const DiscreteSelectionParams& params [[buffer(12)]],
    device float*         scores_batch   [[buffer(13)]],
    uint tid                             [[thread_position_in_threadgroup]],
    uint candidate_gid                   [[threadgroup_position_in_grid]]
) {
    int candidate_id = int(candidate_gid);
    if (candidate_id >= params.n_candidates) return;

    int kind = kinds[candidate_id];
    int fa = feature_a[candidate_id];
    int fb = feature_b[candidate_id];
    int vf = value_feature[candidate_id];
    int direction = directions[candidate_id];
    device const float* p = candidate_params + candidate_id * 4;
    device const float* s = scales + candidate_id * 2;
    float k = sharpness[candidate_id];

    threadgroup atomic_uint hist_in[96];
    threadgroup atomic_uint hist_out[96];
    threadgroup float partials[10 * 256];

    for (uint idx = tid; idx < 96; idx += 256) {
        threadgroup_atomic_float_store(&hist_in[idx], 0.0f);
        threadgroup_atomic_float_store(&hist_out[idx], 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sw = 0.0f, sw2 = 0.0f, swy = 0.0f, swyy = 0.0f;
    float n = 0.0f, sx = 0.0f, sr = 0.0f;
    float sxx = 0.0f, srr = 0.0f, sxr = 0.0f;
    int target_bins = min(max(params.target_bins, 2), 96);

    for (uint row = tid; row < uint(params.n_samples); row += 256) {
        float feature_value = discrete_eval_soft(
            X, params.n_samples, row, kind, fa, fb, vf, direction, p, s, k
        );
        float mask = discrete_eval_mask_soft(
            X, params.n_samples, row, kind, fa, fb, direction, p, s, k
        );
        float target = y[row];
        float res = residual[row];
        if (isnan(feature_value) || isnan(mask) || isnan(target) || isnan(res)) {
            continue;
        }

        mask = clamp(mask, 0.0f, 1.0f);
        float out_w = 1.0f - mask;
        sw += mask;
        sw2 += mask * mask;
        swy += mask * target;
        swyy += mask * target * target;

        n += 1.0f;
        sx += feature_value;
        sr += res;
        sxx += feature_value * feature_value;
        srr += res * res;
        sxr += feature_value * res;

        int yb = y_bins[row];
        if (yb >= 0 && yb < target_bins) {
            threadgroup_atomic_float_add(&hist_in[yb], mask);
            threadgroup_atomic_float_add(&hist_out[yb], out_w);
        }
    }

    partials[tid] = sw;
    partials[256 + tid] = sw2;
    partials[2 * 256 + tid] = swy;
    partials[3 * 256 + tid] = swyy;
    partials[4 * 256 + tid] = n;
    partials[5 * 256 + tid] = sx;
    partials[6 * 256 + tid] = sr;
    partials[7 * 256 + tid] = sxx;
    partials[8 * 256 + tid] = srr;
    partials[9 * 256 + tid] = sxr;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sum_sw = 0.0f, sum_sw2 = 0.0f, sum_swy = 0.0f, sum_swyy = 0.0f;
        float sum_n = 0.0f, sum_sx = 0.0f, sum_sr = 0.0f;
        float sum_sxx = 0.0f, sum_srr = 0.0f, sum_sxr = 0.0f;
        for (uint i = 0; i < 256; i++) {
            sum_sw += partials[i];
            sum_sw2 += partials[256 + i];
            sum_swy += partials[2 * 256 + i];
            sum_swyy += partials[3 * 256 + i];
            sum_n += partials[4 * 256 + i];
            sum_sx += partials[5 * 256 + i];
            sum_sr += partials[6 * 256 + i];
            sum_sxx += partials[7 * 256 + i];
            sum_srr += partials[8 * 256 + i];
            sum_sxr += partials[9 * 256 + i];
        }

        float total_n = max(sum_n, 0.0f);
        float right_w = total_n - sum_sw;
        float right_sw2 = total_n - 2.0f * sum_sw + sum_sw2;
        float effective_in = (sum_sw2 > 1e-12f) ? (sum_sw * sum_sw / sum_sw2) : 0.0f;
        float effective_out = (right_sw2 > 1e-12f) ? (right_w * right_w / right_sw2) : 0.0f;
        float min_support = min(8.0f, max(3.0f, 0.02f * total_n));
        bool support_ok = effective_in >= min_support && effective_out >= min_support;

        float total_sse = params.y_sq_sum - (params.y_sum * params.y_sum) / max(total_n, 1.0f);
        float left_sse = 0.0f;
        if (sum_sw > 1e-9f) {
            left_sse = max(sum_swyy - (sum_swy * sum_swy) / sum_sw, 0.0f);
        }
        float right_swy = params.y_sum - sum_swy;
        float right_swyy = params.y_sq_sum - sum_swyy;
        float right_sse = 0.0f;
        if (right_w > 1e-9f) {
            right_sse = max(right_swyy - (right_swy * right_swy) / right_w, 0.0f);
        }
        float variance_gain = 0.0f;
        if (support_ok && total_sse > 1e-12f) {
            variance_gain = max((total_sse - left_sse - right_sse) / total_sse, 0.0f);
        }

        float cov = sum_sxr - (sum_sx * sum_sr) / max(total_n, 1.0f);
        float var_x = sum_sxx - (sum_sx * sum_sx) / max(total_n, 1.0f);
        float var_r = sum_srr - (sum_sr * sum_sr) / max(total_n, 1.0f);
        float residual_corr = 0.0f;
        float denom = var_x * var_r;
        if (denom > 1e-20f) {
            residual_corr = min(abs(cov / sqrt(denom)), 1.0f);
        }
        float residual_r2 = residual_corr * residual_corr;

        float mutual_info = 0.0f;
        if (support_ok && total_n > 0.0f) {
            int nonzero_y = 0;
            for (int by = 0; by < target_bins; by++) {
                float py_count = threadgroup_atomic_float_load(&hist_in[by])
                    + threadgroup_atomic_float_load(&hist_out[by]);
                if (py_count > 0.0f) {
                    nonzero_y++;
                }
            }
            if (nonzero_y >= 2) {
                float px_in = sum_sw / total_n;
                float px_out = right_w / total_n;
                for (int by = 0; by < target_bins; by++) {
                    float count_in = threadgroup_atomic_float_load(&hist_in[by]);
                    float count_out = threadgroup_atomic_float_load(&hist_out[by]);
                    float y_count = count_in + count_out;
                    if (y_count <= 0.0f) continue;
                    float py = y_count / total_n;
                    if (count_in > 0.0f && px_in > 0.0f) {
                        float pxy = count_in / total_n;
                        mutual_info += pxy * log(pxy / (px_in * py));
                    }
                    if (count_out > 0.0f && px_out > 0.0f) {
                        float pxy = count_out / total_n;
                        mutual_info += pxy * log(pxy / (px_out * py));
                    }
                }
                float bias = float(nonzero_y - 1) / (2.0f * total_n);
                mutual_info = max(mutual_info - bias, 0.0f);
            }
        }

        device float* out = scores_batch + candidate_id * GAFIME_SELECTION_SCORE_SIZE;
        out[GAFIME_SELECTION_MUTUAL_INFO] = mutual_info;
        out[GAFIME_SELECTION_VARIANCE_REDUCTION] = variance_gain;
        out[GAFIME_SELECTION_RESIDUAL_ABS_CORR] = residual_corr;
        out[GAFIME_SELECTION_RESIDUAL_R2_GAIN] = residual_r2;
    }
}

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
