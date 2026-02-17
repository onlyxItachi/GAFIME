/**
 * GAFIME CPU Backend - OpenMP Parallelized
 * 
 * Full-featured CPU fallback matching the GPU kernel's fused map-reduce API.
 * Supports all 11 unary operators, 6 interaction types, and produces the
 * same 12-float stats output (train/val split) as the CUDA kernel.
 * 
 * Uses OpenMP for parallel loop execution with reduction.
 */

#include "../common/interfaces.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CPU UNARY OPERATORS (mirror of GPU apply_op)
// ============================================================================

/**
 * Apply unary transformation to a single value.
 * Safe implementations prevent NaN/Inf propagation.
 * Matches the GPU kernel's apply_op exactly.
 */
static inline float apply_op_cpu(float x, int op) {
    switch (op) {
        case GAFIME_OP_LOG:
            return logf(fabsf(x) + 1e-8f);
            
        case GAFIME_OP_EXP:
            return expf(fminf(fmaxf(x, -20.0f), 20.0f));
            
        case GAFIME_OP_SQRT:
            return sqrtf(fabsf(x));
            
        case GAFIME_OP_TANH: {
            float clamped = fminf(fmaxf(x, -10.0f), 10.0f);
            float exp2x = expf(2.0f * clamped);
            return (exp2x - 1.0f) / (exp2x + 1.0f);
        }
            
        case GAFIME_OP_SIGMOID: {
            float clamped = fminf(fmaxf(x, -20.0f), 20.0f);
            float ex = expf(-clamped);
            return 1.0f / (1.0f + ex);
        }
            
        case GAFIME_OP_SQUARE:
            return x * x;
            
        case GAFIME_OP_NEGATE:
            return -x;
            
        case GAFIME_OP_ABS:
            return fabsf(x);
            
        case GAFIME_OP_INVERSE:
            return 1.0f / (fabsf(x) < 1e-8f ? copysignf(1e-8f, x) : x);
            
        case GAFIME_OP_CUBE:
            return x * x * x;
            
        case GAFIME_OP_ROLLING_MEAN:
        case GAFIME_OP_ROLLING_STD:
            // Rolling ops should be done via CPU preprocessing (Polars)
            return NAN;
            
        case GAFIME_OP_IDENTITY:
        default:
            return x;
    }
}

// ============================================================================
// CPU INTERACTION COMBINERS (mirror of GPU combine)
// ============================================================================

/**
 * Combine two values using the specified interaction type.
 * Matches the GPU kernel's combine function exactly.
 */
static inline float combine_cpu(float a, float b, int interact_type) {
    switch (interact_type) {
        case GAFIME_INTERACT_ADD:
            return a + b;
        case GAFIME_INTERACT_SUB:
            return a - b;
        case GAFIME_INTERACT_DIV:
            return a / (fabsf(b) < 1e-8f ? copysignf(1e-8f, b) : b);
        case GAFIME_INTERACT_MAX:
            return fmaxf(a, b);
        case GAFIME_INTERACT_MIN:
            return fminf(a, b);
        case GAFIME_INTERACT_MULT:
        default:
            return a * b;
    }
}

extern "C" {

// ============================================================================
// CPU AVAILABILITY
// ============================================================================

int gafime_cpu_available(void) {
    return 1;
}

// ============================================================================
// FUSED MAP-REDUCE (matches GPU gafime_fused_interaction signature)
// ============================================================================

/**
 * CPU fused map-reduce: Transform features, combine, reduce to stats.
 * 
 * Produces the same 12-float output as the GPU kernel:
 * [train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy,
 *  val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy]
 * 
 * Uses OpenMP parallel reduction for multi-core performance.
 */
int gafime_fused_interaction_cpu(
    const float** h_inputs,     // Array of pointers to feature columns
    const float* h_target,      // Target vector
    const uint8_t* h_mask,      // Fold mask
    const int* h_ops,           // Unary operator IDs per feature
    int arity,                  // Number of features (2-5)
    int interaction_type,       // Interaction combiner type
    int val_fold_id,            // Validation fold ID
    int n_samples,
    float* h_stats              // Output: 12 floats
) {
    if (arity < 2 || arity > 5) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (!h_inputs || !h_target || !h_mask || !h_ops || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Validate all input pointers
    for (int i = 0; i < arity; i++) {
        if (!h_inputs[i]) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
    }
    
    // Use double precision for accumulation to avoid catastrophic cancellation
    // (same reason the standalone CPU Pearson uses double)
    double train_n = 0, train_sx = 0, train_sy = 0;
    double train_sxx = 0, train_syy = 0, train_sxy = 0;
    double val_n = 0, val_sx = 0, val_sy = 0;
    double val_sxx = 0, val_syy = 0, val_sxy = 0;
    
    #pragma omp parallel for reduction(+:train_n,train_sx,train_sy,train_sxx,train_syy,train_sxy,val_n,val_sx,val_sy,val_sxx,val_syy,val_sxy) schedule(static)
    for (int i = 0; i < n_samples; i++) {
        // Apply unary ops to each feature
        float vals[5];
        for (int f = 0; f < arity; f++) {
            vals[f] = apply_op_cpu(h_inputs[f][i], h_ops[f]);
        }
        
        // Combine features using interaction type (left-to-right reduction)
        float x = combine_cpu(vals[0], vals[1], interaction_type);
        for (int f = 2; f < arity; f++) {
            x = combine_cpu(x, vals[f], interaction_type);
        }
        
        float y = h_target[i];
        
        // NaN guard: skip rows with NaN to prevent poisoning
        if (std::isnan(x) || std::isnan(y)) continue;
        
        // Accumulate statistics into train or val based on fold mask
        double xd = static_cast<double>(x);
        double yd = static_cast<double>(y);
        
        if (h_mask[i] == val_fold_id) {
            val_n += 1.0;
            val_sx += xd;
            val_sy += yd;
            val_sxx += xd * xd;
            val_syy += yd * yd;
            val_sxy += xd * yd;
        } else {
            train_n += 1.0;
            train_sx += xd;
            train_sy += yd;
            train_sxx += xd * xd;
            train_syy += yd * yd;
            train_sxy += xd * yd;
        }
    }
    
    // Write output (cast back to float for API compatibility)
    h_stats[GAFIME_STAT_TRAIN_N]   = static_cast<float>(train_n);
    h_stats[GAFIME_STAT_TRAIN_SX]  = static_cast<float>(train_sx);
    h_stats[GAFIME_STAT_TRAIN_SY]  = static_cast<float>(train_sy);
    h_stats[GAFIME_STAT_TRAIN_SXX] = static_cast<float>(train_sxx);
    h_stats[GAFIME_STAT_TRAIN_SYY] = static_cast<float>(train_syy);
    h_stats[GAFIME_STAT_TRAIN_SXY] = static_cast<float>(train_sxy);
    h_stats[GAFIME_STAT_VAL_N]     = static_cast<float>(val_n);
    h_stats[GAFIME_STAT_VAL_SX]    = static_cast<float>(val_sx);
    h_stats[GAFIME_STAT_VAL_SY]    = static_cast<float>(val_sy);
    h_stats[GAFIME_STAT_VAL_SXX]   = static_cast<float>(val_sxx);
    h_stats[GAFIME_STAT_VAL_SYY]   = static_cast<float>(val_syy);
    h_stats[GAFIME_STAT_VAL_SXY]   = static_cast<float>(val_sxy);
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// FUSED MAP-REDUCE WITH PER-PAIR INTERACTION TYPES
// ============================================================================

/**
 * CPU fused map-reduce with per-pair interaction types.
 * Same as above but allows different interaction types between each pair.
 * Matches the GPU bucket_compute API.
 */
int gafime_fused_interaction_perpair_cpu(
    const float** h_inputs,
    const float* h_target,
    const uint8_t* h_mask,
    const int* h_ops,
    int arity,
    const int* interaction_types,   // Per-pair: arity-1 interaction types
    int val_fold_id,
    int n_samples,
    float* h_stats
) {
    if (arity < 2 || arity > 5) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (!h_inputs || !h_target || !h_mask || !h_ops || !interaction_types || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    for (int i = 0; i < arity; i++) {
        if (!h_inputs[i]) return GAFIME_ERROR_INVALID_ARGS;
    }
    
    double train_n = 0, train_sx = 0, train_sy = 0;
    double train_sxx = 0, train_syy = 0, train_sxy = 0;
    double val_n = 0, val_sx = 0, val_sy = 0;
    double val_sxx = 0, val_syy = 0, val_sxy = 0;
    
    #pragma omp parallel for reduction(+:train_n,train_sx,train_sy,train_sxx,train_syy,train_sxy,val_n,val_sx,val_sy,val_sxx,val_syy,val_sxy) schedule(static)
    for (int i = 0; i < n_samples; i++) {
        float vals[5];
        for (int f = 0; f < arity; f++) {
            vals[f] = apply_op_cpu(h_inputs[f][i], h_ops[f]);
        }
        
        // Per-pair interaction: combine(combine(combine(v0, v1, t0), v2, t1), v3, t2)
        float x = combine_cpu(vals[0], vals[1], interaction_types[0]);
        for (int f = 2; f < arity; f++) {
            x = combine_cpu(x, vals[f], interaction_types[f - 1]);
        }
        
        float y = h_target[i];
        
        if (std::isnan(x) || std::isnan(y)) continue;
        
        double xd = static_cast<double>(x);
        double yd = static_cast<double>(y);
        
        if (h_mask[i] == val_fold_id) {
            val_n += 1.0;
            val_sx += xd;
            val_sy += yd;
            val_sxx += xd * xd;
            val_syy += yd * yd;
            val_sxy += xd * yd;
        } else {
            train_n += 1.0;
            train_sx += xd;
            train_sy += yd;
            train_sxx += xd * xd;
            train_syy += yd * yd;
            train_sxy += xd * yd;
        }
    }
    
    h_stats[GAFIME_STAT_TRAIN_N]   = static_cast<float>(train_n);
    h_stats[GAFIME_STAT_TRAIN_SX]  = static_cast<float>(train_sx);
    h_stats[GAFIME_STAT_TRAIN_SY]  = static_cast<float>(train_sy);
    h_stats[GAFIME_STAT_TRAIN_SXX] = static_cast<float>(train_sxx);
    h_stats[GAFIME_STAT_TRAIN_SYY] = static_cast<float>(train_syy);
    h_stats[GAFIME_STAT_TRAIN_SXY] = static_cast<float>(train_sxy);
    h_stats[GAFIME_STAT_VAL_N]     = static_cast<float>(val_n);
    h_stats[GAFIME_STAT_VAL_SX]    = static_cast<float>(val_sx);
    h_stats[GAFIME_STAT_VAL_SY]    = static_cast<float>(val_sy);
    h_stats[GAFIME_STAT_VAL_SXX]   = static_cast<float>(val_sxx);
    h_stats[GAFIME_STAT_VAL_SYY]   = static_cast<float>(val_syy);
    h_stats[GAFIME_STAT_VAL_SXY]   = static_cast<float>(val_sxy);
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// LEGACY API (unchanged)
// ============================================================================

int gafime_feature_interaction_cpu(
    const float* X,
    const float* means,
    float* output,
    const int32_t* combo_indices,
    const int32_t* combo_offsets,
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
) {
    if (!X || !means || !output || !combo_indices || !combo_offsets) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_combos <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int32_t sample = 0; sample < n_samples; sample++) {
        for (int32_t combo = 0; combo < n_combos; combo++) {
            int32_t start = combo_offsets[combo];
            int32_t end = combo_offsets[combo + 1];
            int32_t combo_size = end - start;
            
            int32_t out_idx = sample * n_combos + combo;
            
            if (combo_size <= 0) {
                output[out_idx] = 0.0f;
                continue;
            }
            
            // Single feature: return raw value
            if (combo_size == 1) {
                int32_t feat = combo_indices[start];
                if (feat >= 0 && feat < n_features) {
                    output[out_idx] = X[sample * n_features + feat];
                } else {
                    output[out_idx] = 0.0f;
                }
                continue;
            }
            
            // Multi-feature: compute centered product
            float prod = 1.0f;
            for (int32_t j = start; j < end; j++) {
                int32_t feat = combo_indices[j];
                if (feat >= 0 && feat < n_features) {
                    float centered = X[sample * n_features + feat] - means[feat];
                    prod *= centered;
                }
            }
            output[out_idx] = prod;
        }
    }
    
    return GAFIME_SUCCESS;
}

int gafime_pearson_cpu(
    const float* x,
    const float* y,
    int32_t n,
    float* result_out
) {
    if (!x || !y || !result_out || n <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    double sum_x = 0.0, sum_y = 0.0;
    double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0;
    
    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_xx,sum_yy,sum_xy)
    for (int32_t i = 0; i < n; i++) {
        double xi = static_cast<double>(x[i]);
        double yi = static_cast<double>(y[i]);
        sum_x += xi;
        sum_y += yi;
        sum_xx += xi * xi;
        sum_yy += yi * yi;
        sum_xy += xi * yi;
    }
    
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double var_x = sum_xx - sum_x * mean_x;
    double var_y = sum_yy - sum_y * mean_y;
    double cov_xy = sum_xy - sum_x * mean_y;
    
    if (var_x <= 0.0 || var_y <= 0.0) {
        *result_out = 0.0f;
    } else {
        *result_out = static_cast<float>(cov_xy / std::sqrt(var_x * var_y));
    }
    
    return GAFIME_SUCCESS;
}

/**
 * Compute feature means (CPU version)
 */
int gafime_compute_means_cpu(
    const float* X,
    float* means,
    int32_t n_samples,
    int32_t n_features
) {
    if (!X || !means || n_samples <= 0 || n_features <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    #pragma omp parallel for
    for (int32_t feat = 0; feat < n_features; feat++) {
        double sum = 0.0;
        for (int32_t i = 0; i < n_samples; i++) {
            sum += static_cast<double>(X[i * n_features + feat]);
        }
        means[feat] = static_cast<float>(sum / n_samples);
    }
    
    return GAFIME_SUCCESS;
}

} // extern "C"
