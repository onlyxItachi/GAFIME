/**
 * GAFIME CPU Backend - OpenMP Parallelized
 * 
 * Target: AMD Ryzen AI 9 (multi-core)
 * 
 * Fallback implementation when CUDA is not available.
 * Uses OpenMP for parallel loop execution.
 */

#include "../common/interfaces.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

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
