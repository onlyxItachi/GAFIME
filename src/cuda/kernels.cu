#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_cuda_v1::kernel {

constexpr int kThreadsPerBlock = gafime_cuda_v1::kThreadsPerBlock;
constexpr int kMiThreadsPerBlock = gafime_cuda_v1::kMiThreadsPerBlock;
constexpr int kTopKThreadsPerBlock = gafime_cuda_v1::kTopKThreadsPerBlock;
constexpr int kCudaWarpSize = 32;
constexpr int kMiWarpsPerBlock = (kMiThreadsPerBlock + kCudaWarpSize - 1) / kCudaWarpSize;
constexpr uint32_t kMaxMutualInfoBins = gafime_cuda_v1::kMaxMutualInfoBins;

#define GAFIME_CUDA_FORCEINLINE __device__ __forceinline__

GAFIME_CUDA_FORCEINLINE float nonfinite_metric() {
    return __uint_as_float(0x7fc00000u);
}

template <typename Float>
GAFIME_CUDA_FORCEINLINE float finalize_correlation(Float variance_x, Float variance_y, Float covariance) {
    if (!isfinite(variance_x) || !isfinite(variance_y) || !isfinite(covariance)) {
        return nonfinite_metric();
    }
    if (variance_x == static_cast<Float>(0) || variance_y == static_cast<Float>(0)) {
        return 0.0f;
    }
    if (variance_x < static_cast<Float>(0) || variance_y < static_cast<Float>(0)) {
        return nonfinite_metric();
    }
    const Float denom = sqrt(variance_x * variance_y);
    if (!isfinite(denom) || denom <= static_cast<Float>(0)) {
        return nonfinite_metric();
    }
    const Float correlation = covariance / denom;
    if (!isfinite(correlation)) {
        return nonfinite_metric();
    }
    return static_cast<float>(fmin(static_cast<Float>(1), fmax(static_cast<Float>(-1), correlation)));
}

GAFIME_CUDA_FORCEINLINE float finalize_r2(float pearson) {
    if (!isfinite(pearson)) {
        return nonfinite_metric();
    }
    return fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
}

GAFIME_CUDA_FORCEINLINE uint32_t fixed_mi_bin(
    float value,
    float minimum,
    float inverse_span,
    uint32_t bins
) {
    const float scaled = (value - minimum) * inverse_span;
    if (isnan(scaled) || scaled <= 0.0f) {
        return 0;
    }
    const uint32_t max_bin = bins - 1;
    if (!isfinite(scaled) || scaled >= static_cast<float>(max_bin)) {
        return max_bin;
    }
    return static_cast<uint32_t>(scaled);
}

GAFIME_CUDA_FORCEINLINE float warp_reduce_sum_float(float value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

GAFIME_CUDA_FORCEINLINE float warp_reduce_min_float(float value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

GAFIME_CUDA_FORCEINLINE float warp_reduce_max_float(float value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

GAFIME_CUDA_FORCEINLINE unsigned int warp_reduce_sum_uint(unsigned int value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <uint32_t Arity>
GAFIME_CUDA_FORCEINLINE float interaction_value(
    const float* features,
    const float* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo
) {
    static_assert(Arity >= 1 && Arity <= gafime_cuda_v1::kTemplateMaxArity);
    if constexpr (Arity == 1) {
        return features[static_cast<uint64_t>(combo[0]) * rows + row];
    } else {
        float value = 1.0f;
#pragma unroll
        for (uint32_t idx = 0; idx < Arity; ++idx) {
            const uint32_t col = combo[idx];
            value *= features[static_cast<uint64_t>(col) * rows + row] - column_means[col];
        }
        return value;
    }
}

GAFIME_CUDA_FORCEINLINE float interaction_value(
    const float* features,
    const float* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo,
    uint32_t arity
) {
    if (arity == 1) {
        return features[static_cast<uint64_t>(combo[0]) * rows + row];
    }
    float value = 1.0f;
    for (uint32_t idx = 0; idx < arity; ++idx) {
        const uint32_t col = combo[idx];
        value *= features[static_cast<uint64_t>(col) * rows + row] - column_means[col];
    }
    return value;
}

template <uint32_t Arity>
GAFIME_CUDA_FORCEINLINE float selected_interaction_value(
    const float* features,
    const float* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo,
    uint32_t runtime_arity
) {
    if constexpr (Arity == 0) {
        return interaction_value(
            features,
            column_means,
            row,
            rows,
            combo,
            runtime_arity
        );
    } else {
        return interaction_value<Arity>(features, column_means, row, rows, combo);
    }
}

__global__ void target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    TargetStatsDevice* target_stats
) {
    float local_sy = 0.0f;
    float local_n = 0.0f;
    uint64_t local_count = 0;

    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            local_sy += y;
            local_n += 1.0f;
            local_count += 1;
        }
    }

    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ uint64_t count[kThreadsPerBlock];
    __shared__ float mean_y;
    __shared__ float syy[kThreadsPerBlock];

    sy[threadIdx.x] = local_sy;
    sn[threadIdx.x] = local_n;
    count[threadIdx.x] = local_count;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            count[threadIdx.x] += count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean_y = sn[0] > 0.0f ? sy[0] / sn[0] : 0.0f;
    }
    __syncthreads();

    float local_syy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const float dy = y - mean_y;
            local_syy += dy * dy;
        }
    }

    syy[threadIdx.x] = local_syy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            syy[threadIdx.x] += syy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        target_stats->mean_y = mean_y;
        target_stats->syy = syy[0];
        target_stats->finite = count[0] == n_samples ? 1u : 0u;
        target_stats->reserved = 0u;
    }
}

__global__ void unary_feature_stats_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    UnaryFeatureStatsDevice* feature_stats
) {
    const uint32_t col = blockIdx.x;
    if (col >= n_features) {
        return;
    }
    const uint64_t base = static_cast<uint64_t>(col) * n_samples;

    float local_sx = 0.0f;
    float local_n = 0.0f;
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = features[base + row];
        if (isfinite(x)) {
            local_sx += x;
            local_n += 1.0f;
            local_count += 1;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ uint64_t count[kThreadsPerBlock];
    __shared__ float mean_x;
    __shared__ float sxx[kThreadsPerBlock];

    sx[threadIdx.x] = local_sx;
    sn[threadIdx.x] = local_n;
    count[threadIdx.x] = local_count;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            count[threadIdx.x] += count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean_x = sn[0] > 0.0f ? sx[0] / sn[0] : 0.0f;
    }
    __syncthreads();

    float local_sxx = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = features[base + row];
        if (isfinite(x)) {
            const float dx = x - mean_x;
            local_sxx += dx * dx;
        }
    }

    sxx[threadIdx.x] = local_sxx;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        feature_stats[col].mean_x = mean_x;
        feature_stats[col].sxx = sxx[0];
        feature_stats[col].finite = count[0] == n_samples ? 1u : 0u;
        feature_stats[col].reserved = 0u;
    }
}

__global__ void score_continuous_unary_all_finite_chunk_kernel(
    const float* features,
    const float* target,
    const TargetStatsDevice* target_stats,
    const UnaryFeatureStatsDevice* feature_stats,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t col = combo_indices[descriptor_offset + combo_row];
    const uint64_t base = static_cast<uint64_t>(col) * n_samples;
    const float mean_x = feature_stats[col].mean_x;
    const float feature_sxx = feature_stats[col].sxx;
    const float mean_y = target_stats->mean_y;

    __shared__ float sxy[kThreadsPerBlock];

    float local_sxy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = features[base + row];
        const float y = target[row];
        const float dx = x - mean_x;
        const float dy = y - mean_y;
        local_sxy += dx * dy;
    }

    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float pearson = finalize_correlation(feature_sxx, target_stats->syy, sxy[0]);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = finalize_r2(pearson);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

__global__ void score_continuous_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    float local_sx = 0.0f;
    float local_sy = 0.0f;
    float local_n = 0.0f;

    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        float x = interaction_value(features, column_means, row, n_samples, combo, arity);
        float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += x;
            local_sy += y;
            local_n += 1.0f;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ float sxx[kThreadsPerBlock];
    __shared__ float syy[kThreadsPerBlock];
    __shared__ float sxy[kThreadsPerBlock];
    __shared__ float mean_x;
    __shared__ float mean_y;

    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    sn[threadIdx.x] = local_n;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            sn[threadIdx.x] += sn[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (sn[0] > 0.0f) {
            mean_x = sx[0] / sn[0];
            mean_y = sy[0] / sn[0];
        } else {
            mean_x = 0.0f;
            mean_y = 0.0f;
        }
    }
    __syncthreads();

    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        float x = interaction_value(features, column_means, row, n_samples, combo, arity);
        float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            const float dx = x - mean_x;
            const float dy = y - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }

    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float pearson = finalize_correlation(sxx[0], syy[0], sxy[0]);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = finalize_r2(pearson);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

template <uint32_t Arity>
__global__ void score_continuous_chunk_kernel_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;

    float local_sx = 0.0f;
    float local_sy = 0.0f;
    float local_n = 0.0f;

    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += x;
            local_sy += y;
            local_n += 1.0f;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ float sxx[kThreadsPerBlock];
    __shared__ float syy[kThreadsPerBlock];
    __shared__ float sxy[kThreadsPerBlock];
    __shared__ float mean_x;
    __shared__ float mean_y;

    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    sn[threadIdx.x] = local_n;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            sn[threadIdx.x] += sn[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (sn[0] > 0.0f) {
            mean_x = sx[0] / sn[0];
            mean_y = sy[0] / sn[0];
        } else {
            mean_x = 0.0f;
            mean_y = 0.0f;
        }
    }
    __syncthreads();

    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            const float dx = x - mean_x;
            const float dy = y - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }

    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float pearson = finalize_correlation(sxx[0], syy[0], sxy[0]);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = finalize_r2(pearson);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

template <uint32_t Arity>
__global__ void score_continuous_scaled_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t runtime_arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t arity = Arity != 0 ? Arity : runtime_arity;
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    float local_scale_x = 0.0f;
    float local_scale_y = 0.0f;
    float local_n = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = selected_interaction_value<Arity>(
            features, column_means, row, n_samples, combo, runtime_arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_scale_x = fmaxf(local_scale_x, fabsf(x));
            local_scale_y = fmaxf(local_scale_y, fabsf(y));
            local_n += 1.0f;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ float sxx[kThreadsPerBlock];
    __shared__ float syy[kThreadsPerBlock];
    __shared__ float sxy[kThreadsPerBlock];
    __shared__ float scale_x;
    __shared__ float scale_y;
    __shared__ float mean_x;
    __shared__ float mean_y;

    sx[threadIdx.x] = local_scale_x;
    sy[threadIdx.x] = local_scale_y;
    sn[threadIdx.x] = local_n;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] = fmaxf(sx[threadIdx.x], sx[threadIdx.x + stride]);
            sy[threadIdx.x] = fmaxf(sy[threadIdx.x], sy[threadIdx.x + stride]);
            sn[threadIdx.x] += sn[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scale_x = sx[0];
        scale_y = sy[0];
    }
    __syncthreads();

    float local_sx = 0.0f;
    float local_sy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = selected_interaction_value<Arity>(
            features, column_means, row, n_samples, combo, runtime_arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += scale_x > 0.0f ? x / scale_x : 0.0f;
            local_sy += scale_y > 0.0f ? y / scale_y : 0.0f;
        }
    }
    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean_x = sn[0] > 0.0f ? sx[0] / sn[0] : 0.0f;
        mean_y = sn[0] > 0.0f ? sy[0] / sn[0] : 0.0f;
    }
    __syncthreads();

    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = selected_interaction_value<Arity>(
            features, column_means, row, n_samples, combo, runtime_arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            const float dx = (scale_x > 0.0f ? x / scale_x : 0.0f) - mean_x;
            const float dy = (scale_y > 0.0f ? y / scale_y : 0.0f) - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }
    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float pearson = finalize_correlation(sxx[0], syy[0], sxy[0]);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = finalize_r2(pearson);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

__global__ void score_mutual_info_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t bins,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    __shared__ float min_x;
    __shared__ float max_x;
    __shared__ float min_y;
    __shared__ float max_y;
    __shared__ unsigned int hist_x[kMaxMutualInfoBins];
    __shared__ unsigned int hist_y[kMaxMutualInfoBins];
    __shared__ unsigned int joint[kMaxMutualInfoBins * kMaxMutualInfoBins];
    __shared__ unsigned int valid_count;

    if (bins == 0 || bins > kMaxMutualInfoBins) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = 0.0f;
        }
        return;
    }

    if (threadIdx.x == 0) {
        min_x = INFINITY;
        max_x = -INFINITY;
        min_y = INFINITY;
        max_y = -INFINITY;
        valid_count = 0;
        for (uint64_t row = 0; row < n_samples; ++row) {
            const float x = interaction_value(features, column_means, row, n_samples, combo, arity);
            const float y = target[row];
            if (isfinite(x) && isfinite(y)) {
                min_x = fminf(min_x, x);
                max_x = fmaxf(max_x, x);
                min_y = fminf(min_y, y);
                max_y = fmaxf(max_y, y);
                ++valid_count;
            }
        }
    }
    for (uint32_t idx = threadIdx.x; idx < bins; idx += blockDim.x) {
        hist_x[idx] = 0;
        hist_y[idx] = 0;
    }
    for (uint32_t idx = threadIdx.x; idx < bins * bins; idx += blockDim.x) {
        joint[idx] = 0;
    }
    __syncthreads();

    if (valid_count <= 1 || max_x <= min_x || max_y <= min_y) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = 0.0f;
        }
        return;
    }

    const float inv_x = static_cast<float>(bins) / (max_x - min_x);
    const float inv_y = static_cast<float>(bins) / (max_y - min_y);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value(features, column_means, row, n_samples, combo, arity);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        const uint32_t xb = fixed_mi_bin(x, min_x, inv_x, bins);
        const uint32_t yb = fixed_mi_bin(y, min_y, inv_y, bins);
        atomicAdd(&hist_x[xb], 1);
        atomicAdd(&hist_y[yb], 1);
        atomicAdd(&joint[xb * bins + yb], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const float total = static_cast<float>(valid_count);
        float mi = 0.0f;
        uint32_t active_x = 0;
        uint32_t active_y = 0;
        for (uint32_t xb = 0; xb < bins; ++xb) {
            if (hist_x[xb] == 0) {
                continue;
            }
            ++active_x;
            const float px = static_cast<float>(hist_x[xb]) / total;
            for (uint32_t yb = 0; yb < bins; ++yb) {
                const unsigned int count = joint[xb * bins + yb];
                if (count == 0 || hist_y[yb] == 0) {
                    continue;
                }
                const float py = static_cast<float>(hist_y[yb]) / total;
                const float pxy = static_cast<float>(count) / total;
                mi += pxy * logf(pxy / (px * py));
            }
        }
        for (uint32_t yb = 0; yb < bins; ++yb) {
            if (hist_y[yb] != 0) {
                ++active_y;
            }
        }
        const float correction = active_x > 0 && active_y > 0
            ? static_cast<float>((active_x - 1) * (active_y - 1)) / (2.0f * total)
            : 0.0f;
        const float corrected = fmaxf(0.0f, mi - correction);
        const uint32_t normalizer_bins = min(active_x, active_y);
        const float normalizer = normalizer_bins > 1
            ? logf(static_cast<float>(normalizer_bins))
            : 0.0f;
        metric_values[combo_row * metric_count + metric_index] =
            normalizer > 0.0f ? corrected / normalizer : 0.0f;
    }
}

template <uint32_t Arity, uint32_t Bins>
__global__ void score_mutual_info_chunk_kernel_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values
) {
    static_assert(Bins >= 2 && Bins <= kMaxMutualInfoBins);
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;

    __shared__ float min_x;
    __shared__ float max_x;
    __shared__ float min_y;
    __shared__ float max_y;
    __shared__ unsigned int hist_x[Bins];
    __shared__ unsigned int hist_y[Bins];
    __shared__ unsigned int joint[Bins * Bins];
    __shared__ unsigned int valid_count;

    float local_min_x = INFINITY;
    float local_max_x = -INFINITY;
    float local_min_y = INFINITY;
    float local_max_y = -INFINITY;
    unsigned int local_valid_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_min_x = fminf(local_min_x, x);
            local_max_x = fmaxf(local_max_x, x);
            local_min_y = fminf(local_min_y, y);
            local_max_y = fmaxf(local_max_y, y);
            ++local_valid_count;
        }
    }

    __shared__ float warp_float0[kMiWarpsPerBlock];
    __shared__ float warp_float1[kMiWarpsPerBlock];
    __shared__ float warp_float2[kMiWarpsPerBlock];
    __shared__ float warp_float3[kMiWarpsPerBlock];
    __shared__ unsigned int warp_uint0[kMiWarpsPerBlock];
    __shared__ unsigned int warp_uint1[kMiWarpsPerBlock];

    local_min_x = warp_reduce_min_float(local_min_x);
    local_max_x = warp_reduce_max_float(local_max_x);
    local_min_y = warp_reduce_min_float(local_min_y);
    local_max_y = warp_reduce_max_float(local_max_y);
    local_valid_count = warp_reduce_sum_uint(local_valid_count);

    const uint32_t lane = threadIdx.x & (kCudaWarpSize - 1);
    const uint32_t warp = threadIdx.x / kCudaWarpSize;
    const uint32_t warp_count = (blockDim.x + kCudaWarpSize - 1) / kCudaWarpSize;
    if (lane == 0) {
        warp_float0[warp] = local_min_x;
        warp_float1[warp] = local_max_x;
        warp_float2[warp] = local_min_y;
        warp_float3[warp] = local_max_y;
        warp_uint0[warp] = local_valid_count;
    }
    __syncthreads();

    float block_min_x = INFINITY;
    float block_max_x = -INFINITY;
    float block_min_y = INFINITY;
    float block_max_y = -INFINITY;
    unsigned int block_valid_count = 0;
    if (threadIdx.x < warp_count) {
        block_min_x = warp_float0[lane];
        block_max_x = warp_float1[lane];
        block_min_y = warp_float2[lane];
        block_max_y = warp_float3[lane];
        block_valid_count = warp_uint0[lane];
    }
    if (warp == 0) {
        block_min_x = warp_reduce_min_float(block_min_x);
        block_max_x = warp_reduce_max_float(block_max_x);
        block_min_y = warp_reduce_min_float(block_min_y);
        block_max_y = warp_reduce_max_float(block_max_y);
        block_valid_count = warp_reduce_sum_uint(block_valid_count);
        if (lane == 0) {
            min_x = block_min_x;
            max_x = block_max_x;
            min_y = block_min_y;
            max_y = block_max_y;
            valid_count = block_valid_count;
        }
    }
    for (uint32_t idx = threadIdx.x; idx < Bins; idx += blockDim.x) {
        hist_x[idx] = 0;
        hist_y[idx] = 0;
    }
    for (uint32_t idx = threadIdx.x; idx < Bins * Bins; idx += blockDim.x) {
        joint[idx] = 0;
    }
    __syncthreads();

    if (valid_count <= 1 || max_x <= min_x || max_y <= min_y) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = 0.0f;
        }
        return;
    }

    const float inv_x = static_cast<float>(Bins) / (max_x - min_x);
    const float inv_y = static_cast<float>(Bins) / (max_y - min_y);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        const uint32_t xb = fixed_mi_bin(x, min_x, inv_x, Bins);
        const uint32_t yb = fixed_mi_bin(y, min_y, inv_y, Bins);
        atomicAdd(&hist_x[xb], 1);
        atomicAdd(&hist_y[yb], 1);
        atomicAdd(&joint[xb * Bins + yb], 1);
    }
    __syncthreads();

    if constexpr (Bins <= 4) {
        if (threadIdx.x == 0) {
            const float total = static_cast<float>(valid_count);
            float mi = 0.0f;
            uint32_t active_x = 0;
            uint32_t active_y = 0;
#pragma unroll
            for (uint32_t xb = 0; xb < Bins; ++xb) {
                if (hist_x[xb] == 0) {
                    continue;
                }
                ++active_x;
                const float px = static_cast<float>(hist_x[xb]) / total;
#pragma unroll
                for (uint32_t yb = 0; yb < Bins; ++yb) {
                    const unsigned int count = joint[xb * Bins + yb];
                    if (count == 0 || hist_y[yb] == 0) {
                        continue;
                    }
                    const float py = static_cast<float>(hist_y[yb]) / total;
                    const float pxy = static_cast<float>(count) / total;
                    mi += pxy * logf(pxy / (px * py));
                }
            }
#pragma unroll
            for (uint32_t yb = 0; yb < Bins; ++yb) {
                if (hist_y[yb] != 0) {
                    ++active_y;
                }
            }
            const float correction = active_x > 0 && active_y > 0
                ? static_cast<float>((active_x - 1) * (active_y - 1)) / (2.0f * total)
                : 0.0f;
            const float corrected = fmaxf(0.0f, mi - correction);
            const uint32_t normalizer_bins = min(active_x, active_y);
            const float normalizer = normalizer_bins > 1
                ? logf(static_cast<float>(normalizer_bins))
                : 0.0f;
            metric_values[combo_row * metric_count + metric_index] =
                normalizer > 0.0f ? corrected / normalizer : 0.0f;
        }
    } else {
        const float total = static_cast<float>(valid_count);
        float local_mi = 0.0f;
        uint32_t local_active_x = 0;
        uint32_t local_active_y = 0;
        for (uint32_t idx = threadIdx.x; idx < Bins * Bins; idx += blockDim.x) {
            const uint32_t xb = idx / Bins;
            const uint32_t yb = idx - xb * Bins;
            const unsigned int count = joint[idx];
            if (count == 0 || hist_x[xb] == 0 || hist_y[yb] == 0) {
                continue;
            }
            const float px = static_cast<float>(hist_x[xb]) / total;
            const float py = static_cast<float>(hist_y[yb]) / total;
            const float pxy = static_cast<float>(count) / total;
            local_mi += pxy * logf(pxy / (px * py));
        }
        for (uint32_t xb = threadIdx.x; xb < Bins; xb += blockDim.x) {
            if (hist_x[xb] != 0) {
                ++local_active_x;
            }
        }
        for (uint32_t yb = threadIdx.x; yb < Bins; yb += blockDim.x) {
            if (hist_y[yb] != 0) {
                ++local_active_y;
            }
        }

        local_mi = warp_reduce_sum_float(local_mi);
        local_active_x = warp_reduce_sum_uint(local_active_x);
        local_active_y = warp_reduce_sum_uint(local_active_y);
        if (lane == 0) {
            warp_float0[warp] = local_mi;
            warp_uint0[warp] = local_active_x;
            warp_uint1[warp] = local_active_y;
        }
        __syncthreads();

        float block_mi = 0.0f;
        uint32_t block_active_x = 0;
        uint32_t block_active_y = 0;
        if (threadIdx.x < warp_count) {
            block_mi = warp_float0[lane];
            block_active_x = warp_uint0[lane];
            block_active_y = warp_uint1[lane];
        }
        if (warp == 0) {
            block_mi = warp_reduce_sum_float(block_mi);
            block_active_x = warp_reduce_sum_uint(block_active_x);
            block_active_y = warp_reduce_sum_uint(block_active_y);
        }

        if (threadIdx.x == 0) {
            const uint32_t active_x = block_active_x;
            const uint32_t active_y = block_active_y;
            const float correction = active_x > 0 && active_y > 0
                ? static_cast<float>((active_x - 1) * (active_y - 1)) / (2.0f * total)
                : 0.0f;
            const float corrected = fmaxf(0.0f, block_mi - correction);
            const uint32_t normalizer_bins = min(active_x, active_y);
            const float normalizer = normalizer_bins > 1
                ? logf(static_cast<float>(normalizer_bins))
                : 0.0f;
            metric_values[combo_row * metric_count + metric_index] =
                normalizer > 0.0f ? corrected / normalizer : 0.0f;
        }
    }
}

// Spearman = Pearson on ranks. Ranks are computed by counting (rank_i = #less +
// 0.5*(#equal - 1), average-tie ranks over the finite pairs) so it matches the
// CPU rankdata exactly; the pearson-of-ranks is accumulated in f64 for stability.
// O(n^2) per candidate (correctness-first; a sort-based fast path is a follow-on).
__global__ void score_spearman_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    double l_srx = 0.0, l_sry = 0.0, l_srxx = 0.0, l_sryy = 0.0, l_srxy = 0.0, l_n = 0.0;
    for (uint64_t i = threadIdx.x; i < n_samples; i += blockDim.x) {
        const float xi = interaction_value(features, column_means, i, n_samples, combo, arity);
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        double less_x = 0.0, eq_x = 0.0, less_y = 0.0, eq_y = 0.0;
        for (uint64_t j = 0; j < n_samples; ++j) {
            const float xj = interaction_value(features, column_means, j, n_samples, combo, arity);
            const float yj = target[j];
            if (!isfinite(xj) || !isfinite(yj)) {
                continue;
            }
            if (xj < xi) {
                less_x += 1.0;
            } else if (xj == xi) {
                eq_x += 1.0;
            }
            if (yj < yi) {
                less_y += 1.0;
            } else if (yj == yi) {
                eq_y += 1.0;
            }
        }
        const double rx = less_x + 0.5 * (eq_x - 1.0);
        const double ry = less_y + 0.5 * (eq_y - 1.0);
        l_srx += rx;
        l_sry += ry;
        l_srxx += rx * rx;
        l_sryy += ry * ry;
        l_srxy += rx * ry;
        l_n += 1.0;
    }

    __shared__ double s_srx[kThreadsPerBlock];
    __shared__ double s_sry[kThreadsPerBlock];
    __shared__ double s_srxx[kThreadsPerBlock];
    __shared__ double s_sryy[kThreadsPerBlock];
    __shared__ double s_srxy[kThreadsPerBlock];
    __shared__ double s_n[kThreadsPerBlock];
    s_srx[threadIdx.x] = l_srx;
    s_sry[threadIdx.x] = l_sry;
    s_srxx[threadIdx.x] = l_srxx;
    s_sryy[threadIdx.x] = l_sryy;
    s_srxy[threadIdx.x] = l_srxy;
    s_n[threadIdx.x] = l_n;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_srx[threadIdx.x] += s_srx[threadIdx.x + stride];
            s_sry[threadIdx.x] += s_sry[threadIdx.x + stride];
            s_srxx[threadIdx.x] += s_srxx[threadIdx.x + stride];
            s_sryy[threadIdx.x] += s_sryy[threadIdx.x + stride];
            s_srxy[threadIdx.x] += s_srxy[threadIdx.x + stride];
            s_n[threadIdx.x] += s_n[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double n = s_n[0];
        float out = 0.0f;
        if (n > 1.0) {
            const double cov = n * s_srxy[0] - s_srx[0] * s_sry[0];
            const double vx = n * s_srxx[0] - s_srx[0] * s_srx[0];
            const double vy = n * s_sryy[0] - s_sry[0] * s_sry[0];
            out = finalize_correlation(vx, vy, cov);
        }
        metric_values[combo_row * metric_count + metric_index] = out;
    }
}

// The finite-unary path reuses these exact count-based target ranks across a
// batch. It deliberately does not sort: ties retain the same average-rank
// calculation as the pairwise fallback.
__global__ void build_spearman_target_ranks_kernel(
    const float* target,
    uint64_t n_samples,
    double* target_ranks
) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = first; i < n_samples; i += stride) {
        const float yi = target[i];
        if (!isfinite(yi)) {
            target_ranks[i] = 0.0;
            continue;
        }
        double less_y = 0.0;
        double eq_y = 0.0;
        for (uint64_t j = 0; j < n_samples; ++j) {
            const float yj = target[j];
            if (!isfinite(yj)) {
                continue;
            }
            if (yj < yi) {
                less_y += 1.0;
            } else if (yj == yi) {
                eq_y += 1.0;
            }
        }
        target_ranks[i] = less_y + 0.5 * (eq_y - 1.0);
    }
}

__global__ void score_spearman_unary_cached_target_ranks_kernel(
    const float* features,
    const float* target,
    const double* target_ranks,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t column = combo_indices[descriptor_offset + combo_row];
    const float* x_values = features + static_cast<uint64_t>(column) * n_samples;

    double l_srx = 0.0, l_sry = 0.0, l_srxx = 0.0, l_sryy = 0.0, l_srxy = 0.0, l_n = 0.0;
    for (uint64_t i = threadIdx.x; i < n_samples; i += blockDim.x) {
        const float xi = x_values[i];
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        double less_x = 0.0;
        double eq_x = 0.0;
        for (uint64_t j = 0; j < n_samples; ++j) {
            const float xj = x_values[j];
            if (!isfinite(xj)) {
                continue;
            }
            if (xj < xi) {
                less_x += 1.0;
            } else if (xj == xi) {
                eq_x += 1.0;
            }
        }
        const double rx = less_x + 0.5 * (eq_x - 1.0);
        const double ry = target_ranks[i];
        l_srx += rx;
        l_sry += ry;
        l_srxx += rx * rx;
        l_sryy += ry * ry;
        l_srxy += rx * ry;
        l_n += 1.0;
    }

    __shared__ double s_srx[kThreadsPerBlock];
    __shared__ double s_sry[kThreadsPerBlock];
    __shared__ double s_srxx[kThreadsPerBlock];
    __shared__ double s_sryy[kThreadsPerBlock];
    __shared__ double s_srxy[kThreadsPerBlock];
    __shared__ double s_n[kThreadsPerBlock];
    s_srx[threadIdx.x] = l_srx;
    s_sry[threadIdx.x] = l_sry;
    s_srxx[threadIdx.x] = l_srxx;
    s_sryy[threadIdx.x] = l_sryy;
    s_srxy[threadIdx.x] = l_srxy;
    s_n[threadIdx.x] = l_n;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_srx[threadIdx.x] += s_srx[threadIdx.x + stride];
            s_sry[threadIdx.x] += s_sry[threadIdx.x + stride];
            s_srxx[threadIdx.x] += s_srxx[threadIdx.x + stride];
            s_sryy[threadIdx.x] += s_sryy[threadIdx.x + stride];
            s_srxy[threadIdx.x] += s_srxy[threadIdx.x + stride];
            s_n[threadIdx.x] += s_n[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double n = s_n[0];
        float out = 0.0f;
        if (n > 1.0) {
            const double cov = n * s_srxy[0] - s_srx[0] * s_sry[0];
            const double vx = n * s_srxx[0] - s_srx[0] * s_srx[0];
            const double vy = n * s_sryy[0] - s_sry[0] * s_sry[0];
            out = finalize_correlation(vx, vy, cov);
        }
        metric_values[combo_row * metric_count + metric_index] = out;
    }
}

template <uint32_t Arity>
__global__ void score_spearman_chunk_kernel_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;

    double l_srx = 0.0, l_sry = 0.0, l_srxx = 0.0, l_sryy = 0.0, l_srxy = 0.0, l_n = 0.0;
    for (uint64_t i = threadIdx.x; i < n_samples; i += blockDim.x) {
        const float xi = interaction_value<Arity>(features, column_means, i, n_samples, combo);
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        double less_x = 0.0, eq_x = 0.0, less_y = 0.0, eq_y = 0.0;
        for (uint64_t j = 0; j < n_samples; ++j) {
            const float xj = interaction_value<Arity>(features, column_means, j, n_samples, combo);
            const float yj = target[j];
            if (!isfinite(xj) || !isfinite(yj)) {
                continue;
            }
            if (xj < xi) {
                less_x += 1.0;
            } else if (xj == xi) {
                eq_x += 1.0;
            }
            if (yj < yi) {
                less_y += 1.0;
            } else if (yj == yi) {
                eq_y += 1.0;
            }
        }
        const double rx = less_x + 0.5 * (eq_x - 1.0);
        const double ry = less_y + 0.5 * (eq_y - 1.0);
        l_srx += rx;
        l_sry += ry;
        l_srxx += rx * rx;
        l_sryy += ry * ry;
        l_srxy += rx * ry;
        l_n += 1.0;
    }

    __shared__ double s_srx[kThreadsPerBlock];
    __shared__ double s_sry[kThreadsPerBlock];
    __shared__ double s_srxx[kThreadsPerBlock];
    __shared__ double s_sryy[kThreadsPerBlock];
    __shared__ double s_srxy[kThreadsPerBlock];
    __shared__ double s_n[kThreadsPerBlock];
    s_srx[threadIdx.x] = l_srx;
    s_sry[threadIdx.x] = l_sry;
    s_srxx[threadIdx.x] = l_srxx;
    s_sryy[threadIdx.x] = l_sryy;
    s_srxy[threadIdx.x] = l_srxy;
    s_n[threadIdx.x] = l_n;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_srx[threadIdx.x] += s_srx[threadIdx.x + stride];
            s_sry[threadIdx.x] += s_sry[threadIdx.x + stride];
            s_srxx[threadIdx.x] += s_srxx[threadIdx.x + stride];
            s_sryy[threadIdx.x] += s_sryy[threadIdx.x + stride];
            s_srxy[threadIdx.x] += s_srxy[threadIdx.x + stride];
            s_n[threadIdx.x] += s_n[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double n = s_n[0];
        float out = 0.0f;
        if (n > 1.0) {
            const double cov = n * s_srxy[0] - s_srx[0] * s_sry[0];
            const double vx = n * s_srxx[0] - s_srx[0] * s_srx[0];
            const double vy = n * s_sryy[0] - s_sry[0] * s_sry[0];
            out = finalize_correlation(vx, vy, cov);
        }
        metric_values[combo_row * metric_count + metric_index] = out;
    }
}

template <bool Descending>
GAFIME_CUDA_FORCEINLINE bool candidate_better_static(
    float candidate_score,
    uint32_t candidate_index,
    float best_score,
    uint32_t best_index
) {
    if (!isfinite(candidate_score)) {
        return false;
    }
    if (best_index == UINT32_MAX) {
        return true;
    }
    if constexpr (Descending) {
        if (candidate_score > best_score) {
            return true;
        }
        if (candidate_score < best_score) {
            return false;
        }
    } else {
        if (candidate_score < best_score) {
            return true;
        }
        if (candidate_score > best_score) {
            return false;
        }
    }
    return candidate_index < best_index;
}

template <bool Descending>
__global__ void select_topk_partials_kernel_static(
    const float* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    float* partial_scores,
    uint32_t* partial_indices
) {
    __shared__ float best_scores[kTopKThreadsPerBlock];
    __shared__ uint32_t best_indices[kTopKThreadsPerBlock];
    __shared__ float previous_score;
    __shared__ uint32_t previous_index;

    const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) * top_k;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        float local_score = Descending ? -INFINITY : INFINITY;
        uint32_t local_index = UINT32_MAX;
        for (uint64_t row = start; row < row_count; row += stride) {
            const uint32_t row_index = static_cast<uint32_t>(row);
            const float score = metric_values[row * metric_count + primary_metric_index];
            if (rank != 0 && !candidate_better_static<Descending>(
                    previous_score,
                    previous_index,
                    score,
                    row_index)) {
                continue;
            }
            if (candidate_better_static<Descending>(
                    score,
                    row_index,
                    local_score,
                    local_index)) {
                local_score = score;
                local_index = row_index;
            }
        }

        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();

        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride) {
                const float score = best_scores[threadIdx.x + reduce_stride];
                const uint32_t index = best_indices[threadIdx.x + reduce_stride];
                if (candidate_better_static<Descending>(
                        score,
                        index,
                        best_scores[threadIdx.x],
                        best_indices[threadIdx.x])) {
                    best_scores[threadIdx.x] = score;
                    best_indices[threadIdx.x] = index;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            partial_scores[block_base + rank] = best_scores[0];
            partial_indices[block_base + rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
            __threadfence_block();
        }
        __syncthreads();
    }
}

template <bool Descending>
__global__ void merge_topk_partials_kernel_static(
    const float* partial_scores,
    const uint32_t* partial_indices,
    uint64_t partial_count,
    uint32_t top_k,
    uint32_t* selected_indices
) {
    __shared__ float best_scores[kTopKThreadsPerBlock];
    __shared__ uint32_t best_indices[kTopKThreadsPerBlock];
    __shared__ float previous_score;
    __shared__ uint32_t previous_index;

    for (uint32_t rank = 0; rank < top_k; ++rank) {
        float local_score = Descending ? -INFINITY : INFINITY;
        uint32_t local_index = UINT32_MAX;
        for (uint64_t item = threadIdx.x; item < partial_count; item += blockDim.x) {
            const uint32_t candidate_index = partial_indices[item];
            if (candidate_index == UINT32_MAX) {
                continue;
            }
            const float score = partial_scores[item];
            if (rank != 0 && !candidate_better_static<Descending>(
                    previous_score,
                    previous_index,
                    score,
                    candidate_index)) {
                continue;
            }
            if (candidate_better_static<Descending>(score, candidate_index, local_score, local_index)) {
                local_score = score;
                local_index = candidate_index;
            }
        }

        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();

        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride) {
                const float score = best_scores[threadIdx.x + reduce_stride];
                const uint32_t index = best_indices[threadIdx.x + reduce_stride];
                if (candidate_better_static<Descending>(
                        score,
                        index,
                        best_scores[threadIdx.x],
                        best_indices[threadIdx.x])) {
                    best_scores[threadIdx.x] = score;
                    best_indices[threadIdx.x] = index;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            selected_indices[rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
            __threadfence_block();
        }
        __syncthreads();
    }
}

__global__ void copy_selected_metric_rows_kernel(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (idx >= total) {
        return;
    }
    const uint64_t selected_row = idx / metric_count;
    const uint32_t metric_idx = static_cast<uint32_t>(idx % metric_count);
    const uint32_t source_row = selected_indices[selected_row];
    selected_metric_values[idx] = metric_values[static_cast<uint64_t>(source_row) * metric_count + metric_idx];
}

__device__ float metric_extremeness(uint32_t metric_id, float value) {
    if (!isfinite(value)) {
        return -INFINITY;
    }
    if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_SPEARMAN) {
        return fabsf(value);
    }
    return value;
}

__global__ void selected_metric_max_kernel(
    const float* metric_values,
    uint64_t row_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_max
) {
    const uint32_t metric_idx = blockIdx.x;
    if (metric_idx >= metric_count) {
        return;
    }
    const uint32_t metric_id = metric_ids[metric_idx];
    float local_max = -INFINITY;
    for (uint64_t row = threadIdx.x; row < row_count; row += blockDim.x) {
        const float value = metric_values[row * metric_count + metric_idx];
        local_max = fmaxf(local_max, metric_extremeness(metric_id, value));
    }

    __shared__ float partial[kThreadsPerBlock];
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        metric_max[metric_idx] = partial[0];
    }
}

__global__ void accumulate_exceedances_kernel(
    const float* metric_max,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    const float* observed_metric_values,
    uint64_t selected_count,
    uint32_t* exceedance_counts
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (idx >= total) {
        return;
    }
    const uint32_t metric_idx = static_cast<uint32_t>(idx % metric_count);
    const float observed = metric_extremeness(metric_ids[metric_idx], observed_metric_values[idx]);
    if (metric_max[metric_idx] >= observed) {
        atomicAdd(&exceedance_counts[idx], 1u);
    }
}

#define GAFIME_CUDA_INSTANTIATE_CONTINUOUS(ARITY) \
    template __global__ void score_continuous_chunk_kernel_static<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, const uint32_t*, uint32_t, float*);

#define GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(ARITY) \
    template __global__ void score_continuous_scaled_chunk_kernel<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint32_t, \
        uint64_t, uint64_t, const uint32_t*, uint32_t, float*);

#define GAFIME_CUDA_INSTANTIATE_SPEARMAN(ARITY) \
    template __global__ void score_spearman_chunk_kernel_static<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, uint32_t, uint32_t, float*);

#define GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, BINS) \
    template __global__ void score_mutual_info_chunk_kernel_static<ARITY, BINS>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, uint32_t, uint32_t, float*);

#define GAFIME_CUDA_INSTANTIATE_MI_ARITY(ARITY) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 2) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 4) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 8) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 12) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 16) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 24) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 32) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 48) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 64) \
    GAFIME_CUDA_INSTANTIATE_MI_BIN(ARITY, 96)

GAFIME_CUDA_INSTANTIATE_CONTINUOUS(1)
GAFIME_CUDA_INSTANTIATE_CONTINUOUS(2)
GAFIME_CUDA_INSTANTIATE_CONTINUOUS(3)
GAFIME_CUDA_INSTANTIATE_CONTINUOUS(4)
GAFIME_CUDA_INSTANTIATE_CONTINUOUS(5)

GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(0)
GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(1)
GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(2)
GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(3)
GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(4)
GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS(5)

GAFIME_CUDA_INSTANTIATE_SPEARMAN(1)
GAFIME_CUDA_INSTANTIATE_SPEARMAN(2)
GAFIME_CUDA_INSTANTIATE_SPEARMAN(3)
GAFIME_CUDA_INSTANTIATE_SPEARMAN(4)
GAFIME_CUDA_INSTANTIATE_SPEARMAN(5)

GAFIME_CUDA_INSTANTIATE_MI_ARITY(1)
GAFIME_CUDA_INSTANTIATE_MI_ARITY(2)
GAFIME_CUDA_INSTANTIATE_MI_ARITY(3)
GAFIME_CUDA_INSTANTIATE_MI_ARITY(4)
GAFIME_CUDA_INSTANTIATE_MI_ARITY(5)

template __global__ void select_topk_partials_kernel_static<true>(
    const float*, uint64_t, uint32_t, uint32_t, uint32_t, float*, uint32_t*);
template __global__ void select_topk_partials_kernel_static<false>(
    const float*, uint64_t, uint32_t, uint32_t, uint32_t, float*, uint32_t*);
template __global__ void merge_topk_partials_kernel_static<true>(
    const float*, const uint32_t*, uint64_t, uint32_t, uint32_t*);
template __global__ void merge_topk_partials_kernel_static<false>(
    const float*, const uint32_t*, uint64_t, uint32_t, uint32_t*);

#undef GAFIME_CUDA_INSTANTIATE_CONTINUOUS
#undef GAFIME_CUDA_INSTANTIATE_SCALED_CONTINUOUS
#undef GAFIME_CUDA_INSTANTIATE_SPEARMAN
#undef GAFIME_CUDA_INSTANTIATE_MI_BIN
#undef GAFIME_CUDA_INSTANTIATE_MI_ARITY
#undef GAFIME_CUDA_FORCEINLINE

}  // namespace gafime_cuda_v1::kernel
