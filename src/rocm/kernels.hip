#include "kernels.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <type_traits>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_rocm_v1::kernel {

constexpr int kThreadsPerBlock = gafime_rocm_v1::kThreadsPerBlock;
constexpr int kMiThreadsPerBlock = gafime_rocm_v1::kMiThreadsPerBlock;
constexpr int kTopKThreadsPerBlock = gafime_rocm_v1::kTopKThreadsPerBlock;
constexpr int kHipMinWavefrontSize = 32;
constexpr int kMiMaxWavesPerBlock =
    (kMiThreadsPerBlock + kHipMinWavefrontSize - 1) / kHipMinWavefrontSize;
#ifndef GAFIME_HIP_WAVE_MI_MASK
#define GAFIME_HIP_WAVE_MI_MASK 1
#endif
#ifndef GAFIME_HIP_TARGET_ARCH_TAG
#define GAFIME_HIP_TARGET_ARCH_TAG "unknown"
#endif
#ifndef GAFIME_GPU_MI_ACCUMULATION_FP64
#define GAFIME_GPU_MI_ACCUMULATION_FP64 0
#endif
#ifndef GAFIME_HIP_PRECISION_PROFILE_MASK
#define GAFIME_HIP_PRECISION_PROFILE_MASK 7
#endif
#define GAFIME_HIP_STRINGIFY_INNER(value) #value
#define GAFIME_HIP_STRINGIFY(value) GAFIME_HIP_STRINGIFY_INNER(value)

[[gnu::used, gnu::section(".gafime.build")]] static const char kGafimeRocmBuildInfo[] =
    "GAFIME_ROCM_BUILD_INFO:arch=" GAFIME_HIP_TARGET_ARCH_TAG
    ";wave_mi_mask=" GAFIME_HIP_STRINGIFY(GAFIME_HIP_WAVE_MI_MASK)
    ";mi_accumulation_fp64=" GAFIME_HIP_STRINGIFY(GAFIME_GPU_MI_ACCUMULATION_FP64)
    ";precision_profile_mask=" GAFIME_HIP_STRINGIFY(GAFIME_HIP_PRECISION_PROFILE_MASK);
constexpr uint32_t kHipWaveMiMask = GAFIME_HIP_WAVE_MI_MASK;
static_assert((kHipWaveMiMask & ~3u) == 0u);
static_assert(
    GAFIME_GPU_MI_ACCUMULATION_FP64 == 0 || GAFIME_GPU_MI_ACCUMULATION_FP64 == 1
);
static_assert(GAFIME_HIP_PRECISION_PROFILE_MASK == 7);
template <uint32_t Bins>
constexpr bool kUseHipWaveMi =
    (Bins == 64 && (kHipWaveMiMask & 1u) != 0u) ||
    (Bins == 96 && (kHipWaveMiMask & 2u) != 0u);
constexpr uint32_t kMaxMutualInfoBins = gafime_rocm_v1::kMaxMutualInfoBins;

#if GAFIME_GPU_MI_ACCUMULATION_FP64
using MutualInfoAccumulator = double;
#else
using MutualInfoAccumulator = float;
#endif

#define GAFIME_HIP_FORCEINLINE __device__ __forceinline__

GAFIME_HIP_FORCEINLINE float nonfinite_metric() {
    return __uint_as_float(0x7fc00000u);
}

template <typename Float>
GAFIME_HIP_FORCEINLINE float finalize_correlation(Float variance_x, Float variance_y, Float covariance) {
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

GAFIME_HIP_FORCEINLINE float finalize_r2(float pearson) {
    if (!isfinite(pearson)) {
        return nonfinite_metric();
    }
    return fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
}

GAFIME_HIP_FORCEINLINE uint32_t fixed_mi_bin(
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

struct MutualInfoProbabilityContext {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    double total;
    double inverse_total;
#else
    float total;
#endif
};

GAFIME_HIP_FORCEINLINE MutualInfoProbabilityContext mutual_info_probability_context(
    unsigned int total
) {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    const double probability_total = static_cast<double>(total);
    return {probability_total, 1.0 / probability_total};
#else
    return {static_cast<float>(total)};
#endif
}

GAFIME_HIP_FORCEINLINE MutualInfoAccumulator mutual_info_contribution(
    unsigned int count,
    unsigned int count_x,
    unsigned int count_y,
    MutualInfoProbabilityContext probability
) {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    const double pxy = static_cast<double>(count) * probability.inverse_total;
    const double ratio =
        (static_cast<double>(count) * probability.total) /
        (static_cast<double>(count_x) * static_cast<double>(count_y));
    return pxy * log(ratio);
#else
    const float px = static_cast<float>(count_x) / probability.total;
    const float py = static_cast<float>(count_y) / probability.total;
    const float pxy = static_cast<float>(count) / probability.total;
    return pxy * logf(pxy / (px * py));
#endif
}

GAFIME_HIP_FORCEINLINE void accumulate_mutual_info(
    MutualInfoAccumulator value,
    MutualInfoAccumulator& sum
) {
    sum += value;
}

GAFIME_HIP_FORCEINLINE float finalize_mutual_info_score(
    MutualInfoAccumulator mi,
    uint32_t active_x,
    uint32_t active_y,
    MutualInfoProbabilityContext probability
) {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    const double correction = active_x > 0 && active_y > 0
        ? (static_cast<double>(active_x - 1) * static_cast<double>(active_y - 1)) /
            (2.0 * probability.total)
        : 0.0;
    const double corrected = fmax(0.0, mi - correction);
    const uint32_t normalizer_bins = min(active_x, active_y);
    const double normalizer = normalizer_bins > 1
        ? log(static_cast<double>(normalizer_bins))
        : 0.0;
    return normalizer > 0.0 ? static_cast<float>(corrected / normalizer) : 0.0f;
#else
    const float correction = active_x > 0 && active_y > 0
        ? static_cast<float>((active_x - 1) * (active_y - 1)) /
            (2.0f * probability.total)
        : 0.0f;
    const float corrected = fmaxf(0.0f, mi - correction);
    const uint32_t normalizer_bins = min(active_x, active_y);
    const float normalizer = normalizer_bins > 1
        ? logf(static_cast<float>(normalizer_bins))
        : 0.0f;
    return normalizer > 0.0f ? corrected / normalizer : 0.0f;
#endif
}

// These helpers are called only by full waves in fixed 256-thread MI blocks.
GAFIME_HIP_FORCEINLINE unsigned int float_order_key(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int sign_mask = (bits & 0x80000000u) != 0 ? 0xffffffffu : 0x80000000u;
    return bits ^ sign_mask;
}

GAFIME_HIP_FORCEINLINE float ordered_key_float(unsigned int key) {
    const unsigned int sign_mask = (key & 0x80000000u) != 0 ? 0x80000000u : 0xffffffffu;
    return __uint_as_float(key ^ sign_mask);
}

GAFIME_HIP_FORCEINLINE float wave_reduce_min_float(float value) {
    return ordered_key_float(__ockl_wfred_min_u32(float_order_key(value)));
}

GAFIME_HIP_FORCEINLINE float wave_reduce_max_float(float value) {
    return ordered_key_float(__ockl_wfred_max_u32(float_order_key(value)));
}

GAFIME_HIP_FORCEINLINE unsigned int wave_reduce_sum_uint(unsigned int value) {
    return __ockl_wfred_add_u32(value);
}

template <uint32_t Arity>
GAFIME_HIP_FORCEINLINE float interaction_value(
    const float* features,
    const float* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo
) {
    static_assert(Arity >= 1 && Arity <= gafime_rocm_v1::kTemplateMaxArity);
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

GAFIME_HIP_FORCEINLINE float interaction_value(
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
GAFIME_HIP_FORCEINLINE float selected_interaction_value(
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
        const float x = interaction_value(features, column_means, row, n_samples, combo, arity);
        const float y = target[row];
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
        const float x = interaction_value(features, column_means, row, n_samples, combo, arity);
        const float y = target[row];
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
        const float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        const float y = target[row];
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
        const float x = interaction_value<Arity>(features, column_means, row, n_samples, combo);
        const float y = target[row];
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

// Mutual-information kernel: one block per candidate, adaptive-range binning into
// shared histograms, finite-sample-corrected + normalized MI. Ported from the
// CUDA host (HIP is source-compatible: atomicAdd / __syncthreads / log / min).
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
        const MutualInfoProbabilityContext probability =
            mutual_info_probability_context(valid_count);
        MutualInfoAccumulator mi = static_cast<MutualInfoAccumulator>(0);
        uint32_t active_x = 0;
        uint32_t active_y = 0;
        for (uint32_t xb = 0; xb < bins; ++xb) {
            if (hist_x[xb] == 0) {
                continue;
            }
            ++active_x;
            for (uint32_t yb = 0; yb < bins; ++yb) {
                const unsigned int count = joint[xb * bins + yb];
                if (count == 0 || hist_y[yb] == 0) {
                    continue;
                }
                accumulate_mutual_info(
                    mutual_info_contribution(
                        count, hist_x[xb], hist_y[yb], probability),
                    mi
                );
            }
        }
        for (uint32_t yb = 0; yb < bins; ++yb) {
            if (hist_y[yb] != 0) {
                ++active_y;
            }
        }
        metric_values[combo_row * metric_count + metric_index] =
            finalize_mutual_info_score(mi, active_x, active_y, probability);
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
    // High-bin variants trade a small amount of code for lower LDS pressure.
    // The floating MI sum always keeps the existing shared-memory tree so its
    // addition order does not change.
    __shared__ float reduce_float0[kMiThreadsPerBlock];
    __shared__ float reduce_float1[kMiThreadsPerBlock];
    __shared__ float reduce_float2[kMiThreadsPerBlock];
    __shared__ float reduce_float3[kMiThreadsPerBlock];
    __shared__ unsigned int reduce_uint0[kMiThreadsPerBlock];
    __shared__ unsigned int reduce_uint1[kMiThreadsPerBlock];
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    __shared__ MutualInfoAccumulator reduce_mi0[kMiThreadsPerBlock];
#endif
    __shared__ float wave_float0[kMiMaxWavesPerBlock];
    __shared__ float wave_float1[kMiMaxWavesPerBlock];
    __shared__ float wave_float2[kMiMaxWavesPerBlock];
    __shared__ float wave_float3[kMiMaxWavesPerBlock];
    __shared__ unsigned int wave_uint0[kMiMaxWavesPerBlock];
    __shared__ unsigned int wave_uint1[kMiMaxWavesPerBlock];

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

    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t wave = threadIdx.x / warpSize;
    const uint32_t wave_count = (blockDim.x + warpSize - 1) / warpSize;
    if constexpr (kUseHipWaveMi<Bins>) {
        local_min_x = wave_reduce_min_float(local_min_x);
        local_max_x = wave_reduce_max_float(local_max_x);
        local_min_y = wave_reduce_min_float(local_min_y);
        local_max_y = wave_reduce_max_float(local_max_y);
        local_valid_count = wave_reduce_sum_uint(local_valid_count);
        if (lane == 0) {
            wave_float0[wave] = local_min_x;
            wave_float1[wave] = local_max_x;
            wave_float2[wave] = local_min_y;
            wave_float3[wave] = local_max_y;
            wave_uint0[wave] = local_valid_count;
        }
        __syncthreads();

        if (wave == 0) {
            float block_min_x = lane < wave_count ? wave_float0[lane] : INFINITY;
            float block_max_x = lane < wave_count ? wave_float1[lane] : -INFINITY;
            float block_min_y = lane < wave_count ? wave_float2[lane] : INFINITY;
            float block_max_y = lane < wave_count ? wave_float3[lane] : -INFINITY;
            unsigned int block_valid_count = lane < wave_count ? wave_uint0[lane] : 0;
            block_min_x = wave_reduce_min_float(block_min_x);
            block_max_x = wave_reduce_max_float(block_max_x);
            block_min_y = wave_reduce_min_float(block_min_y);
            block_max_y = wave_reduce_max_float(block_max_y);
            block_valid_count = wave_reduce_sum_uint(block_valid_count);
            if (lane == 0) {
                min_x = block_min_x;
                max_x = block_max_x;
                min_y = block_min_y;
                max_y = block_max_y;
                valid_count = block_valid_count;
            }
        }
    } else {
        reduce_float0[threadIdx.x] = local_min_x;
        reduce_float1[threadIdx.x] = local_max_x;
        reduce_float2[threadIdx.x] = local_min_y;
        reduce_float3[threadIdx.x] = local_max_y;
        reduce_uint0[threadIdx.x] = local_valid_count;
        __syncthreads();

        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                reduce_float0[threadIdx.x] =
                    fminf(reduce_float0[threadIdx.x], reduce_float0[threadIdx.x + stride]);
                reduce_float1[threadIdx.x] =
                    fmaxf(reduce_float1[threadIdx.x], reduce_float1[threadIdx.x + stride]);
                reduce_float2[threadIdx.x] =
                    fminf(reduce_float2[threadIdx.x], reduce_float2[threadIdx.x + stride]);
                reduce_float3[threadIdx.x] =
                    fmaxf(reduce_float3[threadIdx.x], reduce_float3[threadIdx.x + stride]);
                reduce_uint0[threadIdx.x] += reduce_uint0[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            min_x = reduce_float0[0];
            max_x = reduce_float1[0];
            min_y = reduce_float2[0];
            max_y = reduce_float3[0];
            valid_count = reduce_uint0[0];
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
            const MutualInfoProbabilityContext probability =
                mutual_info_probability_context(valid_count);
            MutualInfoAccumulator mi = static_cast<MutualInfoAccumulator>(0);
            uint32_t active_x = 0;
            uint32_t active_y = 0;
#pragma unroll
            for (uint32_t xb = 0; xb < Bins; ++xb) {
                if (hist_x[xb] == 0) {
                    continue;
                }
                ++active_x;
#pragma unroll
                for (uint32_t yb = 0; yb < Bins; ++yb) {
                    const unsigned int count = joint[xb * Bins + yb];
                    if (count == 0 || hist_y[yb] == 0) {
                        continue;
                    }
                    accumulate_mutual_info(
                        mutual_info_contribution(
                            count, hist_x[xb], hist_y[yb], probability),
                        mi
                    );
                }
            }
#pragma unroll
            for (uint32_t yb = 0; yb < Bins; ++yb) {
                if (hist_y[yb] != 0) {
                    ++active_y;
                }
            }
            metric_values[combo_row * metric_count + metric_index] =
                finalize_mutual_info_score(mi, active_x, active_y, probability);
        }
    } else {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
        const MutualInfoProbabilityContext probability =
            mutual_info_probability_context(valid_count);
        MutualInfoAccumulator local_mi = static_cast<MutualInfoAccumulator>(0);
#else
        const float probability_total = static_cast<float>(valid_count);
        float local_mi = 0.0f;
#endif
        uint32_t local_active_x = 0;
        uint32_t local_active_y = 0;
        for (uint32_t idx = threadIdx.x; idx < Bins * Bins; idx += blockDim.x) {
            const uint32_t xb = idx / Bins;
            const uint32_t yb = idx - xb * Bins;
            const unsigned int count = joint[idx];
            if (count == 0 || hist_x[xb] == 0 || hist_y[yb] == 0) {
                continue;
            }
#if GAFIME_GPU_MI_ACCUMULATION_FP64
            local_mi += mutual_info_contribution(
                count, hist_x[xb], hist_y[yb], probability);
#else
            const float px = static_cast<float>(hist_x[xb]) / probability_total;
            const float py = static_cast<float>(hist_y[yb]) / probability_total;
            const float pxy = static_cast<float>(count) / probability_total;
            local_mi += pxy * logf(pxy / (px * py));
#endif
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

#if GAFIME_GPU_MI_ACCUMULATION_FP64
        reduce_mi0[threadIdx.x] = local_mi;
#else
        reduce_float0[threadIdx.x] = local_mi;
#endif
        if constexpr (kUseHipWaveMi<Bins>) {
            local_active_x = wave_reduce_sum_uint(local_active_x);
            local_active_y = wave_reduce_sum_uint(local_active_y);
            if (lane == 0) {
                wave_uint0[wave] = local_active_x;
                wave_uint1[wave] = local_active_y;
            }
        } else {
            reduce_uint0[threadIdx.x] = local_active_x;
            reduce_uint1[threadIdx.x] = local_active_y;
        }
        __syncthreads();

        if constexpr (kUseHipWaveMi<Bins>) {
            if (wave == 0) {
                unsigned int block_active_x = lane < wave_count ? wave_uint0[lane] : 0;
                unsigned int block_active_y = lane < wave_count ? wave_uint1[lane] : 0;
                block_active_x = wave_reduce_sum_uint(block_active_x);
                block_active_y = wave_reduce_sum_uint(block_active_y);
                if (lane == 0) {
                    wave_uint0[0] = block_active_x;
                    wave_uint1[0] = block_active_y;
                }
            }
        }

        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
#if GAFIME_GPU_MI_ACCUMULATION_FP64
                reduce_mi0[threadIdx.x] += reduce_mi0[threadIdx.x + stride];
#else
                reduce_float0[threadIdx.x] += reduce_float0[threadIdx.x + stride];
#endif
                if constexpr (!kUseHipWaveMi<Bins>) {
                    reduce_uint0[threadIdx.x] += reduce_uint0[threadIdx.x + stride];
                    reduce_uint1[threadIdx.x] += reduce_uint1[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const uint32_t active_x = kUseHipWaveMi<Bins>
                ? wave_uint0[0]
                : reduce_uint0[0];
            const uint32_t active_y = kUseHipWaveMi<Bins>
                ? wave_uint1[0]
                : reduce_uint1[0];
#if GAFIME_GPU_MI_ACCUMULATION_FP64
            const MutualInfoAccumulator block_mi = reduce_mi0[0];
            metric_values[combo_row * metric_count + metric_index] =
                finalize_mutual_info_score(block_mi, active_x, active_y, probability);
#else
            const float correction = active_x > 0 && active_y > 0
                ? static_cast<float>((active_x - 1) * (active_y - 1)) /
                    (2.0f * probability_total)
                : 0.0f;
            const float corrected = fmaxf(0.0f, reduce_float0[0] - correction);
            const uint32_t normalizer_bins = min(active_x, active_y);
            const float normalizer = normalizer_bins > 1
                ? logf(static_cast<float>(normalizer_bins))
                : 0.0f;
            metric_values[combo_row * metric_count + metric_index] =
                normalizer > 0.0f ? corrected / normalizer : 0.0f;
#endif
        }
    }
}

// Spearman = Pearson on ranks (count-based average-tie ranks over finite pairs,
// f64 accumulation). Verbatim port of the CUDA kernel (HIP source-compatible).
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
GAFIME_HIP_FORCEINLINE bool candidate_better_static(
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

// One block diagnoses one caller-selected combo. The launcher validates the
// padded descriptor rows before launch, so this kernel only needs to preserve
// the scoring interaction order while recording non-finite materialization.
__global__ void interaction_diagnostics_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t max_arity,
    uint64_t* overflow_row_counts,
    uint32_t* flags
) {
    const uint64_t combo_row = static_cast<uint64_t>(blockIdx.x);
    const uint32_t* combo = combo_indices + combo_row * max_arity;
    uint32_t arity = 0;
    while (arity < max_arity && combo[arity] != UINT32_MAX) {
        ++arity;
    }

    uint64_t local_overflow_rows = 0;
    uint32_t local_flags = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        bool feature_sources_finite = true;
        if (!isfinite(target[row])) {
            local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
        }

        for (uint32_t idx = 0; idx < arity; ++idx) {
            const uint32_t col = combo[idx];
            const float source = features[static_cast<uint64_t>(col) * n_samples + row];
            const float mean = column_means[col];
            if (!isfinite(source) || !isfinite(mean)) {
                feature_sources_finite = false;
                local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
            }
        }

        if (arity > 1 && feature_sources_finite) {
            float value = 1.0f;
            bool materialization_nonfinite = false;
            for (uint32_t idx = 0; idx < arity; ++idx) {
                const uint32_t col = combo[idx];
                const float centered =
                    features[static_cast<uint64_t>(col) * n_samples + row] - column_means[col];
                materialization_nonfinite = materialization_nonfinite || !isfinite(centered);
                // Keep the same sequential fp32 multiplication order as interaction_value.
                value *= centered;
                materialization_nonfinite = materialization_nonfinite || !isfinite(value);
            }
            if (materialization_nonfinite) {
                ++local_overflow_rows;
            }
        }
    }

    __shared__ uint64_t overflow_partials[kThreadsPerBlock];
    __shared__ uint32_t flag_partials[kThreadsPerBlock];
    overflow_partials[threadIdx.x] = local_overflow_rows;
    flag_partials[threadIdx.x] = local_flags;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            overflow_partials[threadIdx.x] += overflow_partials[threadIdx.x + stride];
            flag_partials[threadIdx.x] |= flag_partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        overflow_row_counts[combo_row] = overflow_partials[0];
        flags[combo_row] = flag_partials[0];
    }
}

// ---------------------------------------------------------------------------
// Precision-profile kernels (ABI 1.1)
//
// No kernel in this namespace accepts a runtime precision discriminator.  The
// host launcher selects a concrete `<Storage, Accumulator, Result>`
// instantiation once when it creates a precision matrix.  In particular the
// mixed lane's `Storage` remains float all the way through interaction
// materialisation, while its reductions and result values are double.
// ---------------------------------------------------------------------------
namespace precision_kernel {

template <typename T>
GAFIME_HIP_FORCEINLINE bool precision_isfinite(T value) {
    return isfinite(value);
}

template <typename T>
GAFIME_HIP_FORCEINLINE T precision_min(T lhs, T rhs) {
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
GAFIME_HIP_FORCEINLINE T precision_max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
GAFIME_HIP_FORCEINLINE T precision_abs(T value) {
    return value < static_cast<T>(0) ? -value : value;
}

template <typename T>
GAFIME_HIP_FORCEINLINE T precision_log(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return logf(value);
    }
    return log(value);
}

template <typename T>
GAFIME_HIP_FORCEINLINE T precision_sqrt(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return sqrtf(value);
    }
    return sqrt(value);
}

template <typename ResultT>
GAFIME_HIP_FORCEINLINE ResultT precision_nonfinite_metric() {
    return static_cast<ResultT>(NAN);
}

template <typename AccumT, typename ResultT>
GAFIME_HIP_FORCEINLINE ResultT precision_finalize_correlation(
    AccumT variance_x,
    AccumT variance_y,
    AccumT covariance
) {
    if (!precision_isfinite(variance_x) || !precision_isfinite(variance_y) ||
        !precision_isfinite(covariance)) {
        return precision_nonfinite_metric<ResultT>();
    }
    if (variance_x == static_cast<AccumT>(0) || variance_y == static_cast<AccumT>(0)) {
        return static_cast<ResultT>(0);
    }
    if (variance_x < static_cast<AccumT>(0) || variance_y < static_cast<AccumT>(0)) {
        return precision_nonfinite_metric<ResultT>();
    }
    const AccumT denominator = precision_sqrt(variance_x * variance_y);
    if (!precision_isfinite(denominator) || denominator <= static_cast<AccumT>(0)) {
        return precision_nonfinite_metric<ResultT>();
    }
    const AccumT correlation = covariance / denominator;
    if (!precision_isfinite(correlation)) {
        return precision_nonfinite_metric<ResultT>();
    }
    return static_cast<ResultT>(precision_min(
        static_cast<AccumT>(1),
        precision_max(static_cast<AccumT>(-1), correlation)
    ));
}

template <typename ResultT>
GAFIME_HIP_FORCEINLINE ResultT precision_finalize_r2(ResultT pearson) {
    if (!precision_isfinite(pearson)) {
        return precision_nonfinite_metric<ResultT>();
    }
    return precision_min(
        static_cast<ResultT>(1),
        precision_max(static_cast<ResultT>(0), pearson * pearson)
    );
}

template <typename StorageT, typename AccumT, uint32_t Arity>
GAFIME_HIP_FORCEINLINE StorageT precision_interaction_value(
    const StorageT* features,
    const AccumT* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo
) {
    static_assert(Arity >= 1 && Arity <= gafime_rocm_v1::kTemplateMaxArity);
    if constexpr (Arity == 1) {
        return features[static_cast<uint64_t>(combo[0]) * rows + row];
    }
    // Keeping this variable in StorageT is intentional.  It makes mixed
    // materialise interaction arithmetic in binary32 before any widening.
    StorageT value = static_cast<StorageT>(1);
#pragma unroll
    for (uint32_t idx = 0; idx < Arity; ++idx) {
        const uint32_t column = combo[idx];
        // The mean is a reduction and therefore uses AccumT.  The explicit
        // cast before subtraction is the mixed profile's fp32 pointwise
        // boundary: it never turns materialised interaction arithmetic into
        // an accidental fp64 lane.
        const StorageT mean = static_cast<StorageT>(column_means[column]);
        const StorageT centered =
            features[static_cast<uint64_t>(column) * rows + row] - mean;
        value = static_cast<StorageT>(value * centered);
    }
    return value;
}

// Diagnostics use the same storage-domain subtraction and multiplication as
// the scorer.  AccumT appears only in the persisted reduction mean; for mixed
// it is narrowed immediately before the fp32 materialisation boundary.
template <typename StorageT, typename AccumT>
__global__ void interaction_diagnostics_kernel(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t max_arity,
    uint64_t* overflow_row_counts,
    uint32_t* flags
) {
    const uint64_t combo_row = static_cast<uint64_t>(blockIdx.x);
    const uint32_t* combo = combo_indices + combo_row * max_arity;
    uint32_t arity = 0;
    while (arity < max_arity && combo[arity] != UINT32_MAX) {
        ++arity;
    }

    uint64_t local_overflow_rows = 0;
    uint32_t local_flags = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        bool feature_sources_finite = true;
        if (!precision_isfinite(target[row])) {
            local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
        }
        for (uint32_t idx = 0; idx < arity; ++idx) {
            const uint32_t col = combo[idx];
            const StorageT source = features[static_cast<uint64_t>(col) * n_samples + row];
            const StorageT mean = static_cast<StorageT>(column_means[col]);
            if (!precision_isfinite(source) || !precision_isfinite(mean)) {
                feature_sources_finite = false;
                local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
            }
        }
        if (arity > 1 && feature_sources_finite) {
            StorageT value = static_cast<StorageT>(1);
            bool materialization_nonfinite = false;
            for (uint32_t idx = 0; idx < arity; ++idx) {
                const uint32_t col = combo[idx];
                const StorageT centered = static_cast<StorageT>(
                    features[static_cast<uint64_t>(col) * n_samples + row] -
                    static_cast<StorageT>(column_means[col]));
                materialization_nonfinite = materialization_nonfinite ||
                    !precision_isfinite(centered);
                value = static_cast<StorageT>(value * centered);
                materialization_nonfinite = materialization_nonfinite ||
                    !precision_isfinite(value);
            }
            if (materialization_nonfinite) {
                ++local_overflow_rows;
            }
        }
    }

    __shared__ uint64_t overflow_partials[kThreadsPerBlock];
    __shared__ uint32_t flag_partials[kThreadsPerBlock];
    overflow_partials[threadIdx.x] = local_overflow_rows;
    flag_partials[threadIdx.x] = local_flags;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            overflow_partials[threadIdx.x] += overflow_partials[threadIdx.x + stride];
            flag_partials[threadIdx.x] |= flag_partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        overflow_row_counts[combo_row] = overflow_partials[0];
        flags[combo_row] = flag_partials[0];
    }
}

template <typename StorageT>
GAFIME_HIP_FORCEINLINE uint32_t precision_fixed_mi_bin(
    StorageT value,
    StorageT minimum,
    StorageT inverse_span,
    uint32_t bins
) {
    // Histogram destinations remain integer and fixed-bin placement is
    // pointwise arithmetic. StorageT therefore keeps mixed bin identity in
    // f32; probability/logarithm/correction/final scoring use AccumT later.
    const StorageT scaled = (value - minimum) * inverse_span;
    if (!precision_isfinite(scaled) || scaled <= static_cast<StorageT>(0)) {
        return 0;
    }
    const uint32_t max_bin = bins - 1;
    if (scaled >= static_cast<StorageT>(max_bin)) {
        return max_bin;
    }
    return static_cast<uint32_t>(scaled);
}

template <typename StorageT, typename AccumT>
__global__ void target_stats_kernel(
    const StorageT* target,
    uint64_t n_samples,
    PrecisionTargetStatsDevice<AccumT>* target_stats
) {
    AccumT local_sum = static_cast<AccumT>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT value = target[row];
        if (precision_isfinite(value)) {
            local_sum += static_cast<AccumT>(value);
            ++local_count;
        }
    }
    __shared__ AccumT sums[kThreadsPerBlock];
    __shared__ uint64_t counts[kThreadsPerBlock];
    __shared__ AccumT mean;
    __shared__ AccumT variances[kThreadsPerBlock];
    sums[threadIdx.x] = local_sum;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean = counts[0] == 0 ? static_cast<AccumT>(0) :
            sums[0] / static_cast<AccumT>(counts[0]);
    }
    __syncthreads();
    AccumT local_variance = static_cast<AccumT>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT value = target[row];
        if (precision_isfinite(value)) {
            const AccumT delta = static_cast<AccumT>(value) - mean;
            local_variance += delta * delta;
        }
    }
    variances[threadIdx.x] = local_variance;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            variances[threadIdx.x] += variances[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        target_stats->mean_y = mean;
        target_stats->syy = variances[0];
        target_stats->finite = counts[0] == n_samples ? 1u : 0u;
        target_stats->reserved = 0u;
    }
}

template <typename StorageT, typename AccumT>
__global__ void unary_feature_stats_kernel(
    const StorageT* features,
    uint64_t n_samples,
    uint32_t n_features,
    PrecisionUnaryFeatureStatsDevice<AccumT>* feature_stats
) {
    const uint32_t column = blockIdx.x;
    if (column >= n_features) {
        return;
    }
    const uint64_t base = static_cast<uint64_t>(column) * n_samples;
    AccumT local_sum = static_cast<AccumT>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT value = features[base + row];
        if (precision_isfinite(value)) {
            local_sum += static_cast<AccumT>(value);
            ++local_count;
        }
    }
    __shared__ AccumT sums[kThreadsPerBlock];
    __shared__ uint64_t counts[kThreadsPerBlock];
    __shared__ AccumT mean;
    __shared__ AccumT variances[kThreadsPerBlock];
    sums[threadIdx.x] = local_sum;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean = counts[0] == 0 ? static_cast<AccumT>(0) :
            sums[0] / static_cast<AccumT>(counts[0]);
    }
    __syncthreads();
    AccumT local_variance = static_cast<AccumT>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT value = features[base + row];
        if (precision_isfinite(value)) {
            const AccumT delta = static_cast<AccumT>(value) - mean;
            local_variance += delta * delta;
        }
    }
    variances[threadIdx.x] = local_variance;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            variances[threadIdx.x] += variances[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        feature_stats[column].mean_x = mean;
        feature_stats[column].sxx = variances[0];
        feature_stats[column].finite = counts[0] == n_samples ? 1u : 0u;
        feature_stats[column].reserved = 0u;
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, bool Scaled>
__global__ void score_continuous_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    ResultT* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;

    // A reduction tree provides no occupancy benefit for a sub-block input,
    // and it changes the row order for nearly constant generated features.
    // Match Core's deterministic row order for both precision modes.  The
    // optional magnitude scan does not change the subsequent two-pass
    // mean/covariance order used by the serial path.
    if (n_samples <= static_cast<uint64_t>(kThreadsPerBlock)) {
        if (threadIdx.x != 0) {
            return;
        }
        AccumT serial_scale_x = static_cast<AccumT>(0);
        AccumT serial_scale_y = static_cast<AccumT>(0);
        uint64_t serial_count = 0;
        for (uint64_t row = 0; row < n_samples; ++row) {
            const StorageT x = precision_interaction_value<StorageT, AccumT, Arity>(
                features, column_means, row, n_samples, combo);
            const StorageT y = target[row];
            if (precision_isfinite(x) && precision_isfinite(y)) {
                if constexpr (Scaled) {
                    const AccumT abs_x = precision_abs(static_cast<AccumT>(x));
                    const AccumT abs_y = precision_abs(static_cast<AccumT>(y));
                    serial_scale_x = precision_max(serial_scale_x, abs_x);
                    serial_scale_y = precision_max(serial_scale_y, abs_y);
                }
                ++serial_count;
            }
        }
        AccumT serial_sum_x = static_cast<AccumT>(0);
        AccumT serial_sum_y = static_cast<AccumT>(0);
        for (uint64_t row = 0; row < n_samples; ++row) {
            const StorageT raw_x = precision_interaction_value<StorageT, AccumT, Arity>(
                features, column_means, row, n_samples, combo);
            const StorageT raw_y = target[row];
            if (precision_isfinite(raw_x) && precision_isfinite(raw_y)) {
                AccumT x = static_cast<AccumT>(raw_x);
                AccumT y = static_cast<AccumT>(raw_y);
                if constexpr (Scaled) {
                    x = serial_scale_x > static_cast<AccumT>(0)
                        ? x / serial_scale_x : static_cast<AccumT>(0);
                    y = serial_scale_y > static_cast<AccumT>(0)
                        ? y / serial_scale_y : static_cast<AccumT>(0);
                }
                serial_sum_x += x;
                serial_sum_y += y;
            }
        }
        const AccumT serial_count_value = static_cast<AccumT>(serial_count);
        const AccumT serial_mean_x = serial_count == 0
            ? static_cast<AccumT>(0) : serial_sum_x / serial_count_value;
        const AccumT serial_mean_y = serial_count == 0
            ? static_cast<AccumT>(0) : serial_sum_y / serial_count_value;
        AccumT serial_sum_xx = static_cast<AccumT>(0);
        AccumT serial_sum_yy = static_cast<AccumT>(0);
        AccumT serial_sum_xy = static_cast<AccumT>(0);
        for (uint64_t row = 0; row < n_samples; ++row) {
            const StorageT raw_x = precision_interaction_value<StorageT, AccumT, Arity>(
                features, column_means, row, n_samples, combo);
            const StorageT raw_y = target[row];
            if (precision_isfinite(raw_x) && precision_isfinite(raw_y)) {
                AccumT x = static_cast<AccumT>(raw_x);
                AccumT y = static_cast<AccumT>(raw_y);
                if constexpr (Scaled) {
                    x = serial_scale_x > static_cast<AccumT>(0)
                        ? x / serial_scale_x : static_cast<AccumT>(0);
                    y = serial_scale_y > static_cast<AccumT>(0)
                        ? y / serial_scale_y : static_cast<AccumT>(0);
                }
                const AccumT delta_x = x - serial_mean_x;
                const AccumT delta_y = y - serial_mean_y;
                serial_sum_xx += delta_x * delta_x;
                serial_sum_yy += delta_y * delta_y;
                serial_sum_xy += delta_x * delta_y;
            }
        }
        const ResultT pearson = precision_finalize_correlation<AccumT, ResultT>(
            serial_sum_xx, serial_sum_yy, serial_sum_xy);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric = metric_ids[metric_idx];
            if (metric == GAFIME_METRIC_PEARSON) {
                metric_values[combo_row * metric_count + metric_idx] = pearson;
            } else if (metric == GAFIME_METRIC_R2) {
                metric_values[combo_row * metric_count + metric_idx] =
                    precision_finalize_r2(pearson);
            } else {
                metric_values[combo_row * metric_count + metric_idx] =
                    static_cast<ResultT>(0);
            }
        }
        return;
    }

    // The unscaled specialization keeps the established two-pass reduction.
    // The scaled specialization uses the guarded three-pass algorithm: find
    // finite magnitudes, reduce scaled means, then reduce scaled covariance.
    AccumT local_first_x = static_cast<AccumT>(0);
    AccumT local_first_y = static_cast<AccumT>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT x = precision_interaction_value<StorageT, AccumT, Arity>(
            features, column_means, row, n_samples, combo);
        const StorageT y = target[row];
        if (precision_isfinite(x) && precision_isfinite(y)) {
            if constexpr (Scaled) {
                local_first_x = precision_max(
                    local_first_x, precision_abs(static_cast<AccumT>(x)));
                local_first_y = precision_max(
                    local_first_y, precision_abs(static_cast<AccumT>(y)));
            } else {
                local_first_x += static_cast<AccumT>(x);
                local_first_y += static_cast<AccumT>(y);
            }
            ++local_count;
        }
    }
    __shared__ AccumT sums_x[kThreadsPerBlock];
    __shared__ AccumT sums_y[kThreadsPerBlock];
    __shared__ uint64_t counts[kThreadsPerBlock];
    __shared__ AccumT scale_x;
    __shared__ AccumT scale_y;
    __shared__ AccumT mean_x;
    __shared__ AccumT mean_y;
    __shared__ AccumT sum_xx[kThreadsPerBlock];
    __shared__ AccumT sum_yy[kThreadsPerBlock];
    __shared__ AccumT sum_xy[kThreadsPerBlock];
    sums_x[threadIdx.x] = local_first_x;
    sums_y[threadIdx.x] = local_first_y;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if constexpr (Scaled) {
                sums_x[threadIdx.x] = precision_max(
                    sums_x[threadIdx.x], sums_x[threadIdx.x + stride]);
                sums_y[threadIdx.x] = precision_max(
                    sums_y[threadIdx.x], sums_y[threadIdx.x + stride]);
            } else {
                sums_x[threadIdx.x] += sums_x[threadIdx.x + stride];
                sums_y[threadIdx.x] += sums_y[threadIdx.x + stride];
            }
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if constexpr (Scaled) {
            scale_x = sums_x[0];
            scale_y = sums_y[0];
        } else {
            scale_x = static_cast<AccumT>(1);
            scale_y = static_cast<AccumT>(1);
            const AccumT count = static_cast<AccumT>(counts[0]);
            mean_x = counts[0] == 0 ? static_cast<AccumT>(0) : sums_x[0] / count;
            mean_y = counts[0] == 0 ? static_cast<AccumT>(0) : sums_y[0] / count;
        }
    }
    __syncthreads();

    if constexpr (Scaled) {
        AccumT local_sum_x = static_cast<AccumT>(0);
        AccumT local_sum_y = static_cast<AccumT>(0);
        for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
            const StorageT raw_x = precision_interaction_value<StorageT, AccumT, Arity>(
                features, column_means, row, n_samples, combo);
            const StorageT raw_y = target[row];
            if (precision_isfinite(raw_x) && precision_isfinite(raw_y)) {
                local_sum_x += scale_x > static_cast<AccumT>(0)
                    ? static_cast<AccumT>(raw_x) / scale_x : static_cast<AccumT>(0);
                local_sum_y += scale_y > static_cast<AccumT>(0)
                    ? static_cast<AccumT>(raw_y) / scale_y : static_cast<AccumT>(0);
            }
        }
        sums_x[threadIdx.x] = local_sum_x;
        sums_y[threadIdx.x] = local_sum_y;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums_x[threadIdx.x] += sums_x[threadIdx.x + stride];
                sums_y[threadIdx.x] += sums_y[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const AccumT count = static_cast<AccumT>(counts[0]);
            mean_x = counts[0] == 0 ? static_cast<AccumT>(0) : sums_x[0] / count;
            mean_y = counts[0] == 0 ? static_cast<AccumT>(0) : sums_y[0] / count;
        }
        __syncthreads();
    }

    AccumT local_xx = static_cast<AccumT>(0);
    AccumT local_yy = static_cast<AccumT>(0);
    AccumT local_xy = static_cast<AccumT>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT raw_x = precision_interaction_value<StorageT, AccumT, Arity>(
            features, column_means, row, n_samples, combo);
        const StorageT raw_y = target[row];
        if (precision_isfinite(raw_x) && precision_isfinite(raw_y)) {
            AccumT x = static_cast<AccumT>(raw_x);
            AccumT y = static_cast<AccumT>(raw_y);
            if constexpr (Scaled) {
                x = scale_x > static_cast<AccumT>(0)
                    ? x / scale_x : static_cast<AccumT>(0);
                y = scale_y > static_cast<AccumT>(0)
                    ? y / scale_y : static_cast<AccumT>(0);
            }
            const AccumT delta_x = x - mean_x;
            const AccumT delta_y = y - mean_y;
            local_xx += delta_x * delta_x;
            local_yy += delta_y * delta_y;
            local_xy += delta_x * delta_y;
        }
    }
    sum_xx[threadIdx.x] = local_xx;
    sum_yy[threadIdx.x] = local_yy;
    sum_xy[threadIdx.x] = local_xy;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_xx[threadIdx.x] += sum_xx[threadIdx.x + stride];
            sum_yy[threadIdx.x] += sum_yy[threadIdx.x + stride];
            sum_xy[threadIdx.x] += sum_xy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const ResultT pearson = precision_finalize_correlation<AccumT, ResultT>(
            sum_xx[0], sum_yy[0], sum_xy[0]);
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric = metric_ids[metric_idx];
            if (metric == GAFIME_METRIC_PEARSON) {
                metric_values[combo_row * metric_count + metric_idx] = pearson;
            } else if (metric == GAFIME_METRIC_R2) {
                metric_values[combo_row * metric_count + metric_idx] = precision_finalize_r2(pearson);
            } else {
                metric_values[combo_row * metric_count + metric_idx] = static_cast<ResultT>(0);
            }
        }
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, uint32_t Bins>
__global__ void score_mutual_info_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;
    StorageT local_min_x = static_cast<StorageT>(INFINITY);
    StorageT local_max_x = static_cast<StorageT>(-INFINITY);
    StorageT local_min_y = static_cast<StorageT>(INFINITY);
    StorageT local_max_y = static_cast<StorageT>(-INFINITY);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT x = precision_interaction_value<StorageT, AccumT, Arity>(
            features, column_means, row, n_samples, combo);
        const StorageT y = target[row];
        if (precision_isfinite(x) && precision_isfinite(y)) {
            local_min_x = precision_min(local_min_x, x);
            local_max_x = precision_max(local_max_x, x);
            local_min_y = precision_min(local_min_y, y);
            local_max_y = precision_max(local_max_y, y);
            ++local_count;
        }
    }
    __shared__ StorageT mins_x[kMiThreadsPerBlock];
    __shared__ StorageT maxs_x[kMiThreadsPerBlock];
    __shared__ StorageT mins_y[kMiThreadsPerBlock];
    __shared__ StorageT maxs_y[kMiThreadsPerBlock];
    __shared__ uint64_t counts[kMiThreadsPerBlock];
    __shared__ StorageT min_x;
    __shared__ StorageT max_x;
    __shared__ StorageT min_y;
    __shared__ StorageT max_y;
    __shared__ uint64_t valid_count;
    __shared__ uint32_t hist_x[Bins];
    __shared__ uint32_t hist_y[Bins];
    __shared__ uint32_t joint[Bins * Bins];
    mins_x[threadIdx.x] = local_min_x;
    maxs_x[threadIdx.x] = local_max_x;
    mins_y[threadIdx.x] = local_min_y;
    maxs_y[threadIdx.x] = local_max_y;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mins_x[threadIdx.x] = precision_min(mins_x[threadIdx.x], mins_x[threadIdx.x + stride]);
            maxs_x[threadIdx.x] = precision_max(maxs_x[threadIdx.x], maxs_x[threadIdx.x + stride]);
            mins_y[threadIdx.x] = precision_min(mins_y[threadIdx.x], mins_y[threadIdx.x + stride]);
            maxs_y[threadIdx.x] = precision_max(maxs_y[threadIdx.x], maxs_y[threadIdx.x + stride]);
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        min_x = mins_x[0];
        max_x = maxs_x[0];
        min_y = mins_y[0];
        max_y = maxs_y[0];
        valid_count = counts[0];
    }
    for (uint32_t idx = threadIdx.x; idx < Bins; idx += blockDim.x) {
        hist_x[idx] = 0;
        hist_y[idx] = 0;
    }
    for (uint32_t idx = threadIdx.x; idx < Bins * Bins; idx += blockDim.x) {
        joint[idx] = 0;
    }
    __syncthreads();
    if (valid_count <= 1 || !(max_x > min_x) || !(max_y > min_y)) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = static_cast<ResultT>(0);
        }
        return;
    }
    const StorageT inv_x = static_cast<StorageT>(Bins) / (max_x - min_x);
    const StorageT inv_y = static_cast<StorageT>(Bins) / (max_y - min_y);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT x = precision_interaction_value<StorageT, AccumT, Arity>(
            features, column_means, row, n_samples, combo);
        const StorageT y = target[row];
        if (precision_isfinite(x) && precision_isfinite(y)) {
            const uint32_t bin_x = precision_fixed_mi_bin<StorageT>(x, min_x, inv_x, Bins);
            const uint32_t bin_y = precision_fixed_mi_bin<StorageT>(y, min_y, inv_y, Bins);
            atomicAdd(&hist_x[bin_x], 1u);
            atomicAdd(&hist_y[bin_y], 1u);
            atomicAdd(&joint[bin_x * Bins + bin_y], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const AccumT total = static_cast<AccumT>(valid_count);
        AccumT mi = static_cast<AccumT>(0);
        uint32_t active_x = 0;
        uint32_t active_y = 0;
        for (uint32_t bin_x = 0; bin_x < Bins; ++bin_x) {
            active_x += hist_x[bin_x] != 0 ? 1u : 0u;
            active_y += hist_y[bin_x] != 0 ? 1u : 0u;
        }
        for (uint32_t bin_x = 0; bin_x < Bins; ++bin_x) {
            const uint32_t count_x = hist_x[bin_x];
            if (count_x == 0) {
                continue;
            }
            for (uint32_t bin_y = 0; bin_y < Bins; ++bin_y) {
                const uint32_t count = joint[bin_x * Bins + bin_y];
                const uint32_t count_y = hist_y[bin_y];
                if (count == 0 || count_y == 0) {
                    continue;
                }
                const AccumT pxy = static_cast<AccumT>(count) / total;
                const AccumT ratio =
                    (static_cast<AccumT>(count) * total) /
                    (static_cast<AccumT>(count_x) * static_cast<AccumT>(count_y));
                mi += pxy * precision_log(ratio);
            }
        }
        const AccumT correction = active_x > 0 && active_y > 0
            ? (static_cast<AccumT>(active_x - 1) * static_cast<AccumT>(active_y - 1)) /
                (static_cast<AccumT>(2) * total)
            : static_cast<AccumT>(0);
        const AccumT corrected = precision_max(static_cast<AccumT>(0), mi - correction);
        const uint32_t normalizer_bins = active_x < active_y ? active_x : active_y;
        const AccumT normalizer = normalizer_bins > 1
            ? precision_log(static_cast<AccumT>(normalizer_bins))
            : static_cast<AccumT>(0);
        metric_values[combo_row * metric_count + metric_index] = normalizer > static_cast<AccumT>(0)
            ? static_cast<ResultT>(corrected / normalizer)
            : static_cast<ResultT>(0);
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity>
__global__ void score_spearman_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * Arity;
    AccumT local_rx = static_cast<AccumT>(0);
    AccumT local_ry = static_cast<AccumT>(0);
    AccumT local_rxx = static_cast<AccumT>(0);
    AccumT local_ryy = static_cast<AccumT>(0);
    AccumT local_rxy = static_cast<AccumT>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const StorageT x = precision_interaction_value<StorageT, AccumT, Arity>(
            features, column_means, row, n_samples, combo);
        const StorageT y = target[row];
        if (!precision_isfinite(x) || !precision_isfinite(y)) {
            continue;
        }
        // Ranking positions are integer counters.  Only the average-tie
        // covariance arithmetic below is widened for mixed/fp64.
        uint64_t less_x = 0;
        uint64_t equal_x = 0;
        uint64_t less_y = 0;
        uint64_t equal_y = 0;
        for (uint64_t other = 0; other < n_samples; ++other) {
            const StorageT candidate_x = precision_interaction_value<StorageT, AccumT, Arity>(
                features, column_means, other, n_samples, combo);
            const StorageT candidate_y = target[other];
            if (!precision_isfinite(candidate_x) || !precision_isfinite(candidate_y)) {
                continue;
            }
            if (candidate_x < x) {
                ++less_x;
            } else if (candidate_x == x) {
                ++equal_x;
            }
            if (candidate_y < y) {
                ++less_y;
            } else if (candidate_y == y) {
                ++equal_y;
            }
        }
        const AccumT rank_x = static_cast<AccumT>(less_x) +
            static_cast<AccumT>(0.5) * (static_cast<AccumT>(equal_x) - static_cast<AccumT>(1));
        const AccumT rank_y = static_cast<AccumT>(less_y) +
            static_cast<AccumT>(0.5) * (static_cast<AccumT>(equal_y) - static_cast<AccumT>(1));
        local_rx += rank_x;
        local_ry += rank_y;
        local_rxx += rank_x * rank_x;
        local_ryy += rank_y * rank_y;
        local_rxy += rank_x * rank_y;
        ++local_count;
    }
    __shared__ AccumT sum_rx[kThreadsPerBlock];
    __shared__ AccumT sum_ry[kThreadsPerBlock];
    __shared__ AccumT sum_rxx[kThreadsPerBlock];
    __shared__ AccumT sum_ryy[kThreadsPerBlock];
    __shared__ AccumT sum_rxy[kThreadsPerBlock];
    __shared__ uint64_t counts[kThreadsPerBlock];
    sum_rx[threadIdx.x] = local_rx;
    sum_ry[threadIdx.x] = local_ry;
    sum_rxx[threadIdx.x] = local_rxx;
    sum_ryy[threadIdx.x] = local_ryy;
    sum_rxy[threadIdx.x] = local_rxy;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_rx[threadIdx.x] += sum_rx[threadIdx.x + stride];
            sum_ry[threadIdx.x] += sum_ry[threadIdx.x + stride];
            sum_rxx[threadIdx.x] += sum_rxx[threadIdx.x + stride];
            sum_ryy[threadIdx.x] += sum_ryy[threadIdx.x + stride];
            sum_rxy[threadIdx.x] += sum_rxy[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ResultT out = static_cast<ResultT>(0);
        if (counts[0] > 1) {
            const AccumT n = static_cast<AccumT>(counts[0]);
            const AccumT covariance = n * sum_rxy[0] - sum_rx[0] * sum_ry[0];
            const AccumT variance_x = n * sum_rxx[0] - sum_rx[0] * sum_rx[0];
            const AccumT variance_y = n * sum_ryy[0] - sum_ry[0] * sum_ry[0];
            out = precision_finalize_correlation<AccumT, ResultT>(
                variance_x, variance_y, covariance);
        }
        metric_values[combo_row * metric_count + metric_index] = out;
    }
}

template <typename ResultT, bool Descending>
GAFIME_HIP_FORCEINLINE bool precision_candidate_better(
    ResultT candidate_score,
    uint32_t candidate_index,
    ResultT best_score,
    uint32_t best_index
) {
    if (!precision_isfinite(candidate_score)) {
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
    // Candidate identity is structural metadata; it remains an integer and is
    // the deterministic final tie-break for every precision profile.
    return candidate_index < best_index;
}

template <typename ResultT, bool Descending>
__global__ void select_topk_partials_kernel_static(
    const ResultT* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    ResultT* partial_scores,
    uint32_t* partial_indices
) {
    __shared__ ResultT best_scores[kTopKThreadsPerBlock];
    __shared__ uint32_t best_indices[kTopKThreadsPerBlock];
    __shared__ ResultT previous_score;
    __shared__ uint32_t previous_index;
    const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) * top_k;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        ResultT local_score = Descending ? static_cast<ResultT>(-INFINITY) :
            static_cast<ResultT>(INFINITY);
        uint32_t local_index = UINT32_MAX;
        for (uint64_t row = start; row < row_count; row += stride) {
            const uint32_t row_index = static_cast<uint32_t>(row);
            const ResultT score = metric_values[row * metric_count + primary_metric_index];
            if (rank != 0 && !precision_candidate_better<ResultT, Descending>(
                    previous_score, previous_index, score, row_index)) {
                continue;
            }
            if (precision_candidate_better<ResultT, Descending>(
                    score, row_index, local_score, local_index)) {
                local_score = score;
                local_index = row_index;
            }
        }
        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride) {
                const ResultT score = best_scores[threadIdx.x + reduce_stride];
                const uint32_t index = best_indices[threadIdx.x + reduce_stride];
                if (precision_candidate_better<ResultT, Descending>(
                        score, index, best_scores[threadIdx.x], best_indices[threadIdx.x])) {
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
        }
        __syncthreads();
    }
}

template <typename ResultT, bool Descending>
__global__ void merge_topk_partials_kernel_static(
    const ResultT* partial_scores,
    const uint32_t* partial_indices,
    uint64_t partial_count,
    uint32_t top_k,
    uint32_t* selected_indices
) {
    __shared__ ResultT best_scores[kTopKThreadsPerBlock];
    __shared__ uint32_t best_indices[kTopKThreadsPerBlock];
    __shared__ ResultT previous_score;
    __shared__ uint32_t previous_index;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        ResultT local_score = Descending ? static_cast<ResultT>(-INFINITY) :
            static_cast<ResultT>(INFINITY);
        uint32_t local_index = UINT32_MAX;
        for (uint64_t item = threadIdx.x; item < partial_count; item += blockDim.x) {
            const uint32_t candidate_index = partial_indices[item];
            if (candidate_index == UINT32_MAX) {
                continue;
            }
            const ResultT score = partial_scores[item];
            if (rank != 0 && !precision_candidate_better<ResultT, Descending>(
                    previous_score, previous_index, score, candidate_index)) {
                continue;
            }
            if (precision_candidate_better<ResultT, Descending>(
                    score, candidate_index, local_score, local_index)) {
                local_score = score;
                local_index = candidate_index;
            }
        }
        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride) {
                const ResultT score = best_scores[threadIdx.x + reduce_stride];
                const uint32_t index = best_indices[threadIdx.x + reduce_stride];
                if (precision_candidate_better<ResultT, Descending>(
                        score, index, best_scores[threadIdx.x], best_indices[threadIdx.x])) {
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
        }
        __syncthreads();
    }
}

template <typename ResultT>
__global__ void copy_selected_metric_rows_kernel(
    const ResultT* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    ResultT* selected_metric_values
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (idx >= total) {
        return;
    }
    const uint64_t selected_row = idx / metric_count;
    const uint32_t metric_index = static_cast<uint32_t>(idx % metric_count);
    const uint32_t source_row = selected_indices[selected_row];
    selected_metric_values[idx] =
        metric_values[static_cast<uint64_t>(source_row) * metric_count + metric_index];
}

}  // namespace precision_kernel

#define GAFIME_HIP_INSTANTIATE_CONTINUOUS(ARITY) \
    template __global__ void score_continuous_chunk_kernel_static<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, const uint32_t*, uint32_t, float*);

#define GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(ARITY) \
    template __global__ void score_continuous_scaled_chunk_kernel<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint32_t, \
        uint64_t, uint64_t, const uint32_t*, uint32_t, float*);

#define GAFIME_HIP_INSTANTIATE_SPEARMAN(ARITY) \
    template __global__ void score_spearman_chunk_kernel_static<ARITY>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, uint32_t, uint32_t, float*);

#define GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, BINS) \
    template __global__ void score_mutual_info_chunk_kernel_static<ARITY, BINS>( \
        const float*, const float*, const float*, const uint32_t*, uint64_t, uint64_t, \
        uint64_t, uint32_t, uint32_t, float*);

#define GAFIME_HIP_INSTANTIATE_MI_ARITY(ARITY) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 2) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 4) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 8) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 12) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 16) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 24) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 32) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 48) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 64) \
    GAFIME_HIP_INSTANTIATE_MI_BIN(ARITY, 96)

GAFIME_HIP_INSTANTIATE_CONTINUOUS(1)
GAFIME_HIP_INSTANTIATE_CONTINUOUS(2)
GAFIME_HIP_INSTANTIATE_CONTINUOUS(3)
GAFIME_HIP_INSTANTIATE_CONTINUOUS(4)
GAFIME_HIP_INSTANTIATE_CONTINUOUS(5)

GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(0)
GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(1)
GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(2)
GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(3)
GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(4)
GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS(5)

GAFIME_HIP_INSTANTIATE_SPEARMAN(1)
GAFIME_HIP_INSTANTIATE_SPEARMAN(2)
GAFIME_HIP_INSTANTIATE_SPEARMAN(3)
GAFIME_HIP_INSTANTIATE_SPEARMAN(4)
GAFIME_HIP_INSTANTIATE_SPEARMAN(5)

GAFIME_HIP_INSTANTIATE_MI_ARITY(1)
GAFIME_HIP_INSTANTIATE_MI_ARITY(2)
GAFIME_HIP_INSTANTIATE_MI_ARITY(3)
GAFIME_HIP_INSTANTIATE_MI_ARITY(4)
GAFIME_HIP_INSTANTIATE_MI_ARITY(5)

template __global__ void select_topk_partials_kernel_static<true>(
    const float*, uint64_t, uint32_t, uint32_t, uint32_t, float*, uint32_t*);
template __global__ void select_topk_partials_kernel_static<false>(
    const float*, uint64_t, uint32_t, uint32_t, uint32_t, float*, uint32_t*);
template __global__ void merge_topk_partials_kernel_static<true>(
    const float*, const uint32_t*, uint64_t, uint32_t, uint32_t*);
template __global__ void merge_topk_partials_kernel_static<false>(
    const float*, const uint32_t*, uint64_t, uint32_t, uint32_t*);

// Emit every precision route into one ROCm payload.  The CMake target has no
// build-time precision selector: fp32, mixed, and fp64 are all concrete HIP
// specialisations selected through the ABI-1.1 dispatch table.
#define GAFIME_HIP_INSTANTIATE_PRECISION_STATS(STORAGE, ACCUM) \
    template __global__ void precision_kernel::target_stats_kernel<STORAGE, ACCUM>( \
        const STORAGE*, uint64_t, PrecisionTargetStatsDevice<ACCUM>*); \
    template __global__ void precision_kernel::unary_feature_stats_kernel<STORAGE, ACCUM>( \
        const STORAGE*, uint64_t, uint32_t, PrecisionUnaryFeatureStatsDevice<ACCUM>*); \
    template __global__ void precision_kernel::interaction_diagnostics_kernel<STORAGE, ACCUM>( \
        const STORAGE*, const STORAGE*, const ACCUM*, const uint32_t*, uint64_t, uint32_t, \
        uint64_t*, uint32_t*);

#define GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS_MODE( \
    STORAGE, ACCUM, RESULT, ARITY, SCALED) \
    template __global__ void precision_kernel::score_continuous_chunk_kernel_static< \
        STORAGE, ACCUM, RESULT, ARITY, SCALED>( \
            const STORAGE*, const STORAGE*, const ACCUM*, const uint32_t*, uint64_t, \
            uint64_t, uint64_t, const uint32_t*, uint32_t, RESULT*);

#define GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, ARITY) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS_MODE( \
        STORAGE, ACCUM, RESULT, ARITY, false) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS_MODE( \
        STORAGE, ACCUM, RESULT, ARITY, true)

#define GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, ARITY) \
    template __global__ void precision_kernel::score_spearman_chunk_kernel_static< \
        STORAGE, ACCUM, RESULT, ARITY>( \
            const STORAGE*, const STORAGE*, const ACCUM*, const uint32_t*, uint64_t, \
            uint64_t, uint64_t, uint32_t, uint32_t, RESULT*);

#define GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, BINS) \
    template __global__ void precision_kernel::score_mutual_info_chunk_kernel_static< \
        STORAGE, ACCUM, RESULT, ARITY, BINS>( \
            const STORAGE*, const STORAGE*, const ACCUM*, const uint32_t*, uint64_t, \
            uint64_t, uint64_t, uint32_t, uint32_t, RESULT*);

#define GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, ARITY) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 2) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 4) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 8) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 12) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 16) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 24) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 32) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 48) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 64) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN(STORAGE, ACCUM, RESULT, ARITY, 96)

#define GAFIME_HIP_INSTANTIATE_PRECISION_RANKING(RESULT) \
    template __global__ void precision_kernel::select_topk_partials_kernel_static<RESULT, true>( \
        const RESULT*, uint64_t, uint32_t, uint32_t, uint32_t, RESULT*, uint32_t*); \
    template __global__ void precision_kernel::select_topk_partials_kernel_static<RESULT, false>( \
        const RESULT*, uint64_t, uint32_t, uint32_t, uint32_t, RESULT*, uint32_t*); \
    template __global__ void precision_kernel::merge_topk_partials_kernel_static<RESULT, true>( \
        const RESULT*, const uint32_t*, uint64_t, uint32_t, uint32_t*); \
    template __global__ void precision_kernel::merge_topk_partials_kernel_static<RESULT, false>( \
        const RESULT*, const uint32_t*, uint64_t, uint32_t, uint32_t*); \
    template __global__ void precision_kernel::copy_selected_metric_rows_kernel<RESULT>( \
        const RESULT*, const uint32_t*, uint64_t, uint32_t, RESULT*);

#define GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(STORAGE, ACCUM, RESULT) \
    GAFIME_HIP_INSTANTIATE_PRECISION_STATS(STORAGE, ACCUM) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, 1) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, 2) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, 3) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, 4) \
    GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS(STORAGE, ACCUM, RESULT, 5) \
    GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, 1) \
    GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, 2) \
    GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, 3) \
    GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, 4) \
    GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN(STORAGE, ACCUM, RESULT, 5) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, 1) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, 2) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, 3) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, 4) \
    GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY(STORAGE, ACCUM, RESULT, 5)

GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(float, float, float)
GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(float, double, double)
GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(double, double, double)
GAFIME_HIP_INSTANTIATE_PRECISION_RANKING(float)
GAFIME_HIP_INSTANTIATE_PRECISION_RANKING(double)

#undef GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE
#undef GAFIME_HIP_INSTANTIATE_PRECISION_RANKING
#undef GAFIME_HIP_INSTANTIATE_PRECISION_MI_ARITY
#undef GAFIME_HIP_INSTANTIATE_PRECISION_MI_BIN
#undef GAFIME_HIP_INSTANTIATE_PRECISION_SPEARMAN
#undef GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS
#undef GAFIME_HIP_INSTANTIATE_PRECISION_CONTINUOUS_MODE
#undef GAFIME_HIP_INSTANTIATE_PRECISION_STATS

#undef GAFIME_HIP_INSTANTIATE_CONTINUOUS
#undef GAFIME_HIP_INSTANTIATE_SCALED_CONTINUOUS
#undef GAFIME_HIP_INSTANTIATE_SPEARMAN
#undef GAFIME_HIP_INSTANTIATE_MI_BIN
#undef GAFIME_HIP_INSTANTIATE_MI_ARITY
#undef GAFIME_HIP_FORCEINLINE
#undef GAFIME_HIP_STRINGIFY
#undef GAFIME_HIP_STRINGIFY_INNER


}  // namespace gafime_rocm_v1::kernel
