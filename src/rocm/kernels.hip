#include "kernels.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_rocm_v1::kernel {

constexpr int kThreadsPerBlock = gafime_rocm_v1::kThreadsPerBlock;
constexpr uint32_t kMaxMutualInfoBins = gafime_rocm_v1::kMaxMutualInfoBins;

__device__ float interaction_value(
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
        float pearson = 0.0f;
        const float denom = sqrtf(fmaxf(feature_sxx * target_stats->syy, 0.0f));
        if (denom > 0.0f) {
            pearson = fminf(fmaxf(sxy[0] / denom, -1.0f), 1.0f);
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
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
        float pearson = 0.0f;
        const float denom = sqrtf(fmaxf(sxx[0] * syy[0], 0.0f));
        if (denom > 0.0f) {
            pearson = fminf(fmaxf(sxy[0] / denom, -1.0f), 1.0f);
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

// Mutual-information kernel: one block per candidate, adaptive-range binning into
// shared histograms, finite-sample-corrected + normalized MI. Ported from the
// CUDA host (HIP is source-compatible: atomicAdd / __syncthreads / logf / min).
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
        uint32_t xb = static_cast<uint32_t>((x - min_x) * inv_x);
        uint32_t yb = static_cast<uint32_t>((y - min_y) * inv_y);
        xb = min(xb, static_cast<uint32_t>(bins - 1));
        yb = min(yb, static_cast<uint32_t>(bins - 1));
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
            const double denom = sqrt(vx * vy);
            if (denom > 0.0) {
                double r = cov / denom;
                r = fmax(-1.0, fmin(1.0, r));
                out = static_cast<float>(r);
            }
        }
        metric_values[combo_row * metric_count + metric_index] = out;
    }
}


}  // namespace gafime_rocm_v1::kernel
