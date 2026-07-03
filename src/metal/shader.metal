#include <metal_stdlib>

using namespace metal;

constant uint GAFIME_METRIC_PEARSON = 1;
constant uint GAFIME_METRIC_SPEARMAN = 2;
constant uint GAFIME_METRIC_MUTUAL_INFO = 3;
constant uint GAFIME_METRIC_R2 = 4;

// Metal has no fp64, so the reductions below accumulate in fp32 (documented
// tolerance vs the f64 CUDA/CPU parity oracle). The mutual-information joint
// histogram lives in threadgroup memory; 48*48 uint = 9216 B fits the Apple
// ~32 KB threadgroup limit, so the requested bins (12/24/48/96) are clamped to
// <= 48 on Metal. Spearman uses a fixed-width threadgroup reduction.
constant uint kMetalMaxMiBins = 48;
constant uint kMetalReduceWidth = 64;

struct MetalChunk {
    uint arity;
    uint mi_bins;
    ulong descriptor_offset;
    ulong combo_count;
    ulong global_row_offset;
};

// Map a global candidate row to its chunk; returns the chunk index or -1.
static inline int locate_candidate_index(
    device const MetalChunk* chunks,
    uint chunk_count,
    ulong candidate,
    thread ulong& local_row
) {
    for (uint idx = 0; idx < chunk_count; ++idx) {
        const MetalChunk chunk = chunks[idx];
        if (candidate >= chunk.global_row_offset &&
            candidate < chunk.global_row_offset + chunk.combo_count) {
            local_row = candidate - chunk.global_row_offset;
            return static_cast<int>(idx);
        }
    }
    return -1;
}

struct MetalLaunchInfo {
    ulong rows;
    uint cols;
    uint metric_count;
    uint chunk_count;
};

static inline float interaction_value(
    device const float* features,
    device const float* column_means,
    ulong row,
    uint cols,
    device const uint* combo,
    uint arity
) {
    if (arity == 1) {
        return features[row * cols + combo[0]];
    }
    float value = 1.0f;
    for (uint idx = 0; idx < arity; ++idx) {
        const uint col = combo[idx];
        value *= features[row * cols + col] - column_means[col];
    }
    return value;
}

kernel void gafime_score_continuous(
    device const float* features [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device const float* column_means [[buffer(2)]],
    device const uint* combo_indices [[buffer(3)]],
    device const uint* metric_ids [[buffer(4)]],
    device const MetalChunk* chunks [[buffer(5)]],
    device float* metric_values [[buffer(6)]],
    constant MetalLaunchInfo& info [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const ulong global_row = static_cast<ulong>(gid);
    device const MetalChunk* selected = nullptr;
    ulong local_row = 0;
    for (uint chunk_idx = 0; chunk_idx < info.chunk_count; ++chunk_idx) {
        const MetalChunk chunk = chunks[chunk_idx];
        if (global_row >= chunk.global_row_offset &&
            global_row < chunk.global_row_offset + chunk.combo_count) {
            selected = chunks + chunk_idx;
            local_row = global_row - chunk.global_row_offset;
            break;
        }
    }
    if (selected == nullptr) {
        return;
    }

    device const uint* combo =
        combo_indices + selected->descriptor_offset + local_row * selected->arity;

    float sx = 0.0f;
    float sy = 0.0f;
    float count = 0.0f;
    for (ulong row = 0; row < info.rows; ++row) {
        const float x = interaction_value(features, column_means, row, info.cols, combo, selected->arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            sx += x;
            sy += y;
            count += 1.0f;
        }
    }

    float pearson = 0.0f;
    if (count > 0.0f) {
        const float mean_x = sx / count;
        const float mean_y = sy / count;
        float sxx = 0.0f;
        float syy = 0.0f;
        float sxy = 0.0f;
        for (ulong row = 0; row < info.rows; ++row) {
            const float x = interaction_value(features, column_means, row, info.cols, combo, selected->arity);
            const float y = target[row];
            if (isfinite(x) && isfinite(y)) {
                const float dx = x - mean_x;
                const float dy = y - mean_y;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
        }
        const float denom = sqrt(max(sxx * syy, 0.0f));
        if (denom > 0.0f) {
            pearson = clamp(sxy / denom, -1.0f, 1.0f);
        }
    }

    for (uint metric_idx = 0; metric_idx < info.metric_count; ++metric_idx) {
        const uint metric_id = metric_ids[metric_idx];
        float out = 0.0f;
        if (metric_id == GAFIME_METRIC_PEARSON) {
            out = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            out = clamp(pearson * pearson, 0.0f, 1.0f);
        }
        metric_values[global_row * info.metric_count + metric_idx] = out;
    }
}

// Fixed-bin mutual information, one threadgroup per candidate. Mirrors the CUDA
// score_mutual_info_chunk_kernel algorithm (min/max scan -> equal-width binning
// -> joint histogram -> bias-corrected, normalized MI). fp32 accumulation and a
// <= kMetalMaxMiBins bin clamp are the Metal-specific tolerances (see header).
kernel void gafime_score_mutual_info(
    device const float* features [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device const float* column_means [[buffer(2)]],
    device const uint* combo_indices [[buffer(3)]],
    device const uint* metric_ids [[buffer(4)]],
    device const MetalChunk* chunks [[buffer(5)]],
    device float* metric_values [[buffer(6)]],
    constant MetalLaunchInfo& info [[buffer(7)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    int metric_index = -1;
    for (uint m = 0; m < info.metric_count; ++m) {
        if (metric_ids[m] == GAFIME_METRIC_MUTUAL_INFO) {
            metric_index = static_cast<int>(m);
            break;
        }
    }
    if (metric_index < 0) {
        return;
    }
    ulong local_row = 0;
    const int ci = locate_candidate_index(chunks, info.chunk_count, static_cast<ulong>(candidate), local_row);
    if (ci < 0) {
        return;
    }
    const MetalChunk chunk = chunks[ci];
    const uint arity = chunk.arity;
    device const uint* combo = combo_indices + chunk.descriptor_offset + local_row * arity;
    uint bins = chunk.mi_bins;
    bins = bins < 2 ? 2 : (bins > kMetalMaxMiBins ? kMetalMaxMiBins : bins);

    threadgroup atomic_uint hist_x[kMetalMaxMiBins];
    threadgroup atomic_uint hist_y[kMetalMaxMiBins];
    threadgroup atomic_uint joint[kMetalMaxMiBins * kMetalMaxMiBins];
    threadgroup float g_min_x;
    threadgroup float g_max_x;
    threadgroup float g_min_y;
    threadgroup float g_max_y;
    threadgroup uint g_valid;

    for (uint i = lane; i < bins; i += lane_count) {
        atomic_store_explicit(&hist_x[i], 0u, memory_order_relaxed);
        atomic_store_explicit(&hist_y[i], 0u, memory_order_relaxed);
    }
    for (uint i = lane; i < bins * bins; i += lane_count) {
        atomic_store_explicit(&joint[i], 0u, memory_order_relaxed);
    }
    if (lane == 0) {
        float min_x = INFINITY;
        float max_x = -INFINITY;
        float min_y = INFINITY;
        float max_y = -INFINITY;
        uint valid = 0;
        for (ulong row = 0; row < info.rows; ++row) {
            const float x = interaction_value(features, column_means, row, info.cols, combo, arity);
            const float y = target[row];
            if (isfinite(x) && isfinite(y)) {
                min_x = min(min_x, x);
                max_x = max(max_x, x);
                min_y = min(min_y, y);
                max_y = max(max_y, y);
                ++valid;
            }
        }
        g_min_x = min_x;
        g_max_x = max_x;
        g_min_y = min_y;
        g_max_y = max_y;
        g_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (g_valid <= 1 || g_max_x <= g_min_x || g_max_y <= g_min_y) {
        if (lane == 0) {
            metric_values[candidate * info.metric_count + metric_index] = 0.0f;
        }
        return;
    }

    const float inv_x = static_cast<float>(bins) / (g_max_x - g_min_x);
    const float inv_y = static_cast<float>(bins) / (g_max_y - g_min_y);
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float x = interaction_value(features, column_means, row, info.cols, combo, arity);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        uint xb = static_cast<uint>((x - g_min_x) * inv_x);
        uint yb = static_cast<uint>((y - g_min_y) * inv_y);
        xb = min(xb, bins - 1);
        yb = min(yb, bins - 1);
        atomic_fetch_add_explicit(&hist_x[xb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist_y[yb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&joint[xb * bins + yb], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const float total = static_cast<float>(g_valid);
        float mi = 0.0f;
        uint active_x = 0;
        uint active_y = 0;
        for (uint xb = 0; xb < bins; ++xb) {
            const uint hx = atomic_load_explicit(&hist_x[xb], memory_order_relaxed);
            if (hx == 0) {
                continue;
            }
            ++active_x;
            const float px = static_cast<float>(hx) / total;
            for (uint yb = 0; yb < bins; ++yb) {
                const uint count = atomic_load_explicit(&joint[xb * bins + yb], memory_order_relaxed);
                const uint hy = atomic_load_explicit(&hist_y[yb], memory_order_relaxed);
                if (count == 0 || hy == 0) {
                    continue;
                }
                const float py = static_cast<float>(hy) / total;
                const float pxy = static_cast<float>(count) / total;
                mi += pxy * log(pxy / (px * py));
            }
        }
        for (uint yb = 0; yb < bins; ++yb) {
            if (atomic_load_explicit(&hist_y[yb], memory_order_relaxed) != 0) {
                ++active_y;
            }
        }
        const float correction = active_x > 0 && active_y > 0
            ? static_cast<float>((active_x - 1) * (active_y - 1)) / (2.0f * total)
            : 0.0f;
        const float corrected = max(0.0f, mi - correction);
        const uint normalizer_bins = min(active_x, active_y);
        const float normalizer = normalizer_bins > 1 ? log(static_cast<float>(normalizer_bins)) : 0.0f;
        metric_values[candidate * info.metric_count + metric_index] =
            normalizer > 0.0f ? corrected / normalizer : 0.0f;
    }
}

// Spearman = Pearson on average-tie ranks, one threadgroup per candidate. Ranks
// are counted (rank_i = #less + 0.5*(#equal - 1)) to match the CPU/CUDA rankdata
// exactly; the pearson-of-ranks is reduced across lanes. O(n^2) per candidate
// (correctness-first). fp32 accumulation is the Metal tolerance (see header).
kernel void gafime_score_spearman(
    device const float* features [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device const float* column_means [[buffer(2)]],
    device const uint* combo_indices [[buffer(3)]],
    device const uint* metric_ids [[buffer(4)]],
    device const MetalChunk* chunks [[buffer(5)]],
    device float* metric_values [[buffer(6)]],
    constant MetalLaunchInfo& info [[buffer(7)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    int metric_index = -1;
    for (uint m = 0; m < info.metric_count; ++m) {
        if (metric_ids[m] == GAFIME_METRIC_SPEARMAN) {
            metric_index = static_cast<int>(m);
            break;
        }
    }
    if (metric_index < 0) {
        return;
    }
    ulong local_row = 0;
    const int ci = locate_candidate_index(chunks, info.chunk_count, static_cast<ulong>(candidate), local_row);
    if (ci < 0) {
        return;
    }
    const MetalChunk chunk = chunks[ci];
    const uint arity = chunk.arity;
    device const uint* combo = combo_indices + chunk.descriptor_offset + local_row * arity;

    float l_srx = 0.0f, l_sry = 0.0f, l_srxx = 0.0f, l_sryy = 0.0f, l_srxy = 0.0f, l_n = 0.0f;
    for (ulong i = lane; i < info.rows; i += lane_count) {
        const float xi = interaction_value(features, column_means, i, info.cols, combo, arity);
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        float less_x = 0.0f, eq_x = 0.0f, less_y = 0.0f, eq_y = 0.0f;
        for (ulong j = 0; j < info.rows; ++j) {
            const float xj = interaction_value(features, column_means, j, info.cols, combo, arity);
            const float yj = target[j];
            if (!isfinite(xj) || !isfinite(yj)) {
                continue;
            }
            if (xj < xi) {
                less_x += 1.0f;
            } else if (xj == xi) {
                eq_x += 1.0f;
            }
            if (yj < yi) {
                less_y += 1.0f;
            } else if (yj == yi) {
                eq_y += 1.0f;
            }
        }
        const float rx = less_x + 0.5f * (eq_x - 1.0f);
        const float ry = less_y + 0.5f * (eq_y - 1.0f);
        l_srx += rx;
        l_sry += ry;
        l_srxx += rx * rx;
        l_sryy += ry * ry;
        l_srxy += rx * ry;
        l_n += 1.0f;
    }

    threadgroup float s_srx[kMetalReduceWidth];
    threadgroup float s_sry[kMetalReduceWidth];
    threadgroup float s_srxx[kMetalReduceWidth];
    threadgroup float s_sryy[kMetalReduceWidth];
    threadgroup float s_srxy[kMetalReduceWidth];
    threadgroup float s_n[kMetalReduceWidth];
    s_srx[lane] = l_srx;
    s_sry[lane] = l_sry;
    s_srxx[lane] = l_srxx;
    s_sryy[lane] = l_sryy;
    s_srxy[lane] = l_srxy;
    s_n[lane] = l_n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_srx[lane] += s_srx[lane + stride];
            s_sry[lane] += s_sry[lane + stride];
            s_srxx[lane] += s_srxx[lane + stride];
            s_sryy[lane] += s_sryy[lane + stride];
            s_srxy[lane] += s_srxy[lane + stride];
            s_n[lane] += s_n[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float n = s_n[0];
        float out = 0.0f;
        if (n > 1.0f) {
            const float cov = n * s_srxy[0] - s_srx[0] * s_sry[0];
            const float vx = n * s_srxx[0] - s_srx[0] * s_srx[0];
            const float vy = n * s_sryy[0] - s_sry[0] * s_sry[0];
            const float denom = sqrt(vx * vy);
            if (denom > 0.0f) {
                out = clamp(cov / denom, -1.0f, 1.0f);
            }
        }
        metric_values[candidate * info.metric_count + metric_index] = out;
    }
}
