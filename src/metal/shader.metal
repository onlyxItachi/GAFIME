#include <metal_stdlib>

using namespace metal;

constant uint GAFIME_METRIC_PEARSON = 1;
constant uint GAFIME_METRIC_SPEARMAN = 2;
constant uint GAFIME_METRIC_MUTUAL_INFO = 3;
constant uint GAFIME_METRIC_R2 = 4;

// Metal has no fp64, so the reductions below accumulate in fp32. Parity
// tolerances account for backend-specific precision and reduction order. The
// mutual-information joint histogram lives in threadgroup memory; 48*48 uint =
// 9216 B fits the Apple
// ~32 KB threadgroup limit, so adaptive templates above 48 bins are clamped to
// 48 on Metal. MI and Spearman use fixed-width threadgroup reductions.
constant uint kMetalMaxMiBins = 48;
constant uint kMetalReduceWidth = 64;
constant uint kInvalidIndex = 0xffffffffu;

static inline uint fixed_mi_bin(
    float value,
    float minimum,
    float inverse_span,
    uint bins
) {
    const float scaled = (value - minimum) * inverse_span;
    if (isnan(scaled) || scaled <= 0.0f) {
        return 0;
    }
    const uint max_bin = bins - 1;
    if (!isfinite(scaled) || scaled >= static_cast<float>(max_bin)) {
        return max_bin;
    }
    return static_cast<uint>(scaled);
}

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

struct MetalRankInfo {
    ulong row_count;
    uint metric_count;
    uint primary_metric_index;
    uint top_k;
    uint partial_block_count;
};

static inline float centered_feature(
    device const float* features,
    device const float* column_means,
    ulong row,
    ulong rows,
    uint col
) {
    return features[static_cast<ulong>(col) * rows + row] - column_means[col];
}

static inline float interaction_value(
    device const float* features,
    device const float* column_means,
    ulong row,
    ulong rows,
    device const uint* combo,
    uint arity
) {
    switch (arity) {
    case 1:
        return features[static_cast<ulong>(combo[0]) * rows + row];
    case 2:
        return centered_feature(features, column_means, row, rows, combo[0]) *
            centered_feature(features, column_means, row, rows, combo[1]);
    case 3:
        return centered_feature(features, column_means, row, rows, combo[0]) *
            centered_feature(features, column_means, row, rows, combo[1]) *
            centered_feature(features, column_means, row, rows, combo[2]);
    case 4:
        return centered_feature(features, column_means, row, rows, combo[0]) *
            centered_feature(features, column_means, row, rows, combo[1]) *
            centered_feature(features, column_means, row, rows, combo[2]) *
            centered_feature(features, column_means, row, rows, combo[3]);
    case 5:
        return centered_feature(features, column_means, row, rows, combo[0]) *
            centered_feature(features, column_means, row, rows, combo[1]) *
            centered_feature(features, column_means, row, rows, combo[2]) *
            centered_feature(features, column_means, row, rows, combo[3]) *
            centered_feature(features, column_means, row, rows, combo[4]);
    default:
        break;
    }
    float value = 1.0f;
    for (uint idx = 0; idx < arity; ++idx) {
        const uint col = combo[idx];
        value *= features[static_cast<ulong>(col) * rows + row] - column_means[col];
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
    uint candidate [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    const ulong global_row = static_cast<ulong>(candidate);
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

    float local_sx = 0.0f;
    float local_sy = 0.0f;
    float local_count = 0.0f;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float x = interaction_value(features, column_means, row, info.rows, combo, selected->arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += x;
            local_sy += y;
            local_count += 1.0f;
        }
    }

    threadgroup float s_sx[kMetalReduceWidth];
    threadgroup float s_sy[kMetalReduceWidth];
    threadgroup float s_n[kMetalReduceWidth];
    threadgroup float s_sxx[kMetalReduceWidth];
    threadgroup float s_syy[kMetalReduceWidth];
    threadgroup float s_sxy[kMetalReduceWidth];
    threadgroup float mean_x;
    threadgroup float mean_y;

    s_sx[lane] = local_sx;
    s_sy[lane] = local_sy;
    s_n[lane] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_sx[lane] += s_sx[lane + stride];
            s_sy[lane] += s_sy[lane + stride];
            s_n[lane] += s_n[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        if (s_n[0] > 0.0f) {
            mean_x = s_sx[0] / s_n[0];
            mean_y = s_sy[0] / s_n[0];
        } else {
            mean_x = 0.0f;
            mean_y = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float x = interaction_value(features, column_means, row, info.rows, combo, selected->arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            const float dx = x - mean_x;
            const float dy = y - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }

    s_sxx[lane] = local_sxx;
    s_syy[lane] = local_syy;
    s_sxy[lane] = local_sxy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_sxx[lane] += s_sxx[lane + stride];
            s_syy[lane] += s_syy[lane + stride];
            s_sxy[lane] += s_sxy[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float pearson = 0.0f;
    if (lane == 0) {
        const float denom = sqrt(max(s_sxx[0] * s_syy[0], 0.0f));
        if (denom > 0.0f) {
            pearson = clamp(s_sxy[0] / denom, -1.0f, 1.0f);
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
    threadgroup float s_float0[kMetalReduceWidth];
    threadgroup float s_float1[kMetalReduceWidth];
    threadgroup float s_float2[kMetalReduceWidth];
    threadgroup float s_float3[kMetalReduceWidth];
    threadgroup uint s_uint0[kMetalReduceWidth];
    threadgroup uint s_uint1[kMetalReduceWidth];

    for (uint i = lane; i < bins; i += lane_count) {
        atomic_store_explicit(&hist_x[i], 0u, memory_order_relaxed);
        atomic_store_explicit(&hist_y[i], 0u, memory_order_relaxed);
    }
    for (uint i = lane; i < bins * bins; i += lane_count) {
        atomic_store_explicit(&joint[i], 0u, memory_order_relaxed);
    }
    float local_min_x = INFINITY;
    float local_max_x = -INFINITY;
    float local_min_y = INFINITY;
    float local_max_y = -INFINITY;
    uint local_valid = 0;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float x = interaction_value(features, column_means, row, info.rows, combo, arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_min_x = min(local_min_x, x);
            local_max_x = max(local_max_x, x);
            local_min_y = min(local_min_y, y);
            local_max_y = max(local_max_y, y);
            ++local_valid;
        }
    }
    s_float0[lane] = local_min_x;
    s_float1[lane] = local_max_x;
    s_float2[lane] = local_min_y;
    s_float3[lane] = local_max_y;
    s_uint0[lane] = local_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_float0[lane] = min(s_float0[lane], s_float0[lane + stride]);
            s_float1[lane] = max(s_float1[lane], s_float1[lane + stride]);
            s_float2[lane] = min(s_float2[lane], s_float2[lane + stride]);
            s_float3[lane] = max(s_float3[lane], s_float3[lane + stride]);
            s_uint0[lane] += s_uint0[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        g_min_x = s_float0[0];
        g_max_x = s_float1[0];
        g_min_y = s_float2[0];
        g_max_y = s_float3[0];
        g_valid = s_uint0[0];
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
        const float x = interaction_value(features, column_means, row, info.rows, combo, arity);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        const uint xb = fixed_mi_bin(x, g_min_x, inv_x, bins);
        const uint yb = fixed_mi_bin(y, g_min_y, inv_y, bins);
        atomic_fetch_add_explicit(&hist_x[xb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist_y[yb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&joint[xb * bins + yb], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float total = static_cast<float>(g_valid);
    float local_mi = 0.0f;
    uint local_active_x = 0;
    uint local_active_y = 0;
    for (uint idx = lane; idx < bins * bins; idx += lane_count) {
        const uint xb = idx / bins;
        const uint yb = idx - xb * bins;
        const uint count = atomic_load_explicit(&joint[idx], memory_order_relaxed);
        const uint hx = atomic_load_explicit(&hist_x[xb], memory_order_relaxed);
        const uint hy = atomic_load_explicit(&hist_y[yb], memory_order_relaxed);
        if (count == 0 || hx == 0 || hy == 0) {
            continue;
        }
        const float px = static_cast<float>(hx) / total;
        const float py = static_cast<float>(hy) / total;
        const float pxy = static_cast<float>(count) / total;
        local_mi += pxy * log(pxy / (px * py));
    }
    for (uint xb = lane; xb < bins; xb += lane_count) {
        if (atomic_load_explicit(&hist_x[xb], memory_order_relaxed) != 0) {
            ++local_active_x;
        }
    }
    for (uint yb = lane; yb < bins; yb += lane_count) {
        if (atomic_load_explicit(&hist_y[yb], memory_order_relaxed) != 0) {
            ++local_active_y;
        }
    }

    s_float0[lane] = local_mi;
    s_uint0[lane] = local_active_x;
    s_uint1[lane] = local_active_y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_float0[lane] += s_float0[lane + stride];
            s_uint0[lane] += s_uint0[lane + stride];
            s_uint1[lane] += s_uint1[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        const float mi = s_float0[0];
        const uint active_x = s_uint0[0];
        const uint active_y = s_uint1[0];
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
        const float xi = interaction_value(features, column_means, i, info.rows, combo, arity);
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        float less_x = 0.0f, eq_x = 0.0f, less_y = 0.0f, eq_y = 0.0f;
        for (ulong j = 0; j < info.rows; ++j) {
            const float xj = interaction_value(features, column_means, j, info.rows, combo, arity);
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

static inline bool candidate_better_desc(
    float candidate_score,
    uint candidate_index,
    float best_score,
    uint best_index
) {
    if (!isfinite(candidate_score)) {
        return false;
    }
    if (best_index == kInvalidIndex) {
        return true;
    }
    if (candidate_score > best_score) {
        return true;
    }
    if (candidate_score < best_score) {
        return false;
    }
    return candidate_index < best_index;
}

static inline bool candidate_better_asc(
    float candidate_score,
    uint candidate_index,
    float best_score,
    uint best_index
) {
    if (!isfinite(candidate_score)) {
        return false;
    }
    if (best_index == kInvalidIndex) {
        return true;
    }
    if (candidate_score < best_score) {
        return true;
    }
    if (candidate_score > best_score) {
        return false;
    }
    return candidate_index < best_index;
}

kernel void gafime_select_topk_partials_desc(
    device const float* metric_values [[buffer(0)]],
    device float* partial_scores [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant MetalRankInfo& rank [[buffer(3)]],
    uint partial_block [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    threadgroup float best_scores[kMetalReduceWidth];
    threadgroup uint best_indices[kMetalReduceWidth];
    threadgroup float previous_score;
    threadgroup uint previous_index;

    const ulong block_base = static_cast<ulong>(partial_block) * rank.top_k;
    const ulong stride = static_cast<ulong>(rank.partial_block_count) * lane_count;
    const ulong start = static_cast<ulong>(partial_block) * lane_count + lane;
    for (uint out_rank = 0; out_rank < rank.top_k; ++out_rank) {
        float local_score = -INFINITY;
        uint local_index = kInvalidIndex;
        for (ulong row = start; row < rank.row_count; row += stride) {
            const uint row_index = static_cast<uint>(row);
            const float score = metric_values[row * rank.metric_count + rank.primary_metric_index];
            if (out_rank != 0 && !candidate_better_desc(
                    previous_score,
                    previous_index,
                    score,
                    row_index)) {
                continue;
            }
            if (candidate_better_desc(score, row_index, local_score, local_index)) {
                local_score = score;
                local_index = row_index;
            }
        }

        best_scores[lane] = local_score;
        best_indices[lane] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float score = best_scores[lane + stride];
                const uint index = best_indices[lane + stride];
                if (candidate_better_desc(score, index, best_scores[lane], best_indices[lane])) {
                    best_scores[lane] = score;
                    best_indices[lane] = index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            partial_scores[block_base + out_rank] = best_scores[0];
            partial_indices[block_base + out_rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gafime_select_topk_partials_asc(
    device const float* metric_values [[buffer(0)]],
    device float* partial_scores [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant MetalRankInfo& rank [[buffer(3)]],
    uint partial_block [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    threadgroup float best_scores[kMetalReduceWidth];
    threadgroup uint best_indices[kMetalReduceWidth];
    threadgroup float previous_score;
    threadgroup uint previous_index;

    const ulong block_base = static_cast<ulong>(partial_block) * rank.top_k;
    const ulong stride = static_cast<ulong>(rank.partial_block_count) * lane_count;
    const ulong start = static_cast<ulong>(partial_block) * lane_count + lane;
    for (uint out_rank = 0; out_rank < rank.top_k; ++out_rank) {
        float local_score = INFINITY;
        uint local_index = kInvalidIndex;
        for (ulong row = start; row < rank.row_count; row += stride) {
            const uint row_index = static_cast<uint>(row);
            const float score = metric_values[row * rank.metric_count + rank.primary_metric_index];
            if (out_rank != 0 && !candidate_better_asc(
                    previous_score,
                    previous_index,
                    score,
                    row_index)) {
                continue;
            }
            if (candidate_better_asc(score, row_index, local_score, local_index)) {
                local_score = score;
                local_index = row_index;
            }
        }

        best_scores[lane] = local_score;
        best_indices[lane] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float score = best_scores[lane + stride];
                const uint index = best_indices[lane + stride];
                if (candidate_better_asc(score, index, best_scores[lane], best_indices[lane])) {
                    best_scores[lane] = score;
                    best_indices[lane] = index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            partial_scores[block_base + out_rank] = best_scores[0];
            partial_indices[block_base + out_rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gafime_merge_topk_partials_desc(
    device const float* partial_scores [[buffer(0)]],
    device const uint* partial_indices [[buffer(1)]],
    device uint* selected_indices [[buffer(2)]],
    constant MetalRankInfo& rank [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    threadgroup float best_scores[kMetalReduceWidth];
    threadgroup uint best_indices[kMetalReduceWidth];
    threadgroup float previous_score;
    threadgroup uint previous_index;

    const ulong partial_count = static_cast<ulong>(rank.partial_block_count) * rank.top_k;
    for (uint out_rank = 0; out_rank < rank.top_k; ++out_rank) {
        float local_score = -INFINITY;
        uint local_index = kInvalidIndex;
        for (ulong item = lane; item < partial_count; item += lane_count) {
            const uint candidate_index = partial_indices[item];
            if (candidate_index == kInvalidIndex) {
                continue;
            }
            const float score = partial_scores[item];
            if (out_rank != 0 && !candidate_better_desc(
                    previous_score,
                    previous_index,
                    score,
                    candidate_index)) {
                continue;
            }
            if (candidate_better_desc(score, candidate_index, local_score, local_index)) {
                local_score = score;
                local_index = candidate_index;
            }
        }

        best_scores[lane] = local_score;
        best_indices[lane] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float score = best_scores[lane + stride];
                const uint index = best_indices[lane + stride];
                if (candidate_better_desc(score, index, best_scores[lane], best_indices[lane])) {
                    best_scores[lane] = score;
                    best_indices[lane] = index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            selected_indices[out_rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gafime_merge_topk_partials_asc(
    device const float* partial_scores [[buffer(0)]],
    device const uint* partial_indices [[buffer(1)]],
    device uint* selected_indices [[buffer(2)]],
    constant MetalRankInfo& rank [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    threadgroup float best_scores[kMetalReduceWidth];
    threadgroup uint best_indices[kMetalReduceWidth];
    threadgroup float previous_score;
    threadgroup uint previous_index;

    const ulong partial_count = static_cast<ulong>(rank.partial_block_count) * rank.top_k;
    for (uint out_rank = 0; out_rank < rank.top_k; ++out_rank) {
        float local_score = INFINITY;
        uint local_index = kInvalidIndex;
        for (ulong item = lane; item < partial_count; item += lane_count) {
            const uint candidate_index = partial_indices[item];
            if (candidate_index == kInvalidIndex) {
                continue;
            }
            const float score = partial_scores[item];
            if (out_rank != 0 && !candidate_better_asc(
                    previous_score,
                    previous_index,
                    score,
                    candidate_index)) {
                continue;
            }
            if (candidate_better_asc(score, candidate_index, local_score, local_index)) {
                local_score = score;
                local_index = candidate_index;
            }
        }

        best_scores[lane] = local_score;
        best_indices[lane] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float score = best_scores[lane + stride];
                const uint index = best_indices[lane + stride];
                if (candidate_better_asc(score, index, best_scores[lane], best_indices[lane])) {
                    best_scores[lane] = score;
                    best_indices[lane] = index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            selected_indices[out_rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gafime_copy_selected_metric_rows(
    device const float* metric_values [[buffer(0)]],
    device const uint* selected_indices [[buffer(1)]],
    device float* selected_metric_values [[buffer(2)]],
    constant MetalRankInfo& rank [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const ulong total = static_cast<ulong>(rank.top_k) * rank.metric_count;
    const ulong idx = static_cast<ulong>(gid);
    if (idx >= total) {
        return;
    }
    const ulong selected_row = idx / rank.metric_count;
    const uint metric_idx = static_cast<uint>(idx - selected_row * rank.metric_count);
    const uint source_row = selected_indices[selected_row];
    if (source_row == kInvalidIndex || source_row >= rank.row_count) {
        selected_metric_values[idx] = 0.0f;
        return;
    }
    selected_metric_values[idx] =
        metric_values[static_cast<ulong>(source_row) * rank.metric_count + metric_idx];
}
