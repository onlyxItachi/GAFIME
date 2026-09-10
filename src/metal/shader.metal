#include <metal_stdlib>

using namespace metal;

constant uint GAFIME_METRIC_PEARSON = 1;
constant uint GAFIME_METRIC_SPEARMAN = 2;
constant uint GAFIME_METRIC_MUTUAL_INFO = 3;
constant uint GAFIME_METRIC_R2 = 4;
constant uint GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE = 0x1u;
constant uint GAFIME_PRECISION_FP32 = 1;
constant uint GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE = 2;
constant uint GAFIME_SEMANTIC_PROGRAM_SOFTSIGN = 3;
constant uint GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT = 4;
constant uint GAFIME_SEMANTIC_REGION_LESS_EQUAL = 1;
constant uint GAFIME_SEMANTIC_REGION_GREATER_THAN = 2;
constant uint GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE = 2;
constant uint GAFIME_SEMANTIC_SCALAR_MEASURED = 1;
constant uint GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT = 2;
constant uint GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND = 3;
constant uint GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION = 4;
constant uint GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION = 5;

// Metal has no fp64, so the reductions below accumulate in fp32. Parity
// tolerances account for backend-specific precision and reduction order. The
// mutual-information joint histogram lives in threadgroup memory; 48*48 uint =
// 9216 B fits the Apple
// ~32 KB threadgroup limit, so adaptive templates above 48 bins are clamped to
// 48 on Metal. MI and Spearman use fixed-width threadgroup reductions.
constant uint kMetalMaxMiBins = 48;
constant uint kMetalReduceWidth = 64;
constant uint kInvalidIndex = 0xffffffffu;

static inline float nonfinite_metric() {
    return as_type<float>(0x7fc00000u);
}

static inline float finalize_correlation(float variance_x, float variance_y, float covariance) {
    if (!isfinite(variance_x) || !isfinite(variance_y) || !isfinite(covariance)) {
        return nonfinite_metric();
    }
    if (variance_x == 0.0f || variance_y == 0.0f) {
        return 0.0f;
    }
    if (variance_x < 0.0f || variance_y < 0.0f) {
        return nonfinite_metric();
    }
    const float denom = sqrt(variance_x * variance_y);
    if (!isfinite(denom) || denom <= 0.0f) {
        return nonfinite_metric();
    }
    const float correlation = covariance / denom;
    return isfinite(correlation) ? clamp(correlation, -1.0f, 1.0f) : nonfinite_metric();
}

static inline float finalize_r2(float pearson) {
    return isfinite(pearson) ? clamp(pearson * pearson, 0.0f, 1.0f) : nonfinite_metric();
}

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
    uint scaled_covariance;
    uint reserved;
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
    uint precision_profile;
};

struct MetalRankInfo {
    ulong row_count;
    uint metric_count;
    uint primary_metric_index;
    uint top_k;
    uint partial_block_count;
};

struct MetalInteractionDiagnosticInfo {
    ulong rows;
    uint max_arity;
    uint combo_count;
};

// These structures are private to the optional semantic arithmetic table.  The
// public C ABI descriptors are copied into compact immutable Metal buffers by
// semantic_launcher.mm, so no semantic/program identity crosses this boundary.
struct MetalSemanticProgramNode {
    uint opcode;
    uint output_slot;
    uint operand_offset;
    uint operand_count;
    uint mean_offset;
    uint mean_count;
    uint region_term_offset;
    uint region_term_count;
};

struct MetalSemanticRegionTerm {
    uint input_slot;
    uint relation;
    ulong threshold_bits;
};

struct MetalSemanticRowsInfo {
    ulong rows;
    uint item_count;
    uint reserved;
};

struct MetalSemanticAssociationInfo {
    ulong rows;
    ulong pair_count;
    uint presentation;
    uint bins;
    uint padded_rows;
    uint reserved;
};

struct MetalSemanticRankInfo {
    ulong rows;
    ulong pair_count;
    uint padded_rows;
    uint stage;
    uint stride;
    uint reserved;
};

struct MetalSemanticEdgeInfo {
    ulong rows;
    ulong edge_count;
    ulong candidate_count;
    ulong reserved;
};

struct MetalSemanticGatherInfo {
    ulong source_rows;
    ulong destination_rows;
    ulong slot_count;
    ulong reserved;
};

struct MetalSemanticEdge {
    ulong left_row;
    ulong right_row;
};

struct MetalSemanticRankRecord {
    float value;
    uint row;
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

// This scans only diagnostic combos that the host-side exponent envelope cannot
// prove finite. Keep the centered subtraction and left-to-right multiplication
// order aligned with interaction_value above: diagnostic counts describe the
// same fp32 materialization used by scoring, not a widened approximation.
kernel void gafime_interaction_diagnostics(
    device const float* features [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device const float* column_means [[buffer(2)]],
    device const uint* combo_indices [[buffer(3)]],
    device ulong* overflow_row_counts [[buffer(4)]],
    device uint* flags [[buffer(5)]],
    constant MetalInteractionDiagnosticInfo& info [[buffer(6)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    if (candidate >= info.combo_count) {
        return;
    }

    device const uint* combo =
        combo_indices + static_cast<ulong>(candidate) * info.max_arity;
    uint arity = 0;
    while (arity < info.max_arity && combo[arity] != kInvalidIndex) {
        ++arity;
    }

    ulong local_overflow_count = 0;
    uint local_source_nonfinite = 0;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        bool source_nonfinite = !isfinite(target[row]);
        bool finite_feature_inputs = true;
        for (uint idx = 0; idx < arity; ++idx) {
            const uint col = combo[idx];
            const float raw = features[static_cast<ulong>(col) * info.rows + row];
            const float mean = column_means[col];
            if (!isfinite(raw) || !isfinite(mean)) {
                source_nonfinite = true;
                finite_feature_inputs = false;
                break;
            }
        }
        if (source_nonfinite) {
            local_source_nonfinite = 1;
        }
        if (!finite_feature_inputs || arity <= 1) {
            continue;
        }

        bool overflowed = false;
        float product = 0.0f;
        for (uint idx = 0; idx < arity; ++idx) {
            const uint col = combo[idx];
            const float centered =
                features[static_cast<ulong>(col) * info.rows + row] - column_means[col];
            if (!isfinite(centered)) {
                overflowed = true;
                break;
            }
            if (idx == 0) {
                product = centered;
            } else {
                product *= centered;
                if (!isfinite(product)) {
                    overflowed = true;
                    break;
                }
            }
        }
        if (overflowed) {
            ++local_overflow_count;
        }
    }

    threadgroup ulong overflow_counts[kMetalReduceWidth];
    threadgroup uint source_nonfinite_flags[kMetalReduceWidth];
    overflow_counts[lane] = local_overflow_count;
    source_nonfinite_flags[lane] = local_source_nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            overflow_counts[lane] += overflow_counts[lane + stride];
            source_nonfinite_flags[lane] |= source_nonfinite_flags[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        overflow_row_counts[candidate] = overflow_counts[0];
        flags[candidate] = source_nonfinite_flags[0] == 0
            ? 0u
            : GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
    }
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

    threadgroup float s_sx[kMetalReduceWidth];
    threadgroup float s_sy[kMetalReduceWidth];
    threadgroup ulong s_n[kMetalReduceWidth];
    threadgroup float s_sxx[kMetalReduceWidth];
    threadgroup float s_syy[kMetalReduceWidth];
    threadgroup float s_sxy[kMetalReduceWidth];
    threadgroup float mean_x;
    threadgroup float mean_y;
    threadgroup float scale_x;
    threadgroup float scale_y;

    const bool scaled_covariance = selected->scaled_covariance != 0;
    // The ABI 1.1 fp32 oracle sums short vectors in row order. Keep the legacy
    // ABI 1.0 route unchanged, and retain the parallel reduction for larger
    // workloads where serialization would compromise the throughput lane.
    const bool core_ordered_fp32_mean =
        info.precision_profile == GAFIME_PRECISION_FP32 &&
        info.rows <= static_cast<ulong>(4u * kMetalReduceWidth);
    float local_sx = 0.0f;
    float local_sy = 0.0f;
    ulong local_count = 0;
    if (scaled_covariance) {
        for (ulong row = lane; row < info.rows; row += lane_count) {
            const float x = interaction_value(
                features, column_means, row, info.rows, combo, selected->arity);
            const float y = target[row];
            if (isfinite(x) && isfinite(y)) {
                local_sx = max(local_sx, fabs(x));
                local_sy = max(local_sy, fabs(y));
                ++local_count;
            }
        }

        s_sx[lane] = local_sx;
        s_sy[lane] = local_sy;
        s_n[lane] = local_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                s_sx[lane] = max(s_sx[lane], s_sx[lane + stride]);
                s_sy[lane] = max(s_sy[lane], s_sy[lane + stride]);
                s_n[lane] += s_n[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            scale_x = s_sx[0];
            scale_y = s_sy[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (core_ordered_fp32_mean) {
            if (lane == 0) {
                local_sx = 0.0f;
                local_sy = 0.0f;
                local_count = 0;
                for (ulong row = 0; row < info.rows; ++row) {
                    const float x = interaction_value(
                        features, column_means, row, info.rows, combo, selected->arity);
                    const float y = target[row];
                    if (isfinite(x) && isfinite(y)) {
                        local_sx += scale_x > 0.0f ? x / scale_x : 0.0f;
                        local_sy += scale_y > 0.0f ? y / scale_y : 0.0f;
                        ++local_count;
                    }
                }
                s_sx[0] = local_sx;
                s_sy[0] = local_sy;
                s_n[0] = local_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            local_sx = 0.0f;
            local_sy = 0.0f;
            for (ulong row = lane; row < info.rows; row += lane_count) {
                const float x = interaction_value(
                    features, column_means, row, info.rows, combo, selected->arity);
                const float y = target[row];
                if (isfinite(x) && isfinite(y)) {
                    local_sx += scale_x > 0.0f ? x / scale_x : 0.0f;
                    local_sy += scale_y > 0.0f ? y / scale_y : 0.0f;
                }
            }

            s_sx[lane] = local_sx;
            s_sy[lane] = local_sy;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    s_sx[lane] += s_sx[lane + stride];
                    s_sy[lane] += s_sy[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        if (core_ordered_fp32_mean) {
            if (lane == 0) {
                for (ulong row = 0; row < info.rows; ++row) {
                    const float x = interaction_value(
                        features, column_means, row, info.rows, combo, selected->arity);
                    const float y = target[row];
                    if (isfinite(x) && isfinite(y)) {
                        local_sx += x;
                        local_sy += y;
                        ++local_count;
                    }
                }
                s_sx[0] = local_sx;
                s_sy[0] = local_sy;
                s_n[0] = local_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (ulong row = lane; row < info.rows; row += lane_count) {
                const float x = interaction_value(
                    features, column_means, row, info.rows, combo, selected->arity);
                const float y = target[row];
                if (isfinite(x) && isfinite(y)) {
                    local_sx += x;
                    local_sy += y;
                    ++local_count;
                }
            }

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
        }
    }

    if (lane == 0) {
        if (s_n[0] > 0) {
            const float count = static_cast<float>(s_n[0]);
            mean_x = s_sx[0] / count;
            mean_y = s_sy[0] / count;
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
            const float covariance_x = scaled_covariance && scale_x > 0.0f ? x / scale_x : x;
            const float covariance_y = scaled_covariance && scale_y > 0.0f ? y / scale_y : y;
            const float dx = covariance_x - mean_x;
            const float dy = covariance_y - mean_y;
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

    if (lane == 0) {
        const float pearson = finalize_correlation(s_sxx[0], s_syy[0], s_sxy[0]);

        for (uint metric_idx = 0; metric_idx < info.metric_count; ++metric_idx) {
            const uint metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = finalize_r2(pearson);
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
    threadgroup float s_float0[kMetalReduceWidth];
    threadgroup float s_float1[kMetalReduceWidth];
    threadgroup float s_float2[kMetalReduceWidth];
    threadgroup float s_float3[kMetalReduceWidth];
    threadgroup uint s_uint0[kMetalReduceWidth];

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
    const float min_x = s_float0[0];
    const float max_x = s_float1[0];
    const float min_y = s_float2[0];
    const float max_y = s_float3[0];
    const uint valid = s_uint0[0];

    if (valid <= 1 || max_x <= min_x || max_y <= min_y) {
        if (lane == 0) {
            metric_values[candidate * info.metric_count + metric_index] = 0.0f;
        }
        return;
    }

    const float inv_x = static_cast<float>(bins) / (max_x - min_x);
    const float inv_y = static_cast<float>(bins) / (max_y - min_y);
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float x = interaction_value(features, column_means, row, info.rows, combo, arity);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        const uint xb = fixed_mi_bin(x, min_x, inv_x, bins);
        const uint yb = fixed_mi_bin(y, min_y, inv_y, bins);
        atomic_fetch_add_explicit(&hist_x[xb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist_y[yb], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&joint[xb * bins + yb], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        // Histogram construction remains parallel and count-exact. Finalize in
        // the same row-major bin order as Core/CUDA/ROCm so the fp32 score has a
        // deterministic, backend-comparable reduction order.
        const float total = static_cast<float>(valid);
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
                const uint count = atomic_load_explicit(
                    &joint[xb * bins + yb], memory_order_relaxed);
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

// The finite-unary path reuses these exact count-based target ranks across a
// batch. It deliberately does not sort: ties retain the same average-rank
// calculation as the pairwise fallback.
kernel void gafime_build_spearman_target_ranks(
    device const float* target [[buffer(0)]],
    device uint* target_ranks_twice [[buffer(1)]],
    constant MetalLaunchInfo& info [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (static_cast<ulong>(row) >= info.rows) {
        return;
    }
    const float yi = target[row];
    if (!isfinite(yi)) {
        target_ranks_twice[row] = 0;
        return;
    }
    uint less_y = 0;
    uint eq_y = 0;
    for (ulong j = 0; j < info.rows; ++j) {
        const float yj = target[j];
        if (!isfinite(yj)) {
            continue;
        }
        if (yj < yi) {
            ++less_y;
        } else if (yj == yi) {
            ++eq_y;
        }
    }
    target_ranks_twice[row] = less_y * 2 + eq_y - 1;
}

// Spearman = Pearson on average-tie ranks, one threadgroup per candidate. Ranks
// are counted (rank_i = #less + 0.5*(#equal - 1)) to match the CPU/CUDA rankdata
// exactly; the pearson-of-ranks is reduced across lanes. The finite-unary path
// reuses target ranks; every other shape retains the O(n^2) pairwise fallback.
// fp32 accumulation is the Metal tolerance (see header).
kernel void gafime_score_spearman(
    device const float* features [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device const float* column_means [[buffer(2)]],
    device const uint* combo_indices [[buffer(3)]],
    device const uint* metric_ids [[buffer(4)]],
    device const MetalChunk* chunks [[buffer(5)]],
    device float* metric_values [[buffer(6)]],
    constant MetalLaunchInfo& info [[buffer(7)]],
    device const uint* target_ranks_twice [[buffer(8)]],
    constant uint& use_cached_target_ranks [[buffer(9)]],
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
    const bool use_cached_target_ranks_for_candidate =
        use_cached_target_ranks != 0 && arity == 1;

    float l_srx = 0.0f, l_sry = 0.0f, l_srxx = 0.0f, l_sryy = 0.0f, l_srxy = 0.0f;
    ulong l_n = 0;
    for (ulong i = lane; i < info.rows; i += lane_count) {
        const float xi = interaction_value(features, column_means, i, info.rows, combo, arity);
        const float yi = target[i];
        if (!isfinite(xi) || !isfinite(yi)) {
            continue;
        }
        uint less_x = 0;
        uint eq_x = 0;
        uint less_y = 0;
        uint eq_y = 0;
        if (use_cached_target_ranks_for_candidate) {
            for (ulong j = 0; j < info.rows; ++j) {
                const float xj = interaction_value(features, column_means, j, info.rows, combo, arity);
                if (!isfinite(xj)) {
                    continue;
                }
                if (xj < xi) {
                    ++less_x;
                } else if (xj == xi) {
                    ++eq_x;
                }
            }
        } else {
            for (ulong j = 0; j < info.rows; ++j) {
                const float xj = interaction_value(features, column_means, j, info.rows, combo, arity);
                const float yj = target[j];
                if (!isfinite(xj) || !isfinite(yj)) {
                    continue;
                }
                if (xj < xi) {
                    ++less_x;
                } else if (xj == xi) {
                    ++eq_x;
                }
                if (yj < yi) {
                    ++less_y;
                } else if (yj == yi) {
                    ++eq_y;
                }
            }
        }
        const uint rx_twice = less_x * 2 + eq_x - 1;
        const uint ry_twice = use_cached_target_ranks_for_candidate
            ? target_ranks_twice[i]
            : less_y * 2 + eq_y - 1;
        const float rx = static_cast<float>(rx_twice) * 0.5f;
        const float ry = static_cast<float>(ry_twice) * 0.5f;
        l_srx += rx;
        l_sry += ry;
        l_srxx += rx * rx;
        l_sryy += ry * ry;
        l_srxy += rx * ry;
        ++l_n;
    }

    threadgroup float s_srx[kMetalReduceWidth];
    threadgroup float s_sry[kMetalReduceWidth];
    threadgroup float s_srxx[kMetalReduceWidth];
    threadgroup float s_sryy[kMetalReduceWidth];
    threadgroup float s_srxy[kMetalReduceWidth];
    threadgroup ulong s_n[kMetalReduceWidth];
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
        const float n = static_cast<float>(s_n[0]);
        float out = 0.0f;
        if (n > 1.0f) {
            const float cov = n * s_srxy[0] - s_srx[0] * s_sry[0];
            const float vx = n * s_srxx[0] - s_srx[0] * s_srx[0];
            const float vy = n * s_sryy[0] - s_sry[0] * s_sry[0];
            out = finalize_correlation(vx, vy, cov);
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

// -------------------------------------------------------------------------
// Optional typed semantic-arithmetic table.  These kernels consume only
// resident physical slots and frozen numeric descriptors; evidence identity,
// fitting provenance, and selection policy stay in Rust.
// -------------------------------------------------------------------------

kernel void gafime_semantic_absolute_difference(
    device float* columns [[buffer(0)]],
    constant MetalSemanticProgramNode& node [[buffer(1)]],
    device const uint* operands [[buffer(2)]],
    constant MetalSemanticRowsInfo& info [[buffer(3)]],
    ulong row [[thread_position_in_grid]]
) {
    if (row >= info.rows) return;
    const uint left_slot = operands[node.operand_offset];
    const uint right_slot = operands[node.operand_offset + 1u];
    const float left = columns[static_cast<ulong>(left_slot) * info.rows + row];
    const float right = columns[static_cast<ulong>(right_slot) * info.rows + row];
    columns[static_cast<ulong>(node.output_slot) * info.rows + row] = abs(left - right);
}

kernel void gafime_semantic_softsign(
    device float* columns [[buffer(0)]],
    constant MetalSemanticProgramNode& node [[buffer(1)]],
    device const uint* operands [[buffer(2)]],
    constant MetalSemanticRowsInfo& info [[buffer(3)]],
    ulong row [[thread_position_in_grid]]
) {
    if (row >= info.rows) return;
    const uint input_slot = operands[node.operand_offset];
    const float value = columns[static_cast<ulong>(input_slot) * info.rows + row];
    columns[static_cast<ulong>(node.output_slot) * info.rows + row] =
        value / (1.0f + abs(value));
}

kernel void gafime_semantic_centered_product(
    device float* columns [[buffer(0)]],
    constant MetalSemanticProgramNode& node [[buffer(1)]],
    device const uint* operands [[buffer(2)]],
    device const ulong* mean_bits [[buffer(3)]],
    constant MetalSemanticRowsInfo& info [[buffer(4)]],
    ulong row [[thread_position_in_grid]]
) {
    if (row >= info.rows) return;
    float product = 1.0f;
    for (uint operand = 0; operand < node.operand_count; ++operand) {
        const uint slot = operands[node.operand_offset + operand];
        const float mean = as_type<float>(static_cast<uint>(mean_bits[node.mean_offset + operand]));
        const float value = columns[static_cast<ulong>(slot) * info.rows + row];
        product *= value - mean;
    }
    columns[static_cast<ulong>(node.output_slot) * info.rows + row] = product;
}

kernel void gafime_semantic_frozen_region_conjunction(
    device float* columns [[buffer(0)]],
    constant MetalSemanticProgramNode& node [[buffer(1)]],
    device const MetalSemanticRegionTerm* terms [[buffer(2)]],
    constant MetalSemanticRowsInfo& info [[buffer(3)]],
    ulong row [[thread_position_in_grid]]
) {
    if (row >= info.rows) return;
    bool undetermined = false;
    for (uint term_index = 0; term_index < node.region_term_count; ++term_index) {
        const MetalSemanticRegionTerm term = terms[node.region_term_offset + term_index];
        const float value = columns[static_cast<ulong>(term.input_slot) * info.rows + row];
        if (isnan(value)) {
            undetermined = true;
            continue;
        }
        const float threshold = as_type<float>(static_cast<uint>(term.threshold_bits));
        const bool holds = term.relation == GAFIME_SEMANTIC_REGION_LESS_EQUAL
            ? value <= threshold
            : value > threshold;
        // False dominates a prior NaN exactly as the frozen predicate contract
        // specifies: only an otherwise-true undetermined conjunction emits NaN.
        if (!holds) {
            columns[static_cast<ulong>(node.output_slot) * info.rows + row] = 0.0f;
            return;
        }
    }
    columns[static_cast<ulong>(node.output_slot) * info.rows + row] = undetermined
        ? nonfinite_metric()
        : 1.0f;
}

kernel void gafime_semantic_reject_nonfinite(
    device const float* columns [[buffer(0)]],
    constant MetalSemanticRowsInfo& info [[buffer(1)]],
    device atomic_uint* nonfinite_out [[buffer(2)]],
    ulong row [[thread_position_in_grid]]
) {
    if (row >= info.rows) return;
    if (!isfinite(columns[static_cast<ulong>(info.item_count) * info.rows + row])) {
        atomic_store_explicit(nonfinite_out, 1u, memory_order_relaxed);
    }
}

kernel void gafime_semantic_pairwise_pearson(
    device const float* left_columns [[buffer(0)]],
    device const float* right_columns [[buffer(1)]],
    device const uint* left_slots [[buffer(2)]],
    device const uint* right_slots [[buffer(3)]],
    device float* values [[buffer(4)]],
    device uint* states [[buffer(5)]],
    device ulong* supports [[buffer(6)]],
    constant MetalSemanticAssociationInfo& info [[buffer(7)]],
    uint pair [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    if (static_cast<ulong>(pair) >= info.pair_count) return;
    device const float* left = left_columns + static_cast<ulong>(left_slots[pair]) * info.rows;
    device const float* right = right_columns + static_cast<ulong>(right_slots[pair]) * info.rows;
    const float left_first = info.rows == 0 ? 0.0f : left[0];
    const float right_first = info.rows == 0 ? 0.0f : right[0];

    float local_left_sum = 0.0f;
    float local_right_sum = 0.0f;
    uint local_nonfinite = 0;
    uint local_left_changed = 0;
    uint local_right_changed = 0;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float left_value = left[row];
        const float right_value = right[row];
        local_nonfinite |= (!isfinite(left_value) || !isfinite(right_value)) ? 1u : 0u;
        local_left_changed |= left_value != left_first;
        local_right_changed |= right_value != right_first;
        local_left_sum += left_value;
        local_right_sum += right_value;
    }

    threadgroup float sums_left[kMetalReduceWidth];
    threadgroup float sums_right[kMetalReduceWidth];
    threadgroup float variances_left[kMetalReduceWidth];
    threadgroup float variances_right[kMetalReduceWidth];
    threadgroup float covariances[kMetalReduceWidth];
    threadgroup uint nonfinite[kMetalReduceWidth];
    threadgroup uint left_changed[kMetalReduceWidth];
    threadgroup uint right_changed[kMetalReduceWidth];
    threadgroup float left_mean;
    threadgroup float right_mean;
    threadgroup uint result_state;

    sums_left[lane] = local_left_sum;
    sums_right[lane] = local_right_sum;
    nonfinite[lane] = local_nonfinite;
    left_changed[lane] = local_left_changed;
    right_changed[lane] = local_right_changed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums_left[lane] += sums_left[lane + stride];
            sums_right[lane] += sums_right[lane + stride];
            nonfinite[lane] |= nonfinite[lane + stride];
            left_changed[lane] |= left_changed[lane + stride];
            right_changed[lane] |= right_changed[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        supports[pair] = info.rows;
        result_state = info.rows < 2 ? GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT :
            nonfinite[0] != 0 ? GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION :
            (left_changed[0] == 0 || right_changed[0] == 0)
                ? GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND
                : GAFIME_SEMANTIC_SCALAR_MEASURED;
        if (result_state == GAFIME_SEMANTIC_SCALAR_MEASURED) {
            left_mean = sums_left[0] / static_cast<float>(info.rows);
            right_mean = sums_right[0] / static_cast<float>(info.rows);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (result_state != GAFIME_SEMANTIC_SCALAR_MEASURED) {
        if (lane == 0) {
            states[pair] = result_state;
            values[pair] = 0.0f;
        }
        return;
    }

    float local_left_variance = 0.0f;
    float local_right_variance = 0.0f;
    float local_covariance = 0.0f;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float left_delta = left[row] - left_mean;
        const float right_delta = right[row] - right_mean;
        local_left_variance += left_delta * left_delta;
        local_right_variance += right_delta * right_delta;
        local_covariance += left_delta * right_delta;
    }
    variances_left[lane] = local_left_variance;
    variances_right[lane] = local_right_variance;
    covariances[lane] = local_covariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            variances_left[lane] += variances_left[lane + stride];
            variances_right[lane] += variances_right[lane + stride];
            covariances[lane] += covariances[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float variance_left = variances_left[0];
        const float variance_right = variances_right[0];
        const float covariance = covariances[0];
        if (variance_left == 0.0f || variance_right == 0.0f ||
            (isfinite(variance_left) && isfinite(variance_right) && isfinite(covariance) &&
                variance_left > 0.0f && variance_right > 0.0f &&
                variance_left * variance_right == 0.0f)) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        float correlation = finalize_correlation(variance_left, variance_right, covariance);
        if (!isfinite(correlation)) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        if (info.presentation == GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE) correlation = abs(correlation);
        states[pair] = GAFIME_SEMANTIC_SCALAR_MEASURED;
        values[pair] = correlation;
    }
}

kernel void gafime_semantic_fixed_corrected_nmi(
    device const float* left_columns [[buffer(0)]],
    device const float* right_columns [[buffer(1)]],
    device const uint* left_slots [[buffer(2)]],
    device const uint* right_slots [[buffer(3)]],
    device float* values [[buffer(4)]],
    device uint* states [[buffer(5)]],
    device ulong* supports [[buffer(6)]],
    constant MetalSemanticAssociationInfo& info [[buffer(7)]],
    uint pair [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    if (static_cast<ulong>(pair) >= info.pair_count) return;
    const uint bins = info.bins;
    device const float* left = left_columns + static_cast<ulong>(left_slots[pair]) * info.rows;
    device const float* right = right_columns + static_cast<ulong>(right_slots[pair]) * info.rows;
    const float left_first = info.rows == 0 ? 0.0f : left[0];
    const float right_first = info.rows == 0 ? 0.0f : right[0];

    threadgroup atomic_uint histogram_left[kMetalMaxMiBins];
    threadgroup atomic_uint histogram_right[kMetalMaxMiBins];
    threadgroup atomic_uint joint[kMetalMaxMiBins * kMetalMaxMiBins];
    threadgroup float minimum_left[kMetalReduceWidth];
    threadgroup float maximum_left[kMetalReduceWidth];
    threadgroup float minimum_right[kMetalReduceWidth];
    threadgroup float maximum_right[kMetalReduceWidth];
    threadgroup uint nonfinite[kMetalReduceWidth];
    threadgroup uint left_changed[kMetalReduceWidth];
    threadgroup uint right_changed[kMetalReduceWidth];
    threadgroup uint result_state;

    for (uint index = lane; index < bins; index += lane_count) {
        atomic_store_explicit(&histogram_left[index], 0u, memory_order_relaxed);
        atomic_store_explicit(&histogram_right[index], 0u, memory_order_relaxed);
    }
    for (uint index = lane; index < bins * bins; index += lane_count) {
        atomic_store_explicit(&joint[index], 0u, memory_order_relaxed);
    }

    float local_minimum_left = INFINITY;
    float local_maximum_left = -INFINITY;
    float local_minimum_right = INFINITY;
    float local_maximum_right = -INFINITY;
    uint local_nonfinite = 0;
    uint local_left_changed = 0;
    uint local_right_changed = 0;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float left_value = left[row];
        const float right_value = right[row];
        const bool finite = isfinite(left_value) && isfinite(right_value);
        local_nonfinite |= finite ? 0u : 1u;
        local_left_changed |= left_value != left_first;
        local_right_changed |= right_value != right_first;
        if (finite) {
            local_minimum_left = min(local_minimum_left, left_value);
            local_maximum_left = max(local_maximum_left, left_value);
            local_minimum_right = min(local_minimum_right, right_value);
            local_maximum_right = max(local_maximum_right, right_value);
        }
    }
    minimum_left[lane] = local_minimum_left;
    maximum_left[lane] = local_maximum_left;
    minimum_right[lane] = local_minimum_right;
    maximum_right[lane] = local_maximum_right;
    nonfinite[lane] = local_nonfinite;
    left_changed[lane] = local_left_changed;
    right_changed[lane] = local_right_changed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            minimum_left[lane] = min(minimum_left[lane], minimum_left[lane + stride]);
            maximum_left[lane] = max(maximum_left[lane], maximum_left[lane + stride]);
            minimum_right[lane] = min(minimum_right[lane], minimum_right[lane + stride]);
            maximum_right[lane] = max(maximum_right[lane], maximum_right[lane + stride]);
            nonfinite[lane] |= nonfinite[lane + stride];
            left_changed[lane] |= left_changed[lane + stride];
            right_changed[lane] |= right_changed[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float min_left = minimum_left[0];
    const float max_left = maximum_left[0];
    const float min_right = minimum_right[0];
    const float max_right = maximum_right[0];
    if (lane == 0) {
        supports[pair] = info.rows;
        const ulong required_support = 8ull * static_cast<ulong>(bins) * static_cast<ulong>(bins);
        const float left_span = max_left - min_left;
        const float right_span = max_right - min_right;
        result_state = info.rows < 2 ? GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT :
            nonfinite[0] != 0 ? GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION :
            (left_changed[0] == 0 || right_changed[0] == 0)
                ? GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND
                : info.rows < required_support
                    ? GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT
                    : (!isfinite(left_span) || !isfinite(right_span))
                        ? GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION
                        : (left_span <= 0.0f || right_span <= 0.0f)
                            ? GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION
                            : GAFIME_SEMANTIC_SCALAR_MEASURED;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (result_state != GAFIME_SEMANTIC_SCALAR_MEASURED) {
        if (lane == 0) {
            states[pair] = result_state;
            values[pair] = 0.0f;
        }
        return;
    }

    const float inverse_left = static_cast<float>(bins) / (max_left - min_left);
    const float inverse_right = static_cast<float>(bins) / (max_right - min_right);
    if (!isfinite(inverse_left) || !isfinite(inverse_right)) {
        if (lane == 0) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
            values[pair] = 0.0f;
        }
        return;
    }
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const uint left_bin = fixed_mi_bin(left[row], min_left, inverse_left, bins);
        const uint right_bin = fixed_mi_bin(right[row], min_right, inverse_right, bins);
        atomic_fetch_add_explicit(&histogram_left[left_bin], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&histogram_right[right_bin], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&joint[left_bin * bins + right_bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const float total = static_cast<float>(info.rows);
        float mutual_information = 0.0f;
        uint active_left = 0;
        uint active_right = 0;
        for (uint left_bin = 0; left_bin < bins; ++left_bin) {
            const uint left_count = atomic_load_explicit(&histogram_left[left_bin], memory_order_relaxed);
            if (left_count == 0) continue;
            ++active_left;
            const float probability_left = static_cast<float>(left_count) / total;
            for (uint right_bin = 0; right_bin < bins; ++right_bin) {
                const uint joint_count = atomic_load_explicit(
                    &joint[left_bin * bins + right_bin], memory_order_relaxed);
                const uint right_count = atomic_load_explicit(
                    &histogram_right[right_bin], memory_order_relaxed);
                if (joint_count == 0 || right_count == 0) continue;
                const float probability_right = static_cast<float>(right_count) / total;
                const float probability_joint = static_cast<float>(joint_count) / total;
                mutual_information += probability_joint * log(
                    probability_joint / (probability_left * probability_right));
            }
        }
        for (uint right_bin = 0; right_bin < bins; ++right_bin) {
            if (atomic_load_explicit(&histogram_right[right_bin], memory_order_relaxed) != 0) {
                ++active_right;
            }
        }
        if (active_left < 2 || active_right < 2) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        const float correction = static_cast<float>((active_left - 1u) * (active_right - 1u)) /
            (2.0f * total);
        const float corrected = max(0.0f, mutual_information - correction);
        const float normalizer = log(static_cast<float>(min(active_left, active_right)));
        const float result = corrected / normalizer;
        if (!isfinite(mutual_information) || !isfinite(correction) || !isfinite(normalizer) ||
            normalizer <= 0.0f || !isfinite(result)) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        states[pair] = GAFIME_SEMANTIC_SCALAR_MEASURED;
        values[pair] = result;
    }
}

kernel void gafime_semantic_rank_prepare(
    device const float* left_columns [[buffer(0)]],
    device const float* right_columns [[buffer(1)]],
    device const uint* left_slots [[buffer(2)]],
    device const uint* right_slots [[buffer(3)]],
    device MetalSemanticRankRecord* left_records [[buffer(4)]],
    device MetalSemanticRankRecord* right_records [[buffer(5)]],
    constant MetalSemanticRankInfo& info [[buffer(6)]],
    ulong item [[thread_position_in_grid]]
) {
    const ulong total = info.pair_count * static_cast<ulong>(info.padded_rows);
    if (item >= total) return;
    const ulong pair = item / static_cast<ulong>(info.padded_rows);
    const uint row = static_cast<uint>(item - pair * static_cast<ulong>(info.padded_rows));
    MetalSemanticRankRecord left_record{};
    MetalSemanticRankRecord right_record{};
    left_record.row = row;
    right_record.row = row;
    if (static_cast<ulong>(row) < info.rows) {
        const float left_value = left_columns[
            static_cast<ulong>(left_slots[pair]) * info.rows + static_cast<ulong>(row)];
        const float right_value = right_columns[
            static_cast<ulong>(right_slots[pair]) * info.rows + static_cast<ulong>(row)];
        // A later definedness pass reports a nonfinite input.  The sentinels
        // keep the bounded sort well-defined without treating nonfinite values
        // as valid rank observations.
        left_record.value = isfinite(left_value) && isfinite(right_value) ? left_value : INFINITY;
        right_record.value = isfinite(left_value) && isfinite(right_value) ? right_value : INFINITY;
    } else {
        left_record.value = INFINITY;
        right_record.value = INFINITY;
    }
    left_records[item] = left_record;
    right_records[item] = right_record;
}

static inline bool semantic_rank_record_greater(
    MetalSemanticRankRecord left,
    MetalSemanticRankRecord right
) {
    return left.value > right.value ||
        (left.value == right.value && left.row > right.row);
}

static inline bool semantic_rank_record_less(
    MetalSemanticRankRecord left,
    MetalSemanticRankRecord right
) {
    return left.value < right.value ||
        (left.value == right.value && left.row < right.row);
}

kernel void gafime_semantic_rank_bitonic_step(
    device MetalSemanticRankRecord* left_records [[buffer(0)]],
    device MetalSemanticRankRecord* right_records [[buffer(1)]],
    constant MetalSemanticRankInfo& info [[buffer(2)]],
    ulong item [[thread_position_in_grid]]
) {
    const ulong total = info.pair_count * static_cast<ulong>(info.padded_rows);
    if (item >= total) return;
    const ulong pair = item / static_cast<ulong>(info.padded_rows);
    const uint local = static_cast<uint>(item - pair * static_cast<ulong>(info.padded_rows));
    const uint peer_local = local ^ info.stride;
    if (peer_local <= local || peer_local >= info.padded_rows) return;
    const ulong peer = pair * static_cast<ulong>(info.padded_rows) + peer_local;
    const bool ascending = (local & info.stage) == 0;

    MetalSemanticRankRecord left_a = left_records[item];
    MetalSemanticRankRecord left_b = left_records[peer];
    const bool swap_left = ascending
        ? semantic_rank_record_greater(left_a, left_b)
        : semantic_rank_record_less(left_a, left_b);
    if (swap_left) {
        left_records[item] = left_b;
        left_records[peer] = left_a;
    }

    MetalSemanticRankRecord right_a = right_records[item];
    MetalSemanticRankRecord right_b = right_records[peer];
    const bool swap_right = ascending
        ? semantic_rank_record_greater(right_a, right_b)
        : semantic_rank_record_less(right_a, right_b);
    if (swap_right) {
        right_records[item] = right_b;
        right_records[peer] = right_a;
    }
}

static inline uint semantic_rank_lower_bound(
    device const MetalSemanticRankRecord* records,
    ulong base,
    uint count,
    float value
) {
    uint begin = 0;
    uint end = count;
    while (begin < end) {
        const uint middle = begin + (end - begin) / 2u;
        if (records[base + middle].value < value) {
            begin = middle + 1u;
        } else {
            end = middle;
        }
    }
    return begin;
}

static inline uint semantic_rank_upper_bound(
    device const MetalSemanticRankRecord* records,
    ulong base,
    uint count,
    float value
) {
    uint begin = 0;
    uint end = count;
    while (begin < end) {
        const uint middle = begin + (end - begin) / 2u;
        if (!(value < records[base + middle].value)) {
            begin = middle + 1u;
        } else {
            end = middle;
        }
    }
    return begin;
}

kernel void gafime_semantic_rank_positions(
    device const MetalSemanticRankRecord* left_records [[buffer(0)]],
    device const MetalSemanticRankRecord* right_records [[buffer(1)]],
    device uint* left_ranks_twice [[buffer(2)]],
    device uint* right_ranks_twice [[buffer(3)]],
    constant MetalSemanticRankInfo& info [[buffer(4)]],
    ulong item [[thread_position_in_grid]]
) {
    const ulong total = info.pair_count * info.rows;
    if (item >= total) return;
    const ulong pair = item / info.rows;
    const ulong row = item - pair * info.rows;
    const ulong record_base = pair * static_cast<ulong>(info.padded_rows);
    const float left_value = left_records[record_base + row].value;
    const float right_value = right_records[record_base + row].value;
    if (!isfinite(left_value) || !isfinite(right_value)) return;
    const uint count = static_cast<uint>(info.rows);
    const uint left_lower = semantic_rank_lower_bound(left_records, record_base, count, left_value);
    const uint left_upper = semantic_rank_upper_bound(left_records, record_base, count, left_value);
    const uint right_lower = semantic_rank_lower_bound(right_records, record_base, count, right_value);
    const uint right_upper = semantic_rank_upper_bound(right_records, record_base, count, right_value);
    const MetalSemanticRankRecord left_record = left_records[record_base + row];
    const MetalSemanticRankRecord right_record = right_records[record_base + row];
    left_ranks_twice[pair * info.rows + left_record.row] = left_lower + left_upper - 1u;
    right_ranks_twice[pair * info.rows + right_record.row] = right_lower + right_upper - 1u;
}

kernel void gafime_semantic_spearman_finalize(
    device const float* left_columns [[buffer(0)]],
    device const float* right_columns [[buffer(1)]],
    device const uint* left_slots [[buffer(2)]],
    device const uint* right_slots [[buffer(3)]],
    device const uint* left_ranks_twice [[buffer(4)]],
    device const uint* right_ranks_twice [[buffer(5)]],
    device float* values [[buffer(6)]],
    device uint* states [[buffer(7)]],
    device ulong* supports [[buffer(8)]],
    constant MetalSemanticAssociationInfo& info [[buffer(9)]],
    uint pair [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    if (static_cast<ulong>(pair) >= info.pair_count) return;
    device const float* left = left_columns + static_cast<ulong>(left_slots[pair]) * info.rows;
    device const float* right = right_columns + static_cast<ulong>(right_slots[pair]) * info.rows;
    const float left_first = info.rows == 0 ? 0.0f : left[0];
    const float right_first = info.rows == 0 ? 0.0f : right[0];

    uint local_nonfinite = 0;
    uint local_left_changed = 0;
    uint local_right_changed = 0;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float left_value = left[row];
        const float right_value = right[row];
        local_nonfinite |= (!isfinite(left_value) || !isfinite(right_value)) ? 1u : 0u;
        local_left_changed |= left_value != left_first;
        local_right_changed |= right_value != right_first;
    }
    threadgroup uint nonfinite[kMetalReduceWidth];
    threadgroup uint left_changed[kMetalReduceWidth];
    threadgroup uint right_changed[kMetalReduceWidth];
    threadgroup float variances_left[kMetalReduceWidth];
    threadgroup float variances_right[kMetalReduceWidth];
    threadgroup float covariances[kMetalReduceWidth];
    threadgroup uint result_state;
    nonfinite[lane] = local_nonfinite;
    left_changed[lane] = local_left_changed;
    right_changed[lane] = local_right_changed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            nonfinite[lane] |= nonfinite[lane + stride];
            left_changed[lane] |= left_changed[lane + stride];
            right_changed[lane] |= right_changed[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        supports[pair] = info.rows;
        result_state = info.rows < 2 ? GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT :
            nonfinite[0] != 0 ? GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION :
            (left_changed[0] == 0 || right_changed[0] == 0)
                ? GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND
                : GAFIME_SEMANTIC_SCALAR_MEASURED;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (result_state != GAFIME_SEMANTIC_SCALAR_MEASURED) {
        if (lane == 0) {
            states[pair] = result_state;
            values[pair] = 0.0f;
        }
        return;
    }

    // Average-tie rank positions are exact integers.  Centering at the exact
    // common mean and normalizing before reduction avoids a large fp32
    // cancellation at the advertised 32,768-row bound without changing the
    // mathematical Pearson correlation of the ranks.
    const float center = static_cast<float>(info.rows - 1ull);
    const float scale = 1.0f / max(1.0f, center);
    float local_left_variance = 0.0f;
    float local_right_variance = 0.0f;
    float local_covariance = 0.0f;
    const ulong rank_base = static_cast<ulong>(pair) * info.rows;
    for (ulong row = lane; row < info.rows; row += lane_count) {
        const float left_rank = (static_cast<float>(left_ranks_twice[rank_base + row]) - center) * scale;
        const float right_rank = (static_cast<float>(right_ranks_twice[rank_base + row]) - center) * scale;
        local_left_variance += left_rank * left_rank;
        local_right_variance += right_rank * right_rank;
        local_covariance += left_rank * right_rank;
    }
    variances_left[lane] = local_left_variance;
    variances_right[lane] = local_right_variance;
    covariances[lane] = local_covariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            variances_left[lane] += variances_left[lane + stride];
            variances_right[lane] += variances_right[lane + stride];
            covariances[lane] += covariances[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        if (variances_left[0] == 0.0f || variances_right[0] == 0.0f) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        float correlation = finalize_correlation(variances_left[0], variances_right[0], covariances[0]);
        if (!isfinite(correlation)) {
            states[pair] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
            values[pair] = 0.0f;
            return;
        }
        if (info.presentation == GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE) correlation = abs(correlation);
        states[pair] = GAFIME_SEMANTIC_SCALAR_MEASURED;
        values[pair] = correlation;
    }
}

kernel void gafime_semantic_column_means(
    device const float* columns [[buffer(0)]],
    device const uint* candidate_slots [[buffer(1)]],
    device float* values [[buffer(2)]],
    device uint* states [[buffer(3)]],
    device ulong* supports [[buffer(4)]],
    constant MetalSemanticRowsInfo& info [[buffer(5)]],
    ulong candidate [[thread_position_in_grid]]
) {
    if (candidate >= static_cast<ulong>(info.item_count)) return;
    supports[candidate] = info.rows;
    if (info.rows == 0) {
        states[candidate] = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        values[candidate] = 0.0f;
        return;
    }
    device const float* column = columns + static_cast<ulong>(candidate_slots[candidate]) * info.rows;
    float sum = 0.0f;
    bool nonfinite = false;
    // Means are intentionally one device thread per requested column.  This
    // preserves the declared row-order fp32 sum used to freeze centered terms;
    // the host only transfers the resulting typed scalar.
    for (ulong row = 0; row < info.rows; ++row) {
        const float value = column[row];
        nonfinite = nonfinite || !isfinite(value);
        sum += value;
    }
    const float mean = sum / static_cast<float>(info.rows);
    if (nonfinite || !isfinite(sum) || !isfinite(mean)) {
        states[candidate] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
        values[candidate] = 0.0f;
        return;
    }
    states[candidate] = GAFIME_SEMANTIC_SCALAR_MEASURED;
    values[candidate] = mean;
}

kernel void gafime_semantic_ordered_edge_energy(
    device const float* columns [[buffer(0)]],
    device const uint* candidate_slots [[buffer(1)]],
    device const MetalSemanticEdge* edges [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* values [[buffer(4)]],
    device uint* states [[buffer(5)]],
    device ulong* supports [[buffer(6)]],
    constant MetalSemanticEdgeInfo& info [[buffer(7)]],
    ulong candidate [[thread_position_in_grid]]
) {
    if (candidate >= info.candidate_count) return;
    supports[candidate] = info.edge_count;
    if (info.rows == 0) {
        states[candidate] = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        values[candidate] = 0.0f;
        return;
    }
    device const float* column = columns + static_cast<ulong>(candidate_slots[candidate]) * info.rows;
    const float first = column[0];
    bool constant = true;
    for (ulong row = 0; row < info.rows; ++row) {
        constant = constant && column[row] == first;
    }
    if (constant) {
        states[candidate] = GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND;
        values[candidate] = 0.0f;
        return;
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    // Edge order is the ABI reduction order.  One device thread per candidate
    // deliberately preserves it rather than silently re-associating weights.
    for (ulong edge_index = 0; edge_index < info.edge_count; ++edge_index) {
        const MetalSemanticEdge edge = edges[edge_index];
        const float left = column[edge.left_row];
        const float right = column[edge.right_row];
        const float weight = weights[edge_index];
        const float difference = left - right;
        numerator += weight * difference * difference;
        denominator += weight * (left * left + right * right);
    }
    const float ratio = numerator / denominator;
    if (!isfinite(numerator) || !isfinite(denominator) || !isfinite(ratio)) {
        states[candidate] = GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION;
        values[candidate] = 0.0f;
        return;
    }
    states[candidate] = GAFIME_SEMANTIC_SCALAR_MEASURED;
    values[candidate] = ratio;
}

kernel void gafime_semantic_sparse_gather(
    device const float* source_columns [[buffer(0)]],
    device float* destination_columns [[buffer(1)]],
    device const uint* source_slots [[buffer(2)]],
    device const uint* destination_slots [[buffer(3)]],
    device const ulong* row_indices [[buffer(4)]],
    constant MetalSemanticGatherInfo& info [[buffer(5)]],
    ulong item [[thread_position_in_grid]]
) {
    const ulong total = info.slot_count * info.destination_rows;
    if (item >= total) return;
    const ulong slot_index = item / info.destination_rows;
    const ulong destination_row = item - slot_index * info.destination_rows;
    const ulong source_row = row_indices[destination_row];
    destination_columns[static_cast<ulong>(destination_slots[slot_index]) * info.destination_rows +
        destination_row] = source_columns[static_cast<ulong>(source_slots[slot_index]) * info.source_rows +
        source_row];
}
