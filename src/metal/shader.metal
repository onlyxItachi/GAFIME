#include <metal_stdlib>

using namespace metal;

constant uint GAFIME_METRIC_PEARSON = 1;
constant uint GAFIME_METRIC_R2 = 4;

struct MetalChunk {
    uint arity;
    uint reserved;
    ulong descriptor_offset;
    ulong combo_count;
    ulong global_row_offset;
};

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
