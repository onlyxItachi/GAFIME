#include "metal_api.hpp"
#include "../common/covariance_policy.hpp"
#include "../common/gpu_abi_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__APPLE__) && __has_include(<Foundation/Foundation.h>) && __has_include(<Metal/Metal.h>)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define GAFIME_HAS_METAL_RUNTIME 1
#else
#define GAFIME_HAS_METAL_RUNTIME 0
#endif

namespace {

// Must match shader.metal: MI joint histogram fits the Apple threadgroup limit at
// <= 48 bins; continuous/MI/Spearman dispatches use a fixed reduction width.
constexpr uint32_t kMetalMaxMiBins = 48;
constexpr uint32_t kMetalReduceWidth = 64;
constexpr uint32_t kMetalTopKMaxPartialBlocks = 4096;
constexpr uint64_t kMetalSpearmanTargetRankCacheMinSamples = 128;
constexpr uint64_t kMetalSpearmanTargetRankCacheMaxSamples = 4096;
constexpr uint64_t kMetalSpearmanTargetRankCacheMinUnaryCandidates = 2;

bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t* result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool host_size_supported(uint64_t value) {
    return value <= std::numeric_limits<size_t>::max();
}

struct MatrixSizes {
    uint64_t feature_elements;
    uint64_t feature_bytes;
    uint64_t target_bytes;
    uint64_t mean_bytes;
};

struct InteractionDiagnosticSizes {
    uint64_t combo_index_count;
    uint64_t combo_bytes;
    uint64_t overflow_bytes;
    uint64_t flags_bytes;
};

bool checked_matrix_sizes(uint64_t rows, uint32_t cols, MatrixSizes* sizes_out) {
    MatrixSizes sizes{};
    if (!checked_mul_u64(rows, cols, &sizes.feature_elements) ||
        !checked_mul_u64(sizes.feature_elements, sizeof(float), &sizes.feature_bytes) ||
        !checked_mul_u64(rows, sizeof(float), &sizes.target_bytes) ||
        !checked_mul_u64(cols, sizeof(float), &sizes.mean_bytes)) {
        return false;
    }
    *sizes_out = sizes;
    return true;
}

int validate_interaction_diagnostic_batch(
    const GafimeInteractionDiagnosticBatch* diagnostics,
    uint32_t cols,
    InteractionDiagnosticSizes* sizes_out
) {
    if (diagnostics == nullptr || sizes_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (diagnostics->max_arity == 0 || diagnostics->max_arity > 5 ||
        diagnostics->max_arity > cols || diagnostics->reserved32 != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t value : diagnostics->reserved) {
        if (value != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (diagnostics->row_count > std::numeric_limits<uint32_t>::max()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    InteractionDiagnosticSizes sizes{};
    if (!checked_mul_u64(
            diagnostics->row_count,
            diagnostics->max_arity,
            &sizes.combo_index_count
        ) ||
        diagnostics->combo_index_count != sizes.combo_index_count ||
        !checked_mul_u64(sizes.combo_index_count, sizeof(uint32_t), &sizes.combo_bytes) ||
        !checked_mul_u64(diagnostics->row_count, sizeof(uint64_t), &sizes.overflow_bytes) ||
        !checked_mul_u64(diagnostics->row_count, sizeof(uint32_t), &sizes.flags_bytes) ||
        !host_size_supported(diagnostics->row_count) ||
        !host_size_supported(sizes.combo_index_count) ||
        !host_size_supported(sizes.combo_bytes) ||
        !host_size_supported(sizes.overflow_bytes) ||
        !host_size_supported(sizes.flags_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->row_count == 0) {
        *sizes_out = sizes;
        return GAFIME_STATUS_OK;
    }
    if (diagnostics->combo_indices == nullptr || diagnostics->overflow_row_counts == nullptr ||
        diagnostics->flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    for (uint64_t combo_row = 0; combo_row < diagnostics->row_count; ++combo_row) {
        const uint64_t combo_base = combo_row * diagnostics->max_arity;
        const uint32_t* combo = diagnostics->combo_indices + combo_base;
        uint32_t arity = 0;
        while (arity < diagnostics->max_arity && combo[arity] != UINT32_MAX) {
            if (combo[arity] >= cols) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            ++arity;
        }
        if (arity == 0 || arity > 5) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t slot = arity; slot < diagnostics->max_arity; ++slot) {
            if (combo[slot] != UINT32_MAX) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    *sizes_out = sizes;
    return GAFIME_STATUS_OK;
}

struct MetalChunk {
    uint32_t arity;
    uint32_t mi_bins;
    uint32_t scaled_covariance;
    uint32_t reserved;
    uint64_t descriptor_offset;
    uint64_t combo_count;
    uint64_t global_row_offset;
};

struct MetalLaunchInfo {
    uint64_t rows;
    uint32_t cols;
    uint32_t metric_count;
    uint32_t chunk_count;
};

struct MetalRankInfo {
    uint64_t row_count;
    uint32_t metric_count;
    uint32_t primary_metric_index;
    uint32_t top_k;
    uint32_t partial_block_count;
};

struct MetalInteractionDiagnosticInfo {
    uint64_t rows;
    uint32_t max_arity;
    uint32_t combo_count;
};

static_assert(sizeof(MetalChunk) == 40, "MetalChunk ABI size changed");
static_assert(sizeof(MetalLaunchInfo) == 24, "MetalLaunchInfo ABI size changed");
static_assert(sizeof(MetalRankInfo) == 24, "MetalRankInfo ABI size changed");
static_assert(
    sizeof(MetalInteractionDiagnosticInfo) == 16,
    "MetalInteractionDiagnosticInfo ABI size changed"
);

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO || metric_id == GAFIME_METRIC_SPEARMAN;
}

bool protocol_has_metric(const GafimeLaunchProtocol* protocol, uint32_t metric_id) {
    for (uint32_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (protocol->metric_ids.ptr[idx] == metric_id) {
            return true;
        }
    }
    return false;
}

// Resolve the MI bin count for a chunk from its shape hint (mirrors the CUDA
// mi_bins_for_chunk), then clamp to the Metal threadgroup-memory ceiling.
uint32_t metal_mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = kMetalMaxMiBins;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t hint = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (hint == 2 || hint == 4 || hint == 8 || hint == 12 ||
            hint == 16 || hint == 24 || hint == 32 || hint == 48 ||
            hint == 64 || hint == 96) {
            bins = hint;
        }
    }
    return bins > kMetalMaxMiBins ? kMetalMaxMiBins : bins;
}

int validate_matrix_desc(const GafimeMatrixDesc* desc, MatrixSizes* sizes_out) {
    if (desc == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->rows == 0 || desc->cols == 0 || desc->flags != 0 ||
        desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    MatrixSizes sizes{};
    if (!checked_matrix_sizes(desc->rows, desc->cols, &sizes) ||
        desc->bytes != sizes.feature_bytes) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *sizes_out = sizes;
    return GAFIME_STATUS_OK;
}

uint64_t planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t total = 0;
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        total += protocol->chunks[idx].combo_count;
    }
    return total;
}

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t total_rows) {
    if (protocol->rank.top_k == 0) {
        return total_rows;
    }
    return std::min<uint64_t>(total_rows, protocol->rank.top_k);
}

uint32_t primary_metric_index(const GafimeLaunchProtocol* protocol) {
    if (protocol->metric_ids.len == 0) {
        return 0;
    }
    for (uint32_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
            return idx;
        }
    }
    return 0;
}

uint32_t metal_topk_partial_block_count(uint64_t row_count, uint64_t top_k) {
    if (row_count == 0 || top_k == 0) {
        return 0;
    }
    const uint64_t target_blocks = 1 + (row_count - 1) / kMetalReduceWidth;
    const uint64_t storage_blocks = 1 + (row_count - 1) / top_k;
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::min(target_blocks, storage_blocks),
        kMetalTopKMaxPartialBlocks
    ));
}

int validate_protocol(const GafimeLaunchProtocol* protocol, uint64_t rows, uint32_t cols) {
    if (protocol == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->backend_kind != GAFIME_BACKEND_METAL) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != rows || protocol->n_features != cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->max_arity == 0 || protocol->max_arity > cols ||
        protocol->family_count != 1) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if ((protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    // MI_APPROX is an accepted planning hint; Metal currently uses the same MI dispatch.
    constexpr uint32_t kKnownLaunchFlags =
        GAFIME_LAUNCH_FLAG_MI_APPROX | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    if ((protocol->flags & ~kKnownLaunchFlags) != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.len > std::numeric_limits<uint32_t>::max()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (!metric_supported(protocol->metric_ids.ptr[idx])) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    if (protocol->rank.top_k != 0) {
        if (protocol->rank.include_ties != 0) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        bool primary_found = false;
        for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
            if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
                primary_found = true;
                break;
            }
        }
        if (!primary_found) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr || protocol->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->shape_hint_count != 0 && protocol->shape_hints == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    const uint64_t max_native_rows = std::numeric_limits<uint32_t>::max();
    uint64_t total_rows = 0;
    uint64_t expected_descriptor_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.family != GAFIME_FAMILY_CONTINUOUS) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (chunk.arity == 0 || chunk.arity > protocol->max_arity) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        if (chunk.combo_count == 0 || chunk.descriptor_count != chunk.combo_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        if (chunk.combo_row_offset != total_rows ||
            chunk.descriptor_offset != expected_descriptor_offset ||
            chunk.local_chunk_id != chunk_idx ||
            (protocol->shape_hint_count != 0 &&
                chunk.shape_hint_index >= protocol->shape_hint_count)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        uint64_t descriptor_span = 0;
        uint64_t descriptor_end = 0;
        if (!checked_mul_u64(chunk.combo_count, chunk.arity, &descriptor_span) ||
            !checked_add_u64(chunk.descriptor_offset, descriptor_span, &descriptor_end) ||
            descriptor_end > protocol->combo_indices.len) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint64_t descriptor_idx = chunk.descriptor_offset;
             descriptor_idx < descriptor_end;
             ++descriptor_idx) {
            if (protocol->combo_indices.ptr[descriptor_idx] >= cols) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!checked_add_u64(total_rows, chunk.combo_count, &total_rows) ||
            total_rows > max_native_rows) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        expected_descriptor_offset = descriptor_end;
    }
    if (expected_descriptor_offset != protocol->combo_indices.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t metric_value_count = 0;
    if (!checked_mul_u64(total_rows, protocol->metric_ids.len, &metric_value_count) ||
        metric_value_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        protocol->combo_indices.len > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr ||
        result->candidate_ids == nullptr || result->row_flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->metric_count < protocol->metric_ids.len || result->max_arity < protocol->max_arity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->capacity < output_row_count(protocol, planned_row_count(protocol))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

void compute_column_means_legacy(
    const float* features,
    uint64_t rows,
    uint32_t cols,
    std::vector<float>& means
) {
    means.assign(cols, 0.0f);
    // ABI 1.0 retains its established host preprocessing semantics so legacy
    // callers are not silently moved onto the additive ABI 1.1 fp32 lane.
    std::vector<double> sums(cols, 0.0);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t base = row * cols;
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += static_cast<double>(features[base + col]);
        }
    }
    for (uint32_t col = 0; col < cols; ++col) {
        means[col] = static_cast<float>(sums[col] / static_cast<double>(rows));
    }
}

void compute_column_means_fp32(
    const float* features,
    uint64_t rows,
    uint32_t cols,
    std::vector<float>& means
) {
    means.assign(cols, 0.0f);
    // Metal is the genuine lane-wide fp32 profile: host preprocessing must not
    // silently widen its statistical reduction before shader execution.
    std::vector<float> sums(cols, 0.0f);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t base = row * cols;
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += features[base + col];
        }
    }
    for (uint32_t col = 0; col < cols; ++col) {
        means[col] = sums[col] / static_cast<float>(rows);
    }
}

void build_feature_major(
    const float* features,
    size_t rows,
    size_t cols,
    size_t feature_element_count,
    std::vector<float>& resident_features,
    std::vector<int>& feature_abs_exponents,
    std::vector<uint8_t>& feature_columns_are_finite
) {
    resident_features.assign(feature_element_count, 0.0f);
    feature_abs_exponents.assign(cols, gafime_gpu_abi::kZeroMagnitudeExponent);
    feature_columns_are_finite.assign(cols, 1u);
    for (size_t col = 0; col < cols; ++col) {
        const size_t feature_base = col * rows;
        for (size_t row = 0; row < rows; ++row) {
            const float value = features[row * cols + col];
            resident_features[feature_base + row] = value;
            if (!std::isfinite(value)) {
                feature_columns_are_finite[col] = 0u;
            }
            feature_abs_exponents[col] = gafime_gpu_abi::update_finite_abs_exponent(
                feature_abs_exponents[col],
                value
            );
        }
    }
}

int centered_difference_upper_exponent(int raw_exponent, float mean) {
    const int mean_exponent = mean == 0.0f
        ? gafime_gpu_abi::kZeroMagnitudeExponent
        : std::ilogb(std::fabs(mean));
    const int largest_exponent = std::max(raw_exponent, mean_exponent);
    if (largest_exponent == gafime_gpu_abi::kZeroMagnitudeExponent) {
        return gafime_gpu_abi::kZeroMagnitudeExponent;
    }
    return largest_exponent > std::numeric_limits<int>::max() - 2
        ? std::numeric_limits<int>::max()
        : largest_exponent + 2;
}

void build_interaction_diagnostic_metadata(
    const std::vector<int>& feature_abs_exponents,
    const std::vector<uint8_t>& feature_columns_are_finite,
    const std::vector<float>& means,
    std::vector<uint8_t>& feature_sources_are_finite,
    std::vector<int>& centered_abs_exponent_upper_bounds
) {
    const size_t cols = means.size();
    feature_sources_are_finite.assign(cols, 0u);
    centered_abs_exponent_upper_bounds.assign(
        cols,
        gafime_gpu_abi::kZeroMagnitudeExponent
    );
    for (size_t col = 0; col < cols; ++col) {
        const bool source_is_finite =
            col < feature_abs_exponents.size() &&
            col < feature_columns_are_finite.size() &&
            feature_columns_are_finite[col] != 0 &&
            std::isfinite(means[col]);
        if (!source_is_finite) {
            continue;
        }
        feature_sources_are_finite[col] = 1u;
        centered_abs_exponent_upper_bounds[col] =
            centered_difference_upper_exponent(feature_abs_exponents[col], means[col]);
    }
}

bool host_values_are_finite(const float* values, size_t value_count) {
    for (size_t idx = 0; idx < value_count; ++idx) {
        if (!std::isfinite(values[idx])) {
            return false;
        }
    }
    return true;
}

bool locate_combo(
    const GafimeLaunchProtocol* protocol,
    uint64_t global_row,
    const GafimeArityChunk** chunk_out,
    uint64_t* local_row_out
) {
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        if (global_row >= chunk.combo_row_offset &&
            global_row - chunk.combo_row_offset < chunk.combo_count) {
            *chunk_out = &chunk;
            *local_row_out = global_row - chunk.combo_row_offset;
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> all_rows(size_t total_rows) {
    std::vector<uint32_t> rows(total_rows);
    std::iota(rows.begin(), rows.end(), 0);
    return rows;
}

int write_result_rows(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    const std::vector<uint32_t>& rows,
    bool compact_metric_rows
) {
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    for (size_t output_row = 0; output_row < rows.size(); ++output_row) {
        const uint64_t global_row = rows[output_row];
        const GafimeArityChunk* chunk = nullptr;
        uint64_t local_row = 0;
        if (!locate_combo(protocol, global_row, &chunk, &local_row)) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const uint64_t combo_base = chunk->descriptor_offset + local_row * chunk->arity;
        for (uint32_t slot = 0; slot < result->max_arity; ++slot) {
            result->combo_indices[output_row * result->max_arity + slot] =
                slot < chunk->arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
        }
        for (uint32_t metric_idx = 0; metric_idx < result->metric_count; ++metric_idx) {
            const uint64_t metric_row = compact_metric_rows ? output_row : global_row;
            result->metric_values[output_row * result->metric_count + metric_idx] =
                metric_idx < metric_count ? metric_values[metric_row * metric_count + metric_idx] : 0.0f;
        }
        result->ranks[output_row] = static_cast<uint32_t>(output_row);
        result->families[output_row] = GAFIME_FAMILY_CONTINUOUS;
        result->candidate_ids[output_row] = global_row;
        result->row_flags[output_row] = 0;
    }
    result->row_count = static_cast<uint64_t>(rows.size());
    return GAFIME_STATUS_OK;
}

#if GAFIME_HAS_METAL_RUNTIME

bool metal_has_unified_memory(id<MTLDevice> device) {
    if (device == nil) {
        return false;
    }
    if ([device respondsToSelector:@selector(hasUnifiedMemory)]) {
        return [device hasUnifiedMemory];
    }
    return [device isLowPower];
}

bool metal_is_apple_family(id<MTLDevice> device) {
    if (device == nil || ![device respondsToSelector:@selector(supportsFamily:)]) {
        return false;
    }
    return [device supportsFamily:MTLGPUFamilyApple1];
}

uint32_t metal_device_flags(id<MTLDevice> device) {
    uint32_t flags = GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL |
        GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION;
    const bool unified = metal_has_unified_memory(device);
    if (unified) {
        flags |= GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY | GAFIME_GPU_DEVICE_FLAG_INTEGRATED;
    } else if ([device isLowPower]) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (metal_is_apple_family(device)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY;
    }
    if (unified && device.recommendedMaxWorkingSetSize >= (8ull * 1024ull * 1024ull * 1024ull)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
    }
    return flags;
}

MTLResourceOptions cpu_visible_storage_options(id<MTLDevice> device) {
    return metal_has_unified_memory(device)
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModeManaged;
}

bool metal_size_supported(uint64_t value) {
    return host_size_supported(value) &&
        value <= std::numeric_limits<NSUInteger>::max();
}

void mark_host_writes(id<MTLBuffer> buffer, NSUInteger length, bool managed_storage) {
    if (managed_storage && buffer != nil && length > 0) {
        [buffer didModifyRange:NSMakeRange(0, length)];
    }
}

struct MetalMatrix {
    uint32_t device_id;
    uint32_t precision_profile;
    bool unified_memory;
    bool managed_storage;
    bool content_valid;
    bool features_are_finite;
    bool target_is_finite;
    bool spearman_target_rank_cache_available;
    bool spearman_target_ranks_ready;
    uint64_t rows;
    uint32_t cols;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> score_pipeline;
    id<MTLComputePipelineState> mi_pipeline;
    id<MTLComputePipelineState> spearman_pipeline;
    id<MTLComputePipelineState> spearman_target_rank_pipeline;
    id<MTLComputePipelineState> partial_topk_desc_pipeline;
    id<MTLComputePipelineState> partial_topk_asc_pipeline;
    id<MTLComputePipelineState> merge_topk_desc_pipeline;
    id<MTLComputePipelineState> merge_topk_asc_pipeline;
    id<MTLComputePipelineState> gather_pipeline;
    id<MTLComputePipelineState> interaction_diagnostics_pipeline;
    id<MTLBuffer> features;
    id<MTLBuffer> target;
    id<MTLBuffer> column_means;
    id<MTLBuffer> spearman_target_ranks;
    id<MTLBuffer> descriptor_combo_buffer;
    id<MTLBuffer> descriptor_metric_id_buffer;
    id<MTLBuffer> descriptor_chunk_buffer;
    id<MTLBuffer> descriptor_info_buffer;
    uint32_t descriptor_profile;
    uint32_t feature_stats_profile;
    uint32_t target_stats_profile;
    uint32_t spearman_target_ranks_profile;
    uint64_t descriptor_generation;
    uint64_t descriptor_combo_len;
    uint64_t descriptor_metric_id_len;
    uint32_t descriptor_chunk_count;
    uint32_t descriptor_shape_hint_count;
    std::vector<int> feature_abs_exponents;
    // Bounded CPU-side diagnostic metadata: one byte plus one int per feature.
    // It owns no additional MTLBuffer and is refreshed with matrix upload.
    std::vector<uint8_t> feature_sources_are_finite;
    std::vector<int> centered_abs_exponent_upper_bounds;
    int target_abs_exponent;
};

bool metal_diagnostic_sources_are_finite(
    const MetalMatrix* matrix,
    const uint32_t* combo,
    uint32_t arity
) {
    if (!matrix->target_is_finite ||
        matrix->feature_sources_are_finite.size() != matrix->cols) {
        return false;
    }
    for (uint32_t idx = 0; idx < arity; ++idx) {
        if (matrix->feature_sources_are_finite[combo[idx]] == 0) {
            return false;
        }
    }
    return true;
}

bool metal_diagnostic_product_is_proven_finite(
    const MetalMatrix* matrix,
    const uint32_t* combo,
    uint32_t arity
) {
    if (arity <= 1 || matrix->centered_abs_exponent_upper_bounds.size() != matrix->cols) {
        return arity <= 1;
    }

    // For a raw/mean exponent e, the centered value is strictly below 2^(e+2).
    // Requiring every prefix below 2^127 is deliberately stronger than the IEEE
    // overflow boundary. Borderline cases take the exact shader path instead of
    // relying on rounding-threshold reasoning.
    bool prefix_is_zero = false;
    int64_t prefix_upper_exponent = 0;
    for (uint32_t idx = 0; idx < arity; ++idx) {
        const int centered_upper_exponent =
            matrix->centered_abs_exponent_upper_bounds[combo[idx]];
        if (centered_upper_exponent == gafime_gpu_abi::kZeroMagnitudeExponent) {
            prefix_is_zero = true;
            continue;
        }
        if (centered_upper_exponent > 127) {
            return false;
        }
        if (!prefix_is_zero) {
            prefix_upper_exponent += centered_upper_exponent;
            if (prefix_upper_exponent > 127) {
                return false;
            }
        }
    }
    return true;
}

bool metal_chunk_requires_scaled_covariance(
    const MetalMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    const GafimeArityChunk& chunk
) {
    if (!protocol_has_metric(protocol, GAFIME_METRIC_PEARSON) &&
        !protocol_has_metric(protocol, GAFIME_METRIC_R2)) {
        return false;
    }
    for (uint64_t combo_idx = 0; combo_idx < chunk.combo_count; ++combo_idx) {
        const uint64_t combo_offset = chunk.descriptor_offset + combo_idx * chunk.arity;
        const int interaction_exponent = gafime_gpu_abi::interaction_abs_exponent(
            matrix->feature_abs_exponents.data(),
            protocol->combo_indices.ptr + combo_offset,
            chunk.arity
        );
        if (gafime_gpu_abi::covariance_requires_scaled_path(
                matrix->rows,
                interaction_exponent,
                matrix->target_abs_exponent
            )) {
            return true;
        }
    }
    return false;
}

bool metal_protocol_descriptors_cacheable(const GafimeLaunchProtocol* protocol) {
    const uint64_t descriptor_generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    return (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0 &&
        descriptor_generation != 0;
}

bool metal_protocol_descriptors_resident(
    const MetalMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    const uint64_t descriptor_generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    return metal_protocol_descriptors_cacheable(protocol) &&
        matrix->descriptor_combo_buffer != nil &&
        matrix->descriptor_metric_id_buffer != nil &&
        matrix->descriptor_chunk_buffer != nil &&
        matrix->descriptor_info_buffer != nil &&
        matrix->descriptor_profile == matrix->precision_profile &&
        matrix->descriptor_generation == descriptor_generation &&
        matrix->descriptor_combo_len == protocol->combo_indices.len &&
        matrix->descriptor_metric_id_len == protocol->metric_ids.len &&
        matrix->descriptor_chunk_count == protocol->chunk_count &&
        matrix->descriptor_shape_hint_count == protocol->shape_hint_count;
}

uint64_t metal_buffer_bytes(id<MTLBuffer> buffer) {
    return buffer == nil ? 0 : static_cast<uint64_t>(buffer.length);
}

uint64_t metal_matrix_fixed_device_bytes(const MetalMatrix* matrix) {
    using gafime_gpu_abi::saturating_add_u64;

    uint64_t bytes = metal_buffer_bytes(matrix->features);
    bytes = saturating_add_u64(bytes, metal_buffer_bytes(matrix->target));
    bytes = saturating_add_u64(bytes, metal_buffer_bytes(matrix->column_means));
    bytes = saturating_add_u64(bytes, metal_buffer_bytes(matrix->spearman_target_ranks));
    return bytes;
}

bool spearman_target_rank_cache_eligible(
    const MetalMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (!matrix->spearman_target_rank_cache_available ||
        !matrix->features_are_finite ||
        !matrix->target_is_finite ||
        matrix->rows < kMetalSpearmanTargetRankCacheMinSamples ||
        matrix->rows > kMetalSpearmanTargetRankCacheMaxSamples ||
        !protocol_has_metric(protocol, GAFIME_METRIC_SPEARMAN)) {
        return false;
    }

    uint64_t unary_candidate_count = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.arity != 1) {
            continue;
        }
        if (unary_candidate_count >= kMetalSpearmanTargetRankCacheMinUnaryCandidates) {
            return true;
        }
        const uint64_t candidates_remaining =
            kMetalSpearmanTargetRankCacheMinUnaryCandidates - unary_candidate_count;
        if (chunk.combo_count >= candidates_remaining) {
            return true;
        }
        unary_candidate_count += chunk.combo_count;
    }
    return false;
}

uint64_t metal_descriptor_cache_device_bytes(const MetalMatrix* matrix) {
    using gafime_gpu_abi::saturating_add_u64;

    uint64_t bytes = metal_buffer_bytes(matrix->descriptor_combo_buffer);
    bytes = saturating_add_u64(
        bytes,
        metal_buffer_bytes(matrix->descriptor_metric_id_buffer)
    );
    bytes = saturating_add_u64(
        bytes,
        metal_buffer_bytes(matrix->descriptor_chunk_buffer)
    );
    bytes = saturating_add_u64(
        bytes,
        metal_buffer_bytes(matrix->descriptor_info_buffer)
    );
    return bytes;
}

uint64_t metal_execution_peak_device_bytes(
    const MetalMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    using gafime_gpu_abi::allocation_bytes;
    using gafime_gpu_abi::saturating_add_u64;
    using gafime_gpu_abi::saturating_mul_u64;

    const uint64_t fixed_bytes = metal_matrix_fixed_device_bytes(matrix);
    const uint64_t old_descriptor_bytes =
        metal_descriptor_cache_device_bytes(matrix);
    const uint64_t resident_bytes =
        saturating_add_u64(fixed_bytes, old_descriptor_bytes);
    uint64_t peak_bytes = resident_bytes;

    const uint64_t total_rows = planned_row_count(protocol);
    if (total_rows == 0) {
        return peak_bytes;
    }

    uint64_t next_descriptor_bytes = allocation_bytes(
        protocol->combo_indices.len,
        sizeof(uint32_t)
    );
    next_descriptor_bytes = saturating_add_u64(
        next_descriptor_bytes,
        allocation_bytes(protocol->metric_ids.len, sizeof(uint32_t))
    );
    next_descriptor_bytes = saturating_add_u64(
        next_descriptor_bytes,
        allocation_bytes(protocol->chunk_count, sizeof(MetalChunk))
    );
    next_descriptor_bytes = saturating_add_u64(
        next_descriptor_bytes,
        sizeof(MetalLaunchInfo)
    );

    uint64_t execution_resident_bytes = resident_bytes;
    if (!metal_protocol_descriptors_resident(matrix, protocol)) {
        const uint64_t replacement_peak = saturating_add_u64(
            resident_bytes,
            next_descriptor_bytes
        );
        peak_bytes = std::max(peak_bytes, replacement_peak);
        execution_resident_bytes = metal_protocol_descriptors_cacheable(protocol)
            ? saturating_add_u64(fixed_bytes, next_descriptor_bytes)
            : replacement_peak;
    }

    const uint64_t metric_items =
        saturating_mul_u64(total_rows, protocol->metric_ids.len);
    uint64_t runtime_bytes = saturating_add_u64(
        execution_resident_bytes,
        allocation_bytes(metric_items, sizeof(float))
    );
    if (protocol->rank.top_k != 0) {
        const uint64_t output_rows = output_row_count(protocol, total_rows);
        runtime_bytes = saturating_add_u64(runtime_bytes, sizeof(MetalRankInfo));
        runtime_bytes = saturating_add_u64(
            runtime_bytes,
            allocation_bytes(output_rows, sizeof(uint32_t))
        );
        runtime_bytes = saturating_add_u64(
            runtime_bytes,
            allocation_bytes(
                saturating_mul_u64(output_rows, protocol->metric_ids.len),
                sizeof(float)
            )
        );
        const uint64_t partial_items = saturating_mul_u64(
            metal_topk_partial_block_count(total_rows, output_rows),
            output_rows
        );
        runtime_bytes = saturating_add_u64(
            runtime_bytes,
            allocation_bytes(partial_items, sizeof(float))
        );
        runtime_bytes = saturating_add_u64(
            runtime_bytes,
            allocation_bytes(partial_items, sizeof(uint32_t))
        );
    }
    return std::max(peak_bytes, runtime_bytes);
}

void invalidate_protocol_descriptor_cache(MetalMatrix* matrix) {
    matrix->descriptor_combo_buffer = nil;
    matrix->descriptor_metric_id_buffer = nil;
    matrix->descriptor_chunk_buffer = nil;
    matrix->descriptor_info_buffer = nil;
    matrix->descriptor_profile = 0;
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->descriptor_chunk_count = 0;
    matrix->descriptor_shape_hint_count = 0;
}

NSArray<id<MTLDevice>>* available_devices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
        if (default_device != nil) {
            devices = @[default_device];
        }
    }
    return devices;
}

id<MTLDevice> device_for_id(uint32_t device_id) {
    NSArray<id<MTLDevice>>* devices = available_devices();
    if (device_id >= devices.count) {
        return nil;
    }
    return devices[device_id];
}

NSString* default_metallib_path() {
#ifdef GAFIME_METAL_DEFAULT_LIBRARY_PATH
    return [NSString stringWithUTF8String:GAFIME_METAL_DEFAULT_LIBRARY_PATH];
#else
    return nil;
#endif
}

id<MTLLibrary> load_library(id<MTLDevice> device) {
    NSString* env_path = [[[NSProcessInfo processInfo] environment] objectForKey:@"GAFIME_METAL_V1_METALLIB"];
    NSString* path = env_path.length > 0 ? env_path : default_metallib_path();
    if (path == nil || path.length == 0) {
        return nil;
    }
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    (void)error;
    return library;
}

#endif

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(uint32_t device_id, GafimeGpuDeviceInfo* info_out) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        if (info_out == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        id<MTLDevice> device = device_for_id(device_id);
        if (device == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        std::memset(info_out, 0, sizeof(*info_out));
        info_out->abi_version = GAFIME_ABI_VERSION;
        info_out->backend_kind = GAFIME_BACKEND_METAL;
        info_out->device_id = device_id;
        info_out->flags = metal_device_flags(device);
        std::snprintf(info_out->name, sizeof(info_out->name), "%s", device.name.UTF8String);
        info_out->total_global_mem_bytes = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        info_out->warp_size = 32;
        info_out->reserved[0] = metal_is_apple_family(device)
            ? GAFIME_GPU_ARCH_APPLE
            : GAFIME_GPU_ARCH_UNKNOWN;
        info_out->reserved[1] = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        info_out->reserved[2] = static_cast<uint64_t>(device.maxThreadgroupMemoryLength);
        info_out->reserved[3] = static_cast<uint64_t>(device.maxThreadsPerThreadgroup.width);
        info_out->reserved[4] = metal_has_unified_memory(device) ? 1ull : 0ull;
        info_out->reserved[5] = [device isLowPower] ? 1ull : 0ull;
        info_out->reserved[6] = [device isRemovable] ? 1ull : 0ull;
        if ([device respondsToSelector:@selector(registryID)]) {
            info_out->reserved[7] = static_cast<uint64_t>(device.registryID);
        }
        return GAFIME_STATUS_OK;
    }
#else
    return gafime_gpu_abi::fill_device_info(device_id, GAFIME_BACKEND_METAL, "metal-unavailable", info_out);
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
) try {
    (void)device_id;
    const int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_METAL,
        GAFIME_GRAPH_UNSUPPORTED,
        capability_out
    );
    if (status == GAFIME_STATUS_OK) {
#if GAFIME_HAS_METAL_RUNTIME
        capability_out->supports_device_ranking = 1;
#endif
        capability_out->stable_pointer_flags = 1;
    }
    return status;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) try {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        MatrixSizes matrix_sizes{};
        int status = validate_matrix_desc(matrix_desc, &matrix_sizes);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        if (!metal_size_supported(matrix_sizes.feature_bytes) ||
            !metal_size_supported(matrix_sizes.target_bytes) ||
            !metal_size_supported(matrix_sizes.mean_bytes)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        id<MTLDevice> device = device_for_id(device_id);
        if (device == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLLibrary> library = load_library(device);
        if (queue == nil || library == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        NSError* error = nil;
        id<MTLFunction> score_function = [library newFunctionWithName:@"gafime_score_continuous"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:score_function error:&error];
        (void)error;
        if (score_function == nil || pipeline == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLFunction> mi_function = [library newFunctionWithName:@"gafime_score_mutual_info"];
        id<MTLComputePipelineState> mi_pipeline = [device newComputePipelineStateWithFunction:mi_function error:&error];
        (void)error;
        id<MTLFunction> spearman_function = [library newFunctionWithName:@"gafime_score_spearman"];
        id<MTLComputePipelineState> spearman_pipeline = [device newComputePipelineStateWithFunction:spearman_function error:&error];
        (void)error;
        id<MTLFunction> spearman_target_rank_function =
            [library newFunctionWithName:@"gafime_build_spearman_target_ranks"];
        id<MTLComputePipelineState> spearman_target_rank_pipeline =
            [device newComputePipelineStateWithFunction:spearman_target_rank_function error:&error];
        (void)error;
        if (mi_function == nil || mi_pipeline == nil ||
            spearman_function == nil || spearman_pipeline == nil ||
            spearman_target_rank_function == nil || spearman_target_rank_pipeline == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLFunction> partial_topk_desc_function = [library newFunctionWithName:@"gafime_select_topk_partials_desc"];
        id<MTLComputePipelineState> partial_topk_desc_pipeline =
            [device newComputePipelineStateWithFunction:partial_topk_desc_function error:&error];
        (void)error;
        id<MTLFunction> partial_topk_asc_function = [library newFunctionWithName:@"gafime_select_topk_partials_asc"];
        id<MTLComputePipelineState> partial_topk_asc_pipeline =
            [device newComputePipelineStateWithFunction:partial_topk_asc_function error:&error];
        (void)error;
        id<MTLFunction> merge_topk_desc_function = [library newFunctionWithName:@"gafime_merge_topk_partials_desc"];
        id<MTLComputePipelineState> merge_topk_desc_pipeline =
            [device newComputePipelineStateWithFunction:merge_topk_desc_function error:&error];
        (void)error;
        id<MTLFunction> merge_topk_asc_function = [library newFunctionWithName:@"gafime_merge_topk_partials_asc"];
        id<MTLComputePipelineState> merge_topk_asc_pipeline =
            [device newComputePipelineStateWithFunction:merge_topk_asc_function error:&error];
        (void)error;
        id<MTLFunction> gather_function = [library newFunctionWithName:@"gafime_copy_selected_metric_rows"];
        id<MTLComputePipelineState> gather_pipeline = [device newComputePipelineStateWithFunction:gather_function error:&error];
        (void)error;
        if (partial_topk_desc_function == nil || partial_topk_desc_pipeline == nil ||
            partial_topk_asc_function == nil || partial_topk_asc_pipeline == nil ||
            merge_topk_desc_function == nil || merge_topk_desc_pipeline == nil ||
            merge_topk_asc_function == nil || merge_topk_asc_pipeline == nil ||
            gather_function == nil || gather_pipeline == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLFunction> interaction_diagnostics_function =
            [library newFunctionWithName:@"gafime_interaction_diagnostics"];
        id<MTLComputePipelineState> interaction_diagnostics_pipeline = nil;
        if (interaction_diagnostics_function != nil) {
            error = nil;
            interaction_diagnostics_pipeline = [device
                newComputePipelineStateWithFunction:interaction_diagnostics_function
                error:&error];
            (void)error;
        }
        if ([pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [mi_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [spearman_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [spearman_target_rank_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [partial_topk_desc_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [partial_topk_asc_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [merge_topk_desc_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [merge_topk_asc_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth ||
            [gather_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (interaction_diagnostics_pipeline != nil &&
            [interaction_diagnostics_pipeline maxTotalThreadsPerThreadgroup] < kMetalReduceWidth) {
            interaction_diagnostics_pipeline = nil;
        }
        const NSUInteger feature_bytes = static_cast<NSUInteger>(matrix_sizes.feature_bytes);
        const NSUInteger target_bytes = static_cast<NSUInteger>(matrix_sizes.target_bytes);
        const NSUInteger mean_bytes = static_cast<NSUInteger>(matrix_sizes.mean_bytes);
        const bool spearman_target_rank_cache_fits =
            matrix_desc->rows <= kMetalSpearmanTargetRankCacheMaxSamples;
        const NSUInteger spearman_target_rank_bytes = spearman_target_rank_cache_fits
            ? static_cast<NSUInteger>(matrix_desc->rows * sizeof(uint32_t))
            : sizeof(uint32_t);
        const MTLResourceOptions storage_options = cpu_visible_storage_options(device);
        const bool managed_storage = storage_options == MTLResourceStorageModeManaged;
        auto* matrix = new MetalMatrix{};
        matrix->device_id = device_id;
        matrix->precision_profile = 0;
        matrix->unified_memory = metal_has_unified_memory(device);
        matrix->managed_storage = managed_storage;
        matrix->content_valid = false;
        matrix->features_are_finite = false;
        matrix->target_is_finite = false;
        matrix->spearman_target_rank_cache_available = false;
        matrix->spearman_target_ranks_ready = false;
        matrix->descriptor_profile = 0;
        matrix->feature_stats_profile = 0;
        matrix->target_stats_profile = 0;
        matrix->spearman_target_ranks_profile = 0;
        matrix->target_abs_exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
        matrix->rows = matrix_desc->rows;
        matrix->cols = matrix_desc->cols;
        matrix->device = device;
        matrix->queue = queue;
        matrix->score_pipeline = pipeline;
        matrix->mi_pipeline = mi_pipeline;
        matrix->spearman_pipeline = spearman_pipeline;
        matrix->spearman_target_rank_pipeline = spearman_target_rank_pipeline;
        matrix->partial_topk_desc_pipeline = partial_topk_desc_pipeline;
        matrix->partial_topk_asc_pipeline = partial_topk_asc_pipeline;
        matrix->merge_topk_desc_pipeline = merge_topk_desc_pipeline;
        matrix->merge_topk_asc_pipeline = merge_topk_asc_pipeline;
        matrix->gather_pipeline = gather_pipeline;
        matrix->interaction_diagnostics_pipeline = interaction_diagnostics_pipeline;
        matrix->features = [device newBufferWithLength:feature_bytes options:storage_options];
        matrix->target = [device newBufferWithLength:target_bytes options:storage_options];
        matrix->column_means = [device newBufferWithLength:mean_bytes options:storage_options];
        matrix->spearman_target_ranks = [device
            newBufferWithLength:spearman_target_rank_bytes
            options:storage_options];
        matrix->spearman_target_rank_cache_available =
            spearman_target_rank_cache_fits && matrix->spearman_target_ranks != nil;
        if (matrix->spearman_target_ranks == nil && spearman_target_rank_cache_fits) {
            // The cache is optional. Keep a one-element binding for the fallback
            // shader path when its bounded resident allocation is unavailable.
            matrix->spearman_target_ranks = [device
                newBufferWithLength:sizeof(uint32_t)
                options:storage_options];
        }
        if (matrix->features == nil || matrix->target == nil || matrix->column_means == nil ||
            matrix->spearman_target_ranks == nil) {
            delete matrix;
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
        return GAFIME_STATUS_OK;
    }
#else
    (void)device_id;
    (void)matrix_desc;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) try {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr ||
        rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    MatrixSizes matrix_sizes{};
    if (!checked_matrix_sizes(rows, cols, &matrix_sizes) ||
        !host_size_supported(rows) ||
        !host_size_supported(cols) ||
        !metal_size_supported(matrix_sizes.feature_elements) ||
        !metal_size_supported(matrix_sizes.feature_bytes) ||
        !metal_size_supported(matrix_sizes.target_bytes) ||
        !metal_size_supported(matrix_sizes.mean_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t row_count_host = static_cast<size_t>(rows);
    const size_t col_count_host = static_cast<size_t>(cols);
    const size_t feature_element_count = static_cast<size_t>(matrix_sizes.feature_elements);
    const size_t feature_bytes_host = static_cast<size_t>(matrix_sizes.feature_bytes);
    const size_t target_bytes_host = static_cast<size_t>(matrix_sizes.target_bytes);
    const size_t mean_bytes_host = static_cast<size_t>(matrix_sizes.mean_bytes);
    const NSUInteger feature_bytes_metal = static_cast<NSUInteger>(matrix_sizes.feature_bytes);
    const NSUInteger target_bytes_metal = static_cast<NSUInteger>(matrix_sizes.target_bytes);
    const NSUInteger mean_bytes_metal = static_cast<NSUInteger>(matrix_sizes.mean_bytes);
    std::vector<float> means;
    if (matrix->precision_profile == GAFIME_PRECISION_FP32) {
        compute_column_means_fp32(features_host, rows, cols, means);
    } else {
        compute_column_means_legacy(features_host, rows, cols, means);
    }
    std::vector<float> resident_features;
    std::vector<int> feature_abs_exponents;
    std::vector<uint8_t> feature_columns_are_finite;
    build_feature_major(
        features_host,
        row_count_host,
        col_count_host,
        feature_element_count,
        resident_features,
        feature_abs_exponents,
        feature_columns_are_finite
    );
    std::vector<uint8_t> feature_sources_are_finite;
    std::vector<int> centered_abs_exponent_upper_bounds;
    build_interaction_diagnostic_metadata(
        feature_abs_exponents,
        feature_columns_are_finite,
        means,
        feature_sources_are_finite,
        centered_abs_exponent_upper_bounds
    );
    matrix->content_valid = false;
    matrix->feature_stats_profile = 0;
    matrix->target_stats_profile = 0;
    matrix->spearman_target_ranks_profile = 0;
    matrix->features_are_finite = host_values_are_finite(features_host, feature_element_count);
    matrix->target_is_finite = host_values_are_finite(target_host, row_count_host);
    matrix->feature_abs_exponents = std::move(feature_abs_exponents);
    matrix->feature_sources_are_finite = std::move(feature_sources_are_finite);
    matrix->centered_abs_exponent_upper_bounds = std::move(centered_abs_exponent_upper_bounds);
    matrix->target_abs_exponent = gafime_gpu_abi::finite_abs_exponent(target_host, rows);
    matrix->spearman_target_ranks_ready = false;
    invalidate_protocol_descriptor_cache(matrix);
    std::memcpy(matrix->features.contents, resident_features.data(), feature_bytes_host);
    std::memcpy(matrix->target.contents, target_host, target_bytes_host);
    std::memcpy(matrix->column_means.contents, means.data(), mean_bytes_host);
    mark_host_writes(matrix->features, feature_bytes_metal, matrix->managed_storage);
    mark_host_writes(matrix->target, target_bytes_metal, matrix->managed_storage);
    mark_host_writes(matrix->column_means, mean_bytes_metal, matrix->managed_storage);
    matrix->feature_stats_profile = matrix->precision_profile;
    matrix->target_stats_profile = matrix->precision_profile;
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
#else
    (void)matrix_handle;
    (void)features_host;
    (void)target_host;
    (void)rows;
    (void)cols;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) try {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || !matrix->content_valid || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t target_bytes = 0;
    if (!checked_mul_u64(rows, sizeof(float), &target_bytes) ||
        !metal_size_supported(target_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t target_bytes_host = static_cast<size_t>(target_bytes);
    const NSUInteger target_bytes_metal = static_cast<NSUInteger>(target_bytes);
    matrix->content_valid = false;
    matrix->target_stats_profile = 0;
    matrix->spearman_target_ranks_profile = 0;
    matrix->target_is_finite = host_values_are_finite(target_host, static_cast<size_t>(rows));
    matrix->target_abs_exponent = gafime_gpu_abi::finite_abs_exponent(target_host, rows);
    matrix->spearman_target_ranks_ready = false;
    invalidate_protocol_descriptor_cache(matrix);
    std::memcpy(matrix->target.contents, target_host, target_bytes_host);
    mark_host_writes(matrix->target, target_bytes_metal, matrix->managed_storage);
    matrix->target_stats_profile = matrix->precision_profile;
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
#else
    (void)matrix_handle;
    (void)target_host;
    (void)rows;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    delete matrix;
#else
    (void)matrix_handle;
#endif
}

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics(
    GafimeGpuMatrix matrix_handle,
    GafimeInteractionDiagnosticBatch* diagnostics
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
        if (matrix == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        InteractionDiagnosticSizes sizes{};
        const int validation_status = validate_interaction_diagnostic_batch(
            diagnostics,
            matrix->cols,
            &sizes
        );
        if (validation_status != GAFIME_STATUS_OK) {
            return validation_status;
        }
        if (!matrix->content_valid ||
            matrix->feature_stats_profile != matrix->precision_profile ||
            matrix->target_stats_profile != matrix->precision_profile) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        if (diagnostics->row_count == 0) {
            return GAFIME_STATUS_OK;
        }
        if (!metal_size_supported(sizes.combo_bytes) ||
            !metal_size_supported(sizes.overflow_bytes) ||
            !metal_size_supported(sizes.flags_bytes)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }

        const size_t diagnostic_row_count = static_cast<size_t>(diagnostics->row_count);
        std::vector<uint32_t> scan_combo_indices;
        std::vector<uint32_t> scan_output_indices;
        for (uint64_t combo_row = 0; combo_row < diagnostics->row_count; ++combo_row) {
            const uint64_t combo_base = combo_row * diagnostics->max_arity;
            const uint32_t* combo = diagnostics->combo_indices + combo_base;
            uint32_t arity = 0;
            while (arity < diagnostics->max_arity && combo[arity] != UINT32_MAX) {
                ++arity;
            }
            const bool sources_are_finite =
                metal_diagnostic_sources_are_finite(matrix, combo, arity);
            if (sources_are_finite &&
                metal_diagnostic_product_is_proven_finite(matrix, combo, arity)) {
                continue;
            }
            scan_output_indices.push_back(static_cast<uint32_t>(combo_row));
            scan_combo_indices.insert(
                scan_combo_indices.end(),
                combo,
                combo + diagnostics->max_arity
            );
        }
        if (scan_output_indices.empty()) {
            std::fill_n(diagnostics->overflow_row_counts, diagnostic_row_count, 0ull);
            std::fill_n(diagnostics->flags, diagnostic_row_count, 0u);
            return GAFIME_STATUS_OK;
        }
        if (matrix->interaction_diagnostics_pipeline == nil) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }

        uint64_t scan_combo_bytes_u64 = 0;
        uint64_t scan_overflow_bytes_u64 = 0;
        uint64_t scan_flags_bytes_u64 = 0;
        const uint64_t scan_combo_count = static_cast<uint64_t>(scan_combo_indices.size());
        const uint64_t scan_count = static_cast<uint64_t>(scan_output_indices.size());
        if (!checked_mul_u64(scan_combo_count, sizeof(uint32_t), &scan_combo_bytes_u64) ||
            !checked_mul_u64(scan_count, sizeof(uint64_t), &scan_overflow_bytes_u64) ||
            !checked_mul_u64(scan_count, sizeof(uint32_t), &scan_flags_bytes_u64) ||
            !metal_size_supported(scan_combo_bytes_u64) ||
            !metal_size_supported(scan_overflow_bytes_u64) ||
            !metal_size_supported(scan_flags_bytes_u64)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        const NSUInteger scan_combo_bytes = static_cast<NSUInteger>(scan_combo_bytes_u64);
        const NSUInteger scan_overflow_bytes = static_cast<NSUInteger>(scan_overflow_bytes_u64);
        const NSUInteger scan_flags_bytes = static_cast<NSUInteger>(scan_flags_bytes_u64);
        const NSUInteger scan_count_metal = static_cast<NSUInteger>(scan_count);
        const MetalInteractionDiagnosticInfo info{
            matrix->rows,
            diagnostics->max_arity,
            static_cast<uint32_t>(scan_count),
        };
        const MTLResourceOptions output_storage = cpu_visible_storage_options(matrix->device);
        id<MTLBuffer> combo_buffer = [matrix->device
            newBufferWithBytes:scan_combo_indices.data()
            length:scan_combo_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> overflow_buffer = [matrix->device
            newBufferWithLength:scan_overflow_bytes
            options:output_storage];
        id<MTLBuffer> flags_buffer = [matrix->device
            newBufferWithLength:scan_flags_bytes
            options:output_storage];
        id<MTLBuffer> info_buffer = [matrix->device
            newBufferWithBytes:&info
            length:sizeof(MetalInteractionDiagnosticInfo)
            options:MTLResourceStorageModeShared];
        if (combo_buffer == nil || overflow_buffer == nil || flags_buffer == nil || info_buffer == nil) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }

        id<MTLCommandBuffer> command_buffer = [matrix->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        [encoder setComputePipelineState:matrix->interaction_diagnostics_pipeline];
        [encoder setBuffer:matrix->features offset:0 atIndex:0];
        [encoder setBuffer:matrix->target offset:0 atIndex:1];
        [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
        [encoder setBuffer:combo_buffer offset:0 atIndex:3];
        [encoder setBuffer:overflow_buffer offset:0 atIndex:4];
        [encoder setBuffer:flags_buffer offset:0 atIndex:5];
        [encoder setBuffer:info_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(scan_count_metal, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kMetalReduceWidth, 1, 1)];
        [encoder endEncoding];
        if (matrix->managed_storage) {
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            if (blit == nil) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            [blit synchronizeResource:overflow_buffer];
            [blit synchronizeResource:flags_buffer];
            [blit endEncoding];
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted ||
            overflow_buffer.contents == nullptr || flags_buffer.contents == nullptr) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }

        const uint64_t* scanned_overflows =
            static_cast<const uint64_t*>(overflow_buffer.contents);
        const uint32_t* scanned_flags = static_cast<const uint32_t*>(flags_buffer.contents);
        std::fill_n(diagnostics->overflow_row_counts, diagnostic_row_count, 0ull);
        std::fill_n(diagnostics->flags, diagnostic_row_count, 0u);
        for (size_t scan_row = 0; scan_row < scan_output_indices.size(); ++scan_row) {
            const uint32_t output_row = scan_output_indices[scan_row];
            diagnostics->overflow_row_counts[output_row] = scanned_overflows[scan_row];
            diagnostics->flags[output_row] = scanned_flags[scan_row];
        }
        return GAFIME_STATUS_OK;
    }
#else
    (void)matrix_handle;
    (void)diagnostics;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) try {
    if (peak_bytes_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *peak_bytes_out = 0;
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
        if (matrix == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        const int status = validate_protocol(protocol, matrix->rows, matrix->cols);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        if (!matrix->content_valid ||
            matrix->feature_stats_profile != matrix->precision_profile ||
            matrix->target_stats_profile != matrix->precision_profile) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // Pipeline, command-buffer, and driver bookkeeping is opaque; this
        // reports all payload-owned MTLBuffer lifetimes for the next launch.
        *peak_bytes_out = metal_execution_peak_device_bytes(matrix, protocol);
        return GAFIME_STATUS_OK;
    }
#else
    (void)matrix_handle;
    (void)protocol;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
        if (matrix == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        int status = validate_protocol(protocol, matrix->rows, matrix->cols);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = validate_result_table(protocol, result_out);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        if (!matrix->content_valid ||
            matrix->feature_stats_profile != matrix->precision_profile ||
            matrix->target_stats_profile != matrix->precision_profile) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        const uint64_t total_rows = planned_row_count(protocol);
        if (total_rows == 0) {
            result_out->row_count = 0;
            return GAFIME_STATUS_OK;
        }
        const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
        const uint64_t output_rows = output_row_count(protocol, total_rows);
        const bool ranked_output = protocol->rank.top_k != 0;
        uint64_t combo_bytes_u64 = 0;
        uint64_t metric_id_bytes_u64 = 0;
        uint64_t chunk_bytes_u64 = 0;
        uint64_t metric_value_count = 0;
        uint64_t metric_bytes_u64 = 0;
        if (!checked_mul_u64(protocol->combo_indices.len, sizeof(uint32_t), &combo_bytes_u64) ||
            !checked_mul_u64(protocol->metric_ids.len, sizeof(uint32_t), &metric_id_bytes_u64) ||
            !checked_mul_u64(protocol->chunk_count, sizeof(MetalChunk), &chunk_bytes_u64) ||
            !checked_mul_u64(total_rows, metric_count, &metric_value_count) ||
            !checked_mul_u64(metric_value_count, sizeof(float), &metric_bytes_u64) ||
            !metal_size_supported(total_rows) ||
            !metal_size_supported(combo_bytes_u64) ||
            !metal_size_supported(metric_id_bytes_u64) ||
            !metal_size_supported(chunk_bytes_u64) ||
            !metal_size_supported(metric_value_count) ||
            !metal_size_supported(metric_bytes_u64)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        const NSUInteger combo_bytes = static_cast<NSUInteger>(combo_bytes_u64);
        const NSUInteger metric_id_bytes = static_cast<NSUInteger>(metric_id_bytes_u64);
        const NSUInteger chunk_bytes = static_cast<NSUInteger>(chunk_bytes_u64);
        const NSUInteger metric_bytes = static_cast<NSUInteger>(metric_bytes_u64);
        const NSUInteger total_rows_metal = static_cast<NSUInteger>(total_rows);
        const size_t total_row_count_host = static_cast<size_t>(total_rows);
        const size_t metric_value_count_host = static_cast<size_t>(metric_value_count);
        const MetalLaunchInfo info{
            matrix->rows,
            matrix->cols,
            metric_count,
            protocol->chunk_count,
        };
        const uint64_t descriptor_generation =
            protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
        const bool cacheable = metal_protocol_descriptors_cacheable(protocol);
        const bool descriptors_resident =
            metal_protocol_descriptors_resident(matrix, protocol);
        const bool use_cached_spearman_target_ranks =
            spearman_target_rank_cache_eligible(matrix, protocol);
        const bool build_spearman_target_ranks =
            use_cached_spearman_target_ranks &&
            (!matrix->spearman_target_ranks_ready ||
             matrix->spearman_target_ranks_profile != matrix->precision_profile);

        id<MTLBuffer> combo_buffer = matrix->descriptor_combo_buffer;
        id<MTLBuffer> metric_id_buffer = matrix->descriptor_metric_id_buffer;
        id<MTLBuffer> chunk_buffer = matrix->descriptor_chunk_buffer;
        id<MTLBuffer> info_buffer = matrix->descriptor_info_buffer;
        if (!descriptors_resident) {
            std::vector<MetalChunk> chunks;
            chunks.reserve(protocol->chunk_count);
            for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
                const GafimeArityChunk& chunk = protocol->chunks[idx];
                chunks.push_back(MetalChunk{
                    chunk.arity,
                    metal_mi_bins_for_chunk(protocol, chunk),
                    metal_chunk_requires_scaled_covariance(matrix, protocol, chunk) ? 1u : 0u,
                    0u,
                    chunk.descriptor_offset,
                    chunk.combo_count,
                    chunk.combo_row_offset,
                });
            }
            combo_buffer = [matrix->device
                newBufferWithBytes:protocol->combo_indices.ptr
                length:combo_bytes
                options:MTLResourceStorageModeShared];
            metric_id_buffer = [matrix->device
                newBufferWithBytes:protocol->metric_ids.ptr
                length:metric_id_bytes
                options:MTLResourceStorageModeShared];
            chunk_buffer = [matrix->device
                newBufferWithBytes:chunks.data()
                length:chunk_bytes
                options:MTLResourceStorageModeShared];
            info_buffer = [matrix->device
                newBufferWithBytes:&info
                length:sizeof(MetalLaunchInfo)
                options:MTLResourceStorageModeShared];
            if (cacheable && combo_buffer != nil && metric_id_buffer != nil &&
                chunk_buffer != nil && info_buffer != nil) {
                matrix->descriptor_combo_buffer = combo_buffer;
                matrix->descriptor_metric_id_buffer = metric_id_buffer;
                matrix->descriptor_chunk_buffer = chunk_buffer;
                matrix->descriptor_info_buffer = info_buffer;
                matrix->descriptor_profile = matrix->precision_profile;
                matrix->descriptor_generation = descriptor_generation;
                matrix->descriptor_combo_len = protocol->combo_indices.len;
                matrix->descriptor_metric_id_len = protocol->metric_ids.len;
                matrix->descriptor_chunk_count = protocol->chunk_count;
                matrix->descriptor_shape_hint_count = protocol->shape_hint_count;
            }
        }
        id<MTLBuffer> metric_buffer = [matrix->device
            newBufferWithLength:metric_bytes
            options:(matrix->managed_storage ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared)];
        if (combo_buffer == nil || metric_id_buffer == nil || chunk_buffer == nil ||
            info_buffer == nil || metric_buffer == nil) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        id<MTLBuffer> rank_info_buffer = nil;
        id<MTLBuffer> selected_index_buffer = nil;
        id<MTLBuffer> selected_metric_buffer = nil;
        id<MTLBuffer> partial_score_buffer = nil;
        id<MTLBuffer> partial_index_buffer = nil;
        NSUInteger gather_item_count = 0;
        size_t output_row_count_host = 0;
        if (ranked_output) {
            const uint32_t partial_blocks = metal_topk_partial_block_count(total_rows, output_rows);
            const MetalRankInfo rank_info{
                total_rows,
                metric_count,
                primary_metric_index(protocol),
                static_cast<uint32_t>(output_rows),
                partial_blocks,
            };
            const MTLResourceOptions result_storage =
                matrix->managed_storage ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared;
            uint64_t selected_index_bytes_u64 = 0;
            uint64_t selected_metric_items = 0;
            uint64_t selected_metric_bytes_u64 = 0;
            uint64_t partial_items = 0;
            uint64_t partial_score_bytes_u64 = 0;
            uint64_t partial_index_bytes_u64 = 0;
            if (!checked_mul_u64(output_rows, sizeof(uint32_t), &selected_index_bytes_u64) ||
                !checked_mul_u64(output_rows, metric_count, &selected_metric_items) ||
                !checked_mul_u64(selected_metric_items, sizeof(float), &selected_metric_bytes_u64) ||
                !checked_mul_u64(partial_blocks, output_rows, &partial_items) ||
                !checked_mul_u64(partial_items, sizeof(float), &partial_score_bytes_u64) ||
                !checked_mul_u64(partial_items, sizeof(uint32_t), &partial_index_bytes_u64) ||
                !metal_size_supported(output_rows) ||
                !metal_size_supported(selected_index_bytes_u64) ||
                !metal_size_supported(selected_metric_items) ||
                !metal_size_supported(selected_metric_bytes_u64) ||
                !metal_size_supported(partial_score_bytes_u64) ||
                !metal_size_supported(partial_index_bytes_u64)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            const NSUInteger selected_index_bytes =
                static_cast<NSUInteger>(selected_index_bytes_u64);
            const NSUInteger selected_metric_bytes =
                static_cast<NSUInteger>(selected_metric_bytes_u64);
            const NSUInteger partial_score_bytes =
                static_cast<NSUInteger>(partial_score_bytes_u64);
            const NSUInteger partial_index_bytes =
                static_cast<NSUInteger>(partial_index_bytes_u64);
            gather_item_count = static_cast<NSUInteger>(selected_metric_items);
            output_row_count_host = static_cast<size_t>(output_rows);
            rank_info_buffer = [matrix->device
                newBufferWithBytes:&rank_info
                length:sizeof(MetalRankInfo)
                options:MTLResourceStorageModeShared];
            selected_index_buffer = [matrix->device
                newBufferWithLength:selected_index_bytes
                options:result_storage];
            selected_metric_buffer = [matrix->device
                newBufferWithLength:selected_metric_bytes
                options:result_storage];
            partial_score_buffer = [matrix->device
                newBufferWithLength:partial_score_bytes
                options:MTLResourceStorageModePrivate];
            partial_index_buffer = [matrix->device
                newBufferWithLength:partial_index_bytes
                options:MTLResourceStorageModePrivate];
            if (rank_info_buffer == nil || selected_index_buffer == nil ||
                selected_metric_buffer == nil || partial_score_buffer == nil ||
                partial_index_buffer == nil) {
                return GAFIME_STATUS_OUT_OF_MEMORY;
            }
        }
        id<MTLCommandBuffer> command_buffer = [matrix->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const MTLSize per_candidate_group = MTLSizeMake(kMetalReduceWidth, 1, 1);
        if (build_spearman_target_ranks) {
            const NSUInteger rank_threadgroups = static_cast<NSUInteger>(
                1 + (matrix->rows - 1) / kMetalReduceWidth
            );
            [encoder setComputePipelineState:matrix->spearman_target_rank_pipeline];
            [encoder setBuffer:matrix->target offset:0 atIndex:0];
            [encoder setBuffer:matrix->spearman_target_ranks offset:0 atIndex:1];
            [encoder setBuffer:info_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(rank_threadgroups, 1, 1)
                threadsPerThreadgroup:per_candidate_group];
        }
        [encoder setComputePipelineState:matrix->score_pipeline];
        [encoder setBuffer:matrix->features offset:0 atIndex:0];
        [encoder setBuffer:matrix->target offset:0 atIndex:1];
        [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
        [encoder setBuffer:combo_buffer offset:0 atIndex:3];
        [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
        [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
        [encoder setBuffer:metric_buffer offset:0 atIndex:6];
        [encoder setBuffer:info_buffer offset:0 atIndex:7];
        const MTLSize per_candidate_grid = MTLSizeMake(total_rows_metal, 1, 1);
        [encoder dispatchThreadgroups:per_candidate_grid threadsPerThreadgroup:per_candidate_group];

        // MI + Spearman also use one threadgroup per candidate. The compute
        // encoder serializes dependent dispatches on the shared metric buffer,
        // so the continuous pass (which zeroes the MI/Spearman slots) is
        // visible before these overwrite them.
        if (protocol_has_metric(protocol, GAFIME_METRIC_MUTUAL_INFO)) {
            [encoder setComputePipelineState:matrix->mi_pipeline];
            [encoder setBuffer:matrix->features offset:0 atIndex:0];
            [encoder setBuffer:matrix->target offset:0 atIndex:1];
            [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
            [encoder setBuffer:combo_buffer offset:0 atIndex:3];
            [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
            [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
            [encoder setBuffer:metric_buffer offset:0 atIndex:6];
            [encoder setBuffer:info_buffer offset:0 atIndex:7];
            [encoder dispatchThreadgroups:per_candidate_grid threadsPerThreadgroup:per_candidate_group];
        }
        if (protocol_has_metric(protocol, GAFIME_METRIC_SPEARMAN)) {
            const uint32_t use_cached_spearman_target_ranks_u32 =
                use_cached_spearman_target_ranks ? 1u : 0u;
            [encoder setComputePipelineState:matrix->spearman_pipeline];
            [encoder setBuffer:matrix->features offset:0 atIndex:0];
            [encoder setBuffer:matrix->target offset:0 atIndex:1];
            [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
            [encoder setBuffer:combo_buffer offset:0 atIndex:3];
            [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
            [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
            [encoder setBuffer:metric_buffer offset:0 atIndex:6];
            [encoder setBuffer:info_buffer offset:0 atIndex:7];
            [encoder setBuffer:matrix->spearman_target_ranks offset:0 atIndex:8];
            [encoder setBytes:&use_cached_spearman_target_ranks_u32
                length:sizeof(use_cached_spearman_target_ranks_u32)
                atIndex:9];
            [encoder dispatchThreadgroups:per_candidate_grid threadsPerThreadgroup:per_candidate_group];
        }
        if (ranked_output) {
            const uint32_t partial_blocks = metal_topk_partial_block_count(total_rows, output_rows);
            id<MTLComputePipelineState> partial_topk_pipeline = protocol->rank.descending != 0
                ? matrix->partial_topk_desc_pipeline
                : matrix->partial_topk_asc_pipeline;
            [encoder setComputePipelineState:partial_topk_pipeline];
            [encoder setBuffer:metric_buffer offset:0 atIndex:0];
            [encoder setBuffer:partial_score_buffer offset:0 atIndex:1];
            [encoder setBuffer:partial_index_buffer offset:0 atIndex:2];
            [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(partial_blocks, 1, 1) threadsPerThreadgroup:per_candidate_group];

            id<MTLComputePipelineState> merge_topk_pipeline = protocol->rank.descending != 0
                ? matrix->merge_topk_desc_pipeline
                : matrix->merge_topk_asc_pipeline;
            [encoder setComputePipelineState:merge_topk_pipeline];
            [encoder setBuffer:partial_score_buffer offset:0 atIndex:0];
            [encoder setBuffer:partial_index_buffer offset:0 atIndex:1];
            [encoder setBuffer:selected_index_buffer offset:0 atIndex:2];
            [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:per_candidate_group];

            const MTLSize gather_grid =
                MTLSizeMake(1 + (gather_item_count - 1) / kMetalReduceWidth, 1, 1);
            [encoder setComputePipelineState:matrix->gather_pipeline];
            [encoder setBuffer:metric_buffer offset:0 atIndex:0];
            [encoder setBuffer:selected_index_buffer offset:0 atIndex:1];
            [encoder setBuffer:selected_metric_buffer offset:0 atIndex:2];
            [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
            [encoder dispatchThreadgroups:gather_grid threadsPerThreadgroup:per_candidate_group];
        }
        [encoder endEncoding];
        if (matrix->managed_storage) {
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            if (blit == nil) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            if (ranked_output) {
                [blit synchronizeResource:selected_index_buffer];
                [blit synchronizeResource:selected_metric_buffer];
            } else {
                [blit synchronizeResource:metric_buffer];
            }
            [blit endEncoding];
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        if (build_spearman_target_ranks) {
            matrix->spearman_target_ranks_ready = true;
            matrix->spearman_target_ranks_profile = matrix->precision_profile;
        }
        result_out->flags = 0;
        if (!ranked_output) {
            const float* values = static_cast<const float*>(metric_buffer.contents);
            std::vector<float> metric_values(values, values + metric_value_count_host);
            std::vector<uint32_t> rows = all_rows(total_row_count_host);
            return write_result_rows(protocol, result_out, metric_values, rows, false);
        }

        const uint32_t* selected_values = static_cast<const uint32_t*>(selected_index_buffer.contents);
        std::vector<uint32_t> rows(selected_values, selected_values + output_row_count_host);
        size_t selected_count = 0;
        while (selected_count < output_row_count_host && rows[selected_count] != UINT32_MAX) {
            ++selected_count;
        }
        rows.resize(selected_count);
        if (selected_count == 0) {
            result_out->row_count = 0;
            return GAFIME_STATUS_OK;
        }

        const float* selected_metric_values = static_cast<const float*>(selected_metric_buffer.contents);
        uint64_t selected_metric_count = 0;
        if (!checked_mul_u64(selected_count, metric_count, &selected_metric_count) ||
            !host_size_supported(selected_metric_count)) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const size_t selected_metric_count_host = static_cast<size_t>(selected_metric_count);
        std::vector<float> metric_values(
            selected_metric_values,
            selected_metric_values + selected_metric_count_host
        );
        return write_result_rows(protocol, result_out, metric_values, rows, true);
    }
#else
    (void)matrix_handle;
    (void)protocol;
    (void)result_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_precision_capabilities(
    uint32_t device_id,
    GafimePrecisionCapabilities* capabilities_out
) try {
    if (capabilities_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    GafimeGpuDeviceInfo info{};
    const int device_status = gafime_gpu_device_info(device_id, &info);
    if (device_status != GAFIME_STATUS_OK) {
        return device_status;
    }
    std::memset(capabilities_out, 0, sizeof(*capabilities_out));
    capabilities_out->abi_version = GAFIME_PRECISION_ABI_VERSION;
    capabilities_out->backend_kind = GAFIME_BACKEND_METAL;
    capabilities_out->profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32;
    capabilities_out->storage_dtype_mask = GAFIME_DTYPE_MASK_F32;
    capabilities_out->result_dtype_mask = GAFIME_DTYPE_MASK_F32;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id,
    const GafimePrecisionMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) try {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    if (matrix_desc == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (matrix_desc->abi_version != GAFIME_PRECISION_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (matrix_desc->profile != GAFIME_PRECISION_FP32) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (matrix_desc->dtype != GAFIME_DTYPE_F32 ||
        matrix_desc->layout != GAFIME_MATRIX_ROW_MAJOR ||
        matrix_desc->flags != 0 || matrix_desc->reserved32 != 0 ||
        matrix_desc->rows == 0 || matrix_desc->cols == 0 ||
        matrix_desc->row_stride != matrix_desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t value : matrix_desc->reserved) {
        if (value != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    MatrixSizes sizes{};
    if (!checked_matrix_sizes(matrix_desc->rows, matrix_desc->cols, &sizes) ||
        matrix_desc->bytes != sizes.feature_bytes) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const GafimeMatrixDesc legacy_desc{
        GAFIME_ABI_VERSION,
        GAFIME_DTYPE_F32,
        GAFIME_MATRIX_ROW_MAJOR,
        0,
        matrix_desc->rows,
        matrix_desc->cols,
        matrix_desc->cols,
        matrix_desc->bytes,
    };
    const int status = gafime_gpu_matrix_alloc(device_id, &legacy_desc, matrix_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(*matrix_out);
    if (matrix == nullptr) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    matrix->precision_profile = GAFIME_PRECISION_FP32;
#endif
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload_f32_v2(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) try {
#if GAFIME_HAS_METAL_RUNTIME
    const auto* matrix = static_cast<const MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->precision_profile != GAFIME_PRECISION_FP32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return gafime_gpu_matrix_upload(matrix_handle, features_host, target_host, rows, cols);
#else
    (void)matrix_handle;
    (void)features_host;
    (void)target_host;
    (void)rows;
    (void)cols;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload_f64_v2(
    GafimeGpuMatrix matrix_handle,
    const double* features_host,
    const double* target_host,
    uint64_t rows,
    uint32_t cols
) {
    (void)matrix_handle;
    (void)features_host;
    (void)target_host;
    (void)rows;
    (void)cols;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f32_v2(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) try {
#if GAFIME_HAS_METAL_RUNTIME
    const auto* matrix = static_cast<const MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->precision_profile != GAFIME_PRECISION_FP32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return gafime_gpu_matrix_update_target(matrix_handle, target_host, rows);
#else
    (void)matrix_handle;
    (void)target_host;
    (void)rows;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f64_v2(
    GafimeGpuMatrix matrix_handle,
    const double* target_host,
    uint64_t rows
) {
    (void)matrix_handle;
    (void)target_host;
    (void)rows;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execute_f32_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result_out
) try {
    if (protocol == nullptr || protocol->abi_version != GAFIME_PRECISION_ABI_VERSION) {
        return protocol == nullptr ? GAFIME_STATUS_INVALID_ARGUMENT : GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->profile != GAFIME_PRECISION_FP32 || protocol->base == nullptr) {
        return protocol->profile == GAFIME_PRECISION_FP32
            ? GAFIME_STATUS_INVALID_ARGUMENT
            : GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    for (uint64_t value : protocol->reserved) {
        if (value != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
#if GAFIME_HAS_METAL_RUNTIME
    const auto* matrix = static_cast<const MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->precision_profile != GAFIME_PRECISION_FP32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
#endif
    return gafime_gpu_execute(matrix_handle, protocol->base, result_out);
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_execute_f64_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result_out
) {
    (void)matrix_handle;
    (void)protocol;
    (void)result_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) try {
    if (protocol == nullptr || protocol->abi_version != GAFIME_PRECISION_ABI_VERSION) {
        return protocol == nullptr ? GAFIME_STATUS_INVALID_ARGUMENT : GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->profile != GAFIME_PRECISION_FP32 || protocol->base == nullptr) {
        return protocol->profile == GAFIME_PRECISION_FP32
            ? GAFIME_STATUS_INVALID_ARGUMENT
            : GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    for (uint64_t value : protocol->reserved) {
        if (value != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
#if GAFIME_HAS_METAL_RUNTIME
    const auto* matrix = static_cast<const MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->precision_profile != GAFIME_PRECISION_FP32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
#endif
    return gafime_gpu_execution_memory_peak(matrix_handle, protocol->base, peak_bytes_out);
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

}  // extern "C"
