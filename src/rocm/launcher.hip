#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rocm_api.hpp"
#include "kernels.hpp"
#include "../common/gpu_abi_impl.hpp"

namespace gafime_rocm_v1 {

hipError_t launch_target_stats(
    const float* target,
    uint64_t n_samples,
    TargetStatsDevice* target_stats,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        kernel::target_stats_kernel,
        dim3(1),
        dim3(kThreadsPerBlock),
        0,
        stream,
        target,
        n_samples,
        target_stats
    );
    return hipGetLastError();
}

hipError_t launch_unary_feature_stats(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    UnaryFeatureStatsDevice* feature_stats,
    hipStream_t stream
) {
    dim3 grid(n_features);
    dim3 block(kThreadsPerBlock);
    hipLaunchKernelGGL(
        kernel::unary_feature_stats_kernel,
        grid,
        block,
        0,
        stream,
        features,
        n_samples,
        n_features,
        feature_stats
    );
    return hipGetLastError();
}

hipError_t launch_continuous_chunk(
    const float* features,
    const float* target,
    const float* column_means,
    const TargetStatsDevice* target_stats,
    const UnaryFeatureStatsDevice* feature_stats,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t features_are_finite,
    uint32_t target_is_finite,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values,
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    if (arity == 1 && features_are_finite != 0u && target_is_finite != 0u &&
        target_stats != nullptr && feature_stats != nullptr) {
        hipLaunchKernelGGL(
            kernel::score_continuous_unary_all_finite_chunk_kernel,
            grid,
            block,
            0,
            stream,
            features,
            target,
            target_stats,
            feature_stats,
            combo_indices,
            n_samples,
            descriptor_offset,
            combo_count,
            metric_ids,
            metric_count,
            metric_values
        );
        return hipGetLastError();
    }
    hipLaunchKernelGGL(
        kernel::score_continuous_chunk_kernel,
        grid,
        block,
        0,
        stream,
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        n_features,
        arity,
        descriptor_offset,
        combo_count,
        metric_ids,
        metric_count,
        metric_values
    );
    return hipGetLastError();
}

hipError_t launch_mutual_info_chunk(
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
    float* metric_values,
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    hipLaunchKernelGGL(
        kernel::score_mutual_info_chunk_kernel,
        grid,
        block,
        0,
        stream,
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        n_features,
        arity,
        descriptor_offset,
        combo_count,
        metric_count,
        metric_index,
        bins,
        metric_values
    );
    return hipGetLastError();
}

hipError_t launch_spearman_chunk(
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
    float* metric_values,
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    hipLaunchKernelGGL(
        kernel::score_spearman_chunk_kernel,
        grid,
        block,
        0,
        stream,
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        n_features,
        arity,
        descriptor_offset,
        combo_count,
        metric_count,
        metric_index,
        metric_values
    );
    return hipGetLastError();
}

}  // namespace gafime_rocm_v1

namespace {


struct GraphChunkShape {
    uint32_t arity;
    uint64_t descriptor_offset;
    uint64_t combo_count;
};

struct HipMatrix {
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t arch_class;
    bool unified_memory;
    uint64_t rows;
    uint32_t cols;
    bool features_are_finite;
    bool target_is_finite;
    float* features;
    float* target;
    float* column_means;
    gafime_rocm_v1::TargetStatsDevice* target_stats;
    gafime_rocm_v1::UnaryFeatureStatsDevice* feature_stats;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    float* metric_values;
    uint64_t metric_value_capacity;
    // Graph capture cache (device-copy stream-capture, mirror of the CUDA host;
    // ROCm has no permutation path so there is no resident-target-copy variant).
    hipStream_t graph_stream;
    hipGraph_t graph;
    hipGraphExec_t graph_exec;
    bool graph_valid;
    uint32_t graph_chunk_count;
    uint64_t graph_combo_len;
    uint64_t graph_metric_id_len;
    uint64_t graph_metric_value_count;
    uintptr_t graph_combo_ptr;
    uintptr_t graph_metric_ids_ptr;
    uintptr_t graph_metric_values_ptr;
    uint64_t graph_metric_signature;
    std::vector<GraphChunkShape> graph_chunk_shapes;
};

int hip_status(hipError_t status) {
    return status == hipSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

bool ascii_contains_ci(const char* text, const char* needle) {
    if (text == nullptr || needle == nullptr || needle[0] == '\0') {
        return false;
    }
    const size_t needle_len = std::strlen(needle);
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        size_t idx = 0;
        while (idx < needle_len && cursor[idx] != '\0') {
            const unsigned char left = static_cast<unsigned char>(cursor[idx]);
            const unsigned char right = static_cast<unsigned char>(needle[idx]);
            if (std::tolower(left) != std::tolower(right)) {
                break;
            }
            ++idx;
        }
        if (idx == needle_len) {
            return true;
        }
    }
    return false;
}

uint64_t parse_gfx_arch(const char* arch) {
    if (arch == nullptr || std::strncmp(arch, "gfx", 3) != 0) {
        return GAFIME_GPU_ARCH_UNKNOWN;
    }
    uint64_t value = 0;
    bool saw_digit = false;
    for (const char* cursor = arch + 3; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            break;
        }
        saw_digit = true;
        value = value * 10 + static_cast<uint64_t>(*cursor - '0');
    }
    return saw_digit ? value : GAFIME_GPU_ARCH_UNKNOWN;
}

bool rocm_arch_is_cdna(const char* arch) {
    if (arch == nullptr) {
        return false;
    }
    return std::strncmp(arch, "gfx908", 6) == 0 ||
        std::strncmp(arch, "gfx90a", 6) == 0 ||
        std::strncmp(arch, "gfx940", 6) == 0 ||
        std::strncmp(arch, "gfx941", 6) == 0 ||
        std::strncmp(arch, "gfx942", 6) == 0 ||
        std::strncmp(arch, "gfx950", 6) == 0;
}

bool rocm_arch_is_rdna(const char* arch) {
    if (arch == nullptr) {
        return false;
    }
    return std::strncmp(arch, "gfx10", 5) == 0 ||
        std::strncmp(arch, "gfx11", 5) == 0 ||
        std::strncmp(arch, "gfx12", 5) == 0;
}

bool rocm_device_is_integrated(const hipDeviceProp_t& props) {
    return props.integrated != 0 ||
        ascii_contains_ci(props.name, "Radeon Graphics") ||
        ascii_contains_ci(props.name, "APU") ||
        ascii_contains_ci(props.name, "iGPU") ||
        ascii_contains_ci(props.name, "780M") ||
        ascii_contains_ci(props.name, "760M") ||
        ascii_contains_ci(props.name, "680M") ||
        ascii_contains_ci(props.name, "660M") ||
        ascii_contains_ci(props.name, "870M") ||
        ascii_contains_ci(props.name, "860M");
}

uint64_t rocm_arch_class(const hipDeviceProp_t& props) {
    if (rocm_arch_is_cdna(props.gcnArchName)) {
        return GAFIME_GPU_ARCH_AMD_CDNA;
    }
    if (rocm_arch_is_rdna(props.gcnArchName)) {
        return GAFIME_GPU_ARCH_AMD_RDNA;
    }
    return parse_gfx_arch(props.gcnArchName);
}

uint32_t rocm_device_flags(const hipDeviceProp_t& props) {
    uint32_t flags = 0;
    if (rocm_device_is_integrated(props)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (props.managedMemory != 0 || props.concurrentManagedAccess != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY;
    }
    if (rocm_arch_is_rdna(props.gcnArchName)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_AMD_RDNA;
    }
    if (rocm_arch_is_cdna(props.gcnArchName)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_AMD_CDNA;
    }
    if (rocm_arch_is_cdna(props.gcnArchName) || props.memoryBusWidth >= 512) {
        flags |= GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
    }
    return flags;
}

bool rocm_use_managed_memory(const hipDeviceProp_t& props) {
    return rocm_device_is_integrated(props);
}

template <typename T>
int allocate_hip_storage(T** ptr, size_t bytes, bool managed) {
    if (bytes == 0) {
        *ptr = nullptr;
        return GAFIME_STATUS_OK;
    }
    if (managed) {
        hipError_t status = hipMallocManaged(reinterpret_cast<void**>(ptr), bytes);
        if (status == hipSuccess) {
            return GAFIME_STATUS_OK;
        }
    }
    return hip_status(hipMalloc(ptr, bytes));
}

int copy_to_hip_storage(void* dst, const void* src, size_t bytes, bool managed) {
    if (bytes == 0) {
        return GAFIME_STATUS_OK;
    }
    if (managed) {
        std::memcpy(dst, src, bytes);
        return GAFIME_STATUS_OK;
    }
    return hip_status(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

int copy_from_hip_storage(void* dst, const void* src, size_t bytes, bool managed) {
    if (bytes == 0) {
        return GAFIME_STATUS_OK;
    }
    if (managed) {
        const int status = hip_status(hipDeviceSynchronize());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        std::memcpy(dst, src, bytes);
        return GAFIME_STATUS_OK;
    }
    return hip_status(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO || metric_id == GAFIME_METRIC_SPEARMAN;
}

int validate_matrix_desc(const GafimeMatrixDesc* desc) {
    if (desc == nullptr || desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (desc->rows == 0 || desc->cols == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, const HipMatrix* matrix) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION || protocol->backend_kind != GAFIME_BACKEND_ROCM) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    if (protocol->rank.top_k != 0 && protocol->rank.include_ties != 0) {
        // Host-side top-k is supported; tie-inclusive ranking is not implemented.
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (!metric_supported(protocol->metric_ids.ptr[idx])) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.family != GAFIME_FAMILY_CONTINUOUS || chunk.arity == 0 || chunk.arity > protocol->max_arity) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        const uint64_t required = chunk.descriptor_offset + chunk.combo_count * chunk.arity;
        if (required > protocol->combo_indices.len) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return GAFIME_STATUS_OK;
}

uint64_t planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t rows = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        rows += protocol->chunks[chunk_idx].combo_count;
    }
    return rows;
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t planned = planned_row_count(protocol);
    // With top-k ranking only the selected rows are written, so the result table
    // need only hold min(planned, top_k).
    const uint64_t rows = protocol->rank.top_k == 0
        ? planned
        : std::min<uint64_t>(planned, protocol->rank.top_k);
    if (result->capacity < rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows > 0 && (
        result->combo_indices == nullptr ||
        result->metric_values == nullptr ||
        result->ranks == nullptr ||
        result->families == nullptr ||
        result->candidate_ids == nullptr ||
        result->row_flags == nullptr
    )) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

template <typename T>
int ensure_device_capacity(T** ptr, uint64_t* capacity, uint64_t required, bool managed) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    T* next = nullptr;
    const uint64_t next_capacity = std::max(required, (*capacity == 0 ? required : *capacity * 2));
    const size_t bytes = static_cast<size_t>(next_capacity) * sizeof(T);
    int status = allocate_hip_storage(&next, bytes, managed);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    static_cast<void>(hipFree(*ptr));
    *ptr = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

void compute_column_means_host(const float* features, uint64_t rows, uint32_t cols, std::vector<float>& means) {
    means.assign(cols, 0.0f);
    std::vector<double> sums(cols, 0.0);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += static_cast<double>(features[row * cols + col]);
        }
    }
    const double inv_rows = 1.0 / static_cast<double>(rows);
    for (uint32_t col = 0; col < cols; ++col) {
        means[col] = static_cast<float>(sums[col] * inv_rows);
    }
}

void build_feature_major_host(
    const float* features,
    uint64_t rows,
    uint32_t cols,
    std::vector<float>& resident_features,
    bool* features_are_finite
) {
    bool finite = true;
    resident_features.assign(static_cast<size_t>(rows) * cols, 0.0f);
    for (uint32_t col = 0; col < cols; ++col) {
        const uint64_t feature_base = static_cast<uint64_t>(col) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const float value = features[static_cast<size_t>(row) * cols + col];
            finite = finite && std::isfinite(value);
            resident_features[static_cast<size_t>(feature_base + row)] =
                value;
        }
    }
    if (features_are_finite != nullptr) {
        *features_are_finite = finite;
    }
}

int refresh_target_stats(HipMatrix* matrix, hipStream_t stream) {
    if (matrix == nullptr || matrix->target == nullptr || matrix->target_stats == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(gafime_rocm_v1::launch_target_stats(
        matrix->target,
        matrix->rows,
        matrix->target_stats,
        stream
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return hip_status(hipStreamSynchronize(stream));
}

int refresh_unary_feature_stats(HipMatrix* matrix, hipStream_t stream) {
    if (matrix == nullptr || matrix->features == nullptr || matrix->feature_stats == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(gafime_rocm_v1::launch_unary_feature_stats(
        matrix->features,
        matrix->rows,
        matrix->cols,
        matrix->feature_stats,
        stream
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return hip_status(hipStreamSynchronize(stream));
}

bool all_finite_host(const float* values, uint64_t len) {
    bool finite = true;
    for (uint64_t idx = 0; idx < len; ++idx) {
        finite = finite && std::isfinite(values[idx]);
    }
    return finite;
}

uint32_t mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = 96;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t hint = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (hint == 12 || hint == 24 || hint == 48 || hint == 96) {
            bins = hint;
        }
    }
    return bins;
}

void write_result_row_at(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    uint64_t dst_row,
    uint64_t src_global_row,
    uint64_t combo_base,
    uint32_t arity,
    uint32_t rank,
    uint64_t candidate_id
) {
    const uint32_t max_arity = result->max_arity;
    const uint32_t metric_count = result->metric_count;
    for (uint32_t slot = 0; slot < max_arity; ++slot) {
        result->combo_indices[dst_row * max_arity + slot] =
            slot < arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
    }
    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const float value = metric_idx < protocol->metric_ids.len
            ? metric_values[src_global_row * protocol->metric_ids.len + metric_idx]
            : 0.0f;
        result->metric_values[dst_row * metric_count + metric_idx] = value;
    }
    result->ranks[dst_row] = rank;
    result->families[dst_row] = GAFIME_FAMILY_CONTINUOUS;
    result->candidate_ids[dst_row] = candidate_id;
    result->row_flags[dst_row] = 0;
}

int write_result_rows_host(const GafimeLaunchProtocol* protocol, GafimeResultTable* result, const std::vector<float>& metric_values) {
    const uint32_t top_k = protocol->rank.top_k;

    if (top_k == 0) {
        uint64_t output_row = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
            const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
            for (uint64_t row = 0; row < chunk.combo_count; ++row) {
                const uint64_t combo_base = chunk.descriptor_offset + row * chunk.arity;
                write_result_row_at(protocol, result, metric_values, output_row, output_row,
                    combo_base, chunk.arity, static_cast<uint32_t>(output_row), output_row);
                ++output_row;
            }
        }
        result->row_count = output_row;
        return GAFIME_STATUS_OK;
    }

    // Host-side top-k selection by the primary metric. The ROCm host already
    // copies every candidate's metrics back to the host, so no device selection
    // kernel is needed. Mirrors the CPU/CUDA ranking: order by the raw primary
    // metric value (descending or ascending), tie-break by candidate id ascending,
    // skip non-finite scores.
    uint32_t primary_index = 0;
    bool found = false;
    for (uint32_t i = 0; i < protocol->metric_ids.len; ++i) {
        if (protocol->metric_ids.ptr[i] == protocol->rank.primary_metric) {
            primary_index = i;
            found = true;
            break;
        }
    }
    if (!found) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const bool descending = protocol->rank.descending != 0;

    struct Candidate {
        uint64_t id;
        float score;
        uint64_t combo_base;
        uint32_t arity;
    };
    std::vector<Candidate> candidates;
    uint64_t global_row = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const float score =
                metric_values[global_row * protocol->metric_ids.len + primary_index];
            if (std::isfinite(score)) {
                candidates.push_back(
                    {global_row, score, chunk.descriptor_offset + row * chunk.arity, chunk.arity});
            }
            ++global_row;
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [descending](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) {
                return descending ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        });
    const uint64_t selected = std::min<uint64_t>(top_k, candidates.size());
    for (uint64_t r = 0; r < selected; ++r) {
        const Candidate& c = candidates[r];
        write_result_row_at(protocol, result, metric_values, r, c.id, c.combo_base, c.arity,
            static_cast<uint32_t>(r), c.id);
    }
    result->row_count = selected;
    return GAFIME_STATUS_OK;
}

uint64_t compute_metric_signature(const GafimeLaunchProtocol* protocol) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    if (protocol->metric_ids.ptr != nullptr) {
        for (uint64_t i = 0; i < protocol->metric_ids.len; ++i) {
            mix(static_cast<uint64_t>(protocol->metric_ids.ptr[i]));
        }
    }
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        mix(static_cast<uint64_t>(chunk.metric_mask));
        mix(static_cast<uint64_t>(mi_bins_for_chunk(protocol, chunk)));
    }
    return hash;
}

// The whole multi-arity sweep (every chunk + metric) launched on one stream, so a
// single stream-capture covers the whole graph. Mirrors the CUDA host.
int launch_score_kernels(HipMatrix* matrix, const GafimeLaunchProtocol* protocol, hipStream_t stream) {
    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.combo_count == 0) {
            continue;
        }
        int status = hip_status(gafime_rocm_v1::launch_continuous_chunk(
            matrix->features,
            matrix->target,
            matrix->column_means,
            matrix->target_stats,
            matrix->feature_stats,
            matrix->combo_indices,
            matrix->rows,
            matrix->cols,
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
            matrix->features_are_finite ? 1u : 0u,
            matrix->target_is_finite ? 1u : 0u,
            matrix->metric_ids,
            static_cast<uint32_t>(protocol->metric_ids.len),
            matrix->metric_values + metric_row_offset * protocol->metric_ids.len,
            stream
        ));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        for (uint32_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
            if (protocol->metric_ids.ptr[metric_idx] != GAFIME_METRIC_MUTUAL_INFO) {
                continue;
            }
            float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
            status = hip_status(gafime_rocm_v1::launch_mutual_info_chunk(
                matrix->features,
                matrix->target,
                matrix->column_means,
                matrix->combo_indices,
                matrix->rows,
                matrix->cols,
                chunk.arity,
                chunk.descriptor_offset,
                chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len),
                metric_idx,
                mi_bins_for_chunk(protocol, chunk),
                out,
                stream
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }
        for (uint32_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
            if (protocol->metric_ids.ptr[metric_idx] != GAFIME_METRIC_SPEARMAN) {
                continue;
            }
            float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
            status = hip_status(gafime_rocm_v1::launch_spearman_chunk(
                matrix->features,
                matrix->target,
                matrix->column_means,
                matrix->combo_indices,
                matrix->rows,
                matrix->cols,
                chunk.arity,
                chunk.descriptor_offset,
                chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len),
                metric_idx,
                out,
                stream
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }
        metric_row_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

bool graph_shape_matches(const HipMatrix* matrix, const GafimeLaunchProtocol* protocol, uint64_t metric_value_count) {
    if (!matrix->graph_valid || matrix->graph_exec == nullptr) {
        return false;
    }
    if (matrix->graph_chunk_count != protocol->chunk_count ||
        matrix->graph_combo_len != protocol->combo_indices.len ||
        matrix->graph_metric_id_len != protocol->metric_ids.len ||
        matrix->graph_metric_value_count != metric_value_count ||
        matrix->graph_combo_ptr != reinterpret_cast<uintptr_t>(matrix->combo_indices) ||
        matrix->graph_metric_ids_ptr != reinterpret_cast<uintptr_t>(matrix->metric_ids) ||
        matrix->graph_metric_values_ptr != reinterpret_cast<uintptr_t>(matrix->metric_values) ||
        matrix->graph_metric_signature != compute_metric_signature(protocol)) {
        return false;
    }
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        const GraphChunkShape& shape = matrix->graph_chunk_shapes[idx];
        if (shape.arity != chunk.arity || shape.descriptor_offset != chunk.descriptor_offset ||
            shape.combo_count != chunk.combo_count) {
            return false;
        }
    }
    return true;
}

void store_graph_shape(HipMatrix* matrix, const GafimeLaunchProtocol* protocol, uint64_t metric_value_count) {
    matrix->graph_chunk_count = protocol->chunk_count;
    matrix->graph_combo_len = protocol->combo_indices.len;
    matrix->graph_metric_id_len = protocol->metric_ids.len;
    matrix->graph_metric_value_count = metric_value_count;
    matrix->graph_combo_ptr = reinterpret_cast<uintptr_t>(matrix->combo_indices);
    matrix->graph_metric_ids_ptr = reinterpret_cast<uintptr_t>(matrix->metric_ids);
    matrix->graph_metric_values_ptr = reinterpret_cast<uintptr_t>(matrix->metric_values);
    matrix->graph_metric_signature = compute_metric_signature(protocol);
    matrix->graph_chunk_shapes.clear();
    matrix->graph_chunk_shapes.reserve(protocol->chunk_count);
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        matrix->graph_chunk_shapes.push_back(
            GraphChunkShape{chunk.arity, chunk.descriptor_offset, chunk.combo_count});
    }
}

void destroy_graph_cache(HipMatrix* matrix) {
    if (matrix == nullptr) {
        return;
    }
    if (matrix->graph_exec != nullptr) {
        static_cast<void>(hipGraphExecDestroy(matrix->graph_exec));
        matrix->graph_exec = nullptr;
    }
    if (matrix->graph != nullptr) {
        static_cast<void>(hipGraphDestroy(matrix->graph));
        matrix->graph = nullptr;
    }
    matrix->graph_valid = false;
    matrix->graph_chunk_shapes.clear();
}

bool graph_requested(const GafimeLaunchProtocol* protocol) {
    return (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0 &&
        protocol->rank.top_k == 0 &&
        protocol->permutations.permutation_count == 0;
}

int execute_score_kernels(HipMatrix* matrix, const GafimeLaunchProtocol* protocol, uint64_t metric_value_count, bool* graph_replayed) {
    *graph_replayed = false;
    if (!graph_requested(protocol)) {
        int status = launch_score_kernels(matrix, protocol, nullptr);
        if (status == GAFIME_STATUS_OK) {
            status = hip_status(hipDeviceSynchronize());
        }
        return status;
    }

    if (matrix->graph_stream == nullptr) {
        int status = hip_status(hipStreamCreateWithFlags(&matrix->graph_stream, hipStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    if (!graph_shape_matches(matrix, protocol, metric_value_count)) {
        destroy_graph_cache(matrix);
        int status = hip_status(hipStreamBeginCapture(matrix->graph_stream, hipStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = launch_score_kernels(matrix, protocol, matrix->graph_stream);
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(hipStreamEndCapture(matrix->graph_stream, &matrix->graph));
            destroy_graph_cache(matrix);
            return status;
        }
        hipGraph_t next_graph = nullptr;
        status = hip_status(hipStreamEndCapture(matrix->graph_stream, &next_graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        hipGraphExec_t next_exec = nullptr;
        status = hip_status(hipGraphInstantiate(&next_exec, next_graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(hipGraphDestroy(next_graph));
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph = next_graph;
        matrix->graph_exec = next_exec;
        matrix->graph_valid = true;
        store_graph_shape(matrix, protocol, metric_value_count);
    }

    int status = hip_status(hipGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    }
    return status;
}

void update_graph_result_flag(GafimeResultTable* result, bool graph_replayed) {
    result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    if (graph_replayed) {
        result->flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    }
}

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(uint32_t device_id, GafimeGpuDeviceInfo* info_out) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    hipError_t status = hipSetDevice(static_cast<int>(device_id));
    if (status != hipSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != hipSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = GAFIME_BACKEND_ROCM;
    info_out->device_id = device_id;
    info_out->flags = rocm_device_flags(props);
    std::strncpy(info_out->name, props.name, sizeof(info_out->name) - 1);
    info_out->name[sizeof(info_out->name) - 1] = '\0';
    info_out->total_global_mem_bytes = static_cast<uint64_t>(props.totalGlobalMem);
    info_out->multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info_out->warp_size = static_cast<uint32_t>(props.warpSize);
    info_out->compute_major = static_cast<uint32_t>(props.major);
    info_out->compute_minor = static_cast<uint32_t>(props.minor);
    int driver_version = 0;
    int runtime_version = 0;
    if (hipDriverGetVersion(&driver_version) == hipSuccess) {
        info_out->driver_version = static_cast<uint32_t>(driver_version);
    }
    if (hipRuntimeGetVersion(&runtime_version) == hipSuccess) {
        info_out->runtime_version = static_cast<uint32_t>(runtime_version);
    }
    info_out->reserved[0] = rocm_arch_class(props);
    info_out->reserved[1] = parse_gfx_arch(props.gcnArchName);
    info_out->reserved[2] = static_cast<uint64_t>(props.sharedMemPerBlock);
    info_out->reserved[3] = static_cast<uint64_t>(props.regsPerBlock);
    info_out->reserved[4] = static_cast<uint64_t>(props.memoryBusWidth);
    info_out->reserved[5] = static_cast<uint64_t>(props.memoryClockRate);
    info_out->reserved[6] = static_cast<uint64_t>(props.maxThreadsPerMultiProcessor);
    info_out->reserved[7] = static_cast<uint64_t>(props.maxThreadsPerBlock);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(uint32_t device_id, GafimeGpuGraphCapability* capability_out) {
    (void)device_id;
    return gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_ROCM,
        GAFIME_GRAPH_STREAM_CAPTURE,
        capability_out
    );
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(uint32_t device_id, const GafimeMatrixDesc* matrix_desc, GafimeGpuMatrix* matrix_out) {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    int status = validate_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(hipSetDevice(static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    hipDeviceProp_t props{};
    status = hip_status(hipGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const bool managed_memory = rocm_use_managed_memory(props);

    auto* matrix = new HipMatrix{};
    matrix->device_id = device_id;
    matrix->device_flags = rocm_device_flags(props);
    matrix->arch_class = rocm_arch_class(props);
    matrix->unified_memory = managed_memory;
    matrix->rows = matrix_desc->rows;
    matrix->cols = matrix_desc->cols;
    matrix->features_are_finite = true;
    matrix->target_is_finite = true;
    matrix->features = nullptr;
    matrix->target = nullptr;
    matrix->column_means = nullptr;
    matrix->target_stats = nullptr;
    matrix->feature_stats = nullptr;
    matrix->combo_indices = nullptr;
    matrix->combo_capacity = 0;
    matrix->metric_ids = nullptr;
    matrix->metric_id_capacity = 0;
    matrix->metric_values = nullptr;
    matrix->metric_value_capacity = 0;

    const size_t feature_bytes = static_cast<size_t>(matrix->rows) * matrix->cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(matrix->cols) * sizeof(float);
    const size_t target_stats_bytes = sizeof(gafime_rocm_v1::TargetStatsDevice);
    const size_t feature_stats_bytes =
        static_cast<size_t>(matrix->cols) * sizeof(gafime_rocm_v1::UnaryFeatureStatsDevice);
    status = allocate_hip_storage(&matrix->features, feature_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        delete matrix;
        return status;
    }
    status = allocate_hip_storage(&matrix->target, target_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(hipFree(matrix->features));
        delete matrix;
        return status;
    }
    status = allocate_hip_storage(&matrix->column_means, mean_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(hipFree(matrix->target));
        static_cast<void>(hipFree(matrix->features));
        delete matrix;
        return status;
    }
    status = allocate_hip_storage(&matrix->target_stats, target_stats_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(hipFree(matrix->column_means));
        static_cast<void>(hipFree(matrix->target));
        static_cast<void>(hipFree(matrix->features));
        delete matrix;
        return status;
    }
    status = allocate_hip_storage(&matrix->feature_stats, feature_stats_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(hipFree(matrix->target_stats));
        static_cast<void>(hipFree(matrix->column_means));
        static_cast<void>(hipFree(matrix->target));
        static_cast<void>(hipFree(matrix->features));
        delete matrix;
        return status;
    }
    *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<float> column_means;
    compute_column_means_host(features_host, rows, cols, column_means);
    std::vector<float> resident_features;
    bool features_are_finite = true;
    build_feature_major_host(features_host, rows, cols, resident_features, &features_are_finite);
    const bool target_is_finite = all_finite_host(target_host, rows);
    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(float);
    status = copy_to_hip_storage(matrix->features, resident_features.data(), feature_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(matrix->target, target_host, target_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(matrix->column_means, column_means.data(), mean_bytes, matrix->unified_memory);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    matrix->features_are_finite = features_are_finite;
    matrix->target_is_finite = target_is_finite;
    destroy_graph_cache(matrix);
    status = refresh_unary_feature_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return refresh_target_stats(matrix, nullptr);
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(GafimeGpuMatrix matrix_handle, const float* target_host, uint64_t rows) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const bool target_is_finite = all_finite_host(target_host, rows);
    int status = copy_to_hip_storage(matrix->target, target_host, target_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK && matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        if (matrix->target_is_finite != target_is_finite) {
            destroy_graph_cache(matrix);
        }
        matrix->target_is_finite = target_is_finite;
        status = refresh_target_stats(matrix, nullptr);
    }
    return status;
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr) {
        return;
    }
    static_cast<void>(hipSetDevice(static_cast<int>(matrix->device_id)));
    destroy_graph_cache(matrix);
    if (matrix->graph_stream != nullptr) {
        static_cast<void>(hipStreamDestroy(matrix->graph_stream));
        matrix->graph_stream = nullptr;
    }
    static_cast<void>(hipFree(matrix->column_means));
    static_cast<void>(hipFree(matrix->target_stats));
    static_cast<void>(hipFree(matrix->feature_stats));
    static_cast<void>(hipFree(matrix->target));
    static_cast<void>(hipFree(matrix->features));
    static_cast<void>(hipFree(matrix->metric_values));
    static_cast<void>(hipFree(matrix->metric_ids));
    static_cast<void>(hipFree(matrix->combo_indices));
    delete matrix;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    if (total_rows == 0) {
        result_out->row_count = 0;
        return GAFIME_STATUS_OK;
    }
    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(
        &matrix->combo_indices,
        &matrix->combo_capacity,
        protocol->combo_indices.len,
        matrix->unified_memory
    );
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &matrix->metric_ids,
            &matrix->metric_id_capacity,
            protocol->metric_ids.len,
            matrix->unified_memory
        );
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &matrix->metric_values,
            &matrix->metric_value_capacity,
            metric_value_count,
            matrix->unified_memory
        );
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = copy_to_hip_storage(
        matrix->combo_indices,
        protocol->combo_indices.ptr,
        static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t),
        matrix->unified_memory
    );
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(
            matrix->metric_ids,
            protocol->metric_ids.ptr,
            static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t),
            matrix->unified_memory
        );
    }
    if (status == GAFIME_STATUS_OK && matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    bool graph_replayed = false;
    status = execute_score_kernels(matrix, protocol, metric_value_count, &graph_replayed);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<float> metric_values(static_cast<size_t>(metric_value_count), 0.0f);
    status = copy_from_hip_storage(
        metric_values.data(),
        matrix->metric_values,
        static_cast<size_t>(metric_value_count) * sizeof(float),
        matrix->unified_memory
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = write_result_rows_host(protocol, result_out, metric_values);
    if (status == GAFIME_STATUS_OK) {
        update_graph_result_flag(result_out, graph_replayed);
    }
    return status;
}

}  // extern "C"
