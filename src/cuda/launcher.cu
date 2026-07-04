#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_api.hpp"
#include "kernels.cuh"
#include "../common/gpu_abi_impl.hpp"

namespace gafime_cuda_v1 {

cudaError_t launch_continuous_chunk(
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
    float* metric_values,
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    kernel::score_continuous_chunk_kernel<<<grid, block, 0, stream>>>(
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
    return cudaGetLastError();
}

cudaError_t launch_mutual_info_chunk(
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
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    kernel::score_mutual_info_chunk_kernel<<<grid, block, 0, stream>>>(
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
    return cudaGetLastError();
}

cudaError_t launch_spearman_chunk(
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
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    kernel::score_spearman_chunk_kernel<<<grid, block, 0, stream>>>(
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
    return cudaGetLastError();
}

cudaError_t launch_select_topk(
    const float* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    uint32_t descending,
    uint32_t* selected_indices,
    cudaStream_t stream
) {
    kernel::select_topk_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
        metric_values,
        row_count,
        metric_count,
        primary_metric_index,
        top_k,
        descending,
        selected_indices
    );
    return cudaGetLastError();
}

cudaError_t launch_copy_selected_metric_rows(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values,
    cudaStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) {
        return cudaSuccess;
    }
    const uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>((total + threads - 1) / threads);
    kernel::copy_selected_metric_rows_kernel<<<blocks, threads, 0, stream>>>(
        metric_values,
        selected_indices,
        selected_count,
        metric_count,
        selected_metric_values
    );
    return cudaGetLastError();
}

cudaError_t launch_selected_metric_max(
    const float* metric_values,
    const uint64_t* candidate_ids,
    uint64_t selected_count,
    uint64_t total_rows,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_max,
    cudaStream_t stream
) {
    if (selected_count == 0 || metric_count == 0) {
        return cudaSuccess;
    }
    dim3 grid(metric_count);
    dim3 block(kThreadsPerBlock);
    kernel::selected_metric_max_kernel<<<grid, block, 0, stream>>>(
        metric_values,
        candidate_ids,
        selected_count,
        total_rows,
        metric_ids,
        metric_count,
        metric_max
    );
    return cudaGetLastError();
}

cudaError_t launch_accumulate_exceedances(
    const float* metric_max,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    const float* observed_metric_values,
    uint64_t selected_count,
    uint32_t* exceedance_counts,
    cudaStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) {
        return cudaSuccess;
    }
    const uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>((total + threads - 1) / threads);
    kernel::accumulate_exceedances_kernel<<<blocks, threads, 0, stream>>>(
        metric_max,
        metric_ids,
        metric_count,
        observed_metric_values,
        selected_count,
        exceedance_counts
    );
    return cudaGetLastError();
}

}  // namespace gafime_cuda_v1

namespace {

struct GraphChunkShape {
    uint32_t arity;
    uint64_t descriptor_offset;
    uint64_t combo_count;
};

struct CudaMatrix {
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t arch_class;
    uint64_t rows;
    uint32_t cols;
    float* features;
    float* target;
    float* column_means;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    float* metric_values;
    uint64_t metric_value_capacity;
    uint32_t* selected_indices;
    uint64_t selected_index_capacity;
    float* selected_metric_values;
    uint64_t selected_metric_value_capacity;
    uint64_t* significance_candidate_ids;
    uint64_t significance_candidate_id_capacity;
    float* significance_observed_values;
    uint64_t significance_observed_value_capacity;
    float* significance_metric_max;
    uint64_t significance_metric_max_capacity;
    uint32_t* significance_exceedance_counts;
    uint64_t significance_exceedance_count_capacity;
    std::vector<float> target_host;
    float* permutation_target_host;
    uint64_t permutation_target_capacity;
    cudaStream_t graph_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_valid;
    bool graph_has_target_copy;
    uint32_t graph_chunk_count;
    uint64_t graph_combo_len;
    uint64_t graph_metric_id_len;
    uint64_t graph_metric_value_count;
    uintptr_t graph_target_copy_ptr;
    uintptr_t graph_combo_ptr;
    uintptr_t graph_metric_ids_ptr;
    uintptr_t graph_metric_values_ptr;
    uint64_t graph_metric_signature;
    std::vector<GraphChunkShape> graph_chunk_shapes;
};

int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

uint64_t cuda_arch_class(const cudaDeviceProp& props) {
    const uint32_t sm = static_cast<uint32_t>(props.major * 10 + props.minor);
    if (props.major >= 10) {
        return GAFIME_GPU_ARCH_NVIDIA_BLACKWELL;
    }
    if (props.major == 9) {
        return GAFIME_GPU_ARCH_NVIDIA_HOPPER;
    }
    if (props.major == 8) {
        return GAFIME_GPU_ARCH_NVIDIA_AMPERE;
    }
    if (props.major == 7 && props.minor >= 5) {
        return GAFIME_GPU_ARCH_NVIDIA_TURING;
    }
    return sm;
}

int cuda_device_attr(uint32_t device_id, cudaDeviceAttr attr) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attr, static_cast<int>(device_id)) != cudaSuccess) {
        return 0;
    }
    return value;
}

uint32_t cuda_device_flags(const cudaDeviceProp& props, uint32_t device_id) {
    uint32_t flags = 0;
    const int integrated = cuda_device_attr(device_id, cudaDevAttrIntegrated);
    const int managed_memory = cuda_device_attr(device_id, cudaDevAttrManagedMemory);
    const int concurrent_managed = cuda_device_attr(device_id, cudaDevAttrConcurrentManagedAccess);
    const int unified_addressing = cuda_device_attr(device_id, cudaDevAttrUnifiedAddressing);
    const int memory_bus_width = cuda_device_attr(device_id, cudaDevAttrGlobalMemoryBusWidth);
    const int l2_cache_size = cuda_device_attr(device_id, cudaDevAttrL2CacheSize);
    if (props.integrated != 0 || integrated != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (props.managedMemory != 0 || props.concurrentManagedAccess != 0 ||
        managed_memory != 0 || concurrent_managed != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY;
    }
    if (props.unifiedAddressing != 0 || unified_addressing != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    }
    if (memory_bus_width >= 384 || l2_cache_size >= (40 * 1024 * 1024)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
    }
    return flags;
}

void tune_cuda_kernels_for_device(const cudaDeviceProp& props) {
    const cudaFuncCache shared_heavy_cache =
        props.major >= 7 ? cudaFuncCachePreferShared : cudaFuncCachePreferL1;
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::score_continuous_chunk_kernel,
        shared_heavy_cache
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::score_mutual_info_chunk_kernel,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::score_spearman_chunk_kernel,
        shared_heavy_cache
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::select_topk_kernel,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::selected_metric_max_kernel,
        cudaFuncCachePreferShared
    ));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    const int carveout = props.major >= 7 ? 100 : 50;
    static_cast<void>(cudaFuncSetAttribute(
        gafime_cuda_v1::kernel::score_mutual_info_chunk_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout
    ));
    static_cast<void>(cudaFuncSetAttribute(
        gafime_cuda_v1::kernel::score_spearman_chunk_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout
    ));
#else
    (void)props;
#endif
    static_cast<void>(cudaGetLastError());
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

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON ||
        metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO ||
        metric_id == GAFIME_METRIC_SPEARMAN;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, const CudaMatrix* matrix) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION || protocol->backend_kind != GAFIME_BACKEND_CUDA) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        const uint64_t expected_offsets =
            static_cast<uint64_t>(protocol->permutations.permutation_count) * matrix->rows;
        if (protocol->permutations.target_offsets.len != 0 &&
            protocol->permutations.target_offsets.len != expected_offsets) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        if (protocol->permutations.target_offsets.len != 0 &&
            protocol->permutations.target_offsets.ptr == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
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
        bool primary_metric_found = false;
        for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
            if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
                primary_metric_found = true;
                break;
            }
        }
        if (!primary_metric_found) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
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

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t planned_rows) {
    if (protocol->rank.top_k == 0) {
        return planned_rows;
    }
    return std::min<uint64_t>(planned_rows, protocol->rank.top_k);
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
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

uint32_t primary_metric_index(const GafimeLaunchProtocol* protocol) {
    if (protocol->rank.top_k == 0) {
        return 0;
    }
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(protocol->metric_ids.len); ++idx) {
        if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
            return idx;
        }
    }
    return 0;
}

void destroy_graph_cache(CudaMatrix* matrix) {
    if (matrix == nullptr) {
        return;
    }
    if (matrix->graph_exec != nullptr) {
        cudaGraphExecDestroy(matrix->graph_exec);
        matrix->graph_exec = nullptr;
    }
    if (matrix->graph != nullptr) {
        cudaGraphDestroy(matrix->graph);
        matrix->graph = nullptr;
    }
    matrix->graph_valid = false;
    matrix->graph_has_target_copy = false;
    matrix->graph_target_copy_ptr = 0;
    matrix->graph_chunk_shapes.clear();
}

template <typename T>
int ensure_device_capacity(T** ptr, uint64_t* capacity, uint64_t required) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    T* next = nullptr;
    const uint64_t next_capacity = std::max(required, (*capacity == 0 ? required : *capacity * 2));
    const size_t bytes = static_cast<size_t>(next_capacity) * sizeof(T);
    int status = cuda_status(cudaMalloc(&next, bytes));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaFree(*ptr);
    *ptr = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

int ensure_pinned_target_capacity(CudaMatrix* matrix, uint64_t required) {
    if (required <= matrix->permutation_target_capacity) {
        return GAFIME_STATUS_OK;
    }
    float* next = nullptr;
    const uint64_t next_capacity = std::max(
        required,
        matrix->permutation_target_capacity == 0 ? required : matrix->permutation_target_capacity * 2
    );
    int status = cuda_status(cudaHostAlloc(&next, static_cast<size_t>(next_capacity) * sizeof(float), cudaHostAllocDefault));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaFreeHost(matrix->permutation_target_host);
    matrix->permutation_target_host = next;
    matrix->permutation_target_capacity = next_capacity;
    destroy_graph_cache(matrix);
    return GAFIME_STATUS_OK;
}

void compute_column_means_host(
    const float* features_host,
    uint64_t rows,
    uint32_t cols,
    std::vector<float>& means
) {
    means.assign(cols, 0.0f);
    std::vector<double> sums(cols, 0.0);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += static_cast<double>(features_host[row * cols + col]);
        }
    }
    const double inv_rows = 1.0 / static_cast<double>(rows);
    for (uint32_t col = 0; col < cols; ++col) {
        means[col] = static_cast<float>(sums[col] * inv_rows);
    }
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

int launch_mi_kernel_for_bins(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    const GafimeArityChunk& chunk,
    uint64_t metric_row_offset,
    uint32_t metric_index,
    uint32_t bins,
    cudaStream_t stream
) {
    float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
    return cuda_status(gafime_cuda_v1::launch_mutual_info_chunk(
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
        metric_index,
        bins,
        out,
        stream
    ));
}

int launch_score_kernels(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    cudaStream_t stream
) {
    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.combo_count == 0) {
            continue;
        }
        const int status = cuda_status(gafime_cuda_v1::launch_continuous_chunk(
            matrix->features,
            matrix->target,
            matrix->column_means,
            matrix->combo_indices,
            matrix->rows,
            matrix->cols,
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
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
            const int mi_status = launch_mi_kernel_for_bins(
                matrix,
                protocol,
                chunk,
                metric_row_offset,
                metric_idx,
                mi_bins_for_chunk(protocol, chunk),
                stream
            );
            if (mi_status != GAFIME_STATUS_OK) {
                return mi_status;
            }
        }
        for (uint32_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
            if (protocol->metric_ids.ptr[metric_idx] != GAFIME_METRIC_SPEARMAN) {
                continue;
            }
            float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
            const int sp_status = cuda_status(gafime_cuda_v1::launch_spearman_chunk(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len), metric_idx, out, stream
            ));
            if (sp_status != GAFIME_STATUS_OK) {
                return sp_status;
            }
        }
        metric_row_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

uint64_t compute_metric_signature(const GafimeLaunchProtocol* protocol) {
    // FNV-1a over the metric ids plus each chunk's metric mask and resolved MI
    // bins. The resident metric_ids buffer keeps a stable address, so the pointer
    // check alone cannot tell [pearson] from [mutual_info]; this signature does,
    // and a graph captured for one metric/bin set is never replayed for another.
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

bool graph_shape_matches(
    const CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool target_copy_required
) {
    if (!matrix->graph_valid || matrix->graph_exec == nullptr) {
        return false;
    }
    const uintptr_t target_copy_ptr = target_copy_required
        ? reinterpret_cast<uintptr_t>(matrix->permutation_target_host)
        : 0;
    if (matrix->graph_has_target_copy != target_copy_required ||
        matrix->graph_target_copy_ptr != target_copy_ptr ||
        matrix->graph_chunk_count != protocol->chunk_count ||
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
        if (shape.arity != chunk.arity ||
            shape.descriptor_offset != chunk.descriptor_offset ||
            shape.combo_count != chunk.combo_count) {
            return false;
        }
    }
    return true;
}

void store_graph_shape(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool target_copy_required
) {
    matrix->graph_has_target_copy = target_copy_required;
    matrix->graph_chunk_count = protocol->chunk_count;
    matrix->graph_combo_len = protocol->combo_indices.len;
    matrix->graph_metric_id_len = protocol->metric_ids.len;
    matrix->graph_metric_value_count = metric_value_count;
    matrix->graph_target_copy_ptr = target_copy_required
        ? reinterpret_cast<uintptr_t>(matrix->permutation_target_host)
        : 0;
    matrix->graph_combo_ptr = reinterpret_cast<uintptr_t>(matrix->combo_indices);
    matrix->graph_metric_ids_ptr = reinterpret_cast<uintptr_t>(matrix->metric_ids);
    matrix->graph_metric_values_ptr = reinterpret_cast<uintptr_t>(matrix->metric_values);
    matrix->graph_chunk_shapes.clear();
    matrix->graph_chunk_shapes.reserve(protocol->chunk_count);
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        matrix->graph_chunk_shapes.push_back(GraphChunkShape{
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
        });
    }
}

bool graph_requested(const GafimeLaunchProtocol* protocol) {
    return (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0 &&
        protocol->rank.top_k == 0;
}

int execute_score_kernels(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool* graph_replayed
) {
    *graph_replayed = false;
    if (!graph_requested(protocol)) {
        int status = launch_score_kernels(matrix, protocol, 0);
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaDeviceSynchronize());
        }
        return status;
    }

    if (matrix->graph_stream == nullptr) {
        int status = cuda_status(cudaStreamCreateWithFlags(&matrix->graph_stream, cudaStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    if (!graph_shape_matches(matrix, protocol, metric_value_count, false)) {
        destroy_graph_cache(matrix);
        int status = cuda_status(cudaStreamBeginCapture(matrix->graph_stream, cudaStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = launch_score_kernels(matrix, protocol, matrix->graph_stream);
        if (status != GAFIME_STATUS_OK) {
            cudaStreamEndCapture(matrix->graph_stream, &matrix->graph);
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraph_t next_graph = nullptr;
        status = cuda_status(cudaStreamEndCapture(matrix->graph_stream, &next_graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraphExec_t next_exec = nullptr;
        status = cuda_status(cudaGraphInstantiate(&next_exec, next_graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            cudaGraphDestroy(next_graph);
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph = next_graph;
        matrix->graph_exec = next_exec;
        matrix->graph_valid = true;
        store_graph_shape(matrix, protocol, metric_value_count, false);
    }

    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    }
    return status;
}

uint64_t splitmix64_next(uint64_t* state) {
    uint64_t value = (*state += 0x9E3779B97F4A7C15ull);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

uint64_t mix_permutation_seed(uint64_t base_seed, uint64_t permutation_index) {
    uint64_t state =
        base_seed ^
        0xA5A5A5A5ull * 0x9E3779B97F4A7C15ull ^
        permutation_index * 0xD1B54A32D192ED03ull;
    return splitmix64_next(&state);
}

uint64_t bounded_random(uint64_t* state, uint64_t bound) {
    if (bound <= 1) {
        return 0;
    }
    return splitmix64_next(state) % bound;
}

int fill_permutation_target(
    const CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint32_t permutation_index
) {
    if (matrix->target_host.size() != matrix->rows || matrix->permutation_target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t rows = matrix->rows;
    if (protocol->permutations.target_offsets.len != 0) {
        const uint64_t base = static_cast<uint64_t>(permutation_index) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const uint64_t source_row = protocol->permutations.target_offsets.ptr[base + row];
            if (source_row >= rows) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            matrix->permutation_target_host[row] = matrix->target_host[static_cast<size_t>(source_row)];
        }
        return GAFIME_STATUS_OK;
    }

    std::vector<uint64_t> order(static_cast<size_t>(rows), 0);
    for (uint64_t row = 0; row < rows; ++row) {
        order[static_cast<size_t>(row)] = row;
    }
    uint64_t rng_state = mix_permutation_seed(
        protocol->permutations.seed,
        static_cast<uint64_t>(permutation_index)
    );
    for (uint64_t row = rows; row > 1; --row) {
        const uint64_t swap_idx = bounded_random(&rng_state, row);
        std::swap(order[static_cast<size_t>(row - 1)], order[static_cast<size_t>(swap_idx)]);
    }
    for (uint64_t row = 0; row < rows; ++row) {
        matrix->permutation_target_host[row] = matrix->target_host[static_cast<size_t>(order[static_cast<size_t>(row)])];
    }
    return GAFIME_STATUS_OK;
}

int execute_permutation_iteration(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool use_graph,
    bool* graph_replayed
) {
    *graph_replayed = false;
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    if (!use_graph) {
        int status = cuda_status(cudaMemcpy(matrix->target, matrix->permutation_target_host, target_bytes, cudaMemcpyHostToDevice));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = launch_score_kernels(matrix, protocol, 0);
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaDeviceSynchronize());
        }
        return status;
    }

    if (matrix->graph_stream == nullptr) {
        int status = cuda_status(cudaStreamCreateWithFlags(&matrix->graph_stream, cudaStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    if (!graph_shape_matches(matrix, protocol, metric_value_count, true)) {
        destroy_graph_cache(matrix);
        int status = cuda_status(cudaStreamBeginCapture(matrix->graph_stream, cudaStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = cuda_status(cudaMemcpyAsync(
            matrix->target,
            matrix->permutation_target_host,
            target_bytes,
            cudaMemcpyHostToDevice,
            matrix->graph_stream
        ));
        if (status == GAFIME_STATUS_OK) {
            status = launch_score_kernels(matrix, protocol, matrix->graph_stream);
        }
        if (status != GAFIME_STATUS_OK) {
            cudaStreamEndCapture(matrix->graph_stream, &matrix->graph);
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraph_t next_graph = nullptr;
        status = cuda_status(cudaStreamEndCapture(matrix->graph_stream, &next_graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraphExec_t next_exec = nullptr;
        status = cuda_status(cudaGraphInstantiate(&next_exec, next_graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            cudaGraphDestroy(next_graph);
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph = next_graph;
        matrix->graph_exec = next_exec;
        matrix->graph_valid = true;
        store_graph_shape(matrix, protocol, metric_value_count, true);
    }

    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
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

bool locate_combo_for_global_row(
    const GafimeLaunchProtocol* protocol,
    uint64_t global_row,
    const GafimeArityChunk** chunk_out,
    uint64_t* local_row_out
) {
    uint64_t row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (global_row < row_offset + chunk.combo_count) {
            *chunk_out = &chunk;
            *local_row_out = global_row - row_offset;
            return true;
        }
        row_offset += chunk.combo_count;
    }
    return false;
}

int write_result_rows_host(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    const std::vector<uint32_t>* selected_indices
) {
    const uint32_t max_arity = result->max_arity;
    const uint32_t metric_count = result->metric_count;
    const uint64_t output_rows = selected_indices == nullptr
        ? planned_row_count(protocol)
        : static_cast<uint64_t>(selected_indices->size());

    for (uint64_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint64_t global_row = selected_indices == nullptr
            ? output_row
            : static_cast<uint64_t>((*selected_indices)[output_row]);
        const GafimeArityChunk* chunk = nullptr;
        uint64_t local_row = 0;
        if (!locate_combo_for_global_row(protocol, global_row, &chunk, &local_row)) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const uint64_t combo_base = chunk->descriptor_offset + local_row * chunk->arity;
        for (uint32_t slot = 0; slot < max_arity; ++slot) {
            result->combo_indices[output_row * max_arity + slot] =
                slot < chunk->arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const float value = metric_idx < protocol->metric_ids.len
                ? metric_values[output_row * protocol->metric_ids.len + metric_idx]
                : 0.0f;
            result->metric_values[output_row * metric_count + metric_idx] = value;
        }
        result->ranks[output_row] = static_cast<uint32_t>(output_row);
        result->families[output_row] = GAFIME_FAMILY_CONTINUOUS;
        result->candidate_ids[output_row] = global_row;
        result->row_flags[output_row] = 0;
    }
    result->row_count = output_rows;
    return GAFIME_STATUS_OK;
}

int prepare_protocol_device_buffers(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) {
    int status = ensure_device_capacity(&matrix->combo_indices, &matrix->combo_capacity, protocol->combo_indices.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_ids, &matrix->metric_id_capacity, protocol->metric_ids.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_values, &matrix->metric_value_capacity, metric_value_count);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const size_t combo_bytes = static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t);
    status = cuda_status(cudaMemcpy(matrix->combo_indices, protocol->combo_indices.ptr, combo_bytes, cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(matrix->metric_ids, protocol->metric_ids.ptr, metric_id_bytes, cudaMemcpyHostToDevice));
    }
    return status;
}

int validate_significance_table(
    const GafimeLaunchProtocol* protocol,
    const CudaMatrix* matrix,
    const GafimePermutationSignificanceTable* significance,
    uint64_t total_rows
) {
    if (significance == nullptr || significance->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->metric_count != protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->row_count == 0) {
        return GAFIME_STATUS_OK;
    }
    if (significance->candidate_ids == nullptr ||
        significance->observed_metric_values == nullptr ||
        significance->p_values == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t row = 0; row < significance->row_count; ++row) {
        if (significance->candidate_ids[row] >= total_rows) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (matrix->target_host.size() != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int execute_permutation_pvalues(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance,
    uint64_t total_rows,
    uint64_t metric_value_count
) {
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    const uint64_t selected_count = significance->row_count;
    if (selected_count == 0) {
        return GAFIME_STATUS_OK;
    }

    int status = ensure_pinned_target_capacity(matrix, matrix->rows);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(
        &matrix->significance_candidate_ids,
        &matrix->significance_candidate_id_capacity,
        selected_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(
        &matrix->significance_observed_values,
        &matrix->significance_observed_value_capacity,
        selected_count * metric_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(
        &matrix->significance_metric_max,
        &matrix->significance_metric_max_capacity,
        metric_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(
        &matrix->significance_exceedance_counts,
        &matrix->significance_exceedance_count_capacity,
        selected_count * metric_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = cuda_status(cudaMemcpy(
        matrix->significance_candidate_ids,
        significance->candidate_ids,
        static_cast<size_t>(selected_count) * sizeof(uint64_t),
        cudaMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            matrix->significance_observed_values,
            significance->observed_metric_values,
            static_cast<size_t>(selected_count * metric_count) * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(
            matrix->significance_exceedance_counts,
            0,
            static_cast<size_t>(selected_count * metric_count) * sizeof(uint32_t)
        ));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const bool use_graph = (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0;
    bool ignored_graph_replayed = false;
    const uint32_t permutation_count = protocol->permutations.permutation_count;
    for (uint32_t permutation_index = 0; permutation_index < permutation_count; ++permutation_index) {
        status = fill_permutation_target(matrix, protocol, permutation_index);
        if (status != GAFIME_STATUS_OK) {
            break;
        }
        status = execute_permutation_iteration(
            matrix,
            protocol,
            metric_value_count,
            use_graph,
            &ignored_graph_replayed
        );
        if (status != GAFIME_STATUS_OK) {
            break;
        }
        status = cuda_status(gafime_cuda_v1::launch_selected_metric_max(
            matrix->metric_values,
            matrix->significance_candidate_ids,
            selected_count,
            total_rows,
            matrix->metric_ids,
            metric_count,
            matrix->significance_metric_max,
            0
        ));
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(gafime_cuda_v1::launch_accumulate_exceedances(
                matrix->significance_metric_max,
                matrix->metric_ids,
                metric_count,
                matrix->significance_observed_values,
                selected_count,
                matrix->significance_exceedance_counts,
                0
            ));
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaDeviceSynchronize());
        }
        if (status != GAFIME_STATUS_OK) {
            break;
        }
    }

    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    const int restore_status = cuda_status(cudaMemcpy(
        matrix->target,
        matrix->target_host.data(),
        target_bytes,
        cudaMemcpyHostToDevice
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (restore_status != GAFIME_STATUS_OK) {
        return restore_status;
    }

    std::vector<uint32_t> counts(static_cast<size_t>(selected_count * metric_count), 0);
    status = cuda_status(cudaMemcpy(
        counts.data(),
        matrix->significance_exceedance_counts,
        counts.size() * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const float denom = static_cast<float>(permutation_count) + 1.0f;
    for (uint64_t idx = 0; idx < selected_count * metric_count; ++idx) {
        significance->p_values[idx] = (static_cast<float>(counts[static_cast<size_t>(idx)]) + 1.0f) / denom;
    }
    return GAFIME_STATUS_OK;
}

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(
    uint32_t device_id,
    GafimeGpuDeviceInfo* info_out
) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t status = cudaSetDevice(static_cast<int>(device_id));
    if (status != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    cudaDeviceProp props{};
    status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = GAFIME_BACKEND_CUDA;
    info_out->device_id = device_id;
    info_out->flags = cuda_device_flags(props, device_id);
    std::strncpy(info_out->name, props.name, sizeof(info_out->name) - 1);
    info_out->name[sizeof(info_out->name) - 1] = '\0';
    info_out->total_global_mem_bytes = static_cast<uint64_t>(props.totalGlobalMem);
    info_out->multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info_out->warp_size = static_cast<uint32_t>(props.warpSize);
    info_out->compute_major = static_cast<uint32_t>(props.major);
    info_out->compute_minor = static_cast<uint32_t>(props.minor);
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        info_out->driver_version = static_cast<uint32_t>(driver_version);
    }
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        info_out->runtime_version = static_cast<uint32_t>(runtime_version);
    }
    info_out->reserved[0] = cuda_arch_class(props);
    const int shared_optin = cuda_device_attr(device_id, cudaDevAttrMaxSharedMemoryPerBlockOptin);
    info_out->reserved[1] = static_cast<uint64_t>(
        shared_optin > 0 ? shared_optin : cuda_device_attr(device_id, cudaDevAttrMaxSharedMemoryPerBlock)
    );
    info_out->reserved[2] = static_cast<uint64_t>(cuda_device_attr(device_id, cudaDevAttrMaxRegistersPerBlock));
    info_out->reserved[3] = static_cast<uint64_t>(cuda_device_attr(device_id, cudaDevAttrL2CacheSize));
    info_out->reserved[4] = static_cast<uint64_t>(cuda_device_attr(device_id, cudaDevAttrGlobalMemoryBusWidth));
    info_out->reserved[5] = static_cast<uint64_t>(cuda_device_attr(device_id, cudaDevAttrMemoryClockRate));
    info_out->reserved[6] = static_cast<uint64_t>(cuda_device_attr(device_id, cudaDevAttrMaxThreadsPerMultiProcessor));
    info_out->reserved[7] = static_cast<uint64_t>(props.maxThreadsPerBlock);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
) {
    (void)device_id;
    const int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_CUDA,
        GAFIME_GRAPH_STREAM_CAPTURE,
        capability_out
    );
    if (status == GAFIME_STATUS_OK) {
        capability_out->supports_kernel_param_update = 0;
        capability_out->supports_device_ranking = 1;
        capability_out->max_captured_nodes = 64;
        capability_out->stable_pointer_flags = 1;
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;

    int status = validate_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaDeviceProp props{};
    status = cuda_status(cudaGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    tune_cuda_kernels_for_device(props);

    auto* matrix = new CudaMatrix{};
    matrix->device_id = device_id;
    matrix->device_flags = cuda_device_flags(props, device_id);
    matrix->arch_class = cuda_arch_class(props);
    matrix->rows = matrix_desc->rows;
    matrix->cols = matrix_desc->cols;
    matrix->features = nullptr;
    matrix->target = nullptr;
    matrix->column_means = nullptr;
    matrix->combo_indices = nullptr;
    matrix->combo_capacity = 0;
    matrix->metric_ids = nullptr;
    matrix->metric_id_capacity = 0;
    matrix->metric_values = nullptr;
    matrix->metric_value_capacity = 0;
    matrix->selected_indices = nullptr;
    matrix->selected_index_capacity = 0;
    matrix->selected_metric_values = nullptr;
    matrix->selected_metric_value_capacity = 0;
    matrix->significance_candidate_ids = nullptr;
    matrix->significance_candidate_id_capacity = 0;
    matrix->significance_observed_values = nullptr;
    matrix->significance_observed_value_capacity = 0;
    matrix->significance_metric_max = nullptr;
    matrix->significance_metric_max_capacity = 0;
    matrix->significance_exceedance_counts = nullptr;
    matrix->significance_exceedance_count_capacity = 0;
    matrix->permutation_target_host = nullptr;
    matrix->permutation_target_capacity = 0;
    matrix->graph_stream = nullptr;
    matrix->graph = nullptr;
    matrix->graph_exec = nullptr;
    matrix->graph_valid = false;
    matrix->graph_has_target_copy = false;
    matrix->graph_chunk_count = 0;
    matrix->graph_combo_len = 0;
    matrix->graph_metric_id_len = 0;
    matrix->graph_metric_value_count = 0;
    matrix->graph_target_copy_ptr = 0;
    matrix->graph_combo_ptr = 0;
    matrix->graph_metric_ids_ptr = 0;
    matrix->graph_metric_values_ptr = 0;
    matrix->graph_metric_signature = 0;

    const size_t feature_bytes = static_cast<size_t>(matrix->rows) * matrix->cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(matrix->cols) * sizeof(float);
    status = cuda_status(cudaMalloc(&matrix->features, feature_bytes));
    if (status != GAFIME_STATUS_OK) {
        delete matrix;
        return status;
    }
    status = cuda_status(cudaMalloc(&matrix->target, target_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->features);
        delete matrix;
        return status;
    }
    status = cuda_status(cudaMalloc(&matrix->column_means, mean_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->target);
        cudaFree(matrix->features);
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
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<float> column_means;
    compute_column_means_host(features_host, rows, cols, column_means);

    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(float);
    status = cuda_status(cudaMemcpy(matrix->features, features_host, feature_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    matrix->target_host.assign(target_host, target_host + rows);
    return cuda_status(cudaMemcpy(matrix->column_means, column_means.data(), mean_bytes, cudaMemcpyHostToDevice));
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        matrix->target_host.assign(target_host, target_host + rows);
    }
    return status;
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr) {
        return;
    }
    cudaSetDevice(static_cast<int>(matrix->device_id));
    destroy_graph_cache(matrix);
    if (matrix->graph_stream != nullptr) {
        cudaStreamDestroy(matrix->graph_stream);
    }
    cudaFree(matrix->column_means);
    cudaFree(matrix->target);
    cudaFree(matrix->features);
    cudaFree(matrix->metric_values);
    cudaFree(matrix->metric_ids);
    cudaFree(matrix->combo_indices);
    cudaFree(matrix->selected_metric_values);
    cudaFree(matrix->selected_indices);
    cudaFree(matrix->significance_exceedance_counts);
    cudaFree(matrix->significance_metric_max);
    cudaFree(matrix->significance_observed_values);
    cudaFree(matrix->significance_candidate_ids);
    cudaFreeHost(matrix->permutation_target_host);
    delete matrix;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    const uint64_t output_rows = output_row_count(protocol, total_rows);
    if (total_rows == 0) {
        result_out->row_count = 0;
        return GAFIME_STATUS_OK;
    }

    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(&matrix->combo_indices, &matrix->combo_capacity, protocol->combo_indices.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_ids, &matrix->metric_id_capacity, protocol->metric_ids.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_values, &matrix->metric_value_capacity, metric_value_count);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const size_t combo_bytes = static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t);
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);
    status = cuda_status(cudaMemcpy(matrix->combo_indices, protocol->combo_indices.ptr, combo_bytes, cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(matrix->metric_ids, protocol->metric_ids.ptr, metric_id_bytes, cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    bool graph_replayed = false;
    status = execute_score_kernels(matrix, protocol, metric_value_count, &graph_replayed);

    if (protocol->rank.top_k == 0) {
        std::vector<float> metric_values(static_cast<size_t>(total_rows) * protocol->metric_ids.len, 0.0f);
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(metric_values.data(), matrix->metric_values, metric_value_bytes, cudaMemcpyDeviceToHost));
        }
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        update_graph_result_flag(result_out, graph_replayed);
        return write_result_rows_host(protocol, result_out, metric_values, nullptr);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = ensure_device_capacity(&matrix->selected_indices, &matrix->selected_index_capacity, output_rows);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t selected_metric_value_count = output_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(
        &matrix->selected_metric_values,
        &matrix->selected_metric_value_capacity,
        selected_metric_value_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = cuda_status(gafime_cuda_v1::launch_select_topk(
        matrix->metric_values,
        total_rows,
        static_cast<uint32_t>(protocol->metric_ids.len),
        primary_metric_index(protocol),
        static_cast<uint32_t>(output_rows),
        protocol->rank.descending,
        matrix->selected_indices,
        0
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t copy_items = selected_metric_value_count;
    if (copy_items > 0) {
        status = cuda_status(gafime_cuda_v1::launch_copy_selected_metric_rows(
            matrix->metric_values,
            matrix->selected_indices,
            output_rows,
            static_cast<uint32_t>(protocol->metric_ids.len),
            matrix->selected_metric_values,
            0
        ));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<uint32_t> selected_indices(static_cast<size_t>(output_rows), UINT32_MAX);
    std::vector<float> selected_metric_values(
        static_cast<size_t>(selected_metric_value_count),
        0.0f
    );
    status = cuda_status(cudaMemcpy(
        selected_indices.data(),
        matrix->selected_indices,
        static_cast<size_t>(output_rows) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    ));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            selected_metric_values.data(),
            matrix->selected_metric_values,
            static_cast<size_t>(selected_metric_value_count) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    update_graph_result_flag(result_out, graph_replayed);
    return write_result_rows_host(protocol, result_out, selected_metric_values, &selected_indices);
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = validate_significance_table(protocol, matrix, significance_out, total_rows);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (total_rows == 0 || significance_out->row_count == 0) {
        return GAFIME_STATUS_OK;
    }

    status = prepare_protocol_device_buffers(matrix, protocol, metric_value_count);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return execute_permutation_pvalues(
        matrix,
        protocol,
        significance_out,
        total_rows,
        metric_value_count
    );
}

}  // extern "C"
