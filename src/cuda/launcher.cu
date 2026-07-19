#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "cuda_api.hpp"
#include "kernels.cuh"
#include "rt_launcher.cuh"
#include "../common/gpu_abi_impl.hpp"

namespace gafime_cuda_v1 {

cudaError_t launch_target_stats(
    const float* target,
    uint64_t n_samples,
    TargetStatsDevice* target_stats,
    cudaStream_t stream
) {
    kernel::target_stats_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
        target,
        n_samples,
        target_stats
    );
    return cudaGetLastError();
}

cudaError_t launch_unary_feature_stats(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    UnaryFeatureStatsDevice* feature_stats,
    cudaStream_t stream
) {
    dim3 grid(n_features);
    dim3 block(kThreadsPerBlock);
    kernel::unary_feature_stats_kernel<<<grid, block, 0, stream>>>(
        features,
        n_samples,
        n_features,
        feature_stats
    );
    return cudaGetLastError();
}

template <uint32_t Arity>
cudaError_t launch_continuous_chunk_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values,
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    kernel::score_continuous_chunk_kernel_static<Arity><<<grid, block, 0, stream>>>(
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        descriptor_offset,
        combo_count,
        metric_ids,
        metric_count,
        metric_values
    );
    return cudaGetLastError();
}

cudaError_t launch_continuous_chunk(
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
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    if (arity == 1 && features_are_finite != 0u && target_is_finite != 0u &&
        target_stats != nullptr && feature_stats != nullptr) {
        kernel::score_continuous_unary_all_finite_chunk_kernel<<<grid, block, 0, stream>>>(
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
        return cudaGetLastError();
    }
    switch (arity) {
    case 1:
        return launch_continuous_chunk_static<1>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
    case 2:
        return launch_continuous_chunk_static<2>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
    case 3:
        return launch_continuous_chunk_static<3>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
    case 4:
        return launch_continuous_chunk_static<4>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
    case 5:
        return launch_continuous_chunk_static<5>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
    default:
        break;
    }
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

template <uint32_t Arity, uint32_t Bins>
cudaError_t launch_mutual_info_chunk_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values,
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kMiThreadsPerBlock);
    kernel::score_mutual_info_chunk_kernel_static<Arity, Bins><<<grid, block, 0, stream>>>(
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        descriptor_offset,
        combo_count,
        metric_count,
        metric_index,
        metric_values
    );
    return cudaGetLastError();
}

template <uint32_t Arity>
cudaError_t launch_mutual_info_chunk_for_bins(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t bins,
    float* metric_values,
    cudaStream_t stream
) {
    switch (bins) {
    case 2:
        return launch_mutual_info_chunk_static<Arity, 2>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 4:
        return launch_mutual_info_chunk_static<Arity, 4>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 8:
        return launch_mutual_info_chunk_static<Arity, 8>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 12:
        return launch_mutual_info_chunk_static<Arity, 12>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 16:
        return launch_mutual_info_chunk_static<Arity, 16>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 24:
        return launch_mutual_info_chunk_static<Arity, 24>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 32:
        return launch_mutual_info_chunk_static<Arity, 32>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 48:
        return launch_mutual_info_chunk_static<Arity, 48>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 64:
        return launch_mutual_info_chunk_static<Arity, 64>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 96:
        return launch_mutual_info_chunk_static<Arity, 96>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    default:
        dim3 grid(static_cast<unsigned int>(combo_count));
        dim3 block(kThreadsPerBlock);
        kernel::score_mutual_info_chunk_kernel<<<grid, block, 0, stream>>>(
            features,
            target,
            column_means,
            combo_indices,
            n_samples,
            0,
            Arity,
            descriptor_offset,
            combo_count,
            metric_count,
            metric_index,
            bins,
            metric_values
        );
        return cudaGetLastError();
    }
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
    switch (arity) {
    case 1:
        return launch_mutual_info_chunk_for_bins<1>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, bins, metric_values, stream);
    case 2:
        return launch_mutual_info_chunk_for_bins<2>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, bins, metric_values, stream);
    case 3:
        return launch_mutual_info_chunk_for_bins<3>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, bins, metric_values, stream);
    case 4:
        return launch_mutual_info_chunk_for_bins<4>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, bins, metric_values, stream);
    case 5:
        return launch_mutual_info_chunk_for_bins<5>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, bins, metric_values, stream);
    default:
        break;
    }
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

template <uint32_t Arity>
cudaError_t launch_spearman_chunk_static(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values,
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    kernel::score_spearman_chunk_kernel_static<Arity><<<grid, block, 0, stream>>>(
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        descriptor_offset,
        combo_count,
        metric_count,
        metric_index,
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
    switch (arity) {
    case 1:
        return launch_spearman_chunk_static<1>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 2:
        return launch_spearman_chunk_static<2>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 3:
        return launch_spearman_chunk_static<3>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 4:
        return launch_spearman_chunk_static<4>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    case 5:
        return launch_spearman_chunk_static<5>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_count, metric_index, metric_values, stream);
    default:
        break;
    }
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
    float* partial_scores,
    uint32_t* partial_indices,
    uint32_t partial_blocks,
    cudaStream_t stream
) {
    if (top_k == 0 || partial_blocks == 0) {
        return cudaSuccess;
    }
    if (descending != 0u) {
        kernel::select_topk_partials_kernel_static<true><<<partial_blocks, kTopKThreadsPerBlock, 0, stream>>>(
            metric_values,
            row_count,
            metric_count,
            primary_metric_index,
            top_k,
            partial_scores,
            partial_indices
        );
        const cudaError_t partial_status = cudaGetLastError();
        if (partial_status != cudaSuccess) {
            return partial_status;
        }
        kernel::merge_topk_partials_kernel_static<true><<<1, kTopKThreadsPerBlock, 0, stream>>>(
            partial_scores,
            partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k,
            top_k,
            selected_indices
        );
    } else {
        kernel::select_topk_partials_kernel_static<false><<<partial_blocks, kTopKThreadsPerBlock, 0, stream>>>(
            metric_values,
            row_count,
            metric_count,
            primary_metric_index,
            top_k,
            partial_scores,
            partial_indices
        );
        const cudaError_t partial_status = cudaGetLastError();
        if (partial_status != cudaSuccess) {
            return partial_status;
        }
        kernel::merge_topk_partials_kernel_static<false><<<1, kTopKThreadsPerBlock, 0, stream>>>(
            partial_scores,
            partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k,
            top_k,
            selected_indices
        );
    }
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
    uint64_t row_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_max,
    cudaStream_t stream
) {
    if (row_count == 0 || metric_count == 0) {
        return cudaSuccess;
    }
    dim3 grid(metric_count);
    dim3 block(kThreadsPerBlock);
    kernel::selected_metric_max_kernel<<<grid, block, 0, stream>>>(
        metric_values,
        row_count,
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

struct DescriptorBufferUpdateTransition {
    bool combo_reserved;
    bool metric_ids_reserved;
    bool combo_uploaded;
    bool metric_ids_uploaded;

    constexpr bool allocations_complete() const {
        return combo_reserved && metric_ids_reserved;
    }

    constexpr bool uploads_complete() const {
        return allocations_complete() && combo_uploaded && metric_ids_uploaded;
    }
};

static_assert(
    !DescriptorBufferUpdateTransition{true, false, false, false}.allocations_complete()
);
static_assert(
    !DescriptorBufferUpdateTransition{true, true, true, false}.uploads_complete()
);

struct CudaMatrix {
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t arch_class;
    uint64_t rows;
    uint32_t cols;
    bool content_valid;
    bool features_are_finite;
    bool target_is_finite;
    uint64_t feature_generation;
    uint64_t target_generation;
    float* features;
    float* target;
    float* column_means;
    gafime_cuda_v1::TargetStatsDevice* target_stats;
    gafime_cuda_v1::UnaryFeatureStatsDevice* feature_stats;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    uint64_t descriptor_generation;
    uint64_t descriptor_combo_len;
    uint64_t descriptor_metric_id_len;
    float* metric_values;
    uint64_t metric_value_capacity;
    uint32_t* selected_indices;
    uint64_t selected_index_capacity;
    float* selected_metric_values;
    uint64_t selected_metric_value_capacity;
    float* topk_partial_scores;
    uint64_t topk_partial_score_capacity;
    uint32_t* topk_partial_indices;
    uint64_t topk_partial_index_capacity;
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

void invalidate_protocol_descriptor_cache(CudaMatrix* matrix) {
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
}

std::atomic<uint64_t> g_cuda_matrix_content_generation{1};

uint64_t next_cuda_matrix_generation() {
    return g_cuda_matrix_content_generation.fetch_add(1, std::memory_order_relaxed);
}

int require_valid_matrix_content(const CudaMatrix* matrix) {
    return matrix != nullptr && matrix->content_valid
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
}

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) {
        return GAFIME_STATUS_OK;
    }
    if (status == cudaErrorMemoryAllocation) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_DEVICE_ERROR;
}

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

bool allocation_fits_size_t(uint64_t count, size_t element_size) {
    return count <= std::numeric_limits<size_t>::max() / element_size;
}

uint64_t cuda_arch_class(const cudaDeviceProp& props) {
    const uint32_t sm = static_cast<uint32_t>(props.major * 10 + props.minor);
    if (props.major >= 10) {
        return GAFIME_GPU_ARCH_NVIDIA_BLACKWELL;
    }
    if (props.major == 9) {
        return GAFIME_GPU_ARCH_NVIDIA_HOPPER;
    }
    if (props.major == 8 && props.minor >= 9) {
        return GAFIME_GPU_ARCH_NVIDIA_ADA;
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
    uint32_t flags = GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL |
        GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION;
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
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    flags |= GAFIME_GPU_DEVICE_FLAG_OPTIX_RT;
#endif
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
        gafime_cuda_v1::kernel::select_topk_partials_kernel_static<true>,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::select_topk_partials_kernel_static<false>,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::merge_topk_partials_kernel_static<true>,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::merge_topk_partials_kernel_static<false>,
        cudaFuncCachePreferShared
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::kernel::selected_metric_max_kernel,
        cudaFuncCachePreferShared
    ));
    gafime_cuda_v1::tune_rt_kernels_for_device(props);

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
    if (desc == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (desc->rows == 0 || desc->cols == 0 || desc->flags != 0 ||
        desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t feature_count = 0;
    uint64_t feature_bytes = 0;
    if (!checked_mul_u64(desc->rows, desc->cols, &feature_count) ||
        !checked_mul_u64(feature_count, sizeof(float), &feature_bytes) ||
        !allocation_fits_size_t(feature_count, sizeof(float)) ||
        !allocation_fits_size_t(desc->rows, sizeof(float)) ||
        !allocation_fits_size_t(desc->cols, sizeof(float)) ||
        !allocation_fits_size_t(desc->cols, sizeof(gafime_cuda_v1::UnaryFeatureStatsDevice))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (desc->bytes != feature_bytes) {
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
    if (protocol->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->backend_kind != GAFIME_BACKEND_CUDA) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->max_arity == 0 || protocol->max_arity > matrix->cols ||
        protocol->family_count != 1) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    constexpr uint32_t kKnownLaunchFlags =
        GAFIME_LAUNCH_FLAG_GRAPH | GAFIME_LAUNCH_FLAG_MI_APPROX |
        GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    if ((protocol->flags & ~kKnownLaunchFlags) != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.len > std::numeric_limits<uint32_t>::max()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        uint64_t expected_offsets = 0;
        if (!checked_mul_u64(
                protocol->permutations.permutation_count,
                matrix->rows,
                &expected_offsets)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
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
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr ||
        protocol->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->shape_hint_count != 0 && protocol->shape_hints == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t total_rows = 0;
    uint64_t expected_descriptor_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.family != GAFIME_FAMILY_CONTINUOUS) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (chunk.arity == 0 || chunk.arity > protocol->max_arity ||
            chunk.combo_count == 0 || chunk.descriptor_count != chunk.combo_count ||
            chunk.combo_row_offset != total_rows ||
            chunk.descriptor_offset != expected_descriptor_offset ||
            chunk.local_chunk_id != chunk_idx ||
            (protocol->shape_hint_count != 0 &&
                chunk.shape_hint_index >= protocol->shape_hint_count)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        if (chunk.combo_count > std::numeric_limits<uint32_t>::max() ||
            total_rows > std::numeric_limits<uint32_t>::max() - chunk.combo_count) {
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
            if (protocol->combo_indices.ptr[descriptor_idx] >= matrix->cols) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
        total_rows += chunk.combo_count;
        expected_descriptor_offset = descriptor_end;
    }
    if (expected_descriptor_offset != protocol->combo_indices.len ||
        !allocation_fits_size_t(protocol->combo_indices.len, sizeof(uint32_t)) ||
        !allocation_fits_size_t(protocol->metric_ids.len, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t metric_value_count = 0;
    if (!checked_mul_u64(total_rows, protocol->metric_ids.len, &metric_value_count) ||
        !allocation_fits_size_t(metric_value_count, sizeof(float))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
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
    if (result == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
    if (result->capacity < rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t combo_value_count = 0;
    uint64_t metric_value_count = 0;
    if (!checked_mul_u64(rows, result->max_arity, &combo_value_count) ||
        !checked_mul_u64(rows, result->metric_count, &metric_value_count) ||
        !allocation_fits_size_t(combo_value_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(metric_value_count, sizeof(float)) ||
        !allocation_fits_size_t(rows, sizeof(uint64_t))) {
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

uint32_t topk_partial_block_count(uint64_t row_count, uint64_t top_k) {
    if (row_count == 0 || top_k == 0) {
        return 0;
    }
    const uint64_t target_blocks =
        1 + (row_count - 1) / gafime_cuda_v1::kTopKThreadsPerBlock;
    const uint64_t storage_blocks = 1 + (row_count - 1) / top_k;
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::min(target_blocks, storage_blocks),
        gafime_cuda_v1::kTopKMaxPartialBlocks
    ));
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
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / sizeof(T);
    if (required > max_capacity) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t grown_capacity = *capacity > max_capacity / 2
        ? max_capacity
        : *capacity * 2;
    const uint64_t next_capacity = std::max(
        required,
        *capacity == 0 ? required : grown_capacity
    );
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

template <typename T>
class CudaDeviceBufferReservation {
public:
    CudaDeviceBufferReservation(T* live_ptr, uint64_t live_capacity)
        : ptr_(live_ptr), capacity_(live_capacity), owns_replacement_(false) {}

    ~CudaDeviceBufferReservation() {
        if (owns_replacement_) {
            static_cast<void>(cudaFree(ptr_));
        }
    }

    CudaDeviceBufferReservation(const CudaDeviceBufferReservation&) = delete;
    CudaDeviceBufferReservation& operator=(const CudaDeviceBufferReservation&) = delete;

    int reserve(uint64_t required) {
        if (required <= capacity_) {
            return GAFIME_STATUS_OK;
        }
        const uint64_t max_capacity = std::numeric_limits<size_t>::max() / sizeof(T);
        if (required > max_capacity) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        const uint64_t grown_capacity = capacity_ > max_capacity / 2
            ? max_capacity
            : capacity_ * 2;
        const uint64_t next_capacity = std::max(
            required,
            capacity_ == 0 ? required : grown_capacity
        );
        T* next = nullptr;
        const size_t bytes = static_cast<size_t>(next_capacity) * sizeof(T);
        const int status = cuda_status(cudaMalloc(&next, bytes));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        ptr_ = next;
        capacity_ = next_capacity;
        owns_replacement_ = true;
        return GAFIME_STATUS_OK;
    }

    T* get() const {
        return ptr_;
    }

    bool replaces_live_buffer() const {
        return owns_replacement_;
    }

    void commit(T** live_ptr, uint64_t* live_capacity) {
        if (!owns_replacement_) {
            return;
        }
        T* previous = *live_ptr;
        *live_ptr = ptr_;
        *live_capacity = capacity_;
        ptr_ = nullptr;
        owns_replacement_ = false;
        static_cast<void>(cudaFree(previous));
    }

private:
    T* ptr_;
    uint64_t capacity_;
    bool owns_replacement_;
};

int ensure_pinned_target_capacity(CudaMatrix* matrix, uint64_t required) {
    if (required <= matrix->permutation_target_capacity) {
        return GAFIME_STATUS_OK;
    }
    float* next = nullptr;
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / sizeof(float);
    if (required > max_capacity) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t grown_capacity = matrix->permutation_target_capacity > max_capacity / 2
        ? max_capacity
        : matrix->permutation_target_capacity * 2;
    const uint64_t next_capacity = std::max(
        required,
        matrix->permutation_target_capacity == 0 ? required : grown_capacity
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

void build_feature_major_host(
    const float* features_host,
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
            const float value = features_host[static_cast<size_t>(row) * cols + col];
            finite = finite && std::isfinite(value);
            resident_features[static_cast<size_t>(feature_base + row)] =
                value;
        }
    }
    if (features_are_finite != nullptr) {
        *features_are_finite = finite;
    }
}

bool all_finite_host(const float* values, uint64_t len) {
    bool finite = true;
    for (uint64_t idx = 0; idx < len; ++idx) {
        finite = finite && std::isfinite(values[idx]);
    }
    return finite;
}

int refresh_target_stats(CudaMatrix* matrix, cudaStream_t stream) {
    if (matrix == nullptr || matrix->target == nullptr || matrix->target_stats == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(gafime_cuda_v1::launch_target_stats(
        matrix->target,
        matrix->rows,
        matrix->target_stats,
        stream
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return cuda_status(cudaStreamSynchronize(stream));
}

int refresh_unary_feature_stats(CudaMatrix* matrix, cudaStream_t stream) {
    if (matrix == nullptr || matrix->features == nullptr || matrix->feature_stats == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(gafime_cuda_v1::launch_unary_feature_stats(
        matrix->features,
        matrix->rows,
        matrix->cols,
        matrix->feature_stats,
        stream
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return cuda_status(cudaStreamSynchronize(stream));
}

uint32_t mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = 96;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t hint = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (hint == 2 || hint == 4 || hint == 8 || hint == 12 || hint == 16 ||
            hint == 24 || hint == 32 || hint == 48 || hint == 64 || hint == 96) {
            bins = hint;
        }
    }
    return bins;
}

bool has_continuous_covariance_metric(const GafimeLaunchProtocol* protocol) {
    if (protocol == nullptr || protocol->metric_ids.ptr == nullptr) {
        return false;
    }
    for (uint64_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
        const uint32_t metric_id = protocol->metric_ids.ptr[metric_idx];
        if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2) {
            return true;
        }
    }
    return false;
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
        if (has_continuous_covariance_metric(protocol)) {
            const bool enable_cuda_unary_target_cache =
                protocol->permutations.permutation_count == 0;
            const int status = cuda_status(gafime_cuda_v1::launch_continuous_chunk(
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
                (enable_cuda_unary_target_cache && matrix->target_is_finite) ? 1u : 0u,
                matrix->metric_ids,
                static_cast<uint32_t>(protocol->metric_ids.len),
                matrix->metric_values + metric_row_offset * protocol->metric_ids.len,
                stream
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
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
        matrix->graph_chunk_shapes.size() != protocol->chunk_count ||
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

int store_graph_shape(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool target_copy_required
) try {
    std::vector<GraphChunkShape> next_shapes;
    next_shapes.reserve(protocol->chunk_count);
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        next_shapes.push_back(GraphChunkShape{
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
        });
    }
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
    matrix->graph_metric_signature = compute_metric_signature(protocol);
    matrix->graph_chunk_shapes.swap(next_shapes);
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
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
        status = store_graph_shape(matrix, protocol, metric_value_count, false);
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }

    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    } else {
        destroy_graph_cache(matrix);
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
) try {
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
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
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
        status = store_graph_shape(matrix, protocol, metric_value_count, true);
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }

    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    } else {
        destroy_graph_cache(matrix);
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
    if (selected_indices == nullptr && protocol->rank.top_k != 0) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
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
    int status = ensure_device_capacity(
        &matrix->metric_values,
        &matrix->metric_value_capacity,
        metric_value_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const bool immutable =
        (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0;
    const uint64_t descriptor_generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    const bool cacheable = immutable && descriptor_generation != 0;
    const bool descriptors_resident = cacheable &&
        matrix->descriptor_generation == descriptor_generation &&
        matrix->descriptor_combo_len == protocol->combo_indices.len &&
        matrix->descriptor_metric_id_len == protocol->metric_ids.len;
    if (descriptors_resident) {
        return GAFIME_STATUS_OK;
    }

    DescriptorBufferUpdateTransition transition{};
    CudaDeviceBufferReservation<uint32_t> combo_reservation(
        matrix->combo_indices,
        matrix->combo_capacity
    );
    status = combo_reservation.reserve(protocol->combo_indices.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    transition.combo_reserved = true;
    CudaDeviceBufferReservation<uint32_t> metric_id_reservation(
        matrix->metric_ids,
        matrix->metric_id_capacity
    );
    status = metric_id_reservation.reserve(protocol->metric_ids.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    transition.metric_ids_reserved = true;
    if (!transition.allocations_complete()) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    // Both growth allocations are now reserved. From this point onward a failed
    // upload leaves no published key, so partially written buffers cannot hit.
    invalidate_protocol_descriptor_cache(matrix);
    const size_t combo_bytes = static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t);
    status = cuda_status(cudaMemcpy(
        combo_reservation.get(),
        protocol->combo_indices.ptr,
        combo_bytes,
        cudaMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK) {
        transition.combo_uploaded = true;
        status = cuda_status(cudaMemcpy(
            metric_id_reservation.get(),
            protocol->metric_ids.ptr,
            metric_id_bytes,
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        transition.metric_ids_uploaded = true;
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (!transition.uploads_complete()) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    if (combo_reservation.replaces_live_buffer() ||
        metric_id_reservation.replaces_live_buffer()) {
        destroy_graph_cache(matrix);
    }
    combo_reservation.commit(&matrix->combo_indices, &matrix->combo_capacity);
    metric_id_reservation.commit(&matrix->metric_ids, &matrix->metric_id_capacity);
    if (cacheable) {
        matrix->descriptor_generation = descriptor_generation;
        matrix->descriptor_combo_len = protocol->combo_indices.len;
        matrix->descriptor_metric_id_len = protocol->metric_ids.len;
    }
    return GAFIME_STATUS_OK;
}

int validate_significance_table(
    const GafimeLaunchProtocol* protocol,
    const CudaMatrix* matrix,
    const GafimePermutationSignificanceTable* significance,
    uint64_t total_rows
) {
    if (significance == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->permutations.permutation_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->metric_count != protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->row_count > total_rows) {
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
    uint64_t selected_metric_count = 0;
    if (!checked_mul_u64(
            significance->row_count,
            significance->metric_count,
            &selected_metric_count) ||
        !allocation_fits_size_t(selected_metric_count, sizeof(float)) ||
        !allocation_fits_size_t(selected_metric_count, sizeof(uint32_t))) {
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
        matrix->significance_observed_values,
        significance->observed_metric_values,
        static_cast<size_t>(selected_count * metric_count) * sizeof(float),
        cudaMemcpyHostToDevice
    ));
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

    matrix->content_valid = false;
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
        // maxT is family-wide: reduce every permuted candidate while keeping
        // observed values and exceedance counters compact to surfaced rows.
        status = cuda_status(gafime_cuda_v1::launch_selected_metric_max(
            matrix->metric_values,
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
    if (restore_status == GAFIME_STATUS_OK) {
        matrix->content_valid = true;
    } else {
        destroy_graph_cache(matrix);
    }
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
) try {
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
    matrix->content_valid = false;
    matrix->features_are_finite = true;
    matrix->target_is_finite = true;
    matrix->feature_generation = 0;
    matrix->target_generation = 0;
    matrix->features = nullptr;
    matrix->target = nullptr;
    matrix->column_means = nullptr;
    matrix->target_stats = nullptr;
    matrix->feature_stats = nullptr;
    matrix->combo_indices = nullptr;
    matrix->combo_capacity = 0;
    matrix->metric_ids = nullptr;
    matrix->metric_id_capacity = 0;
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->metric_values = nullptr;
    matrix->metric_value_capacity = 0;
    matrix->selected_indices = nullptr;
    matrix->selected_index_capacity = 0;
    matrix->selected_metric_values = nullptr;
    matrix->selected_metric_value_capacity = 0;
    matrix->topk_partial_scores = nullptr;
    matrix->topk_partial_score_capacity = 0;
    matrix->topk_partial_indices = nullptr;
    matrix->topk_partial_index_capacity = 0;
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
    const size_t target_stats_bytes = sizeof(gafime_cuda_v1::TargetStatsDevice);
    const size_t feature_stats_bytes =
        static_cast<size_t>(matrix->cols) * sizeof(gafime_cuda_v1::UnaryFeatureStatsDevice);
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
    status = cuda_status(cudaMalloc(&matrix->target_stats, target_stats_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->column_means);
        cudaFree(matrix->target);
        cudaFree(matrix->features);
        delete matrix;
        return status;
    }
    status = cuda_status(cudaMalloc(&matrix->feature_stats, feature_stats_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->target_stats);
        cudaFree(matrix->column_means);
        cudaFree(matrix->target);
        cudaFree(matrix->features);
        delete matrix;
        return status;
    }

    *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
    return GAFIME_STATUS_OK;
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
    std::vector<float> resident_features;
    bool features_are_finite = true;
    build_feature_major_host(features_host, rows, cols, resident_features, &features_are_finite);
    const bool target_is_finite = all_finite_host(target_host, rows);
    std::vector<float> next_target_host(target_host, target_host + rows);

    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(float);
    matrix->content_valid = false;
    destroy_graph_cache(matrix);
    invalidate_protocol_descriptor_cache(matrix);
    status = cuda_status(cudaMemcpy(matrix->features, resident_features.data(), feature_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaMemcpy(matrix->column_means, column_means.data(), mean_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = refresh_unary_feature_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = refresh_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    matrix->feature_generation = next_cuda_matrix_generation();
    matrix->target_generation = next_cuda_matrix_generation();
    matrix->features_are_finite = features_are_finite;
    matrix->target_is_finite = target_is_finite;
    matrix->target_host.swap(next_target_host);
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
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
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (require_valid_matrix_content(matrix) != GAFIME_STATUS_OK) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const bool target_is_finite = all_finite_host(target_host, rows);
    std::vector<float> next_target_host(target_host, target_host + rows);
    matrix->content_valid = false;
    invalidate_protocol_descriptor_cache(matrix);
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        destroy_graph_cache(matrix);
        return status;
    }
    status = refresh_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        destroy_graph_cache(matrix);
        return status;
    }

    matrix->target_generation = next_cuda_matrix_generation();
    matrix->target_host.swap(next_target_host);
    if (matrix->target_is_finite != target_is_finite) {
        destroy_graph_cache(matrix);
    }
    matrix->target_is_finite = target_is_finite;
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr) {
        return;
    }

    int previous_device = 0;
    const bool restore_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    const cudaError_t select_status =
        matrix->device_id <= static_cast<uint32_t>(std::numeric_limits<int>::max())
            ? cudaSetDevice(static_cast<int>(matrix->device_id))
            : cudaErrorInvalidDevice;
    destroy_graph_cache(matrix);
    if (matrix->graph_stream != nullptr) {
        static_cast<void>(cudaStreamDestroy(matrix->graph_stream));
    }
    static_cast<void>(cudaFree(matrix->column_means));
    static_cast<void>(cudaFree(matrix->target_stats));
    static_cast<void>(cudaFree(matrix->feature_stats));
    static_cast<void>(cudaFree(matrix->target));
    static_cast<void>(cudaFree(matrix->features));
    static_cast<void>(cudaFree(matrix->metric_values));
    static_cast<void>(cudaFree(matrix->metric_ids));
    static_cast<void>(cudaFree(matrix->combo_indices));
    static_cast<void>(cudaFree(matrix->selected_metric_values));
    static_cast<void>(cudaFree(matrix->selected_indices));
    static_cast<void>(cudaFree(matrix->topk_partial_indices));
    static_cast<void>(cudaFree(matrix->topk_partial_scores));
    static_cast<void>(cudaFree(matrix->significance_exceedance_counts));
    static_cast<void>(cudaFree(matrix->significance_metric_max));
    static_cast<void>(cudaFree(matrix->significance_observed_values));
    static_cast<void>(cudaFreeHost(matrix->permutation_target_host));
    delete matrix;
    if (restore_previous && select_status == cudaSuccess) {
        static_cast<void>(cudaSetDevice(previous_device));
    }
}

GAFIME_GPU_API int gafime_gpu_decision_path_membership(
    GafimeGpuMatrix matrix_handle,
    const GafimeDecisionPathBatch* paths
) try {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || paths == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (paths->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    const int content_status = require_valid_matrix_content(matrix);
    if (content_status != GAFIME_STATUS_OK) {
        return content_status;
    }
    return gafime_cuda_v1::execute_decision_path_membership(
        matrix->features,
        matrix->rows,
        matrix->cols,
        matrix->device_id,
        matrix->arch_class,
        matrix->device_flags,
        matrix->features_are_finite,
        paths
    );
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_decision_path_score(
    GafimeGpuMatrix matrix_handle,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result_out
) try {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || paths == nullptr || result_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (paths->abi_version != GAFIME_ABI_VERSION ||
        result_out->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    const int content_status = require_valid_matrix_content(matrix);
    if (content_status != GAFIME_STATUS_OK) {
        return content_status;
    }
    return gafime_cuda_v1::execute_decision_path_score(
        matrix->features,
        matrix->target,
        matrix->rows,
        matrix->cols,
        matrix->device_id,
        matrix->arch_class,
        matrix->device_flags,
        matrix->features_are_finite,
        matrix->feature_generation,
        matrix->target_generation,
        paths,
        result_out
    );
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
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = require_valid_matrix_content(matrix);
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
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);
    status = prepare_protocol_device_buffers(matrix, protocol, metric_value_count);
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
    const uint32_t partial_blocks = topk_partial_block_count(total_rows, output_rows);
    const uint64_t partial_items = static_cast<uint64_t>(partial_blocks) * output_rows;
    status = ensure_device_capacity(&matrix->topk_partial_scores, &matrix->topk_partial_score_capacity, partial_items);
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&matrix->topk_partial_indices, &matrix->topk_partial_index_capacity, partial_items);
    }
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
        matrix->topk_partial_scores,
        matrix->topk_partial_indices,
        partial_blocks,
        0
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<uint32_t> selected_indices(static_cast<size_t>(output_rows), UINT32_MAX);
    status = cuda_status(cudaMemcpy(
        selected_indices.data(),
        matrix->selected_indices,
        static_cast<size_t>(output_rows) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    uint64_t selected_count = 0;
    while (selected_count < output_rows &&
        selected_indices[static_cast<size_t>(selected_count)] != UINT32_MAX) {
        ++selected_count;
    }
    selected_indices.resize(static_cast<size_t>(selected_count));
    if (selected_count == 0) {
        result_out->row_count = 0;
        update_graph_result_flag(result_out, graph_replayed);
        return GAFIME_STATUS_OK;
    }

    const uint64_t selected_metric_value_count = selected_count * protocol->metric_ids.len;
    status = ensure_device_capacity(
        &matrix->selected_metric_values,
        &matrix->selected_metric_value_capacity,
        selected_metric_value_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t copy_items = selected_metric_value_count;
    if (copy_items > 0) {
        status = cuda_status(gafime_cuda_v1::launch_copy_selected_metric_rows(
            matrix->metric_values,
            matrix->selected_indices,
            selected_count,
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

    std::vector<float> selected_metric_values(
        static_cast<size_t>(selected_metric_value_count),
        0.0f
    );
    status = cuda_status(cudaMemcpy(
        selected_metric_values.data(),
        matrix->selected_metric_values,
        static_cast<size_t>(selected_metric_value_count) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    update_graph_result_flag(result_out, graph_replayed);
    return write_result_rows_host(protocol, result_out, selected_metric_values, &selected_indices);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
) try {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t total_rows = planned_row_count(protocol);
    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = validate_significance_table(protocol, matrix, significance_out, total_rows);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = require_valid_matrix_content(matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
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
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

}  // extern "C"
