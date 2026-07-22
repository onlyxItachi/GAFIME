#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "rocm_api.hpp"
#include "kernels.hpp"
#include "../common/covariance_policy.hpp"
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

template <uint32_t Arity>
hipError_t launch_continuous_chunk_static(
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
    bool scaled_covariance,
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    if (scaled_covariance) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::score_continuous_scaled_chunk_kernel<Arity>),
            grid,
            block,
            0,
            stream,
            features,
            target,
            column_means,
            combo_indices,
            n_samples,
            Arity,
            descriptor_offset,
            combo_count,
            metric_ids,
            metric_count,
            metric_values
        );
        return hipGetLastError();
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::score_continuous_chunk_kernel_static<Arity>),
        grid,
        block,
        0,
        stream,
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
    uint32_t scaled_covariance,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values,
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    if (scaled_covariance == 0u && arity == 1 && features_are_finite != 0u &&
        target_is_finite != 0u &&
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
    switch (arity) {
    case 1:
        return launch_continuous_chunk_static<1>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values,
            scaled_covariance != 0u, stream);
    case 2:
        return launch_continuous_chunk_static<2>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values,
            scaled_covariance != 0u, stream);
    case 3:
        return launch_continuous_chunk_static<3>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values,
            scaled_covariance != 0u, stream);
    case 4:
        return launch_continuous_chunk_static<4>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values,
            scaled_covariance != 0u, stream);
    case 5:
        return launch_continuous_chunk_static<5>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values,
            scaled_covariance != 0u, stream);
    default:
        break;
    }
    if (scaled_covariance != 0u) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::score_continuous_scaled_chunk_kernel<0>),
            grid,
            block,
            0,
            stream,
            features,
            target,
            column_means,
            combo_indices,
            n_samples,
            arity,
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

template <uint32_t Arity, uint32_t Bins>
hipError_t launch_mutual_info_chunk_static(
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
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kMiThreadsPerBlock);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::score_mutual_info_chunk_kernel_static<Arity, Bins>),
        grid,
        block,
        0,
        stream,
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
    return hipGetLastError();
}

template <uint32_t Arity>
hipError_t launch_mutual_info_chunk_for_bins(
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
    hipStream_t stream
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
        break;
    }
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
        0,
        Arity,
        descriptor_offset,
        combo_count,
        metric_count,
        metric_index,
        bins,
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

hipError_t launch_spearman_target_ranks(
    const float* target,
    uint64_t n_samples,
    double* target_ranks,
    hipStream_t stream
) {
    if (n_samples == 0 || target_ranks == nullptr) {
        return hipSuccess;
    }
    const uint64_t block_count = 1 + (n_samples - 1) / kThreadsPerBlock;
    hipLaunchKernelGGL(
        kernel::build_spearman_target_ranks_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        target,
        n_samples,
        target_ranks
    );
    return hipGetLastError();
}

template <uint32_t Arity>
hipError_t launch_spearman_chunk_static(
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
    hipStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(combo_count));
    dim3 block(kThreadsPerBlock);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::score_spearman_chunk_kernel_static<Arity>),
        grid,
        block,
        0,
        stream,
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
    return hipGetLastError();
}

hipError_t launch_spearman_chunk(
    const float* features,
    const float* target,
    const float* column_means,
    const double* target_ranks,
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
    if (arity == 1 && target_ranks != nullptr) {
        hipLaunchKernelGGL(
            kernel::score_spearman_unary_cached_target_ranks_kernel,
            grid,
            block,
            0,
            stream,
            features,
            target,
            target_ranks,
            combo_indices,
            n_samples,
            descriptor_offset,
            combo_count,
            metric_count,
            metric_index,
            metric_values
        );
        return hipGetLastError();
    }
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

hipError_t launch_select_topk(
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
    hipStream_t stream
) {
    if (top_k == 0 || partial_blocks == 0) {
        return hipSuccess;
    }
    dim3 grid(partial_blocks);
    dim3 block(kTopKThreadsPerBlock);
    if (descending != 0u) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::select_topk_partials_kernel_static<true>),
            grid,
            block,
            0,
            stream,
            metric_values,
            row_count,
            metric_count,
            primary_metric_index,
            top_k,
            partial_scores,
            partial_indices
        );
        hipError_t partial_status = hipGetLastError();
        if (partial_status != hipSuccess) {
            return partial_status;
        }
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::merge_topk_partials_kernel_static<true>),
            dim3(1),
            block,
            0,
            stream,
            partial_scores,
            partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k,
            top_k,
            selected_indices
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::select_topk_partials_kernel_static<false>),
            grid,
            block,
            0,
            stream,
            metric_values,
            row_count,
            metric_count,
            primary_metric_index,
            top_k,
            partial_scores,
            partial_indices
        );
        hipError_t partial_status = hipGetLastError();
        if (partial_status != hipSuccess) {
            return partial_status;
        }
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::merge_topk_partials_kernel_static<false>),
            dim3(1),
            block,
            0,
            stream,
            partial_scores,
            partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k,
            top_k,
            selected_indices
        );
    }
    return hipGetLastError();
}

hipError_t launch_copy_selected_metric_rows(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values,
    hipStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) {
        return hipSuccess;
    }
    constexpr uint32_t kCopyThreads = 256;
    const uint32_t blocks = static_cast<uint32_t>((total + kCopyThreads - 1) / kCopyThreads);
    hipLaunchKernelGGL(
        kernel::copy_selected_metric_rows_kernel,
        dim3(blocks),
        dim3(kCopyThreads),
        0,
        stream,
        metric_values,
        selected_indices,
        selected_count,
        metric_count,
        selected_metric_values
    );
    return hipGetLastError();
}

}  // namespace gafime_rocm_v1

namespace {


struct GraphChunkShape {
    uint32_t arity;
    uint32_t scaled_covariance;
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

struct HipMatrix {
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t arch_class;
    bool unified_memory;
    uint64_t rows;
    uint32_t cols;
    bool content_valid;
    bool features_are_finite;
    bool target_is_finite;
    float* features;
    float* target;
    float* column_means;
    gafime_rocm_v1::TargetStatsDevice* target_stats;
    gafime_rocm_v1::UnaryFeatureStatsDevice* feature_stats;
    double* spearman_target_ranks;
    bool spearman_target_ranks_ready;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    uint64_t descriptor_generation;
    uint64_t descriptor_combo_len;
    uint64_t descriptor_metric_id_len;
    std::vector<int> feature_abs_exponents;
    int target_abs_exponent;
    std::vector<uint8_t> covariance_modes;
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

void invalidate_protocol_descriptor_cache(HipMatrix* matrix) {
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->covariance_modes.clear();
}

int hip_status(hipError_t status) {
    if (status == hipSuccess) {
        return GAFIME_STATUS_OK;
    }
    if (status == hipErrorOutOfMemory) {
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

bool checked_allocation_bytes(uint64_t count, size_t element_size, size_t* bytes_out) {
    if (!allocation_fits_size_t(count, element_size)) {
        return false;
    }
    *bytes_out = static_cast<size_t>(count) * element_size;
    return true;
}

int require_valid_matrix_content(const HipMatrix* matrix) {
    return matrix != nullptr && matrix->content_valid
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
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
    return props.integrated != 0;
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
    uint32_t flags = GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL |
        GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION;
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
    return rocm_device_is_integrated(props) &&
        (props.managedMemory != 0 || props.concurrentManagedAccess != 0);
}

template <typename T>
int allocate_hip_storage(T** ptr, size_t bytes, bool managed) {
    if (bytes == 0) {
        *ptr = nullptr;
        return GAFIME_STATUS_OK;
    }
    if (managed) {
        return hip_status(hipMallocManaged(reinterpret_cast<void**>(ptr), bytes));
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
    if (desc == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (desc->flags != 0 || desc->rows == 0 || desc->cols == 0 ||
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
        !allocation_fits_size_t(
            desc->cols,
            sizeof(gafime_rocm_v1::UnaryFeatureStatsDevice))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (desc->bytes != feature_bytes) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, const HipMatrix* matrix) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->backend_kind != GAFIME_BACKEND_ROCM) {
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
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    if (protocol->rank.top_k != 0 && protocol->rank.include_ties != 0) {
        // Top-k is supported; tie-inclusive ranking is not implemented.
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
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
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr ||
        protocol->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->shape_hint_count != 0 && protocol->shape_hints == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!allocation_fits_size_t(
            protocol->chunk_count,
            sizeof(GafimeArityChunk)) ||
        !allocation_fits_size_t(
            protocol->shape_hint_count,
            sizeof(GafimeShapeHint))) {
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
    if (protocol->rank.top_k != 0) {
        constexpr uint64_t kCopyThreads = 256;
        constexpr uint64_t kMaxCopyItems =
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) *
            kCopyThreads;
        const uint64_t selected_rows =
            std::min<uint64_t>(total_rows, protocol->rank.top_k);
        uint64_t selected_metric_value_count = 0;
        if (!checked_mul_u64(
                selected_rows,
                protocol->metric_ids.len,
                &selected_metric_value_count) ||
            selected_metric_value_count > kMaxCopyItems) {
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

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t planned) {
    if (protocol->rank.top_k == 0) {
        return planned;
    }
    return std::min<uint64_t>(planned, protocol->rank.top_k);
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

uint32_t topk_partial_block_count(uint64_t row_count, uint64_t top_k) {
    if (row_count == 0 || top_k == 0) {
        return 0;
    }
    const uint64_t target_blocks =
        1 + (row_count - 1) / gafime_rocm_v1::kTopKThreadsPerBlock;
    const uint64_t storage_blocks = 1 + (row_count - 1) / top_k;
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::min(target_blocks, storage_blocks),
        gafime_rocm_v1::kTopKMaxPartialBlocks
    ));
}

bool hip_protocol_descriptors_resident(
    const HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    const bool immutable =
        (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0;
    const uint64_t descriptor_generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    return immutable && descriptor_generation != 0 &&
        matrix->descriptor_generation == descriptor_generation &&
        matrix->descriptor_combo_len == protocol->combo_indices.len &&
        matrix->descriptor_metric_id_len == protocol->metric_ids.len;
}

uint64_t hip_matrix_resident_device_bytes(const HipMatrix* matrix) {
    using gafime_gpu_abi::allocation_bytes;
    using gafime_gpu_abi::saturating_add_u64;
    using gafime_gpu_abi::saturating_mul_u64;

    uint64_t bytes = 0;
    const auto add = [&bytes](uint64_t count, size_t element_size) {
        bytes = saturating_add_u64(bytes, allocation_bytes(count, element_size));
    };
    add(saturating_mul_u64(matrix->rows, matrix->cols), sizeof(float));
    add(matrix->rows, sizeof(float));
    add(matrix->cols, sizeof(float));
    add(1, sizeof(gafime_rocm_v1::TargetStatsDevice));
    add(matrix->cols, sizeof(gafime_rocm_v1::UnaryFeatureStatsDevice));
    add(matrix->spearman_target_ranks != nullptr ? matrix->rows : 0, sizeof(double));
    add(matrix->combo_capacity, sizeof(uint32_t));
    add(matrix->metric_id_capacity, sizeof(uint32_t));
    add(matrix->metric_value_capacity, sizeof(float));
    add(matrix->selected_index_capacity, sizeof(uint32_t));
    add(matrix->selected_metric_value_capacity, sizeof(float));
    add(matrix->topk_partial_score_capacity, sizeof(float));
    add(matrix->topk_partial_index_capacity, sizeof(uint32_t));
    return bytes;
}

uint64_t hip_execution_peak_device_bytes(
    const HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    using gafime_gpu_abi::DeviceMemoryPeakTracker;
    using gafime_gpu_abi::saturating_mul_u64;

    DeviceMemoryPeakTracker tracker(hip_matrix_resident_device_bytes(matrix));
    const uint64_t total_rows = planned_row_count(protocol);
    if (total_rows == 0) {
        return tracker.peak_bytes();
    }
    const uint64_t metric_value_count =
        saturating_mul_u64(total_rows, protocol->metric_ids.len);
    tracker.grow(
        matrix->metric_value_capacity,
        metric_value_count,
        sizeof(float)
    );
    if (!hip_protocol_descriptors_resident(matrix, protocol)) {
        tracker.reserve_pair(
            matrix->combo_capacity,
            protocol->combo_indices.len,
            sizeof(uint32_t),
            matrix->metric_id_capacity,
            protocol->metric_ids.len,
            sizeof(uint32_t)
        );
    }
    if (protocol->rank.top_k == 0) {
        return tracker.peak_bytes();
    }

    const uint64_t output_rows = output_row_count(protocol, total_rows);
    tracker.grow(
        matrix->selected_index_capacity,
        output_rows,
        sizeof(uint32_t)
    );
    const uint64_t partial_items = saturating_mul_u64(
        topk_partial_block_count(total_rows, output_rows),
        output_rows
    );
    tracker.grow(
        matrix->topk_partial_score_capacity,
        partial_items,
        sizeof(float)
    );
    tracker.grow(
        matrix->topk_partial_index_capacity,
        partial_items,
        sizeof(uint32_t)
    );
    // The selected count is data-dependent, so admission uses its output_rows ceiling.
    tracker.grow(
        matrix->selected_metric_value_capacity,
        saturating_mul_u64(output_rows, protocol->metric_ids.len),
        sizeof(float)
    );
    return tracker.peak_bytes();
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
    const uint64_t planned = planned_row_count(protocol);
    // With top-k ranking only the selected rows are written, so the result table
    // need only hold min(planned, top_k).
    const uint64_t rows = output_row_count(protocol, planned);
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

template <typename T>
int ensure_device_capacity(T** ptr, uint64_t* capacity, uint64_t required, bool managed) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / sizeof(T);
    if (required > max_capacity) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    T* next = nullptr;
    const uint64_t grown_capacity = *capacity > max_capacity / 2
        ? max_capacity
        : *capacity * 2;
    const uint64_t next_capacity = std::max(required, *capacity == 0 ? required : grown_capacity);
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

template <typename T>
class HipDeviceBufferReservation {
public:
    HipDeviceBufferReservation(T* live_ptr, uint64_t live_capacity, bool managed)
        : ptr_(live_ptr),
          capacity_(live_capacity),
          managed_(managed),
          owns_replacement_(false) {}

    ~HipDeviceBufferReservation() {
        if (owns_replacement_) {
            static_cast<void>(hipFree(ptr_));
        }
    }

    HipDeviceBufferReservation(const HipDeviceBufferReservation&) = delete;
    HipDeviceBufferReservation& operator=(const HipDeviceBufferReservation&) = delete;

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
        const int status = allocate_hip_storage(&next, bytes, managed_);
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
        static_cast<void>(hipFree(previous));
    }

private:
    T* ptr_;
    uint64_t capacity_;
    bool managed_;
    bool owns_replacement_;
};

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
    std::vector<int>& feature_abs_exponents,
    bool* features_are_finite
) {
    bool finite = true;
    resident_features.assign(static_cast<size_t>(rows) * cols, 0.0f);
    feature_abs_exponents.assign(cols, gafime_gpu_abi::kZeroMagnitudeExponent);
    for (uint32_t col = 0; col < cols; ++col) {
        const uint64_t feature_base = static_cast<uint64_t>(col) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const float value = features[static_cast<size_t>(row) * cols + col];
            feature_abs_exponents[col] = gafime_gpu_abi::update_finite_abs_exponent(
                feature_abs_exponents[col],
                value
            );
            finite = finite && std::isfinite(value);
            resident_features[static_cast<size_t>(feature_base + row)] =
                value;
        }
    }
    if (features_are_finite != nullptr) {
        *features_are_finite = finite;
    }
}

bool has_continuous_covariance_metric(const GafimeLaunchProtocol* protocol);

std::vector<uint8_t> covariance_modes_for_protocol(
    const HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    std::vector<uint8_t> modes(protocol->chunk_count, 0);
    if (!has_continuous_covariance_metric(protocol)) {
        return modes;
    }
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const uint64_t descriptor_base =
                chunk.descriptor_offset + row * static_cast<uint64_t>(chunk.arity);
            const int feature_exponent = gafime_gpu_abi::interaction_abs_exponent(
                matrix->feature_abs_exponents.data(),
                protocol->combo_indices.ptr + descriptor_base,
                chunk.arity
            );
            if (gafime_gpu_abi::covariance_requires_scaled_path(
                    matrix->rows,
                    feature_exponent,
                    matrix->target_abs_exponent
                )) {
                modes[chunk_idx] = 1;
                break;
            }
        }
    }
    return modes;
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

bool protocol_has_spearman_metric(const GafimeLaunchProtocol* protocol) {
    if (protocol == nullptr || protocol->metric_ids.ptr == nullptr) {
        return false;
    }
    for (uint64_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
        if (protocol->metric_ids.ptr[metric_idx] == GAFIME_METRIC_SPEARMAN) {
            return true;
        }
    }
    return false;
}

bool spearman_target_rank_cache_eligible(
    const HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (matrix == nullptr || protocol == nullptr || matrix->spearman_target_ranks == nullptr ||
        !matrix->features_are_finite || !matrix->target_is_finite ||
        matrix->rows < gafime_rocm_v1::kSpearmanTargetRankCacheMinSamples ||
        matrix->rows > gafime_rocm_v1::kSpearmanTargetRankCacheMaxSamples ||
        protocol->permutations.permutation_count != 0 || !protocol_has_spearman_metric(protocol)) {
        return false;
    }
    uint64_t unary_candidate_count = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.arity != 1) {
            continue;
        }
        if (unary_candidate_count >=
            gafime_rocm_v1::kSpearmanTargetRankCacheMinUnaryCandidates) {
            return true;
        }
        const uint64_t candidates_remaining =
            gafime_rocm_v1::kSpearmanTargetRankCacheMinUnaryCandidates - unary_candidate_count;
        if (chunk.combo_count >= candidates_remaining) {
            return true;
        }
        unary_candidate_count += chunk.combo_count;
    }
    return false;
}

int prepare_spearman_target_rank_cache(
    HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (!spearman_target_rank_cache_eligible(matrix, protocol) ||
        matrix->spearman_target_ranks_ready) {
        return GAFIME_STATUS_OK;
    }
    int status = hip_status(gafime_rocm_v1::launch_spearman_target_ranks(
        matrix->target,
        matrix->rows,
        matrix->spearman_target_ranks,
        nullptr
    ));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        matrix->spearman_target_ranks_ready = true;
    }
    return status;
}

void write_result_row_at(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    uint64_t dst_row,
    uint64_t src_metric_row,
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
            ? metric_values[src_metric_row * protocol->metric_ids.len + metric_idx]
            : 0.0f;
        result->metric_values[dst_row * metric_count + metric_idx] = value;
    }
    result->ranks[dst_row] = rank;
    result->families[dst_row] = GAFIME_FAMILY_CONTINUOUS;
    result->candidate_ids[dst_row] = candidate_id;
    result->row_flags[dst_row] = 0;
}

int write_result_rows_host(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    const std::vector<uint32_t>* selected_indices
) {
    if (selected_indices != nullptr) {
        for (uint64_t r = 0; r < selected_indices->size(); ++r) {
            const uint64_t candidate_id = static_cast<uint64_t>((*selected_indices)[r]);
            uint64_t row_offset = 0;
            const GafimeArityChunk* selected_chunk = nullptr;
            uint64_t local_row = 0;
            for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
                const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
                if (candidate_id < row_offset + chunk.combo_count) {
                    selected_chunk = &chunk;
                    local_row = candidate_id - row_offset;
                    break;
                }
                row_offset += chunk.combo_count;
            }
            if (selected_chunk == nullptr) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            const uint64_t combo_base =
                selected_chunk->descriptor_offset + local_row * selected_chunk->arity;
            write_result_row_at(protocol, result, metric_values, r, r, combo_base,
                selected_chunk->arity, static_cast<uint32_t>(r), candidate_id);
        }
        result->row_count = static_cast<uint64_t>(selected_indices->size());
        return GAFIME_STATUS_OK;
    }

    if (protocol->rank.top_k != 0) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

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
    const double* spearman_target_ranks =
        spearman_target_rank_cache_eligible(matrix, protocol) &&
            matrix->spearman_target_ranks_ready
        ? matrix->spearman_target_ranks
        : nullptr;
    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.combo_count == 0) {
            continue;
        }
        int status = GAFIME_STATUS_OK;
        if (has_continuous_covariance_metric(protocol)) {
            if (matrix->covariance_modes.size() != protocol->chunk_count) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            status = hip_status(gafime_rocm_v1::launch_continuous_chunk(
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
                matrix->covariance_modes[chunk_idx],
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
                spearman_target_ranks,
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
            shape.scaled_covariance != matrix->covariance_modes[idx] ||
            shape.descriptor_offset != chunk.descriptor_offset ||
            shape.combo_count != chunk.combo_count) {
            return false;
        }
    }
    return true;
}

int store_graph_shape(
    HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) try {
    std::vector<GraphChunkShape> next_shapes;
    next_shapes.reserve(protocol->chunk_count);
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        next_shapes.push_back(GraphChunkShape{
            chunk.arity,
            matrix->covariance_modes[idx],
            chunk.descriptor_offset,
            chunk.combo_count,
        });
    }
    matrix->graph_chunk_count = protocol->chunk_count;
    matrix->graph_combo_len = protocol->combo_indices.len;
    matrix->graph_metric_id_len = protocol->metric_ids.len;
    matrix->graph_metric_value_count = metric_value_count;
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
    // Keep the target-rank build outside stream capture so graph replays only
    // score kernels and a target update can refresh the same resident buffer.
    int status = prepare_spearman_target_rank_cache(matrix, protocol);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (!graph_requested(protocol)) {
        status = launch_score_kernels(matrix, protocol, nullptr);
        if (status == GAFIME_STATUS_OK) {
            status = hip_status(hipDeviceSynchronize());
        }
        return status;
    }

    if (matrix->graph_stream == nullptr) {
        status = hip_status(hipStreamCreateWithFlags(&matrix->graph_stream, hipStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    if (!graph_shape_matches(matrix, protocol, metric_value_count)) {
        destroy_graph_cache(matrix);
        status = hip_status(hipStreamBeginCapture(matrix->graph_stream, hipStreamCaptureModeThreadLocal));
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
        status = store_graph_shape(matrix, protocol, metric_value_count);
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }

    status = hip_status(hipGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipStreamSynchronize(matrix->graph_stream));
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

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(uint32_t device_id, GafimeGpuDeviceInfo* info_out) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (device_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    hipDeviceProp_t props{};
    status = hip_status(hipGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
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
    const int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_ROCM,
        GAFIME_GRAPH_STREAM_CAPTURE,
        capability_out
    );
    if (status == GAFIME_STATUS_OK) {
        capability_out->supports_device_ranking = 1;
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(uint32_t device_id, const GafimeMatrixDesc* matrix_desc, GafimeGpuMatrix* matrix_out) try {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    if (device_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
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
    matrix->content_valid = false;
    matrix->features_are_finite = true;
    matrix->target_is_finite = true;
    matrix->features = nullptr;
    matrix->target = nullptr;
    matrix->column_means = nullptr;
    matrix->target_stats = nullptr;
    matrix->feature_stats = nullptr;
    matrix->spearman_target_ranks = nullptr;
    matrix->spearman_target_ranks_ready = false;
    matrix->combo_indices = nullptr;
    matrix->combo_capacity = 0;
    matrix->metric_ids = nullptr;
    matrix->metric_id_capacity = 0;
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->target_abs_exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
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

    uint64_t feature_count = 0;
    size_t feature_bytes = 0;
    size_t target_bytes = 0;
    size_t mean_bytes = 0;
    size_t feature_stats_bytes = 0;
    if (!checked_mul_u64(matrix->rows, matrix->cols, &feature_count) ||
        !checked_allocation_bytes(feature_count, sizeof(float), &feature_bytes) ||
        !checked_allocation_bytes(matrix->rows, sizeof(float), &target_bytes) ||
        !checked_allocation_bytes(matrix->cols, sizeof(float), &mean_bytes) ||
        !checked_allocation_bytes(
            matrix->cols,
            sizeof(gafime_rocm_v1::UnaryFeatureStatsDevice),
            &feature_stats_bytes)) {
        delete matrix;
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const size_t target_stats_bytes = sizeof(gafime_rocm_v1::TargetStatsDevice);
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
    if (matrix->rows <= gafime_rocm_v1::kSpearmanTargetRankCacheMaxSamples) {
        const int rank_cache_status = allocate_hip_storage(
            &matrix->spearman_target_ranks,
            static_cast<size_t>(matrix->rows) * sizeof(double),
            matrix->unified_memory
        );
        if (rank_cache_status != GAFIME_STATUS_OK) {
            matrix->spearman_target_ranks = nullptr;
        }
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
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t feature_count = 0;
    size_t feature_bytes = 0;
    size_t target_bytes = 0;
    size_t mean_bytes = 0;
    if (!checked_mul_u64(rows, cols, &feature_count) ||
        !checked_allocation_bytes(feature_count, sizeof(float), &feature_bytes) ||
        !checked_allocation_bytes(rows, sizeof(float), &target_bytes) ||
        !checked_allocation_bytes(cols, sizeof(float), &mean_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<float> column_means;
    compute_column_means_host(features_host, rows, cols, column_means);
    std::vector<float> resident_features;
    std::vector<int> feature_abs_exponents;
    bool features_are_finite = true;
    build_feature_major_host(
        features_host,
        rows,
        cols,
        resident_features,
        feature_abs_exponents,
        &features_are_finite
    );
    const bool target_is_finite = all_finite_host(target_host, rows);
    const int target_abs_exponent = gafime_gpu_abi::finite_abs_exponent(target_host, rows);
    matrix->content_valid = false;
    destroy_graph_cache(matrix);
    invalidate_protocol_descriptor_cache(matrix);
    matrix->spearman_target_ranks_ready = false;
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
    status = refresh_unary_feature_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = refresh_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    matrix->features_are_finite = features_are_finite;
    matrix->feature_abs_exponents.swap(feature_abs_exponents);
    matrix->target_abs_exponent = target_abs_exponent;
    matrix->target_is_finite = target_is_finite;
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
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (require_valid_matrix_content(matrix) != GAFIME_STATUS_OK) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    size_t target_bytes = 0;
    if (!checked_allocation_bytes(rows, sizeof(float), &target_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const bool target_is_finite = all_finite_host(target_host, rows);
    const int target_abs_exponent = gafime_gpu_abi::finite_abs_exponent(target_host, rows);
    matrix->content_valid = false;
    invalidate_protocol_descriptor_cache(matrix);
    matrix->spearman_target_ranks_ready = false;
    status = copy_to_hip_storage(matrix->target, target_host, target_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    status = refresh_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (matrix->target_is_finite != target_is_finite) {
        destroy_graph_cache(matrix);
    }
    matrix->target_is_finite = target_is_finite;
    matrix->target_abs_exponent = target_abs_exponent;
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
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
    static_cast<void>(hipFree(matrix->spearman_target_ranks));
    static_cast<void>(hipFree(matrix->target));
    static_cast<void>(hipFree(matrix->features));
    static_cast<void>(hipFree(matrix->metric_values));
    static_cast<void>(hipFree(matrix->selected_metric_values));
    static_cast<void>(hipFree(matrix->selected_indices));
    static_cast<void>(hipFree(matrix->topk_partial_indices));
    static_cast<void>(hipFree(matrix->topk_partial_scores));
    static_cast<void>(hipFree(matrix->metric_ids));
    static_cast<void>(hipFree(matrix->combo_indices));
    delete matrix;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *peak_bytes_out = 0;
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = require_valid_matrix_content(matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    // HIP graph/runtime bookkeeping is opaque to the allocator API; this
    // reports matrix-owned global allocations and their exact growth order.
    *peak_bytes_out = hip_execution_peak_device_bytes(matrix, protocol);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) try {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
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
    status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    const uint64_t output_rows = output_row_count(protocol, total_rows);
    if (total_rows == 0) {
        result_out->row_count = 0;
        return GAFIME_STATUS_OK;
    }
    uint64_t metric_value_count = 0;
    size_t combo_index_bytes = 0;
    size_t metric_id_bytes = 0;
    size_t metric_value_bytes = 0;
    if (!checked_mul_u64(total_rows, protocol->metric_ids.len, &metric_value_count) ||
        !checked_allocation_bytes(
            protocol->combo_indices.len,
            sizeof(uint32_t),
            &combo_index_bytes) ||
        !checked_allocation_bytes(
            protocol->metric_ids.len,
            sizeof(uint32_t),
            &metric_id_bytes) ||
        !checked_allocation_bytes(
            metric_value_count,
            sizeof(float),
            &metric_value_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = ensure_device_capacity(
        &matrix->metric_values,
        &matrix->metric_value_capacity,
        metric_value_count,
        matrix->unified_memory
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const bool immutable =
        (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0;
    const uint64_t descriptor_generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    const bool cacheable = immutable && descriptor_generation != 0;
    const bool descriptors_resident =
        hip_protocol_descriptors_resident(matrix, protocol);
    if (!descriptors_resident) {
        std::vector<uint8_t> next_covariance_modes =
            covariance_modes_for_protocol(matrix, protocol);
        DescriptorBufferUpdateTransition transition{};
        HipDeviceBufferReservation<uint32_t> combo_reservation(
            matrix->combo_indices,
            matrix->combo_capacity,
            matrix->unified_memory
        );
        status = combo_reservation.reserve(protocol->combo_indices.len);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        transition.combo_reserved = true;
        HipDeviceBufferReservation<uint32_t> metric_id_reservation(
            matrix->metric_ids,
            matrix->metric_id_capacity,
            matrix->unified_memory
        );
        status = metric_id_reservation.reserve(protocol->metric_ids.len);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        transition.metric_ids_reserved = true;
        if (!transition.allocations_complete()) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }

        // Reserve every growth allocation before mutating live storage. A failed
        // upload clears the key before any in-place write can become observable.
        invalidate_protocol_descriptor_cache(matrix);
        status = copy_to_hip_storage(
            combo_reservation.get(),
            protocol->combo_indices.ptr,
            combo_index_bytes,
            matrix->unified_memory
        );
        if (status == GAFIME_STATUS_OK) {
            transition.combo_uploaded = true;
            status = copy_to_hip_storage(
                metric_id_reservation.get(),
                protocol->metric_ids.ptr,
                metric_id_bytes,
                matrix->unified_memory
            );
        }
        if (status == GAFIME_STATUS_OK) {
            transition.metric_ids_uploaded = true;
        }
        if (status == GAFIME_STATUS_OK && matrix->unified_memory) {
            status = hip_status(hipDeviceSynchronize());
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
        matrix->covariance_modes.swap(next_covariance_modes);
        if (cacheable) {
            matrix->descriptor_generation = descriptor_generation;
            matrix->descriptor_combo_len = protocol->combo_indices.len;
            matrix->descriptor_metric_id_len = protocol->metric_ids.len;
        }
    }

    bool graph_replayed = false;
    status = execute_score_kernels(matrix, protocol, metric_value_count, &graph_replayed);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    if (protocol->rank.top_k == 0) {
        std::vector<float> metric_values(static_cast<size_t>(metric_value_count), 0.0f);
        status = copy_from_hip_storage(
            metric_values.data(),
            matrix->metric_values,
            metric_value_bytes,
            matrix->unified_memory
        );
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = write_result_rows_host(protocol, result_out, metric_values, nullptr);
        if (status == GAFIME_STATUS_OK) {
            update_graph_result_flag(result_out, graph_replayed);
        }
        return status;
    }

    status = ensure_device_capacity(
        &matrix->selected_indices,
        &matrix->selected_index_capacity,
        output_rows,
        matrix->unified_memory
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint32_t partial_blocks = topk_partial_block_count(total_rows, output_rows);
    uint64_t partial_items = 0;
    if (!checked_mul_u64(partial_blocks, output_rows, &partial_items) ||
        !allocation_fits_size_t(partial_items, sizeof(float)) ||
        !allocation_fits_size_t(partial_items, sizeof(uint32_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_device_capacity(
        &matrix->topk_partial_scores,
        &matrix->topk_partial_score_capacity,
        partial_items,
        matrix->unified_memory
    );
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &matrix->topk_partial_indices,
            &matrix->topk_partial_index_capacity,
            partial_items,
            matrix->unified_memory
        );
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(gafime_rocm_v1::launch_select_topk(
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
        nullptr
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    size_t selected_index_bytes = 0;
    if (!checked_allocation_bytes(
            output_rows,
            sizeof(uint32_t),
            &selected_index_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    std::vector<uint32_t> selected_indices(static_cast<size_t>(output_rows), UINT32_MAX);
    status = copy_from_hip_storage(
        selected_indices.data(),
        matrix->selected_indices,
        selected_index_bytes,
        matrix->unified_memory
    );
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

    uint64_t selected_metric_value_count = 0;
    size_t selected_metric_value_bytes = 0;
    if (!checked_mul_u64(
            selected_count,
            protocol->metric_ids.len,
            &selected_metric_value_count) ||
        !checked_allocation_bytes(
            selected_metric_value_count,
            sizeof(float),
            &selected_metric_value_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_device_capacity(
        &matrix->selected_metric_values,
        &matrix->selected_metric_value_capacity,
        selected_metric_value_count,
        matrix->unified_memory
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(gafime_rocm_v1::launch_copy_selected_metric_rows(
        matrix->metric_values,
        matrix->selected_indices,
        selected_count,
        static_cast<uint32_t>(protocol->metric_ids.len),
        matrix->selected_metric_values,
        nullptr
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<float> selected_metric_values(
        static_cast<size_t>(selected_metric_value_count),
        0.0f
    );
    status = copy_from_hip_storage(
        selected_metric_values.data(),
        matrix->selected_metric_values,
        selected_metric_value_bytes,
        matrix->unified_memory
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = write_result_rows_host(protocol, result_out, selected_metric_values, &selected_indices);
    if (status == GAFIME_STATUS_OK) {
        update_graph_result_flag(result_out, graph_replayed);
    }
    return status;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

}  // extern "C"
