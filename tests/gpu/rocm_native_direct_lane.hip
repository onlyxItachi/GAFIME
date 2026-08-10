#include "rocm_native_direct_lane.hpp"

#include "kernels.hpp"

#include "gafime_gpu_abi.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#ifndef GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API
#define GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API 2
#endif

namespace {

using namespace gafime_rocm_v1;

template <typename StorageT>
__global__ void materialize_candidates_kernel(
    const StorageT* features,
    const uint32_t* combos,
    StorageT* materialized,
    uint64_t rows,
    uint32_t candidates,
    uint32_t arity
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = rows * static_cast<uint64_t>(candidates);
    if (index >= total) return;
    const uint32_t candidate = static_cast<uint32_t>(index / rows);
    const uint64_t row = index % rows;
    StorageT value = static_cast<StorageT>(1);
    for (uint32_t slot = 0; slot < arity; ++slot) {
        value *= features[
            static_cast<uint64_t>(combos[static_cast<uint64_t>(candidate) * arity + slot]) *
                rows + row];
    }
    materialized[index] = value;
}

// The 8df product exposes only the ABI-1.0 float-to-double target-rank
// primitive, while the candidate exposes typed precision specialisations.
// Keep this narrow cache-preparation step in the common harness so the same
// tracked direct source compiles against both products.  Product Spearman
// scoring below always uses the product static rank kernel; this helper is
// measured as the direct lane's target-rank preparation operation and is not
// substituted for a product metric kernel.
template <typename StorageT, typename RankT>
__global__ void build_target_ranks_common_kernel(
    const StorageT* target,
    uint64_t rows,
    RankT* ranks
) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = first; row < rows; row += stride) {
        const StorageT value = target[row];
        if (!isfinite(value)) {
            ranks[row] = static_cast<RankT>(0);
            continue;
        }
        RankT less = static_cast<RankT>(0);
        RankT equal = static_cast<RankT>(0);
        for (uint64_t other = 0; other < rows; ++other) {
            const StorageT candidate = target[other];
            if (!isfinite(candidate)) {
                continue;
            }
            if (candidate < value) {
                less += static_cast<RankT>(1);
            } else if (candidate == value) {
                equal += static_cast<RankT>(1);
            }
        }
        ranks[row] = less + static_cast<RankT>(0.5) *
            (equal - static_cast<RankT>(1));
    }
}

template <typename StorageT>
hipError_t launch_materialize(
    const void* features,
    const uint32_t* combos,
    void* materialized,
    uint64_t rows,
    uint32_t candidates,
    uint32_t arity,
    hipStream_t stream
) {
    if (rows == 0 || candidates == 0 || arity == 0 || arity > 5 ||
        candidates > std::numeric_limits<unsigned int>::max()) {
        return hipErrorInvalidValue;
    }
    const uint64_t total = rows * static_cast<uint64_t>(candidates);
    const uint32_t blocks = static_cast<uint32_t>(
        (total + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    hipLaunchKernelGGL(
        materialize_candidates_kernel<StorageT>,
        dim3(blocks), dim3(kThreadsPerBlock), 0, stream,
        static_cast<const StorageT*>(features), combos,
        static_cast<StorageT*>(materialized), rows, candidates, arity);
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity>
hipError_t launch_continuous_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint64_t candidates,
    const uint32_t* metric_ids,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_continuous_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity, false>)),
        dim3(static_cast<unsigned int>(candidates)), dim3(kThreadsPerBlock), 0, stream,
        features, target, means, combos, rows, 0, candidates, metric_ids, 1,
        metric_values);
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_continuous(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint32_t arity,
    uint64_t candidates,
    const uint32_t* metric_ids,
    ResultT* metric_values,
    hipStream_t stream
) {
    switch (arity) {
    case 1: return launch_continuous_static<StorageT, AccumT, ResultT, 1>(
        features, target, means, combos, rows, candidates, metric_ids, metric_values, stream);
    case 2: return launch_continuous_static<StorageT, AccumT, ResultT, 2>(
        features, target, means, combos, rows, candidates, metric_ids, metric_values, stream);
    case 3: return launch_continuous_static<StorageT, AccumT, ResultT, 3>(
        features, target, means, combos, rows, candidates, metric_ids, metric_values, stream);
    case 4: return launch_continuous_static<StorageT, AccumT, ResultT, 4>(
        features, target, means, combos, rows, candidates, metric_ids, metric_values, stream);
    case 5: return launch_continuous_static<StorageT, AccumT, ResultT, 5>(
        features, target, means, combos, rows, candidates, metric_ids, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, uint32_t Bins>
hipError_t launch_mutual_info_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint64_t candidates,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_mutual_info_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity, Bins>)),
        dim3(static_cast<unsigned int>(candidates)), dim3(kMiThreadsPerBlock), 0, stream,
#if GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API >= 2
        features, target, means, combos, rows, 0, candidates, 1, 0, 0, metric_values);
#else
        features, target, means, combos, rows, 0, candidates, 1, 0, metric_values);
#endif
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity>
hipError_t launch_mutual_info_bins(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint64_t candidates,
    uint32_t bins,
    ResultT* metric_values,
    hipStream_t stream
) {
    switch (bins) {
    case 2: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 2>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 4: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 4>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 8: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 8>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 12: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 12>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 16: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 16>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 24: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 24>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 32: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 32>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 48: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 48>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 64: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 64>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 96: return launch_mutual_info_static<StorageT, AccumT, ResultT, Arity, 96>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_mutual_info(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint32_t arity,
    uint64_t candidates,
    uint32_t bins,
    ResultT* metric_values,
    hipStream_t stream
) {
    switch (arity) {
    case 1: return launch_mutual_info_bins<StorageT, AccumT, ResultT, 1>(
        features, target, means, combos, rows, candidates, bins, metric_values, stream);
    case 2: return launch_mutual_info_bins<StorageT, AccumT, ResultT, 2>(
        features, target, means, combos, rows, candidates, bins, metric_values, stream);
    case 3: return launch_mutual_info_bins<StorageT, AccumT, ResultT, 3>(
        features, target, means, combos, rows, candidates, bins, metric_values, stream);
    case 4: return launch_mutual_info_bins<StorageT, AccumT, ResultT, 4>(
        features, target, means, combos, rows, candidates, bins, metric_values, stream);
    case 5: return launch_mutual_info_bins<StorageT, AccumT, ResultT, 5>(
        features, target, means, combos, rows, candidates, bins, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity>
hipError_t launch_spearman_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint64_t candidates,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_spearman_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity>)),
        dim3(static_cast<unsigned int>(candidates)), dim3(kThreadsPerBlock), 0, stream,
        features, target, means, combos, rows, 0, candidates, 1, 0, metric_values);
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_spearman(
    const StorageT* features,
    const StorageT* target,
    const AccumT* means,
    const uint32_t* combos,
    uint64_t rows,
    uint32_t arity,
    uint64_t candidates,
    const void* target_ranks,
    ResultT* metric_values,
    hipStream_t stream
) {
    // The direct wrapper intentionally uses the product's static Spearman
    // kernels for every arity.  The candidate's cached-unary optimization is
    // a separate payload path; retaining it here would make the baseline
    // 8df helper depend on an API that does not exist there.
    static_cast<void>(target_ranks);
    switch (arity) {
    case 1: return launch_spearman_static<StorageT, AccumT, ResultT, 1>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 2: return launch_spearman_static<StorageT, AccumT, ResultT, 2>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 3: return launch_spearman_static<StorageT, AccumT, ResultT, 3>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 4: return launch_spearman_static<StorageT, AccumT, ResultT, 4>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    case 5: return launch_spearman_static<StorageT, AccumT, ResultT, 5>(
        features, target, means, combos, rows, candidates, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename ResultT, bool Descending>
hipError_t launch_topk_static(
    const ResultT* metric_values,
    uint64_t candidates,
    uint32_t top_k,
    ResultT* partial_scores,
    uint32_t* partial_indices,
    uint32_t* selected_indices,
    uint32_t partial_blocks,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::select_topk_partials_kernel_static<
            ResultT, Descending>)),
        dim3(partial_blocks), dim3(kTopKThreadsPerBlock), 0, stream,
        metric_values, candidates, 1, 0, top_k, partial_scores, partial_indices);
    if (hipGetLastError() != hipSuccess) return hipGetLastError();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::merge_topk_partials_kernel_static<
            ResultT, Descending>)),
        dim3(1), dim3(kTopKThreadsPerBlock), 0, stream, partial_scores, partial_indices,
        static_cast<uint64_t>(partial_blocks) * top_k, top_k, selected_indices);
    return hipGetLastError();
}

template <typename ResultT>
hipError_t launch_topk(
    const void* metric_values,
    uint64_t candidates,
    uint32_t top_k,
    void* partial_scores,
    uint32_t* partial_indices,
    uint32_t* selected_indices,
    uint32_t partial_blocks,
    hipStream_t stream
) {
    if (top_k == 0 || partial_blocks == 0) return hipSuccess;
    return launch_topk_static<ResultT, true>(
        static_cast<const ResultT*>(metric_values), candidates, top_k,
        static_cast<ResultT*>(partial_scores), partial_indices, selected_indices,
        partial_blocks, stream);
}

template <typename ResultT>
hipError_t launch_selected_rows(
    const void* metric_values,
    const uint32_t* selected_indices,
    uint32_t selected_count,
    void* selected_metric_values,
    hipStream_t stream
) {
    const uint64_t total = selected_count;
    const uint32_t blocks = static_cast<uint32_t>((total + 255u) / 256u);
    if (blocks == 0) return hipSuccess;
    hipLaunchKernelGGL(
#if GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API >= 2
        HIP_KERNEL_NAME((kernel::precision_kernel::copy_selected_rows_kernel<ResultT>)),
#else
        HIP_KERNEL_NAME((kernel::precision_kernel::copy_selected_metric_rows_kernel<ResultT>)),
#endif
        dim3(blocks), dim3(256), 0, stream,
        static_cast<const ResultT*>(metric_values), selected_indices, selected_count,
        1, static_cast<ResultT*>(selected_metric_values));
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t dispatch_metric(
    uint32_t metric_id,
    uint32_t arity,
    uint32_t bins,
    const void* features,
    const void* target,
    const void* means,
    const void* target_ranks,
    const uint32_t* combos,
    const uint32_t* metric_ids,
    uint64_t rows,
    uint64_t candidates,
    void* metric_values,
    hipStream_t stream
) {
    if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2) {
        return launch_continuous<StorageT, AccumT, ResultT>(
            static_cast<const StorageT*>(features), static_cast<const StorageT*>(target),
            static_cast<const AccumT*>(means), combos, rows, arity, candidates, metric_ids,
            static_cast<ResultT*>(metric_values), stream);
    }
    if (metric_id == GAFIME_METRIC_MUTUAL_INFO) {
        return launch_mutual_info<StorageT, AccumT, ResultT>(
            static_cast<const StorageT*>(features), static_cast<const StorageT*>(target),
            static_cast<const AccumT*>(means), combos, rows, arity, candidates, bins,
            static_cast<ResultT*>(metric_values), stream);
    }
    if (metric_id == GAFIME_METRIC_SPEARMAN) {
        return launch_spearman<StorageT, AccumT, ResultT>(
            static_cast<const StorageT*>(features), static_cast<const StorageT*>(target),
            static_cast<const AccumT*>(means), combos, rows, arity, candidates,
            target_ranks, static_cast<ResultT*>(metric_values), stream);
    }
    return hipErrorInvalidValue;
}

}  // namespace

namespace gafime_rocm_native_direct {

hipError_t materialize(
    uint32_t profile,
    const void* features,
    const uint32_t* combos,
    void* materialized,
    uint64_t rows,
    uint32_t candidates,
    uint32_t arity,
    hipStream_t stream
) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
    case GAFIME_PRECISION_MIXED:
        return launch_materialize<float>(features, combos, materialized, rows, candidates, arity, stream);
    case GAFIME_PRECISION_FP64:
        return launch_materialize<double>(features, combos, materialized, rows, candidates, arity, stream);
    default: return hipErrorInvalidValue;
    }
}

hipError_t target_ranks(
    uint32_t profile,
    const void* target,
    void* ranks,
    uint64_t rows,
    hipStream_t stream
) {
    if (rows == 0 || rows > std::numeric_limits<unsigned int>::max() *
        static_cast<uint64_t>(kThreadsPerBlock)) return hipErrorInvalidValue;
    const uint32_t blocks = static_cast<uint32_t>((rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((build_target_ranks_common_kernel<float, float>)),
            dim3(blocks), dim3(kThreadsPerBlock), 0, stream,
            static_cast<const float*>(target), rows, static_cast<float*>(ranks));
        break;
    case GAFIME_PRECISION_MIXED:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((build_target_ranks_common_kernel<float, double>)),
            dim3(blocks), dim3(kThreadsPerBlock), 0, stream,
            static_cast<const float*>(target), rows, static_cast<double*>(ranks));
        break;
    case GAFIME_PRECISION_FP64:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((build_target_ranks_common_kernel<double, double>)),
            dim3(blocks), dim3(kThreadsPerBlock), 0, stream,
            static_cast<const double*>(target), rows, static_cast<double*>(ranks));
        break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

hipError_t metric(
    uint32_t profile,
    uint32_t metric_id,
    uint32_t arity,
    uint32_t bins,
    const void* features,
    const void* target,
    const void* means,
    const void* target_ranks,
    const uint32_t* combos,
    const uint32_t* metric_ids,
    uint64_t rows,
    uint64_t candidates,
    void* metric_values,
    hipStream_t stream
) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        return dispatch_metric<float, float, float>(
            metric_id, arity, bins, features, target, means, target_ranks, combos, metric_ids,
            rows, candidates, metric_values, stream);
    case GAFIME_PRECISION_MIXED:
        return dispatch_metric<float, double, double>(
            metric_id, arity, bins, features, target, means, target_ranks, combos, metric_ids,
            rows, candidates, metric_values, stream);
    case GAFIME_PRECISION_FP64:
        return dispatch_metric<double, double, double>(
            metric_id, arity, bins, features, target, means, target_ranks, combos, metric_ids,
            rows, candidates, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

hipError_t ranking_topk(
    uint32_t profile,
    const void* metric_values,
    uint64_t candidates,
    uint32_t top_k,
    void* partial_scores,
    uint32_t* partial_indices,
    uint32_t* selected_indices,
    uint32_t partial_blocks,
    hipStream_t stream
) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        return launch_topk<float>(metric_values, candidates, top_k, partial_scores,
                                  partial_indices, selected_indices, partial_blocks, stream);
    case GAFIME_PRECISION_MIXED:
    case GAFIME_PRECISION_FP64:
        return launch_topk<double>(metric_values, candidates, top_k, partial_scores,
                                   partial_indices, selected_indices, partial_blocks, stream);
    default: return hipErrorInvalidValue;
    }
}

hipError_t selected_rows(
    uint32_t profile,
    const void* metric_values,
    const uint32_t* selected_indices,
    uint32_t selected_count,
    void* selected_metric_values,
    hipStream_t stream
) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        return launch_selected_rows<float>(metric_values, selected_indices, selected_count,
                                           selected_metric_values, stream);
    case GAFIME_PRECISION_MIXED:
    case GAFIME_PRECISION_FP64:
        return launch_selected_rows<double>(metric_values, selected_indices, selected_count,
                                            selected_metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

}  // namespace gafime_rocm_native_direct
