#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include "rocm_api.hpp"
#include "kernels.hpp"
#include "../common/covariance_policy.hpp"
#include "../common/gpu_abi_impl.hpp"
#include "../common/gafime_gpu_internal_abi.hpp"

#ifndef GAFIME_GPU_MI_ACCUMULATION_FP64
#define GAFIME_GPU_MI_ACCUMULATION_FP64 0
#endif

namespace gafime_rocm_v1 {

template <typename StorageT, typename AccumT>
hipError_t launch_precision_interaction_diagnostics(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t combo_count,
    uint64_t n_samples,
    uint32_t max_arity,
    uint64_t* overflow_row_counts,
    uint32_t* flags,
    hipStream_t stream
) {
    if (combo_count == 0) {
        return hipSuccess;
    }
    if (combo_count > std::numeric_limits<unsigned int>::max()) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::interaction_diagnostics_kernel<StorageT, AccumT>)),
        dim3(static_cast<unsigned int>(combo_count)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        features,
        target,
        column_means,
        combo_indices,
        n_samples,
        max_arity,
        overflow_row_counts,
        flags
    );
    return hipGetLastError();
}

// ABI-1.1 profile launch helpers.  All profile selection happens in the host
// dispatch table below; these functions have only concrete scalar template
// parameters and therefore never branch on precision from a hot loop.
template <typename StorageT, typename AccumT>
hipError_t launch_precision_target_stats(
    const StorageT* target,
    uint64_t n_samples,
    PrecisionTargetStatsDevice<AccumT>* target_stats,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::precision_kernel::target_stats_kernel<StorageT, AccumT>),
        dim3(1), dim3(kThreadsPerBlock), 0, stream, target, n_samples, target_stats
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT>
hipError_t launch_precision_unary_feature_stats(
    const StorageT* features,
    uint64_t n_samples,
    uint32_t n_features,
    PrecisionUnaryFeatureStatsDevice<AccumT>* feature_stats,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::precision_kernel::unary_feature_stats_kernel<StorageT, AccumT>),
        dim3(n_features), dim3(kThreadsPerBlock), 0, stream,
        features, n_samples, n_features, feature_stats
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_precision_continuous_unary_all_finite(
    const StorageT* features,
    const StorageT* target,
    const PrecisionTargetStatsDevice<AccumT>* target_stats,
    const PrecisionUnaryFeatureStatsDevice<AccumT>* feature_stats,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_continuous_unary_all_finite_chunk_kernel<
            StorageT, AccumT, ResultT>)),
        dim3(static_cast<unsigned int>(combo_count)), dim3(kThreadsPerBlock), 0, stream,
        features, target, target_stats, feature_stats, combo_indices, n_samples,
        descriptor_offset, combo_count, metric_ids, metric_count, metric_values
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, bool Scaled>
hipError_t launch_precision_continuous_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_continuous_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity, Scaled>)),
        dim3(static_cast<unsigned int>(combo_count)), dim3(kThreadsPerBlock), 0, stream,
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_precision_continuous(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t scaled_covariance,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    ResultT* metric_values,
    hipStream_t stream
) {
    if (scaled_covariance != 0) {
        switch (arity) {
        case 1: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 1, true>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
        case 2: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 2, true>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
        case 3: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 3, true>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
        case 4: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 4, true>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
        case 5: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 5, true>(
            features, target, column_means, combo_indices, n_samples, descriptor_offset,
            combo_count, metric_ids, metric_count, metric_values, stream);
        default: return hipErrorInvalidValue;
        }
    }
    switch (arity) {
    case 1: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 1, false>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values, stream);
    case 2: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 2, false>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values, stream);
    case 3: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 3, false>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values, stream);
    case 4: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 4, false>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values, stream);
    case 5: return launch_precision_continuous_static<StorageT, AccumT, ResultT, 5, false>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_ids, metric_count, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, uint32_t Bins>
hipError_t launch_precision_mutual_info_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values,
    uint32_t legacy_nonfinite,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_mutual_info_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity, Bins>)),
        dim3(static_cast<unsigned int>(combo_count)), dim3(kMiThreadsPerBlock), 0, stream,
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, legacy_nonfinite, metric_values
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity>
hipError_t launch_precision_mutual_info_for_bins(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t bins,
    ResultT* metric_values,
    uint32_t legacy_nonfinite,
    hipStream_t stream
) {
    switch (bins) {
    case 2: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 2>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 4: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 4>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 8: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 8>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 12: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 12>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 16: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 16>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 24: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 24>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 32: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 32>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 48: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 48>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 64: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 64>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    case 96: return launch_precision_mutual_info_static<StorageT, AccumT, ResultT, Arity, 96>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, legacy_nonfinite, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_precision_mutual_info(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t bins,
    ResultT* metric_values,
    uint32_t legacy_nonfinite,
    hipStream_t stream
) {
    switch (arity) {
    case 1: return launch_precision_mutual_info_for_bins<StorageT, AccumT, ResultT, 1>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, bins, metric_values, legacy_nonfinite, stream);
    case 2: return launch_precision_mutual_info_for_bins<StorageT, AccumT, ResultT, 2>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, bins, metric_values, legacy_nonfinite, stream);
    case 3: return launch_precision_mutual_info_for_bins<StorageT, AccumT, ResultT, 3>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, bins, metric_values, legacy_nonfinite, stream);
    case 4: return launch_precision_mutual_info_for_bins<StorageT, AccumT, ResultT, 4>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, bins, metric_values, legacy_nonfinite, stream);
    case 5: return launch_precision_mutual_info_for_bins<StorageT, AccumT, ResultT, 5>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, bins, metric_values, legacy_nonfinite, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity,
          typename MeanT = AccumT>
hipError_t launch_precision_spearman_static(
    const StorageT* features,
    const StorageT* target,
    const MeanT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::score_spearman_chunk_kernel_static<
            StorageT, AccumT, ResultT, Arity, MeanT>)),
        dim3(static_cast<unsigned int>(combo_count)), dim3(kThreadsPerBlock), 0, stream,
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values
    );
    return hipGetLastError();
}

template <typename StorageT, typename AccumT, typename ResultT, typename MeanT = AccumT>
hipError_t launch_precision_spearman(
    const StorageT* features,
    const StorageT* target,
    const MeanT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values,
    hipStream_t stream
) {
    switch (arity) {
    case 1: return launch_precision_spearman_static<StorageT, AccumT, ResultT, 1, MeanT>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, stream);
    case 2: return launch_precision_spearman_static<StorageT, AccumT, ResultT, 2, MeanT>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, stream);
    case 3: return launch_precision_spearman_static<StorageT, AccumT, ResultT, 3, MeanT>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, stream);
    case 4: return launch_precision_spearman_static<StorageT, AccumT, ResultT, 4, MeanT>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, stream);
    case 5: return launch_precision_spearman_static<StorageT, AccumT, ResultT, 5, MeanT>(
        features, target, column_means, combo_indices, n_samples, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, stream);
    default: return hipErrorInvalidValue;
    }
}

template <typename ResultT>
hipError_t launch_precision_select_topk(
    const ResultT* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    uint32_t descending,
    uint32_t* selected_indices,
    ResultT* partial_scores,
    uint32_t* partial_indices,
    uint32_t partial_blocks,
    hipStream_t stream
) {
    if (top_k == 0 || partial_blocks == 0) {
        return hipSuccess;
    }
    const dim3 grid(partial_blocks);
    const dim3 block(kTopKThreadsPerBlock);
    if (descending != 0u) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::precision_kernel::select_topk_partials_kernel_static<ResultT, true>)),
            grid, block, 0, stream, metric_values, row_count, metric_count,
            primary_metric_index, top_k, partial_scores, partial_indices
        );
        if (hipGetLastError() != hipSuccess) {
            return hipGetLastError();
        }
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::precision_kernel::merge_topk_partials_kernel_static<ResultT, true>)),
            dim3(1), block, 0, stream, partial_scores, partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k, top_k, selected_indices
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::precision_kernel::select_topk_partials_kernel_static<ResultT, false>)),
            grid, block, 0, stream, metric_values, row_count, metric_count,
            primary_metric_index, top_k, partial_scores, partial_indices
        );
        if (hipGetLastError() != hipSuccess) {
            return hipGetLastError();
        }
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::precision_kernel::merge_topk_partials_kernel_static<ResultT, false>)),
            dim3(1), block, 0, stream, partial_scores, partial_indices,
            static_cast<uint64_t>(partial_blocks) * top_k, top_k, selected_indices
        );
    }
    return hipGetLastError();
}

template <typename ResultT>
hipError_t launch_precision_copy_selected_metric_rows(
    const ResultT* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    ResultT* selected_metric_values,
    hipStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) {
        return hipSuccess;
    }
    constexpr uint32_t kCopyThreads = 256;
    const uint32_t blocks = static_cast<uint32_t>((total + kCopyThreads - 1) / kCopyThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::copy_selected_rows_kernel<ResultT>)),
        dim3(blocks), dim3(kCopyThreads), 0, stream, metric_values,
        selected_indices, selected_count, metric_count, selected_metric_values
    );
    return hipGetLastError();
}

// ABI 1.0 compatibility adapters.  The legacy entry points retain their
// historical signatures and storage/result dtypes, but shared execution now
// uses the already-instantiated fp32 ABI-1.1 kernels.  Only Spearman keeps a
// separate adapter because ABI 1.0 accumulates ranks/covariance in fp64, and
// MI receives its one historical nonfinite-bin policy bit.
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
    static_cast<void>(target_stats);
    static_cast<void>(feature_stats);
    if (arity > kTemplateMaxArity) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::legacy_compat::score_continuous_chunk_kernel)),
            dim3(static_cast<unsigned int>(combo_count)),
            dim3(kThreadsPerBlock),
            0,
            stream,
            features,
            target,
            column_means,
            combo_indices,
            n_samples,
            n_features,
            arity,
            scaled_covariance,
            descriptor_offset,
            combo_count,
            metric_ids,
            metric_count,
            metric_values
        );
        return hipGetLastError();
    }
    static_cast<void>(n_features);
    if (scaled_covariance == 0u && arity == 1u && features_are_finite != 0u &&
        target_is_finite != 0u && target_stats != nullptr && feature_stats != nullptr) {
        return launch_precision_continuous_unary_all_finite<float, float, float>(
            features, target, target_stats, feature_stats, combo_indices, n_samples,
            descriptor_offset, combo_count, metric_ids, metric_count, metric_values, stream);
    }
    return launch_precision_continuous<float, float, float>(
        features, target, column_means, combo_indices, n_samples, arity,
        descriptor_offset, combo_count, scaled_covariance, metric_ids,
        metric_count, metric_values, stream);
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
    if (arity > kTemplateMaxArity) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::legacy_compat::score_mutual_info_chunk_kernel)),
            dim3(static_cast<unsigned int>(combo_count)),
            dim3(kMiThreadsPerBlock),
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
    static_cast<void>(n_features);
    return launch_precision_mutual_info<float, float, float>(
        features, target, column_means, combo_indices, n_samples, arity,
        descriptor_offset, combo_count, metric_count, metric_index, bins,
        metric_values, 1u, stream);
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
    if (arity > kTemplateMaxArity) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::legacy_compat::score_spearman_chunk_kernel)),
            dim3(static_cast<unsigned int>(combo_count)),
            dim3(kThreadsPerBlock),
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
    static_cast<void>(n_features);
    if (arity == 1u && target_ranks != nullptr) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((kernel::precision_kernel::
                score_spearman_unary_cached_target_ranks_kernel<float, double, float>)),
            dim3(static_cast<unsigned int>(combo_count)),
            dim3(kThreadsPerBlock),
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
    return launch_precision_spearman<float, double, float, float>(
        features, target, column_means, combo_indices, n_samples, arity,
        descriptor_offset, combo_count, metric_count, metric_index,
        metric_values, stream);
}

template <typename StorageT, typename RankT>
hipError_t launch_precision_spearman_target_ranks(
    const StorageT* target,
    uint64_t n_samples,
    RankT* target_ranks,
    hipStream_t stream
) {
    if (n_samples == 0 || target == nullptr || target_ranks == nullptr) {
        return hipSuccess;
    }
    if (n_samples > std::numeric_limits<unsigned int>::max() *
            static_cast<uint64_t>(kThreadsPerBlock)) {
        return hipErrorInvalidValue;
    }
    const uint64_t block_count = 1 + (n_samples - 1) / kThreadsPerBlock;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::
            build_spearman_target_ranks_kernel<StorageT, RankT>)),
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

template <typename StorageT, typename AccumT, typename ResultT>
hipError_t launch_precision_spearman_unary_cached_target_ranks(
    const StorageT* features,
    const StorageT* target,
    const AccumT* target_ranks,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values,
    hipStream_t stream
) {
    if (combo_count == 0) {
        return hipSuccess;
    }
    if (combo_count > std::numeric_limits<unsigned int>::max()) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((kernel::precision_kernel::
            score_spearman_unary_cached_target_ranks_kernel<StorageT, AccumT, ResultT>)),
        dim3(static_cast<unsigned int>(combo_count)),
        dim3(kThreadsPerBlock),
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
    return launch_precision_select_topk<float>(
        metric_values, row_count, metric_count, primary_metric_index, top_k,
        descending, selected_indices, partial_scores, partial_indices,
        partial_blocks, stream);
}

hipError_t launch_copy_selected_metric_rows(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values,
    hipStream_t stream
) {
    return launch_precision_copy_selected_metric_rows<float>(
        metric_values, selected_indices, selected_count, metric_count,
        selected_metric_values, stream);
}

}  // namespace gafime_rocm_v1

namespace {


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

// Both ABI generations now share one opaque resident owner.  The magic guards
// the frozen and generic entry points before either interprets the handle.
constexpr uint64_t kPrecisionHipMatrixMagic = 0x4741464849505032ull;  // GAFHIPP2

struct HipMatrixHeader {
    uint64_t magic;
};

struct PrecisionGraphChunkShape {
    uint32_t arity;
    uint32_t family;
    uint32_t scaled_covariance;
    uint64_t descriptor_offset;
    uint64_t combo_count;
    uint32_t mi_bins;
};

struct PrecisionResultSink {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t* flags;
    uint64_t capacity;
    uint64_t* row_count;
    uint32_t* combo_indices;
    void* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
};

struct PrecisionHipMatrix;

struct PrecisionDispatchTable {
    uint32_t profile;
    uint32_t storage_dtype;
    uint32_t result_dtype;
    uint32_t storage_bytes;
    uint32_t accumulator_bytes;
    uint32_t result_bytes;
    int (*upload)(PrecisionHipMatrix*, const void*, const void*, uint64_t, uint32_t);
    int (*update_target)(PrecisionHipMatrix*, const void*, uint64_t);
    int (*execute)(PrecisionHipMatrix*, const GafimeLaunchProtocol*, PrecisionResultSink*);
    int (*interaction_diagnostics)(
        PrecisionHipMatrix*, GafimeInteractionDiagnosticBatch*
    );
};

struct PrecisionHipMatrix : HipMatrixHeader {
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t arch_class;
    uint32_t profile;
    // Frozen ABI 1.0 is a policy adapter over this same resident owner.  It
    // keeps its established preprocessing/Spearman/MI details without
    // retaining a second host lifecycle or a fourth public precision route.
    bool legacy_abi10;
    bool unified_memory;
    uint64_t rows;
    uint32_t cols;
    bool content_valid;
    bool features_are_finite;
    bool target_is_finite;
    const PrecisionDispatchTable* dispatch;

    // These pointer types are deliberately erased only at the opaque matrix
    // boundary.  Every dispatch-table target immediately casts to its concrete
    // scalar type and launches an explicitly instantiated kernel family.
    void* features;
    void* target;
    void* column_means;
    void* target_stats;
    void* feature_stats;
    // ABI-1.1 Spearman target ranks are owned by the selected route.  The
    // erased pointer is never consumed without the profile-typed dispatch
    // below proving its dtype and readiness.
    void* spearman_target_ranks;
    uint64_t spearman_target_rank_capacity;
    uint32_t spearman_target_ranks_profile;
    bool spearman_target_ranks_ready;
    void* metric_values;
    uint64_t metric_value_capacity;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    uint32_t* selected_indices;
    uint64_t selected_index_capacity;
    void* selected_metric_values;
    uint64_t selected_metric_value_capacity;
    void* topk_partial_scores;
    uint64_t topk_partial_score_capacity;
    uint32_t* topk_partial_indices;
    uint64_t topk_partial_index_capacity;

    // Profile is part of every retained identity.  A descriptor, statistic, or
    // graph created by fp32 cannot be accepted by a mixed/fp64 matrix even if
    // all external generation counters happen to match.
    uint32_t target_stats_profile;
    uint32_t feature_stats_profile;
    uint32_t descriptor_profile;
    uint32_t graph_profile;
    uint64_t descriptor_generation;
    uint64_t descriptor_combo_len;
    uint64_t descriptor_metric_id_len;
    std::vector<int> feature_abs_exponents;
    int target_abs_exponent;
    std::vector<uint8_t> covariance_modes;

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
    std::vector<PrecisionGraphChunkShape> graph_chunk_shapes;

    // Host mirrors preserve the requested storage dtype.  fp64 upload and
    // target replacement never materialise a float staging vector.
    std::vector<float> features_host_f32;
    std::vector<float> target_host_f32;
    std::vector<float> column_means_host_f32;
    std::vector<double> features_host_f64;
    std::vector<double> target_host_f64;
    std::vector<double> column_means_host_f64;
};

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

template <typename MatrixT>
int require_valid_matrix_content(const MatrixT* matrix) {
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
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    flags |= GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64;
#endif
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

template <typename MatrixT>
int validate_protocol(
    const GafimeLaunchProtocol* protocol,
    const MatrixT* matrix
) {
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

constexpr uint32_t kInteractionDiagnosticMaxArity = 5;

struct InteractionDiagnosticScanPlan {
    std::vector<uint32_t> combo_indices;
    std::vector<uint64_t> output_rows;
};

template <typename T>
class HipTransientBuffer {
public:
    explicit HipTransientBuffer(bool managed) : managed_(managed) {}

    ~HipTransientBuffer() {
        static_cast<void>(hipFree(ptr_));
    }

    HipTransientBuffer(const HipTransientBuffer&) = delete;
    HipTransientBuffer& operator=(const HipTransientBuffer&) = delete;

    int allocate(uint64_t count) {
        if (!allocation_fits_size_t(count, sizeof(T))) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        if (count == 0) {
            return GAFIME_STATUS_OK;
        }
        return allocate_hip_storage(&ptr_, static_cast<size_t>(count) * sizeof(T), managed_);
    }

    T* get() const {
        return ptr_;
    }

private:
    T* ptr_ = nullptr;
    bool managed_;
};

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

bool precision_metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO || metric_id == GAFIME_METRIC_SPEARMAN;
}

bool precision_family_supported(uint32_t family) {
    return family == GAFIME_FAMILY_CONTINUOUS ||
        family == GAFIME_FAMILY_TIME_SERIES ||
        family == GAFIME_FAMILY_DECISION_PATH;
}

uint32_t expected_storage_dtype_for_precision(uint32_t profile) {
    return gafime_rocm_v1::precision_lane_uses_f32_storage(profile)
        ? GAFIME_DTYPE_F32
        : GAFIME_DTYPE_F64;
}

uint32_t expected_result_dtype_for_precision(uint32_t profile) {
    return gafime_rocm_v1::precision_lane_uses_f32_results(profile)
        ? GAFIME_DTYPE_F32
        : GAFIME_DTYPE_F64;
}

int validate_precision_matrix_desc(const GafimePrecisionMatrixDesc* desc) {
    if (desc == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->abi_version != GAFIME_PRECISION_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!gafime_rocm_v1::precision_lane_is_valid(desc->profile) ||
        desc->dtype != expected_storage_dtype_for_precision(desc->profile) ||
        desc->layout != GAFIME_MATRIX_ROW_MAJOR || desc->flags != 0 ||
        desc->reserved32 != 0 || desc->rows == 0 || desc->cols == 0 ||
        desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : desc->reserved) {
        if (reserved != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    const size_t storage_bytes = desc->dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double);
    uint64_t element_count = 0;
    uint64_t byte_count = 0;
    if (!checked_mul_u64(desc->rows, desc->cols, &element_count) ||
        !checked_mul_u64(element_count, storage_bytes, &byte_count) ||
        !allocation_fits_size_t(element_count, storage_bytes) ||
        !allocation_fits_size_t(desc->rows, storage_bytes) ||
        !allocation_fits_size_t(desc->cols, storage_bytes) ||
        desc->bytes != byte_count) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

void precision_destroy_graph_cache(PrecisionHipMatrix* matrix) {
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
    matrix->graph_profile = 0;
}

void precision_invalidate_descriptor_cache(PrecisionHipMatrix* matrix) {
    if (matrix == nullptr) {
        return;
    }
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->descriptor_profile = 0;
    matrix->covariance_modes.clear();
}

bool precision_spearman_target_rank_cache_eligible(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (matrix == nullptr || protocol == nullptr || protocol->chunks == nullptr ||
        matrix->spearman_target_ranks == nullptr ||
        matrix->spearman_target_rank_capacity < matrix->rows ||
        !matrix->features_are_finite || !matrix->target_is_finite ||
        matrix->rows < gafime_rocm_v1::kSpearmanTargetRankCacheMinSamples ||
        matrix->rows > gafime_rocm_v1::kSpearmanTargetRankCacheMaxSamples ||
        protocol->permutations.permutation_count != 0 ||
        !protocol_has_spearman_metric(protocol)) {
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

template <typename StorageT, typename AccumT>
int precision_prepare_spearman_target_rank_cache(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (!precision_spearman_target_rank_cache_eligible(matrix, protocol)) {
        return GAFIME_STATUS_OK;
    }
    if (matrix->spearman_target_ranks_ready &&
        matrix->spearman_target_ranks_profile == matrix->profile) {
        return GAFIME_STATUS_OK;
    }
    // Invalidate before issuing the asynchronous build.  A failed build can
    // therefore never leave a stale target/profile pair marked usable.
    matrix->spearman_target_ranks_ready = false;
    matrix->spearman_target_ranks_profile = 0;
    int status = hip_status(gafime_rocm_v1::launch_precision_spearman_target_ranks<StorageT, AccumT>(
        static_cast<const StorageT*>(matrix->target), matrix->rows,
        static_cast<AccumT*>(matrix->spearman_target_ranks), nullptr
    ));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        matrix->spearman_target_ranks_profile = matrix->profile;
        matrix->spearman_target_ranks_ready = true;
    }
    return status;
}

bool precision_descriptors_resident(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (matrix == nullptr || protocol == nullptr) {
        return false;
    }
    const bool immutable = (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0;
    const uint64_t generation =
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    return immutable && generation != 0 && matrix->descriptor_profile == matrix->profile &&
        matrix->descriptor_generation == generation &&
        matrix->descriptor_combo_len == protocol->combo_indices.len &&
        matrix->descriptor_metric_id_len == protocol->metric_ids.len &&
        matrix->covariance_modes.size() == protocol->chunk_count;
}

uint32_t precision_mi_bins_for_chunk(
    const GafimeLaunchProtocol* protocol,
    const GafimeArityChunk& chunk
) {
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

uint64_t precision_planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t rows = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        rows += protocol->chunks[chunk_index].combo_count;
    }
    return rows;
}

uint64_t precision_output_row_count(const GafimeLaunchProtocol* protocol, uint64_t planned) {
    return protocol->rank.top_k == 0 ? planned : std::min<uint64_t>(planned, protocol->rank.top_k);
}

uint32_t precision_primary_metric_index(const GafimeLaunchProtocol* protocol) {
    for (uint32_t metric_index = 0; metric_index < protocol->metric_ids.len; ++metric_index) {
        if (protocol->metric_ids.ptr[metric_index] == protocol->rank.primary_metric) {
            return metric_index;
        }
    }
    return 0;
}

int validate_precision_base_protocol(
    const GafimeLaunchProtocol* protocol,
    const PrecisionHipMatrix* matrix
) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION ||
        protocol->backend_kind != GAFIME_BACKEND_ROCM ||
        protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols ||
        protocol->max_arity == 0 ||
        protocol->max_arity > gafime_rocm_v1::kTemplateMaxArity ||
        protocol->family_count == 0 || protocol->family_count > 3 ||
        protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0 ||
        protocol->metric_ids.len > std::numeric_limits<uint32_t>::max() ||
        protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr ||
        protocol->chunk_count == 0 ||
        (protocol->shape_hint_count != 0 && protocol->shape_hints == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    constexpr uint32_t kKnownLaunchFlags =
        GAFIME_LAUNCH_FLAG_GRAPH | GAFIME_LAUNCH_FLAG_MI_APPROX |
        GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    if ((protocol->flags & ~kKnownLaunchFlags) != 0 ||
        (protocol->rank.top_k != 0 && protocol->rank.include_ties != 0)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    bool has_mi = false;
    for (uint64_t metric_index = 0; metric_index < protocol->metric_ids.len; ++metric_index) {
        const uint32_t metric = protocol->metric_ids.ptr[metric_index];
        if (!precision_metric_supported(metric)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        has_mi = has_mi || metric == GAFIME_METRIC_MUTUAL_INFO;
    }
    if (has_mi && matrix->rows > std::numeric_limits<uint32_t>::max()) {
        // The fixed-bin histogram counts are explicitly u32 and must never be
        // widened into a floating control path.
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->rank.top_k != 0) {
        bool primary_found = false;
        for (uint64_t metric_index = 0; metric_index < protocol->metric_ids.len; ++metric_index) {
            primary_found = primary_found ||
                protocol->metric_ids.ptr[metric_index] == protocol->rank.primary_metric;
        }
        if (!primary_found) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (protocol->permutations.permutation_count != 0) {
        uint64_t expected_offsets = 0;
        if (!checked_mul_u64(
                protocol->permutations.permutation_count, matrix->rows, &expected_offsets) ||
            (protocol->permutations.target_offsets.len != 0 &&
                protocol->permutations.target_offsets.len != expected_offsets) ||
            (protocol->permutations.target_offsets.len != 0 &&
                protocol->permutations.target_offsets.ptr == nullptr)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    uint64_t total_rows = 0;
    uint64_t expected_descriptor_offset = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        if (!precision_family_supported(chunk.family) || chunk.arity == 0 ||
            chunk.arity > protocol->max_arity || chunk.combo_count == 0 ||
            chunk.descriptor_count != chunk.combo_count ||
            chunk.combo_row_offset != total_rows ||
            chunk.descriptor_offset != expected_descriptor_offset ||
            chunk.local_chunk_id != chunk_index ||
            (protocol->shape_hint_count != 0 && chunk.shape_hint_index >= protocol->shape_hint_count)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        uint64_t descriptor_span = 0;
        uint64_t descriptor_end = 0;
        if (!checked_mul_u64(chunk.combo_count, chunk.arity, &descriptor_span) ||
            !checked_add_u64(chunk.descriptor_offset, descriptor_span, &descriptor_end) ||
            descriptor_end > protocol->combo_indices.len ||
            chunk.combo_count > std::numeric_limits<uint32_t>::max() ||
            total_rows > std::numeric_limits<uint32_t>::max() - chunk.combo_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint64_t descriptor_index = chunk.descriptor_offset;
             descriptor_index < descriptor_end;
             ++descriptor_index) {
            if (protocol->combo_indices.ptr[descriptor_index] >= matrix->cols) {
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
    uint64_t metric_values = 0;
    if (!checked_mul_u64(total_rows, protocol->metric_ids.len, &metric_values) ||
        !allocation_fits_size_t(metric_values, matrix->dispatch->result_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_precision_launch_protocol(
    const GafimePrecisionLaunchProtocol* protocol,
    const PrecisionHipMatrix* matrix
) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_PRECISION_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->profile != matrix->profile || protocol->base == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : protocol->reserved) {
        if (reserved != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return validate_precision_base_protocol(protocol->base, matrix);
}

int validate_precision_result_sink(
    const GafimeLaunchProtocol* protocol,
    const PrecisionHipMatrix* matrix,
    const PrecisionResultSink* result
) {
    if (result == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // The fp32 v2 entrypoint deliberately reuses the byte-identical ABI-1.0
    // GafimeResultTable layout; the versioned symbol and precision launch
    // wrapper provide its profile identity.  Only the additive f64 table has
    // the ABI-1.1 layout tag.  CUDA, Metal, and the Rust owner follow the same
    // split, so rejecting the ABI-1.0 fp32 table here breaks public ROCm fp32.
    const uint32_t expected_result_abi = matrix->dispatch->result_dtype == GAFIME_DTYPE_F32
        ? GAFIME_ABI_VERSION
        : GAFIME_PRECISION_ABI_VERSION;
    if (result->abi_version != expected_result_abi ||
        result->max_arity < protocol->max_arity ||
        result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = precision_output_row_count(protocol, precision_planned_row_count(protocol));
    if (result->capacity < rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t combo_values = 0;
    uint64_t metric_values = 0;
    if (!checked_mul_u64(rows, result->max_arity, &combo_values) ||
        !checked_mul_u64(rows, result->metric_count, &metric_values) ||
        !allocation_fits_size_t(combo_values, sizeof(uint32_t)) ||
        !allocation_fits_size_t(metric_values, matrix->dispatch->result_bytes) ||
        !allocation_fits_size_t(rows, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows > 0 && (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr ||
        result->candidate_ids == nullptr || result->row_flags == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

template <typename StorageT, typename AccumT>
void precision_compute_column_means_host(
    const StorageT* features,
    uint64_t rows,
    uint32_t cols,
    std::vector<AccumT>* means_out
) {
    means_out->assign(cols, static_cast<AccumT>(0));
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            (*means_out)[col] += static_cast<AccumT>(features[row * cols + col]);
        }
    }
    for (uint32_t col = 0; col < cols; ++col) {
        // Match Core/CUDA's profile-typed division exactly. Multiplication by
        // a pre-rounded reciprocal can move the fp32 centering constant by an
        // ULP, which is enough to change interaction ties and Spearman ranks.
        (*means_out)[col] /= static_cast<AccumT>(rows);
    }
}

template <typename StorageT>
int precision_update_finite_abs_exponent(int current, StorageT value) {
    if (!std::isfinite(value) || value == static_cast<StorageT>(0)) {
        return current;
    }
    // Keep magnitude inspection in the profile's storage type.  The exponent
    // is structural dispatch metadata; it must not widen fp32 values merely
    // to decide whether a guarded covariance kernel is needed.
    return std::max(current, std::ilogb(std::fabs(value)));
}

template <typename StorageT>
int precision_abs_exponent_host(const StorageT* values, uint64_t length) {
    int exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
    for (uint64_t index = 0; index < length; ++index) {
        exponent = precision_update_finite_abs_exponent(exponent, values[index]);
    }
    return exponent;
}

template <typename StorageT>
void precision_build_feature_major_host(
    const StorageT* features,
    uint64_t rows,
    uint32_t cols,
    std::vector<StorageT>* feature_major_out,
    std::vector<int>* feature_abs_exponents_out,
    bool* finite_out
) {
    feature_major_out->assign(static_cast<size_t>(rows) * cols, static_cast<StorageT>(0));
    feature_abs_exponents_out->assign(cols, gafime_gpu_abi::kZeroMagnitudeExponent);
    bool finite = true;
    for (uint32_t col = 0; col < cols; ++col) {
        const uint64_t base = static_cast<uint64_t>(col) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const StorageT value = features[row * cols + col];
            finite = finite && std::isfinite(value);
            (*feature_abs_exponents_out)[col] = precision_update_finite_abs_exponent(
                (*feature_abs_exponents_out)[col], value);
            (*feature_major_out)[static_cast<size_t>(base + row)] = value;
        }
    }
    *finite_out = finite;
}

template <typename StorageT>
bool precision_all_finite_host(const StorageT* values, uint64_t length) {
    for (uint64_t index = 0; index < length; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

// ABI 1.0's diagnostic batch remains the internal structural container used by
// both generations. We conservatively scan every requested descriptor; the
// shared typed owner keeps this optional path correct for every route.
int validate_precision_interaction_diagnostics(
    const PrecisionHipMatrix* matrix,
    GafimeInteractionDiagnosticBatch* diagnostics,
    InteractionDiagnosticScanPlan* scan_plan
) {
    if (matrix == nullptr || diagnostics == nullptr || scan_plan == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (diagnostics->max_arity == 0 ||
        diagnostics->max_arity > kInteractionDiagnosticMaxArity ||
        diagnostics->reserved32 != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : diagnostics->reserved) {
        if (reserved != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    uint64_t expected_combo_index_count = 0;
    if (!checked_mul_u64(
            diagnostics->row_count,
            static_cast<uint64_t>(diagnostics->max_arity),
            &expected_combo_index_count) ||
        diagnostics->combo_index_count != expected_combo_index_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!allocation_fits_size_t(diagnostics->row_count, sizeof(uint64_t)) ||
        !allocation_fits_size_t(diagnostics->row_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(diagnostics->combo_index_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (diagnostics->row_count == 0) {
        return GAFIME_STATUS_OK;
    }
    if (diagnostics->combo_indices == nullptr ||
        diagnostics->overflow_row_counts == nullptr || diagnostics->flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    scan_plan->combo_indices.clear();
    scan_plan->output_rows.clear();
    scan_plan->combo_indices.reserve(static_cast<size_t>(diagnostics->combo_index_count));
    scan_plan->output_rows.reserve(static_cast<size_t>(diagnostics->row_count));
    for (uint64_t row = 0; row < diagnostics->row_count; ++row) {
        const size_t combo_offset = static_cast<size_t>(
            row * static_cast<uint64_t>(diagnostics->max_arity));
        const uint32_t* combo = diagnostics->combo_indices + combo_offset;
        uint32_t arity = 0;
        bool saw_padding = false;
        for (uint32_t idx = 0; idx < diagnostics->max_arity; ++idx) {
            const uint32_t col = combo[idx];
            if (col == UINT32_MAX) {
                saw_padding = true;
                continue;
            }
            if (saw_padding || col >= matrix->cols) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            ++arity;
        }
        if (arity == 0 || arity > kInteractionDiagnosticMaxArity) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        scan_plan->combo_indices.insert(
            scan_plan->combo_indices.end(), combo, combo + diagnostics->max_arity);
        scan_plan->output_rows.push_back(row);
    }
    std::fill_n(
        diagnostics->overflow_row_counts,
        static_cast<size_t>(diagnostics->row_count),
        uint64_t{0});
    std::fill_n(
        diagnostics->flags, static_cast<size_t>(diagnostics->row_count), uint32_t{0});
    return GAFIME_STATUS_OK;
}

template <typename StorageT, typename AccumT>
int precision_interaction_diagnostics_typed(
    PrecisionHipMatrix* matrix,
    GafimeInteractionDiagnosticBatch* diagnostics
) try {
    if (matrix == nullptr || !matrix->content_valid) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    InteractionDiagnosticScanPlan scan_plan;
    int status = validate_precision_interaction_diagnostics(matrix, diagnostics, &scan_plan);
    if (status != GAFIME_STATUS_OK || scan_plan.output_rows.empty()) {
        return status;
    }
    if (scan_plan.output_rows.size() > std::numeric_limits<unsigned int>::max()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t scan_count = static_cast<uint64_t>(scan_plan.output_rows.size());
    uint64_t scan_combo_index_count = 0;
    if (!checked_mul_u64(
            scan_count, static_cast<uint64_t>(diagnostics->max_arity),
            &scan_combo_index_count) ||
        !allocation_fits_size_t(scan_combo_index_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(scan_count, sizeof(uint64_t)) ||
        !allocation_fits_size_t(scan_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    HipTransientBuffer<uint32_t> device_combos(matrix->unified_memory);
    HipTransientBuffer<uint64_t> device_overflow_counts(matrix->unified_memory);
    HipTransientBuffer<uint32_t> device_flags(matrix->unified_memory);
    status = device_combos.allocate(scan_combo_index_count);
    if (status == GAFIME_STATUS_OK) {
        status = device_overflow_counts.allocate(scan_count);
    }
    if (status == GAFIME_STATUS_OK) {
        status = device_flags.allocate(scan_count);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    size_t combo_bytes = 0;
    size_t count_bytes = 0;
    size_t flag_bytes = 0;
    if (!checked_allocation_bytes(scan_combo_index_count, sizeof(uint32_t), &combo_bytes) ||
        !checked_allocation_bytes(scan_count, sizeof(uint64_t), &count_bytes) ||
        !checked_allocation_bytes(scan_count, sizeof(uint32_t), &flag_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = copy_to_hip_storage(
        device_combos.get(), scan_plan.combo_indices.data(), combo_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(gafime_rocm_v1::launch_precision_interaction_diagnostics<StorageT, AccumT>(
        static_cast<const StorageT*>(matrix->features),
        static_cast<const StorageT*>(matrix->target),
        static_cast<const AccumT*>(matrix->column_means),
        device_combos.get(),
        scan_count,
        matrix->rows,
        diagnostics->max_arity,
        device_overflow_counts.get(),
        device_flags.get(),
        nullptr
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(hipDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<uint64_t> overflow_counts(static_cast<size_t>(scan_count), 0);
    std::vector<uint32_t> flags(static_cast<size_t>(scan_count), 0);
    status = copy_from_hip_storage(
        overflow_counts.data(), device_overflow_counts.get(), count_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = copy_from_hip_storage(
            flags.data(), device_flags.get(), flag_bytes, matrix->unified_memory);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    for (size_t idx = 0; idx < scan_plan.output_rows.size(); ++idx) {
        const uint64_t output_row = scan_plan.output_rows[idx];
        diagnostics->overflow_row_counts[output_row] = overflow_counts[idx];
        diagnostics->flags[output_row] = flags[idx];
    }
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

template <typename StorageT>
std::vector<StorageT>& precision_features_host(PrecisionHipMatrix* matrix) {
    if constexpr (std::is_same_v<StorageT, float>) {
        return matrix->features_host_f32;
    } else {
        return matrix->features_host_f64;
    }
}

template <typename StorageT>
std::vector<StorageT>& precision_target_host(PrecisionHipMatrix* matrix) {
    if constexpr (std::is_same_v<StorageT, float>) {
        return matrix->target_host_f32;
    } else {
        return matrix->target_host_f64;
    }
}

template <typename AccumT>
std::vector<AccumT>& precision_column_means_host(PrecisionHipMatrix* matrix) {
    if constexpr (std::is_same_v<AccumT, float>) {
        return matrix->column_means_host_f32;
    } else {
        return matrix->column_means_host_f64;
    }
}

int allocate_precision_storage(void** pointer, size_t bytes, bool managed) {
    if (bytes == 0) {
        *pointer = nullptr;
        return GAFIME_STATUS_OK;
    }
    if (managed) {
        return hip_status(hipMallocManaged(pointer, bytes));
    }
    return hip_status(hipMalloc(pointer, bytes));
}

int ensure_precision_device_capacity(
    void** pointer,
    uint64_t* capacity,
    uint64_t required,
    size_t element_size,
    bool managed
) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    if (!allocation_fits_size_t(required, element_size)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / element_size;
    const uint64_t doubled = *capacity > max_capacity / 2 ? max_capacity : *capacity * 2;
    const uint64_t next_capacity = std::max(required, *capacity == 0 ? required : doubled);
    void* next = nullptr;
    const int status = allocate_precision_storage(
        &next, static_cast<size_t>(next_capacity) * element_size, managed);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    static_cast<void>(hipFree(*pointer));
    *pointer = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_refresh_stats(PrecisionHipMatrix* matrix) {
    const int target_status = hip_status(gafime_rocm_v1::launch_precision_target_stats<StorageT, AccumT>(
        static_cast<const StorageT*>(matrix->target), matrix->rows,
        static_cast<gafime_rocm_v1::PrecisionTargetStatsDevice<AccumT>*>(matrix->target_stats),
        nullptr
    ));
    if (target_status != GAFIME_STATUS_OK) {
        return target_status;
    }
    const int feature_status = hip_status(gafime_rocm_v1::launch_precision_unary_feature_stats<StorageT, AccumT>(
        static_cast<const StorageT*>(matrix->features), matrix->rows, matrix->cols,
        static_cast<gafime_rocm_v1::PrecisionUnaryFeatureStatsDevice<AccumT>*>(matrix->feature_stats),
        nullptr
    ));
    if (feature_status != GAFIME_STATUS_OK) {
        return feature_status;
    }
    const int sync_status = hip_status(hipDeviceSynchronize());
    if (sync_status == GAFIME_STATUS_OK) {
        matrix->target_stats_profile = matrix->profile;
        matrix->feature_stats_profile = matrix->profile;
    }
    return sync_status;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_upload_typed(
    PrecisionHipMatrix* matrix,
    const void* features_host,
    const void* target_host,
    uint64_t rows,
    uint32_t cols
) try {
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr ||
        rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const auto* typed_features = static_cast<const StorageT*>(features_host);
    const auto* typed_target = static_cast<const StorageT*>(target_host);
    uint64_t feature_count = 0;
    size_t feature_bytes = 0;
    size_t target_bytes = 0;
    size_t mean_bytes = 0;
    if (!checked_mul_u64(rows, cols, &feature_count) ||
        !checked_allocation_bytes(feature_count, sizeof(StorageT), &feature_bytes) ||
        !checked_allocation_bytes(rows, sizeof(StorageT), &target_bytes) ||
        !checked_allocation_bytes(cols, sizeof(AccumT), &mean_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    auto& resident_features = precision_features_host<StorageT>(matrix);
    auto& resident_target = precision_target_host<StorageT>(matrix);
    auto& resident_means = precision_column_means_host<AccumT>(matrix);
    std::vector<int> feature_abs_exponents;
    bool features_are_finite = true;
    if constexpr (std::is_same_v<StorageT, float> && std::is_same_v<AccumT, float>) {
        if (matrix->legacy_abi10) {
            // Frozen ABI 1.0 accumulated host column means in binary64 before
            // storing binary32.  Keep that one preprocessing difference in the
            // adapter without giving it a separate resident engine.
            compute_column_means_host(typed_features, rows, cols, resident_means);
        } else {
            precision_compute_column_means_host<StorageT, AccumT>(
                typed_features, rows, cols, &resident_means);
        }
    } else {
        precision_compute_column_means_host<StorageT, AccumT>(
            typed_features, rows, cols, &resident_means);
    }
    precision_build_feature_major_host(
        typed_features, rows, cols, &resident_features, &feature_abs_exponents,
        &features_are_finite);
    resident_target.assign(typed_target, typed_target + rows);
    const bool target_is_finite = precision_all_finite_host(typed_target, rows);
    const int target_abs_exponent = precision_abs_exponent_host(typed_target, rows);

    matrix->content_valid = false;
    precision_destroy_graph_cache(matrix);
    precision_invalidate_descriptor_cache(matrix);
    matrix->spearman_target_ranks_ready = false;
    matrix->spearman_target_ranks_profile = 0;
    matrix->target_stats_profile = 0;
    matrix->feature_stats_profile = 0;
    status = copy_to_hip_storage(
        matrix->features, resident_features.data(), feature_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(matrix->target, resident_target.data(), target_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(
            matrix->column_means, resident_means.data(), mean_bytes, matrix->unified_memory);
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
    status = precision_refresh_stats<StorageT, AccumT, ResultT>(matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    matrix->features_are_finite = features_are_finite;
    matrix->target_is_finite = target_is_finite;
    matrix->feature_abs_exponents.swap(feature_abs_exponents);
    matrix->target_abs_exponent = target_abs_exponent;
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_update_target_typed(
    PrecisionHipMatrix* matrix,
    const void* target_host,
    uint64_t rows
) try {
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows || !matrix->content_valid) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const auto* typed_target = static_cast<const StorageT*>(target_host);
    size_t target_bytes = 0;
    if (!checked_allocation_bytes(rows, sizeof(StorageT), &target_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    auto& resident_target = precision_target_host<StorageT>(matrix);
    resident_target.assign(typed_target, typed_target + rows);
    const bool target_is_finite = precision_all_finite_host(typed_target, rows);
    const int target_abs_exponent = precision_abs_exponent_host(typed_target, rows);
    matrix->content_valid = false;
    precision_destroy_graph_cache(matrix);
    precision_invalidate_descriptor_cache(matrix);
    matrix->spearman_target_ranks_ready = false;
    matrix->spearman_target_ranks_profile = 0;
    matrix->target_stats_profile = 0;
    status = copy_to_hip_storage(
        matrix->target, resident_target.data(), target_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    status = precision_refresh_stats<StorageT, AccumT, ResultT>(matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
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

bool precision_has_covariance_metric(const GafimeLaunchProtocol* protocol) {
    for (uint64_t metric_index = 0; metric_index < protocol->metric_ids.len; ++metric_index) {
        const uint32_t metric = protocol->metric_ids.ptr[metric_index];
        if (metric == GAFIME_METRIC_PEARSON || metric == GAFIME_METRIC_R2) {
            return true;
        }
    }
    return false;
}

std::vector<uint8_t> precision_covariance_modes_for_protocol(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    std::vector<uint8_t> modes(protocol->chunk_count, 0);
    if (!precision_has_covariance_metric(protocol)) {
        return modes;
    }
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const uint64_t descriptor =
                chunk.descriptor_offset + row * static_cast<uint64_t>(chunk.arity);
            const int feature_exponent = gafime_gpu_abi::interaction_abs_exponent(
                matrix->feature_abs_exponents.data(),
                protocol->combo_indices.ptr + descriptor,
                chunk.arity);
            if (gafime_gpu_abi::covariance_requires_scaled_path(
                    matrix->rows, feature_exponent, matrix->target_abs_exponent)) {
                modes[chunk_index] = 1;
                break;
            }
        }
    }
    return modes;
}

int precision_launch_legacy_score_kernels(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    hipStream_t stream
) {
    if (matrix == nullptr || !matrix->legacy_abi10 ||
        matrix->covariance_modes.size() != protocol->chunk_count) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    const auto* features = static_cast<const float*>(matrix->features);
    const auto* target = static_cast<const float*>(matrix->target);
    const auto* means = static_cast<const float*>(matrix->column_means);
    const auto* target_stats = static_cast<const gafime_rocm_v1::TargetStatsDevice*>(
        matrix->target_stats);
    const auto* feature_stats =
        static_cast<const gafime_rocm_v1::UnaryFeatureStatsDevice*>(matrix->feature_stats);
    const auto* target_ranks = matrix->spearman_target_ranks_ready
        ? static_cast<const double*>(matrix->spearman_target_ranks)
        : nullptr;
    auto* values = static_cast<float*>(matrix->metric_values);
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        float* output = values + metric_row_offset * metric_count;
        int status = GAFIME_STATUS_OK;
        if (precision_has_covariance_metric(protocol)) {
            status = hip_status(gafime_rocm_v1::launch_continuous_chunk(
                features,
                target,
                means,
                target_stats,
                feature_stats,
                matrix->combo_indices,
                matrix->rows,
                matrix->cols,
                chunk.arity,
                chunk.descriptor_offset,
                chunk.combo_count,
                matrix->features_are_finite ? 1u : 0u,
                matrix->target_is_finite ? 1u : 0u,
                matrix->covariance_modes[chunk_index],
                matrix->metric_ids,
                metric_count,
                output,
                stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_MUTUAL_INFO) {
                continue;
            }
            status = hip_status(gafime_rocm_v1::launch_mutual_info_chunk(
                features,
                target,
                means,
                matrix->combo_indices,
                matrix->rows,
                matrix->cols,
                chunk.arity,
                chunk.descriptor_offset,
                chunk.combo_count,
                metric_count,
                metric_index,
                precision_mi_bins_for_chunk(protocol, chunk),
                output,
                stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_SPEARMAN) {
                continue;
            }
            status = hip_status(gafime_rocm_v1::launch_spearman_chunk(
                features,
                target,
                means,
                target_ranks,
                matrix->combo_indices,
                matrix->rows,
                matrix->cols,
                chunk.arity,
                chunk.descriptor_offset,
                chunk.combo_count,
                metric_count,
                metric_index,
                output,
                stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        metric_row_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_launch_score_kernels(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    hipStream_t stream
) {
    uint64_t metric_row_offset = 0;
    const auto* features = static_cast<const StorageT*>(matrix->features);
    const auto* target = static_cast<const StorageT*>(matrix->target);
    const auto* means = static_cast<const AccumT*>(matrix->column_means);
    const auto* target_ranks =
        matrix->spearman_target_ranks_ready &&
            matrix->spearman_target_ranks_profile == matrix->profile &&
            precision_spearman_target_rank_cache_eligible(matrix, protocol)
        ? static_cast<const AccumT*>(matrix->spearman_target_ranks)
        : nullptr;
    auto* values = static_cast<ResultT*>(matrix->metric_values);
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    if (matrix->covariance_modes.size() != protocol->chunk_count) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        ResultT* output = values + metric_row_offset * metric_count;
        int status = GAFIME_STATUS_OK;
        if (precision_has_covariance_metric(protocol)) {
            status = hip_status(gafime_rocm_v1::launch_precision_continuous<StorageT, AccumT, ResultT>(
                features, target, means, matrix->combo_indices, matrix->rows, chunk.arity,
                chunk.descriptor_offset, chunk.combo_count, matrix->covariance_modes[chunk_index],
                matrix->metric_ids, metric_count, output, stream
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_MUTUAL_INFO) {
                continue;
            }
            status = hip_status(gafime_rocm_v1::launch_precision_mutual_info<StorageT, AccumT, ResultT>(
                features, target, means, matrix->combo_indices, matrix->rows, chunk.arity,
                chunk.descriptor_offset, chunk.combo_count, metric_count, metric_index,
                precision_mi_bins_for_chunk(protocol, chunk), output, 0u, stream
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_SPEARMAN) {
                continue;
            }
            if (chunk.arity == 1u && target_ranks != nullptr) {
                status = hip_status(
                    gafime_rocm_v1::launch_precision_spearman_unary_cached_target_ranks<
                        StorageT, AccumT, ResultT>(
                            features, target, target_ranks, matrix->combo_indices,
                            matrix->rows, chunk.descriptor_offset, chunk.combo_count,
                            metric_count, metric_index, output, stream));
            } else {
                status = hip_status(gafime_rocm_v1::launch_precision_spearman<StorageT, AccumT, ResultT>(
                    features, target, means, matrix->combo_indices, matrix->rows, chunk.arity,
                    chunk.descriptor_offset, chunk.combo_count, metric_count, metric_index,
                    output, stream
                ));
            }
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }
        metric_row_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

uint64_t precision_metric_signature(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(matrix->profile);
    mix(matrix->legacy_abi10 ? 1u : 0u);
    for (uint64_t metric_index = 0; metric_index < protocol->metric_ids.len; ++metric_index) {
        mix(protocol->metric_ids.ptr[metric_index]);
    }
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        mix(chunk.metric_mask);
        mix(chunk.family);
        mix(precision_mi_bins_for_chunk(protocol, chunk));
        mix(chunk_index < matrix->covariance_modes.size()
            ? matrix->covariance_modes[chunk_index]
            : std::numeric_limits<uint32_t>::max());
    }
    return hash;
}

int precision_prepare_descriptors(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    if (precision_descriptors_resident(matrix, protocol)) {
        return GAFIME_STATUS_OK;
    }
    std::vector<uint8_t> next_covariance_modes =
        precision_covariance_modes_for_protocol(matrix, protocol);
    size_t combo_bytes = 0;
    size_t metric_id_bytes = 0;
    if (!checked_allocation_bytes(protocol->combo_indices.len, sizeof(uint32_t), &combo_bytes) ||
        !checked_allocation_bytes(protocol->metric_ids.len, sizeof(uint32_t), &metric_id_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    HipDeviceBufferReservation<uint32_t> combo_reservation(
        matrix->combo_indices, matrix->combo_capacity, matrix->unified_memory);
    HipDeviceBufferReservation<uint32_t> metric_reservation(
        matrix->metric_ids, matrix->metric_id_capacity, matrix->unified_memory);
    int status = combo_reservation.reserve(protocol->combo_indices.len);
    if (status == GAFIME_STATUS_OK) {
        status = metric_reservation.reserve(protocol->metric_ids.len);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    // Clear the profile-aware cache key before an in-place upload becomes
    // observable.  A failed upload can therefore never reuse a stale profile.
    precision_invalidate_descriptor_cache(matrix);
    status = copy_to_hip_storage(
        combo_reservation.get(), protocol->combo_indices.ptr, combo_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = copy_to_hip_storage(
            metric_reservation.get(), protocol->metric_ids.ptr, metric_id_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK && matrix->unified_memory) {
        status = hip_status(hipDeviceSynchronize());
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (combo_reservation.replaces_live_buffer() || metric_reservation.replaces_live_buffer()) {
        precision_destroy_graph_cache(matrix);
    }
    combo_reservation.commit(&matrix->combo_indices, &matrix->combo_capacity);
    metric_reservation.commit(&matrix->metric_ids, &matrix->metric_id_capacity);
    matrix->covariance_modes.swap(next_covariance_modes);
    // Even an uncached upload belongs to this profile and covariance-mode
    // identity. A zero generation means it cannot be reused, not that its
    // profile-specialized ownership is ambiguous.
    matrix->descriptor_profile = matrix->profile;
    const bool cacheable =
        (protocol->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0 &&
        protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] != 0;
    if (cacheable) {
        matrix->descriptor_generation =
            protocol->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
        matrix->descriptor_combo_len = protocol->combo_indices.len;
        matrix->descriptor_metric_id_len = protocol->metric_ids.len;
    }
    return GAFIME_STATUS_OK;
}

bool precision_graph_requested(const GafimeLaunchProtocol* protocol) {
    return (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0 &&
        protocol->rank.top_k == 0 && protocol->permutations.permutation_count == 0;
}

bool precision_graph_shape_matches(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) {
    if (!matrix->graph_valid || matrix->graph_exec == nullptr ||
        matrix->graph_profile != matrix->profile ||
        matrix->graph_chunk_count != protocol->chunk_count ||
        matrix->graph_chunk_shapes.size() != protocol->chunk_count ||
        matrix->covariance_modes.size() != protocol->chunk_count ||
        matrix->graph_combo_len != protocol->combo_indices.len ||
        matrix->graph_metric_id_len != protocol->metric_ids.len ||
        matrix->graph_metric_value_count != metric_value_count ||
        matrix->graph_combo_ptr != reinterpret_cast<uintptr_t>(matrix->combo_indices) ||
        matrix->graph_metric_ids_ptr != reinterpret_cast<uintptr_t>(matrix->metric_ids) ||
        matrix->graph_metric_values_ptr != reinterpret_cast<uintptr_t>(matrix->metric_values) ||
        matrix->graph_metric_signature != precision_metric_signature(matrix, protocol)) {
        return false;
    }
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        const PrecisionGraphChunkShape& shape = matrix->graph_chunk_shapes[chunk_index];
        if (shape.arity != chunk.arity || shape.family != chunk.family ||
            shape.scaled_covariance != matrix->covariance_modes[chunk_index] ||
            shape.descriptor_offset != chunk.descriptor_offset ||
            shape.combo_count != chunk.combo_count ||
            shape.mi_bins != precision_mi_bins_for_chunk(protocol, chunk)) {
            return false;
        }
    }
    return true;
}

int precision_store_graph_shape(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) try {
    std::vector<PrecisionGraphChunkShape> shapes;
    shapes.reserve(protocol->chunk_count);
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        shapes.push_back(PrecisionGraphChunkShape{
            chunk.arity,
            chunk.family,
            matrix->covariance_modes[chunk_index],
            chunk.descriptor_offset,
            chunk.combo_count,
            precision_mi_bins_for_chunk(protocol, chunk),
        });
    }
    matrix->graph_profile = matrix->profile;
    matrix->graph_chunk_count = protocol->chunk_count;
    matrix->graph_combo_len = protocol->combo_indices.len;
    matrix->graph_metric_id_len = protocol->metric_ids.len;
    matrix->graph_metric_value_count = metric_value_count;
    matrix->graph_combo_ptr = reinterpret_cast<uintptr_t>(matrix->combo_indices);
    matrix->graph_metric_ids_ptr = reinterpret_cast<uintptr_t>(matrix->metric_ids);
    matrix->graph_metric_values_ptr = reinterpret_cast<uintptr_t>(matrix->metric_values);
    matrix->graph_metric_signature = precision_metric_signature(matrix, protocol);
    matrix->graph_chunk_shapes.swap(shapes);
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_execute_score_kernels(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool* graph_replayed
) {
    *graph_replayed = false;
    if (!precision_graph_requested(protocol)) {
        int status = matrix->legacy_abi10
            ? precision_launch_legacy_score_kernels(matrix, protocol, nullptr)
            : precision_launch_score_kernels<StorageT, AccumT, ResultT>(
                matrix, protocol, nullptr);
        if (status == GAFIME_STATUS_OK) {
            status = hip_status(hipDeviceSynchronize());
        }
        return status;
    }
    if (matrix->graph_stream == nullptr) {
        const int stream_status = hip_status(hipStreamCreateWithFlags(
            &matrix->graph_stream, hipStreamNonBlocking));
        if (stream_status != GAFIME_STATUS_OK) {
            return stream_status;
        }
    }
    if (!precision_graph_shape_matches(matrix, protocol, metric_value_count)) {
        precision_destroy_graph_cache(matrix);
        int status = hip_status(hipStreamBeginCapture(
            matrix->graph_stream, hipStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = matrix->legacy_abi10
            ? precision_launch_legacy_score_kernels(
                matrix, protocol, matrix->graph_stream)
            : precision_launch_score_kernels<StorageT, AccumT, ResultT>(
                matrix, protocol, matrix->graph_stream);
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(hipStreamEndCapture(matrix->graph_stream, &matrix->graph));
            precision_destroy_graph_cache(matrix);
            return status;
        }
        hipGraph_t graph = nullptr;
        status = hip_status(hipStreamEndCapture(matrix->graph_stream, &graph));
        if (status != GAFIME_STATUS_OK) {
            precision_destroy_graph_cache(matrix);
            return status;
        }
        hipGraphExec_t graph_exec = nullptr;
        status = hip_status(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(hipGraphDestroy(graph));
            precision_destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph = graph;
        matrix->graph_exec = graph_exec;
        status = precision_store_graph_shape(matrix, protocol, metric_value_count);
        if (status != GAFIME_STATUS_OK) {
            precision_destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }
    int status = hip_status(hipGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    } else {
        precision_destroy_graph_cache(matrix);
    }
    return status;
}

template <typename ResultT>
void precision_write_result_row(
    const GafimeLaunchProtocol* protocol,
    PrecisionResultSink* result,
    const std::vector<ResultT>& metric_values,
    uint64_t destination_row,
    uint64_t source_metric_row,
    uint64_t combo_base,
    uint32_t arity,
    uint32_t family,
    uint32_t rank,
    uint64_t candidate_id
) {
    for (uint32_t slot = 0; slot < result->max_arity; ++slot) {
        result->combo_indices[destination_row * result->max_arity + slot] =
            slot < arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
    }
    auto* output_values = static_cast<ResultT*>(result->metric_values);
    for (uint32_t metric_index = 0; metric_index < result->metric_count; ++metric_index) {
        output_values[destination_row * result->metric_count + metric_index] =
            metric_index < protocol->metric_ids.len
            ? metric_values[source_metric_row * protocol->metric_ids.len + metric_index]
            : static_cast<ResultT>(0);
    }
    result->ranks[destination_row] = rank;
    result->families[destination_row] = family;
    result->candidate_ids[destination_row] = candidate_id;
    result->row_flags[destination_row] = 0;
}

template <typename ResultT>
int precision_write_result_rows(
    const GafimeLaunchProtocol* protocol,
    PrecisionResultSink* result,
    const std::vector<ResultT>& metric_values,
    const std::vector<uint32_t>* selected_indices
) {
    if (selected_indices != nullptr) {
        for (uint64_t output_row = 0; output_row < selected_indices->size(); ++output_row) {
            const uint64_t candidate_id = (*selected_indices)[output_row];
            uint64_t chunk_base = 0;
            const GafimeArityChunk* selected_chunk = nullptr;
            uint64_t local_row = 0;
            for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
                const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
                if (candidate_id < chunk_base + chunk.combo_count) {
                    selected_chunk = &chunk;
                    local_row = candidate_id - chunk_base;
                    break;
                }
                chunk_base += chunk.combo_count;
            }
            if (selected_chunk == nullptr) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            const uint64_t combo_base =
                selected_chunk->descriptor_offset + local_row * selected_chunk->arity;
            precision_write_result_row(
                protocol, result, metric_values, output_row, output_row, combo_base,
                selected_chunk->arity, selected_chunk->family,
                static_cast<uint32_t>(output_row), candidate_id);
        }
        *result->row_count = selected_indices->size();
        return GAFIME_STATUS_OK;
    }
    uint64_t output_row = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const uint64_t combo_base = chunk.descriptor_offset + row * chunk.arity;
            precision_write_result_row(
                protocol, result, metric_values, output_row, output_row, combo_base,
                chunk.arity, chunk.family, static_cast<uint32_t>(output_row), output_row);
            ++output_row;
        }
    }
    *result->row_count = output_row;
    return GAFIME_STATUS_OK;
}

template <typename StorageT, typename AccumT, typename ResultT>
int precision_execute_typed(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    PrecisionResultSink* result
) try {
    if (matrix == nullptr || !matrix->content_valid ||
        matrix->target_stats_profile != matrix->profile ||
        matrix->feature_stats_profile != matrix->profile) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t total_rows = precision_planned_row_count(protocol);
    const uint64_t output_rows = precision_output_row_count(protocol, total_rows);
    if (total_rows == 0) {
        *result->row_count = 0;
        return GAFIME_STATUS_OK;
    }
    uint64_t metric_value_count = 0;
    if (!checked_mul_u64(total_rows, protocol->metric_ids.len, &metric_value_count)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    void* previous_metric_values = matrix->metric_values;
    status = ensure_precision_device_capacity(
        &matrix->metric_values, &matrix->metric_value_capacity, metric_value_count,
        sizeof(ResultT), matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (previous_metric_values != matrix->metric_values) {
        precision_destroy_graph_cache(matrix);
    }
    status = precision_prepare_descriptors(matrix, protocol);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    // Build the target-rank cache before optional graph capture.  Cache
    // construction is a lifecycle operation, not a host branch inside the
    // captured or replayed hot-loop dispatch.
    if (matrix->legacy_abi10) {
        status = precision_prepare_spearman_target_rank_cache<float, double>(
            matrix, protocol);
    } else {
        status = precision_prepare_spearman_target_rank_cache<StorageT, AccumT>(
            matrix, protocol);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    bool graph_replayed = false;
    status = precision_execute_score_kernels<StorageT, AccumT, ResultT>(
        matrix, protocol, metric_value_count, &graph_replayed);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    size_t metric_value_bytes = 0;
    if (!checked_allocation_bytes(metric_value_count, sizeof(ResultT), &metric_value_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (protocol->rank.top_k == 0) {
        std::vector<ResultT> host_values(static_cast<size_t>(metric_value_count));
        status = copy_from_hip_storage(
            host_values.data(), matrix->metric_values, metric_value_bytes, matrix->unified_memory);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = precision_write_result_rows(protocol, result, host_values, nullptr);
        if (status == GAFIME_STATUS_OK) {
            *result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
            if (graph_replayed) {
                *result->flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
            }
        }
        return status;
    }
    status = ensure_device_capacity(
        &matrix->selected_indices, &matrix->selected_index_capacity, output_rows, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint32_t partial_blocks = topk_partial_block_count(total_rows, output_rows);
    uint64_t partial_items = 0;
    if (!checked_mul_u64(partial_blocks, output_rows, &partial_items)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_precision_device_capacity(
        &matrix->topk_partial_scores, &matrix->topk_partial_score_capacity, partial_items,
        sizeof(ResultT), matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &matrix->topk_partial_indices, &matrix->topk_partial_index_capacity,
            partial_items, matrix->unified_memory);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(gafime_rocm_v1::launch_precision_select_topk<ResultT>(
        static_cast<const ResultT*>(matrix->metric_values), total_rows,
        static_cast<uint32_t>(protocol->metric_ids.len), precision_primary_metric_index(protocol),
        static_cast<uint32_t>(output_rows), protocol->rank.descending, matrix->selected_indices,
        static_cast<ResultT*>(matrix->topk_partial_scores), matrix->topk_partial_indices,
        partial_blocks, nullptr
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    size_t selected_index_bytes = 0;
    if (!checked_allocation_bytes(output_rows, sizeof(uint32_t), &selected_index_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    std::vector<uint32_t> selected_indices(static_cast<size_t>(output_rows), UINT32_MAX);
    status = copy_from_hip_storage(
        selected_indices.data(), matrix->selected_indices, selected_index_bytes, matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    uint64_t selected_count = 0;
    while (selected_count < output_rows && selected_indices[selected_count] != UINT32_MAX) {
        ++selected_count;
    }
    selected_indices.resize(static_cast<size_t>(selected_count));
    if (selected_count == 0) {
        *result->row_count = 0;
        *result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
        return GAFIME_STATUS_OK;
    }
    uint64_t selected_metric_count = 0;
    size_t selected_metric_bytes = 0;
    if (!checked_mul_u64(selected_count, protocol->metric_ids.len, &selected_metric_count) ||
        !checked_allocation_bytes(selected_metric_count, sizeof(ResultT), &selected_metric_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_precision_device_capacity(
        &matrix->selected_metric_values, &matrix->selected_metric_value_capacity,
        selected_metric_count, sizeof(ResultT), matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(gafime_rocm_v1::launch_precision_copy_selected_metric_rows<ResultT>(
        static_cast<const ResultT*>(matrix->metric_values), matrix->selected_indices,
        selected_count, static_cast<uint32_t>(protocol->metric_ids.len),
        static_cast<ResultT*>(matrix->selected_metric_values), nullptr
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<ResultT> selected_values(static_cast<size_t>(selected_metric_count));
    status = copy_from_hip_storage(
        selected_values.data(), matrix->selected_metric_values, selected_metric_bytes,
        matrix->unified_memory);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = precision_write_result_rows(protocol, result, selected_values, &selected_indices);
    if (status == GAFIME_STATUS_OK) {
        *result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    }
    return status;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

const PrecisionDispatchTable kPrecisionFp32Dispatch{
    GAFIME_PRECISION_FP32,
    GAFIME_DTYPE_F32,
    GAFIME_DTYPE_F32,
    static_cast<uint32_t>(sizeof(float)),
    static_cast<uint32_t>(sizeof(float)),
    static_cast<uint32_t>(sizeof(float)),
    &precision_upload_typed<float, float, float>,
    &precision_update_target_typed<float, float, float>,
    &precision_execute_typed<float, float, float>,
    &precision_interaction_diagnostics_typed<float, float>,
};

const PrecisionDispatchTable kPrecisionMixedDispatch{
    GAFIME_PRECISION_MIXED,
    GAFIME_DTYPE_F32,
    GAFIME_DTYPE_F64,
    static_cast<uint32_t>(sizeof(float)),
    static_cast<uint32_t>(sizeof(double)),
    static_cast<uint32_t>(sizeof(double)),
    &precision_upload_typed<float, double, double>,
    &precision_update_target_typed<float, double, double>,
    &precision_execute_typed<float, double, double>,
    &precision_interaction_diagnostics_typed<float, double>,
};

const PrecisionDispatchTable kPrecisionFp64Dispatch{
    GAFIME_PRECISION_FP64,
    GAFIME_DTYPE_F64,
    GAFIME_DTYPE_F64,
    static_cast<uint32_t>(sizeof(double)),
    static_cast<uint32_t>(sizeof(double)),
    static_cast<uint32_t>(sizeof(double)),
    &precision_upload_typed<double, double, double>,
    &precision_update_target_typed<double, double, double>,
    &precision_execute_typed<double, double, double>,
    &precision_interaction_diagnostics_typed<double, double>,
};

const PrecisionDispatchTable* precision_dispatch_for_profile(uint32_t profile) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        return &kPrecisionFp32Dispatch;
    case GAFIME_PRECISION_MIXED:
        return &kPrecisionMixedDispatch;
    case GAFIME_PRECISION_FP64:
        return &kPrecisionFp64Dispatch;
    default:
        return nullptr;
    }
}

PrecisionHipMatrix* precision_matrix_from_handle(GafimeGpuMatrix handle) {
    auto* header = static_cast<HipMatrixHeader*>(handle);
    if (header == nullptr || header->magic != kPrecisionHipMatrixMagic) {
        return nullptr;
    }
    return static_cast<PrecisionHipMatrix*>(handle);
}

void precision_matrix_free(PrecisionHipMatrix* matrix) {
    if (matrix == nullptr) {
        return;
    }
    static_cast<void>(hipSetDevice(static_cast<int>(matrix->device_id)));
    precision_destroy_graph_cache(matrix);
    if (matrix->graph_stream != nullptr) {
        static_cast<void>(hipStreamDestroy(matrix->graph_stream));
    }
    static_cast<void>(hipFree(matrix->features));
    static_cast<void>(hipFree(matrix->target));
    static_cast<void>(hipFree(matrix->column_means));
    static_cast<void>(hipFree(matrix->target_stats));
    static_cast<void>(hipFree(matrix->feature_stats));
    static_cast<void>(hipFree(matrix->spearman_target_ranks));
    static_cast<void>(hipFree(matrix->metric_values));
    static_cast<void>(hipFree(matrix->combo_indices));
    static_cast<void>(hipFree(matrix->metric_ids));
    static_cast<void>(hipFree(matrix->selected_indices));
    static_cast<void>(hipFree(matrix->selected_metric_values));
    static_cast<void>(hipFree(matrix->topk_partial_scores));
    static_cast<void>(hipFree(matrix->topk_partial_indices));
    matrix->magic = 0;
    delete matrix;
}

int precision_matrix_alloc(
    uint32_t device_id,
    const GafimePrecisionMatrixDesc* desc,
    GafimeGpuMatrix* matrix_out,
    bool legacy_abi10
) try {
    if (matrix_out == nullptr || device_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    int status = validate_precision_matrix_desc(desc);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const PrecisionDispatchTable* dispatch = precision_dispatch_for_profile(desc->profile);
    if (dispatch == nullptr || dispatch->storage_dtype != desc->dtype) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
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
    auto* matrix = new PrecisionHipMatrix{};
    matrix->magic = kPrecisionHipMatrixMagic;
    matrix->device_id = device_id;
    matrix->device_flags = rocm_device_flags(props);
    matrix->arch_class = rocm_arch_class(props);
    matrix->profile = desc->profile;
    matrix->legacy_abi10 = legacy_abi10;
    matrix->unified_memory = rocm_use_managed_memory(props);
    matrix->rows = desc->rows;
    matrix->cols = desc->cols;
    matrix->content_valid = false;
    matrix->features_are_finite = true;
    matrix->target_is_finite = true;
    matrix->target_abs_exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
    matrix->dispatch = dispatch;
    matrix->spearman_target_ranks = nullptr;
    matrix->spearman_target_rank_capacity = 0;
    matrix->spearman_target_ranks_profile = 0;
    matrix->spearman_target_ranks_ready = false;
    uint64_t feature_count = 0;
    size_t feature_bytes = 0;
    size_t target_bytes = 0;
    size_t mean_bytes = 0;
    size_t target_stats_bytes = 0;
    size_t feature_stats_bytes = 0;
    size_t spearman_target_rank_bytes = 0;
    if (!checked_mul_u64(matrix->rows, matrix->cols, &feature_count) ||
        !checked_allocation_bytes(feature_count, dispatch->storage_bytes, &feature_bytes) ||
        !checked_allocation_bytes(matrix->rows, dispatch->storage_bytes, &target_bytes) ||
        !checked_allocation_bytes(matrix->cols, dispatch->accumulator_bytes, &mean_bytes) ||
        !checked_allocation_bytes(1, dispatch->accumulator_bytes * 2 + sizeof(uint64_t), &target_stats_bytes) ||
        !checked_allocation_bytes(
            matrix->cols, dispatch->accumulator_bytes * 2 + sizeof(uint64_t), &feature_stats_bytes)) {
        delete matrix;
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    // The typed statistics structs contain two accumulator fields plus two u32
    // fields.  Compute their exact size from the selected profile rather than
    // assuming the legacy float layout.
    if (dispatch->accumulator_bytes == sizeof(float)) {
        target_stats_bytes = sizeof(gafime_rocm_v1::PrecisionTargetStatsDevice<float>);
        feature_stats_bytes = matrix->cols * sizeof(gafime_rocm_v1::PrecisionUnaryFeatureStatsDevice<float>);
    } else {
        target_stats_bytes = sizeof(gafime_rocm_v1::PrecisionTargetStatsDevice<double>);
        feature_stats_bytes = matrix->cols * sizeof(gafime_rocm_v1::PrecisionUnaryFeatureStatsDevice<double>);
    }
    status = allocate_precision_storage(&matrix->features, feature_bytes, matrix->unified_memory);
    if (status == GAFIME_STATUS_OK) {
        status = allocate_precision_storage(&matrix->target, target_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK) {
        status = allocate_precision_storage(&matrix->column_means, mean_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK) {
        status = allocate_precision_storage(&matrix->target_stats, target_stats_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK) {
        status = allocate_precision_storage(&matrix->feature_stats, feature_stats_bytes, matrix->unified_memory);
    }
    if (status == GAFIME_STATUS_OK &&
        matrix->rows >= gafime_rocm_v1::kSpearmanTargetRankCacheMinSamples &&
        matrix->rows <= gafime_rocm_v1::kSpearmanTargetRankCacheMaxSamples &&
        checked_allocation_bytes(
            matrix->rows,
            legacy_abi10 ? sizeof(double) : dispatch->accumulator_bytes,
            &spearman_target_rank_bytes)) {
        // The rank cache is an optional performance optimization.  If the
        // device cannot reserve it, the canonical route remains correct and
        // falls back to the ordinary unary Spearman kernel.
        const int rank_status = allocate_precision_storage(
            &matrix->spearman_target_ranks,
            spearman_target_rank_bytes,
            matrix->unified_memory);
        if (rank_status == GAFIME_STATUS_OK) {
            matrix->spearman_target_rank_capacity = matrix->rows;
        } else {
            matrix->spearman_target_ranks = nullptr;
            matrix->spearman_target_rank_capacity = 0;
        }
    }
    if (status != GAFIME_STATUS_OK) {
        precision_matrix_free(matrix);
        return status;
    }
    *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

uint64_t precision_matrix_resident_device_bytes(const PrecisionHipMatrix* matrix) {
    using gafime_gpu_abi::allocation_bytes;
    using gafime_gpu_abi::saturating_add_u64;
    using gafime_gpu_abi::saturating_mul_u64;
    uint64_t bytes = 0;
    const auto add = [&bytes](uint64_t count, size_t element_size) {
        bytes = saturating_add_u64(bytes, allocation_bytes(count, element_size));
    };
    const PrecisionDispatchTable* dispatch = matrix->dispatch;
    add(saturating_mul_u64(matrix->rows, matrix->cols), dispatch->storage_bytes);
    add(matrix->rows, dispatch->storage_bytes);
    add(matrix->cols, dispatch->accumulator_bytes);
    if (dispatch->accumulator_bytes == sizeof(float)) {
        add(1, sizeof(gafime_rocm_v1::PrecisionTargetStatsDevice<float>));
        add(matrix->cols, sizeof(gafime_rocm_v1::PrecisionUnaryFeatureStatsDevice<float>));
    } else {
        add(1, sizeof(gafime_rocm_v1::PrecisionTargetStatsDevice<double>));
        add(matrix->cols, sizeof(gafime_rocm_v1::PrecisionUnaryFeatureStatsDevice<double>));
    }
    add(
        matrix->spearman_target_rank_capacity,
        matrix->legacy_abi10 ? sizeof(double) : dispatch->accumulator_bytes);
    add(matrix->combo_capacity, sizeof(uint32_t));
    add(matrix->metric_id_capacity, sizeof(uint32_t));
    add(matrix->metric_value_capacity, dispatch->result_bytes);
    add(matrix->selected_index_capacity, sizeof(uint32_t));
    add(matrix->selected_metric_value_capacity, dispatch->result_bytes);
    add(matrix->topk_partial_score_capacity, dispatch->result_bytes);
    add(matrix->topk_partial_index_capacity, sizeof(uint32_t));
    return bytes;
}

uint64_t precision_execution_peak_device_bytes(
    const PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol
) {
    using gafime_gpu_abi::DeviceMemoryPeakTracker;
    using gafime_gpu_abi::saturating_mul_u64;
    DeviceMemoryPeakTracker tracker(precision_matrix_resident_device_bytes(matrix));
    const uint64_t total_rows = precision_planned_row_count(protocol);
    const uint64_t metric_values = saturating_mul_u64(total_rows, protocol->metric_ids.len);
    tracker.grow(matrix->metric_value_capacity, metric_values, matrix->dispatch->result_bytes);
    if (!precision_descriptors_resident(matrix, protocol)) {
        tracker.reserve_pair(
            matrix->combo_capacity, protocol->combo_indices.len, sizeof(uint32_t),
            matrix->metric_id_capacity, protocol->metric_ids.len, sizeof(uint32_t));
    }
    if (protocol->rank.top_k == 0) {
        return tracker.peak_bytes();
    }
    const uint64_t output_rows = precision_output_row_count(protocol, total_rows);
    tracker.grow(matrix->selected_index_capacity, output_rows, sizeof(uint32_t));
    const uint64_t partial_items = saturating_mul_u64(
        topk_partial_block_count(total_rows, output_rows), output_rows);
    tracker.grow(
        matrix->topk_partial_score_capacity, partial_items, matrix->dispatch->result_bytes);
    tracker.grow(matrix->topk_partial_index_capacity, partial_items, sizeof(uint32_t));
    tracker.grow(
        matrix->selected_metric_value_capacity,
        saturating_mul_u64(output_rows, protocol->metric_ids.len),
        matrix->dispatch->result_bytes);
    return tracker.peak_bytes();
}

uint64_t precision_splitmix64_next(uint64_t* state) {
    uint64_t value = (*state += 0x9E3779B97F4A7C15ull);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

uint64_t precision_bounded_random(uint64_t* state, uint64_t bound) {
    return bound <= 1 ? 0 : precision_splitmix64_next(state) % bound;
}

template <typename ResultT>
ResultT precision_metric_extremeness_host(uint32_t metric_id, ResultT value) {
    if (!std::isfinite(value)) return -std::numeric_limits<ResultT>::infinity();
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_SPEARMAN
        ? std::fabs(value)
        : value;
}

template <typename StorageT, typename ResultT>
int precision_permutation_pvalues_host(
    PrecisionHipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    const uint64_t* candidate_ids,
    uint64_t selected_rows,
    const ResultT* observed_metric_values,
    ResultT* p_values
) try {
    if (matrix == nullptr || protocol == nullptr || protocol->permutations.permutation_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t planned_rows = precision_planned_row_count(protocol);
    if (selected_rows > planned_rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    uint64_t selected_values = 0;
    uint64_t all_values = 0;
    uint64_t combo_values = 0;
    if (!checked_mul_u64(selected_rows, metric_count, &selected_values) ||
        !checked_mul_u64(planned_rows, metric_count, &all_values) ||
        !checked_mul_u64(planned_rows, protocol->max_arity, &combo_values) ||
        !allocation_fits_size_t(selected_values, sizeof(ResultT)) ||
        !allocation_fits_size_t(all_values, sizeof(ResultT)) ||
        !allocation_fits_size_t(combo_values, sizeof(uint32_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (selected_rows != 0 &&
        (candidate_ids == nullptr || observed_metric_values == nullptr || p_values == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t row = 0; row < selected_rows; ++row) {
        if (candidate_ids[row] >= planned_rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    const auto& resident_target = precision_target_host<StorageT>(matrix);
    if (resident_target.size() != matrix->rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    const std::vector<StorageT> original_target = resident_target;
    std::vector<StorageT> permuted_target(static_cast<size_t>(matrix->rows));
    std::vector<uint64_t> order(static_cast<size_t>(matrix->rows));
    std::vector<uint32_t> exceedances(static_cast<size_t>(selected_values), 0);
    std::vector<uint32_t> combo_indices(static_cast<size_t>(combo_values));
    std::vector<ResultT> metric_values(static_cast<size_t>(all_values));
    std::vector<uint32_t> ranks(static_cast<size_t>(planned_rows));
    std::vector<uint32_t> families(static_cast<size_t>(planned_rows));
    std::vector<uint64_t> result_ids(static_cast<size_t>(planned_rows));
    std::vector<uint32_t> row_flags(static_cast<size_t>(planned_rows));

    GafimeLaunchProtocol iteration = *protocol;
    iteration.rank.top_k = 0;
    iteration.rank.include_ties = 0;
    iteration.permutations = {};
    uint32_t result_flags = 0;
    uint64_t result_rows = 0;
    PrecisionResultSink sink{
        matrix->dispatch->result_dtype == GAFIME_DTYPE_F32
            ? GAFIME_ABI_VERSION
            : GAFIME_PRECISION_ABI_VERSION,
        iteration.max_arity,
        metric_count,
        &result_flags,
        planned_rows,
        &result_rows,
        combo_indices.data(),
        metric_values.data(),
        ranks.data(),
        families.data(),
        result_ids.data(),
        row_flags.data(),
    };

    int status = GAFIME_STATUS_OK;
    for (uint32_t permutation = 0;
         permutation < protocol->permutations.permutation_count;
         ++permutation) {
        if (protocol->permutations.target_offsets.len != 0) {
            const uint64_t offset = static_cast<uint64_t>(permutation) * matrix->rows;
            for (uint64_t row = 0; row < matrix->rows; ++row) {
                const uint64_t source = protocol->permutations.target_offsets.ptr[offset + row];
                if (source >= matrix->rows) {
                    status = GAFIME_STATUS_INVALID_ARGUMENT;
                    break;
                }
                permuted_target[static_cast<size_t>(row)] =
                    original_target[static_cast<size_t>(source)];
            }
            if (status != GAFIME_STATUS_OK) break;
        } else {
            for (uint64_t row = 0; row < matrix->rows; ++row) order[row] = row;
            uint64_t state = protocol->permutations.seed ^
                static_cast<uint64_t>(permutation) * 0xD1B54A32D192ED03ull ^
                0xA5A5A5A59E3779B9ull;
            for (uint64_t row = matrix->rows; row > 1; --row) {
                std::swap(order[row - 1], order[precision_bounded_random(&state, row)]);
            }
            for (uint64_t row = 0; row < matrix->rows; ++row) {
                permuted_target[row] = original_target[order[row]];
            }
        }
        status = matrix->dispatch->update_target(matrix, permuted_target.data(), matrix->rows);
        if (status != GAFIME_STATUS_OK) break;
        result_rows = 0;
        result_flags = 0;
        status = matrix->dispatch->execute(matrix, &iteration, &sink);
        if (status != GAFIME_STATUS_OK || result_rows != planned_rows) {
            if (status == GAFIME_STATUS_OK) status = GAFIME_STATUS_DEVICE_ERROR;
            break;
        }
        for (uint32_t metric = 0; metric < metric_count; ++metric) {
            ResultT maximum = -std::numeric_limits<ResultT>::infinity();
            for (uint64_t row = 0; row < planned_rows; ++row) {
                maximum = std::max(
                    maximum,
                    precision_metric_extremeness_host(
                        protocol->metric_ids.ptr[metric],
                        metric_values[row * metric_count + metric]));
            }
            for (uint64_t selected = 0; selected < selected_rows; ++selected) {
                const uint64_t index = selected * metric_count + metric;
                const ResultT observed = precision_metric_extremeness_host(
                    protocol->metric_ids.ptr[metric], observed_metric_values[index]);
                if (maximum >= observed) ++exceedances[index];
            }
        }
    }
    const int restore_status =
        matrix->dispatch->update_target(matrix, original_target.data(), matrix->rows);
    if (status != GAFIME_STATUS_OK) return status;
    if (restore_status != GAFIME_STATUS_OK) return restore_status;
    const ResultT denominator =
        static_cast<ResultT>(protocol->permutations.permutation_count) + static_cast<ResultT>(1);
    for (uint64_t index = 0; index < selected_values; ++index) {
        p_values[index] =
            (static_cast<ResultT>(exceedances[index]) + static_cast<ResultT>(1)) / denominator;
    }
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
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

static int rocm_precision_capabilities_internal(
    uint32_t device_id,
    GafimePrecisionCapabilities* capabilities_out
) {
    if (capabilities_out == nullptr ||
        device_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
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
    std::memset(capabilities_out, 0, sizeof(*capabilities_out));
    capabilities_out->abi_version = GAFIME_PRECISION_ABI_VERSION;
    capabilities_out->backend_kind = GAFIME_BACKEND_ROCM;
    capabilities_out->profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32 |
        GAFIME_PRECISION_PROFILE_MASK_MIXED | GAFIME_PRECISION_PROFILE_MASK_FP64;
    capabilities_out->storage_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    capabilities_out->result_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    capabilities_out->flags = rocm_device_flags(props) |
        GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64 |
        GAFIME_GPU_DEVICE_FLAG_F64_STORAGE;
    capabilities_out->reserved[0] = rocm_arch_class(props);
    capabilities_out->reserved[1] = parse_gfx_arch(props.gcnArchName);
    return GAFIME_STATUS_OK;
}

static int rocm_matrix_alloc_internal(
    uint32_t device_id,
    const GafimePrecisionMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) {
    return precision_matrix_alloc(device_id, matrix_desc, matrix_out, false);
}

static int rocm_matrix_upload_f32_internal(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->storage_dtype != GAFIME_DTYPE_F32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->upload(matrix, features_host, target_host, rows, cols);
}

static int rocm_matrix_upload_f64_internal(
    GafimeGpuMatrix matrix_handle,
    const double* features_host,
    const double* target_host,
    uint64_t rows,
    uint32_t cols
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->storage_dtype != GAFIME_DTYPE_F64) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->upload(matrix, features_host, target_host, rows, cols);
}

static int rocm_matrix_update_target_f32_internal(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || target_host == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->storage_dtype != GAFIME_DTYPE_F32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->update_target(matrix, target_host, rows);
}

static int rocm_matrix_update_target_f64_internal(
    GafimeGpuMatrix matrix_handle,
    const double* target_host,
    uint64_t rows
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || target_host == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->storage_dtype != GAFIME_DTYPE_F64) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->update_target(matrix, target_host, rows);
}

static int rocm_execute_f32_internal(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || result_out == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->result_dtype != GAFIME_DTYPE_F32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_precision_launch_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    PrecisionResultSink sink{
        result_out->abi_version, result_out->max_arity, result_out->metric_count,
        &result_out->flags, result_out->capacity, &result_out->row_count,
        result_out->combo_indices, result_out->metric_values, result_out->ranks,
        result_out->families, result_out->candidate_ids, result_out->row_flags,
    };
    status = validate_precision_result_sink(protocol->base, matrix, &sink);
    return status == GAFIME_STATUS_OK ? matrix->dispatch->execute(matrix, protocol->base, &sink) : status;
}

static int rocm_execute_f64_internal(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || result_out == nullptr ||
        matrix->legacy_abi10 || matrix->dispatch->result_dtype != GAFIME_DTYPE_F64) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_precision_launch_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    PrecisionResultSink sink{
        result_out->abi_version, result_out->max_arity, result_out->metric_count,
        &result_out->flags, result_out->capacity, &result_out->row_count,
        result_out->combo_indices, result_out->metric_values, result_out->ranks,
        result_out->families, result_out->candidate_ids, result_out->row_flags,
    };
    status = validate_precision_result_sink(protocol->base, matrix, &sink);
    return status == GAFIME_STATUS_OK ? matrix->dispatch->execute(matrix, protocol->base, &sink) : status;
}

static int rocm_execution_memory_peak_internal(
    GafimeGpuMatrix matrix_handle,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *peak_bytes_out = 0;
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix != nullptr && matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const int status = validate_precision_launch_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK || !matrix->content_valid) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }
    *peak_bytes_out = precision_execution_peak_device_bytes(matrix, protocol->base);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_numeric_routes_v2(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    uint32_t route_stride,
    GafimeNumericRoute* routes_out,
    uint32_t route_capacity,
    uint32_t* route_count_out
) {
    if (device_id > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        hipSetDevice(static_cast<int>(device_id)) != hipSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    constexpr uint32_t profiles[] = {
        GAFIME_PRECISION_FP32,
        GAFIME_PRECISION_MIXED,
        GAFIME_PRECISION_FP64,
    };
    return gafime_gpu_abi::enumerate_numeric_routes(
        consumer_abi_version,
        route_stride,
        routes_out,
        route_capacity,
        route_count_out,
        profiles,
        static_cast<uint32_t>(sizeof(profiles) / sizeof(profiles[0])));
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id,
    const GafimeNumericMatrixDesc* desc,
    GafimeGpuMatrix* matrix_out
) {
    int status = gafime_gpu_abi::validate_numeric_matrix_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionMatrixDesc internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = desc->route.profile;
    internal.dtype = desc->route.storage_dtype;
    internal.layout = desc->layout;
    internal.rows = desc->rows;
    internal.cols = desc->cols;
    internal.row_stride = desc->row_stride;
    internal.bytes = desc->bytes;
    return rocm_matrix_alloc_internal(device_id, &internal, matrix_out);
}

GAFIME_GPU_API int gafime_gpu_matrix_upload_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* features,
    const GafimeConstBufferView* target,
    uint64_t rows,
    uint32_t cols
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->profile != matrix->profile || rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t feature_count = 0;
    if (!checked_mul_u64(rows, cols, &feature_count)) return GAFIME_STATUS_INVALID_ARGUMENT;
    status = gafime_gpu_abi::validate_const_buffer(
        features, route->storage_dtype, feature_count);
    if (status == GAFIME_STATUS_OK) {
        status = gafime_gpu_abi::validate_const_buffer(target, route->storage_dtype, rows);
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (route->storage_dtype == GAFIME_DTYPE_F32) {
        return rocm_matrix_upload_f32_internal(
            matrix_handle,
            static_cast<const float*>(features->data),
            static_cast<const float*>(target->data),
            rows,
            cols);
    }
    if (route->storage_dtype == GAFIME_DTYPE_F64) {
        return rocm_matrix_upload_f64_internal(
            matrix_handle,
            static_cast<const double*>(features->data),
            static_cast<const double*>(target->data),
            rows,
            cols);
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* target,
    uint64_t rows
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->profile != matrix->profile || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_gpu_abi::validate_const_buffer(target, route->storage_dtype, rows);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->storage_dtype == GAFIME_DTYPE_F32) {
        return rocm_matrix_update_target_f32_internal(
            matrix_handle, static_cast<const float*>(target->data), rows);
    }
    if (route->storage_dtype == GAFIME_DTYPE_F64) {
        return rocm_matrix_update_target_f64_internal(
            matrix_handle, static_cast<const double*>(target->data), rows);
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execute_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericResultTable* result_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_gpu_abi::validate_numeric_result_table(
        result_out, protocol->route.result_dtype);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal_protocol{};
    internal_protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal_protocol.profile = protocol->route.profile;
    internal_protocol.base = protocol->base;
    if (protocol->route.result_dtype == GAFIME_DTYPE_F32) {
        GafimeResultTable internal{};
        internal.abi_version = GAFIME_ABI_VERSION;
        internal.max_arity = result_out->max_arity;
        internal.metric_count = result_out->metric_count;
        internal.flags = result_out->flags;
        internal.capacity = result_out->capacity;
        internal.row_count = result_out->row_count;
        internal.combo_indices = result_out->combo_indices;
        internal.metric_values = static_cast<float*>(result_out->metric_values.data);
        internal.ranks = result_out->ranks;
        internal.families = result_out->families;
        internal.candidate_ids = result_out->candidate_ids;
        internal.row_flags = result_out->row_flags;
        status = rocm_execute_f32_internal(matrix_handle, &internal_protocol, &internal);
        result_out->flags = internal.flags;
        result_out->row_count = internal.row_count;
        return status;
    }
    if (protocol->route.result_dtype == GAFIME_DTYPE_F64) {
        GafimeResultTableF64 internal{};
        internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
        internal.max_arity = result_out->max_arity;
        internal.metric_count = result_out->metric_count;
        internal.flags = result_out->flags;
        internal.capacity = result_out->capacity;
        internal.row_count = result_out->row_count;
        internal.combo_indices = result_out->combo_indices;
        internal.metric_values = static_cast<double*>(result_out->metric_values.data);
        internal.ranks = result_out->ranks;
        internal.families = result_out->families;
        internal.candidate_ids = result_out->candidate_ids;
        internal.row_flags = result_out->row_flags;
        status = rocm_execute_f64_internal(matrix_handle, &internal_protocol, &internal);
        result_out->flags = internal.flags;
        result_out->row_count = internal.row_count;
        return status;
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = protocol->route.profile;
    internal.base = protocol->base;
    return rocm_execution_memory_peak_internal(matrix_handle, &internal, peak_bytes_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10 || peak_bytes_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *peak_bytes_out = 0;
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK || !matrix->content_valid || protocol->base == nullptr ||
        protocol->base->permutations.permutation_count == 0 ||
        selected_row_count > precision_planned_row_count(protocol->base)) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }
    *peak_bytes_out = precision_execution_peak_device_bytes(matrix, protocol->base);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericSignificanceTable* significance
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK || !matrix->content_valid) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }
    GafimePrecisionLaunchProtocol internal_protocol{};
    internal_protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal_protocol.profile = protocol->route.profile;
    internal_protocol.base = protocol->base;
    status = validate_precision_launch_protocol(&internal_protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_gpu_abi::validate_numeric_significance_table(
        significance, protocol->route.result_dtype);
    if (status != GAFIME_STATUS_OK || protocol->base == nullptr ||
        significance->metric_count != protocol->base->metric_ids.len) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }
    if (matrix->profile == GAFIME_PRECISION_FP32) {
        return precision_permutation_pvalues_host<float, float>(
            matrix,
            protocol->base,
            significance->candidate_ids,
            significance->row_count,
            static_cast<const float*>(significance->observed_metric_values.data),
            static_cast<float*>(significance->p_values.data));
    }
    if (matrix->profile == GAFIME_PRECISION_MIXED) {
        return precision_permutation_pvalues_host<float, double>(
            matrix,
            protocol->base,
            significance->candidate_ids,
            significance->row_count,
            static_cast<const double*>(significance->observed_metric_values.data),
            static_cast<double*>(significance->p_values.data));
    }
    if (matrix->profile == GAFIME_PRECISION_FP64) {
        return precision_permutation_pvalues_host<double, double>(
            matrix,
            protocol->base,
            significance->candidate_ids,
            significance->row_count,
            static_cast<const double*>(significance->observed_metric_values.data),
            static_cast<double*>(significance->p_values.data));
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics_v2(
    GafimeGpuMatrix matrix_handle,
    GafimeNumericInteractionDiagnosticBatch* diagnostics
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || matrix->legacy_abi10) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_diagnostics(diagnostics, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    GafimeInteractionDiagnosticBatch internal{};
    internal.abi_version = GAFIME_ABI_VERSION;
    internal.max_arity = diagnostics->max_arity;
    internal.row_count = diagnostics->row_count;
    internal.combo_indices = diagnostics->combo_indices;
    internal.combo_index_count = diagnostics->combo_index_count;
    internal.overflow_row_counts = diagnostics->overflow_row_counts;
    internal.flags = diagnostics->row_flags;
    return matrix->dispatch->interaction_diagnostics(matrix, &internal);
}

GAFIME_GPU_API int gafime_gpu_matrix_free_v2(GafimeGpuMatrix matrix_handle) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    // Teardown is generation-neutral: after magic validation either ABI
    // generation may release the one canonical resident owner.
    precision_matrix_free(matrix);
    return GAFIME_STATUS_OK;
}

// Frozen ABI 1.0 exports are deliberately thin adapters into the canonical
// resident owner above.  The adapter selects the narrow legacy arithmetic
// policy once at allocation; it does not own a second matrix, graph, ranking,
// descriptor, or result lifecycle.
GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) try {
    if (matrix_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *matrix_out = nullptr;
    int status = validate_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionMatrixDesc internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = GAFIME_PRECISION_FP32;
    internal.dtype = GAFIME_DTYPE_F32;
    internal.layout = matrix_desc->layout;
    internal.rows = matrix_desc->rows;
    internal.cols = matrix_desc->cols;
    internal.row_stride = matrix_desc->row_stride;
    internal.bytes = matrix_desc->bytes;
    return precision_matrix_alloc(device_id, &internal, matrix_out, true);
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
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || !matrix->legacy_abi10 || features_host == nullptr ||
        target_host == nullptr || matrix->dispatch->storage_dtype != GAFIME_DTYPE_F32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->upload(
        matrix, features_host, target_host, rows, cols);
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || !matrix->legacy_abi10 || target_host == nullptr ||
        matrix->dispatch->storage_dtype != GAFIME_DTYPE_F32) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->update_target(matrix, target_host, rows);
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    // ABI 1.0 historically accepts null and also tears down an additive matrix
    // owner.  Both generations now share this single destruction path.
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix != nullptr) precision_matrix_free(matrix);
}

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics(
    GafimeGpuMatrix matrix_handle,
    GafimeInteractionDiagnosticBatch* diagnostics
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || !matrix->legacy_abi10 || diagnostics == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return matrix->dispatch->interaction_diagnostics(matrix, diagnostics);
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *peak_bytes_out = 0;
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || !matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = require_valid_matrix_content(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    *peak_bytes_out = precision_execution_peak_device_bytes(matrix, protocol);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    PrecisionHipMatrix* matrix = precision_matrix_from_handle(matrix_handle);
    if (matrix == nullptr || !matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) return status;
    status = require_valid_matrix_content(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    PrecisionResultSink sink{
        result_out->abi_version,
        result_out->max_arity,
        result_out->metric_count,
        &result_out->flags,
        result_out->capacity,
        &result_out->row_count,
        result_out->combo_indices,
        result_out->metric_values,
        result_out->ranks,
        result_out->families,
        result_out->candidate_ids,
        result_out->row_flags,
    };
    return matrix->dispatch->execute(matrix, protocol, &sink);
}

}  // extern "C"
