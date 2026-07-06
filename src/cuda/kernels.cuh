#ifndef GAFIME_CUDA_KERNELS_CUH
#define GAFIME_CUDA_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace gafime_cuda_v1 {

constexpr int kThreadsPerBlock = 256;
constexpr uint32_t kMaxMutualInfoBins = 96;

struct TargetStatsDevice {
    float mean_y;
    float syy;
    uint32_t finite;
    uint32_t reserved;
};

namespace kernel {

__global__ void target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    TargetStatsDevice* target_stats
);

__global__ void score_continuous_unary_all_finite_chunk_kernel(
    const float* features,
    const float* target,
    const TargetStatsDevice* target_stats,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
);

__global__ void score_continuous_chunk_kernel(
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
    float* metric_values
);

__global__ void score_mutual_info_chunk_kernel(
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
    float* metric_values
);

__global__ void score_spearman_chunk_kernel(
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
    float* metric_values
);

__global__ void select_topk_kernel(
    const float* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    uint32_t descending,
    uint32_t* selected_indices
);

__global__ void copy_selected_metric_rows_kernel(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values
);

__global__ void selected_metric_max_kernel(
    const float* metric_values,
    const uint64_t* candidate_ids,
    uint64_t selected_count,
    uint64_t total_rows,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_max
);

__global__ void accumulate_exceedances_kernel(
    const float* metric_max,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    const float* observed_metric_values,
    uint64_t selected_count,
    uint32_t* exceedance_counts
);

}  // namespace kernel

cudaError_t launch_target_stats(
    const float* target,
    uint64_t n_samples,
    TargetStatsDevice* target_stats,
    cudaStream_t stream
);

cudaError_t launch_continuous_chunk(
    const float* features,
    const float* target,
    const float* column_means,
    const TargetStatsDevice* target_stats,
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
);

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
);

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
);

cudaError_t launch_select_topk(
    const float* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    uint32_t descending,
    uint32_t* selected_indices,
    cudaStream_t stream
);

cudaError_t launch_copy_selected_metric_rows(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values,
    cudaStream_t stream
);

cudaError_t launch_selected_metric_max(
    const float* metric_values,
    const uint64_t* candidate_ids,
    uint64_t selected_count,
    uint64_t total_rows,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_max,
    cudaStream_t stream
);

cudaError_t launch_accumulate_exceedances(
    const float* metric_max,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    const float* observed_metric_values,
    uint64_t selected_count,
    uint32_t* exceedance_counts,
    cudaStream_t stream
);

}  // namespace gafime_cuda_v1

#endif  // GAFIME_CUDA_KERNELS_CUH
