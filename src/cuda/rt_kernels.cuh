#ifndef GAFIME_CUDA_RT_KERNELS_CUH
#define GAFIME_CUDA_RT_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_cuda_v1::rt_kernel {

struct GafimeRtBox {
    float lo_x;
    float lo_y;
    float lo_z;
    float hi_x;
    float hi_y;
    float hi_z;
    uint32_t open_lo_mask;
    uint32_t dims;
};

struct GafimeRtTriVertex {
    float x;
    float y;
    float z;
};

struct GafimeRtTriIndex {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

__global__ void pack_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t axis0,
    uint32_t axis1,
    uint32_t axis2,
    uint32_t dims,
    float* points_xyz
);

__global__ void pack_grouped_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    const uint32_t* group_axes,
    const uint32_t* group_dims,
    uint32_t group_count,
    uint32_t point_stride,
    float* points_xyz
);

__global__ void decision_path_membership_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership
);

__global__ void decision_path_bitset_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    uint32_t words_per_path,
    uint32_t* membership_words
);

__global__ void score_decision_path_bitset_kernel(
    const uint32_t* membership_words,
    const float* target,
    uint64_t n_samples,
    uint32_t path_count,
    uint32_t words_per_path,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
);

__global__ void decision_path_target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    float* target_stats
);

__global__ void score_decision_path_direct_stats_kernel(
    const uint32_t* inside_counts,
    const float* inside_sum_y,
    const float* target_stats,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
);

__global__ void score_decision_path_direct_stats_scatter_kernel(
    const uint32_t* inside_counts,
    const float* inside_sum_y,
    const float* target_stats,
    const uint32_t* original_paths,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* final_metric_values
);

__global__ void scatter_decision_path_score_metrics_kernel(
    const float* group_metric_values,
    const uint32_t* original_paths,
    uint32_t group_path_count,
    uint32_t metric_count,
    float* final_metric_values
);

}  // namespace gafime_cuda_v1::rt_kernel

#endif  // GAFIME_CUDA_RT_KERNELS_CUH
