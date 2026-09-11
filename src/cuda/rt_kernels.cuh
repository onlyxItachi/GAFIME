#ifndef GAFIME_CUDA_RT_KERNELS_CUH
#define GAFIME_CUDA_RT_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "rt_abi.hpp"

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

constexpr uint32_t kRtFloatBucketShift = 9u;
constexpr uint64_t kRtFloatEncodingVersion = 1u;

__global__ void validate_rt_feature_domain_kernel(
    const float* features,
    uint64_t value_count,
    uint32_t* invalid_out
);

/* The local semantic RT entry scans only its resolved physical input slots.
 * It rejects non-finite and subnormal fp32 values before OptiX traversal so
 * the exact predicate path never turns an unavailable value into membership. */
__global__ void validate_semantic_region_input_domain_kernel(
    const float* columns,
    uint64_t rows,
    const uint32_t* input_slots,
    uint32_t input_slot_count,
    uint32_t* invalid_out
);

/* Maps path-major OptiX membership into fresh semantic-bank output slots.  A
 * bad temporary membership value leaves slots uncommitted at the host layer. */
__global__ void scatter_semantic_region_membership_kernel(
    const float* membership,
    uint64_t rows,
    uint32_t region_count,
    const uint32_t* output_slots,
    float* columns,
    uint32_t* invalid_out
);

/* Compact binary regional-statistics helpers.  They never materialize a
 * path-major float matrix: overlap-safe paths retain one bit per
 * (region,row), then reduce exact integer counts. */
__global__ void semantic_region_binary_label_masks_kernel(
    const uint64_t* row_indices,
    const uint8_t* values,
    uint64_t label_count,
    uint32_t* label_zero_words,
    uint32_t* label_one_words
);

__global__ void semantic_region_membership_masks_sm_kernel(
    const float* primary_columns,
    const float* paired_columns,
    uint64_t rows,
    const GafimeDecisionPathTerm* primary_terms,
    const GafimeDecisionPathTerm* paired_terms,
    const uint32_t* region_offsets,
    uint32_t region_count,
    uint32_t words_per_region,
    uint32_t* primary_membership_words,
    uint32_t* paired_membership_words
);

/* Exact SM baseline for grouped regions.  A conservative query-bound x-bin
 * (ordered-float only for unbounded fallback groups) narrows candidates, then
 * the same original predicate terms guard every candidate before its
 * overlap-safe membership bit is set. */
__global__ void semantic_region_membership_masks_binned_sm_kernel(
    const float* primary_points_xyz,
    const float* paired_points_xyz,
    uint64_t rows,
    const GafimeDecisionPathTerm* exact_terms,
    const uint32_t* exact_region_offsets,
    const uint32_t* group_path_offsets,
    const uint32_t* group_region_ids,
    const uint32_t* bin_offsets,
    const uint32_t* bin_candidates,
    const float* bin_lo,
    const float* bin_inv_span,
    uint32_t group_count,
    uint32_t point_stride,
    uint32_t point_group_stride,
    uint32_t words_per_region,
    uint32_t* primary_membership_words,
    uint32_t* paired_membership_words
);

__global__ void reduce_semantic_region_membership_masks_kernel(
    const uint32_t* primary_membership_words,
    const uint32_t* paired_membership_words,
    const uint32_t* label_zero_words,
    const uint32_t* label_one_words,
    uint64_t rows,
    uint32_t region_count,
    uint32_t words_per_region,
    uint32_t statistic_mask,
    uint64_t label_zero_count,
    uint64_t label_one_count,
    GafimeSemanticRtRegionExactStats* stats
);

__global__ void finalize_semantic_region_stats_kernel(
    uint32_t region_count,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    uint64_t rows,
    uint64_t label_zero_count,
    uint64_t label_one_count,
    GafimeSemanticRtRegionExactStats* stats
);

/* One exact target-free pointwise feature: the number of submitted regions
 * whose retained binary membership contains each row. */
__global__ void materialize_semantic_region_coverage_kernel(
    const uint32_t* primary_membership_words,
    uint64_t rows,
    uint32_t region_count,
    uint32_t words_per_region,
    float* output_column
);

/* The proof-gated first-hit RT path keeps a per-row exact membership count
 * rather than an R by N bitset.  This final scatter is intentionally tiny
 * and shares the same fresh-slot commit gate as the bitset path. */
__global__ void scatter_semantic_region_coverage_counts_kernel(
    const uint32_t* coverage_counts,
    uint64_t rows,
    float* output_column
);

__host__ __device__ inline uint32_t rt_canonical_float_bits(float value) {
#if defined(__CUDA_ARCH__)
    uint32_t bits = __float_as_uint(value);
#else
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    return (bits & 0x7fffffffu) == 0u ? 0u : bits;
}

__host__ __device__ inline uint32_t rt_ordered_float_key(float value) {
    const uint32_t bits = rt_canonical_float_bits(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

__host__ __device__ inline uint32_t rt_float_bucket(float value) {
    return rt_ordered_float_key(value) >> kRtFloatBucketShift;
}

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
    const double* target_stats,
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
    double* target_stats
);

__global__ void score_decision_path_direct_stats_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
);

__global__ void score_decision_path_direct_stats_scatter_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
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
