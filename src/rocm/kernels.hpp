#ifndef GAFIME_ROCM_KERNELS_HPP
#define GAFIME_ROCM_KERNELS_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

#include "precision.hpp"
#include "../common/gafime_semantic_primitives_abi.hpp"

namespace gafime_rocm_v1 {

constexpr int kThreadsPerBlock = 256;
constexpr int kMiThreadsPerBlock = kThreadsPerBlock;
constexpr int kTopKThreadsPerBlock = kThreadsPerBlock;
constexpr uint32_t kTopKMaxPartialBlocks = 4096;
constexpr uint32_t kTemplateMaxArity = 5;
constexpr uint32_t kMaxMutualInfoBins = 96;
constexpr uint64_t kSpearmanTargetRankCacheMinSamples = 128;
constexpr uint64_t kSpearmanTargetRankCacheMaxSamples = 4096;
constexpr uint64_t kSpearmanTargetRankCacheMinUnaryCandidates = 2;

// ABI 1.1 precision-profile caches are deliberately typed.  In particular,
// mixed stores its matrix and pointwise means as fp32 while retaining fp64
// target/feature statistical state; fp64 has no fp32 member in this path.
template <typename AccumT>
struct PrecisionTargetStatsDevice {
    AccumT mean_y;
    AccumT syy;
    uint32_t finite;
    uint32_t reserved;
};

template <typename AccumT>
struct PrecisionUnaryFeatureStatsDevice {
    AccumT mean_x;
    AccumT sxx;
    uint32_t finite;
    uint32_t reserved;
};

// ABI 1.0's cache layouts are byte-identical to the fp32 ABI 1.1 cache
// layouts.  Keep the old names at the opaque host boundary while making the
// device implementation use the one shared float precision family.
using TargetStatsDevice = PrecisionTargetStatsDevice<float>;
using UnaryFeatureStatsDevice = PrecisionUnaryFeatureStatsDevice<float>;

// Precision-profile kernels are the single device implementation for the
// ABI-1.1 routes and the ABI-1.0 adapters.  The legacy adapters bind only the
// float storage/accumulation specialisations, so a legacy pointer is never
// reinterpreted as a double pointer.  The host selects each instantiation once
// and stores the resulting function table on the precision matrix.
namespace kernel::precision_kernel {

/* Optional semantic-arithmetic kernels.  These operate only on physical
 * resident columns; host code owns semantic IDs, contexts and policy. */
template <typename StorageT>
__global__ void semantic_absolute_difference_kernel(
    StorageT* columns, uint64_t rows, uint32_t left_slot, uint32_t right_slot, uint32_t output_slot);

template <typename StorageT>
__global__ void semantic_softsign_kernel(
    StorageT* columns, uint64_t rows, uint32_t input_slot, uint32_t output_slot);

template <typename StorageT>
__global__ void semantic_centered_product_kernel(
    StorageT* columns, uint64_t rows, const uint32_t* operand_slots, const uint64_t* mean_bits,
    uint32_t operand_count, uint32_t output_slot);

template <typename StorageT>
__global__ void semantic_reject_nonfinite_output_kernel(
    const StorageT* columns, uint64_t rows, uint32_t slot, uint32_t* nonfinite_out);

template <typename StorageT, typename AccumT, typename ResultT>
__global__ void semantic_pairwise_pearson_kernel(
    const StorageT* left_columns, const StorageT* right_columns, uint64_t rows,
    const uint32_t* left_slots, const uint32_t* right_slots, uint64_t pair_count, uint32_t mode,
    ResultT* values, uint32_t* states, uint64_t* supports);

template <typename StorageT, typename AccumT, typename ResultT>
__global__ void semantic_ordered_edge_energy_kernel(
    const StorageT* columns, uint64_t rows, const uint32_t* candidate_slots,
    uint64_t candidate_count, const GafimeSemanticEdge* edges, const StorageT* weights,
    uint64_t edge_count, ResultT* values, uint32_t* states, uint64_t* supports);

template <typename StorageT>
__global__ void semantic_sparse_gather_kernel(
    const StorageT* source_columns, uint64_t source_rows, StorageT* destination_columns,
    uint64_t destination_rows, const uint32_t* source_slots, const uint32_t* destination_slots,
    uint64_t slot_count, const uint64_t* row_indices);

template <typename StorageT, typename AccumT>
__global__ void target_stats_kernel(
    const StorageT* target,
    uint64_t n_samples,
    PrecisionTargetStatsDevice<AccumT>* target_stats
);

template <typename StorageT, typename AccumT>
__global__ void unary_feature_stats_kernel(
    const StorageT* features,
    uint64_t n_samples,
    uint32_t n_features,
    PrecisionUnaryFeatureStatsDevice<AccumT>* feature_stats
);

template <typename StorageT, typename AccumT, typename ResultT>
__global__ void score_continuous_unary_all_finite_chunk_kernel(
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
    ResultT* metric_values
);

template <typename StorageT, typename AccumT>
__global__ void interaction_diagnostics_kernel(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t max_arity,
    uint64_t* overflow_row_counts,
    uint32_t* flags
);

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, bool Scaled>
__global__ void score_continuous_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    ResultT* metric_values
);

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity, uint32_t Bins>
__global__ void score_mutual_info_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const AccumT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t legacy_nonfinite,
    ResultT* metric_values
);

template <typename StorageT, typename AccumT, typename ResultT, uint32_t Arity,
          typename MeanT = AccumT>
__global__ void score_spearman_chunk_kernel_static(
    const StorageT* features,
    const StorageT* target,
    const MeanT* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values
);

// Narrow shared primitives used by the ABI-1.0 adapter and canonical ABI-1.1
// unary Spearman cache. They are templates so each current route can reuse
// the mechanism without creating another complete scoring engine.
template <typename StorageT, typename RankT>
__global__ void build_spearman_target_ranks_kernel(
    const StorageT* target,
    uint64_t n_samples,
    RankT* target_ranks
);

template <typename StorageT, typename AccumT, typename ResultT>
__global__ void score_spearman_unary_cached_target_ranks_kernel(
    const StorageT* features,
    const StorageT* target,
    const AccumT* target_ranks,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    ResultT* metric_values
);

template <typename ResultT, bool Descending>
__global__ void select_topk_partials_kernel_static(
    const ResultT* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    ResultT* partial_scores,
    uint32_t* partial_indices
);

template <typename ResultT, bool Descending>
__global__ void merge_topk_partials_kernel_static(
    const ResultT* partial_scores,
    const uint32_t* partial_indices,
    uint64_t partial_count,
    uint32_t top_k,
    uint32_t* selected_indices
);

template <typename ResultT>
__global__ void copy_selected_rows_kernel(
    const ResultT* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    ResultT* selected_metric_values
);

}  // namespace kernel::precision_kernel

// Frozen ABI 1.0 accepted arbitrary runtime arities.  Keep only one compact
// float compatibility kernel per scoring family for arity values above the
// ABI-1.1 template ceiling; the normal ABI-1.0 arities are dispatched through
// the shared precision specialisations above.  These declarations are private
// launch targets, not exported host ABI symbols.
namespace kernel::legacy_compat {

__global__ void score_continuous_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint32_t scaled_covariance,
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

}  // namespace kernel::legacy_compat

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
);

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
);

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
);

// Generic ABI-1.1 route-typed variants.  These wrappers keep the public
// operation generic while selecting the concrete device specialization once
// per profile at the host dispatch boundary.
template <typename StorageT, typename RankT>
hipError_t launch_precision_spearman_target_ranks(
    const StorageT* target,
    uint64_t n_samples,
    RankT* target_ranks,
    hipStream_t stream
);

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
);

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
);

hipError_t launch_copy_selected_metric_rows(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values,
    hipStream_t stream
);

}  // namespace gafime_rocm_v1

#endif  // GAFIME_ROCM_KERNELS_HPP
