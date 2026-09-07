#ifndef GAFIME_CUDA_PRECISION_KERNELS_CUH
#define GAFIME_CUDA_PRECISION_KERNELS_CUH

// Profile-specialized CUDA kernels owned by the canonical numeric-route ABI.
// Frozen ABI 1.0 calls are thin adapters into these same device primitives;
// kernels.cuh now owns only shared launch and architecture policy.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_api.hpp"
#include "kernels.cuh"
#include "../common/gafime_semantic_primitives_abi.hpp"

namespace gafime_cuda_v1 {

// All entries in this table are selected once when a v2 matrix is allocated.
// They intentionally use erased pointers at the host boundary only; each
// entry points to a compile-time typed kernel family.  Thus no precision
// policy branch is present in a device scoring/reduction/ranking hot loop.
struct CudaPrecisionKernelSet {
    size_t storage_bytes;
    size_t accumulation_bytes;
    size_t result_bytes;
    size_t target_stats_bytes;
    size_t feature_stats_bytes;

    cudaError_t (*target_stats)(
        const void* target,
        uint64_t n_samples,
        void* target_stats,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*feature_stats)(
        const void* features,
        uint64_t n_samples,
        uint32_t n_features,
        void* feature_stats,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*interaction_diagnostics)(
        const void* features,
        const void* target,
        const void* column_means,
        const uint32_t* combo_indices,
        uint64_t combo_count,
        uint64_t n_samples,
        uint32_t max_arity,
        uint64_t* overflow_row_counts,
        uint32_t* flags,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*continuous)(
        const void* features,
        const void* target,
        const void* column_means,
        const uint32_t* combo_indices,
        uint64_t n_samples,
        uint32_t arity,
        uint64_t descriptor_offset,
        uint64_t combo_count,
        uint32_t scaled_covariance,
        const uint32_t* metric_ids,
        uint32_t metric_count,
        void* metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    // Finite unary covariance fast path.  It is a shared profile primitive;
    // the ABI 1.0 adapter selects the same typed implementation when its
    // historical all-finite admission conditions hold.
    cudaError_t (*continuous_unary)(
        const void* features,
        const void* target,
        const void* target_stats,
        const void* feature_stats,
        const uint32_t* combo_indices,
        uint64_t n_samples,
        uint64_t descriptor_offset,
        uint64_t combo_count,
        const uint32_t* metric_ids,
        uint32_t metric_count,
        void* metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*mutual_info)(
        const void* features,
        const void* target,
        const void* column_means,
        const uint32_t* combo_indices,
        uint64_t n_samples,
        uint32_t arity,
        uint64_t descriptor_offset,
        uint64_t combo_count,
        uint32_t metric_count,
        uint32_t metric_index,
        uint32_t bins,
        void* metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*build_target_ranks)(
        const void* target,
        uint64_t n_samples,
        uint64_t* target_ranks_twice,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*spearman)(
        const void* features,
        const void* target,
        const void* column_means,
        const uint64_t* target_ranks_twice,
        const uint32_t* combo_indices,
        uint64_t n_samples,
        uint32_t arity,
        uint64_t descriptor_offset,
        uint64_t combo_count,
        uint32_t metric_count,
        uint32_t metric_index,
        void* metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    // ABI 1.0's Spearman primitive keeps f32 storage and visible results but
    // accumulates ranks/covariance in f64.  It is an adapter-only primitive;
    // all other ABI 1.0 metrics use the FP32 profile above.
    cudaError_t (*legacy_spearman)(
        const void* features,
        const void* target,
        const void* column_means,
        const uint64_t* target_ranks_twice,
        const uint32_t* combo_indices,
        uint64_t n_samples,
        uint32_t arity,
        uint64_t descriptor_offset,
        uint64_t combo_count,
        uint32_t metric_count,
        uint32_t metric_index,
        void* metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*select_topk)(
        const void* metric_values,
        uint64_t row_count,
        uint32_t metric_count,
        uint32_t primary_metric_index,
        uint32_t top_k,
        uint32_t descending,
        uint32_t* selected_indices,
        void* partial_scores,
        uint32_t* partial_indices,
        uint32_t partial_blocks,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*copy_selected_rows)(
        const void* metric_values,
        const uint32_t* selected_indices,
        uint64_t selected_count,
        uint32_t metric_count,
        void* selected_metric_values,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*selected_metric_max)(
        const void* metric_values,
        uint64_t row_count,
        const uint32_t* metric_ids,
        uint32_t metric_count,
        void* metric_max,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*accumulate_exceedances)(
        const void* metric_max,
        const uint32_t* metric_ids,
        uint32_t metric_count,
        const void* observed_metric_values,
        uint64_t selected_count,
        uint32_t* exceedance_counts,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
};

const CudaPrecisionKernelSet* cuda_precision_kernel_set(GafimePrecisionProfile profile);

// Internal ABI 1.0 adapter set. This is not a fourth public profile.
const CudaPrecisionKernelSet* cuda_legacy_kernel_set();

/* Compiler-owned typed arithmetic used by the optional resident semantic
 * table.  This intentionally does not carry target pointers, metric IDs,
 * evidence IDs, or selection state. */
struct CudaSemanticKernelSet {
    size_t storage_bytes;
    size_t accumulation_bytes;
    size_t result_bytes;

    cudaError_t (*absolute_difference)(
        void* columns,
        uint64_t rows,
        uint32_t left_slot,
        uint32_t right_slot,
        uint32_t output_slot,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*softsign)(
        void* columns,
        uint64_t rows,
        uint32_t input_slot,
        uint32_t output_slot,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*centered_product)(
        void* columns,
        uint64_t rows,
        const uint32_t* operand_slots,
        const uint64_t* mean_bits,
        uint32_t operand_count,
        uint32_t output_slot,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*reject_nonfinite_output)(
        const void* columns,
        uint64_t rows,
        uint32_t slot,
        uint32_t* nonfinite_out,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*pairwise_pearson)(
        const void* left_columns,
        const void* right_columns,
        uint64_t rows,
        const uint32_t* left_slots,
        const uint32_t* right_slots,
        uint64_t pair_count,
        uint32_t mode,
        void* values,
        uint32_t* states,
        uint64_t* supports,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*ordered_edge_energy)(
        const void* columns,
        uint64_t rows,
        const uint32_t* candidate_slots,
        uint64_t candidate_count,
        const GafimeSemanticEdge* edges,
        const void* weights,
        uint64_t edge_count,
        void* values,
        uint32_t* states,
        uint64_t* supports,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
    cudaError_t (*sparse_gather)(
        const void* source_columns,
        uint64_t source_rows,
        void* destination_columns,
        uint64_t destination_rows,
        const uint32_t* source_slots,
        const uint32_t* destination_slots,
        uint64_t slot_count,
        const uint64_t* row_indices,
        const CudaKernelLaunchPolicy& launch_policy,
        cudaStream_t stream
    );
};

const CudaSemanticKernelSet* cuda_semantic_kernel_set(GafimePrecisionProfile profile);

}  // namespace gafime_cuda_v1

namespace gafime_cuda_v1::detail {

// The ABI deliberately keeps one matrix-free entry point.  The legacy
// launcher delegates only recognized v2 handles here before interpreting its
// private ABI-1.0 matrix layout.
bool free_precision_cuda_matrix(GafimeGpuMatrix matrix_handle);

// Route the ABI-1.0 diagnostic batch layout through the scalar specialization
// owned by an ABI-1.1 resident matrix.  The bool distinguishes a legacy handle
// from a recognized precision handle; `status_out` carries the actual ABI
// status for the recognized case.
bool interaction_diagnostics_precision_cuda_matrix(
    GafimeGpuMatrix matrix_handle,
    GafimeInteractionDiagnosticBatch* diagnostics,
    int* status_out
);

}  // namespace gafime_cuda_v1::detail

#endif  // GAFIME_CUDA_PRECISION_KERNELS_CUH
