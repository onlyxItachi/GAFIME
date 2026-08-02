#ifndef GAFIME_CUDA_PRECISION_KERNELS_CUH
#define GAFIME_CUDA_PRECISION_KERNELS_CUH

// Profile-specialized CUDA kernels used exclusively by the additive precision
// ABI.  The v1 ABI continues to use kernels.cuh while callers migrate through
// the versioned surfaces in gafime_gpu_abi.hpp.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_api.hpp"
#include "kernels.cuh"

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

// CUDA-local test instrumentation.  This is deliberately not part of the
// public C ABI: it lets the payload's physical smoke prove that the resident
// statistics, descriptor cache, and graph replay state all carry the profile
// identity that is also used for cache matching.
struct PrecisionCudaMatrixIdentity {
    uint32_t profile;
    uint32_t feature_stats_profile;
    uint32_t target_stats_profile;
    uint32_t descriptor_profile;
    uint32_t graph_profile;
    uint32_t graph_valid;
    uint32_t storage_bytes;
    uint32_t accumulation_bytes;
    uint32_t result_bytes;
    uint64_t feature_generation;
    uint64_t target_generation;
    uint64_t descriptor_generation;
    uint64_t graph_metric_signature;
    uintptr_t resident_features;
    uintptr_t resident_target;
    uintptr_t descriptor_combos;
    uintptr_t graph_exec;
};

int inspect_precision_cuda_matrix(
    GafimeGpuMatrix matrix_handle,
    PrecisionCudaMatrixIdentity* identity_out
);

}  // namespace gafime_cuda_v1::detail

#endif  // GAFIME_CUDA_PRECISION_KERNELS_CUH
