/**
 * GAFIME Metal Backend - C Interface Header
 * 
 * C-compatible declarations for the Metal compute backend.
 * Apple Silicon only (M1/M2/M3/M4 with Metal 3.0 support).
 */

#ifndef GAFIME_METAL_BACKEND_H
#define GAFIME_METAL_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STATUS CODES (shared with CUDA/CPU)
// ============================================================================

#ifndef GAFIME_SUCCESS
#define GAFIME_SUCCESS              0
#define GAFIME_ERROR_INVALID_ARGS  -1
#define GAFIME_ERROR_OUT_OF_MEMORY -3
#define GAFIME_ERROR_KERNEL_FAILED -4
#endif

#define GAFIME_ERROR_METAL_NOT_AVAILABLE -10

// ============================================================================
// DEVICE DETECTION
// ============================================================================

/**
 * Check if Metal compute is available (Apple Silicon only).
 * @return 1 if available, 0 otherwise
 */
int gafime_metal_available(void);

/**
 * Get Metal GPU device info.
 * @param name_out        Buffer for device name (at least 256 chars)
 * @param memory_mb_out   Unified memory available to GPU (MB)
 * @param gpu_family_out  Metal GPU family version
 * @return GAFIME_SUCCESS or error code
 */
int gafime_metal_get_device_info(
    char* name_out,
    int* memory_mb_out,
    int* gpu_family_out
);

// ============================================================================
// BUCKET MANAGEMENT (UMA zero-copy optimized)
// ============================================================================

/**
 * Opaque handle to a Metal compute bucket.
 * On Apple Silicon, buffers use storageModeShared for zero-copy UMA access.
 */
typedef void* GafimeMetalBucket;

/**
 * Allocate a Metal compute bucket.
 * Creates shared MTLBuffers — CPU and GPU share the same physical memory.
 */
int gafime_metal_bucket_alloc(
    int n_samples,
    int n_features,
    GafimeMetalBucket* bucket_out
);

/**
 * Upload feature column to bucket.
 * On UMA, this is a memcpy within the same physical memory — no PCIe transfer.
 */
int gafime_metal_bucket_upload_feature(
    GafimeMetalBucket bucket,
    int feature_index,
    const float* data,
    int n_samples
);

/**
 * Upload target vector to bucket.
 */
int gafime_metal_bucket_upload_target(
    GafimeMetalBucket bucket,
    const float* data,
    int n_samples
);

/**
 * Upload fold mask to bucket.
 */
int gafime_metal_bucket_upload_mask(
    GafimeMetalBucket bucket,
    const uint8_t* data,
    int n_samples
);

/**
 * Compute fused interaction on Metal GPU.
 * Dispatches compute pipeline and writes 12-float stats.
 */
int gafime_metal_bucket_compute(
    GafimeMetalBucket bucket,
    const int* ops,
    int arity,
    const int* interaction_types,
    int val_fold_id,
    float* stats_out
);

/**
 * Free Metal bucket resources.
 */
int gafime_metal_bucket_free(GafimeMetalBucket bucket);

// ============================================================================
// STANDALONE FUSED API
// ============================================================================

/**
 * Standalone fused map-reduce (allocates temp buffers internally).
 * Same signature pattern as GPU/CPU fused interaction.
 */
int gafime_metal_fused_interaction(
    const float** h_inputs,
    const float* h_target,
    const uint8_t* h_mask,
    const int* h_ops,
    int arity,
    int interaction_type,
    int val_fold_id,
    int n_samples,
    float* h_stats
);

#ifdef __cplusplus
}
#endif

#endif // GAFIME_METAL_BACKEND_H
