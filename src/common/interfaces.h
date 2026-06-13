/**
 * GAFIME Native Backend Interface Definitions
 * 
 * Common C interfaces for both CUDA and CPU backends.
 * All functions use extern "C" for ctypes compatibility.
 */

#ifndef GAFIME_INTERFACES_H
#define GAFIME_INTERFACES_H

#include <stdint.h>

// Windows DLL export macro
#ifdef _WIN32
    #ifdef GAFIME_BUILDING_DLL
        #define GAFIME_API __declspec(dllexport)
    #else
        #define GAFIME_API __declspec(dllimport)
    #endif
#else
    #define GAFIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STATUS CODES
// ============================================================================

#define GAFIME_SUCCESS 0
#define GAFIME_ERROR_INVALID_ARGS -1
#define GAFIME_ERROR_CUDA_NOT_AVAILABLE -2
#define GAFIME_ERROR_OUT_OF_MEMORY -3
#define GAFIME_ERROR_KERNEL_FAILED -4
#define GAFIME_ERROR_PIPELINE_FULL -5
#define GAFIME_ERROR_NO_RESULT -6
#define GAFIME_ERROR_HIP_NOT_AVAILABLE -7

// Opaque handle types
typedef void* GafimeCudaMatrix;
typedef void* GafimeRocmMatrix;
typedef void* GafimeRocmBucket;

// ============================================================================
// UNARY OPERATORS
// ============================================================================

/**
 * Unary operators applied to individual features before combination.
 * Each operator transforms a single value: x' = op(x)
 * 
 * SFU-Heavy (Special Function Unit): LOG, EXP, SQRT, TANH, SIGMOID
 * ALU-Heavy (CUDA Core): IDENTITY, SQUARE, NEGATE, ABS, INVERSE, CUBE
 */
// Point operators (SFU-heavy)
#define GAFIME_OP_IDENTITY  0   // x' = x (ALU)
#define GAFIME_OP_LOG       1   // x' = log(|x| + eps) (SFU)
#define GAFIME_OP_EXP       2   // x' = exp(clamp(x)) (SFU)
#define GAFIME_OP_SQRT      3   // x' = sqrt(|x|) (SFU)
#define GAFIME_OP_TANH      4   // x' = tanh(x) (SFU)
#define GAFIME_OP_SIGMOID   5   // x' = 1 / (1 + exp(-x)) (SFU)

// Point operators (ALU-heavy)
#define GAFIME_OP_SQUARE    6   // x' = x^2 (ALU)
#define GAFIME_OP_NEGATE    7   // x' = -x (ALU)
#define GAFIME_OP_ABS       8   // x' = |x| (ALU)
#define GAFIME_OP_INVERSE   9   // x' = 1/x (ALU)
#define GAFIME_OP_CUBE      10  // x' = x^3 (ALU)

// ============================================================================
// INTERACTION TYPES
// ============================================================================

/**
 * Binary operators for combining transformed features.
 * For arity > 2, applied sequentially: result = op(op(x0, x1), x2)...
 */
#define GAFIME_INTERACT_MULT  0   // X = x0 * x1 * ...
#define GAFIME_INTERACT_ADD   1   // X = x0 + x1 + ...
#define GAFIME_INTERACT_SUB   2   // X = x0 - x1 (arity=2)
#define GAFIME_INTERACT_DIV   3   // X = x0 / x1 (arity=2, safe)
#define GAFIME_INTERACT_MAX   4   // X = max(x0, x1, ...)
#define GAFIME_INTERACT_MIN   5   // X = min(x0, x1, ...)

// ============================================================================
// EXPLICIT FEATURE CANDIDATE KINDS
// ============================================================================

#define GAFIME_CANDIDATE_CONTINUOUS          0
#define GAFIME_CANDIDATE_TS_LAG              1
#define GAFIME_CANDIDATE_TS_DELTA            2
#define GAFIME_CANDIDATE_TS_VELOCITY         3
#define GAFIME_CANDIDATE_TS_ACCELERATION     4
#define GAFIME_CANDIDATE_TS_ROLLING_MEAN     5
#define GAFIME_CANDIDATE_TS_ROLLING_STD      6
#define GAFIME_CANDIDATE_TS_ROLLING_SUM      7

// ============================================================================
// STATISTICS OUTPUT LAYOUT
// ============================================================================

/**
 * Output stats array layout (12 floats):
 * 
 * Train split (first 6):
 *   [0] N      - count of training samples
 *   [1] ΣX     - sum of interaction values
 *   [2] ΣY     - sum of target values
 *   [3] ΣX²    - sum of squared interactions
 *   [4] ΣY²    - sum of squared targets
 *   [5] ΣXY    - sum of interaction * target
 * 
 * Validation split (next 6):
 *   [6-11] same as above for validation fold
 * 
 * Pearson formula: r = (NΣxy - ΣxΣy) / sqrt((NΣx² - (Σx)²)(NΣy² - (Σy)²))
 */
#define GAFIME_STATS_SIZE 12

#define GAFIME_STAT_TRAIN_N     0
#define GAFIME_STAT_TRAIN_SX    1
#define GAFIME_STAT_TRAIN_SY    2
#define GAFIME_STAT_TRAIN_SXX   3
#define GAFIME_STAT_TRAIN_SYY   4
#define GAFIME_STAT_TRAIN_SXY   5
#define GAFIME_STAT_VAL_N       6
#define GAFIME_STAT_VAL_SX      7
#define GAFIME_STAT_VAL_SY      8
#define GAFIME_STAT_VAL_SXX     9
#define GAFIME_STAT_VAL_SYY     10
#define GAFIME_STAT_VAL_SXY     11

// ============================================================================
// DEVICE FUNCTIONS
// ============================================================================

/**
 * Check if CUDA is available on this system.
 * @return 1 if CUDA is available, 0 otherwise
 */
GAFIME_API int gafime_cuda_available(void);

/**
 * Get GPU device information.
 * @param device_id GPU device index
 * @param name_out Buffer for device name (at least 256 chars)
 * @param memory_mb_out Total memory in MB
 * @param compute_cap_major_out Compute capability major version
 * @param compute_cap_minor_out Compute capability minor version
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_get_device_info(
    int device_id,
    char* name_out,
    int* memory_mb_out,
    int* compute_cap_major_out,
    int* compute_cap_minor_out
);

/**
 * Get GPU auto-tuned configuration.
 * 
 * Queries GPU properties and returns optimal kernel parameters.
 * Auto-tunes for different GPU architectures (Turing, Ampere, Ada, Hopper, Blackwell).
 * 
 * @param block_size_out     Optimal threads per block
 * @param max_blocks_out     Max blocks for grid
 * @param sm_count_out       Number of streaming multiprocessors
 * @param compute_major_out  Compute capability major
 * @param compute_minor_out  Compute capability minor
 * @param l2_cache_bytes_out L2 cache size in bytes
 * @param gpu_name_out       GPU name (at least 256 chars)
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_get_gpu_config(
    int* block_size_out,
    int* max_blocks_out,
    int* sm_count_out,
    int* compute_major_out,
    int* compute_minor_out,
    int* l2_cache_bytes_out,
    char* gpu_name_out
);

// ============================================================================
// STATIC VRAM BUCKET MANAGEMENT
// ============================================================================

/**
 * Opaque handle to a pre-allocated VRAM bucket.
 * Stores device pointers for features, target, mask, and stats.
 */
typedef void* GafimeBucket;

/**
 * Maximum number of feature columns in a bucket.
 */
#define GAFIME_MAX_FEATURES 5

/**
 * Allocate a static VRAM bucket.
 * 
 * The bucket pre-allocates all GPU memory needed for n_samples and n_features.
 * Call this ONCE at initialization, then use the bucket for millions of iterations.
 * 
 * @param n_samples     Number of samples (rows)
 * @param n_features    Number of feature columns (max 5)
 * @param bucket_out    Output: handle to the allocated bucket
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_alloc(
    int n_samples,
    int n_features,
    GafimeBucket* bucket_out
);

/**
 * Upload feature data to the bucket (host -> device copy).
 * 
 * Call this when data changes (new batch from streamer).
 * Feature columns can be uploaded individually (feature_idx) or use -1 for all.
 * 
 * @param bucket        The bucket handle
 * @param feature_idx   Which feature to upload (0 to n_features-1), or -1 for all
 * @param h_data        Host pointer to feature data [n_samples] float32
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_upload_feature(
    GafimeBucket bucket,
    int feature_idx,
    const float* h_data
);

/**
 * Upload target vector to the bucket.
 * 
 * @param bucket        The bucket handle
 * @param h_target      Host pointer to target data [n_samples] float32
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_upload_target(
    GafimeBucket bucket,
    const float* h_target
);

/**
 * Upload fold mask to the bucket.
 * 
 * @param bucket        The bucket handle
 * @param h_mask        Host pointer to mask data [n_samples] uint8
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_upload_mask(
    GafimeBucket bucket,
    const uint8_t* h_mask
);

/**
 * Execute fused computation on pre-uploaded bucket data.
 * 
 * NO cudaMalloc/cudaFree inside! All memory is pre-allocated in the bucket.
 * Safe to call millions of times in a tight loop.
 * 
 * @param bucket            The bucket handle (with data already uploaded)
 * @param feature_indices   Which features to use [arity] (0 to n_features-1)
 * @param ops               Unary operator IDs for each feature [arity]
 * @param arity             Number of features to combine (1-5)
 * @param interaction_types Array of (arity-1) interaction types for each pair
 *                          e.g., for A*B+C: [MULT, ADD] 
 * @param val_fold_id       Validation fold ID
 * @param h_stats           Host output array [12 floats]
 * @return GAFIME_SUCCESS or error code
 */
/**
 * Free the VRAM bucket and all associated GPU memory.
 * 
 * Call this ONCE at shutdown.
 * 
 * @param bucket        The bucket handle to free
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_free(GafimeBucket bucket);

// ============================================================================
// LOCAL BUCKET BATCHED COMPUTE API
// ============================================================================

/**
 * Maximum batch size for batched compute.
 */
#define GAFIME_MAX_BATCH_SIZE 1024

/**
 * Compute N local-bucket feature interactions in ONE kernel launch.
 * 
 * This API uses bucket-local feature slots. It is intended for compact working
 * sets such as explicit time-series transforms. Broad continuous scans should
 * use the full-matrix API below so arity-5 candidates can reference global
 * feature indices without being isolated into one-candidate buckets.
 * 
 * @param bucket            Pre-allocated bucket with uploaded data
 * @param batch_kinds       [N] candidate kind IDs
 * @param batch_indices     [N * arity] feature indices for each interaction
 * @param batch_ops         [N * arity] unary operator IDs for each
 * @param batch_interact    [N * (arity - 1)] interaction types
 * @param batch_ts_params   [N * 4] time-series parameters (lag/window/etc.)
 * @param arity             Homogeneous candidate arity for this batch (1-5)
 * @param batch_size        Number of interactions (1 to 1024)
 * @param val_fold_id       Validation fold ID
 * @param h_stats_batch     Output: [N * 12] statistics array
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_compute_batch(
    GafimeBucket bucket,
    const int* batch_kinds,
    const int* batch_indices,
    const int* batch_ops,
    const int* batch_interact,
    const int* batch_ts_params,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
);

/**
 * Allocate a full column-major CUDA matrix for broad continuous scans.
 *
 * Continuous feature interactions use global feature indices in
 * gafime_cuda_matrix_compute_batch. This keeps X/y/mask/means resident on the
 * device and lets Rust schedule cache-local homogeneous arity batches without
 * the old 5-slot bucket constraint.
 */
GAFIME_API int gafime_cuda_matrix_alloc(
    int n_samples,
    int n_features,
    int max_batch_size,
    GafimeCudaMatrix* matrix_out
);

GAFIME_API int gafime_cuda_matrix_upload(
    GafimeCudaMatrix matrix,
    const float* h_X_colmajor,
    const float* h_y,
    const uint8_t* h_mask,
    const float* h_means
);

GAFIME_API int gafime_cuda_matrix_compute_batch(
    GafimeCudaMatrix matrix,
    const int* h_batch_indices,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
);

GAFIME_API int gafime_cuda_matrix_free(GafimeCudaMatrix matrix);

// ============================================================================
// DISCRETE SOFT FUNCTION FAMILY
// ============================================================================

#define GAFIME_DISCRETE_SOFT_THRESHOLD          0
#define GAFIME_DISCRETE_SOFT_INTERVAL           1
#define GAFIME_DISCRETE_VALUE_GATED_THRESHOLD   2
#define GAFIME_DISCRETE_SOFT_RECTANGLE          3
#define GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE 4

#define GAFIME_DISCRETE_DIRECTION_GE 0
#define GAFIME_DISCRETE_DIRECTION_LE 1

#define GAFIME_SELECTION_SCORE_SIZE 4
#define GAFIME_SELECTION_MUTUAL_INFO 0
#define GAFIME_SELECTION_VARIANCE_REDUCTION 1
#define GAFIME_SELECTION_RESIDUAL_ABS_CORR 2
#define GAFIME_SELECTION_RESIDUAL_R2_GAIN 3

/**
 * CUDA soft discrete feature family batch API.
 *
 * X is column-major [n_features][n_samples]. Each candidate writes one
 * 12-float stats block using the standard GAFIME_STATS_SIZE layout. This API
 * intentionally supports only soft/vectorized discrete functions for GPU use.
 */
GAFIME_API int gafime_discrete_soft_batch_cuda(
    const float* X,
    const float* y,
    const int* kinds,
    const int* feature_a,
    const int* feature_b,
    const int* value_feature,
    const int* directions,
    const float* params,
    const float* scales,
    const float* sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    float* h_stats_batch
);

/**
 * CUDA split-aware discrete selection scoring API.
 *
 * X is column-major [n_features][n_samples]. This computes selection/ranking
 * diagnostics for soft discrete candidates:
 *   - binned mutual information between candidate mask and target,
 *   - soft variance/impurity reduction from candidate mask,
 *   - residual absolute correlation using the candidate feature value,
 *   - residual R2 gain (squared residual correlation).
 *
 * Output layout: [n_candidates][GAFIME_SELECTION_SCORE_SIZE].
 */
GAFIME_API int gafime_discrete_selection_batch_cuda(
    const float* X,
    const float* y,
    const float* residual,
    const int* kinds,
    const int* feature_a,
    const int* feature_b,
    const int* value_feature,
    const int* directions,
    const float* params,
    const float* scales,
    const float* sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    int mi_bins,
    float y_min,
    float y_max,
    float y_sum,
    float y_sq_sum,
    float* h_scores_batch
);

/**
 * CUDA split-aware discrete selection scoring API using adaptive soft-binary
 * MI box templates.
 *
 * This receives precomputed target-bin ids and an explicit compile-time
 * histogram template in {2,4,8,16,32,64,96}. Batches should be homogeneous by
 * template so the host dispatch can switch directly to the matching static
 * CUDA kernel shape.
 */
GAFIME_API int gafime_discrete_selection_adaptive_cuda(
    const float* X,
    const float* y,
    const float* residual,
    const int* y_bins,
    const int* kinds,
    const int* feature_a,
    const int* feature_b,
    const int* value_feature,
    const int* directions,
    const float* params,
    const float* scales,
    const float* sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    int target_bin_template,
    float y_sum,
    float y_sq_sum,
    float* h_scores_batch
);

// ============================================================================
// ROCm/HIP BACKEND API
// ============================================================================

GAFIME_API int gafime_rocm_available(void);

GAFIME_API int gafime_rocm_get_device_info(
    int device_id,
    char* name_out,
    int* memory_mb_out,
    int* compute_cap_major_out,
    int* compute_cap_minor_out
);

GAFIME_API int gafime_rocm_get_gpu_config(
    int* block_size_out,
    int* max_blocks_out,
    int* sm_count_out,
    int* compute_major_out,
    int* compute_minor_out,
    int* l2_cache_bytes_out,
    char* gpu_name_out
);

GAFIME_API int gafime_rocm_bucket_alloc(
    int n_samples,
    int n_features,
    GafimeRocmBucket* bucket_out
);

GAFIME_API int gafime_rocm_bucket_upload_feature(
    GafimeRocmBucket bucket,
    int feature_idx,
    const float* h_data
);

GAFIME_API int gafime_rocm_bucket_upload_target(
    GafimeRocmBucket bucket,
    const float* h_target
);

GAFIME_API int gafime_rocm_bucket_upload_mask(
    GafimeRocmBucket bucket,
    const uint8_t* h_mask
);

GAFIME_API int gafime_rocm_bucket_compute_batch(
    GafimeRocmBucket bucket,
    const int* batch_kinds,
    const int* batch_indices,
    const int* batch_ops,
    const int* batch_interact,
    const int* batch_ts_params,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
);

GAFIME_API int gafime_rocm_bucket_free(GafimeRocmBucket bucket);

GAFIME_API int gafime_rocm_matrix_alloc(
    int n_samples,
    int n_features,
    int max_batch_size,
    GafimeRocmMatrix* matrix_out
);

GAFIME_API int gafime_rocm_matrix_upload(
    GafimeRocmMatrix matrix,
    const float* h_X_colmajor,
    const float* h_y,
    const uint8_t* h_mask,
    const float* h_means
);

GAFIME_API int gafime_rocm_matrix_compute_batch(
    GafimeRocmMatrix matrix,
    const int* h_batch_indices,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
);

GAFIME_API int gafime_rocm_matrix_free(GafimeRocmMatrix matrix);

GAFIME_API int gafime_discrete_soft_batch_rocm(
    const float* X,
    const float* y,
    const int* kinds,
    const int* feature_a,
    const int* feature_b,
    const int* value_feature,
    const int* directions,
    const float* params,
    const float* scales,
    const float* sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    float* h_stats_batch
);

GAFIME_API int gafime_discrete_selection_adaptive_rocm(
    const float* X,
    const float* y,
    const float* residual,
    const int* y_bins,
    const int* kinds,
    const int* feature_a,
    const int* feature_b,
    const int* value_feature,
    const int* directions,
    const float* params,
    const float* scales,
    const float* sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    int target_bin_template,
    float y_sum,
    float y_sq_sum,
    float* h_scores_batch
);

// ============================================================================
// TENSOR CORE FUTUREPROOFING
// ============================================================================

/**
 * Check if tensor core acceleration is available.
 * @param precision_mode Output: 0=unavailable, 1=FP16, 2=TF32, 3=FP8
 * @return 1 if tensor cores available, 0 otherwise
 */
GAFIME_API int gafime_tensor_core_available(int* precision_mode);

#ifdef __cplusplus
}
#endif

#endif // GAFIME_INTERFACES_H
