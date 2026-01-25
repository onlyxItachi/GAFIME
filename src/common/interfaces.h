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

// Opaque handle types
typedef void* GafimePipeline;
typedef void* ContiguousBucket;

// ============================================================================
// UNARY OPERATORS
// ============================================================================

/**
 * Unary operators applied to individual features before combination.
 * Each operator transforms a single value: x' = op(x)
 * 
 * SFU-Heavy (Special Function Unit): LOG, EXP, SQRT, TANH, SIGMOID
 * ALU-Heavy (CUDA Core): IDENTITY, SQUARE, NEGATE, ABS, INVERSE, CUBE
 * Time-Series (Memory + ALU): ROLLING_MEAN, ROLLING_STD
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

// Time-series operators (Memory + ALU)
#define GAFIME_OP_ROLLING_MEAN  11  // x' = mean(x[i-w:i])
#define GAFIME_OP_ROLLING_STD   12  // x' = std(x[i-w:i]) - Welford's algorithm

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
 * Auto-tunes for different GPU architectures (Pascal, Turing, Ampere, Ada).
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
 * @param arity             Number of features to combine (2-5)
 * @param interaction_types Array of (arity-1) interaction types for each pair
 *                          e.g., for A*B+C: [MULT, ADD] 
 * @param val_fold_id       Validation fold ID
 * @param h_stats           Host output array [12 floats]
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_compute(
    GafimeBucket bucket,
    const int* feature_indices,
    const int* ops,
    int arity,
    const int* interaction_types,
    int val_fold_id,
    float* h_stats
);

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
// BATCHED COMPUTE API (Process N interactions in ONE kernel launch)
// ============================================================================

/**
 * Maximum batch size for batched compute.
 */
#define GAFIME_MAX_BATCH_SIZE 1024

/**
 * Compute N feature interactions in ONE kernel launch.
 * 
 * Eliminates per-iteration kernel launch overhead by processing
 * batch_size interactions in parallel.
 * 
 * @param bucket            Pre-allocated bucket with uploaded data
 * @param batch_indices     [N * 2] feature indices for each interaction
 * @param batch_ops         [N * 2] unary operator IDs for each
 * @param batch_interact    [N] interaction types
 * @param batch_size        Number of interactions (1 to 1024)
 * @param val_fold_id       Validation fold ID
 * @param h_stats_batch     Output: [N * 12] statistics array
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_bucket_compute_batch(
    GafimeBucket bucket,
    const int* batch_indices,
    const int* batch_ops,
    const int* batch_interact,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
);

// ============================================================================
// DUAL-ISSUE INTERLEAVED KERNEL (SFU+ALU Parallelism)
// ============================================================================

/**
 * Execute TWO feature interactions in parallel (interleaved pipeline).
 * 
 * Slot A: SFU-heavy operations (log, exp, tanh, sigmoid)
 * Slot B: ALU-heavy operations (square, cube, rolling_mean, rolling_std)
 * 
 * While Slot A stalls on SFU, Slot B executes on CUDA cores - doubling throughput.
 * 
 * @param bucket            Pre-allocated bucket with data
 * @param feature_indices_A Slot A feature indices [arity_A]
 * @param ops_A             Slot A unary operators [arity_A] (prefer SFU: LOG, EXP, TANH)
 * @param arity_A           Slot A feature count (2-5)
 * @param interact_A        Slot A interaction type
 * @param feature_indices_B Slot B feature indices [arity_B]
 * @param ops_B             Slot B unary operators [arity_B] (prefer ALU: SQUARE, CUBE, ROLLING_*)
 * @param arity_B           Slot B feature count (2-5)
 * @param interact_B        Slot B interaction type
 * @param window_size       Rolling window size for time-series operators (0 = disabled)
 * @param val_fold_id       Validation fold ID
 * @param h_stats_A         Host output for Slot A [12 floats]
 * @param h_stats_B         Host output for Slot B [12 floats]
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_interleaved_compute(
    GafimeBucket bucket,
    const int* feature_indices_A,
    const int* ops_A,
    int arity_A,
    int interact_A,
    const int* feature_indices_B,
    const int* ops_B,
    int arity_B,
    int interact_B,
    int window_size,
    int val_fold_id,
    float* h_stats_A,
    float* h_stats_B
);

// ============================================================================
// LEGACY: Fused interaction (allocates per-call, for backwards compatibility)
// ============================================================================

/**
 * Fused feature interaction with per-call allocation (DEPRECATED).
 * 
 * For new code, prefer gafime_bucket_* functions.
 */
GAFIME_API int gafime_fused_interaction(
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

// ============================================================================
// LEGACY API (backwards compatibility)
// ============================================================================

/**
 * Legacy feature interaction (writes full vector to global memory).
 * Prefer gafime_fused_interaction for new code.
 */
GAFIME_API int gafime_feature_interaction_cuda(
    const float* X,
    const float* means,
    float* output,
    const int32_t* combo_indices,
    const int32_t* combo_offsets,
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
);

/**
 * CPU version of feature interaction (OpenMP parallelized).
 */
GAFIME_API int gafime_feature_interaction_cpu(
    const float* X,
    const float* means,
    float* output,
    const int32_t* combo_indices,
    const int32_t* combo_offsets,
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
);

/**
 * Legacy Pearson correlation.
 */
GAFIME_API int gafime_pearson_cuda(
    const float* x,
    const float* y,
    int32_t n,
    float* result_out
);

GAFIME_API int gafime_pearson_cpu(
    const float* x,
    const float* y,
    int32_t n,
    float* result_out
);

// ============================================================================
// ASYNC 4-SLOT RING BUFFER PIPELINE
// ============================================================================
// 
// Enables overlapping batch creation (CPU/Rust) with GPU execution.
// 4 slots allow producer to fill slots while GPU executes others.
// ============================================================================

#define PIPELINE_SLOTS 4
#define PIPELINE_MAX_BATCH 1024

/**
 * Initialize async pipeline with 4 pre-allocated slots.
 * Call once after bucket is ready with data uploaded.
 */
GAFIME_API int gafime_pipeline_init(
    GafimeBucket bucket,
    int val_fold_id,
    GafimePipeline* pipeline_out
);

/**
 * Submit batch to pipeline (non-blocking).
 * Returns GAFIME_ERROR_PIPELINE_FULL if all 4 slots are busy.
 */
GAFIME_API int gafime_pipeline_submit(
    GafimePipeline pipeline,
    const int* h_indices,    // [batch_size * 2]
    const int* h_ops,        // [batch_size * 2]
    const int* h_interact,   // [batch_size]
    int batch_size,
    int* slot_id_out
);

/**
 * Poll for completed results (non-blocking).
 * Returns GAFIME_ERROR_NO_RESULT if nothing is ready yet.
 */
GAFIME_API int gafime_pipeline_poll(
    GafimePipeline pipeline,
    float* h_stats_out,      // [batch_size * 12]
    int* batch_size_out
);

/**
 * Wait for next result (blocking).
 */
GAFIME_API int gafime_pipeline_wait(
    GafimePipeline pipeline,
    float* h_stats_out,
    int* batch_size_out
);

/**
 * Get number of pending batches in pipeline (0-4).
 */
GAFIME_API int gafime_pipeline_pending(GafimePipeline pipeline);

/**
 * Flush all pending batches and collect results (blocking).
 */
GAFIME_API int gafime_pipeline_flush(
    GafimePipeline pipeline,
    float* h_all_stats_out,
    int* total_batch_size_out
);

/**
 * Free pipeline resources.
 */
GAFIME_API int gafime_pipeline_free(GafimePipeline pipeline);

// ============================================================================
// CONTIGUOUS MEMORY API
// ============================================================================

/**
 * Allocate contiguous bucket with single VRAM allocation.
 */
GAFIME_API int gafime_contiguous_bucket_alloc(
    int n_samples,
    int n_features,
    ContiguousBucket* bucket_out
);

/**
 * Upload all data to contiguous bucket.
 */
GAFIME_API int gafime_contiguous_bucket_upload(
    ContiguousBucket bucket,
    const float* h_data,
    const uint8_t* h_mask
);

/**
 * Compute interaction on contiguous bucket.
 */
GAFIME_API int gafime_contiguous_bucket_compute(
    ContiguousBucket bucket,
    int feature_a_idx,
    int feature_b_idx,
    int op_a,
    int op_b,
    int interact_type,
    int val_fold_id,
    float* h_stats_out
);

/**
 * Free contiguous bucket resources.
 */
GAFIME_API int gafime_contiguous_bucket_free(ContiguousBucket bucket);

/**
 * Get contiguous bucket info.
 */
GAFIME_API int gafime_contiguous_bucket_info(
    ContiguousBucket bucket,
    int* n_samples_out,
    int* n_features_out
);

#ifdef __cplusplus
}
#endif

#endif // GAFIME_INTERFACES_H
