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

// ============================================================================
// UNARY OPERATORS
// ============================================================================

/**
 * Unary operators applied to individual features before combination.
 * Each operator transforms a single value: x' = op(x)
 */
#define GAFIME_OP_IDENTITY  0   // x' = x
#define GAFIME_OP_LOG       1   // x' = log(|x| + eps)
#define GAFIME_OP_EXP       2   // x' = exp(clamp(x))
#define GAFIME_OP_SQRT      3   // x' = sqrt(|x|)
#define GAFIME_OP_TANH      4   // x' = tanh(x)
#define GAFIME_OP_SIGMOID   5   // x' = 1 / (1 + exp(-x))
#define GAFIME_OP_SQUARE    6   // x' = x^2
#define GAFIME_OP_NEGATE    7   // x' = -x
#define GAFIME_OP_ABS       8   // x' = |x|
#define GAFIME_OP_INVERSE   9   // x' = 1/x (with safety)
#define GAFIME_OP_CUBE      10  // x' = x^3

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

// ============================================================================
// NEW: FUSED MAP-REDUCE KERNEL
// ============================================================================

/**
 * Fused feature interaction kernel with on-chip reduction.
 * 
 * Transforms features with unary ops, combines with interaction op,
 * then reduces to summary statistics (no intermediate global memory).
 * 
 * @param h_inputs     Array of device pointers to feature columns [arity]
 * @param d_target     Device pointer to target vector [n_samples]
 * @param d_mask       Device pointer to fold mask [n_samples], uint8
 * @param h_ops        Host array of unary operator IDs [arity]
 * @param arity        Number of features to combine (2-5)
 * @param interaction_type  GAFIME_INTERACT_* constant
 * @param val_fold_id  Validation fold ID (samples with mask[i]==val_fold_id go to val stats)
 * @param n_samples    Number of samples
 * @param h_stats      Host output array [12 floats] for train/val statistics
 * 
 * @return GAFIME_SUCCESS or error code
 */
GAFIME_API int gafime_fused_interaction(
    const float** h_inputs,
    const float* d_target,
    const uint8_t* d_mask,
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

#ifdef __cplusplus
}
#endif

#endif // GAFIME_INTERFACES_H
