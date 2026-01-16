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

// Status codes
#define GAFIME_SUCCESS 0
#define GAFIME_ERROR_INVALID_ARGS -1
#define GAFIME_ERROR_CUDA_NOT_AVAILABLE -2
#define GAFIME_ERROR_OUT_OF_MEMORY -3
#define GAFIME_ERROR_KERNEL_FAILED -4

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
 * Compute feature interaction vectors for a batch of combinations.
 * 
 * For each combination, computes the product of centered features:
 *   output[i, c] = prod_{j in combo[c]} (X[i, j] - mean[j])
 * 
 * @param X Feature matrix [n_samples x n_features], row-major, float32
 * @param means Pre-computed feature means [n_features], float32
 * @param output Output matrix [n_samples x n_combos], row-major, float32
 * @param combo_indices Flat array of feature indices for all combos
 * @param combo_offsets Start offsets for each combo (length n_combos + 1)
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @param n_combos Number of combinations
 * @return GAFIME_SUCCESS or error code
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
 * Same signature as CUDA version.
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
 * Compute Pearson correlation between two vectors.
 * @param x First vector [n], float32
 * @param y Second vector [n], float32
 * @param n Vector length
 * @param result_out Pointer to store result
 * @return GAFIME_SUCCESS or error code
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
