#ifndef GAFIME_GPU_INTERNAL_ABI_HPP
#define GAFIME_GPU_INTERNAL_ABI_HPP

/*
 * Private host adapters behind the canonical generic ABI in
 * gafime_gpu_abi.hpp. These layouts are not exported ABI 1.1 types and must
 * never appear in the dynamic symbol surface. They let one generic checked
 * boundary reuse statically specialized f32/f64 hot loops without duplicating
 * those loops.
 */

#include "gafime_gpu_abi.hpp"

typedef struct GafimePrecisionMatrixDesc {
    uint32_t abi_version;
    uint32_t profile;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
    uint64_t reserved[8];
} GafimePrecisionMatrixDesc;

typedef struct GafimePrecisionCapabilities {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t profile_mask;
    uint32_t storage_dtype_mask;
    uint32_t result_dtype_mask;
    uint32_t flags;
    uint64_t reserved[8];
} GafimePrecisionCapabilities;

typedef struct GafimePrecisionLaunchProtocol {
    uint32_t abi_version;
    uint32_t profile;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
} GafimePrecisionLaunchProtocol;

typedef struct GafimeResultTableF64 {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    double* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    void* backend_private;
    uint64_t reserved[8];
} GafimeResultTableF64;

typedef struct GafimePermutationSignificanceTableF64 {
    uint32_t abi_version;
    uint32_t metric_count;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    const double* observed_metric_values;
    double* p_values;
    uint64_t reserved[8];
} GafimePermutationSignificanceTableF64;

#endif  // GAFIME_GPU_INTERNAL_ABI_HPP
