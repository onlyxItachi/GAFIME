/*
 * External consumer for the pre-freeze, dtype-suffixed ABI 1.1 surface.
 *
 * This source deliberately does not include the repository ABI header.  Its
 * declarations freeze the exact public layouts exported by the preserved
 * PR #70 baseline so comparative evidence cannot silently compile against the
 * candidate generic numeric-route ABI.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "abi_dynamic_load.h"

#define GAFIME_ABI_1_0 ((1u << 16) | 0u)
#define GAFIME_ABI_1_1 ((1u << 16) | 1u)
#define GAFIME_STATUS_OK 0
#define GAFIME_STATUS_UNSUPPORTED_BACKEND -2
#define GAFIME_STATUS_DEVICE_ERROR -4
#define GAFIME_DTYPE_F32 1u
#define GAFIME_DTYPE_F64 2u
#define GAFIME_DTYPE_MASK_F32 0x1u
#define GAFIME_DTYPE_MASK_F64 0x2u
#define GAFIME_PROFILE_FP32 1u
#define GAFIME_PROFILE_MIXED 2u
#define GAFIME_PROFILE_FP64 3u
#define GAFIME_PROFILE_MASK_FP32 0x1u
#define GAFIME_PROFILE_MASK_MIXED 0x2u
#define GAFIME_PROFILE_MASK_FP64 0x4u
#define GAFIME_MATRIX_ROW_MAJOR 1u
#define GAFIME_METRIC_PEARSON 1u
#define GAFIME_FAMILY_CONTINUOUS 1u

typedef void* GafimeGpuMatrix11;

typedef struct SliceU32_11 {
    const uint32_t* ptr;
    uint64_t len;
} SliceU32_11;

typedef struct SliceU64_11 {
    const uint64_t* ptr;
    uint64_t len;
} SliceU64_11;

typedef struct PrecisionMatrixDesc_11 {
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
} PrecisionMatrixDesc_11;

typedef struct PrecisionCapabilities_11 {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t profile_mask;
    uint32_t storage_dtype_mask;
    uint32_t result_dtype_mask;
    uint32_t flags;
    uint64_t reserved[8];
} PrecisionCapabilities_11;

typedef struct ArityChunk_10 {
    uint32_t arity;
    uint32_t family;
    uint32_t metric_mask;
    uint32_t shape_hint_index;
    uint64_t combo_row_offset;
    uint64_t combo_count;
    uint32_t local_chunk_id;
    uint32_t flags;
    uint64_t descriptor_offset;
    uint64_t descriptor_count;
} ArityChunk_10;

typedef struct RankSpec_10 {
    uint32_t top_k;
    uint32_t primary_metric;
    uint32_t descending;
    uint32_t include_ties;
    uint64_t reserved[4];
} RankSpec_10;

typedef struct PermutationSchedule_10 {
    uint32_t permutation_count;
    uint32_t mode;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t seed;
    SliceU64_11 target_offsets;
    uint64_t reserved[4];
} PermutationSchedule_10;

typedef struct LaunchProtocol_10 {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t flags;
    uint32_t max_arity;
    uint64_t n_samples;
    uint32_t n_features;
    uint32_t family_count;
    SliceU32_11 combo_indices;
    SliceU32_11 metric_ids;
    const ArityChunk_10* chunks;
    uint32_t chunk_count;
    uint32_t reserved32_a;
    const void* shape_hints;
    uint32_t shape_hint_count;
    uint32_t reserved32_b;
    RankSpec_10 rank;
    PermutationSchedule_10 permutations;
    uint64_t reserved[8];
} LaunchProtocol_10;

typedef struct PrecisionLaunchProtocol_11 {
    uint32_t abi_version;
    uint32_t profile;
    const LaunchProtocol_10* base;
    uint64_t reserved[8];
} PrecisionLaunchProtocol_11;

#define DEFINE_RESULT_TABLE(name, value_type) \
    typedef struct name {                     \
        uint32_t abi_version;                 \
        uint32_t max_arity;                   \
        uint32_t metric_count;                \
        uint32_t flags;                       \
        uint64_t capacity;                    \
        uint64_t row_count;                   \
        uint32_t* combo_indices;              \
        value_type* metric_values;            \
        uint32_t* ranks;                      \
        uint32_t* families;                   \
        uint64_t* candidate_ids;              \
        uint32_t* row_flags;                  \
        void* backend_private;                \
        uint64_t reserved[8];                 \
    } name

DEFINE_RESULT_TABLE(ResultTableF32_11, float);
DEFINE_RESULT_TABLE(ResultTableF64_11, double);

#define DEFINE_SIGNIFICANCE_TABLE(name, value_type) \
    typedef struct name {                           \
        uint32_t abi_version;                       \
        uint32_t metric_count;                      \
        uint64_t row_count;                         \
        const uint64_t* candidate_ids;              \
        const value_type* observed_metric_values;   \
        value_type* p_values;                       \
        uint64_t reserved[8];                       \
    } name

DEFINE_SIGNIFICANCE_TABLE(SignificanceF32_11, float);
DEFINE_SIGNIFICANCE_TABLE(SignificanceF64_11, double);

typedef struct InteractionDiagnosticBatch_11 {
    uint32_t abi_version;
    uint32_t max_arity;
    uint64_t row_count;
    const uint32_t* combo_indices;
    uint64_t combo_index_count;
    uint64_t* overflow_row_counts;
    uint32_t* flags;
    uint32_t reserved32;
    uint64_t reserved[7];
} InteractionDiagnosticBatch_11;

_Static_assert(sizeof(PrecisionMatrixDesc_11) == 112,
               "typed ABI 1.1 matrix size drifted");
_Static_assert(_Alignof(PrecisionMatrixDesc_11) == 8,
               "typed ABI 1.1 matrix alignment drifted");
_Static_assert(offsetof(PrecisionMatrixDesc_11, rows) == 24,
               "typed ABI 1.1 matrix row offset drifted");
_Static_assert(offsetof(PrecisionMatrixDesc_11, reserved) == 48,
               "typed ABI 1.1 matrix reserved offset drifted");
_Static_assert(sizeof(PrecisionCapabilities_11) == 88,
               "typed ABI 1.1 capability size drifted");
_Static_assert(sizeof(ArityChunk_10) == 56, "ABI 1.0 chunk size drifted");
_Static_assert(sizeof(RankSpec_10) == 48, "ABI 1.0 rank size drifted");
_Static_assert(sizeof(PermutationSchedule_10) == 72,
               "ABI 1.0 permutation size drifted");
_Static_assert(sizeof(LaunchProtocol_10) == 280,
               "ABI 1.0 launch size drifted");
_Static_assert(offsetof(LaunchProtocol_10, rank) == 96,
               "ABI 1.0 rank offset drifted");
_Static_assert(offsetof(LaunchProtocol_10, permutations) == 144,
               "ABI 1.0 permutation offset drifted");
_Static_assert(sizeof(PrecisionLaunchProtocol_11) == 80,
               "typed ABI 1.1 protocol size drifted");
_Static_assert(offsetof(PrecisionLaunchProtocol_11, base) == 8,
               "typed ABI 1.1 protocol base offset drifted");
_Static_assert(sizeof(ResultTableF32_11) == 152,
               "typed ABI 1.1 f32 result size drifted");
_Static_assert(sizeof(ResultTableF64_11) == 152,
               "typed ABI 1.1 f64 result size drifted");
_Static_assert(offsetof(ResultTableF32_11, metric_values) == 40,
               "typed ABI 1.1 result value offset drifted");
_Static_assert(sizeof(SignificanceF32_11) == 104,
               "typed ABI 1.1 f32 significance size drifted");
_Static_assert(sizeof(SignificanceF64_11) == 104,
               "typed ABI 1.1 f64 significance size drifted");
_Static_assert(sizeof(InteractionDiagnosticBatch_11) == 112,
               "typed ABI 1.1 diagnostic size drifted");

typedef int (*CapabilitiesFn)(uint32_t, PrecisionCapabilities_11*);
typedef int (*AllocFn)(uint32_t, const PrecisionMatrixDesc_11*, GafimeGpuMatrix11*);
typedef int (*UploadF32Fn)(GafimeGpuMatrix11, const float*, const float*, uint64_t,
                           uint32_t);
typedef int (*UploadF64Fn)(GafimeGpuMatrix11, const double*, const double*, uint64_t,
                           uint32_t);
typedef int (*UpdateF32Fn)(GafimeGpuMatrix11, const float*, uint64_t);
typedef int (*UpdateF64Fn)(GafimeGpuMatrix11, const double*, uint64_t);
typedef int (*ExecuteF32Fn)(GafimeGpuMatrix11, const PrecisionLaunchProtocol_11*,
                            ResultTableF32_11*);
typedef int (*ExecuteF64Fn)(GafimeGpuMatrix11, const PrecisionLaunchProtocol_11*,
                            ResultTableF64_11*);
typedef int (*ExecutionMemoryFn)(GafimeGpuMatrix11, const PrecisionLaunchProtocol_11*,
                                 uint64_t*);
typedef int (*PermutationMemoryFn)(GafimeGpuMatrix11,
                                   const PrecisionLaunchProtocol_11*, uint64_t,
                                   uint64_t*);
typedef int (*PermutationF32Fn)(GafimeGpuMatrix11, const PrecisionLaunchProtocol_11*,
                                SignificanceF32_11*);
typedef int (*PermutationF64Fn)(GafimeGpuMatrix11, const PrecisionLaunchProtocol_11*,
                                SignificanceF64_11*);
typedef int (*DiagnosticsFn)(GafimeGpuMatrix11, InteractionDiagnosticBatch_11*);
typedef void (*FreeFn)(GafimeGpuMatrix11);

typedef struct TypedApi11 {
    CapabilitiesFn capabilities;
    AllocFn alloc;
    UploadF32Fn upload_f32;
    UploadF64Fn upload_f64;
    UpdateF32Fn update_f32;
    UpdateF64Fn update_f64;
    ExecuteF32Fn execute_f32;
    ExecuteF64Fn execute_f64;
    ExecutionMemoryFn execution_memory;
    PermutationMemoryFn permutation_memory;
    PermutationF32Fn permutation_f32;
    PermutationF64Fn permutation_f64;
    DiagnosticsFn diagnostics;
    FreeFn free_matrix;
} TypedApi11;

#define LOAD_OPTIONAL_FUNCTION(library, destination, symbol_name)                \
    do {                                                                         \
        GafimeTestSymbol typed_symbol =                                          \
            gafime_test_library_symbol((library), (symbol_name));                \
        _Static_assert(sizeof(destination) == sizeof(typed_symbol),              \
                       "function and dynamic-symbol pointers must match");      \
        if (typed_symbol != NULL) {                                               \
            memcpy(&(destination), &typed_symbol, sizeof(destination));          \
        }                                                                        \
    } while (0)

static int unavailable_status(int status) {
    return status == GAFIME_STATUS_UNSUPPORTED_BACKEND ||
        status == GAFIME_STATUS_DEVICE_ERROR;
}

static uint32_t profile_bit(uint32_t profile) {
    return profile == GAFIME_PROFILE_FP32 ? GAFIME_PROFILE_MASK_FP32 :
        (profile == GAFIME_PROFILE_MIXED ? GAFIME_PROFILE_MASK_MIXED :
         GAFIME_PROFILE_MASK_FP64);
}

static uint32_t storage_dtype(uint32_t profile) {
    return profile == GAFIME_PROFILE_FP64 ? GAFIME_DTYPE_F64 : GAFIME_DTYPE_F32;
}

static uint32_t result_dtype(uint32_t profile) {
    return profile == GAFIME_PROFILE_FP32 ? GAFIME_DTYPE_F32 : GAFIME_DTYPE_F64;
}

static int run_profile(const TypedApi11* api, uint32_t backend_kind, uint32_t profile,
                       int permutation_available) {
    const float features_f32[8] = {
        1.0f, 7.0f, 2.0f, 5.0f, 3.0f, 3.0f, 4.0f, 1.0f,
    };
    const float target_f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const double epsilon = 1.0 / 1073741824.0;
    const double features_f64[8] = {
        1.0 + epsilon, 7.0, 2.0 + epsilon, 5.0,
        3.0 + epsilon, 3.0, 4.0 + epsilon, 1.0,
    };
    const double target_f64[4] = {
        1.0 + epsilon, 2.0 + epsilon, 3.0 + epsilon, 4.0 + epsilon,
    };
    const uint32_t storage = storage_dtype(profile);
    const uint32_t result_type = result_dtype(profile);
    PrecisionMatrixDesc_11 desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = GAFIME_ABI_1_1;
    desc.profile = profile;
    desc.dtype = storage;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 4;
    desc.cols = 2;
    desc.row_stride = 2;
    desc.bytes = 8 * (storage == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double));

    GafimeGpuMatrix11 matrix = NULL;
    int status = api->alloc(0, &desc, &matrix);
    if (status != GAFIME_STATUS_OK || matrix == NULL) {
        fprintf(stderr, "typed profile %u allocation failed: %d\n", profile, status);
        return 1;
    }
    int failed = 0;
    if (storage == GAFIME_DTYPE_F32) {
        status = api->upload_f32(matrix, features_f32, target_f32, 4, 2);
        if (status == GAFIME_STATUS_OK) {
            status = api->update_f32(matrix, target_f32, 4);
        }
    } else {
        status = api->upload_f64(matrix, features_f64, target_f64, 4, 2);
        if (status == GAFIME_STATUS_OK) {
            status = api->update_f64(matrix, target_f64, 4);
        }
    }
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "typed profile %u upload/update failed: %d\n", profile, status);
        failed = 1;
        goto cleanup;
    }

    const uint32_t combo = 0;
    const uint32_t metric = GAFIME_METRIC_PEARSON;
    ArityChunk_10 chunk;
    LaunchProtocol_10 base;
    PrecisionLaunchProtocol_11 protocol;
    memset(&chunk, 0, sizeof(chunk));
    memset(&base, 0, sizeof(base));
    memset(&protocol, 0, sizeof(protocol));
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 1;
    chunk.descriptor_count = 1;
    base.abi_version = GAFIME_ABI_1_0;
    base.backend_kind = backend_kind;
    base.max_arity = 1;
    base.n_samples = 4;
    base.n_features = 2;
    base.family_count = 1;
    base.combo_indices.ptr = &combo;
    base.combo_indices.len = 1;
    base.metric_ids.ptr = &metric;
    base.metric_ids.len = 1;
    base.chunks = &chunk;
    base.chunk_count = 1;
    base.rank.top_k = 1;
    base.rank.primary_metric = GAFIME_METRIC_PEARSON;
    protocol.abi_version = GAFIME_ABI_1_1;
    protocol.profile = profile;
    protocol.base = &base;

    uint64_t execution_peak = 0;
    status = api->execution_memory(matrix, &protocol, &execution_peak);
    if (status != GAFIME_STATUS_OK || execution_peak == 0) {
        fprintf(stderr, "typed profile %u forecast failed: %d/%llu\n", profile,
                status, (unsigned long long)execution_peak);
        failed = 1;
    }

    uint32_t combo_out = UINT32_MAX;
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate_id = UINT64_MAX;
    uint32_t row_flags = UINT32_MAX;
    float metric_f32 = 0.0f;
    double metric_f64 = 0.0;
    if (result_type == GAFIME_DTYPE_F32) {
        ResultTableF32_11 result;
        memset(&result, 0, sizeof(result));
        result.abi_version = GAFIME_ABI_1_0;
        result.max_arity = 1;
        result.metric_count = 1;
        result.capacity = 1;
        result.combo_indices = &combo_out;
        result.metric_values = &metric_f32;
        result.ranks = &rank;
        result.families = &family;
        result.candidate_ids = &candidate_id;
        result.row_flags = &row_flags;
        status = api->execute_f32(matrix, &protocol, &result);
        if (status != GAFIME_STATUS_OK || result.row_count != 1 ||
            !isfinite(metric_f32) || fabs((double)metric_f32 - 1.0) > 1.0e-5) {
            fprintf(stderr, "typed fp32 execute mismatch: %d/%llu/%.9g\n", status,
                    (unsigned long long)result.row_count, (double)metric_f32);
            failed = 1;
        }
    } else {
        ResultTableF64_11 result;
        memset(&result, 0, sizeof(result));
        result.abi_version = GAFIME_ABI_1_1;
        result.max_arity = 1;
        result.metric_count = 1;
        result.capacity = 1;
        result.combo_indices = &combo_out;
        result.metric_values = &metric_f64;
        result.ranks = &rank;
        result.families = &family;
        result.candidate_ids = &candidate_id;
        result.row_flags = &row_flags;
        status = api->execute_f64(matrix, &protocol, &result);
        if (status != GAFIME_STATUS_OK || result.row_count != 1 ||
            !isfinite(metric_f64) || fabs(metric_f64 - 1.0) > 1.0e-5) {
            fprintf(stderr, "typed f64 execute mismatch: %d/%llu/%.17g\n", status,
                    (unsigned long long)result.row_count, metric_f64);
            failed = 1;
        }
    }
    if (combo_out != 0 || family != GAFIME_FAMILY_CONTINUOUS || row_flags != 0) {
        fprintf(stderr, "typed profile %u structural result mismatch\n", profile);
        failed = 1;
    }

    if (permutation_available) {
        uint64_t permutation_peak = 0;
        base.permutations.permutation_count = 2;
        base.permutations.seed = UINT64_C(0x12345678);
        status = api->permutation_memory(matrix, &protocol, 1, &permutation_peak);
        if (status != GAFIME_STATUS_OK || permutation_peak == 0) {
            fprintf(stderr, "typed profile %u permutation forecast failed: %d/%llu\n",
                    profile, status, (unsigned long long)permutation_peak);
            failed = 1;
        } else if (result_type == GAFIME_DTYPE_F32) {
            float p_value = 0.0f;
            SignificanceF32_11 significance;
            memset(&significance, 0, sizeof(significance));
            significance.abi_version = GAFIME_ABI_1_0;
            significance.metric_count = 1;
            significance.row_count = 1;
            significance.candidate_ids = &candidate_id;
            significance.observed_metric_values = &metric_f32;
            significance.p_values = &p_value;
            status = api->permutation_f32(matrix, &protocol, &significance);
            if (status != GAFIME_STATUS_OK || !isfinite(p_value) ||
                p_value < 0.0f || p_value > 1.0f) {
                fprintf(stderr, "typed fp32 permutation failed: %d/%.9g\n", status,
                        (double)p_value);
                failed = 1;
            }
        } else {
            double p_value = 0.0;
            SignificanceF64_11 significance;
            memset(&significance, 0, sizeof(significance));
            significance.abi_version = GAFIME_ABI_1_1;
            significance.metric_count = 1;
            significance.row_count = 1;
            significance.candidate_ids = &candidate_id;
            significance.observed_metric_values = &metric_f64;
            significance.p_values = &p_value;
            status = api->permutation_f64(matrix, &protocol, &significance);
            if (status != GAFIME_STATUS_OK || !isfinite(p_value) ||
                p_value < 0.0 || p_value > 1.0) {
                fprintf(stderr, "typed f64 permutation failed: %d/%.17g\n", status,
                        p_value);
                failed = 1;
            }
        }
    }

    {
        uint64_t overflow = UINT64_MAX;
        uint32_t flags = UINT32_MAX;
        InteractionDiagnosticBatch_11 diagnostics;
        memset(&diagnostics, 0, sizeof(diagnostics));
        diagnostics.abi_version = GAFIME_ABI_1_0;
        diagnostics.max_arity = 1;
        diagnostics.row_count = 1;
        diagnostics.combo_indices = &combo;
        diagnostics.combo_index_count = 1;
        diagnostics.overflow_row_counts = &overflow;
        diagnostics.flags = &flags;
        status = api->diagnostics(matrix, &diagnostics);
        if (status != GAFIME_STATUS_OK || overflow != 0 || flags != 0) {
            fprintf(stderr, "typed profile %u diagnostics failed: %d/%llu/%u\n",
                    profile, status, (unsigned long long)overflow, flags);
            failed = 1;
        }
    }

cleanup:
    api->free_matrix(matrix);
    return failed;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s PAYLOAD BACKEND_KIND EXPECTED_PROFILE_COUNT\n", argv[0]);
        return 2;
    }
    const uint32_t backend_kind = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t expected_count = (uint32_t)strtoul(argv[3], NULL, 10);
    GafimeTestLibrary library = gafime_test_library_open(argv[1]);
    if (library == NULL) {
        fprintf(stderr, "could not load %s: %s\n", argv[1], gafime_test_library_error());
        return 1;
    }

    TypedApi11 api;
    memset(&api, 0, sizeof(api));
    GAFIME_TEST_LOAD_FUNCTION(library, api.capabilities,
                              "gafime_gpu_precision_capabilities");
    GAFIME_TEST_LOAD_FUNCTION(library, api.alloc, "gafime_gpu_matrix_alloc_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.upload_f32,
                              "gafime_gpu_matrix_upload_f32_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.upload_f64,
                              "gafime_gpu_matrix_upload_f64_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.update_f32,
                              "gafime_gpu_matrix_update_target_f32_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.update_f64,
                              "gafime_gpu_matrix_update_target_f64_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execute_f32, "gafime_gpu_execute_f32_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execute_f64, "gafime_gpu_execute_f64_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execution_memory,
                              "gafime_gpu_execution_memory_peak_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.diagnostics,
                              "gafime_gpu_interaction_diagnostics");
    GAFIME_TEST_LOAD_FUNCTION(library, api.free_matrix, "gafime_gpu_matrix_free");
    LOAD_OPTIONAL_FUNCTION(library, api.permutation_memory,
                           "gafime_gpu_permutation_memory_peak_v2");
    LOAD_OPTIONAL_FUNCTION(library, api.permutation_f32,
                           "gafime_gpu_permutation_pvalues_f32_v2");
    LOAD_OPTIONAL_FUNCTION(library, api.permutation_f64,
                           "gafime_gpu_permutation_pvalues_f64_v2");

    const int any_permutation = api.permutation_memory != NULL ||
        api.permutation_f32 != NULL || api.permutation_f64 != NULL;
    const int all_permutation = api.permutation_memory != NULL &&
        api.permutation_f32 != NULL && api.permutation_f64 != NULL;
    if (any_permutation && !all_permutation) {
        fprintf(stderr, "typed payload exposes an incomplete optional permutation family\n");
        gafime_test_library_close(library);
        return 1;
    }

    PrecisionCapabilities_11 capabilities;
    memset(&capabilities, 0xa5, sizeof(capabilities));
    int status = api.capabilities(0, &capabilities);
    if (unavailable_status(status)) {
        gafime_test_library_close(library);
        return 77;
    }
    const uint32_t expected_mask = expected_count == 1 ?
        GAFIME_PROFILE_MASK_FP32 :
        (GAFIME_PROFILE_MASK_FP32 | GAFIME_PROFILE_MASK_MIXED |
         GAFIME_PROFILE_MASK_FP64);
    const uint32_t expected_dtype_mask = expected_count == 1 ?
        GAFIME_DTYPE_MASK_F32 : (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64);
    if (status != GAFIME_STATUS_OK || capabilities.abi_version != GAFIME_ABI_1_1 ||
        capabilities.backend_kind != backend_kind ||
        capabilities.profile_mask != expected_mask ||
        capabilities.storage_dtype_mask != expected_dtype_mask ||
        capabilities.result_dtype_mask != expected_dtype_mask) {
        fprintf(stderr, "typed capability record mismatch: %d/%u/%u/%u\n", status,
                capabilities.abi_version, capabilities.backend_kind,
                capabilities.profile_mask);
        gafime_test_library_close(library);
        return 1;
    }

    int failed = 0;
    uint32_t observed = 0;
    for (uint32_t profile = GAFIME_PROFILE_FP32; profile <= GAFIME_PROFILE_FP64;
         ++profile) {
        const uint32_t bit = profile_bit(profile);
        if ((capabilities.profile_mask & bit) == 0) {
            continue;
        }
        const uint32_t storage_bit = storage_dtype(profile) == GAFIME_DTYPE_F32 ?
            GAFIME_DTYPE_MASK_F32 : GAFIME_DTYPE_MASK_F64;
        const uint32_t result_bit = result_dtype(profile) == GAFIME_DTYPE_F32 ?
            GAFIME_DTYPE_MASK_F32 : GAFIME_DTYPE_MASK_F64;
        if ((capabilities.storage_dtype_mask & storage_bit) == 0 ||
            (capabilities.result_dtype_mask & result_bit) == 0) {
            fprintf(stderr, "typed profile %u has contradictory dtype masks\n", profile);
            failed = 1;
            continue;
        }
        observed += 1;
        failed |= run_profile(&api, backend_kind, profile, all_permutation);
    }
    gafime_test_library_close(library);
    if (observed != expected_count) {
        fprintf(stderr, "typed profile count mismatch: %u/%u\n", observed, expected_count);
        failed = 1;
    }
    if (!failed) {
        printf(
            "{\"schema\":\"gafime.abi-1.1-typed-consumer-result.v1\","
            "\"status\":\"pass\",\"abi_surface\":\"precision-typed-v1.1\","
            "\"backend_kind\":%u,\"route_count\":%u,\"profile_mask\":%u,"
            "\"permutation_payload_operations\":%s,\"operations\":["
            "\"precision_capabilities\",\"matrix_alloc\",\"matrix_upload\","
            "\"matrix_update_target\",\"execute\",\"execution_memory_peak\","
            "\"interaction_diagnostics\",\"matrix_free\"]}\n",
            backend_kind, observed, capabilities.profile_mask,
            all_permutation ? "true" : "false");
    }
    return failed;
}
