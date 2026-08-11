/*
 * Standalone ABI 1.0 consumer.  Deliberately does not include the current
 * GAFIME header: these declarations are the frozen 1.0 prefix a third-party C
 * program compiled before ABI 1.1 existed.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "abi_dynamic_load.h"

#define GAFIME_ABI_1_0 ((1u << 16) | 0u)
#define GAFIME_STATUS_OK 0
#define GAFIME_STATUS_UNSUPPORTED_BACKEND -2
#define GAFIME_STATUS_DEVICE_ERROR -4
#define GAFIME_DTYPE_F32 1u
#define GAFIME_MATRIX_ROW_MAJOR 1u
#define GAFIME_METRIC_PEARSON 1u
#define GAFIME_FAMILY_CONTINUOUS 1u

typedef void* GafimeGpuMatrix10;

typedef struct GafimeSliceU32_10 {
    const uint32_t* ptr;
    uint64_t len;
} GafimeSliceU32_10;

typedef struct GafimeSliceU64_10 {
    const uint64_t* ptr;
    uint64_t len;
} GafimeSliceU64_10;

typedef struct GafimeMatrixDesc_10 {
    uint32_t abi_version;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
} GafimeMatrixDesc_10;

typedef struct GafimeArityChunk_10 {
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
} GafimeArityChunk_10;

typedef struct GafimeShapeHint_10 {
    uint32_t threads_per_block;
    uint32_t items_per_thread;
    uint32_t blocks_per_sm;
    uint32_t min_blocks;
    uint32_t shared_bytes;
    uint32_t register_budget;
    uint32_t occupancy_target_pct;
    uint32_t vendor_hint;
    uint64_t reserved[4];
} GafimeShapeHint_10;

typedef struct GafimeRankSpec_10 {
    uint32_t top_k;
    uint32_t primary_metric;
    uint32_t descending;
    uint32_t include_ties;
    uint64_t reserved[4];
} GafimeRankSpec_10;

typedef struct GafimePermutationSchedule_10 {
    uint32_t permutation_count;
    uint32_t mode;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t seed;
    GafimeSliceU64_10 target_offsets;
    uint64_t reserved[4];
} GafimePermutationSchedule_10;

typedef struct GafimeLaunchProtocol_10 {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t flags;
    uint32_t max_arity;
    uint64_t n_samples;
    uint32_t n_features;
    uint32_t family_count;
    GafimeSliceU32_10 combo_indices;
    GafimeSliceU32_10 metric_ids;
    const GafimeArityChunk_10* chunks;
    uint32_t chunk_count;
    uint32_t reserved32_a;
    const GafimeShapeHint_10* shape_hints;
    uint32_t shape_hint_count;
    uint32_t reserved32_b;
    GafimeRankSpec_10 rank;
    GafimePermutationSchedule_10 permutations;
    uint64_t reserved[8];
} GafimeLaunchProtocol_10;

typedef struct GafimeResultTable_10 {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    float* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    void* backend_private;
    uint64_t reserved[8];
} GafimeResultTable_10;

_Static_assert(sizeof(GafimeMatrixDesc_10) == 40, "ABI 1.0 matrix layout changed");
_Static_assert(_Alignof(GafimeMatrixDesc_10) == 8, "ABI 1.0 matrix alignment changed");
_Static_assert(sizeof(GafimeArityChunk_10) == 56, "ABI 1.0 chunk layout changed");
_Static_assert(sizeof(GafimeShapeHint_10) == 64, "ABI 1.0 shape layout changed");
_Static_assert(sizeof(GafimeRankSpec_10) == 48, "ABI 1.0 rank layout changed");
_Static_assert(sizeof(GafimePermutationSchedule_10) == 72,
               "ABI 1.0 permutation layout changed");
_Static_assert(sizeof(GafimeLaunchProtocol_10) == 280,
               "ABI 1.0 launch layout changed");
_Static_assert(_Alignof(GafimeLaunchProtocol_10) == 8,
               "ABI 1.0 launch alignment changed");
_Static_assert(sizeof(GafimeResultTable_10) == 152, "ABI 1.0 result layout changed");
_Static_assert(_Alignof(GafimeResultTable_10) == 8,
               "ABI 1.0 result alignment changed");
_Static_assert(offsetof(GafimeLaunchProtocol_10, rank) == 96,
               "ABI 1.0 launch rank offset changed");
_Static_assert(offsetof(GafimeLaunchProtocol_10, permutations) == 144,
               "ABI 1.0 launch permutation offset changed");
_Static_assert(offsetof(GafimeResultTable_10, metric_values) == 40,
               "ABI 1.0 metric pointer offset changed");

typedef int (*MatrixAllocFn)(uint32_t, const GafimeMatrixDesc_10*, GafimeGpuMatrix10*);
typedef int (*MatrixUploadFn)(GafimeGpuMatrix10, const float*, const float*, uint64_t, uint32_t);
typedef int (*ExecuteFn)(GafimeGpuMatrix10, const GafimeLaunchProtocol_10*, GafimeResultTable_10*);
typedef void (*MatrixFreeFn)(GafimeGpuMatrix10);

static int unavailable_status(int status) {
    return status == GAFIME_STATUS_UNSUPPORTED_BACKEND || status == GAFIME_STATUS_DEVICE_ERROR;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s PAYLOAD BACKEND_KIND\n", argv[0]);
        return 2;
    }
    const uint32_t backend_kind = (uint32_t)strtoul(argv[2], NULL, 10);
    GafimeTestLibrary library = gafime_test_library_open(argv[1]);
    if (library == NULL) {
        fprintf(stderr, "could not load %s: %s\n", argv[1], gafime_test_library_error());
        return 1;
    }

    MatrixAllocFn matrix_alloc = NULL;
    MatrixUploadFn matrix_upload = NULL;
    ExecuteFn execute = NULL;
    MatrixFreeFn matrix_free = NULL;
    GAFIME_TEST_LOAD_FUNCTION(library, matrix_alloc, "gafime_gpu_matrix_alloc");
    GAFIME_TEST_LOAD_FUNCTION(library, matrix_upload, "gafime_gpu_matrix_upload");
    GAFIME_TEST_LOAD_FUNCTION(library, execute, "gafime_gpu_execute");
    GAFIME_TEST_LOAD_FUNCTION(library, matrix_free, "gafime_gpu_matrix_free");

    const float features[8] = {1.0f, 7.0f, 2.0f, 5.0f, 3.0f, 3.0f, 4.0f, 1.0f};
    const float target[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GafimeMatrixDesc_10 desc = {0};
    desc.abi_version = GAFIME_ABI_1_0;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 4;
    desc.cols = 2;
    desc.row_stride = 2;
    desc.bytes = sizeof(features);

    GafimeGpuMatrix10 matrix = NULL;
    int status = matrix_alloc(0, &desc, &matrix);
    if (unavailable_status(status)) {
        gafime_test_library_close(library);
        return 77;
    }
    if (status != GAFIME_STATUS_OK || matrix == NULL) {
        fprintf(stderr, "ABI 1.0 allocation failed: %d\n", status);
        gafime_test_library_close(library);
        return 1;
    }
    status = matrix_upload(matrix, features, target, 4, 2);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "ABI 1.0 upload failed: %d\n", status);
        matrix_free(matrix);
        gafime_test_library_close(library);
        return 1;
    }

    const uint32_t combo = 0;
    const uint32_t metric = GAFIME_METRIC_PEARSON;
    GafimeArityChunk_10 chunk = {0};
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 1;
    chunk.descriptor_count = 1;
    GafimeLaunchProtocol_10 protocol = {0};
    protocol.abi_version = GAFIME_ABI_1_0;
    protocol.backend_kind = backend_kind;
    protocol.max_arity = 1;
    protocol.n_samples = 4;
    protocol.n_features = 2;
    protocol.family_count = 1;
    protocol.combo_indices.ptr = &combo;
    protocol.combo_indices.len = 1;
    protocol.metric_ids.ptr = &metric;
    protocol.metric_ids.len = 1;
    protocol.chunks = &chunk;
    protocol.chunk_count = 1;

    uint32_t combo_out = UINT32_MAX;
    float value_out = 0.0f;
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate_id = UINT64_MAX;
    uint32_t row_flags = UINT32_MAX;
    GafimeResultTable_10 result = {0};
    result.abi_version = GAFIME_ABI_1_0;
    result.max_arity = 1;
    result.metric_count = 1;
    result.capacity = 1;
    result.combo_indices = &combo_out;
    result.metric_values = &value_out;
    result.ranks = &rank;
    result.families = &family;
    result.candidate_ids = &candidate_id;
    result.row_flags = &row_flags;

    status = execute(matrix, &protocol, &result);
    const int failed = status != GAFIME_STATUS_OK || result.row_count != 1 || combo_out != 0 ||
        !isfinite(value_out) || fabsf(value_out - 1.0f) > 1.0e-5f;
    if (failed) {
        fprintf(stderr,
                "ABI 1.0 execute mismatch: status=%d rows=%llu combo=%u value=%.9g\n",
                status, (unsigned long long)result.row_count, combo_out, value_out);
    }
    matrix_free(matrix);
    gafime_test_library_close(library);
    return failed;
}
