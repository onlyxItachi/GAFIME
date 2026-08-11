/*
 * Frozen ABI 1.0 regression: a third-party C consumer may submit a runtime
 * arity above the ABI 1.1 fixed-kernel ceiling.  This fixture intentionally
 * keeps the old declarations local instead of including the current header.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "abi_dynamic_load.h"

#define GAFIME_ABI_1_0 ((1u << 16) | 0u)
#define GAFIME_STATUS_OK 0
#define GAFIME_STATUS_INVALID_ARGUMENT -1
#define GAFIME_STATUS_UNSUPPORTED_BACKEND -2
#define GAFIME_STATUS_DEVICE_ERROR -4
#define GAFIME_STATUS_GRAPH_UNSUPPORTED -5
#define GAFIME_DTYPE_F32 1u
#define GAFIME_MATRIX_ROW_MAJOR 1u
#define GAFIME_FAMILY_CONTINUOUS 1u
#define GAFIME_METRIC_PEARSON 1u
#define GAFIME_METRIC_SPEARMAN 2u
#define GAFIME_METRIC_MUTUAL_INFO 3u

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
    const void* shape_hints;
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

    enum { rows = 16, cols = 6, arity = 6, metric_count = 3 };
    float features[rows * cols];
    float target[rows];
    for (uint32_t row = 0; row < rows; ++row) {
        target[row] = 0.5f + (float)row;
        for (uint32_t col = 0; col < cols; ++col) {
            features[row * cols + col] =
                0.25f * (float)(row + 1) * (float)(col + 2) +
                0.01f * (float)(row * row + col);
        }
    }

    GafimeMatrixDesc_10 desc = {0};
    desc.abi_version = GAFIME_ABI_1_0;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = rows;
    desc.cols = cols;
    desc.row_stride = cols;
    desc.bytes = sizeof(features);

    GafimeGpuMatrix10 matrix = NULL;
    int status = matrix_alloc(0, &desc, &matrix);
    if (unavailable_status(status)) {
        gafime_test_library_close(library);
        return 77;
    }
    if (status != GAFIME_STATUS_OK || matrix == NULL) {
        fprintf(stderr, "ABI 1.0 arity-6 allocation failed: %d\n", status);
        gafime_test_library_close(library);
        return 1;
    }
    status = matrix_upload(matrix, features, target, rows, cols);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "ABI 1.0 arity-6 upload failed: %d\n", status);
        matrix_free(matrix);
        gafime_test_library_close(library);
        return 1;
    }

    const uint32_t combos[arity] = {0, 1, 2, 3, 4, 5};
    const uint32_t metrics[metric_count] = {
        GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_SPEARMAN,
    };
    GafimeArityChunk_10 chunk = {0};
    chunk.arity = arity;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 1;
    chunk.descriptor_count = 1;

    GafimeLaunchProtocol_10 protocol = {0};
    protocol.abi_version = GAFIME_ABI_1_0;
    protocol.backend_kind = backend_kind;
    protocol.max_arity = arity;
    protocol.n_samples = rows;
    protocol.n_features = cols;
    protocol.family_count = 1;
    protocol.combo_indices.ptr = combos;
    protocol.combo_indices.len = arity;
    protocol.metric_ids.ptr = metrics;
    protocol.metric_ids.len = metric_count;
    protocol.chunks = &chunk;
    protocol.chunk_count = 1;

    /* ABI 1.0 execute never treated an inline permutation schedule as an
     * ordinary scoring launch. Preserve each frozen backend's validation
     * order: CUDA rejects the missing result first; ROCm/Metal reject the
     * unsupported schedule first. */
    protocol.permutations.permutation_count = 1;
    protocol.permutations.seed = UINT64_C(0x0123456789abcdef);
    status = execute(matrix, &protocol, NULL);
    const int expected_permutation_status = backend_kind == 2u ?
        GAFIME_STATUS_INVALID_ARGUMENT : GAFIME_STATUS_GRAPH_UNSUPPORTED;
    if (status != expected_permutation_status) {
        fprintf(stderr, "ABI 1.0 permutation rejection changed: %d\n", status);
        matrix_free(matrix);
        gafime_test_library_close(library);
        return 1;
    }
    protocol.permutations.permutation_count = 0;

    uint32_t combo_out[arity];
    float values[metric_count];
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate_id = UINT64_MAX;
    uint32_t row_flags = UINT32_MAX;
    GafimeResultTable_10 result = {0};
    result.abi_version = GAFIME_ABI_1_0;
    result.max_arity = arity;
    result.metric_count = metric_count;
    result.capacity = 1;
    result.combo_indices = combo_out;
    result.metric_values = values;
    result.ranks = &rank;
    result.families = &family;
    result.candidate_ids = &candidate_id;
    result.row_flags = &row_flags;

    status = execute(matrix, &protocol, &result);
    printf(
        "ABI 1.0 arity-6 evidence backend=%u status=%d values=(%.9g, %.9g, %.9g)\n",
        backend_kind, status, values[0], values[1], values[2]);
    int failed = status != GAFIME_STATUS_OK || result.row_count != 1;
    for (uint32_t idx = 0; idx < arity; ++idx) {
        failed = failed || combo_out[idx] != combos[idx];
    }
    for (uint32_t idx = 0; idx < metric_count; ++idx) {
        failed = failed || !isfinite(values[idx]);
    }
    /* These deterministic frozen-baseline expectations are independent of the
     * candidate payload. The tolerance is the existing cross-device fp32
     * reduction-order bound. */
    failed = failed || fabsf(values[0] - 0x1.8183b4p-3f) > 0.00005f;
    failed = failed || fabsf(values[1]) > 0.00005f;
    failed = failed || fabsf(values[2] - 0x1.515152p-6f) > 0.00005f;
    if (failed) {
        fprintf(stderr,
                "ABI 1.0 arity-6 mismatch: status=%d rows=%llu values=(%.9g, %.9g, %.9g)\n",
                status, (unsigned long long)result.row_count,
                values[0], values[1], values[2]);
    }
    matrix_free(matrix);
    gafime_test_library_close(library);
    return failed;
}
