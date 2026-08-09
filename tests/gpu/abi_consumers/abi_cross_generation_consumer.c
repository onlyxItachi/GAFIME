/*
 * Standalone cross-generation ABI consumer.
 *
 * The two matrix lifecycles are intentionally exercised through the published
 * ABI 1.0 and ABI 1.1 symbols.  Non-free calls must reject an opposite-
 * generation owner before interpreting its route/protocol, while either free
 * symbol may tear down a validated owner.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/common/gafime_gpu_abi.hpp"
#include "abi_dynamic_load.h"

typedef int (*LegacyAllocFn)(uint32_t, const GafimeMatrixDesc*, GafimeGpuMatrix*);
typedef int (*LegacyUploadFn)(GafimeGpuMatrix, const float*, const float*, uint64_t, uint32_t);
typedef int (*LegacyUpdateFn)(GafimeGpuMatrix, const float*, uint64_t);
typedef int (*LegacyExecuteFn)(GafimeGpuMatrix, const GafimeLaunchProtocol*, GafimeResultTable*);
typedef int (*LegacyPeakFn)(GafimeGpuMatrix, const GafimeLaunchProtocol*, uint64_t*);
typedef int (*LegacyDiagnosticsFn)(GafimeGpuMatrix, GafimeInteractionDiagnosticBatch*);
typedef void (*LegacyFreeFn)(GafimeGpuMatrix);

typedef int (*RoutesFn)(uint32_t, uint32_t, uint32_t, GafimeNumericRoute*, uint32_t, uint32_t*);
typedef int (*AllocV2Fn)(uint32_t, const GafimeNumericMatrixDesc*, GafimeGpuMatrix*);
typedef int (*UploadV2Fn)(
    GafimeGpuMatrix, const GafimeNumericRoute*, const GafimeConstBufferView*,
    const GafimeConstBufferView*, uint64_t, uint32_t);
typedef int (*UpdateV2Fn)(
    GafimeGpuMatrix, const GafimeNumericRoute*, const GafimeConstBufferView*, uint64_t);
typedef int (*ExecuteV2Fn)(
    GafimeGpuMatrix, const GafimeNumericLaunchProtocol*, GafimeNumericResultTable*);
typedef int (*PeakV2Fn)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*, uint64_t*);
typedef int (*PermutationPeakV2Fn)(
    GafimeGpuMatrix, const GafimeNumericLaunchProtocol*, uint64_t, uint64_t*);
typedef int (*PermutationV2Fn)(
    GafimeGpuMatrix, const GafimeNumericLaunchProtocol*, GafimeNumericSignificanceTable*);
typedef int (*DiagnosticsV2Fn)(GafimeGpuMatrix, GafimeNumericInteractionDiagnosticBatch*);
typedef int (*FreeV2Fn)(GafimeGpuMatrix);

typedef struct CrossApi {
    LegacyAllocFn legacy_alloc;
    LegacyUploadFn legacy_upload;
    LegacyUpdateFn legacy_update;
    LegacyExecuteFn legacy_execute;
    LegacyPeakFn legacy_peak;
    LegacyDiagnosticsFn legacy_diagnostics;
    LegacyFreeFn legacy_free;
    RoutesFn routes;
    AllocV2Fn alloc_v2;
    UploadV2Fn upload_v2;
    UpdateV2Fn update_v2;
    ExecuteV2Fn execute_v2;
    PeakV2Fn peak_v2;
    PermutationPeakV2Fn permutation_peak_v2;
    PermutationV2Fn permutation_v2;
    DiagnosticsV2Fn diagnostics_v2;
    FreeV2Fn free_v2;
} CrossApi;

static int unavailable_status(int status) {
    return status == GAFIME_STATUS_UNSUPPORTED_BACKEND || status == GAFIME_STATUS_DEVICE_ERROR;
}

static GafimeConstBufferView const_view(const void* data, uint32_t dtype, uint64_t count) {
    GafimeConstBufferView view;
    memset(&view, 0, sizeof(view));
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = count;
    view.byte_stride = dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double);
    view.byte_length = count * view.byte_stride;
    return view;
}

static GafimeMutableBufferView mutable_view(void* data, uint32_t dtype, uint64_t count) {
    GafimeMutableBufferView view;
    memset(&view, 0, sizeof(view));
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = count;
    view.byte_stride = dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double);
    view.byte_length = count * view.byte_stride;
    return view;
}

static int require_invalid(int status, const char* operation) {
    if (status != GAFIME_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "%s: expected INVALID_ARGUMENT, got %d\n", operation, status);
        return 1;
    }
    return 0;
}

static GafimeNumericMatrixDesc numeric_desc(const GafimeNumericRoute* route) {
    GafimeNumericMatrixDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = *route;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 1;
    desc.cols = 1;
    desc.row_stride = 1;
    desc.bytes = sizeof(float);
    return desc;
}

static GafimeMatrixDesc legacy_desc(void) {
    GafimeMatrixDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 1;
    desc.cols = 1;
    desc.row_stride = 1;
    desc.bytes = sizeof(float);
    return desc;
}

static void make_protocol(
    uint32_t backend_kind,
    uint32_t* combo,
    uint32_t* metric,
    GafimeArityChunk* chunk,
    GafimeLaunchProtocol* base,
    GafimeNumericLaunchProtocol* numeric,
    const GafimeNumericRoute* route
) {
    memset(chunk, 0, sizeof(*chunk));
    chunk->arity = 1;
    chunk->family = GAFIME_FAMILY_CONTINUOUS;
    chunk->combo_count = 1;
    chunk->descriptor_count = 1;

    memset(base, 0, sizeof(*base));
    base->abi_version = GAFIME_ABI_VERSION;
    base->backend_kind = backend_kind;
    base->max_arity = 1;
    base->n_samples = 1;
    base->n_features = 1;
    base->family_count = 1;
    base->combo_indices.ptr = combo;
    base->combo_indices.len = 1;
    base->metric_ids.ptr = metric;
    base->metric_ids.len = 1;
    base->chunks = chunk;
    base->chunk_count = 1;

    memset(numeric, 0, sizeof(*numeric));
    numeric->abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric->struct_size = sizeof(*numeric);
    numeric->route = *route;
    numeric->base = base;
}

static int exercise_cross_generation(
    const CrossApi* api,
    uint32_t backend_kind,
    const GafimeNumericRoute* route
) {
    const float features[1] = {1.0f};
    const float target[1] = {2.0f};
    const GafimeConstBufferView feature_view = const_view(features, GAFIME_DTYPE_F32, 1);
    const GafimeConstBufferView target_view = const_view(target, GAFIME_DTYPE_F32, 1);
    const GafimeNumericMatrixDesc v2_desc = numeric_desc(route);
    const GafimeMatrixDesc v1_desc = legacy_desc();
    GafimeGpuMatrix legacy_matrix = NULL;
    GafimeGpuMatrix numeric_matrix = NULL;
    int status = api->legacy_alloc(0, &v1_desc, &legacy_matrix);
    if (unavailable_status(status)) return 77;
    if (status != GAFIME_STATUS_OK || legacy_matrix == NULL) {
        fprintf(stderr, "ABI 1.0 cross-generation allocation failed: %d\n", status);
        return 1;
    }
    status = api->alloc_v2(0, &v2_desc, &numeric_matrix);
    if (unavailable_status(status)) {
        api->legacy_free(legacy_matrix);
        return 77;
    }
    if (status != GAFIME_STATUS_OK || numeric_matrix == NULL) {
        fprintf(stderr, "ABI 1.1 cross-generation allocation failed: %d\n", status);
        api->legacy_free(legacy_matrix);
        return 1;
    }

    uint32_t combo = 0;
    uint32_t metric = GAFIME_METRIC_PEARSON;
    GafimeArityChunk chunk;
    GafimeLaunchProtocol base;
    GafimeNumericLaunchProtocol numeric_protocol;
    make_protocol(
        backend_kind, &combo, &metric, &chunk, &base, &numeric_protocol, route);

    uint32_t combo_out = UINT32_MAX;
    float metric_out = 0.0f;
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate_id = UINT64_MAX;
    uint32_t row_flags = 0;
    GafimeResultTable legacy_result;
    memset(&legacy_result, 0, sizeof(legacy_result));
    legacy_result.abi_version = GAFIME_ABI_VERSION;
    legacy_result.max_arity = 1;
    legacy_result.metric_count = 1;
    legacy_result.capacity = 1;
    legacy_result.combo_indices = &combo_out;
    legacy_result.metric_values = &metric_out;
    legacy_result.ranks = &rank;
    legacy_result.families = &family;
    legacy_result.candidate_ids = &candidate_id;
    legacy_result.row_flags = &row_flags;

    GafimeNumericResultTable numeric_result;
    memset(&numeric_result, 0, sizeof(numeric_result));
    numeric_result.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric_result.struct_size = sizeof(numeric_result);
    numeric_result.max_arity = 1;
    numeric_result.metric_count = 1;
    numeric_result.capacity = 1;
    numeric_result.combo_indices = &combo_out;
    numeric_result.metric_values = mutable_view(&metric_out, GAFIME_DTYPE_F32, 1);
    numeric_result.ranks = &rank;
    numeric_result.families = &family;
    numeric_result.candidate_ids = &candidate_id;
    numeric_result.row_flags = &row_flags;

    uint64_t peak = 0;
    int failed = 0;
    failed |= require_invalid(
        api->upload_v2(legacy_matrix, route, &feature_view, &target_view, 1, 1),
        "ABI 1.1 upload on ABI 1.0 owner");
    failed |= require_invalid(
        api->update_v2(legacy_matrix, route, &target_view, 1),
        "ABI 1.1 update on ABI 1.0 owner");
    failed |= require_invalid(
        api->execute_v2(legacy_matrix, &numeric_protocol, &numeric_result),
        "ABI 1.1 execute on ABI 1.0 owner");
    failed |= require_invalid(
        api->peak_v2(legacy_matrix, &numeric_protocol, &peak),
        "ABI 1.1 execution peak on ABI 1.0 owner");
    failed |= require_invalid(
        api->permutation_peak_v2(legacy_matrix, &numeric_protocol, 0, &peak),
        "ABI 1.1 permutation peak on ABI 1.0 owner");

    uint64_t overflow = 0;
    uint32_t diagnostic_flags = 0;
    GafimeNumericInteractionDiagnosticBatch numeric_diagnostics;
    memset(&numeric_diagnostics, 0, sizeof(numeric_diagnostics));
    numeric_diagnostics.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric_diagnostics.struct_size = sizeof(numeric_diagnostics);
    numeric_diagnostics.route = *route;
    numeric_diagnostics.max_arity = 1;
    numeric_diagnostics.overflow_row_counts = &overflow;
    numeric_diagnostics.row_flags = &diagnostic_flags;
    failed |= require_invalid(
        api->diagnostics_v2(legacy_matrix, &numeric_diagnostics),
        "ABI 1.1 diagnostics on ABI 1.0 owner");

    GafimeNumericSignificanceTable significance;
    memset(&significance, 0, sizeof(significance));
    significance.abi_version = GAFIME_PRECISION_ABI_VERSION;
    significance.struct_size = sizeof(significance);
    significance.metric_count = 1;
    significance.observed_metric_values = const_view(NULL, GAFIME_DTYPE_F32, 0);
    significance.p_values = mutable_view(NULL, GAFIME_DTYPE_F32, 0);
    failed |= require_invalid(
        api->permutation_v2(legacy_matrix, &numeric_protocol, &significance),
        "ABI 1.1 permutation p-values on ABI 1.0 owner");

    failed |= require_invalid(
        api->legacy_upload(numeric_matrix, features, target, 1, 1),
        "ABI 1.0 upload on ABI 1.1 owner");
    failed |= require_invalid(
        api->legacy_update(numeric_matrix, target, 1),
        "ABI 1.0 update on ABI 1.1 owner");
    failed |= require_invalid(
        api->legacy_execute(numeric_matrix, &base, &legacy_result),
        "ABI 1.0 execute on ABI 1.1 owner");
    failed |= require_invalid(
        api->legacy_peak(numeric_matrix, &base, &peak),
        "ABI 1.0 execution peak on ABI 1.1 owner");
    GafimeInteractionDiagnosticBatch legacy_diagnostics;
    memset(&legacy_diagnostics, 0, sizeof(legacy_diagnostics));
    legacy_diagnostics.abi_version = GAFIME_ABI_VERSION;
    legacy_diagnostics.max_arity = 1;
    failed |= require_invalid(
        api->legacy_diagnostics(numeric_matrix, &legacy_diagnostics),
        "ABI 1.0 diagnostics on ABI 1.1 owner");

    /* Cross-generation free is the one intentionally accepted operation. */
    status = api->free_v2(legacy_matrix);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "ABI 1.1 free of ABI 1.0 owner failed: %d\n", status);
        failed = 1;
    }
    legacy_matrix = NULL;
    api->legacy_free(numeric_matrix);
    numeric_matrix = NULL;
    return failed;
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

    CrossApi api;
    memset(&api, 0, sizeof(api));
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_alloc, "gafime_gpu_matrix_alloc");
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_upload, "gafime_gpu_matrix_upload");
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_update, "gafime_gpu_matrix_update_target");
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_execute, "gafime_gpu_execute");
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_peak, "gafime_gpu_execution_memory_peak");
    GAFIME_TEST_LOAD_FUNCTION(
        library, api.legacy_diagnostics, "gafime_gpu_interaction_diagnostics");
    GAFIME_TEST_LOAD_FUNCTION(library, api.legacy_free, "gafime_gpu_matrix_free");
    GAFIME_TEST_LOAD_FUNCTION(library, api.routes, "gafime_gpu_numeric_routes_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.alloc_v2, "gafime_gpu_matrix_alloc_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.upload_v2, "gafime_gpu_matrix_upload_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.update_v2, "gafime_gpu_matrix_update_target_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execute_v2, "gafime_gpu_execute_v2");
    GAFIME_TEST_LOAD_FUNCTION(
        library, api.peak_v2, "gafime_gpu_execution_memory_peak_v2");
    GAFIME_TEST_LOAD_FUNCTION(
        library, api.permutation_peak_v2, "gafime_gpu_permutation_memory_peak_v2");
    GAFIME_TEST_LOAD_FUNCTION(
        library, api.permutation_v2, "gafime_gpu_permutation_pvalues_v2");
    GAFIME_TEST_LOAD_FUNCTION(
        library, api.diagnostics_v2, "gafime_gpu_interaction_diagnostics_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.free_v2, "gafime_gpu_matrix_free_v2");

    uint32_t route_count = 0;
    int status = api.routes(
        0, GAFIME_PRECISION_ABI_VERSION, sizeof(GafimeNumericRoute), NULL, 0, &route_count);
    if (unavailable_status(status)) {
        gafime_test_library_close(library);
        return 77;
    }
    if (status != GAFIME_STATUS_OK || route_count == 0 || route_count > 16) {
        fprintf(stderr, "route count query failed: %d/%u\n", status, route_count);
        gafime_test_library_close(library);
        return 1;
    }
    GafimeNumericRoute routes[16];
    memset(routes, 0, sizeof(routes));
    status = api.routes(
        0, GAFIME_PRECISION_ABI_VERSION, sizeof(GafimeNumericRoute), routes, route_count,
        &route_count);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "route enumeration failed: %d\n", status);
        gafime_test_library_close(library);
        return 1;
    }
    const GafimeNumericRoute* fp32_route = NULL;
    for (uint32_t index = 0; index < route_count; ++index) {
        if (routes[index].profile == GAFIME_PRECISION_FP32) {
            fp32_route = &routes[index];
            break;
        }
    }
    if (fp32_route == NULL) {
        fprintf(stderr, "payload did not advertise the required fp32 route\n");
        gafime_test_library_close(library);
        return 1;
    }
    const int result = exercise_cross_generation(&api, backend_kind, fp32_route);
    gafime_test_library_close(library);
    if (result == 0) {
        printf(
            "{\"schema\":\"gafime.cross-generation-consumer-result.v1\","
            "\"status\":\"pass\",\"backend_kind\":%u,"
            "\"policy\":\"non-free-generation-strict-free-generation-neutral\"}\n",
            backend_kind);
    }
    return result;
}
