/* Standalone C consumer of the published canonical ABI 1.1 surface. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/common/gafime_gpu_abi.hpp"
#include "abi_dynamic_load.h"

_Static_assert(sizeof(GafimeNumericRoute) == 104, "numeric route size drifted");
_Static_assert(_Alignof(GafimeNumericRoute) == 8, "numeric route alignment drifted");
_Static_assert(offsetof(GafimeNumericRoute, route_id) == 8, "route id offset drifted");
_Static_assert(offsetof(GafimeNumericRoute, result_dtype) == 28,
               "result dtype offset drifted");
_Static_assert(offsetof(GafimeNumericRoute, reserved) == 40,
               "route stable-prefix size drifted");
_Static_assert(sizeof(GafimeConstBufferView) == 80, "const buffer size drifted");
_Static_assert(_Alignof(GafimeConstBufferView) == 8, "const buffer alignment drifted");
_Static_assert(sizeof(GafimeMutableBufferView) == 80, "mutable buffer size drifted");
_Static_assert(_Alignof(GafimeMutableBufferView) == 8, "mutable buffer alignment drifted");
_Static_assert(offsetof(GafimeConstBufferView, data) == 16, "const data offset drifted");
_Static_assert(offsetof(GafimeConstBufferView, reserved) == 48,
               "const buffer stable-prefix size drifted");
_Static_assert(sizeof(GafimeNumericMatrixDesc) == 208, "matrix descriptor size drifted");
_Static_assert(_Alignof(GafimeNumericMatrixDesc) == 8,
               "matrix descriptor alignment drifted");
_Static_assert(offsetof(GafimeNumericMatrixDesc, route) == 8, "matrix route offset drifted");
_Static_assert(offsetof(GafimeNumericMatrixDesc, reserved) == 144,
               "matrix descriptor stable-prefix size drifted");
_Static_assert(sizeof(GafimeNumericLaunchProtocol) == 184,
               "numeric launch protocol size drifted");
_Static_assert(_Alignof(GafimeNumericLaunchProtocol) == 8,
               "numeric launch protocol alignment drifted");
_Static_assert(offsetof(GafimeNumericLaunchProtocol, base) == 112,
               "numeric launch base offset drifted");
_Static_assert(sizeof(GafimeNumericResultTable) == 224, "numeric result size drifted");
_Static_assert(_Alignof(GafimeNumericResultTable) == 8,
               "numeric result alignment drifted");
_Static_assert(offsetof(GafimeNumericResultTable, metric_values) == 48,
               "numeric result value-view offset drifted");
_Static_assert(sizeof(GafimeNumericSignificanceTable) == 256,
               "numeric significance size drifted");
_Static_assert(_Alignof(GafimeNumericSignificanceTable) == 8,
               "numeric significance alignment drifted");
_Static_assert(offsetof(GafimeNumericSignificanceTable, p_values) == 112,
               "numeric significance p-value offset drifted");
_Static_assert(sizeof(GafimeNumericInteractionDiagnosticBatch) == 224,
               "numeric diagnostic size drifted");
_Static_assert(_Alignof(GafimeNumericInteractionDiagnosticBatch) == 8,
               "numeric diagnostic alignment drifted");

#define MAX_ROUTE_RECORDS 16u

/* A future payload may append fields to an enumerated route record. */
typedef struct FutureRouteRecord {
    GafimeNumericRoute known;
    uint64_t future_fields[2];
} FutureRouteRecord;

typedef struct FutureConstBufferView {
    GafimeConstBufferView known;
    uint64_t future_field;
} FutureConstBufferView;

typedef struct FutureMutableBufferView {
    GafimeMutableBufferView known;
    uint64_t future_field;
} FutureMutableBufferView;

_Static_assert(sizeof(FutureRouteRecord) == 120, "future route fixture size drifted");
_Static_assert(_Alignof(FutureRouteRecord) == 8, "future route fixture alignment drifted");
_Static_assert(sizeof(FutureConstBufferView) == 88, "future const view fixture size drifted");
_Static_assert(sizeof(FutureMutableBufferView) == 88, "future mutable view fixture size drifted");

typedef int (*NumericRoutesFn)(uint32_t, uint32_t, uint32_t, GafimeNumericRoute*,
                               uint32_t, uint32_t*);
typedef int (*MatrixAllocV2Fn)(uint32_t, const GafimeNumericMatrixDesc*, GafimeGpuMatrix*);
typedef int (*MatrixUploadV2Fn)(GafimeGpuMatrix, const GafimeNumericRoute*,
                                const GafimeConstBufferView*, const GafimeConstBufferView*,
                                uint64_t, uint32_t);
typedef int (*MatrixUpdateTargetV2Fn)(GafimeGpuMatrix, const GafimeNumericRoute*,
                                      const GafimeConstBufferView*, uint64_t);
typedef int (*ExecuteV2Fn)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                           GafimeNumericResultTable*);
typedef int (*ExecutionMemoryV2Fn)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                                   uint64_t*);
typedef int (*PermutationMemoryV2Fn)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                                     uint64_t, uint64_t*);
typedef int (*PermutationV2Fn)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                               GafimeNumericSignificanceTable*);
typedef int (*DiagnosticsV2Fn)(GafimeGpuMatrix, GafimeNumericInteractionDiagnosticBatch*);
typedef int (*MatrixFreeV2Fn)(GafimeGpuMatrix);

typedef struct Api11 {
    NumericRoutesFn routes;
    MatrixAllocV2Fn alloc;
    MatrixUploadV2Fn upload;
    MatrixUpdateTargetV2Fn update_target;
    ExecuteV2Fn execute;
    ExecutionMemoryV2Fn execution_memory;
    PermutationMemoryV2Fn permutation_memory;
    PermutationV2Fn permutation;
    DiagnosticsV2Fn diagnostics;
    MatrixFreeV2Fn free_matrix;
} Api11;

static int unavailable_status(int status) {
    return status == GAFIME_STATUS_UNSUPPORTED_BACKEND || status == GAFIME_STATUS_DEVICE_ERROR;
}

static uint64_t dtype_size(uint32_t dtype) {
    return dtype == GAFIME_DTYPE_F32 ? sizeof(float) :
        (dtype == GAFIME_DTYPE_F64 ? sizeof(double) : 0);
}

static GafimeConstBufferView const_view(uint32_t dtype, const void* data, uint64_t count) {
    GafimeConstBufferView view;
    memset(&view, 0, sizeof(view));
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = count;
    view.byte_stride = dtype_size(dtype);
    view.byte_length = count * view.byte_stride;
    return view;
}

static GafimeMutableBufferView mutable_view(uint32_t dtype, void* data, uint64_t count) {
    GafimeMutableBufferView view;
    memset(&view, 0, sizeof(view));
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = count;
    view.byte_stride = dtype_size(dtype);
    view.byte_length = count * view.byte_stride;
    return view;
}

static int canonical_route(const GafimeNumericRoute* route) {
    if (route->overflow_policy != GAFIME_OVERFLOW_IEEE ||
        (route->flags & GAFIME_ABI_REQUIRED_FLAG_MASK) != 0) {
        return 0;
    }
    switch (route->route_id) {
    case GAFIME_NUMERIC_ROUTE_FP32:
        return route->profile == GAFIME_PRECISION_FP32 &&
            route->storage_dtype == GAFIME_DTYPE_F32 &&
            route->pointwise_dtype == GAFIME_DTYPE_F32 &&
            route->reduction_dtype == GAFIME_DTYPE_F32 &&
            route->result_dtype == GAFIME_DTYPE_F32;
    case GAFIME_NUMERIC_ROUTE_MIXED:
        return route->profile == GAFIME_PRECISION_MIXED &&
            route->storage_dtype == GAFIME_DTYPE_F32 &&
            route->pointwise_dtype == GAFIME_DTYPE_F32 &&
            route->reduction_dtype == GAFIME_DTYPE_F64 &&
            route->result_dtype == GAFIME_DTYPE_F64;
    case GAFIME_NUMERIC_ROUTE_FP64:
        return route->profile == GAFIME_PRECISION_FP64 &&
            route->storage_dtype == GAFIME_DTYPE_F64 &&
            route->pointwise_dtype == GAFIME_DTYPE_F64 &&
            route->reduction_dtype == GAFIME_DTYPE_F64 &&
            route->result_dtype == GAFIME_DTYPE_F64;
    default:
        return 0;
    }
}

static int known_route_id(uint32_t route_id) {
    return route_id == GAFIME_NUMERIC_ROUTE_FP32 ||
        route_id == GAFIME_NUMERIC_ROUTE_MIXED ||
        route_id == GAFIME_NUMERIC_ROUTE_FP64;
}

static int route_id_seen(const uint32_t* seen_ids, uint32_t seen_count, uint32_t route_id) {
    for (uint32_t index = 0; index < seen_count; ++index) {
        if (seen_ids[index] == route_id) return 1;
    }
    return 0;
}

/* Returns 1 for a known route, 0 for an unknown additive route, -1 if invalid. */
static int parse_route_record(
    const void* record,
    uint32_t route_stride,
    uint32_t* seen_ids,
    uint32_t* seen_count,
    GafimeNumericRoute* known_out
) {
    if (record == NULL || seen_ids == NULL || seen_count == NULL || known_out == NULL ||
        route_stride < (uint32_t)offsetof(GafimeNumericRoute, reserved) ||
        *seen_count >= MAX_ROUTE_RECORDS) {
        return -1;
    }

    GafimeNumericRoute route;
    memset(&route, 0, sizeof(route));
    const size_t copy_size = route_stride < sizeof(route) ? route_stride : sizeof(route);
    memcpy(&route, record, copy_size);
    /* `struct_size` describes the producer record. It may exceed the
     * caller-provided stride; only `copy_size` bytes were copied above, so the
     * unknown tail is deliberately ignored. */
    if (GAFIME_ABI_VERSION_MAJOR_OF(route.abi_version) !=
            GAFIME_PRECISION_ABI_VERSION_MAJOR ||
        GAFIME_ABI_VERSION_MINOR_OF(route.abi_version) < GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR ||
        route.struct_size < (uint32_t)offsetof(GafimeNumericRoute, reserved) ||
        route.route_id == 0 ||
        (route.flags & GAFIME_ABI_REQUIRED_FLAG_MASK) != 0) {
        return -1;
    }
    if (copy_size >= sizeof(route) && route.struct_size >= sizeof(route) &&
        memcmp(route.reserved, (uint64_t[8]){0}, sizeof(route.reserved)) != 0) {
        return -1;
    }
    if (route_id_seen(seen_ids, *seen_count, route.route_id)) return -1;
    seen_ids[(*seen_count)++] = route.route_id;

    if (!known_route_id(route.route_id)) {
        /* Unknown profile/dtype/overflow values are never dispatched. */
        return 0;
    }
    if (!canonical_route(&route)) return -1;

    /* Copy the known prefix before embedding it in a fixed ABI 1.1 structure. */
    *known_out = route;
    known_out->struct_size = sizeof(*known_out);
    return 1;
}

static uint32_t expected_route_mask(uint32_t expected_count) {
    if (expected_count == 1) return 1u << GAFIME_NUMERIC_ROUTE_FP32;
    if (expected_count == 3) {
        return (1u << GAFIME_NUMERIC_ROUTE_FP32) |
            (1u << GAFIME_NUMERIC_ROUTE_MIXED) |
            (1u << GAFIME_NUMERIC_ROUTE_FP64);
    }
    return 0;
}

static int collect_route_records(
    const void* records,
    uint32_t count,
    uint32_t route_stride,
    uint32_t expected_mask,
    GafimeNumericRoute* known_routes,
    uint32_t* known_count_out,
    uint32_t* known_mask_out
) {
    if (records == NULL || known_routes == NULL || known_count_out == NULL ||
        known_mask_out == NULL || count > MAX_ROUTE_RECORDS || route_stride <
            (uint32_t)offsetof(GafimeNumericRoute, reserved)) {
        return 1;
    }
    uint32_t seen_ids[MAX_ROUTE_RECORDS] = {0};
    uint32_t seen_count = 0;
    uint32_t known_count = 0;
    uint32_t known_mask = 0;
    int failed = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const unsigned char* raw = (const unsigned char*)records +
            (size_t)index * route_stride;
        GafimeNumericRoute route;
        const int result = parse_route_record(
            raw, route_stride, seen_ids, &seen_count, &route);
        if (result < 0) {
            fprintf(stderr, "invalid or duplicate route record at index %u\n", index);
            failed = 1;
        } else if (result > 0) {
            if (known_count >= MAX_ROUTE_RECORDS) {
                failed = 1;
            } else {
                known_routes[known_count++] = route;
                known_mask |= 1u << route.route_id;
            }
        }
    }
    *known_count_out = known_count;
    *known_mask_out = known_mask;
    if (known_mask != expected_mask) {
        fprintf(stderr, "known route mask 0x%x does not match expected 0x%x\n",
                known_mask, expected_mask);
        failed = 1;
    }
    return failed;
}

static GafimeNumericMatrixDesc matrix_desc_for(const GafimeNumericRoute* route) {
    GafimeNumericMatrixDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = *route;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 4;
    desc.cols = 2;
    desc.row_stride = 2;
    desc.bytes = 8 * dtype_size(route->storage_dtype);
    return desc;
}

static int require_rejected(int actual, int expected, const char* label) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected status %d, got %d\n", label, expected, actual);
        return 1;
    }
    return 0;
}

static int validate_fail_closed_inputs(const Api11* api, const GafimeNumericRoute* route) {
    int failed = 0;
    GafimeNumericMatrixDesc desc = matrix_desc_for(route);
    GafimeGpuMatrix matrix = NULL;

    desc.route.abi_version = (2u << 16) | 0u;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_ABI_MISMATCH,
                               "major-version mismatch");
    desc = matrix_desc_for(route);
    desc.route.struct_size = (uint32_t)offsetof(GafimeNumericRoute, reserved) - 1;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_ABI_MISMATCH,
                               "short route prefix");
    desc = matrix_desc_for(route);
    desc.struct_size = (uint32_t)offsetof(GafimeNumericMatrixDesc, reserved) - 1;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_ABI_MISMATCH,
                               "short matrix prefix");
    desc = matrix_desc_for(route);
    desc.route.reserved[0] = 1;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_INVALID_ARGUMENT,
                               "nonzero reserved route field");
    desc = matrix_desc_for(route);
    desc.route.flags = 0x1u;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_INVALID_ARGUMENT,
                               "unknown required route flag");
    desc = matrix_desc_for(route);
    desc.route.storage_dtype = 0x1000u;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_UNSUPPORTED_BACKEND,
                               "future dtype requested from ABI 1.1 payload");
    desc = matrix_desc_for(route);
    desc.route.result_dtype = route->result_dtype == GAFIME_DTYPE_F32 ?
        GAFIME_DTYPE_F64 : GAFIME_DTYPE_F32;
    failed |= require_rejected(api->alloc(0, &desc, &matrix), GAFIME_STATUS_UNSUPPORTED_BACKEND,
                               "contradictory route");
    desc = matrix_desc_for(route);
    desc.rows = UINT64_MAX;
    desc.cols = 2;
    desc.row_stride = 2;
    desc.bytes = UINT64_MAX;
    failed |= require_rejected(
        api->alloc(0, &desc, &matrix), GAFIME_STATUS_INVALID_ARGUMENT,
        "matrix element and byte-size overflow");
    {
        _Alignas(8) unsigned char misaligned_desc[sizeof(GafimeNumericMatrixDesc) + 1];
        failed |= require_rejected(
            api->alloc(
                0,
                (const GafimeNumericMatrixDesc*)(const void*)(misaligned_desc + 1),
                &matrix),
            GAFIME_STATUS_INVALID_ARGUMENT, "misaligned matrix descriptor");
    }
    desc = matrix_desc_for(route);
    desc.route.struct_size = sizeof(GafimeNumericRoute) + sizeof(uint64_t);
    failed |= require_rejected(
        api->alloc(0, &desc, &matrix), GAFIME_STATUS_INVALID_ARGUMENT,
        "oversized route claim inside fixed matrix descriptor");

    /* Upper-half flags are explicitly ignorable. */
    desc = matrix_desc_for(route);
    desc.route.flags = 0x80000000u;
    int status = api->alloc(0, &desc, &matrix);
    if (status != GAFIME_STATUS_OK || matrix == NULL) {
        fprintf(stderr, "ignorable future flag was not accepted: %d\n", status);
        failed = 1;
    } else {
        failed |= require_rejected(api->free_matrix(matrix), GAFIME_STATUS_OK,
                                   "free ignorable-flag matrix");
    }

    /* A newer additive minor and a larger outer structure preserve the prefix. */
    {
        struct FutureMatrixDesc {
            GafimeNumericMatrixDesc known;
            uint64_t future_fields[2];
        } future;
        memset(&future, 0, sizeof(future));
        future.known = matrix_desc_for(route);
        future.known.abi_version = (1u << 16) | 2u;
        future.known.struct_size = sizeof(future);
        future.known.route.abi_version = (1u << 16) | 2u;
        status = api->alloc(0, &future.known, &matrix);
        if (status != GAFIME_STATUS_OK || matrix == NULL) {
            fprintf(stderr, "newer additive matrix prefix was not accepted: %d\n", status);
            failed = 1;
        } else {
            failed |= require_rejected(api->free_matrix(matrix), GAFIME_STATUS_OK,
                                       "free newer-minor matrix");
        }
    }
    return failed;
}

static int run_route(const Api11* api, const GafimeNumericRoute* route, uint32_t backend_kind) {
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
    const void* features = route->storage_dtype == GAFIME_DTYPE_F32 ?
        (const void*)features_f32 : (const void*)features_f64;
    const void* target = route->storage_dtype == GAFIME_DTYPE_F32 ?
        (const void*)target_f32 : (const void*)target_f64;
    GafimeConstBufferView feature_view = const_view(route->storage_dtype, features, 8);
    GafimeConstBufferView target_view = const_view(route->storage_dtype, target, 4);
    GafimeNumericMatrixDesc desc = matrix_desc_for(route);
    GafimeGpuMatrix matrix = NULL;
    int status = api->alloc(0, &desc, &matrix);
    if (status != GAFIME_STATUS_OK || matrix == NULL) {
        fprintf(stderr, "route %u allocation failed: %d\n", route->route_id, status);
        return 1;
    }

    int failed = 0;
    {
        /* Standalone typed views remain forward-extensible. */
        FutureConstBufferView future_features;
        FutureConstBufferView future_target;
        memset(&future_features, 0, sizeof(future_features));
        memset(&future_target, 0, sizeof(future_target));
        future_features.known = feature_view;
        future_features.known.struct_size = sizeof(future_features);
        future_features.future_field = UINT64_C(0x1234);
        future_target.known = target_view;
        future_target.known.struct_size = sizeof(future_target);
        future_target.future_field = UINT64_C(0x5678);
        status = api->upload(
            matrix, route, &future_features.known, &future_target.known, 4, 2);
        if (status != GAFIME_STATUS_OK) {
            fprintf(stderr, "route %u rejected standalone typed-view tail: %d\n",
                    route->route_id, status);
            failed = 1;
            goto cleanup;
        }
    }
    status = api->upload(matrix, route, &feature_view, &target_view, 4, 2);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "route %u upload failed: %d\n", route->route_id, status);
        failed = 1;
        goto cleanup;
    }
    {
        struct FutureRoute {
            GafimeNumericRoute known;
            uint64_t future_fields[2];
        } future;
        memset(&future, 0, sizeof(future));
        future.known = *route;
        future.known.abi_version = (1u << 16) | 2u;
        future.known.struct_size = sizeof(future);
        future.future_fields[0] = UINT64_C(0x12);
        status = api->upload(matrix, &future.known, &feature_view, &target_view, 4, 2);
        if (status != GAFIME_STATUS_OK) {
            fprintf(stderr, "route %u rejected an additive route tail: %d\n",
                    route->route_id, status);
            failed = 1;
            goto cleanup;
        }
    }
    status = api->update_target(matrix, route, &target_view, 4);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "route %u target update failed: %d\n", route->route_id, status);
        failed = 1;
        goto cleanup;
    }

    /* Fail closed on malformed typed views before native execution. */
    {
        GafimeConstBufferView invalid = feature_view;
        invalid.byte_length -= 1;
        failed |= require_rejected(
            api->upload(matrix, route, &invalid, &target_view, 4, 2),
            GAFIME_STATUS_INVALID_ARGUMENT, "short feature byte length");
        invalid = feature_view;
        invalid.byte_stride *= 2;
        failed |= require_rejected(
            api->upload(matrix, route, &invalid, &target_view, 4, 2),
            GAFIME_STATUS_INVALID_ARGUMENT, "invalid feature stride");
        invalid = feature_view;
        invalid.dtype = invalid.dtype == GAFIME_DTYPE_F32 ? GAFIME_DTYPE_F64 : GAFIME_DTYPE_F32;
        failed |= require_rejected(
            api->upload(matrix, route, &invalid, &target_view, 4, 2),
            GAFIME_STATUS_INVALID_ARGUMENT, "wrong feature dtype");
        invalid = feature_view;
        invalid.data = NULL;
        failed |= require_rejected(
            api->upload(matrix, route, &invalid, &target_view, 4, 2),
            GAFIME_STATUS_INVALID_ARGUMENT, "null nonempty feature buffer");
        _Alignas(8) unsigned char misaligned_storage[8 * sizeof(double) + 1];
        invalid = feature_view;
        invalid.data = misaligned_storage + 1;
        failed |= require_rejected(
            api->upload(matrix, route, &invalid, &target_view, 4, 2),
            GAFIME_STATUS_INVALID_ARGUMENT, "misaligned feature buffer");
    }

    {
        const uint32_t combo = 0;
        const uint32_t metric = GAFIME_METRIC_PEARSON;
        GafimeArityChunk chunk;
        memset(&chunk, 0, sizeof(chunk));
        chunk.arity = 1;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.combo_count = 1;
        chunk.descriptor_count = 1;
        GafimeLaunchProtocol base;
        memset(&base, 0, sizeof(base));
        base.abi_version = GAFIME_ABI_VERSION;
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
        GafimeNumericLaunchProtocol protocol;
        memset(&protocol, 0, sizeof(protocol));
        protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
        protocol.struct_size = sizeof(protocol);
        protocol.route = *route;
        protocol.base = &base;

        uint32_t combo_out = UINT32_MAX;
        float metric_f32 = 0.0f;
        double metric_f64 = 0.0;
        void* metric_data = route->result_dtype == GAFIME_DTYPE_F32 ?
            (void*)&metric_f32 : (void*)&metric_f64;
        uint32_t rank = 0;
        uint32_t family = 0;
        uint64_t candidate_id = UINT64_MAX;
        uint32_t row_flags = UINT32_MAX;
        GafimeNumericResultTable result;
        memset(&result, 0, sizeof(result));
        result.abi_version = GAFIME_PRECISION_ABI_VERSION;
        result.struct_size = sizeof(result);
        result.max_arity = 1;
        result.metric_count = 1;
        result.capacity = 1;
        result.combo_indices = &combo_out;
        result.metric_values = mutable_view(route->result_dtype, metric_data, 1);
        result.ranks = &rank;
        result.families = &family;
        result.candidate_ids = &candidate_id;
        result.row_flags = &row_flags;

        {
            FutureMutableBufferView future_metric_values;
            memset(&future_metric_values, 0, sizeof(future_metric_values));
            future_metric_values.known = result.metric_values;
            future_metric_values.known.struct_size = sizeof(future_metric_values);
            future_metric_values.future_field = UINT64_C(0x9abc);
            result.metric_values = future_metric_values.known;
            failed |= require_rejected(
                api->execute(matrix, &protocol, &result),
                GAFIME_STATUS_INVALID_ARGUMENT,
                "embedded result view with future tail");
            result.metric_values = mutable_view(route->result_dtype, metric_data, 1);
        }

        uint64_t execution_peak = 0;
        uint64_t permutation_peak = 0;
        status = api->execution_memory(matrix, &protocol, &execution_peak);
        if (status != GAFIME_STATUS_OK || execution_peak == 0) {
            fprintf(stderr, "route %u execution forecast failed: %d/%llu\n", route->route_id,
                    status, (unsigned long long)execution_peak);
            failed = 1;
        }
        result.row_count = result.capacity + 1;
        failed |= require_rejected(
            api->execute(matrix, &protocol, &result),
            GAFIME_STATUS_INVALID_ARGUMENT, "result row count exceeds capacity");
        result.row_count = 0;
        {
            _Alignas(8) unsigned char misaligned_rank[sizeof(uint32_t) + 1];
            result.ranks = (uint32_t*)(void*)(misaligned_rank + 1);
            failed |= require_rejected(
                api->execute(matrix, &protocol, &result),
                GAFIME_STATUS_INVALID_ARGUMENT, "misaligned structural result buffer");
            result.ranks = &rank;
        }
        {
            const GafimeNumericResultTable valid_result = result;
            result.max_arity = 5;
            result.capacity = UINT64_MAX;
            result.metric_values.element_capacity = UINT64_MAX;
            result.metric_values.byte_length = UINT64_MAX;
            failed |= require_rejected(
                api->execute(matrix, &protocol, &result),
                GAFIME_STATUS_INVALID_ARGUMENT,
                "result structural and numeric byte-size overflow");
            result = valid_result;
        }
        status = api->execute(matrix, &protocol, &result);
        const double visible = route->result_dtype == GAFIME_DTYPE_F32 ?
            (double)metric_f32 : metric_f64;
        if (status != GAFIME_STATUS_OK || result.row_count != 1 || combo_out != 0 ||
            !isfinite(visible) || fabs(visible - 1.0) > 1.0e-5) {
            fprintf(stderr,
                    "route %u execute mismatch: status=%d rows=%llu combo=%u value=%.17g\n",
                    route->route_id, status, (unsigned long long)result.row_count,
                    combo_out, visible);
            failed = 1;
        }

        base.permutations.permutation_count = 2;
        base.permutations.seed = 0x12345678u;
        status = api->permutation_memory(matrix, &protocol, 1, &permutation_peak);
        if (status != GAFIME_STATUS_OK || permutation_peak == 0) {
            fprintf(stderr, "route %u permutation forecast failed: %d/%llu\n",
                    route->route_id, status, (unsigned long long)permutation_peak);
            failed = 1;
        }
        float p_f32 = 0.0f;
        double p_f64 = 0.0;
        void* p_data = route->result_dtype == GAFIME_DTYPE_F32 ?
            (void*)&p_f32 : (void*)&p_f64;
        GafimeNumericSignificanceTable significance;
        memset(&significance, 0, sizeof(significance));
        significance.abi_version = GAFIME_PRECISION_ABI_VERSION;
        significance.struct_size = sizeof(significance);
        significance.metric_count = 1;
        significance.row_count = 1;
        significance.candidate_ids = &candidate_id;
        significance.observed_metric_values = const_view(route->result_dtype, metric_data, 1);
        significance.p_values = mutable_view(route->result_dtype, p_data, 1);
        {
            FutureConstBufferView future_observed;
            FutureMutableBufferView future_p_values;
            memset(&future_observed, 0, sizeof(future_observed));
            memset(&future_p_values, 0, sizeof(future_p_values));
            future_observed.known = significance.observed_metric_values;
            future_observed.known.struct_size = sizeof(future_observed);
            future_observed.future_field = UINT64_C(0xdef0);
            significance.observed_metric_values = future_observed.known;
            failed |= require_rejected(
                api->permutation(matrix, &protocol, &significance),
                GAFIME_STATUS_INVALID_ARGUMENT,
                "embedded significance const view with future tail");
            significance.observed_metric_values = const_view(route->result_dtype, metric_data, 1);
            future_p_values.known = significance.p_values;
            future_p_values.known.struct_size = sizeof(future_p_values);
            future_p_values.future_field = UINT64_C(0x1357);
            significance.p_values = future_p_values.known;
            failed |= require_rejected(
                api->permutation(matrix, &protocol, &significance),
                GAFIME_STATUS_INVALID_ARGUMENT,
                "embedded significance mutable view with future tail");
            significance.p_values = mutable_view(route->result_dtype, p_data, 1);
        }
        {
            _Alignas(8) unsigned char misaligned_id[sizeof(uint64_t) + 1];
            significance.candidate_ids = (const uint64_t*)(const void*)(misaligned_id + 1);
            failed |= require_rejected(
                api->permutation(matrix, &protocol, &significance),
                GAFIME_STATUS_INVALID_ARGUMENT, "misaligned significance candidate IDs");
            significance.candidate_ids = &candidate_id;
        }
        status = api->permutation(matrix, &protocol, &significance);
        const double p_value = route->result_dtype == GAFIME_DTYPE_F32 ?
            (double)p_f32 : p_f64;
        if (status != GAFIME_STATUS_OK || !isfinite(p_value) || p_value < 0.0 || p_value > 1.0) {
            fprintf(stderr, "route %u significance mismatch: %d/%.17g\n",
                    route->route_id, status, p_value);
            failed = 1;
        }
        {
            uint64_t duplicate_candidate_ids[2] = {0, 0};
            float observed_f32_two[2] = {1.0f, 1.0f};
            float p_f32_two[2] = {0.0f, 0.0f};
            double observed_f64_two[2] = {1.0, 1.0};
            double p_f64_two[2] = {0.0, 0.0};
            const void* observed_two = route->result_dtype == GAFIME_DTYPE_F32 ?
                (const void*)observed_f32_two : (const void*)observed_f64_two;
            void* p_two = route->result_dtype == GAFIME_DTYPE_F32 ?
                (void*)p_f32_two : (void*)p_f64_two;
            significance.row_count = 2;
            significance.candidate_ids = duplicate_candidate_ids;
            significance.observed_metric_values = const_view(route->result_dtype, observed_two, 2);
            significance.p_values = mutable_view(route->result_dtype, p_two, 2);
            failed |= require_rejected(
                api->permutation(matrix, &protocol, &significance),
                GAFIME_STATUS_INVALID_ARGUMENT,
                "significance rows exceed planned rows");
            significance.row_count = 1;
            significance.candidate_ids = &candidate_id;
            significance.observed_metric_values = const_view(route->result_dtype, metric_data, 1);
            significance.p_values = mutable_view(route->result_dtype, p_data, 1);
        }

        uint64_t overflow_count = UINT64_MAX;
        uint32_t diagnostic_flags = UINT32_MAX;
        GafimeNumericInteractionDiagnosticBatch diagnostics;
        memset(&diagnostics, 0, sizeof(diagnostics));
        diagnostics.abi_version = GAFIME_PRECISION_ABI_VERSION;
        diagnostics.struct_size = sizeof(diagnostics);
        diagnostics.route = *route;
        diagnostics.max_arity = 1;
        diagnostics.row_count = 1;
        diagnostics.combo_indices = &combo;
        diagnostics.combo_index_count = 1;
        diagnostics.overflow_row_counts = &overflow_count;
        diagnostics.row_flags = &diagnostic_flags;
        {
            _Alignas(8) unsigned char misaligned_overflow[sizeof(uint64_t) + 1];
            diagnostics.overflow_row_counts = (uint64_t*)(void*)(misaligned_overflow + 1);
            failed |= require_rejected(
                api->diagnostics(matrix, &diagnostics),
                GAFIME_STATUS_INVALID_ARGUMENT, "misaligned diagnostic structural buffer");
            diagnostics.overflow_row_counts = &overflow_count;
        }
        status = api->diagnostics(matrix, &diagnostics);
        if (status != GAFIME_STATUS_OK || overflow_count != 0 || diagnostic_flags != 0) {
            fprintf(stderr, "route %u diagnostics mismatch: %d/%llu/%u\n",
                    route->route_id, status, (unsigned long long)overflow_count,
                    diagnostic_flags);
            failed = 1;
        }
    }

cleanup:
    status = api->free_matrix(matrix);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "route %u free failed: %d\n", route->route_id, status);
        failed = 1;
    }
    return failed;
}

static FutureRouteRecord unknown_future_route(void) {
    FutureRouteRecord record;
    memset(&record, 0, sizeof(record));
    record.known.abi_version = (1u << 16) | 2u;
    record.known.struct_size = sizeof(record);
    record.known.route_id = 0x10001u;
    record.known.profile = 0x10001u;
    record.known.storage_dtype = 0x10001u;
    record.known.pointwise_dtype = 0x10001u;
    record.known.reduction_dtype = 0x10001u;
    record.known.result_dtype = 0x10001u;
    record.known.overflow_policy = 0x10001u;
    record.known.flags = GAFIME_ABI_IGNORABLE_FLAG_MASK;
    record.future_fields[0] = UINT64_C(0x123456789abcdef0);
    return record;
}

static int expect_collection_failure(
    const FutureRouteRecord* records,
    uint32_t count,
    uint32_t expected_mask
) {
    GafimeNumericRoute known_routes[MAX_ROUTE_RECORDS];
    uint32_t known_count = 0;
    uint32_t known_mask = 0;
    return collect_route_records(
        records, count, sizeof(FutureRouteRecord), expected_mask,
        known_routes, &known_count, &known_mask) != 0;
}

/* Exercise the complete route-selection path with a future record in-band. */
static int test_future_route_records(
    const Api11* api,
    uint32_t backend_kind,
    const GafimeNumericRoute* current_routes,
    uint32_t current_count,
    uint32_t expected_mask
) {
    if (current_count == 0 || current_count > 3) return 1;
    FutureRouteRecord records[MAX_ROUTE_RECORDS];
    memset(records, 0, sizeof(records));
    for (uint32_t index = 0; index < current_count; ++index) {
        records[index].known = current_routes[index];
        records[index].known.abi_version = (1u << 16) | 2u;
        records[index].known.struct_size = sizeof(FutureRouteRecord);
    }
    records[current_count] = unknown_future_route();
    const uint32_t count = current_count + 1;

    GafimeNumericRoute known_routes[MAX_ROUTE_RECORDS];
    uint32_t known_count = 0;
    uint32_t known_mask = 0;
    int failed = collect_route_records(
        records, count, sizeof(FutureRouteRecord), expected_mask,
        known_routes, &known_count, &known_mask);
    if (!failed) {
        for (uint32_t index = 0; index < known_count; ++index) {
            failed |= run_route(api, &known_routes[index], backend_kind);
        }
    }

    /* Duplicate unknown IDs are rejected even though their semantics are skipped. */
    FutureRouteRecord duplicate_unknown[MAX_ROUTE_RECORDS];
    memcpy(duplicate_unknown, records, sizeof(records));
    duplicate_unknown[count] = records[current_count];
    if (!expect_collection_failure(duplicate_unknown, count + 1, expected_mask)) {
        fprintf(stderr, "duplicate unknown route ID was accepted\n");
        failed = 1;
    }

    /* A recognized route ID with a contradictory dtype is not an unknown route. */
    FutureRouteRecord contradictory[MAX_ROUTE_RECORDS];
    memcpy(contradictory, records, sizeof(records));
    contradictory[0].known.result_dtype = contradictory[0].known.result_dtype ==
        GAFIME_DTYPE_F32 ? GAFIME_DTYPE_F64 : GAFIME_DTYPE_F32;
    contradictory[0].known.struct_size = sizeof(FutureRouteRecord);
    if (!expect_collection_failure(contradictory, current_count, expected_mask)) {
        fprintf(stderr, "contradictory known route was accepted\n");
        failed = 1;
    }

    FutureRouteRecord required_flag[MAX_ROUTE_RECORDS];
    memcpy(required_flag, records, sizeof(records));
    required_flag[current_count].known.flags = 1u;
    if (!expect_collection_failure(required_flag, count, expected_mask)) {
        fprintf(stderr, "unknown required route flag was accepted\n");
        failed = 1;
    }

    FutureRouteRecord major_mismatch[MAX_ROUTE_RECORDS];
    memcpy(major_mismatch, records, sizeof(records));
    major_mismatch[current_count].known.abi_version = 2u << 16;
    if (!expect_collection_failure(major_mismatch, count, expected_mask)) {
        fprintf(stderr, "future route major mismatch was accepted\n");
        failed = 1;
    }

    FutureRouteRecord larger_producer_claim[MAX_ROUTE_RECORDS];
    memcpy(larger_producer_claim, records, sizeof(records));
    larger_producer_claim[current_count].known.struct_size = sizeof(FutureRouteRecord) + 8;
    if (expect_collection_failure(larger_producer_claim, count, expected_mask)) {
        fprintf(stderr, "larger producer route record was rejected\n");
        failed = 1;
    }
    return failed;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s PAYLOAD BACKEND_KIND EXPECTED_ROUTE_COUNT\n", argv[0]);
        return 2;
    }
    const uint32_t backend_kind = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t expected_count = (uint32_t)strtoul(argv[3], NULL, 10);
    GafimeTestLibrary library = gafime_test_library_open(argv[1]);
    if (library == NULL) {
        fprintf(stderr, "could not load %s: %s\n", argv[1], gafime_test_library_error());
        return 1;
    }
    Api11 api;
    memset(&api, 0, sizeof(api));
    GAFIME_TEST_LOAD_FUNCTION(library, api.routes, "gafime_gpu_numeric_routes_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.alloc, "gafime_gpu_matrix_alloc_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.upload, "gafime_gpu_matrix_upload_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.update_target, "gafime_gpu_matrix_update_target_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execute, "gafime_gpu_execute_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.execution_memory,
                              "gafime_gpu_execution_memory_peak_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.permutation_memory,
                              "gafime_gpu_permutation_memory_peak_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.permutation,
                              "gafime_gpu_permutation_pvalues_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.diagnostics,
                              "gafime_gpu_interaction_diagnostics_v2");
    GAFIME_TEST_LOAD_FUNCTION(library, api.free_matrix, "gafime_gpu_matrix_free_v2");

    uint32_t count = 0;
    const uint32_t route_stride = sizeof(FutureRouteRecord);
    const uint32_t expected_mask = expected_route_mask(expected_count);
    if (expected_mask == 0) {
        fprintf(stderr, "unsupported expected route count: %u\n", expected_count);
        gafime_test_library_close(library);
        return 2;
    }
    int status = api.routes(0, GAFIME_PRECISION_ABI_VERSION, route_stride,
                            NULL, 0, &count);
    if (unavailable_status(status)) {
        gafime_test_library_close(library);
        return 77;
    }
    if (status != GAFIME_STATUS_OK || count < expected_count || count > MAX_ROUTE_RECORDS) {
        fprintf(stderr, "route count mismatch: status=%d actual=%u expected=%u\n",
                status, count, expected_count);
        gafime_test_library_close(library);
        return 1;
    }
    if (api.routes(0, (2u << 16) | 1u, sizeof(GafimeNumericRoute), NULL, 0, &count) !=
        GAFIME_STATUS_ABI_MISMATCH) {
        fprintf(stderr, "route enumeration accepted an incompatible major version\n");
        gafime_test_library_close(library);
        return 1;
    }

    FutureRouteRecord routes[MAX_ROUTE_RECORDS];
    memset(routes, 0xa5, sizeof(routes));
    status = api.routes(0, GAFIME_PRECISION_ABI_VERSION, route_stride,
                        (GafimeNumericRoute*)routes, count, &count);
    if (status != GAFIME_STATUS_OK) {
        fprintf(stderr, "route enumeration failed: %d\n", status);
        gafime_test_library_close(library);
        return 1;
    }
    if (api.routes(0, GAFIME_PRECISION_ABI_VERSION,
                   (uint32_t)offsetof(GafimeNumericRoute, reserved) - 1,
                   (GafimeNumericRoute*)routes, count, &count) != GAFIME_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "route enumeration accepted a short record stride\n");
        gafime_test_library_close(library);
        return 1;
    }
    if (api.routes(0, GAFIME_PRECISION_ABI_VERSION,
                   (uint32_t)offsetof(GafimeNumericRoute, reserved) + 1,
                   (GafimeNumericRoute*)routes, count, &count) != GAFIME_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "route enumeration accepted a misaligned record stride\n");
        gafime_test_library_close(library);
        return 1;
    }
    {
        _Alignas(8) unsigned char misaligned_routes[sizeof(routes) + 1];
        if (api.routes(
                0, GAFIME_PRECISION_ABI_VERSION, route_stride,
                (GafimeNumericRoute*)(void*)(misaligned_routes + 1), count, &count) !=
            GAFIME_STATUS_INVALID_ARGUMENT) {
            fprintf(stderr, "route enumeration accepted a misaligned output buffer\n");
            gafime_test_library_close(library);
            return 1;
        }
    }
    {
        _Alignas(8) unsigned char misaligned_count[sizeof(uint32_t) + 1];
        if (api.routes(
                0, GAFIME_PRECISION_ABI_VERSION, sizeof(GafimeNumericRoute),
                NULL, 0, (uint32_t*)(void*)(misaligned_count + 1)) !=
            GAFIME_STATUS_INVALID_ARGUMENT) {
            fprintf(stderr, "route enumeration accepted a misaligned count pointer\n");
            gafime_test_library_close(library);
            return 1;
        }
    }
    {
        enum {
            SHORT_ROUTE_STRIDE = offsetof(GafimeNumericRoute, reserved),
        };
        _Alignas(GafimeNumericRoute)
            unsigned char short_routes[MAX_ROUTE_RECORDS][SHORT_ROUTE_STRIDE];
        uint32_t short_count = count;
        memset(short_routes, 0xa5, sizeof(short_routes));
        status = api.routes(
            0, GAFIME_PRECISION_ABI_VERSION, SHORT_ROUTE_STRIDE,
            (GafimeNumericRoute*)(void*)short_routes, short_count, &short_count);
        if (status != GAFIME_STATUS_OK || short_count != count) {
            fprintf(stderr,
                    "short-stride route enumeration failed: status=%d actual=%u expected=%u\n",
                    status, short_count, count);
            gafime_test_library_close(library);
            return 1;
        }
        for (uint32_t index = 0; index < short_count; ++index) {
            uint32_t producer_size = 0;
            memcpy(
                &producer_size,
                short_routes[index] + offsetof(GafimeNumericRoute, struct_size),
                sizeof(producer_size));
            if (producer_size != sizeof(GafimeNumericRoute)) {
                fprintf(stderr,
                        "short-stride route %u reported producer size %u, expected %zu\n",
                        index, producer_size, sizeof(GafimeNumericRoute));
                gafime_test_library_close(library);
                return 1;
            }
        }
    }

    int failed = 0;
    GafimeNumericRoute known_routes[MAX_ROUTE_RECORDS];
    uint32_t known_count = 0;
    uint32_t known_mask = 0;
    failed |= collect_route_records(
        routes, count, route_stride, expected_mask,
        known_routes, &known_count, &known_mask);
    if (known_count != expected_count) {
        fprintf(stderr, "known route count %u does not match expected %u\n",
                known_count, expected_count);
        failed = 1;
    }
    if (known_count != 0) {
        failed |= validate_fail_closed_inputs(&api, &known_routes[0]);
    }
    failed |= test_future_route_records(
        &api, backend_kind, known_routes, known_count, expected_mask);

    gafime_test_library_close(library);
    if (!failed) {
        printf(
            "{\"schema\":\"gafime.abi-1.1-consumer-result.v1\","
            "\"status\":\"pass\",\"abi_surface\":\"numeric-route-v2\","
            "\"backend_kind\":%u,\"route_count\":%u,"
            "\"route_mask\":%u,\"operations\":["
            "\"numeric_routes\",\"matrix_alloc\",\"matrix_upload\","
            "\"matrix_update_target\",\"execute\",\"execution_memory_peak\","
            "\"permutation_memory_peak\",\"permutation_pvalues\","
            "\"interaction_diagnostics\",\"matrix_free\"]}\n",
            backend_kind, count, known_mask);
    }
    return failed;
}
