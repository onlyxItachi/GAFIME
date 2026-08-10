/*
 * Synthetic future payload.  It advertises ABI 1.2-sized route records and an
 * unknown numeric route without claiming that GAFIME implements integer
 * execution today.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../../../src/common/gafime_gpu_abi.hpp"

typedef struct FutureNumericRoute12 {
    GafimeNumericRoute known_prefix;
    uint32_t future_rounding_policy;
    uint32_t future_accumulator_width;
    uint64_t future_reserved;
} FutureNumericRoute12;

_Static_assert(sizeof(FutureNumericRoute12) > sizeof(GafimeNumericRoute),
               "the synthetic ABI 1.2 record must be larger than ABI 1.1");

static GafimeNumericRoute route(uint32_t id) {
    GafimeNumericRoute value;
    memset(&value, 0, sizeof(value));
    value.abi_version = (1u << 16) | 2u;
    value.struct_size = sizeof(FutureNumericRoute12);
    value.route_id = id;
    value.overflow_policy = GAFIME_OVERFLOW_IEEE;
    switch (id) {
    case GAFIME_NUMERIC_ROUTE_FP32:
        value.profile = GAFIME_PRECISION_FP32;
        value.storage_dtype = GAFIME_DTYPE_F32;
        value.pointwise_dtype = GAFIME_DTYPE_F32;
        value.reduction_dtype = GAFIME_DTYPE_F32;
        value.result_dtype = GAFIME_DTYPE_F32;
        break;
    case GAFIME_NUMERIC_ROUTE_MIXED:
        value.profile = GAFIME_PRECISION_MIXED;
        value.storage_dtype = GAFIME_DTYPE_F32;
        value.pointwise_dtype = GAFIME_DTYPE_F32;
        value.reduction_dtype = GAFIME_DTYPE_F64;
        value.result_dtype = GAFIME_DTYPE_F64;
        value.flags = 0x80000000u; /* Unknown but explicitly ignorable. */
        break;
    case GAFIME_NUMERIC_ROUTE_FP64:
        value.profile = GAFIME_PRECISION_FP64;
        value.storage_dtype = GAFIME_DTYPE_F64;
        value.pointwise_dtype = GAFIME_DTYPE_F64;
        value.reduction_dtype = GAFIME_DTYPE_F64;
        value.result_dtype = GAFIME_DTYPE_F64;
        break;
    default:
        value.route_id = 0x10001u;
        value.profile = 0x10001u;
        value.storage_dtype = 0x1000u;
        value.pointwise_dtype = 0x1001u;
        value.reduction_dtype = 0x1002u;
        value.result_dtype = GAFIME_DTYPE_F64;
        value.flags = 0x40000000u; /* Unknown but explicitly ignorable. */
        break;
    }
    return value;
}

GAFIME_GPU_API int gafime_gpu_numeric_routes_v2(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    uint32_t route_stride,
    GafimeNumericRoute* routes_out,
    uint32_t route_capacity,
    uint32_t* route_count_out
) {
    (void)device_id;
    const uint32_t stable_prefix = (uint32_t)offsetof(GafimeNumericRoute, reserved);
    if (route_count_out == NULL) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *route_count_out = 4;
    if (GAFIME_ABI_VERSION_MAJOR_OF(consumer_abi_version) != 1u ||
        GAFIME_ABI_VERSION_MINOR_OF(consumer_abi_version) < 1u) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (routes_out == NULL) {
        return route_capacity == 0 ? GAFIME_STATUS_OK : GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (route_capacity < 4 || route_stride < stable_prefix) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t route_ids[4] = {
        GAFIME_NUMERIC_ROUTE_FP32,
        0x10001u,
        GAFIME_NUMERIC_ROUTE_MIXED,
        GAFIME_NUMERIC_ROUTE_FP64,
    };
    for (uint32_t index = 0; index < 4; ++index) {
        FutureNumericRoute12 future;
        memset(&future, 0, sizeof(future));
        future.known_prefix = route(route_ids[index]);
        future.future_rounding_policy = 0x1200u + index;
        future.future_accumulator_width = 128u;
        const uint32_t write_size = route_stride < sizeof(future) ?
            route_stride : (uint32_t)sizeof(future);
        /* The producer record remains ABI 1.2-sized even when an older
         * consumer supplies only the ABI 1.1 prefix stride. The copy below is
         * still bounded by that caller-owned stride. */
        future.known_prefix.struct_size = sizeof(FutureNumericRoute12);
        unsigned char* destination = (unsigned char*)routes_out +
            (size_t)index * route_stride;
        memset(destination, 0, route_stride);
        memcpy(destination, &future, write_size);
    }
    return GAFIME_STATUS_OK;
}
