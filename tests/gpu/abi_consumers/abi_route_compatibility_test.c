/* Independent ABI 1.1 route parser exercised against a synthetic ABI 1.2 payload. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/common/gafime_gpu_abi.hpp"
#include "abi_dynamic_load.h"

typedef int (*NumericRoutesFn)(uint32_t, uint32_t, uint32_t, GafimeNumericRoute*,
                               uint32_t, uint32_t*);

enum ParseStatus {
    PARSE_OK = 0,
    PARSE_BAD_MAJOR,
    PARSE_BAD_MINOR,
    PARSE_SHORT_PREFIX,
    PARSE_BAD_FLAGS,
    PARSE_NONZERO_RESERVED,
    PARSE_DUPLICATE,
    PARSE_ZERO_ROUTE_ID,
    PARSE_CONTRADICTORY
};

static int tuple_matches(const GafimeNumericRoute* route) {
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

static enum ParseStatus parse_routes(
    const unsigned char* records,
    uint32_t count,
    uint32_t stride,
    uint32_t* known_mask_out,
    uint32_t* unknown_count_out
) {
    const uint32_t stable_prefix = (uint32_t)offsetof(GafimeNumericRoute, reserved);
    uint32_t known_mask = 0;
    uint32_t unknown_count = 0;
    uint32_t seen_route_ids[32];
    uint32_t seen_route_count = 0;
    if (stride < stable_prefix) {
        return PARSE_SHORT_PREFIX;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const GafimeNumericRoute* route =
            (const GafimeNumericRoute*)(records + (size_t)index * stride);
        if (GAFIME_ABI_VERSION_MAJOR_OF(route->abi_version) != 1u) {
            return PARSE_BAD_MAJOR;
        }
        if (GAFIME_ABI_VERSION_MINOR_OF(route->abi_version) <
            GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR) {
            return PARSE_BAD_MINOR;
        }
        if (route->struct_size < stable_prefix) {
            return PARSE_SHORT_PREFIX;
        }
        if ((route->flags & GAFIME_ABI_REQUIRED_FLAG_MASK) != 0) {
            return PARSE_BAD_FLAGS;
        }
        if (route->route_id == 0u) {
            return PARSE_ZERO_ROUTE_ID;
        }
        for (uint32_t seen = 0; seen < seen_route_count; ++seen) {
            if (seen_route_ids[seen] == route->route_id) {
                return PARSE_DUPLICATE;
            }
        }
        if (seen_route_count >= 32) {
            return PARSE_CONTRADICTORY;
        }
        seen_route_ids[seen_route_count++] = route->route_id;
        if (stride >= sizeof(GafimeNumericRoute) &&
            route->struct_size >= sizeof(GafimeNumericRoute)) {
            for (uint32_t slot = 0; slot < 8; ++slot) {
                if (route->reserved[slot] != 0) {
                    return PARSE_NONZERO_RESERVED;
                }
            }
        }
        if (route->route_id < GAFIME_NUMERIC_ROUTE_FP32 ||
            route->route_id > GAFIME_NUMERIC_ROUTE_FP64) {
            ++unknown_count;
            continue;
        }
        if (route->overflow_policy != GAFIME_OVERFLOW_IEEE || !tuple_matches(route)) {
            return PARSE_CONTRADICTORY;
        }
        const uint32_t bit = 1u << route->route_id;
        if ((known_mask & bit) != 0) {
            return PARSE_DUPLICATE;
        }
        known_mask |= bit;
    }
    *known_mask_out = known_mask;
    *unknown_count_out = unknown_count;
    return PARSE_OK;
}

static int expect_parse(
    enum ParseStatus expected,
    const unsigned char* records,
    uint32_t count,
    uint32_t stride,
    const char* label
) {
    uint32_t mask = 0;
    uint32_t unknown = 0;
    const enum ParseStatus actual = parse_routes(records, count, stride, &mask, &unknown);
    if (actual != expected) {
        fprintf(stderr, "%s: expected parser status %d, got %d\n", label, expected, actual);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SYNTHETIC_ABI_1_2_PAYLOAD\n", argv[0]);
        return 2;
    }
    GafimeTestLibrary library = gafime_test_library_open(argv[1]);
    if (library == NULL) {
        fprintf(stderr, "could not load %s: %s\n", argv[1], gafime_test_library_error());
        return 1;
    }
    NumericRoutesFn enumerate = NULL;
    GAFIME_TEST_LOAD_FUNCTION(library, enumerate, "gafime_gpu_numeric_routes_v2");

    uint32_t count = 0;
    if (enumerate(0, GAFIME_PRECISION_ABI_VERSION, 128, NULL, 0, &count) !=
            GAFIME_STATUS_OK || count != 4) {
        fprintf(stderr, "synthetic ABI 1.2 count query failed\n");
        gafime_test_library_close(library);
        return 1;
    }
    if (enumerate(0, (2u << 16) | 0u, 128, NULL, 0, &count) !=
        GAFIME_STATUS_ABI_MISMATCH) {
        fprintf(stderr, "synthetic payload accepted incompatible consumer major\n");
        gafime_test_library_close(library);
        return 1;
    }
    _Alignas(GafimeNumericRoute) unsigned char records[4][128];
    memset(records, 0, sizeof(records));
    if (enumerate(0, GAFIME_PRECISION_ABI_VERSION, 128,
                  (GafimeNumericRoute*)records, 4, &count) != GAFIME_STATUS_OK) {
        fprintf(stderr, "synthetic ABI 1.2 route enumeration failed\n");
        gafime_test_library_close(library);
        return 1;
    }
    if (enumerate(0, GAFIME_PRECISION_ABI_VERSION,
                  (uint32_t)offsetof(GafimeNumericRoute, reserved) - 1,
                  (GafimeNumericRoute*)records, 4, &count) !=
        GAFIME_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "synthetic payload accepted a short caller record\n");
        gafime_test_library_close(library);
        return 1;
    }

    uint32_t known_mask = 0;
    uint32_t unknown_count = 0;
    int failed = 0;
    enum ParseStatus status = parse_routes(&records[0][0], 4, 128,
                                           &known_mask, &unknown_count);
    const uint32_t expected_mask =
        (1u << GAFIME_NUMERIC_ROUTE_FP32) |
        (1u << GAFIME_NUMERIC_ROUTE_MIXED) |
        (1u << GAFIME_NUMERIC_ROUTE_FP64);
    if (status != PARSE_OK || known_mask != expected_mask || unknown_count != 1) {
        fprintf(stderr, "ABI 1.1 parser lost known routes: status=%d mask=%#x unknown=%u\n",
                status, known_mask, unknown_count);
        failed = 1;
    }
    _Alignas(GafimeNumericRoute)
        unsigned char consumer_prefixes[4][sizeof(GafimeNumericRoute)];
    for (uint32_t index = 0; index < 4; ++index) {
        memcpy(consumer_prefixes[index], records[index], sizeof(GafimeNumericRoute));
    }
    known_mask = 0;
    unknown_count = 0;
    status = parse_routes(
        &consumer_prefixes[0][0], 4, (uint32_t)sizeof(GafimeNumericRoute),
        &known_mask, &unknown_count);
    if (status != PARSE_OK || known_mask != expected_mask || unknown_count != 1) {
        fprintf(stderr,
                "ABI 1.1 parser rejected larger producer records copied into known prefixes: "
                "status=%d mask=%#x unknown=%u\n",
                status, known_mask, unknown_count);
        failed = 1;
    }
    for (uint32_t index = 0; index < 4; ++index) {
        const GafimeNumericRoute* route = (const GafimeNumericRoute*)records[index];
        if (route->struct_size <= sizeof(GafimeNumericRoute)) {
            fprintf(stderr, "synthetic route %u was not a larger ABI 1.2 record\n", index);
            failed = 1;
        }
    }

    _Alignas(GafimeNumericRoute) unsigned char malformed[4][128];
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->abi_version = (2u << 16) | 0u;
    failed |= expect_parse(PARSE_BAD_MAJOR, &malformed[0][0], 4, 128, "record major");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->abi_version = (1u << 16) | 0u;
    failed |= expect_parse(PARSE_BAD_MINOR, &malformed[0][0], 4, 128, "record minor");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->struct_size =
        (uint32_t)offsetof(GafimeNumericRoute, reserved) - 1;
    failed |= expect_parse(PARSE_SHORT_PREFIX, &malformed[0][0], 4, 128, "short prefix");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->flags |= 0x1u;
    failed |= expect_parse(PARSE_BAD_FLAGS, &malformed[0][0], 4, 128, "required flag");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->route_id = 0u;
    failed |= expect_parse(PARSE_ZERO_ROUTE_ID, &malformed[0][0], 4, 128,
                           "zero route ID");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->reserved[0] = 1;
    failed |= expect_parse(PARSE_NONZERO_RESERVED, &malformed[0][0], 4, 128,
                           "reserved field");
    memcpy(malformed, records, sizeof(malformed));
    memcpy(malformed[1], malformed[0], sizeof(malformed[1]));
    failed |= expect_parse(PARSE_DUPLICATE, &malformed[0][0], 4, 128, "duplicate route");
    memcpy(malformed, records, sizeof(malformed));
    memcpy(malformed[0], malformed[1], sizeof(malformed[0]));
    failed |= expect_parse(PARSE_DUPLICATE, &malformed[0][0], 4, 128,
                           "duplicate unknown route");
    memcpy(malformed, records, sizeof(malformed));
    ((GafimeNumericRoute*)malformed[0])->reduction_dtype = GAFIME_DTYPE_F64;
    failed |= expect_parse(PARSE_CONTRADICTORY, &malformed[0][0], 4, 128,
                           "contradictory route");

    gafime_test_library_close(library);
    return failed;
}
