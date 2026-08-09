#ifndef GAFIME_GPU_ABI_IMPL_HPP
#define GAFIME_GPU_ABI_IMPL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

#include "gafime_gpu_abi.hpp"

namespace gafime_gpu_abi {

constexpr uint32_t kNumericRouteStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericRoute, reserved));
constexpr uint32_t kConstBufferStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeConstBufferView, reserved));
constexpr uint32_t kMutableBufferStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeMutableBufferView, reserved));
constexpr uint32_t kNumericMatrixDescStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericMatrixDesc, reserved));
constexpr uint32_t kNumericLaunchProtocolStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericLaunchProtocol, reserved));
constexpr uint32_t kNumericResultTableStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericResultTable, reserved));
constexpr uint32_t kNumericSignificanceTableStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericSignificanceTable, reserved));
constexpr uint32_t kNumericDiagnosticStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeNumericInteractionDiagnosticBatch, reserved));

inline bool abi_1_1_compatible(uint32_t abi_version, uint32_t struct_size, uint32_t prefix_size) {
    return GAFIME_ABI_VERSION_MAJOR_OF(abi_version) == GAFIME_PRECISION_ABI_VERSION_MAJOR &&
        GAFIME_ABI_VERSION_MINOR_OF(abi_version) >= GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR &&
        struct_size >= prefix_size;
}

template <typename T, size_t N>
inline bool all_zero(const T (&values)[N]) {
    for (const T value : values) {
        if (value != 0) return false;
    }
    return true;
}

inline bool flags_supported(uint32_t flags, uint32_t known_required_flags) {
    const uint32_t unknown_required =
        (flags & GAFIME_ABI_REQUIRED_FLAG_MASK) & ~known_required_flags;
    return unknown_required == 0;
}

inline size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case GAFIME_DTYPE_F32: return sizeof(float);
    case GAFIME_DTYPE_F64: return sizeof(double);
    default: return 0;
    }
}

inline bool fits_host_bytes(uint64_t element_count, size_t element_size) {
    return element_size != 0 &&
        element_count <= static_cast<uint64_t>(std::numeric_limits<size_t>::max() / element_size);
}

template <typename T>
inline bool naturally_aligned(const T* pointer) {
    return pointer == nullptr ||
        reinterpret_cast<uintptr_t>(pointer) % alignof(T) == 0;
}

inline GafimeNumericRoute numeric_route(uint32_t profile) {
    GafimeNumericRoute route{};
    route.abi_version = GAFIME_PRECISION_ABI_VERSION;
    route.struct_size = sizeof(GafimeNumericRoute);
    route.profile = profile;
    route.overflow_policy = GAFIME_OVERFLOW_IEEE;
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        route.route_id = GAFIME_NUMERIC_ROUTE_FP32;
        route.storage_dtype = GAFIME_DTYPE_F32;
        route.pointwise_dtype = GAFIME_DTYPE_F32;
        route.reduction_dtype = GAFIME_DTYPE_F32;
        route.result_dtype = GAFIME_DTYPE_F32;
        break;
    case GAFIME_PRECISION_MIXED:
        route.route_id = GAFIME_NUMERIC_ROUTE_MIXED;
        route.storage_dtype = GAFIME_DTYPE_F32;
        route.pointwise_dtype = GAFIME_DTYPE_F32;
        route.reduction_dtype = GAFIME_DTYPE_F64;
        route.result_dtype = GAFIME_DTYPE_F64;
        break;
    case GAFIME_PRECISION_FP64:
        route.route_id = GAFIME_NUMERIC_ROUTE_FP64;
        route.storage_dtype = GAFIME_DTYPE_F64;
        route.pointwise_dtype = GAFIME_DTYPE_F64;
        route.reduction_dtype = GAFIME_DTYPE_F64;
        route.result_dtype = GAFIME_DTYPE_F64;
        break;
    default:
        route = {};
        break;
    }
    return route;
}

inline bool route_fields_equal(const GafimeNumericRoute& lhs, const GafimeNumericRoute& rhs) {
    return lhs.route_id == rhs.route_id && lhs.profile == rhs.profile &&
        lhs.storage_dtype == rhs.storage_dtype && lhs.pointwise_dtype == rhs.pointwise_dtype &&
        lhs.reduction_dtype == rhs.reduction_dtype && lhs.result_dtype == rhs.result_dtype &&
        lhs.overflow_policy == rhs.overflow_policy;
}

inline int validate_numeric_route(const GafimeNumericRoute* route) {
    if (route == nullptr || !naturally_aligned(route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(route->abi_version, route->struct_size, kNumericRouteStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!flags_supported(route->flags, 0)) return GAFIME_STATUS_INVALID_ARGUMENT;
    const GafimeNumericRoute expected = numeric_route(route->profile);
    if (expected.route_id == 0 || !route_fields_equal(*route, expected)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (route->struct_size >= sizeof(GafimeNumericRoute) && !all_zero(route->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_embedded_numeric_route(const GafimeNumericRoute* route) {
    const int status = validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    // ABI 1.1 outer structures embed exactly the fixed 1.1 selection prefix.
    // A larger standalone enumeration record must be truncated before it is
    // copied here so its tail can never overlap the following outer fields.
    return route->struct_size <= sizeof(GafimeNumericRoute)
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
}

inline int enumerate_numeric_routes(
    uint32_t consumer_abi_version,
    uint32_t route_stride,
    GafimeNumericRoute* routes_out,
    uint32_t route_capacity,
    uint32_t* route_count_out,
    const uint32_t* profiles,
    uint32_t profile_count
) {
    if (route_count_out == nullptr || !naturally_aligned(route_count_out) ||
        profiles == nullptr || !naturally_aligned(profiles) || profile_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *route_count_out = profile_count;
    if (GAFIME_ABI_VERSION_MAJOR_OF(consumer_abi_version) !=
            GAFIME_PRECISION_ABI_VERSION_MAJOR ||
        GAFIME_ABI_VERSION_MINOR_OF(consumer_abi_version) <
            GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (routes_out == nullptr) {
        return route_capacity == 0 ? GAFIME_STATUS_OK : GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!naturally_aligned(routes_out) ||
        route_stride % alignof(GafimeNumericRoute) != 0 ||
        route_capacity < profile_count || route_stride < kNumericRouteStablePrefixSize) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (profile_count != 0 &&
        static_cast<size_t>(route_stride) >
            std::numeric_limits<size_t>::max() / profile_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0; index < profile_count; ++index) {
        GafimeNumericRoute route = numeric_route(profiles[index]);
        if (route.route_id == 0) return GAFIME_STATUS_INVALID_ARGUMENT;
        const size_t write_size = std::min<size_t>(route_stride, sizeof(route));
        auto* destination = reinterpret_cast<unsigned char*>(routes_out) +
            static_cast<size_t>(index) * route_stride;
        std::memset(destination, 0, route_stride);
        route.struct_size = static_cast<uint32_t>(write_size);
        std::memcpy(destination, &route, write_size);
    }
    return GAFIME_STATUS_OK;
}

inline int validate_buffer_common(
    uint32_t abi_version,
    uint32_t struct_size,
    uint32_t dtype,
    uint32_t flags,
    const void* data,
    uint64_t element_count,
    uint64_t byte_length,
    uint64_t byte_stride,
    uint32_t expected_dtype,
    uint64_t expected_elements,
    uint32_t prefix_size
) {
    if (!abi_1_1_compatible(abi_version, struct_size, prefix_size)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    constexpr uint32_t required_flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    if (!flags_supported(flags, required_flags) ||
        (flags & required_flags) != required_flags || dtype != expected_dtype) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t element_size = dtype_size(dtype);
    if (element_size == 0 || byte_stride != element_size || element_count != expected_elements) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (element_count != 0 && element_count > std::numeric_limits<uint64_t>::max() / byte_stride) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t expected_bytes = element_count * byte_stride;
    if (byte_length != expected_bytes ||
        !fits_host_bytes(element_count, element_size) ||
        (element_count != 0 && data == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (data != nullptr && reinterpret_cast<uintptr_t>(data) % element_size != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_const_buffer(
    const GafimeConstBufferView* view,
    uint32_t expected_dtype,
    uint64_t expected_elements
) {
    if (view == nullptr || !naturally_aligned(view)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const int status = validate_buffer_common(
        view->abi_version, view->struct_size, view->dtype, view->flags, view->data,
        view->element_count, view->byte_length, view->byte_stride, expected_dtype,
        expected_elements, kConstBufferStablePrefixSize);
    if (status != GAFIME_STATUS_OK) return status;
    if (view->struct_size >= sizeof(GafimeConstBufferView) && !all_zero(view->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_mutable_buffer(
    const GafimeMutableBufferView* view,
    uint32_t expected_dtype,
    uint64_t expected_elements
) {
    if (view == nullptr || !naturally_aligned(view)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const int status = validate_buffer_common(
        view->abi_version, view->struct_size, view->dtype, view->flags, view->data,
        view->element_capacity, view->byte_length, view->byte_stride, expected_dtype,
        expected_elements, kMutableBufferStablePrefixSize);
    if (status != GAFIME_STATUS_OK) return status;
    if (view->struct_size >= sizeof(GafimeMutableBufferView) && !all_zero(view->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_numeric_matrix_desc(const GafimeNumericMatrixDesc* desc) {
    if (desc == nullptr || !naturally_aligned(desc)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(
            desc->abi_version, desc->struct_size, kNumericMatrixDescStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = validate_embedded_numeric_route(&desc->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (!flags_supported(desc->flags, 0) || desc->layout != GAFIME_MATRIX_ROW_MAJOR ||
        desc->rows == 0 || desc->cols == 0 || desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t elements = 0;
    const size_t element_size = dtype_size(desc->route.storage_dtype);
    if (element_size == 0 ||
        (desc->cols != 0 && desc->rows > std::numeric_limits<uint64_t>::max() / desc->cols)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    elements = desc->rows * desc->cols;
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        !fits_host_bytes(elements, element_size) ||
        desc->bytes != elements * element_size) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->struct_size >= sizeof(GafimeNumericMatrixDesc) && !all_zero(desc->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_numeric_launch_protocol(
    const GafimeNumericLaunchProtocol* protocol,
    uint32_t expected_profile
) {
    if (protocol == nullptr || !naturally_aligned(protocol)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(
            protocol->abi_version, protocol->struct_size,
            kNumericLaunchProtocolStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = validate_embedded_numeric_route(&protocol->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (protocol->route.profile != expected_profile || protocol->base == nullptr ||
        !naturally_aligned(protocol->base)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->struct_size >= sizeof(GafimeNumericLaunchProtocol) &&
        !all_zero(protocol->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_numeric_result_table(
    GafimeNumericResultTable* result,
    uint32_t expected_dtype
) {
    if (result == nullptr || !naturally_aligned(result)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(
            result->abi_version, result->struct_size, kNumericResultTableStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!flags_supported(result->flags, GAFIME_RESULT_FLAG_GRAPH_REPLAYED) ||
        result->reserved32 != 0 || result->max_arity == 0 || result->max_arity > 5 ||
        result->metric_count == 0 || result->row_count > result->capacity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t metric_elements = 0;
    if (result->metric_count != 0 &&
        result->capacity > std::numeric_limits<uint64_t>::max() / result->metric_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    metric_elements = result->capacity * result->metric_count;
    const int buffer_status =
        validate_mutable_buffer(&result->metric_values, expected_dtype, metric_elements);
    if (buffer_status != GAFIME_STATUS_OK) return buffer_status;
    if (result->capacity != 0 &&
        (result->combo_indices == nullptr || result->ranks == nullptr ||
         result->families == nullptr || result->candidate_ids == nullptr ||
         result->row_flags == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->capacity >
        std::numeric_limits<uint64_t>::max() / result->max_arity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t combo_elements = result->capacity * result->max_arity;
    if (!naturally_aligned(result->combo_indices) || !naturally_aligned(result->ranks) ||
        !naturally_aligned(result->families) || !naturally_aligned(result->candidate_ids) ||
        !naturally_aligned(result->row_flags) ||
        !fits_host_bytes(combo_elements, sizeof(uint32_t)) ||
        !fits_host_bytes(result->capacity, sizeof(uint32_t)) ||
        !fits_host_bytes(result->capacity, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->struct_size >= sizeof(GafimeNumericResultTable) && !all_zero(result->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_numeric_significance_table(
    GafimeNumericSignificanceTable* significance,
    uint32_t expected_dtype
) {
    if (significance == nullptr || !naturally_aligned(significance)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(
            significance->abi_version, significance->struct_size,
            kNumericSignificanceTableStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!flags_supported(significance->flags, 0) || significance->metric_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->metric_count != 0 &&
        significance->row_count >
            std::numeric_limits<uint64_t>::max() / significance->metric_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t values = significance->row_count * significance->metric_count;
    int status = validate_const_buffer(
        &significance->observed_metric_values, expected_dtype, values);
    if (status == GAFIME_STATUS_OK) {
        status = validate_mutable_buffer(&significance->p_values, expected_dtype, values);
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (significance->row_count != 0 && significance->candidate_ids == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!naturally_aligned(significance->candidate_ids)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!fits_host_bytes(significance->row_count, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->struct_size >= sizeof(GafimeNumericSignificanceTable) &&
        !all_zero(significance->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_numeric_diagnostics(
    GafimeNumericInteractionDiagnosticBatch* diagnostics,
    uint32_t expected_profile
) {
    if (diagnostics == nullptr || !naturally_aligned(diagnostics)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_1_1_compatible(
            diagnostics->abi_version, diagnostics->struct_size,
            kNumericDiagnosticStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = validate_embedded_numeric_route(&diagnostics->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (diagnostics->route.profile != expected_profile ||
        !flags_supported(diagnostics->flags, 0) || diagnostics->reserved32 != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->max_arity == 0 || diagnostics->max_arity > 5 ||
        (diagnostics->row_count != 0 &&
         (diagnostics->combo_indices == nullptr ||
          diagnostics->overflow_row_counts == nullptr || diagnostics->row_flags == nullptr))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->row_count != 0 &&
        diagnostics->row_count >
            std::numeric_limits<uint64_t>::max() / diagnostics->max_arity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->combo_index_count != diagnostics->row_count * diagnostics->max_arity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!fits_host_bytes(diagnostics->combo_index_count, sizeof(uint32_t)) ||
        !fits_host_bytes(diagnostics->row_count, sizeof(uint64_t)) ||
        !fits_host_bytes(diagnostics->row_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!naturally_aligned(diagnostics->combo_indices) ||
        !naturally_aligned(diagnostics->overflow_row_counts) ||
        !naturally_aligned(diagnostics->row_flags)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->struct_size >= sizeof(GafimeNumericInteractionDiagnosticBatch) &&
        !all_zero(diagnostics->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int invalid_if_null(const void* ptr) {
    return ptr == nullptr ? GAFIME_STATUS_INVALID_ARGUMENT : GAFIME_STATUS_OK;
}

inline int fill_device_info(
    uint32_t device_id,
    uint32_t backend_kind,
    const char* name,
    GafimeGpuDeviceInfo* info_out
) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = backend_kind;
    info_out->device_id = device_id;
    info_out->warp_size = 32;
    std::snprintf(info_out->name, sizeof(info_out->name), "%s", name);
    return GAFIME_STATUS_OK;
}

inline int fill_graph_capability(
    uint32_t backend_kind,
    uint32_t graph_mode,
    GafimeGpuGraphCapability* capability_out
) {
    if (capability_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::memset(capability_out, 0, sizeof(*capability_out));
    capability_out->abi_version = GAFIME_ABI_VERSION;
    capability_out->backend_kind = backend_kind;
    capability_out->graph_mode = graph_mode;
    return GAFIME_STATUS_OK;
}

inline int unsupported_until_p3_device_loop() {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

inline uint64_t saturating_add_u64(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs + rhs;
}

inline uint64_t saturating_mul_u64(uint64_t lhs, uint64_t rhs) {
    return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs * rhs;
}

inline uint64_t allocation_bytes(uint64_t capacity, size_t element_size) {
    return saturating_mul_u64(capacity, static_cast<uint64_t>(element_size));
}

inline uint64_t next_allocation_capacity(
    uint64_t capacity,
    uint64_t required,
    size_t element_size
) {
    if (required <= capacity) {
        return capacity;
    }
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / element_size;
    if (required > max_capacity) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t grown_capacity = capacity > max_capacity / 2
        ? max_capacity
        : capacity * 2;
    return std::max(required, capacity == 0 ? required : grown_capacity);
}

class DeviceMemoryPeakTracker {
public:
    explicit DeviceMemoryPeakTracker(uint64_t resident_bytes)
        : resident_bytes_(resident_bytes), peak_bytes_(resident_bytes) {}

    void grow(uint64_t capacity, uint64_t required, size_t element_size) {
        if (required <= capacity) {
            return;
        }
        const uint64_t next_capacity =
            next_allocation_capacity(capacity, required, element_size);
        const uint64_t old_bytes = allocation_bytes(capacity, element_size);
        const uint64_t next_bytes = allocation_bytes(next_capacity, element_size);
        observe_transient(next_bytes);
        replace_resident(old_bytes, next_bytes);
    }

    void reserve_pair(
        uint64_t first_capacity,
        uint64_t first_required,
        size_t first_element_size,
        uint64_t second_capacity,
        uint64_t second_required,
        size_t second_element_size
    ) {
        const uint64_t first_next = next_allocation_capacity(
            first_capacity, first_required, first_element_size);
        const uint64_t second_next = next_allocation_capacity(
            second_capacity, second_required, second_element_size);
        uint64_t transient_bytes = 0;
        if (first_required > first_capacity) {
            transient_bytes = saturating_add_u64(
                transient_bytes,
                allocation_bytes(first_next, first_element_size));
        }
        if (second_required > second_capacity) {
            transient_bytes = saturating_add_u64(
                transient_bytes,
                allocation_bytes(second_next, second_element_size));
        }
        observe_transient(transient_bytes);
        if (first_required > first_capacity) {
            replace_resident(
                allocation_bytes(first_capacity, first_element_size),
                allocation_bytes(first_next, first_element_size));
        }
        if (second_required > second_capacity) {
            replace_resident(
                allocation_bytes(second_capacity, second_element_size),
                allocation_bytes(second_next, second_element_size));
        }
    }

    void observe_transient(uint64_t transient_bytes) {
        peak_bytes_ = std::max(
            peak_bytes_, saturating_add_u64(resident_bytes_, transient_bytes));
    }

    uint64_t resident_bytes() const {
        return resident_bytes_;
    }

    uint64_t peak_bytes() const {
        return peak_bytes_;
    }

private:
    void replace_resident(uint64_t old_bytes, uint64_t next_bytes) {
        if (resident_bytes_ == std::numeric_limits<uint64_t>::max() ||
            old_bytes > resident_bytes_) {
            resident_bytes_ = std::numeric_limits<uint64_t>::max();
        } else {
            resident_bytes_ = saturating_add_u64(
                resident_bytes_ - old_bytes, next_bytes);
        }
        peak_bytes_ = std::max(peak_bytes_, resident_bytes_);
    }

    uint64_t resident_bytes_;
    uint64_t peak_bytes_;
};

}  // namespace gafime_gpu_abi

#endif  // GAFIME_GPU_ABI_IMPL_HPP
