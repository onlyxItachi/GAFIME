#ifndef GAFIME_PRECISION_ABI_SMOKE_ADAPTER_HPP
#define GAFIME_PRECISION_ABI_SMOKE_ADAPTER_HPP

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "gafime_gpu_abi.hpp"
#include "gafime_gpu_internal_abi.hpp"

/*
 * Test-only source adapters keep the pre-existing physical precision suites
 * focused on arithmetic and lifecycle coverage while driving only the final
 * generic ABI 1.1 symbols. They are ordinary inline C++ helpers and therefore
 * cannot become payload exports or independent implementation owners.
 */
namespace gafime_precision_smoke {

inline GafimeNumericRoute route_for_profile(uint32_t profile) {
    GafimeNumericRoute route{};
    route.abi_version = GAFIME_PRECISION_ABI_VERSION;
    route.struct_size = sizeof(route);
    route.profile = profile;
    route.overflow_policy = GAFIME_OVERFLOW_IEEE;
    switch (profile) {
        case GAFIME_PRECISION_FP32: {
            route.route_id = GAFIME_NUMERIC_ROUTE_FP32;
            route.storage_dtype = GAFIME_DTYPE_F32;
            route.pointwise_dtype = GAFIME_DTYPE_F32;
            route.reduction_dtype = GAFIME_DTYPE_F32;
            route.result_dtype = GAFIME_DTYPE_F32;
            return route;
        }
        case GAFIME_PRECISION_MIXED: {
            route.route_id = GAFIME_NUMERIC_ROUTE_MIXED;
            route.storage_dtype = GAFIME_DTYPE_F32;
            route.pointwise_dtype = GAFIME_DTYPE_F32;
            route.reduction_dtype = GAFIME_DTYPE_F64;
            route.result_dtype = GAFIME_DTYPE_F64;
            return route;
        }
        case GAFIME_PRECISION_FP64: {
            route.route_id = GAFIME_NUMERIC_ROUTE_FP64;
            route.storage_dtype = GAFIME_DTYPE_F64;
            route.pointwise_dtype = GAFIME_DTYPE_F64;
            route.reduction_dtype = GAFIME_DTYPE_F64;
            route.result_dtype = GAFIME_DTYPE_F64;
            return route;
        }
        default:
            return {};
    }
}

inline std::unordered_map<GafimeGpuMatrix, GafimeNumericRoute>& matrix_routes() {
    static std::unordered_map<GafimeGpuMatrix, GafimeNumericRoute> routes;
    return routes;
}

inline GafimeConstBufferView const_view(
    uint32_t dtype, const void* data, uint64_t element_count
) {
    const uint64_t element_bytes = dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double);
    GafimeConstBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = element_count;
    view.byte_length = element_count * element_bytes;
    view.byte_stride = element_bytes;
    return view;
}

inline GafimeMutableBufferView mutable_view(
    uint32_t dtype, void* data, uint64_t element_count
) {
    const uint64_t element_bytes = dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double);
    GafimeMutableBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = element_count;
    view.byte_length = element_count * element_bytes;
    view.byte_stride = element_bytes;
    return view;
}

inline int matrix_alloc(
    uint32_t device_id, const GafimePrecisionMatrixDesc* desc, GafimeGpuMatrix* matrix_out
) {
    if (desc == nullptr || matrix_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    GafimeNumericMatrixDesc numeric{};
    numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric.struct_size = sizeof(numeric);
    numeric.route = route_for_profile(desc->profile);
    numeric.layout = desc->layout;
    numeric.flags = desc->flags;
    numeric.rows = desc->rows;
    numeric.cols = desc->cols;
    numeric.row_stride = desc->row_stride;
    numeric.bytes = desc->bytes;
    const int status = gafime_gpu_matrix_alloc_v2(device_id, &numeric, matrix_out);
    if (status == GAFIME_STATUS_OK) matrix_routes()[*matrix_out] = numeric.route;
    return status;
}

inline int matrix_upload(
    GafimeGpuMatrix matrix,
    uint32_t dtype,
    const void* features,
    const void* target,
    uint64_t rows,
    uint32_t cols
) {
    const auto found = matrix_routes().find(matrix);
    if (found == matrix_routes().end()) return GAFIME_STATUS_INVALID_ARGUMENT;
    GafimeNumericRoute route = found->second;
    if (route.storage_dtype != dtype) {
        route = route_for_profile(
            dtype == GAFIME_DTYPE_F32 ? GAFIME_PRECISION_FP32 : GAFIME_PRECISION_FP64);
    }
    const auto feature_view = const_view(dtype, features, rows * cols);
    const auto target_view = const_view(dtype, target, rows);
    return gafime_gpu_matrix_upload_v2(
        matrix, &route, &feature_view, &target_view, rows, cols);
}

inline int matrix_update_target(
    GafimeGpuMatrix matrix, uint32_t dtype, const void* target, uint64_t rows
) {
    const auto found = matrix_routes().find(matrix);
    if (found == matrix_routes().end()) return GAFIME_STATUS_INVALID_ARGUMENT;
    GafimeNumericRoute route = found->second;
    if (route.storage_dtype != dtype) {
        route = route_for_profile(
            dtype == GAFIME_DTYPE_F32 ? GAFIME_PRECISION_FP32 : GAFIME_PRECISION_FP64);
    }
    const auto target_view = const_view(dtype, target, rows);
    return gafime_gpu_matrix_update_target_v2(matrix, &route, &target_view, rows);
}

inline int matrix_upload_f32(
    GafimeGpuMatrix matrix,
    const float* features,
    const float* target,
    uint64_t rows,
    uint32_t cols
) {
    return matrix_upload(matrix, GAFIME_DTYPE_F32, features, target, rows, cols);
}

inline int matrix_upload_f64(
    GafimeGpuMatrix matrix,
    const double* features,
    const double* target,
    uint64_t rows,
    uint32_t cols
) {
    return matrix_upload(matrix, GAFIME_DTYPE_F64, features, target, rows, cols);
}

inline int matrix_update_target_f32(
    GafimeGpuMatrix matrix, const float* target, uint64_t rows
) {
    return matrix_update_target(matrix, GAFIME_DTYPE_F32, target, rows);
}

inline int matrix_update_target_f64(
    GafimeGpuMatrix matrix, const double* target, uint64_t rows
) {
    return matrix_update_target(matrix, GAFIME_DTYPE_F64, target, rows);
}

inline GafimeNumericLaunchProtocol launch(const GafimePrecisionLaunchProtocol* protocol) {
    GafimeNumericLaunchProtocol numeric{};
    numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric.struct_size = sizeof(numeric);
    if (protocol != nullptr) {
        numeric.route = route_for_profile(protocol->profile);
        numeric.base = protocol->base;
    }
    return numeric;
}

template <typename ResultTable>
inline int execute(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    ResultTable* result,
    uint32_t dtype
) {
    if (result == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    auto numeric_protocol = launch(protocol);
    GafimeNumericResultTable numeric{};
    numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric.struct_size = sizeof(numeric);
    numeric.max_arity = result->max_arity;
    numeric.metric_count = result->metric_count;
    numeric.flags = result->flags;
    numeric.capacity = result->capacity;
    numeric.row_count = result->row_count;
    numeric.combo_indices = result->combo_indices;
    numeric.metric_values = mutable_view(
        dtype,
        static_cast<void*>(result->metric_values),
        result->capacity * result->metric_count);
    numeric.ranks = result->ranks;
    numeric.families = result->families;
    numeric.candidate_ids = result->candidate_ids;
    numeric.row_flags = result->row_flags;
    const int status = gafime_gpu_execute_v2(matrix, &numeric_protocol, &numeric);
    result->flags = numeric.flags;
    result->row_count = numeric.row_count;
    return status;
}

inline int execute_f32(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result
) {
    return execute(matrix, protocol, result, GAFIME_DTYPE_F32);
}

inline int execute_f64(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result
) {
    return execute(matrix, protocol, result, GAFIME_DTYPE_F64);
}

inline int execution_memory_peak(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    auto numeric = launch(protocol);
    return gafime_gpu_execution_memory_peak_v2(matrix, &numeric, peak_bytes_out);
}

inline int permutation_memory_peak(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
) {
    auto numeric = launch(protocol);
    return gafime_gpu_permutation_memory_peak_v2(
        matrix, &numeric, selected_row_count, peak_bytes_out);
}

template <typename SignificanceTable>
inline int permutation_pvalues(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    SignificanceTable* significance,
    uint32_t dtype
) {
    if (significance == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    auto numeric_protocol = launch(protocol);
    const uint64_t value_count = significance->row_count * significance->metric_count;
    GafimeNumericSignificanceTable numeric{};
    numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric.struct_size = sizeof(numeric);
    numeric.metric_count = significance->metric_count;
    numeric.row_count = significance->row_count;
    numeric.candidate_ids = significance->candidate_ids;
    numeric.observed_metric_values = const_view(
        dtype, significance->observed_metric_values, value_count);
    numeric.p_values = mutable_view(dtype, significance->p_values, value_count);
    return gafime_gpu_permutation_pvalues_v2(matrix, &numeric_protocol, &numeric);
}

inline int permutation_pvalues_f32(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance
) {
    return permutation_pvalues(matrix, protocol, significance, GAFIME_DTYPE_F32);
}

inline int permutation_pvalues_f64(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTableF64* significance
) {
    return permutation_pvalues(matrix, protocol, significance, GAFIME_DTYPE_F64);
}

inline int interaction_diagnostics(
    GafimeGpuMatrix matrix, GafimeInteractionDiagnosticBatch* diagnostics
) {
    if (diagnostics == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    const auto found = matrix_routes().find(matrix);
    if (found == matrix_routes().end()) return GAFIME_STATUS_INVALID_ARGUMENT;
    GafimeNumericInteractionDiagnosticBatch numeric{};
    numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
    numeric.struct_size = sizeof(numeric);
    numeric.route = found->second;
    numeric.max_arity = diagnostics->max_arity;
    numeric.row_count = diagnostics->row_count;
    numeric.combo_indices = diagnostics->combo_indices;
    numeric.combo_index_count = diagnostics->combo_index_count;
    numeric.overflow_row_counts = diagnostics->overflow_row_counts;
    numeric.row_flags = diagnostics->flags;
    return gafime_gpu_interaction_diagnostics_v2(matrix, &numeric);
}

inline int matrix_free(GafimeGpuMatrix matrix) {
    matrix_routes().erase(matrix);
    return gafime_gpu_matrix_free_v2(matrix);
}

}  // namespace gafime_precision_smoke

#endif
