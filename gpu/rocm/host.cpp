#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../common/gpu_abi_impl.h"

namespace {

constexpr int kThreadsPerBlock = 256;

struct HipMatrix {
    uint32_t device_id;
    uint64_t rows;
    uint32_t cols;
    float* features;
    float* target;
    float* column_means;
    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    float* metric_values;
    uint64_t metric_value_capacity;
};

int hip_status(hipError_t status) {
    return status == hipSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2;
}

int validate_matrix_desc(const GafimeMatrixDesc* desc) {
    if (desc == nullptr || desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (desc->rows == 0 || desc->cols == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, const HipMatrix* matrix) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION || protocol->backend_kind != GAFIME_BACKEND_ROCM) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->rank.top_k != 0 || protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (!metric_supported(protocol->metric_ids.ptr[idx])) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.family != GAFIME_FAMILY_CONTINUOUS || chunk.arity == 0 || chunk.arity > protocol->max_arity) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        const uint64_t required = chunk.descriptor_offset + chunk.combo_count * chunk.arity;
        if (required > protocol->combo_indices.len) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return GAFIME_STATUS_OK;
}

uint64_t planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t rows = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        rows += protocol->chunks[chunk_idx].combo_count;
    }
    return rows;
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = planned_row_count(protocol);
    if (result->capacity < rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows > 0 && (
        result->combo_indices == nullptr ||
        result->metric_values == nullptr ||
        result->ranks == nullptr ||
        result->families == nullptr ||
        result->candidate_ids == nullptr ||
        result->row_flags == nullptr
    )) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

template <typename T>
int ensure_device_capacity(T** ptr, uint64_t* capacity, uint64_t required) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    T* next = nullptr;
    const uint64_t next_capacity = std::max(required, (*capacity == 0 ? required : *capacity * 2));
    const size_t bytes = static_cast<size_t>(next_capacity) * sizeof(T);
    int status = hip_status(hipMalloc(&next, bytes));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    hipFree(*ptr);
    *ptr = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

void compute_column_means_host(const float* features, uint64_t rows, uint32_t cols, std::vector<float>& means) {
    means.assign(cols, 0.0f);
    std::vector<double> sums(cols, 0.0);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += static_cast<double>(features[row * cols + col]);
        }
    }
    const double inv_rows = 1.0 / static_cast<double>(rows);
    for (uint32_t col = 0; col < cols; ++col) {
        means[col] = static_cast<float>(sums[col] * inv_rows);
    }
}

__device__ float interaction_value(
    const float* features,
    const float* column_means,
    uint64_t row,
    uint32_t cols,
    const uint32_t* combo,
    uint32_t arity
) {
    if (arity == 1) {
        return features[row * cols + combo[0]];
    }
    float value = 1.0f;
    for (uint32_t idx = 0; idx < arity; ++idx) {
        const uint32_t col = combo[idx];
        value *= features[row * cols + col] - column_means[col];
    }
    return value;
}

__global__ void score_continuous_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    float local_sx = 0.0f;
    float local_sy = 0.0f;
    float local_n = 0.0f;

    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value(features, column_means, row, n_features, combo, arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += x;
            local_sy += y;
            local_n += 1.0f;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sn[kThreadsPerBlock];
    __shared__ float sxx[kThreadsPerBlock];
    __shared__ float syy[kThreadsPerBlock];
    __shared__ float sxy[kThreadsPerBlock];
    __shared__ float mean_x;
    __shared__ float mean_y;

    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    sn[threadIdx.x] = local_n;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            sn[threadIdx.x] += sn[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (sn[0] > 0.0f) {
            mean_x = sx[0] / sn[0];
            mean_y = sy[0] / sn[0];
        } else {
            mean_x = 0.0f;
            mean_y = 0.0f;
        }
    }
    __syncthreads();

    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value(features, column_means, row, n_features, combo, arity);
        const float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            const float dx = x - mean_x;
            const float dy = y - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }

    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float pearson = 0.0f;
        const float denom = sqrtf(fmaxf(sxx[0] * syy[0], 0.0f));
        if (denom > 0.0f) {
            pearson = fminf(fmaxf(sxy[0] / denom, -1.0f), 1.0f);
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

int write_result_rows_host(const GafimeLaunchProtocol* protocol, GafimeResultTable* result, const std::vector<float>& metric_values) {
    const uint32_t max_arity = result->max_arity;
    const uint32_t metric_count = result->metric_count;
    uint64_t output_row = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const uint64_t combo_base = chunk.descriptor_offset + row * chunk.arity;
            for (uint32_t slot = 0; slot < max_arity; ++slot) {
                result->combo_indices[output_row * max_arity + slot] =
                    slot < chunk.arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
            }
            for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
                const float value = metric_idx < protocol->metric_ids.len
                    ? metric_values[output_row * protocol->metric_ids.len + metric_idx]
                    : 0.0f;
                result->metric_values[output_row * metric_count + metric_idx] = value;
            }
            result->ranks[output_row] = static_cast<uint32_t>(output_row);
            result->families[output_row] = GAFIME_FAMILY_CONTINUOUS;
            result->candidate_ids[output_row] = output_row;
            result->row_flags[output_row] = 0;
            ++output_row;
        }
    }
    result->row_count = output_row;
    return GAFIME_STATUS_OK;
}

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(uint32_t device_id, GafimeGpuDeviceInfo* info_out) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    hipError_t status = hipSetDevice(static_cast<int>(device_id));
    if (status != hipSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != hipSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = GAFIME_BACKEND_ROCM;
    info_out->device_id = device_id;
    std::strncpy(info_out->name, props.name, sizeof(info_out->name) - 1);
    info_out->total_global_mem_bytes = static_cast<uint64_t>(props.totalGlobalMem);
    info_out->multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info_out->warp_size = static_cast<uint32_t>(props.warpSize);
    info_out->compute_major = static_cast<uint32_t>(props.major);
    info_out->compute_minor = static_cast<uint32_t>(props.minor);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(uint32_t device_id, GafimeGpuGraphCapability* capability_out) {
    (void)device_id;
    return gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_ROCM,
        GAFIME_GRAPH_UNSUPPORTED,
        capability_out
    );
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(uint32_t device_id, const GafimeMatrixDesc* matrix_desc, GafimeGpuMatrix* matrix_out) {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    int status = validate_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(hipSetDevice(static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    auto* matrix = new HipMatrix{};
    matrix->device_id = device_id;
    matrix->rows = matrix_desc->rows;
    matrix->cols = matrix_desc->cols;
    matrix->features = nullptr;
    matrix->target = nullptr;
    matrix->column_means = nullptr;
    matrix->combo_indices = nullptr;
    matrix->combo_capacity = 0;
    matrix->metric_ids = nullptr;
    matrix->metric_id_capacity = 0;
    matrix->metric_values = nullptr;
    matrix->metric_value_capacity = 0;

    const size_t feature_bytes = static_cast<size_t>(matrix->rows) * matrix->cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(matrix->cols) * sizeof(float);
    status = hip_status(hipMalloc(&matrix->features, feature_bytes));
    if (status != GAFIME_STATUS_OK) {
        delete matrix;
        return status;
    }
    status = hip_status(hipMalloc(&matrix->target, target_bytes));
    if (status != GAFIME_STATUS_OK) {
        hipFree(matrix->features);
        delete matrix;
        return status;
    }
    status = hip_status(hipMalloc(&matrix->column_means, mean_bytes));
    if (status != GAFIME_STATUS_OK) {
        hipFree(matrix->target);
        hipFree(matrix->features);
        delete matrix;
        return status;
    }
    *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::vector<float> column_means;
    compute_column_means_host(features_host, rows, cols, column_means);
    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(float);
    status = hip_status(hipMemcpy(matrix->features, features_host, feature_bytes, hipMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipMemcpy(matrix->target, target_host, target_bytes, hipMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipMemcpy(matrix->column_means, column_means.data(), mean_bytes, hipMemcpyHostToDevice));
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(GafimeGpuMatrix matrix_handle, const float* target_host, uint64_t rows) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    return hip_status(hipMemcpy(matrix->target, target_host, target_bytes, hipMemcpyHostToDevice));
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    if (matrix == nullptr) {
        return;
    }
    hipSetDevice(static_cast<int>(matrix->device_id));
    hipFree(matrix->column_means);
    hipFree(matrix->target);
    hipFree(matrix->features);
    hipFree(matrix->metric_values);
    hipFree(matrix->metric_ids);
    hipFree(matrix->combo_indices);
    delete matrix;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<HipMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = hip_status(hipSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    if (total_rows == 0) {
        result_out->row_count = 0;
        return GAFIME_STATUS_OK;
    }
    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(&matrix->combo_indices, &matrix->combo_capacity, protocol->combo_indices.len);
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&matrix->metric_ids, &matrix->metric_id_capacity, protocol->metric_ids.len);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&matrix->metric_values, &matrix->metric_value_capacity, metric_value_count);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = hip_status(hipMemcpy(
        matrix->combo_indices,
        protocol->combo_indices.ptr,
        static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t),
        hipMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK) {
        status = hip_status(hipMemcpy(
            matrix->metric_ids,
            protocol->metric_ids.ptr,
            static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t),
            hipMemcpyHostToDevice
        ));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.combo_count == 0) {
            continue;
        }
        dim3 grid(static_cast<unsigned int>(chunk.combo_count));
        dim3 block(kThreadsPerBlock);
        score_continuous_chunk_kernel<<<grid, block>>>(
            matrix->features,
            matrix->target,
            matrix->column_means,
            matrix->combo_indices,
            matrix->rows,
            matrix->cols,
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
            matrix->metric_ids,
            static_cast<uint32_t>(protocol->metric_ids.len),
            matrix->metric_values + metric_row_offset * protocol->metric_ids.len
        );
        status = hip_status(hipGetLastError());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        metric_row_offset += chunk.combo_count;
    }
    status = hip_status(hipDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<float> metric_values(static_cast<size_t>(metric_value_count), 0.0f);
    status = hip_status(hipMemcpy(
        metric_values.data(),
        matrix->metric_values,
        static_cast<size_t>(metric_value_count) * sizeof(float),
        hipMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return write_result_rows_host(protocol, result_out, metric_values);
}

}  // extern "C"
