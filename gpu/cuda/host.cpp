#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../common/gpu_abi_impl.h"

namespace {

constexpr int kThreadsPerBlock = 256;

struct GraphChunkShape {
    uint32_t arity;
    uint64_t descriptor_offset;
    uint64_t combo_count;
};

struct CudaMatrix {
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
    uint32_t* selected_indices;
    uint64_t selected_index_capacity;
    float* selected_metric_values;
    uint64_t selected_metric_value_capacity;
    cudaStream_t graph_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_valid;
    uint32_t graph_chunk_count;
    uint64_t graph_combo_len;
    uint64_t graph_metric_id_len;
    uint64_t graph_metric_value_count;
    uintptr_t graph_combo_ptr;
    uintptr_t graph_metric_ids_ptr;
    uintptr_t graph_metric_values_ptr;
    std::vector<GraphChunkShape> graph_chunk_shapes;
};

int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
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

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON ||
        metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, const CudaMatrix* matrix) {
    if (protocol == nullptr || matrix == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION || protocol->backend_kind != GAFIME_BACKEND_CUDA) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != matrix->rows || protocol->n_features != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (!metric_supported(protocol->metric_ids.ptr[idx])) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    if (protocol->rank.top_k != 0) {
        if (protocol->rank.include_ties != 0) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        bool primary_metric_found = false;
        for (uint64_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
            if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
                primary_metric_found = true;
                break;
            }
        }
        if (!primary_metric_found) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
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

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t planned_rows) {
    if (protocol->rank.top_k == 0) {
        return planned_rows;
    }
    return std::min<uint64_t>(planned_rows, protocol->rank.top_k);
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
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

uint32_t primary_metric_index(const GafimeLaunchProtocol* protocol) {
    if (protocol->rank.top_k == 0) {
        return 0;
    }
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(protocol->metric_ids.len); ++idx) {
        if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
            return idx;
        }
    }
    return 0;
}

void destroy_graph_cache(CudaMatrix* matrix) {
    if (matrix == nullptr) {
        return;
    }
    if (matrix->graph_exec != nullptr) {
        cudaGraphExecDestroy(matrix->graph_exec);
        matrix->graph_exec = nullptr;
    }
    if (matrix->graph != nullptr) {
        cudaGraphDestroy(matrix->graph);
        matrix->graph = nullptr;
    }
    matrix->graph_valid = false;
    matrix->graph_chunk_shapes.clear();
}

template <typename T>
int ensure_device_capacity(T** ptr, uint64_t* capacity, uint64_t required) {
    if (required <= *capacity) {
        return GAFIME_STATUS_OK;
    }
    T* next = nullptr;
    const uint64_t next_capacity = std::max(required, (*capacity == 0 ? required : *capacity * 2));
    const size_t bytes = static_cast<size_t>(next_capacity) * sizeof(T);
    int status = cuda_status(cudaMalloc(&next, bytes));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaFree(*ptr);
    *ptr = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

void compute_column_means_host(
    const float* features_host,
    uint64_t rows,
    uint32_t cols,
    std::vector<float>& means
) {
    means.assign(cols, 0.0f);
    std::vector<double> sums(cols, 0.0);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            sums[col] += static_cast<double>(features_host[row * cols + col]);
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
    float local_sxx = 0.0f;
    float local_syy = 0.0f;
    float local_sxy = 0.0f;

    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        float x = interaction_value(features, column_means, row, n_features, combo, arity);
        float y = target[row];
        if (isfinite(x) && isfinite(y)) {
            local_sx += x;
            local_sy += y;
            local_sxx += x * x;
            local_syy += y * y;
            local_sxy += x * y;
        }
    }

    __shared__ float sx[kThreadsPerBlock];
    __shared__ float sy[kThreadsPerBlock];
    __shared__ float sxx[kThreadsPerBlock];
    __shared__ float syy[kThreadsPerBlock];
    __shared__ float sxy[kThreadsPerBlock];

    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float n = static_cast<float>(n_samples);
        const float numerator = n * sxy[0] - sx[0] * sy[0];
        const float denom_x = n * sxx[0] - sx[0] * sx[0];
        const float denom_y = n * syy[0] - sy[0] * sy[0];
        float pearson = 0.0f;
        const float denom = sqrtf(fmaxf(denom_x * denom_y, 0.0f));
        if (denom > 0.0f) {
            pearson = numerator / denom;
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = pearson * pearson;
            }
            metric_values[combo_row * metric_count + metric_idx] = out;
        }
    }
}

template <int kBins>
__global__ void score_mutual_info_chunk_kernel(
    const float* features,
    const float* target,
    const float* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t n_features,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    float* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    __shared__ float min_x;
    __shared__ float max_x;
    __shared__ float min_y;
    __shared__ float max_y;
    __shared__ unsigned int hist_x[kBins];
    __shared__ unsigned int hist_y[kBins];
    __shared__ unsigned int joint[kBins * kBins];
    __shared__ unsigned int valid_count;

    if (threadIdx.x == 0) {
        min_x = INFINITY;
        max_x = -INFINITY;
        min_y = INFINITY;
        max_y = -INFINITY;
        valid_count = 0;
        for (uint64_t row = 0; row < n_samples; ++row) {
            const float x = interaction_value(features, column_means, row, n_features, combo, arity);
            const float y = target[row];
            if (isfinite(x) && isfinite(y)) {
                min_x = fminf(min_x, x);
                max_x = fmaxf(max_x, x);
                min_y = fminf(min_y, y);
                max_y = fmaxf(max_y, y);
                ++valid_count;
            }
        }
    }
    for (uint32_t idx = threadIdx.x; idx < kBins; idx += blockDim.x) {
        hist_x[idx] = 0;
        hist_y[idx] = 0;
    }
    for (uint32_t idx = threadIdx.x; idx < kBins * kBins; idx += blockDim.x) {
        joint[idx] = 0;
    }
    __syncthreads();

    if (valid_count <= 1 || max_x <= min_x || max_y <= min_y) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = 0.0f;
        }
        return;
    }

    const float inv_x = static_cast<float>(kBins) / (max_x - min_x);
    const float inv_y = static_cast<float>(kBins) / (max_y - min_y);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float x = interaction_value(features, column_means, row, n_features, combo, arity);
        const float y = target[row];
        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }
        uint32_t xb = static_cast<uint32_t>((x - min_x) * inv_x);
        uint32_t yb = static_cast<uint32_t>((y - min_y) * inv_y);
        xb = min(xb, static_cast<uint32_t>(kBins - 1));
        yb = min(yb, static_cast<uint32_t>(kBins - 1));
        atomicAdd(&hist_x[xb], 1);
        atomicAdd(&hist_y[yb], 1);
        atomicAdd(&joint[xb * kBins + yb], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const float total = static_cast<float>(valid_count);
        float mi = 0.0f;
        uint32_t active_x = 0;
        uint32_t active_y = 0;
        for (uint32_t xb = 0; xb < kBins; ++xb) {
            if (hist_x[xb] == 0) {
                continue;
            }
            ++active_x;
            const float px = static_cast<float>(hist_x[xb]) / total;
            for (uint32_t yb = 0; yb < kBins; ++yb) {
                const unsigned int count = joint[xb * kBins + yb];
                if (count == 0 || hist_y[yb] == 0) {
                    continue;
                }
                const float py = static_cast<float>(hist_y[yb]) / total;
                const float pxy = static_cast<float>(count) / total;
                mi += pxy * logf(pxy / (px * py));
            }
        }
        for (uint32_t yb = 0; yb < kBins; ++yb) {
            if (hist_y[yb] != 0) {
                ++active_y;
            }
        }
        const float correction = active_x > 0 && active_y > 0
            ? static_cast<float>((active_x - 1) * (active_y - 1)) / (2.0f * total)
            : 0.0f;
        const float corrected = fmaxf(0.0f, mi - correction);
        const uint32_t normalizer_bins = min(active_x, active_y);
        const float normalizer = normalizer_bins > 1
            ? logf(static_cast<float>(normalizer_bins))
            : 0.0f;
        metric_values[combo_row * metric_count + metric_index] =
            normalizer > 0.0f ? corrected / normalizer : 0.0f;
    }
}

uint32_t mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = 96;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t hint = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (hint == 12 || hint == 24 || hint == 48 || hint == 96) {
            bins = hint;
        }
    }
    return bins;
}

int launch_mi_kernel_for_bins(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    const GafimeArityChunk& chunk,
    uint64_t metric_row_offset,
    uint32_t metric_index,
    uint32_t bins,
    cudaStream_t stream
) {
    dim3 grid(static_cast<unsigned int>(chunk.combo_count));
    dim3 block(kThreadsPerBlock);
    float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
    switch (bins) {
        case 12:
            score_mutual_info_chunk_kernel<12><<<grid, block, 0, stream>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len), metric_index, out
            );
            break;
        case 24:
            score_mutual_info_chunk_kernel<24><<<grid, block, 0, stream>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len), metric_index, out
            );
            break;
        case 48:
            score_mutual_info_chunk_kernel<48><<<grid, block, 0, stream>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len), metric_index, out
            );
            break;
        default:
            score_mutual_info_chunk_kernel<96><<<grid, block, 0, stream>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(protocol->metric_ids.len), metric_index, out
            );
            break;
    }
    return cuda_status(cudaGetLastError());
}

int launch_score_kernels(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    cudaStream_t stream
) {
    uint64_t metric_row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (chunk.combo_count == 0) {
            continue;
        }
        dim3 grid(static_cast<unsigned int>(chunk.combo_count));
        dim3 block(kThreadsPerBlock);
        score_continuous_chunk_kernel<<<grid, block, 0, stream>>>(
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
        const int status = cuda_status(cudaGetLastError());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        for (uint32_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
            if (protocol->metric_ids.ptr[metric_idx] != GAFIME_METRIC_MUTUAL_INFO) {
                continue;
            }
            const int mi_status = launch_mi_kernel_for_bins(
                matrix,
                protocol,
                chunk,
                metric_row_offset,
                metric_idx,
                mi_bins_for_chunk(protocol, chunk),
                stream
            );
            if (mi_status != GAFIME_STATUS_OK) {
                return mi_status;
            }
        }
        metric_row_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

bool graph_shape_matches(
    const CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) {
    if (!matrix->graph_valid || matrix->graph_exec == nullptr) {
        return false;
    }
    if (matrix->graph_chunk_count != protocol->chunk_count ||
        matrix->graph_combo_len != protocol->combo_indices.len ||
        matrix->graph_metric_id_len != protocol->metric_ids.len ||
        matrix->graph_metric_value_count != metric_value_count ||
        matrix->graph_combo_ptr != reinterpret_cast<uintptr_t>(matrix->combo_indices) ||
        matrix->graph_metric_ids_ptr != reinterpret_cast<uintptr_t>(matrix->metric_ids) ||
        matrix->graph_metric_values_ptr != reinterpret_cast<uintptr_t>(matrix->metric_values)) {
        return false;
    }
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        const GraphChunkShape& shape = matrix->graph_chunk_shapes[idx];
        if (shape.arity != chunk.arity ||
            shape.descriptor_offset != chunk.descriptor_offset ||
            shape.combo_count != chunk.combo_count) {
            return false;
        }
    }
    return true;
}

void store_graph_shape(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count
) {
    matrix->graph_chunk_count = protocol->chunk_count;
    matrix->graph_combo_len = protocol->combo_indices.len;
    matrix->graph_metric_id_len = protocol->metric_ids.len;
    matrix->graph_metric_value_count = metric_value_count;
    matrix->graph_combo_ptr = reinterpret_cast<uintptr_t>(matrix->combo_indices);
    matrix->graph_metric_ids_ptr = reinterpret_cast<uintptr_t>(matrix->metric_ids);
    matrix->graph_metric_values_ptr = reinterpret_cast<uintptr_t>(matrix->metric_values);
    matrix->graph_chunk_shapes.clear();
    matrix->graph_chunk_shapes.reserve(protocol->chunk_count);
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        matrix->graph_chunk_shapes.push_back(GraphChunkShape{
            chunk.arity,
            chunk.descriptor_offset,
            chunk.combo_count,
        });
    }
}

bool graph_requested(const GafimeLaunchProtocol* protocol) {
    return (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0 &&
        protocol->rank.top_k == 0 &&
        protocol->permutations.permutation_count == 0;
}

int execute_score_kernels(
    CudaMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool* graph_replayed
) {
    *graph_replayed = false;
    if (!graph_requested(protocol)) {
        int status = launch_score_kernels(matrix, protocol, 0);
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaDeviceSynchronize());
        }
        return status;
    }

    if (matrix->graph_stream == nullptr) {
        int status = cuda_status(cudaStreamCreateWithFlags(&matrix->graph_stream, cudaStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    if (!graph_shape_matches(matrix, protocol, metric_value_count)) {
        destroy_graph_cache(matrix);
        int status = cuda_status(cudaStreamBeginCapture(matrix->graph_stream, cudaStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = launch_score_kernels(matrix, protocol, matrix->graph_stream);
        if (status != GAFIME_STATUS_OK) {
            cudaStreamEndCapture(matrix->graph_stream, &matrix->graph);
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraph_t next_graph = nullptr;
        status = cuda_status(cudaStreamEndCapture(matrix->graph_stream, &next_graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_graph_cache(matrix);
            return status;
        }
        cudaGraphExec_t next_exec = nullptr;
        status = cuda_status(cudaGraphInstantiate(&next_exec, next_graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            cudaGraphDestroy(next_graph);
            destroy_graph_cache(matrix);
            return status;
        }
        matrix->graph = next_graph;
        matrix->graph_exec = next_exec;
        matrix->graph_valid = true;
        store_graph_shape(matrix, protocol, metric_value_count);
    }

    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    }
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    }
    return status;
}

__device__ bool candidate_better(
    float candidate_score,
    uint32_t candidate_index,
    float best_score,
    uint32_t best_index,
    bool descending
) {
    if (!isfinite(candidate_score)) {
        return false;
    }
    if (best_index == UINT32_MAX) {
        return true;
    }
    if (descending) {
        if (candidate_score > best_score) {
            return true;
        }
        if (candidate_score < best_score) {
            return false;
        }
    } else {
        if (candidate_score < best_score) {
            return true;
        }
        if (candidate_score > best_score) {
            return false;
        }
    }
    return candidate_index < best_index;
}

__global__ void select_topk_kernel(
    const float* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    uint32_t descending,
    uint32_t* selected_indices
) {
    __shared__ float best_scores[kThreadsPerBlock];
    __shared__ uint32_t best_indices[kThreadsPerBlock];

    const bool sort_descending = descending != 0;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        float local_score = sort_descending ? -INFINITY : INFINITY;
        uint32_t local_index = UINT32_MAX;
        for (uint64_t row = threadIdx.x; row < row_count; row += blockDim.x) {
            bool already_selected = false;
            for (uint32_t prev = 0; prev < rank; ++prev) {
                if (selected_indices[prev] == static_cast<uint32_t>(row)) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }
            const float score = metric_values[row * metric_count + primary_metric_index];
            if (candidate_better(score, static_cast<uint32_t>(row), local_score, local_index, sort_descending)) {
                local_score = score;
                local_index = static_cast<uint32_t>(row);
            }
        }

        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();

        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float score = best_scores[threadIdx.x + stride];
                const uint32_t index = best_indices[threadIdx.x + stride];
                if (candidate_better(score, index, best_scores[threadIdx.x], best_indices[threadIdx.x], sort_descending)) {
                    best_scores[threadIdx.x] = score;
                    best_indices[threadIdx.x] = index;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            selected_indices[rank] = best_indices[0];
        }
        __syncthreads();
    }
}

__global__ void copy_selected_metric_rows_kernel(
    const float* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    float* selected_metric_values
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (idx >= total) {
        return;
    }
    const uint64_t selected_row = idx / metric_count;
    const uint32_t metric_idx = static_cast<uint32_t>(idx % metric_count);
    const uint32_t source_row = selected_indices[selected_row];
    selected_metric_values[idx] = metric_values[static_cast<uint64_t>(source_row) * metric_count + metric_idx];
}

bool locate_combo_for_global_row(
    const GafimeLaunchProtocol* protocol,
    uint64_t global_row,
    const GafimeArityChunk** chunk_out,
    uint64_t* local_row_out
) {
    uint64_t row_offset = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        if (global_row < row_offset + chunk.combo_count) {
            *chunk_out = &chunk;
            *local_row_out = global_row - row_offset;
            return true;
        }
        row_offset += chunk.combo_count;
    }
    return false;
}

int write_result_rows_host(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    const std::vector<uint32_t>* selected_indices
) {
    const uint32_t max_arity = result->max_arity;
    const uint32_t metric_count = result->metric_count;
    const uint64_t output_rows = selected_indices == nullptr
        ? planned_row_count(protocol)
        : static_cast<uint64_t>(selected_indices->size());

    for (uint64_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint64_t global_row = selected_indices == nullptr
            ? output_row
            : static_cast<uint64_t>((*selected_indices)[output_row]);
        const GafimeArityChunk* chunk = nullptr;
        uint64_t local_row = 0;
        if (!locate_combo_for_global_row(protocol, global_row, &chunk, &local_row)) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const uint64_t combo_base = chunk->descriptor_offset + local_row * chunk->arity;
        for (uint32_t slot = 0; slot < max_arity; ++slot) {
            result->combo_indices[output_row * max_arity + slot] =
                slot < chunk->arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const float value = metric_idx < protocol->metric_ids.len
                ? metric_values[output_row * protocol->metric_ids.len + metric_idx]
                : 0.0f;
            result->metric_values[output_row * metric_count + metric_idx] = value;
        }
        result->ranks[output_row] = static_cast<uint32_t>(output_row);
        result->families[output_row] = GAFIME_FAMILY_CONTINUOUS;
        result->candidate_ids[output_row] = global_row;
        result->row_flags[output_row] = 0;
    }
    result->row_count = output_rows;
    return GAFIME_STATUS_OK;
}

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(
    uint32_t device_id,
    GafimeGpuDeviceInfo* info_out
) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t status = cudaSetDevice(static_cast<int>(device_id));
    if (status != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    cudaDeviceProp props{};
    status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = GAFIME_BACKEND_CUDA;
    info_out->device_id = device_id;
    std::strncpy(info_out->name, props.name, sizeof(info_out->name) - 1);
    info_out->name[sizeof(info_out->name) - 1] = '\0';
    info_out->total_global_mem_bytes = static_cast<uint64_t>(props.totalGlobalMem);
    info_out->multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info_out->warp_size = static_cast<uint32_t>(props.warpSize);
    info_out->compute_major = static_cast<uint32_t>(props.major);
    info_out->compute_minor = static_cast<uint32_t>(props.minor);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
) {
    (void)device_id;
    const int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_CUDA,
        GAFIME_GRAPH_STREAM_CAPTURE,
        capability_out
    );
    if (status == GAFIME_STATUS_OK) {
        capability_out->supports_kernel_param_update = 0;
        capability_out->supports_device_ranking = 1;
        capability_out->max_captured_nodes = 64;
        capability_out->stable_pointer_flags = 1;
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;

    int status = validate_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    auto* matrix = new CudaMatrix{};
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
    matrix->selected_indices = nullptr;
    matrix->selected_index_capacity = 0;
    matrix->selected_metric_values = nullptr;
    matrix->selected_metric_value_capacity = 0;
    matrix->graph_stream = nullptr;
    matrix->graph = nullptr;
    matrix->graph_exec = nullptr;
    matrix->graph_valid = false;
    matrix->graph_chunk_count = 0;
    matrix->graph_combo_len = 0;
    matrix->graph_metric_id_len = 0;
    matrix->graph_metric_value_count = 0;
    matrix->graph_combo_ptr = 0;
    matrix->graph_metric_ids_ptr = 0;
    matrix->graph_metric_values_ptr = 0;

    const size_t feature_bytes = static_cast<size_t>(matrix->rows) * matrix->cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(matrix->cols) * sizeof(float);
    status = cuda_status(cudaMalloc(&matrix->features, feature_bytes));
    if (status != GAFIME_STATUS_OK) {
        delete matrix;
        return status;
    }
    status = cuda_status(cudaMalloc(&matrix->target, target_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->features);
        delete matrix;
        return status;
    }
    status = cuda_status(cudaMalloc(&matrix->column_means, mean_bytes));
    if (status != GAFIME_STATUS_OK) {
        cudaFree(matrix->target);
        cudaFree(matrix->features);
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
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<float> column_means;
    compute_column_means_host(features_host, rows, cols, column_means);

    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(float);
    status = cuda_status(cudaMemcpy(matrix->features, features_host, feature_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return cuda_status(cudaMemcpy(matrix->column_means, column_means.data(), mean_bytes, cudaMemcpyHostToDevice));
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(float);
    return cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    if (matrix == nullptr) {
        return;
    }
    cudaSetDevice(static_cast<int>(matrix->device_id));
    destroy_graph_cache(matrix);
    if (matrix->graph_stream != nullptr) {
        cudaStreamDestroy(matrix->graph_stream);
    }
    cudaFree(matrix->column_means);
    cudaFree(matrix->target);
    cudaFree(matrix->features);
    cudaFree(matrix->metric_values);
    cudaFree(matrix->metric_ids);
    cudaFree(matrix->combo_indices);
    cudaFree(matrix->selected_metric_values);
    cudaFree(matrix->selected_indices);
    delete matrix;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<CudaMatrix*>(matrix_handle);
    int status = validate_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = validate_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t total_rows = planned_row_count(protocol);
    const uint64_t output_rows = output_row_count(protocol, total_rows);
    if (total_rows == 0) {
        result_out->row_count = 0;
        return GAFIME_STATUS_OK;
    }

    const uint64_t metric_value_count = total_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(&matrix->combo_indices, &matrix->combo_capacity, protocol->combo_indices.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_ids, &matrix->metric_id_capacity, protocol->metric_ids.len);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(&matrix->metric_values, &matrix->metric_value_capacity, metric_value_count);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const size_t combo_bytes = static_cast<size_t>(protocol->combo_indices.len) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(protocol->metric_ids.len) * sizeof(uint32_t);
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);
    status = cuda_status(cudaMemcpy(matrix->combo_indices, protocol->combo_indices.ptr, combo_bytes, cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(matrix->metric_ids, protocol->metric_ids.ptr, metric_id_bytes, cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    bool graph_replayed = false;
    status = execute_score_kernels(matrix, protocol, metric_value_count, &graph_replayed);
    result_out->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    if (graph_replayed) {
        result_out->flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    }

    if (protocol->rank.top_k == 0) {
        std::vector<float> metric_values(static_cast<size_t>(total_rows) * protocol->metric_ids.len, 0.0f);
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(metric_values.data(), matrix->metric_values, metric_value_bytes, cudaMemcpyDeviceToHost));
        }
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        return write_result_rows_host(protocol, result_out, metric_values, nullptr);
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    status = ensure_device_capacity(&matrix->selected_indices, &matrix->selected_index_capacity, output_rows);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const uint64_t selected_metric_value_count = output_rows * protocol->metric_ids.len;
    status = ensure_device_capacity(
        &matrix->selected_metric_values,
        &matrix->selected_metric_value_capacity,
        selected_metric_value_count
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    select_topk_kernel<<<1, kThreadsPerBlock>>>(
        matrix->metric_values,
        total_rows,
        static_cast<uint32_t>(protocol->metric_ids.len),
        primary_metric_index(protocol),
        static_cast<uint32_t>(output_rows),
        protocol->rank.descending,
        matrix->selected_indices
    );
    status = cuda_status(cudaGetLastError());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t copy_items = selected_metric_value_count;
    if (copy_items > 0) {
        const uint32_t copy_threads = 256;
        const uint32_t copy_blocks = static_cast<uint32_t>((copy_items + copy_threads - 1) / copy_threads);
        copy_selected_metric_rows_kernel<<<copy_blocks, copy_threads>>>(
            matrix->metric_values,
            matrix->selected_indices,
            output_rows,
            static_cast<uint32_t>(protocol->metric_ids.len),
            matrix->selected_metric_values
        );
        status = cuda_status(cudaGetLastError());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    std::vector<uint32_t> selected_indices(static_cast<size_t>(output_rows), UINT32_MAX);
    std::vector<float> selected_metric_values(
        static_cast<size_t>(selected_metric_value_count),
        0.0f
    );
    status = cuda_status(cudaMemcpy(
        selected_indices.data(),
        matrix->selected_indices,
        static_cast<size_t>(output_rows) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    ));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            selected_metric_values.data(),
            matrix->selected_metric_values,
            static_cast<size_t>(selected_metric_value_count) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    return write_result_rows_host(protocol, result_out, selected_metric_values, &selected_indices);
}

}  // extern "C"
