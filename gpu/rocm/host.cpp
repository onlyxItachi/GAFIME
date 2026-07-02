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
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO;
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
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    if (protocol->rank.top_k != 0 && protocol->rank.include_ties != 0) {
        // Host-side top-k is supported; tie-inclusive ranking is not implemented.
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
    const uint64_t planned = planned_row_count(protocol);
    // With top-k ranking only the selected rows are written, so the result table
    // need only hold min(planned, top_k).
    const uint64_t rows = protocol->rank.top_k == 0
        ? planned
        : std::min<uint64_t>(planned, protocol->rank.top_k);
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

// Mutual-information kernel: one block per candidate, adaptive-range binning into
// shared histograms, finite-sample-corrected + normalized MI. Ported from the
// CUDA host (HIP is source-compatible: atomicAdd / __syncthreads / logf / min).
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

int launch_hip_mi_for_bins(
    HipMatrix* matrix,
    const GafimeLaunchProtocol* protocol,
    const GafimeArityChunk& chunk,
    uint64_t metric_row_offset,
    uint32_t metric_index,
    uint32_t bins
) {
    dim3 grid(static_cast<unsigned int>(chunk.combo_count));
    dim3 block(kThreadsPerBlock);
    float* out = matrix->metric_values + metric_row_offset * protocol->metric_ids.len;
    const uint32_t mc = static_cast<uint32_t>(protocol->metric_ids.len);
    switch (bins) {
        case 12:
            score_mutual_info_chunk_kernel<12><<<grid, block>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                mc, metric_index, out);
            break;
        case 24:
            score_mutual_info_chunk_kernel<24><<<grid, block>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                mc, metric_index, out);
            break;
        case 48:
            score_mutual_info_chunk_kernel<48><<<grid, block>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                mc, metric_index, out);
            break;
        default:
            score_mutual_info_chunk_kernel<96><<<grid, block>>>(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, matrix->cols, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                mc, metric_index, out);
            break;
    }
    return hip_status(hipGetLastError());
}

void write_result_row_at(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    uint64_t dst_row,
    uint64_t src_global_row,
    uint64_t combo_base,
    uint32_t arity,
    uint32_t rank,
    uint64_t candidate_id
) {
    const uint32_t max_arity = result->max_arity;
    const uint32_t metric_count = result->metric_count;
    for (uint32_t slot = 0; slot < max_arity; ++slot) {
        result->combo_indices[dst_row * max_arity + slot] =
            slot < arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
    }
    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const float value = metric_idx < protocol->metric_ids.len
            ? metric_values[src_global_row * protocol->metric_ids.len + metric_idx]
            : 0.0f;
        result->metric_values[dst_row * metric_count + metric_idx] = value;
    }
    result->ranks[dst_row] = rank;
    result->families[dst_row] = GAFIME_FAMILY_CONTINUOUS;
    result->candidate_ids[dst_row] = candidate_id;
    result->row_flags[dst_row] = 0;
}

int write_result_rows_host(const GafimeLaunchProtocol* protocol, GafimeResultTable* result, const std::vector<float>& metric_values) {
    const uint32_t top_k = protocol->rank.top_k;

    if (top_k == 0) {
        uint64_t output_row = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
            const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
            for (uint64_t row = 0; row < chunk.combo_count; ++row) {
                const uint64_t combo_base = chunk.descriptor_offset + row * chunk.arity;
                write_result_row_at(protocol, result, metric_values, output_row, output_row,
                    combo_base, chunk.arity, static_cast<uint32_t>(output_row), output_row);
                ++output_row;
            }
        }
        result->row_count = output_row;
        return GAFIME_STATUS_OK;
    }

    // Host-side top-k selection by the primary metric. The ROCm host already
    // copies every candidate's metrics back to the host, so no device selection
    // kernel is needed. Mirrors the CPU/CUDA ranking: order by the raw primary
    // metric value (descending or ascending), tie-break by candidate id ascending,
    // skip non-finite scores.
    uint32_t primary_index = 0;
    bool found = false;
    for (uint32_t i = 0; i < protocol->metric_ids.len; ++i) {
        if (protocol->metric_ids.ptr[i] == protocol->rank.primary_metric) {
            primary_index = i;
            found = true;
            break;
        }
    }
    if (!found) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const bool descending = protocol->rank.descending != 0;

    struct Candidate {
        uint64_t id;
        float score;
        uint64_t combo_base;
        uint32_t arity;
    };
    std::vector<Candidate> candidates;
    uint64_t global_row = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < protocol->chunk_count; ++chunk_idx) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_idx];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const float score =
                metric_values[global_row * protocol->metric_ids.len + primary_index];
            if (std::isfinite(score)) {
                candidates.push_back(
                    {global_row, score, chunk.descriptor_offset + row * chunk.arity, chunk.arity});
            }
            ++global_row;
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [descending](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) {
                return descending ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        });
    const uint64_t selected = std::min<uint64_t>(top_k, candidates.size());
    for (uint64_t r = 0; r < selected; ++r) {
        const Candidate& c = candidates[r];
        write_result_row_at(protocol, result, metric_values, r, c.id, c.combo_base, c.arity,
            static_cast<uint32_t>(r), c.id);
    }
    result->row_count = selected;
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
        // Mutual-information metrics get a dedicated histogram kernel per chunk
        // (mirrors the CUDA host), writing into the same metric column.
        for (uint32_t metric_idx = 0; metric_idx < protocol->metric_ids.len; ++metric_idx) {
            if (protocol->metric_ids.ptr[metric_idx] != GAFIME_METRIC_MUTUAL_INFO) {
                continue;
            }
            status = launch_hip_mi_for_bins(
                matrix, protocol, chunk, metric_row_offset, metric_idx,
                mi_bins_for_chunk(protocol, chunk));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
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
