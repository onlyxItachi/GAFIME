#include "rt_kernels.cuh"

#ifdef GAFIME_CUDA_RT_OPTIX_DEVICE

#include <optix.h>
#include <optix_device.h>

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes;
    const float* target;
    const double* target_stats;
    float* membership;
    uint32_t* membership_words;
    uint32_t* direct_inside_counts;
    double* direct_inside_sum_y;
    uint32_t rows;
    uint32_t path_count;
    uint32_t geometry_mode;
    uint32_t words_per_path;
    const uint32_t* group_path_offsets;
    uint32_t group_count;
    uint32_t point_group_stride;
    uint32_t point_stride;
    uint32_t direct_first_hit;
};

extern "C" {
__constant__ GafimeRtParams params;
}

static __forceinline__ __device__ bool inside_dim(
    float value,
    float lo,
    float hi,
    bool lo_open
) {
    return lo_open ? (value > lo && value <= hi) : (value >= lo && value <= hi);
}

static __forceinline__ __device__ uint32_t current_group_index()
{
    return params.group_count > 1u ? optixGetInstanceId() : 0u;
}

static __forceinline__ __device__ uint64_t semantic_point_offset(uint32_t row, uint32_t group_idx)
{
    const uint64_t point_stride = static_cast<uint64_t>(params.point_stride);
    return params.group_count > 1u
        ? static_cast<uint64_t>(group_idx) * params.point_group_stride +
            static_cast<uint64_t>(row) * point_stride
        : static_cast<uint64_t>(row) * point_stride;
}

static __forceinline__ __device__ bool inside_box(float3 point, uint32_t path_idx)
{
    const gafime_cuda_v1::rt_kernel::GafimeRtBox box = params.boxes[path_idx];
    bool inside = inside_dim(point.x, box.lo_x, box.hi_x, (box.open_lo_mask & 1u) != 0u);
    if (box.dims > 1u) {
        inside = inside && inside_dim(point.y, box.lo_y, box.hi_y, (box.open_lo_mask & 2u) != 0u);
    }
    if (box.dims > 2u) {
        inside = inside && inside_dim(point.z, box.lo_z, box.hi_z, (box.open_lo_mask & 4u) != 0u);
    }
    return inside;
}

extern "C" __global__ void __raygen__gafime_dp()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint32_t row = launch_idx.x;
    const uint32_t group_idx = launch_idx.y;
    if (row >= params.rows || (params.group_count > 1u && group_idx >= params.group_count)) {
        return;
    }

    const uint64_t point_offset = semantic_point_offset(row, group_idx);
    const float x = params.points_xyz[point_offset + 0u];
    const float y = params.points_xyz[point_offset + 1u];
    const bool triangle_2d_instanced = params.geometry_mode == 2u;
    uint32_t payload_row = row;
    const float group_z = params.geometry_mode != 0u ? static_cast<float>(group_idx) * 4.0f : 0.0f;
    const float3 origin = triangle_2d_instanced
        ? make_float3(x, y, group_z - 1.0f)
        : make_float3(
            static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(x)),
            static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(y)),
            group_z - 2.0f
        );
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const float tmax = triangle_2d_instanced ? 2.0f : 4.0f;

    const unsigned int ray_flags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
        (params.direct_first_hit != 0u ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : 0u);

    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        ray_flags,
        0,
        1,
        0,
        payload_row
    );
}

extern "C" __global__ void __miss__gafime_dp() {}

extern "C" __global__ void __intersection__gafime_dp_box()
{
    const uint32_t row = optixGetPayload_0();
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    const uint32_t group_idx = current_group_index();
    const uint32_t path_base = params.group_path_offsets != nullptr ? params.group_path_offsets[group_idx] : 0u;
    const uint32_t path_idx = path_base + primitive_idx;
    if (path_idx >= params.path_count) {
        return;
    }
    const uint64_t point_offset = semantic_point_offset(row, group_idx);
    const float3 point = make_float3(
        params.points_xyz[point_offset + 0u],
        params.points_xyz[point_offset + 1u],
        params.points_xyz[point_offset + 2u]
    );
    if (inside_box(point, path_idx)) {
        optixReportIntersection(2.0f, 0, path_idx);
    }
}

extern "C" __global__ void __anyhit__gafime_dp_mark()
{
    const uint32_t row = optixGetPayload_0();
    const bool triangle_2d_instanced = params.geometry_mode == 2u;
    const uint32_t group_idx = current_group_index();
    const uint32_t path_base = params.group_path_offsets != nullptr ? params.group_path_offsets[group_idx] : 0u;
    const uint32_t path_idx = triangle_2d_instanced
        ? path_base + (optixGetPrimitiveIndex() >> 1u)
        : optixGetAttribute_0();
    if (path_idx < params.path_count) {
        if (triangle_2d_instanced && !inside_box(optixGetWorldRayOrigin(), path_idx)) {
            optixIgnoreIntersection();
            return;
        }
        if (params.direct_inside_counts != nullptr) {
            bool first_callback = true;
            if (params.membership_words != nullptr) {
                const uint64_t word_idx =
                    static_cast<uint64_t>(path_idx) * params.words_per_path + (row >> 5u);
                const uint32_t row_mask = 1u << (row & 31u);
                first_callback = (atomicOr(&params.membership_words[word_idx], row_mask) & row_mask) == 0u;
            }
            const float y = params.target[row];
            if (first_callback && isfinite(y)) {
                atomicAdd(&params.direct_inside_counts[path_idx], 1u);
                const double centered_y = static_cast<double>(y) - params.target_stats[1];
                atomicAdd(&params.direct_inside_sum_y[path_idx], centered_y);
            }
            if (params.direct_first_hit != 0u) {
                optixTerminateRay();
                return;
            }
        } else if (params.membership_words != nullptr) {
            const uint64_t word_idx =
                static_cast<uint64_t>(path_idx) * params.words_per_path + (row >> 5u);
            atomicOr(&params.membership_words[word_idx], 1u << (row & 31u));
        } else {
            const uint64_t out_idx = static_cast<uint64_t>(path_idx) * params.rows + row;
            params.membership[out_idx] = 1.0f;
        }
    }
    optixIgnoreIntersection();
}

#else

#include <cuda_runtime.h>

#include <cmath>

namespace gafime_cuda_v1::rt_kernel {

__global__ void pack_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t axis0,
    uint32_t axis1,
    uint32_t axis2,
    uint32_t dims,
    float* points_xyz
) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_samples) {
        return;
    }
    const uint32_t axes[3] = {axis0, axis1, axis2};
    for (uint32_t dim = 0; dim < 3u; ++dim) {
        const float value = dim < dims ? features[static_cast<uint64_t>(axes[dim]) * n_samples + row] : 0.0f;
        points_xyz[row * 3u + dim] = value;
    }
}

__global__ void pack_grouped_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    const uint32_t* group_axes,
    const uint32_t* group_dims,
    uint32_t group_count,
    uint32_t point_stride,
    float* points_xyz
) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t group_idx = blockIdx.y;
    if (row >= n_samples || group_idx >= group_count) {
        return;
    }
    const uint32_t dims = group_dims[group_idx];
    const uint32_t* axes = group_axes + static_cast<uint64_t>(group_idx) * 3u;
    const uint64_t point_base =
        static_cast<uint64_t>(group_idx) * n_samples * point_stride +
        row * point_stride;
    for (uint32_t dim = 0; dim < point_stride; ++dim) {
        const float value = dim < dims ? features[static_cast<uint64_t>(axes[dim]) * n_samples + row] : 0.0f;
        points_xyz[point_base + dim] = value;
    }
}

__global__ void decision_path_membership_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = row_offset + static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count || row >= n_samples) {
        return;
    }

    const uint32_t begin = path_offsets[path_idx];
    const uint32_t end = path_offsets[path_idx + 1];
    bool member = true;
    bool undetermined = false;

    for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
        const GafimeDecisionPathTerm term = terms[term_idx];
        if (term.feature >= n_features) {
            member = false;
            break;
        }
        const float x = features[static_cast<uint64_t>(term.feature) * n_samples + row];
        if (isnan(x)) {
            undetermined = true;
            continue;
        }
        const bool holds =
            term.sign == GAFIME_DECISION_PATH_SIGN_LE ? x <= term.threshold : x > term.threshold;
        if (!holds) {
            member = false;
            break;
        }
    }

    membership[static_cast<uint64_t>(path_idx) * n_samples + row] =
        member ? (undetermined ? nanf("") : 1.0f) : 0.0f;
}

__global__ void decision_path_bitset_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    uint32_t words_per_path,
    uint32_t* membership_words
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = row_offset + static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count || row >= n_samples) {
        return;
    }

    const uint32_t begin = path_offsets[path_idx];
    const uint32_t end = path_offsets[path_idx + 1];
    bool member = true;
    for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
        const GafimeDecisionPathTerm term = terms[term_idx];
        if (term.feature >= n_features) {
            member = false;
            break;
        }
        const float x = features[static_cast<uint64_t>(term.feature) * n_samples + row];
        if (!isfinite(x)) {
            member = false;
            break;
        }
        const bool holds =
            term.sign == GAFIME_DECISION_PATH_SIGN_LE ? x <= term.threshold : x > term.threshold;
        if (!holds) {
            member = false;
            break;
        }
    }

    if (member) {
        const uint64_t word_idx =
            static_cast<uint64_t>(path_idx) * words_per_path + (row >> 5u);
        atomicOr(&membership_words[word_idx], 1u << (row & 31u));
    }
}

__global__ void score_decision_path_bitset_kernel(
    const uint32_t* membership_words,
    const float* target,
    const double* target_stats,
    uint64_t n_samples,
    uint32_t path_count,
    uint32_t words_per_path,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint32_t path_idx = blockIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    uint64_t local_sx = 0u;
    double local_sxy = 0.0;
    const double mean_y = target_stats[1];
    const uint64_t path_offset = static_cast<uint64_t>(path_idx) * words_per_path;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const uint32_t word = membership_words[path_offset + (row >> 5u)];
            if (((word >> (row & 31u)) & 1u) != 0u) {
                ++local_sx;
                local_sxy += static_cast<double>(y) - mean_y;
            }
        }
    }

    __shared__ uint64_t sx[256];
    __shared__ double sxy[256];
    sx[threadIdx.x] = local_sx;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double pearson = 0.0;
        const double n = target_stats[0];
        if (n > 0.0) {
            const double count = static_cast<double>(sx[0]);
            const double sxx = fmax(count - count * count / n, 0.0);
            const double syy = fmax(target_stats[2], 0.0);
            const double denom = sqrt(fmax(sxx * syy, 0.0));
            if (denom > 0.0) {
                pearson = fmin(fmax(sxy[0] / denom, -1.0), 1.0);
            }
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = static_cast<float>(pearson);
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = static_cast<float>(fmin(fmax(pearson * pearson, 0.0), 1.0));
            }
            metric_values[static_cast<uint64_t>(path_idx) * metric_count + metric_idx] = out;
        }
    }
}

__global__ void decision_path_target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    double* target_stats
) {
    uint64_t local_n = 0u;
    double local_sy = 0.0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            ++local_n;
            local_sy += static_cast<double>(y);
        }
    }

    __shared__ uint64_t sn[256];
    __shared__ double sy[256];
    __shared__ double mean_y;
    __shared__ double syy[256];
    sn[threadIdx.x] = local_n;
    sy[threadIdx.x] = local_sy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean_y = sn[0] > 0u ? sy[0] / static_cast<double>(sn[0]) : 0.0;
    }
    __syncthreads();

    double local_syy = 0.0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const double dy = static_cast<double>(y) - mean_y;
            local_syy += dy * dy;
        }
    }
    syy[threadIdx.x] = local_syy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            syy[threadIdx.x] += syy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        target_stats[0] = static_cast<double>(sn[0]);
        target_stats[1] = mean_y;
        target_stats[2] = syy[0];
    }
}

static __forceinline__ __device__ float decision_path_binary_pearson(
    uint32_t inside_count,
    double centered_inside_sum,
    const double* target_stats
) {
    const double n = target_stats[0];
    if (n <= 0.0) {
        return 0.0f;
    }
    const double count = static_cast<double>(inside_count);
    const double sxx = fmax(count - count * count / n, 0.0);
    const double syy = fmax(target_stats[2], 0.0);
    const double denom = sqrt(fmax(sxx * syy, 0.0));
    if (denom <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(fmin(fmax(centered_inside_sum / denom, -1.0), 1.0));
}

__global__ void score_decision_path_direct_stats_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint32_t path_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    const float pearson = decision_path_binary_pearson(
        inside_counts[path_idx],
        inside_sum_y[path_idx],
        target_stats
    );

    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const uint32_t metric_id = metric_ids[metric_idx];
        float out = 0.0f;
        if (metric_id == GAFIME_METRIC_PEARSON) {
            out = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
        }
        metric_values[static_cast<uint64_t>(path_idx) * metric_count + metric_idx] = out;
    }
}

__global__ void score_decision_path_direct_stats_scatter_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
    const uint32_t* original_paths,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* final_metric_values
) {
    const uint32_t path_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    const float pearson = decision_path_binary_pearson(
        inside_counts[path_idx],
        inside_sum_y[path_idx],
        target_stats
    );

    const uint32_t original_path = original_paths[path_idx];
    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const uint32_t metric_id = metric_ids[metric_idx];
        float out = 0.0f;
        if (metric_id == GAFIME_METRIC_PEARSON) {
            out = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
        }
        final_metric_values[static_cast<uint64_t>(original_path) * metric_count + metric_idx] = out;
    }
}

__global__ void scatter_decision_path_score_metrics_kernel(
    const float* group_metric_values,
    const uint32_t* original_paths,
    uint32_t group_path_count,
    uint32_t metric_count,
    float* final_metric_values
) {
    const uint64_t value_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t value_count = static_cast<uint64_t>(group_path_count) * metric_count;
    if (value_idx >= value_count) {
        return;
    }
    const uint32_t local_path = static_cast<uint32_t>(value_idx / metric_count);
    const uint32_t metric_idx = static_cast<uint32_t>(value_idx - static_cast<uint64_t>(local_path) * metric_count);
    const uint32_t original_path = original_paths[local_path];
    final_metric_values[static_cast<uint64_t>(original_path) * metric_count + metric_idx] =
        group_metric_values[value_idx];
}

}  // namespace gafime_cuda_v1::rt_kernel

#endif
