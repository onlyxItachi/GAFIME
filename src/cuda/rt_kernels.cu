#include "rt_kernels.cuh"

#ifdef GAFIME_CUDA_RT_OPTIX_DEVICE

#include <optix.h>
#include <optix_device.h>

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes;
    const float* target;
    float* membership;
    uint32_t* membership_words;
    uint32_t* direct_inside_counts;
    float* direct_inside_sum_y;
    uint32_t rows;
    uint32_t path_count;
    uint32_t geometry_mode;
    uint32_t words_per_path;
    const uint32_t* group_path_offsets;
    uint32_t group_count;
    uint32_t point_group_stride;
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

extern "C" __global__ void __raygen__gafime_dp()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint32_t row = launch_idx.x;
    const uint32_t group_idx = launch_idx.y;
    if (row >= params.rows || (params.group_count > 1u && group_idx >= params.group_count)) {
        return;
    }

    const uint64_t point_offset = params.group_count > 1u
        ? static_cast<uint64_t>(group_idx) * params.point_group_stride + static_cast<uint64_t>(row) * 3u
        : static_cast<uint64_t>(row) * 3u;
    const float x = params.points_xyz[point_offset + 0u];
    const float y = params.points_xyz[point_offset + 1u];
    const float z = params.points_xyz[point_offset + 2u];
    uint32_t payload_row = row;
    const bool triangle_2d = params.geometry_mode == 1u || params.geometry_mode == 2u;
    const float group_z = params.geometry_mode == 2u ? static_cast<float>(group_idx) * 4.0f : 0.0f;
    const float3 origin = triangle_2d ? make_float3(x, y, group_z - 1.0f) : make_float3(x, y, z);
    const float3 direction = triangle_2d ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    const float tmax = triangle_2d ? 2.0f : 1.0e-7f;

    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0,
        1,
        0,
        payload_row
    );
}

extern "C" __global__ void __miss__gafime_dp() {}

extern "C" __global__ void __intersection__gafime_dp_box()
{
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    const float3 point = optixGetWorldRayOrigin();
    const gafime_cuda_v1::rt_kernel::GafimeRtBox box = params.boxes[primitive_idx];

    bool inside = inside_dim(point.x, box.lo_x, box.hi_x, (box.open_lo_mask & 1u) != 0u);
    if (box.dims > 1u) {
        inside = inside && inside_dim(point.y, box.lo_y, box.hi_y, (box.open_lo_mask & 2u) != 0u);
    }
    if (box.dims > 2u) {
        inside = inside && inside_dim(point.z, box.lo_z, box.hi_z, (box.open_lo_mask & 4u) != 0u);
    }
    if (inside) {
        optixReportIntersection(0.0f, 0);
    }
}

extern "C" __global__ void __anyhit__gafime_dp_mark()
{
    const uint32_t row = optixGetPayload_0();
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    const uint32_t group_idx = params.group_count > 1u ? optixGetInstanceId() : 0u;
    const uint32_t local_path_idx =
        (params.geometry_mode == 1u || params.geometry_mode == 2u) ? (primitive_idx >> 1u) : primitive_idx;
    const uint32_t path_base = params.group_path_offsets != nullptr ? params.group_path_offsets[group_idx] : 0u;
    const uint32_t path_idx = path_base + local_path_idx;
    if (path_idx < params.path_count) {
        const float3 point = optixGetWorldRayOrigin();
        const gafime_cuda_v1::rt_kernel::GafimeRtBox box = params.boxes[path_idx];
        bool inside = inside_dim(point.x, box.lo_x, box.hi_x, (box.open_lo_mask & 1u) != 0u);
        if (box.dims > 1u) {
            inside = inside && inside_dim(point.y, box.lo_y, box.hi_y, (box.open_lo_mask & 2u) != 0u);
        }
        if (box.dims > 2u) {
            inside = inside && inside_dim(point.z, box.lo_z, box.hi_z, (box.open_lo_mask & 4u) != 0u);
        }
        if (inside) {
            const uint64_t out_idx = static_cast<uint64_t>(path_idx) * params.rows + row;
            if (params.direct_inside_counts != nullptr) {
                bool owns_hit = true;
                if (params.geometry_mode == 1u) {
                    const float width = box.hi_x - box.lo_x;
                    const float height = box.hi_y - box.lo_y;
                    if (width > 0.0f && height > 0.0f) {
                        const float nx = (point.x - box.lo_x) / width;
                        const float ny = (point.y - box.lo_y) / height;
                        owns_hit = (primitive_idx & 1u) == 0u ? ny <= nx : ny > nx;
                    }
                }
                const float y = params.target[row];
                if (owns_hit && isfinite(y)) {
                    atomicAdd(&params.direct_inside_counts[path_idx], 1u);
                    atomicAdd(&params.direct_inside_sum_y[path_idx], y);
                }
            } else if (params.membership_words != nullptr) {
                const uint64_t word_idx =
                    static_cast<uint64_t>(path_idx) * params.words_per_path + (row >> 5u);
                atomicOr(&params.membership_words[word_idx], 1u << (row & 31u));
            } else {
                params.membership[out_idx] = 1.0f;
            }
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
        (static_cast<uint64_t>(group_idx) * n_samples + row) * 3u;
    for (uint32_t dim = 0; dim < 3u; ++dim) {
        const float value = dim < dims ? features[static_cast<uint64_t>(axes[dim]) * n_samples + row] : 0.0f;
        points_xyz[point_base + dim] = value;
    }
}

__global__ void decision_path_membership_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
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
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    uint32_t words_per_path,
    uint32_t* membership_words
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
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

    float local_n = 0.0f;
    float local_sx = 0.0f;
    float local_sy = 0.0f;
    float local_syy_raw = 0.0f;
    float local_sxy_raw = 0.0f;
    const uint64_t path_offset = static_cast<uint64_t>(path_idx) * words_per_path;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const uint32_t word = membership_words[path_offset + (row >> 5u)];
            const float x = ((word >> (row & 31u)) & 1u) != 0u ? 1.0f : 0.0f;
            local_n += 1.0f;
            local_sx += x;
            local_sy += y;
            local_syy_raw += y * y;
            local_sxy_raw += x * y;
        }
    }

    __shared__ float sn[256];
    __shared__ float sx[256];
    __shared__ float sy[256];
    __shared__ float syy_raw[256];
    __shared__ float sxy_raw[256];
    sn[threadIdx.x] = local_n;
    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    syy_raw[threadIdx.x] = local_syy_raw;
    sxy_raw[threadIdx.x] = local_sxy_raw;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            syy_raw[threadIdx.x] += syy_raw[threadIdx.x + stride];
            sxy_raw[threadIdx.x] += sxy_raw[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float pearson = 0.0f;
        if (sn[0] > 0.0f) {
            const float inv_n = 1.0f / sn[0];
            const float sxx = fmaxf(sx[0] - sx[0] * sx[0] * inv_n, 0.0f);
            const float syy = fmaxf(syy_raw[0] - sy[0] * sy[0] * inv_n, 0.0f);
            const float sxy = sxy_raw[0] - sx[0] * sy[0] * inv_n;
            const float denom = sqrtf(fmaxf(sxx * syy, 0.0f));
            if (denom > 0.0f) {
                pearson = fminf(fmaxf(sxy / denom, -1.0f), 1.0f);
            }
        }
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
}

__global__ void decision_path_target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    float* target_stats
) {
    float local_n = 0.0f;
    float local_sy = 0.0f;
    float local_syy = 0.0f;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            local_n += 1.0f;
            local_sy += y;
            local_syy += y * y;
        }
    }

    __shared__ float sn[256];
    __shared__ float sy[256];
    __shared__ float syy[256];
    sn[threadIdx.x] = local_n;
    sy[threadIdx.x] = local_sy;
    syy[threadIdx.x] = local_syy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        target_stats[0] = sn[0];
        target_stats[1] = sy[0];
        target_stats[2] = syy[0];
    }
}

__global__ void score_decision_path_direct_stats_kernel(
    const uint32_t* inside_counts,
    const float* inside_sum_y,
    const float* target_stats,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint32_t path_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    float pearson = 0.0f;
    const float n = target_stats[0];
    if (n > 0.0f) {
        const float sx = static_cast<float>(inside_counts[path_idx]);
        const float sy = target_stats[1];
        const float syy_raw = target_stats[2];
        const float sxy_raw = inside_sum_y[path_idx];
        const float inv_n = 1.0f / n;
        const float sxx = fmaxf(sx - sx * sx * inv_n, 0.0f);
        const float syy = fmaxf(syy_raw - sy * sy * inv_n, 0.0f);
        const float sxy = sxy_raw - sx * sy * inv_n;
        const float denom = sqrtf(fmaxf(sxx * syy, 0.0f));
        if (denom > 0.0f) {
            pearson = fminf(fmaxf(sxy / denom, -1.0f), 1.0f);
        }
    }

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
