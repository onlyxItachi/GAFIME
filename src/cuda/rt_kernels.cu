#include "rt_kernels.cuh"

#ifdef GAFIME_CUDA_RT_OPTIX_DEVICE

#include <optix.h>
#include <optix_device.h>

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes;
    float* membership;
    uint32_t rows;
    uint32_t path_count;
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
    const uint32_t row = optixGetLaunchIndex().x;
    if (row >= params.rows) {
        return;
    }

    const float x = params.points_xyz[row * 3u + 0u];
    const float y = params.points_xyz[row * 3u + 1u];
    const float z = params.points_xyz[row * 3u + 2u];
    uint32_t payload_row = row;

    optixTrace(
        params.handle,
        make_float3(x, y, z),
        make_float3(1.0f, 0.0f, 0.0f),
        0.0f,
        1.0e-7f,
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
    params.membership[static_cast<uint64_t>(primitive_idx) * params.rows + row] = 1.0f;
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

}  // namespace gafime_cuda_v1::rt_kernel

#endif
