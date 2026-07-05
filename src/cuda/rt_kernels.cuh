#ifndef GAFIME_CUDA_RT_KERNELS_CUH
#define GAFIME_CUDA_RT_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_cuda_v1::rt_kernel {

__global__ void decision_path_membership_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership
);

}  // namespace gafime_cuda_v1::rt_kernel

#endif  // GAFIME_CUDA_RT_KERNELS_CUH
