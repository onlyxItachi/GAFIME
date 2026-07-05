#ifndef GAFIME_CUDA_RT_LAUNCHER_CUH
#define GAFIME_CUDA_RT_LAUNCHER_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/gafime_gpu_abi.hpp"

namespace gafime_cuda_v1 {

void tune_rt_kernels_for_device(const cudaDeviceProp& props);

cudaError_t launch_decision_path_membership(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership,
    cudaStream_t stream
);

int execute_decision_path_membership(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* paths
);

}  // namespace gafime_cuda_v1

#endif  // GAFIME_CUDA_RT_LAUNCHER_CUH
