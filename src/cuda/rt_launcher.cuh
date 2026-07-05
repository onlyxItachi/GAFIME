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
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    const GafimeDecisionPathBatch* paths
);

int execute_decision_path_score(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
);

}  // namespace gafime_cuda_v1

#endif  // GAFIME_CUDA_RT_LAUNCHER_CUH
