#include "rt_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace gafime_cuda_v1::rt_kernel {

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
