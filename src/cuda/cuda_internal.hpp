#ifndef GAFIME_CUDA_INTERNAL_HPP
#define GAFIME_CUDA_INTERNAL_HPP

#include <cstdint>

#include "cuda_api.hpp"

namespace gafime_cuda_v1::detail {

struct CudaMatrixView {
    float* features;
    float* target;
    uint64_t rows;
    uint32_t cols;
    uint32_t device_id;
    uint64_t architecture_class;
    uint32_t device_flags;
    bool features_are_finite;
    uint64_t feature_generation;
    uint64_t target_generation;
};

int inspect_cuda_matrix(GafimeGpuMatrix matrix, CudaMatrixView* view_out);

}  // namespace gafime_cuda_v1::detail

#endif /* GAFIME_CUDA_INTERNAL_HPP */
