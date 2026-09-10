#ifndef GAFIME_CUDA_INTERNAL_HPP
#define GAFIME_CUDA_INTERNAL_HPP

#include <cstdint>
#include <vector>

#include "cuda_api.hpp"
#include "../common/gafime_semantic_primitives_abi.hpp"

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

/*
 * Immediate, RT-free view of a typed semantic bank.  This is deliberately a
 * C++-internal bridge rather than an extension of the frozen standard ABI:
 * the RT launcher receives only physical resident slots and must not acquire
 * a second semantic catalog.  `initialized_slots` remains owned by the bank
 * and is valid only while the caller holds its synchronous bank call.
 */
struct CudaSemanticBankView {
    float* columns;
    uint64_t rows;
    uint32_t source_slots;
    uint32_t slot_capacity;
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t architecture_class;
    GafimeNumericRoute route;
    const std::vector<uint8_t>* initialized_slots;
};

int inspect_cuda_semantic_bank(
    GafimeGpuSemanticBank bank,
    CudaSemanticBankView* view_out
);

/* Marks a fresh derived-output set only after an RT launcher has synchronously
 * completed and checked every output.  The operation is all-or-nothing and
 * intentionally knows nothing about region semantics or OptiX state. */
int commit_cuda_semantic_bank_outputs(
    GafimeGpuSemanticBank bank,
    const uint32_t* output_slots,
    uint32_t output_count
);

}  // namespace gafime_cuda_v1::detail

#endif /* GAFIME_CUDA_INTERNAL_HPP */
