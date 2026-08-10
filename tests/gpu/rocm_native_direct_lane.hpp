#ifndef GAFIME_ROCM_NATIVE_DIRECT_LANE_HPP
#define GAFIME_ROCM_NATIVE_DIRECT_LANE_HPP

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace gafime_rocm_native_direct {

// The benchmark helper owns these buffers.  The direct lane deliberately
// carries untyped pointers only across this small C++ boundary; the HIP
// translation unit selects the concrete Storage/Accum/Result types once from
// the numeric route and never branches on dtype in a hot loop.
struct Buffers {
    void* materialized = nullptr;
    void* target_ranks = nullptr;
    void* metric_values = nullptr;
    void* partial_scores = nullptr;
    uint32_t* partial_indices = nullptr;
    uint32_t* selected_indices = nullptr;
    void* selected_metric_values = nullptr;
};

hipError_t materialize(
    uint32_t profile,
    const void* features,
    const uint32_t* combos,
    void* materialized,
    uint64_t rows,
    uint32_t candidates,
    uint32_t arity,
    hipStream_t stream
);

hipError_t target_ranks(
    uint32_t profile,
    const void* target,
    void* target_ranks,
    uint64_t rows,
    hipStream_t stream
);

hipError_t metric(
    uint32_t profile,
    uint32_t metric_id,
    uint32_t arity,
    uint32_t bins,
    const void* features,
    const void* target,
    const void* column_means,
    const void* target_ranks,
    const uint32_t* combos,
    const uint32_t* metric_ids,
    uint64_t rows,
    uint64_t candidates,
    void* metric_values,
    hipStream_t stream
);

hipError_t ranking_topk(
    uint32_t profile,
    const void* metric_values,
    uint64_t candidates,
    uint32_t top_k,
    void* partial_scores,
    uint32_t* partial_indices,
    uint32_t* selected_indices,
    uint32_t partial_blocks,
    hipStream_t stream
);

hipError_t selected_rows(
    uint32_t profile,
    const void* metric_values,
    const uint32_t* selected_indices,
    uint32_t selected_count,
    void* selected_metric_values,
    hipStream_t stream
);

}  // namespace gafime_rocm_native_direct

#endif  // GAFIME_ROCM_NATIVE_DIRECT_LANE_HPP
