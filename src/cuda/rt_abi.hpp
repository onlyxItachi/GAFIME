#ifndef GAFIME_CUDA_RT_ABI_HPP
#define GAFIME_CUDA_RT_ABI_HPP

#include "cuda_api.hpp"
#include "../common/gafime_semantic_primitives_abi.hpp"

#define GAFIME_GPU_DEVICE_FLAG_OPTIX_RT 0x100u
#define GAFIME_DECISION_PATH_SIGN_LE 1u
#define GAFIME_DECISION_PATH_SIGN_GT 2u
#define GAFIME_DECISION_PATH_FLAG_REQUIRE_RT 0x1u
/* Conservative historical path-count ceiling retained by the local u32 ABI. */
#define GAFIME_MAX_DECISION_PATH_COUNT (UINT32_MAX / 4u)

/* Local OptiX semantic-region experiment bounds.  They deliberately describe
 * physical execution only: Rust remains the owner of program identity,
 * candidate policy, provenance and run partitioning. */
#define GAFIME_CUDA_RT_SEMANTIC_MAX_ROWS 65536u
#define GAFIME_CUDA_RT_SEMANTIC_MAX_REGIONS 256u
#define GAFIME_CUDA_RT_SEMANTIC_MAX_AXES 3u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GafimeDecisionPathTerm {
    uint32_t feature;
    uint32_t sign;
    float threshold;
    uint32_t reserved32;
    uint64_t reserved[2];
} GafimeDecisionPathTerm;

typedef struct GafimeDecisionPathBatch {
    uint32_t abi_version;
    uint32_t path_count;
    uint32_t term_count;
    uint32_t flags;
    const GafimeDecisionPathTerm* terms;
    const uint32_t* path_offsets;
    float* membership_host;
    uint64_t reserved[8];
} GafimeDecisionPathBatch;

typedef struct GafimeDecisionPathScoreBatch {
    uint32_t abi_version;
    uint32_t path_count;
    uint32_t term_count;
    uint32_t flags;
    const GafimeDecisionPathTerm* terms;
    const uint32_t* path_offsets;
    const uint32_t* metric_ids;
    uint32_t metric_count;
    uint32_t reserved32;
    uint64_t reserved[7];
} GafimeDecisionPathScoreBatch;

GAFIME_GPU_API int gafime_gpu_decision_path_membership(
    GafimeGpuMatrix matrix,
    const GafimeDecisionPathBatch* paths
);

GAFIME_GPU_API int gafime_gpu_decision_path_score(
    GafimeGpuMatrix matrix,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result_out
);

GAFIME_GPU_API int gafime_gpu_decision_path_release_device_state(uint32_t device_id);

/*
 * Local-only, strict-RT lowering for an fp32 semantic bank and a v1.3
 * physical FROZEN_REGION_CONJUNCTION batch.  It does not participate in the
 * normative semantic primitive table and never falls back to an SM/host path.
 *
 * `max_temporary_bytes` bounds every explicit temporary host/device buffer
 * used by this cold, stateless call.  After structural validation and OptiX
 * sizing, `peak_bytes_out` receives the exact planned explicit-buffer peak
 * before an over-budget request fails with OUT_OF_MEMORY.  A lower budget may
 * reject earlier, before the shared validator can copy the bank's initialized
 * slot mask or the fixed SBT records can be allocated; in that case the output
 * is the bounded prequery minimum, not a full plan peak.  The bank allocation
 * and its preallocated output slots, plus opaque CUDA/OptiX driver/context overhead, are
 * intentionally outside the count.  `peak_bytes_out` is zeroed on an invalid
 * or unavailable request.  Successful output slots are committed only after
 * finite-value verification and full device synchronization.
 */
GAFIME_GPU_API int gafime_gpu_semantic_region_materialize_rt_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticProgramBatch* batch,
    uint64_t max_temporary_bytes,
    uint64_t* peak_bytes_out
);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_CUDA_RT_ABI_HPP */
