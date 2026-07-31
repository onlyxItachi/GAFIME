#ifndef GAFIME_CUDA_RT_ABI_HPP
#define GAFIME_CUDA_RT_ABI_HPP

#include "cuda_api.hpp"

#define GAFIME_GPU_DEVICE_FLAG_OPTIX_RT 0x100u
#define GAFIME_DECISION_PATH_SIGN_LE 1u
#define GAFIME_DECISION_PATH_SIGN_GT 2u
#define GAFIME_DECISION_PATH_FLAG_REQUIRE_RT 0x1u
/* Conservative historical path-count ceiling retained by the local u32 ABI. */
#define GAFIME_MAX_DECISION_PATH_COUNT (UINT32_MAX / 4u)

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

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_CUDA_RT_ABI_HPP */
