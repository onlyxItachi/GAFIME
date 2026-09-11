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

/* The reusable compact query has its own larger bounded envelope.  Do not
 * silently inherit the cold dense-materialization limits above. */
#define GAFIME_CUDA_RT_REGION_QUERY_MAX_ROWS 262144u
#define GAFIME_CUDA_RT_REGION_QUERY_MAX_REGIONS 8192u
#define GAFIME_CUDA_RT_REGION_QUERY_MAX_TERMS_PER_REGION 64u
#define GAFIME_CUDA_RT_REGION_QUERY_MAX_GROUPS 64u

/*
 * The compact regional-statistics query is a local OptiX experiment beside
 * the normal semantic primitive table.  Its records deliberately contain
 * only physical slots, exact binary sufficient statistics, and optional
 * profile-native arithmetic finalizations.  Rust retains program/candidate
 * identity, contextual-evidence meaning, labels policy, and selection.
 */
#define GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION 0x00010000u

/* Each query must select exactly one route flag. */
#define GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT 0x1u
/* Local validation/benchmark hook: use the native exact spatial-bin SM path
 * instead of OptiX.  It is never a geometry-mode selector. */
#define GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM 0x2u
/* Diagnostic exact comparator: evaluate every submitted conjunction for
 * every row.  It is mutually exclusive with REQUIRE_RT and the binned SM
 * lane; production callers should use FORCE_SM only for the spatial-bin
 * comparator. */
#define GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE 0x4u

#define GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY 0x1u
#define GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED 0x2u
#define GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED 0x4u
#define GAFIME_SEMANTIC_RT_REGION_STAT_MASK_ALL \
    (GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY | \
     GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED | \
     GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED)

#define GAFIME_SEMANTIC_RT_REGION_FINALIZE_OCCUPANCY 0x1u
#define GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_AGREEMENT 0x2u
#define GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_IOU 0x4u
#define GAFIME_SEMANTIC_RT_REGION_FINALIZE_LABELED_GINI_GAIN 0x8u
#define GAFIME_SEMANTIC_RT_REGION_FINALIZE_MASK_ALL \
    (GAFIME_SEMANTIC_RT_REGION_FINALIZE_OCCUPANCY | \
     GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_AGREEMENT | \
     GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_IOU | \
     GAFIME_SEMANTIC_RT_REGION_FINALIZE_LABELED_GINI_GAIN)

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

/* Opaque query ownership is deliberately separate from a semantic bank.  The
 * caller must keep both referenced bank handles alive until query free; the
 * Rust owner does this with retained bank Arcs.  Create synchronously copies
 * every descriptor below and retains its own device geometry/points/SBT, so
 * neither descriptor pointers nor caller labels become cache identities. */
typedef void* GafimeGpuSemanticRegionQuery;

/* `terms` and `region_offsets` form physical frozen conjunctions in submitted
 * result order.  `paired_term_slots`, when `paired_bank` is present, supplies
 * the paired bank's physical input slot for each term; it need not equal the
 * primary slot because retained accepted atoms can have different layouts.
 * A partition is a contiguous range of those ordinals, not a candidate-id
 * range and not an assertion that boxes are disjoint.  Native code preserves
 * this ordinal mapping through internal grouped physical work and independently
 * proves any first-hit eligibility. */
typedef struct GafimeSemanticRtRegionQueryDesc {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeGpuSemanticBank primary_bank;
    GafimeGpuSemanticBank paired_bank;
    const GafimeSemanticFrozenRegionTerm* terms;
    const uint32_t* paired_term_slots;
    const uint32_t* region_offsets;
    const uint32_t* partition_offsets;
    uint64_t term_count;
    uint32_t region_count;
    uint32_t partition_count;
    uint32_t flags;
    uint32_t reserved32;
    /* Bounds a conservative reservation for explicit query-owned create
     * resources: retained host snapshots and construction staging plus CUDA
     * descriptors/points/masks, SBT/params, GAS/IAS and retained build
     * workspaces.  Opaque vendor driver/context/module/pipeline overhead and
     * the caller-owned semantic banks remain outside this boundary. */
    uint64_t max_persistent_bytes;
    uint64_t reserved[7];
} GafimeSemanticRtRegionQueryDesc;

/* Actual supplied labels only.  The caller validates the canonical binary
 * label domain before this boundary; native defensively rechecks sorted,
 * unique, in-range rows and literal {0,1} values.  A present zero-count
 * context may use null arrays and means labeled support zero, not missing
 * labels.  This object is copied for one execute call and is never retained
 * or keyed by its host pointers. */
typedef struct GafimeSemanticRtBinaryLabelContext {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint64_t* row_indices;
    const uint8_t* values;
    uint64_t count;
    uint64_t reserved[6];
} GafimeSemanticRtBinaryLabelContext;

typedef struct GafimeSemanticRtRegionExecuteDesc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t statistic_mask;
    uint32_t finalizer_mask;
    GafimeSemanticRtBinaryLabelContext labels;
    /* Bounds explicit execute-only staging.  Query-owned retained scratch is
     * reported by create and does not silently grow during execute. */
    uint64_t max_temporary_bytes;
    uint64_t reserved[7];
} GafimeSemanticRtRegionExecuteDesc;

/* All count fields are exact integer sufficient statistics.  A channel not
 * requested by `statistic_mask` is zero.  The four scalar fields are optional
 * fp32 finalizations performed on device from these integers; a state of zero
 * means that finalizer was not requested.  No score has target semantics. */
typedef struct GafimeSemanticRtRegionExactStats {
    uint64_t row_count;
    uint64_t label_support;
    uint64_t occupancy_inside;
    uint64_t paired_n00;
    uint64_t paired_n01;
    uint64_t paired_n10;
    uint64_t paired_n11;
    uint64_t label_outside_0;
    uint64_t label_outside_1;
    uint64_t label_inside_0;
    uint64_t label_inside_1;
    float occupancy;
    float paired_agreement;
    float paired_iou;
    float labeled_gini_gain;
    uint32_t occupancy_state;
    uint32_t paired_agreement_state;
    uint32_t paired_iou_state;
    uint32_t labeled_gini_gain_state;
    uint64_t reserved[4];
} GafimeSemanticRtRegionExactStats;

typedef struct GafimeSemanticRtRegionStatsTable {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t requested_statistic_mask;
    uint32_t finalized_mask;
    uint64_t capacity;
    uint64_t count;
    GafimeSemanticRtRegionExactStats* records;
    uint64_t reserved[8];
} GafimeSemanticRtRegionStatsTable;

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

/* Local-only reusable compact region query.  Create consumes immutable bank
 * snapshots and descriptor bytes; execute changes only contextual labels and
 * the requested physical count/finalizer channels.  `persistent_bytes_out`
 * reports the conservative explicit create/retained reservation above (not
 * an exact live-VRAM measurement); an early admission failure may report its
 * bounded prequery reservation.  `temporary_peak_out` reports explicit
 * execute-only staging. */
GAFIME_GPU_API int gafime_gpu_semantic_region_query_create_rt_v1(
    const GafimeSemanticRtRegionQueryDesc* desc,
    GafimeGpuSemanticRegionQuery* query_out,
    uint64_t* persistent_bytes_out
);

GAFIME_GPU_API int gafime_gpu_semantic_region_query_execute_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    const GafimeSemanticRtRegionExecuteDesc* desc,
    GafimeSemanticRtRegionStatsTable* stats_out,
    uint64_t* temporary_peak_out
);

/* Materializes exactly one fresh fp32 primary-bank slot from the successful
 * most-recent query membership: each row receives the integer count of
 * matching submitted regions (0 through region_count).  It is deliberately
 * not a weighted score, target channel, or generic expression operation.
 * The caller must retain the query's primary bank and choose a fresh physical
 * output slot; native commits that slot only after synchronized completion. */
GAFIME_GPU_API int gafime_gpu_semantic_region_query_materialize_coverage_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    uint32_t output_slot,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
);

/* Materializes exactly one fresh fp32 primary-bank slot from the successful
 * most-recent query membership using one finite f32 weight per submitted
 * region.  The weighted endpoint admits exactly two through sixty-four
 * regions; `region_weights` has exactly `region_weight_count` entries and is
 * copied synchronously before launch; entries remain aligned with the query's
 * submitted canonical region order.  Native rejects a null/misaligned array,
 * an out-of-range or mismatched count, and every NaN or infinity.
 *
 * Every row starts from +0 and conditionally adds matching weights in that
 * same ordinal order.  This is a distinct local physical operation from
 * integer coverage, including all-one weights.  Its explicit temporary peak
 * contains the copied device weight vector plus its finite-output validation
 * flag.  A finite weight vector whose fp32 row accumulation overflows returns
 * INVALID_ARGUMENT without committing the output slot; finite subnormal
 * outputs remain valid. */
GAFIME_GPU_API int gafime_gpu_semantic_region_query_materialize_weighted_sum_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    uint32_t output_slot,
    const float* region_weights,
    uint64_t region_weight_count,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
);

/* If teardown cannot select the owning CUDA device, this returns an error and
 * retains caller ownership of the query so release can be retried. */
GAFIME_GPU_API int gafime_gpu_semantic_region_query_free_rt_v1(
    GafimeGpuSemanticRegionQuery query
);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_CUDA_RT_ABI_HPP */
