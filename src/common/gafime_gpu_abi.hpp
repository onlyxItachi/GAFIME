/*
 * GAFIME v1 GPU ABI
 *
 * This is the stable Rust<->GPU host boundary. Python never calls this ABI
 * directly. Vendor payloads implement it; crates/gafime-gpu-sys bindgens it.
 */

#ifndef GAFIME_GPU_ABI_HPP
#define GAFIME_GPU_ABI_HPP

#include <stdint.h>

#ifdef _WIN32
  #ifdef GAFIME_GPU_BUILDING_DLL
    #define GAFIME_GPU_API __declspec(dllexport)
  #else
    #define GAFIME_GPU_API __declspec(dllimport)
  #endif
#else
  #define GAFIME_GPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GAFIME_ABI_VERSION_MAJOR 1u
#define GAFIME_ABI_VERSION_MINOR 0u
#define GAFIME_ABI_VERSION ((GAFIME_ABI_VERSION_MAJOR << 16) | GAFIME_ABI_VERSION_MINOR)

#define GAFIME_LAUNCH_FLAG_GRAPH 0x1u
#define GAFIME_LAUNCH_FLAG_MI_APPROX 0x2u
#define GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL 0x4u
#define GAFIME_RESULT_FLAG_GRAPH_REPLAYED 0x1u

/*
 * reserved[0] carries a nonzero caller-owned immutable descriptor generation.
 * Zero keeps older same-ABI callers valid, but payloads must upload descriptors
 * on every execution because no stable content identity was supplied.
 */
#define GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT 0u

#define GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY 0x1u
#define GAFIME_GPU_DEVICE_FLAG_INTEGRATED 0x2u
#define GAFIME_GPU_DEVICE_FLAG_DISCRETE 0x4u
#define GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY 0x8u
#define GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH 0x10u
#define GAFIME_GPU_DEVICE_FLAG_AMD_RDNA 0x20u
#define GAFIME_GPU_DEVICE_FLAG_AMD_CDNA 0x40u
#define GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY 0x80u
#define GAFIME_GPU_DEVICE_FLAG_OPTIX_RT 0x100u
/* Legacy ABI 1.0 capability; this does not imply generation-token support. */
#define GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL 0x200u
/* Payload keys immutable launch descriptors by reserved[0] generation. */
#define GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION 0x400u

#define GAFIME_GPU_ARCH_UNKNOWN 0u
#define GAFIME_GPU_ARCH_NVIDIA_TURING 75u
#define GAFIME_GPU_ARCH_NVIDIA_AMPERE 80u
#define GAFIME_GPU_ARCH_NVIDIA_ADA 89u
#define GAFIME_GPU_ARCH_NVIDIA_HOPPER 90u
#define GAFIME_GPU_ARCH_NVIDIA_BLACKWELL 100u
#define GAFIME_GPU_ARCH_AMD_RDNA 1000u
#define GAFIME_GPU_ARCH_AMD_CDNA 2000u
#define GAFIME_GPU_ARCH_APPLE 3000u

#define GAFIME_DECISION_PATH_SIGN_LE 1u
#define GAFIME_DECISION_PATH_SIGN_GT 2u
#define GAFIME_DECISION_PATH_FLAG_REQUIRE_RT 0x1u
/* Four vertices are emitted per RT triangle path and indexed with uint32_t. */
#define GAFIME_MAX_DECISION_PATH_COUNT (UINT32_MAX / 4u)

typedef enum GafimeStatus {
    GAFIME_STATUS_OK = 0,
    GAFIME_STATUS_INVALID_ARGUMENT = -1,
    GAFIME_STATUS_UNSUPPORTED_BACKEND = -2,
    GAFIME_STATUS_OUT_OF_MEMORY = -3,
    GAFIME_STATUS_DEVICE_ERROR = -4,
    GAFIME_STATUS_GRAPH_UNSUPPORTED = -5,
    GAFIME_STATUS_ABI_MISMATCH = -6
} GafimeStatus;

typedef enum GafimeBackendKind {
    GAFIME_BACKEND_CPU = 1,
    GAFIME_BACKEND_CUDA = 2,
    GAFIME_BACKEND_ROCM = 3,
    GAFIME_BACKEND_METAL = 4
} GafimeBackendKind;

typedef enum GafimeMetricId {
    GAFIME_METRIC_PEARSON = 1,
    GAFIME_METRIC_SPEARMAN = 2,
    GAFIME_METRIC_MUTUAL_INFO = 3,
    GAFIME_METRIC_R2 = 4
} GafimeMetricId;

typedef enum GafimeCandidateFamily {
    GAFIME_FAMILY_CONTINUOUS = 1,
    GAFIME_FAMILY_DECISION_PATH = 2,
    GAFIME_FAMILY_TIME_SERIES = 3
} GafimeCandidateFamily;

typedef enum GafimeDataType {
    GAFIME_DTYPE_F32 = 1
} GafimeDataType;

typedef enum GafimeMatrixLayout {
    GAFIME_MATRIX_ROW_MAJOR = 1,
    GAFIME_MATRIX_COLUMN_MAJOR = 2,
    GAFIME_MATRIX_ARROW_COLUMNAR = 3,
    GAFIME_MATRIX_DEVICE_NATIVE = 4
} GafimeMatrixLayout;

typedef enum GafimeGraphMode {
    GAFIME_GRAPH_UNSUPPORTED = 0,
    GAFIME_GRAPH_STREAM_CAPTURE = 1,
    GAFIME_GRAPH_HOST_REPLAY = 2
} GafimeGraphMode;

typedef void* GafimeGpuMatrix;

typedef struct GafimeSliceU32 {
    const uint32_t* ptr;
    uint64_t len;
} GafimeSliceU32;

typedef struct GafimeSliceU64 {
    const uint64_t* ptr;
    uint64_t len;
} GafimeSliceU64;

typedef struct GafimeSliceF32 {
    const float* ptr;
    uint64_t len;
} GafimeSliceF32;

typedef struct GafimeMatrixDesc {
    uint32_t abi_version;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
} GafimeMatrixDesc;

typedef struct GafimeGpuDeviceInfo {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t device_id;
    uint32_t flags;
    char name[128];
    uint64_t total_global_mem_bytes;
    uint32_t multiprocessor_count;
    uint32_t warp_size;
    uint32_t compute_major;
    uint32_t compute_minor;
    uint32_t driver_version;
    uint32_t runtime_version;
    uint64_t reserved[8];
} GafimeGpuDeviceInfo;

typedef struct GafimeGpuGraphCapability {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t graph_mode;
    uint32_t flags;
    uint32_t supports_memcpy_nodes;
    uint32_t supports_kernel_param_update;
    uint32_t supports_device_ranking;
    uint32_t max_captured_nodes;
    uint64_t stable_pointer_flags;
    uint64_t reserved[8];
} GafimeGpuGraphCapability;

typedef struct GafimeShapeHint {
    uint32_t threads_per_block;
    uint32_t items_per_thread;
    uint32_t blocks_per_sm;
    uint32_t min_blocks;
    uint32_t shared_bytes;
    uint32_t register_budget;
    uint32_t occupancy_target_pct;
    uint32_t vendor_hint;
    uint64_t reserved[4];
} GafimeShapeHint;

typedef struct GafimeArityChunk {
    uint32_t arity;
    uint32_t family;
    uint32_t metric_mask;
    uint32_t shape_hint_index;
    uint64_t combo_row_offset;
    uint64_t combo_count;
    uint32_t local_chunk_id;
    uint32_t flags;
    uint64_t descriptor_offset;
    uint64_t descriptor_count;
} GafimeArityChunk;

typedef struct GafimeRankSpec {
    uint32_t top_k;
    uint32_t primary_metric;
    uint32_t descending;
    uint32_t include_ties;
    uint64_t reserved[4];
} GafimeRankSpec;

typedef struct GafimePermutationSchedule {
    uint32_t permutation_count;
    uint32_t mode;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t seed;
    GafimeSliceU64 target_offsets;
    uint64_t reserved[4];
} GafimePermutationSchedule;

typedef struct GafimeLaunchProtocol {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t flags;
    uint32_t max_arity;
    uint64_t n_samples;
    uint32_t n_features;
    uint32_t family_count;
    GafimeSliceU32 combo_indices;
    GafimeSliceU32 metric_ids;
    const GafimeArityChunk* chunks;
    uint32_t chunk_count;
    uint32_t reserved32_a;
    const GafimeShapeHint* shape_hints;
    uint32_t shape_hint_count;
    uint32_t reserved32_b;
    GafimeRankSpec rank;
    GafimePermutationSchedule permutations;
    uint64_t reserved[8];
} GafimeLaunchProtocol;

typedef struct GafimeResultTable {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    float* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    void* backend_private;
    uint64_t reserved[8];
} GafimeResultTable;

typedef struct GafimePermutationSignificanceTable {
    uint32_t abi_version;
    uint32_t metric_count;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    const float* observed_metric_values;
    float* p_values;
    uint64_t reserved[8];
} GafimePermutationSignificanceTable;

typedef struct GafimeDecisionPathTerm {
    uint32_t feature;
    uint32_t sign;
    float threshold;
    uint32_t reserved32;
    uint64_t reserved[2];
} GafimeDecisionPathTerm;

/*
 * Optional CUDA-only v1.1 spike ABI for decision_path materialization. Rust owns
 * path discovery/planning and passes validated terms; the backend only computes
 * hard-AND membership over the resident feature-major matrix. `path_offsets`
 * has length path_count + 1. `membership_host` has path_count * matrix.rows
 * f32s laid out path-major, matching Rust CPU `path_membership`.
 */
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

/*
 * Optional CUDA-only v1.1 spike ABI for compact decision_path scoring. Rust owns
 * path discovery/planning and passes validated terms plus metric ids. The
 * backend computes path membership over the resident matrix and returns compact
 * result rows without copying path_count * rows membership to host.
 */
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

GAFIME_GPU_API int gafime_gpu_device_info(
    uint32_t device_id,
    GafimeGpuDeviceInfo* info_out
);

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
);

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
);

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
);

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix,
    const float* target_host,
    uint64_t rows
);

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix);

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
);

/*
 * Optional backend capability. Payloads that expose this symbol compute
 * permutation-test p-values for already-surfaced result rows. Rust must treat a
 * missing symbol as "not supported" and must not infer p-values from
 * gafime_gpu_execute alone.
 */
GAFIME_GPU_API int gafime_gpu_permutation_pvalues(
    GafimeGpuMatrix matrix,
    const GafimeLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
);

GAFIME_GPU_API int gafime_gpu_decision_path_membership(
    GafimeGpuMatrix matrix,
    const GafimeDecisionPathBatch* paths
);

GAFIME_GPU_API int gafime_gpu_decision_path_score(
    GafimeGpuMatrix matrix,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result_out
);

/* Optional CUDA RT lifecycle capability for releasing per-device native state. */
GAFIME_GPU_API int gafime_gpu_decision_path_release_device_state(uint32_t device_id);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_GPU_ABI_HPP */
