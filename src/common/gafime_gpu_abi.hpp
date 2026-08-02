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
#define GAFIME_PRECISION_ABI_VERSION_MAJOR 1u
#define GAFIME_PRECISION_ABI_VERSION_MINOR 1u
#define GAFIME_PRECISION_ABI_VERSION \
    ((GAFIME_PRECISION_ABI_VERSION_MAJOR << 16) | GAFIME_PRECISION_ABI_VERSION_MINOR)

#define GAFIME_LAUNCH_FLAG_GRAPH 0x1u
#define GAFIME_LAUNCH_FLAG_MI_APPROX 0x2u
#define GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL 0x4u
#define GAFIME_RESULT_FLAG_GRAPH_REPLAYED 0x1u
#define GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE 0x1u

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
/* Legacy ABI 1.0 capability; this does not imply generation-token support. */
#define GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL 0x200u
/* Payload keys immutable launch descriptors by reserved[0] generation. */
#define GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION 0x400u
/* Legacy ABI 1.0 payload-wide MI mode; ABI 1.1 follows the requested profile. */
#define GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64 0x800u
/* Legacy device flag. ABI 1.1 f64 is authoritative in storage_dtype_mask. */
#define GAFIME_GPU_DEVICE_FLAG_F64_STORAGE 0x1000u

#define GAFIME_GPU_ARCH_UNKNOWN 0u
#define GAFIME_GPU_ARCH_NVIDIA_TURING 75u
#define GAFIME_GPU_ARCH_NVIDIA_AMPERE 80u
#define GAFIME_GPU_ARCH_NVIDIA_ADA 89u
#define GAFIME_GPU_ARCH_NVIDIA_HOPPER 90u
#define GAFIME_GPU_ARCH_NVIDIA_BLACKWELL 100u
#define GAFIME_GPU_ARCH_AMD_RDNA 1000u
#define GAFIME_GPU_ARCH_AMD_CDNA 2000u
#define GAFIME_GPU_ARCH_APPLE 3000u

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
    GAFIME_DTYPE_F32 = 1,
    GAFIME_DTYPE_F64 = 2
} GafimeDataType;

#define GAFIME_DTYPE_MASK_F32 0x1u
#define GAFIME_DTYPE_MASK_F64 0x2u

typedef enum GafimePrecisionProfile {
    GAFIME_PRECISION_FP32 = 1,
    GAFIME_PRECISION_MIXED = 2,
    GAFIME_PRECISION_FP64 = 3
} GafimePrecisionProfile;

#define GAFIME_PRECISION_PROFILE_MASK_FP32 0x1u
#define GAFIME_PRECISION_PROFILE_MASK_MIXED 0x2u
#define GAFIME_PRECISION_PROFILE_MASK_FP64 0x4u

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

typedef struct GafimeSliceF64 {
    const double* ptr;
    uint64_t len;
} GafimeSliceF64;

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

typedef struct GafimePrecisionMatrixDesc {
    uint32_t abi_version;
    uint32_t profile;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
    uint64_t reserved[8];
} GafimePrecisionMatrixDesc;

typedef struct GafimePrecisionCapabilities {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t profile_mask;
    uint32_t storage_dtype_mask;
    uint32_t result_dtype_mask;
    uint32_t flags;
    uint64_t reserved[8];
} GafimePrecisionCapabilities;

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

typedef struct GafimePrecisionLaunchProtocol {
    uint32_t abi_version;
    uint32_t profile;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
} GafimePrecisionLaunchProtocol;

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

typedef struct GafimeResultTableF64 {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    double* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    void* backend_private;
    uint64_t reserved[8];
} GafimeResultTableF64;

typedef struct GafimePermutationSignificanceTable {
    uint32_t abi_version;
    uint32_t metric_count;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    const float* observed_metric_values;
    float* p_values;
    uint64_t reserved[8];
} GafimePermutationSignificanceTable;

typedef struct GafimePermutationSignificanceTableF64 {
    uint32_t abi_version;
    uint32_t metric_count;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    const double* observed_metric_values;
    double* p_values;
    uint64_t reserved[8];
} GafimePermutationSignificanceTableF64;

/*
 * Optional post-selection precision diagnostic. `combo_indices` contains
 * row_count rows padded to max_arity with UINT32_MAX. Payloads report the
 * number of sample rows whose finite inputs overflow during centered
 * interaction materialization in the resident profile's pointwise dtype,
 * separately from source non-finite values.
 */
typedef struct GafimeInteractionDiagnosticBatch {
    uint32_t abi_version;
    uint32_t max_arity;
    uint64_t row_count;
    const uint32_t* combo_indices;
    uint64_t combo_index_count;
    uint64_t* overflow_row_counts;
    uint32_t* flags;
    uint32_t reserved32;
    uint64_t reserved[7];
} GafimeInteractionDiagnosticBatch;

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
 * Optional state-aware admission capability. Reports the peak device bytes for
 * the next execution, including live matrix/cache allocations and any
 * old-plus-new growth transition. The query must not mutate backend state.
 */
GAFIME_GPU_API int gafime_gpu_execution_memory_peak(
    GafimeGpuMatrix matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
);

/*
 * Optional CUDA significance admission capability. Reports the peak device
 * bytes for gafime_gpu_permutation_pvalues with selected_row_count surfaced
 * rows. The query must not mutate backend state. Callers with an active device
 * budget must use a budgeted fallback when this symbol is absent.
 */
GAFIME_GPU_API int gafime_gpu_permutation_memory_peak(
    GafimeGpuMatrix matrix,
    const GafimeLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
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

/*
 * Additive ABI 1.1 precision-profile surface. ABI 1.0 entry points above stay
 * byte-for-byte compatible. Current payloads expose the capability, typed
 * matrix, execution, and execution-peak symbols below for every advertised
 * profile. The permutation-peak and permutation-pvalue symbols are optional:
 * absence selects the documented Rust-orchestrated significance path. An older
 * payload may omit ABI 1.1 entirely and must then fail a profile request closed.
 */
GAFIME_GPU_API int gafime_gpu_precision_capabilities(
    uint32_t device_id,
    GafimePrecisionCapabilities* capabilities_out
);

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id,
    const GafimePrecisionMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
);

GAFIME_GPU_API int gafime_gpu_matrix_upload_f32_v2(
    GafimeGpuMatrix matrix,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
);

GAFIME_GPU_API int gafime_gpu_matrix_upload_f64_v2(
    GafimeGpuMatrix matrix,
    const double* features_host,
    const double* target_host,
    uint64_t rows,
    uint32_t cols
);

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f32_v2(
    GafimeGpuMatrix matrix,
    const float* target_host,
    uint64_t rows
);

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f64_v2(
    GafimeGpuMatrix matrix,
    const double* target_host,
    uint64_t rows
);

GAFIME_GPU_API int gafime_gpu_execute_f32_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result_out
);

GAFIME_GPU_API int gafime_gpu_execute_f64_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result_out
);

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
);

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
);

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_f32_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
);

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_f64_v2(
    GafimeGpuMatrix matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTableF64* significance_out
);

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics(
    GafimeGpuMatrix matrix,
    GafimeInteractionDiagnosticBatch* diagnostics
);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_GPU_ABI_HPP */
