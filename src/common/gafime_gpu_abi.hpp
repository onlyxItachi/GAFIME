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
  #define GAFIME_GPU_API __attribute__((visibility("default")))
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
/* Stable floor for the generic numeric-route structures and operations. */
#define GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR 1u

/*
 * ABI 1.1 extensibility rules.
 *
 * The high 16 bits of abi_version are the incompatible major version.  The
 * low 16 bits are additive.  Consumers accept a newer minor version when the
 * structure's known stable prefix is present.  Flag bits in the upper half of
 * a flags word are explicitly ignorable hints; unknown lower-half bits are
 * required semantics and fail closed.
 */
#define GAFIME_ABI_VERSION_MAJOR_OF(version) ((uint32_t)(version) >> 16)
#define GAFIME_ABI_VERSION_MINOR_OF(version) ((uint32_t)(version) & 0xffffu)
#define GAFIME_ABI_IGNORABLE_FLAG_MASK 0xffff0000u
#define GAFIME_ABI_REQUIRED_FLAG_MASK 0x0000ffffu

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
/* Legacy device flag. ABI 1.1 f64 support is authoritative in numeric routes. */
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

typedef enum GafimeOverflowPolicy {
    GAFIME_OVERFLOW_IEEE = 1
} GafimeOverflowPolicy;

#define GAFIME_NUMERIC_ROUTE_FP32 1u
#define GAFIME_NUMERIC_ROUTE_MIXED 2u
#define GAFIME_NUMERIC_ROUTE_FP64 3u

/* Current ABI calls accept caller-owned, contiguous host buffers only. */
#define GAFIME_BUFFER_FLAG_HOST 0x1u
#define GAFIME_BUFFER_FLAG_CONTIGUOUS 0x2u

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

/*
 * One authoritative numeric route.  The four dtype domains are a supported
 * combination, never four independently configurable knobs.  Future ABI 1.2
 * payloads may append fields and advertise additional route records.
 */
typedef struct GafimeNumericRoute {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t route_id;
    uint32_t profile;
    uint32_t storage_dtype;
    uint32_t pointwise_dtype;
    uint32_t reduction_dtype;
    uint32_t result_dtype;
    uint32_t overflow_policy;
    uint32_t flags;
    uint64_t reserved[8];
} GafimeNumericRoute;

typedef struct GafimeConstBufferView {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t flags;
    const void* data;
    uint64_t element_count;
    uint64_t byte_length;
    uint64_t byte_stride;
    uint64_t reserved[4];
} GafimeConstBufferView;

typedef struct GafimeMutableBufferView {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t flags;
    void* data;
    uint64_t element_capacity;
    uint64_t byte_length;
    uint64_t byte_stride;
    uint64_t reserved[4];
} GafimeMutableBufferView;

typedef struct GafimeNumericMatrixDesc {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    uint32_t layout;
    uint32_t flags;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
    uint64_t reserved[8];
} GafimeNumericMatrixDesc;

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

typedef struct GafimeNumericLaunchProtocol {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
} GafimeNumericLaunchProtocol;

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

typedef struct GafimeNumericResultTable {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    GafimeMutableBufferView metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    uint64_t reserved[8];
} GafimeNumericResultTable;

typedef struct GafimePermutationSignificanceTable {
    uint32_t abi_version;
    uint32_t metric_count;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    const float* observed_metric_values;
    float* p_values;
    uint64_t reserved[8];
} GafimePermutationSignificanceTable;

typedef struct GafimeNumericSignificanceTable {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t row_count;
    const uint64_t* candidate_ids;
    GafimeConstBufferView observed_metric_values;
    GafimeMutableBufferView p_values;
    uint64_t reserved[8];
} GafimeNumericSignificanceTable;

typedef struct GafimeNumericInteractionDiagnosticBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    uint32_t max_arity;
    uint32_t flags;
    uint64_t row_count;
    const uint32_t* combo_indices;
    uint64_t combo_index_count;
    uint64_t* overflow_row_counts;
    uint32_t* row_flags;
    uint32_t reserved32;
    uint64_t reserved[7];
} GafimeNumericInteractionDiagnosticBatch;

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

/*
 * Teardown is generation-neutral.  After validating the opaque owner, either
 * ABI generation's free symbol may release it; all non-free operations remain
 * generation-strict and reject an opposite-generation handle.
 */
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
 * Canonical ABI 1.1 numeric-route surface. These ten v2 declarations form one
 * complete operation table: a payload advertising numeric_routes_v2 must
 * export every one, and consumers reject a partial table before allocation.
 * ABI 1.0 entry points above stay byte-for-byte compatible and their optional
 * capability symbols retain their legacy semantics. Route records and all
 * typed buffers are caller-owned for the duration of a synchronous call.
 * `route_stride` is the byte distance between caller-owned records and must
 * contain the stable prefix through `flags`. A producer may report a
 * `struct_size` larger than that stride, but must never write beyond the
 * caller-provided stride. Passing routes_out == NULL with route_capacity == 0
 * performs a count query.
 */
GAFIME_GPU_API int gafime_gpu_numeric_routes_v2(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    uint32_t route_stride,
    GafimeNumericRoute* routes_out,
    uint32_t route_capacity,
    uint32_t* route_count_out
);

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id,
    const GafimeNumericMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
);

GAFIME_GPU_API int gafime_gpu_matrix_upload_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* features,
    const GafimeConstBufferView* target,
    uint64_t rows,
    uint32_t cols
);

GAFIME_GPU_API int gafime_gpu_matrix_update_target_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* target,
    uint64_t rows
);

GAFIME_GPU_API int gafime_gpu_execute_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericResultTable* result_out
);

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
);

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
);

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_v2(
    GafimeGpuMatrix matrix,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericSignificanceTable* significance_out
);

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics_v2(
    GafimeGpuMatrix matrix,
    GafimeNumericInteractionDiagnosticBatch* diagnostics
);

/* Accepts a validated ABI 1.0 or ABI 1.1 owner for cross-generation teardown. */
GAFIME_GPU_API int gafime_gpu_matrix_free_v2(GafimeGpuMatrix matrix);

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics(
    GafimeGpuMatrix matrix,
    GafimeInteractionDiagnosticBatch* diagnostics
);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_GPU_ABI_HPP */
