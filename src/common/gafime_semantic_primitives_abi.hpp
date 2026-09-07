/*
 * Optional GAFIME semantic-arithmetic GPU ABI.
 *
 * This is an additive operation table beside the frozen v1 / v1.1 matrix
 * ABI.  It intentionally contains only typed resident-column arithmetic and
 * reductions.  Feature identities, evidence identities, provenance, policy,
 * labels-as-optional-context, and selection remain Rust-owned.
 */

#ifndef GAFIME_SEMANTIC_PRIMITIVES_ABI_HPP
#define GAFIME_SEMANTIC_PRIMITIVES_ABI_HPP

#include "gafime_gpu_abi.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR 1u
/* Minor 2 makes program-descriptor forecast inputs exact for immutable,
 * batch-wide descriptor storage.  Minor 1's reusable maximum span cannot
 * bound the resident operand and mean arrays of this lowering. */
#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR 2u
#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION \
    ((GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | \
        GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR)

typedef void* GafimeGpuSemanticBank;

typedef enum GafimeSemanticProgramOp {
    /* A source feature is already resident in a source slot; this is a
       validated no-op that lets a program batch retain its complete DAG. */
    GAFIME_SEMANTIC_PROGRAM_SOURCE = 1,
    GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE = 2,
    GAFIME_SEMANTIC_PROGRAM_SOFTSIGN = 3,
    GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT = 4
} GafimeSemanticProgramOp;

#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE 0x1u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE 0x2u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN 0x4u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT 0x8u

typedef enum GafimeSemanticPrimitiveKind {
    GAFIME_SEMANTIC_PRIMITIVE_PAIRWISE_PEARSON = 1,
    GAFIME_SEMANTIC_PRIMITIVE_ORDERED_EDGE_ENERGY = 2,
    GAFIME_SEMANTIC_PRIMITIVE_SPARSE_GATHER = 3
} GafimeSemanticPrimitiveKind;

#define GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON 0x1u
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY 0x2u
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER 0x4u

/* Association statistics are negotiated independently from generic operand
 * primitives.  A GPU may expose Pearson arithmetic while explicitly declining
 * rank- or histogram-based statistics; callers must not substitute Core work
 * behind an explicit GPU selection. */
#define GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON 0x1u
#define GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN 0x2u
#define GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI 0x4u

typedef enum GafimeSemanticPearsonMode {
    GAFIME_SEMANTIC_PEARSON_SIGNED = 1,
    GAFIME_SEMANTIC_PEARSON_ABSOLUTE = 2
} GafimeSemanticPearsonMode;

/* Native result states describe arithmetic definedness only.  Rust maps these
 * to its evidence vocabulary and applies missingness policy. */
typedef enum GafimeSemanticScalarState {
    GAFIME_SEMANTIC_SCALAR_MEASURED = 1,
    GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT = 2,
    GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND = 3,
    GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION = 4,
    GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION = 5
} GafimeSemanticScalarState;

/* All capacities are physical slot counts.  No semantic FeatureId enters this
 * ABI: Rust maps its IDs to these positions before every call. */
typedef struct GafimeSemanticCapabilities {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend_kind;
    uint32_t device_id;
    uint32_t profile_mask;
    uint32_t program_op_mask;
    uint32_t primitive_mask;
    uint32_t association_statistic_mask;
    uint32_t flags;
    uint32_t max_program_nodes;
    uint32_t max_slot_count;
    uint64_t max_rows;
    uint64_t max_gather_rows;
    uint64_t reserved[8];
} GafimeSemanticCapabilities;

/* Columns are stored column-major in a typed resident bank.  `source_slots`
 * are populated by upload; later program nodes may populate any remaining
 * slots up through `slot_capacity`. */
typedef struct GafimeSemanticBankDesc {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    uint32_t layout;
    uint32_t flags;
    uint64_t rows;
    uint32_t source_slots;
    uint32_t slot_capacity;
    uint64_t bytes;
    uint64_t reserved[8];
} GafimeSemanticBankDesc;

typedef struct GafimeSemanticProgramNode {
    uint32_t opcode;
    uint32_t output_slot;
    uint32_t operand_offset;
    uint32_t operand_count;
    uint32_t mean_offset;
    uint32_t mean_count;
    uint64_t reserved[2];
} GafimeSemanticProgramNode;

/* `operand_slots` and `mean_bits` are one contiguous program descriptor.
 * `mean_bits` contains f32 bit patterns zero-extended for fp32/mixed and raw
 * f64 bits for fp64.  The implementation never recomputes frozen means. */
typedef struct GafimeSemanticProgramBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    const GafimeSemanticProgramNode* nodes;
    uint32_t node_count;
    uint32_t reserved32;
    GafimeSliceU32 operand_slots;
    GafimeSliceU64 mean_bits;
    uint64_t reserved[8];
} GafimeSemanticProgramBatch;

/* Corresponding entries in left_slots and right_slots form one generic
 * arithmetic pair.  A caller repeats a reference slot when comparing many
 * candidates with one reference; this remains a physical lowering, not a
 * target protocol. */
typedef struct GafimeSemanticPearsonBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t mode;
    uint32_t flags;
    GafimeSliceU32 left_slots;
    GafimeSliceU32 right_slots;
    uint64_t reserved[8];
} GafimeSemanticPearsonBatch;

typedef struct GafimeSemanticEdge {
    uint64_t left_row;
    uint64_t right_row;
} GafimeSemanticEdge;

/* The declared edge order is the reduction order.  Weights use the bank's
 * storage dtype, so fp32 stays fp32 while mixed widens only for reduction. */
typedef struct GafimeSemanticEdgeEnergyBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t flags;
    uint32_t reserved32;
    const GafimeSemanticEdge* edges;
    uint64_t edge_count;
    GafimeConstBufferView weights;
    GafimeSliceU32 candidate_slots;
    uint64_t reserved[8];
} GafimeSemanticEdgeEnergyBatch;

/* Gathers selected source rows into the corresponding destination slots in a
 * separate resident bank.  This is sufficient to form an optional-label
 * subset without treating labels as a native target. */
typedef struct GafimeSemanticSparseGatherBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t flags;
    uint32_t reserved32;
    GafimeSliceU32 source_slots;
    GafimeSliceU32 destination_slots;
    GafimeSliceU64 row_indices;
    uint64_t reserved[8];
} GafimeSemanticSparseGatherBatch;

/* Values, states, and supports are caller-owned result buffers.  Values use
 * `route.result_dtype`; state/support arrays contain one element per result.
 */
typedef struct GafimeSemanticScalarResultTable {
    uint32_t abi_version;
    uint32_t struct_size;
    GafimeNumericRoute route;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t capacity;
    uint64_t count;
    GafimeMutableBufferView values;
    uint32_t* states;
    uint64_t* supports;
    uint64_t reserved[8];
} GafimeSemanticScalarResultTable;

typedef struct GafimeSemanticForecastRequest {
    uint32_t abi_version;
    uint32_t struct_size;
    /* Preserved from semantic ABI 1.1: largest operand span in one node.
       It remains descriptive but does not size the immutable batch buffers. */
    uint64_t program_max_operand_count;
    uint64_t pair_count;
    uint64_t graph_candidate_count;
    uint64_t graph_edge_count;
    uint64_t gather_slot_count;
    uint64_t gather_row_count;
    uint64_t retained_slot_count;
    /* Exact flattened descriptor lengths for one immutable program batch.
       `program_operand_count` counts u32 physical slots and
       `program_mean_count` counts u64 frozen-mean bit patterns.  These are
       intentionally distinct: only centered products contribute means. */
    uint64_t program_operand_count;
    uint64_t program_mean_count;
    uint64_t reserved[8];
} GafimeSemanticForecastRequest;

typedef struct GafimeSemanticMemoryForecast {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t resident_bytes;
    uint64_t transient_bytes;
    uint64_t retained_bytes;
    uint64_t reserved[8];
} GafimeSemanticMemoryForecast;

/* The following eleven symbols form one optional operation table.  A payload
 * exporting any one must export all of them; consumers reject partial tables
 * and old payloads simply report semantic lowering unavailable. */
GAFIME_GPU_API int gafime_gpu_semantic_capabilities_v1(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    GafimeSemanticCapabilities* capabilities_out
);

GAFIME_GPU_API int gafime_gpu_semantic_bank_alloc_v1(
    uint32_t device_id,
    const GafimeSemanticBankDesc* desc,
    GafimeGpuSemanticBank* bank_out
);

GAFIME_GPU_API int gafime_gpu_semantic_bank_upload_v1(
    GafimeGpuSemanticBank bank,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* source_columns
);

GAFIME_GPU_API int gafime_gpu_semantic_materialize_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticProgramBatch* batch
);

GAFIME_GPU_API int gafime_gpu_semantic_pairwise_pearson_v1(
    GafimeGpuSemanticBank left_bank,
    GafimeGpuSemanticBank right_bank,
    const GafimeSemanticPearsonBatch* batch,
    GafimeSemanticScalarResultTable* results_out
);

GAFIME_GPU_API int gafime_gpu_semantic_ordered_edge_energy_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticEdgeEnergyBatch* batch,
    GafimeSemanticScalarResultTable* results_out
);

GAFIME_GPU_API int gafime_gpu_semantic_sparse_gather_v1(
    GafimeGpuSemanticBank source_bank,
    GafimeGpuSemanticBank destination_bank,
    const GafimeSemanticSparseGatherBatch* batch
);

GAFIME_GPU_API int gafime_gpu_semantic_forecast_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticForecastRequest* request,
    GafimeSemanticMemoryForecast* forecast_out
);

// On a non-OK return this output is normally null.  If copying fails after a
// retained bank was allocated and native cleanup also fails, it remains
// non-null as free-only caller ownership: do not dispatch it, call the free
// function below for best-effort release or diagnostics.
GAFIME_GPU_API int gafime_gpu_semantic_bank_retain_v1(
    GafimeGpuSemanticBank source_bank,
    GafimeSliceU32 slots,
    GafimeGpuSemanticBank* retained_bank_out
);

GAFIME_GPU_API int gafime_gpu_semantic_bank_download_v1(
    GafimeGpuSemanticBank bank,
    GafimeSliceU32 slots,
    const GafimeNumericRoute* route,
    GafimeMutableBufferView* columns_out
);

// Returns the native cleanup status.  A failed release preserves caller
// ownership of the non-null handle so a direct caller can retry or diagnose;
// safe Rust Drop can only make its best-effort attempt.
GAFIME_GPU_API int gafime_gpu_semantic_bank_free_v1(GafimeGpuSemanticBank bank);

#ifdef __cplusplus
}
#endif

#endif /* GAFIME_SEMANTIC_PRIMITIVES_ABI_HPP */
