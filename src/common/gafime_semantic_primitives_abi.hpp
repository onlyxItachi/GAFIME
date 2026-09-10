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
 * bound the resident operand and mean arrays of this lowering.  Minor 3 adds
 * generic pairwise association and its exact fixed-NMI capability envelope.
 * It leaves the frozen matrix ABI untouched, but extends the program-node
 * element stride; v1.2 semantic consumers must fail negotiation before they
 * can interpret a v1.3 program array. */
#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR 3u
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
    GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT = 4,
    /* A closed hard-AND over frozen <=/> terms.  The terms remain typed
       physical arithmetic; candidate identity and fitting provenance stay in
       Rust. */
    GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION = 5
} GafimeSemanticProgramOp;

#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE 0x1u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE 0x2u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN 0x4u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT 0x8u
#define GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION 0x10u

typedef enum GafimeSemanticPrimitiveKind {
    GAFIME_SEMANTIC_PRIMITIVE_PAIRWISE_ASSOCIATION = 1,
    /* Source-compatible spelling for the v1.2 Pearson-only lowering. */
    GAFIME_SEMANTIC_PRIMITIVE_PAIRWISE_PEARSON =
        GAFIME_SEMANTIC_PRIMITIVE_PAIRWISE_ASSOCIATION,
    GAFIME_SEMANTIC_PRIMITIVE_ORDERED_EDGE_ENERGY = 2,
    GAFIME_SEMANTIC_PRIMITIVE_SPARSE_GATHER = 3,
    GAFIME_SEMANTIC_PRIMITIVE_COLUMN_MEANS = 4
} GafimeSemanticPrimitiveKind;

#define GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION 0x1u
/* Source-compatible spelling for the same physical primitive bit. */
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON \
    GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY 0x2u
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER 0x4u
#define GAFIME_SEMANTIC_PRIMITIVE_MASK_COLUMN_MEANS 0x8u

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

/* Association arithmetic is generic over resident physical slots.  Rust
 * chooses the statistic/presentation from its already-declared evidence
 * channel; native code neither receives nor owns evidence semantics. */
typedef enum GafimeSemanticAssociationStatistic {
    GAFIME_SEMANTIC_ASSOCIATION_PEARSON = 1,
    GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN = 2,
    GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI = 3
} GafimeSemanticAssociationStatistic;

typedef enum GafimeSemanticAssociationPresentation {
    GAFIME_SEMANTIC_ASSOCIATION_SIGNED = 1,
    GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE = 2,
    GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE = 3
} GafimeSemanticAssociationPresentation;

/* The bin mask is intentionally a capability bitset rather than an implicit
 * integer range: backends may decline a costly static histogram specialization
 * and callers fail closed for that exact requested bin count. */
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2 0x001u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_4 0x002u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_8 0x004u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_12 0x008u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_16 0x010u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_24 0x020u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_32 0x040u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_48 0x080u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_64 0x100u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_96 0x200u
#define GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_MASK_ALL 0x3ffu

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
    /* v1.2 reserved prefix: it remains zero and its offsets never move. */
    uint64_t reserved[8];
    /* v1.3 association/program tail.  Each statistic-specific limit is
     * explicit rather than hiding nonlinear rank work or histogram-counter
     * bounds behind generic `max_rows`. */
    uint32_t fixed_corrected_nmi_bin_mask;
    uint32_t max_region_terms;
    uint64_t max_association_pairs;
    uint64_t max_spearman_rows;
    uint64_t max_fixed_corrected_nmi_rows;
    uint64_t reserved_v3[5];
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
    /* v1.2 reserved prefix: it remains zero and its offsets never move. */
    uint64_t reserved[2];
    /* v1.3 region-term range.  Only FROZEN_REGION_CONJUNCTION consumes it;
     * every older operation requires both fields to be zero. */
    uint32_t region_term_offset;
    uint32_t region_term_count;
    uint64_t reserved_v3[2];
} GafimeSemanticProgramNode;

typedef enum GafimeSemanticRegionRelation {
    GAFIME_SEMANTIC_REGION_LESS_EQUAL = 1,
    GAFIME_SEMANTIC_REGION_GREATER_THAN = 2
} GafimeSemanticRegionRelation;

/* Frozen threshold bits follow the program profile: f32 bits are zero-extended
 * for fp32/mixed, f64 bits are raw for fp64.  Each term names only a physical
 * initialized input slot; source versus accepted-atom eligibility is Rust
 * policy and never crosses this boundary. */
typedef struct GafimeSemanticFrozenRegionTerm {
    uint32_t input_slot;
    uint32_t relation;
    uint64_t threshold_bits;
} GafimeSemanticFrozenRegionTerm;

typedef struct GafimeSemanticRegionTermSlice {
    const GafimeSemanticFrozenRegionTerm* ptr;
    uint64_t len;
} GafimeSemanticRegionTermSlice;

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
    /* v1.2 reserved prefix: it remains zero and its offsets never move. */
    uint64_t reserved[8];
    /* v1.3 immutable region descriptor storage. */
    GafimeSemanticRegionTermSlice region_terms;
    uint64_t reserved_v3[6];
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

/* Corresponding left/right slots form one association pair.  `presentation`
 * is arithmetic post-processing only: Pearson and Spearman allow signed or
 * absolute values; fixed corrected NMI requires nonnegative presentation.
 * `fixed_nmi_bins` is zero for Pearson/Spearman and one advertised exact bin
 * count for fixed corrected NMI. */
typedef struct GafimeSemanticAssociationBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t statistic;
    uint32_t presentation;
    uint32_t fixed_nmi_bins;
    uint32_t flags;
    GafimeSliceU32 left_slots;
    GafimeSliceU32 right_slots;
    uint64_t reserved[8];
} GafimeSemanticAssociationBatch;

/* One typed resident column per requested output.  A mean is defined for a
 * finite constant column; only empty or nonfinite inputs are unavailable.
 * Values/states/supports use the same route-typed scalar table as reductions. */
typedef struct GafimeSemanticColumnMeanBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t flags;
    uint32_t reserved32;
    GafimeSliceU32 candidate_slots;
    uint64_t reserved[8];
} GafimeSemanticColumnMeanBatch;

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
    /* v1.2 reserved prefix: it remains zero and its offsets never move. */
    uint64_t reserved[8];
    /* v1.3 exact counts for independently allocated transient descriptors. */
    uint64_t mean_slot_count;
    uint64_t program_region_term_count;
    uint64_t reserved_v3[6];
} GafimeSemanticForecastRequest;

typedef struct GafimeSemanticMemoryForecast {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t resident_bytes;
    uint64_t transient_bytes;
    uint64_t retained_bytes;
    uint64_t reserved[8];
} GafimeSemanticMemoryForecast;

/* The following thirteen symbols form one optional operation table.  A payload
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

/* v1.3 generic association entry.  The v1.2 Pearson entry above remains an
 * adapter so existing direct consumers retain their exact descriptor/symbol. */
GAFIME_GPU_API int gafime_gpu_semantic_pairwise_association_v1(
    GafimeGpuSemanticBank left_bank,
    GafimeGpuSemanticBank right_bank,
    const GafimeSemanticAssociationBatch* batch,
    GafimeSemanticScalarResultTable* results_out
);

GAFIME_GPU_API int gafime_gpu_semantic_column_means_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticColumnMeanBatch* batch,
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
