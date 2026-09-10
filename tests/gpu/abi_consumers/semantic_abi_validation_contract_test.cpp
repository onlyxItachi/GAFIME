// Host-only descriptor tests. These exercise the shared validation used by
// CUDA/HIP; passing them is not evidence of physical device execution.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "../../../src/common/semantic_primitives_abi_impl.hpp"

namespace {

template <typename T>
T descriptor() {
    T result{};
    result.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    result.struct_size = sizeof(result);
    return result;
}

int expect(int actual, int expected, const char* label) {
    if (actual == expected) return 0;
    std::fprintf(stderr, "%s: expected %d, got %d\n", label, expected, actual);
    return 1;
}

int capabilities_and_versions() {
    auto caps = descriptor<GafimeSemanticCapabilities>();
    caps.backend_kind = GAFIME_BACKEND_CUDA;
    caps.profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32;
    caps.program_op_mask = GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE;
    caps.primitive_mask = GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON;
    caps.association_statistic_mask = GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON;
    caps.max_program_nodes = 8;
    caps.max_slot_count = 8;
    caps.max_rows = 32;
    caps.max_gather_rows = 32;
    caps.max_association_pairs = 8;
    const auto check = [&]() {
        return gafime_semantic_abi::validate_capabilities(&caps, GAFIME_BACKEND_CUDA, 0);
    };
    int failed = expect(check(), GAFIME_STATUS_OK, "valid capabilities");
    caps.flags = UINT32_C(1) << 31;
    failed |= expect(check(), GAFIME_STATUS_OK, "ignorable capability hint");
    caps.flags = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "unknown required capability");
    caps.flags = 0;
    caps.abi_version = (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 2u;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH,
        "v1.2 consumer capability record rejected");
    caps.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION + 1;
    failed |= expect(check(), GAFIME_STATUS_OK, "future compatible semantic minor");
    caps.struct_size = gafime_semantic_abi::kCapabilitiesV13StablePrefixSize - 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "truncated capability prefix");
    caps.struct_size = sizeof(caps);
    caps.reserved[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "reserved capability field");
    caps.reserved[0] = 0;
    caps.reserved_v3[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "v1.3 reserved capability field");
    caps.reserved_v3[0] = 0;
    caps.max_program_nodes = 0;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "zero program capacity");
    return failed;
}

int bank_and_program_shapes() {
    auto bank = descriptor<GafimeSemanticBankDesc>();
    bank.route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    bank.layout = GAFIME_MATRIX_COLUMN_MAJOR;
    bank.rows = 4;
    bank.source_slots = 2;
    bank.slot_capacity = 3;
    bank.bytes = 4 * 3 * sizeof(float);
    int failed = expect(gafime_semantic_abi::validate_bank_desc(&bank),
        GAFIME_STATUS_OK, "valid physical bank");
    bank.bytes -= 1;
    failed |= expect(gafime_semantic_abi::validate_bank_desc(&bank),
        GAFIME_STATUS_INVALID_ARGUMENT, "undersized bank storage");
    bank.rows = std::numeric_limits<uint64_t>::max();
    failed |= expect(gafime_semantic_abi::validate_bank_desc(&bank),
        GAFIME_STATUS_INVALID_ARGUMENT, "bank byte-count overflow");

    uint32_t slots[] = {0, 1};
    GafimeSemanticProgramNode node{};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE;
    node.output_slot = 2;
    node.operand_count = 2;
    auto batch = descriptor<GafimeSemanticProgramBatch>();
    batch.route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    batch.nodes = &node;
    batch.node_count = 1;
    batch.operand_slots = {slots, 2};
    const std::vector<uint8_t> initialized = {1, 1, 0};
    const auto check = [&]() {
        return gafime_semantic_abi::validate_program_batch(
            &batch, GAFIME_PRECISION_FP32, 2, 3, initialized,
            gafime_semantic_abi::kSemanticMaxRegionTerms);
    };
    failed |= expect(check(), GAFIME_STATUS_OK, "valid absolute-difference node");
    batch.struct_size = gafime_semantic_abi::kProgramBatchV13StablePrefixSize;
    batch.reserved_v3[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_OK,
        "program stable prefix does not expose its absent reserved tail");
    // Allocate only the promised prefix. ASan catches a validator that reads
    // the physically absent tail even if ordinary fixtures happen to have it.
    std::unique_ptr<void, decltype(&std::free)> prefix(
        std::malloc(batch.struct_size), &std::free);
    if (!prefix) return failed | 1;
    std::memcpy(prefix.get(), &batch, batch.struct_size);
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        static_cast<const GafimeSemanticProgramBatch*>(prefix.get()),
        GAFIME_PRECISION_FP32, 2, 3, initialized,
        gafime_semantic_abi::kSemanticMaxRegionTerms), GAFIME_STATUS_OK,
        "allocated exact program prefix is readable without a tail");
    batch.struct_size = sizeof(batch);
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT,
        "present program reserved tail still must be zero");
    batch.reserved_v3[0] = 0;
    batch.struct_size = gafime_semantic_abi::kProgramBatchV13StablePrefixSize - 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH,
        "truncated program stable prefix fails closed");
    batch.struct_size = sizeof(batch);
    batch.abi_version = (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 2u;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH,
        "v1.2 program descriptor rejected before node stride access");
    batch.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    const std::vector<uint8_t> previously_written = {1, 1, 1};
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 2, 3, previously_written,
        gafime_semantic_abi::kSemanticMaxRegionTerms),
        GAFIME_STATUS_INVALID_ARGUMENT, "derived slots cannot overwrite a prior valid value");
    const GafimeSemanticProgramNode repeated_nodes[] = {node, node};
    batch.nodes = repeated_nodes;
    batch.node_count = 2;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT,
        "derived slots cannot be written twice in one batch");
    batch.nodes = &node;
    batch.node_count = 1;
    node.output_slot = 0;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "program cannot overwrite input");
    node.output_slot = 2;
    node.operand_offset = std::numeric_limits<uint32_t>::max();
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "operand offset overflow");
    node.operand_offset = 0;
    slots[1] = 3;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "operand outside bank");
    slots[1] = 2;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "uninitialized operand");
    slots[1] = 1;
    uint64_t means[] = {0, UINT64_C(1) << 32};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT;
    node.mean_count = 2;
    batch.mean_bits = {means, 2};
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "fp32 mean upper bits");

    // Gathered accepted values are initialized even when not upload-source
    // slots. Their use must be legal, without allowing an uninitialized atom.
    const std::vector<uint8_t> gathered = {1, 0};
    const std::vector<uint8_t> absent = {0, 0};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_SOFTSIGN;
    node.output_slot = 1;
    node.operand_count = 1;
    node.mean_count = 0;
    batch.operand_slots.len = 1;
    batch.mean_bits = {nullptr, 0};
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 0, 2, gathered,
        gafime_semantic_abi::kSemanticMaxRegionTerms), GAFIME_STATUS_OK,
        "gathered accepted atom may feed a later program");
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 0, 2, absent,
        gafime_semantic_abi::kSemanticMaxRegionTerms), GAFIME_STATUS_INVALID_ARGUMENT,
        "ungathered accepted atom is not initialized");

    GafimeSemanticFrozenRegionTerm term{};
    term.input_slot = 0;
    term.relation = GAFIME_SEMANTIC_REGION_LESS_EQUAL;
    term.threshold_bits = UINT32_C(0x3f800000);
    node = {};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    node.output_slot = 1;
    node.region_term_count = 1;
    batch.nodes = &node;
    batch.node_count = 1;
    batch.operand_slots = {nullptr, 0};
    batch.mean_bits = {nullptr, 0};
    batch.region_terms = {&term, 1};
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 0, 2, gathered,
        gafime_semantic_abi::kSemanticMaxRegionTerms), GAFIME_STATUS_OK,
        "valid frozen region physical term");
    term.threshold_bits = UINT32_C(0x7fc00000);
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 0, 2, gathered,
        gafime_semantic_abi::kSemanticMaxRegionTerms), GAFIME_STATUS_INVALID_ARGUMENT,
        "nonfinite frozen region threshold");
    return failed;
}

int forecast_versions() {
    auto request = descriptor<GafimeSemanticForecastRequest>();
    request.program_max_operand_count = 2;
    request.program_operand_count = 7;
    request.program_mean_count = 4;
    request.mean_slot_count = 3;
    request.program_region_term_count = 2;
    const auto check = [&]() {
        return gafime_semantic_abi::validate_forecast_request(&request);
    };
    int failed = expect(check(), GAFIME_STATUS_OK, "immutable batch descriptor forecast");
    request.abi_version = (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 2u;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "v1.2 forecast rejected");
    request.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    request.struct_size = gafime_semantic_abi::kForecastRequestV13StablePrefixSize - 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "truncated total-count prefix");
    request.struct_size = gafime_semantic_abi::kForecastRequestV13StablePrefixSize;
    failed |= expect(check(), GAFIME_STATUS_OK, "complete stable forecast prefix");
    request.struct_size = sizeof(request);
    request.reserved[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "reserved forecast field");
    request.reserved[0] = 0;
    request.reserved_v3[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "v1.3 reserved forecast field");
    return failed;
}

int association_and_mean_shapes() {
    uint32_t left[] = {0, 1};
    uint32_t right[] = {1, 0};
    auto association = descriptor<GafimeSemanticAssociationBatch>();
    association.statistic = GAFIME_SEMANTIC_ASSOCIATION_PEARSON;
    association.presentation = GAFIME_SEMANTIC_ASSOCIATION_SIGNED;
    association.left_slots = {left, 2};
    association.right_slots = {right, 2};
    const auto check_association = [&]() {
        return gafime_semantic_abi::validate_association_batch(
            &association, 2, 2,
            GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON |
                GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN |
                GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI,
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2, 2, 8, 8, 4);
    };
    int failed = expect(check_association(), GAFIME_STATUS_OK, "valid generic Pearson");
    association.statistic = GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI;
    association.fixed_nmi_bins = 2;
    association.presentation = GAFIME_SEMANTIC_ASSOCIATION_SIGNED;
    failed |= expect(check_association(), GAFIME_STATUS_INVALID_ARGUMENT,
        "fixed NMI signed presentation rejected before work");
    association.presentation = GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE;
    association.fixed_nmi_bins = 4;
    failed |= expect(check_association(), GAFIME_STATUS_UNSUPPORTED_BACKEND,
        "unsupported fixed NMI bins reject before work");
    association.fixed_nmi_bins = 2;
    failed |= expect(check_association(), GAFIME_STATUS_OK, "valid fixed NMI descriptor");
    association.abi_version = (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 2u;
    failed |= expect(check_association(), GAFIME_STATUS_ABI_MISMATCH,
        "v1.2 association descriptor rejected");

    uint32_t candidates[] = {0, 1};
    auto means = descriptor<GafimeSemanticColumnMeanBatch>();
    means.candidate_slots = {candidates, 2};
    failed |= expect(gafime_semantic_abi::validate_column_mean_batch(&means, 2),
        GAFIME_STATUS_OK, "valid ordered mean slots");
    candidates[1] = 0;
    failed |= expect(gafime_semantic_abi::validate_column_mean_batch(&means, 2),
        GAFIME_STATUS_INVALID_ARGUMENT, "duplicate ordered mean slot");
    return failed;
}

int gathering_and_outputs() {
    uint32_t sources[] = {0};
    uint32_t destinations[] = {1};
    uint64_t rows[] = {3, 0};
    auto gather = descriptor<GafimeSemanticSparseGatherBatch>();
    gather.source_slots = {sources, 1};
    gather.destination_slots = {destinations, 1};
    gather.row_indices = {rows, 2};
    const auto check = [&]() {
        return gafime_semantic_abi::validate_gather_batch(&gather, 4, 2, 2, 2);
    };
    int failed = expect(check(), GAFIME_STATUS_OK, "valid row permutation");
    rows[0] = 4;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "row outside source");
    rows[0] = 3;
    gather.destination_slots.len = 0;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "mismatched gather slots");
    uint32_t duplicate_sources[] = {0, 1};
    uint32_t duplicate_destinations[] = {1, 1};
    auto duplicate_gather = descriptor<GafimeSemanticSparseGatherBatch>();
    duplicate_gather.source_slots = {duplicate_sources, 2};
    duplicate_gather.destination_slots = {duplicate_destinations, 2};
    duplicate_gather.row_indices = {rows, 2};
    failed |= expect(gafime_semantic_abi::validate_gather_batch(
        &duplicate_gather, 4, 2, 2, 2), GAFIME_STATUS_INVALID_ARGUMENT,
        "duplicate gather destination slots");

    auto results = descriptor<GafimeSemanticScalarResultTable>();
    results.route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    results.capacity = 1;
    float value = 0;
    uint32_t state = 0;
    uint64_t support = 0;
    results.values.abi_version = GAFIME_PRECISION_ABI_VERSION;
    results.values.struct_size = sizeof(results.values);
    results.values.dtype = GAFIME_DTYPE_F32;
    results.values.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    results.values.data = &value;
    results.values.element_capacity = 1;
    results.values.byte_length = sizeof(value);
    results.values.byte_stride = sizeof(value);
    results.states = &state;
    results.supports = &support;
    failed |= expect(gafime_semantic_abi::validate_scalar_results(&results, results.route, 1),
        GAFIME_STATUS_OK, "valid typed scalar output");
    results.count = 1;
    failed |= expect(gafime_semantic_abi::validate_scalar_results(&results, results.route, 1),
        GAFIME_STATUS_INVALID_ARGUMENT, "output count must start empty");
    results.count = 0;
    results.supports = nullptr;
    failed |= expect(gafime_semantic_abi::validate_scalar_results(&results, results.route, 1),
        GAFIME_STATUS_INVALID_ARGUMENT, "missing support buffer");
    return failed;
}

}  // namespace

static_assert(std::is_same_v<decltype(&gafime_gpu_semantic_bank_free_v1),
    int (*)(GafimeGpuSemanticBank)>, "semantic free reports native status");
static_assert(offsetof(GafimeSemanticForecastRequest, pair_count) == 16);
static_assert(offsetof(GafimeSemanticForecastRequest, retained_slot_count) == 56);
static_assert(offsetof(GafimeSemanticForecastRequest, program_operand_count) == 64);
static_assert(offsetof(GafimeSemanticForecastRequest, program_mean_count) == 72);
static_assert(offsetof(GafimeSemanticForecastRequest, reserved) == 80);
static_assert(offsetof(GafimeSemanticForecastRequest, mean_slot_count) == 144);
static_assert(offsetof(GafimeSemanticForecastRequest, program_region_term_count) == 152);
static_assert(offsetof(GafimeSemanticForecastRequest, reserved_v3) == 160);
static_assert(sizeof(GafimeSemanticForecastRequest) == 208);
static_assert(offsetof(GafimeSemanticProgramNode, region_term_offset) == 40);
static_assert(offsetof(GafimeSemanticProgramNode, reserved_v3) == 48);
static_assert(sizeof(GafimeSemanticProgramNode) == 64);
static_assert(offsetof(GafimeSemanticProgramBatch, region_terms) == 224);
static_assert(sizeof(GafimeSemanticProgramBatch) == 288);
static_assert(offsetof(GafimeSemanticCapabilities, fixed_corrected_nmi_bin_mask) == 128);
static_assert(offsetof(GafimeSemanticCapabilities, max_association_pairs) == 136);
static_assert(sizeof(GafimeSemanticCapabilities) == 200);
static_assert(offsetof(GafimeSemanticAssociationBatch, left_slots) == 24);
static_assert(sizeof(GafimeSemanticAssociationBatch) == 120);
static_assert(offsetof(GafimeSemanticColumnMeanBatch, candidate_slots) == 16);
static_assert(sizeof(GafimeSemanticColumnMeanBatch) == 96);

int main() {
    return capabilities_and_versions() | bank_and_program_shapes() |
        forecast_versions() | association_and_mean_shapes() | gathering_and_outputs();
}
