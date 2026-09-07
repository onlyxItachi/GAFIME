// Host-only descriptor tests. These exercise the shared validation used by
// CUDA/HIP; passing them is not evidence of physical device execution.
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

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
    const auto check = [&]() {
        return gafime_semantic_abi::validate_capabilities(&caps, GAFIME_BACKEND_CUDA, 0);
    };
    int failed = expect(check(), GAFIME_STATUS_OK, "valid capabilities");
    caps.flags = UINT32_C(1) << 31;
    failed |= expect(check(), GAFIME_STATUS_OK, "ignorable capability hint");
    caps.flags = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "unknown required capability");
    caps.flags = 0;
    caps.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "old forecast draft rejected");
    caps.abi_version |= 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "reusable descriptor forecast rejected");
    caps.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION + 1;
    failed |= expect(check(), GAFIME_STATUS_OK, "future compatible semantic minor");
    caps.struct_size = gafime_semantic_abi::kCapabilitiesStablePrefixSize - 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "truncated capability prefix");
    caps.struct_size = sizeof(caps);
    caps.reserved[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "reserved capability field");
    caps.reserved[0] = 0;
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
            &batch, GAFIME_PRECISION_FP32, 2, 3, initialized);
    };
    failed |= expect(check(), GAFIME_STATUS_OK, "valid absolute-difference node");
    const std::vector<uint8_t> previously_written = {1, 1, 1};
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 2, 3, previously_written),
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
        &batch, GAFIME_PRECISION_FP32, 0, 2, gathered), GAFIME_STATUS_OK,
        "gathered accepted atom may feed a later program");
    failed |= expect(gafime_semantic_abi::validate_program_batch(
        &batch, GAFIME_PRECISION_FP32, 0, 2, absent), GAFIME_STATUS_INVALID_ARGUMENT,
        "ungathered accepted atom is not initialized");
    return failed;
}

int forecast_versions() {
    auto request = descriptor<GafimeSemanticForecastRequest>();
    request.program_max_operand_count = 2;
    request.program_operand_count = 7;
    request.program_mean_count = 4;
    const auto check = [&]() {
        return gafime_semantic_abi::validate_forecast_request(&request);
    };
    int failed = expect(check(), GAFIME_STATUS_OK, "immutable batch descriptor forecast");
    request.abi_version = (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "old max-span forecast rejected");
    request.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    request.struct_size = gafime_semantic_abi::kForecastRequestStablePrefixSize - 1;
    failed |= expect(check(), GAFIME_STATUS_ABI_MISMATCH, "truncated total-count prefix");
    request.struct_size = gafime_semantic_abi::kForecastRequestStablePrefixSize;
    failed |= expect(check(), GAFIME_STATUS_OK, "complete stable forecast prefix");
    request.struct_size = sizeof(request);
    request.reserved[0] = 1;
    failed |= expect(check(), GAFIME_STATUS_INVALID_ARGUMENT, "reserved forecast field");
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
static_assert(sizeof(GafimeSemanticForecastRequest) == 144);

int main() {
    return capabilities_and_versions() | bank_and_program_shapes() |
        forecast_versions() | gathering_and_outputs();
}
