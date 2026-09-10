// Direct physical-Metal exercise of the optional semantic arithmetic table.
// This deliberately calls the payload symbols rather than a Rust/Python
// fallback: a passing CTest means the resident Metal bank, shaders, and result
// downloads all executed on the selected Apple device.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../src/common/semantic_primitives_abi_impl.hpp"

namespace {

constexpr uint64_t kRows = 256;
constexpr uint32_t kSourceSlots = 3;
constexpr uint32_t kSlotCapacity = 8;

int fail(const char* message, int status = GAFIME_STATUS_OK) {
    if (status == GAFIME_STATUS_OK) {
        std::fprintf(stderr, "metal semantic native test: %s\n", message);
    } else {
        std::fprintf(stderr, "metal semantic native test: %s (status %d)\n", message, status);
    }
    return 1;
}

bool near(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::fabs(actual - expected) <= tolerance;
}

uint64_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

GafimeConstBufferView const_f32_view(const std::vector<float>& values) {
    GafimeConstBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_count = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

GafimeMutableBufferView mutable_f32_view(std::vector<float>& values) {
    GafimeMutableBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_capacity = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

struct ScalarResults {
    std::vector<float> values;
    std::vector<uint32_t> states;
    std::vector<uint64_t> supports;
    GafimeSemanticScalarResultTable table{};

    ScalarResults(const GafimeNumericRoute& route, uint64_t count)
        : values(count), states(count), supports(count) {
        table.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
        table.struct_size = sizeof(table);
        table.route = route;
        table.capacity = count;
        table.values = mutable_f32_view(values);
        table.states = states.data();
        table.supports = supports.data();
    }
};

GafimeSemanticBankDesc bank_desc(
    const GafimeNumericRoute& route,
    uint64_t rows,
    uint32_t source_slots,
    uint32_t slot_capacity
) {
    GafimeSemanticBankDesc desc{};
    desc.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = route;
    desc.layout = GAFIME_MATRIX_COLUMN_MAJOR;
    desc.rows = rows;
    desc.source_slots = source_slots;
    desc.slot_capacity = slot_capacity;
    desc.bytes = rows * static_cast<uint64_t>(slot_capacity) * sizeof(float);
    return desc;
}

int check_capabilities(const GafimeNumericRoute& fp32) {
    GafimeSemanticCapabilities capabilities{};
    int status = gafime_gpu_semantic_capabilities_v1(
        0, GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION, &capabilities);
    if (status != GAFIME_STATUS_OK) return fail("semantic capabilities were unavailable", status);
    if (capabilities.abi_version != GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION ||
        capabilities.backend_kind != GAFIME_BACKEND_METAL ||
        capabilities.profile_mask != GAFIME_PRECISION_PROFILE_MASK_FP32 ||
        capabilities.program_op_mask !=
            (GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE |
             GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE |
             GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN |
             GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT |
             GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION) ||
        capabilities.primitive_mask !=
            (GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION |
             GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY |
             GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER |
             GAFIME_SEMANTIC_PRIMITIVE_MASK_COLUMN_MEANS) ||
        capabilities.association_statistic_mask !=
            (GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON |
             GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN |
             GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI) ||
        capabilities.fixed_corrected_nmi_bin_mask !=
            (GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_4 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_8 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_12 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_16 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_24 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_32 |
             GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_48) ||
        capabilities.max_region_terms != gafime_semantic_abi::kSemanticMaxRegionTerms ||
        capabilities.max_association_pairs != 65'536 ||
        capabilities.max_spearman_rows != 32'768 ||
        capabilities.max_fixed_corrected_nmi_rows != 32'768) {
        return fail("semantic capability envelope was not the declared fp32-only Metal envelope");
    }
    status = gafime_gpu_semantic_capabilities_v1(
        0,
        (GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR << 16) | 2u,
        &capabilities);
    if (status != GAFIME_STATUS_ABI_MISMATCH) {
        return fail("semantic ABI 1.2 capability consumer did not fail before descriptor use", status);
    }

    GafimeGpuSemanticBank rejected = nullptr;
    auto mixed = bank_desc(gafime_gpu_abi::numeric_route(GAFIME_PRECISION_MIXED), 4, 1, 1);
    status = gafime_gpu_semantic_bank_alloc_v1(0, &mixed, &rejected);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || rejected != nullptr) {
        return fail("mixed semantic bank did not fail closed on Metal", status);
    }
    auto fp64 = bank_desc(gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP64), 4, 1, 1);
    fp64.bytes = 4 * sizeof(double);
    status = gafime_gpu_semantic_bank_alloc_v1(0, &fp64, &rejected);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || rejected != nullptr) {
        return fail("fp64 semantic bank did not fail closed on Metal", status);
    }
    (void)fp32;
    return 0;
}

int check_tied_spearman_and_largest_nmi_template(const GafimeNumericRoute& route) {
    // This small orthogonal tie pattern has exact average-tie Spearman zero.
    // It catches an implementation that merely preserves sort order within a
    // tie instead of assigning the shared integer rank position.
    constexpr uint64_t kTieRows = 4;
    const std::vector<float> tied_columns = {
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
    };
    auto tied_desc = bank_desc(route, kTieRows, 2, 2);
    GafimeGpuSemanticBank tied_bank = nullptr;
    int status = gafime_gpu_semantic_bank_alloc_v1(0, &tied_desc, &tied_bank);
    if (status != GAFIME_STATUS_OK || tied_bank == nullptr) {
        return fail("tied-rank semantic bank allocation failed", status);
    }
    const GafimeConstBufferView tied_view = const_f32_view(tied_columns);
    status = gafime_gpu_semantic_bank_upload_v1(tied_bank, &route, &tied_view);
    if (status != GAFIME_STATUS_OK) return fail("tied-rank semantic upload failed", status);
    const uint32_t left_slot[] = {0};
    const uint32_t right_slot[] = {1};
    GafimeSemanticAssociationBatch spearman{};
    spearman.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    spearman.struct_size = sizeof(spearman);
    spearman.statistic = GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN;
    spearman.presentation = GAFIME_SEMANTIC_ASSOCIATION_SIGNED;
    spearman.left_slots = {left_slot, 1};
    spearman.right_slots = {right_slot, 1};
    ScalarResults tie_results(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(
        tied_bank, tied_bank, &spearman, &tie_results.table);
    if (status != GAFIME_STATUS_OK ||
        tie_results.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        tie_results.supports[0] != kTieRows || !near(tie_results.values[0], 0.0f)) {
        return fail("parallel tied Spearman ranks did not preserve average-tie semantics", status);
    }
    status = gafime_gpu_semantic_bank_free_v1(tied_bank);
    if (status != GAFIME_STATUS_OK) return fail("tied-rank semantic bank release failed", status);

    // The capability mask advertises 48 as the largest real Metal histogram.
    // Execute that exact template at its declared minimum support instead of
    // treating a mask-only assertion as evidence that the threadgroup storage
    // and integer histogram launch are usable on Apple hardware.
    constexpr uint64_t kLargestNmiBins = 48;
    constexpr uint64_t kLargestNmiRows = 8 * kLargestNmiBins * kLargestNmiBins;
    std::vector<float> largest_nmi_columns(kLargestNmiRows * 2);
    for (uint64_t row = 0; row < kLargestNmiRows; ++row) {
        largest_nmi_columns[row] = static_cast<float>(row % kLargestNmiBins);
        largest_nmi_columns[kLargestNmiRows + row] =
            static_cast<float>((row * 7) % kLargestNmiBins);
    }
    auto largest_nmi_desc = bank_desc(route, kLargestNmiRows, 2, 2);
    GafimeGpuSemanticBank largest_nmi_bank = nullptr;
    status = gafime_gpu_semantic_bank_alloc_v1(0, &largest_nmi_desc, &largest_nmi_bank);
    if (status != GAFIME_STATUS_OK || largest_nmi_bank == nullptr) {
        return fail("largest-template NMI semantic bank allocation failed", status);
    }
    const GafimeConstBufferView largest_nmi_view = const_f32_view(largest_nmi_columns);
    status = gafime_gpu_semantic_bank_upload_v1(largest_nmi_bank, &route, &largest_nmi_view);
    if (status != GAFIME_STATUS_OK) return fail("largest-template NMI semantic upload failed", status);
    GafimeSemanticAssociationBatch nmi{};
    nmi.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    nmi.struct_size = sizeof(nmi);
    nmi.statistic = GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI;
    nmi.presentation = GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE;
    nmi.fixed_nmi_bins = static_cast<uint32_t>(kLargestNmiBins);
    nmi.left_slots = {left_slot, 1};
    nmi.right_slots = {right_slot, 1};
    ScalarResults largest_nmi_results(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(
        largest_nmi_bank, largest_nmi_bank, &nmi, &largest_nmi_results.table);
    if (status != GAFIME_STATUS_OK ||
        largest_nmi_results.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        largest_nmi_results.supports[0] != kLargestNmiRows ||
        !std::isfinite(largest_nmi_results.values[0]) || largest_nmi_results.values[0] < 0.0f) {
        return fail("largest advertised fixed-NMI template was not measured", status);
    }
    nmi.fixed_nmi_bins = 64;
    ScalarResults unsupported_nmi_results(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(
        largest_nmi_bank, largest_nmi_bank, &nmi, &unsupported_nmi_results.table);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
        return fail("unsupported 64-bin NMI request was silently coerced", status);
    }
    status = gafime_gpu_semantic_bank_free_v1(largest_nmi_bank);
    if (status != GAFIME_STATUS_OK) {
        return fail("largest-template NMI semantic bank release failed", status);
    }
    return 0;
}

int run_native_semantic_lifecycle() {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    if (const int result = check_capabilities(route); result != 0) return result;

    std::vector<float> source(kRows * kSourceSlots);
    for (uint64_t row = 0; row < kRows; ++row) {
        source[row] = static_cast<float>(row);
        source[kRows + row] = static_cast<float>(kRows - 1 - row);
        source[2 * kRows + row] = static_cast<float>(row % 17);
    }
    auto desc = bank_desc(route, kRows, kSourceSlots, kSlotCapacity);
    GafimeGpuSemanticBank bank = nullptr;
    int status = gafime_gpu_semantic_bank_alloc_v1(0, &desc, &bank);
    if (status != GAFIME_STATUS_OK || bank == nullptr) return fail("fp32 semantic bank allocation failed", status);
    const GafimeConstBufferView source_view = const_f32_view(source);
    status = gafime_gpu_semantic_bank_upload_v1(bank, &route, &source_view);
    if (status != GAFIME_STATUS_OK) return fail("fp32 semantic source upload failed", status);

    const uint32_t mean_slots[] = {0, 1};
    GafimeSemanticColumnMeanBatch mean_batch{};
    mean_batch.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    mean_batch.struct_size = sizeof(mean_batch);
    mean_batch.candidate_slots = {mean_slots, 2};
    ScalarResults means(route, 2);
    status = gafime_gpu_semantic_column_means_v1(bank, &mean_batch, &means.table);
    if (status != GAFIME_STATUS_OK || means.table.count != 2 ||
        means.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        means.states[1] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        means.supports[0] != kRows || means.supports[1] != kRows ||
        !near(means.values[0], 127.5f) || !near(means.values[1], 127.5f)) {
        return fail("serial native fp32 column means did not freeze the expected values", status);
    }

    const uint32_t operands[] = {0, 1, 3, 0, 1};
    const uint64_t frozen_means[] = {f32_bits(means.values[0]), f32_bits(means.values[1])};
    const GafimeSemanticFrozenRegionTerm region_terms[] = {
        {0, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(10.0f)},
        {0, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(200.0f)},
    };
    GafimeSemanticProgramNode nodes[4]{};
    nodes[0].opcode = GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE;
    nodes[0].output_slot = 3;
    nodes[0].operand_offset = 0;
    nodes[0].operand_count = 2;
    nodes[1].opcode = GAFIME_SEMANTIC_PROGRAM_SOFTSIGN;
    nodes[1].output_slot = 4;
    nodes[1].operand_offset = 2;
    nodes[1].operand_count = 1;
    nodes[2].opcode = GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT;
    nodes[2].output_slot = 5;
    nodes[2].operand_offset = 3;
    nodes[2].operand_count = 2;
    nodes[2].mean_offset = 0;
    nodes[2].mean_count = 2;
    nodes[3].opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    nodes[3].output_slot = 6;
    nodes[3].region_term_offset = 0;
    nodes[3].region_term_count = 2;
    GafimeSemanticProgramBatch program{};
    program.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    program.struct_size = sizeof(program);
    program.route = route;
    program.nodes = nodes;
    program.node_count = 4;
    program.operand_slots = {operands, 5};
    program.mean_bits = {frozen_means, 2};
    program.region_terms = {region_terms, 2};
    status = gafime_gpu_semantic_materialize_v1(bank, &program);
    if (status != GAFIME_STATUS_OK) return fail("native semantic program materialization failed", status);

    const uint32_t derived_slots[] = {3, 4, 5, 6};
    std::vector<float> derived(kRows * 4);
    GafimeMutableBufferView derived_view = mutable_f32_view(derived);
    status = gafime_gpu_semantic_bank_download_v1(bank, {derived_slots, 4}, &route, &derived_view);
    if (status != GAFIME_STATUS_OK) return fail("native derived-column download failed", status);
    for (uint64_t row = 0; row < kRows; ++row) {
        const float x = static_cast<float>(row);
        const float y = static_cast<float>(kRows - 1 - row);
        const float difference = std::fabs(x - y);
        const float expected_region = row > 10 && row <= 200 ? 1.0f : 0.0f;
        if (!near(derived[row], difference) ||
            !near(derived[kRows + row], difference / (1.0f + difference)) ||
            !near(derived[2 * kRows + row], (x - 127.5f) * (y - 127.5f)) ||
            derived[3 * kRows + row] != expected_region) {
            return fail("native materialized column differed from the frozen fp32 descriptor");
        }
    }

    const uint32_t left_slot[] = {0};
    const uint32_t right_slot[] = {1};
    GafimeSemanticAssociationBatch association{};
    association.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    association.struct_size = sizeof(association);
    association.statistic = GAFIME_SEMANTIC_ASSOCIATION_PEARSON;
    association.presentation = GAFIME_SEMANTIC_ASSOCIATION_SIGNED;
    association.left_slots = {left_slot, 1};
    association.right_slots = {right_slot, 1};
    ScalarResults pearson(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(bank, bank, &association, &pearson.table);
    if (status != GAFIME_STATUS_OK || pearson.table.count != 1 ||
        pearson.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        pearson.supports[0] != kRows || pearson.values[0] > -0.999f) {
        return fail("native signed Pearson association was not measured on the resident bank", status);
    }

    GafimeSemanticPearsonBatch legacy{};
    legacy.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    legacy.struct_size = sizeof(legacy);
    legacy.mode = GAFIME_SEMANTIC_PEARSON_ABSOLUTE;
    legacy.left_slots = {left_slot, 1};
    legacy.right_slots = {right_slot, 1};
    ScalarResults legacy_pearson(route, 1);
    status = gafime_gpu_semantic_pairwise_pearson_v1(bank, bank, &legacy, &legacy_pearson.table);
    if (status != GAFIME_STATUS_OK || legacy_pearson.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        legacy_pearson.values[0] < 0.999f) {
        return fail("legacy Pearson adapter did not use the native association lane", status);
    }

    association.statistic = GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN;
    ScalarResults spearman(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(bank, bank, &association, &spearman.table);
    if (status != GAFIME_STATUS_OK || spearman.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        spearman.supports[0] != kRows || spearman.values[0] > -0.999f) {
        return fail("bounded native Spearman association was not measured", status);
    }

    association.statistic = GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI;
    association.presentation = GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE;
    association.fixed_nmi_bins = 2;
    ScalarResults nmi(route, 1);
    status = gafime_gpu_semantic_pairwise_association_v1(bank, bank, &association, &nmi.table);
    if (status != GAFIME_STATUS_OK || nmi.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        nmi.supports[0] != kRows || !std::isfinite(nmi.values[0]) || nmi.values[0] < 0.0f) {
        return fail("native fixed corrected NMI association was not measured", status);
    }

    const GafimeSemanticEdge edges[] = {{0, 1}, {2, 3}, {5, 7}};
    const std::vector<float> weights = {1.0f, 0.5f, 2.0f};
    const uint32_t edge_candidates[] = {0, 3};
    GafimeSemanticEdgeEnergyBatch edge_batch{};
    edge_batch.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    edge_batch.struct_size = sizeof(edge_batch);
    edge_batch.edges = edges;
    edge_batch.edge_count = 3;
    edge_batch.weights = const_f32_view(weights);
    edge_batch.candidate_slots = {edge_candidates, 2};
    ScalarResults edge_results(route, 2);
    status = gafime_gpu_semantic_ordered_edge_energy_v1(bank, &edge_batch, &edge_results.table);
    if (status != GAFIME_STATUS_OK || edge_results.states[0] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        edge_results.states[1] != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        !std::isfinite(edge_results.values[0]) || !std::isfinite(edge_results.values[1])) {
        return fail("native ordered edge-energy context was not measured", status);
    }

    GafimeSemanticForecastRequest forecast_request{};
    forecast_request.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    forecast_request.struct_size = sizeof(forecast_request);
    forecast_request.program_max_operand_count = 2;
    forecast_request.program_operand_count = 5;
    forecast_request.program_mean_count = 2;
    forecast_request.program_region_term_count = 2;
    forecast_request.mean_slot_count = 2;
    forecast_request.pair_count = 1;
    forecast_request.graph_candidate_count = 2;
    forecast_request.graph_edge_count = 3;
    forecast_request.gather_slot_count = 2;
    forecast_request.gather_row_count = kRows / 2;
    forecast_request.retained_slot_count = 2;
    GafimeSemanticMemoryForecast forecast{};
    forecast.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    forecast.struct_size = sizeof(forecast);
    status = gafime_gpu_semantic_forecast_v1(bank, &forecast_request, &forecast);
    const uint64_t expected_rank_workspace =
        2 * kRows * (sizeof(float) + sizeof(uint32_t)) + 2 * kRows * sizeof(uint32_t) +
        2 * sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint64_t);
    if (status != GAFIME_STATUS_OK || forecast.resident_bytes != kRows * kSlotCapacity * sizeof(float) ||
        forecast.retained_bytes != kRows * 2 * sizeof(float) ||
        forecast.transient_bytes != expected_rank_workspace) {
        return fail("Metal forecast did not include the bounded rank workspace", status);
    }

    const uint32_t retain_slots[] = {3, 4};
    GafimeGpuSemanticBank retained = nullptr;
    status = gafime_gpu_semantic_bank_retain_v1(bank, {retain_slots, 2}, &retained);
    if (status != GAFIME_STATUS_OK || retained == nullptr) return fail("native retain copy failed", status);
    const uint32_t retained_download_slots[] = {0, 1};
    std::vector<float> retained_columns(kRows * 2);
    GafimeMutableBufferView retained_view = mutable_f32_view(retained_columns);
    status = gafime_gpu_semantic_bank_download_v1(
        retained, {retained_download_slots, 2}, &route, &retained_view);
    if (status != GAFIME_STATUS_OK ||
        !std::equal(retained_columns.begin(), retained_columns.end(), derived.begin())) {
        return fail("native retained columns did not preserve device-to-device values", status);
    }

    if (const int result = check_tied_spearman_and_largest_nmi_template(route); result != 0) {
        return result;
    }

    auto gathered_desc = bank_desc(route, kRows / 2, 0, 2);
    GafimeGpuSemanticBank gathered = nullptr;
    status = gafime_gpu_semantic_bank_alloc_v1(0, &gathered_desc, &gathered);
    if (status != GAFIME_STATUS_OK || gathered == nullptr) return fail("native gather destination allocation failed", status);
    std::vector<uint64_t> rows(kRows / 2);
    for (uint64_t row = 0; row < rows.size(); ++row) rows[row] = row * 2;
    const uint32_t gather_sources[] = {0, 1};
    const uint32_t gather_destinations[] = {0, 1};
    GafimeSemanticSparseGatherBatch gather_batch{};
    gather_batch.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    gather_batch.struct_size = sizeof(gather_batch);
    gather_batch.source_slots = {gather_sources, 2};
    gather_batch.destination_slots = {gather_destinations, 2};
    gather_batch.row_indices = {rows.data(), rows.size()};
    status = gafime_gpu_semantic_sparse_gather_v1(bank, gathered, &gather_batch);
    if (status != GAFIME_STATUS_OK) return fail("native sparse gather failed", status);
    std::vector<float> gathered_columns(rows.size() * 2);
    GafimeMutableBufferView gathered_view = mutable_f32_view(gathered_columns);
    status = gafime_gpu_semantic_bank_download_v1(
        gathered, {gather_destinations, 2}, &route, &gathered_view);
    if (status != GAFIME_STATUS_OK) return fail("native gathered-column download failed", status);
    for (uint64_t row = 0; row < rows.size(); ++row) {
        if (gathered_columns[row] != source[rows[row]] ||
            gathered_columns[rows.size() + row] != source[kRows + rows[row]]) {
            return fail("native sparse gather changed a selected row");
        }
    }

    const int gathered_free = gafime_gpu_semantic_bank_free_v1(gathered);
    const int retained_free = gafime_gpu_semantic_bank_free_v1(retained);
    const int bank_free = gafime_gpu_semantic_bank_free_v1(bank);
    if (gathered_free != GAFIME_STATUS_OK || retained_free != GAFIME_STATUS_OK ||
        bank_free != GAFIME_STATUS_OK) {
        return fail("native semantic bank release failed");
    }
    return 0;
}

}  // namespace

int main() {
    return run_native_semantic_lifecycle();
}
