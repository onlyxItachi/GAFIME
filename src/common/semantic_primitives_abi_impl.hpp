#ifndef GAFIME_SEMANTIC_PRIMITIVES_ABI_IMPL_HPP
#define GAFIME_SEMANTIC_PRIMITIVES_ABI_IMPL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "gpu_abi_impl.hpp"
#include "gafime_semantic_primitives_abi.hpp"

/* Validation shared by CUDA and ROCm.  These helpers validate only physical
 * typed buffers, slot maps and arithmetic descriptors.  They deliberately do
 * not know FeatureId, EvidenceId, context, provenance or policy. */
namespace gafime_semantic_abi {

constexpr uint32_t kCapabilitiesStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticCapabilities, reserved));
constexpr uint32_t kBankDescStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticBankDesc, reserved));
constexpr uint32_t kProgramBatchStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticProgramBatch, reserved));
constexpr uint32_t kPearsonBatchStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticPearsonBatch, reserved));
constexpr uint32_t kEdgeEnergyBatchStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticEdgeEnergyBatch, reserved));
constexpr uint32_t kGatherBatchStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticSparseGatherBatch, reserved));
constexpr uint32_t kScalarResultStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticScalarResultTable, reserved));
constexpr uint32_t kForecastRequestStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticForecastRequest, reserved));
constexpr uint32_t kForecastStablePrefixSize =
    static_cast<uint32_t>(offsetof(GafimeSemanticMemoryForecast, reserved));

// These bounds align the physical native table with the canonical Rust
// program/frame limits.  They are advertised rather than inferred so a
// caller can reject an oversized lowering before a descriptor allocation.
constexpr uint32_t kSemanticMaxProgramNodes = 65'536u;
constexpr uint64_t kSemanticMaxGatherRows = 1'000'000ull;

inline bool abi_compatible(uint32_t abi_version, uint32_t struct_size, uint32_t prefix_size) {
    return GAFIME_ABI_VERSION_MAJOR_OF(abi_version) ==
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR &&
        GAFIME_ABI_VERSION_MINOR_OF(abi_version) >=
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR &&
        struct_size >= prefix_size;
}

inline bool range_within(uint64_t offset, uint64_t count, uint64_t length) {
    return offset <= length && count <= length - offset;
}

inline bool fits_host_elements(uint64_t count, size_t element_size) {
    return element_size != 0 &&
        count <= static_cast<uint64_t>(std::numeric_limits<size_t>::max() / element_size);
}

template <typename T>
inline bool aligned_or_empty(const T* pointer, uint64_t len) {
    return (len == 0 || pointer != nullptr) && gafime_gpu_abi::naturally_aligned(pointer);
}

inline int validate_slot_slice(GafimeSliceU32 slots, uint32_t capacity) {
    if (!aligned_or_empty(slots.ptr, slots.len) || !fits_host_elements(slots.len, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t index = 0; index < slots.len; ++index) {
        if (slots.ptr[index] >= capacity) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

// Gather destination uniqueness is a physical descriptor invariant, not a
// scheduling or ranking decision.  Keep it in the common validator so CUDA
// and ROCm reject the same malformed batch before examining bank state.  The
// set is bounded by the request itself (which validate_slot_slice has already
// made representable), never by an independently advertised bank capacity.
inline int validate_distinct_slot_slice(GafimeSliceU32 slots) {
    if (slots.len < 2) return GAFIME_STATUS_OK;
    std::unordered_set<uint32_t> claimed;
    claimed.reserve(static_cast<size_t>(slots.len));
    for (uint64_t index = 0; index < slots.len; ++index) {
        if (!claimed.insert(slots.ptr[index]).second) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return GAFIME_STATUS_OK;
}

inline int validate_row_slice(GafimeSliceU64 rows, uint64_t row_count) {
    if (!aligned_or_empty(rows.ptr, rows.len) || !fits_host_elements(rows.len, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t index = 0; index < rows.len; ++index) {
        if (rows.ptr[index] >= row_count) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_capabilities(
    const GafimeSemanticCapabilities* capabilities,
    uint32_t expected_backend,
    uint32_t expected_device
) {
    if (capabilities == nullptr || !gafime_gpu_abi::naturally_aligned(capabilities)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(
            capabilities->abi_version, capabilities->struct_size, kCapabilitiesStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (capabilities->backend_kind != expected_backend || capabilities->device_id != expected_device ||
        capabilities->profile_mask == 0 || capabilities->program_op_mask == 0 ||
        capabilities->primitive_mask == 0 || capabilities->association_statistic_mask == 0 ||
        !gafime_gpu_abi::flags_supported(capabilities->flags, 0) ||
        capabilities->max_program_nodes == 0 || capabilities->max_slot_count == 0 ||
        capabilities->max_rows == 0 || capabilities->max_gather_rows == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (capabilities->struct_size >= sizeof(GafimeSemanticCapabilities) &&
        !gafime_gpu_abi::all_zero(capabilities->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_bank_desc(const GafimeSemanticBankDesc* desc) {
    if (desc == nullptr || !gafime_gpu_abi::naturally_aligned(desc)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(desc->abi_version, desc->struct_size, kBankDescStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = gafime_gpu_abi::validate_embedded_numeric_route(&desc->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (!gafime_gpu_abi::flags_supported(desc->flags, 0) ||
        desc->layout != GAFIME_MATRIX_COLUMN_MAJOR || desc->rows == 0 ||
        desc->slot_capacity == 0 || desc->source_slots > desc->slot_capacity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const size_t element_size = gafime_gpu_abi::dtype_size(desc->route.storage_dtype);
    if (element_size == 0 || desc->rows > std::numeric_limits<uint64_t>::max() / desc->slot_capacity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t element_count = desc->rows * static_cast<uint64_t>(desc->slot_capacity);
    if (element_count > std::numeric_limits<uint64_t>::max() / element_size ||
        !fits_host_elements(element_count, element_size) ||
        desc->bytes != element_count * element_size) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->struct_size >= sizeof(GafimeSemanticBankDesc) &&
        !gafime_gpu_abi::all_zero(desc->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_program_batch(
    const GafimeSemanticProgramBatch* batch,
    uint32_t expected_profile,
    uint32_t source_slots,
    uint32_t slot_capacity,
    const std::vector<uint8_t>& initialized_slots
) {
    if (batch == nullptr || !gafime_gpu_abi::naturally_aligned(batch)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(batch->abi_version, batch->struct_size, kProgramBatchStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = gafime_gpu_abi::validate_embedded_numeric_route(&batch->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (batch->route.profile != expected_profile || batch->reserved32 != 0 ||
        !aligned_or_empty(batch->nodes, batch->node_count) ||
        !fits_host_elements(batch->node_count, sizeof(GafimeSemanticProgramNode)) ||
        (batch->struct_size >= sizeof(GafimeSemanticProgramBatch) &&
            !gafime_gpu_abi::all_zero(batch->reserved))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = validate_slot_slice(batch->operand_slots, slot_capacity);
    if (status != GAFIME_STATUS_OK || !aligned_or_empty(batch->mean_bits.ptr, batch->mean_bits.len) ||
        !fits_host_elements(batch->mean_bits.len, sizeof(uint64_t))) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }

    // The physical bank may already contain D2D-gathered retained values in
    // non-source slots.  Validate against that authoritative initialization
    // state rather than assuming only the source prefix can be consumed.
    if (initialized_slots.size() != slot_capacity) return GAFIME_STATUS_INVALID_ARGUMENT;
    std::vector<uint8_t> initialized = initialized_slots;
    for (uint32_t node_index = 0; node_index < batch->node_count; ++node_index) {
        const GafimeSemanticProgramNode& node = batch->nodes[node_index];
        if (!gafime_gpu_abi::all_zero(node.reserved) ||
            !range_within(node.operand_offset, node.operand_count, batch->operand_slots.len) ||
            !range_within(node.mean_offset, node.mean_count, batch->mean_bits.len) ||
            node.output_slot >= slot_capacity) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // A gathered accepted value may occupy a non-source physical slot.
        // It remains a legal operand, but no derived program may overwrite it.
        // The evolving mask additionally rejects duplicate derived outputs in
        // one batch: the ABI has no scratch-slot redefinition contract.
        const bool output_was_initialized = initialized_slots[node.output_slot] != 0;
        const bool output_was_written = initialized[node.output_slot] != 0;
        const auto operand = [&](uint32_t offset) { return batch->operand_slots.ptr[node.operand_offset + offset]; };
        for (uint32_t offset = 0; offset < node.operand_count; ++offset) {
            if (!initialized[operand(offset)]) return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        switch (node.opcode) {
        case GAFIME_SEMANTIC_PROGRAM_SOURCE:
            if (node.operand_count != 1 || node.mean_count != 0 ||
                node.output_slot != operand(0) || node.output_slot >= source_slots) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            break;
        case GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE:
            if (node.operand_count != 2 || node.mean_count != 0 || node.output_slot < source_slots ||
                output_was_initialized || output_was_written) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            break;
        case GAFIME_SEMANTIC_PROGRAM_SOFTSIGN:
            if (node.operand_count != 1 || node.mean_count != 0 || node.output_slot < source_slots ||
                output_was_initialized || output_was_written) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            break;
        case GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT:
            if (node.operand_count == 0 || node.mean_count != node.operand_count ||
                node.output_slot < source_slots || output_was_initialized || output_was_written) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            break;
        default: return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        for (uint32_t offset = 0; offset < node.operand_count; ++offset) {
            if (node.output_slot == operand(offset) && node.opcode != GAFIME_SEMANTIC_PROGRAM_SOURCE) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
        if (expected_profile != GAFIME_PRECISION_FP64) {
            for (uint32_t offset = 0; offset < node.mean_count; ++offset) {
                if ((batch->mean_bits.ptr[node.mean_offset + offset] >> 32) != 0) {
                    return GAFIME_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        initialized[node.output_slot] = 1;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_pearson_batch(
    const GafimeSemanticPearsonBatch* batch,
    uint32_t left_capacity,
    uint32_t right_capacity
) {
    if (batch == nullptr || !gafime_gpu_abi::naturally_aligned(batch)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(batch->abi_version, batch->struct_size, kPearsonBatchStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!gafime_gpu_abi::flags_supported(batch->flags, 0) ||
        (batch->mode != GAFIME_SEMANTIC_PEARSON_SIGNED &&
            batch->mode != GAFIME_SEMANTIC_PEARSON_ABSOLUTE) ||
        batch->left_slots.len != batch->right_slots.len ||
        (batch->struct_size >= sizeof(GafimeSemanticPearsonBatch) &&
            !gafime_gpu_abi::all_zero(batch->reserved))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_slot_slice(batch->left_slots, left_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    return validate_slot_slice(batch->right_slots, right_capacity);
}

inline int validate_edge_energy_batch(
    const GafimeSemanticEdgeEnergyBatch* batch,
    const GafimeNumericRoute& route,
    uint64_t rows,
    uint32_t slot_capacity
) {
    if (batch == nullptr || !gafime_gpu_abi::naturally_aligned(batch)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(batch->abi_version, batch->struct_size, kEdgeEnergyBatchStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!gafime_gpu_abi::flags_supported(batch->flags, 0) || batch->reserved32 != 0 ||
        !aligned_or_empty(batch->edges, batch->edge_count) ||
        !fits_host_elements(batch->edge_count, sizeof(GafimeSemanticEdge)) ||
        (batch->struct_size >= sizeof(GafimeSemanticEdgeEnergyBatch) &&
            !gafime_gpu_abi::all_zero(batch->reserved))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_embedded_const_buffer(
        &batch->weights, route.storage_dtype, batch->edge_count);
    if (status != GAFIME_STATUS_OK) return status;
    status = validate_slot_slice(batch->candidate_slots, slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    for (uint64_t index = 0; index < batch->edge_count; ++index) {
        if (batch->edges[index].left_row >= rows || batch->edges[index].right_row >= rows) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return GAFIME_STATUS_OK;
}

inline int validate_gather_batch(
    const GafimeSemanticSparseGatherBatch* batch,
    uint64_t source_rows,
    uint32_t source_capacity,
    uint64_t destination_rows,
    uint32_t destination_capacity
) {
    if (batch == nullptr || !gafime_gpu_abi::naturally_aligned(batch)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(batch->abi_version, batch->struct_size, kGatherBatchStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!gafime_gpu_abi::flags_supported(batch->flags, 0) || batch->reserved32 != 0 ||
        batch->source_slots.len != batch->destination_slots.len ||
        batch->row_indices.len != destination_rows ||
        (batch->struct_size >= sizeof(GafimeSemanticSparseGatherBatch) &&
            !gafime_gpu_abi::all_zero(batch->reserved))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_slot_slice(batch->source_slots, source_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = validate_slot_slice(batch->destination_slots, destination_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = validate_distinct_slot_slice(batch->destination_slots);
    if (status != GAFIME_STATUS_OK) return status;
    return validate_row_slice(batch->row_indices, source_rows);
}

inline int validate_scalar_results(
    GafimeSemanticScalarResultTable* results,
    const GafimeNumericRoute& route,
    uint64_t expected_count
) {
    if (results == nullptr || !gafime_gpu_abi::naturally_aligned(results)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(results->abi_version, results->struct_size, kScalarResultStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    int status = gafime_gpu_abi::validate_embedded_numeric_route(&results->route);
    if (status != GAFIME_STATUS_OK) return status;
    if (!gafime_gpu_abi::route_fields_equal(results->route, route) ||
        !gafime_gpu_abi::flags_supported(results->flags, 0) || results->reserved32 != 0 ||
        results->count != 0 || results->capacity < expected_count ||
        (results->struct_size >= sizeof(GafimeSemanticScalarResultTable) &&
            !gafime_gpu_abi::all_zero(results->reserved))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_gpu_abi::validate_embedded_mutable_buffer(
        &results->values, route.result_dtype, results->capacity);
    if (status != GAFIME_STATUS_OK) return status;
    if ((results->capacity != 0 && (results->states == nullptr || results->supports == nullptr)) ||
        !gafime_gpu_abi::naturally_aligned(results->states) ||
        !gafime_gpu_abi::naturally_aligned(results->supports) ||
        !fits_host_elements(results->capacity, sizeof(uint32_t)) ||
        !fits_host_elements(results->capacity, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_forecast_request(const GafimeSemanticForecastRequest* request) {
    if (request == nullptr || !gafime_gpu_abi::naturally_aligned(request)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(request->abi_version, request->struct_size, kForecastRequestStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (request->struct_size >= sizeof(GafimeSemanticForecastRequest) &&
        !gafime_gpu_abi::all_zero(request->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_forecast(GafimeSemanticMemoryForecast* forecast) {
    if (forecast == nullptr || !gafime_gpu_abi::naturally_aligned(forecast)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_compatible(forecast->abi_version, forecast->struct_size, kForecastStablePrefixSize)) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (forecast->struct_size >= sizeof(GafimeSemanticMemoryForecast) &&
        !gafime_gpu_abi::all_zero(forecast->reserved)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

inline int validate_download(
    GafimeSliceU32 slots,
    uint32_t slot_capacity,
    uint64_t rows,
    const GafimeNumericRoute* route,
    uint32_t expected_profile,
    GafimeMutableBufferView* columns_out
) {
    int status = validate_slot_slice(slots, slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->profile != expected_profile ||
        (slots.len != 0 && rows > std::numeric_limits<uint64_t>::max() / slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return gafime_gpu_abi::validate_mutable_buffer(
        columns_out, route->storage_dtype, rows * slots.len);
}

}  // namespace gafime_semantic_abi

#endif  // GAFIME_SEMANTIC_PRIMITIVES_ABI_IMPL_HPP
