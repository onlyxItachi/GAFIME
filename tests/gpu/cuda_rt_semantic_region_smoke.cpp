// Direct local-RT exercise of the resident semantic-bank lowering.  This is
// intentionally a correctness/lifecycle fixture, not a timing benchmark: it
// proves the OptiX path writes fresh physical bank slots without a host-column
// roundtrip and preserves their ordinary semantic-bank lifecycle.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../../src/common/gpu_abi_impl.hpp"
#include "../../src/cuda/rt_abi.hpp"

namespace {

constexpr uint64_t kRows = 4u;
constexpr uint32_t kSourceSlots = 2u;
constexpr uint32_t kSlotCapacity = 4u;

int fail(const char* message, int status = GAFIME_STATUS_OK) {
    if (status == GAFIME_STATUS_OK) {
        std::fprintf(stderr, "CUDA RT semantic-region smoke: %s\n", message);
    } else {
        std::fprintf(stderr, "CUDA RT semantic-region smoke: %s (status %d)\n", message, status);
    }
    return 1;
}

uint64_t f32_bits(float value) {
    uint32_t bits = 0u;
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

GafimeSemanticBankDesc bank_desc(
    const GafimeNumericRoute& route,
    uint32_t source_slots = kSourceSlots,
    uint32_t slot_capacity = kSlotCapacity
) {
    GafimeSemanticBankDesc desc{};
    desc.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = route;
    desc.layout = GAFIME_MATRIX_COLUMN_MAJOR;
    desc.rows = kRows;
    desc.source_slots = source_slots;
    desc.slot_capacity = slot_capacity;
    desc.bytes = kRows * static_cast<uint64_t>(slot_capacity) *
        gafime_gpu_abi::dtype_size(route.storage_dtype);
    return desc;
}

GafimeSemanticProgramBatch region_batch(
    const GafimeNumericRoute& route,
    const GafimeSemanticProgramNode* nodes,
    uint32_t node_count,
    const GafimeSemanticFrozenRegionTerm* terms,
    uint64_t term_count
) {
    GafimeSemanticProgramBatch batch{};
    batch.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    batch.struct_size = sizeof(batch);
    batch.route = route;
    batch.nodes = nodes;
    batch.node_count = node_count;
    batch.region_terms = {terms, term_count};
    return batch;
}

int allocate_uploaded_f32_bank(
    const std::vector<float>& source,
    uint32_t source_slots,
    uint32_t slot_capacity,
    GafimeGpuSemanticBank* bank_out
) {
    if (bank_out == nullptr || source.size() != kRows * static_cast<uint64_t>(source_slots)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *bank_out = nullptr;
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    const GafimeSemanticBankDesc desc = bank_desc(route, source_slots, slot_capacity);
    int status = gafime_gpu_semantic_bank_alloc_v1(0u, &desc, bank_out);
    if (status != GAFIME_STATUS_OK) return status;
    const GafimeConstBufferView source_view = const_f32_view(source);
    status = gafime_gpu_semantic_bank_upload_v1(*bank_out, &route, &source_view);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(gafime_gpu_semantic_bank_free_v1(*bank_out));
        *bank_out = nullptr;
    }
    return status;
}

int require_f32_slots_uninitialized(
    GafimeGpuSemanticBank bank,
    const GafimeNumericRoute& route,
    const uint32_t* slots,
    uint64_t slot_count,
    const char* context
) {
    std::vector<float> output(kRows * slot_count, -1.0f);
    GafimeMutableBufferView output_view = mutable_f32_view(output);
    const int status = gafime_gpu_semantic_bank_download_v1(
        bank, {slots, slot_count}, &route, &output_view);
    return status == GAFIME_STATUS_INVALID_ARGUMENT
        ? 0
        : fail(context, status);
}

int require_unsupported_profile(uint32_t profile) {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(profile);
    const GafimeSemanticBankDesc desc = bank_desc(route, 1u, 2u);
    GafimeGpuSemanticBank bank = nullptr;
    int status = gafime_gpu_semantic_bank_alloc_v1(0u, &desc, &bank);
    if (status != GAFIME_STATUS_OK || bank == nullptr) {
        return fail("unsupported-profile semantic-bank allocation failed", status);
    }
    const GafimeSemanticFrozenRegionTerm term[] = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(0.0f)},
    };
    GafimeSemanticProgramNode node{};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    node.output_slot = 1u;
    node.region_term_count = 1u;
    const GafimeSemanticProgramBatch batch = region_batch(route, &node, 1u, term, 1u);
    uint64_t peak = UINT64_MAX;
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, UINT64_MAX, &peak);
    const int free_status = gafime_gpu_semantic_bank_free_v1(bank);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || peak != 0u) {
        return fail("local RT accepted a non-fp32 semantic-bank route", status);
    }
    return free_status == GAFIME_STATUS_OK
        ? 0
        : fail("unsupported-profile semantic-bank free failed", free_status);
}

int require_source_rejection(float invalid_value, const char* context) {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    std::vector<float> source = {
        0.0f, invalid_value, 2.0f, 3.0f,
        3.0f, 2.0f, 1.0f, 0.0f,
    };
    GafimeGpuSemanticBank bank = nullptr;
    int status = allocate_uploaded_f32_bank(source, kSourceSlots, kSlotCapacity, &bank);
    if (status != GAFIME_STATUS_OK) return fail("invalid-source semantic-bank setup failed", status);
    const GafimeSemanticFrozenRegionTerm term[] = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
    };
    GafimeSemanticProgramNode node{};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    node.output_slot = 2u;
    node.region_term_count = 1u;
    const GafimeSemanticProgramBatch batch = region_batch(route, &node, 1u, term, 1u);
    uint64_t peak = UINT64_MAX;
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, UINT64_MAX, &peak);
    const uint32_t output_slot[] = {2u};
    const int uninitialized = require_f32_slots_uninitialized(
        bank, route, output_slot, 1u, context);
    const int free_status = gafime_gpu_semantic_bank_free_v1(bank);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || peak != 0u) {
        return fail(context, status);
    }
    if (uninitialized != 0) return uninitialized;
    return free_status == GAFIME_STATUS_OK ? 0 : fail("invalid-source semantic-bank free failed", free_status);
}

int require_threshold_rejection(float threshold, int expected_status, const char* context) {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    const std::vector<float> source = {
        0.0f, 1.0f, 2.0f, 3.0f,
        3.0f, 2.0f, 1.0f, 0.0f,
    };
    GafimeGpuSemanticBank bank = nullptr;
    int status = allocate_uploaded_f32_bank(source, kSourceSlots, kSlotCapacity, &bank);
    if (status != GAFIME_STATUS_OK) return fail("invalid-threshold semantic-bank setup failed", status);
    const GafimeSemanticFrozenRegionTerm term[] = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(threshold)},
    };
    GafimeSemanticProgramNode node{};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    node.output_slot = 2u;
    node.region_term_count = 1u;
    const GafimeSemanticProgramBatch batch = region_batch(route, &node, 1u, term, 1u);
    uint64_t peak = UINT64_MAX;
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, UINT64_MAX, &peak);
    const uint32_t output_slot[] = {2u};
    const int uninitialized = require_f32_slots_uninitialized(
        bank, route, output_slot, 1u, context);
    const int free_status = gafime_gpu_semantic_bank_free_v1(bank);
    if (status != expected_status || peak != 0u) return fail(context, status);
    if (uninitialized != 0) return uninitialized;
    return free_status == GAFIME_STATUS_OK ? 0 : fail("invalid-threshold semantic-bank free failed", free_status);
}

int require_axis_cap_rejection() {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    const std::vector<float> source = {
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
    };
    GafimeGpuSemanticBank bank = nullptr;
    int status = allocate_uploaded_f32_bank(source, 4u, 5u, &bank);
    if (status != GAFIME_STATUS_OK) return fail("axis-cap semantic-bank setup failed", status);
    const GafimeSemanticFrozenRegionTerm terms[] = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {2u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {3u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
    };
    GafimeSemanticProgramNode node{};
    node.opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    node.output_slot = 4u;
    node.region_term_count = 4u;
    const GafimeSemanticProgramBatch batch = region_batch(route, &node, 1u, terms, 4u);
    uint64_t peak = UINT64_MAX;
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, UINT64_MAX, &peak);
    const uint32_t output_slot[] = {4u};
    const int uninitialized = require_f32_slots_uninitialized(
        bank, route, output_slot, 1u, "over-axis RT region committed an output slot");
    const int free_status = gafime_gpu_semantic_bank_free_v1(bank);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || peak != 0u) {
        return fail("local RT accepted a region with more than three physical axes", status);
    }
    if (uninitialized != 0) return uninitialized;
    return free_status == GAFIME_STATUS_OK ? 0 : fail("axis-cap semantic-bank free failed", free_status);
}

int run() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    int caller_device = 0;
    if (cudaGetDevice(&caller_device) != cudaSuccess) return 77;

    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    const std::vector<float> source = {
        0.0f, 1.0f, 2.0f, 3.0f,
        3.0f, 2.0f, 1.0f, 0.0f,
    };
    const GafimeSemanticBankDesc desc = bank_desc(route);
    GafimeGpuSemanticBank bank = nullptr;
    int status = gafime_gpu_semantic_bank_alloc_v1(0u, &desc, &bank);
    if (status != GAFIME_STATUS_OK || bank == nullptr) {
        return fail("semantic-bank allocation failed", status);
    }
    const auto free_bank = [&]() { return gafime_gpu_semantic_bank_free_v1(bank); };
    const GafimeConstBufferView source_view = const_f32_view(source);
    status = gafime_gpu_semantic_bank_upload_v1(bank, &route, &source_view);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(free_bank());
        return fail("semantic-bank source upload failed", status);
    }

    const GafimeSemanticFrozenRegionTerm terms[] = {
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(0.0f)},
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(1.0f)},
        {1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)},
    };
    GafimeSemanticProgramNode nodes[2]{};
    nodes[0].opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    nodes[0].output_slot = 2u;
    nodes[0].region_term_offset = 0u;
    nodes[0].region_term_count = 2u;
    nodes[1].opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    nodes[1].output_slot = 3u;
    nodes[1].region_term_offset = 2u;
    nodes[1].region_term_count = 2u;
    const GafimeSemanticProgramBatch batch = region_batch(route, nodes, 2u, terms, 4u);

    // Generic semantic batches permit a later node to consume an earlier
    // output.  This local lowering is intentionally parallel and must reject
    // that shape for Rust to split into ordered RT calls rather than reading
    // an uninitialized output slot.
    const GafimeSemanticFrozenRegionTerm dependent_terms[] = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {2u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(0.0f)},
    };
    GafimeSemanticProgramNode dependent_nodes[2]{};
    dependent_nodes[0].opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    dependent_nodes[0].output_slot = 2u;
    dependent_nodes[0].region_term_count = 1u;
    dependent_nodes[1].opcode = GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION;
    dependent_nodes[1].output_slot = 3u;
    dependent_nodes[1].region_term_offset = 1u;
    dependent_nodes[1].region_term_count = 1u;
    const GafimeSemanticProgramBatch dependent_batch = region_batch(
        route, dependent_nodes, 2u, dependent_terms, 2u);
    uint64_t dependent_peak = 0u;
    status = gafime_gpu_semantic_region_materialize_rt_v1(
        bank, &dependent_batch, UINT64_MAX, &dependent_peak);
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || dependent_peak != 0u) {
        static_cast<void>(free_bank());
        return fail("parallel RT lowering accepted a dependent region batch", status);
    }

    const uint32_t output_slots[] = {2u, 3u};
    uint64_t early_peak = 0u;
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, 0u, &early_peak);
    if (status != GAFIME_STATUS_OUT_OF_MEMORY || early_peak == 0u) {
        static_cast<void>(free_bank());
        return fail("zero RT temporary budget did not report its prequery minimum", status);
    }
    if (require_f32_slots_uninitialized(
            bank, route, output_slots, 2u, "zero-budget RT call committed output slots") != 0) {
        static_cast<void>(free_bank());
        return 1;
    }

    uint64_t peak = 0u;
    // The prequery admission includes both the one-byte-per-slot validator
    // copy and three fixed SBT records.  This modest bound clears that fixed
    // gate but remains below the real AS plan, proving the later exact peak.
    status = gafime_gpu_semantic_region_materialize_rt_v1(
        bank, &batch, 1024u, &peak);
    if (status == GAFIME_STATUS_UNSUPPORTED_BACKEND) {
        static_cast<void>(free_bank());
        return 77;
    }
    if (status != GAFIME_STATUS_OUT_OF_MEMORY || peak == 0u) {
        static_cast<void>(free_bank());
        return fail("bounded temporary budget did not fail with its planned peak", status);
    }

    if (require_f32_slots_uninitialized(
            bank, route, output_slots, 2u, "over-budget RT call committed output slots") != 0) {
        static_cast<void>(free_bank());
        return 1;
    }

    // The preceding call established actual local RT availability, so each
    // following negative exercises its stated semantic/physical boundary
    // rather than treating a missing OptiX capability as a false positive.
    if (require_unsupported_profile(GAFIME_PRECISION_MIXED) != 0 ||
        require_unsupported_profile(GAFIME_PRECISION_FP64) != 0 ||
        require_source_rejection(
            std::numeric_limits<float>::denorm_min(),
            "subnormal source was accepted by local RT") != 0 ||
        require_source_rejection(
            std::numeric_limits<float>::infinity(),
            "infinite source was accepted by local RT") != 0 ||
        require_threshold_rejection(
            std::numeric_limits<float>::denorm_min(),
            GAFIME_STATUS_UNSUPPORTED_BACKEND,
            "subnormal threshold was accepted by local RT") != 0 ||
        require_threshold_rejection(
            std::numeric_limits<float>::infinity(),
            GAFIME_STATUS_INVALID_ARGUMENT,
            "infinite threshold was not rejected by the semantic boundary") != 0 ||
        require_axis_cap_rejection() != 0) {
        static_cast<void>(free_bank());
        return 1;
    }

    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, peak, &peak);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(free_bank());
        return fail("strict RT region materialization failed", status);
    }
    int restored_device = -1;
    if (cudaGetDevice(&restored_device) != cudaSuccess || restored_device != caller_device) {
        static_cast<void>(free_bank());
        return fail("RT materialization did not restore the caller CUDA device");
    }
    std::vector<float> output(kRows * 2u, -1.0f);
    GafimeMutableBufferView output_view = mutable_f32_view(output);
    status = gafime_gpu_semantic_bank_download_v1(bank, {output_slots, 2u}, &route, &output_view);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(free_bank());
        return fail("RT output slots were not downloadable through the semantic bank", status);
    }
    const float expected_first[] = {0.0f, 1.0f, 1.0f, 0.0f};
    const float expected_second[] = {0.0f, 0.0f, 1.0f, 1.0f};
    for (uint64_t row = 0; row < kRows; ++row) {
        if (output[row] != expected_first[row] || output[kRows + row] != expected_second[row]) {
            static_cast<void>(free_bank());
            return fail("RT membership differed from exact frozen-region predicates");
        }
    }
    status = gafime_gpu_semantic_region_materialize_rt_v1(bank, &batch, peak, &peak);
    if (status != GAFIME_STATUS_INVALID_ARGUMENT) {
        static_cast<void>(free_bank());
        return fail("RT lowering overwrote a previously initialized output slot", status);
    }

    const int free_status = free_bank();
    return free_status == GAFIME_STATUS_OK ? 0 : fail("semantic-bank free failed", free_status);
}

}  // namespace

int main() {
    return run();
}
