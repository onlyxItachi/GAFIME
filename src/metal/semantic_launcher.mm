#include "metal_api.hpp"
#include "../common/semantic_primitives_abi_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#if defined(__APPLE__) && __has_include(<Foundation/Foundation.h>) && __has_include(<Metal/Metal.h>)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define GAFIME_HAS_METAL_RUNTIME 1
#else
#define GAFIME_HAS_METAL_RUNTIME 0
#endif

namespace {

#if GAFIME_HAS_METAL_RUNTIME

constexpr uint64_t kMetalSemanticBankMagic = 0x474146534d544c33ull;  // GAFSMTL3
constexpr uint64_t kMetalSemanticMaxRows = 32'768ull;
constexpr uint32_t kMetalSemanticMaxSlots = 65'536u;
constexpr uint64_t kMetalSemanticMaxAssociationPairs = 65'536ull;
constexpr uint32_t kMetalSemanticReduceWidth = 64u;
// MSL exposes thread_position_in_grid as uint.  Keep every scheduled semantic
// invocation representable before the shader promotes it to ulong for offset
// arithmetic.  The round-down also accounts for the final partial group.
constexpr uint64_t kMetalSemanticMaxIndexedDispatchItems =
    (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) /
        static_cast<uint64_t>(kMetalSemanticReduceWidth)) *
    static_cast<uint64_t>(kMetalSemanticReduceWidth);

bool semantic_checked_add(uint64_t left, uint64_t right, uint64_t* out) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    *out = left + right;
    return true;
}

bool semantic_checked_mul(uint64_t left, uint64_t right, uint64_t* out) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    *out = left * right;
    return true;
}

bool semantic_host_size_supported(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

bool semantic_fp32_route(const GafimeNumericRoute& route) {
    const GafimeNumericRoute expected = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    return gafime_gpu_abi::route_fields_equal(route, expected);
}

bool semantic_rank_padded_rows(uint64_t rows, uint32_t* padded_rows_out) {
    if (padded_rows_out == nullptr || rows == 0 || rows > kMetalSemanticMaxRows) return false;
    uint64_t padded = 1;
    while (padded < rows) {
        if (padded > kMetalSemanticMaxRows / 2) return false;
        padded <<= 1;
    }
    *padded_rows_out = static_cast<uint32_t>(padded);
    return true;
}

bool semantic_has_unified_memory(id<MTLDevice> device) {
    if (device == nil) return false;
    if ([device respondsToSelector:@selector(hasUnifiedMemory)]) {
        return [device hasUnifiedMemory];
    }
    return [device isLowPower];
}

MTLResourceOptions semantic_cpu_visible_storage_options(id<MTLDevice> device) {
    return semantic_has_unified_memory(device)
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModeManaged;
}

bool semantic_metal_size_supported(uint64_t value) {
    return semantic_host_size_supported(value) &&
        value <= static_cast<uint64_t>(std::numeric_limits<NSUInteger>::max());
}

void semantic_mark_host_writes(id<MTLBuffer> buffer, NSUInteger length, bool managed_storage) {
    if (managed_storage && buffer != nil && length != 0) {
        [buffer didModifyRange:NSMakeRange(0, length)];
    }
}

NSArray<id<MTLDevice>>* semantic_available_devices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
        if (default_device != nil) {
            devices = @[default_device];
        }
    }
    return devices;
}

id<MTLDevice> semantic_device_for_id(uint32_t device_id) {
    NSArray<id<MTLDevice>>* devices = semantic_available_devices();
    return device_id < devices.count ? devices[device_id] : nil;
}

NSString* semantic_default_metallib_path() {
#ifdef GAFIME_METAL_DEFAULT_LIBRARY_PATH
    return [NSString stringWithUTF8String:GAFIME_METAL_DEFAULT_LIBRARY_PATH];
#else
    return nil;
#endif
}

id<MTLLibrary> semantic_load_library(id<MTLDevice> device) {
    NSString* env_path = [[[NSProcessInfo processInfo] environment]
        objectForKey:@"GAFIME_METAL_V1_METALLIB"];
    NSString* path = env_path.length > 0 ? env_path : semantic_default_metallib_path();
    if (device == nil || path == nil || path.length == 0) return nil;
    NSError* error = nil;
    return [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
}

struct MetalSemanticProgramNode {
    uint32_t opcode;
    uint32_t output_slot;
    uint32_t operand_offset;
    uint32_t operand_count;
    uint32_t mean_offset;
    uint32_t mean_count;
    uint32_t region_term_offset;
    uint32_t region_term_count;
};

struct MetalSemanticRowsInfo {
    uint64_t rows;
    uint32_t item_count;
    uint32_t reserved;
};

struct MetalSemanticAssociationInfo {
    uint64_t rows;
    uint64_t pair_count;
    uint32_t presentation;
    uint32_t bins;
    uint32_t padded_rows;
    uint32_t reserved;
};

struct MetalSemanticRankInfo {
    uint64_t rows;
    uint64_t pair_count;
    uint32_t padded_rows;
    uint32_t stage;
    uint32_t stride;
    uint32_t reserved;
};

struct MetalSemanticEdgeInfo {
    uint64_t rows;
    uint64_t edge_count;
    uint64_t candidate_count;
    uint64_t reserved;
};

struct MetalSemanticGatherInfo {
    uint64_t source_rows;
    uint64_t destination_rows;
    uint64_t slot_count;
    uint64_t reserved;
};

struct MetalSemanticRankRecord {
    float value;
    uint32_t row;
};

static_assert(sizeof(MetalSemanticProgramNode) == 32, "Metal semantic node ABI changed");
static_assert(sizeof(MetalSemanticRowsInfo) == 16, "Metal semantic rows ABI changed");
static_assert(sizeof(MetalSemanticAssociationInfo) == 32, "Metal association ABI changed");
static_assert(sizeof(MetalSemanticRankInfo) == 32, "Metal rank ABI changed");
static_assert(sizeof(MetalSemanticEdgeInfo) == 32, "Metal edge ABI changed");
static_assert(sizeof(MetalSemanticGatherInfo) == 32, "Metal gather ABI changed");
static_assert(sizeof(MetalSemanticRankRecord) == 8, "Metal rank record ABI changed");

struct MetalSemanticPipelines {
    id<MTLComputePipelineState> absolute_difference;
    id<MTLComputePipelineState> softsign;
    id<MTLComputePipelineState> centered_product;
    id<MTLComputePipelineState> frozen_region;
    id<MTLComputePipelineState> reject_nonfinite;
    id<MTLComputePipelineState> pearson;
    id<MTLComputePipelineState> column_means;
    id<MTLComputePipelineState> fixed_nmi;
    id<MTLComputePipelineState> edge_energy;
    id<MTLComputePipelineState> sparse_gather;
    id<MTLComputePipelineState> rank_prepare;
    id<MTLComputePipelineState> rank_sort;
    id<MTLComputePipelineState> rank_positions;
    id<MTLComputePipelineState> spearman_finalize;
};

id<MTLComputePipelineState> semantic_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    if (device == nil || library == nil) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    return [device newComputePipelineStateWithFunction:function error:&error];
}

bool semantic_pipeline_usable(id<MTLComputePipelineState> pipeline) {
    return pipeline != nil && [pipeline maxTotalThreadsPerThreadgroup] >= kMetalSemanticReduceWidth;
}

bool semantic_make_pipelines(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    MetalSemanticPipelines* pipelines_out
) {
    if (pipelines_out == nullptr || device == nil || library == nil) return false;
    MetalSemanticPipelines pipelines{};
    pipelines.absolute_difference = semantic_pipeline(device, library, @"gafime_semantic_absolute_difference");
    pipelines.softsign = semantic_pipeline(device, library, @"gafime_semantic_softsign");
    pipelines.centered_product = semantic_pipeline(device, library, @"gafime_semantic_centered_product");
    pipelines.frozen_region = semantic_pipeline(device, library, @"gafime_semantic_frozen_region_conjunction");
    pipelines.reject_nonfinite = semantic_pipeline(device, library, @"gafime_semantic_reject_nonfinite");
    pipelines.pearson = semantic_pipeline(device, library, @"gafime_semantic_pairwise_pearson");
    pipelines.column_means = semantic_pipeline(device, library, @"gafime_semantic_column_means");
    pipelines.fixed_nmi = semantic_pipeline(device, library, @"gafime_semantic_fixed_corrected_nmi");
    pipelines.edge_energy = semantic_pipeline(device, library, @"gafime_semantic_ordered_edge_energy");
    pipelines.sparse_gather = semantic_pipeline(device, library, @"gafime_semantic_sparse_gather");
    pipelines.rank_prepare = semantic_pipeline(device, library, @"gafime_semantic_rank_prepare");
    pipelines.rank_sort = semantic_pipeline(device, library, @"gafime_semantic_rank_bitonic_step");
    pipelines.rank_positions = semantic_pipeline(device, library, @"gafime_semantic_rank_positions");
    pipelines.spearman_finalize = semantic_pipeline(device, library, @"gafime_semantic_spearman_finalize");
    if (!semantic_pipeline_usable(pipelines.absolute_difference) ||
        !semantic_pipeline_usable(pipelines.softsign) ||
        !semantic_pipeline_usable(pipelines.centered_product) ||
        !semantic_pipeline_usable(pipelines.frozen_region) ||
        !semantic_pipeline_usable(pipelines.reject_nonfinite) ||
        !semantic_pipeline_usable(pipelines.pearson) ||
        !semantic_pipeline_usable(pipelines.column_means) ||
        !semantic_pipeline_usable(pipelines.fixed_nmi) ||
        !semantic_pipeline_usable(pipelines.edge_energy) ||
        !semantic_pipeline_usable(pipelines.sparse_gather) ||
        !semantic_pipeline_usable(pipelines.rank_prepare) ||
        !semantic_pipeline_usable(pipelines.rank_sort) ||
        !semantic_pipeline_usable(pipelines.rank_positions) ||
        !semantic_pipeline_usable(pipelines.spearman_finalize) ||
        device.maxThreadgroupMemoryLength < 12u * 1024u) {
        return false;
    }
    *pipelines_out = pipelines;
    return true;
}

struct MetalSemanticBank {
    uint64_t magic;
    uint32_t device_id;
    GafimeNumericRoute route;
    uint64_t rows;
    uint32_t source_slots;
    uint32_t slot_capacity;
    bool managed_storage;
    bool sources_uploaded;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> columns;
    MetalSemanticPipelines pipelines;
    std::vector<uint8_t> initialized_slots;
};

bool semantic_bank_valid(const MetalSemanticBank* bank) {
    return bank != nullptr && bank->magic == kMetalSemanticBankMagic;
}

MetalSemanticBank* semantic_bank_from_handle(GafimeGpuSemanticBank handle) {
    auto* bank = static_cast<MetalSemanticBank*>(handle);
    return semantic_bank_valid(bank) ? bank : nullptr;
}

bool semantic_slots_initialized(
    const MetalSemanticBank* bank,
    const uint32_t* slots,
    uint64_t slot_count
) {
    if (!semantic_bank_valid(bank) || (slot_count != 0 && slots == nullptr)) return false;
    for (uint64_t index = 0; index < slot_count; ++index) {
        if (slots[index] >= bank->slot_capacity || bank->initialized_slots[slots[index]] == 0) {
            return false;
        }
    }
    return true;
}

id<MTLBuffer> semantic_allocate_buffer(
    id<MTLDevice> device,
    uint64_t bytes,
    MTLResourceOptions options
) {
    if (device == nil || !semantic_metal_size_supported(bytes)) return nil;
    const NSUInteger length = static_cast<NSUInteger>(bytes == 0 ? 1 : bytes);
    return [device newBufferWithLength:length options:options];
}

id<MTLBuffer> semantic_upload_buffer(
    id<MTLDevice> device,
    const void* source,
    uint64_t bytes,
    MTLResourceOptions options,
    bool managed_storage
) {
    id<MTLBuffer> buffer = semantic_allocate_buffer(device, bytes, options);
    if (buffer == nil) return nil;
    if (bytes != 0) {
        if (source == nullptr || buffer.contents == nullptr) return nil;
        std::memcpy(buffer.contents, source, static_cast<size_t>(bytes));
        semantic_mark_host_writes(buffer, static_cast<NSUInteger>(bytes), managed_storage);
    }
    return buffer;
}

int semantic_finish_command(
    id<MTLCommandBuffer> command_buffer,
    bool managed_storage,
    id<MTLBuffer> synchronize_first = nil,
    id<MTLBuffer> synchronize_second = nil,
    id<MTLBuffer> synchronize_third = nil
) {
    if (command_buffer == nil) return GAFIME_STATUS_DEVICE_ERROR;
    if (managed_storage &&
        (synchronize_first != nil || synchronize_second != nil || synchronize_third != nil)) {
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (blit == nil) return GAFIME_STATUS_DEVICE_ERROR;
        if (synchronize_first != nil) [blit synchronizeResource:synchronize_first];
        if (synchronize_second != nil) [blit synchronizeResource:synchronize_second];
        if (synchronize_third != nil) [blit synchronizeResource:synchronize_third];
        [blit endEncoding];
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_DEVICE_ERROR;
}

bool semantic_dispatch_width(uint64_t items, MTLSize* grid_out) {
    if (grid_out == nullptr || !semantic_metal_size_supported(items) ||
        items > kMetalSemanticMaxIndexedDispatchItems) {
        return false;
    }
    const uint64_t groups = items == 0 ? 0 :
        1 + (items - 1) / static_cast<uint64_t>(kMetalSemanticReduceWidth);
    if (!semantic_metal_size_supported(groups) ||
        groups > kMetalSemanticMaxIndexedDispatchItems /
            static_cast<uint64_t>(kMetalSemanticReduceWidth)) {
        return false;
    }
    *grid_out = MTLSizeMake(static_cast<NSUInteger>(groups), 1, 1);
    return true;
}

int semantic_copy_results(
    GafimeSemanticScalarResultTable* results,
    uint64_t count,
    id<MTLBuffer> values,
    id<MTLBuffer> states,
    id<MTLBuffer> supports
) {
    uint64_t value_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t support_bytes = 0;
    if (!semantic_checked_mul(count, sizeof(float), &value_bytes) ||
        !semantic_checked_mul(count, sizeof(uint32_t), &state_bytes) ||
        !semantic_checked_mul(count, sizeof(uint64_t), &support_bytes) ||
        !semantic_host_size_supported(value_bytes) || !semantic_host_size_supported(state_bytes) ||
        !semantic_host_size_supported(support_bytes) || values == nil || states == nil || supports == nil ||
        values.contents == nullptr || states.contents == nullptr || supports.contents == nullptr) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    if (count != 0) {
        std::memcpy(results->values.data, values.contents, static_cast<size_t>(value_bytes));
        std::memcpy(results->states, states.contents, static_cast<size_t>(state_bytes));
        std::memcpy(results->supports, supports.contents, static_cast<size_t>(support_bytes));
    }
    results->count = count;
    return GAFIME_STATUS_OK;
}

int semantic_bank_alloc_internal(
    uint32_t device_id,
    const GafimeSemanticBankDesc* desc,
    GafimeGpuSemanticBank* bank_out
) {
    if (bank_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *bank_out = nullptr;
    int status = gafime_semantic_abi::validate_bank_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_fp32_route(desc->route)) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    if (desc->rows > kMetalSemanticMaxRows || desc->slot_capacity > kMetalSemanticMaxSlots ||
        !semantic_metal_size_supported(desc->bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    id<MTLDevice> device = semantic_device_for_id(device_id);
    if (device == nil) return GAFIME_STATUS_DEVICE_ERROR;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLLibrary> library = semantic_load_library(device);
    MetalSemanticPipelines pipelines{};
    if (queue == nil || library == nil || !semantic_make_pipelines(device, library, &pipelines)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const MTLResourceOptions options = semantic_cpu_visible_storage_options(device);
    id<MTLBuffer> columns = semantic_allocate_buffer(device, desc->bytes, options);
    if (columns == nil) return GAFIME_STATUS_OUT_OF_MEMORY;
    // Keep construction owned until the initialized-slot vector has allocated.
    // The exported catch boundary maps bad_alloc to OUT_OF_MEMORY, but it must
    // not strand an already-created Metal buffer or command queue on that
    // exceptional path.
    auto bank = std::make_unique<MetalSemanticBank>();
    bank->magic = kMetalSemanticBankMagic;
    bank->device_id = device_id;
    bank->route = desc->route;
    bank->rows = desc->rows;
    bank->source_slots = desc->source_slots;
    bank->slot_capacity = desc->slot_capacity;
    bank->managed_storage = options == MTLResourceStorageModeManaged;
    bank->sources_uploaded = false;
    bank->device = device;
    bank->queue = queue;
    bank->columns = columns;
    bank->pipelines = pipelines;
    bank->initialized_slots.assign(desc->slot_capacity, uint8_t{0});
    *bank_out = static_cast<GafimeGpuSemanticBank>(bank.release());
    return GAFIME_STATUS_OK;
}

int semantic_bank_upload_internal(
    MetalSemanticBank* bank,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* source_columns
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (bank->sources_uploaded) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (!gafime_gpu_abi::route_fields_equal(*route, bank->route)) return GAFIME_STATUS_INVALID_ARGUMENT;
    uint64_t source_elements = 0;
    if (!semantic_checked_mul(bank->rows, bank->source_slots, &source_elements)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_gpu_abi::validate_const_buffer(source_columns, GAFIME_DTYPE_F32, source_elements);
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t source_bytes = 0;
    if (!semantic_checked_mul(source_elements, sizeof(float), &source_bytes) ||
        !semantic_host_size_supported(source_bytes) || bank->columns.contents == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (source_bytes != 0) {
        std::memcpy(bank->columns.contents, source_columns->data, static_cast<size_t>(source_bytes));
        semantic_mark_host_writes(bank->columns, static_cast<NSUInteger>(source_bytes), bank->managed_storage);
    }
    std::fill(bank->initialized_slots.begin(), bank->initialized_slots.begin() + bank->source_slots, uint8_t{1});
    bank->sources_uploaded = true;
    return GAFIME_STATUS_OK;
}

int semantic_pair_banks(
    GafimeGpuSemanticBank left_handle,
    GafimeGpuSemanticBank right_handle,
    MetalSemanticBank** left_out,
    MetalSemanticBank** right_out
) {
    auto* left = semantic_bank_from_handle(left_handle);
    auto* right = semantic_bank_from_handle(right_handle);
    if (!semantic_bank_valid(left) || !semantic_bank_valid(right) || left_out == nullptr || right_out == nullptr ||
        left->device_id != right->device_id || left->rows != right->rows ||
        !gafime_gpu_abi::route_fields_equal(left->route, right->route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *left_out = left;
    *right_out = right;
    return GAFIME_STATUS_OK;
}

int semantic_materialize_internal(
    MetalSemanticBank* bank,
    const GafimeSemanticProgramBatch* batch
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    const int validation = gafime_semantic_abi::validate_program_batch(
        batch,
        GAFIME_PRECISION_FP32,
        bank->source_slots,
        bank->slot_capacity,
        bank->initialized_slots,
        gafime_semantic_abi::kSemanticMaxRegionTerms
    );
    if (validation != GAFIME_STATUS_OK) return validation;
    if (batch->node_count > gafime_semantic_abi::kSemanticMaxProgramNodes) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->node_count == 0) return GAFIME_STATUS_OK;
    bool has_derived_node = false;
    for (uint32_t index = 0; index < batch->node_count; ++index) {
        if (batch->nodes[index].opcode != GAFIME_SEMANTIC_PROGRAM_SOURCE) {
            has_derived_node = true;
            break;
        }
    }
    // A source-only batch is a validated no-op.  Avoid allocating placeholder
    // Metal buffers for it, which keeps the descriptor forecast independent of
    // opaque zero-length MTLBuffer implementation details.
    if (!has_derived_node) return GAFIME_STATUS_OK;

    uint64_t operand_bytes = 0;
    uint64_t mean_bytes = 0;
    uint64_t term_bytes = 0;
    if (!semantic_checked_mul(batch->operand_slots.len, sizeof(uint32_t), &operand_bytes) ||
        !semantic_checked_mul(batch->mean_bits.len, sizeof(uint64_t), &mean_bytes) ||
        !semantic_checked_mul(batch->region_terms.len, sizeof(GafimeSemanticFrozenRegionTerm), &term_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const MTLResourceOptions visible = semantic_cpu_visible_storage_options(bank->device);
    id<MTLBuffer> operands = operand_bytes == 0 ? nil : semantic_upload_buffer(
        bank->device, batch->operand_slots.ptr, operand_bytes, visible, bank->managed_storage);
    id<MTLBuffer> means = mean_bytes == 0 ? nil : semantic_upload_buffer(
        bank->device, batch->mean_bits.ptr, mean_bytes, visible, bank->managed_storage);
    id<MTLBuffer> terms = term_bytes == 0 ? nil : semantic_upload_buffer(
        bank->device, batch->region_terms.ptr, term_bytes, visible, bank->managed_storage);
    id<MTLBuffer> nonfinite = semantic_allocate_buffer(bank->device, sizeof(uint32_t), visible);
    if ((operand_bytes != 0 && operands == nil) || (mean_bytes != 0 && means == nil) ||
        (term_bytes != 0 && terms == nil) || nonfinite == nil ||
        nonfinite.contents == nullptr) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    std::memset(nonfinite.contents, 0, sizeof(uint32_t));
    semantic_mark_host_writes(nonfinite, sizeof(uint32_t), bank->managed_storage);

    id<MTLCommandBuffer> command = [bank->queue commandBuffer];
    if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
    const MetalSemanticRowsInfo rows{bank->rows, 0u, 0u};
    const MTLSize threads = MTLSizeMake(kMetalSemanticReduceWidth, 1, 1);
    std::vector<uint8_t> initialized = bank->initialized_slots;
    for (uint32_t index = 0; index < batch->node_count; ++index) {
        const GafimeSemanticProgramNode& node = batch->nodes[index];
        if (node.opcode == GAFIME_SEMANTIC_PROGRAM_SOURCE) {
            continue;
        }
        const MetalSemanticProgramNode metal_node{
            node.opcode,
            node.output_slot,
            node.operand_offset,
            node.operand_count,
            node.mean_offset,
            node.mean_count,
            node.region_term_offset,
            node.region_term_count,
        };
        id<MTLComputePipelineState> pipeline = nil;
        switch (node.opcode) {
        case GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE:
            pipeline = bank->pipelines.absolute_difference;
            break;
        case GAFIME_SEMANTIC_PROGRAM_SOFTSIGN:
            pipeline = bank->pipelines.softsign;
            break;
        case GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT:
            pipeline = bank->pipelines.centered_product;
            break;
        case GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION:
            pipeline = bank->pipelines.frozen_region;
            break;
        default:
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        MTLSize grid{};
        if (!semantic_dispatch_width(bank->rows, &grid)) return GAFIME_STATUS_INVALID_ARGUMENT;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bank->columns offset:0 atIndex:0];
        [encoder setBytes:&metal_node length:sizeof(metal_node) atIndex:1];
        if (node.opcode == GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION) {
            [encoder setBuffer:terms offset:0 atIndex:2];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        } else {
            [encoder setBuffer:operands offset:0 atIndex:2];
            if (node.opcode == GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT) {
                [encoder setBuffer:means offset:0 atIndex:3];
                [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
            } else {
                [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            }
        }
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];

        MetalSemanticRowsInfo scan{bank->rows, node.output_slot, 0u};
        encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:bank->pipelines.reject_nonfinite];
        [encoder setBuffer:bank->columns offset:0 atIndex:0];
        [encoder setBytes:&scan length:sizeof(scan) atIndex:1];
        [encoder setBuffer:nonfinite offset:0 atIndex:2];
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        initialized[node.output_slot] = 1;
    }
    const int status = semantic_finish_command(command, bank->managed_storage, nonfinite);
    if (status != GAFIME_STATUS_OK) return status;
    if (*static_cast<const uint32_t*>(nonfinite.contents) != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    bank->initialized_slots = std::move(initialized);
    return GAFIME_STATUS_OK;
}

int semantic_association_internal(
    MetalSemanticBank* left,
    MetalSemanticBank* right,
    const GafimeSemanticAssociationBatch* batch,
    GafimeSemanticScalarResultTable* results
) {
    if (!semantic_bank_valid(left) || !semantic_bank_valid(right)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    constexpr uint32_t kAssociationMask =
        GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON |
        GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN |
        GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI;
    constexpr uint32_t kNmiBinMask =
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_4 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_8 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_12 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_16 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_24 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_32 |
        GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_48;
    int status = gafime_semantic_abi::validate_association_batch(
        batch,
        left->slot_capacity,
        right->slot_capacity,
        kAssociationMask,
        kNmiBinMask,
        kMetalSemanticMaxAssociationPairs,
        kMetalSemanticMaxRows,
        kMetalSemanticMaxRows,
        left->rows
    );
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_scalar_results(results, left->route, batch->left_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_slots_initialized(left, batch->left_slots.ptr, batch->left_slots.len) ||
        !semantic_slots_initialized(right, batch->right_slots.ptr, batch->right_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->left_slots.len == 0) return GAFIME_STATUS_OK;

    uint64_t slot_bytes = 0;
    uint64_t value_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t support_bytes = 0;
    if (!semantic_checked_mul(batch->left_slots.len, sizeof(uint32_t), &slot_bytes) ||
        !semantic_checked_mul(batch->left_slots.len, sizeof(float), &value_bytes) ||
        !semantic_checked_mul(batch->left_slots.len, sizeof(uint32_t), &state_bytes) ||
        !semantic_checked_mul(batch->left_slots.len, sizeof(uint64_t), &support_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const MTLResourceOptions visible = semantic_cpu_visible_storage_options(left->device);
    id<MTLBuffer> left_slots = semantic_upload_buffer(
        left->device, batch->left_slots.ptr, slot_bytes, visible, left->managed_storage);
    id<MTLBuffer> right_slots = semantic_upload_buffer(
        left->device, batch->right_slots.ptr, slot_bytes, visible, left->managed_storage);
    id<MTLBuffer> values = semantic_allocate_buffer(left->device, value_bytes, visible);
    id<MTLBuffer> states = semantic_allocate_buffer(left->device, state_bytes, visible);
    id<MTLBuffer> supports = semantic_allocate_buffer(left->device, support_bytes, visible);
    if (left_slots == nil || right_slots == nil || values == nil || states == nil || supports == nil) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    MetalSemanticAssociationInfo info{
        left->rows,
        batch->left_slots.len,
        batch->presentation,
        batch->fixed_nmi_bins,
        0u,
        0u,
    };
    id<MTLCommandBuffer> command = [left->queue commandBuffer];
    if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
    const MTLSize threads = MTLSizeMake(kMetalSemanticReduceWidth, 1, 1);
    MTLSize groups = MTLSizeMake(static_cast<NSUInteger>(batch->left_slots.len), 1, 1);

    if (batch->statistic == GAFIME_SEMANTIC_ASSOCIATION_PEARSON ||
        batch->statistic == GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:batch->statistic == GAFIME_SEMANTIC_ASSOCIATION_PEARSON
            ? left->pipelines.pearson : left->pipelines.fixed_nmi];
        [encoder setBuffer:left->columns offset:0 atIndex:0];
        [encoder setBuffer:right->columns offset:0 atIndex:1];
        [encoder setBuffer:left_slots offset:0 atIndex:2];
        [encoder setBuffer:right_slots offset:0 atIndex:3];
        [encoder setBuffer:values offset:0 atIndex:4];
        [encoder setBuffer:states offset:0 atIndex:5];
        [encoder setBuffer:supports offset:0 atIndex:6];
        [encoder setBytes:&info length:sizeof(info) atIndex:7];
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];
    } else if (batch->statistic == GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN) {
        uint32_t padded_rows = 0;
        if (!semantic_rank_padded_rows(left->rows, &padded_rows)) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        info.padded_rows = padded_rows;
        uint64_t record_count = 0;
        uint64_t record_bytes = 0;
        uint64_t rank_count = 0;
        uint64_t rank_bytes = 0;
        if (!semantic_checked_mul(batch->left_slots.len, padded_rows, &record_count) ||
            !semantic_checked_mul(record_count, sizeof(MetalSemanticRankRecord), &record_bytes) ||
            !semantic_checked_mul(batch->left_slots.len, left->rows, &rank_count) ||
            !semantic_checked_mul(rank_count, sizeof(uint32_t), &rank_bytes)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        id<MTLBuffer> left_records = semantic_allocate_buffer(
            left->device, record_bytes, MTLResourceStorageModePrivate);
        id<MTLBuffer> right_records = semantic_allocate_buffer(
            left->device, record_bytes, MTLResourceStorageModePrivate);
        id<MTLBuffer> left_ranks = semantic_allocate_buffer(
            left->device, rank_bytes, MTLResourceStorageModePrivate);
        id<MTLBuffer> right_ranks = semantic_allocate_buffer(
            left->device, rank_bytes, MTLResourceStorageModePrivate);
        if (left_records == nil || right_records == nil || left_ranks == nil || right_ranks == nil) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        MetalSemanticRankInfo rank_info{left->rows, batch->left_slots.len, padded_rows, 0u, 0u, 0u};
        MTLSize item_grid{};
        if (!semantic_dispatch_width(record_count, &item_grid)) return GAFIME_STATUS_INVALID_ARGUMENT;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:left->pipelines.rank_prepare];
        [encoder setBuffer:left->columns offset:0 atIndex:0];
        [encoder setBuffer:right->columns offset:0 atIndex:1];
        [encoder setBuffer:left_slots offset:0 atIndex:2];
        [encoder setBuffer:right_slots offset:0 atIndex:3];
        [encoder setBuffer:left_records offset:0 atIndex:4];
        [encoder setBuffer:right_records offset:0 atIndex:5];
        [encoder setBytes:&rank_info length:sizeof(rank_info) atIndex:6];
        [encoder dispatchThreadgroups:item_grid threadsPerThreadgroup:threads];
        [encoder endEncoding];

        for (uint32_t stage = 2u; stage <= padded_rows; stage <<= 1u) {
            for (uint32_t stride = stage >> 1u; stride != 0; stride >>= 1u) {
                rank_info.stage = stage;
                rank_info.stride = stride;
                encoder = [command computeCommandEncoder];
                if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
                [encoder setComputePipelineState:left->pipelines.rank_sort];
                [encoder setBuffer:left_records offset:0 atIndex:0];
                [encoder setBuffer:right_records offset:0 atIndex:1];
                [encoder setBytes:&rank_info length:sizeof(rank_info) atIndex:2];
                [encoder dispatchThreadgroups:item_grid threadsPerThreadgroup:threads];
                [encoder endEncoding];
            }
        }
        if (padded_rows == 0) return GAFIME_STATUS_DEVICE_ERROR;
        MTLSize rank_grid{};
        if (!semantic_dispatch_width(rank_count, &rank_grid)) return GAFIME_STATUS_INVALID_ARGUMENT;
        encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:left->pipelines.rank_positions];
        [encoder setBuffer:left_records offset:0 atIndex:0];
        [encoder setBuffer:right_records offset:0 atIndex:1];
        [encoder setBuffer:left_ranks offset:0 atIndex:2];
        [encoder setBuffer:right_ranks offset:0 atIndex:3];
        [encoder setBytes:&rank_info length:sizeof(rank_info) atIndex:4];
        [encoder dispatchThreadgroups:rank_grid threadsPerThreadgroup:threads];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
        [encoder setComputePipelineState:left->pipelines.spearman_finalize];
        [encoder setBuffer:left->columns offset:0 atIndex:0];
        [encoder setBuffer:right->columns offset:0 atIndex:1];
        [encoder setBuffer:left_slots offset:0 atIndex:2];
        [encoder setBuffer:right_slots offset:0 atIndex:3];
        [encoder setBuffer:left_ranks offset:0 atIndex:4];
        [encoder setBuffer:right_ranks offset:0 atIndex:5];
        [encoder setBuffer:values offset:0 atIndex:6];
        [encoder setBuffer:states offset:0 atIndex:7];
        [encoder setBuffer:supports offset:0 atIndex:8];
        [encoder setBytes:&info length:sizeof(info) atIndex:9];
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];
    } else {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    status = semantic_finish_command(command, left->managed_storage, values, states, supports);
    if (status != GAFIME_STATUS_OK) return status;
    return semantic_copy_results(results, batch->left_slots.len, values, states, supports);
}

int semantic_column_means_internal(
    MetalSemanticBank* bank,
    const GafimeSemanticColumnMeanBatch* batch,
    GafimeSemanticScalarResultTable* results
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_semantic_abi::validate_column_mean_batch(batch, bank->slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_scalar_results(results, bank->route, batch->candidate_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_slots_initialized(bank, batch->candidate_slots.ptr, batch->candidate_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->candidate_slots.len == 0) return GAFIME_STATUS_OK;

    uint64_t slot_bytes = 0;
    uint64_t value_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t support_bytes = 0;
    if (!semantic_checked_mul(batch->candidate_slots.len, sizeof(uint32_t), &slot_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(float), &value_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(uint32_t), &state_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(uint64_t), &support_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const MTLResourceOptions visible = semantic_cpu_visible_storage_options(bank->device);
    id<MTLBuffer> slots = semantic_upload_buffer(
        bank->device, batch->candidate_slots.ptr, slot_bytes, visible, bank->managed_storage);
    id<MTLBuffer> values = semantic_allocate_buffer(bank->device, value_bytes, visible);
    id<MTLBuffer> states = semantic_allocate_buffer(bank->device, state_bytes, visible);
    id<MTLBuffer> supports = semantic_allocate_buffer(bank->device, support_bytes, visible);
    if (slots == nil || values == nil || states == nil || supports == nil) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const MetalSemanticRowsInfo info{bank->rows, static_cast<uint32_t>(batch->candidate_slots.len), 0u};
    MTLSize groups{};
    if (!semantic_dispatch_width(batch->candidate_slots.len, &groups)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    id<MTLCommandBuffer> command = [bank->queue commandBuffer];
    if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
    [encoder setComputePipelineState:bank->pipelines.column_means];
    [encoder setBuffer:bank->columns offset:0 atIndex:0];
    [encoder setBuffer:slots offset:0 atIndex:1];
    [encoder setBuffer:values offset:0 atIndex:2];
    [encoder setBuffer:states offset:0 atIndex:3];
    [encoder setBuffer:supports offset:0 atIndex:4];
    [encoder setBytes:&info length:sizeof(info) atIndex:5];
    [encoder dispatchThreadgroups:groups
        threadsPerThreadgroup:MTLSizeMake(kMetalSemanticReduceWidth, 1, 1)];
    [encoder endEncoding];
    status = semantic_finish_command(command, bank->managed_storage, values, states, supports);
    if (status != GAFIME_STATUS_OK) return status;
    return semantic_copy_results(results, batch->candidate_slots.len, values, states, supports);
}

int semantic_edge_energy_internal(
    MetalSemanticBank* bank,
    const GafimeSemanticEdgeEnergyBatch* batch,
    GafimeSemanticScalarResultTable* results
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_semantic_abi::validate_edge_energy_batch(
        batch, bank->route, bank->rows, bank->slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_scalar_results(results, bank->route, batch->candidate_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_slots_initialized(bank, batch->candidate_slots.ptr, batch->candidate_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->candidate_slots.len == 0) return GAFIME_STATUS_OK;

    uint64_t slot_bytes = 0;
    uint64_t edge_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t value_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t support_bytes = 0;
    if (!semantic_checked_mul(batch->candidate_slots.len, sizeof(uint32_t), &slot_bytes) ||
        !semantic_checked_mul(batch->edge_count, sizeof(GafimeSemanticEdge), &edge_bytes) ||
        !semantic_checked_mul(batch->edge_count, sizeof(float), &weight_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(float), &value_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(uint32_t), &state_bytes) ||
        !semantic_checked_mul(batch->candidate_slots.len, sizeof(uint64_t), &support_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const MTLResourceOptions visible = semantic_cpu_visible_storage_options(bank->device);
    id<MTLBuffer> slots = semantic_upload_buffer(
        bank->device, batch->candidate_slots.ptr, slot_bytes, visible, bank->managed_storage);
    id<MTLBuffer> edges = edge_bytes == 0 ? nil : semantic_upload_buffer(
        bank->device, batch->edges, edge_bytes, visible, bank->managed_storage);
    id<MTLBuffer> weights = weight_bytes == 0 ? nil : semantic_upload_buffer(
        bank->device, batch->weights.data, weight_bytes, visible, bank->managed_storage);
    id<MTLBuffer> values = semantic_allocate_buffer(bank->device, value_bytes, visible);
    id<MTLBuffer> states = semantic_allocate_buffer(bank->device, state_bytes, visible);
    id<MTLBuffer> supports = semantic_allocate_buffer(bank->device, support_bytes, visible);
    if (slots == nil || (edge_bytes != 0 && edges == nil) ||
        (weight_bytes != 0 && weights == nil) || values == nil || states == nil || supports == nil) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const MetalSemanticEdgeInfo info{bank->rows, batch->edge_count, batch->candidate_slots.len, 0u};
    MTLSize groups{};
    if (!semantic_dispatch_width(batch->candidate_slots.len, &groups)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    id<MTLCommandBuffer> command = [bank->queue commandBuffer];
    if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
    [encoder setComputePipelineState:bank->pipelines.edge_energy];
    [encoder setBuffer:bank->columns offset:0 atIndex:0];
    [encoder setBuffer:slots offset:0 atIndex:1];
    // Edge-count zero is a valid scalar-context request.  Keep all declared
    // Metal arguments bound without allocating a zero-length transient; the
    // shader never dereferences these placeholders when edge_count is zero.
    [encoder setBuffer:edge_bytes == 0 ? bank->columns : edges offset:0 atIndex:2];
    [encoder setBuffer:weight_bytes == 0 ? bank->columns : weights offset:0 atIndex:3];
    [encoder setBuffer:values offset:0 atIndex:4];
    [encoder setBuffer:states offset:0 atIndex:5];
    [encoder setBuffer:supports offset:0 atIndex:6];
    [encoder setBytes:&info length:sizeof(info) atIndex:7];
    [encoder dispatchThreadgroups:groups
        threadsPerThreadgroup:MTLSizeMake(kMetalSemanticReduceWidth, 1, 1)];
    [encoder endEncoding];
    status = semantic_finish_command(command, bank->managed_storage, values, states, supports);
    if (status != GAFIME_STATUS_OK) return status;
    return semantic_copy_results(results, batch->candidate_slots.len, values, states, supports);
}

int semantic_sparse_gather_internal(
    MetalSemanticBank* source,
    MetalSemanticBank* destination,
    const GafimeSemanticSparseGatherBatch* batch
) {
    if (!semantic_bank_valid(source) || !semantic_bank_valid(destination) || source == destination ||
        source->device_id != destination->device_id ||
        !gafime_gpu_abi::route_fields_equal(source->route, destination->route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_semantic_abi::validate_gather_batch(
        batch,
        source->rows,
        source->slot_capacity,
        destination->rows,
        destination->slot_capacity
    );
    if (status != GAFIME_STATUS_OK) return status;
    if (batch->row_indices.len > kMetalSemanticMaxRows ||
        !semantic_slots_initialized(source, batch->source_slots.ptr, batch->source_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t index = 0; index < batch->destination_slots.len; ++index) {
        if (destination->initialized_slots[batch->destination_slots.ptr[index]] != 0) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (batch->source_slots.len == 0 || batch->row_indices.len == 0) return GAFIME_STATUS_OK;

    uint64_t slot_bytes = 0;
    uint64_t row_bytes = 0;
    uint64_t items = 0;
    if (!semantic_checked_mul(batch->source_slots.len, sizeof(uint32_t), &slot_bytes) ||
        !semantic_checked_mul(batch->row_indices.len, sizeof(uint64_t), &row_bytes) ||
        !semantic_checked_mul(batch->source_slots.len, destination->rows, &items)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const MTLResourceOptions visible = semantic_cpu_visible_storage_options(source->device);
    id<MTLBuffer> source_slots = semantic_upload_buffer(
        source->device, batch->source_slots.ptr, slot_bytes, visible, source->managed_storage);
    id<MTLBuffer> destination_slots = semantic_upload_buffer(
        source->device, batch->destination_slots.ptr, slot_bytes, visible, source->managed_storage);
    id<MTLBuffer> rows = semantic_upload_buffer(
        source->device, batch->row_indices.ptr, row_bytes, visible, source->managed_storage);
    if (source_slots == nil || destination_slots == nil || rows == nil) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    MTLSize groups{};
    if (!semantic_dispatch_width(items, &groups)) return GAFIME_STATUS_INVALID_ARGUMENT;
    const MetalSemanticGatherInfo info{
        source->rows,
        destination->rows,
        batch->source_slots.len,
        0u,
    };
    id<MTLCommandBuffer> command = [source->queue commandBuffer];
    if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return GAFIME_STATUS_DEVICE_ERROR;
    [encoder setComputePipelineState:source->pipelines.sparse_gather];
    [encoder setBuffer:source->columns offset:0 atIndex:0];
    [encoder setBuffer:destination->columns offset:0 atIndex:1];
    [encoder setBuffer:source_slots offset:0 atIndex:2];
    [encoder setBuffer:destination_slots offset:0 atIndex:3];
    [encoder setBuffer:rows offset:0 atIndex:4];
    [encoder setBytes:&info length:sizeof(info) atIndex:5];
    [encoder dispatchThreadgroups:groups
        threadsPerThreadgroup:MTLSizeMake(kMetalSemanticReduceWidth, 1, 1)];
    [encoder endEncoding];
    status = semantic_finish_command(command, destination->managed_storage);
    if (status != GAFIME_STATUS_OK) return status;
    for (uint64_t index = 0; index < batch->destination_slots.len; ++index) {
        destination->initialized_slots[batch->destination_slots.ptr[index]] = 1;
    }
    return GAFIME_STATUS_OK;
}

int semantic_forecast_internal(
    MetalSemanticBank* bank,
    const GafimeSemanticForecastRequest* request,
    GafimeSemanticMemoryForecast* forecast
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_semantic_abi::validate_forecast_request(request);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_forecast(forecast);
    if (status != GAFIME_STATUS_OK) return status;
    if (request->pair_count > kMetalSemanticMaxAssociationPairs ||
        request->gather_row_count > kMetalSemanticMaxRows ||
        request->retained_slot_count > kMetalSemanticMaxSlots ||
        request->mean_slot_count > kMetalSemanticMaxSlots ||
        request->program_region_term_count >
            static_cast<uint64_t>(gafime_semantic_abi::kSemanticMaxProgramNodes) *
                gafime_semantic_abi::kSemanticMaxRegionTerms) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    uint64_t resident = 0;
    uint64_t retained = 0;
    uint64_t program_operand_bytes = 0;
    uint64_t program_mean_bytes = 0;
    uint64_t program_term_bytes = 0;
    uint64_t program_descriptor_bytes = 0;
    uint64_t program_flag_bytes = 0;
    uint64_t pair_slot_bytes = 0;
    uint64_t pair_result_bytes = 0;
    uint64_t pair_record_bytes = 0;
    uint64_t pair_rank_bytes = 0;
    uint64_t pair_bytes = 0;
    uint64_t mean_slot_bytes = 0;
    uint64_t mean_result_bytes = 0;
    uint64_t mean_bytes = 0;
    uint64_t graph_candidate_bytes = 0;
    uint64_t graph_edge_bytes = 0;
    uint64_t graph_bytes = 0;
    uint64_t gather_slot_bytes = 0;
    uint64_t gather_row_bytes = 0;
    uint64_t gather_bytes = 0;
    uint32_t padded_rows = 0;
    if (!semantic_rank_padded_rows(bank->rows, &padded_rows) ||
        !semantic_checked_mul(bank->rows, bank->slot_capacity, &resident) ||
        !semantic_checked_mul(resident, sizeof(float), &resident) ||
        !semantic_checked_mul(bank->rows, request->retained_slot_count, &retained) ||
        !semantic_checked_mul(retained, sizeof(float), &retained) ||
        !semantic_checked_mul(request->program_operand_count, sizeof(uint32_t), &program_operand_bytes) ||
        !semantic_checked_mul(request->program_mean_count, sizeof(uint64_t), &program_mean_bytes) ||
        !semantic_checked_mul(
            request->program_region_term_count, sizeof(GafimeSemanticFrozenRegionTerm), &program_term_bytes) ||
        !semantic_checked_add(program_operand_bytes, program_mean_bytes, &program_descriptor_bytes) ||
        !semantic_checked_add(program_descriptor_bytes, program_term_bytes, &program_descriptor_bytes) ||
        !semantic_checked_mul(
            (request->program_operand_count != 0 || request->program_mean_count != 0 ||
                request->program_region_term_count != 0) ? 1u : 0u,
            sizeof(uint32_t),
            &program_flag_bytes) ||
        !semantic_checked_add(program_descriptor_bytes, program_flag_bytes, &program_descriptor_bytes) ||
        !semantic_checked_mul(request->pair_count, 2u * sizeof(uint32_t), &pair_slot_bytes) ||
        !semantic_checked_mul(request->pair_count, sizeof(float) + sizeof(uint32_t) + sizeof(uint64_t),
            &pair_result_bytes) ||
        !semantic_checked_mul(request->pair_count, padded_rows, &pair_record_bytes) ||
        !semantic_checked_mul(pair_record_bytes, 2u * sizeof(MetalSemanticRankRecord), &pair_record_bytes) ||
        !semantic_checked_mul(request->pair_count, bank->rows, &pair_rank_bytes) ||
        !semantic_checked_mul(pair_rank_bytes, 2u * sizeof(uint32_t), &pair_rank_bytes) ||
        !semantic_checked_add(pair_slot_bytes, pair_result_bytes, &pair_bytes) ||
        !semantic_checked_add(pair_bytes, pair_record_bytes, &pair_bytes) ||
        !semantic_checked_add(pair_bytes, pair_rank_bytes, &pair_bytes) ||
        !semantic_checked_mul(request->mean_slot_count, sizeof(uint32_t), &mean_slot_bytes) ||
        !semantic_checked_mul(request->mean_slot_count, sizeof(float) + sizeof(uint32_t) + sizeof(uint64_t),
            &mean_result_bytes) ||
        !semantic_checked_add(mean_slot_bytes, mean_result_bytes, &mean_bytes) ||
        !semantic_checked_mul(request->graph_candidate_count,
            sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint64_t),
            &graph_candidate_bytes) ||
        !semantic_checked_mul(request->graph_edge_count, sizeof(GafimeSemanticEdge) + sizeof(float),
            &graph_edge_bytes) ||
        !semantic_checked_add(graph_candidate_bytes, graph_edge_bytes, &graph_bytes) ||
        !semantic_checked_mul(request->gather_slot_count, 2u * sizeof(uint32_t), &gather_slot_bytes) ||
        !semantic_checked_mul(request->gather_row_count, sizeof(uint64_t), &gather_row_bytes) ||
        !semantic_checked_add(gather_slot_bytes, gather_row_bytes, &gather_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // Pair association has no statistic field in the forecast.  Charge the
    // bounded rank records/ranks on every nonempty pair request so a caller
    // cannot pass an optimistic Pearson-only forecast before selecting
    // Spearman.  This is the actual Metal launch workspace, not CPU staging.
    if (request->pair_count == 0) pair_bytes = 0;
    if (request->mean_slot_count == 0) mean_bytes = 0;
    if (request->graph_candidate_count == 0) graph_bytes = 0;
    if (request->gather_slot_count == 0 || request->gather_row_count == 0) gather_bytes = 0;
    forecast->resident_bytes = resident;
    forecast->retained_bytes = retained;
    forecast->transient_bytes = std::max(
        std::max(pair_bytes, mean_bytes),
        std::max(std::max(program_descriptor_bytes, graph_bytes), gather_bytes)
    );
    return GAFIME_STATUS_OK;
}

int semantic_retain_internal(
    MetalSemanticBank* source,
    GafimeSliceU32 slots,
    GafimeGpuSemanticBank* retained_out
) {
    if (retained_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *retained_out = nullptr;
    if (!semantic_bank_valid(source)) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_semantic_abi::validate_slot_slice(slots, source->slot_capacity);
    if (status != GAFIME_STATUS_OK || slots.len == 0 || slots.len > kMetalSemanticMaxSlots ||
        !semantic_slots_initialized(source, slots.ptr, slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    if (!semantic_checked_mul(source->rows, slots.len, &elements) ||
        !semantic_checked_mul(elements, sizeof(float), &bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // Retention is an in-device copy of an already-supported physical bank.
    // Reuse its device, queue and immutable pipeline states rather than
    // reopening the metallib and recompiling every PSO for each retain call.
    // No process-global cache is involved, so there is no unkeyed cross-device
    // state to invalidate.
    const MTLResourceOptions storage = source->managed_storage
        ? MTLResourceStorageModeManaged
        : MTLResourceStorageModeShared;
    auto retained = std::make_unique<MetalSemanticBank>();
    retained->magic = kMetalSemanticBankMagic;
    retained->device_id = source->device_id;
    retained->route = source->route;
    retained->rows = source->rows;
    retained->source_slots = static_cast<uint32_t>(slots.len);
    retained->slot_capacity = static_cast<uint32_t>(slots.len);
    retained->managed_storage = source->managed_storage;
    retained->sources_uploaded = false;
    retained->device = source->device;
    retained->queue = source->queue;
    retained->columns = semantic_allocate_buffer(source->device, bytes, storage);
    retained->pipelines = source->pipelines;
    if (retained->columns == nil) return GAFIME_STATUS_OUT_OF_MEMORY;
    retained->initialized_slots.assign(static_cast<size_t>(slots.len), uint8_t{0});
    uint64_t column_bytes = 0;
    if (!semantic_checked_mul(source->rows, sizeof(float), &column_bytes) ||
        !semantic_metal_size_supported(column_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    id<MTLCommandBuffer> command = [source->queue commandBuffer];
    if (command == nil) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    for (uint64_t index = 0; index < slots.len; ++index) {
        uint64_t source_offset = 0;
        uint64_t destination_offset = 0;
        if (!semantic_checked_mul(slots.ptr[index], column_bytes, &source_offset) ||
            !semantic_checked_mul(index, column_bytes, &destination_offset) ||
            !semantic_metal_size_supported(source_offset) || !semantic_metal_size_supported(destination_offset)) {
            [blit endEncoding];
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        [blit copyFromBuffer:source->columns
            sourceOffset:static_cast<NSUInteger>(source_offset)
            toBuffer:retained->columns
            destinationOffset:static_cast<NSUInteger>(destination_offset)
            size:static_cast<NSUInteger>(column_bytes)];
    }
    [blit endEncoding];
    status = semantic_finish_command(command, false);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    std::fill(retained->initialized_slots.begin(), retained->initialized_slots.end(), uint8_t{1});
    retained->sources_uploaded = true;
    *retained_out = static_cast<GafimeGpuSemanticBank>(retained.release());
    return GAFIME_STATUS_OK;
}

int semantic_download_internal(
    MetalSemanticBank* bank,
    GafimeSliceU32 slots,
    const GafimeNumericRoute* route,
    GafimeMutableBufferView* columns_out
) {
    if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
    int status = gafime_semantic_abi::validate_download(
        slots, bank->slot_capacity, bank->rows, route, GAFIME_PRECISION_FP32, columns_out);
    if (status != GAFIME_STATUS_OK || !gafime_gpu_abi::route_fields_equal(*route, bank->route) ||
        !semantic_slots_initialized(bank, slots.ptr, slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (slots.len == 0) return GAFIME_STATUS_OK;
    if (bank->managed_storage) {
        id<MTLCommandBuffer> command = [bank->queue commandBuffer];
        if (command == nil) return GAFIME_STATUS_DEVICE_ERROR;
        const int synchronize_status = semantic_finish_command(command, true, bank->columns);
        if (synchronize_status != GAFIME_STATUS_OK) return synchronize_status;
    }
    uint64_t column_bytes = 0;
    if (!semantic_checked_mul(bank->rows, sizeof(float), &column_bytes) ||
        !semantic_host_size_supported(column_bytes) || bank->columns.contents == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const auto* source = static_cast<const uint8_t*>(bank->columns.contents);
    auto* destination = static_cast<uint8_t*>(columns_out->data);
    for (uint64_t index = 0; index < slots.len; ++index) {
        uint64_t source_offset = 0;
        uint64_t destination_offset = 0;
        if (!semantic_checked_mul(slots.ptr[index], column_bytes, &source_offset) ||
            !semantic_checked_mul(index, column_bytes, &destination_offset) ||
            !semantic_host_size_supported(source_offset) || !semantic_host_size_supported(destination_offset)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        std::memcpy(
            destination + static_cast<size_t>(destination_offset),
            source + static_cast<size_t>(source_offset),
            static_cast<size_t>(column_bytes)
        );
    }
    return GAFIME_STATUS_OK;
}

#endif  // GAFIME_HAS_METAL_RUNTIME

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_semantic_capabilities_v1(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    GafimeSemanticCapabilities* capabilities_out
) try {
    if (capabilities_out == nullptr || !gafime_gpu_abi::naturally_aligned(capabilities_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (GAFIME_ABI_VERSION_MAJOR_OF(consumer_abi_version) !=
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR ||
        GAFIME_ABI_VERSION_MINOR_OF(consumer_abi_version) <
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        id<MTLDevice> device = semantic_device_for_id(device_id);
        if (device == nil) return GAFIME_STATUS_DEVICE_ERROR;
        id<MTLLibrary> library = semantic_load_library(device);
        MetalSemanticPipelines pipelines{};
        if (library == nil || !semantic_make_pipelines(device, library, &pipelines)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        std::memset(capabilities_out, 0, sizeof(*capabilities_out));
        capabilities_out->abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
        capabilities_out->struct_size = sizeof(*capabilities_out);
        capabilities_out->backend_kind = GAFIME_BACKEND_METAL;
        capabilities_out->device_id = device_id;
        capabilities_out->profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32;
        capabilities_out->program_op_mask = GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE |
            GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE |
            GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN |
            GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT |
            GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION;
        capabilities_out->primitive_mask = GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION |
            GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY |
            GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER |
            GAFIME_SEMANTIC_PRIMITIVE_MASK_COLUMN_MEANS;
        capabilities_out->association_statistic_mask = GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON |
            GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN |
            GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI;
        capabilities_out->max_program_nodes = gafime_semantic_abi::kSemanticMaxProgramNodes;
        capabilities_out->max_slot_count = kMetalSemanticMaxSlots;
        capabilities_out->max_rows = kMetalSemanticMaxRows;
        capabilities_out->max_gather_rows = kMetalSemanticMaxRows;
        capabilities_out->fixed_corrected_nmi_bin_mask =
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_4 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_8 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_12 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_16 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_24 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_32 |
            GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_48;
        capabilities_out->max_region_terms = gafime_semantic_abi::kSemanticMaxRegionTerms;
        capabilities_out->max_association_pairs = kMetalSemanticMaxAssociationPairs;
        capabilities_out->max_spearman_rows = kMetalSemanticMaxRows;
        capabilities_out->max_fixed_corrected_nmi_rows = kMetalSemanticMaxRows;
        return GAFIME_STATUS_OK;
    }
#else
    (void)device_id;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_alloc_v1(
    uint32_t device_id,
    const GafimeSemanticBankDesc* desc,
    GafimeGpuSemanticBank* bank_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_bank_alloc_internal(device_id, desc, bank_out);
    }
#else
    (void)device_id;
    (void)desc;
    if (bank_out != nullptr) *bank_out = nullptr;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_upload_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* source_columns
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_bank_upload_internal(semantic_bank_from_handle(bank_handle), route, source_columns);
    }
#else
    (void)bank_handle;
    (void)route;
    (void)source_columns;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_materialize_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticProgramBatch* batch
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_materialize_internal(semantic_bank_from_handle(bank_handle), batch);
    }
#else
    (void)bank_handle;
    (void)batch;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_pairwise_pearson_v1(
    GafimeGpuSemanticBank left_bank,
    GafimeGpuSemanticBank right_bank,
    const GafimeSemanticPearsonBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        MetalSemanticBank* left = nullptr;
        MetalSemanticBank* right = nullptr;
        int status = semantic_pair_banks(left_bank, right_bank, &left, &right);
        if (status != GAFIME_STATUS_OK) return status;
        status = gafime_semantic_abi::validate_pearson_batch(
            batch, left->slot_capacity, right->slot_capacity);
        if (status != GAFIME_STATUS_OK) return status;
        GafimeSemanticAssociationBatch association{};
        association.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
        association.struct_size = sizeof(association);
        association.statistic = GAFIME_SEMANTIC_ASSOCIATION_PEARSON;
        association.presentation = batch->mode == GAFIME_SEMANTIC_PEARSON_ABSOLUTE
            ? GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE
            : GAFIME_SEMANTIC_ASSOCIATION_SIGNED;
        association.left_slots = batch->left_slots;
        association.right_slots = batch->right_slots;
        return semantic_association_internal(left, right, &association, results_out);
    }
#else
    (void)left_bank;
    (void)right_bank;
    (void)batch;
    (void)results_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_pairwise_association_v1(
    GafimeGpuSemanticBank left_bank,
    GafimeGpuSemanticBank right_bank,
    const GafimeSemanticAssociationBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        MetalSemanticBank* left = nullptr;
        MetalSemanticBank* right = nullptr;
        const int status = semantic_pair_banks(left_bank, right_bank, &left, &right);
        if (status != GAFIME_STATUS_OK) return status;
        return semantic_association_internal(left, right, batch, results_out);
    }
#else
    (void)left_bank;
    (void)right_bank;
    (void)batch;
    (void)results_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_column_means_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticColumnMeanBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_column_means_internal(semantic_bank_from_handle(bank_handle), batch, results_out);
    }
#else
    (void)bank_handle;
    (void)batch;
    (void)results_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_ordered_edge_energy_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticEdgeEnergyBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_edge_energy_internal(semantic_bank_from_handle(bank_handle), batch, results_out);
    }
#else
    (void)bank_handle;
    (void)batch;
    (void)results_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_sparse_gather_v1(
    GafimeGpuSemanticBank source_bank,
    GafimeGpuSemanticBank destination_bank,
    const GafimeSemanticSparseGatherBatch* batch
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_sparse_gather_internal(
            semantic_bank_from_handle(source_bank),
            semantic_bank_from_handle(destination_bank),
            batch
        );
    }
#else
    (void)source_bank;
    (void)destination_bank;
    (void)batch;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_forecast_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticForecastRequest* request,
    GafimeSemanticMemoryForecast* forecast_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_forecast_internal(semantic_bank_from_handle(bank_handle), request, forecast_out);
    }
#else
    (void)bank_handle;
    (void)request;
    (void)forecast_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_retain_v1(
    GafimeGpuSemanticBank source_bank,
    GafimeSliceU32 slots,
    GafimeGpuSemanticBank* retained_bank_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_retain_internal(semantic_bank_from_handle(source_bank), slots, retained_bank_out);
    }
#else
    (void)source_bank;
    (void)slots;
    if (retained_bank_out != nullptr) *retained_bank_out = nullptr;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_download_v1(
    GafimeGpuSemanticBank bank_handle,
    GafimeSliceU32 slots,
    const GafimeNumericRoute* route,
    GafimeMutableBufferView* columns_out
) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        return semantic_download_internal(semantic_bank_from_handle(bank_handle), slots, route, columns_out);
    }
#else
    (void)bank_handle;
    (void)slots;
    (void)route;
    (void)columns_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_free_v1(GafimeGpuSemanticBank bank_handle) try {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        auto* bank = semantic_bank_from_handle(bank_handle);
        if (!semantic_bank_valid(bank)) return GAFIME_STATUS_INVALID_ARGUMENT;
        bank->magic = 0;
        delete bank;
        return GAFIME_STATUS_OK;
    }
#else
    (void)bank_handle;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

}  // extern "C"
