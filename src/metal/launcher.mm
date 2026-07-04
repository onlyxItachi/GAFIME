#include "metal_api.hpp"
#include "../common/gpu_abi_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#if defined(__APPLE__) && __has_include(<Foundation/Foundation.h>) && __has_include(<Metal/Metal.h>)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define GAFIME_HAS_METAL_RUNTIME 1
#else
#define GAFIME_HAS_METAL_RUNTIME 0
#endif

namespace {

constexpr uint32_t kMetalThreadsPerThreadgroup = 64;
// Must match shader.metal: MI joint histogram fits the Apple threadgroup limit at
// <= 48 bins; spearman/MI threadgroup dispatches use a fixed reduction width.
constexpr uint32_t kMetalMaxMiBins = 48;
constexpr uint32_t kMetalReduceWidth = 64;

struct MetalChunk {
    uint32_t arity;
    uint32_t mi_bins;
    uint64_t descriptor_offset;
    uint64_t combo_count;
    uint64_t global_row_offset;
};

struct MetalLaunchInfo {
    uint64_t rows;
    uint32_t cols;
    uint32_t metric_count;
    uint32_t chunk_count;
};

bool metric_supported(uint32_t metric_id) {
    return metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2 ||
        metric_id == GAFIME_METRIC_MUTUAL_INFO || metric_id == GAFIME_METRIC_SPEARMAN;
}

bool protocol_has_metric(const GafimeLaunchProtocol* protocol, uint32_t metric_id) {
    for (uint32_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (protocol->metric_ids.ptr[idx] == metric_id) {
            return true;
        }
    }
    return false;
}

// Resolve the MI bin count for a chunk from its shape hint (mirrors the CUDA
// mi_bins_for_chunk), then clamp to the Metal threadgroup-memory ceiling.
uint32_t metal_mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = 96;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t hint = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (hint == 12 || hint == 24 || hint == 48 || hint == 96) {
            bins = hint;
        }
    }
    return bins > kMetalMaxMiBins ? kMetalMaxMiBins : bins;
}

int validate_matrix_desc(const GafimeMatrixDesc* desc) {
    if (desc == nullptr || desc->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (desc->rows == 0 || desc->cols == 0 || desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t expected = desc->rows * static_cast<uint64_t>(desc->cols) * sizeof(float);
    return desc->bytes == expected ? GAFIME_STATUS_OK : GAFIME_STATUS_INVALID_ARGUMENT;
}

uint64_t planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t total = 0;
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        total += protocol->chunks[idx].combo_count;
    }
    return total;
}

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t total_rows) {
    if (protocol->rank.top_k == 0) {
        return total_rows;
    }
    return std::min<uint64_t>(total_rows, protocol->rank.top_k);
}

uint32_t primary_metric_index(const GafimeLaunchProtocol* protocol) {
    if (protocol->metric_ids.len == 0) {
        return 0;
    }
    for (uint32_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (protocol->metric_ids.ptr[idx] == protocol->rank.primary_metric) {
            return idx;
        }
    }
    return 0;
}

int validate_protocol(const GafimeLaunchProtocol* protocol, uint64_t rows, uint32_t cols) {
    if (protocol == nullptr || protocol->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->backend_kind != GAFIME_BACKEND_METAL) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->n_samples != rows || protocol->n_features != cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.ptr == nullptr || protocol->metric_ids.len == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t idx = 0; idx < protocol->metric_ids.len; ++idx) {
        if (!metric_supported(protocol->metric_ids.ptr[idx])) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    if (protocol->combo_indices.ptr == nullptr || protocol->chunks == nullptr || protocol->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->permutations.permutation_count != 0) {
        return GAFIME_STATUS_GRAPH_UNSUPPORTED;
    }
    return GAFIME_STATUS_OK;
}

int validate_result_table(const GafimeLaunchProtocol* protocol, const GafimeResultTable* result) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr ||
        result->candidate_ids == nullptr || result->row_flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->metric_count < protocol->metric_ids.len || result->max_arity < protocol->max_arity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->capacity < output_row_count(protocol, planned_row_count(protocol))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

void compute_column_means(const float* features, uint64_t rows, uint32_t cols, std::vector<float>& means) {
    means.assign(cols, 0.0f);
    std::vector<uint64_t> counts(cols, 0);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t base = row * cols;
        for (uint32_t col = 0; col < cols; ++col) {
            const float value = features[base + col];
            if (std::isfinite(value)) {
                means[col] += value;
                counts[col] += 1;
            }
        }
    }
    for (uint32_t col = 0; col < cols; ++col) {
        if (counts[col] != 0) {
            means[col] /= static_cast<float>(counts[col]);
        }
    }
}

bool locate_combo(
    const GafimeLaunchProtocol* protocol,
    uint64_t global_row,
    const GafimeArityChunk** chunk_out,
    uint64_t* local_row_out
) {
    uint64_t offset = 0;
    for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
        const GafimeArityChunk& chunk = protocol->chunks[idx];
        if (global_row >= offset && global_row < offset + chunk.combo_count) {
            *chunk_out = &chunk;
            *local_row_out = global_row - offset;
            return true;
        }
        offset += chunk.combo_count;
    }
    return false;
}

std::vector<uint32_t> selected_rows(
    const GafimeLaunchProtocol* protocol,
    const std::vector<float>& metric_values,
    uint64_t total_rows
) {
    std::vector<uint32_t> rows(static_cast<size_t>(total_rows));
    std::iota(rows.begin(), rows.end(), 0);
    const uint64_t output_rows = output_row_count(protocol, total_rows);
    if (output_rows == total_rows) {
        return rows;
    }
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    const uint32_t metric_index = primary_metric_index(protocol);
    const bool descending = protocol->rank.descending != 0;
    std::stable_sort(rows.begin(), rows.end(), [&](uint32_t left, uint32_t right) {
        const float lhs = metric_values[static_cast<uint64_t>(left) * metric_count + metric_index];
        const float rhs = metric_values[static_cast<uint64_t>(right) * metric_count + metric_index];
        if (lhs == rhs) {
            return left < right;
        }
        return descending ? lhs > rhs : lhs < rhs;
    });
    rows.resize(static_cast<size_t>(output_rows));
    return rows;
}

int write_result_rows(
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result,
    const std::vector<float>& metric_values,
    const std::vector<uint32_t>& rows
) {
    const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
    for (uint64_t output_row = 0; output_row < rows.size(); ++output_row) {
        const uint64_t global_row = rows[static_cast<size_t>(output_row)];
        const GafimeArityChunk* chunk = nullptr;
        uint64_t local_row = 0;
        if (!locate_combo(protocol, global_row, &chunk, &local_row)) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const uint64_t combo_base = chunk->descriptor_offset + local_row * chunk->arity;
        for (uint32_t slot = 0; slot < result->max_arity; ++slot) {
            result->combo_indices[output_row * result->max_arity + slot] =
                slot < chunk->arity ? protocol->combo_indices.ptr[combo_base + slot] : UINT32_MAX;
        }
        for (uint32_t metric_idx = 0; metric_idx < result->metric_count; ++metric_idx) {
            result->metric_values[output_row * result->metric_count + metric_idx] =
                metric_idx < metric_count ? metric_values[global_row * metric_count + metric_idx] : 0.0f;
        }
        result->ranks[output_row] = static_cast<uint32_t>(output_row);
        result->families[output_row] = GAFIME_FAMILY_CONTINUOUS;
        result->candidate_ids[output_row] = global_row;
        result->row_flags[output_row] = 0;
    }
    result->row_count = static_cast<uint64_t>(rows.size());
    return GAFIME_STATUS_OK;
}

#if GAFIME_HAS_METAL_RUNTIME

bool metal_has_unified_memory(id<MTLDevice> device) {
    if (device == nil) {
        return false;
    }
    if ([device respondsToSelector:@selector(hasUnifiedMemory)]) {
        return [device hasUnifiedMemory];
    }
    return [device isLowPower];
}

bool metal_is_apple_family(id<MTLDevice> device) {
    if (device == nil || ![device respondsToSelector:@selector(supportsFamily:)]) {
        return false;
    }
    return [device supportsFamily:MTLGPUFamilyApple1];
}

uint32_t metal_device_flags(id<MTLDevice> device) {
    uint32_t flags = 0;
    const bool unified = metal_has_unified_memory(device);
    if (unified) {
        flags |= GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY | GAFIME_GPU_DEVICE_FLAG_INTEGRATED;
    } else if ([device isLowPower]) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (metal_is_apple_family(device)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY;
    }
    if (unified && device.recommendedMaxWorkingSetSize >= (8ull * 1024ull * 1024ull * 1024ull)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
    }
    return flags;
}

MTLResourceOptions cpu_visible_storage_options(id<MTLDevice> device) {
    return metal_has_unified_memory(device)
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModeManaged;
}

void mark_host_writes(id<MTLBuffer> buffer, NSUInteger length, bool managed_storage) {
    if (managed_storage && buffer != nil && length > 0) {
        [buffer didModifyRange:NSMakeRange(0, length)];
    }
}

struct MetalMatrix {
    uint32_t device_id;
    bool unified_memory;
    bool managed_storage;
    uint64_t rows;
    uint32_t cols;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> score_pipeline;
    id<MTLComputePipelineState> mi_pipeline;
    id<MTLComputePipelineState> spearman_pipeline;
    id<MTLBuffer> features;
    id<MTLBuffer> target;
    id<MTLBuffer> column_means;
};

NSArray<id<MTLDevice>>* available_devices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
        if (default_device != nil) {
            devices = @[default_device];
        }
    }
    return devices;
}

id<MTLDevice> device_for_id(uint32_t device_id) {
    NSArray<id<MTLDevice>>* devices = available_devices();
    if (device_id >= devices.count) {
        return nil;
    }
    return devices[device_id];
}

NSString* default_metallib_path() {
#ifdef GAFIME_METAL_DEFAULT_LIBRARY_PATH
    return [NSString stringWithUTF8String:GAFIME_METAL_DEFAULT_LIBRARY_PATH];
#else
    return nil;
#endif
}

id<MTLLibrary> load_library(id<MTLDevice> device) {
    NSString* env_path = [[[NSProcessInfo processInfo] environment] objectForKey:@"GAFIME_METAL_V1_METALLIB"];
    NSString* path = env_path.length > 0 ? env_path : default_metallib_path();
    if (path == nil || path.length == 0) {
        return nil;
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
    (void)error;
    return library;
}

#endif

}  // namespace

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(uint32_t device_id, GafimeGpuDeviceInfo* info_out) {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        if (info_out == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        id<MTLDevice> device = device_for_id(device_id);
        if (device == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        std::memset(info_out, 0, sizeof(*info_out));
        info_out->abi_version = GAFIME_ABI_VERSION;
        info_out->backend_kind = GAFIME_BACKEND_METAL;
        info_out->device_id = device_id;
        info_out->flags = metal_device_flags(device);
        std::snprintf(info_out->name, sizeof(info_out->name), "%s", device.name.UTF8String);
        info_out->total_global_mem_bytes = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        info_out->warp_size = 32;
        info_out->reserved[0] = metal_is_apple_family(device)
            ? GAFIME_GPU_ARCH_APPLE
            : GAFIME_GPU_ARCH_UNKNOWN;
        info_out->reserved[1] = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        info_out->reserved[2] = static_cast<uint64_t>(device.maxThreadgroupMemoryLength);
        info_out->reserved[3] = static_cast<uint64_t>(device.maxThreadsPerThreadgroup.width);
        info_out->reserved[4] = metal_has_unified_memory(device) ? 1ull : 0ull;
        info_out->reserved[5] = [device isLowPower] ? 1ull : 0ull;
        info_out->reserved[6] = [device isRemovable] ? 1ull : 0ull;
        if ([device respondsToSelector:@selector(registryID)]) {
            info_out->reserved[7] = static_cast<uint64_t>(device.registryID);
        }
        return GAFIME_STATUS_OK;
    }
#else
    return gafime_gpu_abi::fill_device_info(device_id, GAFIME_BACKEND_METAL, "metal-unavailable", info_out);
#endif
}

GAFIME_GPU_API int gafime_gpu_graph_capability(uint32_t device_id, GafimeGpuGraphCapability* capability_out) {
    (void)device_id;
    const int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_METAL,
        GAFIME_GRAPH_UNSUPPORTED,
        capability_out
    );
    if (status == GAFIME_STATUS_OK) {
        capability_out->stable_pointer_flags = 1;
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) {
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        int status = validate_matrix_desc(matrix_desc);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        id<MTLDevice> device = device_for_id(device_id);
        if (device == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLLibrary> library = load_library(device);
        if (queue == nil || library == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        NSError* error = nil;
        id<MTLFunction> score_function = [library newFunctionWithName:@"gafime_score_continuous"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:score_function error:&error];
        (void)error;
        if (score_function == nil || pipeline == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        id<MTLFunction> mi_function = [library newFunctionWithName:@"gafime_score_mutual_info"];
        id<MTLComputePipelineState> mi_pipeline = [device newComputePipelineStateWithFunction:mi_function error:&error];
        (void)error;
        id<MTLFunction> spearman_function = [library newFunctionWithName:@"gafime_score_spearman"];
        id<MTLComputePipelineState> spearman_pipeline = [device newComputePipelineStateWithFunction:spearman_function error:&error];
        (void)error;
        if (mi_function == nil || mi_pipeline == nil ||
            spearman_function == nil || spearman_pipeline == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const NSUInteger feature_bytes = static_cast<NSUInteger>(matrix_desc->rows) * matrix_desc->cols * sizeof(float);
        const NSUInteger target_bytes = static_cast<NSUInteger>(matrix_desc->rows) * sizeof(float);
        const NSUInteger mean_bytes = static_cast<NSUInteger>(matrix_desc->cols) * sizeof(float);
        const MTLResourceOptions storage_options = cpu_visible_storage_options(device);
        const bool managed_storage = storage_options == MTLResourceStorageModeManaged;
        auto* matrix = new MetalMatrix{};
        matrix->device_id = device_id;
        matrix->unified_memory = metal_has_unified_memory(device);
        matrix->managed_storage = managed_storage;
        matrix->rows = matrix_desc->rows;
        matrix->cols = matrix_desc->cols;
        matrix->device = device;
        matrix->queue = queue;
        matrix->score_pipeline = pipeline;
        matrix->mi_pipeline = mi_pipeline;
        matrix->spearman_pipeline = spearman_pipeline;
        matrix->features = [device newBufferWithLength:feature_bytes options:storage_options];
        matrix->target = [device newBufferWithLength:target_bytes options:storage_options];
        matrix->column_means = [device newBufferWithLength:mean_bytes options:storage_options];
        if (matrix->features == nil || matrix->target == nil || matrix->column_means == nil) {
            delete matrix;
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
        return GAFIME_STATUS_OK;
    }
#else
    (void)device_id;
    (void)matrix_desc;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || features_host == nullptr || target_host == nullptr ||
        rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::vector<float> means;
    compute_column_means(features_host, rows, cols, means);
    const NSUInteger feature_bytes = static_cast<NSUInteger>(rows) * cols * sizeof(float);
    const NSUInteger target_bytes = static_cast<NSUInteger>(rows) * sizeof(float);
    const NSUInteger mean_bytes = static_cast<NSUInteger>(cols) * sizeof(float);
    std::memcpy(matrix->features.contents, features_host, static_cast<size_t>(feature_bytes));
    std::memcpy(matrix->target.contents, target_host, static_cast<size_t>(target_bytes));
    std::memcpy(matrix->column_means.contents, means.data(), static_cast<size_t>(mean_bytes));
    mark_host_writes(matrix->features, feature_bytes, matrix->managed_storage);
    mark_host_writes(matrix->target, target_bytes, matrix->managed_storage);
    mark_host_writes(matrix->column_means, mean_bytes, matrix->managed_storage);
    return GAFIME_STATUS_OK;
#else
    (void)matrix_handle;
    (void)features_host;
    (void)target_host;
    (void)rows;
    (void)cols;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    if (matrix == nullptr || target_host == nullptr || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const NSUInteger target_bytes = static_cast<NSUInteger>(rows) * sizeof(float);
    std::memcpy(matrix->target.contents, target_host, static_cast<size_t>(target_bytes));
    mark_host_writes(matrix->target, target_bytes, matrix->managed_storage);
    return GAFIME_STATUS_OK;
#else
    (void)matrix_handle;
    (void)target_host;
    (void)rows;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
#if GAFIME_HAS_METAL_RUNTIME
    auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
    delete matrix;
#else
    (void)matrix_handle;
#endif
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
#if GAFIME_HAS_METAL_RUNTIME
    @autoreleasepool {
        auto* matrix = static_cast<MetalMatrix*>(matrix_handle);
        if (matrix == nullptr) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        int status = validate_protocol(protocol, matrix->rows, matrix->cols);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        status = validate_result_table(protocol, result_out);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        const uint64_t total_rows = planned_row_count(protocol);
        if (total_rows == 0) {
            result_out->row_count = 0;
            return GAFIME_STATUS_OK;
        }
        const uint32_t metric_count = static_cast<uint32_t>(protocol->metric_ids.len);
        std::vector<MetalChunk> chunks;
        chunks.reserve(protocol->chunk_count);
        uint64_t offset = 0;
        for (uint32_t idx = 0; idx < protocol->chunk_count; ++idx) {
            const GafimeArityChunk& chunk = protocol->chunks[idx];
            chunks.push_back(MetalChunk{
                chunk.arity,
                metal_mi_bins_for_chunk(protocol, chunk),
                chunk.descriptor_offset,
                chunk.combo_count,
                offset,
            });
            offset += chunk.combo_count;
        }
        const MetalLaunchInfo info{
            matrix->rows,
            matrix->cols,
            metric_count,
            protocol->chunk_count,
        };
        id<MTLBuffer> combo_buffer = [matrix->device
            newBufferWithBytes:protocol->combo_indices.ptr
            length:static_cast<NSUInteger>(protocol->combo_indices.len * sizeof(uint32_t))
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> metric_id_buffer = [matrix->device
            newBufferWithBytes:protocol->metric_ids.ptr
            length:static_cast<NSUInteger>(protocol->metric_ids.len * sizeof(uint32_t))
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> chunk_buffer = [matrix->device
            newBufferWithBytes:chunks.data()
            length:static_cast<NSUInteger>(chunks.size() * sizeof(MetalChunk))
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> info_buffer = [matrix->device
            newBufferWithBytes:&info
            length:sizeof(MetalLaunchInfo)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> metric_buffer = [matrix->device
            newBufferWithLength:static_cast<NSUInteger>(total_rows * metric_count * sizeof(float))
            options:(matrix->managed_storage ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared)];
        if (combo_buffer == nil || metric_id_buffer == nil || chunk_buffer == nil ||
            info_buffer == nil || metric_buffer == nil) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        id<MTLCommandBuffer> command_buffer = [matrix->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        [encoder setComputePipelineState:matrix->score_pipeline];
        [encoder setBuffer:matrix->features offset:0 atIndex:0];
        [encoder setBuffer:matrix->target offset:0 atIndex:1];
        [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
        [encoder setBuffer:combo_buffer offset:0 atIndex:3];
        [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
        [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
        [encoder setBuffer:metric_buffer offset:0 atIndex:6];
        [encoder setBuffer:info_buffer offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(static_cast<NSUInteger>(total_rows), 1, 1);
        MTLSize group = MTLSizeMake(kMetalThreadsPerThreadgroup, 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:group];

        // Mutual information + Spearman use one threadgroup per candidate (they
        // need cooperative histogram/reduction state), so they are dispatched
        // separately. The compute encoder serializes dependent dispatches on the
        // shared metric buffer, so the continuous pass (which zeroes the MI/
        // Spearman slots) is visible before these overwrite them.
        const MTLSize per_candidate_group = MTLSizeMake(kMetalReduceWidth, 1, 1);
        const MTLSize per_candidate_grid = MTLSizeMake(static_cast<NSUInteger>(total_rows), 1, 1);
        if (protocol_has_metric(protocol, GAFIME_METRIC_MUTUAL_INFO)) {
            [encoder setComputePipelineState:matrix->mi_pipeline];
            [encoder setBuffer:matrix->features offset:0 atIndex:0];
            [encoder setBuffer:matrix->target offset:0 atIndex:1];
            [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
            [encoder setBuffer:combo_buffer offset:0 atIndex:3];
            [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
            [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
            [encoder setBuffer:metric_buffer offset:0 atIndex:6];
            [encoder setBuffer:info_buffer offset:0 atIndex:7];
            [encoder dispatchThreadgroups:per_candidate_grid threadsPerThreadgroup:per_candidate_group];
        }
        if (protocol_has_metric(protocol, GAFIME_METRIC_SPEARMAN)) {
            [encoder setComputePipelineState:matrix->spearman_pipeline];
            [encoder setBuffer:matrix->features offset:0 atIndex:0];
            [encoder setBuffer:matrix->target offset:0 atIndex:1];
            [encoder setBuffer:matrix->column_means offset:0 atIndex:2];
            [encoder setBuffer:combo_buffer offset:0 atIndex:3];
            [encoder setBuffer:metric_id_buffer offset:0 atIndex:4];
            [encoder setBuffer:chunk_buffer offset:0 atIndex:5];
            [encoder setBuffer:metric_buffer offset:0 atIndex:6];
            [encoder setBuffer:info_buffer offset:0 atIndex:7];
            [encoder dispatchThreadgroups:per_candidate_grid threadsPerThreadgroup:per_candidate_group];
        }
        [encoder endEncoding];
        if (matrix->managed_storage) {
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            if (blit == nil) {
                return GAFIME_STATUS_DEVICE_ERROR;
            }
            [blit synchronizeResource:metric_buffer];
            [blit endEncoding];
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        const float* values = static_cast<const float*>(metric_buffer.contents);
        std::vector<float> metric_values(values, values + total_rows * metric_count);
        std::vector<uint32_t> rows = selected_rows(protocol, metric_values, total_rows);
        result_out->flags = 0;
        return write_result_rows(protocol, result_out, metric_values, rows);
    }
#else
    (void)matrix_handle;
    (void)protocol;
    (void)result_out;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
}

}  // extern "C"
