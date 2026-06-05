/**
 * GAFIME Metal Backend - Objective-C++ Host Wrapper
 * 
 * Bridges the C API to Apple's Metal framework for GPU compute.
 * 
 * Apple Silicon UMA Architecture:
 * - MTLBuffer with storageModeShared gives CPU+GPU zero-copy access
 * - No host↔device transfers needed (unlike CUDA cudaMemcpy)
 * - memcpy into shared buffer is a RAM→RAM copy, not PCIe transfer
 * 
 * Requires: macOS 13+ with Apple Silicon (arm64)
 * Compile:  clang++ -framework Metal -framework Foundation -fobjc-arc
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_backend.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <algorithm>

// ============================================================================
// SINGLETON DEVICE + LIBRARY (initialized once)
// ============================================================================

static id<MTLDevice>        g_device        = nil;
static id<MTLCommandQueue>  g_command_queue = nil;
static id<MTLLibrary>       g_library       = nil;
static bool                 g_initialized   = false;
static char                 g_library_path[4096] = {0};

/**
 * Initialize Metal device, command queue, and shader library.
 * Only succeeds on Apple Silicon (not Intel Macs).
 */
static bool metal_init(void) {
    if (g_initialized) return (g_device != nil);
    g_initialized = true;
    
    @autoreleasepool {
        // Get default GPU device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return false;
        
        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            g_device = nil;
            return false;
        }
        
        // Load pre-compiled Metal shader library (metallib)
        // Try multiple search paths
        NSMutableArray<NSString*>* paths = [NSMutableArray array];
        if (g_library_path[0] != '\0') {
            [paths addObject:[NSString stringWithUTF8String:g_library_path]];
        }
        [paths addObject:@"gafime_kernels.metallib"];
        [paths addObject:[[[NSBundle mainBundle] bundlePath]
            stringByAppendingPathComponent:@"gafime_kernels.metallib"]];
        
        NSError* error = nil;
        for (NSString* path in paths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                g_library = [g_device newLibraryWithFile:path error:&error];
                if (g_library) break;
            }
        }
        
        // Fallback: compile from source at runtime (development mode)
        if (!g_library) {
            NSArray<NSString*>* sourcePaths = @[
                @"src/metal/gafime_kernels.metal",
                @"gafime_kernels.metal",
            ];
            
            for (NSString* path in sourcePaths) {
                if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                    NSString* source = [NSString stringWithContentsOfFile:path
                                                                encoding:NSUTF8StringEncoding
                                                                   error:&error];
                    if (source) {
                        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                        options.fastMathEnabled = NO;
                        options.languageVersion = MTLLanguageVersion3_0;
                        g_library = [g_device newLibraryWithSource:source
                                                           options:options
                                                             error:&error];
                        if (g_library) break;
                    }
                }
            }
        }
        
        if (!g_library) {
            fprintf(stderr, "GAFIME Metal: Failed to load shader library: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            g_device = nil;
            g_command_queue = nil;
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// FUSED KERNEL PARAMS (must match Metal shader struct layout)
// ============================================================================

struct FusedParams {
    int ops[5];
    int interaction_types[4];
    int arity;
    int val_fold_id;
    int n_samples;
    int padding;
};

struct DiscreteBatchParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int padding;
};

struct MatrixBatchParams {
    int batch_size;
    int val_fold_id;
    int n_samples;
    int n_features;
};

struct DiscreteSelectionParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int target_bins;
    float y_sum;
    float y_sq_sum;
    int padding0;
    int padding1;
};

struct TimeSeriesBatchParams {
    int n_samples;
    int n_features;
    int n_candidates;
    int padding;
};

// ============================================================================
// BUCKET IMPLEMENTATION
// ============================================================================

struct MetalBucketImpl {
    id<MTLBuffer> feature_buffers[5];   // Shared-mode MTLBuffers (zero-copy UMA)
    id<MTLBuffer> target_buffer;
    id<MTLBuffer> mask_buffer;
    id<MTLBuffer> stats_buffer;         // 12 floats output
    id<MTLBuffer> params_buffer;        // FusedParams
    
    int n_samples;
    int n_features;
    
    id<MTLComputePipelineState> fused_pipeline;
};

struct MetalMatrixImpl {
    id<MTLBuffer> X_buffer;
    id<MTLBuffer> target_buffer;
    id<MTLBuffer> mask_buffer;
    id<MTLBuffer> means_buffer;
    id<MTLBuffer> batch_indices_buffer;
    id<MTLBuffer> stats_buffer;
    id<MTLBuffer> params_buffer;

    int n_samples;
    int n_features;
    int max_batch_size;

    id<MTLComputePipelineState> continuous_pipelines[6];
};

static id<MTLComputePipelineState> get_continuous_pipeline(MetalMatrixImpl* matrix, int arity) {
    if (!matrix || arity < 1 || arity > 5) return nil;
    if (matrix->continuous_pipelines[arity]) {
        return matrix->continuous_pipelines[arity];
    }

    @autoreleasepool {
        NSError* error = nil;
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        int arity_value = arity;
        [constants setConstantValue:&arity_value type:MTLDataTypeInt atIndex:0];
        id<MTLFunction> func = [g_library newFunctionWithName:@"gafime_global_continuous_kernel"
                                                constantValues:constants
                                                         error:&error];
        if (!func) {
            fprintf(stderr, "GAFIME Metal: Failed to specialize global continuous kernel: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return nil;
        }
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            fprintf(stderr, "GAFIME Metal: Failed to create global continuous pipeline: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return nil;
        }
        matrix->continuous_pipelines[arity] = pipeline;
        return pipeline;
    }
}

// ============================================================================
// C API IMPLEMENTATIONS
// ============================================================================

extern "C" {

int gafime_metal_available(void) {
    return metal_init() ? 1 : 0;
}

int gafime_metal_get_device_info(
    char* name_out,
    int* memory_mb_out,
    int* gpu_family_out
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!name_out || !memory_mb_out || !gpu_family_out) return GAFIME_ERROR_INVALID_ARGS;
    
    @autoreleasepool {
        const char* name = [[g_device name] UTF8String];
        strncpy(name_out, name, 255);
        name_out[255] = '\0';
        
        // On Apple Silicon, "recommended" working set size ≈ usable unified memory
        uint64_t mem_bytes = [g_device recommendedMaxWorkingSetSize];
        *memory_mb_out = (int)(mem_bytes / (1024 * 1024));
        
        // Detect GPU family (Apple7=M1, Apple8=M2, Apple9=M3/M4)
        if ([g_device supportsFamily:MTLGPUFamilyApple9]) {
            *gpu_family_out = 9;
        } else if ([g_device supportsFamily:MTLGPUFamilyApple8]) {
            *gpu_family_out = 8;
        } else if ([g_device supportsFamily:MTLGPUFamilyApple7]) {
            *gpu_family_out = 7;
        } else {
            *gpu_family_out = 0;
        }
    }
    
    return GAFIME_SUCCESS;
}

int gafime_metal_set_library_path(const char* metallib_path) {
    if (!metallib_path) return GAFIME_ERROR_INVALID_ARGS;
    strncpy(g_library_path, metallib_path, sizeof(g_library_path) - 1);
    g_library_path[sizeof(g_library_path) - 1] = '\0';
    if (g_initialized && !g_library) {
        g_initialized = false;
    }
    return GAFIME_SUCCESS;
}

// ============================================================================
// GLOBAL MATRIX BATCH API
// ============================================================================

int gafime_metal_matrix_alloc(
    int n_samples,
    int n_features,
    int max_batch_size,
    GafimeMetalMatrix* matrix_out
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!matrix_out || n_samples <= 0 || n_features <= 0 ||
        max_batch_size <= 0 || max_batch_size > 1024) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    @autoreleasepool {
        MetalMatrixImpl* matrix = new (std::nothrow) MetalMatrixImpl;
        if (!matrix) return GAFIME_ERROR_OUT_OF_MEMORY;
        memset(matrix, 0, sizeof(MetalMatrixImpl));
        matrix->n_samples = n_samples;
        matrix->n_features = n_features;
        matrix->max_batch_size = max_batch_size;

        size_t n = (size_t)n_samples;
        size_t f = (size_t)n_features;
        size_t X_bytes = n * f * sizeof(float);
        size_t vec_bytes = n * sizeof(float);
        size_t mask_bytes = n * sizeof(uint8_t);
        size_t means_bytes = f * sizeof(float);
        size_t indices_bytes = (size_t)max_batch_size * 5 * sizeof(int);
        size_t stats_bytes = (size_t)max_batch_size * 12 * sizeof(float);

        MTLResourceOptions shared_write = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        MTLResourceOptions shared_read = MTLResourceStorageModeShared;
        matrix->X_buffer = [g_device newBufferWithLength:X_bytes options:shared_write];
        matrix->target_buffer = [g_device newBufferWithLength:vec_bytes options:shared_write];
        matrix->mask_buffer = [g_device newBufferWithLength:mask_bytes options:shared_write];
        matrix->means_buffer = [g_device newBufferWithLength:means_bytes options:shared_write];
        matrix->batch_indices_buffer = [g_device newBufferWithLength:indices_bytes options:shared_write];
        matrix->stats_buffer = [g_device newBufferWithLength:stats_bytes options:shared_read];
        matrix->params_buffer = [g_device newBufferWithLength:sizeof(MatrixBatchParams) options:shared_write];

        if (!matrix->X_buffer || !matrix->target_buffer || !matrix->mask_buffer ||
            !matrix->means_buffer || !matrix->batch_indices_buffer ||
            !matrix->stats_buffer || !matrix->params_buffer) {
            gafime_metal_matrix_free(matrix);
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }

        *matrix_out = (GafimeMetalMatrix)matrix;
    }

    return GAFIME_SUCCESS;
}

int gafime_metal_matrix_upload(
    GafimeMetalMatrix matrix_handle,
    const float* h_X_colmajor,
    const float* h_y,
    const uint8_t* h_mask,
    const float* h_means
) {
    if (!matrix_handle || !h_X_colmajor || !h_y || !h_mask || !h_means) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    MetalMatrixImpl* matrix = (MetalMatrixImpl*)matrix_handle;
    size_t n = (size_t)matrix->n_samples;
    size_t f = (size_t)matrix->n_features;
    memcpy([matrix->X_buffer contents], h_X_colmajor, n * f * sizeof(float));
    memcpy([matrix->target_buffer contents], h_y, n * sizeof(float));
    memcpy([matrix->mask_buffer contents], h_mask, n * sizeof(uint8_t));
    memcpy([matrix->means_buffer contents], h_means, f * sizeof(float));
    return GAFIME_SUCCESS;
}

int gafime_metal_matrix_compute_batch(
    GafimeMetalMatrix matrix_handle,
    const int* h_batch_indices,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
) {
    if (!matrix_handle || !h_batch_indices || !h_stats_batch) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (arity < 1 || arity > 5 || batch_size <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    MetalMatrixImpl* matrix = (MetalMatrixImpl*)matrix_handle;
    if (batch_size > matrix->max_batch_size) return GAFIME_ERROR_INVALID_ARGS;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_continuous_pipeline(matrix, arity);
        if (!pipeline) return GAFIME_ERROR_KERNEL_FAILED;

        size_t indices_bytes = (size_t)batch_size * (size_t)arity * sizeof(int);
        size_t stats_bytes = (size_t)batch_size * 12 * sizeof(float);
        memcpy([matrix->batch_indices_buffer contents], h_batch_indices, indices_bytes);
        memset([matrix->stats_buffer contents], 0, stats_bytes);

        MatrixBatchParams* params = (MatrixBatchParams*)[matrix->params_buffer contents];
        params->batch_size = batch_size;
        params->val_fold_id = val_fold_id;
        params->n_samples = matrix->n_samples;
        params->n_features = matrix->n_features;

        id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix->X_buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix->target_buffer offset:0 atIndex:1];
        [encoder setBuffer:matrix->mask_buffer offset:0 atIndex:2];
        [encoder setBuffer:matrix->means_buffer offset:0 atIndex:3];
        [encoder setBuffer:matrix->batch_indices_buffer offset:0 atIndex:4];
        [encoder setBuffer:matrix->params_buffer offset:0 atIndex:5];
        [encoder setBuffer:matrix->stats_buffer offset:0 atIndex:6];

        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        NSUInteger numThreadGroups = ((NSUInteger)matrix->n_samples + threadGroupSize - 1) / threadGroupSize;
        if (numThreadGroups > 64) numThreadGroups = 64;
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadGroups, (NSUInteger)batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "GAFIME Metal: global matrix GPU error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        memcpy(h_stats_batch, [matrix->stats_buffer contents], stats_bytes);
    }

    return GAFIME_SUCCESS;
}

int gafime_metal_matrix_free(GafimeMetalMatrix matrix_handle) {
    if (!matrix_handle) return GAFIME_SUCCESS;
    MetalMatrixImpl* matrix = (MetalMatrixImpl*)matrix_handle;
    matrix->X_buffer = nil;
    matrix->target_buffer = nil;
    matrix->mask_buffer = nil;
    matrix->means_buffer = nil;
    matrix->batch_indices_buffer = nil;
    matrix->stats_buffer = nil;
    matrix->params_buffer = nil;
    for (int i = 0; i < 6; i++) {
        matrix->continuous_pipelines[i] = nil;
    }
    delete matrix;
    return GAFIME_SUCCESS;
}

// ============================================================================
// BUCKET MANAGEMENT
// ============================================================================

int gafime_metal_bucket_alloc(
    int n_samples,
    int n_features,
    GafimeMetalBucket* bucket_out
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!bucket_out || n_samples <= 0 || n_features <= 0 || n_features > 5) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    @autoreleasepool {
        MetalBucketImpl* bucket = new (std::nothrow) MetalBucketImpl;
        if (!bucket) return GAFIME_ERROR_OUT_OF_MEMORY;
        
        memset(bucket, 0, sizeof(MetalBucketImpl));
        bucket->n_samples = n_samples;
        bucket->n_features = n_features;
        
        size_t float_bytes = (size_t)n_samples * sizeof(float);
        size_t mask_bytes = (size_t)n_samples * sizeof(uint8_t);
        
        // Allocate shared-mode buffers (UMA zero-copy)
        // storageModeShared: CPU and GPU share the same physical memory
        MTLResourceOptions opts = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        
        for (int i = 0; i < n_features; i++) {
            bucket->feature_buffers[i] = [g_device newBufferWithLength:float_bytes
                                                               options:opts];
            if (!bucket->feature_buffers[i]) {
                gafime_metal_bucket_free(bucket);
                return GAFIME_ERROR_OUT_OF_MEMORY;
            }
        }
        
        bucket->target_buffer = [g_device newBufferWithLength:float_bytes options:opts];
        bucket->mask_buffer = [g_device newBufferWithLength:mask_bytes options:opts];
        bucket->stats_buffer = [g_device newBufferWithLength:12 * sizeof(float) options:opts];
        bucket->params_buffer = [g_device newBufferWithLength:sizeof(FusedParams) options:opts];
        
        if (!bucket->target_buffer || !bucket->mask_buffer ||
            !bucket->stats_buffer || !bucket->params_buffer) {
            gafime_metal_bucket_free(bucket);
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }
        
        // Create compute pipeline for fused kernel
        NSError* error = nil;
        id<MTLFunction> func = [g_library newFunctionWithName:@"gafime_fused_kernel"];
        if (!func) {
            fprintf(stderr, "GAFIME Metal: Failed to find gafime_fused_kernel function\n");
            gafime_metal_bucket_free(bucket);
            return GAFIME_ERROR_KERNEL_FAILED;
        }
        
        bucket->fused_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!bucket->fused_pipeline) {
            fprintf(stderr, "GAFIME Metal: Failed to create pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            gafime_metal_bucket_free(bucket);
            return GAFIME_ERROR_KERNEL_FAILED;
        }
        
        *bucket_out = (GafimeMetalBucket)bucket;
    }
    
    return GAFIME_SUCCESS;
}

int gafime_metal_bucket_upload_feature(
    GafimeMetalBucket bucket_handle,
    int feature_index,
    const float* data,
    int n_samples
) {
    if (!bucket_handle || !data) return GAFIME_ERROR_INVALID_ARGS;
    
    MetalBucketImpl* bucket = (MetalBucketImpl*)bucket_handle;
    if (feature_index < 0 || feature_index >= bucket->n_features) return GAFIME_ERROR_INVALID_ARGS;
    if (n_samples != bucket->n_samples) return GAFIME_ERROR_INVALID_ARGS;
    
    // UMA zero-copy: this is a RAM→RAM memcpy, NOT a PCIe transfer
    memcpy([bucket->feature_buffers[feature_index] contents], data, (size_t)n_samples * sizeof(float));
    
    return GAFIME_SUCCESS;
}

int gafime_metal_bucket_upload_target(
    GafimeMetalBucket bucket_handle,
    const float* data,
    int n_samples
) {
    if (!bucket_handle || !data) return GAFIME_ERROR_INVALID_ARGS;
    
    MetalBucketImpl* bucket = (MetalBucketImpl*)bucket_handle;
    if (n_samples != bucket->n_samples) return GAFIME_ERROR_INVALID_ARGS;
    
    memcpy([bucket->target_buffer contents], data, (size_t)n_samples * sizeof(float));
    return GAFIME_SUCCESS;
}

int gafime_metal_bucket_upload_mask(
    GafimeMetalBucket bucket_handle,
    const uint8_t* data,
    int n_samples
) {
    if (!bucket_handle || !data) return GAFIME_ERROR_INVALID_ARGS;
    
    MetalBucketImpl* bucket = (MetalBucketImpl*)bucket_handle;
    if (n_samples != bucket->n_samples) return GAFIME_ERROR_INVALID_ARGS;
    
    memcpy([bucket->mask_buffer contents], data, (size_t)n_samples * sizeof(uint8_t));
    return GAFIME_SUCCESS;
}

// ============================================================================
// COMPUTE DISPATCH
// ============================================================================

int gafime_metal_bucket_compute(
    GafimeMetalBucket bucket_handle,
    const int* ops,
    int arity,
    const int* interaction_types,
    int val_fold_id,
    float* stats_out
) {
    if (!bucket_handle || !ops || !interaction_types || !stats_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (arity < 2 || arity > 5) return GAFIME_ERROR_INVALID_ARGS;
    
    MetalBucketImpl* bucket = (MetalBucketImpl*)bucket_handle;
    
    @autoreleasepool {
        // Zero out stats buffer
        memset([bucket->stats_buffer contents], 0, 12 * sizeof(float));
        
        // Fill params
        FusedParams* params = (FusedParams*)[bucket->params_buffer contents];
        memset(params, 0, sizeof(FusedParams));
        for (int i = 0; i < arity; i++) params->ops[i] = ops[i];
        for (int i = 0; i < arity - 1; i++) params->interaction_types[i] = interaction_types[i];
        params->arity = arity;
        params->val_fold_id = val_fold_id;
        params->n_samples = bucket->n_samples;
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:bucket->fused_pipeline];
        
        // Bind buffers — feature buffers use index 0-4
        for (int i = 0; i < 5; i++) {
            if (i < arity) {
                [encoder setBuffer:bucket->feature_buffers[i] offset:0 atIndex:i];
            } else {
                // Bind a dummy buffer for unused feature slots
                [encoder setBuffer:bucket->feature_buffers[0] offset:0 atIndex:i];
            }
        }
        [encoder setBuffer:bucket->target_buffer offset:0 atIndex:5];
        [encoder setBuffer:bucket->mask_buffer   offset:0 atIndex:6];
        [encoder setBuffer:bucket->params_buffer offset:0 atIndex:7];
        [encoder setBuffer:bucket->stats_buffer  offset:0 atIndex:8];
        
        // Calculate dispatch dimensions
        NSUInteger threadGroupSize = bucket->fused_pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;  // Cap to 256 like CUDA
        
        NSUInteger numThreadGroups = ((NSUInteger)bucket->n_samples + threadGroupSize - 1) / threadGroupSize;
        if (numThreadGroups > 1024) numThreadGroups = 1024; // Cap like CUDA
        
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadGroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        // Check for GPU errors
        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "GAFIME Metal: GPU error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }
        
        // Read back stats (UMA: already in shared memory)
        memcpy(stats_out, [bucket->stats_buffer contents], 12 * sizeof(float));
    }
    
    return GAFIME_SUCCESS;
}

int gafime_metal_bucket_free(GafimeMetalBucket bucket_handle) {
    if (!bucket_handle) return GAFIME_SUCCESS;
    
    MetalBucketImpl* bucket = (MetalBucketImpl*)bucket_handle;
    
    // ARC handles MTLBuffer release automatically when set to nil
    for (int i = 0; i < 5; i++) {
        bucket->feature_buffers[i] = nil;
    }
    bucket->target_buffer = nil;
    bucket->mask_buffer = nil;
    bucket->stats_buffer = nil;
    bucket->params_buffer = nil;
    bucket->fused_pipeline = nil;
    
    delete bucket;
    return GAFIME_SUCCESS;
}

// ============================================================================
// STANDALONE FUSED API
// ============================================================================

int gafime_metal_fused_interaction(
    const float** h_inputs,
    const float* h_target,
    const uint8_t* h_mask,
    const int* h_ops,
    int arity,
    int interaction_type,
    int val_fold_id,
    int n_samples,
    float* h_stats
) {
    if (arity < 2 || arity > 5) return GAFIME_ERROR_INVALID_ARGS;
    if (!h_inputs || !h_target || !h_mask || !h_ops || !h_stats) return GAFIME_ERROR_INVALID_ARGS;
    if (n_samples <= 0) return GAFIME_ERROR_INVALID_ARGS;
    
    // Use bucket API internally (allocate → upload → compute → free)
    GafimeMetalBucket bucket = nullptr;
    int ret = gafime_metal_bucket_alloc(n_samples, arity, &bucket);
    if (ret != GAFIME_SUCCESS) return ret;
    
    // Upload features
    for (int i = 0; i < arity; i++) {
        ret = gafime_metal_bucket_upload_feature(bucket, i, h_inputs[i], n_samples);
        if (ret != GAFIME_SUCCESS) { gafime_metal_bucket_free(bucket); return ret; }
    }
    
    // Upload target and mask
    ret = gafime_metal_bucket_upload_target(bucket, h_target, n_samples);
    if (ret != GAFIME_SUCCESS) { gafime_metal_bucket_free(bucket); return ret; }
    
    ret = gafime_metal_bucket_upload_mask(bucket, h_mask, n_samples);
    if (ret != GAFIME_SUCCESS) { gafime_metal_bucket_free(bucket); return ret; }
    
    // Build uniform interaction_types array
    int interact_types[4] = {interaction_type, interaction_type, interaction_type, interaction_type};
    
    // Compute
    ret = gafime_metal_bucket_compute(bucket, h_ops, arity, interact_types, val_fold_id, h_stats);
    
    gafime_metal_bucket_free(bucket);
    return ret;
}

// ============================================================================
// DISCRETE SOFT FUNCTION FAMILY
// ============================================================================

int gafime_metal_discrete_soft_batch(
    const float* h_X,
    const float* h_y,
    const int* h_kinds,
    const int* h_feature_a,
    const int* h_feature_b,
    const int* h_value_feature,
    const int* h_directions,
    const float* h_params,
    const float* h_scales,
    const float* h_sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    float* h_stats
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!h_X || !h_y || !h_kinds || !h_feature_a || !h_feature_b ||
        !h_value_feature || !h_directions || !h_params || !h_scales ||
        !h_sharpness || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_candidates <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    for (int i = 0; i < n_candidates; i++) {
        int kind = h_kinds[i];
        int fa = h_feature_a[i];
        int fb = h_feature_b[i];
        int vf = h_value_feature[i];
        if (kind < GAFIME_DISCRETE_SOFT_THRESHOLD ||
            kind > GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE ||
            fa < 0 || fa >= n_features) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
        if ((kind == GAFIME_DISCRETE_SOFT_RECTANGLE ||
             kind == GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE) &&
            (fb < 0 || fb >= n_features)) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
        if ((kind == GAFIME_DISCRETE_VALUE_GATED_THRESHOLD ||
             kind == GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE) &&
            (vf < 0 || vf >= n_features)) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
    }

    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> func = [g_library newFunctionWithName:@"gafime_discrete_soft_batch_kernel"];
        if (!func) {
            fprintf(stderr, "GAFIME Metal: Failed to find gafime_discrete_soft_batch_kernel function\n");
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            fprintf(stderr, "GAFIME Metal: Failed to create discrete pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        size_t X_bytes = (size_t)n_samples * (size_t)n_features * sizeof(float);
        size_t y_bytes = (size_t)n_samples * sizeof(float);
        size_t int_bytes = (size_t)n_candidates * sizeof(int);
        size_t params_bytes = (size_t)n_candidates * 4 * sizeof(float);
        size_t scales_bytes = (size_t)n_candidates * 2 * sizeof(float);
        size_t sharpness_bytes = (size_t)n_candidates * sizeof(float);
        size_t stats_bytes = (size_t)n_candidates * 12 * sizeof(float);

        MTLResourceOptions opts = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        id<MTLBuffer> X_buf = [g_device newBufferWithLength:X_bytes options:opts];
        id<MTLBuffer> y_buf = [g_device newBufferWithLength:y_bytes options:opts];
        id<MTLBuffer> kinds_buf = [g_device newBufferWithLength:int_bytes options:opts];
        id<MTLBuffer> feature_a_buf = [g_device newBufferWithLength:int_bytes options:opts];
        id<MTLBuffer> feature_b_buf = [g_device newBufferWithLength:int_bytes options:opts];
        id<MTLBuffer> value_feature_buf = [g_device newBufferWithLength:int_bytes options:opts];
        id<MTLBuffer> directions_buf = [g_device newBufferWithLength:int_bytes options:opts];
        id<MTLBuffer> params_buf = [g_device newBufferWithLength:params_bytes options:opts];
        id<MTLBuffer> scales_buf = [g_device newBufferWithLength:scales_bytes options:opts];
        id<MTLBuffer> sharpness_buf = [g_device newBufferWithLength:sharpness_bytes options:opts];
        id<MTLBuffer> batch_params_buf = [g_device newBufferWithLength:sizeof(DiscreteBatchParams) options:opts];
        id<MTLBuffer> stats_buf = [g_device newBufferWithLength:stats_bytes options:opts];

        if (!X_buf || !y_buf || !kinds_buf || !feature_a_buf || !feature_b_buf ||
            !value_feature_buf || !directions_buf || !params_buf || !scales_buf ||
            !sharpness_buf || !batch_params_buf || !stats_buf) {
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }

        memcpy([X_buf contents], h_X, X_bytes);
        memcpy([y_buf contents], h_y, y_bytes);
        memcpy([kinds_buf contents], h_kinds, int_bytes);
        memcpy([feature_a_buf contents], h_feature_a, int_bytes);
        memcpy([feature_b_buf contents], h_feature_b, int_bytes);
        memcpy([value_feature_buf contents], h_value_feature, int_bytes);
        memcpy([directions_buf contents], h_directions, int_bytes);
        memcpy([params_buf contents], h_params, params_bytes);
        memcpy([scales_buf contents], h_scales, scales_bytes);
        memcpy([sharpness_buf contents], h_sharpness, sharpness_bytes);
        memset([stats_buf contents], 0, stats_bytes);

        DiscreteBatchParams* params = (DiscreteBatchParams*)[batch_params_buf contents];
        params->n_samples = n_samples;
        params->n_features = n_features;
        params->n_candidates = n_candidates;
        params->padding = 0;

        id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:X_buf offset:0 atIndex:0];
        [encoder setBuffer:y_buf offset:0 atIndex:1];
        [encoder setBuffer:kinds_buf offset:0 atIndex:2];
        [encoder setBuffer:feature_a_buf offset:0 atIndex:3];
        [encoder setBuffer:feature_b_buf offset:0 atIndex:4];
        [encoder setBuffer:value_feature_buf offset:0 atIndex:5];
        [encoder setBuffer:directions_buf offset:0 atIndex:6];
        [encoder setBuffer:params_buf offset:0 atIndex:7];
        [encoder setBuffer:scales_buf offset:0 atIndex:8];
        [encoder setBuffer:sharpness_buf offset:0 atIndex:9];
        [encoder setBuffer:batch_params_buf offset:0 atIndex:10];
        [encoder setBuffer:stats_buf offset:0 atIndex:11];

        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        NSUInteger numThreadGroups = ((NSUInteger)n_samples + threadGroupSize - 1) / threadGroupSize;
        if (numThreadGroups > 1024) numThreadGroups = 1024;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadGroups, (NSUInteger)n_candidates, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "GAFIME Metal: discrete GPU error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        memcpy(h_stats, [stats_buf contents], stats_bytes);
    }

    return GAFIME_SUCCESS;
}

int gafime_metal_discrete_selection_adaptive(
    const float* h_X,
    const float* h_y,
    const float* h_residual,
    const int* h_y_bins,
    const int* h_kinds,
    const int* h_feature_a,
    const int* h_feature_b,
    const int* h_value_feature,
    const int* h_directions,
    const float* h_params,
    const float* h_scales,
    const float* h_sharpness,
    int n_samples,
    int n_features,
    int n_candidates,
    int target_bin_template,
    float y_sum,
    float y_sq_sum,
    float* h_scores
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!h_X || !h_y || !h_residual || !h_y_bins || !h_kinds || !h_feature_a ||
        !h_feature_b || !h_value_feature || !h_directions || !h_params ||
        !h_scales || !h_sharpness || !h_scores) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_candidates <= 0 ||
        (target_bin_template != 2 && target_bin_template != 4 &&
         target_bin_template != 8 && target_bin_template != 16 &&
         target_bin_template != 32 && target_bin_template != 64 &&
         target_bin_template != 96)) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    for (int i = 0; i < n_candidates; i++) {
        int kind = h_kinds[i];
        int fa = h_feature_a[i];
        int fb = h_feature_b[i];
        int vf = h_value_feature[i];
        if (kind < GAFIME_DISCRETE_SOFT_THRESHOLD ||
            kind > GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE ||
            fa < 0 || fa >= n_features) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
        if ((kind == GAFIME_DISCRETE_SOFT_RECTANGLE ||
             kind == GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE) &&
            (fb < 0 || fb >= n_features)) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
        if ((kind == GAFIME_DISCRETE_VALUE_GATED_THRESHOLD ||
             kind == GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE) &&
            (vf < 0 || vf >= n_features)) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
    }

    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> func = [g_library newFunctionWithName:@"gafime_discrete_selection_adaptive_kernel"];
        if (!func) {
            fprintf(stderr, "GAFIME Metal: Failed to find gafime_discrete_selection_adaptive_kernel function\n");
            return GAFIME_ERROR_KERNEL_FAILED;
        }
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            fprintf(stderr, "GAFIME Metal: Failed to create discrete selection pipeline: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        size_t X_bytes = (size_t)n_samples * (size_t)n_features * sizeof(float);
        size_t vec_bytes = (size_t)n_samples * sizeof(float);
        size_t sample_int_bytes = (size_t)n_samples * sizeof(int);
        size_t int_bytes = (size_t)n_candidates * sizeof(int);
        size_t params_bytes = (size_t)n_candidates * 4 * sizeof(float);
        size_t scales_bytes = (size_t)n_candidates * 2 * sizeof(float);
        size_t sharpness_bytes = (size_t)n_candidates * sizeof(float);
        size_t scores_bytes = (size_t)n_candidates * GAFIME_SELECTION_SCORE_SIZE * sizeof(float);

        MTLResourceOptions shared_write = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        MTLResourceOptions shared_read = MTLResourceStorageModeShared;
        id<MTLBuffer> X_buf = [g_device newBufferWithLength:X_bytes options:shared_write];
        id<MTLBuffer> y_buf = [g_device newBufferWithLength:vec_bytes options:shared_write];
        id<MTLBuffer> residual_buf = [g_device newBufferWithLength:vec_bytes options:shared_write];
        id<MTLBuffer> y_bins_buf = [g_device newBufferWithLength:sample_int_bytes options:shared_write];
        id<MTLBuffer> kinds_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> feature_a_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> feature_b_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> value_feature_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> directions_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> params_buf = [g_device newBufferWithLength:params_bytes options:shared_write];
        id<MTLBuffer> scales_buf = [g_device newBufferWithLength:scales_bytes options:shared_write];
        id<MTLBuffer> sharpness_buf = [g_device newBufferWithLength:sharpness_bytes options:shared_write];
        id<MTLBuffer> selection_params_buf = [g_device newBufferWithLength:sizeof(DiscreteSelectionParams) options:shared_write];
        id<MTLBuffer> scores_buf = [g_device newBufferWithLength:scores_bytes options:shared_read];

        if (!X_buf || !y_buf || !residual_buf || !y_bins_buf || !kinds_buf ||
            !feature_a_buf || !feature_b_buf || !value_feature_buf ||
            !directions_buf || !params_buf || !scales_buf || !sharpness_buf ||
            !selection_params_buf || !scores_buf) {
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }

        memcpy([X_buf contents], h_X, X_bytes);
        memcpy([y_buf contents], h_y, vec_bytes);
        memcpy([residual_buf contents], h_residual, vec_bytes);
        memcpy([y_bins_buf contents], h_y_bins, sample_int_bytes);
        memcpy([kinds_buf contents], h_kinds, int_bytes);
        memcpy([feature_a_buf contents], h_feature_a, int_bytes);
        memcpy([feature_b_buf contents], h_feature_b, int_bytes);
        memcpy([value_feature_buf contents], h_value_feature, int_bytes);
        memcpy([directions_buf contents], h_directions, int_bytes);
        memcpy([params_buf contents], h_params, params_bytes);
        memcpy([scales_buf contents], h_scales, scales_bytes);
        memcpy([sharpness_buf contents], h_sharpness, sharpness_bytes);
        memset([scores_buf contents], 0, scores_bytes);

        DiscreteSelectionParams* params = (DiscreteSelectionParams*)[selection_params_buf contents];
        params->n_samples = n_samples;
        params->n_features = n_features;
        params->n_candidates = n_candidates;
        params->target_bins = target_bin_template;
        params->y_sum = y_sum;
        params->y_sq_sum = y_sq_sum;
        params->padding0 = 0;
        params->padding1 = 0;

        id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:X_buf offset:0 atIndex:0];
        [encoder setBuffer:y_buf offset:0 atIndex:1];
        [encoder setBuffer:residual_buf offset:0 atIndex:2];
        [encoder setBuffer:y_bins_buf offset:0 atIndex:3];
        [encoder setBuffer:kinds_buf offset:0 atIndex:4];
        [encoder setBuffer:feature_a_buf offset:0 atIndex:5];
        [encoder setBuffer:feature_b_buf offset:0 atIndex:6];
        [encoder setBuffer:value_feature_buf offset:0 atIndex:7];
        [encoder setBuffer:directions_buf offset:0 atIndex:8];
        [encoder setBuffer:params_buf offset:0 atIndex:9];
        [encoder setBuffer:scales_buf offset:0 atIndex:10];
        [encoder setBuffer:sharpness_buf offset:0 atIndex:11];
        [encoder setBuffer:selection_params_buf offset:0 atIndex:12];
        [encoder setBuffer:scores_buf offset:0 atIndex:13];

        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize < 256) return GAFIME_ERROR_KERNEL_FAILED;
        threadGroupSize = 256;
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)n_candidates, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "GAFIME Metal: discrete selection GPU error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        memcpy(h_scores, [scores_buf contents], scores_bytes);
    }

    return GAFIME_SUCCESS;
}

int gafime_metal_time_series_batch(
    const float* h_X,
    const float* h_y,
    const int* h_kinds,
    const int* h_feature_index,
    const int* h_lags,
    const int* h_windows,
    int n_samples,
    int n_features,
    int n_candidates,
    float* h_stats
) {
    if (!metal_init()) return GAFIME_ERROR_METAL_NOT_AVAILABLE;
    if (!h_X || !h_y || !h_kinds || !h_feature_index || !h_lags ||
        !h_windows || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_candidates <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    for (int i = 0; i < n_candidates; i++) {
        int kind = h_kinds[i];
        int feature = h_feature_index[i];
        if (kind < 1 || kind > 7 || feature < 0 || feature >= n_features) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
    }

    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> func = [g_library newFunctionWithName:@"gafime_time_series_batch_kernel"];
        if (!func) {
            fprintf(stderr, "GAFIME Metal: Failed to find gafime_time_series_batch_kernel function\n");
            return GAFIME_ERROR_KERNEL_FAILED;
        }
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            fprintf(stderr, "GAFIME Metal: Failed to create time-series pipeline: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        size_t X_bytes = (size_t)n_samples * (size_t)n_features * sizeof(float);
        size_t y_bytes = (size_t)n_samples * sizeof(float);
        size_t int_bytes = (size_t)n_candidates * sizeof(int);
        size_t stats_bytes = (size_t)n_candidates * 12 * sizeof(float);

        MTLResourceOptions shared_write = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        MTLResourceOptions shared_read = MTLResourceStorageModeShared;
        id<MTLBuffer> X_buf = [g_device newBufferWithLength:X_bytes options:shared_write];
        id<MTLBuffer> y_buf = [g_device newBufferWithLength:y_bytes options:shared_write];
        id<MTLBuffer> kinds_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> feature_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> lags_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> windows_buf = [g_device newBufferWithLength:int_bytes options:shared_write];
        id<MTLBuffer> params_buf = [g_device newBufferWithLength:sizeof(TimeSeriesBatchParams) options:shared_write];
        id<MTLBuffer> stats_buf = [g_device newBufferWithLength:stats_bytes options:shared_read];

        if (!X_buf || !y_buf || !kinds_buf || !feature_buf || !lags_buf ||
            !windows_buf || !params_buf || !stats_buf) {
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }

        memcpy([X_buf contents], h_X, X_bytes);
        memcpy([y_buf contents], h_y, y_bytes);
        memcpy([kinds_buf contents], h_kinds, int_bytes);
        memcpy([feature_buf contents], h_feature_index, int_bytes);
        memcpy([lags_buf contents], h_lags, int_bytes);
        memcpy([windows_buf contents], h_windows, int_bytes);
        memset([stats_buf contents], 0, stats_bytes);

        TimeSeriesBatchParams* params = (TimeSeriesBatchParams*)[params_buf contents];
        params->n_samples = n_samples;
        params->n_features = n_features;
        params->n_candidates = n_candidates;
        params->padding = 0;

        id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:X_buf offset:0 atIndex:0];
        [encoder setBuffer:y_buf offset:0 atIndex:1];
        [encoder setBuffer:kinds_buf offset:0 atIndex:2];
        [encoder setBuffer:feature_buf offset:0 atIndex:3];
        [encoder setBuffer:lags_buf offset:0 atIndex:4];
        [encoder setBuffer:windows_buf offset:0 atIndex:5];
        [encoder setBuffer:params_buf offset:0 atIndex:6];
        [encoder setBuffer:stats_buf offset:0 atIndex:7];

        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        NSUInteger numThreadGroups = ((NSUInteger)n_samples + threadGroupSize - 1) / threadGroupSize;
        if (numThreadGroups > 1024) numThreadGroups = 1024;
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadGroups, (NSUInteger)n_candidates, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "GAFIME Metal: time-series GPU error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return GAFIME_ERROR_KERNEL_FAILED;
        }

        memcpy(h_stats, [stats_buf contents], stats_bytes);
    }

    return GAFIME_SUCCESS;
}

} // extern "C"
