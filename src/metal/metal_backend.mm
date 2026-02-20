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

// ============================================================================
// SINGLETON DEVICE + LIBRARY (initialized once)
// ============================================================================

static id<MTLDevice>        g_device        = nil;
static id<MTLCommandQueue>  g_command_queue = nil;
static id<MTLLibrary>       g_library       = nil;
static bool                 g_initialized   = false;

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
        
        // Reject Intel Macs — they don't have UMA or Metal 3.0
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            // Apple7 = M1 and later (Apple Silicon)
            g_device = nil;
            return false;
        }
        
        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            g_device = nil;
            return false;
        }
        
        // Load pre-compiled Metal shader library (metallib)
        // Try multiple search paths
        NSArray<NSString*>* paths = @[
            // Same directory as the dylib
            [[[NSBundle mainBundle] bundlePath]
                stringByAppendingPathComponent:@"gafime_kernels.metallib"],
            // Alongside the Python package
            @"gafime_kernels.metallib",
        ];
        
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
                        options.fastMathEnabled = YES;
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

} // extern "C"
