/**
 * GAFIME CUDA Kernels - Operator-Fused Map-Reduce Architecture
 * 
 * AUTO-TUNING for different GPU architectures:
 * - Queries GPU properties at runtime
 * - Adjusts block size, grid size based on SM count
 * - Optimizes for compute capability
 * 
 * Design Philosophy:
 * 1. Fused Operations: Apply unary ops + interaction in single pass
 * 2. On-Chip Reduction: Accumulate stats in registers, NOT global memory
 * 3. Train/Val Split: Use byte mask for cross-validation fold separation
 * 4. Output: Only 12 floats (6 train + 6 val statistics)
 * 
 * Statistics accumulated: N, ΣX, ΣY, ΣX², ΣY², ΣXY
 * Pearson formula: r = (NΣxy - ΣxΣy) / sqrt((NΣx² - (Σx)²)(NΣy² - (Σy)²))
 */

#include "interfaces.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <new>  // for std::nothrow

#define MAX_BATCH_SLOTS 4096

// ============================================================================
// GPU AUTO-DETECTION AND AUTO-TUNING SYSTEM
// ============================================================================

// Default values (will be overridden by auto-tune)
#define DEFAULT_BLOCK_SIZE 256
#define WARP_SIZE 32

// Cached GPU configuration (set once on first kernel call)
struct GpuConfig {
    int block_size;           // Optimal threads per block
    int max_blocks;           // Max blocks for grid
    int sm_count;             // Number of streaming multiprocessors
    int compute_major;        // Compute capability major
    int compute_minor;        // Compute capability minor
    int l2_cache_size;        // L2 cache size in bytes
    int max_shared_memory;    // Max shared memory per block
    int warp_size;            // Warp size (always 32 for NVIDIA)
    bool is_initialized;      // Config has been set
    char gpu_name[256];       // GPU name for logging
};

static GpuConfig g_gpu_config = {
    DEFAULT_BLOCK_SIZE, 256, 0, 0, 0, 0, 0, WARP_SIZE, false, ""
};

/**
 * Query GPU properties and set optimal parameters.
 * Called automatically on first kernel invocation.
 */
static void auto_tune_for_gpu(int device_id = 0) {
    if (g_gpu_config.is_initialized) return;
    
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        // Fallback to defaults
        g_gpu_config.is_initialized = true;
        return;
    }
    
    // Store GPU info
    strncpy(g_gpu_config.gpu_name, props.name, 255);
    g_gpu_config.gpu_name[255] = '\0';
    g_gpu_config.sm_count = props.multiProcessorCount;
    g_gpu_config.compute_major = props.major;
    g_gpu_config.compute_minor = props.minor;
    g_gpu_config.l2_cache_size = props.l2CacheSize;
    g_gpu_config.max_shared_memory = props.sharedMemPerBlock;
    g_gpu_config.warp_size = props.warpSize;
    
    // =========================================================================
    // AUTO-TUNE BLOCK SIZE based on compute capability
    // =========================================================================
    int compute_cap = props.major * 10 + props.minor;
    
    if (compute_cap >= 89) {
        // Ada Lovelace (RTX 40 series) - 128 CUDA cores per SM
        g_gpu_config.block_size = 256;  // 8 warps, good occupancy
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else if (compute_cap >= 80) {
        // Ampere (RTX 30 series, A100) - 64/128 cores per SM
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else if (compute_cap >= 75) {
        // Turing (RTX 20 series) - 64 cores per SM
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 2;
    } else if (compute_cap >= 60) {
        // Pascal (GTX 10 series) - 64/128 cores per SM
        g_gpu_config.block_size = 128;
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else {
        // Older architectures
        g_gpu_config.block_size = 128;
        g_gpu_config.max_blocks = props.multiProcessorCount * 2;
    }
    
    g_gpu_config.is_initialized = true;
    
    // Log GPU info (optional, helps with debugging)
    fprintf(stderr, "[GAFIME] Auto-tuned for: %s\n", props.name);
    fprintf(stderr, "[GAFIME]   SM count: %d, Compute: %d.%d\n", 
            props.multiProcessorCount, props.major, props.minor);
    fprintf(stderr, "[GAFIME]   Block size: %d, Max blocks: %d\n",
            g_gpu_config.block_size, g_gpu_config.max_blocks);
    fprintf(stderr, "[GAFIME]   L2 cache: %.1f MB, Shared mem: %d KB\n",
            props.l2CacheSize / (1024.0 * 1024.0), props.sharedMemPerBlock / 1024);
}

/**
 * Get current GPU configuration (for Python introspection)
 */
GAFIME_API int gafime_get_gpu_config(
    int* block_size_out,
    int* max_blocks_out,
    int* sm_count_out,
    int* compute_major_out,
    int* compute_minor_out,
    int* l2_cache_bytes_out,
    char* gpu_name_out
) {
    auto_tune_for_gpu();
    
    if (block_size_out) *block_size_out = g_gpu_config.block_size;
    if (max_blocks_out) *max_blocks_out = g_gpu_config.max_blocks;
    if (sm_count_out) *sm_count_out = g_gpu_config.sm_count;
    if (compute_major_out) *compute_major_out = g_gpu_config.compute_major;
    if (compute_minor_out) *compute_minor_out = g_gpu_config.compute_minor;
    if (l2_cache_bytes_out) *l2_cache_bytes_out = g_gpu_config.l2_cache_size;
    if (gpu_name_out) strcpy(gpu_name_out, g_gpu_config.gpu_name);
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// COMPILE-TIME VS RUNTIME TUNING
// ============================================================================
// 
// CUDA Constraint: Shared memory size MUST be a compile-time constant.
// Therefore:
//   - BLOCK_SIZE: Compile-time constant (256), used for shared memory sizing
//   - max_blocks: Runtime-tuned based on GPU SM count
//
// The block size of 256 is optimal for all modern CUDA architectures (Pascal+)
// and provides good occupancy. The real tuning happens in grid dimension.
// ============================================================================

#define BLOCK_SIZE 256  // Compile-time constant for shared memory

// Helper macro to get runtime-tuned max blocks
#define GET_MAX_BLOCKS() (g_gpu_config.is_initialized ? g_gpu_config.max_blocks : 256)

// ============================================================================
// UNARY OPERATORS (Standard math library)
// ============================================================================

/**
 * Apply unary transformation to a single value.
 * Safe implementations prevent NaN/Inf propagation.
 */
__device__ __forceinline__ float apply_op(float x, int op) {
    // Optimized implementation using Fast Intrinsics (SFU)
    switch (op) {
        case GAFIME_OP_LOG:
            // __logf is ~10x faster than logf
            return __logf(fabsf(x) + 1e-8f);
            
        case GAFIME_OP_EXP:
            // __expf is significantly faster, clamp to avoid Inf
            return __expf(fminf(fmaxf(x, -20.0f), 20.0f));
            
        case GAFIME_OP_SQRT:
            // __fsqrt_rn maps directly to hardware unit
            return __fsqrt_rn(fabsf(x));
            
        case GAFIME_OP_TANH: {
            // Fast approximation: tanh(x) = (e^2x - 1) / (e^2x + 1)
            float exp2x = __expf(2.0f * fminf(fmaxf(x, -10.0f), 10.0f));
            return (exp2x - 1.0f) / (exp2x + 1.0f);
        }
            
        case GAFIME_OP_SIGMOID: {
            // Fast sigmoid: 1 / (1 + e^-x)
            float ex = __expf(-fminf(fmaxf(x, -20.0f), 20.0f));
            return __fdividef(1.0f, 1.0f + ex);
        }
            
        case GAFIME_OP_SQUARE:
            return x * x;
            
        case GAFIME_OP_NEGATE:
            return -x;
            
        case GAFIME_OP_ABS:
            return fabsf(x);
            
        case GAFIME_OP_INVERSE:
            // __fdividef is much faster than standard division
            return __fdividef(1.0f, fabsf(x) < 1e-8f ? copysignf(1e-8f, x) : x);
            
        case GAFIME_OP_CUBE:
            return x * x * x;
            
        case GAFIME_OP_IDENTITY:
        default:
            return x;
    }
}

// ============================================================================
// FAST INTRINSICS + TIME-SERIES OPERATORS (for interleaved kernel)
// ============================================================================

/**
 * Apply unary transformation using NVIDIA fast intrinsics.
 * Uses __logf, __expf, __fsqrt_rn for SFU acceleration.
 * Supports rolling window operators for time-series data.
 * 
 * @param col       Pointer to feature column
 * @param idx       Current row index
 * @param n_rows    Total rows (for boundary check)
 * @param op        Operator ID
 * @param window    Window size for rolling ops (0 = point op)
 */
__device__ __forceinline__ float apply_op_fast(
    const float* __restrict__ col, int idx, int n_rows, int op, int window
) {
    float x = col[idx];
    
    switch (op) {
        // SFU-Heavy operations (fast intrinsics)
        case GAFIME_OP_LOG:
            return __logf(fabsf(x) + 1e-8f);
        
        case GAFIME_OP_EXP:
            return __expf(__saturatef(x * 0.05f) * 20.0f);  // Clamp to [-20,20]
        
        case GAFIME_OP_SQRT:
            return __fsqrt_rn(fabsf(x));
        
        case GAFIME_OP_TANH: {
            // Fast tanh approximation using exp intrinsic
            float exp2x = __expf(2.0f * fminf(fmaxf(x, -10.0f), 10.0f));
            return (exp2x - 1.0f) / (exp2x + 1.0f);
        }
        
        case GAFIME_OP_SIGMOID: {
            float ex = __expf(-fminf(fmaxf(x, -20.0f), 20.0f));
            return __fdividef(1.0f, 1.0f + ex);
        }
        
        // ALU-Heavy operations
        case GAFIME_OP_SQUARE:
            return x * x;
        
        case GAFIME_OP_NEGATE:
            return -x;
        
        case GAFIME_OP_ABS:
            return fabsf(x);
        
        case GAFIME_OP_INVERSE:
            return __fdividef(1.0f, fabsf(x) < 1e-8f ? copysignf(1e-8f, x) : x);
        
        case GAFIME_OP_CUBE:
            return x * x * x;
        
        // =====================================================================
        // DEPRECATED: Rolling operators removed from GPU
        // These have O(window) serial memory access per thread, destroying
        // memory coalescing and GPU performance.
        // 
        // Use gafime.preprocessors.TimeSeriesPreprocessor for rolling features.
        // It uses Polars with vectorized ops - 10-50x faster.
        // =====================================================================
        case GAFIME_OP_ROLLING_MEAN:
        case GAFIME_OP_ROLLING_STD:
            // Return NaN to indicate "use CPU preprocessing"
            return NAN;
        
        case GAFIME_OP_IDENTITY:
        default:
            return x;
    }
}

// ============================================================================
// INTERACTION COMBINERS
// ============================================================================

/**
 * Combine two values using the specified interaction type.
 */
__device__ __forceinline__ float combine(float a, float b, int interaction_type) {
    switch (interaction_type) {
        case GAFIME_INTERACT_ADD:
            return a + b;
        
        case GAFIME_INTERACT_SUB:
            return a - b;
        
        case GAFIME_INTERACT_DIV:
            // Safe division
            return a / (fabsf(b) < 1e-8f ? copysignf(1e-8f, b) : b);
        
        case GAFIME_INTERACT_MAX:
            return fmaxf(a, b);
        
        case GAFIME_INTERACT_MIN:
            return fminf(a, b);
        
        case GAFIME_INTERACT_MULT:
        default:
            return a * b;
    }
}

// ============================================================================
// SHARED MEMORY REDUCTION HELPERS
// ============================================================================

/**
 * Warp-level reduction using shuffle instructions.
 */
__device__ __forceinline__ void warp_reduce_6(
    float& n, float& sx, float& sy, float& sxx, float& syy, float& sxy
) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        n   += __shfl_down_sync(0xffffffff, n, offset);
        sx  += __shfl_down_sync(0xffffffff, sx, offset);
        sy  += __shfl_down_sync(0xffffffff, sy, offset);
        sxx += __shfl_down_sync(0xffffffff, sxx, offset);
        syy += __shfl_down_sync(0xffffffff, syy, offset);
        sxy += __shfl_down_sync(0xffffffff, sxy, offset);
    }
}

// ============================================================================
// FUSED MAP-REDUCE KERNEL (Template for Arity)
// ============================================================================

/**
 * Main fused kernel: Transform features, combine, reduce to stats.
 * 
 * Template parameter Arity: Number of input features (2-5)
 * 
 * Inputs are loaded, transformed with unary ops, combined with interaction,
 * then accumulated into train/val statistics based on fold mask.
 */
template<int Arity>
__global__ void gafime_fused_kernel(
    const float* __restrict__ input0,
    const float* __restrict__ input1,
    const float* __restrict__ input2,  // Used if Arity >= 3
    const float* __restrict__ input3,  // Used if Arity >= 4
    const float* __restrict__ input4,  // Used if Arity >= 5
    const float* __restrict__ target,
    const uint8_t* __restrict__ mask,
    int op0, int op1, int op2, int op3, int op4,
    int interact0, int interact1, int interact2, int interact3,  // Per-pair interaction types
    int val_fold_id,
    int N,
    float* __restrict__ global_stats  // Output: 12 floats
) {
    // Thread-local accumulators (registers)
    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;
    float val_n = 0.0f, val_sx = 0.0f, val_sy = 0.0f;
    float val_sxx = 0.0f, val_syy = 0.0f, val_sxy = 0.0f;
    
    // Grid-stride loop for large datasets
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        // Load and transform features
        float x0 = apply_op(input0[i], op0);
        float x1 = apply_op(input1[i], op1);
        
        // Combine based on arity with PER-PAIR interaction types
        float X = combine(x0, x1, interact0);  // First pair uses interact0
        
        if constexpr (Arity >= 3) {
            float x2 = apply_op(input2[i], op2);
            X = combine(X, x2, interact1);  // Second pair uses interact1
        }
        if constexpr (Arity >= 4) {
            float x3 = apply_op(input3[i], op3);
            X = combine(X, x3, interact2);  // Third pair uses interact2
        }
        if constexpr (Arity >= 5) {
            float x4 = apply_op(input4[i], op4);
            X = combine(X, x4, interact3);  // Fourth pair uses interact3
        }
        
        float Y = target[i];
        uint8_t fold = mask[i];
        
        // Accumulate to appropriate split
        if (fold == val_fold_id) {
            val_n += 1.0f;
            val_sx += X;
            val_sy += Y;
            val_sxx += X * X;
            val_syy += Y * Y;
            val_sxy += X * Y;
        } else {
            train_n += 1.0f;
            train_sx += X;
            train_sy += Y;
            train_sxx += X * X;
            train_syy += Y * Y;
            train_sxy += X * Y;
        }
    }
    
    // Warp-level reduction
    warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    warp_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);
    
    // Shared memory for block-level reduction
    __shared__ float shared_train[6 * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ float shared_val[6 * (BLOCK_SIZE / WARP_SIZE)];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_train[warp_id * 6 + 0] = train_n;
        shared_train[warp_id * 6 + 1] = train_sx;
        shared_train[warp_id * 6 + 2] = train_sy;
        shared_train[warp_id * 6 + 3] = train_sxx;
        shared_train[warp_id * 6 + 4] = train_syy;
        shared_train[warp_id * 6 + 5] = train_sxy;
        
        shared_val[warp_id * 6 + 0] = val_n;
        shared_val[warp_id * 6 + 1] = val_sx;
        shared_val[warp_id * 6 + 2] = val_sy;
        shared_val[warp_id * 6 + 3] = val_sxx;
        shared_val[warp_id * 6 + 4] = val_syy;
        shared_val[warp_id * 6 + 5] = val_sxy;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0 && lane < num_warps) {
        train_n = shared_train[lane * 6 + 0];
        train_sx = shared_train[lane * 6 + 1];
        train_sy = shared_train[lane * 6 + 2];
        train_sxx = shared_train[lane * 6 + 3];
        train_syy = shared_train[lane * 6 + 4];
        train_sxy = shared_train[lane * 6 + 5];
        
        val_n = shared_val[lane * 6 + 0];
        val_sx = shared_val[lane * 6 + 1];
        val_sy = shared_val[lane * 6 + 2];
        val_sxx = shared_val[lane * 6 + 3];
        val_syy = shared_val[lane * 6 + 4];
        val_sxy = shared_val[lane * 6 + 5];
        
        // Reduce across warps
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            train_n += __shfl_down_sync(0xffffffff, train_n, offset);
            train_sx += __shfl_down_sync(0xffffffff, train_sx, offset);
            train_sy += __shfl_down_sync(0xffffffff, train_sy, offset);
            train_sxx += __shfl_down_sync(0xffffffff, train_sxx, offset);
            train_syy += __shfl_down_sync(0xffffffff, train_syy, offset);
            train_sxy += __shfl_down_sync(0xffffffff, train_sxy, offset);
            
            val_n += __shfl_down_sync(0xffffffff, val_n, offset);
            val_sx += __shfl_down_sync(0xffffffff, val_sx, offset);
            val_sy += __shfl_down_sync(0xffffffff, val_sy, offset);
            val_sxx += __shfl_down_sync(0xffffffff, val_sxx, offset);
            val_syy += __shfl_down_sync(0xffffffff, val_syy, offset);
            val_sxy += __shfl_down_sync(0xffffffff, val_sxy, offset);
        }
        
        // Thread 0 writes to global memory with atomics
        if (lane == 0) {
            atomicAdd(&global_stats[0], train_n);
            atomicAdd(&global_stats[1], train_sx);
            atomicAdd(&global_stats[2], train_sy);
            atomicAdd(&global_stats[3], train_sxx);
            atomicAdd(&global_stats[4], train_syy);
            atomicAdd(&global_stats[5], train_sxy);
            
            atomicAdd(&global_stats[6], val_n);
            atomicAdd(&global_stats[7], val_sx);
            atomicAdd(&global_stats[8], val_sy);
            atomicAdd(&global_stats[9], val_sxx);
            atomicAdd(&global_stats[10], val_syy);
            atomicAdd(&global_stats[11], val_sxy);
        }
    }
}

// ============================================================================
// HOST API (extern "C" for ctypes)
// ============================================================================

extern "C" {

GAFIME_API int gafime_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

GAFIME_API int gafime_get_device_info(
    int device_id,
    char* name_out,
    int* memory_mb_out,
    int* compute_cap_major_out,
    int* compute_cap_minor_out
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return GAFIME_ERROR_CUDA_NOT_AVAILABLE;
    }
    
    if (name_out) {
        strncpy(name_out, prop.name, 255);
        name_out[255] = '\0';
    }
    if (memory_mb_out) {
        *memory_mb_out = static_cast<int>(prop.totalGlobalMem / (1024 * 1024));
    }
    if (compute_cap_major_out) {
        *compute_cap_major_out = prop.major;
    }
    if (compute_cap_minor_out) {
        *compute_cap_minor_out = prop.minor;
    }
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// STATIC VRAM BUCKET IMPLEMENTATION
// ============================================================================

/**
 * Internal bucket structure - holds pre-allocated device memory.
 */
struct GafimeBucketImpl {
    int n_samples;
    int n_features;
    float* d_features[GAFIME_MAX_FEATURES];  // Device pointers to feature columns
    float* d_target;                          // Device pointer to target vector
    uint8_t* d_mask;                          // Device pointer to fold mask
    float* d_stats;                           // Device pointer to stats output A (12 floats)
    float* d_stats_B;                         // Device pointer to stats output B (12 floats) for interleaved
    
    // Priority 4: Async operations
    cudaStream_t stream;                      // Compute stream for async operations
    float* h_stats_pinned;                    // Pinned host memory for zero-copy D2H
    
    // Priority 5: L2 cache hints (stored for potential future use)
    size_t total_data_bytes;                  // Total bytes of feature data
};

GAFIME_API int gafime_bucket_alloc(
    int n_samples,
    int n_features,
    GafimeBucket* bucket_out
) {
    if (n_samples <= 0 || n_features <= 0 || n_features > GAFIME_MAX_FEATURES) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (!bucket_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Allocate bucket struct on host
    GafimeBucketImpl* bucket = new (std::nothrow) GafimeBucketImpl;
    if (!bucket) {
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    bucket->n_samples = n_samples;
    bucket->n_features = n_features;
    bucket->d_target = nullptr;
    bucket->d_mask = nullptr;
    bucket->d_stats = nullptr;
    bucket->d_stats_B = nullptr;
    bucket->stream = nullptr;
    bucket->h_stats_pinned = nullptr;
    for (int i = 0; i < GAFIME_MAX_FEATURES; i++) {
        bucket->d_features[i] = nullptr;
    }
    
    size_t vec_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t mask_bytes = static_cast<size_t>(n_samples) * sizeof(uint8_t);
    cudaError_t err;
    
    // =========================================================================
    // Priority 4: Create CUDA stream for async operations
    // =========================================================================
    err = cudaStreamCreate(&bucket->stream);
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Allocate pinned host memory for zero-copy D2H (12 floats * 2 for A+B)
    err = cudaMallocHost(&bucket->h_stats_pinned, 24 * sizeof(float));
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate feature columns
    for (int i = 0; i < n_features; i++) {
        err = cudaMalloc(&bucket->d_features[i], vec_bytes);
        if (err != cudaSuccess) {
            gafime_bucket_free(bucket);
            return GAFIME_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Allocate target
    err = cudaMalloc(&bucket->d_target, vec_bytes);
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate mask
    err = cudaMalloc(&bucket->d_mask, mask_bytes);
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate stats A (12 floats)
    err = cudaMalloc(&bucket->d_stats, 12 * sizeof(float));
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate stats B for interleaved kernel (12 floats)
    err = cudaMalloc(&bucket->d_stats_B, 12 * sizeof(float));
    if (err != cudaSuccess) {
        gafime_bucket_free(bucket);
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // =========================================================================
    // Priority 5: Calculate total data size for L2 cache hints
    // =========================================================================
    bucket->total_data_bytes = n_features * vec_bytes + vec_bytes + mask_bytes;
    
    // Note: L2 cache persistence requires cudaStreamSetAttribute with
    // cudaStreamAttributeAccessPolicyWindow. This is available on Ampere+
    // but requires careful tuning. For now, we rely on natural L2 caching.
    
    *bucket_out = static_cast<GafimeBucket>(bucket);
    return GAFIME_SUCCESS;
}

GAFIME_API int gafime_bucket_upload_feature(
    GafimeBucket bucket,
    int feature_idx,
    const float* h_data
) {
    if (!bucket || !h_data) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    
    if (feature_idx < 0 || feature_idx >= impl->n_features) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    size_t bytes = static_cast<size_t>(impl->n_samples) * sizeof(float);
    cudaError_t err = cudaMemcpy(impl->d_features[feature_idx], h_data, bytes, cudaMemcpyHostToDevice);
    
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_bucket_upload_target(
    GafimeBucket bucket,
    const float* h_target
) {
    if (!bucket || !h_target) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    size_t bytes = static_cast<size_t>(impl->n_samples) * sizeof(float);
    
    cudaError_t err = cudaMemcpy(impl->d_target, h_target, bytes, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_bucket_upload_mask(
    GafimeBucket bucket,
    const uint8_t* h_mask
) {
    if (!bucket || !h_mask) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    size_t bytes = static_cast<size_t>(impl->n_samples) * sizeof(uint8_t);
    
    cudaError_t err = cudaMemcpy(impl->d_mask, h_mask, bytes, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_bucket_compute(
    GafimeBucket bucket,
    const int* feature_indices,
    const int* ops,
    int arity,
    const int* interaction_types,
    int val_fold_id,
    float* h_stats
) {
    if (!bucket || !feature_indices || !ops || !interaction_types || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (arity < 2 || arity > GAFIME_MAX_FEATURES) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    
    // Validate feature indices
    for (int i = 0; i < arity; i++) {
        if (feature_indices[i] < 0 || feature_indices[i] >= impl->n_features) {
            return GAFIME_ERROR_INVALID_ARGS;
        }
    }
    
    // Get device pointers for selected features
    const float* d_input0 = impl->d_features[feature_indices[0]];
    const float* d_input1 = impl->d_features[feature_indices[1]];
    const float* d_input2 = (arity >= 3) ? impl->d_features[feature_indices[2]] : nullptr;
    const float* d_input3 = (arity >= 4) ? impl->d_features[feature_indices[3]] : nullptr;
    const float* d_input4 = (arity >= 5) ? impl->d_features[feature_indices[4]] : nullptr;
    
    // Extract per-pair interaction types (arity-1 interactions needed)
    int interact0 = interaction_types[0];
    int interact1 = (arity >= 3) ? interaction_types[1] : 0;
    int interact2 = (arity >= 4) ? interaction_types[2] : 0;
    int interact3 = (arity >= 5) ? interaction_types[3] : 0;
    
    // Zero stats buffer (NO cudaMalloc!)
    cudaError_t err = cudaMemset(impl->d_stats, 0, 12 * sizeof(float));
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Auto-tune for GPU on first call (lazy initialization)
    auto_tune_for_gpu();
    
    // Calculate grid dimensions (BLOCK_SIZE is compile-time, max_blocks is runtime-tuned)
    int num_blocks = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, GET_MAX_BLOCKS());
    
    // Launch kernel based on arity (NO cudaMalloc/cudaFree!)
    switch (arity) {
        case 2:
            gafime_fused_kernel<2><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, nullptr, nullptr, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], 0, 0, 0,
                interact0, 0, 0, 0,
                val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 3:
            gafime_fused_kernel<3><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, nullptr, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], 0, 0,
                interact0, interact1, 0, 0,
                val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 4:
            gafime_fused_kernel<4><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, d_input3, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], ops[3], 0,
                interact0, interact1, interact2, 0,
                val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 5:
            gafime_fused_kernel<5><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, d_input3, d_input4,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], ops[3], ops[4],
                interact0, interact1, interact2, interact3,
                val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Copy stats back (NO cudaFree!)
    err = cudaMemcpy(h_stats, impl->d_stats, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_bucket_free(GafimeBucket bucket) {
    if (!bucket) {
        return GAFIME_SUCCESS;  // Nothing to free
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    
    // Free all device memory
    for (int i = 0; i < GAFIME_MAX_FEATURES; i++) {
        if (impl->d_features[i]) {
            cudaFree(impl->d_features[i]);
        }
    }
    if (impl->d_target) cudaFree(impl->d_target);
    if (impl->d_mask) cudaFree(impl->d_mask);
    if (impl->d_stats) cudaFree(impl->d_stats);
    if (impl->d_stats_B) cudaFree(impl->d_stats_B);
    
    // Free Priority 4 resources
    if (impl->stream) cudaStreamDestroy(impl->stream);
    if (impl->h_stats_pinned) cudaFreeHost(impl->h_stats_pinned);
    
    delete impl;
    return GAFIME_SUCCESS;
}

// ============================================================================
// BATCHED COMPUTE API (Priority 3: Minimize kernel launch overhead)
// ============================================================================

/**
 * Maximum batch size for batched compute.
 * Each interaction needs 12 floats output.
 */
#define GAFIME_MAX_BATCH_SIZE 1024

/**
 * Batched fused kernel: Compute N feature interactions in ONE kernel launch.
 * 
 * Each block computes ONE interaction from the batch.
 * All N interactions run in parallel.
 * 
 * @param d_features     Array of device pointers to feature columns
 * @param d_target       Target vector
 * @param d_mask         Fold mask
 * @param batch_indices  [N * 2] - feature indices for each interaction
 * @param batch_ops      [N * 2] - ops for each interaction
 * @param batch_interact [N] - interaction type for each
 * @param batch_size     Number of interactions in batch
 * @param val_fold_id    Validation fold
 * @param n_samples      Samples per feature
 * @param d_stats_batch  Output: [N * 12] stats
 */
__global__ void gafime_batched_kernel(
    float* __restrict__ d_features_0,
    float* __restrict__ d_features_1,
    float* __restrict__ d_features_2,
    float* __restrict__ d_features_3,
    float* __restrict__ d_features_4,
    const float* __restrict__ d_target,
    const uint8_t* __restrict__ d_mask,
    const int* __restrict__ batch_indices,  // [N * 2]
    const int* __restrict__ batch_ops,      // [N * 2]
    const int* __restrict__ batch_interact, // [N]
    int batch_size,
    int val_fold_id,
    int n_samples,
    float* __restrict__ d_stats_batch       // [N * 12]
) {
    // Each block handles one interaction from the batch
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;
    
    // Get feature pointers array
    const float* features[5] = {d_features_0, d_features_1, d_features_2, d_features_3, d_features_4};
    
    // Load this interaction's parameters
    int f0_idx = batch_indices[batch_id * 2 + 0];
    int f1_idx = batch_indices[batch_id * 2 + 1];
    int op0 = batch_ops[batch_id * 2 + 0];
    int op1 = batch_ops[batch_id * 2 + 1];
    int interact = batch_interact[batch_id];
    
    const float* f0 = features[f0_idx];
    const float* f1 = features[f1_idx];
    
    // Thread-local accumulators
    float train_n = 0, train_sx = 0, train_sy = 0;
    float train_sxx = 0, train_syy = 0, train_sxy = 0;
    float val_n = 0, val_sx = 0, val_sy = 0;
    float val_sxx = 0, val_syy = 0, val_sxy = 0;
    
    // Grid-stride loop within this interaction
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_samples; i += blockDim.x * gridDim.x) {
        float x0 = apply_op(f0[i], op0);
        float x1 = apply_op(f1[i], op1);
        float X = combine(x0, x1, interact);
        float Y = d_target[i];
        uint8_t fold = d_mask[i];
        
        if (fold == val_fold_id) {
            val_n += 1.0f; val_sx += X; val_sy += Y;
            val_sxx += X*X; val_syy += Y*Y; val_sxy += X*Y;
        } else {
            train_n += 1.0f; train_sx += X; train_sy += Y;
            train_sxx += X*X; train_syy += Y*Y; train_sxy += X*Y;
        }
    }
    
    // Warp-level reduction
    warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    warp_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);
    
    // Shared memory for block reduction
    __shared__ float shared_train[6 * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ float shared_val[6 * (BLOCK_SIZE / WARP_SIZE)];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        shared_train[warp_id * 6 + 0] = train_n;
        shared_train[warp_id * 6 + 1] = train_sx;
        shared_train[warp_id * 6 + 2] = train_sy;
        shared_train[warp_id * 6 + 3] = train_sxx;
        shared_train[warp_id * 6 + 4] = train_syy;
        shared_train[warp_id * 6 + 5] = train_sxy;
        
        shared_val[warp_id * 6 + 0] = val_n;
        shared_val[warp_id * 6 + 1] = val_sx;
        shared_val[warp_id * 6 + 2] = val_sy;
        shared_val[warp_id * 6 + 3] = val_sxx;
        shared_val[warp_id * 6 + 4] = val_syy;
        shared_val[warp_id * 6 + 5] = val_sxy;
    }
    __syncthreads();
    
    // First warp reduces and writes to global output
    if (warp_id == 0 && lane < num_warps) {
        train_n = shared_train[lane * 6 + 0];
        train_sx = shared_train[lane * 6 + 1];
        train_sy = shared_train[lane * 6 + 2];
        train_sxx = shared_train[lane * 6 + 3];
        train_syy = shared_train[lane * 6 + 4];
        train_sxy = shared_train[lane * 6 + 5];
        
        val_n = shared_val[lane * 6 + 0];
        val_sx = shared_val[lane * 6 + 1];
        val_sy = shared_val[lane * 6 + 2];
        val_sxx = shared_val[lane * 6 + 3];
        val_syy = shared_val[lane * 6 + 4];
        val_sxy = shared_val[lane * 6 + 5];
        
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            train_n += __shfl_down_sync(0xffffffff, train_n, offset);
            train_sx += __shfl_down_sync(0xffffffff, train_sx, offset);
            train_sy += __shfl_down_sync(0xffffffff, train_sy, offset);
            train_sxx += __shfl_down_sync(0xffffffff, train_sxx, offset);
            train_syy += __shfl_down_sync(0xffffffff, train_syy, offset);
            train_sxy += __shfl_down_sync(0xffffffff, train_sxy, offset);
            
            val_n += __shfl_down_sync(0xffffffff, val_n, offset);
            val_sx += __shfl_down_sync(0xffffffff, val_sx, offset);
            val_sy += __shfl_down_sync(0xffffffff, val_sy, offset);
            val_sxx += __shfl_down_sync(0xffffffff, val_sxx, offset);
            val_syy += __shfl_down_sync(0xffffffff, val_syy, offset);
            val_sxy += __shfl_down_sync(0xffffffff, val_sxy, offset);
        }
        
        // Thread 0 writes to this batch's output slot
        if (lane == 0) {
            float* out = &d_stats_batch[batch_id * 12];
            atomicAdd(&out[0], train_n);
            atomicAdd(&out[1], train_sx);
            atomicAdd(&out[2], train_sy);
            atomicAdd(&out[3], train_sxx);
            atomicAdd(&out[4], train_syy);
            atomicAdd(&out[5], train_sxy);
            atomicAdd(&out[6], val_n);
            atomicAdd(&out[7], val_sx);
            atomicAdd(&out[8], val_sy);
            atomicAdd(&out[9], val_sxx);
            atomicAdd(&out[10], val_syy);
            atomicAdd(&out[11], val_sxy);
        }
    }
}

/**
 * Host API: Compute N interactions in ONE kernel launch.
 * 
 * @param bucket          Bucket with uploaded features
 * @param batch_indices   [N * 2] feature indices
 * @param batch_ops       [N * 2] operator IDs
 * @param batch_interact  [N] interaction types
 * @param batch_size      Number of interactions (max 1024)
 * @param val_fold_id     Validation fold
 * @param h_stats_batch   Output: [N * 12] host array
 */
GAFIME_API int gafime_bucket_compute_batch(
    GafimeBucket bucket,
    const int* h_batch_indices,
    const int* h_batch_ops,
    const int* h_batch_interact,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
) {
    if (!bucket || !h_batch_indices || !h_batch_ops || !h_batch_interact || !h_stats_batch) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (batch_size <= 0 || batch_size > GAFIME_MAX_BATCH_SIZE) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    cudaError_t err;
    
    // Allocate device memory for batch parameters (small, temporary)
    int* d_batch_indices;
    int* d_batch_ops;
    int* d_batch_interact;
    float* d_stats_batch;
    
    size_t indices_bytes = batch_size * 2 * sizeof(int);
    size_t ops_bytes = batch_size * 2 * sizeof(int);
    size_t interact_bytes = batch_size * sizeof(int);
    size_t stats_bytes = batch_size * 12 * sizeof(float);
    
    err = cudaMalloc(&d_batch_indices, indices_bytes);
    if (err != cudaSuccess) return GAFIME_ERROR_OUT_OF_MEMORY;
    err = cudaMalloc(&d_batch_ops, ops_bytes);
    if (err != cudaSuccess) { cudaFree(d_batch_indices); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&d_batch_interact, interact_bytes);
    if (err != cudaSuccess) { cudaFree(d_batch_indices); cudaFree(d_batch_ops); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&d_stats_batch, stats_bytes);
    if (err != cudaSuccess) { cudaFree(d_batch_indices); cudaFree(d_batch_ops); cudaFree(d_batch_interact); return GAFIME_ERROR_OUT_OF_MEMORY; }
    
    // Copy batch params to device
    cudaMemcpy(d_batch_indices, h_batch_indices, indices_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_ops, h_batch_ops, ops_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_interact, h_batch_interact, interact_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_stats_batch, 0, stats_bytes);
    
    // Auto-tune for GPU on first call
    auto_tune_for_gpu();
    
    // Launch using auto-tuned max_blocks (BLOCK_SIZE is compile-time)
    int blocks_per_interaction = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks_per_interaction = min(blocks_per_interaction, 64);  // Limit for atomics
    
    dim3 grid(blocks_per_interaction, batch_size);
    dim3 block(BLOCK_SIZE);
    
    gafime_batched_kernel<<<grid, block>>>(
        impl->d_features[0], impl->d_features[1], impl->d_features[2],
        impl->d_features[3], impl->d_features[4],
        impl->d_target, impl->d_mask,
        d_batch_indices, d_batch_ops, d_batch_interact,
        batch_size, val_fold_id, impl->n_samples,
        d_stats_batch
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_batch_indices); cudaFree(d_batch_ops);
        cudaFree(d_batch_interact); cudaFree(d_stats_batch);
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_batch_indices); cudaFree(d_batch_ops);
        cudaFree(d_batch_interact); cudaFree(d_stats_batch);
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Copy results back
    cudaMemcpy(h_stats_batch, d_stats_batch, stats_bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_batch_indices);
    cudaFree(d_batch_ops);
    cudaFree(d_batch_interact);
    cudaFree(d_stats_batch);
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// DUAL-ISSUE INTERLEAVED KERNEL (SFU + ALU Parallelism)
// ============================================================================

/**
 * Interleaved kernel processing TWO feature interactions per thread.
 * 
 * Slot A: SFU-heavy operations (log, exp, tanh, sigmoid)
 * Slot B: ALU-heavy operations (square, cube, rolling_mean, rolling_std)
 * 
 * While Slot A stalls waiting for SFU, Slot B executes on CUDA cores.
 */
__global__ void gafime_interleaved_kernel(
    // Slot A inputs
    const float* __restrict__ col_A0,
    const float* __restrict__ col_A1,
    // Slot B inputs
    const float* __restrict__ col_B0,
    const float* __restrict__ col_B1,
    // Shared
    const float* __restrict__ target,
    const uint8_t* __restrict__ mask,
    // Params
    int op_A0, int op_A1, int interact_A,
    int op_B0, int op_B1, int interact_B,
    int window_size,
    int val_fold_id,
    int n_rows,
    // Outputs
    float* __restrict__ stats_A,
    float* __restrict__ stats_B
) {
    // Per-thread accumulators for BOTH slots (registers)
    float train_n_A = 0, train_sx_A = 0, train_sy_A = 0;
    float train_sxx_A = 0, train_syy_A = 0, train_sxy_A = 0;
    float val_n_A = 0, val_sx_A = 0, val_sy_A = 0;
    float val_sxx_A = 0, val_syy_A = 0, val_sxy_A = 0;
    
    float train_n_B = 0, train_sx_B = 0, train_sy_B = 0;
    float train_sxx_B = 0, train_syy_B = 0, train_sxy_B = 0;
    float val_n_B = 0, val_sx_B = 0, val_sy_B = 0;
    float val_sxx_B = 0, val_syy_B = 0, val_sxy_B = 0;
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_rows; idx += blockDim.x * gridDim.x) {
        // === PIPELINE STUFFING: Issue SFU ops first (high latency) ===
        float val_A0 = apply_op_fast(col_A0, idx, n_rows, op_A0, window_size);
        float val_A1 = apply_op_fast(col_A1, idx, n_rows, op_A1, window_size);
        
        // === Issue ALU ops while SFU is busy (low latency, executes immediately) ===
        float val_B0 = apply_op_fast(col_B0, idx, n_rows, op_B0, window_size);
        float val_B1 = apply_op_fast(col_B1, idx, n_rows, op_B1, window_size);
        
        // Combine interactions
        float res_A = combine(val_A0, val_A1, interact_A);
        float res_B = combine(val_B0, val_B1, interact_B);
        
        float y = target[idx];
        uint8_t fold = mask[idx];
        
        // Accumulate Slot A
        if (fold == val_fold_id) {
            val_n_A += 1; val_sx_A += res_A; val_sy_A += y;
            val_sxx_A += res_A * res_A; val_syy_A += y * y; val_sxy_A += res_A * y;
        } else {
            train_n_A += 1; train_sx_A += res_A; train_sy_A += y;
            train_sxx_A += res_A * res_A; train_syy_A += y * y; train_sxy_A += res_A * y;
        }
        
        // Accumulate Slot B
        if (fold == val_fold_id) {
            val_n_B += 1; val_sx_B += res_B; val_sy_B += y;
            val_sxx_B += res_B * res_B; val_syy_B += y * y; val_sxy_B += res_B * y;
        } else {
            train_n_B += 1; train_sx_B += res_B; train_sy_B += y;
            train_sxx_B += res_B * res_B; train_syy_B += y * y; train_sxy_B += res_B * y;
        }
    }
    
    // Warp-level reduction for Slot A
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        train_n_A += __shfl_down_sync(0xffffffff, train_n_A, offset);
        train_sx_A += __shfl_down_sync(0xffffffff, train_sx_A, offset);
        train_sy_A += __shfl_down_sync(0xffffffff, train_sy_A, offset);
        train_sxx_A += __shfl_down_sync(0xffffffff, train_sxx_A, offset);
        train_syy_A += __shfl_down_sync(0xffffffff, train_syy_A, offset);
        train_sxy_A += __shfl_down_sync(0xffffffff, train_sxy_A, offset);
        val_n_A += __shfl_down_sync(0xffffffff, val_n_A, offset);
        val_sx_A += __shfl_down_sync(0xffffffff, val_sx_A, offset);
        val_sy_A += __shfl_down_sync(0xffffffff, val_sy_A, offset);
        val_sxx_A += __shfl_down_sync(0xffffffff, val_sxx_A, offset);
        val_syy_A += __shfl_down_sync(0xffffffff, val_syy_A, offset);
        val_sxy_A += __shfl_down_sync(0xffffffff, val_sxy_A, offset);
    }
    
    // Warp-level reduction for Slot B
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        train_n_B += __shfl_down_sync(0xffffffff, train_n_B, offset);
        train_sx_B += __shfl_down_sync(0xffffffff, train_sx_B, offset);
        train_sy_B += __shfl_down_sync(0xffffffff, train_sy_B, offset);
        train_sxx_B += __shfl_down_sync(0xffffffff, train_sxx_B, offset);
        train_syy_B += __shfl_down_sync(0xffffffff, train_syy_B, offset);
        train_sxy_B += __shfl_down_sync(0xffffffff, train_sxy_B, offset);
        val_n_B += __shfl_down_sync(0xffffffff, val_n_B, offset);
        val_sx_B += __shfl_down_sync(0xffffffff, val_sx_B, offset);
        val_sy_B += __shfl_down_sync(0xffffffff, val_sy_B, offset);
        val_sxx_B += __shfl_down_sync(0xffffffff, val_sxx_B, offset);
        val_syy_B += __shfl_down_sync(0xffffffff, val_syy_B, offset);
        val_sxy_B += __shfl_down_sync(0xffffffff, val_sxy_B, offset);
    }
    
    // Block-level reduction via shared memory
    __shared__ float shared_A[12 * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ float shared_B[12 * (BLOCK_SIZE / WARP_SIZE)];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        // Store Slot A
        shared_A[warp_id * 12 + 0] = train_n_A;
        shared_A[warp_id * 12 + 1] = train_sx_A;
        shared_A[warp_id * 12 + 2] = train_sy_A;
        shared_A[warp_id * 12 + 3] = train_sxx_A;
        shared_A[warp_id * 12 + 4] = train_syy_A;
        shared_A[warp_id * 12 + 5] = train_sxy_A;
        shared_A[warp_id * 12 + 6] = val_n_A;
        shared_A[warp_id * 12 + 7] = val_sx_A;
        shared_A[warp_id * 12 + 8] = val_sy_A;
        shared_A[warp_id * 12 + 9] = val_sxx_A;
        shared_A[warp_id * 12 + 10] = val_syy_A;
        shared_A[warp_id * 12 + 11] = val_sxy_A;
        
        // Store Slot B
        shared_B[warp_id * 12 + 0] = train_n_B;
        shared_B[warp_id * 12 + 1] = train_sx_B;
        shared_B[warp_id * 12 + 2] = train_sy_B;
        shared_B[warp_id * 12 + 3] = train_sxx_B;
        shared_B[warp_id * 12 + 4] = train_syy_B;
        shared_B[warp_id * 12 + 5] = train_sxy_B;
        shared_B[warp_id * 12 + 6] = val_n_B;
        shared_B[warp_id * 12 + 7] = val_sx_B;
        shared_B[warp_id * 12 + 8] = val_sy_B;
        shared_B[warp_id * 12 + 9] = val_sxx_B;
        shared_B[warp_id * 12 + 10] = val_syy_B;
        shared_B[warp_id * 12 + 11] = val_sxy_B;
    }
    __syncthreads();
    
    // First warp does final reduction
    if (warp_id == 0 && lane < 12) {
        float sum_A = 0, sum_B = 0;
        for (int w = 0; w < num_warps; w++) {
            sum_A += shared_A[w * 12 + lane];
            sum_B += shared_B[w * 12 + lane];
        }
        atomicAdd(&stats_A[lane], sum_A);
        atomicAdd(&stats_B[lane], sum_B);
    }
}

/**
 * Host API for interleaved compute.
 */
GAFIME_API int gafime_interleaved_compute(
    GafimeBucket bucket,
    const int* feature_indices_A,
    const int* ops_A,
    int arity_A,
    int interact_A,
    const int* feature_indices_B,
    const int* ops_B,
    int arity_B,
    int interact_B,
    int window_size,
    int val_fold_id,
    float* h_stats_A,
    float* h_stats_B
) {
    if (!bucket || !feature_indices_A || !ops_A || !feature_indices_B || !ops_B || !h_stats_A || !h_stats_B) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    // For now, only support arity=2 for simplicity
    if (arity_A != 2 || arity_B != 2) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    
    // Validate feature indices
    if (feature_indices_A[0] < 0 || feature_indices_A[0] >= impl->n_features ||
        feature_indices_A[1] < 0 || feature_indices_A[1] >= impl->n_features ||
        feature_indices_B[0] < 0 || feature_indices_B[0] >= impl->n_features ||
        feature_indices_B[1] < 0 || feature_indices_B[1] >= impl->n_features) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Zero both stats buffers
    cudaError_t err = cudaMemset(impl->d_stats, 0, 12 * sizeof(float));
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemset(impl->d_stats_B, 0, 12 * sizeof(float));
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    
    int num_blocks = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 1024);
    
    gafime_interleaved_kernel<<<num_blocks, BLOCK_SIZE>>>(
        impl->d_features[feature_indices_A[0]],
        impl->d_features[feature_indices_A[1]],
        impl->d_features[feature_indices_B[0]],
        impl->d_features[feature_indices_B[1]],
        impl->d_target,
        impl->d_mask,
        ops_A[0], ops_A[1], interact_A,
        ops_B[0], ops_B[1], interact_B,
        window_size,
        val_fold_id,
        impl->n_samples,
        impl->d_stats,
        impl->d_stats_B
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Interleaved kernel error: %s\n", cudaGetErrorString(err));
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    
    // Copy both stats back
    err = cudaMemcpy(h_stats_A, impl->d_stats, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemcpy(h_stats_B, impl->d_stats_B, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// LEGACY API (per-call allocation, DEPRECATED)
// ============================================================================

/**
 * Fused feature interaction kernel with on-chip reduction.
 * 
 * All inputs are HOST pointers - this function copies to device.
 * 
 * Returns 12 floats: [train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy,
 *                     val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy]
 */
GAFIME_API int gafime_fused_interaction(
    const float** h_inputs,     // Array of HOST pointers to feature columns
    const float* h_target,      // Host pointer to target vector
    const uint8_t* h_mask,      // Host pointer to fold mask
    const int* h_ops,           // Host array of unary operator IDs
    int arity,                  // Number of features (2-5)
    int interaction_type,       // Mult, Add, Div, etc.
    int val_fold_id,            // Current validation fold
    int n_samples,
    float* h_stats              // Output: 12 floats (host)
) {
    if (arity < 2 || arity > 5) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (!h_inputs || !h_target || !h_mask || !h_ops || !h_stats) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    cudaError_t err;
    size_t vec_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t mask_bytes = static_cast<size_t>(n_samples) * sizeof(uint8_t);
    
    // Allocate device memory for inputs
    float* d_inputs[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    float* d_target = nullptr;
    uint8_t* d_mask = nullptr;
    float* d_stats = nullptr;
    
    // Allocate and copy feature columns
    for (int i = 0; i < arity; i++) {
        err = cudaMalloc(&d_inputs[i], vec_bytes);
        if (err != cudaSuccess) goto fused_cleanup;
        err = cudaMemcpy(d_inputs[i], h_inputs[i], vec_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fused_cleanup;
    }
    
    // Allocate and copy target
    err = cudaMalloc(&d_target, vec_bytes);
    if (err != cudaSuccess) goto fused_cleanup;
    err = cudaMemcpy(d_target, h_target, vec_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fused_cleanup;
    
    // Allocate and copy mask
    err = cudaMalloc(&d_mask, mask_bytes);
    if (err != cudaSuccess) goto fused_cleanup;
    err = cudaMemcpy(d_mask, h_mask, mask_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fused_cleanup;
    
    // Allocate and zero output stats
    err = cudaMalloc(&d_stats, 12 * sizeof(float));
    if (err != cudaSuccess) goto fused_cleanup;
    err = cudaMemset(d_stats, 0, 12 * sizeof(float));
    if (err != cudaSuccess) goto fused_cleanup;
    
    // Calculate grid dimensions
    {
        int num_blocks = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks = min(num_blocks, 1024);
        
        // Launch appropriate kernel based on arity
        // Legacy API: use same interaction_type for all pairs (backward compatible)
        switch (arity) {
            case 2:
                gafime_fused_kernel<2><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], nullptr, nullptr, nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], 0, 0, 0,
                    interaction_type, 0, 0, 0,
                    val_fold_id, n_samples, d_stats
                );
                break;
            case 3:
                gafime_fused_kernel<3><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], nullptr, nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], 0, 0,
                    interaction_type, interaction_type, 0, 0,
                    val_fold_id, n_samples, d_stats
                );
                break;
            case 4:
                gafime_fused_kernel<4><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3], nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], h_ops[3], 0,
                    interaction_type, interaction_type, interaction_type, 0,
                    val_fold_id, n_samples, d_stats
                );
                break;
            case 5:
                gafime_fused_kernel<5><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3], d_inputs[4],
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], h_ops[3], h_ops[4],
                    interaction_type, interaction_type, interaction_type, interaction_type,
                    val_fold_id, n_samples, d_stats
                );
                break;
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
            goto fused_cleanup;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto fused_cleanup;
    }
    
    // Copy stats back to host
    err = cudaMemcpy(h_stats, d_stats, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fused_cleanup;
    
    // Cleanup and return success
    for (int i = 0; i < arity; i++) cudaFree(d_inputs[i]);
    cudaFree(d_target);
    cudaFree(d_mask);
    cudaFree(d_stats);
    return GAFIME_SUCCESS;

fused_cleanup:
    for (int i = 0; i < 5; i++) if (d_inputs[i]) cudaFree(d_inputs[i]);
    if (d_target) cudaFree(d_target);
    if (d_mask) cudaFree(d_mask);
    if (d_stats) cudaFree(d_stats);
    return GAFIME_ERROR_KERNEL_FAILED;
}

// ============================================================================
// LEGACY API (for backwards compatibility)
// ============================================================================

GAFIME_API int gafime_feature_interaction_cuda(
    const float* X,
    const float* means,
    float* output,
    const int32_t* combo_indices,
    const int32_t* combo_offsets,
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
) {
    // Legacy implementation kept for backwards compatibility
    // This is the simple multiply kernel we had before
    
    if (!X || !means || !output || !combo_indices || !combo_offsets) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_combos <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // For now, fall back to CPU computation for legacy API
    // TODO: Implement GPU version if needed
    for (int32_t c = 0; c < n_combos; c++) {
        int32_t start = combo_offsets[c];
        int32_t end = combo_offsets[c + 1];
        int32_t combo_size = end - start;
        
        for (int32_t i = 0; i < n_samples; i++) {
            float prod = 1.0f;
            for (int32_t j = start; j < end; j++) {
                int32_t feat = combo_indices[j];
                if (feat >= 0 && feat < n_features) {
                    if (combo_size == 1) {
                        prod = X[i * n_features + feat];
                    } else {
                        prod *= (X[i * n_features + feat] - means[feat]);
                    }
                }
            }
            output[i * n_combos + c] = prod;
        }
    }
    
    return GAFIME_SUCCESS;
}

GAFIME_API int gafime_pearson_cuda(
    const float* x,
    const float* y,
    int32_t n,
    float* result_out
) {
    if (!x || !y || !result_out || n <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // CPU fallback for now
    float sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xy = 0;
    for (int32_t i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_yy += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }
    
    float mean_x = sum_x / n;
    float mean_y = sum_y / n;
    float var_x = sum_xx - sum_x * mean_x;
    float var_y = sum_yy - sum_y * mean_y;
    float cov_xy = sum_xy - sum_x * mean_y;
    
    if (var_x <= 0 || var_y <= 0) {
        *result_out = 0.0f;
    } else {
        *result_out = cov_xy / sqrtf(var_x * var_y);
    }
    
    return GAFIME_SUCCESS;
}

// ============================================================================
// ASYNC 4-SLOT RING BUFFER PIPELINE
// ============================================================================
// 
// Enables overlapping batch creation (Rust/CPU) with GPU execution.
// 4 slots allow Rust to fill slots while GPU executes, preventing idle time.
// ============================================================================

#define PIPELINE_SLOTS 4
#define PIPELINE_MAX_BATCH 1024

/**
 * Single slot in the async pipeline.
 */
struct PipelineSlot {
    // Pre-allocated device buffers
    int* d_indices;      // [PIPELINE_MAX_BATCH * 2]
    int* d_ops;          // [PIPELINE_MAX_BATCH * 2]
    int* d_interact;     // [PIPELINE_MAX_BATCH]
    float* d_stats;      // [PIPELINE_MAX_BATCH * 12]
    
    // CUDA async primitives
    cudaStream_t stream;
    cudaEvent_t done_event;
    
    // Slot state
    int batch_size;      // Current batch size (0 if empty)
    bool is_submitted;   // Kernel has been launched
    bool is_complete;    // Kernel has finished
};

/**
 * Async pipeline implementation.
 */
struct GafimePipelineImpl {
    GafimeBucketImpl* bucket;   // Reference to data bucket
    PipelineSlot slots[PIPELINE_SLOTS];
    int write_idx;              // Next slot to write (producer)
    int read_idx;               // Next slot to read results (consumer)
    int pending_count;          // Number of slots with pending work
    int val_fold_id;            // Current validation fold
};

/**
 * Initialize async pipeline (call once after bucket is ready).
 */
GAFIME_API int gafime_pipeline_init(
    GafimeBucket bucket,
    int val_fold_id,
    GafimePipeline* pipeline_out
) {
    if (!bucket || !pipeline_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimePipelineImpl* pipeline = new (std::nothrow) GafimePipelineImpl;
    if (!pipeline) {
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    pipeline->bucket = static_cast<GafimeBucketImpl*>(bucket);
    pipeline->write_idx = 0;
    pipeline->read_idx = 0;
    pipeline->pending_count = 0;
    pipeline->val_fold_id = val_fold_id;
    
    cudaError_t err;
    
    // Initialize all slots
    for (int i = 0; i < PIPELINE_SLOTS; i++) {
        PipelineSlot& slot = pipeline->slots[i];
        
        slot.batch_size = 0;
        slot.is_submitted = false;
        slot.is_complete = false;
        
        // Create stream and event
        err = cudaStreamCreate(&slot.stream);
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaEventCreate(&slot.done_event);
        if (err != cudaSuccess) goto cleanup;
        
        // Allocate device buffers
        err = cudaMalloc(&slot.d_indices, PIPELINE_MAX_BATCH * 2 * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMalloc(&slot.d_ops, PIPELINE_MAX_BATCH * 2 * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMalloc(&slot.d_interact, PIPELINE_MAX_BATCH * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMalloc(&slot.d_stats, PIPELINE_MAX_BATCH * 12 * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
    }
    
    *pipeline_out = static_cast<GafimePipeline>(pipeline);
    return GAFIME_SUCCESS;
    
cleanup:
    // Free any allocated resources
    for (int i = 0; i < PIPELINE_SLOTS; i++) {
        PipelineSlot& slot = pipeline->slots[i];
        if (slot.stream) cudaStreamDestroy(slot.stream);
        if (slot.done_event) cudaEventDestroy(slot.done_event);
        if (slot.d_indices) cudaFree(slot.d_indices);
        if (slot.d_ops) cudaFree(slot.d_ops);
        if (slot.d_interact) cudaFree(slot.d_interact);
        if (slot.d_stats) cudaFree(slot.d_stats);
    }
    delete pipeline;
    return GAFIME_ERROR_OUT_OF_MEMORY;
}

/**
 * Submit batch to pipeline (non-blocking).
 * Returns slot_id on success, or -1 if pipeline is full.
 */
GAFIME_API int gafime_pipeline_submit(
    GafimePipeline pipeline_handle,
    const int* h_indices,
    const int* h_ops,
    const int* h_interact,
    int batch_size,
    int* slot_id_out
) {
    if (!pipeline_handle || !h_indices || !h_ops || !h_interact) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (batch_size <= 0 || batch_size > PIPELINE_MAX_BATCH) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    
    // Check if pipeline is full (backpressure)
    if (pipeline->pending_count >= PIPELINE_SLOTS) {
        if (slot_id_out) *slot_id_out = -1;
        return GAFIME_ERROR_PIPELINE_FULL;  // Caller should wait
    }
    
    // Get next write slot
    int slot_idx = pipeline->write_idx;
    PipelineSlot& slot = pipeline->slots[slot_idx];
    
    // Async copy batch data to device
    cudaMemcpyAsync(slot.d_indices, h_indices, batch_size * 2 * sizeof(int),
                    cudaMemcpyHostToDevice, slot.stream);
    cudaMemcpyAsync(slot.d_ops, h_ops, batch_size * 2 * sizeof(int),
                    cudaMemcpyHostToDevice, slot.stream);
    cudaMemcpyAsync(slot.d_interact, h_interact, batch_size * sizeof(int),
                    cudaMemcpyHostToDevice, slot.stream);
    cudaMemsetAsync(slot.d_stats, 0, batch_size * 12 * sizeof(float), slot.stream);
    
    // Auto-tune for GPU
    auto_tune_for_gpu();
    
    // Launch kernel asynchronously
    int blocks_per_interaction = (pipeline->bucket->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks_per_interaction = min(blocks_per_interaction, 64);
    
    dim3 grid(blocks_per_interaction, batch_size);
    dim3 block(BLOCK_SIZE);
    
    gafime_batched_kernel<<<grid, block, 0, slot.stream>>>(
        pipeline->bucket->d_features[0],
        pipeline->bucket->d_features[1],
        pipeline->bucket->d_features[2],
        pipeline->bucket->d_features[3],
        pipeline->bucket->d_features[4],
        pipeline->bucket->d_target,
        pipeline->bucket->d_mask,
        slot.d_indices, slot.d_ops, slot.d_interact,
        batch_size, pipeline->val_fold_id, pipeline->bucket->n_samples,
        slot.d_stats
    );
    
    // Record completion event
    cudaEventRecord(slot.done_event, slot.stream);
    
    // Update slot state
    slot.batch_size = batch_size;
    slot.is_submitted = true;
    slot.is_complete = false;
    
    // Advance write pointer
    pipeline->write_idx = (pipeline->write_idx + 1) % PIPELINE_SLOTS;
    pipeline->pending_count++;
    
    if (slot_id_out) *slot_id_out = slot_idx;
    return GAFIME_SUCCESS;
}

/**
 * Poll for completed results (non-blocking).
 * Returns GAFIME_SUCCESS if a result is ready, writes to stats_out and batch_size_out.
 * Returns GAFIME_ERROR_NO_RESULT if nothing is ready yet.
 */
GAFIME_API int gafime_pipeline_poll(
    GafimePipeline pipeline_handle,
    float* h_stats_out,
    int* batch_size_out
) {
    if (!pipeline_handle) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    
    if (pipeline->pending_count == 0) {
        return GAFIME_ERROR_NO_RESULT;
    }
    
    // Check oldest pending slot
    int slot_idx = pipeline->read_idx;
    PipelineSlot& slot = pipeline->slots[slot_idx];
    
    if (!slot.is_submitted) {
        return GAFIME_ERROR_NO_RESULT;
    }
    
    // Check if kernel is complete (non-blocking)
    cudaError_t err = cudaEventQuery(slot.done_event);
    if (err == cudaErrorNotReady) {
        return GAFIME_ERROR_NO_RESULT;  // Still running
    }
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Copy results back
    if (h_stats_out) {
        cudaMemcpy(h_stats_out, slot.d_stats, slot.batch_size * 12 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    if (batch_size_out) {
        *batch_size_out = slot.batch_size;
    }
    
    // Reset slot
    slot.batch_size = 0;
    slot.is_submitted = false;
    slot.is_complete = false;
    
    // Advance read pointer
    pipeline->read_idx = (pipeline->read_idx + 1) % PIPELINE_SLOTS;
    pipeline->pending_count--;
    
    return GAFIME_SUCCESS;
}

/**
 * Wait for next result (blocking).
 */
GAFIME_API int gafime_pipeline_wait(
    GafimePipeline pipeline_handle,
    float* h_stats_out,
    int* batch_size_out
) {
    if (!pipeline_handle) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    
    if (pipeline->pending_count == 0) {
        return GAFIME_ERROR_NO_RESULT;
    }
    
    // Wait on oldest pending slot
    int slot_idx = pipeline->read_idx;
    PipelineSlot& slot = pipeline->slots[slot_idx];
    
    // Block until kernel completes
    cudaEventSynchronize(slot.done_event);
    
    // Copy results back
    if (h_stats_out) {
        cudaMemcpy(h_stats_out, slot.d_stats, slot.batch_size * 12 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    if (batch_size_out) {
        *batch_size_out = slot.batch_size;
    }
    
    // Reset slot
    int size = slot.batch_size;
    slot.batch_size = 0;
    slot.is_submitted = false;
    slot.is_complete = false;
    
    // Advance read pointer
    pipeline->read_idx = (pipeline->read_idx + 1) % PIPELINE_SLOTS;
    pipeline->pending_count--;
    
    return GAFIME_SUCCESS;
}

/**
 * Get number of pending batches in pipeline.
 */
GAFIME_API int gafime_pipeline_pending(GafimePipeline pipeline_handle) {
    if (!pipeline_handle) return 0;
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    return pipeline->pending_count;
}

/**
 * Flush all pending batches (blocking).
 */
GAFIME_API int gafime_pipeline_flush(
    GafimePipeline pipeline_handle,
    float* h_all_stats_out,
    int* total_batch_size_out
) {
    if (!pipeline_handle) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    int total = 0;
    float* write_ptr = h_all_stats_out;
    
    while (pipeline->pending_count > 0) {
        int batch_size = 0;
        int result = gafime_pipeline_wait(pipeline_handle, write_ptr, &batch_size);
        if (result != GAFIME_SUCCESS) {
            return result;
        }
        total += batch_size;
        if (write_ptr) write_ptr += batch_size * 12;
    }
    
    if (total_batch_size_out) *total_batch_size_out = total;
    return GAFIME_SUCCESS;
}

/**
 * Free pipeline resources.
 */
GAFIME_API int gafime_pipeline_free(GafimePipeline pipeline_handle) {
    if (!pipeline_handle) {
        return GAFIME_SUCCESS;
    }
    
    GafimePipelineImpl* pipeline = static_cast<GafimePipelineImpl*>(pipeline_handle);
    
    // Wait for all pending work
    for (int i = 0; i < PIPELINE_SLOTS; i++) {
        PipelineSlot& slot = pipeline->slots[i];
        if (slot.is_submitted) {
            cudaEventSynchronize(slot.done_event);
        }
    }
    
    // Free all resources
    for (int i = 0; i < PIPELINE_SLOTS; i++) {
        PipelineSlot& slot = pipeline->slots[i];
        if (slot.stream) cudaStreamDestroy(slot.stream);
        if (slot.done_event) cudaEventDestroy(slot.done_event);
        if (slot.d_indices) cudaFree(slot.d_indices);
        if (slot.d_ops) cudaFree(slot.d_ops);
        if (slot.d_interact) cudaFree(slot.d_interact);
        if (slot.d_stats) cudaFree(slot.d_stats);
    }
    
    delete pipeline;
    return GAFIME_SUCCESS;
}

// ============================================================================
// CONTIGUOUS MEMORY BUCKET (V2) - Column-Major Layout
// ============================================================================
//
// Single contiguous allocation for optimal memory coalescing.
// Layout: [Feature0][Feature1]...[FeatureN][Target][Mask]
//         ^         ^             ^         ^       ^
//         0         N             N*k       N*(k+1) N*(k+2)
//
// Access: feature[i] = d_data + i * n_samples
//         target     = d_data + n_features * n_samples
//         mask       = (uint8_t*)(d_data + (n_features + 1) * n_samples)
// ============================================================================

struct ContiguousBucketImpl {
    int n_samples;
    int n_features;
    
    // Single contiguous allocation
    float* d_data;                  // [n_features * n_samples + n_samples] floats
    uint8_t* d_mask;                // [n_samples] bytes (separate for alignment)
    
    // Stats output
    float* d_stats;
    
    // Async primitives
    cudaStream_t stream;
    float* h_stats_pinned;
    int* h_batch_indices;  // [5 * MAX_BATCH_SLOTS] pinned memory
    int* d_batch_indices;  // [5 * MAX_BATCH_SLOTS] device memory
    
    // Computed offsets (for convenience)
    size_t target_offset;           // = n_features * n_samples
    size_t total_floats;            // = (n_features + 1) * n_samples
};

// Opaque handle
// typedef void* ContiguousBucket; // Defined in interfaces.h

/**
 * Allocate contiguous bucket with single VRAM allocation.
 * Rust will prepare the data layout on CPU, then upload in one transfer.
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_alloc(
    int n_samples,
    int n_features,
    ContiguousBucket* bucket_out
) {
    if (n_samples <= 0 || n_features <= 0 || !bucket_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* bucket = new (std::nothrow) ContiguousBucketImpl;
    if (!bucket) {
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    bucket->n_samples = n_samples;
    bucket->n_features = n_features;
    bucket->d_data = nullptr;
    bucket->d_mask = nullptr;
    bucket->d_stats = nullptr;
    bucket->stream = nullptr;
    bucket->h_stats_pinned = nullptr;
    
    // Calculate sizes
    size_t n = static_cast<size_t>(n_samples);
    size_t k = static_cast<size_t>(n_features);
    bucket->target_offset = k * n;
    bucket->total_floats = (k + 1) * n;  // features + target
    
    size_t data_bytes = bucket->total_floats * sizeof(float);
    size_t mask_bytes = n * sizeof(uint8_t);
    
    cudaError_t err;
    
    // Create stream
    err = cudaStreamCreate(&bucket->stream);
    if (err != cudaSuccess) {
        delete bucket;
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Allocate pinned host memory for stats
    // Allocate pinned host memory for stats ring buffer and batch indices
    err = cudaMallocHost(&bucket->h_stats_pinned, MAX_BATCH_SLOTS * 12 * sizeof(float));
    if (err != cudaSuccess) {
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate host pinned buffer for indices
    // 5 arrays * MAX_BATCH_SLOTS * sizeof(int)
    err = cudaMallocHost(&bucket->h_batch_indices, 5 * MAX_BATCH_SLOTS * sizeof(int));
    if (err != cudaSuccess) {
        cudaFreeHost(bucket->h_stats_pinned);
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate device batch indices
    err = cudaMalloc(&bucket->d_batch_indices, 5 * MAX_BATCH_SLOTS * sizeof(int));
    if (err != cudaSuccess) {
        cudaFreeHost(bucket->h_stats_pinned);
        cudaFreeHost(bucket->h_batch_indices);
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate SINGLE contiguous buffer for all data
    err = cudaMalloc(&bucket->d_data, data_bytes);
    if (err != cudaSuccess) {
        cudaFreeHost(bucket->h_stats_pinned);
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate mask separately (different type, alignment)
    err = cudaMalloc(&bucket->d_mask, mask_bytes);
    if (err != cudaSuccess) {
        cudaFree(bucket->d_data);
        cudaFreeHost(bucket->h_stats_pinned);
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate stats output
    // Allocate stats output ring buffer
    err = cudaMalloc(&bucket->d_stats, MAX_BATCH_SLOTS * 12 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(bucket->d_mask);
        cudaFree(bucket->d_data);
        cudaFreeHost(bucket->h_stats_pinned);
        cudaStreamDestroy(bucket->stream);
        delete bucket;
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    
    *bucket_out = static_cast<ContiguousBucket>(bucket);
    
    printf("[GAFIME] Contiguous bucket allocated: %d samples × %d features\n", n_samples, n_features);
    printf("[GAFIME]   Data: %.2f MB (single allocation)\n", data_bytes / (1024.0 * 1024.0));
    printf("[GAFIME]   Layout: [F0|F1|...|F%d|Target] column-major\n", n_features - 1);
    
    return GAFIME_SUCCESS;
}

/**
 * Upload ALL data in one transfer.
 * h_data must be prepared by Rust in column-major layout:
 * [feature0_samples][feature1_samples]...[target_samples]
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_upload(
    ContiguousBucket bucket,
    const float* h_data,      // [n_features * n_samples + n_samples] floats
    const uint8_t* h_mask     // [n_samples] bytes
) {
    if (!bucket || !h_data || !h_mask) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    size_t data_bytes = impl->total_floats * sizeof(float);
    size_t mask_bytes = static_cast<size_t>(impl->n_samples) * sizeof(uint8_t);
    
    cudaError_t err;
    
    // Upload all data in ONE transfer
    err = cudaMemcpy(impl->d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Upload mask
    err = cudaMemcpy(impl->d_mask, h_mask, mask_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    return GAFIME_SUCCESS;
}

/**
 * Compute kernel for contiguous bucket.
 * Uses pointer arithmetic for memory access.
 */
__global__ void gafime_contiguous_kernel(
    const float* __restrict__ d_data,    // Contiguous data
    const uint8_t* __restrict__ d_mask,
    int n_samples,
    int n_features,
    int feature_a_idx,
    int feature_b_idx,
    int op_a,
    int op_b,
    int interact_type,
    int val_fold_id,
    float* __restrict__ d_stats_out
) {
    // Shared memory for warp-level reduction
    __shared__ float shared_train[6 * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ float shared_val[6 * (BLOCK_SIZE / WARP_SIZE)];
    
    // Calculate feature pointers using pointer arithmetic
    const float* feature_a = d_data + feature_a_idx * n_samples;
    const float* feature_b = d_data + feature_b_idx * n_samples;
    const float* target = d_data + n_features * n_samples;
    
    // Per-thread accumulators
    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;
    float val_n = 0.0f, val_sx = 0.0f, val_sy = 0.0f;
    float val_sxx = 0.0f, val_syy = 0.0f, val_sxy = 0.0f;
    
    // Grid-stride loop for processing samples
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n_samples; 
         i += blockDim.x * gridDim.x) {
        
        // Load data - sequential access pattern for coalescing!
        float a = feature_a[i];
        float b = feature_b[i];
        float y = target[i];
        uint8_t fold = d_mask[i];
        
        // Apply unary operators
        float xa = apply_op(a, op_a);
        float xb = apply_op(b, op_b);
        
        // Apply interaction
        float x;
        switch (interact_type) {
            case GAFIME_INTERACT_MULT: x = xa * xb; break;
            case GAFIME_INTERACT_ADD: x = xa + xb; break;
            case GAFIME_INTERACT_SUB: x = xa - xb; break;
            case GAFIME_INTERACT_DIV: x = xa / (xb + 1e-8f); break;
            case GAFIME_INTERACT_MAX: x = fmaxf(xa, xb); break;
            case GAFIME_INTERACT_MIN: x = fminf(xa, xb); break;
            default: x = xa * xb; break;
        }
        
        // Accumulate statistics
        if (fold == val_fold_id) {
            val_n += 1.0f;
            val_sx += x;
            val_sy += y;
            val_sxx += x * x;
            val_syy += y * y;
            val_sxy += x * y;
        } else {
            train_n += 1.0f;
            train_sx += x;
            train_sy += y;
            train_sxx += x * x;
            train_syy += y * y;
            train_sxy += x * y;
        }
    }
    
    // Warp-level reduction
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        train_n += __shfl_down_sync(0xffffffff, train_n, offset);
        train_sx += __shfl_down_sync(0xffffffff, train_sx, offset);
        train_sy += __shfl_down_sync(0xffffffff, train_sy, offset);
        train_sxx += __shfl_down_sync(0xffffffff, train_sxx, offset);
        train_syy += __shfl_down_sync(0xffffffff, train_syy, offset);
        train_sxy += __shfl_down_sync(0xffffffff, train_sxy, offset);
        
        val_n += __shfl_down_sync(0xffffffff, val_n, offset);
        val_sx += __shfl_down_sync(0xffffffff, val_sx, offset);
        val_sy += __shfl_down_sync(0xffffffff, val_sy, offset);
        val_sxx += __shfl_down_sync(0xffffffff, val_sxx, offset);
        val_syy += __shfl_down_sync(0xffffffff, val_syy, offset);
        val_sxy += __shfl_down_sync(0xffffffff, val_sxy, offset);
    }
    
    // First thread in warp writes to shared memory
    if (lane == 0) {
        shared_train[warp_id * 6 + 0] = train_n;
        shared_train[warp_id * 6 + 1] = train_sx;
        shared_train[warp_id * 6 + 2] = train_sy;
        shared_train[warp_id * 6 + 3] = train_sxx;
        shared_train[warp_id * 6 + 4] = train_syy;
        shared_train[warp_id * 6 + 5] = train_sxy;
        
        shared_val[warp_id * 6 + 0] = val_n;
        shared_val[warp_id * 6 + 1] = val_sx;
        shared_val[warp_id * 6 + 2] = val_sy;
        shared_val[warp_id * 6 + 3] = val_sxx;
        shared_val[warp_id * 6 + 4] = val_syy;
        shared_val[warp_id * 6 + 5] = val_sxy;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        int n_warps = BLOCK_SIZE / WARP_SIZE;
        
        float final_train[6] = {0};
        float final_val[6] = {0};
        
        for (int w = lane; w < n_warps; w += WARP_SIZE) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += shared_train[w * 6 + j];
                final_val[j] += shared_val[w * 6 + j];
            }
        }
        
        // Final warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += __shfl_down_sync(0xffffffff, final_train[j], offset);
                final_val[j] += __shfl_down_sync(0xffffffff, final_val[j], offset);
            }
        }
        
        // First thread writes to global memory
        if (lane == 0) {
            atomicAdd(&d_stats_out[0], final_train[0]);
            atomicAdd(&d_stats_out[1], final_train[1]);
            atomicAdd(&d_stats_out[2], final_train[2]);
            atomicAdd(&d_stats_out[3], final_train[3]);
            atomicAdd(&d_stats_out[4], final_train[4]);
            atomicAdd(&d_stats_out[5], final_train[5]);
            
            atomicAdd(&d_stats_out[6], final_val[0]);
            atomicAdd(&d_stats_out[7], final_val[1]);
            atomicAdd(&d_stats_out[8], final_val[2]);
            atomicAdd(&d_stats_out[9], final_val[3]);
            atomicAdd(&d_stats_out[10], final_val[4]);
            atomicAdd(&d_stats_out[11], final_val[5]);
        }
    }
}

/**
 * Compute single interaction on contiguous bucket.
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_compute(
    ContiguousBucket bucket,
    int feature_a_idx,
    int feature_b_idx,
    int op_a,
    int op_b,
    int interact_type,
    int val_fold_id,
    float* h_stats_out
) {
    if (!bucket || !h_stats_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    if (feature_a_idx < 0 || feature_a_idx >= impl->n_features ||
        feature_b_idx < 0 || feature_b_idx >= impl->n_features) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Clear stats
    cudaMemsetAsync(impl->d_stats, 0, 12 * sizeof(float), impl->stream);
    
    // Launch kernel
    auto_tune_for_gpu();
    int num_blocks = min((impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, GET_MAX_BLOCKS());
    
    gafime_contiguous_kernel<<<num_blocks, BLOCK_SIZE, 0, impl->stream>>>(
        impl->d_data,
        impl->d_mask,
        impl->n_samples,
        impl->n_features,
        feature_a_idx,
        feature_b_idx,
        op_a,
        op_b,
        interact_type,
        val_fold_id,
        impl->d_stats
    );
    
    // Copy results back
    cudaMemcpyAsync(h_stats_out, impl->d_stats, 12 * sizeof(float),
                    cudaMemcpyDeviceToHost, impl->stream);
    cudaStreamSynchronize(impl->stream);
    
    return GAFIME_SUCCESS;
}

/**
 * Free contiguous bucket.
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_free(ContiguousBucket bucket) {
    if (!bucket) {
        return GAFIME_SUCCESS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    if (impl->d_stats) cudaFree(impl->d_stats);
    if (impl->d_mask) cudaFree(impl->d_mask);
    if (impl->d_data) cudaFree(impl->d_data);
    if (impl->h_stats_pinned) cudaFreeHost(impl->h_stats_pinned);
    if (impl->stream) cudaStreamDestroy(impl->stream);
    
    delete impl;
    return GAFIME_SUCCESS;
}

/**
 * Get bucket info.
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_info(
    ContiguousBucket bucket,
    int* n_samples_out,
    int* n_features_out
) {
    if (!bucket) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    if (n_samples_out) *n_samples_out = impl->n_samples;
    if (n_features_out) *n_features_out = impl->n_features;
    
    return GAFIME_SUCCESS;
}

/**
 * Async Compute (Launch Only)
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_compute_async(
    ContiguousBucket bucket,
    int feature_a_idx,
    int feature_b_idx,
    int op_a,
    int op_b,
    int interact_type,
    int val_fold_id,
    int slot_id
) {
    if (!bucket || slot_id < 0 || slot_id >= MAX_BATCH_SLOTS) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    // Check bounds
    if (feature_a_idx < 0 || feature_a_idx >= impl->n_features ||
        feature_b_idx < 0 || feature_b_idx >= impl->n_features) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Pointer math for this slot
    float* d_stats_slot = impl->d_stats + (slot_id * 12);
    
    // Clear stats (Async)
    cudaMemsetAsync(d_stats_slot, 0, 12 * sizeof(float), impl->stream);
    
    // Launch kernel (Async)
    int num_blocks = min((impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, GET_MAX_BLOCKS());
    
    gafime_contiguous_kernel<<<num_blocks, BLOCK_SIZE, 0, impl->stream>>>(
        impl->d_data,
        impl->d_mask,
        impl->n_samples,
        impl->n_features,
        feature_a_idx,
        feature_b_idx,
        op_a,
        op_b,
        interact_type,
        val_fold_id,
        d_stats_slot
    );
    
    // Copy back to specific pinned slot (Async)
    float* h_stats_slot = impl->h_stats_pinned + (slot_id * 12);
    cudaMemcpyAsync(h_stats_slot, d_stats_slot, 12 * sizeof(float),
                    cudaMemcpyDeviceToHost, impl->stream);
                    
    // NO SYNC HERE!
    
    return GAFIME_SUCCESS;
}

/**
 * Sync Stream
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_sync(ContiguousBucket bucket) {
    if (!bucket) return GAFIME_ERROR_INVALID_ARGS;
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    cudaStreamSynchronize(impl->stream);
    return GAFIME_SUCCESS;
}

/**
 * Read Result from Pinned Memory (CPU-side only)
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_read_result(
    ContiguousBucket bucket,
    int slot_id,
    float* h_stats_out
) {
    if (!bucket || slot_id < 0 || slot_id >= MAX_BATCH_SLOTS || !h_stats_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    float* h_stats_slot = impl->h_stats_pinned + (slot_id * 12);
    
    // Direct memcpy from pinned memory (safe if synced)
    memcpy(h_stats_out, h_stats_slot, 12 * sizeof(float));
    
    return GAFIME_SUCCESS;
}

/**
 * Batched Kernel
 * Grid computes N interactions in parallel.
 * blockIdx.y determines which interaction (0..n_interactions-1)
 * blockIdx.x handles samples for that interaction.
 */
__global__ void gafime_contiguous_batched_kernel(
    const float* __restrict__ data,
    const uint8_t* __restrict__ mask,
    int n_samples,
    int n_features,
    const int* __restrict__ d_feature_a,
    const int* __restrict__ d_feature_b,
    const int* __restrict__ d_op_a,
    const int* __restrict__ d_op_b,
    const int* __restrict__ d_interact_type,
    int val_fold_id,
    float* __restrict__ d_stats_base
) {
    int interaction_idx = blockIdx.y;
    
    // Load interaction params
    int fa = d_feature_a[interaction_idx];
    int fb = d_feature_b[interaction_idx];
    int oa = d_op_a[interaction_idx];
    int ob = d_op_b[interaction_idx];
    int type = d_interact_type[interaction_idx];
    
    // Pointer to this interaction's stats output
    float* d_stats = d_stats_base + (interaction_idx * 12);
    
    // Shared memory for reduction - allocated per block
    extern __shared__ float shared_mem[];
    float* shared_train = shared_mem;
    float* shared_val = shared_mem + (BLOCK_SIZE / WARP_SIZE) * 6;
    
    // Per-thread accumulators
    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;
    float val_n = 0.0f, val_sx = 0.0f, val_sy = 0.0f;
    float val_sxx = 0.0f, val_syy = 0.0f, val_sxy = 0.0f;
    
    // Get pointers to features
    const float* feature_a = data + (static_cast<size_t>(fa) * n_samples);
    const float* feature_b = data + (static_cast<size_t>(fb) * n_samples);
    const float* target = data + (static_cast<size_t>(n_features) * n_samples);
    
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n_samples; 
         i += blockDim.x * gridDim.x) {
        
        float a = feature_a[i];
        float b = feature_b[i];
        float y = target[i];
        uint8_t fold = mask[i];
        
        float xa = apply_op(a, oa);
        float xb = apply_op(b, ob);
        
        float x;
        switch (type) {
            case GAFIME_INTERACT_MULT: x = xa * xb; break;
            case GAFIME_INTERACT_ADD: x = xa + xb; break;
            case GAFIME_INTERACT_SUB: x = xa - xb; break;
            case GAFIME_INTERACT_DIV: x = xa / (xb + 1e-8f); break;
            case GAFIME_INTERACT_MAX: x = fmaxf(xa, xb); break;
            case GAFIME_INTERACT_MIN: x = fminf(xa, xb); break;
            default: x = xa * xb; break;
        }
        
        if (fold == val_fold_id) {
            val_n += 1.0f; val_sx += x; val_sy += y;
            val_sxx += x * x; val_syy += y * y; val_sxy += x * y;
        } else {
            train_n += 1.0f; train_sx += x; train_sy += y;
            train_sxx += x * x; train_syy += y * y; train_sxy += x * y;
        }
    }
    
    // Warp Reduction
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        train_n += __shfl_down_sync(0xffffffff, train_n, offset);
        train_sx += __shfl_down_sync(0xffffffff, train_sx, offset);
        train_sy += __shfl_down_sync(0xffffffff, train_sy, offset);
        train_sxx += __shfl_down_sync(0xffffffff, train_sxx, offset);
        train_syy += __shfl_down_sync(0xffffffff, train_syy, offset);
        train_sxy += __shfl_down_sync(0xffffffff, train_sxy, offset);
        
        val_n += __shfl_down_sync(0xffffffff, val_n, offset);
        val_sx += __shfl_down_sync(0xffffffff, val_sx, offset);
        val_sy += __shfl_down_sync(0xffffffff, val_sy, offset);
        val_sxx += __shfl_down_sync(0xffffffff, val_sxx, offset);
        val_syy += __shfl_down_sync(0xffffffff, val_syy, offset);
        val_sxy += __shfl_down_sync(0xffffffff, val_sxy, offset);
    }
    
    if (lane == 0) {
        shared_train[warp_id * 6 + 0] = train_n;
        shared_train[warp_id * 6 + 1] = train_sx;
        shared_train[warp_id * 6 + 2] = train_sy;
        shared_train[warp_id * 6 + 3] = train_sxx;
        shared_train[warp_id * 6 + 4] = train_syy;
        shared_train[warp_id * 6 + 5] = train_sxy;
        
        shared_val[warp_id * 6 + 0] = val_n;
        shared_val[warp_id * 6 + 1] = val_sx;
        shared_val[warp_id * 6 + 2] = val_sy;
        shared_val[warp_id * 6 + 3] = val_sxx;
        shared_val[warp_id * 6 + 4] = val_syy;
        shared_val[warp_id * 6 + 5] = val_sxy;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        int n_warps = BLOCK_SIZE / WARP_SIZE;
        float final_train[6] = {0};
        float final_val[6] = {0};
        
        for (int w = lane; w < n_warps; w += WARP_SIZE) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += shared_train[w * 6 + j];
                final_val[j] += shared_val[w * 6 + j];
            }
        }
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            for (int j = 0; j < 6; j++) {
                final_train[j] += __shfl_down_sync(0xffffffff, final_train[j], offset);
                final_val[j] += __shfl_down_sync(0xffffffff, final_val[j], offset);
            }
        }
        
        if (lane == 0) {
            atomicAdd(&d_stats[0], final_train[0]);
            atomicAdd(&d_stats[1], final_train[1]);
            atomicAdd(&d_stats[2], final_train[2]);
            atomicAdd(&d_stats[3], final_train[3]);
            atomicAdd(&d_stats[4], final_train[4]);
            atomicAdd(&d_stats[5], final_train[5]);
            
            atomicAdd(&d_stats[6], final_val[0]);
            atomicAdd(&d_stats[7], final_val[1]);
            atomicAdd(&d_stats[8], final_val[2]);
            atomicAdd(&d_stats[9], final_val[3]);
            atomicAdd(&d_stats[10], final_val[4]);
            atomicAdd(&d_stats[11], final_val[5]);
        }
    }
}

/**
 * Batched Compute (N interactions in one launch)
 * Pass pointers to arrays of arguments (length = n_interactions)
 */
extern "C" GAFIME_API int gafime_contiguous_bucket_compute_batched(
    ContiguousBucket bucket,
    const int* feature_a_indices,
    const int* feature_b_indices,
    const int* op_a_indices,
    const int* op_b_indices,
    const int* interact_types,
    int n_interactions,
    int val_fold_id,
    float* h_stats_out_flat // [n_interactions * 12]
) {
    if (!bucket || !feature_a_indices || !h_stats_out_flat || n_interactions <= 0 || n_interactions > MAX_BATCH_SLOTS) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
    
    // 1. Copy indices to Pinned Host Memory first
    size_t batch_bytes = n_interactions * sizeof(int);
    
    int* dst = impl->h_batch_indices;
    memcpy(dst + 0*MAX_BATCH_SLOTS, feature_a_indices, batch_bytes);
    memcpy(dst + 1*MAX_BATCH_SLOTS, feature_b_indices, batch_bytes);
    memcpy(dst + 2*MAX_BATCH_SLOTS, op_a_indices, batch_bytes);
    memcpy(dst + 3*MAX_BATCH_SLOTS, op_b_indices, batch_bytes);
    memcpy(dst + 4*MAX_BATCH_SLOTS, interact_types, batch_bytes);
    
    // 2. Async Upload Indices
    int* d_dst = impl->d_batch_indices;
    cudaMemcpyAsync(d_dst + 0*MAX_BATCH_SLOTS, dst + 0*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(d_dst + 1*MAX_BATCH_SLOTS, dst + 1*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(d_dst + 2*MAX_BATCH_SLOTS, dst + 2*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(d_dst + 3*MAX_BATCH_SLOTS, dst + 3*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(d_dst + 4*MAX_BATCH_SLOTS, dst + 4*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
    
    // 3. Clear Stats Output (N * 12 floats)
    cudaMemsetAsync(impl->d_stats, 0, n_interactions * 12 * sizeof(float), impl->stream);
    
    // 4. Launch ONE Kernel
    int blocks_per_sample = min((impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, GET_MAX_BLOCKS());
    dim3 grid(blocks_per_sample, n_interactions);
    
    int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * 2 * 6 * sizeof(float);
    
    gafime_contiguous_batched_kernel<<<grid, BLOCK_SIZE, shared_mem_size, impl->stream>>>(
        impl->d_data,
        impl->d_mask,
        impl->n_samples,
        impl->n_features,
        d_dst + 0*MAX_BATCH_SLOTS,
        d_dst + 1*MAX_BATCH_SLOTS,
        d_dst + 2*MAX_BATCH_SLOTS,
        d_dst + 3*MAX_BATCH_SLOTS,
        d_dst + 4*MAX_BATCH_SLOTS,
        val_fold_id,
        impl->d_stats
    );
    
    // 5. Copy Results Back
    cudaMemcpyAsync(impl->h_stats_pinned, impl->d_stats, n_interactions * 12 * sizeof(float), cudaMemcpyDeviceToHost, impl->stream);
    
    // 6. Synchronize
    cudaStreamSynchronize(impl->stream);
    
    // 7. Copy to user output
    memcpy(h_stats_out_flat, impl->h_stats_pinned, n_interactions * 12 * sizeof(float));
    
    return GAFIME_SUCCESS;
}

} // extern "C"

// ============================================================================
// PIVOT KERNEL V2 (Tiled Register Reuse)
// ============================================================================

#define K_TILE 4

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void gafime_contiguous_pivot_kernel_tiled(
    const float* __restrict__ data,
    const uint8_t* __restrict__ mask,
    int n_samples,
    int n_features,
    int fa_fixed, 
    int oa_fixed, 
    const int* __restrict__ d_fb_indices, 
    const int* __restrict__ d_ob_indices, 
    const int* __restrict__ d_type_indices, 
    int n_candidates, 
    int val_fold_id, 
    float* __restrict__ d_stats_base
) {
    // Shared memory for reduction - Just for one set of stats at a time?
    // We reduce sequentially at end of tile.
    extern __shared__ float shared_mem[];
    float* shared_train = shared_mem;
    float* shared_val = shared_mem + (BLOCK_SIZE / WARP_SIZE) * 6;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float* feature_a_ptr = data + (static_cast<size_t>(fa_fixed) * n_samples);
    const float* target_ptr = data + (static_cast<size_t>(n_features) * n_samples);

    // OUTER LOOP: Iterate over Candidates in Tiles of K_TILE
    for (int k_base = 0; k_base < n_candidates; k_base += K_TILE) {
        
        // Register Accumulators for the Tile
        float train_N[K_TILE] = {0}; float train_SX[K_TILE] = {0}; float train_SY[K_TILE] = {0};
        float train_SXX[K_TILE] = {0}; float train_SYY[K_TILE] = {0}; float train_SXY[K_TILE] = {0};
        
        float val_N[K_TILE] = {0}; float val_SX[K_TILE] = {0}; float val_SY[K_TILE] = {0};
        float val_SXX[K_TILE] = {0}; float val_SYY[K_TILE] = {0}; float val_SXY[K_TILE] = {0};

        int valid_k = (K_TILE < (n_candidates - k_base)) ? K_TILE : (n_candidates - k_base);
        
        // Pre-load B params for this tile to registers
        int ob_tile[K_TILE];
        int type_tile[K_TILE];
        const float* b_ptrs[K_TILE];
        
        for(int t=0; t<valid_k; t++) {
            int k = k_base + t;
            int fb = d_fb_indices[k];
            ob_tile[t] = d_ob_indices[k];
            type_tile[t] = d_type_indices[k];
            b_ptrs[t] = data + (static_cast<size_t>(fb) * n_samples);
        }

        // INNER GRID STRIDE LOOP over samples
        for (int i = tid; i < n_samples; i += blockDim.x * gridDim.x) {
            
            float a_raw = feature_a_ptr[i]; // Load A ONCE per sample for this Tile (Register Reuse!)
            float val_a = apply_op(a_raw, oa_fixed);
            float y = target_ptr[i];
            uint8_t fold = mask[i];
            
            for(int t=0; t<valid_k; t++) {
                float b_raw = b_ptrs[t][i];
                float val_b = apply_op(b_raw, ob_tile[t]);
                
                float x;
                switch (type_tile[t]) {
                    case GAFIME_INTERACT_MULT: x = val_a * val_b; break;
                    case GAFIME_INTERACT_ADD: x = val_a + val_b; break;
                    case GAFIME_INTERACT_SUB: x = val_a - val_b; break;
                    case GAFIME_INTERACT_DIV: x = val_a / (val_b + 1e-8f); break;
                    case GAFIME_INTERACT_MAX: x = fmaxf(val_a, val_b); break;
                    case GAFIME_INTERACT_MIN: x = fminf(val_a, val_b); break;
                    default: x = val_a * val_b; break;
                }
                
                if (fold == val_fold_id) {
                    val_N[t] += 1.0f; val_SX[t] += x; val_SY[t] += y;
                    val_SXX[t] += x*x; val_SYY[t] += y*y; val_SXY[t] += x*y;
                } else {
                    train_N[t] += 1.0f; train_SX[t] += x; train_SY[t] += y;
                    train_SXX[t] += x*x; train_SYY[t] += y*y; train_SXY[t] += x*y;
                }
            }
        }
        
        // REDUCTION FOR THIS TILE
        
        unsigned int lane = threadIdx.x % WARP_SIZE;
        unsigned int warp_id = threadIdx.x / WARP_SIZE;
        
        for(int t=0; t<valid_k; t++) {
            int global_k = k_base + t;
            float* d_stats = d_stats_base + (global_k * 12);
            
            // Warp Reduce Train
            float tn = train_N[t], tsx = train_SX[t], tsy = train_SY[t];
            float tsxx = train_SXX[t], tsyy = train_SYY[t], tsxy = train_SXY[t];
            
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                tn += __shfl_down_sync(0xffffffff, tn, offset);
                tsx += __shfl_down_sync(0xffffffff, tsx, offset);
                tsy += __shfl_down_sync(0xffffffff, tsy, offset);
                tsxx += __shfl_down_sync(0xffffffff, tsxx, offset);
                tsyy += __shfl_down_sync(0xffffffff, tsyy, offset);
                tsxy += __shfl_down_sync(0xffffffff, tsxy, offset);
            }
            
             if (lane == 0) {
                shared_train[warp_id * 6 + 0] = tn;
                shared_train[warp_id * 6 + 1] = tsx;
                shared_train[warp_id * 6 + 2] = tsy;
                shared_train[warp_id * 6 + 3] = tsxx;
                shared_train[warp_id * 6 + 4] = tsyy;
                shared_train[warp_id * 6 + 5] = tsxy;
            }
            __syncthreads();
            
            // Block Reduce Train
             if (warp_id == 0) {
                int n_warps = BLOCK_SIZE / WARP_SIZE;
                float final_train[6] = {0};
                for (int w = lane; w < n_warps; w += WARP_SIZE) {
                    for (int j = 0; j < 6; j++) final_train[j] += shared_train[w * 6 + j];
                }
                 #pragma unroll
                 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
                    for (int j=0; j<6; j++) final_train[j] += __shfl_down_sync(0xffffffff, final_train[j], offset);
                 
                  if (lane == 0) {
                    atomicAdd(&d_stats[0], final_train[0]);
                    atomicAdd(&d_stats[1], final_train[1]);
                    atomicAdd(&d_stats[2], final_train[2]);
                    atomicAdd(&d_stats[3], final_train[3]);
                    atomicAdd(&d_stats[4], final_train[4]);
                    atomicAdd(&d_stats[5], final_train[5]);
                  }
             }
             __syncthreads();
             
              // Warp Reduce Val
              float vn = val_N[t], vsx = val_SX[t], vsy = val_SY[t];
              float vsxx = val_SXX[t], vsyy = val_SYY[t], vsxy = val_SXY[t];
              
              #pragma unroll
              for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                  vn += __shfl_down_sync(0xffffffff, vn, offset);
                  vsx += __shfl_down_sync(0xffffffff, vsx, offset);
                  vsy += __shfl_down_sync(0xffffffff, vsy, offset);
                  vsxx += __shfl_down_sync(0xffffffff, vsxx, offset);
                  vsyy += __shfl_down_sync(0xffffffff, vsyy, offset);
                  vsxy += __shfl_down_sync(0xffffffff, vsxy, offset);
              }
               if (lane == 0) {
                shared_val[warp_id * 6 + 0] = vn;
                shared_val[warp_id * 6 + 1] = vsx;
                shared_val[warp_id * 6 + 2] = vsy;
                shared_val[warp_id * 6 + 3] = vsxx;
                shared_val[warp_id * 6 + 4] = vsyy;
                shared_val[warp_id * 6 + 5] = vsxy;
            }
            __syncthreads();
            
            if (warp_id == 0) {
                int n_warps = BLOCK_SIZE / WARP_SIZE;
                float final_val[6] = {0};
                for (int w = lane; w < n_warps; w += WARP_SIZE) {
                    for (int j = 0; j < 6; j++) final_val[j] += shared_val[w * 6 + j];
                }
                 #pragma unroll
                 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
                    for (int j=0; j<6; j++) final_val[j] += __shfl_down_sync(0xffffffff, final_val[j], offset);
                 
                  if (lane == 0) {
                    atomicAdd(&d_stats[6], final_val[0]);
                    atomicAdd(&d_stats[7], final_val[1]);
                    atomicAdd(&d_stats[8], final_val[2]);
                    atomicAdd(&d_stats[9], final_val[3]);
                    atomicAdd(&d_stats[10], final_val[4]);
                    atomicAdd(&d_stats[11], final_val[5]);
                  }
             }
             __syncthreads();
        }
    }
}

extern "C" {

GAFIME_API int gafime_contiguous_bucket_compute_pivot_v2(
    ContiguousBucket bucket,
    int feature_a_idx,
    int op_a_idx,
    const int* feature_b_indices,
    const int* op_b_indices,
    const int* interact_types,
    int n_candidates,
    int val_fold_id,
    float* h_stats_out_flat
) {
    if (!bucket || !feature_b_indices || !h_stats_out_flat || n_candidates <= 0 || n_candidates > MAX_BATCH_SLOTS) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
     ContiguousBucketImpl* impl = static_cast<ContiguousBucketImpl*>(bucket);
     
     // 1. Copy B arrays to Pinned
     size_t batch_bytes = n_candidates * sizeof(int);
     int* dst = impl->h_batch_indices;
     memcpy(dst + 1*MAX_BATCH_SLOTS, feature_b_indices, batch_bytes); // Slot 1 for B
     memcpy(dst + 3*MAX_BATCH_SLOTS, op_b_indices, batch_bytes);      // Slot 3 for OpB
     memcpy(dst + 4*MAX_BATCH_SLOTS, interact_types, batch_bytes);    // Slot 4 for Type
     
     // 2. Upload
     int* d_dst = impl->d_batch_indices;
     cudaMemcpyAsync(d_dst + 1*MAX_BATCH_SLOTS, dst + 1*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
     cudaMemcpyAsync(d_dst + 3*MAX_BATCH_SLOTS, dst + 3*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
     cudaMemcpyAsync(d_dst + 4*MAX_BATCH_SLOTS, dst + 4*MAX_BATCH_SLOTS, batch_bytes, cudaMemcpyHostToDevice, impl->stream);
     
     // 3. Clear Stats
     cudaMemsetAsync(impl->d_stats, 0, n_candidates * 12 * sizeof(float), impl->stream);
     
     // 4. Launch Tiled Kernel
     int calculated_blocks = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
     int max_blks = GET_MAX_BLOCKS();
     int blocks_per_sample = (calculated_blocks < max_blks) ? calculated_blocks : max_blks;
     int shared_mem = (BLOCK_SIZE / WARP_SIZE) * 2 * 6 * sizeof(float);
     
     gafime_contiguous_pivot_kernel_tiled<<<blocks_per_sample, BLOCK_SIZE, shared_mem, impl->stream>>>(
         impl->d_data,
         impl->d_mask,
         impl->n_samples,
         impl->n_features,
         feature_a_idx,
         op_a_idx,
         d_dst + 1*MAX_BATCH_SLOTS, // B
         d_dst + 3*MAX_BATCH_SLOTS, // OpB
         d_dst + 4*MAX_BATCH_SLOTS, // Type
         n_candidates,
         val_fold_id,
         impl->d_stats
     );
     
     // 5. Read Back
     cudaMemcpyAsync(impl->h_stats_pinned, impl->d_stats, n_candidates * 12 * sizeof(float), cudaMemcpyDeviceToHost, impl->stream);
     
     cudaStreamSynchronize(impl->stream);
     
     memcpy(h_stats_out_flat, impl->h_stats_pinned, n_candidates * 12 * sizeof(float));
     return GAFIME_SUCCESS;
}

} // extern "C"


