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
#include <new>      // for std::nothrow
#include <mutex>    // for std::call_once (thread-safe init)

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
static std::once_flag g_gpu_config_flag;

static void auto_tune_for_gpu_impl(int device_id) {
    
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
    cudaDeviceGetAttribute(&g_gpu_config.compute_major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&g_gpu_config.compute_minor, cudaDevAttrComputeCapabilityMinor, device_id);
    g_gpu_config.l2_cache_size = props.l2CacheSize;
    g_gpu_config.max_shared_memory = props.sharedMemPerBlock;
    g_gpu_config.warp_size = props.warpSize;
    
    // =========================================================================
    // AUTO-TUNE BLOCK SIZE based on compute capability
    // =========================================================================
    int compute_cap = g_gpu_config.compute_major * 10 + g_gpu_config.compute_minor;
    
    if (compute_cap >= 120) {
        // Blackwell consumer (RTX 50 series, GB20x) - 5th gen tensor cores
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 8;
    } else if (compute_cap >= 100) {
        // Blackwell datacenter (GB100/GB200) - massive parallelism
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 8;
    } else if (compute_cap >= 90) {
        // Hopper (H100/H200) - 4th gen tensor cores, wgmma
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else if (compute_cap >= 89) {
        // Ada Lovelace (RTX 40 series) - 128 CUDA cores per SM
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else if (compute_cap >= 80) {
        // Ampere (RTX 30 series, A100) - 64/128 cores per SM
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 4;
    } else if (compute_cap >= 75) {
        // Turing (RTX 20 series) - 64 cores per SM, minimum supported
        g_gpu_config.block_size = 256;
        g_gpu_config.max_blocks = props.multiProcessorCount * 2;
    } else {
        // Pre-Turing: deprecated, may be removed in a future release
        fprintf(stderr, "[GAFIME] WARNING: GPU compute capability %d.%d (pre-Turing) is deprecated.\n",
                g_gpu_config.compute_major, g_gpu_config.compute_minor);
        fprintf(stderr, "[GAFIME]   Minimum supported architecture is Turing (sm_75).\n");
        g_gpu_config.block_size = 128;
        g_gpu_config.max_blocks = props.multiProcessorCount * 2;
    }
    
    g_gpu_config.is_initialized = true;
    
    // Log GPU info (runs once during initialization)
    fprintf(stderr, "[GAFIME] Auto-tuned for: %s\n", g_gpu_config.gpu_name);
    fprintf(stderr, "[GAFIME]   SM count: %d, Compute: %d.%d\n", 
            g_gpu_config.sm_count, g_gpu_config.compute_major, g_gpu_config.compute_minor);
    fprintf(stderr, "[GAFIME]   Block size: %d, Max blocks: %d\n",
            g_gpu_config.block_size, g_gpu_config.max_blocks);
    fprintf(stderr, "[GAFIME]   L2 cache: %.1f MB, Shared mem: %d KB\n",
            g_gpu_config.l2_cache_size / (1024.0 * 1024.0), g_gpu_config.max_shared_memory / 1024);
}

static void auto_tune_for_gpu(int device_id = 0) {
    std::call_once(g_gpu_config_flag, auto_tune_for_gpu_impl, device_id);
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
    if (gpu_name_out) {
        strncpy(gpu_name_out, g_gpu_config.gpu_name, 255);
        gpu_name_out[255] = '\0';
    }
    
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

__device__ __forceinline__ float apply_op_fast_value(float x, int op) {
    switch (op) {
        case GAFIME_OP_LOG:
            return __logf(fabsf(x) + 1e-8f);
            
        case GAFIME_OP_EXP:
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
    (void)n_rows;
    (void)window;
    float x = col[idx];
    return apply_op_fast_value(x, op);
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
// CONTINUOUS GLOBAL MATRIX BATCH KERNEL
// ============================================================================
// v0.4.5 continuous scoring uses a full column-major feature matrix and global
// feature indices. This avoids forcing broad arity-5 scans through the old
// 5-slot local bucket contract.

// ============================================================================
// HOST API (extern "C" for ctypes)
// ============================================================================

template <int ARITY>
__global__ void gafime_global_continuous_kernel(
    const float* __restrict__ d_X_colmajor,
    const float* __restrict__ d_target,
    const uint8_t* __restrict__ d_mask,
    const float* __restrict__ d_means,
    const int* __restrict__ batch_indices,
    int batch_size,
    int val_fold_id,
    int n_samples,
    int n_features,
    float* __restrict__ d_stats_batch
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    float train_n = 0, train_sx = 0, train_sy = 0;
    float train_sxx = 0, train_syy = 0, train_sxy = 0;
    float val_n = 0, val_sx = 0, val_sy = 0;
    float val_sxx = 0, val_syy = 0, val_sxy = 0;

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n_samples; row += blockDim.x * gridDim.x) {
        int f0 = batch_indices[batch_id * ARITY + 0];
        if (f0 < 0 || f0 >= n_features) continue;
        float x = d_X_colmajor[static_cast<size_t>(f0) * n_samples + row];
        if constexpr (ARITY > 1) {
            x -= d_means[f0];
        }

        #pragma unroll
        for (int slot = 1; slot < ARITY; ++slot) {
            int feature_idx = batch_indices[batch_id * ARITY + slot];
            if (feature_idx < 0 || feature_idx >= n_features) {
                x = NAN;
                break;
            }
            float value = d_X_colmajor[static_cast<size_t>(feature_idx) * n_samples + row] - d_means[feature_idx];
            x *= value;
        }

        float y = d_target[row];
        uint8_t fold = d_mask[row];
        if (isnan(x) || isnan(y)) continue;

        if (fold == val_fold_id) {
            val_n += 1.0f; val_sx += x; val_sy += y;
            val_sxx += x * x; val_syy += y * y; val_sxy += x * y;
        } else {
            train_n += 1.0f; train_sx += x; train_sy += y;
            train_sxx += x * x; train_syy += y * y; train_sxy += x * y;
        }
    }

    warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    warp_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);

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

    if (warp_id == 0) {
        train_n   = (lane < num_warps) ? shared_train[lane * 6 + 0] : 0.0f;
        train_sx  = (lane < num_warps) ? shared_train[lane * 6 + 1] : 0.0f;
        train_sy  = (lane < num_warps) ? shared_train[lane * 6 + 2] : 0.0f;
        train_sxx = (lane < num_warps) ? shared_train[lane * 6 + 3] : 0.0f;
        train_syy = (lane < num_warps) ? shared_train[lane * 6 + 4] : 0.0f;
        train_sxy = (lane < num_warps) ? shared_train[lane * 6 + 5] : 0.0f;
        val_n   = (lane < num_warps) ? shared_val[lane * 6 + 0] : 0.0f;
        val_sx  = (lane < num_warps) ? shared_val[lane * 6 + 1] : 0.0f;
        val_sy  = (lane < num_warps) ? shared_val[lane * 6 + 2] : 0.0f;
        val_sxx = (lane < num_warps) ? shared_val[lane * 6 + 3] : 0.0f;
        val_syy = (lane < num_warps) ? shared_val[lane * 6 + 4] : 0.0f;
        val_sxy = (lane < num_warps) ? shared_val[lane * 6 + 5] : 0.0f;

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
            float* out = &d_stats_batch[batch_id * 12];
            atomicAdd(&out[0], train_n); atomicAdd(&out[1], train_sx); atomicAdd(&out[2], train_sy);
            atomicAdd(&out[3], train_sxx); atomicAdd(&out[4], train_syy); atomicAdd(&out[5], train_sxy);
            atomicAdd(&out[6], val_n); atomicAdd(&out[7], val_sx); atomicAdd(&out[8], val_sy);
            atomicAdd(&out[9], val_sxx); atomicAdd(&out[10], val_syy); atomicAdd(&out[11], val_sxy);
        }
    }
}

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
 * Maximum batch size for batched compute.
 * Each interaction needs 12 floats output.
 */
#define GAFIME_MAX_BATCH_SIZE 1024

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
    
    // Pre-allocated batch compute buffers (avoids per-call cudaMalloc)
    int* d_batch_kinds;                       // [GAFIME_MAX_BATCH_SIZE]
    int* d_batch_indices;                     // [GAFIME_MAX_BATCH_SIZE * GAFIME_MAX_FEATURES]
    int* d_batch_ops;                         // [GAFIME_MAX_BATCH_SIZE * GAFIME_MAX_FEATURES]
    int* d_batch_interact;                    // [GAFIME_MAX_BATCH_SIZE * (GAFIME_MAX_FEATURES - 1)]
    int* d_batch_ts_params;                   // [GAFIME_MAX_BATCH_SIZE * 4]
    float* d_batch_stats;                     // [GAFIME_MAX_BATCH_SIZE * 12]
    
    // Priority 4: Async operations
    cudaStream_t stream;                      // Compute stream for async operations
    float* h_stats_pinned;                    // Pinned host memory for zero-copy D2H
    
    // Introspection only. GAFIME does not reserve or pin L2 cache; locality is
    // controlled by host-side launch ordering and normal hardware caching.
    size_t total_data_bytes;                  // Total bytes of feature data
};

/**
 * Full feature-matrix resident CUDA layout for broad continuous scans.
 *
 * The 5-slot bucket is efficient for a small local working set. It is not a
 * valid memory contract for broad arity-5 search, where each candidate may use
 * a different 5-feature universe. This matrix handle keeps all features in one
 * column-major allocation and uses global feature indices in the batch
 * descriptor.
 */
struct GafimeCudaMatrixImpl {
    int n_samples;
    int n_features;
    int max_batch_size;
    float* d_X_colmajor;
    float* d_target;
    uint8_t* d_mask;
    float* d_means;
    int* d_batch_indices;
    float* d_batch_stats;
    cudaStream_t stream;
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
    bucket->d_batch_kinds = nullptr;
    bucket->d_batch_indices = nullptr;
    bucket->d_batch_ops = nullptr;
    bucket->d_batch_interact = nullptr;
    bucket->d_batch_ts_params = nullptr;
    bucket->d_batch_stats = nullptr;
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
    // Pre-allocate batch compute buffers (avoids per-call cudaMalloc)
    // =========================================================================
    err = cudaMalloc(&bucket->d_batch_kinds, GAFIME_MAX_BATCH_SIZE * sizeof(int));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&bucket->d_batch_indices, GAFIME_MAX_BATCH_SIZE * GAFIME_MAX_FEATURES * sizeof(int));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&bucket->d_batch_ops, GAFIME_MAX_BATCH_SIZE * GAFIME_MAX_FEATURES * sizeof(int));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&bucket->d_batch_interact, GAFIME_MAX_BATCH_SIZE * (GAFIME_MAX_FEATURES - 1) * sizeof(int));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&bucket->d_batch_ts_params, GAFIME_MAX_BATCH_SIZE * 4 * sizeof(int));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&bucket->d_batch_stats, GAFIME_MAX_BATCH_SIZE * 12 * sizeof(float));
    if (err != cudaSuccess) { gafime_bucket_free(bucket); return GAFIME_ERROR_OUT_OF_MEMORY; }
    
    // Track total resident bytes for diagnostics and future sizing heuristics.
    // This is not a CUDA L2 persistence or access-policy-window setup.
    bucket->total_data_bytes = n_features * vec_bytes + vec_bytes + mask_bytes;
    
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
    if (impl->d_batch_kinds) cudaFree(impl->d_batch_kinds);
    if (impl->d_batch_indices) cudaFree(impl->d_batch_indices);
    if (impl->d_batch_ops) cudaFree(impl->d_batch_ops);
    if (impl->d_batch_interact) cudaFree(impl->d_batch_interact);
    if (impl->d_batch_ts_params) cudaFree(impl->d_batch_ts_params);
    if (impl->d_batch_stats) cudaFree(impl->d_batch_stats);
    
    // Free Priority 4 resources
    if (impl->stream) cudaStreamDestroy(impl->stream);
    if (impl->h_stats_pinned) cudaFreeHost(impl->h_stats_pinned);
    
    delete impl;
    return GAFIME_SUCCESS;
}

}  // extern "C"

// ============================================================================
// BATCHED COMPUTE API (Priority 3: Minimize kernel launch overhead)
// ============================================================================


__device__ __forceinline__ float ts_candidate_value(
    const float* __restrict__ col,
    int idx,
    int n_samples,
    int candidate_kind,
    const int* __restrict__ params
) {
    int lag = max(params[0], 1);
    int window = max(params[1], 1);
    if (candidate_kind == GAFIME_CANDIDATE_CONTINUOUS) {
        return col[idx];
    }
    int lag_idx = max(idx - lag, 0);
    if (candidate_kind == GAFIME_CANDIDATE_TS_LAG) {
        return col[lag_idx];
    }
    if (candidate_kind == GAFIME_CANDIDATE_TS_DELTA) {
        return col[idx] - col[lag_idx];
    }
    if (candidate_kind == GAFIME_CANDIDATE_TS_VELOCITY) {
        return (col[idx] - col[lag_idx]) / static_cast<float>(lag);
    }
    if (candidate_kind == GAFIME_CANDIDATE_TS_ACCELERATION) {
        int lag2_idx = max(idx - 2 * lag, 0);
        float lag_span = static_cast<float>(lag * lag);
        return (col[idx] - 2.0f * col[lag_idx] + col[lag2_idx]) / lag_span;
    }
    if (candidate_kind == GAFIME_CANDIDATE_TS_ROLLING_SUM ||
        candidate_kind == GAFIME_CANDIDATE_TS_ROLLING_MEAN ||
        candidate_kind == GAFIME_CANDIDATE_TS_ROLLING_STD) {
        int start = max(0, idx - window + 1);
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;
        for (int i = start; i <= idx && i < n_samples; ++i) {
            float value = col[i];
            sum += value;
            sum_sq += value * value;
            count += 1;
        }
        if (candidate_kind == GAFIME_CANDIDATE_TS_ROLLING_SUM) {
            return sum;
        }
        float mean = sum / static_cast<float>(max(count, 1));
        if (candidate_kind == GAFIME_CANDIDATE_TS_ROLLING_MEAN) {
            return mean;
        }
        float variance = fmaxf(sum_sq / static_cast<float>(max(count, 1)) - mean * mean, 0.0f);
        return sqrtf(variance);
    }
    return col[idx];
}

template <int ARITY>
__global__ void gafime_batched_kernel(
    float* __restrict__ d_features_0,
    float* __restrict__ d_features_1,
    float* __restrict__ d_features_2,
    float* __restrict__ d_features_3,
    float* __restrict__ d_features_4,
    const float* __restrict__ d_target,
    const uint8_t* __restrict__ d_mask,
    const int* __restrict__ batch_kinds,      // [N]
    const int* __restrict__ batch_indices,    // [N * ARITY]
    const int* __restrict__ batch_ops,        // [N * ARITY]
    const int* __restrict__ batch_interact,   // [N * (ARITY - 1)]
    const int* __restrict__ batch_ts_params,  // [N * 4]
    int batch_size,
    int val_fold_id,
    int n_samples,
    float* __restrict__ d_stats_batch
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const float* features[5] = {d_features_0, d_features_1, d_features_2, d_features_3, d_features_4};
    int candidate_kind = batch_kinds ? batch_kinds[batch_id] : GAFIME_CANDIDATE_CONTINUOUS;
    const int* ts_params = batch_ts_params ? &batch_ts_params[batch_id * 4] : nullptr;
    int default_ts_params[4] = {1, 1, 0, 0};
    if (!ts_params) {
        ts_params = default_ts_params;
    }

    float train_n = 0, train_sx = 0, train_sy = 0;
    float train_sxx = 0, train_syy = 0, train_sxy = 0;
    float val_n = 0, val_sx = 0, val_sy = 0;
    float val_sxx = 0, val_syy = 0, val_sxy = 0;

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n_samples; row += blockDim.x * gridDim.x) {
        int f0_idx = batch_indices[batch_id * ARITY + 0];
        int op0 = batch_ops[batch_id * ARITY + 0];
        float x = apply_op_fast_value(
            ts_candidate_value(features[f0_idx], row, n_samples, candidate_kind, ts_params), op0
        );

        #pragma unroll
        for (int slot = 1; slot < ARITY; ++slot) {
            int feature_idx = batch_indices[batch_id * ARITY + slot];
            int op = batch_ops[batch_id * ARITY + slot];
            int interaction = batch_interact[batch_id * (ARITY - 1) + (slot - 1)];
            float value = apply_op_fast_value(
                ts_candidate_value(features[feature_idx], row, n_samples, candidate_kind, ts_params), op
            );
            x = combine(x, value, interaction);
        }

        float y = d_target[row];
        uint8_t fold = d_mask[row];
        if (isnan(x) || isnan(y)) continue;

        if (fold == val_fold_id) {
            val_n += 1.0f; val_sx += x; val_sy += y;
            val_sxx += x * x; val_syy += y * y; val_sxy += x * y;
        } else {
            train_n += 1.0f; train_sx += x; train_sy += y;
            train_sxx += x * x; train_syy += y * y; train_sxy += x * y;
        }
    }

    warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);
    warp_reduce_6(val_n, val_sx, val_sy, val_sxx, val_syy, val_sxy);

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

    if (warp_id == 0) {
        train_n   = (lane < num_warps) ? shared_train[lane * 6 + 0] : 0.0f;
        train_sx  = (lane < num_warps) ? shared_train[lane * 6 + 1] : 0.0f;
        train_sy  = (lane < num_warps) ? shared_train[lane * 6 + 2] : 0.0f;
        train_sxx = (lane < num_warps) ? shared_train[lane * 6 + 3] : 0.0f;
        train_syy = (lane < num_warps) ? shared_train[lane * 6 + 4] : 0.0f;
        train_sxy = (lane < num_warps) ? shared_train[lane * 6 + 5] : 0.0f;
        val_n   = (lane < num_warps) ? shared_val[lane * 6 + 0] : 0.0f;
        val_sx  = (lane < num_warps) ? shared_val[lane * 6 + 1] : 0.0f;
        val_sy  = (lane < num_warps) ? shared_val[lane * 6 + 2] : 0.0f;
        val_sxx = (lane < num_warps) ? shared_val[lane * 6 + 3] : 0.0f;
        val_syy = (lane < num_warps) ? shared_val[lane * 6 + 4] : 0.0f;
        val_sxy = (lane < num_warps) ? shared_val[lane * 6 + 5] : 0.0f;

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
            float* out = &d_stats_batch[batch_id * 12];
            atomicAdd(&out[0], train_n); atomicAdd(&out[1], train_sx); atomicAdd(&out[2], train_sy);
            atomicAdd(&out[3], train_sxx); atomicAdd(&out[4], train_syy); atomicAdd(&out[5], train_sxy);
            atomicAdd(&out[6], val_n); atomicAdd(&out[7], val_sx); atomicAdd(&out[8], val_sy);
            atomicAdd(&out[9], val_sxx); atomicAdd(&out[10], val_syy); atomicAdd(&out[11], val_sxy);
        }
    }
}

extern "C" {

GAFIME_API int gafime_cuda_matrix_alloc(
    int n_samples,
    int n_features,
    int max_batch_size,
    GafimeCudaMatrix* matrix_out
) {
    if (n_samples <= 0 || n_features <= 0 || max_batch_size <= 0 || max_batch_size > GAFIME_MAX_BATCH_SIZE || !matrix_out) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    GafimeCudaMatrixImpl* matrix = new (std::nothrow) GafimeCudaMatrixImpl;
    if (!matrix) {
        return GAFIME_ERROR_OUT_OF_MEMORY;
    }
    matrix->n_samples = n_samples;
    matrix->n_features = n_features;
    matrix->max_batch_size = max_batch_size;
    matrix->d_X_colmajor = nullptr;
    matrix->d_target = nullptr;
    matrix->d_mask = nullptr;
    matrix->d_means = nullptr;
    matrix->d_batch_indices = nullptr;
    matrix->d_batch_stats = nullptr;
    matrix->stream = nullptr;

    cudaError_t err = cudaStreamCreate(&matrix->stream);
    if (err != cudaSuccess) {
        delete matrix;
        return GAFIME_ERROR_KERNEL_FAILED;
    }

    size_t n = static_cast<size_t>(n_samples);
    size_t f = static_cast<size_t>(n_features);
    err = cudaMalloc(&matrix->d_X_colmajor, n * f * sizeof(float));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&matrix->d_target, n * sizeof(float));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&matrix->d_mask, n * sizeof(uint8_t));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&matrix->d_means, f * sizeof(float));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&matrix->d_batch_indices, static_cast<size_t>(max_batch_size) * GAFIME_MAX_FEATURES * sizeof(int));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }
    err = cudaMalloc(&matrix->d_batch_stats, static_cast<size_t>(max_batch_size) * 12 * sizeof(float));
    if (err != cudaSuccess) { gafime_cuda_matrix_free(matrix); return GAFIME_ERROR_OUT_OF_MEMORY; }

    *matrix_out = static_cast<GafimeCudaMatrix>(matrix);
    return GAFIME_SUCCESS;
}

GAFIME_API int gafime_cuda_matrix_upload(
    GafimeCudaMatrix matrix_handle,
    const float* h_X_colmajor,
    const float* h_y,
    const uint8_t* h_mask,
    const float* h_means
) {
    if (!matrix_handle || !h_X_colmajor || !h_y || !h_mask || !h_means) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    GafimeCudaMatrixImpl* matrix = static_cast<GafimeCudaMatrixImpl*>(matrix_handle);
    size_t n = static_cast<size_t>(matrix->n_samples);
    size_t f = static_cast<size_t>(matrix->n_features);
    cudaError_t err;
    err = cudaMemcpyAsync(matrix->d_X_colmajor, h_X_colmajor, n * f * sizeof(float), cudaMemcpyHostToDevice, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemcpyAsync(matrix->d_target, h_y, n * sizeof(float), cudaMemcpyHostToDevice, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemcpyAsync(matrix->d_mask, h_mask, n * sizeof(uint8_t), cudaMemcpyHostToDevice, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemcpyAsync(matrix->d_means, h_means, f * sizeof(float), cudaMemcpyHostToDevice, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaStreamSynchronize(matrix->stream);
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_cuda_matrix_compute_batch(
    GafimeCudaMatrix matrix_handle,
    const int* h_batch_indices,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
) {
    if (!matrix_handle || !h_batch_indices || !h_stats_batch) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    GafimeCudaMatrixImpl* matrix = static_cast<GafimeCudaMatrixImpl*>(matrix_handle);
    if (arity < 1 || arity > GAFIME_MAX_FEATURES || batch_size <= 0 || batch_size > matrix->max_batch_size) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    size_t indices_bytes = static_cast<size_t>(batch_size) * arity * sizeof(int);
    size_t stats_bytes = static_cast<size_t>(batch_size) * 12 * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(matrix->d_batch_indices, h_batch_indices, indices_bytes, cudaMemcpyHostToDevice, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemsetAsync(matrix->d_batch_stats, 0, stats_bytes, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;

    auto_tune_for_gpu();
    int blocks_per_candidate = (matrix->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks_per_candidate = (blocks_per_candidate < 64) ? blocks_per_candidate : 64;
    dim3 grid(blocks_per_candidate, batch_size);
    dim3 block(BLOCK_SIZE);

    switch (arity) {
        case 1:
            gafime_global_continuous_kernel<1><<<grid, block, 0, matrix->stream>>>(
                matrix->d_X_colmajor, matrix->d_target, matrix->d_mask, matrix->d_means,
                matrix->d_batch_indices, batch_size, val_fold_id, matrix->n_samples, matrix->n_features, matrix->d_batch_stats);
            break;
        case 2:
            gafime_global_continuous_kernel<2><<<grid, block, 0, matrix->stream>>>(
                matrix->d_X_colmajor, matrix->d_target, matrix->d_mask, matrix->d_means,
                matrix->d_batch_indices, batch_size, val_fold_id, matrix->n_samples, matrix->n_features, matrix->d_batch_stats);
            break;
        case 3:
            gafime_global_continuous_kernel<3><<<grid, block, 0, matrix->stream>>>(
                matrix->d_X_colmajor, matrix->d_target, matrix->d_mask, matrix->d_means,
                matrix->d_batch_indices, batch_size, val_fold_id, matrix->n_samples, matrix->n_features, matrix->d_batch_stats);
            break;
        case 4:
            gafime_global_continuous_kernel<4><<<grid, block, 0, matrix->stream>>>(
                matrix->d_X_colmajor, matrix->d_target, matrix->d_mask, matrix->d_means,
                matrix->d_batch_indices, batch_size, val_fold_id, matrix->n_samples, matrix->n_features, matrix->d_batch_stats);
            break;
        case 5:
            gafime_global_continuous_kernel<5><<<grid, block, 0, matrix->stream>>>(
                matrix->d_X_colmajor, matrix->d_target, matrix->d_mask, matrix->d_means,
                matrix->d_batch_indices, batch_size, val_fold_id, matrix->n_samples, matrix->n_features, matrix->d_batch_stats);
            break;
        default:
            return GAFIME_ERROR_INVALID_ARGS;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaMemcpyAsync(h_stats_batch, matrix->d_batch_stats, stats_bytes, cudaMemcpyDeviceToHost, matrix->stream);
    if (err != cudaSuccess) return GAFIME_ERROR_KERNEL_FAILED;
    err = cudaStreamSynchronize(matrix->stream);
    return (err == cudaSuccess) ? GAFIME_SUCCESS : GAFIME_ERROR_KERNEL_FAILED;
}

GAFIME_API int gafime_cuda_matrix_free(GafimeCudaMatrix matrix_handle) {
    if (!matrix_handle) {
        return GAFIME_SUCCESS;
    }
    GafimeCudaMatrixImpl* matrix = static_cast<GafimeCudaMatrixImpl*>(matrix_handle);
    if (matrix->d_batch_stats) cudaFree(matrix->d_batch_stats);
    if (matrix->d_batch_indices) cudaFree(matrix->d_batch_indices);
    if (matrix->d_means) cudaFree(matrix->d_means);
    if (matrix->d_mask) cudaFree(matrix->d_mask);
    if (matrix->d_target) cudaFree(matrix->d_target);
    if (matrix->d_X_colmajor) cudaFree(matrix->d_X_colmajor);
    if (matrix->stream) cudaStreamDestroy(matrix->stream);
    delete matrix;
    return GAFIME_SUCCESS;
}

GAFIME_API int gafime_bucket_compute_batch(
    GafimeBucket bucket,
    const int* h_batch_kinds,
    const int* h_batch_indices,
    const int* h_batch_ops,
    const int* h_batch_interact,
    const int* h_batch_ts_params,
    int arity,
    int batch_size,
    int val_fold_id,
    float* h_stats_batch
) {
    if (!bucket || !h_batch_indices || !h_batch_ops || !h_batch_interact || !h_stats_batch) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (arity < 1 || arity > GAFIME_MAX_FEATURES || batch_size <= 0 || batch_size > GAFIME_MAX_BATCH_SIZE) {
        return GAFIME_ERROR_INVALID_ARGS;
    }

    GafimeBucketImpl* impl = static_cast<GafimeBucketImpl*>(bucket);
    cudaError_t err;

    size_t kinds_bytes = batch_size * sizeof(int);
    size_t indices_bytes = static_cast<size_t>(batch_size) * arity * sizeof(int);
    size_t ops_bytes = static_cast<size_t>(batch_size) * arity * sizeof(int);
    size_t interact_bytes = static_cast<size_t>(batch_size) * max(arity - 1, 1) * sizeof(int);
    size_t ts_params_bytes = static_cast<size_t>(batch_size) * 4 * sizeof(int);
    size_t stats_bytes = static_cast<size_t>(batch_size) * 12 * sizeof(float);

    cudaMemcpyAsync(impl->d_batch_indices, h_batch_indices, indices_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(impl->d_batch_ops, h_batch_ops, ops_bytes, cudaMemcpyHostToDevice, impl->stream);
    cudaMemcpyAsync(impl->d_batch_interact, h_batch_interact, interact_bytes, cudaMemcpyHostToDevice, impl->stream);
    if (h_batch_kinds) {
        cudaMemcpyAsync(impl->d_batch_kinds, h_batch_kinds, kinds_bytes, cudaMemcpyHostToDevice, impl->stream);
    } else {
        cudaMemsetAsync(impl->d_batch_kinds, 0, kinds_bytes, impl->stream);
    }
    if (h_batch_ts_params) {
        cudaMemcpyAsync(impl->d_batch_ts_params, h_batch_ts_params, ts_params_bytes, cudaMemcpyHostToDevice, impl->stream);
    } else {
        cudaMemsetAsync(impl->d_batch_ts_params, 0, ts_params_bytes, impl->stream);
    }
    cudaMemsetAsync(impl->d_batch_stats, 0, stats_bytes, impl->stream);

    auto_tune_for_gpu();
    int blocks_per_interaction = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks_per_interaction = min(blocks_per_interaction, 64);
    dim3 grid(blocks_per_interaction, batch_size);
    dim3 block(BLOCK_SIZE);

    switch (arity) {
        case 1:
            gafime_batched_kernel<1><<<grid, block, 0, impl->stream>>>(
                impl->d_features[0], impl->d_features[1], impl->d_features[2], impl->d_features[3], impl->d_features[4],
                impl->d_target, impl->d_mask, impl->d_batch_kinds, impl->d_batch_indices, impl->d_batch_ops,
                impl->d_batch_interact, impl->d_batch_ts_params, batch_size, val_fold_id, impl->n_samples, impl->d_batch_stats);
            break;
        case 2:
            gafime_batched_kernel<2><<<grid, block, 0, impl->stream>>>(
                impl->d_features[0], impl->d_features[1], impl->d_features[2], impl->d_features[3], impl->d_features[4],
                impl->d_target, impl->d_mask, impl->d_batch_kinds, impl->d_batch_indices, impl->d_batch_ops,
                impl->d_batch_interact, impl->d_batch_ts_params, batch_size, val_fold_id, impl->n_samples, impl->d_batch_stats);
            break;
        case 3:
            gafime_batched_kernel<3><<<grid, block, 0, impl->stream>>>(
                impl->d_features[0], impl->d_features[1], impl->d_features[2], impl->d_features[3], impl->d_features[4],
                impl->d_target, impl->d_mask, impl->d_batch_kinds, impl->d_batch_indices, impl->d_batch_ops,
                impl->d_batch_interact, impl->d_batch_ts_params, batch_size, val_fold_id, impl->n_samples, impl->d_batch_stats);
            break;
        case 4:
            gafime_batched_kernel<4><<<grid, block, 0, impl->stream>>>(
                impl->d_features[0], impl->d_features[1], impl->d_features[2], impl->d_features[3], impl->d_features[4],
                impl->d_target, impl->d_mask, impl->d_batch_kinds, impl->d_batch_indices, impl->d_batch_ops,
                impl->d_batch_interact, impl->d_batch_ts_params, batch_size, val_fold_id, impl->n_samples, impl->d_batch_stats);
            break;
        case 5:
            gafime_batched_kernel<5><<<grid, block, 0, impl->stream>>>(
                impl->d_features[0], impl->d_features[1], impl->d_features[2], impl->d_features[3], impl->d_features[4],
                impl->d_target, impl->d_mask, impl->d_batch_kinds, impl->d_batch_indices, impl->d_batch_ops,
                impl->d_batch_interact, impl->d_batch_ts_params, batch_size, val_fold_id, impl->n_samples, impl->d_batch_stats);
            break;
        default:
            return GAFIME_ERROR_INVALID_ARGS;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    err = cudaStreamSynchronize(impl->stream);
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    cudaMemcpy(h_stats_batch, impl->d_batch_stats, stats_bytes, cudaMemcpyDeviceToHost);
    return GAFIME_SUCCESS;
}


// ============================================================================
// LEGACY PER-CALL, INTERLEAVED, AND CONTIGUOUS CUDA PATHS REMOVED
// ============================================================================
// v0.4.5 routes broad continuous work through gafime_cuda_matrix_compute_batch.
// The bucket batch path remains for explicit time-series candidate transforms.

// ============================================================================
// DISCRETE SOFT FUNCTION FAMILY
// ============================================================================

__device__ __forceinline__ float discrete_sigmoid(float z) {
    float z_clamped = fminf(fmaxf(z, -60.0f), 60.0f);
    return __fdividef(1.0f, 1.0f + __expf(-z_clamped));
}

__device__ __forceinline__ float discrete_scale(float scale) {
    return scale > 1e-12f ? scale : 1.0f;
}

__device__ __forceinline__ float discrete_threshold_gate(
    float x,
    float threshold,
    int direction,
    float scale,
    float sharpness
) {
    float sign = (direction == GAFIME_DISCRETE_DIRECTION_LE) ? -1.0f : 1.0f;
    float z = sharpness * sign * (x - threshold) / discrete_scale(scale);
    return discrete_sigmoid(z);
}

__device__ __forceinline__ float discrete_interval_gate(
    float x,
    float low,
    float high,
    float scale,
    float sharpness
) {
    float safe_scale = discrete_scale(scale);
    float left = discrete_sigmoid(sharpness * (x - low) / safe_scale);
    float right = discrete_sigmoid(sharpness * (high - x) / safe_scale);
    return left * right;
}

__device__ __forceinline__ float discrete_eval_soft(
    const float* __restrict__ X,
    int n_samples,
    int i,
    int kind,
    int feature_a,
    int feature_b,
    int value_feature,
    int direction,
    const float* __restrict__ params,
    const float* __restrict__ scales,
    float sharpness
) {
    const float* feature0 = X + feature_a * n_samples;
    float a = feature0[i];

    switch (kind) {
        case GAFIME_DISCRETE_SOFT_THRESHOLD:
            return discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );

        case GAFIME_DISCRETE_SOFT_INTERVAL:
            return discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );

        case GAFIME_DISCRETE_VALUE_GATED_THRESHOLD: {
            const float* value_col = X + value_feature * n_samples;
            float gate = discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );
            return value_col[i] * gate;
        }

        case GAFIME_DISCRETE_SOFT_RECTANGLE: {
            const float* feature1 = X + feature_b * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[i], params[2], params[3], scales[1], sharpness
            );
            return mask0 * mask1;
        }

        case GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE: {
            const float* feature1 = X + feature_b * n_samples;
            const float* value_col = X + value_feature * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[i], params[2], params[3], scales[1], sharpness
            );
            return value_col[i] * mask0 * mask1;
        }

        default:
            return NAN;
    }
}

__device__ __forceinline__ float discrete_eval_mask_soft(
    const float* __restrict__ X,
    int n_samples,
    int i,
    int kind,
    int feature_a,
    int feature_b,
    int direction,
    const float* __restrict__ params,
    const float* __restrict__ scales,
    float sharpness
) {
    const float* feature0 = X + feature_a * n_samples;
    float a = feature0[i];

    switch (kind) {
        case GAFIME_DISCRETE_SOFT_THRESHOLD:
        case GAFIME_DISCRETE_VALUE_GATED_THRESHOLD:
            return discrete_threshold_gate(
                a, params[0], direction, scales[0], sharpness
            );

        case GAFIME_DISCRETE_SOFT_INTERVAL:
            return discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );

        case GAFIME_DISCRETE_SOFT_RECTANGLE:
        case GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE: {
            const float* feature1 = X + feature_b * n_samples;
            float mask0 = discrete_interval_gate(
                a, params[0], params[1], scales[0], sharpness
            );
            float mask1 = discrete_interval_gate(
                feature1[i], params[2], params[3], scales[1], sharpness
            );
            return mask0 * mask1;
        }

        default:
            return NAN;
    }
}

__global__ void gafime_discrete_soft_batch_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    const int* __restrict__ kinds,
    const int* __restrict__ feature_a,
    const int* __restrict__ feature_b,
    const int* __restrict__ value_feature,
    const int* __restrict__ directions,
    const float* __restrict__ params,
    const float* __restrict__ scales,
    const float* __restrict__ sharpness,
    int n_samples,
    int n_candidates,
    float* __restrict__ stats_batch
) {
    int candidate_id = blockIdx.y;
    if (candidate_id >= n_candidates) return;

    int kind = kinds[candidate_id];
    int fa = feature_a[candidate_id];
    int fb = feature_b[candidate_id];
    int vf = value_feature[candidate_id];
    int direction = directions[candidate_id];
    const float* candidate_params = params + candidate_id * 4;
    const float* candidate_scales = scales + candidate_id * 2;
    float k = sharpness[candidate_id];

    float train_n = 0.0f, train_sx = 0.0f, train_sy = 0.0f;
    float train_sxx = 0.0f, train_syy = 0.0f, train_sxy = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_samples;
         i += blockDim.x * gridDim.x) {
        float x = discrete_eval_soft(
            X, n_samples, i, kind, fa, fb, vf, direction,
            candidate_params, candidate_scales, k
        );
        float target = y[i];
        if (isnan(x) || isnan(target)) continue;

        train_n += 1.0f;
        train_sx += x;
        train_sy += target;
        train_sxx += x * x;
        train_syy += target * target;
        train_sxy += x * target;
    }

    warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);

    __shared__ float shared_train[6 * (BLOCK_SIZE / WARP_SIZE)];
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
    }
    __syncthreads();

    if (warp_id == 0) {
        train_n = (lane < num_warps) ? shared_train[lane * 6 + 0] : 0.0f;
        train_sx = (lane < num_warps) ? shared_train[lane * 6 + 1] : 0.0f;
        train_sy = (lane < num_warps) ? shared_train[lane * 6 + 2] : 0.0f;
        train_sxx = (lane < num_warps) ? shared_train[lane * 6 + 3] : 0.0f;
        train_syy = (lane < num_warps) ? shared_train[lane * 6 + 4] : 0.0f;
        train_sxy = (lane < num_warps) ? shared_train[lane * 6 + 5] : 0.0f;

        warp_reduce_6(train_n, train_sx, train_sy, train_sxx, train_syy, train_sxy);

        if (lane == 0) {
            float* out = stats_batch + candidate_id * GAFIME_STATS_SIZE;
            atomicAdd(&out[GAFIME_STAT_TRAIN_N], train_n);
            atomicAdd(&out[GAFIME_STAT_TRAIN_SX], train_sx);
            atomicAdd(&out[GAFIME_STAT_TRAIN_SY], train_sy);
            atomicAdd(&out[GAFIME_STAT_TRAIN_SXX], train_sxx);
            atomicAdd(&out[GAFIME_STAT_TRAIN_SYY], train_syy);
            atomicAdd(&out[GAFIME_STAT_TRAIN_SXY], train_sxy);
        }
    }
}

GAFIME_API int gafime_discrete_soft_batch_cuda(
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
    float* h_stats_batch
) {
    if (!h_X || !h_y || !h_kinds || !h_feature_a || !h_feature_b ||
        !h_value_feature || !h_directions || !h_params || !h_scales ||
        !h_sharpness || !h_stats_batch) {
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

    float* d_X = nullptr;
    float* d_y = nullptr;
    int* d_kinds = nullptr;
    int* d_feature_a = nullptr;
    int* d_feature_b = nullptr;
    int* d_value_feature = nullptr;
    int* d_directions = nullptr;
    float* d_params = nullptr;
    float* d_scales = nullptr;
    float* d_sharpness = nullptr;
    float* d_stats = nullptr;

    size_t X_bytes = static_cast<size_t>(n_samples) * n_features * sizeof(float);
    size_t y_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t int_bytes = static_cast<size_t>(n_candidates) * sizeof(int);
    size_t params_bytes = static_cast<size_t>(n_candidates) * 4 * sizeof(float);
    size_t scales_bytes = static_cast<size_t>(n_candidates) * 2 * sizeof(float);
    size_t sharpness_bytes = static_cast<size_t>(n_candidates) * sizeof(float);
    size_t stats_bytes = static_cast<size_t>(n_candidates) * GAFIME_STATS_SIZE * sizeof(float);

    cudaError_t err;
    int status = GAFIME_SUCCESS;

    err = cudaMalloc(&d_X, X_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_y, y_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_kinds, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_a, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_b, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_value_feature, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_directions, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_params, params_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_scales, scales_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_sharpness, sharpness_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_stats, stats_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }

    err = cudaMemcpy(d_X, h_X, X_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_y, h_y, y_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_kinds, h_kinds, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_a, h_feature_a, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_b, h_feature_b, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_value_feature, h_value_feature, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_directions, h_directions, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_scales, h_scales, scales_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_sharpness, h_sharpness, sharpness_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemset(d_stats, 0, stats_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

    auto_tune_for_gpu();
    {
        int blocks_per_candidate = min((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, GET_MAX_BLOCKS());
        dim3 grid(blocks_per_candidate, n_candidates);
        dim3 block(BLOCK_SIZE);
        gafime_discrete_soft_batch_kernel<<<grid, block>>>(
            d_X, d_y, d_kinds, d_feature_a, d_feature_b, d_value_feature,
            d_directions, d_params, d_scales, d_sharpness,
            n_samples, n_candidates, d_stats
        );
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(h_stats_batch, d_stats, stats_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

cleanup:
    if (d_stats) cudaFree(d_stats);
    if (d_sharpness) cudaFree(d_sharpness);
    if (d_scales) cudaFree(d_scales);
    if (d_params) cudaFree(d_params);
    if (d_directions) cudaFree(d_directions);
    if (d_value_feature) cudaFree(d_value_feature);
    if (d_feature_b) cudaFree(d_feature_b);
    if (d_feature_a) cudaFree(d_feature_a);
    if (d_kinds) cudaFree(d_kinds);
    if (d_y) cudaFree(d_y);
    if (d_X) cudaFree(d_X);
    return status;
}

__global__ void gafime_discrete_selection_batch_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    const float* __restrict__ residual,
    const int* __restrict__ kinds,
    const int* __restrict__ feature_a,
    const int* __restrict__ feature_b,
    const int* __restrict__ value_feature,
    const int* __restrict__ directions,
    const float* __restrict__ params,
    const float* __restrict__ scales,
    const float* __restrict__ sharpness,
    int n_samples,
    int n_candidates,
    int mi_bins,
    float y_min,
    float y_max,
    float y_sum,
    float y_sq_sum,
    float* __restrict__ scores_batch
) {
    int candidate_id = blockIdx.x;
    if (candidate_id >= n_candidates) return;

    int bins = min(max(mi_bins, 2), 16);
    int kind = kinds[candidate_id];
    int fa = feature_a[candidate_id];
    int fb = feature_b[candidate_id];
    int vf = value_feature[candidate_id];
    int direction = directions[candidate_id];
    const float* candidate_params = params + candidate_id * 4;
    const float* candidate_scales = scales + candidate_id * 2;
    float k = sharpness[candidate_id];

    __shared__ unsigned int hist_xy[16 * 16];
    __shared__ unsigned int hist_x[16];
    __shared__ unsigned int hist_y[16];
    __shared__ float partials[9 * BLOCK_SIZE];

    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        hist_xy[idx] = 0;
    }
    for (int idx = threadIdx.x; idx < 16; idx += blockDim.x) {
        hist_x[idx] = 0;
        hist_y[idx] = 0;
    }
    __syncthreads();

    float sw = 0.0f, swy = 0.0f, swyy = 0.0f;
    float n = 0.0f, sx = 0.0f, sr = 0.0f;
    float sxx = 0.0f, srr = 0.0f, sxr = 0.0f;
    float y_range = y_max - y_min;

    for (int i = threadIdx.x; i < n_samples; i += blockDim.x) {
        float feature_value = discrete_eval_soft(
            X, n_samples, i, kind, fa, fb, vf, direction,
            candidate_params, candidate_scales, k
        );
        float mask = discrete_eval_mask_soft(
            X, n_samples, i, kind, fa, fb, direction,
            candidate_params, candidate_scales, k
        );
        float target = y[i];
        float res = residual[i];
        if (isnan(feature_value) || isnan(mask) || isnan(target) || isnan(res)) {
            continue;
        }

        mask = fminf(fmaxf(mask, 0.0f), 1.0f);
        sw += mask;
        swy += mask * target;
        swyy += mask * target * target;

        n += 1.0f;
        sx += feature_value;
        sr += res;
        sxx += feature_value * feature_value;
        srr += res * res;
        sxr += feature_value * res;

        int xb = min(max(static_cast<int>(mask * bins), 0), bins - 1);
        int yb = 0;
        if (y_range > 1e-12f) {
            yb = min(max(static_cast<int>(((target - y_min) / y_range) * bins), 0), bins - 1);
        }
        atomicAdd(&hist_xy[xb * 16 + yb], 1u);
        atomicAdd(&hist_x[xb], 1u);
        atomicAdd(&hist_y[yb], 1u);
    }

    int t = threadIdx.x;
    partials[t] = sw;
    partials[BLOCK_SIZE + t] = swy;
    partials[2 * BLOCK_SIZE + t] = swyy;
    partials[3 * BLOCK_SIZE + t] = n;
    partials[4 * BLOCK_SIZE + t] = sx;
    partials[5 * BLOCK_SIZE + t] = sr;
    partials[6 * BLOCK_SIZE + t] = sxx;
    partials[7 * BLOCK_SIZE + t] = srr;
    partials[8 * BLOCK_SIZE + t] = sxr;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_sw = 0.0f, sum_swy = 0.0f, sum_swyy = 0.0f;
        float sum_n = 0.0f, sum_sx = 0.0f, sum_sr = 0.0f;
        float sum_sxx = 0.0f, sum_srr = 0.0f, sum_sxr = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum_sw += partials[i];
            sum_swy += partials[BLOCK_SIZE + i];
            sum_swyy += partials[2 * BLOCK_SIZE + i];
            sum_n += partials[3 * BLOCK_SIZE + i];
            sum_sx += partials[4 * BLOCK_SIZE + i];
            sum_sr += partials[5 * BLOCK_SIZE + i];
            sum_sxx += partials[6 * BLOCK_SIZE + i];
            sum_srr += partials[7 * BLOCK_SIZE + i];
            sum_sxr += partials[8 * BLOCK_SIZE + i];
        }

        float total_n = fmaxf(sum_n, 0.0f);
        float total_sse = y_sq_sum - (y_sum * y_sum) / fmaxf(total_n, 1.0f);
        float left_sse = 0.0f;
        if (sum_sw > 1e-9f) {
            left_sse = sum_swyy - (sum_swy * sum_swy) / sum_sw;
        }
        float right_w = total_n - sum_sw;
        float right_swy = y_sum - sum_swy;
        float right_swyy = y_sq_sum - sum_swyy;
        float right_sse = 0.0f;
        if (right_w > 1e-9f) {
            right_sse = right_swyy - (right_swy * right_swy) / right_w;
        }
        float variance_gain = 0.0f;
        if (total_sse > 1e-12f && sum_sw > 1e-9f && right_w > 1e-9f) {
            variance_gain = fmaxf((total_sse - left_sse - right_sse) / total_sse, 0.0f);
        }

        float cov = sum_sxr - (sum_sx * sum_sr) / fmaxf(total_n, 1.0f);
        float var_x = sum_sxx - (sum_sx * sum_sx) / fmaxf(total_n, 1.0f);
        float var_r = sum_srr - (sum_sr * sum_sr) / fmaxf(total_n, 1.0f);
        float residual_corr = 0.0f;
        float denom = var_x * var_r;
        if (denom > 1e-20f) {
            residual_corr = fabsf(cov / sqrtf(denom));
        }
        float residual_r2 = residual_corr * residual_corr;

        float mutual_info = 0.0f;
        if (total_n > 0.0f) {
            for (int bx = 0; bx < bins; bx++) {
                float px = static_cast<float>(hist_x[bx]) / total_n;
                if (px <= 0.0f) continue;
                for (int by = 0; by < bins; by++) {
                    unsigned int count = hist_xy[bx * 16 + by];
                    if (count == 0u) continue;
                    float pxy = static_cast<float>(count) / total_n;
                    float py = static_cast<float>(hist_y[by]) / total_n;
                    if (py > 0.0f) {
                        mutual_info += pxy * logf(pxy / (px * py));
                    }
                }
            }
        }

        float* out = scores_batch + candidate_id * GAFIME_SELECTION_SCORE_SIZE;
        out[GAFIME_SELECTION_MUTUAL_INFO] = mutual_info;
        out[GAFIME_SELECTION_VARIANCE_REDUCTION] = variance_gain;
        out[GAFIME_SELECTION_RESIDUAL_ABS_CORR] = residual_corr;
        out[GAFIME_SELECTION_RESIDUAL_R2_GAIN] = residual_r2;
    }
}

GAFIME_API int gafime_discrete_selection_batch_cuda(
    const float* h_X,
    const float* h_y,
    const float* h_residual,
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
    int mi_bins,
    float y_min,
    float y_max,
    float y_sum,
    float y_sq_sum,
    float* h_scores_batch
) {
    if (!h_X || !h_y || !h_residual || !h_kinds || !h_feature_a || !h_feature_b ||
        !h_value_feature || !h_directions || !h_params || !h_scales ||
        !h_sharpness || !h_scores_batch) {
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

    float* d_X = nullptr;
    float* d_y = nullptr;
    float* d_residual = nullptr;
    int* d_kinds = nullptr;
    int* d_feature_a = nullptr;
    int* d_feature_b = nullptr;
    int* d_value_feature = nullptr;
    int* d_directions = nullptr;
    float* d_params = nullptr;
    float* d_scales = nullptr;
    float* d_sharpness = nullptr;
    float* d_scores = nullptr;

    size_t X_bytes = static_cast<size_t>(n_samples) * n_features * sizeof(float);
    size_t vec_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t int_bytes = static_cast<size_t>(n_candidates) * sizeof(int);
    size_t params_bytes = static_cast<size_t>(n_candidates) * 4 * sizeof(float);
    size_t scales_bytes = static_cast<size_t>(n_candidates) * 2 * sizeof(float);
    size_t sharpness_bytes = static_cast<size_t>(n_candidates) * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(n_candidates) * GAFIME_SELECTION_SCORE_SIZE * sizeof(float);

    cudaError_t err;
    int status = GAFIME_SUCCESS;

    err = cudaMalloc(&d_X, X_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_y, vec_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_residual, vec_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_kinds, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_a, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_b, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_value_feature, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_directions, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_params, params_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_scales, scales_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_sharpness, sharpness_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_scores, scores_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }

    err = cudaMemcpy(d_X, h_X, X_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_y, h_y, vec_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_residual, h_residual, vec_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_kinds, h_kinds, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_a, h_feature_a, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_b, h_feature_b, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_value_feature, h_value_feature, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_directions, h_directions, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_scales, h_scales, scales_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_sharpness, h_sharpness, sharpness_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemset(d_scores, 0, scores_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

    auto_tune_for_gpu();
    gafime_discrete_selection_batch_kernel<<<n_candidates, BLOCK_SIZE>>>(
        d_X, d_y, d_residual, d_kinds, d_feature_a, d_feature_b,
        d_value_feature, d_directions, d_params, d_scales, d_sharpness,
        n_samples, n_candidates, mi_bins, y_min, y_max, y_sum, y_sq_sum,
        d_scores
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(h_scores_batch, d_scores, scores_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

cleanup:
    if (d_scores) cudaFree(d_scores);
    if (d_sharpness) cudaFree(d_sharpness);
    if (d_scales) cudaFree(d_scales);
    if (d_params) cudaFree(d_params);
    if (d_directions) cudaFree(d_directions);
    if (d_value_feature) cudaFree(d_value_feature);
    if (d_feature_b) cudaFree(d_feature_b);
    if (d_feature_a) cudaFree(d_feature_a);
    if (d_kinds) cudaFree(d_kinds);
    if (d_residual) cudaFree(d_residual);
    if (d_y) cudaFree(d_y);
    if (d_X) cudaFree(d_X);
    return status;
}

} // extern "C"

template <int TARGET_BIN_CAPACITY>
__global__ void gafime_discrete_selection_adaptive_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    const float* __restrict__ residual,
    const int* __restrict__ y_bins,
    const int* __restrict__ kinds,
    const int* __restrict__ feature_a,
    const int* __restrict__ feature_b,
    const int* __restrict__ value_feature,
    const int* __restrict__ directions,
    const float* __restrict__ params,
    const float* __restrict__ scales,
    const float* __restrict__ sharpness,
    int n_samples,
    int n_candidates,
    float y_sum,
    float y_sq_sum,
    float* __restrict__ scores_batch
) {
    int candidate_id = blockIdx.x;
    if (candidate_id >= n_candidates) return;

    int kind = kinds[candidate_id];
    int fa = feature_a[candidate_id];
    int fb = feature_b[candidate_id];
    int vf = value_feature[candidate_id];
    int direction = directions[candidate_id];
    const float* candidate_params = params + candidate_id * 4;
    const float* candidate_scales = scales + candidate_id * 2;
    float k = sharpness[candidate_id];

    __shared__ float hist_in[TARGET_BIN_CAPACITY];
    __shared__ float hist_out[TARGET_BIN_CAPACITY];
    __shared__ float partials[10 * BLOCK_SIZE];

    for (int idx = threadIdx.x; idx < TARGET_BIN_CAPACITY; idx += blockDim.x) {
        hist_in[idx] = 0.0f;
        hist_out[idx] = 0.0f;
    }
    __syncthreads();

    float sw = 0.0f, sw2 = 0.0f, swy = 0.0f, swyy = 0.0f;
    float n = 0.0f, sx = 0.0f, sr = 0.0f;
    float sxx = 0.0f, srr = 0.0f, sxr = 0.0f;

    for (int i = threadIdx.x; i < n_samples; i += blockDim.x) {
        float feature_value = discrete_eval_soft(
            X, n_samples, i, kind, fa, fb, vf, direction,
            candidate_params, candidate_scales, k
        );
        float mask = discrete_eval_mask_soft(
            X, n_samples, i, kind, fa, fb, direction,
            candidate_params, candidate_scales, k
        );
        float target = y[i];
        float res = residual[i];
        if (isnan(feature_value) || isnan(mask) || isnan(target) || isnan(res)) {
            continue;
        }

        mask = fminf(fmaxf(mask, 0.0f), 1.0f);
        float out_w = 1.0f - mask;
        sw += mask;
        sw2 += mask * mask;
        swy += mask * target;
        swyy += mask * target * target;

        n += 1.0f;
        sx += feature_value;
        sr += res;
        sxx += feature_value * feature_value;
        srr += res * res;
        sxr += feature_value * res;

        int yb = y_bins[i];
        if (yb >= 0 && yb < TARGET_BIN_CAPACITY) {
            atomicAdd(&hist_in[yb], mask);
            atomicAdd(&hist_out[yb], out_w);
        }
    }

    int t = threadIdx.x;
    partials[t] = sw;
    partials[BLOCK_SIZE + t] = sw2;
    partials[2 * BLOCK_SIZE + t] = swy;
    partials[3 * BLOCK_SIZE + t] = swyy;
    partials[4 * BLOCK_SIZE + t] = n;
    partials[5 * BLOCK_SIZE + t] = sx;
    partials[6 * BLOCK_SIZE + t] = sr;
    partials[7 * BLOCK_SIZE + t] = sxx;
    partials[8 * BLOCK_SIZE + t] = srr;
    partials[9 * BLOCK_SIZE + t] = sxr;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_sw = 0.0f, sum_sw2 = 0.0f, sum_swy = 0.0f, sum_swyy = 0.0f;
        float sum_n = 0.0f, sum_sx = 0.0f, sum_sr = 0.0f;
        float sum_sxx = 0.0f, sum_srr = 0.0f, sum_sxr = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum_sw += partials[i];
            sum_sw2 += partials[BLOCK_SIZE + i];
            sum_swy += partials[2 * BLOCK_SIZE + i];
            sum_swyy += partials[3 * BLOCK_SIZE + i];
            sum_n += partials[4 * BLOCK_SIZE + i];
            sum_sx += partials[5 * BLOCK_SIZE + i];
            sum_sr += partials[6 * BLOCK_SIZE + i];
            sum_sxx += partials[7 * BLOCK_SIZE + i];
            sum_srr += partials[8 * BLOCK_SIZE + i];
            sum_sxr += partials[9 * BLOCK_SIZE + i];
        }

        float total_n = fmaxf(sum_n, 0.0f);
        float right_w = total_n - sum_sw;
        float right_sw2 = total_n - 2.0f * sum_sw + sum_sw2;
        float effective_in = (sum_sw2 > 1e-12f) ? (sum_sw * sum_sw / sum_sw2) : 0.0f;
        float effective_out = (right_sw2 > 1e-12f) ? (right_w * right_w / right_sw2) : 0.0f;
        float min_support = fminf(8.0f, fmaxf(3.0f, 0.02f * total_n));
        bool support_ok = effective_in >= min_support && effective_out >= min_support;

        float total_sse = y_sq_sum - (y_sum * y_sum) / fmaxf(total_n, 1.0f);
        float left_sse = 0.0f;
        if (sum_sw > 1e-9f) {
            left_sse = fmaxf(sum_swyy - (sum_swy * sum_swy) / sum_sw, 0.0f);
        }
        float right_swy = y_sum - sum_swy;
        float right_swyy = y_sq_sum - sum_swyy;
        float right_sse = 0.0f;
        if (right_w > 1e-9f) {
            right_sse = fmaxf(right_swyy - (right_swy * right_swy) / right_w, 0.0f);
        }
        float variance_gain = 0.0f;
        if (support_ok && total_sse > 1e-12f) {
            variance_gain = fmaxf((total_sse - left_sse - right_sse) / total_sse, 0.0f);
        }

        float cov = sum_sxr - (sum_sx * sum_sr) / fmaxf(total_n, 1.0f);
        float var_x = sum_sxx - (sum_sx * sum_sx) / fmaxf(total_n, 1.0f);
        float var_r = sum_srr - (sum_sr * sum_sr) / fmaxf(total_n, 1.0f);
        float residual_corr = 0.0f;
        float denom = var_x * var_r;
        if (denom > 1e-20f) {
            residual_corr = fminf(fabsf(cov / sqrtf(denom)), 1.0f);
        }
        float residual_r2 = residual_corr * residual_corr;

        float mutual_info = 0.0f;
        if (support_ok && total_n > 0.0f) {
            int nonzero_y = 0;
            #pragma unroll 1
            for (int by = 0; by < TARGET_BIN_CAPACITY; by++) {
                float py_count = hist_in[by] + hist_out[by];
                if (py_count > 0.0f) {
                    nonzero_y++;
                }
            }
            if (nonzero_y >= 2) {
                float px_in = sum_sw / total_n;
                float px_out = right_w / total_n;
                #pragma unroll 1
                for (int by = 0; by < TARGET_BIN_CAPACITY; by++) {
                    float y_count = hist_in[by] + hist_out[by];
                    if (y_count <= 0.0f) continue;
                    float py = y_count / total_n;
                    float count_in = hist_in[by];
                    if (count_in > 0.0f && px_in > 0.0f) {
                        float pxy = count_in / total_n;
                        mutual_info += pxy * logf(pxy / (px_in * py));
                    }
                    float count_out = hist_out[by];
                    if (count_out > 0.0f && px_out > 0.0f) {
                        float pxy = count_out / total_n;
                        mutual_info += pxy * logf(pxy / (px_out * py));
                    }
                }
                float bias = static_cast<float>(nonzero_y - 1) / (2.0f * total_n);
                mutual_info = fmaxf(mutual_info - bias, 0.0f);
            }
        }

        float* out = scores_batch + candidate_id * GAFIME_SELECTION_SCORE_SIZE;
        out[GAFIME_SELECTION_MUTUAL_INFO] = mutual_info;
        out[GAFIME_SELECTION_VARIANCE_REDUCTION] = variance_gain;
        out[GAFIME_SELECTION_RESIDUAL_ABS_CORR] = residual_corr;
        out[GAFIME_SELECTION_RESIDUAL_R2_GAIN] = residual_r2;
    }
}

template <int TARGET_BIN_CAPACITY>
static void launch_discrete_selection_adaptive_kernel(
    const float* d_X,
    const float* d_y,
    const float* d_residual,
    const int* d_y_bins,
    const int* d_kinds,
    const int* d_feature_a,
    const int* d_feature_b,
    const int* d_value_feature,
    const int* d_directions,
    const float* d_params,
    const float* d_scales,
    const float* d_sharpness,
    int n_samples,
    int n_candidates,
    float y_sum,
    float y_sq_sum,
    float* d_scores
) {
    gafime_discrete_selection_adaptive_kernel<TARGET_BIN_CAPACITY><<<n_candidates, BLOCK_SIZE>>>(
        d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a, d_feature_b,
        d_value_feature, d_directions, d_params, d_scales, d_sharpness,
        n_samples, n_candidates, y_sum, y_sq_sum, d_scores
    );
}

extern "C" GAFIME_API int gafime_discrete_selection_adaptive_cuda(
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
    float* h_scores_batch
) {
    if (!h_X || !h_y || !h_residual || !h_y_bins || !h_kinds || !h_feature_a ||
        !h_feature_b || !h_value_feature || !h_directions || !h_params ||
        !h_scales || !h_sharpness || !h_scores_batch) {
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

    float* d_X = nullptr;
    float* d_y = nullptr;
    float* d_residual = nullptr;
    int* d_y_bins = nullptr;
    int* d_kinds = nullptr;
    int* d_feature_a = nullptr;
    int* d_feature_b = nullptr;
    int* d_value_feature = nullptr;
    int* d_directions = nullptr;
    float* d_params = nullptr;
    float* d_scales = nullptr;
    float* d_sharpness = nullptr;
    float* d_scores = nullptr;

    size_t X_bytes = static_cast<size_t>(n_samples) * n_features * sizeof(float);
    size_t vec_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t sample_int_bytes = static_cast<size_t>(n_samples) * sizeof(int);
    size_t int_bytes = static_cast<size_t>(n_candidates) * sizeof(int);
    size_t params_bytes = static_cast<size_t>(n_candidates) * 4 * sizeof(float);
    size_t scales_bytes = static_cast<size_t>(n_candidates) * 2 * sizeof(float);
    size_t sharpness_bytes = static_cast<size_t>(n_candidates) * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(n_candidates) * GAFIME_SELECTION_SCORE_SIZE * sizeof(float);

    cudaError_t err;
    int status = GAFIME_SUCCESS;

    err = cudaMalloc(&d_X, X_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_y, vec_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_residual, vec_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_y_bins, sample_int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_kinds, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_a, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_feature_b, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_value_feature, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_directions, int_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_params, params_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_scales, scales_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_sharpness, sharpness_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }
    err = cudaMalloc(&d_scores, scores_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_OUT_OF_MEMORY; goto cleanup; }

    err = cudaMemcpy(d_X, h_X, X_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_y, h_y, vec_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_residual, h_residual, vec_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_y_bins, h_y_bins, sample_int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_kinds, h_kinds, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_a, h_feature_a, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_feature_b, h_feature_b, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_value_feature, h_value_feature, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_directions, h_directions, int_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_params, h_params, params_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_scales, h_scales, scales_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(d_sharpness, h_sharpness, sharpness_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemset(d_scores, 0, scores_bytes); if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

    auto_tune_for_gpu();
    switch (target_bin_template) {
        case 2:
            launch_discrete_selection_adaptive_kernel<2>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 4:
            launch_discrete_selection_adaptive_kernel<4>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 8:
            launch_discrete_selection_adaptive_kernel<8>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 16:
            launch_discrete_selection_adaptive_kernel<16>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 32:
            launch_discrete_selection_adaptive_kernel<32>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 64:
            launch_discrete_selection_adaptive_kernel<64>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        case 96:
            launch_discrete_selection_adaptive_kernel<96>(
                d_X, d_y, d_residual, d_y_bins, d_kinds, d_feature_a,
                d_feature_b, d_value_feature, d_directions, d_params,
                d_scales, d_sharpness, n_samples, n_candidates, y_sum,
                y_sq_sum, d_scores
            );
            break;
        default:
            status = GAFIME_ERROR_INVALID_ARGS;
            goto cleanup;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }
    err = cudaMemcpy(h_scores_batch, d_scores, scores_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = GAFIME_ERROR_KERNEL_FAILED; goto cleanup; }

cleanup:
    if (d_scores) cudaFree(d_scores);
    if (d_sharpness) cudaFree(d_sharpness);
    if (d_scales) cudaFree(d_scales);
    if (d_params) cudaFree(d_params);
    if (d_directions) cudaFree(d_directions);
    if (d_value_feature) cudaFree(d_value_feature);
    if (d_feature_b) cudaFree(d_feature_b);
    if (d_feature_a) cudaFree(d_feature_a);
    if (d_kinds) cudaFree(d_kinds);
    if (d_y_bins) cudaFree(d_y_bins);
    if (d_residual) cudaFree(d_residual);
    if (d_y) cudaFree(d_y);
    if (d_X) cudaFree(d_X);
    return status;
}

extern "C" {

// ============================================================================
// TENSOR CORE FUTUREPROOFING
// ============================================================================
// Tensor cores can accelerate Pearson correlation (which decomposes into dot
// products / GEMM). This stub provides the interface for future tensor core
// acceleration paths.
//
// Architecture support:
//   Turing  (sm_75):  2nd gen TC — FP16 WMMA
//   Ampere  (sm_80+): 3rd gen TC — TF32, BF16, FP16 WMMA
//   Ada     (sm_89):  4th gen TC — FP8, TF32, BF16, FP16
//   Hopper  (sm_90):  4th gen TC — FP8, wgmma instructions
//   Blackwell (sm_100/120): 5th gen TC — FP4, FP8, TF32, BF16
//
// Recommended integration path:
//   1. Use cuBLAS GEMM for all-pairs correlation (auto-uses TC)
//   2. Fused WMMA kernel for transform→correlate→reduce
//   3. Architecture-adaptive: TF32 on Ampere+, FP16 on Turing

#define GAFIME_TC_AVAILABLE(compute_cap) ((compute_cap) >= 75)
#define GAFIME_TC_TF32_AVAILABLE(compute_cap) ((compute_cap) >= 80)
#define GAFIME_TC_FP8_AVAILABLE(compute_cap) ((compute_cap) >= 89)

/**
 * Check if tensor core acceleration is available on the current GPU.
 * Returns 1 if available, 0 if not. Writes the recommended precision mode.
 *
 * precision_mode: 0 = not available, 1 = FP16, 2 = TF32, 3 = FP8
 */
GAFIME_API int gafime_tensor_core_available(int* precision_mode) {
    if (!g_gpu_config.is_initialized) {
        if (precision_mode) *precision_mode = 0;
        return 0;
    }
    
    int compute_cap = g_gpu_config.compute_major * 10 + g_gpu_config.compute_minor;
    
    if (GAFIME_TC_FP8_AVAILABLE(compute_cap)) {
        if (precision_mode) *precision_mode = 3; // FP8
        return 1;
    } else if (GAFIME_TC_TF32_AVAILABLE(compute_cap)) {
        if (precision_mode) *precision_mode = 2; // TF32
        return 1;
    } else if (GAFIME_TC_AVAILABLE(compute_cap)) {
        if (precision_mode) *precision_mode = 1; // FP16
        return 1;
    }
    
    if (precision_mode) *precision_mode = 0;
    return 0;
}

} // extern "C"
