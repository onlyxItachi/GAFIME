/**
 * GAFIME CUDA Kernels - Operator-Fused Map-Reduce Architecture
 * 
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM89)
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

// Block size tuned for RTX 4060 (Ada Lovelace SM89)
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// UNARY OPERATORS (Standard math library)
// ============================================================================

/**
 * Apply unary transformation to a single value.
 * Safe implementations prevent NaN/Inf propagation.
 */
__device__ __forceinline__ float apply_op(float x, int op) {
    switch (op) {
        case GAFIME_OP_LOG:
            return logf(fabsf(x) + 1e-8f);
        case GAFIME_OP_EXP:
            return expf(fminf(fmaxf(x, -20.0f), 20.0f));
        case GAFIME_OP_SQRT:
            return sqrtf(fabsf(x));
        case GAFIME_OP_TANH:
            return tanhf(x);
        case GAFIME_OP_SIGMOID:
            return 1.0f / (1.0f + expf(-fminf(fmaxf(x, -20.0f), 20.0f)));
        case GAFIME_OP_SQUARE:
            return x * x;
        case GAFIME_OP_NEGATE:
            return -x;
        case GAFIME_OP_ABS:
            return fabsf(x);
        case GAFIME_OP_INVERSE:
            return 1.0f / (fabsf(x) < 1e-8f ? copysignf(1e-8f, x) : x);
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
        
        // Time-Series operators (Memory + ALU)
        case GAFIME_OP_ROLLING_MEAN: {
            int w = (window > 0) ? window : 10;  // Default window=10
            int start = max(0, idx - w + 1);
            int count = idx - start + 1;
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = start; i <= idx; i++) {
                sum += col[i];
            }
            return sum / (float)count;
        }
        
        case GAFIME_OP_ROLLING_STD: {
            // Welford's algorithm for numerical stability
            int w = (window > 0) ? window : 10;
            int start = max(0, idx - w + 1);
            int count = 0;
            float mean = 0.0f;
            float M2 = 0.0f;
            
            #pragma unroll 4
            for (int i = start; i <= idx; i++) {
                count++;
                float delta = col[i] - mean;
                mean += delta / (float)count;
                float delta2 = col[i] - mean;
                M2 += delta * delta2;
            }
            
            if (count < 2) return 0.0f;
            return __fsqrt_rn(M2 / (float)(count - 1));
        }
        
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
    int interaction_type,
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
        
        // Combine based on arity
        float X = combine(x0, x1, interaction_type);
        
        if constexpr (Arity >= 3) {
            float x2 = apply_op(input2[i], op2);
            X = combine(X, x2, interaction_type);
        }
        if constexpr (Arity >= 4) {
            float x3 = apply_op(input3[i], op3);
            X = combine(X, x3, interaction_type);
        }
        if constexpr (Arity >= 5) {
            float x4 = apply_op(input4[i], op4);
            X = combine(X, x4, interaction_type);
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
    bucket->d_stats_B = nullptr;  // For interleaved kernel
    for (int i = 0; i < GAFIME_MAX_FEATURES; i++) {
        bucket->d_features[i] = nullptr;
    }
    
    size_t vec_bytes = static_cast<size_t>(n_samples) * sizeof(float);
    size_t mask_bytes = static_cast<size_t>(n_samples) * sizeof(uint8_t);
    cudaError_t err;
    
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
    int interaction_type,
    int val_fold_id,
    float* h_stats
) {
    if (!bucket || !feature_indices || !ops || !h_stats) {
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
    
    // Zero stats buffer (NO cudaMalloc!)
    cudaError_t err = cudaMemset(impl->d_stats, 0, 12 * sizeof(float));
    if (err != cudaSuccess) {
        return GAFIME_ERROR_KERNEL_FAILED;
    }
    
    // Calculate grid dimensions
    int num_blocks = (impl->n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 1024);
    
    // Launch kernel based on arity (NO cudaMalloc/cudaFree!)
    switch (arity) {
        case 2:
            gafime_fused_kernel<2><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, nullptr, nullptr, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], 0, 0, 0,
                interaction_type, val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 3:
            gafime_fused_kernel<3><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, nullptr, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], 0, 0,
                interaction_type, val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 4:
            gafime_fused_kernel<4><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, d_input3, nullptr,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], ops[3], 0,
                interaction_type, val_fold_id, impl->n_samples, impl->d_stats
            );
            break;
        case 5:
            gafime_fused_kernel<5><<<num_blocks, BLOCK_SIZE>>>(
                d_input0, d_input1, d_input2, d_input3, d_input4,
                impl->d_target, impl->d_mask,
                ops[0], ops[1], ops[2], ops[3], ops[4],
                interaction_type, val_fold_id, impl->n_samples, impl->d_stats
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
    if (impl->d_stats_B) cudaFree(impl->d_stats_B);  // Free second stats buffer
    
    delete impl;
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
        switch (arity) {
            case 2:
                gafime_fused_kernel<2><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], nullptr, nullptr, nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], 0, 0, 0,
                    interaction_type, val_fold_id, n_samples, d_stats
                );
                break;
            case 3:
                gafime_fused_kernel<3><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], nullptr, nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], 0, 0,
                    interaction_type, val_fold_id, n_samples, d_stats
                );
                break;
            case 4:
                gafime_fused_kernel<4><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3], nullptr,
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], h_ops[3], 0,
                    interaction_type, val_fold_id, n_samples, d_stats
                );
                break;
            case 5:
                gafime_fused_kernel<5><<<num_blocks, BLOCK_SIZE>>>(
                    d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3], d_inputs[4],
                    d_target, d_mask,
                    h_ops[0], h_ops[1], h_ops[2], h_ops[3], h_ops[4],
                    interaction_type, val_fold_id, n_samples, d_stats
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

} // extern "C"
