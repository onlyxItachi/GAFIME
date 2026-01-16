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

// Block size tuned for RTX 4060 (Ada Lovelace SM89)
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// UNARY OPERATORS
// ============================================================================

/**
 * Apply unary transformation to a single value.
 * Safe implementations prevent NaN/Inf propagation.
 */
__device__ __forceinline__ float apply_op(float x, int op) {
    switch (op) {
        case GAFIME_OP_LOG:
            // log(|x| + eps) to handle negatives and zeros
            return logf(fabsf(x) + 1e-8f);
        
        case GAFIME_OP_EXP:
            // Clamp to prevent overflow (exp(20) ≈ 4.8e8)
            return expf(fminf(fmaxf(x, -20.0f), 20.0f));
        
        case GAFIME_OP_SQRT:
            // sqrt(|x|) to handle negatives
            return sqrtf(fabsf(x));
        
        case GAFIME_OP_TANH:
            return tanhf(x);
        
        case GAFIME_OP_SIGMOID:
            // 1 / (1 + exp(-x)) with stability
            return 1.0f / (1.0f + expf(-fminf(fmaxf(x, -20.0f), 20.0f)));
        
        case GAFIME_OP_SQUARE:
            return x * x;
        
        case GAFIME_OP_NEGATE:
            return -x;
        
        case GAFIME_OP_ABS:
            return fabsf(x);
        
        case GAFIME_OP_INVERSE:
            // 1/x with safety for small values
            return 1.0f / (fabsf(x) < 1e-8f ? copysignf(1e-8f, x) : x);
        
        case GAFIME_OP_CUBE:
            return x * x * x;
        
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
