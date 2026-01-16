/**
 * GAFIME CUDA Kernels - Feature Interaction Mining
 * 
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM89)
 * 
 * Design Principles:
 * 1. Safety First: All kernels have bounds checking (if (idx >= N) return;)
 * 2. Memory Coalescing: 128-byte aligned global memory accesses
 * 3. Register Pressure: Minimize registers per thread for SM occupancy
 * 4. Precision: FP32 for standard operations
 */

#include "../common/interfaces.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

// Block size tuned for RTX 4060 (Ada Lovelace SM89)
// 256 threads is a good balance for occupancy
#define BLOCK_SIZE 256

// Maximum combo size (features per interaction)
#define MAX_COMBO_SIZE 8

/**
 * Kernel: Compute feature means
 * Each thread handles one feature column
 */
__global__ void compute_means_kernel(
    const float* __restrict__ X,
    float* __restrict__ means,
    int32_t n_samples,
    int32_t n_features
) {
    int32_t feat = blockIdx.x * blockDim.x + threadIdx.x;
    
    // CRITICAL: Bounds check to prevent illegal memory access
    if (feat >= n_features) return;
    
    float sum = 0.0f;
    for (int32_t i = 0; i < n_samples; i++) {
        // Coalesced access: threads read adjacent features
        sum += X[i * n_features + feat];
    }
    means[feat] = sum / static_cast<float>(n_samples);
}

/**
 * Kernel: Feature Interaction Mining
 * 
 * For each (sample, combo) pair, compute:
 *   output[sample, combo] = prod_{j in combo} (X[sample, j] - mean[j])
 * 
 * Grid: (ceil(n_samples * n_combos / BLOCK_SIZE),)
 * Block: (BLOCK_SIZE,)
 */
__global__ void feature_interaction_kernel(
    const float* __restrict__ X,        // [n_samples x n_features]
    const float* __restrict__ means,    // [n_features]
    float* __restrict__ output,         // [n_samples x n_combos]
    const int32_t* __restrict__ combo_indices,  // Flat combo feature indices
    const int32_t* __restrict__ combo_offsets,  // [n_combos + 1] start offsets
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_work = n_samples * n_combos;
    
    // CRITICAL: Bounds check to prevent kernel panic
    if (idx >= total_work) return;
    
    int32_t sample = idx / n_combos;
    int32_t combo = idx % n_combos;
    
    // Get combo range from offsets
    int32_t start = combo_offsets[combo];
    int32_t end = combo_offsets[combo + 1];
    int32_t combo_size = end - start;
    
    // Safety: skip empty combos
    if (combo_size <= 0) {
        output[idx] = 0.0f;
        return;
    }
    
    // For single-feature combos, just return the raw value
    if (combo_size == 1) {
        int32_t feat = combo_indices[start];
        // Bounds check on feature index
        if (feat >= 0 && feat < n_features) {
            output[idx] = X[sample * n_features + feat];
        } else {
            output[idx] = 0.0f;
        }
        return;
    }
    
    // Multi-feature combo: compute centered product
    float prod = 1.0f;
    for (int32_t j = start; j < end && j < start + MAX_COMBO_SIZE; j++) {
        int32_t feat = combo_indices[j];
        // Bounds check on each feature index
        if (feat >= 0 && feat < n_features) {
            float centered = X[sample * n_features + feat] - means[feat];
            prod *= centered;
        }
    }
    output[idx] = prod;
}

/**
 * Kernel: Pearson Correlation (reduction-based)
 * Uses shared memory for partial sums
 */
__global__ void pearson_reduce_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,  // [5 * num_blocks]: sum_x, sum_y, sum_xx, sum_yy, sum_xy
    int32_t n
) {
    extern __shared__ float sdata[];
    
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t block_offset = blockIdx.x * 5;
    
    // Initialize shared memory for 5 accumulators
    float* s_sum_x = sdata;
    float* s_sum_y = sdata + blockDim.x;
    float* s_sum_xx = sdata + 2 * blockDim.x;
    float* s_sum_yy = sdata + 3 * blockDim.x;
    float* s_sum_xy = sdata + 4 * blockDim.x;
    
    // Load and accumulate with grid-stride loop
    float local_sum_x = 0.0f, local_sum_y = 0.0f;
    float local_sum_xx = 0.0f, local_sum_yy = 0.0f, local_sum_xy = 0.0f;
    
    for (int32_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        float xi = x[i];
        float yi = y[i];
        local_sum_x += xi;
        local_sum_y += yi;
        local_sum_xx += xi * xi;
        local_sum_yy += yi * yi;
        local_sum_xy += xi * yi;
    }
    
    s_sum_x[tid] = local_sum_x;
    s_sum_y[tid] = local_sum_y;
    s_sum_xx[tid] = local_sum_xx;
    s_sum_yy[tid] = local_sum_yy;
    s_sum_xy[tid] = local_sum_xy;
    __syncthreads();
    
    // Reduction in shared memory
    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_x[tid] += s_sum_x[tid + s];
            s_sum_y[tid] += s_sum_y[tid + s];
            s_sum_xx[tid] += s_sum_xx[tid + s];
            s_sum_yy[tid] += s_sum_yy[tid + s];
            s_sum_xy[tid] += s_sum_xy[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[block_offset + 0] = s_sum_x[0];
        partial_sums[block_offset + 1] = s_sum_y[0];
        partial_sums[block_offset + 2] = s_sum_xx[0];
        partial_sums[block_offset + 3] = s_sum_yy[0];
        partial_sums[block_offset + 4] = s_sum_xy[0];
    }
}

// ============================================================================
// Host API (extern "C" for ctypes)
// ============================================================================

extern "C" {

int gafime_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

int gafime_get_device_info(
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

int gafime_feature_interaction_cuda(
    const float* X,
    const float* means,
    float* output,
    const int32_t* combo_indices,
    const int32_t* combo_offsets,
    int32_t n_samples,
    int32_t n_features,
    int32_t n_combos
) {
    if (!X || !means || !output || !combo_indices || !combo_offsets) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    if (n_samples <= 0 || n_features <= 0 || n_combos <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // Allocate device memory
    float *d_X = nullptr, *d_means = nullptr, *d_output = nullptr;
    int32_t *d_combo_indices = nullptr, *d_combo_offsets = nullptr;
    
    size_t X_size = static_cast<size_t>(n_samples) * n_features * sizeof(float);
    size_t means_size = static_cast<size_t>(n_features) * sizeof(float);
    size_t output_size = static_cast<size_t>(n_samples) * n_combos * sizeof(float);
    
    // Get total indices from last offset
    int32_t total_indices = combo_offsets[n_combos];
    size_t indices_size = static_cast<size_t>(total_indices) * sizeof(int32_t);
    size_t offsets_size = static_cast<size_t>(n_combos + 1) * sizeof(int32_t);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_X, X_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_means, means_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_output, output_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_combo_indices, indices_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_combo_offsets, offsets_size);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy data to device
    err = cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_means, means, means_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_combo_indices, combo_indices, indices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_combo_offsets, combo_offsets, offsets_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel
    {
        int32_t total_work = n_samples * n_combos;
        int32_t num_blocks = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        feature_interaction_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_X, d_means, d_output,
            d_combo_indices, d_combo_offsets,
            n_samples, n_features, n_combos
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Copy result back
    err = cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_means);
    cudaFree(d_output);
    cudaFree(d_combo_indices);
    cudaFree(d_combo_offsets);
    
    return GAFIME_SUCCESS;

cleanup:
    if (d_X) cudaFree(d_X);
    if (d_means) cudaFree(d_means);
    if (d_output) cudaFree(d_output);
    if (d_combo_indices) cudaFree(d_combo_indices);
    if (d_combo_offsets) cudaFree(d_combo_offsets);
    return GAFIME_ERROR_KERNEL_FAILED;
}

int gafime_pearson_cuda(
    const float* x,
    const float* y,
    int32_t n,
    float* result_out
) {
    if (!x || !y || !result_out || n <= 0) {
        return GAFIME_ERROR_INVALID_ARGS;
    }
    
    // For small arrays, just compute on CPU
    if (n < 1024) {
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
    
    // GPU implementation for larger arrays
    float *d_x = nullptr, *d_y = nullptr, *d_partial = nullptr;
    
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 256);  // Cap blocks for reduction
    
    size_t vec_size = static_cast<size_t>(n) * sizeof(float);
    size_t partial_size = static_cast<size_t>(num_blocks * 5) * sizeof(float);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_x, vec_size);
    if (err != cudaSuccess) goto pearson_cleanup;
    
    err = cudaMalloc(&d_y, vec_size);
    if (err != cudaSuccess) goto pearson_cleanup;
    
    err = cudaMalloc(&d_partial, partial_size);
    if (err != cudaSuccess) goto pearson_cleanup;
    
    err = cudaMemcpy(d_x, x, vec_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pearson_cleanup;
    
    err = cudaMemcpy(d_y, y, vec_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pearson_cleanup;
    
    // Launch reduction kernel
    {
        size_t shared_mem = 5 * BLOCK_SIZE * sizeof(float);
        pearson_reduce_kernel<<<num_blocks, BLOCK_SIZE, shared_mem>>>(
            d_x, d_y, d_partial, n
        );
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pearson_cleanup;
    }
    
    // Copy partial sums back and reduce on host
    {
        float* h_partial = new float[num_blocks * 5];
        err = cudaMemcpy(h_partial, d_partial, partial_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] h_partial;
            goto pearson_cleanup;
        }
        
        float sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xy = 0;
        for (int i = 0; i < num_blocks; i++) {
            sum_x += h_partial[i * 5 + 0];
            sum_y += h_partial[i * 5 + 1];
            sum_xx += h_partial[i * 5 + 2];
            sum_yy += h_partial[i * 5 + 3];
            sum_xy += h_partial[i * 5 + 4];
        }
        delete[] h_partial;
        
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
    }
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial);
    return GAFIME_SUCCESS;

pearson_cleanup:
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_partial) cudaFree(d_partial);
    return GAFIME_ERROR_KERNEL_FAILED;
}

} // extern "C"
