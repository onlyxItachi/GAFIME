#include "rt_launcher.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "rt_kernels.cuh"

namespace {

int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

bool decision_path_sign_supported(uint32_t sign) {
    return sign == GAFIME_DECISION_PATH_SIGN_LE || sign == GAFIME_DECISION_PATH_SIGN_GT;
}

int validate_decision_path_batch(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* batch
) {
    if (resident_features == nullptr || batch == nullptr || batch->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->path_count == UINT32_MAX || batch->term_count == 0 ||
        batch->terms == nullptr || batch->path_offsets == nullptr || batch->membership_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_offsets[0] != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t path_idx = 0; path_idx < batch->path_count; ++path_idx) {
        const uint32_t begin = batch->path_offsets[path_idx];
        const uint32_t end = batch->path_offsets[path_idx + 1];
        if (begin >= end || end > batch->term_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = batch->terms[term_idx];
            if (term.feature >= cols || !decision_path_sign_supported(term.sign)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (batch->path_offsets[batch->path_count] != batch->term_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != 0 && batch->path_count > UINT64_MAX / rows) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t output_count = rows * static_cast<uint64_t>(batch->path_count);
    if (output_count > static_cast<uint64_t>(SIZE_MAX / sizeof(float))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

}  // namespace

namespace gafime_cuda_v1 {

void tune_rt_kernels_for_device(const cudaDeviceProp& props) {
    const cudaFuncCache cache_mode = props.major >= 7 ? cudaFuncCachePreferShared : cudaFuncCachePreferL1;
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::rt_kernel::decision_path_membership_kernel,
        cache_mode
    ));
}

cudaError_t launch_decision_path_membership(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership,
    cudaStream_t stream
) {
    if (path_count == 0 || n_samples == 0) {
        return cudaSuccess;
    }
    constexpr uint32_t threads = 256;
    const uint32_t row_blocks = static_cast<uint32_t>((n_samples + threads - 1) / threads);
    dim3 grid(path_count, row_blocks);
    dim3 block(threads);
    rt_kernel::decision_path_membership_kernel<<<grid, block, 0, stream>>>(
        features,
        n_samples,
        n_features,
        terms,
        path_offsets,
        path_count,
        membership
    );
    return cudaGetLastError();
}

int execute_decision_path_membership(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* paths
) {
    int status = validate_decision_path_batch(resident_features, rows, cols, paths);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t output_count = rows * static_cast<uint64_t>(paths->path_count);
    const size_t term_bytes = static_cast<size_t>(paths->term_count) * sizeof(GafimeDecisionPathTerm);
    const size_t offset_bytes = static_cast<size_t>(paths->path_count + 1u) * sizeof(uint32_t);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);

    GafimeDecisionPathTerm* terms_device = nullptr;
    uint32_t* offsets_device = nullptr;
    float* membership_device = nullptr;

    status = cuda_status(cudaMalloc(&terms_device, term_bytes));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&offsets_device, offset_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&membership_device, output_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(terms_device, paths->terms, term_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(offsets_device, paths->path_offsets, offset_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(gafime_cuda_v1::launch_decision_path_membership(
            resident_features,
            rows,
            cols,
            terms_device,
            offsets_device,
            paths->path_count,
            membership_device,
            0
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(paths->membership_host, membership_device, output_bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(membership_device);
    cudaFree(offsets_device);
    cudaFree(terms_device);
    return status;
}

}  // namespace gafime_cuda_v1
