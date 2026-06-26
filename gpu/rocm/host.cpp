#include "../common/gpu_abi_impl.h"

extern "C" {

GAFIME_GPU_API int gafime_gpu_device_info(
    uint32_t device_id,
    GafimeGpuDeviceInfo* info_out
) {
    return gafime_gpu_abi::fill_device_info(
        device_id,
        GAFIME_BACKEND_ROCM,
        "rocm-v1-skeleton",
        info_out
    );
}

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
) {
    (void)device_id;
    return gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_ROCM,
        GAFIME_GRAPH_UNSUPPORTED,
        capability_out
    );
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) {
    (void)device_id;
    (void)matrix_desc;
    if (matrix_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *matrix_out = nullptr;
    return gafime_gpu_abi::unsupported_until_p3_device_loop();
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
    (void)matrix;
    (void)features_host;
    (void)target_host;
    (void)rows;
    (void)cols;
    return gafime_gpu_abi::unsupported_until_p3_device_loop();
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix,
    const float* target_host,
    uint64_t rows
) {
    (void)matrix;
    (void)target_host;
    (void)rows;
    return gafime_gpu_abi::unsupported_until_p3_device_loop();
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix) {
    (void)matrix;
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    (void)matrix;
    (void)protocol;
    (void)result_out;
    return gafime_gpu_abi::unsupported_until_p3_device_loop();
}

}  // extern "C"
