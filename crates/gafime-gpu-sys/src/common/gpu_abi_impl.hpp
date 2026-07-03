#ifndef GAFIME_GPU_ABI_IMPL_HPP
#define GAFIME_GPU_ABI_IMPL_HPP

#include <cstdio>
#include <cstring>

#include "gafime_gpu_abi.hpp"

namespace gafime_gpu_abi {

inline int invalid_if_null(const void* ptr) {
    return ptr == nullptr ? GAFIME_STATUS_INVALID_ARGUMENT : GAFIME_STATUS_OK;
}

inline int fill_device_info(
    uint32_t device_id,
    uint32_t backend_kind,
    const char* name,
    GafimeGpuDeviceInfo* info_out
) {
    if (info_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = backend_kind;
    info_out->device_id = device_id;
    info_out->warp_size = 32;
    std::snprintf(info_out->name, sizeof(info_out->name), "%s", name);
    return GAFIME_STATUS_OK;
}

inline int fill_graph_capability(
    uint32_t backend_kind,
    uint32_t graph_mode,
    GafimeGpuGraphCapability* capability_out
) {
    if (capability_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::memset(capability_out, 0, sizeof(*capability_out));
    capability_out->abi_version = GAFIME_ABI_VERSION;
    capability_out->backend_kind = backend_kind;
    capability_out->graph_mode = graph_mode;
    return GAFIME_STATUS_OK;
}

inline int unsupported_until_p3_device_loop() {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

}  // namespace gafime_gpu_abi

#endif  // GAFIME_GPU_ABI_IMPL_HPP
