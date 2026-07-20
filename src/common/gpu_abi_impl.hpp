#ifndef GAFIME_GPU_ABI_IMPL_HPP
#define GAFIME_GPU_ABI_IMPL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>

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

inline uint64_t saturating_add_u64(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs + rhs;
}

inline uint64_t saturating_mul_u64(uint64_t lhs, uint64_t rhs) {
    return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs * rhs;
}

inline uint64_t allocation_bytes(uint64_t capacity, size_t element_size) {
    return saturating_mul_u64(capacity, static_cast<uint64_t>(element_size));
}

inline uint64_t next_allocation_capacity(
    uint64_t capacity,
    uint64_t required,
    size_t element_size
) {
    if (required <= capacity) {
        return capacity;
    }
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / element_size;
    if (required > max_capacity) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t grown_capacity = capacity > max_capacity / 2
        ? max_capacity
        : capacity * 2;
    return std::max(required, capacity == 0 ? required : grown_capacity);
}

class DeviceMemoryPeakTracker {
public:
    explicit DeviceMemoryPeakTracker(uint64_t resident_bytes)
        : resident_bytes_(resident_bytes), peak_bytes_(resident_bytes) {}

    void grow(uint64_t capacity, uint64_t required, size_t element_size) {
        if (required <= capacity) {
            return;
        }
        const uint64_t next_capacity =
            next_allocation_capacity(capacity, required, element_size);
        const uint64_t old_bytes = allocation_bytes(capacity, element_size);
        const uint64_t next_bytes = allocation_bytes(next_capacity, element_size);
        observe_transient(next_bytes);
        replace_resident(old_bytes, next_bytes);
    }

    void reserve_pair(
        uint64_t first_capacity,
        uint64_t first_required,
        size_t first_element_size,
        uint64_t second_capacity,
        uint64_t second_required,
        size_t second_element_size
    ) {
        const uint64_t first_next = next_allocation_capacity(
            first_capacity, first_required, first_element_size);
        const uint64_t second_next = next_allocation_capacity(
            second_capacity, second_required, second_element_size);
        uint64_t transient_bytes = 0;
        if (first_required > first_capacity) {
            transient_bytes = saturating_add_u64(
                transient_bytes,
                allocation_bytes(first_next, first_element_size));
        }
        if (second_required > second_capacity) {
            transient_bytes = saturating_add_u64(
                transient_bytes,
                allocation_bytes(second_next, second_element_size));
        }
        observe_transient(transient_bytes);
        if (first_required > first_capacity) {
            replace_resident(
                allocation_bytes(first_capacity, first_element_size),
                allocation_bytes(first_next, first_element_size));
        }
        if (second_required > second_capacity) {
            replace_resident(
                allocation_bytes(second_capacity, second_element_size),
                allocation_bytes(second_next, second_element_size));
        }
    }

    void observe_transient(uint64_t transient_bytes) {
        peak_bytes_ = std::max(
            peak_bytes_, saturating_add_u64(resident_bytes_, transient_bytes));
    }

    uint64_t resident_bytes() const {
        return resident_bytes_;
    }

    uint64_t peak_bytes() const {
        return peak_bytes_;
    }

private:
    void replace_resident(uint64_t old_bytes, uint64_t next_bytes) {
        if (resident_bytes_ == std::numeric_limits<uint64_t>::max() ||
            old_bytes > resident_bytes_) {
            resident_bytes_ = std::numeric_limits<uint64_t>::max();
        } else {
            resident_bytes_ = saturating_add_u64(
                resident_bytes_ - old_bytes, next_bytes);
        }
        peak_bytes_ = std::max(peak_bytes_, resident_bytes_);
    }

    uint64_t resident_bytes_;
    uint64_t peak_bytes_;
};

}  // namespace gafime_gpu_abi

#endif  // GAFIME_GPU_ABI_IMPL_HPP
