#include <cstdint>
#include <cstdio>
#include <memory>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../../src/cuda/rt_launcher.cuh"

namespace {

struct TestDeviceState {
    explicit TestDeviceState(uint32_t value) : device_id(value) {}

    uint32_t device_id;
    bool released = false;
};

constexpr uint32_t kMaxGridY = 65'535u;
constexpr uint64_t kRowsAtGridBoundary =
    static_cast<uint64_t>(gafime_cuda_v1::detail::kDecisionPathThreads) * kMaxGridY;

#if defined(_WIN32)
using DynamicLibrary = HMODULE;

DynamicLibrary open_library(const char* path) {
    return LoadLibraryA(path);
}

void* load_symbol(DynamicLibrary library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(library, name));
}

bool close_library(DynamicLibrary library) {
    return FreeLibrary(library) != 0;
}
#else
using DynamicLibrary = void*;

DynamicLibrary open_library(const char* path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void* load_symbol(DynamicLibrary library, const char* name) {
    return dlsym(library, name);
}

bool close_library(DynamicLibrary library) {
    return dlclose(library) == 0;
}
#endif

static_assert(
    gafime_cuda_v1::detail::decision_path_group_count_fits_grid(kMaxGridY, kMaxGridY)
);

bool require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}
static_assert(
    !gafime_cuda_v1::detail::decision_path_group_count_fits_grid(
        static_cast<size_t>(kMaxGridY) + 1u,
        kMaxGridY
    )
);
static_assert(
    gafime_cuda_v1::detail::decision_path_row_tile_count(kRowsAtGridBoundary, kMaxGridY) == 1u
);
static_assert(
    gafime_cuda_v1::detail::decision_path_row_tile_count(kRowsAtGridBoundary + 1u, kMaxGridY) == 2u
);
static_assert(
    gafime_cuda_v1::detail::decision_path_row_tile(
        kRowsAtGridBoundary + 1u,
        kMaxGridY,
        0u
    ).block_count == kMaxGridY
);
static_assert(
    gafime_cuda_v1::detail::decision_path_row_tile(
        kRowsAtGridBoundary + 1u,
        kMaxGridY,
        1u
    ).row_offset == kRowsAtGridBoundary
);
static_assert(
    gafime_cuda_v1::detail::decision_path_row_tile(
        kRowsAtGridBoundary + 1u,
        kMaxGridY,
        1u
    ).block_count == 1u
);

}  // namespace

int main(int argc, char** argv) {
#if defined(GAFIME_RT_TEST_QUERY_DEVICE)
    int device_count = 0;
    if (!require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed") ||
        !require(device_count > 0, "no CUDA device is available")) {
        return 1;
    }
    int max_grid_y = 0;
    if (!require(
            cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, 0) == cudaSuccess,
            "cudaDeviceGetAttribute(cudaDevAttrMaxGridDimY) failed"
        )) {
        return 1;
    }
    std::printf("cuda_devices=%d device0_max_grid_y=%d\n", device_count, max_grid_y);
#endif

    gafime_cuda_v1::detail::DeviceStateMap<TestDeviceState> states;
    const auto factory = [](uint32_t device_id) {
        return std::make_shared<TestDeviceState>(device_id);
    };

    const std::shared_ptr<TestDeviceState> device0_first = states.get_or_create(0u, factory);
    const std::shared_ptr<TestDeviceState> device0_second = states.get_or_create(0u, factory);
    const std::shared_ptr<TestDeviceState> device1 = states.get_or_create(1u, factory);
    if (!require(device0_first == device0_second, "device 0 state was not reused") ||
        !require(device0_first != device1, "different devices shared one state") ||
        !require(device0_first->device_id == 0u, "device 0 state has the wrong id") ||
        !require(device1->device_id == 1u, "device 1 state has the wrong id") ||
        !require(states.size() == 2u, "state registry has the wrong initial size")) {
        return 1;
    }

    const int release_status = states.release(0u, [](TestDeviceState& state) {
        state.released = true;
        return GAFIME_STATUS_OK;
    });
    if (!require(release_status == GAFIME_STATUS_OK, "device 0 state release failed") ||
        !require(device0_first->released, "device 0 release callback did not run") ||
        !require(!device1->released, "device 1 was released with device 0") ||
        !require(states.size() == 1u, "state registry did not erase device 0")) {
        return 1;
    }

    const std::shared_ptr<TestDeviceState> device0_recreated = states.get_or_create(0u, factory);
    if (!require(device0_recreated != device0_first, "released device state was reused") ||
        !require(device0_recreated != device1, "recreated state aliases another device") ||
        !require(states.size() == 2u, "state registry did not recreate device 0")) {
        return 1;
    }

    if (argc == 2) {
        DynamicLibrary payload = open_library(argv[1]);
        if (!require(payload != nullptr, "failed to open CUDA RT payload")) {
            return 1;
        }
        void* symbol = load_symbol(payload, "gafime_gpu_decision_path_release_device_state");
        if (!require(symbol != nullptr, "CUDA RT payload is missing the release symbol")) {
            static_cast<void>(close_library(payload));
            return 1;
        }
        using ReleaseDeviceState = decltype(&gafime_gpu_decision_path_release_device_state);
        const auto release_device_state = reinterpret_cast<ReleaseDeviceState>(symbol);
        const int payload_release_status = release_device_state(0u);
        const bool closed = close_library(payload);
        if (!require(payload_release_status == GAFIME_STATUS_OK, "payload RT release failed") ||
            !require(closed, "failed to close CUDA RT payload")) {
            return 1;
        }
    }
    return 0;
}
