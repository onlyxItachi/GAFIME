#include <cstdint>
#include <cstdio>

#include "../../src/cuda/kernels.cuh"

namespace {

using gafime_cuda_v1::CudaArchitecturePolicyClass;
using gafime_cuda_v1::CudaKernelLaunchPolicy;

bool expect_policy(
    uint32_t compute_major,
    uint32_t max_threads,
    CudaArchitecturePolicyClass expected_class,
    uint32_t expected_threads,
    const char* label
) {
    const CudaKernelLaunchPolicy policy =
        gafime_cuda_v1::cuda_kernel_launch_policy_for_device(compute_major, max_threads);
    if (policy.architecture_class != expected_class ||
        policy.threads_per_block != expected_threads) {
        std::fprintf(stderr, "%s selected an unexpected CUDA launch policy\n", label);
        return false;
    }
    return gafime_cuda_v1::cuda_kernel_launch_policy_supported(policy);
}

}  // namespace

int main() {
    bool passed = true;
    passed = expect_policy(
        7,
        1024,
        CudaArchitecturePolicyClass::kPreAmpere,
        128,
        "pre_ampere"
    ) && passed;
    passed = expect_policy(
        8,
        1024,
        CudaArchitecturePolicyClass::kAmpereAda,
        256,
        "ampere_ada"
    ) && passed;
    passed = expect_policy(
        9,
        1024,
        CudaArchitecturePolicyClass::kHopper,
        256,
        "hopper"
    ) && passed;
    passed = expect_policy(
        10,
        1024,
        CudaArchitecturePolicyClass::kBlackwell,
        256,
        "blackwell"
    ) && passed;

    const CudaKernelLaunchPolicy unsupported =
        gafime_cuda_v1::cuda_kernel_launch_policy_for_device(8, 128);
    if (gafime_cuda_v1::cuda_kernel_launch_policy_supported(unsupported)) {
        std::fprintf(stderr, "insufficient device block capacity was accepted\n");
        passed = false;
    }
    const CudaKernelLaunchPolicy unknown =
        gafime_cuda_v1::cuda_kernel_launch_policy_for_device(0, 1024);
    if (gafime_cuda_v1::cuda_kernel_launch_policy_supported(unknown)) {
        std::fprintf(stderr, "unknown CUDA capability was accepted\n");
        passed = false;
    }
    return passed ? 0 : 1;
}
