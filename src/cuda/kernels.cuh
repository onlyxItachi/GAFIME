#ifndef GAFIME_CUDA_KERNELS_CUH
#define GAFIME_CUDA_KERNELS_CUH

#include <cstdint>

namespace gafime_cuda_v1 {

enum class CudaArchitecturePolicyClass : uint8_t {
    kPreAmpere,
    kAmpereAda,
    kHopper,
    kBlackwell,
};

struct CudaKernelLaunchPolicy {
    CudaArchitecturePolicyClass architecture_class;
    uint32_t threads_per_block;
};

constexpr uint32_t kCudaPreAmpereThreadsPerBlock = 128;
constexpr uint32_t kCudaModernThreadsPerBlock = 256;
constexpr uint32_t kCudaKernelMaxThreadsPerBlock = kCudaModernThreadsPerBlock;

constexpr CudaKernelLaunchPolicy cuda_kernel_launch_policy_for_device(
    uint32_t compute_major,
    uint32_t max_threads_per_block
) {
    CudaArchitecturePolicyClass architecture_class = CudaArchitecturePolicyClass::kPreAmpere;
    uint32_t threads = kCudaPreAmpereThreadsPerBlock;
    if (compute_major == 0) {
        threads = 0;
    } else if (compute_major >= 10) {
        architecture_class = CudaArchitecturePolicyClass::kBlackwell;
        threads = kCudaModernThreadsPerBlock;
    } else if (compute_major == 9) {
        architecture_class = CudaArchitecturePolicyClass::kHopper;
        threads = kCudaModernThreadsPerBlock;
    } else if (compute_major == 8) {
        architecture_class = CudaArchitecturePolicyClass::kAmpereAda;
        threads = kCudaModernThreadsPerBlock;
    }
    if (max_threads_per_block < threads) {
        threads = 0;
    }
    return {architecture_class, threads};
}

constexpr bool cuda_kernel_launch_policy_supported(const CudaKernelLaunchPolicy& policy) {
    return policy.threads_per_block != 0 &&
        policy.threads_per_block <= kCudaKernelMaxThreadsPerBlock;
}

// Device reductions reserve their maximum static storage; the launcher selects
// the geometry from the actual CUDA device at matrix allocation time.
constexpr int kThreadsPerBlock = static_cast<int>(kCudaKernelMaxThreadsPerBlock);
constexpr int kMiThreadsPerBlock = static_cast<int>(kCudaModernThreadsPerBlock);
constexpr int kTopKThreadsPerBlock = static_cast<int>(kCudaModernThreadsPerBlock);
constexpr uint32_t kTopKMaxPartialBlocks = 4096;
constexpr uint32_t kTemplateMaxArity = 5;
constexpr uint32_t kMaxMutualInfoBins = 96;
constexpr uint64_t kSpearmanTargetRankCacheMinSamples = 128;
constexpr uint64_t kSpearmanTargetRankCacheMaxSamples = 4096;
constexpr uint64_t kSpearmanTargetRankCacheMinUnaryCandidates = 2;

}  // namespace gafime_cuda_v1

#endif  // GAFIME_CUDA_KERNELS_CUH
