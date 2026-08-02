#ifndef GAFIME_ROCM_PRECISION_HPP
#define GAFIME_ROCM_PRECISION_HPP

#include <cstdint>

// The precision ABI is intentionally additive.  ABI 1.0 continues to use the
// original float-only kernels, while the ABI 1.1 entry points select one of
// these fully specialised kernel families once when a resident matrix is
// constructed.  Device kernels receive concrete scalar types, never a profile
// discriminator.
namespace gafime_rocm_v1 {

enum class PrecisionLane : uint32_t {
    Fp32 = 1u,
    Mixed = 2u,
    Fp64 = 3u,
};

template <PrecisionLane Lane>
struct PrecisionTraits;

template <>
struct PrecisionTraits<PrecisionLane::Fp32> {
    using Storage = float;
    using Accumulator = float;
    using Result = float;
    static constexpr uint32_t kProfile = static_cast<uint32_t>(PrecisionLane::Fp32);
};

template <>
struct PrecisionTraits<PrecisionLane::Mixed> {
    using Storage = float;
    using Accumulator = double;
    using Result = double;
    static constexpr uint32_t kProfile = static_cast<uint32_t>(PrecisionLane::Mixed);
};

template <>
struct PrecisionTraits<PrecisionLane::Fp64> {
    using Storage = double;
    using Accumulator = double;
    using Result = double;
    static constexpr uint32_t kProfile = static_cast<uint32_t>(PrecisionLane::Fp64);
};

constexpr bool precision_lane_is_valid(uint32_t profile) {
    return profile == static_cast<uint32_t>(PrecisionLane::Fp32) ||
        profile == static_cast<uint32_t>(PrecisionLane::Mixed) ||
        profile == static_cast<uint32_t>(PrecisionLane::Fp64);
}

constexpr bool precision_lane_uses_f32_storage(uint32_t profile) {
    return profile == static_cast<uint32_t>(PrecisionLane::Fp32) ||
        profile == static_cast<uint32_t>(PrecisionLane::Mixed);
}

constexpr bool precision_lane_uses_f32_results(uint32_t profile) {
    return profile == static_cast<uint32_t>(PrecisionLane::Fp32);
}

}  // namespace gafime_rocm_v1

#endif  // GAFIME_ROCM_PRECISION_HPP
