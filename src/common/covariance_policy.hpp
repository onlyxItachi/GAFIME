#ifndef GAFIME_COVARIANCE_POLICY_HPP
#define GAFIME_COVARIANCE_POLICY_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gafime_gpu_abi {

constexpr int kZeroMagnitudeExponent = std::numeric_limits<int>::min() / 4;
constexpr int kCovarianceSafeExponent = 120;

inline int update_finite_abs_exponent(int current, float value) {
    if (!std::isfinite(value) || value == 0.0f) {
        return current;
    }
    return std::max(current, std::ilogb(std::fabs(value)));
}

inline int finite_abs_exponent(const float* values, uint64_t count) {
    int exponent = kZeroMagnitudeExponent;
    for (uint64_t idx = 0; idx < count; ++idx) {
        exponent = update_finite_abs_exponent(exponent, values[idx]);
    }
    return exponent;
}

inline int ceil_log2(uint64_t value) {
    int exponent = 0;
    uint64_t covered = 1;
    while (covered < value && exponent < 63) {
        covered <<= 1;
        ++exponent;
    }
    return exponent;
}

inline int interaction_abs_exponent(
    const int* feature_abs_exponents,
    const uint32_t* combo,
    uint32_t arity
) {
    if (arity == 1) {
        return feature_abs_exponents[combo[0]];
    }
    int64_t exponent = 0;
    for (uint32_t idx = 0; idx < arity; ++idx) {
        const int feature_exponent = feature_abs_exponents[combo[idx]];
        if (feature_exponent == kZeroMagnitudeExponent) {
            return kZeroMagnitudeExponent;
        }
        // A centered value is bounded by twice the raw absolute maximum.
        exponent += static_cast<int64_t>(feature_exponent) + 1;
    }
    return static_cast<int>(std::clamp<int64_t>(
        exponent,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    ));
}

inline bool covariance_requires_scaled_path(
    uint64_t rows,
    int feature_exponent,
    int target_exponent
) {
    if (rows == 0 || feature_exponent == kZeroMagnitudeExponent ||
        target_exponent == kZeroMagnitudeExponent) {
        return false;
    }
    const int64_t row_exponent = ceil_log2(rows);
    const int64_t feature_variance_exponent =
        2 * static_cast<int64_t>(feature_exponent) + row_exponent + 2;
    const int64_t target_variance_exponent =
        2 * static_cast<int64_t>(target_exponent) + row_exponent + 2;
    const int64_t denominator_product_exponent =
        feature_variance_exponent + target_variance_exponent;
    const int64_t safe = kCovarianceSafeExponent;
    return feature_variance_exponent > safe || target_variance_exponent > safe ||
        denominator_product_exponent > safe || feature_variance_exponent < -safe ||
        target_variance_exponent < -safe || denominator_product_exponent < -safe ||
        static_cast<int64_t>(feature_exponent) + row_exponent > safe ||
        static_cast<int64_t>(target_exponent) + row_exponent > safe;
}

}  // namespace gafime_gpu_abi

#endif  // GAFIME_COVARIANCE_POLICY_HPP
