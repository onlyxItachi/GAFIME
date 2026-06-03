#include "gafime_simd.h"

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace {

bool arm_neon_available() {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__) && defined(HWCAP_ASIMD)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

}  // namespace

GafimeAccumStats gafime_accumulate_vector_stats_dispatch(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
#if !defined(GAFIME_USE_DOUBLE_PRECISION)
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        return gafime_accumulate_vector_stats_x86_avx512(x, y_centered, n);
    }
    if (__builtin_cpu_supports("avx2")) {
        return gafime_accumulate_vector_stats_x86_avx2(x, y_centered, n);
    }
    if (__builtin_cpu_supports("sse4.2")) {
        return gafime_accumulate_vector_stats_x86_sse42(x, y_centered, n);
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (arm_neon_available()) {
        return gafime_accumulate_vector_stats_arm_neon(x, y_centered, n);
    }
#endif
#endif
    return gafime_accumulate_vector_stats_scalar(x, y_centered, n);
}

std::string gafime_cpu_dispatch_target() {
#if defined(GAFIME_USE_DOUBLE_PRECISION)
    return "Default";
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        return "AVX512";
    }
    if (__builtin_cpu_supports("avx2")) {
        return "AVX2";
    }
    if (__builtin_cpu_supports("sse4.2")) {
        return "SSE4.2";
    }
    return "Default";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return arm_neon_available() ? "NEON" : "Default";
#else
    return "Default";
#endif
}
