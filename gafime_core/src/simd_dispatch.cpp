#include "gafime_simd.h"

#include <cstdlib>

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace {

std::string normalize_target_name(const char *value) {
    if (value == nullptr) {
        return {};
    }
    std::string out(value);
    for (char &ch : out) {
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - 'a' + 'A');
        } else if (ch == '_' || ch == '-') {
            ch = '.';
        }
    }
    if (out == "SSE42") {
        return "SSE4.2";
    }
    if (out == "SCALAR") {
        return "Default";
    }
    return out;
}

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

bool x86_sse42_available() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 1);
    return (info[2] & (1 << 20)) != 0;
#else
    return false;
#endif
}

bool x86_avx2_available() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 1);
    bool osxsave = (info[2] & (1 << 27)) != 0;
    bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || ((_xgetbv(0) & 0x6) != 0x6)) {
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

bool x86_avx512_available() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 1);
    bool osxsave = (info[2] & (1 << 27)) != 0;
    bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || ((_xgetbv(0) & 0xE6) != 0xE6)) {
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 16)) != 0;
#else
    return false;
#endif
}

std::string forced_dispatch_target() {
    return normalize_target_name(std::getenv("GAFIME_CPU_DISPATCH"));
}

}  // namespace

GafimeAccumStats gafime_accumulate_vector_stats_dispatch(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
#if !defined(GAFIME_USE_DOUBLE_PRECISION)
#if defined(__x86_64__) || defined(_M_X64)
    std::string forced = forced_dispatch_target();
    if (forced == "Default") {
        return gafime_accumulate_vector_stats_scalar(x, y_centered, n);
    }
    if (forced == "AVX512" && x86_avx512_available()) {
        return gafime_accumulate_vector_stats_x86_avx512(x, y_centered, n);
    }
    if (forced == "AVX2" && x86_avx2_available()) {
        return gafime_accumulate_vector_stats_x86_avx2(x, y_centered, n);
    }
    if (forced == "SSE4.2" && x86_sse42_available()) {
        return gafime_accumulate_vector_stats_x86_sse42(x, y_centered, n);
    }
    if (x86_avx512_available()) {
        return gafime_accumulate_vector_stats_x86_avx512(x, y_centered, n);
    }
    if (x86_avx2_available()) {
        return gafime_accumulate_vector_stats_x86_avx2(x, y_centered, n);
    }
    if (x86_sse42_available()) {
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
#elif defined(__x86_64__) || defined(_M_X64)
    std::string forced = forced_dispatch_target();
    if (forced == "Default") {
        return "Default";
    }
    if (forced == "AVX512" && x86_avx512_available()) {
        return "AVX512";
    }
    if (forced == "AVX2" && x86_avx2_available()) {
        return "AVX2";
    }
    if (forced == "SSE4.2" && x86_sse42_available()) {
        return "SSE4.2";
    }
    if (x86_avx512_available()) {
        return "AVX512";
    }
    if (x86_avx2_available()) {
        return "AVX2";
    }
    if (x86_sse42_available()) {
        return "SSE4.2";
    }
    return "Default";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return arm_neon_available() ? "NEON" : "Default";
#else
    return "Default";
#endif
}

std::vector<std::string> gafime_available_cpu_dispatch_targets() {
    std::vector<std::string> targets{"Default"};
#if !defined(GAFIME_USE_DOUBLE_PRECISION)
#if defined(__x86_64__) || defined(_M_X64)
    if (x86_sse42_available()) {
        targets.push_back("SSE4.2");
    }
    if (x86_avx2_available()) {
        targets.push_back("AVX2");
    }
    if (x86_avx512_available()) {
        targets.push_back("AVX512");
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (arm_neon_available()) {
        targets.push_back("NEON");
    }
#endif
#endif
    return targets;
}
