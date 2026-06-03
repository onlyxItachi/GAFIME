#pragma once

#include "gafime_real.h"

#include <cstddef>
#include <string>
#include <vector>

struct GafimeAccumStats {
    real_t sum_x = real_t{0};
    real_t sum_x2 = real_t{0};
    real_t dot_xy = real_t{0};
};

GafimeAccumStats gafime_accumulate_vector_stats_scalar(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);

GafimeAccumStats gafime_accumulate_vector_stats_dispatch(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);

std::string gafime_cpu_dispatch_target();
std::vector<std::string> gafime_available_cpu_dispatch_targets();

#if !defined(GAFIME_USE_DOUBLE_PRECISION)
#if defined(__x86_64__) || defined(_M_X64)
GafimeAccumStats gafime_accumulate_vector_stats_x86_sse42(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);

GafimeAccumStats gafime_accumulate_vector_stats_x86_avx2(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);

GafimeAccumStats gafime_accumulate_vector_stats_x86_avx512(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
GafimeAccumStats gafime_accumulate_vector_stats_arm_neon(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n);
#endif
#endif
