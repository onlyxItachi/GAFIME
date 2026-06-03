#include "gafime_simd.h"

#if !defined(GAFIME_USE_DOUBLE_PRECISION) && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

namespace {

float horizontal_sum_128(__m128 value) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, value);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

}  // namespace

GafimeAccumStats gafime_accumulate_vector_stats_x86_sse42(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
    __m128 sum_x = _mm_setzero_ps();
    __m128 sum_x2 = _mm_setzero_ps();
    __m128 dot_xy = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 xv = _mm_loadu_ps(x + i);
        __m128 yv = _mm_loadu_ps(y_centered + i);
        sum_x = _mm_add_ps(sum_x, xv);
        sum_x2 = _mm_add_ps(sum_x2, _mm_mul_ps(xv, xv));
        dot_xy = _mm_add_ps(dot_xy, _mm_mul_ps(xv, yv));
    }
    GafimeAccumStats stats{
        horizontal_sum_128(sum_x),
        horizontal_sum_128(sum_x2),
        horizontal_sum_128(dot_xy),
    };
    for (; i < n; ++i) {
        real_t value = x[i];
        stats.sum_x += value;
        stats.sum_x2 += value * value;
        stats.dot_xy += value * y_centered[i];
    }
    return stats;
}

#endif
