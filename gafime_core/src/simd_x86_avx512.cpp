#include "gafime_simd.h"

#if !defined(GAFIME_USE_DOUBLE_PRECISION) && (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))

#include <immintrin.h>

GafimeAccumStats gafime_accumulate_vector_stats_x86_avx512(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
    __m512 sum_x = _mm512_setzero_ps();
    __m512 sum_x2 = _mm512_setzero_ps();
    __m512 dot_xy = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 yv = _mm512_loadu_ps(y_centered + i);
        sum_x = _mm512_add_ps(sum_x, xv);
        sum_x2 = _mm512_add_ps(sum_x2, _mm512_mul_ps(xv, xv));
        dot_xy = _mm512_add_ps(dot_xy, _mm512_mul_ps(xv, yv));
    }
    GafimeAccumStats stats{
        _mm512_reduce_add_ps(sum_x),
        _mm512_reduce_add_ps(sum_x2),
        _mm512_reduce_add_ps(dot_xy),
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
