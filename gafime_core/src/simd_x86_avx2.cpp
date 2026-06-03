#include "gafime_simd.h"

#if !defined(GAFIME_USE_DOUBLE_PRECISION) && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

namespace {

float horizontal_sum_256(__m256 value) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    float sum = 0.0f;
    for (float item : tmp) {
        sum += item;
    }
    return sum;
}

}  // namespace

GafimeAccumStats gafime_accumulate_vector_stats_x86_avx2(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
    __m256 sum_x = _mm256_setzero_ps();
    __m256 sum_x2 = _mm256_setzero_ps();
    __m256 dot_xy = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y_centered + i);
        sum_x = _mm256_add_ps(sum_x, xv);
        sum_x2 = _mm256_add_ps(sum_x2, _mm256_mul_ps(xv, xv));
        dot_xy = _mm256_add_ps(dot_xy, _mm256_mul_ps(xv, yv));
    }
    GafimeAccumStats stats{
        horizontal_sum_256(sum_x),
        horizontal_sum_256(sum_x2),
        horizontal_sum_256(dot_xy),
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
