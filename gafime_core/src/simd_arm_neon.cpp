#include "gafime_simd.h"

#if !defined(GAFIME_USE_DOUBLE_PRECISION) && (defined(__aarch64__) || defined(_M_ARM64))

#include <arm_neon.h>

GafimeAccumStats gafime_accumulate_vector_stats_arm_neon(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
    float32x4_t sum_x = vdupq_n_f32(0.0f);
    float32x4_t sum_x2 = vdupq_n_f32(0.0f);
    float32x4_t dot_xy = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t yv = vld1q_f32(y_centered + i);
        sum_x = vaddq_f32(sum_x, xv);
        sum_x2 = vmlaq_f32(sum_x2, xv, xv);
        dot_xy = vmlaq_f32(dot_xy, xv, yv);
    }
    GafimeAccumStats stats{
        vaddvq_f32(sum_x),
        vaddvq_f32(sum_x2),
        vaddvq_f32(dot_xy),
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
