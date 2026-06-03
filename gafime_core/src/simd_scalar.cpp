#include "gafime_simd.h"

GafimeAccumStats gafime_accumulate_vector_stats_scalar(
    const real_t *x,
    const real_t *y_centered,
    std::size_t n) {
    GafimeAccumStats stats;
    for (std::size_t i = 0; i < n; ++i) {
        real_t value = x[i];
        stats.sum_x += value;
        stats.sum_x2 += value * value;
        stats.dot_xy += value * y_centered[i];
    }
    return stats;
}
