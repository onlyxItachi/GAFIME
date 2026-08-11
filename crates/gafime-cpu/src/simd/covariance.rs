use super::{covariance_common::EqualVectorParts, isa::finite_dispatch_isa};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PearsonSums {
    pub n: usize,
    pub sx: f64,
    pub sy: f64,
    pub sxx: f64,
    pub syy: f64,
    pub sxy: f64,
}

impl PearsonSums {
    pub fn pearson(self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        if !self.sxx.is_finite() || !self.syy.is_finite() || !self.sxy.is_finite() {
            return f32::NAN;
        }
        if self.sxx < 0.0 || self.syy < 0.0 {
            return f32::NAN;
        }
        if self.sxx == 0.0 || self.syy == 0.0 {
            return 0.0;
        }
        let product = self.sxx * self.syy;
        if !product.is_finite() || product <= 0.0 {
            return f32::NAN;
        }
        let correlation = self.sxy / product.sqrt();
        if correlation.is_finite() {
            correlation.clamp(-1.0, 1.0) as f32
        } else {
            f32::NAN
        }
    }
}

pub fn pearson_corr(x: &[f32], y: &[f32]) -> f32 {
    pearson_sums(x, y).pearson()
}

pub fn r2_score(x: &[f32], y: &[f32]) -> f32 {
    let corr = pearson_corr(x, y);
    (corr * corr).clamp(0.0, 1.0)
}

pub fn pearson_sums(x: &[f32], y: &[f32]) -> PearsonSums {
    if x.len() != y.len() || x.is_empty() {
        return PearsonSums::default();
    }
    const EARLY_NONFINITE_PROBE_ROWS: usize = 16;
    for index in 0..x.len().min(EARLY_NONFINITE_PROBE_ROWS) {
        if !x[index].is_finite() || !y[index].is_finite() {
            return pearson_sums_scalar(x, y);
        }
    }
    pearson_sums_dispatched(x, y)
}

pub fn pearson_sums_scalar(x: &[f32], y: &[f32]) -> PearsonSums {
    if x.len() != y.len() {
        return PearsonSums::default();
    }
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            n += 1;
            sx += x_value as f64;
            sy += y_value as f64;
        }
    }
    if n == 0 {
        return PearsonSums::default();
    }
    let mean_x = sx / n as f64;
    let mean_y = sy / n as f64;
    let mut centered = PearsonSums {
        n,
        sx: mean_x,
        sy: mean_y,
        sxx: 0.0,
        syy: 0.0,
        sxy: 0.0,
    };
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            let dx = x_value as f64 - mean_x;
            let dy = y_value as f64 - mean_y;
            centered.sxx += dx * dx;
            centered.syy += dy * dy;
            centered.sxy += dx * dy;
        }
    }
    centered
}

fn pearson_sums_dispatched(x: &[f32], y: &[f32]) -> PearsonSums {
    match finite_dispatch_isa() {
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Avx512 => {
            // SAFETY: The shared selector returns this rung only after its
            // matching runtime feature check. The callee validates all loads.
            unsafe { pearson_sums_avx512(x, y) }
        }
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Avx2 => {
            // SAFETY: As above, the selected target feature is present and
            // the callee validates all raw vector loads.
            unsafe { pearson_sums_avx2(x, y) }
        }
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Sse42 => {
            // SAFETY: As above, the selected target feature is present and
            // the callee validates all raw vector loads.
            unsafe { pearson_sums_sse42(x, y) }
        }
        #[cfg(target_arch = "aarch64")]
        super::isa::IsaLevel::Neon => {
            // SAFETY: NEON is part of the AArch64 baseline. The callee
            // validates slice shape and raw vector-load bounds.
            unsafe { pearson_sums_neon(x, y) }
        }
        _ => pearson_sums_scalar(x, y),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Computes Pearson sums with the AVX-512F implementation.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
unsafe fn pearson_sums_avx512(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f32, 8>::new(x, y) else {
        return PearsonSums::default();
    };
    // 8 fp32 lanes widened to f64 accumulators per iteration (the widest x86 rung).
    let chunks = parts.chunks();
    let mut sx = _mm512_setzero_pd();
    let mut sy = _mm512_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let xv = _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(offset)));
        let yv = _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(offset)));
        if !all_finite_avx512_pd(xv, yv) {
            return pearson_sums_scalar(x, y);
        }
        sx = _mm512_add_pd(sx, xv);
        sy = _mm512_add_pd(sy, yv);
    }

    let n = parts.len();
    let mut sum_x = _mm512_reduce_add_pd(sx);
    let mut sum_y = _mm512_reduce_add_pd(sy);
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_sums_scalar(x, y);
        }
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_avx512(parts, sum_x / n as f64, sum_y / n as f64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Checks two widened AVX-512F vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F.
unsafe fn all_finite_avx512_pd(
    x: std::arch::x86_64::__m512d,
    y: std::arch::x86_64::__m512d,
) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm512_set1_pd(f64::INFINITY);
    let lower = _mm512_set1_pd(f64::NEG_INFINITY);
    let x_mask =
        _mm512_cmp_pd_mask(x, upper, _CMP_LT_OQ) & _mm512_cmp_pd_mask(x, lower, _CMP_GT_OQ);
    let y_mask =
        _mm512_cmp_pd_mask(y, upper, _CMP_LT_OQ) & _mm512_cmp_pd_mask(y, lower, _CMP_GT_OQ);
    x_mask == u8::MAX && y_mask == u8::MAX
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Computes centered finite-input sums over validated AVX-512F vector parts.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F. The two prefixes
/// must have equal lengths divisible by 8, and the two tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds release-safely.
unsafe fn centered_sums_avx512(
    parts: EqualVectorParts<'_, f32, 8>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonSums {
    use std::arch::x86_64::*;

    let chunks = parts.chunks();
    let mean_x_vec = _mm512_set1_pd(mean_x);
    let mean_y_vec = _mm512_set1_pd(mean_y);
    // Four independent accumulator chains per quantity (ILP). Latency-bound f64
    // add/mul is ~4 cycles at ~2/cycle throughput on modern x86, so several
    // independent chains are needed to keep both ports busy; three quantities x
    // four chains = twelve in-flight sums (well under the 32 zmm registers).
    // Non-FMA (add of mul) to stay bit-close to the scalar oracle; the f64 sums
    // are re-associated (regrouped) and the parity tolerance (1e-10) absorbs the
    // ~1e-15 regrouping difference.
    let mut sxx0 = _mm512_setzero_pd();
    let mut sxx1 = _mm512_setzero_pd();
    let mut sxx2 = _mm512_setzero_pd();
    let mut sxx3 = _mm512_setzero_pd();
    let mut syy0 = _mm512_setzero_pd();
    let mut syy1 = _mm512_setzero_pd();
    let mut syy2 = _mm512_setzero_pd();
    let mut syy3 = _mm512_setzero_pd();
    let mut sxy0 = _mm512_setzero_pd();
    let mut sxy1 = _mm512_setzero_pd();
    let mut sxy2 = _mm512_setzero_pd();
    let mut sxy3 = _mm512_setzero_pd();

    let quads = chunks / 4;
    for q in 0..quads {
        let base = 4 * q;
        let o0 = base * 8;
        let dx0 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(o0))),
            mean_x_vec,
        );
        let dy0 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(o0))),
            mean_y_vec,
        );
        sxx0 = _mm512_add_pd(sxx0, _mm512_mul_pd(dx0, dx0));
        syy0 = _mm512_add_pd(syy0, _mm512_mul_pd(dy0, dy0));
        sxy0 = _mm512_add_pd(sxy0, _mm512_mul_pd(dx0, dy0));
        let o1 = (base + 1) * 8;
        let dx1 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(o1))),
            mean_x_vec,
        );
        let dy1 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(o1))),
            mean_y_vec,
        );
        sxx1 = _mm512_add_pd(sxx1, _mm512_mul_pd(dx1, dx1));
        syy1 = _mm512_add_pd(syy1, _mm512_mul_pd(dy1, dy1));
        sxy1 = _mm512_add_pd(sxy1, _mm512_mul_pd(dx1, dy1));
        let o2 = (base + 2) * 8;
        let dx2 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(o2))),
            mean_x_vec,
        );
        let dy2 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(o2))),
            mean_y_vec,
        );
        sxx2 = _mm512_add_pd(sxx2, _mm512_mul_pd(dx2, dx2));
        syy2 = _mm512_add_pd(syy2, _mm512_mul_pd(dy2, dy2));
        sxy2 = _mm512_add_pd(sxy2, _mm512_mul_pd(dx2, dy2));
        let o3 = (base + 3) * 8;
        let dx3 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(o3))),
            mean_x_vec,
        );
        let dy3 = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(o3))),
            mean_y_vec,
        );
        sxx3 = _mm512_add_pd(sxx3, _mm512_mul_pd(dx3, dx3));
        syy3 = _mm512_add_pd(syy3, _mm512_mul_pd(dy3, dy3));
        sxy3 = _mm512_add_pd(sxy3, _mm512_mul_pd(dx3, dy3));
    }
    for c in (quads * 4)..chunks {
        let o = c * 8;
        let dx = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.x_prefix.as_ptr().add(o))),
            mean_x_vec,
        );
        let dy = _mm512_sub_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(parts.y_prefix.as_ptr().add(o))),
            mean_y_vec,
        );
        sxx0 = _mm512_add_pd(sxx0, _mm512_mul_pd(dx, dx));
        syy0 = _mm512_add_pd(syy0, _mm512_mul_pd(dy, dy));
        sxy0 = _mm512_add_pd(sxy0, _mm512_mul_pd(dx, dy));
    }

    let sxx = _mm512_add_pd(_mm512_add_pd(sxx0, sxx1), _mm512_add_pd(sxx2, sxx3));
    let syy = _mm512_add_pd(_mm512_add_pd(syy0, syy1), _mm512_add_pd(syy2, syy3));
    let sxy = _mm512_add_pd(_mm512_add_pd(sxy0, sxy1), _mm512_add_pd(sxy2, sxy3));

    let mut out = PearsonSums {
        n: parts.len(),
        sx: mean_x,
        sy: mean_y,
        sxx: _mm512_reduce_add_pd(sxx),
        syy: _mm512_reduce_add_pd(syy),
        sxy: _mm512_reduce_add_pd(sxy),
    };
    accumulate_centered_tail(&mut out, parts.x_tail, parts.y_tail);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Computes Pearson sums with the AVX2 implementation.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
unsafe fn pearson_sums_avx2(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f32, 4>::new(x, y) else {
        return PearsonSums::default();
    };
    let chunks = parts.chunks();
    let mut sx = _mm256_setzero_pd();
    let mut sy = _mm256_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let xv = _mm256_cvtps_pd(_mm_loadu_ps(parts.x_prefix.as_ptr().add(offset)));
        let yv = _mm256_cvtps_pd(_mm_loadu_ps(parts.y_prefix.as_ptr().add(offset)));
        if !all_finite_avx2_pd(xv, yv) {
            return pearson_sums_scalar(x, y);
        }
        sx = _mm256_add_pd(sx, xv);
        sy = _mm256_add_pd(sy, yv);
    }

    let n = parts.len();
    let mut sum_x = horizontal_sum_avx2_pd(sx);
    let mut sum_y = horizontal_sum_avx2_pd(sy);
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_sums_scalar(x, y);
        }
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_avx2(parts, sum_x / n as f64, sum_y / n as f64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Checks two widened AVX2 vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
unsafe fn all_finite_avx2_pd(x: std::arch::x86_64::__m256d, y: std::arch::x86_64::__m256d) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm256_set1_pd(f64::INFINITY);
    let lower = _mm256_set1_pd(f64::NEG_INFINITY);
    let x_mask = _mm256_and_pd(
        _mm256_cmp_pd(x, upper, _CMP_LT_OQ),
        _mm256_cmp_pd(x, lower, _CMP_GT_OQ),
    );
    let y_mask = _mm256_and_pd(
        _mm256_cmp_pd(y, upper, _CMP_LT_OQ),
        _mm256_cmp_pd(y, lower, _CMP_GT_OQ),
    );
    _mm256_movemask_pd(x_mask) == 0b1111 && _mm256_movemask_pd(y_mask) == 0b1111
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Computes centered finite-input sums over validated AVX2 vector parts.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2. The two prefixes must
/// have equal lengths divisible by 4, and the two tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds release-safely.
unsafe fn centered_sums_avx2(
    parts: EqualVectorParts<'_, f32, 4>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonSums {
    use std::arch::x86_64::*;

    let chunks = parts.chunks();
    let mean_x_vec = _mm256_set1_pd(mean_x);
    let mean_y_vec = _mm256_set1_pd(mean_y);
    let mut sxx = _mm256_setzero_pd();
    let mut syy = _mm256_setzero_pd();
    let mut sxy = _mm256_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let dx = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm_loadu_ps(parts.x_prefix.as_ptr().add(offset))),
            mean_x_vec,
        );
        let dy = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm_loadu_ps(parts.y_prefix.as_ptr().add(offset))),
            mean_y_vec,
        );
        sxx = _mm256_add_pd(sxx, _mm256_mul_pd(dx, dx));
        syy = _mm256_add_pd(syy, _mm256_mul_pd(dy, dy));
        sxy = _mm256_add_pd(sxy, _mm256_mul_pd(dx, dy));
    }

    let mut out = PearsonSums {
        n: parts.len(),
        sx: mean_x,
        sy: mean_y,
        sxx: horizontal_sum_avx2_pd(sxx),
        syy: horizontal_sum_avx2_pd(syy),
        sxy: horizontal_sum_avx2_pd(sxy),
    };
    accumulate_centered_tail(&mut out, parts.x_tail, parts.y_tail);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Reduces four f64 AVX2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
unsafe fn horizontal_sum_avx2_pd(values: std::arch::x86_64::__m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    std::arch::x86_64::_mm256_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
/// Computes Pearson sums with the SSE4.2 implementation.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2. Input shape and
/// unchecked-index bounds are validated before any element is read.
unsafe fn pearson_sums_sse42(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f32, 2>::new(x, y) else {
        return PearsonSums::default();
    };
    let chunks = parts.chunks();
    let mut sx = _mm_setzero_pd();
    let mut sy = _mm_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = _mm_set_pd(
            *parts.x_prefix.get_unchecked(offset + 1) as f64,
            *parts.x_prefix.get_unchecked(offset) as f64,
        );
        let yv = _mm_set_pd(
            *parts.y_prefix.get_unchecked(offset + 1) as f64,
            *parts.y_prefix.get_unchecked(offset) as f64,
        );
        if !all_finite_sse_pd(xv, yv) {
            return pearson_sums_scalar(x, y);
        }
        sx = _mm_add_pd(sx, xv);
        sy = _mm_add_pd(sy, yv);
    }

    let n = parts.len();
    let mut sum_x = horizontal_sum_sse_pd(sx);
    let mut sum_y = horizontal_sum_sse_pd(sy);
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_sums_scalar(x, y);
        }
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_sse42(parts, sum_x / n as f64, sum_y / n as f64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
/// Checks two widened SSE vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2.
unsafe fn all_finite_sse_pd(x: std::arch::x86_64::__m128d, y: std::arch::x86_64::__m128d) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm_set1_pd(f64::INFINITY);
    let lower = _mm_set1_pd(f64::NEG_INFINITY);
    let x_mask = _mm_and_pd(_mm_cmplt_pd(x, upper), _mm_cmpgt_pd(x, lower));
    let y_mask = _mm_and_pd(_mm_cmplt_pd(y, upper), _mm_cmpgt_pd(y, lower));
    _mm_movemask_pd(x_mask) == 0b11 && _mm_movemask_pd(y_mask) == 0b11
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
/// Computes centered finite-input sums over validated SSE4.2 vector parts.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2. The two prefixes
/// must have equal lengths divisible by 2, and the two tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds release-safely.
unsafe fn centered_sums_sse42(
    parts: EqualVectorParts<'_, f32, 2>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonSums {
    use std::arch::x86_64::*;

    let chunks = parts.chunks();
    let mean_x_vec = _mm_set1_pd(mean_x);
    let mean_y_vec = _mm_set1_pd(mean_y);
    let mut sxx = _mm_setzero_pd();
    let mut syy = _mm_setzero_pd();
    let mut sxy = _mm_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = _mm_set_pd(
            *parts.x_prefix.get_unchecked(offset + 1) as f64,
            *parts.x_prefix.get_unchecked(offset) as f64,
        );
        let yv = _mm_set_pd(
            *parts.y_prefix.get_unchecked(offset + 1) as f64,
            *parts.y_prefix.get_unchecked(offset) as f64,
        );
        let dx = _mm_sub_pd(xv, mean_x_vec);
        let dy = _mm_sub_pd(yv, mean_y_vec);
        sxx = _mm_add_pd(sxx, _mm_mul_pd(dx, dx));
        syy = _mm_add_pd(syy, _mm_mul_pd(dy, dy));
        sxy = _mm_add_pd(sxy, _mm_mul_pd(dx, dy));
    }

    let mut out = PearsonSums {
        n: parts.len(),
        sx: mean_x,
        sy: mean_y,
        sxx: horizontal_sum_sse_pd(sxx),
        syy: horizontal_sum_sse_pd(syy),
        sxy: horizontal_sum_sse_pd(sxy),
    };
    accumulate_centered_tail(&mut out, parts.x_tail, parts.y_tail);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
/// Reduces two f64 SSE4.2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2.
unsafe fn horizontal_sum_sse_pd(values: std::arch::x86_64::__m128d) -> f64 {
    let mut lanes = [0.0f64; 2];
    std::arch::x86_64::_mm_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.iter().sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Computes Pearson sums with the AArch64 NEON implementation.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
unsafe fn pearson_sums_neon(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::aarch64::*;

    let Some(parts) = EqualVectorParts::<f32, 2>::new(x, y) else {
        return PearsonSums::default();
    };
    let chunks = parts.chunks();
    let mut sx = vdupq_n_f64(0.0);
    let mut sy = vdupq_n_f64(0.0);
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = f64x2_from_f32_pair(parts.x_prefix.as_ptr().add(offset));
        let yv = f64x2_from_f32_pair(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_neon_f64(xv, yv) {
            return pearson_sums_scalar(x, y);
        }
        sx = vaddq_f64(sx, xv);
        sy = vaddq_f64(sy, yv);
    }

    let n = parts.len();
    let mut sum_x = vaddvq_f64(sx);
    let mut sum_y = vaddvq_f64(sy);
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_sums_scalar(x, y);
        }
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_neon(parts, sum_x / n as f64, sum_y / n as f64)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Checks two widened NEON vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
unsafe fn all_finite_neon_f64(
    x: std::arch::aarch64::float64x2_t,
    y: std::arch::aarch64::float64x2_t,
) -> bool {
    use std::arch::aarch64::*;

    let upper = vdupq_n_f64(f64::INFINITY);
    let lower = vdupq_n_f64(f64::NEG_INFINITY);
    let x_mask = vandq_u64(vcltq_f64(x, upper), vcgtq_f64(x, lower));
    let y_mask = vandq_u64(vcltq_f64(y, upper), vcgtq_f64(y, lower));
    vgetq_lane_u64(x_mask, 0) == u64::MAX
        && vgetq_lane_u64(x_mask, 1) == u64::MAX
        && vgetq_lane_u64(y_mask, 0) == u64::MAX
        && vgetq_lane_u64(y_mask, 1) == u64::MAX
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Computes centered finite-input sums over validated NEON vector parts.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. The two prefixes must
/// have equal lengths divisible by 2, and the two tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds release-safely.
unsafe fn centered_sums_neon(
    parts: EqualVectorParts<'_, f32, 2>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonSums {
    use std::arch::aarch64::*;

    let chunks = parts.chunks();
    let mean_x_vec = vdupq_n_f64(mean_x);
    let mean_y_vec = vdupq_n_f64(mean_y);
    let mut sxx = vdupq_n_f64(0.0);
    let mut syy = vdupq_n_f64(0.0);
    let mut sxy = vdupq_n_f64(0.0);
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let dx = vsubq_f64(
            f64x2_from_f32_pair(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = vsubq_f64(
            f64x2_from_f32_pair(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        sxx = vaddq_f64(sxx, vmulq_f64(dx, dx));
        syy = vaddq_f64(syy, vmulq_f64(dy, dy));
        sxy = vaddq_f64(sxy, vmulq_f64(dx, dy));
    }

    let mut out = PearsonSums {
        n: parts.len(),
        sx: mean_x,
        sy: mean_y,
        sxx: vaddvq_f64(sxx),
        syy: vaddvq_f64(syy),
        sxy: vaddvq_f64(sxy),
    };
    accumulate_centered_tail(&mut out, parts.x_tail, parts.y_tail);
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Widens two contiguous f32 values into a NEON f64 vector.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON and `ptr` points to at
/// least two initialized, readable `f32` values. Callers derive that pointer
/// only from a validated two-lane vector prefix.
unsafe fn f64x2_from_f32_pair(ptr: *const f32) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::*;

    let mut out = vdupq_n_f64(0.0);
    out = vsetq_lane_f64(*ptr as f64, out, 0);
    vsetq_lane_f64(*ptr.add(1) as f64, out, 1)
}

fn accumulate_centered_tail(out: &mut PearsonSums, x: &[f32], y: &[f32]) {
    for (&x_value, &y_value) in x.iter().zip(y) {
        let dx = x_value as f64 - out.sx;
        let dy = y_value as f64 - out.sy;
        out.sxx += dx * dx;
        out.syy += dy * dy;
        out.sxy += dx * dy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_len(len: usize) -> (Vec<f32>, Vec<f32>) {
        let x = (0..len)
            .map(|idx| {
                let value = idx as f32 + 1.0;
                value.sin() * 0.25 + value * 0.01
            })
            .collect::<Vec<_>>();
        let y = (0..len)
            .map(|idx| {
                let value = idx as f32 + 3.0;
                value.cos() * 0.15 + value * 0.02
            })
            .collect::<Vec<_>>();
        (x, y)
    }

    fn dataset() -> (Vec<f32>, Vec<f32>) {
        dataset_len(257)
    }

    #[test]
    fn vector_parts_carry_equal_shape_and_lane_bounds() {
        assert!(EqualVectorParts::<f32, 8>::new(&[], &[]).is_none());
        assert!(EqualVectorParts::<f32, 8>::new(&[1.0, 2.0], &[1.0]).is_none());

        let x = [0.0; 19];
        let y = [1.0; 19];
        let parts = EqualVectorParts::<f32, 8>::new(&x, &y).expect("equal finite shape");
        assert_eq!(parts.x_prefix.len(), 16);
        assert_eq!(parts.y_prefix.len(), 16);
        assert_eq!(parts.x_tail.len(), 3);
        assert_eq!(parts.y_tail.len(), 3);
        assert_eq!(parts.chunks(), 2);
        assert_eq!(parts.len(), 19);
    }

    #[test]
    fn dispatched_pearson_matches_scalar_reduction() {
        let (x, y) = dataset();
        let scalar = pearson_sums_scalar(&x, &y).pearson();
        let dispatched = pearson_corr(&x, &y);
        assert!((scalar - dispatched).abs() <= 5.0e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_paths_match_scalar_centered_covariance() {
        let (x, y) = dataset();
        let scalar = pearson_sums_scalar(&x, &y);
        // SAFETY: Each direct target-feature call is guarded by its runtime
        // feature check, and the test data has equal non-empty slices.
        unsafe {
            if std::is_x86_feature_detected!("sse4.2") {
                assert_close_sums(scalar, pearson_sums_sse42(&x, &y));
            }
            if std::is_x86_feature_detected!("avx2") {
                assert_close_sums(scalar, pearson_sums_avx2(&x, &y));
            }
            if std::is_x86_feature_detected!("avx512f") {
                assert_close_sums(scalar, pearson_sums_avx512(&x, &y));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_paths_match_scalar_across_lane_boundaries() {
        // SAFETY: Each direct target-feature call is guarded by its runtime
        // feature check, and every generated pair has equal non-empty slices.
        unsafe {
            for len in 1..=65 {
                let (x, y) = dataset_len(len);
                let scalar = pearson_sums_scalar(&x, &y);
                if std::is_x86_feature_detected!("sse4.2") {
                    assert_close_sums(scalar, pearson_sums_sse42(&x, &y));
                }
                if std::is_x86_feature_detected!("avx2") {
                    assert_close_sums(scalar, pearson_sums_avx2(&x, &y));
                }
                if std::is_x86_feature_detected!("avx512f") {
                    assert_close_sums(scalar, pearson_sums_avx512(&x, &y));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_sum_pass_detects_nonfinite_vectors_and_tails() {
        let (base_x, base_y) = dataset_len(19);
        let cases = [
            (true, 0, f32::NAN),
            (true, 7, f32::INFINITY),
            (false, 8, f32::NEG_INFINITY),
            (false, 18, f32::NAN),
        ];
        // SAFETY: Each direct target-feature call is guarded by its runtime
        // feature check, and every mutated pair retains equal non-empty shape.
        unsafe {
            for (mutate_x, index, value) in cases {
                let mut x = base_x.clone();
                let mut y = base_y.clone();
                if mutate_x {
                    x[index] = value;
                } else {
                    y[index] = value;
                }
                let scalar = pearson_sums_scalar(&x, &y);
                if std::is_x86_feature_detected!("sse4.2") {
                    assert_eq!(scalar, pearson_sums_sse42(&x, &y));
                }
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(scalar, pearson_sums_avx2(&x, &y));
                }
                if std::is_x86_feature_detected!("avx512f") {
                    assert_eq!(scalar, pearson_sums_avx512(&x, &y));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_paths_reject_invalid_shapes_before_raw_loads() {
        let x = [1.0, 2.0];
        let y = [1.0];
        // SAFETY: Each direct target-feature call is guarded by its runtime
        // feature check. Invalid slice shapes are intentionally supplied to
        // verify that release-safe validation runs before any raw load.
        unsafe {
            if std::is_x86_feature_detected!("sse4.2") {
                assert_eq!(pearson_sums_sse42(&x, &y), PearsonSums::default());
            }
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(pearson_sums_avx2(&x, &y), PearsonSums::default());
            }
            if std::is_x86_feature_detected!("avx512f") {
                assert_eq!(pearson_sums_avx512(&x, &y), PearsonSums::default());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_path_matches_scalar_and_rejects_invalid_shapes() {
        // SAFETY: NEON is part of the AArch64 target baseline. Every generated
        // pair has equal non-empty slices; the final mismatched pair verifies
        // that shape validation runs before any raw load.
        unsafe {
            for len in 1..=65 {
                let (x, y) = dataset_len(len);
                assert_close_sums(pearson_sums_scalar(&x, &y), pearson_sums_neon(&x, &y));
            }
            assert_eq!(
                pearson_sums_neon(&[1.0, 2.0], &[1.0]),
                PearsonSums::default()
            );
            let (mut x, y) = dataset_len(19);
            x[18] = f32::INFINITY;
            assert_eq!(pearson_sums_neon(&x, &y), pearson_sums_scalar(&x, &y));
        }
    }

    #[test]
    fn r2_uses_dispatched_pearson() {
        let (x, y) = dataset();
        let corr = pearson_corr(&x, &y);
        assert!((r2_score(&x, &y) - corr * corr).abs() <= f32::EPSILON);
    }

    #[test]
    fn high_offset_low_variance_pearson_stays_stable() {
        let x = (0..256)
            .map(|idx| 1.0e6f32 + (idx % 7) as f32 * 0.125)
            .collect::<Vec<_>>();
        let y = (0..256)
            .map(|idx| -3.0e6f32 + (idx % 7) as f32 * 0.25)
            .collect::<Vec<_>>();

        let corr = pearson_corr(&x, &y);
        let r2 = r2_score(&x, &y);

        assert!((corr - 1.0).abs() < 1.0e-6);
        assert!((r2 - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn pearson_and_r2_are_clamped_to_valid_ranges() {
        let x = [1.0e20, 1.0e20 + 1.0e14, 1.0e20 + 2.0e14];
        let y = [1.0, 2.0, 3.0];

        let corr = pearson_corr(&x, &y);
        let r2 = r2_score(&x, &y);

        assert!((-1.0..=1.0).contains(&corr));
        assert!((0.0..=1.0).contains(&r2));
    }

    #[test]
    fn non_finite_inputs_fall_back_to_scalar_filtering() {
        let x = [1.0, 2.0, f32::NAN, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(pearson_sums(&x, &y), pearson_sums_scalar(&x, &y));
    }

    #[test]
    fn pearson_finalization_does_not_mask_nonfinite_variance() {
        let x_zero = PearsonSums {
            n: 2,
            sxx: 0.0,
            syy: f64::INFINITY,
            ..PearsonSums::default()
        };
        let y_zero = PearsonSums {
            n: 2,
            sxx: f64::INFINITY,
            syy: 0.0,
            ..PearsonSums::default()
        };
        assert!(x_zero.pearson().is_nan());
        assert!(y_zero.pearson().is_nan());
        assert_eq!(
            PearsonSums {
                n: 2,
                syy: 1.0,
                ..PearsonSums::default()
            }
            .pearson(),
            0.0
        );
    }

    fn assert_close_sums(left: PearsonSums, right: PearsonSums) {
        assert_eq!(left.n, right.n);
        assert!((left.sx - right.sx).abs() <= 1.0e-12);
        assert!((left.sy - right.sy).abs() <= 1.0e-12);
        assert!((left.sxx - right.sxx).abs() <= 1.0e-10);
        assert!((left.syy - right.syy).abs() <= 1.0e-10);
        assert!((left.sxy - right.sxy).abs() <= 1.0e-10);
        assert!((left.pearson() - right.pearson()).abs() <= 1.0e-6);
    }
}
