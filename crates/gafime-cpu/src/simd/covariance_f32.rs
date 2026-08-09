//! Binary32-only Pearson reduction kernels.
//!
//! This module is deliberately separate from the historical stable reduction
//! ladder. Keeping the arithmetic here in one physical type makes the fp32
//! execution contract auditable while retaining runtime ISA dispatch.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PearsonMomentsF32 {
    n: usize,
    variance_x: f32,
    variance_y: f32,
    covariance: f32,
}

// Metal ABI 1.1 preserves Core's row-ordered fp32 correlation reference for
// short inputs before its parallel covariance pass.  Keep that contracted
// oracle stable while vectorizing the release-sized Core workloads.
const ORDERED_REFERENCE_MAX_ROWS: usize = 256;

impl PearsonMomentsF32 {
    #[inline]
    fn finish(self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        if !self.variance_x.is_finite()
            || !self.variance_y.is_finite()
            || !self.covariance.is_finite()
        {
            return f32::NAN;
        }
        if self.variance_x == 0.0 || self.variance_y == 0.0 {
            return 0.0;
        }
        if self.variance_x < 0.0 || self.variance_y < 0.0 {
            return f32::NAN;
        }
        let denominator = (self.variance_x * self.variance_y).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return f32::NAN;
        }
        let correlation = self.covariance / denominator;
        if correlation.is_finite() {
            correlation.clamp(-1.0, 1.0)
        } else {
            f32::NAN
        }
    }
}

#[derive(Clone, Copy)]
struct EqualVectorParts<'a, const LANES: usize> {
    x_prefix: &'a [f32],
    y_prefix: &'a [f32],
    x_tail: &'a [f32],
    y_tail: &'a [f32],
}

impl<'a, const LANES: usize> EqualVectorParts<'a, LANES> {
    #[inline]
    fn new(x: &'a [f32], y: &'a [f32]) -> Option<Self> {
        assert!(LANES > 0, "SIMD lane count must be non-zero");
        if x.len() != y.len() || x.is_empty() {
            return None;
        }
        let prefix_len = (x.len() / LANES) * LANES;
        let (x_prefix, x_tail) = x.split_at(prefix_len);
        let (y_prefix, y_tail) = y.split_at(prefix_len);
        Some(Self {
            x_prefix,
            y_prefix,
            x_tail,
            y_tail,
        })
    }

    #[inline]
    fn chunks(self) -> usize {
        self.x_prefix.len() / LANES
    }

    #[inline]
    fn len(self) -> usize {
        self.x_prefix.len() + self.x_tail.len()
    }
}

/// Computes Pearson correlation entirely in binary32 arithmetic.
///
/// Non-finite pairs retain the public filtering behavior by falling back to
/// the binary32 scalar implementation. Equal finite inputs use the widest
/// runtime-supported vector implementation.
#[inline]
pub fn pearson_corr_f32(x: &[f32], y: &[f32]) -> f32 {
    pearson_moments_f32(x, y).finish()
}

#[inline]
fn pearson_moments_f32(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    if x.len() <= ORDERED_REFERENCE_MAX_ROWS {
        return pearson_moments_scalar_f32(x, y);
    }
    if x.len() != y.len() || x.is_empty() {
        return PearsonMomentsF32::default();
    }
    if let Some(n) = zero_variance_pair_count_f32(x, y) {
        return PearsonMomentsF32 {
            n,
            ..PearsonMomentsF32::default()
        };
    }

    #[cfg(target_arch = "x86_64")]
    // SAFETY: Every target-feature function is called only after its matching
    // runtime feature check. Each implementation validates slice shape and
    // vector-load bounds before dereferencing raw pointers.
    unsafe {
        if std::is_x86_feature_detected!("avx512f") {
            return pearson_moments_avx512(x, y);
        }
        if std::is_x86_feature_detected!("avx2") {
            return pearson_moments_avx2(x, y);
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return pearson_moments_sse42(x, y);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 baseline. The implementation
        // validates slice shape and load bounds before raw pointer use.
        unsafe { pearson_moments_neon(x, y) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        pearson_moments_scalar_f32(x, y)
    }
}

#[inline(never)]
fn pearson_moments_scalar_f32(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    if x.len() != y.len() || x.is_empty() {
        return PearsonMomentsF32::default();
    }
    if let Some(n) = zero_variance_pair_count_f32(x, y) {
        return PearsonMomentsF32 {
            n,
            ..PearsonMomentsF32::default()
        };
    }
    let mut n = 0usize;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            n += 1;
            sum_x += x_value;
            sum_y += y_value;
        }
    }
    if n == 0 {
        return PearsonMomentsF32::default();
    }
    centered_moments_scalar_f32(x, y, n, sum_x / n as f32, sum_y / n as f32)
}

#[inline]
fn zero_variance_pair_count_f32(x: &[f32], y: &[f32]) -> Option<usize> {
    let mut n = 0usize;
    let mut first_x = 0.0f32;
    let mut first_y = 0.0f32;
    let mut x_constant = true;
    let mut y_constant = true;
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            if n == 0 {
                first_x = x_value;
                first_y = y_value;
            } else {
                if x_constant && x_value != first_x {
                    x_constant = false;
                }
                if y_constant && y_value != first_y {
                    y_constant = false;
                }
                if !x_constant && !y_constant {
                    return None;
                }
            }
            n += 1;
        }
    }
    Some(n)
}

#[inline(never)]
fn centered_moments_scalar_f32(
    x: &[f32],
    y: &[f32],
    n: usize,
    mean_x: f32,
    mean_y: f32,
) -> PearsonMomentsF32 {
    let mut out = PearsonMomentsF32 {
        n,
        ..PearsonMomentsF32::default()
    };
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            let dx = x_value - mean_x;
            let dy = y_value - mean_y;
            out.variance_x += dx * dx;
            out.variance_y += dy * dy;
            out.covariance += dx * dy;
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn pearson_moments_avx512(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<16>::new(x, y) else {
        return PearsonMomentsF32::default();
    };
    let chunks = parts.chunks();
    let mut sum_x0 = _mm512_setzero_ps();
    let mut sum_x1 = _mm512_setzero_ps();
    let mut sum_y0 = _mm512_setzero_ps();
    let mut sum_y1 = _mm512_setzero_ps();
    for chunk in 0..chunks {
        let offset = chunk * 16;
        let xv = _mm512_loadu_ps(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm512_loadu_ps(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_avx512_ps(xv, yv) {
            return pearson_moments_scalar_f32(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm512_add_ps(sum_x0, xv);
            sum_y0 = _mm512_add_ps(sum_y0, yv);
        } else {
            sum_x1 = _mm512_add_ps(sum_x1, xv);
            sum_y1 = _mm512_add_ps(sum_y1, yv);
        }
    }

    let mut sum_x = _mm512_reduce_add_ps(_mm512_add_ps(sum_x0, sum_x1));
    let mut sum_y = _mm512_reduce_add_ps(_mm512_add_ps(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f32(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_avx512(parts, sum_x / n as f32, sum_y / n as f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn all_finite_avx512_ps(x: std::arch::x86_64::__m512, y: std::arch::x86_64::__m512) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm512_set1_ps(f32::INFINITY);
    let lower = _mm512_set1_ps(f32::NEG_INFINITY);
    let x_mask =
        _mm512_cmp_ps_mask(x, upper, _CMP_LT_OQ) & _mm512_cmp_ps_mask(x, lower, _CMP_GT_OQ);
    let y_mask =
        _mm512_cmp_ps_mask(y, upper, _CMP_LT_OQ) & _mm512_cmp_ps_mask(y, lower, _CMP_GT_OQ);
    x_mask == u16::MAX && y_mask == u16::MAX
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn centered_moments_avx512(
    parts: EqualVectorParts<'_, 16>,
    mean_x: f32,
    mean_y: f32,
) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm512_set1_ps(mean_x);
    let mean_y_vec = _mm512_set1_ps(mean_y);
    let mut xx0 = _mm512_setzero_ps();
    let mut xx1 = _mm512_setzero_ps();
    let mut yy0 = _mm512_setzero_ps();
    let mut yy1 = _mm512_setzero_ps();
    let mut xy0 = _mm512_setzero_ps();
    let mut xy1 = _mm512_setzero_ps();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 16;
        let dx = _mm512_sub_ps(
            _mm512_loadu_ps(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm512_sub_ps(
            _mm512_loadu_ps(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm512_add_ps(xx0, _mm512_mul_ps(dx, dx));
            yy0 = _mm512_add_ps(yy0, _mm512_mul_ps(dy, dy));
            xy0 = _mm512_add_ps(xy0, _mm512_mul_ps(dx, dy));
        } else {
            xx1 = _mm512_add_ps(xx1, _mm512_mul_ps(dx, dx));
            yy1 = _mm512_add_ps(yy1, _mm512_mul_ps(dy, dy));
            xy1 = _mm512_add_ps(xy1, _mm512_mul_ps(dx, dy));
        }
    }

    let mut out = PearsonMomentsF32 {
        n: parts.len(),
        variance_x: _mm512_reduce_add_ps(_mm512_add_ps(xx0, xx1)),
        variance_y: _mm512_reduce_add_ps(_mm512_add_ps(yy0, yy1)),
        covariance: _mm512_reduce_add_ps(_mm512_add_ps(xy0, xy1)),
    };
    accumulate_centered_tail_f32(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn pearson_moments_avx2(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<8>::new(x, y) else {
        return PearsonMomentsF32::default();
    };
    let mut sum_x0 = _mm256_setzero_ps();
    let mut sum_x1 = _mm256_setzero_ps();
    let mut sum_y0 = _mm256_setzero_ps();
    let mut sum_y1 = _mm256_setzero_ps();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 8;
        let xv = _mm256_loadu_ps(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm256_loadu_ps(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_avx2_ps(xv, yv) {
            return pearson_moments_scalar_f32(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm256_add_ps(sum_x0, xv);
            sum_y0 = _mm256_add_ps(sum_y0, yv);
        } else {
            sum_x1 = _mm256_add_ps(sum_x1, xv);
            sum_y1 = _mm256_add_ps(sum_y1, yv);
        }
    }

    let mut sum_x = horizontal_sum_avx2_ps(_mm256_add_ps(sum_x0, sum_x1));
    let mut sum_y = horizontal_sum_avx2_ps(_mm256_add_ps(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f32(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_avx2(parts, sum_x / n as f32, sum_y / n as f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn all_finite_avx2_ps(x: std::arch::x86_64::__m256, y: std::arch::x86_64::__m256) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm256_set1_ps(f32::INFINITY);
    let lower = _mm256_set1_ps(f32::NEG_INFINITY);
    let x_mask = _mm256_and_ps(
        _mm256_cmp_ps(x, upper, _CMP_LT_OQ),
        _mm256_cmp_ps(x, lower, _CMP_GT_OQ),
    );
    let y_mask = _mm256_and_ps(
        _mm256_cmp_ps(y, upper, _CMP_LT_OQ),
        _mm256_cmp_ps(y, lower, _CMP_GT_OQ),
    );
    _mm256_movemask_ps(x_mask) == 0xff && _mm256_movemask_ps(y_mask) == 0xff
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn centered_moments_avx2(
    parts: EqualVectorParts<'_, 8>,
    mean_x: f32,
    mean_y: f32,
) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm256_set1_ps(mean_x);
    let mean_y_vec = _mm256_set1_ps(mean_y);
    let mut xx0 = _mm256_setzero_ps();
    let mut xx1 = _mm256_setzero_ps();
    let mut yy0 = _mm256_setzero_ps();
    let mut yy1 = _mm256_setzero_ps();
    let mut xy0 = _mm256_setzero_ps();
    let mut xy1 = _mm256_setzero_ps();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 8;
        let dx = _mm256_sub_ps(
            _mm256_loadu_ps(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm256_sub_ps(
            _mm256_loadu_ps(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm256_add_ps(xx0, _mm256_mul_ps(dx, dx));
            yy0 = _mm256_add_ps(yy0, _mm256_mul_ps(dy, dy));
            xy0 = _mm256_add_ps(xy0, _mm256_mul_ps(dx, dy));
        } else {
            xx1 = _mm256_add_ps(xx1, _mm256_mul_ps(dx, dx));
            yy1 = _mm256_add_ps(yy1, _mm256_mul_ps(dy, dy));
            xy1 = _mm256_add_ps(xy1, _mm256_mul_ps(dx, dy));
        }
    }

    let mut out = PearsonMomentsF32 {
        n: parts.len(),
        variance_x: horizontal_sum_avx2_ps(_mm256_add_ps(xx0, xx1)),
        variance_y: horizontal_sum_avx2_ps(_mm256_add_ps(yy0, yy1)),
        covariance: horizontal_sum_avx2_ps(_mm256_add_ps(xy0, xy1)),
    };
    accumulate_centered_tail_f32(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2_ps(values: std::arch::x86_64::__m256) -> f32 {
    let mut lanes = [0.0f32; 8];
    std::arch::x86_64::_mm256_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
unsafe fn pearson_moments_sse42(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<4>::new(x, y) else {
        return PearsonMomentsF32::default();
    };
    let mut sum_x0 = _mm_setzero_ps();
    let mut sum_x1 = _mm_setzero_ps();
    let mut sum_y0 = _mm_setzero_ps();
    let mut sum_y1 = _mm_setzero_ps();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let xv = _mm_loadu_ps(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm_loadu_ps(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_sse_ps(xv, yv) {
            return pearson_moments_scalar_f32(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm_add_ps(sum_x0, xv);
            sum_y0 = _mm_add_ps(sum_y0, yv);
        } else {
            sum_x1 = _mm_add_ps(sum_x1, xv);
            sum_y1 = _mm_add_ps(sum_y1, yv);
        }
    }

    let mut sum_x = horizontal_sum_sse_ps(_mm_add_ps(sum_x0, sum_x1));
    let mut sum_y = horizontal_sum_sse_ps(_mm_add_ps(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f32(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_sse42(parts, sum_x / n as f32, sum_y / n as f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn all_finite_sse_ps(x: std::arch::x86_64::__m128, y: std::arch::x86_64::__m128) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm_set1_ps(f32::INFINITY);
    let lower = _mm_set1_ps(f32::NEG_INFINITY);
    let x_mask = _mm_and_ps(_mm_cmplt_ps(x, upper), _mm_cmpgt_ps(x, lower));
    let y_mask = _mm_and_ps(_mm_cmplt_ps(y, upper), _mm_cmpgt_ps(y, lower));
    _mm_movemask_ps(x_mask) == 0b1111 && _mm_movemask_ps(y_mask) == 0b1111
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
unsafe fn centered_moments_sse42(
    parts: EqualVectorParts<'_, 4>,
    mean_x: f32,
    mean_y: f32,
) -> PearsonMomentsF32 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm_set1_ps(mean_x);
    let mean_y_vec = _mm_set1_ps(mean_y);
    let mut xx0 = _mm_setzero_ps();
    let mut xx1 = _mm_setzero_ps();
    let mut yy0 = _mm_setzero_ps();
    let mut yy1 = _mm_setzero_ps();
    let mut xy0 = _mm_setzero_ps();
    let mut xy1 = _mm_setzero_ps();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let dx = _mm_sub_ps(
            _mm_loadu_ps(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm_sub_ps(
            _mm_loadu_ps(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm_add_ps(xx0, _mm_mul_ps(dx, dx));
            yy0 = _mm_add_ps(yy0, _mm_mul_ps(dy, dy));
            xy0 = _mm_add_ps(xy0, _mm_mul_ps(dx, dy));
        } else {
            xx1 = _mm_add_ps(xx1, _mm_mul_ps(dx, dx));
            yy1 = _mm_add_ps(yy1, _mm_mul_ps(dy, dy));
            xy1 = _mm_add_ps(xy1, _mm_mul_ps(dx, dy));
        }
    }

    let mut out = PearsonMomentsF32 {
        n: parts.len(),
        variance_x: horizontal_sum_sse_ps(_mm_add_ps(xx0, xx1)),
        variance_y: horizontal_sum_sse_ps(_mm_add_ps(yy0, yy1)),
        covariance: horizontal_sum_sse_ps(_mm_add_ps(xy0, xy1)),
    };
    accumulate_centered_tail_f32(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn horizontal_sum_sse_ps(values: std::arch::x86_64::__m128) -> f32 {
    let mut lanes = [0.0f32; 4];
    std::arch::x86_64::_mm_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.into_iter().sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn pearson_moments_neon(x: &[f32], y: &[f32]) -> PearsonMomentsF32 {
    use std::arch::aarch64::*;

    let Some(parts) = EqualVectorParts::<4>::new(x, y) else {
        return PearsonMomentsF32::default();
    };
    let mut sum_x0 = vdupq_n_f32(0.0);
    let mut sum_x1 = vdupq_n_f32(0.0);
    let mut sum_y0 = vdupq_n_f32(0.0);
    let mut sum_y1 = vdupq_n_f32(0.0);
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let xv = vld1q_f32(parts.x_prefix.as_ptr().add(offset));
        let yv = vld1q_f32(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_neon_f32(xv, yv) {
            return pearson_moments_scalar_f32(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = vaddq_f32(sum_x0, xv);
            sum_y0 = vaddq_f32(sum_y0, yv);
        } else {
            sum_x1 = vaddq_f32(sum_x1, xv);
            sum_y1 = vaddq_f32(sum_y1, yv);
        }
    }

    let mut sum_x = vaddvq_f32(vaddq_f32(sum_x0, sum_x1));
    let mut sum_y = vaddvq_f32(vaddq_f32(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f32(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_neon(parts, sum_x / n as f32, sum_y / n as f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn all_finite_neon_f32(
    x: std::arch::aarch64::float32x4_t,
    y: std::arch::aarch64::float32x4_t,
) -> bool {
    use std::arch::aarch64::*;

    let upper = vdupq_n_f32(f32::INFINITY);
    let lower = vdupq_n_f32(f32::NEG_INFINITY);
    let x_mask = vandq_u32(vcltq_f32(x, upper), vcgtq_f32(x, lower));
    let y_mask = vandq_u32(vcltq_f32(y, upper), vcgtq_f32(y, lower));
    vminvq_u32(x_mask) == u32::MAX && vminvq_u32(y_mask) == u32::MAX
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn centered_moments_neon(
    parts: EqualVectorParts<'_, 4>,
    mean_x: f32,
    mean_y: f32,
) -> PearsonMomentsF32 {
    use std::arch::aarch64::*;

    let mean_x_vec = vdupq_n_f32(mean_x);
    let mean_y_vec = vdupq_n_f32(mean_y);
    let mut xx0 = vdupq_n_f32(0.0);
    let mut xx1 = vdupq_n_f32(0.0);
    let mut yy0 = vdupq_n_f32(0.0);
    let mut yy1 = vdupq_n_f32(0.0);
    let mut xy0 = vdupq_n_f32(0.0);
    let mut xy1 = vdupq_n_f32(0.0);
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let dx = vsubq_f32(vld1q_f32(parts.x_prefix.as_ptr().add(offset)), mean_x_vec);
        let dy = vsubq_f32(vld1q_f32(parts.y_prefix.as_ptr().add(offset)), mean_y_vec);
        if chunk & 1 == 0 {
            xx0 = vaddq_f32(xx0, vmulq_f32(dx, dx));
            yy0 = vaddq_f32(yy0, vmulq_f32(dy, dy));
            xy0 = vaddq_f32(xy0, vmulq_f32(dx, dy));
        } else {
            xx1 = vaddq_f32(xx1, vmulq_f32(dx, dx));
            yy1 = vaddq_f32(yy1, vmulq_f32(dy, dy));
            xy1 = vaddq_f32(xy1, vmulq_f32(dx, dy));
        }
    }

    let mut out = PearsonMomentsF32 {
        n: parts.len(),
        variance_x: vaddvq_f32(vaddq_f32(xx0, xx1)),
        variance_y: vaddvq_f32(vaddq_f32(yy0, yy1)),
        covariance: vaddvq_f32(vaddq_f32(xy0, xy1)),
    };
    accumulate_centered_tail_f32(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

#[inline]
fn accumulate_centered_tail_f32(
    out: &mut PearsonMomentsF32,
    x: &[f32],
    y: &[f32],
    mean_x: f32,
    mean_y: f32,
) {
    for (&x_value, &y_value) in x.iter().zip(y) {
        let dx = x_value - mean_x;
        let dy = y_value - mean_y;
        out.variance_x += dx * dx;
        out.variance_y += dy * dy;
        out.covariance += dx * dy;
    }
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, time::Instant};

    use super::*;

    fn dataset(len: usize) -> (Vec<f32>, Vec<f32>) {
        let x = (0..len)
            .map(|index| {
                let value = index as f32 + 1.0;
                value.sin() * 0.25 + value * 0.001
            })
            .collect::<Vec<_>>();
        let y = (0..len)
            .map(|index| {
                let value = index as f32 + 3.0;
                value.cos() * 0.15 + value * 0.002
            })
            .collect::<Vec<_>>();
        (x, y)
    }

    fn assert_close(left: PearsonMomentsF32, right: PearsonMomentsF32) {
        assert_eq!(left.n, right.n);
        let scale_x = left.variance_x.abs().max(right.variance_x.abs()).max(1.0);
        let scale_y = left.variance_y.abs().max(right.variance_y.abs()).max(1.0);
        let scale_xy = left.covariance.abs().max(right.covariance.abs()).max(1.0);
        assert!((left.variance_x - right.variance_x).abs() <= 3.0e-5 * scale_x);
        assert!((left.variance_y - right.variance_y).abs() <= 3.0e-5 * scale_y);
        assert!((left.covariance - right.covariance).abs() <= 3.0e-5 * scale_xy);
        let left_corr = left.finish();
        let right_corr = right.finish();
        assert!((left_corr - right_corr).abs() <= 3.0e-5);
    }

    #[test]
    fn dispatched_binary32_matches_scalar_across_vector_tails() {
        for len in 1..=129 {
            let (x, y) = dataset(len);
            assert_eq!(
                pearson_moments_scalar_f32(&x, &y),
                pearson_moments_f32(&x, &y),
            );
        }
    }

    #[test]
    fn short_ordered_reference_boundary_is_exact() {
        for len in [255, 256] {
            let (x, y) = dataset(len);
            assert_eq!(
                pearson_moments_scalar_f32(&x, &y),
                pearson_moments_f32(&x, &y),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_x86_binary32_rung_matches_scalar() {
        let (x, y) = dataset(4099);
        let scalar = pearson_moments_scalar_f32(&x, &y);
        // SAFETY: Each call is guarded by its exact runtime feature check and
        // receives equal, initialized input slices.
        unsafe {
            if std::is_x86_feature_detected!("sse4.2") {
                assert_close(scalar, pearson_moments_sse42(&x, &y));
            }
            if std::is_x86_feature_detected!("avx2") {
                assert_close(scalar, pearson_moments_avx2(&x, &y));
            }
            if std::is_x86_feature_detected!("avx512f") {
                assert_close(scalar, pearson_moments_avx512(&x, &y));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_x86_rung_filters_nonfinite_vectors_and_tails() {
        let (base_x, base_y) = dataset(131);
        let cases = [
            (true, 0, f32::NAN),
            (false, 17, f32::INFINITY),
            (true, 130, f32::NEG_INFINITY),
        ];
        // SAFETY: Each call is guarded by its exact runtime feature check and
        // all mutated inputs retain equal initialized slice lengths.
        unsafe {
            for (mutate_x, index, value) in cases {
                let mut x = base_x.clone();
                let mut y = base_y.clone();
                if mutate_x {
                    x[index] = value;
                } else {
                    y[index] = value;
                }
                let scalar = pearson_moments_scalar_f32(&x, &y);
                if std::is_x86_feature_detected!("sse4.2") {
                    assert_eq!(scalar, pearson_moments_sse42(&x, &y));
                }
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(scalar, pearson_moments_avx2(&x, &y));
                }
                if std::is_x86_feature_detected!("avx512f") {
                    assert_eq!(scalar, pearson_moments_avx512(&x, &y));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_x86_rung_rejects_mismatched_shapes_before_loading() {
        let x = [1.0, 2.0];
        let y = [1.0];
        // SAFETY: Each call is guarded by its exact runtime feature check. The
        // invalid shape is intentional and must be rejected before raw loads.
        unsafe {
            if std::is_x86_feature_detected!("sse4.2") {
                assert_eq!(pearson_moments_sse42(&x, &y), PearsonMomentsF32::default());
            }
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(pearson_moments_avx2(&x, &y), PearsonMomentsF32::default());
            }
            if std::is_x86_feature_detected!("avx512f") {
                assert_eq!(pearson_moments_avx512(&x, &y), PearsonMomentsF32::default());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_binary32_rung_matches_scalar() {
        let (mut x, y) = dataset(4099);
        // SAFETY: NEON is an AArch64 baseline feature and the inputs are equal,
        // initialized slices. The second call verifies the scalar filtering
        // fallback for a non-finite tail.
        unsafe {
            assert_close(
                pearson_moments_scalar_f32(&x, &y),
                pearson_moments_neon(&x, &y),
            );
            x[4098] = f32::NAN;
            assert_eq!(
                pearson_moments_scalar_f32(&x, &y),
                pearson_moments_neon(&x, &y)
            );
            assert_eq!(
                pearson_moments_neon(&[1.0, 2.0], &[1.0]),
                PearsonMomentsF32::default()
            );
        }
    }

    #[test]
    fn nonfinite_pairs_are_filtered_and_zero_variance_is_preserved() {
        let x = [1.0, 2.0, f32::NAN, 4.0, f32::INFINITY, 6.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, f32::NEG_INFINITY];
        assert_eq!(
            pearson_moments_f32(&x, &y),
            pearson_moments_scalar_f32(&x, &y)
        );
        assert_eq!(pearson_corr_f32(&[2.0; 64], &[1.0; 64]), 0.0);
        assert_eq!(pearson_corr_f32(&[0.1; 131], &[0.3; 131]), 0.0);
        assert_eq!(pearson_corr_f32(&[], &[]), 0.0);
        assert_eq!(pearson_corr_f32(&[1.0], &[]), 0.0);
    }

    #[test]
    fn binary32_overflow_remains_visible_as_nan() {
        let x = [1.0e20, -1.0e20, 1.0e20, -1.0e20];
        let y = [1.0, -1.0, -1.0, 1.0];
        assert!(pearson_corr_f32(&x, &y).is_nan());
    }

    #[test]
    #[ignore = "manual release microbenchmark; run with --release --ignored --nocapture"]
    fn binary32_pearson_release_microbenchmark() {
        fn scalar(x: &[f32], y: &[f32]) -> f32 {
            pearson_moments_scalar_f32(x, y).finish()
        }

        fn measure(kernel: fn(&[f32], &[f32]) -> f32, x: &[f32], y: &[f32], reps: u32) -> u128 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(kernel(black_box(x), black_box(y)));
            }
            started.elapsed().as_nanos() / u128::from(reps)
        }

        let (x, y) = dataset(1 << 20);
        for _ in 0..10 {
            black_box(scalar(black_box(&x), black_box(&y)));
            black_box(pearson_corr_f32(black_box(&x), black_box(&y)));
        }
        let blocks = 6;
        let repetitions_per_block = 10;
        let mut scalar_samples = Vec::with_capacity(blocks);
        let mut simd_samples = Vec::with_capacity(blocks);
        for block in 0..blocks {
            if block & 1 == 0 {
                scalar_samples.push(measure(scalar, &x, &y, repetitions_per_block));
                simd_samples.push(measure(pearson_corr_f32, &x, &y, repetitions_per_block));
            } else {
                simd_samples.push(measure(pearson_corr_f32, &x, &y, repetitions_per_block));
                scalar_samples.push(measure(scalar, &x, &y, repetitions_per_block));
            }
        }
        scalar_samples.sort_unstable();
        simd_samples.sort_unstable();
        let scalar_ns = scalar_samples[blocks / 2];
        let simd_ns = simd_samples[blocks / 2];
        eprintln!(
            "fp32 Pearson rows={} blocks={} reps_per_block={} scalar_median_ns={} simd_median_ns={} speedup={:.3} scalar_samples_ns={:?} simd_samples_ns={:?}",
            x.len(),
            blocks,
            repetitions_per_block,
            scalar_ns,
            simd_ns,
            scalar_ns as f32 / simd_ns as f32,
            scalar_samples,
            simd_samples,
        );
    }
}
