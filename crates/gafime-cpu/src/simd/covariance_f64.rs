//! Binary64-only Pearson reduction kernels.
//!
//! The public fp64 profile reaches this module with binary64 resident values.
//! Every SIMD rung loads, reduces, and finalizes those values as `f64`; there is
//! no narrower staging type and no fused multiply-add arithmetic.

use super::{
    covariance_common::{finalize_correlation_f64, EqualVectorParts},
    isa::finite_dispatch_isa,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PearsonMomentsF64 {
    n: usize,
    variance_x: f64,
    variance_y: f64,
    covariance: f64,
}

impl PearsonMomentsF64 {
    #[inline]
    fn finish(self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        finalize_correlation_f64(self.variance_x, self.variance_y, self.covariance)
    }

    /// Preserve the reduction's numeric result while exposing whether its
    /// binary64 normalization is defined. None means zero or underflowed
    /// normalization; Some(NaN) preserves an arithmetic failure.
    #[inline]
    fn finish_checked(self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        if !self.variance_x.is_finite()
            || !self.variance_y.is_finite()
            || !self.covariance.is_finite()
            || self.variance_x < 0.0
            || self.variance_y < 0.0
        {
            return Some(self.finish());
        }
        if self.variance_x == 0.0 || self.variance_y == 0.0 {
            return None;
        }
        let product = self.variance_x * self.variance_y;
        if product == 0.0 {
            return None;
        }
        Some(self.finish())
    }
}

/// Computes Pearson correlation entirely in binary64 arithmetic.
///
/// Finite equal-length inputs use the widest supported ISA rung. If either
/// input contains a non-finite value, the scalar implementation preserves the
/// established pair-filtering behavior.
#[inline]
pub fn pearson_corr_f64(x: &[f64], y: &[f64]) -> f64 {
    pearson_moments_f64(x, y).finish()
}

/// Crate-internal Pearson result preserving definedness from the same f64
/// moments used by pearson_corr_f64. Public callers retain legacy zero
/// behavior for a zero variance.
#[inline]
pub(crate) fn pearson_corr_f64_checked(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson_moments_f64(x, y).finish_checked()
}

#[inline]
fn pearson_moments_f64(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    if x.len() != y.len() || x.is_empty() {
        return PearsonMomentsF64::default();
    }
    if let Some(axes) = finite_variance_axes_f64(x, y) {
        if axes.n == 0 {
            return PearsonMomentsF64::default();
        }
        if axes.x_constant || axes.y_constant {
            return constant_axis_moments_f64(x, y, axes);
        }
    }

    match finite_dispatch_isa() {
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Avx512 => {
            // SAFETY: The shared selector returns this rung only after its
            // matching runtime feature check. The implementation validates
            // shape and vector bounds before raw pointer use.
            unsafe { pearson_moments_avx512(x, y) }
        }
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Avx2 => {
            // SAFETY: As above, the selected target feature is present and
            // the callee validates every raw vector load.
            unsafe { pearson_moments_avx2(x, y) }
        }
        #[cfg(target_arch = "x86_64")]
        super::isa::IsaLevel::Sse42 => {
            // SAFETY: As above, the selected target feature is present and
            // the callee validates every raw vector load.
            unsafe { pearson_moments_sse42(x, y) }
        }
        #[cfg(target_arch = "aarch64")]
        super::isa::IsaLevel::Neon => {
            // SAFETY: NEON is part of the AArch64 baseline. The callee
            // validates shape and vector bounds before raw pointer use.
            unsafe { pearson_moments_neon(x, y) }
        }
        _ => pearson_moments_scalar_f64(x, y),
    }
}

#[inline(never)]
fn pearson_moments_scalar_f64(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    if x.len() != y.len() || x.is_empty() {
        return PearsonMomentsF64::default();
    }
    if let Some(axes) = finite_variance_axes_f64(x, y) {
        if axes.n == 0 {
            return PearsonMomentsF64::default();
        }
        if axes.x_constant || axes.y_constant {
            return constant_axis_moments_f64(x, y, axes);
        }
    }
    let mut n = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            n += 1;
            sum_x += x_value;
            sum_y += y_value;
        }
    }
    if n == 0 {
        return PearsonMomentsF64::default();
    }
    centered_moments_scalar_f64(x, y, n, sum_x / n as f64, sum_y / n as f64)
}

struct FiniteVarianceAxesF64 {
    n: usize,
    x_constant: bool,
    y_constant: bool,
}

#[inline]
fn finite_variance_axes_f64(x: &[f64], y: &[f64]) -> Option<FiniteVarianceAxesF64> {
    let mut n = 0usize;
    let mut first_x = 0.0f64;
    let mut first_y = 0.0f64;
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
    Some(FiniteVarianceAxesF64 {
        n,
        x_constant,
        y_constant,
    })
}

#[inline(never)]
fn constant_axis_moments_f64(
    x: &[f64],
    y: &[f64],
    axes: FiniteVarianceAxesF64,
) -> PearsonMomentsF64 {
    debug_assert!(axes.n > 0);
    debug_assert!(axes.x_constant || axes.y_constant);
    let mut out = PearsonMomentsF64 {
        n: axes.n,
        ..PearsonMomentsF64::default()
    };
    if axes.x_constant && axes.y_constant {
        return out;
    }

    if axes.x_constant {
        let mut sum_y = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                sum_y += y_value;
            }
        }
        let mean_y = sum_y / axes.n as f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                let dy = y_value - mean_y;
                out.variance_y += dy * dy;
            }
        }
    } else {
        let mut sum_x = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                sum_x += x_value;
            }
        }
        let mean_x = sum_x / axes.n as f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                let dx = x_value - mean_x;
                out.variance_x += dx * dx;
            }
        }
    }
    out
}

#[inline(never)]
fn centered_moments_scalar_f64(
    x: &[f64],
    y: &[f64],
    n: usize,
    mean_x: f64,
    mean_y: f64,
) -> PearsonMomentsF64 {
    let mut out = PearsonMomentsF64 {
        n,
        ..PearsonMomentsF64::default()
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

/// Computes f64 Pearson moments with AVX-512F lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn pearson_moments_avx512(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f64, 8>::new(x, y) else {
        return PearsonMomentsF64::default();
    };
    let mut sum_x0 = _mm512_setzero_pd();
    let mut sum_x1 = _mm512_setzero_pd();
    let mut sum_y0 = _mm512_setzero_pd();
    let mut sum_y1 = _mm512_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 8;
        let xv = _mm512_loadu_pd(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm512_loadu_pd(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_avx512_pd(xv, yv) {
            return pearson_moments_scalar_f64(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm512_add_pd(sum_x0, xv);
            sum_y0 = _mm512_add_pd(sum_y0, yv);
        } else {
            sum_x1 = _mm512_add_pd(sum_x1, xv);
            sum_y1 = _mm512_add_pd(sum_y1, yv);
        }
    }

    let mut sum_x = _mm512_reduce_add_pd(_mm512_add_pd(sum_x0, sum_x1));
    let mut sum_y = _mm512_reduce_add_pd(_mm512_add_pd(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f64(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_avx512(parts, sum_x / n as f64, sum_y / n as f64)
}

/// Checks two AVX-512F vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
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

/// Accumulates centered f64 moments with AVX-512F lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F. The two prefixes
/// must have equal lengths divisible by 8, and the tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn centered_moments_avx512(
    parts: EqualVectorParts<'_, f64, 8>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm512_set1_pd(mean_x);
    let mean_y_vec = _mm512_set1_pd(mean_y);
    let mut xx0 = _mm512_setzero_pd();
    let mut xx1 = _mm512_setzero_pd();
    let mut yy0 = _mm512_setzero_pd();
    let mut yy1 = _mm512_setzero_pd();
    let mut xy0 = _mm512_setzero_pd();
    let mut xy1 = _mm512_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 8;
        let dx = _mm512_sub_pd(
            _mm512_loadu_pd(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm512_sub_pd(
            _mm512_loadu_pd(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm512_add_pd(xx0, _mm512_mul_pd(dx, dx));
            yy0 = _mm512_add_pd(yy0, _mm512_mul_pd(dy, dy));
            xy0 = _mm512_add_pd(xy0, _mm512_mul_pd(dx, dy));
        } else {
            xx1 = _mm512_add_pd(xx1, _mm512_mul_pd(dx, dx));
            yy1 = _mm512_add_pd(yy1, _mm512_mul_pd(dy, dy));
            xy1 = _mm512_add_pd(xy1, _mm512_mul_pd(dx, dy));
        }
    }

    let mut out = PearsonMomentsF64 {
        n: parts.len(),
        variance_x: _mm512_reduce_add_pd(_mm512_add_pd(xx0, xx1)),
        variance_y: _mm512_reduce_add_pd(_mm512_add_pd(yy0, yy1)),
        covariance: _mm512_reduce_add_pd(_mm512_add_pd(xy0, xy1)),
    };
    accumulate_centered_tail_f64(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

/// Computes f64 Pearson moments with AVX2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn pearson_moments_avx2(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f64, 4>::new(x, y) else {
        return PearsonMomentsF64::default();
    };
    let mut sum_x0 = _mm256_setzero_pd();
    let mut sum_x1 = _mm256_setzero_pd();
    let mut sum_y0 = _mm256_setzero_pd();
    let mut sum_y1 = _mm256_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let xv = _mm256_loadu_pd(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm256_loadu_pd(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_avx2_pd(xv, yv) {
            return pearson_moments_scalar_f64(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm256_add_pd(sum_x0, xv);
            sum_y0 = _mm256_add_pd(sum_y0, yv);
        } else {
            sum_x1 = _mm256_add_pd(sum_x1, xv);
            sum_y1 = _mm256_add_pd(sum_y1, yv);
        }
    }

    let mut sum_x = horizontal_sum_avx2_pd(_mm256_add_pd(sum_x0, sum_x1));
    let mut sum_y = horizontal_sum_avx2_pd(_mm256_add_pd(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f64(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_avx2(parts, sum_x / n as f64, sum_y / n as f64)
}

/// Checks two AVX2 vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
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

/// Accumulates centered f64 moments with AVX2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2. The two prefixes must
/// have equal lengths divisible by 4, and the tails must have equal lengths.
/// `EqualVectorParts::new` establishes those bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn centered_moments_avx2(
    parts: EqualVectorParts<'_, f64, 4>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm256_set1_pd(mean_x);
    let mean_y_vec = _mm256_set1_pd(mean_y);
    let mut xx0 = _mm256_setzero_pd();
    let mut xx1 = _mm256_setzero_pd();
    let mut yy0 = _mm256_setzero_pd();
    let mut yy1 = _mm256_setzero_pd();
    let mut xy0 = _mm256_setzero_pd();
    let mut xy1 = _mm256_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 4;
        let dx = _mm256_sub_pd(
            _mm256_loadu_pd(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm256_sub_pd(
            _mm256_loadu_pd(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm256_add_pd(xx0, _mm256_mul_pd(dx, dx));
            yy0 = _mm256_add_pd(yy0, _mm256_mul_pd(dy, dy));
            xy0 = _mm256_add_pd(xy0, _mm256_mul_pd(dx, dy));
        } else {
            xx1 = _mm256_add_pd(xx1, _mm256_mul_pd(dx, dx));
            yy1 = _mm256_add_pd(yy1, _mm256_mul_pd(dy, dy));
            xy1 = _mm256_add_pd(xy1, _mm256_mul_pd(dx, dy));
        }
    }

    let mut out = PearsonMomentsF64 {
        n: parts.len(),
        variance_x: horizontal_sum_avx2_pd(_mm256_add_pd(xx0, xx1)),
        variance_y: horizontal_sum_avx2_pd(_mm256_add_pd(yy0, yy1)),
        covariance: horizontal_sum_avx2_pd(_mm256_add_pd(xy0, xy1)),
    };
    accumulate_centered_tail_f64(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

/// Reduces four f64 AVX2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2_pd(values: std::arch::x86_64::__m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    std::arch::x86_64::_mm256_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.into_iter().sum()
}

/// Computes f64 Pearson moments with SSE4.2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
unsafe fn pearson_moments_sse42(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let Some(parts) = EqualVectorParts::<f64, 2>::new(x, y) else {
        return PearsonMomentsF64::default();
    };
    let mut sum_x0 = _mm_setzero_pd();
    let mut sum_x1 = _mm_setzero_pd();
    let mut sum_y0 = _mm_setzero_pd();
    let mut sum_y1 = _mm_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 2;
        let xv = _mm_loadu_pd(parts.x_prefix.as_ptr().add(offset));
        let yv = _mm_loadu_pd(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_sse_pd(xv, yv) {
            return pearson_moments_scalar_f64(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = _mm_add_pd(sum_x0, xv);
            sum_y0 = _mm_add_pd(sum_y0, yv);
        } else {
            sum_x1 = _mm_add_pd(sum_x1, xv);
            sum_y1 = _mm_add_pd(sum_y1, yv);
        }
    }

    let mut sum_x = horizontal_sum_sse_pd(_mm_add_pd(sum_x0, sum_x1));
    let mut sum_y = horizontal_sum_sse_pd(_mm_add_pd(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f64(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_sse42(parts, sum_x / n as f64, sum_y / n as f64)
}

/// Checks two SSE4.2 vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn all_finite_sse_pd(x: std::arch::x86_64::__m128d, y: std::arch::x86_64::__m128d) -> bool {
    use std::arch::x86_64::*;

    let upper = _mm_set1_pd(f64::INFINITY);
    let lower = _mm_set1_pd(f64::NEG_INFINITY);
    let x_mask = _mm_and_pd(_mm_cmplt_pd(x, upper), _mm_cmpgt_pd(x, lower));
    let y_mask = _mm_and_pd(_mm_cmplt_pd(y, upper), _mm_cmpgt_pd(y, lower));
    _mm_movemask_pd(x_mask) == 0b11 && _mm_movemask_pd(y_mask) == 0b11
}

/// Accumulates centered f64 moments with SSE4.2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2. The two prefixes
/// must have equal lengths divisible by 2, and the tails must have equal
/// lengths. `EqualVectorParts::new` establishes those bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
unsafe fn centered_moments_sse42(
    parts: EqualVectorParts<'_, f64, 2>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonMomentsF64 {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm_set1_pd(mean_x);
    let mean_y_vec = _mm_set1_pd(mean_y);
    let mut xx0 = _mm_setzero_pd();
    let mut xx1 = _mm_setzero_pd();
    let mut yy0 = _mm_setzero_pd();
    let mut yy1 = _mm_setzero_pd();
    let mut xy0 = _mm_setzero_pd();
    let mut xy1 = _mm_setzero_pd();
    for chunk in 0..parts.chunks() {
        let offset = chunk * 2;
        let dx = _mm_sub_pd(
            _mm_loadu_pd(parts.x_prefix.as_ptr().add(offset)),
            mean_x_vec,
        );
        let dy = _mm_sub_pd(
            _mm_loadu_pd(parts.y_prefix.as_ptr().add(offset)),
            mean_y_vec,
        );
        if chunk & 1 == 0 {
            xx0 = _mm_add_pd(xx0, _mm_mul_pd(dx, dx));
            yy0 = _mm_add_pd(yy0, _mm_mul_pd(dy, dy));
            xy0 = _mm_add_pd(xy0, _mm_mul_pd(dx, dy));
        } else {
            xx1 = _mm_add_pd(xx1, _mm_mul_pd(dx, dx));
            yy1 = _mm_add_pd(yy1, _mm_mul_pd(dy, dy));
            xy1 = _mm_add_pd(xy1, _mm_mul_pd(dx, dy));
        }
    }

    let mut out = PearsonMomentsF64 {
        n: parts.len(),
        variance_x: horizontal_sum_sse_pd(_mm_add_pd(xx0, xx1)),
        variance_y: horizontal_sum_sse_pd(_mm_add_pd(yy0, yy1)),
        covariance: horizontal_sum_sse_pd(_mm_add_pd(xy0, xy1)),
    };
    accumulate_centered_tail_f64(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

/// Reduces two f64 SSE4.2 lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn horizontal_sum_sse_pd(values: std::arch::x86_64::__m128d) -> f64 {
    let mut lanes = [0.0f64; 2];
    std::arch::x86_64::_mm_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.into_iter().sum()
}

/// Computes f64 Pearson moments with NEON lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. Input shape and
/// vector-load bounds are validated before any raw pointer is dereferenced.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn pearson_moments_neon(x: &[f64], y: &[f64]) -> PearsonMomentsF64 {
    use std::arch::aarch64::*;

    let Some(parts) = EqualVectorParts::<f64, 2>::new(x, y) else {
        return PearsonMomentsF64::default();
    };
    let mut sum_x0 = vdupq_n_f64(0.0);
    let mut sum_x1 = vdupq_n_f64(0.0);
    let mut sum_y0 = vdupq_n_f64(0.0);
    let mut sum_y1 = vdupq_n_f64(0.0);
    for chunk in 0..parts.chunks() {
        let offset = chunk * 2;
        let xv = vld1q_f64(parts.x_prefix.as_ptr().add(offset));
        let yv = vld1q_f64(parts.y_prefix.as_ptr().add(offset));
        if !all_finite_neon_f64(xv, yv) {
            return pearson_moments_scalar_f64(x, y);
        }
        if chunk & 1 == 0 {
            sum_x0 = vaddq_f64(sum_x0, xv);
            sum_y0 = vaddq_f64(sum_y0, yv);
        } else {
            sum_x1 = vaddq_f64(sum_x1, xv);
            sum_y1 = vaddq_f64(sum_y1, yv);
        }
    }

    let mut sum_x = vaddvq_f64(vaddq_f64(sum_x0, sum_x1));
    let mut sum_y = vaddvq_f64(vaddq_f64(sum_y0, sum_y1));
    for (&x_value, &y_value) in parts.x_tail.iter().zip(parts.y_tail) {
        if !x_value.is_finite() || !y_value.is_finite() {
            return pearson_moments_scalar_f64(x, y);
        }
        sum_x += x_value;
        sum_y += y_value;
    }
    let n = parts.len();
    centered_moments_neon(parts, sum_x / n as f64, sum_y / n as f64)
}

/// Checks two NEON vectors for finite values.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
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

/// Accumulates centered f64 moments with NEON lanes.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. The two prefixes must
/// have equal lengths divisible by 2, and the tails must have equal lengths.
/// `EqualVectorParts::new` establishes those bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn centered_moments_neon(
    parts: EqualVectorParts<'_, f64, 2>,
    mean_x: f64,
    mean_y: f64,
) -> PearsonMomentsF64 {
    use std::arch::aarch64::*;

    let mean_x_vec = vdupq_n_f64(mean_x);
    let mean_y_vec = vdupq_n_f64(mean_y);
    let mut xx0 = vdupq_n_f64(0.0);
    let mut xx1 = vdupq_n_f64(0.0);
    let mut yy0 = vdupq_n_f64(0.0);
    let mut yy1 = vdupq_n_f64(0.0);
    let mut xy0 = vdupq_n_f64(0.0);
    let mut xy1 = vdupq_n_f64(0.0);
    for chunk in 0..parts.chunks() {
        let offset = chunk * 2;
        let dx = vsubq_f64(vld1q_f64(parts.x_prefix.as_ptr().add(offset)), mean_x_vec);
        let dy = vsubq_f64(vld1q_f64(parts.y_prefix.as_ptr().add(offset)), mean_y_vec);
        if chunk & 1 == 0 {
            xx0 = vaddq_f64(xx0, vmulq_f64(dx, dx));
            yy0 = vaddq_f64(yy0, vmulq_f64(dy, dy));
            xy0 = vaddq_f64(xy0, vmulq_f64(dx, dy));
        } else {
            xx1 = vaddq_f64(xx1, vmulq_f64(dx, dx));
            yy1 = vaddq_f64(yy1, vmulq_f64(dy, dy));
            xy1 = vaddq_f64(xy1, vmulq_f64(dx, dy));
        }
    }

    let mut out = PearsonMomentsF64 {
        n: parts.len(),
        variance_x: vaddvq_f64(vaddq_f64(xx0, xx1)),
        variance_y: vaddvq_f64(vaddq_f64(yy0, yy1)),
        covariance: vaddvq_f64(vaddq_f64(xy0, xy1)),
    };
    accumulate_centered_tail_f64(&mut out, parts.x_tail, parts.y_tail, mean_x, mean_y);
    out
}

#[inline]
fn accumulate_centered_tail_f64(
    out: &mut PearsonMomentsF64,
    x: &[f64],
    y: &[f64],
    mean_x: f64,
    mean_y: f64,
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

    use super::super::covariance_common::{finalize_r2_f64, FP64_SIMD_REGROUPING_TOLERANCE};
    use super::*;

    fn dataset(len: usize) -> (Vec<f64>, Vec<f64>) {
        let x = (0..len)
            .map(|index| {
                let value = index as f64 + 1.0;
                value.sin() * 0.25 + value * 0.001
            })
            .collect::<Vec<_>>();
        let y = (0..len)
            .map(|index| {
                let value = index as f64 + 3.0;
                value.cos() * 0.15 + value * 0.002
            })
            .collect::<Vec<_>>();
        (x, y)
    }

    fn assert_close(left: PearsonMomentsF64, right: PearsonMomentsF64) {
        assert_eq!(left.n, right.n);
        let scale_x = left.variance_x.abs().max(right.variance_x.abs()).max(1.0);
        let scale_y = left.variance_y.abs().max(right.variance_y.abs()).max(1.0);
        let scale_xy = left.covariance.abs().max(right.covariance.abs()).max(1.0);
        assert!(
            (left.variance_x - right.variance_x).abs() <= FP64_SIMD_REGROUPING_TOLERANCE * scale_x
        );
        assert!(
            (left.variance_y - right.variance_y).abs() <= FP64_SIMD_REGROUPING_TOLERANCE * scale_y
        );
        assert!(
            (left.covariance - right.covariance).abs() <= FP64_SIMD_REGROUPING_TOLERANCE * scale_xy
        );
        assert!((left.finish() - right.finish()).abs() <= FP64_SIMD_REGROUPING_TOLERANCE);
    }

    #[test]
    fn dispatched_binary64_matches_scalar_across_vector_tails() {
        for len in 1..=65 {
            let (x, y) = dataset(len);
            assert_close(
                pearson_moments_scalar_f64(&x, &y),
                pearson_moments_f64(&x, &y),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_x86_binary64_rung_matches_scalar() {
        let (x, y) = dataset(4099);
        let scalar = pearson_moments_scalar_f64(&x, &y);
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
        let (base_x, base_y) = dataset(67);
        let cases = [
            (true, 0, f64::NAN),
            (false, 17, f64::INFINITY),
            (true, 66, f64::NEG_INFINITY),
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
                let scalar = pearson_moments_scalar_f64(&x, &y);
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
        // SAFETY: Calls are guarded by their exact runtime feature check. The
        // invalid shape verifies validation occurs before any raw load.
        unsafe {
            if std::is_x86_feature_detected!("sse4.2") {
                assert_eq!(pearson_moments_sse42(&x, &y), PearsonMomentsF64::default());
            }
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(pearson_moments_avx2(&x, &y), PearsonMomentsF64::default());
            }
            if std::is_x86_feature_detected!("avx512f") {
                assert_eq!(pearson_moments_avx512(&x, &y), PearsonMomentsF64::default());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_binary64_rung_matches_scalar() {
        let (mut x, y) = dataset(4099);
        // SAFETY: NEON is an AArch64 baseline feature and the inputs are equal,
        // initialized slices. The second call verifies non-finite fallback.
        unsafe {
            assert_close(
                pearson_moments_scalar_f64(&x, &y),
                pearson_moments_neon(&x, &y),
            );
            x[4098] = f64::NAN;
            assert_eq!(
                pearson_moments_scalar_f64(&x, &y),
                pearson_moments_neon(&x, &y)
            );
            assert_eq!(
                pearson_moments_neon(&[1.0, 2.0], &[1.0]),
                PearsonMomentsF64::default()
            );
        }
    }

    #[test]
    fn nonfinite_filter_zero_variance_and_overflow_match_scalar_contract() {
        let x = [1.0, 2.0, f64::NAN, 4.0, f64::INFINITY, 6.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, f64::NEG_INFINITY];
        assert_eq!(
            pearson_moments_f64(&x, &y),
            pearson_moments_scalar_f64(&x, &y)
        );
        assert_eq!(pearson_corr_f64(&[2.0; 64], &[1.0; 64]), 0.0);
        assert_eq!(pearson_corr_f64(&[0.1; 259], &[0.3; 259]), 0.0);
        assert_eq!(pearson_corr_f64(&[], &[]), 0.0);
        assert_eq!(pearson_corr_f64(&[1.0], &[]), 0.0);

        let overflow_x = [1.0e200, -1.0e200, 1.0e200, -1.0e200];
        let overflow_y = [1.0, -1.0, -1.0, 1.0];
        assert!(pearson_corr_f64(&overflow_x, &overflow_y).is_nan());
        assert!(finalize_r2_f64(pearson_corr_f64(&overflow_x, &overflow_y)).is_nan());
    }

    #[test]
    fn constant_axis_does_not_mask_symmetric_binary64_overflow() {
        let varying = (0..512)
            .map(|index| if index % 2 == 0 { 1.0e200 } else { -1.0e200 })
            .collect::<Vec<_>>();
        let constant = vec![2.0f64; varying.len()];
        assert!(pearson_corr_f64(&constant, &varying).is_nan());
        assert!(pearson_corr_f64(&varying, &constant).is_nan());

        let finite = (0..varying.len())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        assert_eq!(pearson_corr_f64(&constant, &finite), 0.0);
        assert_eq!(pearson_corr_f64(&finite, &constant), 0.0);
    }

    #[test]
    fn binary64_dynamic_range_survives_values_that_collapse_in_binary32() {
        let x = (0..4099)
            .map(|index| 1.0f64 + (index % 97) as f64 * 5.0e-10)
            .collect::<Vec<_>>();
        let y = (0..4099)
            .map(|index| {
                let centered = (index % 97) as f64 - 48.0;
                centered * 0.25 + ((index * 13 % 17) as f64 - 8.0) * 0.01
            })
            .collect::<Vec<_>>();
        assert!(x
            .iter()
            .map(|&value| value as f32)
            .all(|value| value == 1.0));

        let scalar = pearson_moments_scalar_f64(&x, &y).finish();
        let dispatched = pearson_corr_f64(&x, &y);
        assert!(scalar.abs() > 0.9);
        assert!((scalar - dispatched).abs() <= FP64_SIMD_REGROUPING_TOLERANCE);
        assert!((finalize_r2_f64(dispatched) - dispatched * dispatched).abs() <= f64::EPSILON);
    }

    #[test]
    #[ignore = "supplemental single-core leaf diagnostic; run with --release --ignored --nocapture"]
    fn binary64_pearson_release_leaf_diagnostic() {
        fn scalar(x: &[f64], y: &[f64]) -> f64 {
            pearson_moments_scalar_f64(x, y).finish()
        }

        fn measure(kernel: fn(&[f64], &[f64]) -> f64, x: &[f64], y: &[f64], reps: u32) -> u128 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(kernel(black_box(x), black_box(y)));
            }
            started.elapsed().as_nanos() / u128::from(reps)
        }

        let (x, y) = dataset(1 << 20);
        for _ in 0..10 {
            black_box(scalar(black_box(&x), black_box(&y)));
            black_box(pearson_corr_f64(black_box(&x), black_box(&y)));
        }
        let blocks = 6;
        let repetitions_per_block = 10;
        let mut scalar_samples = Vec::with_capacity(blocks);
        let mut simd_samples = Vec::with_capacity(blocks);
        for block in 0..blocks {
            if block & 1 == 0 {
                scalar_samples.push(measure(scalar, &x, &y, repetitions_per_block));
                simd_samples.push(measure(pearson_corr_f64, &x, &y, repetitions_per_block));
            } else {
                simd_samples.push(measure(pearson_corr_f64, &x, &y, repetitions_per_block));
                scalar_samples.push(measure(scalar, &x, &y, repetitions_per_block));
            }
        }
        scalar_samples.sort_unstable();
        simd_samples.sort_unstable();
        let scalar_ns = scalar_samples[blocks / 2];
        let simd_ns = simd_samples[blocks / 2];
        eprintln!(
            "supplemental_fp64_leaf rows={} blocks={} reps_per_block={} scalar_median_ns={} simd_median_ns={} speedup={:.3} scalar_samples_ns={:?} simd_samples_ns={:?}",
            x.len(),
            blocks,
            repetitions_per_block,
            scalar_ns,
            simd_ns,
            scalar_ns as f64 / simd_ns as f64,
            scalar_samples,
            simd_samples,
        );
    }
}
