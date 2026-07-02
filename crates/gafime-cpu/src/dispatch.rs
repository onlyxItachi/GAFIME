#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaLevel {
    Scalar,
    Sse42,
    Avx2,
    Avx512,
    Neon,
}

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
        let denom = (self.sxx * self.syy).max(0.0).sqrt();
        if denom <= 0.0 {
            0.0
        } else {
            (self.sxy / denom).clamp(-1.0, 1.0) as f32
        }
    }
}

pub fn detect_isa() -> IsaLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return IsaLevel::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return IsaLevel::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return IsaLevel::Sse42;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return IsaLevel::Neon;
    }

    IsaLevel::Scalar
}

pub fn finite_dispatch_isa() -> IsaLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return IsaLevel::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return IsaLevel::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return IsaLevel::Sse42;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return IsaLevel::Neon;
    }

    IsaLevel::Scalar
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
    if !all_pairs_finite(x, y) {
        return pearson_sums_scalar(x, y);
    }
    pearson_sums_finite(x, y)
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

fn pearson_sums_finite(x: &[f32], y: &[f32]) -> PearsonSums {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if std::is_x86_feature_detected!("avx512f") {
            return pearson_sums_avx512(x, y);
        }
        if std::is_x86_feature_detected!("avx2") {
            return pearson_sums_avx2(x, y);
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return pearson_sums_sse42(x, y);
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return pearson_sums_neon(x, y);
    }

    pearson_sums_scalar(x, y)
}

fn all_pairs_finite(x: &[f32], y: &[f32]) -> bool {
    x.iter()
        .zip(y)
        .all(|(&x_value, &y_value)| x_value.is_finite() && y_value.is_finite())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pearson_sums_avx512(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    // 8 fp32 lanes widened to f64 accumulators per iteration (the widest x86 rung).
    let chunks = x.len() / 8;
    let mut sx = _mm512_setzero_pd();
    let mut sy = _mm512_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let xv = _mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(offset)));
        let yv = _mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(offset)));
        sx = _mm512_add_pd(sx, xv);
        sy = _mm512_add_pd(sy, yv);
    }

    let mut n = chunks * 8;
    let mut sum_x = _mm512_reduce_add_pd(sx);
    let mut sum_y = _mm512_reduce_add_pd(sy);
    for (&x_value, &y_value) in x[chunks * 8..].iter().zip(&y[chunks * 8..]) {
        n += 1;
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_avx512(x, y, n, sum_x / n as f64, sum_y / n as f64, chunks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn centered_sums_avx512(
    x: &[f32],
    y: &[f32],
    n: usize,
    mean_x: f64,
    mean_y: f64,
    chunks: usize,
) -> PearsonSums {
    use std::arch::x86_64::*;

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
        let dx0 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(o0))), mean_x_vec);
        let dy0 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(o0))), mean_y_vec);
        sxx0 = _mm512_add_pd(sxx0, _mm512_mul_pd(dx0, dx0));
        syy0 = _mm512_add_pd(syy0, _mm512_mul_pd(dy0, dy0));
        sxy0 = _mm512_add_pd(sxy0, _mm512_mul_pd(dx0, dy0));
        let o1 = (base + 1) * 8;
        let dx1 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(o1))), mean_x_vec);
        let dy1 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(o1))), mean_y_vec);
        sxx1 = _mm512_add_pd(sxx1, _mm512_mul_pd(dx1, dx1));
        syy1 = _mm512_add_pd(syy1, _mm512_mul_pd(dy1, dy1));
        sxy1 = _mm512_add_pd(sxy1, _mm512_mul_pd(dx1, dy1));
        let o2 = (base + 2) * 8;
        let dx2 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(o2))), mean_x_vec);
        let dy2 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(o2))), mean_y_vec);
        sxx2 = _mm512_add_pd(sxx2, _mm512_mul_pd(dx2, dx2));
        syy2 = _mm512_add_pd(syy2, _mm512_mul_pd(dy2, dy2));
        sxy2 = _mm512_add_pd(sxy2, _mm512_mul_pd(dx2, dy2));
        let o3 = (base + 3) * 8;
        let dx3 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(o3))), mean_x_vec);
        let dy3 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(o3))), mean_y_vec);
        sxx3 = _mm512_add_pd(sxx3, _mm512_mul_pd(dx3, dx3));
        syy3 = _mm512_add_pd(syy3, _mm512_mul_pd(dy3, dy3));
        sxy3 = _mm512_add_pd(sxy3, _mm512_mul_pd(dx3, dy3));
    }
    for c in (quads * 4)..chunks {
        let o = c * 8;
        let dx = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(o))), mean_x_vec);
        let dy = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(y.as_ptr().add(o))), mean_y_vec);
        sxx0 = _mm512_add_pd(sxx0, _mm512_mul_pd(dx, dx));
        syy0 = _mm512_add_pd(syy0, _mm512_mul_pd(dy, dy));
        sxy0 = _mm512_add_pd(sxy0, _mm512_mul_pd(dx, dy));
    }

    let sxx = _mm512_add_pd(_mm512_add_pd(sxx0, sxx1), _mm512_add_pd(sxx2, sxx3));
    let syy = _mm512_add_pd(_mm512_add_pd(syy0, syy1), _mm512_add_pd(syy2, syy3));
    let sxy = _mm512_add_pd(_mm512_add_pd(sxy0, sxy1), _mm512_add_pd(sxy2, sxy3));

    let mut out = PearsonSums {
        n,
        sx: mean_x,
        sy: mean_y,
        sxx: _mm512_reduce_add_pd(sxx),
        syy: _mm512_reduce_add_pd(syy),
        sxy: _mm512_reduce_add_pd(sxy),
    };
    accumulate_centered_tail(&mut out, x, y, chunks * 8);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pearson_sums_avx2(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let chunks = x.len() / 4;
    let mut sx = _mm256_setzero_pd();
    let mut sy = _mm256_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let xv = _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(offset)));
        let yv = _mm256_cvtps_pd(_mm_loadu_ps(y.as_ptr().add(offset)));
        sx = _mm256_add_pd(sx, xv);
        sy = _mm256_add_pd(sy, yv);
    }

    let mut n = chunks * 4;
    let mut sum_x = horizontal_sum_avx2_pd(sx);
    let mut sum_y = horizontal_sum_avx2_pd(sy);
    for (&x_value, &y_value) in x[chunks * 4..].iter().zip(&y[chunks * 4..]) {
        n += 1;
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_avx2(x, y, n, sum_x / n as f64, sum_y / n as f64, chunks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn centered_sums_avx2(
    x: &[f32],
    y: &[f32],
    n: usize,
    mean_x: f64,
    mean_y: f64,
    chunks: usize,
) -> PearsonSums {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm256_set1_pd(mean_x);
    let mean_y_vec = _mm256_set1_pd(mean_y);
    let mut sxx = _mm256_setzero_pd();
    let mut syy = _mm256_setzero_pd();
    let mut sxy = _mm256_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let dx = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(offset))),
            mean_x_vec,
        );
        let dy = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm_loadu_ps(y.as_ptr().add(offset))),
            mean_y_vec,
        );
        sxx = _mm256_add_pd(sxx, _mm256_mul_pd(dx, dx));
        syy = _mm256_add_pd(syy, _mm256_mul_pd(dy, dy));
        sxy = _mm256_add_pd(sxy, _mm256_mul_pd(dx, dy));
    }

    let mut out = PearsonSums {
        n,
        sx: mean_x,
        sy: mean_y,
        sxx: horizontal_sum_avx2_pd(sxx),
        syy: horizontal_sum_avx2_pd(syy),
        sxy: horizontal_sum_avx2_pd(sxy),
    };
    accumulate_centered_tail(&mut out, x, y, chunks * 4);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2_pd(values: std::arch::x86_64::__m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    std::arch::x86_64::_mm256_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn pearson_sums_sse42(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let chunks = x.len() / 2;
    let mut sx = _mm_setzero_pd();
    let mut sy = _mm_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = _mm_set_pd(
            *x.get_unchecked(offset + 1) as f64,
            *x.get_unchecked(offset) as f64,
        );
        let yv = _mm_set_pd(
            *y.get_unchecked(offset + 1) as f64,
            *y.get_unchecked(offset) as f64,
        );
        sx = _mm_add_pd(sx, xv);
        sy = _mm_add_pd(sy, yv);
    }

    let mut n = chunks * 2;
    let mut sum_x = horizontal_sum_sse_pd(sx);
    let mut sum_y = horizontal_sum_sse_pd(sy);
    for (&x_value, &y_value) in x[chunks * 2..].iter().zip(&y[chunks * 2..]) {
        n += 1;
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_sse42(x, y, n, sum_x / n as f64, sum_y / n as f64, chunks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn centered_sums_sse42(
    x: &[f32],
    y: &[f32],
    n: usize,
    mean_x: f64,
    mean_y: f64,
    chunks: usize,
) -> PearsonSums {
    use std::arch::x86_64::*;

    let mean_x_vec = _mm_set1_pd(mean_x);
    let mean_y_vec = _mm_set1_pd(mean_y);
    let mut sxx = _mm_setzero_pd();
    let mut syy = _mm_setzero_pd();
    let mut sxy = _mm_setzero_pd();
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = _mm_set_pd(
            *x.get_unchecked(offset + 1) as f64,
            *x.get_unchecked(offset) as f64,
        );
        let yv = _mm_set_pd(
            *y.get_unchecked(offset + 1) as f64,
            *y.get_unchecked(offset) as f64,
        );
        let dx = _mm_sub_pd(xv, mean_x_vec);
        let dy = _mm_sub_pd(yv, mean_y_vec);
        sxx = _mm_add_pd(sxx, _mm_mul_pd(dx, dx));
        syy = _mm_add_pd(syy, _mm_mul_pd(dy, dy));
        sxy = _mm_add_pd(sxy, _mm_mul_pd(dx, dy));
    }

    let mut out = PearsonSums {
        n,
        sx: mean_x,
        sy: mean_y,
        sxx: horizontal_sum_sse_pd(sxx),
        syy: horizontal_sum_sse_pd(syy),
        sxy: horizontal_sum_sse_pd(sxy),
    };
    accumulate_centered_tail(&mut out, x, y, chunks * 2);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn horizontal_sum_sse_pd(values: std::arch::x86_64::__m128d) -> f64 {
    let mut lanes = [0.0f64; 2];
    std::arch::x86_64::_mm_storeu_pd(lanes.as_mut_ptr(), values);
    lanes.iter().sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pearson_sums_neon(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::aarch64::*;

    let chunks = x.len() / 2;
    let mut sx = vdupq_n_f64(0.0);
    let mut sy = vdupq_n_f64(0.0);
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let xv = f64x2_from_f32_pair(x.as_ptr().add(offset));
        let yv = f64x2_from_f32_pair(y.as_ptr().add(offset));
        sx = vaddq_f64(sx, xv);
        sy = vaddq_f64(sy, yv);
    }

    let mut n = chunks * 2;
    let mut sum_x = vaddvq_f64(sx);
    let mut sum_y = vaddvq_f64(sy);
    for (&x_value, &y_value) in x[chunks * 2..].iter().zip(&y[chunks * 2..]) {
        n += 1;
        sum_x += x_value as f64;
        sum_y += y_value as f64;
    }
    centered_sums_neon(x, y, n, sum_x / n as f64, sum_y / n as f64, chunks)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn centered_sums_neon(
    x: &[f32],
    y: &[f32],
    n: usize,
    mean_x: f64,
    mean_y: f64,
    chunks: usize,
) -> PearsonSums {
    use std::arch::aarch64::*;

    let mean_x_vec = vdupq_n_f64(mean_x);
    let mean_y_vec = vdupq_n_f64(mean_y);
    let mut sxx = vdupq_n_f64(0.0);
    let mut syy = vdupq_n_f64(0.0);
    let mut sxy = vdupq_n_f64(0.0);
    for chunk in 0..chunks {
        let offset = chunk * 2;
        let dx = vsubq_f64(f64x2_from_f32_pair(x.as_ptr().add(offset)), mean_x_vec);
        let dy = vsubq_f64(f64x2_from_f32_pair(y.as_ptr().add(offset)), mean_y_vec);
        sxx = vaddq_f64(sxx, vmulq_f64(dx, dx));
        syy = vaddq_f64(syy, vmulq_f64(dy, dy));
        sxy = vaddq_f64(sxy, vmulq_f64(dx, dy));
    }

    let mut out = PearsonSums {
        n,
        sx: mean_x,
        sy: mean_y,
        sxx: vaddvq_f64(sxx),
        syy: vaddvq_f64(syy),
        sxy: vaddvq_f64(sxy),
    };
    accumulate_centered_tail(&mut out, x, y, chunks * 2);
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f64x2_from_f32_pair(ptr: *const f32) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::*;

    let mut out = vdupq_n_f64(0.0);
    out = vsetq_lane_f64(*ptr as f64, out, 0);
    vsetq_lane_f64(*ptr.add(1) as f64, out, 1)
}

fn accumulate_centered_tail(out: &mut PearsonSums, x: &[f32], y: &[f32], offset: usize) {
    for (&x_value, &y_value) in x[offset..].iter().zip(&y[offset..]) {
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

    fn dataset() -> (Vec<f32>, Vec<f32>) {
        let x = (0..257)
            .map(|idx| {
                let value = idx as f32 + 1.0;
                value.sin() * 0.25 + value * 0.01
            })
            .collect::<Vec<_>>();
        let y = (0..257)
            .map(|idx| {
                let value = idx as f32 + 3.0;
                value.cos() * 0.15 + value * 0.02
            })
            .collect::<Vec<_>>();
        (x, y)
    }

    #[test]
    fn dispatched_pearson_matches_scalar_reduction() {
        let (x, y) = dataset();
        let scalar = pearson_sums_scalar(&x, &y).pearson();
        let dispatched = pearson_corr(&x, &y);
        assert!((scalar - dispatched).abs() <= 5.0e-5);
    }

    #[test]
    fn finite_dispatch_reports_detected_simd_when_available() {
        let isa = finite_dispatch_isa();
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                assert_eq!(isa, IsaLevel::Avx512);
            } else if std::is_x86_feature_detected!("avx2") {
                assert_eq!(isa, IsaLevel::Avx2);
            } else if std::is_x86_feature_detected!("sse4.2") {
                assert_eq!(isa, IsaLevel::Sse42);
            }
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(isa, IsaLevel::Neon);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_paths_match_scalar_centered_covariance() {
        let (x, y) = dataset();
        let scalar = pearson_sums_scalar(&x, &y);
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
