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
        let n = self.n as f64;
        let numerator = n * self.sxy - self.sx * self.sy;
        let denom_x = n * self.sxx - self.sx * self.sx;
        let denom_y = n * self.syy - self.sy * self.sy;
        let denom = (denom_x * denom_y).max(0.0).sqrt();
        if denom <= 0.0 {
            0.0
        } else {
            (numerator / denom) as f32
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

pub fn pearson_corr(x: &[f32], y: &[f32]) -> f32 {
    pearson_sums(x, y).pearson()
}

pub fn r2_score(x: &[f32], y: &[f32]) -> f32 {
    let corr = pearson_corr(x, y);
    corr * corr
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
    let mut sums = PearsonSums::default();
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            let xv = x_value as f64;
            let yv = y_value as f64;
            sums.n += 1;
            sums.sx += xv;
            sums.sy += yv;
            sums.sxx += xv * xv;
            sums.syy += yv * yv;
            sums.sxy += xv * yv;
        }
    }
    sums
}

fn pearson_sums_finite(x: &[f32], y: &[f32]) -> PearsonSums {
    #[cfg(target_arch = "x86_64")]
    unsafe {
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
#[target_feature(enable = "avx2")]
unsafe fn pearson_sums_avx2(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let mut sx = _mm256_setzero_ps();
    let mut sy = _mm256_setzero_ps();
    let mut sxx = _mm256_setzero_ps();
    let mut syy = _mm256_setzero_ps();
    let mut sxy = _mm256_setzero_ps();
    let chunks = x.len() / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let xv = _mm256_loadu_ps(x.as_ptr().add(offset));
        let yv = _mm256_loadu_ps(y.as_ptr().add(offset));
        sx = _mm256_add_ps(sx, xv);
        sy = _mm256_add_ps(sy, yv);
        sxx = _mm256_add_ps(sxx, _mm256_mul_ps(xv, xv));
        syy = _mm256_add_ps(syy, _mm256_mul_ps(yv, yv));
        sxy = _mm256_add_ps(sxy, _mm256_mul_ps(xv, yv));
    }
    let mut sums = PearsonSums {
        n: chunks * 8,
        sx: horizontal_sum_avx2(sx),
        sy: horizontal_sum_avx2(sy),
        sxx: horizontal_sum_avx2(sxx),
        syy: horizontal_sum_avx2(syy),
        sxy: horizontal_sum_avx2(sxy),
    };
    accumulate_tail(&mut sums, x, y, chunks * 8);
    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(values: std::arch::x86_64::__m256) -> f64 {
    let mut lanes = [0.0f32; 8];
    std::arch::x86_64::_mm256_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.iter().map(|&value| value as f64).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn pearson_sums_sse42(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::x86_64::*;

    let mut sx = _mm_setzero_ps();
    let mut sy = _mm_setzero_ps();
    let mut sxx = _mm_setzero_ps();
    let mut syy = _mm_setzero_ps();
    let mut sxy = _mm_setzero_ps();
    let chunks = x.len() / 4;
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let xv = _mm_loadu_ps(x.as_ptr().add(offset));
        let yv = _mm_loadu_ps(y.as_ptr().add(offset));
        sx = _mm_add_ps(sx, xv);
        sy = _mm_add_ps(sy, yv);
        sxx = _mm_add_ps(sxx, _mm_mul_ps(xv, xv));
        syy = _mm_add_ps(syy, _mm_mul_ps(yv, yv));
        sxy = _mm_add_ps(sxy, _mm_mul_ps(xv, yv));
    }
    let mut sums = PearsonSums {
        n: chunks * 4,
        sx: horizontal_sum_sse(sx),
        sy: horizontal_sum_sse(sy),
        sxx: horizontal_sum_sse(sxx),
        syy: horizontal_sum_sse(syy),
        sxy: horizontal_sum_sse(sxy),
    };
    accumulate_tail(&mut sums, x, y, chunks * 4);
    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn horizontal_sum_sse(values: std::arch::x86_64::__m128) -> f64 {
    let mut lanes = [0.0f32; 4];
    std::arch::x86_64::_mm_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.iter().map(|&value| value as f64).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pearson_sums_neon(x: &[f32], y: &[f32]) -> PearsonSums {
    use std::arch::aarch64::*;

    let mut sx = vdupq_n_f32(0.0);
    let mut sy = vdupq_n_f32(0.0);
    let mut sxx = vdupq_n_f32(0.0);
    let mut syy = vdupq_n_f32(0.0);
    let mut sxy = vdupq_n_f32(0.0);
    let chunks = x.len() / 4;
    for chunk in 0..chunks {
        let offset = chunk * 4;
        let xv = vld1q_f32(x.as_ptr().add(offset));
        let yv = vld1q_f32(y.as_ptr().add(offset));
        sx = vaddq_f32(sx, xv);
        sy = vaddq_f32(sy, yv);
        sxx = vaddq_f32(sxx, vmulq_f32(xv, xv));
        syy = vaddq_f32(syy, vmulq_f32(yv, yv));
        sxy = vaddq_f32(sxy, vmulq_f32(xv, yv));
    }
    let mut sums = PearsonSums {
        n: chunks * 4,
        sx: vaddvq_f32(sx) as f64,
        sy: vaddvq_f32(sy) as f64,
        sxx: vaddvq_f32(sxx) as f64,
        syy: vaddvq_f32(syy) as f64,
        sxy: vaddvq_f32(sxy) as f64,
    };
    accumulate_tail(&mut sums, x, y, chunks * 4);
    sums
}

fn accumulate_tail(sums: &mut PearsonSums, x: &[f32], y: &[f32], offset: usize) {
    for (&x_value, &y_value) in x[offset..].iter().zip(&y[offset..]) {
        let xv = x_value as f64;
        let yv = y_value as f64;
        sums.n += 1;
        sums.sx += xv;
        sums.sy += yv;
        sums.sxx += xv * xv;
        sums.syy += yv * yv;
        sums.sxy += xv * yv;
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
    fn r2_uses_dispatched_pearson() {
        let (x, y) = dataset();
        let corr = pearson_corr(&x, &y);
        assert!((r2_score(&x, &y) - corr * corr).abs() <= f32::EPSILON);
    }

    #[test]
    fn non_finite_inputs_fall_back_to_scalar_filtering() {
        let x = [1.0, 2.0, f32::NAN, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(pearson_sums(&x, &y), pearson_sums_scalar(&x, &y));
    }
}
