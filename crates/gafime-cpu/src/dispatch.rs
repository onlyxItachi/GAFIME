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
    pearson_sums_scalar(x, y)
}

fn all_pairs_finite(x: &[f32], y: &[f32]) -> bool {
    x.iter()
        .zip(y)
        .all(|(&x_value, &y_value)| x_value.is_finite() && y_value.is_finite())
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
}
