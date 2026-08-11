//! Shared non-floating scaffolding for profile-specialized covariance kernels.
//!
//! The typed SIMD rungs stay in their profile modules so their intrinsic width,
//! reduction order, and target-feature code generation remain explicit.  This
//! module owns only slice partitioning and the common binary64 result policy.

#[cfg(test)]
pub(crate) const FP64_SIMD_REGROUPING_TOLERANCE: f64 = 2.0e-12;

pub(crate) struct EqualVectorParts<'a, T, const LANES: usize> {
    pub(crate) x_prefix: &'a [T],
    pub(crate) y_prefix: &'a [T],
    pub(crate) x_tail: &'a [T],
    pub(crate) y_tail: &'a [T],
}

impl<'a, T, const LANES: usize> EqualVectorParts<'a, T, LANES> {
    #[inline]
    pub(crate) fn new(x: &'a [T], y: &'a [T]) -> Option<Self> {
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
    pub(crate) fn chunks(&self) -> usize {
        self.x_prefix.len() / LANES
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.x_prefix.len() + self.x_tail.len()
    }
}

/// Finalizes a non-empty binary64 covariance reduction.
///
/// SIMD paths may regroup independent f64 accumulator lanes, but this policy
/// keeps zero-variance and arithmetic-failure classifications exact.
#[inline]
pub(crate) fn finalize_correlation_f64(variance_x: f64, variance_y: f64, covariance: f64) -> f64 {
    if !variance_x.is_finite() || !variance_y.is_finite() || !covariance.is_finite() {
        return f64::NAN;
    }
    if variance_x == 0.0 || variance_y == 0.0 {
        return 0.0;
    }
    if variance_x < 0.0 || variance_y < 0.0 {
        return f64::NAN;
    }
    let denominator = (variance_x * variance_y).sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return f64::NAN;
    }
    let correlation = covariance / denominator;
    if correlation.is_finite() {
        correlation.clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

#[inline]
pub(crate) fn finalize_r2_f64(correlation: f64) -> f64 {
    if correlation.is_finite() {
        (correlation * correlation).clamp(0.0, 1.0)
    } else {
        f64::NAN
    }
}
