//! Decision-path (GBDT-method) split finding for the `decision_path` family.
//!
//! Core primitive: the CART/GBDT variance-reduction best-split of a feature vs
//! the target (or residual). A decision_path candidate is a conjunction of such
//! splits (a root→leaf path); its materialized feature is the membership
//! indicator of that region, scored by the continuous engine. depth-k recursion
//! and residual boosting build on `best_variance_split`.

/// A single threshold split and its variance-reduction gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    pub threshold: f32,
    pub gain: f32,
}

/// Find the threshold on `feature` that maximizes variance reduction of `y`.
/// O(n log n): sort by feature, sweep boundaries maintaining running
/// left/right (sum, sum²) for incremental child variances. Returns `None` for a
/// constant feature, fewer than 2 finite pairs, or zero parent variance.
pub fn best_variance_split(feature: &[f32], y: &[f32]) -> Option<Split> {
    let n = feature.len().min(y.len());
    let mut pairs: Vec<(f32, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let (x, t) = (feature[i], y[i]);
        if x.is_finite() && t.is_finite() {
            pairs.push((x, t as f64));
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let total = pairs.len() as f64;
    let sum_all: f64 = pairs.iter().map(|p| p.1).sum();
    let sum2_all: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let parent_var = sum2_all / total - (sum_all / total).powi(2);
    if parent_var <= 0.0 {
        return None;
    }

    let mut left_sum = 0.0f64;
    let mut left_sum2 = 0.0f64;
    let mut best: Option<Split> = None;
    for i in 0..pairs.len() - 1 {
        left_sum += pairs[i].1;
        left_sum2 += pairs[i].1 * pairs[i].1;
        if pairs[i].0 == pairs[i + 1].0 {
            continue; // can't split between equal feature values
        }
        let n_left = (i + 1) as f64;
        let n_right = total - n_left;
        let var_left = (left_sum2 / n_left - (left_sum / n_left).powi(2)).max(0.0);
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        let var_right = (right_sum2 / n_right - (right_sum / n_right).powi(2)).max(0.0);
        let weighted = (n_left * var_left + n_right * var_right) / total;
        let gain = (parent_var - weighted) as f32;
        let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
        if best.map_or(true, |b| gain > b.gain) {
            best = Some(Split { threshold, gain });
        }
    }
    best
}

/// Materialize a split's membership indicator into `out`: 1.0 where
/// `feature >= threshold`, 0.0 where `< threshold`, NaN where the feature is NaN
/// (so the finite-pair scoring skips it).
pub fn split_indicator(feature: &[f32], threshold: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(feature.len());
    for &x in feature {
        out.push(if x.is_nan() {
            f32::NAN
        } else if x >= threshold {
            1.0
        } else {
            0.0
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_obvious_threshold() {
        // clean step: y jumps from 0 to 10 at x=3.5
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0f32, 0.0, 0.0, 10.0, 10.0, 10.0];
        let s = best_variance_split(&x, &y).unwrap();
        assert!((s.threshold - 3.5).abs() < 1e-6, "threshold={}", s.threshold);
        assert!(s.gain > 0.0);
    }

    #[test]
    fn constant_feature_has_no_split() {
        let x = vec![2.0f32, 2.0, 2.0, 2.0];
        let y = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn constant_target_has_no_split() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 5.0, 5.0, 5.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn indicator_materializes_membership_and_skips_nan() {
        let x = vec![1.0f32, 4.0, f32::NAN, 5.0];
        let mut out = Vec::new();
        split_indicator(&x, 3.5, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert!(out[2].is_nan());
        assert_eq!(out[3], 1.0);
    }
}
