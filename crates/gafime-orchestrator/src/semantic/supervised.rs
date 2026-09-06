//! Canonical supervised unary-screening strength policy.
//!
//! This module deliberately owns only the typed reduction and ordering policy
//! that turns already-computed unary metric rows into feature strengths.  It
//! does not own candidate construction, metric execution, or a Python-facing
//! boundary.  Keeping the f32 and f64 variants distinct preserves the legacy
//! public-profile ordering contract: a mixed/fp64 result must never be
//! narrowed before higher-order planning or shortlist ranking.

use gafime_types::{GAFIME_METRIC_PEARSON, GAFIME_METRIC_SPEARMAN};

use crate::plan::combos::{legacy_higher_feature_order, legacy_higher_feature_order_f64};

use super::{SemanticError, SemanticResult};

/// Typed unary strengths used by supervised candidate planning.
///
/// The vector order is meaningful.  In particular, stable score ties retain
/// the producer's feature order until the seeded higher-order shuffle runs.
#[derive(Clone, Debug, PartialEq)]
pub enum SupervisedStrengths {
    F32(Vec<(u32, f32)>),
    F64(Vec<(u32, f64)>),
}

impl SupervisedStrengths {
    /// Reduce row-major fp32 unary metric values into one strength per feature.
    ///
    /// Pearson and Spearman use absolute value; every other metric, including
    /// future or unknown IDs, retains its raw value.  That is the established
    /// compatibility rule, rather than a metric-validation boundary.
    pub fn from_f32(
        row_values: &[f32],
        features: &[u32],
        metric_ids: &[u32],
    ) -> SemanticResult<Self> {
        validate_row_values(row_values.len(), features.len(), metric_ids.len())?;
        if features.is_empty() {
            return Ok(Self::F32(Vec::new()));
        }
        if metric_ids.is_empty() {
            return Ok(Self::F32(
                features
                    .iter()
                    .copied()
                    .map(|feature| (feature, 0.0))
                    .collect(),
            ));
        }

        let mut strengths = Vec::with_capacity(features.len());
        for (&feature, values) in features
            .iter()
            .zip(row_values.chunks_exact(metric_ids.len()))
        {
            let mut strength = None::<f32>;
            for (&metric_id, &value) in metric_ids.iter().zip(values) {
                let candidate = supervised_metric_strength_f32(metric_id, value);
                strength = Some(strength.map_or(candidate, |current| current.max(candidate)));
            }
            strengths.push((feature, strength.unwrap_or(0.0)));
        }
        Ok(Self::F32(strengths))
    }

    /// Reduce row-major fp64 unary metric values into one strength per feature.
    ///
    /// This intentionally does not share an f32 conversion path with
    /// [`Self::from_f32`].  The f64 lane affects stable shortlist ordering.
    pub fn from_f64(
        row_values: &[f64],
        features: &[u32],
        metric_ids: &[u32],
    ) -> SemanticResult<Self> {
        validate_row_values(row_values.len(), features.len(), metric_ids.len())?;
        if features.is_empty() {
            return Ok(Self::F64(Vec::new()));
        }
        if metric_ids.is_empty() {
            return Ok(Self::F64(
                features
                    .iter()
                    .copied()
                    .map(|feature| (feature, 0.0))
                    .collect(),
            ));
        }

        let mut strengths = Vec::with_capacity(features.len());
        for (&feature, values) in features
            .iter()
            .zip(row_values.chunks_exact(metric_ids.len()))
        {
            let mut strength = None::<f64>;
            for (&metric_id, &value) in metric_ids.iter().zip(values) {
                let candidate = supervised_metric_strength_f64(metric_id, value);
                strength = Some(strength.map_or(candidate, |current| current.max(candidate)));
            }
            strengths.push((feature, strength.unwrap_or(0.0)));
        }
        Ok(Self::F64(strengths))
    }

    /// Restore Core's established deterministic feature-ID source order before
    /// applying the stable score sort used by higher-order planning.
    pub fn sort_by_feature(&mut self) {
        match self {
            Self::F32(values) => values.sort_by_key(|(feature, _)| *feature),
            Self::F64(values) => values.sort_by_key(|(feature, _)| *feature),
        }
    }

    /// Select the source features for higher-order planning with the established
    /// profile-specific stable ordering and seeded shuffle.
    pub fn higher_feature_order(
        &self,
        candidate_cols: u32,
        max_combinations_per_k: u64,
        top_features_for_higher_k: u32,
        planning_seed_words: &[u32],
    ) -> Vec<u32> {
        match self {
            Self::F32(values) => legacy_higher_feature_order(
                candidate_cols,
                max_combinations_per_k,
                top_features_for_higher_k,
                planning_seed_words,
                values,
            ),
            Self::F64(values) => legacy_higher_feature_order_f64(
                candidate_cols,
                max_combinations_per_k,
                top_features_for_higher_k,
                planning_seed_words,
                values,
            ),
        }
    }

    /// Return the descending supervised shortlist, preserving the source order
    /// for equal and unordered (`NaN`) comparisons exactly as the legacy path.
    pub fn into_ranked_features(mut self, top_k: u32) -> Vec<u32> {
        match &mut self {
            Self::F32(values) => {
                values.sort_by(|left, right| {
                    right
                        .1
                        .partial_cmp(&left.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                values.truncate(top_k as usize);
                values.iter().map(|(feature, _)| *feature).collect()
            }
            Self::F64(values) => {
                values.sort_by(|left, right| {
                    right
                        .1
                        .partial_cmp(&left.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                values.truncate(top_k as usize);
                values.iter().map(|(feature, _)| *feature).collect()
            }
        }
    }
}

fn validate_row_values(
    row_value_len: usize,
    feature_len: usize,
    metric_len: usize,
) -> SemanticResult<()> {
    if feature_len == 0 || metric_len == 0 {
        return Ok(());
    }
    let required = feature_len
        .checked_mul(metric_len)
        .ok_or(SemanticError::Invalid(
            "supervised strength metric grid exceeds host address space",
        ))?;
    if row_value_len < required {
        return Err(SemanticError::Invalid(
            "supervised strength metric rows do not match feature/metric shape",
        ));
    }
    Ok(())
}

fn supervised_metric_strength_f32(metric_id: u32, value: f32) -> f32 {
    if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
        value.abs()
    } else {
        value
    }
}

fn supervised_metric_strength_f64(metric_id: u32, value: f64) -> f64 {
    if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
        value.abs()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN};

    #[test]
    fn preserves_metric_extremeness_and_unknown_metric_compatibility() {
        let strengths = SupervisedStrengths::from_f32(
            &[-0.25, -0.5, 0.75, -0.125],
            &[7, 3],
            &[GAFIME_METRIC_PEARSON, u32::MAX],
        )
        .unwrap();
        assert_eq!(
            strengths,
            SupervisedStrengths::F32(vec![(7, 0.25), (3, 0.75)])
        );

        let unknown = SupervisedStrengths::from_f64(&[-0.25], &[9], &[u32::MAX]).unwrap();
        assert_eq!(unknown, SupervisedStrengths::F64(vec![(9, -0.25)]));
    }

    #[test]
    fn preserves_nan_max_and_stable_unordered_ties() {
        let strengths = SupervisedStrengths::from_f32(
            &[f32::NAN, 0.5, f32::NAN, f32::NAN],
            &[7, 3],
            &[GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .unwrap();
        match strengths {
            SupervisedStrengths::F32(values) => {
                assert_eq!(values[0], (7, 0.5));
                assert_eq!(values[1].0, 3);
                assert!(values[1].1.is_nan());
            }
            SupervisedStrengths::F64(_) => panic!("fp32 values must retain their numeric lane"),
        }

        let ranked =
            SupervisedStrengths::F32(vec![(7, f32::NAN), (3, f32::NAN)]).into_ranked_features(2);
        assert_eq!(ranked, vec![7, 3]);
    }

    #[test]
    fn accepts_empty_metric_rows_with_legacy_zero_strengths() {
        assert_eq!(
            SupervisedStrengths::from_f32(&[], &[4, 1], &[]).unwrap(),
            SupervisedStrengths::F32(vec![(4, 0.0), (1, 0.0)])
        );
        assert_eq!(
            SupervisedStrengths::from_f64(&[42.0], &[], &[GAFIME_METRIC_SPEARMAN]).unwrap(),
            SupervisedStrengths::F64(Vec::new())
        );
    }

    #[test]
    fn rejects_missing_metric_rows_without_reading_past_the_boundary() {
        assert!(SupervisedStrengths::from_f32(
            &[0.5],
            &[0],
            &[GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2]
        )
        .is_err());
    }

    #[test]
    fn preserves_f64_bits_in_shortlist_and_seeded_higher_order_policy() {
        let lower = 1.0f64;
        let higher = f64::from_bits(lower.to_bits() + 1);
        assert_eq!(lower as f32, higher as f32);
        assert_eq!(
            SupervisedStrengths::from_f64(&[lower, higher], &[0, 1], &[GAFIME_METRIC_R2])
                .unwrap()
                .into_ranked_features(1),
            vec![1]
        );

        let strengths = SupervisedStrengths::F32(vec![(4, 0.5), (0, 0.1), (5, 0.6)]);
        assert_eq!(strengths.higher_feature_order(6, 3, 3, &[7]), vec![4, 0, 5]);
    }
}
