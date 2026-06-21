use crate::compile_combinatorics::{
    chunk_count, saturating_comb, saturating_u128_add, saturating_u128_mul, saturating_u64_add,
    UINT32_MAX,
};
use crate::compile_descriptor::{
    ChunkRange, ContinuousDescriptor, DiscreteDescriptor, ScenarioPlan, TimeSeriesDescriptor,
};

pub const DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP: u64 = 1024;

#[derive(Clone, Debug)]
pub struct CompilePlanConfig {
    pub plan: bool,
    pub max_comb_size: i64,
    pub max_combinations_per_k: i64,
    pub top_features_for_higher_k: i64,
    pub max_discrete_candidates: i64,
    pub max_thresholds_per_feature: i64,
    pub max_intervals_per_feature: i64,
    pub max_feature_pairs_for_rectangles: i64,
    pub top_k_features_for_discrete: i64,
    pub max_time_series_candidates: i64,
    pub top_k_features_for_time_series: i64,
    pub max_feature_candidate: Option<i64>,
    pub enable_discrete_functions: bool,
    pub enable_time_series_functions: bool,
    pub discrete_quantiles: Vec<f64>,
    pub time_series_lags: Vec<i64>,
    pub time_series_windows: Vec<i64>,
    pub chunk_size: u64,
}

pub fn build_plan(n_samples: u64, n_features: u64, config: &CompilePlanConfig) -> ScenarioPlan {
    if !config.plan {
        return ScenarioPlan::empty(n_samples, n_features);
    }

    let mut warnings = Vec::new();
    let feature_count = feature_candidate_count(n_features, config, &mut warnings);
    let (continuous, mut offset, _) = continuous_descriptors(feature_count, config, &mut warnings);
    let discrete = discrete_descriptor(feature_count, config, offset);
    if let Some(ref descriptor) = discrete {
        offset = saturating_u64_add(offset, descriptor.planned_count);
    }
    let time_series = time_series_descriptor(feature_count, config, offset);

    ScenarioPlan {
        n_samples,
        n_features,
        feature_candidate_count: feature_count,
        continuous,
        discrete,
        time_series,
        warnings,
    }
}

pub fn continuous_combos_for_ordered_features(
    feature_indices: &[u64],
    min_arity: u64,
    max_arity: u64,
    max_combinations_per_k: i64,
) -> (Vec<Vec<u64>>, Vec<String>) {
    let mut combos = Vec::new();
    let mut warnings = Vec::new();
    if feature_indices.is_empty() || max_arity < min_arity {
        return (combos, warnings);
    }

    let cap = if max_combinations_per_k < 0 {
        usize::MAX
    } else {
        max_combinations_per_k.max(0) as usize
    };
    if cap == 0 {
        return (combos, warnings);
    }

    for arity in min_arity..=max_arity {
        let arity_usize = arity as usize;
        if feature_indices.len() < arity_usize {
            break;
        }
        let mut current = Vec::with_capacity(arity_usize);
        let mut produced = 0usize;
        generate_combos(
            feature_indices,
            arity_usize,
            0,
            cap,
            &mut current,
            &mut produced,
            &mut combos,
        );
        if produced >= cap {
            warnings.push(format!(
                "k={} combinations capped by max_combinations_per_k.",
                arity
            ));
        }
    }

    (combos, warnings)
}

pub fn time_series_candidates_for_features(
    feature_indices: &[u64],
    lags: &[i64],
    windows: &[i64],
    max_candidates: i64,
) -> (Vec<TimeSeriesCandidateDescriptor>, Vec<String>) {
    let mut candidates = Vec::new();
    let mut warnings = Vec::new();
    if max_candidates < 1 {
        warnings
            .push("max_time_series_candidates < 1; time-series functions disabled.".to_string());
        return (candidates, warnings);
    }
    let max_candidates = max_candidates as usize;

    for feature in feature_indices {
        for lag in lags {
            let lag = (*lag).max(1);
            for kind in [
                "time_series_lag",
                "time_series_delta",
                "time_series_velocity",
                "time_series_acceleration",
            ] {
                if !push_time_series_candidate(
                    &mut candidates,
                    max_candidates,
                    kind,
                    *feature,
                    lag,
                    1,
                ) {
                    warnings.push(
                        "Time-series candidates capped by max_time_series_candidates.".to_string(),
                    );
                    return (candidates, warnings);
                }
            }
        }
        for window in windows {
            let window = (*window).max(1);
            for kind in [
                "time_series_rolling_mean",
                "time_series_rolling_std",
                "time_series_rolling_sum",
            ] {
                if !push_time_series_candidate(
                    &mut candidates,
                    max_candidates,
                    kind,
                    *feature,
                    1,
                    window,
                ) {
                    warnings.push(
                        "Time-series candidates capped by max_time_series_candidates.".to_string(),
                    );
                    return (candidates, warnings);
                }
            }
        }
    }

    (candidates, warnings)
}

pub fn discrete_candidates_for_features(
    feature_indices: &[u64],
    thresholds_by_feature: &[Vec<f64>],
    intervals_by_feature: &[Vec<(f64, f64)>],
    scales_by_feature: &[f64],
    feature_pairs: &[(u64, u64)],
    max_candidates: i64,
) -> (Vec<DiscreteCandidateDescriptor>, Vec<String>) {
    let mut candidates = Vec::new();
    let mut warnings = Vec::new();
    if max_candidates < 1 {
        warnings.push("max_discrete_candidates < 1; discrete functions disabled.".to_string());
        return (candidates, warnings);
    }
    let max_candidates = max_candidates as usize;

    for (feature_pos, feature) in feature_indices.iter().enumerate() {
        let thresholds = thresholds_by_feature
            .get(feature_pos)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let intervals = intervals_by_feature
            .get(feature_pos)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let scale = scales_by_feature.get(feature_pos).copied().unwrap_or(1.0);

        for threshold in thresholds {
            for direction in ["ge", "le"] {
                if !push_discrete_candidate(
                    &mut candidates,
                    max_candidates,
                    "discrete_function_soft_threshold",
                    &[*feature],
                    &[*threshold],
                    &[],
                    direction,
                    None,
                    &[scale],
                ) {
                    warnings
                        .push("Discrete candidates capped by max_discrete_candidates.".to_string());
                    return (candidates, warnings);
                }
                if !push_discrete_candidate(
                    &mut candidates,
                    max_candidates,
                    "discrete_function_value_gated_threshold",
                    &[*feature],
                    &[*threshold],
                    &[],
                    direction,
                    Some(*feature),
                    &[scale],
                ) {
                    warnings
                        .push("Discrete candidates capped by max_discrete_candidates.".to_string());
                    return (candidates, warnings);
                }
            }
        }

        for interval in intervals {
            if !push_discrete_candidate(
                &mut candidates,
                max_candidates,
                "discrete_function_soft_interval",
                &[*feature],
                &[],
                &[*interval],
                "ge",
                None,
                &[scale],
            ) {
                warnings.push("Discrete candidates capped by max_discrete_candidates.".to_string());
                return (candidates, warnings);
            }
        }
    }

    for (feature_a, feature_b) in feature_pairs {
        let Some(pos_a) = feature_position(feature_indices, *feature_a) else {
            continue;
        };
        let Some(pos_b) = feature_position(feature_indices, *feature_b) else {
            continue;
        };
        let intervals_a = intervals_by_feature
            .get(pos_a)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let intervals_b = intervals_by_feature
            .get(pos_b)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if intervals_a.is_empty() || intervals_b.is_empty() {
            continue;
        }
        let scale_a = scales_by_feature.get(pos_a).copied().unwrap_or(1.0);
        let scale_b = scales_by_feature.get(pos_b).copied().unwrap_or(1.0);
        for interval_a in intervals_a {
            for interval_b in intervals_b {
                for (kind, value_feature) in [
                    ("discrete_function_soft_rectangle", None),
                    (
                        "discrete_function_value_in_soft_rectangle",
                        Some(*feature_a),
                    ),
                    (
                        "discrete_function_value_in_soft_rectangle",
                        Some(*feature_b),
                    ),
                ] {
                    if !push_discrete_candidate(
                        &mut candidates,
                        max_candidates,
                        kind,
                        &[*feature_a, *feature_b],
                        &[],
                        &[*interval_a, *interval_b],
                        "ge",
                        value_feature,
                        &[scale_a, scale_b],
                    ) {
                        warnings.push(
                            "Discrete candidates capped by max_discrete_candidates.".to_string(),
                        );
                        return (candidates, warnings);
                    }
                }
            }
        }
    }

    (candidates, warnings)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeSeriesCandidateDescriptor {
    pub kind: String,
    pub feature_index: u64,
    pub lag: i64,
    pub window: i64,
    pub candidate_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteCandidateDescriptor {
    pub kind: String,
    pub feature_indices: Vec<u64>,
    pub thresholds: Vec<f64>,
    pub intervals: Vec<(f64, f64)>,
    pub direction: String,
    pub value_feature: Option<u64>,
    pub scales: Vec<f64>,
}

fn push_time_series_candidate(
    candidates: &mut Vec<TimeSeriesCandidateDescriptor>,
    max_candidates: usize,
    kind: &str,
    feature_index: u64,
    lag: i64,
    window: i64,
) -> bool {
    if candidates.len() >= max_candidates {
        return false;
    }
    candidates.push(TimeSeriesCandidateDescriptor {
        kind: kind.to_string(),
        feature_index,
        lag,
        window,
        candidate_id: format!(
            "{}|feature={}|lag={}|window={}",
            kind, feature_index, lag, window
        ),
    });
    true
}

#[allow(clippy::too_many_arguments)]
fn push_discrete_candidate(
    candidates: &mut Vec<DiscreteCandidateDescriptor>,
    max_candidates: usize,
    kind: &str,
    feature_indices: &[u64],
    thresholds: &[f64],
    intervals: &[(f64, f64)],
    direction: &str,
    value_feature: Option<u64>,
    scales: &[f64],
) -> bool {
    if candidates.len() >= max_candidates {
        return false;
    }
    candidates.push(DiscreteCandidateDescriptor {
        kind: kind.to_string(),
        feature_indices: feature_indices.to_vec(),
        thresholds: thresholds.to_vec(),
        intervals: intervals.to_vec(),
        direction: direction.to_string(),
        value_feature,
        scales: scales.to_vec(),
    });
    true
}

fn feature_position(feature_indices: &[u64], feature: u64) -> Option<usize> {
    feature_indices
        .iter()
        .position(|candidate| *candidate == feature)
}

fn generate_combos(
    feature_indices: &[u64],
    arity: usize,
    start: usize,
    cap: usize,
    current: &mut Vec<u64>,
    produced: &mut usize,
    combos: &mut Vec<Vec<u64>>,
) {
    if *produced >= cap {
        return;
    }
    if current.len() == arity {
        combos.push(current.clone());
        *produced += 1;
        return;
    }

    let remaining = arity - current.len();
    if feature_indices.len().saturating_sub(start) < remaining {
        return;
    }
    let last_start = feature_indices.len() - remaining;
    for idx in start..=last_start {
        if *produced >= cap {
            break;
        }
        current.push(feature_indices[idx]);
        generate_combos(
            feature_indices,
            arity,
            idx + 1,
            cap,
            current,
            produced,
            combos,
        );
        current.pop();
    }
}

fn feature_candidate_count(
    n_features: u64,
    config: &CompilePlanConfig,
    warnings: &mut Vec<String>,
) -> u64 {
    match config.max_feature_candidate {
        None => n_features,
        Some(value) if value >= 0 => n_features.min(value as u64),
        Some(_) => {
            if has_explicit_power_user_limits(config) {
                return n_features;
            }
            let capped = n_features.min(DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP);
            if n_features > capped {
                warnings.push(format!(
                    "max_feature_candidate=-1 without explicit limits; applying practical safety cap of {} features.",
                    DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP
                ));
            }
            capped
        }
    }
}

fn has_explicit_power_user_limits(config: &CompilePlanConfig) -> bool {
    config.max_comb_size != 2
        || config.max_combinations_per_k != 5000
        || config.top_features_for_higher_k != 50
        || config.max_discrete_candidates != 100_000
        || config.top_k_features_for_discrete != 50
        || config.max_time_series_candidates != 100_000
        || config.top_k_features_for_time_series != 50
}

fn continuous_descriptors(
    n_features: u64,
    config: &CompilePlanConfig,
    warnings: &mut Vec<String>,
) -> (Vec<ContinuousDescriptor>, u64, u32) {
    let mut descriptors = Vec::new();
    let mut offset: u64 = 0;
    let mut chunk_id: u32 = 0;
    let max_arity = (config.max_comb_size.max(1) as u64).min(n_features);
    for arity in 1..=max_arity {
        let feature_stop = if arity > 1 {
            n_features.min(config.top_features_for_higher_k.max(0) as u64)
        } else {
            n_features
        };
        let (universe_count, saturated) = saturating_comb(feature_stop, arity);
        let planned_count = apply_count_cap(universe_count, config.max_combinations_per_k);
        let chunks = chunk_count(planned_count, config.chunk_size);
        if chunks > UINT32_MAX {
            warnings.push(format!(
                "arity={} chunk count exceeds uint32; native launch ids will be split by checkpoint.",
                arity
            ));
        }
        let local_chunks = chunks.min(UINT32_MAX) as u32;
        descriptors.push(ContinuousDescriptor {
            arity,
            feature_start: 0,
            feature_stop,
            universe_count,
            planned_count,
            offset,
            chunk_range: ChunkRange {
                first_chunk_id: chunk_id,
                chunk_count: local_chunks,
                chunk_size: config.chunk_size.max(1),
            },
            saturated,
        });
        offset = saturating_u64_add(offset, planned_count);
        chunk_id = chunk_id.saturating_add(local_chunks);
    }
    (descriptors, offset, chunk_id)
}

fn discrete_descriptor(
    n_features: u64,
    config: &CompilePlanConfig,
    offset: u64,
) -> Option<DiscreteDescriptor> {
    if !config.enable_discrete_functions {
        return None;
    }
    let feature_count = n_features.min(config.top_k_features_for_discrete.max(0) as u64);
    let valid_quantiles = config
        .discrete_quantiles
        .iter()
        .filter(|value| **value > 0.0 && **value < 1.0)
        .count() as u64;
    let threshold_count = valid_quantiles.min(config.max_thresholds_per_feature.max(0) as u64);
    let interval_universe = threshold_count.saturating_mul(threshold_count.saturating_sub(1)) / 2;
    let interval_count = interval_universe.min(config.max_intervals_per_feature.max(0) as u64);
    let pair_universe = feature_count.saturating_mul(feature_count.saturating_sub(1)) / 2;
    let rectangle_pair_count =
        pair_universe.min(config.max_feature_pairs_for_rectangles.max(0) as u64);
    let per_feature_templates = threshold_count
        .saturating_mul(4)
        .saturating_add(interval_count);
    let rectangle_templates = interval_count
        .saturating_mul(interval_count)
        .saturating_mul(3);
    let universe_count = saturating_u128_add(
        saturating_u128_mul(feature_count as u128, per_feature_templates as u128),
        saturating_u128_mul(rectangle_pair_count as u128, rectangle_templates as u128),
    );
    let planned_count = universe_count.min(config.max_discrete_candidates.max(0) as u128);
    Some(DiscreteDescriptor {
        feature_start: 0,
        feature_stop: feature_count,
        threshold_count,
        interval_count,
        rectangle_pair_count,
        template_count: per_feature_templates.saturating_add(rectangle_templates),
        universe_count,
        planned_count,
        offset,
        saturated: universe_count == u128::MAX,
    })
}

fn time_series_descriptor(
    n_features: u64,
    config: &CompilePlanConfig,
    offset: u64,
) -> Option<TimeSeriesDescriptor> {
    if !config.enable_time_series_functions {
        return None;
    }
    let feature_count = n_features.min(config.top_k_features_for_time_series.max(0) as u64);
    let lag_count = config.time_series_lags.len() as u64;
    let window_count = config.time_series_windows.len() as u64;
    let template_count = lag_count
        .saturating_mul(4)
        .saturating_add(window_count.saturating_mul(3));
    let universe_count = saturating_u128_mul(feature_count as u128, template_count as u128);
    let planned_count = universe_count.min(config.max_time_series_candidates.max(0) as u128);
    Some(TimeSeriesDescriptor {
        feature_start: 0,
        feature_stop: feature_count,
        lag_count,
        window_count,
        template_count,
        universe_count,
        planned_count,
        offset,
        saturated: universe_count == u128::MAX,
    })
}

fn apply_count_cap(count: u128, cap: i64) -> u128 {
    if cap < 0 {
        count
    } else {
        count.min(cap as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::{continuous_combos_for_ordered_features, discrete_candidates_for_features};

    #[test]
    fn continuous_combos_respect_order_and_cap() {
        let (combos, warnings) = continuous_combos_for_ordered_features(&[3, 1, 2, 0], 2, 3, 3);
        assert_eq!(
            combos,
            vec![
                vec![3, 1],
                vec![3, 2],
                vec![3, 0],
                vec![3, 1, 2],
                vec![3, 1, 0],
                vec![3, 2, 0],
            ]
        );
        assert_eq!(
            warnings,
            vec![
                "k=2 combinations capped by max_combinations_per_k.",
                "k=3 combinations capped by max_combinations_per_k.",
            ]
        );
    }

    #[test]
    fn time_series_candidates_match_python_order_and_cap() {
        let (candidates, warnings) =
            super::time_series_candidates_for_features(&[2, 0], &[1, 2], &[3], 5);
        assert_eq!(candidates.len(), 5);
        assert_eq!(candidates[0].kind, "time_series_lag");
        assert_eq!(
            candidates[0].candidate_id,
            "time_series_lag|feature=2|lag=1|window=1"
        );
        assert_eq!(
            candidates[4].candidate_id,
            "time_series_lag|feature=2|lag=2|window=1"
        );
        assert_eq!(
            warnings,
            vec!["Time-series candidates capped by max_time_series_candidates."]
        );
    }

    #[test]
    fn discrete_candidates_match_python_order_and_cap() {
        let (candidates, warnings) = discrete_candidates_for_features(
            &[1, 0],
            &[vec![0.5, 1.5], vec![2.5]],
            &[vec![(0.5, 1.5)], vec![]],
            &[1.0, 2.0],
            &[(1, 0)],
            6,
        );
        assert_eq!(candidates.len(), 6);
        assert_eq!(candidates[0].kind, "discrete_function_soft_threshold");
        assert_eq!(candidates[0].feature_indices, vec![1]);
        assert_eq!(candidates[0].thresholds, vec![0.5]);
        assert_eq!(
            candidates[1].kind,
            "discrete_function_value_gated_threshold"
        );
        assert_eq!(candidates[1].value_feature, Some(1));
        assert_eq!(candidates[2].direction, "le");
        assert_eq!(candidates[4].thresholds, vec![1.5]);
        assert_eq!(
            warnings,
            vec!["Discrete candidates capped by max_discrete_candidates."]
        );
    }
}
