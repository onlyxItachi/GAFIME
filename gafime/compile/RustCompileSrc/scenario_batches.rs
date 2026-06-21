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
    let per_feature_templates = threshold_count.saturating_mul(4).saturating_add(interval_count);
    let rectangle_templates = interval_count.saturating_mul(interval_count).saturating_mul(3);
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
    let template_count = lag_count.saturating_mul(4).saturating_add(window_count.saturating_mul(3));
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
