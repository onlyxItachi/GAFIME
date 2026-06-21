use pyo3::prelude::*;
use std::collections::HashMap;

use crate::compile_descriptor::{u128_text, ScenarioPlan};
use crate::compile_scenario_batches::{
    build_plan, continuous_combos_for_ordered_features, time_series_candidates_for_features,
    CompilePlanConfig,
};

#[pyclass(name = "CompilePlanBuilder")]
pub struct PyCompilePlanBuilder;

#[pymethods]
impl PyCompilePlanBuilder {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        n_samples: u64,
        n_features: u64,
        plan: bool,
        max_comb_size: i64,
        max_combinations_per_k: i64,
        top_features_for_higher_k: i64,
        max_discrete_candidates: i64,
        max_thresholds_per_feature: i64,
        max_intervals_per_feature: i64,
        max_feature_pairs_for_rectangles: i64,
        top_k_features_for_discrete: i64,
        max_time_series_candidates: i64,
        top_k_features_for_time_series: i64,
        max_feature_candidate: i64,
        enable_discrete_functions: bool,
        enable_time_series_functions: bool,
        discrete_quantiles: Vec<f64>,
        time_series_lags: Vec<i64>,
        time_series_windows: Vec<i64>,
        chunk_size: u64,
    ) -> PyScenarioPlan {
        let config = CompilePlanConfig {
            plan,
            max_comb_size,
            max_combinations_per_k,
            top_features_for_higher_k,
            max_discrete_candidates,
            max_thresholds_per_feature,
            max_intervals_per_feature,
            max_feature_pairs_for_rectangles,
            top_k_features_for_discrete,
            max_time_series_candidates,
            top_k_features_for_time_series,
            max_feature_candidate: if max_feature_candidate < -1 {
                None
            } else {
                Some(max_feature_candidate)
            },
            enable_discrete_functions,
            enable_time_series_functions,
            discrete_quantiles,
            time_series_lags,
            time_series_windows,
            chunk_size: chunk_size.max(1),
        };
        PyScenarioPlan {
            inner: build_plan(n_samples, n_features, &config),
        }
    }

    fn continuous_combos(
        &self,
        feature_indices: Vec<u64>,
        min_arity: u64,
        max_arity: u64,
        max_combinations_per_k: i64,
    ) -> (Vec<Vec<u64>>, Vec<String>) {
        continuous_combos_for_ordered_features(
            &feature_indices,
            min_arity,
            max_arity,
            max_combinations_per_k,
        )
    }

    fn time_series_candidates(
        &self,
        feature_indices: Vec<u64>,
        lags: Vec<i64>,
        windows: Vec<i64>,
        max_candidates: i64,
    ) -> (Vec<HashMap<String, String>>, Vec<String>) {
        let (candidates, warnings) = time_series_candidates_for_features(
            &feature_indices,
            &lags,
            &windows,
            max_candidates,
        );
        let rows = candidates
            .into_iter()
            .map(|candidate| {
                let mut row = HashMap::new();
                row.insert("kind".to_string(), candidate.kind);
                row.insert("feature_index".to_string(), candidate.feature_index.to_string());
                row.insert("lag".to_string(), candidate.lag.to_string());
                row.insert("window".to_string(), candidate.window.to_string());
                row.insert("candidate_id".to_string(), candidate.candidate_id);
                row
            })
            .collect();
        (rows, warnings)
    }
}

#[pyclass(name = "ScenarioPlan")]
pub struct PyScenarioPlan {
    pub inner: ScenarioPlan,
}

#[pymethods]
impl PyScenarioPlan {
    #[getter]
    fn n_samples(&self) -> u64 {
        self.inner.n_samples
    }

    #[getter]
    fn n_features(&self) -> u64 {
        self.inner.n_features
    }

    #[getter]
    fn feature_candidate_count(&self) -> u64 {
        self.inner.feature_candidate_count
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    fn continuous_descriptors(&self) -> Vec<HashMap<String, String>> {
        self.inner
            .continuous
            .iter()
            .map(|descriptor| {
                let mut row = HashMap::new();
                row.insert("arity".to_string(), descriptor.arity.to_string());
                row.insert("feature_start".to_string(), descriptor.feature_start.to_string());
                row.insert("feature_stop".to_string(), descriptor.feature_stop.to_string());
                row.insert("universe_count".to_string(), u128_text(descriptor.universe_count));
                row.insert("planned_count".to_string(), u128_text(descriptor.planned_count));
                row.insert("offset".to_string(), descriptor.offset.to_string());
                row.insert(
                    "first_chunk_id".to_string(),
                    descriptor.chunk_range.first_chunk_id.to_string(),
                );
                row.insert(
                    "chunk_count".to_string(),
                    descriptor.chunk_range.chunk_count.to_string(),
                );
                row.insert(
                    "chunk_size".to_string(),
                    descriptor.chunk_range.chunk_size.to_string(),
                );
                row.insert("saturated".to_string(), descriptor.saturated.to_string());
                row
            })
            .collect()
    }

    fn discrete_descriptor(&self) -> Option<HashMap<String, String>> {
        self.inner.discrete.as_ref().map(|descriptor| {
            let mut row = HashMap::new();
            row.insert("feature_start".to_string(), descriptor.feature_start.to_string());
            row.insert("feature_stop".to_string(), descriptor.feature_stop.to_string());
            row.insert("threshold_count".to_string(), descriptor.threshold_count.to_string());
            row.insert("interval_count".to_string(), descriptor.interval_count.to_string());
            row.insert(
                "rectangle_pair_count".to_string(),
                descriptor.rectangle_pair_count.to_string(),
            );
            row.insert("template_count".to_string(), descriptor.template_count.to_string());
            row.insert("universe_count".to_string(), u128_text(descriptor.universe_count));
            row.insert("planned_count".to_string(), u128_text(descriptor.planned_count));
            row.insert("offset".to_string(), descriptor.offset.to_string());
            row.insert("saturated".to_string(), descriptor.saturated.to_string());
            row
        })
    }

    fn time_series_descriptor(&self) -> Option<HashMap<String, String>> {
        self.inner.time_series.as_ref().map(|descriptor| {
            let mut row = HashMap::new();
            row.insert("feature_start".to_string(), descriptor.feature_start.to_string());
            row.insert("feature_stop".to_string(), descriptor.feature_stop.to_string());
            row.insert("lag_count".to_string(), descriptor.lag_count.to_string());
            row.insert("window_count".to_string(), descriptor.window_count.to_string());
            row.insert("template_count".to_string(), descriptor.template_count.to_string());
            row.insert("universe_count".to_string(), u128_text(descriptor.universe_count));
            row.insert("planned_count".to_string(), u128_text(descriptor.planned_count));
            row.insert("offset".to_string(), descriptor.offset.to_string());
            row.insert("saturated".to_string(), descriptor.saturated.to_string());
            row
        })
    }
}
