use std::collections::HashSet;

use gafime_orchestrator::{config::EngineConfig, plan::combos::legacy_unary_feature_order};
use gafime_types::GAFIME_BACKEND_CPU;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::artifact::{compile_continuous_rows, PyCompiledContinuousArtifact};
#[cfg(not(feature = "local-cmake-experiment"))]
use crate::common::ContinuousReport;
use crate::common::{validate_shape, DecisionPathResultParams, PyBoundaryError};
use crate::continuous::{analyze_continuous_rows_once, unary_strengths_from_table};
use crate::py_api::PyContinuousReport;
use crate::runtime::{get_u32, parse_engine_config};

#[cfg(feature = "local-cmake-experiment")]
pub(crate) mod local_cmake_experiment;

fn row_major_feature_prefix(
    features: &[f32],
    rows: usize,
    source_cols: usize,
    selected_cols: usize,
) -> Vec<f32> {
    let mut selected = Vec::with_capacity(rows.saturating_mul(selected_cols));
    for row in 0..rows {
        let start = row * source_cols;
        selected.extend_from_slice(&features[start..start + selected_cols]);
    }
    selected
}

fn column_major_feature_selection(
    features: &[f32],
    rows: usize,
    source_cols: usize,
    selected_features: &[u32],
) -> Result<Vec<f32>, PyBoundaryError> {
    if selected_features
        .iter()
        .any(|&feature| feature as usize >= source_cols)
    {
        return Err(PyBoundaryError::InvalidInput(
            "generated-family source feature is out of range".to_string(),
        ));
    }
    let capacity = rows.checked_mul(selected_features.len()).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "generated-family source selection exceeds host address space".to_string(),
        )
    })?;
    let mut selected = Vec::with_capacity(capacity);
    for &feature in selected_features {
        let feature = feature as usize;
        for row in 0..rows {
            selected.push(features[row * source_cols + feature]);
        }
    }
    Ok(selected)
}

fn column_major_feature_prefix(
    features: &[f32],
    rows: usize,
    source_cols: usize,
    selected_cols: usize,
) -> Vec<f32> {
    let mut selected = Vec::with_capacity(rows.saturating_mul(selected_cols));
    for feature in 0..selected_cols {
        for row in 0..rows {
            selected.push(features[row * source_cols + feature]);
        }
    }
    selected
}

fn select_generated_source_features(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    features: &[f32],
    target: &[f32],
    top_k: u32,
) -> Result<Vec<u32>, PyBoundaryError> {
    let candidate_cols = config.effective_feature_candidate_count(cols);
    if candidate_cols == 0 || top_k == 0 {
        return Ok(Vec::new());
    }

    let planning_seed_words = config.effective_planning_seed_words();
    let unary_features = legacy_unary_feature_order(
        candidate_cols,
        config.budget.max_combinations_per_k,
        &planning_seed_words,
    );
    if unary_features.is_empty() {
        return Ok(Vec::new());
    }

    let mut screening_config = config.clone();
    screening_config.budget.max_comb_size = 1;
    screening_config.permutation_tests = 0;
    screening_config.num_repeats = 1;
    screening_config.graph_requested = false;
    let screening = analyze_continuous_rows_once(
        screening_config,
        rows,
        cols,
        features.to_vec(),
        target.to_vec(),
    )?;
    let mut strengths =
        unary_strengths_from_table(&screening.table, &unary_features, &config.metric_ids)?;
    if config.backend_kind == GAFIME_BACKEND_CPU {
        // v0.5's score dictionary was inserted in ascending feature order on
        // Core. Python's stable score sort therefore used feature id for ties.
        strengths.sort_by_key(|(feature, _)| *feature);
    }
    strengths.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    strengths.truncate(top_k as usize);
    Ok(strengths.into_iter().map(|(feature, _)| feature).collect())
}

fn bounded_time_series_descriptors(
    rows: usize,
    source_features: &[u32],
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
    limit: usize,
) -> Vec<gafime_cpu::time_series::TimeSeriesFeature> {
    use gafime_cpu::time_series::{TimeSeriesFeature, TimeSeriesOp};

    let operations_per_feature = lags
        .len()
        .saturating_mul(if velocity { 4 } else { 1 })
        .saturating_add(windows.len().saturating_mul(3));
    let universe = source_features.len().saturating_mul(operations_per_feature);
    let mut descriptors = Vec::with_capacity(limit.min(universe));
    'features: for &base in source_features {
        for &lag in lags {
            let lag_rows = lag as usize;
            if lag_rows == 0 || lag_rows >= rows {
                continue;
            }
            let mut push = |op| {
                if descriptors.len() == limit {
                    return false;
                }
                descriptors.push(TimeSeriesFeature {
                    base_feature: base,
                    op,
                });
                true
            };
            if !push(TimeSeriesOp::Lag(lag)) {
                break 'features;
            }
            if velocity
                && (!push(TimeSeriesOp::Delta(lag))
                    || !push(TimeSeriesOp::Velocity(lag))
                    || (lag_rows.saturating_mul(2) < rows
                        && !push(TimeSeriesOp::Acceleration(lag))))
            {
                break 'features;
            }
        }
        for &window in windows {
            let window_rows = window as usize;
            if window_rows < 2 || window_rows > rows {
                continue;
            }
            for op in [
                TimeSeriesOp::RollingMean(window),
                TimeSeriesOp::RollingStd(window),
                TimeSeriesOp::RollingSum(window),
            ] {
                if descriptors.len() == limit {
                    break 'features;
                }
                descriptors.push(TimeSeriesFeature {
                    base_feature: base,
                    op,
                });
            }
        }
    }
    descriptors
}

fn generated_feature_limit(
    configured_limit: u64,
    rows: usize,
    base_candidate_cols: usize,
) -> usize {
    let configured_limit = usize::try_from(configured_limit).unwrap_or(usize::MAX);
    let addressable_limit = (usize::MAX / rows).saturating_sub(base_candidate_cols);
    let column_limit = (u32::MAX as usize).saturating_sub(base_candidate_cols);
    configured_limit.min(addressable_limit).min(column_limit)
}

fn expanded_column_count(cols: usize) -> PyResult<u32> {
    u32::try_from(cols).map_err(|_| PyValueError::new_err("expanded feature count exceeds u32"))
}

fn append_unique_generated_names(
    names: &mut Vec<String>,
    generated_names: impl IntoIterator<Item = String>,
) {
    let mut used: HashSet<String> = names.iter().cloned().collect();
    for generated_name in generated_names {
        if used.insert(generated_name.clone()) {
            names.push(generated_name);
            continue;
        }
        for suffix in 1usize.. {
            let unique_name = format!("{generated_name}#generated{suffix}");
            if used.insert(unique_name.clone()) {
                names.push(unique_name);
                break;
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "bounded generation keeps source shape, candidate prefix, operation sets, and admission limit explicit"
)]
fn expand_time_series_bounded(
    features: &[f32],
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    source_features: &[u32],
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
    generated_limit: usize,
) -> Result<
    (
        Vec<f32>,
        usize,
        Vec<gafime_cpu::time_series::TimeSeriesFeature>,
    ),
    PyBoundaryError,
> {
    use gafime_cpu::time_series::TimeSeriesOp;

    if base_candidate_cols > source_cols
        || source_features
            .iter()
            .any(|&feature| feature as usize >= base_candidate_cols)
    {
        return Err(PyBoundaryError::InvalidInput(
            "time-series source feature is outside the candidate prefix".to_string(),
        ));
    }
    let descriptors = bounded_time_series_descriptors(
        rows,
        source_features,
        lags,
        windows,
        velocity,
        generated_limit,
    );
    let expanded_cols = base_candidate_cols + descriptors.len();
    let mut expanded = vec![0.0f32; rows * expanded_cols];
    for row in 0..rows {
        let source = row * source_cols;
        let destination = row * expanded_cols;
        expanded[destination..destination + base_candidate_cols]
            .copy_from_slice(&features[source..source + base_candidate_cols]);
    }

    for (generated_index, descriptor) in descriptors.iter().enumerate() {
        let base = descriptor.base_feature as usize;
        let destination_col = base_candidate_cols + generated_index;
        match descriptor.op {
            TimeSeriesOp::Lag(lag) => {
                let lag = lag as usize;
                for row in lag..rows {
                    expanded[row * expanded_cols + destination_col] =
                        features[(row - lag) * source_cols + base];
                }
                for row in 0..lag {
                    expanded[row * expanded_cols + destination_col] = f32::NAN;
                }
            }
            TimeSeriesOp::Delta(lag) | TimeSeriesOp::Velocity(lag) => {
                let lag = lag as usize;
                let scale = if matches!(descriptor.op, TimeSeriesOp::Velocity(_)) {
                    lag as f32
                } else {
                    1.0
                };
                for row in 0..lag {
                    expanded[row * expanded_cols + destination_col] = f32::NAN;
                }
                for row in lag..rows {
                    let delta = features[row * source_cols + base]
                        - features[(row - lag) * source_cols + base];
                    expanded[row * expanded_cols + destination_col] = delta / scale;
                }
            }
            TimeSeriesOp::Acceleration(lag) => {
                let lag = lag as usize;
                let history = lag * 2;
                let scale = (lag * lag) as f32;
                for row in 0..history {
                    expanded[row * expanded_cols + destination_col] = f32::NAN;
                }
                for row in history..rows {
                    expanded[row * expanded_cols + destination_col] = (features
                        [row * source_cols + base]
                        - 2.0 * features[(row - lag) * source_cols + base]
                        + features[(row - history) * source_cols + base])
                        / scale;
                }
            }
            TimeSeriesOp::RollingMean(window)
            | TimeSeriesOp::RollingStd(window)
            | TimeSeriesOp::RollingSum(window) => {
                let window = window as usize;
                let mut sum = 0.0f64;
                let mut sum2 = 0.0f64;
                let mut invalid = 0usize;
                for row in 0..rows {
                    let value = features[row * source_cols + base];
                    if value.is_finite() {
                        let value = value as f64;
                        sum += value;
                        sum2 += value * value;
                    } else {
                        invalid += 1;
                    }
                    if row >= window {
                        let old = features[(row - window) * source_cols + base];
                        if old.is_finite() {
                            let old = old as f64;
                            sum -= old;
                            sum2 -= old * old;
                        } else {
                            invalid -= 1;
                        }
                    }
                    let output = if row + 1 < window || invalid != 0 {
                        f32::NAN
                    } else {
                        let mean = sum / window as f64;
                        match descriptor.op {
                            TimeSeriesOp::RollingMean(_) => mean as f32,
                            TimeSeriesOp::RollingStd(_) => {
                                (sum2 / window as f64 - mean * mean).max(0.0).sqrt() as f32
                            }
                            TimeSeriesOp::RollingSum(_) => sum as f32,
                            _ => unreachable!(),
                        }
                    };
                    expanded[row * expanded_cols + destination_col] = output;
                }
            }
        }
    }
    Ok((expanded, expanded_cols, descriptors))
}

fn discover_decision_paths_bounded(
    features: &[f32],
    target: &[f32],
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    discovery_features: &[u32],
    params: &gafime_cpu::decision_path::DecisionPathParams,
) -> Result<Vec<gafime_cpu::decision_path::DecisionPath>, PyBoundaryError> {
    if base_candidate_cols > source_cols
        || discovery_features
            .iter()
            .any(|&feature| feature as usize >= base_candidate_cols)
    {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path source feature is outside the candidate prefix".to_string(),
        ));
    }
    if discovery_features.is_empty() || params.max_paths == 0 {
        return Ok(Vec::new());
    }
    let discovery_columns =
        column_major_feature_selection(features, rows, source_cols, discovery_features)?;
    let discovery_cols = discovery_features.len();
    let mut paths = gafime_cpu::decision_path::find_decision_paths(
        &discovery_columns,
        rows,
        discovery_cols,
        target,
        params,
    );
    for path in &mut paths {
        for node in &mut path.nodes {
            node.feature = discovery_features
                .get(node.feature as usize)
                .copied()
                .ok_or_else(|| {
                    PyBoundaryError::InvalidInput(
                        "decision-path discovery returned an out-of-range node".to_string(),
                    )
                })?;
        }
    }
    Ok(paths)
}

fn materialize_decision_path_expansion(
    features: &[f32],
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    paths: &[gafime_cpu::decision_path::DecisionPath],
) -> Result<(Vec<f32>, usize), PyBoundaryError> {
    if base_candidate_cols > source_cols
        || paths
            .iter()
            .flat_map(|path| &path.nodes)
            .any(|node| node.feature as usize >= base_candidate_cols || !node.threshold.is_finite())
    {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path source feature is outside the candidate prefix".to_string(),
        ));
    }
    let expanded_cols = base_candidate_cols
        .checked_add(paths.len())
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "decision-path expanded column count overflows".to_string(),
            )
        })?;
    let base_features = row_major_feature_prefix(features, rows, source_cols, base_candidate_cols);
    if paths.is_empty() {
        return Ok((base_features, expanded_cols));
    }

    // Membership remains intentionally confined to the established fallback.
    // The compact CUDA score route consumes descriptors directly instead.
    let base_columns =
        column_major_feature_prefix(features, rows, source_cols, base_candidate_cols);
    let memberships = paths
        .iter()
        .map(|path| gafime_cpu::decision_path::path_membership(&base_columns, rows, &path.nodes))
        .collect::<Vec<_>>();
    let capacity = rows.checked_mul(expanded_cols).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "decision-path expanded matrix exceeds host address space".to_string(),
        )
    })?;
    let mut expanded = vec![0.0f32; capacity];
    for row in 0..rows {
        let base_source = row * base_candidate_cols;
        let destination = row * expanded_cols;
        expanded[destination..destination + base_candidate_cols]
            .copy_from_slice(&base_features[base_source..base_source + base_candidate_cols]);
        for (path_index, membership) in memberships.iter().enumerate() {
            expanded[destination + base_candidate_cols + path_index] = membership[row];
        }
    }
    Ok((expanded, expanded_cols))
}

#[cfg(test)]
fn expand_decision_path_bounded(
    features: &[f32],
    target: &[f32],
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    discovery_features: &[u32],
    params: &gafime_cpu::decision_path::DecisionPathParams,
) -> Result<
    (
        Vec<f32>,
        usize,
        Vec<gafime_cpu::decision_path::DecisionPath>,
    ),
    PyBoundaryError,
> {
    let paths = discover_decision_paths_bounded(
        features,
        target,
        rows,
        source_cols,
        base_candidate_cols,
        discovery_features,
        params,
    )?;
    let (expanded, expanded_cols) = materialize_decision_path_expansion(
        features,
        rows,
        source_cols,
        base_candidate_cols,
        &paths,
    )?;
    Ok((expanded, expanded_cols, paths))
}

fn validate_decision_path_permutation_config(config: &EngineConfig) -> Result<(), PyBoundaryError> {
    if config.permutation_tests == 0 {
        return Ok(());
    }
    Err(PyBoundaryError::UnsupportedFeature(
        "decision-path permutation significance requires path rediscovery for every permuted target and is not supported by this boundary"
            .to_string(),
    ))
}

/// time_series family: expand the feature matrix with lag/delta/velocity/
/// acceleration and rolling mean/std/sum columns, then mine the expanded matrix
/// through the normal continuous path
/// (which dispatches to CPU or GPU per config). The expanded matrix stays
/// native; only the report + expanded feature names cross back. Returns
/// (report, all_feature_names = base ++ time-series).
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, lags, windows, velocity=true))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python time-series API exposes each user-configurable input explicitly"
)]
pub(crate) fn analyze_time_series(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    lags: Vec<u32>,
    windows: Vec<u32>,
    velocity: bool,
) -> PyResult<(PyContinuousReport, Vec<String>)> {
    validate_shape(rows, cols, features.len(), target.len()).map_err(PyErr::from)?;
    let mut parsed = parse_engine_config(config)?;
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyValueError::new_err("rows exceed host address space"))?;
    let cols_usize = cols as usize;
    let base_candidate_cols = parsed.effective_feature_candidate_count(cols) as usize;
    let generated_limit = generated_feature_limit(
        parsed.budget.max_time_series_candidates,
        rows_usize,
        base_candidate_cols,
    );
    let source_features = if base_candidate_cols == 0 || generated_limit == 0 {
        Vec::new()
    } else {
        select_generated_source_features(
            &parsed,
            rows,
            cols,
            &features,
            &target,
            parsed.budget.top_k_features_for_time_series,
        )
        .map_err(PyErr::from)?
    };
    let (expanded, expanded_cols, descriptors) = if base_candidate_cols == 0 {
        (features, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_time_series_bounded(
            &features,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &source_features,
            &lags,
            &windows,
            velocity,
            generated_limit,
        )
        .map_err(PyErr::from)?
    };
    let report = analyze_continuous_rows_once(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
        target,
    )
    .map(PyContinuousReport::from)
    .map_err(PyErr::from)?;
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        descriptors.iter().map(|descriptor| {
            let base = base_names
                .get(descriptor.base_feature as usize)
                .map(String::as_str)
                .unwrap_or("feature");
            gafime_cpu::time_series::feature_label(base, descriptor.op)
        }),
    );
    Ok((report, names))
}

/// time_series compile path: expand native lag/delta/velocity/acceleration and
/// rolling mean/std/sum columns, then return a resident compiled continuous
/// artifact over the expanded matrix.
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, lags, windows, velocity=true))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python time-series compile API mirrors the analyze boundary"
)]
pub(crate) fn compile_time_series(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    lags: Vec<u32>,
    windows: Vec<u32>,
    velocity: bool,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    validate_shape(rows, cols, features.len(), target.len()).map_err(PyErr::from)?;
    let mut parsed = parse_engine_config(config)?;
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyValueError::new_err("rows exceed host address space"))?;
    let cols_usize = cols as usize;
    let base_candidate_cols = parsed.effective_feature_candidate_count(cols) as usize;
    let generated_limit = generated_feature_limit(
        parsed.budget.max_time_series_candidates,
        rows_usize,
        base_candidate_cols,
    );
    let source_features = if base_candidate_cols == 0 || generated_limit == 0 {
        Vec::new()
    } else {
        select_generated_source_features(
            &parsed,
            rows,
            cols,
            &features,
            &target,
            parsed.budget.top_k_features_for_time_series,
        )
        .map_err(PyErr::from)?
    };
    let (expanded, expanded_cols, descriptors) = if base_candidate_cols == 0 {
        (features, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_time_series_bounded(
            &features,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &source_features,
            &lags,
            &windows,
            velocity,
            generated_limit,
        )
        .map_err(PyErr::from)?
    };
    let artifact = compile_continuous_rows(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
        target,
    )
    .map_err(PyErr::from)?;
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        descriptors.iter().map(|descriptor| {
            let base = base_names
                .get(descriptor.base_feature as usize)
                .map(String::as_str)
                .unwrap_or("feature");
            gafime_cpu::time_series::feature_label(base, descriptor.op)
        }),
    );
    Ok((artifact, names))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_feature_names_are_unique_without_changing_clean_labels() {
        let mut names = vec![
            "signal".to_string(),
            "signal_lag1".to_string(),
            "signal_lag1#generated1".to_string(),
        ];

        append_unique_generated_names(
            &mut names,
            ["signal_lag1", "other_lag1", "signal_lag1"]
                .into_iter()
                .map(str::to_string),
        );

        assert_eq!(
            names,
            [
                "signal",
                "signal_lag1",
                "signal_lag1#generated1",
                "signal_lag1#generated2",
                "other_lag1",
                "signal_lag1#generated3",
            ]
        );
    }

    #[test]
    fn time_series_expansion_honors_source_and_candidate_caps() {
        let features = vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 4.0, 40.0, 400.0, 8.0, 80.0, 800.0,
        ];
        let (expanded, cols, descriptors) =
            expand_time_series_bounded(&features, 4, 3, 3, &[0], &[1], &[], true, 2).unwrap();

        assert_eq!(cols, 5);
        assert_eq!(descriptors.len(), 2);
        assert!(descriptors.iter().all(|item| item.base_feature == 0));
        assert!(expanded[3].is_nan() && expanded[4].is_nan());
        assert_eq!(&expanded[5..10], &[2.0, 20.0, 200.0, 1.0, 1.0]);

        let (_, uncapped_base_cols, none) =
            expand_time_series_bounded(&features, 4, 3, 3, &[0, 1, 2], &[1], &[2], true, 0)
                .unwrap();
        assert_eq!(uncapped_base_cols, 3);
        assert!(none.is_empty());
    }

    #[test]
    fn decision_path_expansion_honors_zero_and_feature_caps() {
        let features = vec![
            0.0, 100.0, 0.1, 90.0, 0.2, 80.0, 0.8, 70.0, 0.9, 60.0, 1.0, 50.0,
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut params = gafime_cpu::decision_path::DecisionPathParams {
            max_depth: 1,
            rounds: 1,
            max_paths: 0,
            max_bins: 0,
            min_leaf: 2,
            learning_rate: 1.0,
        };
        let (_, cols, paths) =
            expand_decision_path_bounded(&features, &target, 6, 2, 2, &[0], &params).unwrap();
        assert_eq!(cols, 2);
        assert!(paths.is_empty());

        params.max_paths = 1;
        let (_, cols, paths) =
            expand_decision_path_bounded(&features, &target, 6, 2, 2, &[0], &params).unwrap();
        assert_eq!(cols, 3);
        assert_eq!(paths.len(), 1);
        assert!(paths[0].nodes.iter().all(|node| node.feature == 0));
    }

    #[test]
    fn decision_path_permutations_are_rejected_until_rediscovery_is_available() {
        let mut config = EngineConfig {
            permutation_tests: 1,
            ..Default::default()
        };
        let error = validate_decision_path_permutation_config(&config).unwrap_err();
        assert!(error.to_string().contains("rediscovery"));

        config.permutation_tests = 0;
        validate_decision_path_permutation_config(&config).unwrap();
    }
}

/// decision_path family: discover depth-k GBDT conjunction paths (with residual
/// boosting) natively. CUDA can score the exactly equivalent complete unary
/// base-plus-path request through compact descriptors; all other shapes retain
/// the established membership expansion and continuous scoring path. Returns
/// (report, all_feature_names = base ++ path labels).
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, max_bins, min_leaf, learning_rate))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python decision-path API exposes each discovery parameter explicitly"
)]
pub(crate) fn analyze_decision_path(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    max_depth: u32,
    rounds: u32,
    max_paths: u32,
    max_bins: u32,
    min_leaf: u32,
    learning_rate: f32,
) -> PyResult<(PyContinuousReport, Vec<String>)> {
    validate_shape(rows, cols, features.len(), target.len()).map_err(PyErr::from)?;
    let mut parsed = parse_engine_config(config)?;
    validate_decision_path_permutation_config(&parsed).map_err(PyErr::from)?;
    let params = gafime_cpu::decision_path::DecisionPathParams {
        max_depth,
        rounds,
        max_paths,
        max_bins,
        min_leaf,
        learning_rate,
    };
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyValueError::new_err("rows exceed host address space"))?;
    let cols_usize = cols as usize;
    let base_candidate_cols = parsed.effective_feature_candidate_count(cols) as usize;
    let top_k_features = get_u32(config, "decision_path_top_k_features", 50)?;
    let discovery_features = if base_candidate_cols == 0 || params.max_paths == 0 {
        Vec::new()
    } else {
        select_generated_source_features(&parsed, rows, cols, &features, &target, top_k_features)
            .map_err(PyErr::from)?
    };
    let (paths, native_report) = if base_candidate_cols == 0 {
        (
            Vec::new(),
            analyze_continuous_rows_once(parsed, rows, cols, features, target)
                .map_err(PyErr::from)?,
        )
    } else {
        parsed.budget.max_feature_candidate = -2;
        let paths = discover_decision_paths_bounded(
            &features,
            &target,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?;
        #[cfg(feature = "local-cmake-experiment")]
        let local_report = local_cmake_experiment::try_analyze(
            &parsed,
            rows,
            cols_usize,
            base_candidate_cols,
            &features,
            &target,
            &paths,
        )
        .map_err(PyErr::from)?;
        #[cfg(not(feature = "local-cmake-experiment"))]
        let local_report: Option<ContinuousReport> = None;
        let native_report = match local_report {
            Some(report) => report,
            None => {
                let (expanded, expanded_cols) = materialize_decision_path_expansion(
                    &features,
                    rows_usize,
                    cols_usize,
                    base_candidate_cols,
                    &paths,
                )
                .map_err(PyErr::from)?;
                analyze_continuous_rows_once(
                    parsed,
                    rows,
                    expanded_column_count(expanded_cols)?,
                    expanded,
                    target,
                )
                .map_err(PyErr::from)?
            }
        };
        (paths, native_report)
    };
    let mut report = PyContinuousReport::from(native_report);
    report.decision_path_params = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            DecisionPathResultParams::from_path((base_candidate_cols + index) as u32, path)
        })
        .collect();
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        paths
            .iter()
            .map(|path| gafime_cpu::decision_path::path_label(&base_names, &path.nodes)),
    );
    Ok((report, names))
}

/// decision_path compile path: discover native GBDT conjunction paths, retain a
/// compact CUDA score artifact for the exact unary route, or otherwise return a
/// resident continuous artifact over the established expanded matrix.
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, max_bins, min_leaf, learning_rate))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python decision-path compile API mirrors the analyze boundary"
)]
pub(crate) fn compile_decision_path(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    max_depth: u32,
    rounds: u32,
    max_paths: u32,
    max_bins: u32,
    min_leaf: u32,
    learning_rate: f32,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    validate_shape(rows, cols, features.len(), target.len()).map_err(PyErr::from)?;
    let mut parsed = parse_engine_config(config)?;
    validate_decision_path_permutation_config(&parsed).map_err(PyErr::from)?;
    let params = gafime_cpu::decision_path::DecisionPathParams {
        max_depth,
        rounds,
        max_paths,
        max_bins,
        min_leaf,
        learning_rate,
    };
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyValueError::new_err("rows exceed host address space"))?;
    let cols_usize = cols as usize;
    let base_candidate_cols = parsed.effective_feature_candidate_count(cols) as usize;
    let top_k_features = get_u32(config, "decision_path_top_k_features", 50)?;
    let discovery_features = if base_candidate_cols == 0 || params.max_paths == 0 {
        Vec::new()
    } else {
        select_generated_source_features(&parsed, rows, cols, &features, &target, top_k_features)
            .map_err(PyErr::from)?
    };
    let (mut artifact, paths) = if base_candidate_cols == 0 {
        (
            compile_continuous_rows(parsed, rows, cols, features, target).map_err(PyErr::from)?,
            Vec::new(),
        )
    } else {
        parsed.budget.max_feature_candidate = -2;
        let paths = discover_decision_paths_bounded(
            &features,
            &target,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?;
        #[cfg(feature = "local-cmake-experiment")]
        let local_artifact = local_cmake_experiment::try_compile(
            &parsed,
            rows,
            cols_usize,
            base_candidate_cols,
            &features,
            &target,
            &paths,
        )
        .map_err(PyErr::from)?;
        #[cfg(not(feature = "local-cmake-experiment"))]
        let local_artifact: Option<PyCompiledContinuousArtifact> = None;
        if let Some(artifact) = local_artifact {
            (artifact, paths)
        } else {
            let (expanded, expanded_cols) = materialize_decision_path_expansion(
                &features,
                rows_usize,
                cols_usize,
                base_candidate_cols,
                &paths,
            )
            .map_err(PyErr::from)?;
            (
                compile_continuous_rows(
                    parsed,
                    rows,
                    expanded_column_count(expanded_cols)?,
                    expanded,
                    target,
                )
                .map_err(PyErr::from)?,
                paths,
            )
        }
    };
    artifact.decision_path_params = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            DecisionPathResultParams::from_path((base_candidate_cols + index) as u32, path)
        })
        .collect();
    artifact.target_updates_supported = false;
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        paths
            .iter()
            .map(|path| gafime_cpu::decision_path::path_label(&base_names, &path.nodes)),
    );
    Ok((artifact, names))
}
