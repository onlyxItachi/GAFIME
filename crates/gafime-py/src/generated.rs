use std::{cell::RefCell, collections::HashSet};

use gafime_cpu::result::OwnedResultTable;
use gafime_gpu_sys::{DecisionPathRtPolicy, GpuBackend, GpuSysError, OwnedGpuMatrix};
use gafime_orchestrator::{
    config::EngineConfig, plan::combos::legacy_unary_feature_order,
    prepare_continuous_execution_for_feature_orders, PreparedContinuousExecution,
};
use gafime_types::{
    GafimeDecisionPathTerm, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_DECISION_PATH_SIGN_GT,
    GAFIME_DECISION_PATH_SIGN_LE, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::artifact::{compile_continuous_rows, PyCompiledContinuousArtifact};
use crate::common::{
    report_from_table, validate_shape, ContinuousReport, DecisionPathResultParams, PyBoundaryError,
};
use crate::continuous::{
    analyze_continuous_rows_once, build_continuous_state, unary_strengths_from_table,
};
use crate::py_api::PyContinuousReport;
use crate::runtime::{get_u32, parse_engine_config, RuntimeCacheCounters};

pub(crate) struct CompactDecisionPathFallbackData {
    features: Vec<f32>,
    target: Vec<f32>,
    source_cols: usize,
    paths: Vec<gafime_cpu::decision_path::DecisionPath>,
}

/// CUDA-resident compact score state for the only decision-path request shape
/// that is exactly equivalent to the legacy expanded unary plan. The optional
/// fallback data is retained only by compiled artifacts so an ABI-level
/// unsupported response can materialize the established host path later.
pub(crate) struct CompactDecisionPathState {
    backend: RefCell<GpuBackend>,
    matrix: OwnedGpuMatrix,
    base_prepared: PreparedContinuousExecution,
    terms: Vec<GafimeDecisionPathTerm>,
    path_offsets: Vec<u32>,
    base_candidate_cols: u32,
    expanded_cols: u32,
    policy: DecisionPathRtPolicy,
    fallback: Option<CompactDecisionPathFallbackData>,
}

impl CompactDecisionPathState {
    pub(crate) fn uses_fp64_mi_accumulation(&self) -> bool {
        self.backend.borrow().uses_fp64_mi_accumulation()
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactDecisionPathFallbackReason {
    Backend,
    Metrics,
    Geometry,
    CandidateShape,
    CandidateOrder,
    Significance,
    Policy,
    Payload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactDecisionPathRoute {
    Compact,
    Fallback(CompactDecisionPathFallbackReason),
}

fn compact_decision_path_fallback_message(
    reason: CompactDecisionPathFallbackReason,
) -> &'static str {
    match reason {
        CompactDecisionPathFallbackReason::Backend => "the request is not on the CUDA backend",
        CompactDecisionPathFallbackReason::Metrics => {
            "the request contains a metric other than Pearson or R2"
        }
        CompactDecisionPathFallbackReason::Geometry => {
            "the request does not have finite RT-representable path geometry"
        }
        CompactDecisionPathFallbackReason::CandidateShape => {
            "the request contains mixed or higher-arity candidates"
        }
        CompactDecisionPathFallbackReason::CandidateOrder => {
            "the unary candidate limit would change base-plus-path ordering"
        }
        CompactDecisionPathFallbackReason::Significance => {
            "the request requires significance evaluation"
        }
        CompactDecisionPathFallbackReason::Policy => {
            "the request uses an incompatible execution policy"
        }
        CompactDecisionPathFallbackReason::Payload => {
            "the request exceeds the compact CUDA score ABI shape"
        }
    }
}

fn compact_expanded_column_count(base_candidate_cols: usize, path_count: usize) -> Option<u32> {
    base_candidate_cols
        .checked_add(path_count)
        .and_then(|cols| u32::try_from(cols).ok())
}

fn compact_decision_path_route(
    config: &EngineConfig,
    rows: u64,
    base_candidate_cols: usize,
    paths: &[gafime_cpu::decision_path::DecisionPath],
    candidate_features: &[f32],
    target: &[f32],
) -> CompactDecisionPathRoute {
    if config.backend_kind != GAFIME_BACKEND_CUDA {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Backend);
    }
    if config.graph_requested {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Policy);
    }
    if config.permutation_tests != 0 || config.num_repeats != 1 {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Significance);
    }
    if config.budget.max_comb_size != 1 {
        return CompactDecisionPathRoute::Fallback(
            CompactDecisionPathFallbackReason::CandidateShape,
        );
    }
    if config.metric_ids.is_empty()
        || config
            .metric_ids
            .iter()
            .any(|&metric| !matches!(metric, GAFIME_METRIC_PEARSON | GAFIME_METRIC_R2))
    {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Metrics);
    }
    let Some(expanded_cols) = compact_expanded_column_count(base_candidate_cols, paths.len())
    else {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Payload);
    };
    if base_candidate_cols == 0 || paths.is_empty() || rows > u64::from(u32::MAX) {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Payload);
    }
    if config.budget.max_combinations_per_k < u64::from(expanded_cols) {
        return CompactDecisionPathRoute::Fallback(
            CompactDecisionPathFallbackReason::CandidateOrder,
        );
    }
    let expected_feature_len = usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(base_candidate_cols));
    if expected_feature_len != Some(candidate_features.len())
        || usize::try_from(rows).ok() != Some(target.len())
    {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Payload);
    }
    if candidate_features.iter().any(|value| !value.is_finite())
        || target.iter().any(|value| !value.is_finite())
    {
        return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Geometry);
    }
    for path in paths {
        if path.nodes.is_empty() {
            return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Geometry);
        }
        let mut axes = HashSet::new();
        for node in &path.nodes {
            if node.feature as usize >= base_candidate_cols
                || !node.threshold.is_finite()
                || node.threshold.is_subnormal()
            {
                return CompactDecisionPathRoute::Fallback(
                    CompactDecisionPathFallbackReason::Geometry,
                );
            }
            axes.insert(node.feature);
        }
        if axes.len() > 3 {
            return CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Geometry);
        }
    }
    CompactDecisionPathRoute::Compact
}

fn compact_unsupported(
    policy: DecisionPathRtPolicy,
    reason: &'static str,
) -> Result<bool, PyBoundaryError> {
    match policy {
        DecisionPathRtPolicy::AllowSmFallback => Ok(false),
        DecisionPathRtPolicy::RequireRt => Err(PyBoundaryError::UnsupportedFeature(format!(
            "CUDA decision-path RT score is required but {reason}"
        ))),
    }
}

fn compact_decision_path_backend_available(
    backend: &GpuBackend,
    policy: DecisionPathRtPolicy,
) -> Result<bool, PyBoundaryError> {
    if !backend.supports_decision_path_score() {
        return compact_unsupported(
            policy,
            "the CUDA payload omits gafime_gpu_decision_path_score",
        );
    }
    let profile = match backend.device_profile() {
        Ok(profile) => profile,
        Err(error) => {
            return match policy {
                DecisionPathRtPolicy::AllowSmFallback => Ok(false),
                DecisionPathRtPolicy::RequireRt => Err(error.into()),
            }
        }
    };
    if profile.backend_kind != GAFIME_BACKEND_CUDA || !profile.optix_rt {
        return compact_unsupported(
            policy,
            "the selected CUDA device does not advertise OptiX RT",
        );
    }
    Ok(true)
}

fn compact_score_abi_result(
    result: Result<bool, GpuSysError>,
    policy: DecisionPathRtPolicy,
) -> Result<bool, PyBoundaryError> {
    match result {
        Ok(true) => Ok(true),
        Ok(false) => {
            compact_unsupported(policy, "the CUDA payload does not support compact scoring")
        }
        Err(GpuSysError::BackendStatus {
            status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            ..
        }) => compact_unsupported(policy, "the CUDA payload rejected the score request"),
        Err(error) => Err(error.into()),
    }
}

fn compact_decision_path_descriptors(
    paths: &[gafime_cpu::decision_path::DecisionPath],
) -> Result<(Vec<GafimeDecisionPathTerm>, Vec<u32>), PyBoundaryError> {
    let mut terms = Vec::new();
    let mut path_offsets = Vec::with_capacity(paths.len().saturating_add(1));
    path_offsets.push(0);
    for path in paths {
        if path.nodes.is_empty() {
            return Err(PyBoundaryError::InvalidInput(
                "compact decision-path scoring requires nonempty paths".to_string(),
            ));
        }
        for node in &path.nodes {
            terms.push(GafimeDecisionPathTerm {
                feature: node.feature,
                sign: match node.sign {
                    gafime_cpu::decision_path::SplitSign::Le => GAFIME_DECISION_PATH_SIGN_LE,
                    gafime_cpu::decision_path::SplitSign::Gt => GAFIME_DECISION_PATH_SIGN_GT,
                },
                threshold: node.threshold,
                ..Default::default()
            });
        }
        path_offsets.push(u32::try_from(terms.len()).map_err(|_| {
            PyBoundaryError::InvalidInput(
                "compact decision-path term count exceeds the CUDA ABI".to_string(),
            )
        })?);
    }
    Ok((terms, path_offsets))
}

#[allow(
    clippy::too_many_arguments,
    reason = "compact RT admission requires explicit config, shape, candidate prefix, data, paths, and execution policy"
)]
fn try_build_compact_decision_path_state(
    config: &EngineConfig,
    rows: u64,
    source_cols: usize,
    base_candidate_cols: usize,
    features: &[f32],
    target: &[f32],
    paths: &[gafime_cpu::decision_path::DecisionPath],
    policy: DecisionPathRtPolicy,
) -> Result<Option<CompactDecisionPathState>, PyBoundaryError> {
    if base_candidate_cols > source_cols {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path candidate prefix exceeds source columns".to_string(),
        ));
    }
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyBoundaryError::InvalidInput("rows exceed host address space".to_string()))?;
    let candidate_features =
        row_major_feature_prefix(features, rows_usize, source_cols, base_candidate_cols);
    if let CompactDecisionPathRoute::Fallback(reason) = compact_decision_path_route(
        config,
        rows,
        base_candidate_cols,
        paths,
        &candidate_features,
        target,
    ) {
        let _ = compact_unsupported(policy, compact_decision_path_fallback_message(reason))?;
        return Ok(None);
    }
    let base_candidate_cols = u32::try_from(base_candidate_cols).map_err(|_| {
        PyBoundaryError::InvalidInput("decision-path candidate count exceeds u32".to_string())
    })?;
    let expanded_cols = compact_expanded_column_count(base_candidate_cols as usize, paths.len())
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "decision-path expanded column count exceeds u32".to_string(),
            )
        })?;
    let metric_count = u32::try_from(config.metric_ids.len()).map_err(|_| {
        PyBoundaryError::InvalidInput("decision-path metric count exceeds u32".to_string())
    })?;
    let (terms, path_offsets) = compact_decision_path_descriptors(paths)?;

    let backend = GpuBackend::cuda_from_env(config.device_id)?;
    if !compact_decision_path_backend_available(&backend, policy)? {
        return Ok(None);
    }
    let matrix = backend.alloc_matrix(rows, base_candidate_cols)?;
    matrix.upload(&candidate_features, target)?;

    let mut base_config = config.clone();
    base_config.budget.max_comb_size = 1;
    base_config.permutation_tests = 0;
    base_config.num_repeats = 1;
    base_config.graph_requested = false;
    let base_features = (0..base_candidate_cols).collect::<Vec<_>>();
    let base_prepared = prepare_continuous_execution_for_feature_orders(
        &base_config,
        rows,
        base_candidate_cols,
        &base_features,
        &[],
        true,
        false,
    )?;
    if base_prepared.result_capacity() != u64::from(base_candidate_cols)
        || base_prepared.result_max_arity() != 1
        || base_prepared.result_metric_count() != metric_count
    {
        return Ok(None);
    }

    Ok(Some(CompactDecisionPathState {
        backend: RefCell::new(backend),
        matrix,
        base_prepared,
        terms,
        path_offsets,
        base_candidate_cols,
        expanded_cols,
        policy,
        fallback: None,
    }))
}

fn normalize_compact_decision_path_rows(
    result: &mut gafime_types::GafimeResultTable,
    expected_window: &gafime_types::GafimeResultTable,
    base_candidate_cols: u32,
    path_count: usize,
    metric_count: u32,
) -> bool {
    if result.abi_version != expected_window.abi_version
        || result.capacity != expected_window.capacity
        || result.row_count != path_count as u64
        || result.max_arity != 1
        || result.metric_count != metric_count
        || result.combo_indices != expected_window.combo_indices
        || result.metric_values != expected_window.metric_values
        || result.ranks != expected_window.ranks
        || result.families != expected_window.families
        || result.candidate_ids != expected_window.candidate_ids
        || result.row_flags != expected_window.row_flags
    {
        return false;
    }

    // SAFETY: OwnedResultTable::with_raw_rows_mut creates this bounded raw
    // view over owned Vec storage. The checks above verify that the ABI left
    // every owned buffer pointer unchanged before the exact path_count
    // elements are read or rewritten.
    unsafe {
        let combos = std::slice::from_raw_parts_mut(result.combo_indices, path_count);
        let ranks = std::slice::from_raw_parts_mut(result.ranks, path_count);
        let candidate_ids = std::slice::from_raw_parts_mut(result.candidate_ids, path_count);
        for path_index in 0..path_count {
            let local = path_index as u32;
            if combos[path_index] != local
                || ranks[path_index] != local
                || candidate_ids[path_index] != u64::from(local)
            {
                return false;
            }
        }
        for (path_index, combo) in combos.iter_mut().enumerate() {
            *combo = base_candidate_cols + path_index as u32;
        }
    }
    true
}

pub(crate) fn execute_compact_decision_path_state(
    config: &EngineConfig,
    rows: u64,
    state: &mut CompactDecisionPathState,
) -> Result<Option<ContinuousReport>, PyBoundaryError> {
    let path_count = state.path_offsets.len().saturating_sub(1);
    if path_count == 0 {
        return Ok(None);
    }
    let metric_count = u32::try_from(config.metric_ids.len()).map_err(|_| {
        PyBoundaryError::InvalidInput("decision-path metric count exceeds u32".to_string())
    })?;
    let result_capacity = u64::from(state.base_candidate_cols)
        .checked_add(path_count as u64)
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput("decision-path result capacity overflows".to_string())
        })?;
    let mut combined = OwnedResultTable::new(result_capacity, 1, metric_count);
    let base_execution = state.base_prepared.execute(
        &mut *state.backend.borrow_mut(),
        state.matrix.handle(),
        combined.raw_mut(),
    )?;
    if base_execution.rows_written != u64::from(state.base_candidate_cols)
        || combined.row_count() != state.base_candidate_cols as usize
    {
        return Err(PyBoundaryError::InvalidInput(
            "compact decision-path base plan did not emit every base candidate".to_string(),
        ));
    }

    let start = combined.row_count() as u64;
    let (score_result, path_rows) = combined
        .with_raw_rows_mut(
            start,
            path_count as u64,
            |raw| -> Result<bool, PyBoundaryError> {
                let expected_window = *raw;
                let scored = compact_score_abi_result(
                    state.backend.borrow_mut().decision_path_score_with_policy(
                        state.matrix.handle(),
                        &state.terms,
                        &state.path_offsets,
                        &config.metric_ids,
                        raw,
                        state.policy,
                    ),
                    state.policy,
                )?;
                if !scored {
                    return Ok(false);
                }
                Ok(normalize_compact_decision_path_rows(
                    raw,
                    &expected_window,
                    state.base_candidate_cols,
                    path_count,
                    metric_count,
                ))
            },
        )
        .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
    if !score_result? || path_rows != path_count as u64 {
        return Ok(None);
    }
    combined
        .commit_appended_rows(start, path_rows, start)
        .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
    if combined.row_count() != result_capacity as usize {
        return Err(PyBoundaryError::InvalidInput(
            "compact decision-path result rows are incomplete".to_string(),
        ));
    }
    Ok(Some(report_from_table(
        rows,
        state.expanded_cols,
        1,
        config.metric_ids.clone(),
        config.backend_kind,
        state.uses_fp64_mi_accumulation(),
        combined,
        Vec::new(),
    )))
}

pub(crate) fn fallback_compiled_compact_decision_path(
    artifact: &mut PyCompiledContinuousArtifact,
) -> Result<(), PyBoundaryError> {
    let mut compact = artifact.compact_decision_path_state.take().ok_or_else(|| {
        PyBoundaryError::InvalidInput("compiled compact decision-path state is missing".to_string())
    })?;
    let fallback = compact.fallback.take().ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "compiled compact decision-path fallback inputs are missing".to_string(),
        )
    })?;
    let expected_cols = compact.expanded_cols;
    let base_candidate_cols = compact.base_candidate_cols as usize;
    drop(compact);

    let rows = usize::try_from(artifact.rows)
        .map_err(|_| PyBoundaryError::InvalidInput("rows exceed host address space".to_string()))?;
    let (expanded, expanded_cols) = materialize_decision_path_expansion(
        &fallback.features,
        rows,
        fallback.source_cols,
        base_candidate_cols,
        &fallback.paths,
    )?;
    let expanded_cols = u32::try_from(expanded_cols).map_err(|_| {
        PyBoundaryError::InvalidInput("decision-path expanded column count exceeds u32".to_string())
    })?;
    if expanded_cols != expected_cols {
        return Err(PyBoundaryError::InvalidInput(
            "compact decision-path fallback changed the planned candidate columns".to_string(),
        ));
    }
    let state = build_continuous_state(
        &artifact.config,
        artifact.rows,
        expanded_cols,
        expanded,
        fallback.target,
    )?;
    artifact.cols = expanded_cols;
    artifact.max_arity = state.result_max_arity;
    artifact.state = Some(state);
    Ok(())
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

    use crate::artifact::execute_compiled_artifact;
    use crate::continuous::analyze_continuous_rows_once;
    use gafime_types::{GAFIME_BACKEND_CPU, GAFIME_METRIC_SPEARMAN};

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

    fn compact_decision_path_test_config() -> EngineConfig {
        EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            permutation_tests: 0,
            num_repeats: 1,
            budget: gafime_types::GafimeComputeBudget {
                max_comb_size: 1,
                max_combinations_per_k: 4,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn compact_decision_path_test_paths() -> Vec<gafime_cpu::decision_path::DecisionPath> {
        use gafime_cpu::decision_path::{DecisionPath, PathNode, SplitSign};

        vec![
            DecisionPath {
                nodes: vec![PathNode {
                    feature: 1,
                    threshold: 0.4,
                    sign: SplitSign::Gt,
                }],
                gain: 1.0,
                support: 3,
                round: 0,
            },
            DecisionPath {
                nodes: vec![
                    PathNode {
                        feature: 0,
                        threshold: 0.6,
                        sign: SplitSign::Le,
                    },
                    PathNode {
                        feature: 1,
                        threshold: 0.2,
                        sign: SplitSign::Gt,
                    },
                ],
                gain: 0.5,
                support: 2,
                round: 1,
            },
        ]
    }

    #[test]
    fn compact_decision_path_route_requires_an_exact_unary_plan() {
        let config = compact_decision_path_test_config();
        let paths = compact_decision_path_test_paths();
        let features = vec![0.1, 0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.9];
        let target = vec![0.0, 1.0, 0.5, 1.5];

        assert_eq!(
            compact_decision_path_route(&config, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Compact
        );

        let mut unsupported_metric = config.clone();
        unsupported_metric.metric_ids = vec![GAFIME_METRIC_SPEARMAN];
        assert_eq!(
            compact_decision_path_route(&unsupported_metric, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Metrics)
        );

        let mut higher_arity = config.clone();
        higher_arity.budget.max_comb_size = 2;
        assert_eq!(
            compact_decision_path_route(&higher_arity, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::CandidateShape)
        );

        let mut truncated = config.clone();
        truncated.budget.max_combinations_per_k = 3;
        assert_eq!(
            compact_decision_path_route(&truncated, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::CandidateOrder)
        );

        let mut stability = config.clone();
        stability.num_repeats = 2;
        assert_eq!(
            compact_decision_path_route(&stability, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Significance)
        );

        let mut graph = config.clone();
        graph.graph_requested = true;
        assert_eq!(
            compact_decision_path_route(&graph, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Policy)
        );

        let mut nonfinite = features.clone();
        nonfinite[0] = f32::NAN;
        assert_eq!(
            compact_decision_path_route(&config, 4, 2, &paths, &nonfinite, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Geometry)
        );

        let mut cpu = config.clone();
        cpu.backend_kind = GAFIME_BACKEND_CPU;
        assert_eq!(
            compact_decision_path_route(&cpu, 4, 2, &paths, &features, &target),
            CompactDecisionPathRoute::Fallback(CompactDecisionPathFallbackReason::Backend)
        );
    }

    #[test]
    fn compact_decision_path_descriptor_and_row_order_preserve_global_ids() {
        let paths = compact_decision_path_test_paths();
        let (terms, offsets) = compact_decision_path_descriptors(&paths).unwrap();
        assert_eq!(offsets, vec![0, 1, 3]);
        assert_eq!(
            terms
                .iter()
                .map(|term| (term.feature, term.sign, term.threshold))
                .collect::<Vec<_>>(),
            vec![
                (1, GAFIME_DECISION_PATH_SIGN_GT, 0.4),
                (0, GAFIME_DECISION_PATH_SIGN_LE, 0.6),
                (1, GAFIME_DECISION_PATH_SIGN_GT, 0.2),
            ]
        );

        let mut table = OwnedResultTable::new(4, 1, 2);
        table.raw_mut().row_count = 2;
        let (normalized, rows) = table
            .with_raw_rows_mut(2, 2, |raw| {
                let expected_window = *raw;
                // SAFETY: with_raw_rows_mut supplies a two-row bounded view.
                unsafe {
                    *raw.combo_indices.add(0) = 0;
                    *raw.combo_indices.add(1) = 1;
                    *raw.ranks.add(0) = 0;
                    *raw.ranks.add(1) = 1;
                    *raw.candidate_ids.add(0) = 0;
                    *raw.candidate_ids.add(1) = 1;
                }
                raw.row_count = 2;
                normalize_compact_decision_path_rows(raw, &expected_window, 2, 2, 2)
            })
            .unwrap();
        assert!(normalized);
        assert_eq!(rows, 2);
        table.commit_appended_rows(2, rows, 2).unwrap();
        assert_eq!(&table.combo_indices()[2..4], &[2, 3]);
        assert_eq!(&table.candidate_ids()[2..4], &[2, 3]);
        assert_eq!(&table.ranks()[2..4], &[2, 3]);

        let mut malformed = OwnedResultTable::new(2, 1, 2);
        let (normalized, _) = malformed
            .with_raw_rows_mut(0, 2, |raw| {
                let expected_window = *raw;
                raw.combo_indices = core::ptr::NonNull::<u32>::dangling().as_ptr();
                raw.row_count = 2;
                normalize_compact_decision_path_rows(raw, &expected_window, 0, 2, 2)
            })
            .unwrap();
        assert!(!normalized);
    }

    #[test]
    fn compact_decision_path_require_rt_fails_closed_on_unsupported_score() {
        let unsupported = || {
            Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_score",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            })
        };

        assert!(
            !compact_score_abi_result(unsupported(), DecisionPathRtPolicy::AllowSmFallback)
                .unwrap()
        );
        let error =
            compact_score_abi_result(unsupported(), DecisionPathRtPolicy::RequireRt).unwrap_err();
        assert!(error.to_string().contains("RT score is required"));

        let mut unsupported_metric = compact_decision_path_test_config();
        unsupported_metric.metric_ids = vec![GAFIME_METRIC_SPEARMAN];
        let paths = compact_decision_path_test_paths();
        let features = vec![0.1, 0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.9];
        let target = vec![0.0, 1.0, 0.5, 1.5];
        let fallback = try_build_compact_decision_path_state(
            &unsupported_metric,
            4,
            2,
            2,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::AllowSmFallback,
        )
        .expect("unsupported metrics should select the established fallback");
        assert!(fallback.is_none());
        let error = match try_build_compact_decision_path_state(
            &unsupported_metric,
            4,
            2,
            2,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::RequireRt,
        ) {
            Err(error) => error,
            Ok(_) => panic!("RequireRt must reject static compact-route incompatibility"),
        };
        assert!(error.to_string().contains("RT score is required"));
    }

    #[test]
    fn cuda_rt_compact_decision_path_matches_expanded_unary_when_available() {
        let Ok(backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_score() {
            return;
        }
        let Ok(profile) = backend.device_profile() else {
            return;
        };
        if !profile.optix_rt {
            return;
        }
        drop(backend);

        let rows = 24u64;
        let mut features = Vec::with_capacity(rows as usize * 2);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let first = (row % 6) as f32 / 5.0;
            let second = (row / 6) as f32 / 3.0;
            features.extend([first, second]);
            target.push(
                first * 0.25
                    + if first > 0.4 && second > 0.2 {
                        1.0
                    } else {
                        0.0
                    },
            );
        }
        let mut config = compact_decision_path_test_config();
        config.budget.max_combinations_per_k = 16;
        let paths = compact_decision_path_test_paths();
        let (expanded, expanded_cols) =
            materialize_decision_path_expansion(&features, rows as usize, 2, 2, &paths).unwrap();
        let expected = analyze_continuous_rows_once(
            config.clone(),
            rows,
            expanded_cols as u32,
            expanded,
            target.clone(),
        )
        .unwrap();

        let mut eager_state = try_build_compact_decision_path_state(
            &config,
            rows,
            2,
            2,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::AllowSmFallback,
        )
        .unwrap()
        .expect("RT payload should admit the finite unary compact route");
        let eager = execute_compact_decision_path_state(&config, rows, &mut eager_state)
            .unwrap()
            .expect("RT payload should score the finite unary compact route");

        let mut compiled_state = try_build_compact_decision_path_state(
            &config,
            rows,
            2,
            2,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::AllowSmFallback,
        )
        .unwrap()
        .expect("RT payload should build the compiled compact route");
        compiled_state.fallback = Some(CompactDecisionPathFallbackData {
            features: features.clone(),
            target: target.clone(),
            source_cols: 2,
            paths: paths.clone(),
        });
        let metric_ids = config.metric_ids.clone();
        let mut artifact = PyCompiledContinuousArtifact {
            config: config.clone(),
            rows,
            cols: compiled_state.expanded_cols,
            max_arity: 1,
            metric_ids,
            significance_top_n: config.significance_top_n,
            state: None,
            compact_decision_path_state: Some(compiled_state),
            runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
            decision_path_params: Vec::new(),
            target_updates_supported: false,
            closed: false,
        };
        let compiled = execute_compiled_artifact(&mut artifact).unwrap();
        assert!(artifact.compact_decision_path_state.is_some());

        for actual in [&eager, &compiled] {
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.table.combo_indices(), expected.table.combo_indices());
            assert_eq!(actual.table.candidate_ids(), expected.table.candidate_ids());
            assert_eq!(actual.table.ranks(), expected.table.ranks());
            for (actual, expected) in actual
                .table
                .metric_values()
                .iter()
                .zip(expected.table.metric_values())
            {
                assert!(
                    (actual - expected).abs() <= 1.1e-4,
                    "compact={actual}, expanded={expected}"
                );
            }
        }
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
        let compact_report = try_build_compact_decision_path_state(
            &parsed,
            rows,
            cols_usize,
            base_candidate_cols,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::AllowSmFallback,
        )
        .map_err(PyErr::from)?
        .map(|mut state| execute_compact_decision_path_state(&parsed, rows, &mut state))
        .transpose()
        .map_err(PyErr::from)?
        .flatten();
        let native_report = match compact_report {
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
        let compact = try_build_compact_decision_path_state(
            &parsed,
            rows,
            cols_usize,
            base_candidate_cols,
            &features,
            &target,
            &paths,
            DecisionPathRtPolicy::AllowSmFallback,
        )
        .map_err(PyErr::from)?;
        if let Some(mut compact) = compact {
            compact.fallback = Some(CompactDecisionPathFallbackData {
                features,
                target,
                source_cols: cols_usize,
                paths: paths.clone(),
            });
            let metric_ids = parsed.metric_ids.clone();
            let significance_top_n = parsed.significance_top_n;
            let artifact = PyCompiledContinuousArtifact {
                config: parsed,
                rows,
                cols: compact.expanded_cols,
                max_arity: 1,
                metric_ids,
                significance_top_n,
                state: None,
                compact_decision_path_state: Some(compact),
                runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
                decision_path_params: Vec::new(),
                target_updates_supported: false,
                closed: false,
            };
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
