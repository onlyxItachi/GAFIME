use std::collections::HashSet;

use gafime_cpu::{
    kernels::MetricKernel,
    precision::{CpuPrecisionScalar, CpuPrecisionSlice, CpuPrecisionValues},
    result::PrecisionOwnedResultTable,
    significance::{self, ExpandedDecisionPathSearchSpec, SignificanceParams},
};
use gafime_orchestrator::{config::EngineConfig, plan::combos::legacy_unary_feature_order};
use gafime_types::{
    PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::artifact::{compile_continuous_input, PyCompiledContinuousArtifact};
use crate::common::{
    combo_from_table, validate_shape, ContinuousReport, DecisionPathResultParams,
    OwnedNumericInput, PyBoundaryError, ResultTableView, SignificanceEntry,
};
use crate::continuous::{
    analyze_continuous_input_once, bounded_ranked_indices,
    execute_device_decision_path_null_maxima, unary_strengths_from_table,
};
use crate::py_api::PyContinuousReport;
use crate::runtime::{get_u32, parse_engine_config};

#[cfg(feature = "local-cmake-experiment")]
pub(crate) mod local_cmake_experiment;

fn extract_generated_input(
    precision: PrecisionProfile,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<OwnedNumericInput> {
    match precision {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => OwnedNumericInput::from_f32(
            precision,
            features.extract::<Vec<f32>>()?,
            target.extract::<Vec<f32>>()?,
        )
        .map_err(PyErr::from),
        PrecisionProfile::Fp64 => OwnedNumericInput::from_f64(
            precision,
            features.extract::<Vec<f64>>()?,
            target.extract::<Vec<f64>>()?,
        )
        .map_err(PyErr::from),
    }
}

fn input_slices(input: &OwnedNumericInput) -> (CpuPrecisionSlice<'_>, CpuPrecisionSlice<'_>) {
    match input {
        OwnedNumericInput::F32 { features, target } => (
            CpuPrecisionSlice::F32(features),
            CpuPrecisionSlice::F32(target),
        ),
        OwnedNumericInput::F64 { features, target } => (
            CpuPrecisionSlice::F64(features),
            CpuPrecisionSlice::F64(target),
        ),
    }
}

fn column_major_feature_selection_precision(
    input: &OwnedNumericInput,
    rows: usize,
    source_cols: usize,
    selected_features: &[u32],
) -> Result<CpuPrecisionValues, PyBoundaryError> {
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
    match input {
        OwnedNumericInput::F32 { features, .. } => {
            let mut selected = Vec::with_capacity(capacity);
            for &feature in selected_features {
                let feature = feature as usize;
                for row in 0..rows {
                    selected.push(features[row * source_cols + feature]);
                }
            }
            Ok(CpuPrecisionValues::F32(selected))
        }
        OwnedNumericInput::F64 { features, .. } => {
            let mut selected = Vec::with_capacity(capacity);
            for &feature in selected_features {
                let feature = feature as usize;
                for row in 0..rows {
                    selected.push(features[row * source_cols + feature]);
                }
            }
            Ok(CpuPrecisionValues::F64(selected))
        }
    }
}

fn select_generated_source_features_precision(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    input: &OwnedNumericInput,
    top_k: u32,
) -> Result<Vec<u32>, PyBoundaryError> {
    let candidate_cols = config.effective_feature_candidate_count(cols);
    if candidate_cols == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let unary_features = legacy_unary_feature_order(
        candidate_cols,
        config.budget.max_combinations_per_k,
        &config.effective_planning_seed_words(),
    );
    if unary_features.is_empty() {
        return Ok(Vec::new());
    }
    let mut screening_config = config.clone();
    screening_config.budget.max_comb_size = 1;
    screening_config.permutation_tests = 0;
    screening_config.num_repeats = 1;
    screening_config.graph_requested = false;
    let screening = analyze_continuous_input_once(screening_config, rows, cols, input.clone())?;
    let mut strengths =
        unary_strengths_from_table(&screening.table, &unary_features, &config.metric_ids)?;
    if config.backend_kind == GAFIME_BACKEND_CPU {
        strengths.sort_by_feature();
    }
    Ok(strengths.into_ranked_features(top_k))
}

#[cfg(any(test, feature = "local-cmake-experiment"))]
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

#[cfg(test)]
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

#[cfg(any(test, feature = "local-cmake-experiment"))]
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
    reason = "profile-aware generated-family expansion keeps shape, selected sources, operations, and admission explicit"
)]
fn expand_time_series_precision(
    profile: PrecisionProfile,
    input: OwnedNumericInput,
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
        OwnedNumericInput,
        usize,
        Vec<gafime_cpu::time_series::TimeSeriesFeature>,
    ),
    PyBoundaryError,
> {
    if base_candidate_cols > source_cols
        || source_features
            .iter()
            .any(|&feature| feature as usize >= base_candidate_cols)
    {
        return Err(PyBoundaryError::InvalidInput(
            "time-series source feature is outside the candidate prefix".to_string(),
        ));
    }
    let selected =
        column_major_feature_selection_precision(&input, rows, source_cols, source_features)?;
    let selected_slice = match &selected {
        CpuPrecisionValues::F32(values) => CpuPrecisionSlice::F32(values),
        CpuPrecisionValues::F64(values) => CpuPrecisionSlice::F64(values),
    };
    let (generated, mut descriptors) = gafime_cpu::time_series::time_series_columns_precision(
        profile,
        selected_slice,
        rows,
        source_features.len(),
        lags,
        windows,
        velocity,
    )?;
    for descriptor in &mut descriptors {
        descriptor.base_feature = source_features
            .get(descriptor.base_feature as usize)
            .copied()
            .ok_or_else(|| {
                PyBoundaryError::InvalidInput(
                    "time-series generation returned an out-of-range source feature".to_string(),
                )
            })?;
    }
    descriptors.truncate(generated_limit);
    let generated_count = descriptors.len();
    let expanded_cols = base_candidate_cols
        .checked_add(generated_count)
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput("time-series expanded column count overflows".to_string())
        })?;
    let capacity = rows.checked_mul(expanded_cols).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "time-series expanded matrix exceeds host address space".to_string(),
        )
    })?;
    match (input, generated) {
        (OwnedNumericInput::F32 { features, target }, CpuPrecisionValues::F32(generated)) => {
            let mut expanded = vec![0.0f32; capacity];
            for row in 0..rows {
                let source = row * source_cols;
                let destination = row * expanded_cols;
                expanded[destination..destination + base_candidate_cols]
                    .copy_from_slice(&features[source..source + base_candidate_cols]);
                for generated_index in 0..generated_count {
                    expanded[destination + base_candidate_cols + generated_index] =
                        generated[generated_index * rows + row];
                }
            }
            Ok((
                OwnedNumericInput::F32 {
                    features: expanded,
                    target,
                },
                expanded_cols,
                descriptors,
            ))
        }
        (OwnedNumericInput::F64 { features, target }, CpuPrecisionValues::F64(generated)) => {
            let mut expanded = vec![0.0f64; capacity];
            for row in 0..rows {
                let source = row * source_cols;
                let destination = row * expanded_cols;
                expanded[destination..destination + base_candidate_cols]
                    .copy_from_slice(&features[source..source + base_candidate_cols]);
                for generated_index in 0..generated_count {
                    expanded[destination + base_candidate_cols + generated_index] =
                        generated[generated_index * rows + row];
                }
            }
            Ok((
                OwnedNumericInput::F64 {
                    features: expanded,
                    target,
                },
                expanded_cols,
                descriptors,
            ))
        }
        _ => Err(PyBoundaryError::InvalidInput(
            "time-series generated dtype does not match the requested precision profile"
                .to_string(),
        )),
    }
}

#[cfg(test)]
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

#[cfg(any(test, feature = "local-cmake-experiment"))]
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

fn discover_decision_paths_precision(
    profile: PrecisionProfile,
    input: &OwnedNumericInput,
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    discovery_features: &[u32],
    params: &gafime_cpu::decision_path::DecisionPathParams,
) -> Result<Vec<gafime_cpu::decision_path::PrecisionDecisionPath>, PyBoundaryError> {
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
    let selected =
        column_major_feature_selection_precision(input, rows, source_cols, discovery_features)?;
    let columns = match &selected {
        CpuPrecisionValues::F32(values) => CpuPrecisionSlice::F32(values),
        CpuPrecisionValues::F64(values) => CpuPrecisionSlice::F64(values),
    };
    let (_, target) = input_slices(input);
    let mut paths = gafime_cpu::decision_path::find_decision_paths_precision(
        profile,
        columns,
        rows,
        discovery_features.len(),
        target,
        params,
    )?;
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

fn materialize_decision_path_expansion_precision(
    profile: PrecisionProfile,
    input: OwnedNumericInput,
    rows: usize,
    source_cols: usize,
    base_candidate_cols: usize,
    paths: &[gafime_cpu::decision_path::PrecisionDecisionPath],
) -> Result<(OwnedNumericInput, usize), PyBoundaryError> {
    if base_candidate_cols > source_cols
        || paths
            .iter()
            .flat_map(|path| &path.nodes)
            .any(|node| node.feature as usize >= base_candidate_cols)
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
    let base_features = (0..base_candidate_cols)
        .map(|feature| feature as u32)
        .collect::<Vec<_>>();
    let base_columns =
        column_major_feature_selection_precision(&input, rows, source_cols, &base_features)?;
    let base_slice = match &base_columns {
        CpuPrecisionValues::F32(values) => CpuPrecisionSlice::F32(values),
        CpuPrecisionValues::F64(values) => CpuPrecisionSlice::F64(values),
    };
    let memberships = paths
        .iter()
        .map(|path| {
            gafime_cpu::decision_path::path_membership_precision(
                profile,
                base_slice,
                rows,
                &path.nodes,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let capacity = rows.checked_mul(expanded_cols).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "decision-path expanded matrix exceeds host address space".to_string(),
        )
    })?;
    match input {
        OwnedNumericInput::F32 { features, target } => {
            let mut expanded = vec![0.0f32; capacity];
            for row in 0..rows {
                let source = row * source_cols;
                let destination = row * expanded_cols;
                expanded[destination..destination + base_candidate_cols]
                    .copy_from_slice(&features[source..source + base_candidate_cols]);
                for (path_index, membership) in memberships.iter().enumerate() {
                    let CpuPrecisionValues::F32(membership) = membership else {
                        return Err(PyBoundaryError::InvalidInput(
                            "f32 decision-path expansion received f64 membership".to_string(),
                        ));
                    };
                    expanded[destination + base_candidate_cols + path_index] = membership[row];
                }
            }
            Ok((
                OwnedNumericInput::F32 {
                    features: expanded,
                    target,
                },
                expanded_cols,
            ))
        }
        OwnedNumericInput::F64 { features, target } => {
            let mut expanded = vec![0.0f64; capacity];
            for row in 0..rows {
                let source = row * source_cols;
                let destination = row * expanded_cols;
                expanded[destination..destination + base_candidate_cols]
                    .copy_from_slice(&features[source..source + base_candidate_cols]);
                for (path_index, membership) in memberships.iter().enumerate() {
                    let CpuPrecisionValues::F64(membership) = membership else {
                        return Err(PyBoundaryError::InvalidInput(
                            "fp64 decision-path expansion received f32 membership".to_string(),
                        ));
                    };
                    expanded[destination + base_candidate_cols + path_index] = membership[row];
                }
            }
            Ok((
                OwnedNumericInput::F64 {
                    features: expanded,
                    target,
                },
                expanded_cols,
            ))
        }
    }
}

fn precision_path_label(
    feature_names: &[String],
    path: &gafime_cpu::decision_path::PrecisionDecisionPath,
) -> String {
    let parts = path
        .nodes
        .iter()
        .map(|node| {
            let name = feature_names
                .get(node.feature as usize)
                .map(String::as_str)
                .unwrap_or("f");
            let op = match node.sign {
                gafime_cpu::decision_path::SplitSign::Le => "<=",
                gafime_cpu::decision_path::SplitSign::Gt => ">",
            };
            let threshold = match node.threshold {
                CpuPrecisionScalar::F32(value) => f64::from(value),
                CpuPrecisionScalar::F64(value) => value,
            };
            format!("{name}{op}{threshold:.4}")
        })
        .collect::<Vec<_>>();
    format!("path[{}]", parts.join(" & "))
}

fn decision_path_extremeness_f32(value: f32, kernel: MetricKernel) -> f32 {
    if !value.is_finite() {
        return f32::NEG_INFINITY;
    }
    match kernel {
        MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
        MetricKernel::MutualInfo | MetricKernel::R2 => value,
    }
}

fn decision_path_extremeness_f64(value: f64, kernel: MetricKernel) -> f64 {
    if !value.is_finite() {
        return f64::NEG_INFINITY;
    }
    match kernel {
        MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
        MetricKernel::MutualInfo | MetricKernel::R2 => value,
    }
}

fn update_decision_path_device_exceedances(
    counts: &mut [Vec<u32>],
    observed: &[CpuPrecisionValues],
    maxima: &CpuPrecisionValues,
    kernels: &[MetricKernel],
) -> Result<(), PyBoundaryError> {
    if counts.len() != observed.len() || counts.iter().any(|row| row.len() != kernels.len()) {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path device exceedance table has the wrong shape".to_string(),
        ));
    }
    match maxima {
        CpuPrecisionValues::F32(maxima) if maxima.len() == kernels.len() => {
            for (counts, observed) in counts.iter_mut().zip(observed) {
                let CpuPrecisionValues::F32(observed) = observed else {
                    return Err(PyBoundaryError::InvalidInput(
                        "fp32 decision-path device maximum received an f64 oracle".to_string(),
                    ));
                };
                if observed.len() != kernels.len() {
                    return Err(PyBoundaryError::InvalidInput(
                        "fp32 decision-path observed metric width is invalid".to_string(),
                    ));
                }
                for (metric_index, &kernel) in kernels.iter().enumerate() {
                    if decision_path_extremeness_f32(maxima[metric_index], kernel)
                        >= decision_path_extremeness_f32(observed[metric_index], kernel)
                    {
                        counts[metric_index] += 1;
                    }
                }
            }
            Ok(())
        }
        CpuPrecisionValues::F64(maxima) if maxima.len() == kernels.len() => {
            for (counts, observed) in counts.iter_mut().zip(observed) {
                let CpuPrecisionValues::F64(observed) = observed else {
                    return Err(PyBoundaryError::InvalidInput(
                        "mixed/fp64 decision-path device maximum received an f32 oracle"
                            .to_string(),
                    ));
                };
                if observed.len() != kernels.len() {
                    return Err(PyBoundaryError::InvalidInput(
                        "mixed/fp64 decision-path observed metric width is invalid".to_string(),
                    ));
                }
                for (metric_index, &kernel) in kernels.iter().enumerate() {
                    if decision_path_extremeness_f64(maxima[metric_index], kernel)
                        >= decision_path_extremeness_f64(observed[metric_index], kernel)
                    {
                        counts[metric_index] += 1;
                    }
                }
            }
            Ok(())
        }
        CpuPrecisionValues::F32(_) | CpuPrecisionValues::F64(_) => {
            Err(PyBoundaryError::InvalidInput(
                "decision-path device maximum width does not match its metrics".to_string(),
            ))
        }
    }
}

/// Original typed decision-path inputs and structural discovery policy retained
/// by a compiled artifact. Target replacement rebuilds a fresh expanded matrix
/// and execution state, then swaps it atomically into the public artifact.
pub(crate) struct PrecisionDecisionPathCompiledState {
    input: OwnedNumericInput,
    selection_config: EngineConfig,
    source_cols: u32,
    base_candidate_cols: usize,
    base_names: Vec<String>,
    top_k_features: u32,
    params: gafime_cpu::decision_path::DecisionPathParams,
    current_discovery_features: Vec<u32>,
    current_paths: Vec<gafime_cpu::decision_path::PrecisionDecisionPath>,
}

pub(crate) struct PrecisionDecisionPathRebuild {
    pub(crate) artifact: PyCompiledContinuousArtifact,
    pub(crate) paths: Vec<gafime_cpu::decision_path::PrecisionDecisionPath>,
    pub(crate) feature_names: Vec<String>,
}

impl PrecisionDecisionPathCompiledState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: OwnedNumericInput,
        selection_config: EngineConfig,
        source_cols: u32,
        base_candidate_cols: usize,
        base_names: Vec<String>,
        top_k_features: u32,
        params: gafime_cpu::decision_path::DecisionPathParams,
        current_discovery_features: Vec<u32>,
        current_paths: Vec<gafime_cpu::decision_path::PrecisionDecisionPath>,
    ) -> Self {
        Self {
            input,
            selection_config,
            source_cols,
            base_candidate_cols,
            base_names,
            top_k_features,
            params,
            current_discovery_features,
            current_paths,
        }
    }

    pub(crate) fn base_candidate_cols(&self) -> usize {
        self.base_candidate_cols
    }

    pub(crate) fn execution_config(&self, current_config: &EngineConfig) -> EngineConfig {
        let mut config = current_config.clone();
        // Expanded decision-path significance is target-dependent and runs
        // through the family-specific rediscovery oracle below. The ordinary
        // continuous significance path would incorrectly freeze observed paths.
        config.permutation_tests = 0;
        config.num_repeats = 1;
        config
    }

    fn device_permutation_pvalues(
        &self,
        current_config: &EngineConfig,
        metric_ids: &[u32],
        kernels: &[MetricKernel],
        observed: &[CpuPrecisionValues],
    ) -> Result<Vec<CpuPrecisionValues>, PyBoundaryError> {
        let requested = &self.selection_config;
        if requested.permutation_tests == 0 {
            return Err(PyBoundaryError::InvalidInput(
                "decision-path device p-values require at least one permutation".to_string(),
            ));
        }
        if current_config.precision != requested.precision {
            return Err(PyBoundaryError::InvalidInput(
                "decision-path significance precision differs from the compiled artifact"
                    .to_string(),
            ));
        }
        if !matches!(
            current_config.backend_kind,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) {
            return Err(PyBoundaryError::InvalidInput(
                "decision-path device p-values require an explicit GPU backend".to_string(),
            ));
        }
        if metric_ids.len() != kernels.len()
            || observed.iter().any(|values| match values {
                CpuPrecisionValues::F32(values) => {
                    requested.precision != PrecisionProfile::Fp32
                        || values.len() != metric_ids.len()
                }
                CpuPrecisionValues::F64(values) => {
                    requested.precision == PrecisionProfile::Fp32
                        || values.len() != metric_ids.len()
                }
            })
        {
            return Err(PyBoundaryError::InvalidInput(
                "decision-path device significance oracle does not match its precision lane"
                    .to_string(),
            ));
        }

        let (_, original_target) = input_slices(&self.input);
        let rows = self.input.target_len();
        let mut counts = vec![vec![0u32; metric_ids.len()]; observed.len()];
        let mut null_config = current_config.clone();
        null_config.metric_ids = metric_ids.to_vec();
        null_config.budget.max_feature_candidate = -2;
        null_config.graph_requested = false;

        for permutation_index in 0..requested.permutation_tests {
            let target = significance::precision_permutation_target(
                requested.precision,
                original_target,
                current_config.random_seed,
                permutation_index,
            )?;
            let permuted_input = match (&self.input, target) {
                (OwnedNumericInput::F32 { features, .. }, CpuPrecisionValues::F32(target)) => {
                    OwnedNumericInput::F32 {
                        features: features.clone(),
                        target,
                    }
                }
                (OwnedNumericInput::F64 { features, .. }, CpuPrecisionValues::F64(target)) => {
                    OwnedNumericInput::F64 {
                        features: features.clone(),
                        target,
                    }
                }
                _ => {
                    return Err(PyBoundaryError::InvalidInput(
                        "decision-path permutation target changed the resident dtype".to_string(),
                    ))
                }
            };
            let paths = discover_decision_paths_precision(
                requested.precision,
                &permuted_input,
                rows,
                self.source_cols as usize,
                self.base_candidate_cols,
                &self.current_discovery_features,
                &self.params,
            )?;
            let (expanded, expanded_cols) = materialize_decision_path_expansion_precision(
                requested.precision,
                permuted_input,
                rows,
                self.source_cols as usize,
                self.base_candidate_cols,
                &paths,
            )?;
            let maxima = execute_device_decision_path_null_maxima(
                &null_config,
                rows as u64,
                u32::try_from(expanded_cols).map_err(|_| {
                    PyBoundaryError::InvalidInput(
                        "decision-path null-family feature count exceeds u32".to_string(),
                    )
                })?,
                expanded,
            )?;
            update_decision_path_device_exceedances(&mut counts, observed, &maxima, kernels)?;
        }

        Ok(match requested.precision {
            PrecisionProfile::Fp32 => counts
                .into_iter()
                .map(|row| {
                    CpuPrecisionValues::F32(
                        row.into_iter()
                            .map(|count| {
                                (count + 1) as f32 / (requested.permutation_tests + 1) as f32
                            })
                            .collect(),
                    )
                })
                .collect(),
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => counts
                .into_iter()
                .map(|row| {
                    CpuPrecisionValues::F64(
                        row.into_iter()
                            .map(|count| {
                                f64::from(count + 1) / f64::from(requested.permutation_tests + 1)
                            })
                            .collect(),
                    )
                })
                .collect(),
        })
    }

    pub(crate) fn apply_significance(
        &self,
        current_config: &EngineConfig,
        report: &mut ContinuousReport,
    ) -> Result<(), PyBoundaryError> {
        let requested = &self.selection_config;
        if requested.permutation_tests == 0 && requested.num_repeats <= 1 {
            report.significance.clear();
            return Ok(());
        }
        if report.table.row_count() == 0 {
            report.significance.clear();
            return Ok(());
        }
        let order = bounded_ranked_indices(
            &report.table,
            &report.metric_ids,
            None,
            true,
            requested.significance_top_n.max(1) as usize,
        );
        let kernels = report
            .metric_ids
            .iter()
            .copied()
            .map(MetricKernel::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                PyBoundaryError::InvalidInput(
                    "unknown metric id for decision-path significance".to_string(),
                )
            })?;
        let mut combos = Vec::with_capacity(order.len());
        let mut observed = Vec::with_capacity(order.len());
        let mut candidate_ids = Vec::with_capacity(order.len());
        for &row in &order {
            combos.push(combo_from_table(&report.table, row).ok_or_else(|| {
                PyBoundaryError::InvalidInput(
                    "decision-path significance combo row is out of range".to_string(),
                )
            })?);
            candidate_ids.push(report.table.candidate_ids()[row]);
            let base = row * report.metric_ids.len();
            observed.push(match &report.table {
                PrecisionOwnedResultTable::Fp32(table) => CpuPrecisionValues::F32(
                    table.metric_values()[base..base + report.metric_ids.len()].to_vec(),
                ),
                PrecisionOwnedResultTable::F64 { table, .. } => CpuPrecisionValues::F64(
                    table.metric_values()[base..base + report.metric_ids.len()].to_vec(),
                ),
            });
        }
        let (source_features, target) = input_slices(&self.input);
        let planning_seed_words = requested.effective_planning_seed_words();
        let search = ExpandedDecisionPathSearchSpec {
            base_candidate_cols: self.base_candidate_cols as u32,
            discovery_features: &self.current_discovery_features,
            discovery: self.params,
            max_arity: requested.budget.max_comb_size,
            max_combinations_per_arity: requested.budget.max_combinations_per_k,
            top_features_for_higher_arity: requested.budget.top_features_for_higher_k,
            planning_seed_words: &planning_seed_words,
        };
        let params = SignificanceParams {
            permutation_tests: requested.permutation_tests,
            num_repeats: requested.num_repeats,
            random_seed: current_config.random_seed,
            mi_bins: requested.mi_bins,
            backend_kind: current_config.backend_kind,
            mi_approximate: requested.mi_approximate,
        };
        let evaluated = if current_config.backend_kind == GAFIME_BACKEND_CPU {
            significance::evaluate_precision_expanded_decision_path_family(
                requested.precision,
                source_features,
                self.input.target_len(),
                self.source_cols as usize,
                target,
                &self.current_paths,
                &combos,
                &observed,
                &candidate_ids,
                &kernels,
                &params,
                &search,
            )?
        } else {
            if !matches!(
                current_config.backend_kind,
                GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
            ) {
                return Err(PyBoundaryError::UnsupportedFeature(
                    "decision-path significance requires Core, CUDA, ROCm, or Metal".to_string(),
                ));
            }
            // Host work is limited to target permutation, path rediscovery,
            // membership materialization, and bootstrap statistics. Setting
            // permutations to zero is the fail-closed boundary that prevents
            // the CPU significance scorer from computing a GPU maxT maximum.
            let mut bootstrap_params = params;
            bootstrap_params.permutation_tests = 0;
            let mut evaluated = significance::evaluate_precision_expanded_decision_path_family(
                requested.precision,
                source_features,
                self.input.target_len(),
                self.source_cols as usize,
                target,
                &self.current_paths,
                &combos,
                &observed,
                &candidate_ids,
                &kernels,
                &bootstrap_params,
                &search,
            )?;
            if requested.permutation_tests > 0 {
                let pvalues = self.device_permutation_pvalues(
                    current_config,
                    &report.metric_ids,
                    &kernels,
                    &observed,
                )?;
                if pvalues.len() != evaluated.len() {
                    return Err(PyBoundaryError::InvalidInput(
                        "decision-path device p-values differ from the surfaced shortlist"
                            .to_string(),
                    ));
                }
                for (candidate, pvalues) in evaluated.iter_mut().zip(pvalues) {
                    candidate.pvalues = pvalues;
                }
            }
            evaluated
        };
        report.significance = order
            .into_iter()
            .zip(evaluated)
            .zip(candidate_ids)
            .map(|((row, significance), expected_candidate_id)| {
                if significance.candidate_id != expected_candidate_id {
                    return Err(PyBoundaryError::InvalidInput(
                        "decision-path significance candidate identity changed".to_string(),
                    ));
                }
                Ok(SignificanceEntry {
                    row,
                    pvalues: significance.pvalues,
                    means: significance.means,
                    stds: significance.stds,
                })
            })
            .collect::<Result<Vec<_>, PyBoundaryError>>()?;
        Ok(())
    }

    pub(crate) fn rebuild_target_f32(
        &mut self,
        current_config: &EngineConfig,
        target: Vec<f32>,
    ) -> Result<PrecisionDecisionPathRebuild, PyBoundaryError> {
        let OwnedNumericInput::F32 { features, .. } = &self.input else {
            return Err(PyBoundaryError::InvalidInput(
                "fp64 decision-path artifact requires an f64 target update".to_string(),
            ));
        };
        self.rebuild(
            current_config,
            OwnedNumericInput::F32 {
                features: features.clone(),
                target,
            },
        )
    }

    pub(crate) fn rebuild_target_f64(
        &mut self,
        current_config: &EngineConfig,
        target: Vec<f64>,
    ) -> Result<PrecisionDecisionPathRebuild, PyBoundaryError> {
        let OwnedNumericInput::F64 { features, .. } = &self.input else {
            return Err(PyBoundaryError::InvalidInput(
                "fp32/mixed decision-path artifact requires an f32 target update".to_string(),
            ));
        };
        self.rebuild(
            current_config,
            OwnedNumericInput::F64 {
                features: features.clone(),
                target,
            },
        )
    }

    pub(crate) fn rebuild_current(
        &mut self,
        current_config: &EngineConfig,
    ) -> Result<PrecisionDecisionPathRebuild, PyBoundaryError> {
        self.rebuild(current_config, self.input.clone())
    }

    fn rebuild(
        &mut self,
        current_config: &EngineConfig,
        input: OwnedNumericInput,
    ) -> Result<PrecisionDecisionPathRebuild, PyBoundaryError> {
        let mut selection_config = self.selection_config.clone();
        selection_config.random_seed = current_config.random_seed;
        selection_config.planning_seed_words = current_config.planning_seed_words.clone();
        selection_config.graph_requested = current_config.graph_requested;
        let rows = input.target_len() as u64;
        validate_shape(
            rows,
            self.source_cols,
            input.feature_len(),
            input.target_len(),
        )?;
        let source_features = if self.base_candidate_cols == 0 || self.params.max_paths == 0 {
            Vec::new()
        } else {
            select_generated_source_features_precision(
                &selection_config,
                rows,
                self.source_cols,
                &input,
                self.top_k_features,
            )?
        };
        let paths = discover_decision_paths_precision(
            selection_config.precision,
            &input,
            rows as usize,
            self.source_cols as usize,
            self.base_candidate_cols,
            &source_features,
            &self.params,
        )?;
        let (expanded, expanded_cols) = materialize_decision_path_expansion_precision(
            selection_config.precision,
            input.clone(),
            rows as usize,
            self.source_cols as usize,
            self.base_candidate_cols,
            &paths,
        )?;
        let mut execution_config = selection_config.clone();
        execution_config.budget.max_feature_candidate = -2;
        let artifact = compile_continuous_input(
            execution_config,
            rows,
            u32::try_from(expanded_cols).map_err(|_| {
                PyBoundaryError::InvalidInput(
                    "decision-path expanded feature count exceeds u32".to_string(),
                )
            })?,
            expanded,
        )?;
        let mut feature_names =
            self.base_names[..self.base_candidate_cols.min(self.base_names.len())].to_vec();
        append_unique_generated_names(
            &mut feature_names,
            paths
                .iter()
                .map(|path| precision_path_label(&self.base_names, path)),
        );
        self.input = input;
        self.selection_config = selection_config;
        self.current_discovery_features = source_features;
        self.current_paths.clone_from(&paths);
        Ok(PrecisionDecisionPathRebuild {
            artifact,
            paths,
            feature_names,
        })
    }
}

#[cfg(feature = "local-cmake-experiment")]
fn fp32_precision_paths_as_legacy(
    paths: &[gafime_cpu::decision_path::PrecisionDecisionPath],
) -> Result<Vec<gafime_cpu::decision_path::DecisionPath>, PyBoundaryError> {
    paths
        .iter()
        .map(|path| {
            let nodes = path
                .nodes
                .iter()
                .map(|node| {
                    let CpuPrecisionScalar::F32(threshold) = node.threshold else {
                        return Err(PyBoundaryError::InvalidInput(
                            "local RT experiment cannot consume an fp64 decision-path threshold"
                                .to_string(),
                        ));
                    };
                    Ok(gafime_cpu::decision_path::PathNode {
                        feature: node.feature,
                        threshold,
                        sign: node.sign,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let CpuPrecisionScalar::F32(gain) = path.gain else {
                return Err(PyBoundaryError::InvalidInput(
                    "local RT experiment cannot consume an fp64 decision-path score".to_string(),
                ));
            };
            Ok(gafime_cpu::decision_path::DecisionPath {
                nodes,
                gain,
                support: path.support,
                round: path.round,
            })
        })
        .collect()
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
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    lags: Vec<u32>,
    windows: Vec<u32>,
    velocity: bool,
) -> PyResult<(PyContinuousReport, Vec<String>)> {
    let mut parsed = parse_engine_config(config)?;
    let input = extract_generated_input(parsed.precision, features, target)?;
    validate_shape(rows, cols, input.feature_len(), input.target_len()).map_err(PyErr::from)?;
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
        select_generated_source_features_precision(
            &parsed,
            rows,
            cols,
            &input,
            parsed.budget.top_k_features_for_time_series,
        )
        .map_err(PyErr::from)?
    };
    let (expanded, expanded_cols, descriptors) = if base_candidate_cols == 0 {
        (input, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_time_series_precision(
            parsed.precision,
            input,
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
    let report = analyze_continuous_input_once(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
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
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    lags: Vec<u32>,
    windows: Vec<u32>,
    velocity: bool,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    let mut parsed = parse_engine_config(config)?;
    let input = extract_generated_input(parsed.precision, features, target)?;
    validate_shape(rows, cols, input.feature_len(), input.target_len()).map_err(PyErr::from)?;
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
        select_generated_source_features_precision(
            &parsed,
            rows,
            cols,
            &input,
            parsed.budget.top_k_features_for_time_series,
        )
        .map_err(PyErr::from)?
    };
    let (expanded, expanded_cols, descriptors) = if base_candidate_cols == 0 {
        (input, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_time_series_precision(
            parsed.precision,
            input,
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
    let artifact = compile_continuous_input(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
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
        let input = OwnedNumericInput::F32 {
            features: features.clone(),
            target: vec![0.0; 4],
        };
        let (expanded, cols, descriptors) = expand_time_series_precision(
            PrecisionProfile::Fp32,
            input,
            4,
            3,
            3,
            &[0],
            &[1],
            &[],
            true,
            2,
        )
        .unwrap();
        let OwnedNumericInput::F32 {
            features: expanded, ..
        } = expanded
        else {
            panic!("fp32 time-series expansion must retain f32 storage")
        };

        assert_eq!(cols, 5);
        assert_eq!(descriptors.len(), 2);
        assert!(descriptors.iter().all(|item| item.base_feature == 0));
        assert!(expanded[3].is_nan() && expanded[4].is_nan());
        assert_eq!(&expanded[5..10], &[2.0, 20.0, 200.0, 1.0, 1.0]);

        let (_, uncapped_base_cols, none) = expand_time_series_precision(
            PrecisionProfile::Fp32,
            OwnedNumericInput::F32 {
                features,
                target: vec![0.0; 4],
            },
            4,
            3,
            3,
            &[0, 1, 2],
            &[1],
            &[2],
            true,
            0,
        )
        .unwrap();
        assert_eq!(uncapped_base_cols, 3);
        assert!(none.is_empty());
    }

    #[test]
    fn fp64_time_series_expansion_never_stages_through_fp32() {
        let base = 1.0f64;
        let adjacent = f64::from_bits(base.to_bits() + 1);
        assert_eq!(base as f32, adjacent as f32);
        let input = OwnedNumericInput::F64 {
            features: vec![base, adjacent, 2.0],
            target: vec![0.0, 1.0, 2.0],
        };
        let (expanded, cols, descriptors) = expand_time_series_precision(
            PrecisionProfile::Fp64,
            input,
            3,
            1,
            1,
            &[0],
            &[1],
            &[],
            false,
            1,
        )
        .unwrap();
        let OwnedNumericInput::F64 { features, .. } = expanded else {
            panic!("fp64 time-series expansion must retain f64 storage")
        };

        assert_eq!(cols, 2);
        assert_eq!(descriptors.len(), 1);
        assert_eq!(features[2].to_bits(), adjacent.to_bits());
        assert_eq!(features[3].to_bits(), base.to_bits());
    }

    #[test]
    fn fp64_decision_path_membership_stays_in_f64_expansion() {
        let base = 1.0f64;
        let adjacent = f64::from_bits(base.to_bits() + 1);
        let path = gafime_cpu::decision_path::PrecisionDecisionPath {
            nodes: vec![gafime_cpu::decision_path::PrecisionPathNode {
                feature: 0,
                threshold: CpuPrecisionScalar::F64(base),
                sign: gafime_cpu::decision_path::SplitSign::Gt,
            }],
            gain: CpuPrecisionScalar::F64(1.0),
            support: 1,
            round: 0,
        };
        let input = OwnedNumericInput::F64 {
            features: vec![base, adjacent],
            target: vec![0.0, 1.0],
        };
        let (expanded, cols) = materialize_decision_path_expansion_precision(
            PrecisionProfile::Fp64,
            input,
            2,
            1,
            1,
            &[path],
        )
        .unwrap();
        let OwnedNumericInput::F64 { features, .. } = expanded else {
            panic!("fp64 decision-path expansion must retain f64 storage")
        };

        assert_eq!(cols, 2);
        assert_eq!(features, vec![base, 0.0, adjacent, 1.0]);
    }

    #[test]
    fn device_decision_path_exceedances_preserve_fp32_metric_semantics() {
        let kernels = [MetricKernel::Pearson, MetricKernel::R2];
        let observed = vec![
            CpuPrecisionValues::F32(vec![-0.75, 0.50]),
            CpuPrecisionValues::F32(vec![0.90, 0.20]),
        ];
        let mut counts = vec![vec![0; kernels.len()]; observed.len()];

        update_decision_path_device_exceedances(
            &mut counts,
            &observed,
            &CpuPrecisionValues::F32(vec![0.80, 0.30]),
            &kernels,
        )
        .unwrap();

        assert_eq!(counts, vec![vec![1, 0], vec![0, 1]]);
    }

    #[test]
    fn device_decision_path_exceedances_preserve_fp64_metric_semantics() {
        let kernels = [MetricKernel::Spearman, MetricKernel::MutualInfo];
        let observed = vec![CpuPrecisionValues::F64(vec![-0.75, 0.50])];
        let mut counts = vec![vec![0; kernels.len()]];

        update_decision_path_device_exceedances(
            &mut counts,
            &observed,
            &CpuPrecisionValues::F64(vec![0.80, 0.40]),
            &kernels,
        )
        .unwrap();

        assert_eq!(counts, vec![vec![1, 0]]);
    }

    #[test]
    fn decision_path_device_maxima_reject_core_before_building_state() {
        let error = execute_device_decision_path_null_maxima(
            &EngineConfig::default(),
            0,
            0,
            OwnedNumericInput::F32 {
                features: Vec::new(),
                target: Vec::new(),
            },
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("requires an explicit GPU backend"));
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
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    max_depth: u32,
    rounds: u32,
    max_paths: u32,
    max_bins: u32,
    min_leaf: u32,
    learning_rate: f64,
) -> PyResult<(PyContinuousReport, Vec<String>)> {
    let mut parsed = parse_engine_config(config)?;
    let input = extract_generated_input(parsed.precision, features, target)?;
    validate_shape(rows, cols, input.feature_len(), input.target_len()).map_err(PyErr::from)?;
    let retained_input = input.clone();
    let selection_config = parsed.clone();
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
        select_generated_source_features_precision(&parsed, rows, cols, &input, top_k_features)
            .map_err(PyErr::from)?
    };
    let (paths, native_report) = if base_candidate_cols == 0 {
        (
            Vec::new(),
            analyze_continuous_input_once(parsed, rows, cols, input).map_err(PyErr::from)?,
        )
    } else {
        parsed.budget.max_feature_candidate = -2;
        let paths = discover_decision_paths_precision(
            parsed.precision,
            &input,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?;
        let mut execution_config = parsed.clone();
        execution_config.permutation_tests = 0;
        execution_config.num_repeats = 1;
        #[cfg(feature = "local-cmake-experiment")]
        let local_report = if execution_config.precision == PrecisionProfile::Fp32 {
            let legacy_paths = fp32_precision_paths_as_legacy(&paths).map_err(PyErr::from)?;
            let OwnedNumericInput::F32 { features, target } = &input else {
                unreachable!("fp32 precision input was validated before local RT admission")
            };
            local_cmake_experiment::try_analyze(
                &execution_config,
                rows,
                cols_usize,
                base_candidate_cols,
                features,
                target,
                &legacy_paths,
            )
            .map_err(PyErr::from)?
        } else {
            None
        };
        #[cfg(not(feature = "local-cmake-experiment"))]
        let local_report: Option<ContinuousReport> = None;
        let native_report = match local_report {
            Some(report) => report,
            None => {
                let (expanded, expanded_cols) = materialize_decision_path_expansion_precision(
                    parsed.precision,
                    input,
                    rows_usize,
                    cols_usize,
                    base_candidate_cols,
                    &paths,
                )
                .map_err(PyErr::from)?;
                analyze_continuous_input_once(
                    execution_config,
                    rows,
                    expanded_column_count(expanded_cols)?,
                    expanded,
                )
                .map_err(PyErr::from)?
            }
        };
        (paths, native_report)
    };
    let mut native_report = native_report;
    if base_candidate_cols != 0 {
        let significance_state = PrecisionDecisionPathCompiledState::new(
            retained_input,
            selection_config.clone(),
            cols,
            base_candidate_cols,
            base_names.clone(),
            top_k_features,
            params,
            discovery_features,
            paths.clone(),
        );
        significance_state
            .apply_significance(&selection_config, &mut native_report)
            .map_err(PyErr::from)?;
    }
    let mut report = PyContinuousReport::from(native_report);
    report.decision_path_params = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            DecisionPathResultParams::from_precision_path(
                (base_candidate_cols + index) as u32,
                selection_config.precision,
                path,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyErr::from)?;
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        paths
            .iter()
            .map(|path| precision_path_label(&base_names, path)),
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
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
    base_names: Vec<String>,
    max_depth: u32,
    rounds: u32,
    max_paths: u32,
    max_bins: u32,
    min_leaf: u32,
    learning_rate: f64,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    let mut parsed = parse_engine_config(config)?;
    let input = extract_generated_input(parsed.precision, features, target)?;
    validate_shape(rows, cols, input.feature_len(), input.target_len()).map_err(PyErr::from)?;
    let retained_input = input.clone();
    let selection_config = parsed.clone();
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
        select_generated_source_features_precision(&parsed, rows, cols, &input, top_k_features)
            .map_err(PyErr::from)?
    };
    let (mut artifact, paths) = if base_candidate_cols == 0 {
        (
            compile_continuous_input(parsed, rows, cols, input).map_err(PyErr::from)?,
            Vec::new(),
        )
    } else {
        parsed.budget.max_feature_candidate = -2;
        let paths = discover_decision_paths_precision(
            parsed.precision,
            &input,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?;
        #[cfg(feature = "local-cmake-experiment")]
        let local_artifact = if parsed.precision == PrecisionProfile::Fp32 {
            let legacy_paths = fp32_precision_paths_as_legacy(&paths).map_err(PyErr::from)?;
            let OwnedNumericInput::F32 { features, target } = &input else {
                unreachable!("fp32 precision input was validated before local RT admission")
            };
            local_cmake_experiment::try_compile(
                &parsed,
                rows,
                cols_usize,
                base_candidate_cols,
                features,
                target,
                &legacy_paths,
            )
            .map_err(PyErr::from)?
        } else {
            None
        };
        #[cfg(not(feature = "local-cmake-experiment"))]
        let local_artifact: Option<PyCompiledContinuousArtifact> = None;
        if let Some(artifact) = local_artifact {
            (artifact, paths)
        } else {
            let (expanded, expanded_cols) = materialize_decision_path_expansion_precision(
                parsed.precision,
                input,
                rows_usize,
                cols_usize,
                base_candidate_cols,
                &paths,
            )
            .map_err(PyErr::from)?;
            (
                compile_continuous_input(
                    parsed,
                    rows,
                    expanded_column_count(expanded_cols)?,
                    expanded,
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
            DecisionPathResultParams::from_precision_path(
                (base_candidate_cols + index) as u32,
                selection_config.precision,
                path,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyErr::from)?;
    let mut names = if base_candidate_cols == 0 {
        base_names.clone()
    } else {
        base_names[..base_candidate_cols.min(base_names.len())].to_vec()
    };
    append_unique_generated_names(
        &mut names,
        paths
            .iter()
            .map(|path| precision_path_label(&base_names, path)),
    );
    artifact.feature_names.clone_from(&names);
    if base_candidate_cols != 0 {
        artifact.decision_path_state = Some(PrecisionDecisionPathCompiledState::new(
            retained_input,
            selection_config,
            cols,
            base_candidate_cols,
            base_names,
            top_k_features,
            params,
            discovery_features,
            paths.clone(),
        ));
    }
    artifact.target_updates_supported = true;
    Ok((artifact, names))
}
