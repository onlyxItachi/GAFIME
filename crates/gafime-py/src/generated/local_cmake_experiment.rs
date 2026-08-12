use std::{cell::RefCell, collections::HashSet};

use gafime_cpu::result::OwnedResultTable;
use gafime_gpu_sys::{DecisionPathRtPolicy, GpuBackend, GpuSysError, OwnedGpuMatrix};
use gafime_orchestrator::{
    config::EngineConfig, prepare_continuous_execution_for_feature_orders,
    PreparedContinuousExecution,
};
use gafime_types::{
    GafimeDecisionPathTerm, GAFIME_BACKEND_CUDA, GAFIME_DECISION_PATH_SIGN_GT,
    GAFIME_DECISION_PATH_SIGN_LE, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
#[cfg(test)]
use gafime_types::{GAFIME_BACKEND_CPU, GAFIME_METRIC_SPEARMAN};

#[cfg(test)]
use crate::{artifact::execute_compiled_artifact, continuous::analyze_continuous_rows_once};
use crate::{
    artifact::PyCompiledContinuousArtifact,
    common::{report_from_table, ContinuousReport, PyBoundaryError},
    continuous::build_continuous_state,
    runtime::RuntimeCacheCounters,
};

use super::{materialize_decision_path_expansion, row_major_feature_prefix};

pub(crate) struct CompactDecisionPathFallbackData {
    pub(crate) features: Vec<f32>,
    pub(crate) target: Vec<f32>,
    pub(crate) source_cols: usize,
    pub(crate) paths: Vec<gafime_cpu::decision_path::DecisionPath>,
}

/// CUDA-resident compact score state for the only decision-path request shape
/// that is exactly equivalent to the legacy expanded unary plan. The optional
/// fallback data is retained only by compiled artifacts so an ABI-level
/// unsupported response can materialize the established host path later.
pub(crate) struct CompactDecisionPathState {
    pub(crate) backend: RefCell<GpuBackend>,
    pub(crate) matrix: OwnedGpuMatrix,
    pub(crate) base_prepared: PreparedContinuousExecution,
    pub(crate) terms: Vec<GafimeDecisionPathTerm>,
    pub(crate) path_offsets: Vec<u32>,
    pub(crate) base_candidate_cols: u32,
    pub(crate) expanded_cols: u32,
    pub(crate) policy: DecisionPathRtPolicy,
    pub(crate) fallback: Option<CompactDecisionPathFallbackData>,
}

impl CompactDecisionPathState {
    pub(crate) fn uses_fp64_mi_accumulation(&self) -> bool {
        self.backend.borrow().uses_fp64_mi_accumulation()
    }
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
    if profile.backend_kind != GAFIME_BACKEND_CUDA || !profile.local_cmake_experiment_available() {
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
    // SAFETY: `combined` owns all output buffers and is uniquely borrowed;
    // `base_prepared` owns the live immutable descriptor graph for this call.
    let base_execution = unsafe {
        state.base_prepared.execute(
            &mut *state.backend.borrow_mut(),
            state.matrix.handle(),
            combined.raw_mut(),
        )
    }?;
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
                // SAFETY: `with_raw_rows_mut` created a checked, uniquely
                // borrowed output window over `combined`. The state-owned term
                // and offset slices remain immutable for this synchronous call.
                let scored = compact_score_abi_result(
                    unsafe {
                        state.backend.borrow_mut().decision_path_score_with_policy(
                            state.matrix.handle(),
                            &state.terms,
                            &state.path_offsets,
                            &config.metric_ids,
                            raw,
                            state.policy,
                        )
                    },
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
        None,
        combined,
        Vec::new(),
    )))
}

pub(crate) fn fallback_compiled_compact_decision_path(
    artifact: &mut PyCompiledContinuousArtifact,
) -> Result<(), PyBoundaryError> {
    let mut compact = artifact
        .local_cmake_experiment_state
        .take()
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "compiled compact decision-path state is missing".to_string(),
            )
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_analyze(
    config: &EngineConfig,
    rows: u64,
    source_cols: usize,
    base_candidate_cols: usize,
    features: &[f32],
    target: &[f32],
    paths: &[gafime_cpu::decision_path::DecisionPath],
) -> Result<Option<ContinuousReport>, PyBoundaryError> {
    let Some(mut state) = try_build_compact_decision_path_state(
        config,
        rows,
        source_cols,
        base_candidate_cols,
        features,
        target,
        paths,
        DecisionPathRtPolicy::AllowSmFallback,
    )?
    else {
        return Ok(None);
    };
    execute_compact_decision_path_state(config, rows, &mut state)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_compile(
    config: &EngineConfig,
    rows: u64,
    source_cols: usize,
    base_candidate_cols: usize,
    features: &[f32],
    target: &[f32],
    paths: &[gafime_cpu::decision_path::DecisionPath],
) -> Result<Option<PyCompiledContinuousArtifact>, PyBoundaryError> {
    let Some(mut state) = try_build_compact_decision_path_state(
        config,
        rows,
        source_cols,
        base_candidate_cols,
        features,
        target,
        paths,
        DecisionPathRtPolicy::AllowSmFallback,
    )?
    else {
        return Ok(None);
    };
    state.fallback = Some(CompactDecisionPathFallbackData {
        features: features.to_vec(),
        target: target.to_vec(),
        source_cols,
        paths: paths.to_vec(),
    });
    let metric_ids = config.metric_ids.clone();
    Ok(Some(PyCompiledContinuousArtifact {
        config: config.clone(),
        rows,
        cols: state.expanded_cols,
        max_arity: 1,
        metric_ids,
        significance_top_n: config.significance_top_n,
        state: None,
        local_cmake_experiment_state: Some(state),
        runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
        decision_path_params: Vec::new(),
        decision_path_state: None,
        feature_names: Vec::new(),
        target_updates_supported: false,
        closed: false,
    }))
}

pub(crate) fn execute_compiled(
    artifact: &mut PyCompiledContinuousArtifact,
) -> Result<Option<ContinuousReport>, PyBoundaryError> {
    let Some(state) = artifact.local_cmake_experiment_state.as_mut() else {
        return Ok(None);
    };
    *artifact.runtime_cache_counters.borrow_mut() = RuntimeCacheCounters::default();
    let result = execute_compact_decision_path_state(&artifact.config, artifact.rows, state)?;
    if result.is_some() {
        return Ok(result);
    }
    fallback_compiled_compact_decision_path(artifact)?;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        if !profile.local_cmake_experiment_available() {
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
            local_cmake_experiment_state: Some(compiled_state),
            runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
            decision_path_params: Vec::new(),
            decision_path_state: None,
            feature_names: Vec::new(),
            target_updates_supported: false,
            closed: false,
        };
        let compiled = execute_compiled_artifact(&mut artifact).unwrap();
        assert!(artifact.local_cmake_experiment_state.is_some());

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
}
