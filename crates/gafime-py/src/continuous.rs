use std::cell::RefCell;

use gafime_cpu::{
    kernels::MetricKernel,
    matrix::CpuMatrix,
    result::OwnedResultTable,
    significance::{self, AdaptiveSearchSpec, SignificanceParams},
    CpuBackend,
};
use gafime_gpu_sys::{GpuBackend, OwnedGpuMatrix};
use gafime_orchestrator::config::EngineConfig;
use gafime_orchestrator::{
    plan::combos::{
        legacy_higher_feature_order, legacy_unary_feature_order,
        DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES,
    },
    prepare_continuous_execution_for_feature_orders, PreparedContinuousExecution,
};
use gafime_types::{
    GafimeRankSpec, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
};

use crate::common::{
    combo_from_table, metric_values_from_table, report_from_table, validate_metric_ids,
    validate_shape, ContinuousReport, InteractionPrecisionDiagnostic, PyBoundaryError,
    ResultTableView, SignificanceEntry,
};
use crate::runtime::RuntimeCacheCounters;

pub fn analyze_continuous_cpu_rows(
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Vec<u32>,
) -> Result<ContinuousReport, PyBoundaryError> {
    let metric_ids = if metric_ids.is_empty() {
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2]
    } else {
        metric_ids
    };
    let config = continuous_config_for_cpu(max_arity, max_combinations_per_k, metric_ids)?;
    analyze_continuous_rows_once(config, rows, cols, features, target)
}

pub(crate) fn continuous_config_for_cpu(
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Vec<u32>,
) -> Result<EngineConfig, PyBoundaryError> {
    if max_arity == 0 {
        return Err(PyBoundaryError::InvalidInput(
            "max_comb_size must be greater than zero".to_string(),
        ));
    }
    if max_combinations_per_k == 0 {
        return Err(PyBoundaryError::InvalidInput(
            "max_combinations_per_k must be greater than zero".to_string(),
        ));
    }
    let mut config = EngineConfig {
        backend_kind: GAFIME_BACKEND_CPU,
        metric_ids: validate_metric_ids(metric_ids)?,
        permutation_tests: 0,
        num_repeats: 1,
        ..Default::default()
    };
    config.budget.max_comb_size = max_arity;
    config.budget.max_combinations_per_k = max_combinations_per_k;
    // Significance is opt-in via the full-config `compile_continuous` path; the
    // low-level convenience entrypoints stay raw/fast (no permutation/stability).
    Ok(config)
}
pub(crate) fn analyze_continuous_rows_once(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<ContinuousReport, PyBoundaryError> {
    validate_shape(rows, cols, features.len(), target.len())?;
    let mut state = build_continuous_state(&config, rows, cols, features, target)?;
    let counters = RefCell::new(RuntimeCacheCounters::default());
    execute_continuous_state(
        &config,
        rows,
        cols,
        &config.metric_ids,
        config.significance_top_n,
        &mut state,
        &counters,
    )
}

pub(crate) fn build_continuous_state(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<ContinuousRunState, PyBoundaryError> {
    let needs_significance = config.permutation_tests > 0 || config.num_repeats > 1;
    // For GPU runs that still need CPU-side significance, build the host copy by
    // MOVING the ingest buffers into CpuMatrix after device upload borrows them.
    // CUDA payloads exposing the optional permutation p-value ABI can skip that
    // copy only for static families without bootstrap stability. Adaptive search
    // must retain the host matrix until the ABI can re-screen each permutation.
    let (backend, significance_matrix) =
        match config.backend_kind {
            GAFIME_BACKEND_CPU => (
                CompiledContinuousBackend::Cpu {
                    matrix: CpuMatrix::from_row_major(rows, cols, features, target)?,
                },
                None,
            ),
            GAFIME_BACKEND_CUDA => {
                let backend = GpuBackend::cuda_from_env(config.device_id)?;
                let matrix = backend.alloc_matrix(rows, cols)?;
                matrix.upload(&features, &target)?;
                let use_device_pvalues = needs_significance
                    && device_permutation_pvalues_are_valid(
                        config,
                        cols,
                        backend.supports_permutation_pvalues(),
                    );
                let significance_matrix =
                    if needs_significance && (!use_device_pvalues || config.num_repeats > 1) {
                        Some(CpuMatrix::from_row_major(rows, cols, features, target)?)
                    } else {
                        None
                    };
                (
                    CompiledContinuousBackend::Cuda {
                        backend: RefCell::new(backend),
                        matrix,
                    },
                    significance_matrix,
                )
            }
            GAFIME_BACKEND_ROCM => {
                let backend = GpuBackend::rocm_from_env(config.device_id)?;
                let matrix = backend.alloc_matrix(rows, cols)?;
                matrix.upload(&features, &target)?;
                let significance_matrix = if needs_significance {
                    Some(CpuMatrix::from_row_major(rows, cols, features, target)?)
                } else {
                    None
                };
                (
                    CompiledContinuousBackend::Rocm {
                        backend: RefCell::new(backend),
                        matrix,
                    },
                    significance_matrix,
                )
            }
            GAFIME_BACKEND_METAL => {
                let backend = GpuBackend::metal_from_env(config.device_id)?;
                let matrix = backend.alloc_matrix(rows, cols)?;
                matrix.upload(&features, &target)?;
                let significance_matrix = if needs_significance {
                    Some(CpuMatrix::from_row_major(rows, cols, features, target)?)
                } else {
                    None
                };
                (
                    CompiledContinuousBackend::Metal {
                        backend: RefCell::new(backend),
                        matrix,
                    },
                    significance_matrix,
                )
            }
            _ => return Err(PyBoundaryError::UnsupportedFeature(
                "continuous v1 execution currently supports CPU, explicit CUDA, ROCm, and Metal"
                    .to_string(),
            )),
        };
    let run = prepare_screened_continuous_execution(config, rows, cols, &backend)?;
    Ok(ContinuousRunState {
        significance_matrix,
        backend,
        primary: run.primary,
        screened: run.screened,
        result_capacity: run.result_capacity,
        result_max_arity: run.result_max_arity,
        result_metric_count: run.result_metric_count,
    })
}

fn has_adaptive_higher_order_search(config: &EngineConfig, cols: u32) -> bool {
    let candidate_cols = config.effective_feature_candidate_count(cols);
    let unary_count = u64::from(candidate_cols)
        .min(config.budget.max_combinations_per_k)
        .min(usize::MAX as u64) as usize;
    config.budget.max_comb_size >= 2
        && config.budget.top_features_for_higher_k >= 2
        && unary_count >= 2
}

fn device_permutation_pvalues_are_valid(
    config: &EngineConfig,
    cols: u32,
    backend_supports_pvalues: bool,
) -> bool {
    config.permutation_tests > 0
        && backend_supports_pvalues
        && !has_adaptive_higher_order_search(config, cols)
}

#[derive(Debug)]
struct ScreenedContinuousExecution {
    unary_table: OwnedResultTable,
    higher: PreparedContinuousExecution,
}

pub(crate) struct PreparedContinuousRun {
    /// Direct execution plan, or the complete screened family retained only for
    /// graph/significance protocols. Stateless screened runs do not build it.
    primary: Option<PreparedContinuousExecution>,
    screened: Option<ScreenedContinuousExecution>,
    result_capacity: u64,
    pub(crate) result_max_arity: u32,
    result_metric_count: u32,
}

impl PreparedContinuousRun {
    fn direct(prepared: PreparedContinuousExecution) -> Self {
        Self {
            result_capacity: prepared.result_capacity(),
            result_max_arity: prepared.result_max_arity(),
            result_metric_count: prepared.result_metric_count(),
            primary: Some(prepared),
            screened: None,
        }
    }

    fn empty(metric_count: usize) -> Self {
        Self {
            primary: None,
            screened: None,
            result_capacity: 0,
            result_max_arity: 1,
            result_metric_count: metric_count as u32,
        }
    }
}

fn execute_prepared_continuous(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
) -> Result<OwnedResultTable, PyBoundaryError> {
    let mut table = OwnedResultTable::new(
        prepared.result_capacity(),
        prepared.result_max_arity(),
        prepared.result_metric_count(),
    );
    execute_prepared_continuous_into(backend, prepared, table.raw_mut())?;
    Ok(table)
}

fn execute_prepared_continuous_into(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    result: &mut gafime_types::GafimeResultTable,
) -> Result<gafime_orchestrator::BackendExecutionStats, PyBoundaryError> {
    match backend {
        CompiledContinuousBackend::Cpu { matrix } => {
            let mut backend = CpuBackend;
            Ok(prepared.execute(&mut backend, &matrix.handle(), result)?)
        }
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => {
            Ok(prepared.execute(&mut *backend.borrow_mut(), matrix.handle(), result)?)
        }
    }
}

fn collect_interaction_diagnostics(
    backend: &CompiledContinuousBackend,
    table: &OwnedResultTable,
) -> Result<Option<Vec<InteractionPrecisionDiagnostic>>, PyBoundaryError> {
    let row_count = table.row_count();
    let max_arity = table.max_arity();
    let combo_len = row_count.checked_mul(max_arity).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "interaction diagnostic result shape exceeds host address space".to_string(),
        )
    })?;
    let combo_indices = &table.combo_indices()[..combo_len];
    match backend {
        CompiledContinuousBackend::Cpu { matrix } => Ok(Some(
            gafime_cpu::diagnostics::interaction_diagnostics(
                matrix,
                combo_indices,
                max_arity,
                row_count,
            )?
            .into_iter()
            .map(|diagnostic| InteractionPrecisionDiagnostic {
                overflow_row_count: diagnostic.overflow_row_count,
                source_nonfinite: diagnostic.source_nonfinite,
            })
            .collect(),
        )),
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => Ok(backend
            .borrow()
            .interaction_diagnostics(matrix.handle(), combo_indices, max_arity as u32, row_count)?
            .map(|diagnostics| {
                diagnostics
                    .into_iter()
                    .map(|diagnostic| InteractionPrecisionDiagnostic {
                        overflow_row_count: diagnostic.overflow_row_count,
                        source_nonfinite: diagnostic.source_nonfinite,
                    })
                    .collect()
            })),
    }
}

fn execute_continuous_plan_set(
    backend: &CompiledContinuousBackend,
    primary: Option<&PreparedContinuousExecution>,
    screened: Option<&ScreenedContinuousExecution>,
    result_capacity: u64,
    result_max_arity: u32,
    result_metric_count: u32,
) -> Result<OwnedResultTable, PyBoundaryError> {
    if result_capacity == 0 {
        return Ok(OwnedResultTable::new(
            0,
            result_max_arity,
            result_metric_count,
        ));
    }
    if let Some(screened) = screened {
        let mut combined =
            OwnedResultTable::new(result_capacity, result_max_arity, result_metric_count);
        combined
            .append_rows_from(&screened.unary_table, 0)
            .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
        let start = screened.unary_table.row_count() as u64;
        let (execution, row_count) = combined
            .with_raw_rows_mut(start, screened.higher.result_capacity(), |raw| {
                execute_prepared_continuous_into(backend, &screened.higher, raw)
            })
            .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
        let execution = execution?;
        if execution.rows_written != row_count {
            return Err(PyBoundaryError::InvalidInput(
                "backend result count differs from screened higher-order rows".to_string(),
            ));
        }
        combined
            .commit_appended_rows(start, row_count, start)
            .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
        return Ok(combined);
    }
    let prepared = primary.ok_or_else(|| {
        PyBoundaryError::InvalidInput("direct continuous plan is missing".to_string())
    })?;
    execute_prepared_continuous(backend, prepared)
}

fn result_table_storage_bytes(capacity: u64, max_arity: u32, metric_count: u32) -> u64 {
    const U32_BYTES: u64 = 4;
    const U64_BYTES: u64 = 8;
    let row_bytes = u64::from(max_arity)
        .saturating_mul(U32_BYTES)
        .saturating_add(u64::from(metric_count).saturating_mul(U32_BYTES))
        .saturating_add(U32_BYTES) // rank
        .saturating_add(U32_BYTES) // family
        .saturating_add(U64_BYTES) // candidate id
        .saturating_add(U32_BYTES); // row flags
    capacity.saturating_mul(row_bytes)
}

fn screened_candidate_storage_bytes(
    unary_rows: u64,
    higher_descriptor_words: u64,
    combined_rows: u64,
    combined_max_arity: u32,
    metric_count: u32,
    retained_complete_descriptor_words: u64,
) -> u64 {
    higher_descriptor_words
        .saturating_mul(core::mem::size_of::<u32>() as u64)
        .saturating_add(result_table_storage_bytes(unary_rows, 1, metric_count))
        .saturating_add(result_table_storage_bytes(
            combined_rows,
            combined_max_arity,
            metric_count,
        ))
        .saturating_add(
            retained_complete_descriptor_words.saturating_mul(core::mem::size_of::<u32>() as u64),
        )
}

pub(crate) fn prepare_screened_continuous_execution(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    backend: &CompiledContinuousBackend,
) -> Result<PreparedContinuousRun, PyBoundaryError> {
    let planning_seed_words = config.effective_planning_seed_words();
    let candidate_cols = config.effective_feature_candidate_count(cols);
    if candidate_cols == 0 {
        return Ok(PreparedContinuousRun::empty(config.metric_ids.len()));
    }
    let unary_features = legacy_unary_feature_order(
        candidate_cols,
        config.budget.max_combinations_per_k,
        &planning_seed_words,
    );
    let needs_screening = config.budget.max_comb_size >= 2
        && config.budget.top_features_for_higher_k >= 2
        && unary_features.len() >= 2;
    if !needs_screening {
        let prepared = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            &[],
            true,
            true,
        )?;
        return Ok(PreparedContinuousRun::direct(prepared));
    }

    let unary_prepared = prepare_continuous_execution_for_feature_orders(
        config,
        rows,
        cols,
        &unary_features,
        &[],
        true,
        false,
    )?;
    let unary_table = execute_prepared_continuous(backend, &unary_prepared)?;
    let mut unary_strengths =
        unary_strengths_from_table(&unary_table, &unary_features, &config.metric_ids)?;
    if config.backend_kind == GAFIME_BACKEND_CPU {
        // Published Core inserted scheduler results by ascending feature ID,
        // while GPU mappings retained unary-plan order. Stable score sorting
        // therefore used this order only as the Core tie-break contract.
        unary_strengths.sort_by_key(|(feature, _)| *feature);
    }
    let higher_features = legacy_higher_feature_order(
        candidate_cols,
        config.budget.max_combinations_per_k,
        config.budget.top_features_for_higher_k,
        &planning_seed_words,
        &unary_strengths,
    );
    if config.graph_requested {
        let prepared = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            &higher_features,
            true,
            true,
        )?;
        return Ok(PreparedContinuousRun::direct(prepared));
    }
    let higher = prepare_continuous_execution_for_feature_orders(
        config,
        rows,
        cols,
        &[],
        &higher_features,
        false,
        false,
    )?;
    let result_capacity = (unary_table.row_count() as u64)
        .checked_add(higher.result_capacity())
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput("continuous result capacity overflow".to_string())
        })?;
    let result_max_arity = higher
        .result_max_arity()
        .max(unary_table.max_arity() as u32);
    let result_metric_count = higher.result_metric_count();
    let needs_null_family = config.permutation_tests > 0 || config.num_repeats > 1;
    let complete_descriptor_words = higher
        .plan()
        .logical_descriptor_words()
        .saturating_add(unary_table.row_count() as u64);
    let retained_complete_descriptor_words = if needs_null_family {
        complete_descriptor_words
    } else {
        0
    };
    let screened_storage = screened_candidate_storage_bytes(
        unary_table.row_count() as u64,
        higher.plan().materialized_descriptor_words() as u64,
        result_capacity,
        result_max_arity,
        result_metric_count,
        retained_complete_descriptor_words,
    );
    if screened_storage > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES {
        drop(higher);
        drop(unary_table);
        let primary = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            &higher_features,
            true,
            true,
        )?;
        return Ok(PreparedContinuousRun::direct(primary));
    }
    let primary = if needs_null_family {
        Some(prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            &higher_features,
            true,
            true,
        )?)
    } else {
        None
    };
    Ok(PreparedContinuousRun {
        primary,
        screened: Some(ScreenedContinuousExecution {
            unary_table,
            higher,
        }),
        result_capacity,
        result_max_arity,
        result_metric_count,
    })
}

pub(crate) fn unary_strengths_from_table(
    table: &OwnedResultTable,
    unary_features: &[u32],
    metric_ids: &[u32],
) -> Result<Vec<(u32, f32)>, PyBoundaryError> {
    if table.row_count() != unary_features.len() || table.metric_count() != metric_ids.len() {
        return Err(PyBoundaryError::InvalidInput(
            "unary screening result shape does not match its plan".to_string(),
        ));
    }
    let mut strengths = Vec::with_capacity(unary_features.len());
    for (row, &feature) in unary_features.iter().enumerate() {
        let values = metric_values_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("unary screening metric row is missing".to_string())
        })?;
        let mut strength = None::<f32>;
        for (&metric_id, value) in metric_ids.iter().zip(values) {
            let candidate = if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
                value.abs()
            } else {
                value
            };
            strength = Some(strength.map_or(candidate, |current| current.max(candidate)));
        }
        strengths.push((feature, strength.unwrap_or(0.0)));
    }
    Ok(strengths)
}

pub(crate) fn execute_continuous_state(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    metric_ids: &[u32],
    significance_top_n: u32,
    state: &mut ContinuousRunState,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
) -> Result<ContinuousReport, PyBoundaryError> {
    *runtime_cache_counters.borrow_mut() = RuntimeCacheCounters::default();
    let table = execute_continuous_plan_set(
        &state.backend,
        state.primary.as_ref(),
        state.screened.as_ref(),
        state.result_capacity,
        state.result_max_arity,
        state.result_metric_count,
    )?;
    let interaction_diagnostics = collect_interaction_diagnostics(&state.backend, &table)?;

    if table.row_count() == 0 {
        return Ok(report_from_table(
            rows,
            cols,
            state.result_max_arity,
            metric_ids.to_vec(),
            config.backend_kind,
            state.backend.uses_fp64_mi_accumulation(),
            interaction_diagnostics,
            table,
            Vec::new(),
        ));
    }

    let significance = compute_significance(
        config,
        cols,
        metric_ids,
        significance_top_n,
        runtime_cache_counters,
        state,
        &table,
    )?;

    Ok(report_from_table(
        rows,
        cols,
        state.result_max_arity,
        metric_ids.to_vec(),
        config.backend_kind,
        state.backend.uses_fp64_mi_accumulation(),
        interaction_diagnostics,
        table,
        significance,
    ))
}

fn compute_significance(
    config: &EngineConfig,
    cols: u32,
    metric_ids: &[u32],
    significance_top_n: u32,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
    state: &mut ContinuousRunState,
    table: &OwnedResultTable,
) -> Result<Vec<SignificanceEntry>, PyBoundaryError> {
    let adaptive_search = has_adaptive_higher_order_search(config, cols);
    let generated_family = state
        .complete_family()
        .is_ok_and(|prepared| prepared.plan().uses_generated_descriptors());
    let device_significance = if adaptive_search || generated_family {
        compute_host_orchestrated_gpu_permutation_pvalues(
            config,
            metric_ids,
            significance_top_n,
            runtime_cache_counters,
            state,
            table,
            adaptive_search,
        )?
    } else {
        let native = compute_gpu_permutation_pvalues(
            config,
            cols,
            metric_ids,
            significance_top_n,
            runtime_cache_counters,
            state,
            table,
        )?;
        if native.is_some() {
            native
        } else {
            compute_host_orchestrated_gpu_permutation_pvalues(
                config,
                metric_ids,
                significance_top_n,
                runtime_cache_counters,
                state,
                table,
                false,
            )?
        }
    };
    if let Some(mut significance) = device_significance {
        if config.num_repeats > 1 {
            let device_counters = *runtime_cache_counters.borrow();
            let mut stability_config = config.clone();
            stability_config.permutation_tests = 0;
            let stability = compute_cpu_significance(
                &stability_config,
                metric_ids,
                significance_top_n,
                runtime_cache_counters,
                state,
                table,
            )?;
            merge_significance_stability(&mut significance, stability)?;
            *runtime_cache_counters.borrow_mut() = device_counters;
        }
        return Ok(significance);
    }
    compute_cpu_significance(
        config,
        metric_ids,
        significance_top_n,
        runtime_cache_counters,
        state,
        table,
    )
}

pub(crate) fn compare_ranked_rows(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    left: usize,
    right: usize,
    metric_index: Option<usize>,
    descending: bool,
) -> std::cmp::Ordering {
    let left_value = rank_value_at(table, metric_ids, left, metric_index);
    let right_value = rank_value_at(table, metric_ids, right, metric_index);
    compare_rank_values(left_value, right_value, descending)
        .then_with(|| table.candidate_ids()[left].cmp(&table.candidate_ids()[right]))
        .then_with(|| left.cmp(&right))
}

pub(crate) fn bounded_ranked_indices(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    metric_index: Option<usize>,
    descending: bool,
    limit: usize,
) -> Vec<usize> {
    let limit = limit.min(table.row_count());
    let mut selected = Vec::with_capacity(limit);
    for row in 0..table.row_count() {
        let mut low = 0;
        let mut high = selected.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if compare_ranked_rows(
                table,
                metric_ids,
                selected[middle],
                row,
                metric_index,
                descending,
            ) == std::cmp::Ordering::Greater
            {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        if low < limit {
            selected.insert(low, row);
            if selected.len() > limit {
                selected.pop();
            }
        }
    }
    selected
}

fn significance_order(
    table: &OwnedResultTable,
    metric_ids: &[u32],
    significance_top_n: u32,
) -> Vec<usize> {
    bounded_ranked_indices(
        table,
        metric_ids,
        None,
        true,
        significance_top_n.max(1) as usize,
    )
}

fn metric_extremeness(metric_id: u32, value: f32) -> f32 {
    if !value.is_finite() {
        return f32::NEG_INFINITY;
    }
    if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
        value.abs()
    } else {
        value
    }
}

fn update_gpu_target(
    backend: &CompiledContinuousBackend,
    target: &[f32],
) -> Result<(), PyBoundaryError> {
    match backend {
        CompiledContinuousBackend::Cuda { matrix, .. }
        | CompiledContinuousBackend::Rocm { matrix, .. }
        | CompiledContinuousBackend::Metal { matrix, .. } => {
            matrix.update_target(target).map_err(PyBoundaryError::from)
        }
        CompiledContinuousBackend::Cpu { .. } => Err(PyBoundaryError::InvalidInput(
            "device significance requires a GPU matrix".to_string(),
        )),
    }
}

fn require_device_ranking(backend: &CompiledContinuousBackend) -> Result<(), PyBoundaryError> {
    let supports_device_ranking = match backend {
        CompiledContinuousBackend::Cpu { .. } => false,
        CompiledContinuousBackend::Cuda { backend, .. }
        | CompiledContinuousBackend::Rocm { backend, .. }
        | CompiledContinuousBackend::Metal { backend, .. } => {
            backend.borrow().graph_capability()?.supports_device_ranking != 0
        }
    };
    if supports_device_ranking {
        Ok(())
    } else {
        Err(PyBoundaryError::UnsupportedFeature(
            "GPU permutation significance requires device-ranking capability".to_string(),
        ))
    }
}

fn merge_significance_stability(
    permutation: &mut [SignificanceEntry],
    stability: Vec<SignificanceEntry>,
) -> Result<(), PyBoundaryError> {
    if permutation.len() != stability.len() {
        return Err(PyBoundaryError::InvalidInput(
            "device permutation and host stability rows differ in length".to_string(),
        ));
    }
    for entry in permutation {
        let stable = stability
            .iter()
            .find(|candidate| candidate.row == entry.row)
            .ok_or_else(|| {
                PyBoundaryError::InvalidInput(
                    "device permutation row is missing from host stability".to_string(),
                )
            })?;
        entry.means.clone_from(&stable.means);
        entry.stds.clone_from(&stable.stds);
    }
    Ok(())
}

fn ranked_metric_value(
    result: &OwnedResultTable,
    metric_index: usize,
) -> Result<f32, PyBoundaryError> {
    if result.row_count() == 0 {
        return Ok(f32::NEG_INFINITY);
    }
    let values = metric_values_from_table(result, 0).ok_or_else(|| {
        PyBoundaryError::InvalidInput("ranked significance metric row is missing".to_string())
    })?;
    values.get(metric_index).copied().ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "ranked significance result has the wrong metric width".to_string(),
        )
    })
}

fn execute_ranked_metric_extremum(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    metric_id: u32,
    descending: bool,
) -> Result<f32, PyBoundaryError> {
    let metric_index = prepared
        .plan()
        .metric_ids()
        .iter()
        .position(|&candidate| candidate == metric_id)
        .ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "ranked significance metric is missing from the prepared plan".to_string(),
            )
        })?;
    let rank = GafimeRankSpec {
        top_k: 1,
        primary_metric: metric_id,
        descending: u32::from(descending),
        include_ties: 0,
        reserved: [0; 4],
    };
    let capacity = prepared.ranked_result_capacity(rank)?;
    let mut result = OwnedResultTable::new(
        capacity,
        prepared.result_max_arity(),
        prepared.result_metric_count(),
    );
    match backend {
        CompiledContinuousBackend::Cpu { matrix } => {
            let mut backend = CpuBackend;
            prepared.execute_ranked(rank, &mut backend, &matrix.handle(), result.raw_mut())?;
        }
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => {
            prepared.execute_ranked(
                rank,
                &mut *backend.borrow_mut(),
                matrix.handle(),
                result.raw_mut(),
            )?;
        }
    }
    ranked_metric_value(&result, metric_index)
}

fn update_ranked_plan_maxima(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    metric_ids: &[u32],
    maxima: &mut [f32],
) -> Result<(), PyBoundaryError> {
    for (metric_index, &metric_id) in metric_ids.iter().enumerate() {
        let high = execute_ranked_metric_extremum(backend, prepared, metric_id, true)?;
        maxima[metric_index] = maxima[metric_index].max(metric_extremeness(metric_id, high));
        if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
            let low = execute_ranked_metric_extremum(backend, prepared, metric_id, false)?;
            maxima[metric_index] = maxima[metric_index].max(metric_extremeness(metric_id, low));
        }
    }
    Ok(())
}

fn update_table_maxima(
    table: &OwnedResultTable,
    metric_ids: &[u32],
    maxima: &mut [f32],
) -> Result<(), PyBoundaryError> {
    for row in 0..table.row_count() {
        let values = metric_values_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("GPU significance metric row is missing".to_string())
        })?;
        for (metric_index, &metric_id) in metric_ids.iter().enumerate() {
            maxima[metric_index] =
                maxima[metric_index].max(metric_extremeness(metric_id, values[metric_index]));
        }
    }
    Ok(())
}

fn compute_host_orchestrated_gpu_permutation_pvalues(
    config: &EngineConfig,
    metric_ids: &[u32],
    significance_top_n: u32,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
    state: &mut ContinuousRunState,
    table: &OwnedResultTable,
    adaptive_search: bool,
) -> Result<Option<Vec<SignificanceEntry>>, PyBoundaryError> {
    if config.permutation_tests == 0
        || matches!(&state.backend, CompiledContinuousBackend::Cpu { .. })
    {
        return Ok(None);
    }
    require_device_ranking(&state.backend)?;
    let host_matrix = state.significance_matrix.as_ref().ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "GPU significance requires the retained host target".to_string(),
        )
    })?;
    if table.row_count() == 0 {
        return Ok(Some(Vec::new()));
    }

    let order = significance_order(table, metric_ids, significance_top_n);
    let metric_count = metric_ids.len();
    let mut observed_flat = Vec::with_capacity(order.len() * metric_count);
    for &row in &order {
        observed_flat.extend(metric_values_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance metric row out of range".to_string())
        })?);
    }
    let original_target = host_matrix.target().to_vec();
    let rows = host_matrix.rows();
    let cols = host_matrix.cols();
    let mut permutation_config = config.clone();
    permutation_config.permutation_tests = 0;
    permutation_config.num_repeats = 1;
    permutation_config.graph_requested = false;

    let permutation_result = (|| -> Result<Vec<u32>, PyBoundaryError> {
        let mut counts = vec![0u32; observed_flat.len()];
        for permutation_index in 0..config.permutation_tests {
            let target = significance::permutation_target(
                &original_target,
                config.random_seed,
                permutation_index,
            );
            update_gpu_target(&state.backend, &target)?;
            let mut maxima = vec![f32::NEG_INFINITY; metric_count];
            if adaptive_search {
                let run = prepare_screened_continuous_execution(
                    &permutation_config,
                    rows,
                    cols,
                    &state.backend,
                )?;
                let screened = run.screened.as_ref().ok_or_else(|| {
                    PyBoundaryError::InvalidInput(
                        "adaptive GPU significance did not produce a screened plan".to_string(),
                    )
                })?;
                update_table_maxima(&screened.unary_table, metric_ids, &mut maxima)?;
                update_ranked_plan_maxima(
                    &state.backend,
                    &screened.higher,
                    metric_ids,
                    &mut maxima,
                )?;
            } else {
                update_ranked_plan_maxima(
                    &state.backend,
                    state.complete_family()?,
                    metric_ids,
                    &mut maxima,
                )?;
            }
            for candidate_index in 0..order.len() {
                for metric_index in 0..metric_count {
                    let flat_index = candidate_index * metric_count + metric_index;
                    let observed =
                        metric_extremeness(metric_ids[metric_index], observed_flat[flat_index]);
                    if maxima[metric_index] >= observed {
                        counts[flat_index] += 1;
                    }
                }
            }
        }
        Ok(counts)
    })();
    let restore_result = update_gpu_target(&state.backend, &original_target);
    let counts = match (permutation_result, restore_result) {
        (Ok(counts), Ok(())) => counts,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(restore_error)) => {
            return Err(PyBoundaryError::InvalidInput(format!(
                "{error}; restoring the observed GPU target also failed: {restore_error}"
            )))
        }
    };

    let null_family_rows = state.complete_family()?.plan().planned_row_count();
    *runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(config.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: order.len() as u64,
    };
    let denominator = config.permutation_tests as f32 + 1.0;
    Ok(Some(
        order
            .into_iter()
            .enumerate()
            .map(|(position, row)| {
                let base = position * metric_count;
                SignificanceEntry {
                    row,
                    pvalues: counts[base..base + metric_count]
                        .iter()
                        .map(|&count| (count as f32 + 1.0) / denominator)
                        .collect(),
                    means: observed_flat[base..base + metric_count].to_vec(),
                    stds: vec![0.0; metric_count],
                }
            })
            .collect(),
    ))
}

fn compute_gpu_permutation_pvalues(
    config: &EngineConfig,
    cols: u32,
    metric_ids: &[u32],
    significance_top_n: u32,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
    state: &ContinuousRunState,
    table: &OwnedResultTable,
) -> Result<Option<Vec<SignificanceEntry>>, PyBoundaryError> {
    if config.permutation_tests == 0 {
        return Ok(None);
    }
    let CompiledContinuousBackend::Cuda { backend, matrix } = &state.backend else {
        return Ok(None);
    };
    if !device_permutation_pvalues_are_valid(
        config,
        cols,
        backend.borrow().supports_permutation_pvalues(),
    ) {
        return Ok(None);
    }
    let row_count = table.row_count();
    if row_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let order = significance_order(table, metric_ids, significance_top_n);

    let metric_count = metric_ids.len();
    let mut candidate_ids = Vec::with_capacity(order.len());
    let mut observed_flat = Vec::with_capacity(order.len() * metric_count);
    for &row in &order {
        let metrics = metric_values_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance metric row out of range".to_string())
        })?;
        candidate_ids.push(table.candidate_ids()[row]);
        observed_flat.extend(metrics);
    }

    let handle = matrix.handle();
    let complete_family = state.complete_family()?;
    let null_family_rows = complete_family.plan().planned_row_count();
    *runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(config.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: candidate_ids.len() as u64,
    };
    let protocol = complete_family.try_launch_protocol()?;
    let device_budget_bytes = (config.budget.vram_budget_mb != 0)
        .then(|| config.budget.vram_budget_mb.saturating_mul(1024 * 1024));
    let Some(pvalues) = backend.borrow_mut().permutation_pvalues_with_budget(
        handle,
        &protocol,
        &candidate_ids,
        &observed_flat,
        metric_count as u32,
        device_budget_bytes,
    )?
    else {
        // Older same-ABI payloads may provide native p-values without the
        // state-aware significance preflight. The caller will use the normal
        // budgeted host-orchestrated maxT path instead.
        return Ok(None);
    };
    if pvalues.len() != observed_flat.len() {
        return Err(PyBoundaryError::InvalidInput(
            "GPU p-value grid length does not match observed metrics".to_string(),
        ));
    }

    Ok(Some(
        order
            .into_iter()
            .enumerate()
            .map(|(position, row)| {
                let base = position * metric_count;
                SignificanceEntry {
                    row,
                    pvalues: pvalues[base..base + metric_count].to_vec(),
                    means: observed_flat[base..base + metric_count].to_vec(),
                    stds: vec![0.0; metric_count],
                }
            })
            .collect(),
    ))
}

/// Permutation + stability significance (P-A) for the top-N surfaced rows. The
/// reported rows stay bounded. Static maxT permutations stream the compiled
/// family; adaptive permutations re-score unary candidates and rebuild the
/// higher-order shortlist. CUDA's fixed-plan ABI is used only for static families.
fn compute_cpu_significance(
    config: &EngineConfig,
    metric_ids: &[u32],
    significance_top_n: u32,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
    state: &ContinuousRunState,
    table: &OwnedResultTable,
) -> Result<Vec<SignificanceEntry>, PyBoundaryError> {
    if config.permutation_tests == 0 && config.num_repeats <= 1 {
        return Ok(Vec::new());
    }
    // CPU uses the backend's matrix directly; a GPU run uses the retained host copy.
    // The fallback keeps the observed backend kind in SignificanceParams, which
    // preserves its adaptive MI ceiling and forces GPU-compatible fixed-width MI.
    let matrix = match &state.backend {
        CompiledContinuousBackend::Cpu { matrix } => matrix,
        CompiledContinuousBackend::Cuda { .. }
        | CompiledContinuousBackend::Rocm { .. }
        | CompiledContinuousBackend::Metal { .. } => match &state.significance_matrix {
            Some(matrix) => matrix,
            None => return Ok(Vec::new()),
        },
    };
    let row_count = table.row_count();
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let order = significance_order(table, metric_ids, significance_top_n);

    let kernels = metric_ids
        .iter()
        .copied()
        .map(MetricKernel::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            PyBoundaryError::InvalidInput("unknown metric id for significance".to_string())
        })?;

    let mut combos = Vec::with_capacity(order.len());
    let mut observed = Vec::with_capacity(order.len());
    for &row in &order {
        let combo = combo_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance combo row out of range".to_string())
        })?;
        let metrics = metric_values_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance metric row out of range".to_string())
        })?;
        combos.push(combo);
        observed.push(metrics);
    }

    let complete_family = state.complete_family()?;
    let null_family_rows = complete_family.plan().planned_row_count();
    *runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(config.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: combos.len() as u64,
    };

    let backend_kind = config.backend_kind;
    let params = SignificanceParams {
        permutation_tests: config.permutation_tests,
        num_repeats: config.num_repeats,
        random_seed: config.random_seed,
        mi_bins: config.mi_bins,
        backend_kind,
        mi_approximate: config.mi_approximate,
    };
    let evaluated = if config.permutation_tests > 0
        && has_adaptive_higher_order_search(config, matrix.cols())
    {
        let planning_seed_words = config.effective_planning_seed_words();
        let candidate_cols = config.effective_feature_candidate_count(matrix.cols());
        let unary_features = legacy_unary_feature_order(
            candidate_cols,
            config.budget.max_combinations_per_k,
            &planning_seed_words,
        );
        let search = AdaptiveSearchSpec {
            unary_features: &unary_features,
            candidate_feature_count: candidate_cols,
            max_arity: config.budget.max_comb_size,
            max_combinations_per_arity: config.budget.max_combinations_per_k,
            top_features_for_higher_arity: config.budget.top_features_for_higher_k,
            planning_seed_words: &planning_seed_words,
        };
        significance::evaluate_with_adaptive_search(
            matrix,
            &combos,
            &observed,
            complete_family.plan(),
            &kernels,
            &params,
            &search,
        )?
    } else {
        significance::evaluate_with_null_family(
            matrix,
            &combos,
            &observed,
            complete_family.plan(),
            &kernels,
            &params,
        )?
    };
    Ok(order
        .into_iter()
        .zip(evaluated)
        .map(|(row, sig)| SignificanceEntry {
            row,
            pvalues: sig.pvalues,
            means: sig.means,
            stds: sig.stds,
        })
        .collect())
}
fn rank_value_at(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    row: usize,
    metric_index: Option<usize>,
) -> Option<f32> {
    if row >= table.row_count() {
        return None;
    }
    let metric_count = table.metric_count();
    let metric_base = row.checked_mul(metric_count)?;
    let metrics = &table.metric_values()[metric_base..metric_base + metric_count];
    if let Some(metric_index) = metric_index {
        return metrics.get(metric_index).copied();
    }
    metrics
        .iter()
        .enumerate()
        .map(
            |(idx, &value)| match metric_ids.get(idx).copied().unwrap_or_default() {
                GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN => value.abs(),
                _ => value,
            },
        )
        .reduce(f32::max)
}

fn compare_rank_values(
    left: Option<f32>,
    right: Option<f32>,
    descending: bool,
) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => {
            let ordering = left
                .partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

pub(crate) enum CompiledContinuousBackend {
    Cpu {
        matrix: CpuMatrix,
    },
    Cuda {
        backend: RefCell<GpuBackend>,
        matrix: OwnedGpuMatrix,
    },
    Rocm {
        backend: RefCell<GpuBackend>,
        matrix: OwnedGpuMatrix,
    },
    Metal {
        backend: RefCell<GpuBackend>,
        matrix: OwnedGpuMatrix,
    },
}

impl CompiledContinuousBackend {
    pub(crate) fn uses_fp64_mi_accumulation(&self) -> bool {
        match self {
            Self::Cpu { .. } => true,
            Self::Cuda { backend, .. } | Self::Rocm { backend, .. } => {
                backend.borrow().uses_fp64_mi_accumulation()
            }
            Self::Metal { .. } => false,
        }
    }
}

pub(crate) struct ContinuousRunState {
    pub(crate) significance_matrix: Option<CpuMatrix>,
    pub(crate) backend: CompiledContinuousBackend,
    pub(crate) primary: Option<PreparedContinuousExecution>,
    screened: Option<ScreenedContinuousExecution>,
    pub(crate) result_capacity: u64,
    pub(crate) result_max_arity: u32,
    pub(crate) result_metric_count: u32,
}

impl ContinuousRunState {
    fn complete_family(&self) -> Result<&PreparedContinuousExecution, PyBoundaryError> {
        self.primary.as_ref().ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "continuous significance protocol is missing its complete family".to_string(),
            )
        })
    }

    pub(crate) fn replace_plans(&mut self, run: PreparedContinuousRun) {
        self.primary = run.primary;
        self.screened = run.screened;
        self.result_capacity = run.result_capacity;
        self.result_max_arity = run.result_max_arity;
        self.result_metric_count = run.result_metric_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::artifact::{compile_continuous_rows, execute_compiled_artifact};

    #[test]
    fn screened_host_admission_counts_cached_and_combined_result_owners() {
        let bytes = screened_candidate_storage_bytes(10_000_000, 0, 10_000_000, 5, 1, 0);

        assert_eq!(bytes, 720_000_000);
        assert!(bytes > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES);
    }

    #[test]
    fn continuous_cpu_boundary_scores_flat_f32_rows() {
        let report = analyze_continuous_cpu_rows(
            4,
            3,
            vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
            1,
            10,
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .unwrap();

        assert_eq!(report.rows, 4);
        assert_eq!(report.cols, 3);
        assert_eq!(report.len(), 3);
        assert_eq!(report.combo(0).unwrap(), vec![0]);
        assert!((report.metric_values(0).unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((report.metric_values(1).unwrap()[0] + 1.0).abs() < 1e-6);
        assert_eq!(report.metric_values(2).unwrap()[0], 0.0);
    }

    #[test]
    fn one_shot_and_compiled_cpu_execution_are_exact() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 10;
        config.budget.top_features_for_higher_k = 3;
        let features = vec![1.0, 4.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 0.0, 4.0, 1.0, -1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];

        let one_shot =
            analyze_continuous_rows_once(config.clone(), 4, 3, features.clone(), target.clone())
                .unwrap();
        let mut compiled = compile_continuous_rows(config, 4, 3, features, target).unwrap();
        assert!(compiled.state.as_ref().unwrap().primary.is_none());
        let repeated = execute_compiled_artifact(&mut compiled).unwrap();

        assert_eq!(
            one_shot.table.candidate_ids(),
            repeated.table.candidate_ids()
        );
        assert_eq!(
            one_shot.table.combo_indices(),
            repeated.table.combo_indices()
        );
        assert_eq!(
            one_shot.table.metric_values(),
            repeated.table.metric_values()
        );
    }

    #[test]
    fn adaptive_search_bypasses_the_static_cuda_permutation_abi() {
        let config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 199,
            num_repeats: 1,
            budget: gafime_types::GafimeComputeBudget {
                max_comb_size: 2,
                max_combinations_per_k: 5_000,
                top_features_for_higher_k: 5,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(has_adaptive_higher_order_search(&config, 18));
        assert!(!device_permutation_pvalues_are_valid(&config, 18, true));
    }

    #[test]
    fn static_family_keeps_supported_device_permutation_route() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 31,
            num_repeats: 1,
            budget: gafime_types::GafimeComputeBudget {
                max_comb_size: 1,
                max_combinations_per_k: 64,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!has_adaptive_higher_order_search(&config, 18));
        assert!(device_permutation_pvalues_are_valid(&config, 18, true));
        assert!(!device_permutation_pvalues_are_valid(&config, 18, false));

        config.num_repeats = 2;
        assert!(device_permutation_pvalues_are_valid(&config, 18, true));
    }

    #[test]
    fn ranked_extremum_uses_the_bounded_prepared_execution_api() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_R2, GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = 8;
        let state = build_continuous_state(
            &config,
            4,
            2,
            vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let prepared = state.primary.as_ref().unwrap();

        assert_eq!(
            execute_ranked_metric_extremum(&state.backend, prepared, GAFIME_METRIC_PEARSON, true,)
                .unwrap(),
            1.0
        );
        assert_eq!(
            execute_ranked_metric_extremum(&state.backend, prepared, GAFIME_METRIC_PEARSON, false,)
                .unwrap(),
            -1.0
        );
    }

    #[test]
    fn ranked_extremum_accepts_a_valid_empty_device_result() {
        let table = OwnedResultTable::new(1, 5, 1);

        assert_eq!(ranked_metric_value(&table, 0).unwrap(), f32::NEG_INFINITY);
    }

    #[test]
    fn bounded_selection_matches_full_stable_order() {
        let report = analyze_continuous_cpu_rows(
            5,
            5,
            vec![
                1.0, 5.0, 1.0, 2.0, 9.0, 2.0, 4.0, 1.0, 2.0, 9.0, 3.0, 3.0, 1.0, 2.0, 9.0, 4.0,
                2.0, 1.0, 2.0, 9.0, 5.0, 1.0, 1.0, 2.0, 9.0,
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            1,
            10,
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .unwrap();
        for metric_index in [None, Some(0), Some(1)] {
            for descending in [false, true] {
                let mut full = (0..report.table.row_count()).collect::<Vec<_>>();
                full.sort_by(|&left, &right| {
                    compare_ranked_rows(
                        &report.table,
                        &report.metric_ids,
                        left,
                        right,
                        metric_index,
                        descending,
                    )
                });
                for limit in 0..=full.len() {
                    assert_eq!(
                        bounded_ranked_indices(
                            &report.table,
                            &report.metric_ids,
                            metric_index,
                            descending,
                            limit,
                        ),
                        full[..limit]
                    );
                }
            }
        }
    }
}
