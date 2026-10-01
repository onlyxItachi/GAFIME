use std::cell::RefCell;

use gafime_cpu::{
    kernels::MetricKernel,
    precision::{CpuPrecisionMatrix, CpuPrecisionSlice, CpuPrecisionValues},
    result::{OwnedResultTable, OwnedResultTableF64, PrecisionOwnedResultTable},
    significance::{self, AdaptiveSearchSpec, SignificanceParams},
    CpuBackend,
};
use gafime_gpu_sys::{GpuBackend, OwnedGpuMatrix};
use gafime_orchestrator::config::EngineConfig;
use gafime_orchestrator::semantic::supervised::SupervisedStrengths as PrecisionUnaryStrengths;
use gafime_orchestrator::{
    continuous_staged_device_footprint_bytes,
    plan::combos::{legacy_unary_feature_order, DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES},
    prepare_continuous_execution_for_feature_orders, OrchestratorError,
    PreparedContinuousExecution,
};
use gafime_types::{
    GafimePrecisionLaunchProtocol, GafimeRankSpec, PrecisionProfile, GAFIME_BACKEND_CPU,
    GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_METRIC_PEARSON,
    GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN, GAFIME_PRECISION_ABI_VERSION,
};

use crate::common::{
    combo_from_table, report_from_table, validate_metric_ids, validate_shape, ContinuousReport,
    InteractionPrecisionDiagnostic, OwnedNumericInput, PyBoundaryError, ResultTableView,
    SignificanceEntry,
};
use crate::runtime::RuntimeCacheCounters;

/// Typed resident values for the additive profile-aware Core convenience API.
///
/// The variant must match the requested profile: `Fp32` and `Mixed` consume
/// `F32`, while `Fp64` consumes `F64`. Mismatches fail closed instead of
/// quantizing through an intermediate buffer.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuousCpuInput {
    F32 {
        features: Vec<f32>,
        target: Vec<f32>,
    },
    F64 {
        features: Vec<f64>,
        target: Vec<f64>,
    },
}

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

/// Analyze typed Core rows under an explicit precision profile.
///
/// This is additive to [`analyze_continuous_cpu_rows`], whose existing
/// signature and default-mixed behavior remain source compatible.
pub fn analyze_continuous_cpu_rows_with_precision(
    precision: PrecisionProfile,
    rows: u64,
    cols: u32,
    input: ContinuousCpuInput,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Vec<u32>,
) -> Result<ContinuousReport, PyBoundaryError> {
    let input = match input {
        ContinuousCpuInput::F32 { features, target } => {
            OwnedNumericInput::from_f32(precision, features, target)?
        }
        ContinuousCpuInput::F64 { features, target } => {
            OwnedNumericInput::from_f64(precision, features, target)?
        }
    };
    analyze_continuous_cpu_input_rows(
        precision,
        rows,
        cols,
        input,
        max_arity,
        max_combinations_per_k,
        metric_ids,
    )
}

pub(crate) fn analyze_continuous_cpu_input_rows(
    precision: PrecisionProfile,
    rows: u64,
    cols: u32,
    input: OwnedNumericInput,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Vec<u32>,
) -> Result<ContinuousReport, PyBoundaryError> {
    let metric_ids = if metric_ids.is_empty() {
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2]
    } else {
        metric_ids
    };
    let mut config = continuous_config_for_cpu(max_arity, max_combinations_per_k, metric_ids)?;
    config.precision = precision;
    analyze_continuous_input_once(config, rows, cols, input)
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
    let precision = config.precision;
    let input = OwnedNumericInput::from_f32(precision, features, target)?;
    analyze_continuous_input_once(config, rows, cols, input)
}

pub(crate) fn analyze_continuous_input_once(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    input: OwnedNumericInput,
) -> Result<ContinuousReport, PyBoundaryError> {
    validate_shape(rows, cols, input.feature_len(), input.target_len())?;
    let mut state = build_continuous_state(&config, rows, cols, input)?;
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
    input: OwnedNumericInput,
) -> Result<ContinuousRunState, PyBoundaryError> {
    let precision = config.precision;
    if config.backend_kind == GAFIME_BACKEND_METAL && precision != PrecisionProfile::Fp32 {
        return Err(PyBoundaryError::UnsupportedFeature(
            "Metal supports precision=\"fp32\" only; use backend=\"auto\" or backend=\"core\" for mixed/fp64"
                .to_string(),
        ));
    }
    if requires_gpu_budget_preflight(config) {
        preflight_screened_continuous_execution(config, rows, cols)?;
    }
    let needs_significance = config.permutation_tests > 0 || config.num_repeats > 1;
    // For GPU runs that still need CPU-side significance, build the host copy by
    // MOVING the ingest buffers into CpuMatrix after device upload borrows them.
    // CUDA payloads exposing the optional permutation p-value ABI can skip that
    // copy only for static families without bootstrap stability. Adaptive search
    // must retain the host matrix until the ABI can re-screen each permutation.
    let (backend, significance_matrix) =
        match config.backend_kind {
            GAFIME_BACKEND_CPU => {
                let matrix = cpu_matrix_from_input(precision, rows, cols, input)?;
                (CompiledContinuousBackend::Cpu { matrix }, None)
            }
            GAFIME_BACKEND_CUDA => {
                let backend = GpuBackend::cuda_from_env(config.device_id)?;
                let matrix = backend.alloc_matrix_for_profile(precision, rows, cols)?;
                upload_precision_matrix(&matrix, &input)?;
                let use_device_pvalues = needs_significance
                    && device_permutation_pvalues_are_valid(
                        config,
                        cols,
                        backend.supports_precision_permutation_pvalues(precision),
                    );
                let significance_matrix =
                    if needs_significance && (!use_device_pvalues || config.num_repeats > 1) {
                        Some(cpu_matrix_from_input(precision, rows, cols, input)?)
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
                let matrix = backend.alloc_matrix_for_profile(precision, rows, cols)?;
                upload_precision_matrix(&matrix, &input)?;
                let significance_matrix = if needs_significance {
                    Some(cpu_matrix_from_input(precision, rows, cols, input)?)
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
                let matrix = backend.alloc_matrix_for_profile(precision, rows, cols)?;
                upload_precision_matrix(&matrix, &input)?;
                let significance_matrix = if needs_significance {
                    Some(cpu_matrix_from_input(precision, rows, cols, input)?)
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
        interaction_diagnostics_cache: None,
        result_capacity: run.result_capacity,
        result_max_arity: run.result_max_arity,
        result_metric_count: run.result_metric_count,
    })
}

fn cpu_matrix_from_input(
    precision: PrecisionProfile,
    rows: u64,
    cols: u32,
    input: OwnedNumericInput,
) -> Result<CpuPrecisionMatrix, PyBoundaryError> {
    match input {
        OwnedNumericInput::F32 { features, target } => Ok(CpuPrecisionMatrix::from_row_major_f32(
            precision, rows, cols, features, target,
        )?),
        OwnedNumericInput::F64 { features, target } => Ok(CpuPrecisionMatrix::from_row_major_f64(
            precision, rows, cols, features, target,
        )?),
    }
}

fn upload_precision_matrix(
    matrix: &OwnedGpuMatrix,
    input: &OwnedNumericInput,
) -> Result<(), PyBoundaryError> {
    match input {
        OwnedNumericInput::F32 { features, target } => {
            matrix.upload_f32_v2(features, target)?;
        }
        OwnedNumericInput::F64 { features, target } => {
            matrix.upload_f64_v2(features, target)?;
        }
    }
    Ok(())
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
    unary_table: PrecisionOwnedResultTable,
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
) -> Result<PrecisionOwnedResultTable, PyBoundaryError> {
    let precision = backend.precision();
    let mut table = PrecisionOwnedResultTable::new(
        precision,
        prepared.result_capacity(),
        prepared.result_max_arity(),
        prepared.result_metric_count(),
    );
    execute_prepared_continuous_into(backend, prepared, &mut table)?;
    Ok(table)
}

fn execute_prepared_continuous_into(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    result: &mut PrecisionOwnedResultTable,
) -> Result<gafime_orchestrator::BackendExecutionStats, PyBoundaryError> {
    match result {
        PrecisionOwnedResultTable::Fp32(table) => {
            // SAFETY: `table` owns all fp32 ABI output buffers and keeps them
            // uniquely borrowed for the synchronous prepared execution.
            unsafe { execute_prepared_fp32_into(backend, prepared, table.raw_mut()) }
        }
        PrecisionOwnedResultTable::F64 { profile, table } => {
            if *profile != backend.precision() {
                return Err(PyBoundaryError::InvalidInput(
                    "result profile does not match resident backend identity".to_string(),
                ));
            }
            // SAFETY: `table` owns all f64 ABI output buffers and keeps them
            // uniquely borrowed for the synchronous prepared execution.
            unsafe { execute_prepared_f64_into(backend, prepared, table.raw_mut()) }
        }
    }
}

/// Execute into a caller-provided raw fp32 result descriptor.
///
/// # Safety
///
/// `result` must point only into uniquely borrowed live allocations covering
/// its declared capacity and strides for this synchronous call.
unsafe fn execute_prepared_fp32_into(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    result: &mut gafime_types::GafimeResultTable,
) -> Result<gafime_orchestrator::BackendExecutionStats, PyBoundaryError> {
    match backend {
        CompiledContinuousBackend::Cpu { matrix } => {
            let mut cpu = CpuBackend;
            // SAFETY: upheld by this helper's caller; the CPU matrix guard and
            // prepared descriptor storage remain live for the call.
            Ok(unsafe { prepared.execute_precision_fp32(&mut cpu, &matrix.handle(), result) }?)
        }
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => {
            // SAFETY: upheld by this helper's caller; the resident matrix and
            // prepared descriptor storage remain live for the call.
            Ok(unsafe {
                prepared.execute_precision_fp32(&mut *backend.borrow_mut(), matrix.handle(), result)
            }?)
        }
    }
}

/// Execute into a caller-provided raw f64 result descriptor.
///
/// # Safety
///
/// `result` must point only into uniquely borrowed live allocations covering
/// its declared capacity and strides for this synchronous call.
unsafe fn execute_prepared_f64_into(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    result: &mut gafime_types::GafimeResultTableF64,
) -> Result<gafime_orchestrator::BackendExecutionStats, PyBoundaryError> {
    match backend {
        CompiledContinuousBackend::Cpu { matrix } => {
            let mut cpu = CpuBackend;
            // SAFETY: upheld by this helper's caller; the CPU matrix guard and
            // prepared descriptor storage remain live for the call.
            Ok(unsafe { prepared.execute_precision_f64(&mut cpu, &matrix.handle(), result) }?)
        }
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix } => {
            // SAFETY: upheld by this helper's caller; the resident matrix and
            // prepared descriptor storage remain live for the call.
            Ok(unsafe {
                prepared.execute_precision_f64(&mut *backend.borrow_mut(), matrix.handle(), result)
            }?)
        }
        CompiledContinuousBackend::Metal { .. } => Err(PyBoundaryError::UnsupportedFeature(
            "Metal supports precision=\"fp32\" only".to_string(),
        )),
    }
}

fn collect_interaction_diagnostics(
    state: &mut ContinuousRunState,
    table: &PrecisionOwnedResultTable,
) -> Result<Option<Vec<InteractionPrecisionDiagnostic>>, PyBoundaryError> {
    let row_count = table.row_count();
    let max_arity = table.max_arity();
    let combo_len = row_count.checked_mul(max_arity).ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "interaction diagnostic result shape exceeds host address space".to_string(),
        )
    })?;
    let combo_indices = &table.combo_indices()[..combo_len];
    if let Some(cache) = state.interaction_diagnostics_cache.as_ref() {
        if cache.row_count == row_count
            && cache.max_arity == max_arity
            && cache.combo_indices == combo_indices
        {
            return Ok(cache.diagnostics.clone());
        }
    }

    let diagnostics: Option<Vec<InteractionPrecisionDiagnostic>> = match &state.backend {
        CompiledContinuousBackend::Cpu { matrix } => Some(
            gafime_cpu::diagnostics::interaction_diagnostics_precision(
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
        ),
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => backend
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
            }),
    };
    state.interaction_diagnostics_cache = Some(InteractionDiagnosticsCache {
        row_count,
        max_arity,
        combo_indices: combo_indices.to_vec(),
        diagnostics: diagnostics.clone(),
    });
    Ok(diagnostics)
}

fn execute_continuous_plan_set(
    backend: &CompiledContinuousBackend,
    primary: Option<&PreparedContinuousExecution>,
    screened: Option<&ScreenedContinuousExecution>,
    result_capacity: u64,
    result_max_arity: u32,
    result_metric_count: u32,
) -> Result<PrecisionOwnedResultTable, PyBoundaryError> {
    let precision = backend.precision();
    if result_capacity == 0 {
        return Ok(PrecisionOwnedResultTable::new(
            precision,
            0,
            result_max_arity,
            result_metric_count,
        ));
    }
    if let Some(screened) = screened {
        return combine_screened_results(
            backend,
            screened,
            result_capacity,
            result_max_arity,
            result_metric_count,
        );
    }
    let prepared = primary.ok_or_else(|| {
        PyBoundaryError::InvalidInput("direct continuous plan is missing".to_string())
    })?;
    execute_prepared_continuous(backend, prepared)
}

fn combine_screened_results(
    backend: &CompiledContinuousBackend,
    screened: &ScreenedContinuousExecution,
    result_capacity: u64,
    result_max_arity: u32,
    result_metric_count: u32,
) -> Result<PrecisionOwnedResultTable, PyBoundaryError> {
    match &screened.unary_table {
        PrecisionOwnedResultTable::Fp32(unary) => {
            let mut combined =
                OwnedResultTable::new(result_capacity, result_max_arity, result_metric_count);
            combined
                .append_rows_from(unary, 0)
                .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
            let start = unary.row_count() as u64;
            let (execution, row_count) = combined
                .with_raw_rows_mut(start, screened.higher.result_capacity(), |raw| {
                    // SAFETY: `with_raw_rows_mut` binds this descriptor to a
                    // checked, uniquely borrowed window of `combined`.
                    unsafe { execute_prepared_fp32_into(backend, &screened.higher, raw) }
                })
                .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
            finish_combined_rows(&mut combined, start, execution?, row_count)?;
            Ok(PrecisionOwnedResultTable::Fp32(combined))
        }
        PrecisionOwnedResultTable::F64 {
            profile,
            table: unary,
        } => {
            let mut combined =
                OwnedResultTableF64::new(result_capacity, result_max_arity, result_metric_count);
            combined
                .append_rows_from(unary, 0)
                .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))?;
            let start = unary.row_count() as u64;
            let (execution, row_count) = combined
                .with_raw_rows_mut(start, screened.higher.result_capacity(), |raw| {
                    // SAFETY: `with_raw_rows_mut` binds this descriptor to a
                    // checked, uniquely borrowed window of `combined`.
                    unsafe { execute_prepared_f64_into(backend, &screened.higher, raw) }
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
            Ok(PrecisionOwnedResultTable::F64 {
                profile: *profile,
                table: combined,
            })
        }
    }
}

fn finish_combined_rows(
    combined: &mut OwnedResultTable,
    start: u64,
    execution: gafime_orchestrator::BackendExecutionStats,
    row_count: u64,
) -> Result<(), PyBoundaryError> {
    if execution.rows_written != row_count {
        return Err(PyBoundaryError::InvalidInput(
            "backend result count differs from screened higher-order rows".to_string(),
        ));
    }
    combined
        .commit_appended_rows(start, row_count, start)
        .map_err(|message| PyBoundaryError::InvalidInput(message.to_string()))
}

fn result_table_storage_bytes(
    precision: PrecisionProfile,
    capacity: u64,
    max_arity: u32,
    metric_count: u32,
) -> u64 {
    const U32_BYTES: u64 = 4;
    const U64_BYTES: u64 = 8;
    let metric_bytes = if precision == PrecisionProfile::Fp32 {
        U32_BYTES
    } else {
        U64_BYTES
    };
    let row_bytes = u64::from(max_arity)
        .saturating_mul(U32_BYTES)
        .saturating_add(u64::from(metric_count).saturating_mul(metric_bytes))
        .saturating_add(U32_BYTES) // rank
        .saturating_add(U32_BYTES) // family
        .saturating_add(U64_BYTES) // candidate id
        .saturating_add(U32_BYTES); // row flags
    capacity.saturating_mul(row_bytes)
}

fn screened_candidate_storage_bytes(
    precision: PrecisionProfile,
    unary_rows: u64,
    higher_descriptor_words: u64,
    combined_rows: u64,
    combined_max_arity: u32,
    metric_count: u32,
    retained_complete_descriptor_words: u64,
) -> u64 {
    higher_descriptor_words
        .saturating_mul(core::mem::size_of::<u32>() as u64)
        .saturating_add(result_table_storage_bytes(
            precision,
            unary_rows,
            1,
            metric_count,
        ))
        .saturating_add(result_table_storage_bytes(
            precision,
            combined_rows,
            combined_max_arity,
            metric_count,
        ))
        .saturating_add(
            retained_complete_descriptor_words.saturating_mul(core::mem::size_of::<u32>() as u64),
        )
}

fn requires_gpu_budget_preflight(config: &EngineConfig) -> bool {
    config.budget.vram_budget_mb > 0
        && matches!(
            config.backend_kind,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        )
}

/// Validate every structural plan shape that adaptive screening can select
/// before loading a payload or allocating its resident matrix. Feature
/// identities do not affect byte counts, so the eventual score-selected
/// higher-order features can be represented by any equally sized unique set.
fn preflight_screened_continuous_execution(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
) -> Result<(), PyBoundaryError> {
    let planning_seed_words = config.effective_planning_seed_words();
    let candidate_cols = config.effective_feature_candidate_count(cols);
    if candidate_cols == 0 {
        return Ok(());
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
        let _prepared = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            &[],
            true,
            true,
        )?;
        return Ok(());
    }

    let unary = prepare_continuous_execution_for_feature_orders(
        config,
        rows,
        cols,
        &unary_features,
        &[],
        true,
        false,
    )?;
    let higher_feature_count = unary_features
        .len()
        .min(config.budget.top_features_for_higher_k as usize);
    let higher_features = &unary_features[..higher_feature_count];
    if config.graph_requested {
        let _combined = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            higher_features,
            true,
            true,
        )?;
        return Ok(());
    }

    let higher = prepare_continuous_execution_for_feature_orders(
        config,
        rows,
        cols,
        &[],
        higher_features,
        false,
        false,
    )?;
    let staged_footprint = continuous_staged_device_footprint_bytes(&[&unary, &higher]);
    let budget_bytes = config.budget.vram_budget_mb.saturating_mul(1024 * 1024);
    if staged_footprint > budget_bytes {
        return Err(OrchestratorError::Unsupported(
            "continuous plan device footprint exceeds budget.vram_budget_mb",
        )
        .into());
    }
    let result_capacity = unary
        .result_capacity()
        .saturating_add(higher.result_capacity());
    let result_max_arity = unary.result_max_arity().max(higher.result_max_arity());
    let result_metric_count = higher.result_metric_count();
    let needs_complete_family = config.permutation_tests > 0 || config.num_repeats > 1;
    let complete_descriptor_words = higher
        .plan()
        .logical_descriptor_words()
        .saturating_add(unary.result_capacity());
    let retained_complete_descriptor_words = if needs_complete_family {
        complete_descriptor_words
    } else {
        0
    };
    let screened_storage = screened_candidate_storage_bytes(
        config.precision,
        unary.result_capacity(),
        higher.plan().materialized_descriptor_words() as u64,
        result_capacity,
        result_max_arity,
        result_metric_count,
        retained_complete_descriptor_words,
    );
    if needs_complete_family || screened_storage > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES {
        let _combined = prepare_continuous_execution_for_feature_orders(
            config,
            rows,
            cols,
            &unary_features,
            higher_features,
            true,
            true,
        )?;
    }
    Ok(())
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
        unary_strengths.sort_by_feature();
    }
    let higher_features = unary_strengths.higher_feature_order(
        candidate_cols,
        config.budget.max_combinations_per_k,
        config.budget.top_features_for_higher_k,
        &planning_seed_words,
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
        config.precision,
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
    table: &impl ResultTableView,
    unary_features: &[u32],
    metric_ids: &[u32],
) -> Result<PrecisionUnaryStrengths, PyBoundaryError> {
    if table.row_count() != unary_features.len() || table.metric_count() != metric_ids.len() {
        return Err(PyBoundaryError::InvalidInput(
            "unary screening result shape does not match its plan".to_string(),
        ));
    }
    match table.metric_values() {
        crate::common::MetricValuesRef::F32(values) => {
            PrecisionUnaryStrengths::from_f32(values, unary_features, metric_ids)
        }
        crate::common::MetricValuesRef::F64(values) => {
            PrecisionUnaryStrengths::from_f64(values, unary_features, metric_ids)
        }
    }
    .map_err(|error| PyBoundaryError::InvalidInput(error.to_string()))
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
    let interaction_diagnostics = collect_interaction_diagnostics(state, &table)?;

    if table.row_count() == 0 {
        return Ok(report_from_table(
            rows,
            cols,
            state.result_max_arity,
            metric_ids.to_vec(),
            config.backend_kind,
            config.precision,
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
        config.precision,
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
    table: &PrecisionOwnedResultTable,
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
        if rank_value_at(table, metric_ids, row, metric_index).is_none() {
            continue;
        }
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

pub(crate) fn row_is_rankable(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    row: usize,
    metric_index: Option<usize>,
) -> bool {
    rank_value_at(table, metric_ids, row, metric_index).is_some()
}

fn significance_order(
    table: &impl ResultTableView,
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

trait HostSignificanceScalar: Copy + PartialOrd {
    const NEG_INFINITY: Self;
    const ZERO: Self;

    fn metric_extremeness(metric_id: u32, value: Self) -> Self;
    fn pvalue(count: u32, permutations: u32) -> Self;
    fn into_precision_values(values: Vec<Self>) -> CpuPrecisionValues;

    fn metric_values(table: &impl ResultTableView) -> Result<&[Self], PyBoundaryError>;
}

impl HostSignificanceScalar for f32 {
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const ZERO: Self = 0.0;

    fn metric_extremeness(metric_id: u32, value: Self) -> Self {
        if !value.is_finite() {
            return Self::NEG_INFINITY;
        }
        if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
            value.abs()
        } else {
            value
        }
    }

    fn pvalue(count: u32, permutations: u32) -> Self {
        (count as f32 + 1.0) / (permutations as f32 + 1.0)
    }

    fn into_precision_values(values: Vec<Self>) -> CpuPrecisionValues {
        CpuPrecisionValues::F32(values)
    }

    fn metric_values(table: &impl ResultTableView) -> Result<&[Self], PyBoundaryError> {
        match table.metric_values() {
            crate::common::MetricValuesRef::F32(values) => Ok(values),
            crate::common::MetricValuesRef::F64(_) => Err(PyBoundaryError::InvalidInput(
                "fp32 significance received an fp64 result table".to_string(),
            )),
        }
    }
}

impl HostSignificanceScalar for f64 {
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const ZERO: Self = 0.0;

    fn metric_extremeness(metric_id: u32, value: Self) -> Self {
        if !value.is_finite() {
            return Self::NEG_INFINITY;
        }
        if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
            value.abs()
        } else {
            value
        }
    }

    fn pvalue(count: u32, permutations: u32) -> Self {
        (f64::from(count) + 1.0) / (f64::from(permutations) + 1.0)
    }

    fn into_precision_values(values: Vec<Self>) -> CpuPrecisionValues {
        CpuPrecisionValues::F64(values)
    }

    fn metric_values(table: &impl ResultTableView) -> Result<&[Self], PyBoundaryError> {
        match table.metric_values() {
            crate::common::MetricValuesRef::F64(values) => Ok(values),
            crate::common::MetricValuesRef::F32(_) => Err(PyBoundaryError::InvalidInput(
                "mixed/fp64 significance received an fp32 result table".to_string(),
            )),
        }
    }
}

fn precision_values_slice(values: &CpuPrecisionValues) -> CpuPrecisionSlice<'_> {
    match values {
        CpuPrecisionValues::F32(values) => CpuPrecisionSlice::F32(values),
        CpuPrecisionValues::F64(values) => CpuPrecisionSlice::F64(values),
    }
}

fn permutation_target_precision(
    target: &CpuPrecisionValues,
    base_seed: u64,
    permutation_index: u32,
) -> CpuPrecisionValues {
    let mut state = base_seed
        ^ 0xA5A5_A5A5u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(permutation_index).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let seed = splitmix64(&mut state);
    match target {
        CpuPrecisionValues::F32(values) => {
            CpuPrecisionValues::F32(shuffle_precision_values(values, seed))
        }
        CpuPrecisionValues::F64(values) => {
            CpuPrecisionValues::F64(shuffle_precision_values(values, seed))
        }
    }
}

fn shuffle_precision_values<T: Copy>(values: &[T], seed: u64) -> Vec<T> {
    let mut values = values.to_vec();
    let mut state = seed;
    for index in (1..values.len()).rev() {
        let swap = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
        values.swap(index, swap);
    }
    values
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn update_gpu_target(
    backend: &CompiledContinuousBackend,
    target: CpuPrecisionSlice<'_>,
) -> Result<(), PyBoundaryError> {
    match backend {
        CompiledContinuousBackend::Cuda { matrix, .. }
        | CompiledContinuousBackend::Rocm { matrix, .. }
        | CompiledContinuousBackend::Metal { matrix, .. } => {
            match target {
                CpuPrecisionSlice::F32(target) => matrix.update_target_f32_v2(target)?,
                CpuPrecisionSlice::F64(target) => matrix.update_target_f64_v2(target)?,
            }
            Ok(())
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

fn ranked_metric_value<T: HostSignificanceScalar>(
    result: &impl ResultTableView,
    metric_index: usize,
) -> Result<T, PyBoundaryError> {
    if result.row_count() == 0 {
        return Ok(T::NEG_INFINITY);
    }
    let values = T::metric_values(result)?;
    values.get(metric_index).copied().ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "ranked significance result has the wrong metric width".to_string(),
        )
    })
}

fn execute_ranked_metric_extremum<T: HostSignificanceScalar>(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    metric_id: u32,
    descending: bool,
) -> Result<T, PyBoundaryError> {
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
    let mut result = PrecisionOwnedResultTable::new(
        backend.precision(),
        capacity,
        prepared.result_max_arity(),
        prepared.result_metric_count(),
    );
    execute_ranked_precision_into(backend, prepared, rank, &mut result)?;
    ranked_metric_value::<T>(&result, metric_index)
}

fn execute_ranked_precision_into(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    rank: GafimeRankSpec,
    result: &mut PrecisionOwnedResultTable,
) -> Result<(), PyBoundaryError> {
    match (backend, result) {
        (CompiledContinuousBackend::Cpu { matrix }, PrecisionOwnedResultTable::Fp32(table)) => {
            let mut cpu = CpuBackend;
            // SAFETY: `table` owns all declared output buffers and is uniquely
            // borrowed for this synchronous prepared execution.
            unsafe {
                prepared.execute_precision_ranked_fp32(
                    rank,
                    &mut cpu,
                    &matrix.handle(),
                    table.raw_mut(),
                )
            }?;
        }
        (
            CompiledContinuousBackend::Cuda { backend, matrix }
            | CompiledContinuousBackend::Rocm { backend, matrix }
            | CompiledContinuousBackend::Metal { backend, matrix },
            PrecisionOwnedResultTable::Fp32(table),
        ) => {
            // SAFETY: `table` owns all declared output buffers and is uniquely
            // borrowed for this synchronous prepared execution.
            unsafe {
                prepared.execute_precision_ranked_fp32(
                    rank,
                    &mut *backend.borrow_mut(),
                    matrix.handle(),
                    table.raw_mut(),
                )
            }?;
        }
        (
            CompiledContinuousBackend::Cpu { matrix },
            PrecisionOwnedResultTable::F64 { table, .. },
        ) => {
            let mut cpu = CpuBackend;
            // SAFETY: `table` owns all declared typed output buffers and is
            // uniquely borrowed for this synchronous prepared execution.
            unsafe {
                prepared.execute_precision_ranked_f64(
                    rank,
                    &mut cpu,
                    &matrix.handle(),
                    table.raw_mut(),
                )
            }?;
        }
        (
            CompiledContinuousBackend::Cuda { backend, matrix }
            | CompiledContinuousBackend::Rocm { backend, matrix },
            PrecisionOwnedResultTable::F64 { table, .. },
        ) => {
            // SAFETY: `table` owns all declared typed output buffers and is
            // uniquely borrowed for this synchronous prepared execution.
            unsafe {
                prepared.execute_precision_ranked_f64(
                    rank,
                    &mut *backend.borrow_mut(),
                    matrix.handle(),
                    table.raw_mut(),
                )
            }?;
        }
        (CompiledContinuousBackend::Metal { .. }, PrecisionOwnedResultTable::F64 { .. }) => {
            return Err(PyBoundaryError::UnsupportedFeature(
                "Metal supports precision=\"fp32\" only".to_string(),
            ));
        }
    }
    Ok(())
}

fn update_ranked_plan_maxima<T: HostSignificanceScalar>(
    backend: &CompiledContinuousBackend,
    prepared: &PreparedContinuousExecution,
    metric_ids: &[u32],
    maxima: &mut [T],
) -> Result<(), PyBoundaryError> {
    for (metric_index, &metric_id) in metric_ids.iter().enumerate() {
        let high = execute_ranked_metric_extremum::<T>(backend, prepared, metric_id, true)?;
        let high = T::metric_extremeness(metric_id, high);
        if high > maxima[metric_index] {
            maxima[metric_index] = high;
        }
        if matches!(metric_id, GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN) {
            let low = execute_ranked_metric_extremum::<T>(backend, prepared, metric_id, false)?;
            let low = T::metric_extremeness(metric_id, low);
            if low > maxima[metric_index] {
                maxima[metric_index] = low;
            }
        }
    }
    Ok(())
}

/// Build one permutation-specific expanded decision-path family on the
/// requested GPU backend and return its lane-typed maxT extrema.
///
/// Decision-path discovery and membership materialization remain host-owned,
/// but every score contributing to a GPU null-family maximum is evaluated and
/// ranked through the device's bounded `top_k=1` execution surface. Retaining
/// the complete screened plan is deliberate: the unary screening result may
/// select a different higher-order family for every permuted target.
pub(crate) fn execute_device_decision_path_null_maxima(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    input: OwnedNumericInput,
) -> Result<CpuPrecisionValues, PyBoundaryError> {
    if !matches!(
        config.backend_kind,
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
    ) {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path device maxT requires an explicit GPU backend".to_string(),
        ));
    }

    let mut null_config = config.clone();
    null_config.budget.max_feature_candidate = -2;
    // This flag retains the complete permutation-specific screened family. The
    // helper never invokes ordinary significance recursively.
    null_config.permutation_tests = 1;
    null_config.num_repeats = 1;
    null_config.graph_requested = false;
    let metric_ids = null_config.metric_ids.clone();
    let state = build_continuous_state(&null_config, rows, cols, input)?;
    require_device_ranking(&state.backend)?;
    let complete_family = state.complete_family()?;

    match null_config.precision {
        PrecisionProfile::Fp32 => {
            let mut maxima = vec![f32::NEG_INFINITY; metric_ids.len()];
            update_ranked_plan_maxima::<f32>(
                &state.backend,
                complete_family,
                &metric_ids,
                &mut maxima,
            )?;
            Ok(CpuPrecisionValues::F32(maxima))
        }
        PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
            let mut maxima = vec![f64::NEG_INFINITY; metric_ids.len()];
            update_ranked_plan_maxima::<f64>(
                &state.backend,
                complete_family,
                &metric_ids,
                &mut maxima,
            )?;
            Ok(CpuPrecisionValues::F64(maxima))
        }
    }
}

fn update_table_maxima<T: HostSignificanceScalar>(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    maxima: &mut [T],
) -> Result<(), PyBoundaryError> {
    let all_values = T::metric_values(table)?;
    for row in 0..table.row_count() {
        let base = row.checked_mul(table.metric_count()).ok_or_else(|| {
            PyBoundaryError::InvalidInput("GPU significance metric offset overflow".to_string())
        })?;
        let values = all_values
            .get(base..base + table.metric_count())
            .ok_or_else(|| {
                PyBoundaryError::InvalidInput("GPU significance metric row is missing".to_string())
            })?;
        for (metric_index, &metric_id) in metric_ids.iter().enumerate() {
            let value = T::metric_extremeness(metric_id, values[metric_index]);
            if value > maxima[metric_index] {
                maxima[metric_index] = value;
            }
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
    table: &PrecisionOwnedResultTable,
    adaptive_search: bool,
) -> Result<Option<Vec<SignificanceEntry>>, PyBoundaryError> {
    if config.permutation_tests == 0
        || matches!(&state.backend, CompiledContinuousBackend::Cpu { .. })
    {
        return Ok(None);
    }
    require_device_ranking(&state.backend)?;
    if state.significance_matrix.is_none() {
        return Err(PyBoundaryError::InvalidInput(
            "GPU significance requires the retained host target".to_string(),
        ));
    }
    if table.row_count() == 0 {
        return Ok(Some(Vec::new()));
    }

    match table {
        PrecisionOwnedResultTable::Fp32(_) => {
            compute_host_orchestrated_gpu_permutation_pvalues_typed::<f32>(
                config,
                metric_ids,
                significance_top_n,
                runtime_cache_counters,
                state,
                table,
                adaptive_search,
            )
        }
        PrecisionOwnedResultTable::F64 { .. } => {
            compute_host_orchestrated_gpu_permutation_pvalues_typed::<f64>(
                config,
                metric_ids,
                significance_top_n,
                runtime_cache_counters,
                state,
                table,
                adaptive_search,
            )
        }
    }
}

fn compute_host_orchestrated_gpu_permutation_pvalues_typed<T: HostSignificanceScalar>(
    config: &EngineConfig,
    metric_ids: &[u32],
    significance_top_n: u32,
    runtime_cache_counters: &RefCell<RuntimeCacheCounters>,
    state: &mut ContinuousRunState,
    table: &PrecisionOwnedResultTable,
    adaptive_search: bool,
) -> Result<Option<Vec<SignificanceEntry>>, PyBoundaryError> {
    let host_matrix = state.significance_matrix.as_ref().ok_or_else(|| {
        PyBoundaryError::InvalidInput(
            "GPU significance requires the retained host target".to_string(),
        )
    })?;
    let order = significance_order(table, metric_ids, significance_top_n);
    let metric_count = metric_ids.len();
    let mut observed_flat = Vec::with_capacity(order.len() * metric_count);
    let metric_values = T::metric_values(table)?;
    for &row in &order {
        let base = row.checked_mul(metric_count).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance metric offset overflow".to_string())
        })?;
        observed_flat.extend_from_slice(metric_values.get(base..base + metric_count).ok_or_else(
            || PyBoundaryError::InvalidInput("significance metric row out of range".to_string()),
        )?);
    }
    let original_target = match host_matrix.target() {
        CpuPrecisionSlice::F32(target) => CpuPrecisionValues::F32(target.to_vec()),
        CpuPrecisionSlice::F64(target) => CpuPrecisionValues::F64(target.to_vec()),
    };
    let rows = host_matrix.rows();
    let cols = host_matrix.cols();
    let mut permutation_config = config.clone();
    permutation_config.permutation_tests = 0;
    permutation_config.num_repeats = 1;
    permutation_config.graph_requested = false;

    let permutation_result = (|| -> Result<Vec<u32>, PyBoundaryError> {
        let mut counts = vec![0u32; observed_flat.len()];
        for permutation_index in 0..config.permutation_tests {
            let target = permutation_target_precision(
                &original_target,
                config.random_seed,
                permutation_index,
            );
            update_gpu_target(&state.backend, precision_values_slice(&target))?;
            let mut maxima = vec![T::NEG_INFINITY; metric_count];
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
                update_table_maxima::<T>(&screened.unary_table, metric_ids, &mut maxima)?;
                update_ranked_plan_maxima::<T>(
                    &state.backend,
                    &screened.higher,
                    metric_ids,
                    &mut maxima,
                )?;
            } else {
                update_ranked_plan_maxima::<T>(
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
                        T::metric_extremeness(metric_ids[metric_index], observed_flat[flat_index]);
                    if maxima[metric_index] >= observed {
                        counts[flat_index] += 1;
                    }
                }
            }
        }
        Ok(counts)
    })();
    let restore_result =
        update_gpu_target(&state.backend, precision_values_slice(&original_target));
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
    Ok(Some(
        order
            .into_iter()
            .enumerate()
            .map(|(position, row)| {
                let base = position * metric_count;
                SignificanceEntry {
                    row,
                    pvalues: T::into_precision_values(
                        counts[base..base + metric_count]
                            .iter()
                            .map(|&count| T::pvalue(count, config.permutation_tests))
                            .collect(),
                    ),
                    means: T::into_precision_values(
                        observed_flat[base..base + metric_count].to_vec(),
                    ),
                    stds: T::into_precision_values(vec![T::ZERO; metric_count]),
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
    table: &PrecisionOwnedResultTable,
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
        backend
            .borrow()
            .supports_precision_permutation_pvalues(config.precision),
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
    for &row in &order {
        candidate_ids.push(table.candidate_ids()[row]);
    }

    let handle = matrix.handle();
    let complete_family = state.complete_family()?;
    let null_family_rows = complete_family.plan().planned_row_count();
    *runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(config.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: candidate_ids.len() as u64,
    };
    let base_protocol = complete_family.try_launch_protocol()?;
    let protocol = GafimePrecisionLaunchProtocol {
        abi_version: GAFIME_PRECISION_ABI_VERSION,
        profile: config.precision as u32,
        base: &base_protocol,
        reserved: [0; 8],
    };
    let device_budget_bytes = (config.budget.vram_budget_mb != 0)
        .then(|| config.budget.vram_budget_mb.saturating_mul(1024 * 1024));
    match table {
        PrecisionOwnedResultTable::Fp32(table) => {
            let mut observed = Vec::with_capacity(order.len() * metric_count);
            for &row in &order {
                let base = row * metric_count;
                observed.extend_from_slice(&table.metric_values()[base..base + metric_count]);
            }
            // SAFETY: `base_protocol` is materialized from `complete_family`,
            // which stays live and immutable through this synchronous call;
            // `protocol.base` points to that stack-local descriptor.
            let Some(pvalues) = (unsafe {
                backend
                    .borrow_mut()
                    .permutation_pvalues_fp32_v2_with_budget(
                        handle,
                        &protocol,
                        &candidate_ids,
                        &observed,
                        metric_count as u32,
                        device_budget_bytes,
                    )
            })?
            else {
                return Ok(None);
            };
            Ok(Some(
                order
                    .into_iter()
                    .enumerate()
                    .map(|(position, row)| {
                        let base = position * metric_count;
                        SignificanceEntry {
                            row,
                            pvalues: CpuPrecisionValues::F32(
                                pvalues[base..base + metric_count].to_vec(),
                            ),
                            means: CpuPrecisionValues::F32(
                                observed[base..base + metric_count].to_vec(),
                            ),
                            stds: CpuPrecisionValues::F32(vec![0.0; metric_count]),
                        }
                    })
                    .collect(),
            ))
        }
        PrecisionOwnedResultTable::F64 { table, .. } => {
            let mut observed = Vec::with_capacity(order.len() * metric_count);
            for &row in &order {
                let base = row * metric_count;
                observed.extend_from_slice(&table.metric_values()[base..base + metric_count]);
            }
            // SAFETY: `base_protocol` is materialized from `complete_family`,
            // which stays live and immutable through this synchronous call;
            // `protocol.base` points to that stack-local descriptor.
            let Some(pvalues) = (unsafe {
                backend.borrow_mut().permutation_pvalues_f64_v2_with_budget(
                    handle,
                    &protocol,
                    &candidate_ids,
                    &observed,
                    metric_count as u32,
                    device_budget_bytes,
                )
            })?
            else {
                return Ok(None);
            };
            Ok(Some(
                order
                    .into_iter()
                    .enumerate()
                    .map(|(position, row)| {
                        let base = position * metric_count;
                        SignificanceEntry {
                            row,
                            pvalues: CpuPrecisionValues::F64(
                                pvalues[base..base + metric_count].to_vec(),
                            ),
                            means: CpuPrecisionValues::F64(
                                observed[base..base + metric_count].to_vec(),
                            ),
                            stds: CpuPrecisionValues::F64(vec![0.0; metric_count]),
                        }
                    })
                    .collect(),
            ))
        }
    }
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
    table: &PrecisionOwnedResultTable,
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
    let mut candidate_ids = Vec::with_capacity(order.len());
    let mut observed = Vec::with_capacity(order.len());
    for &row in &order {
        let combo = combo_from_table(table, row).ok_or_else(|| {
            PyBoundaryError::InvalidInput("significance combo row out of range".to_string())
        })?;
        combos.push(combo);
        candidate_ids.push(table.candidate_ids()[row]);
        let metric_base = row * metric_ids.len();
        observed.push(match table {
            PrecisionOwnedResultTable::Fp32(table) => CpuPrecisionValues::F32(
                table.metric_values()[metric_base..metric_base + metric_ids.len()].to_vec(),
            ),
            PrecisionOwnedResultTable::F64 { table, .. } => CpuPrecisionValues::F64(
                table.metric_values()[metric_base..metric_base + metric_ids.len()].to_vec(),
            ),
        });
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
        significance::evaluate_precision_with_adaptive_search(
            matrix,
            &combos,
            &observed,
            &candidate_ids,
            complete_family.plan(),
            &kernels,
            &params,
            &search,
        )?
    } else {
        significance::evaluate_precision_with_null_family(
            matrix,
            &combos,
            &observed,
            &candidate_ids,
            complete_family.plan(),
            &kernels,
            &params,
        )?
    };
    order
        .into_iter()
        .zip(evaluated)
        .zip(candidate_ids)
        .map(|((row, sig), expected_candidate_id)| {
            if sig.candidate_id != expected_candidate_id {
                return Err(PyBoundaryError::InvalidInput(
                    "precision significance candidate identity changed".to_string(),
                ));
            }
            Ok(SignificanceEntry {
                row,
                pvalues: sig.pvalues,
                means: sig.means,
                stds: sig.stds,
            })
        })
        .collect::<Result<Vec<_>, PyBoundaryError>>()
}

fn rank_value_at(
    table: &impl ResultTableView,
    metric_ids: &[u32],
    row: usize,
    metric_index: Option<usize>,
) -> Option<PrecisionRankValue> {
    if row >= table.row_count() {
        return None;
    }
    let metric_count = table.metric_count();
    let metric_base = row.checked_mul(metric_count)?;
    match table.metric_values() {
        crate::common::MetricValuesRef::F32(values) => {
            let metrics = &values[metric_base..metric_base + metric_count];
            if let Some(metric_index) = metric_index {
                return metrics
                    .get(metric_index)
                    .copied()
                    .filter(|value| value.is_finite())
                    .map(PrecisionRankValue::F32);
            }
            metrics
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(idx, &value)| {
                    if matches!(
                        metric_ids.get(idx).copied().unwrap_or_default(),
                        GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN
                    ) {
                        value.abs()
                    } else {
                        value
                    }
                })
                .reduce(f32::max)
                .map(PrecisionRankValue::F32)
        }
        crate::common::MetricValuesRef::F64(values) => {
            let metrics = &values[metric_base..metric_base + metric_count];
            if let Some(metric_index) = metric_index {
                return metrics
                    .get(metric_index)
                    .copied()
                    .filter(|value| value.is_finite())
                    .map(PrecisionRankValue::F64);
            }
            metrics
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(idx, &value)| {
                    if matches!(
                        metric_ids.get(idx).copied().unwrap_or_default(),
                        GAFIME_METRIC_PEARSON | GAFIME_METRIC_SPEARMAN
                    ) {
                        value.abs()
                    } else {
                        value
                    }
                })
                .reduce(f64::max)
                .map(PrecisionRankValue::F64)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PrecisionRankValue {
    F32(f32),
    F64(f64),
}

fn compare_rank_values(
    left: Option<PrecisionRankValue>,
    right: Option<PrecisionRankValue>,
    descending: bool,
) -> std::cmp::Ordering {
    match (left, right) {
        (Some(PrecisionRankValue::F32(left)), Some(PrecisionRankValue::F32(right))) => {
            let ordering = left
                .partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
        (Some(PrecisionRankValue::F64(left)), Some(PrecisionRankValue::F64(right))) => {
            let ordering = left
                .partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
        (Some(PrecisionRankValue::F32(_)), Some(PrecisionRankValue::F64(_)))
        | (Some(PrecisionRankValue::F64(_)), Some(PrecisionRankValue::F32(_))) => {
            debug_assert!(false, "one result table cannot mix public score dtypes");
            std::cmp::Ordering::Equal
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

pub(crate) enum CompiledContinuousBackend {
    Cpu {
        matrix: CpuPrecisionMatrix,
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
    pub(crate) fn precision(&self) -> PrecisionProfile {
        match self {
            Self::Cpu { matrix } => matrix.profile(),
            Self::Cuda { matrix, .. } | Self::Rocm { matrix, .. } | Self::Metal { matrix, .. } => {
                matrix.precision()
            }
        }
    }

    pub(crate) fn uses_fp64_mi_accumulation(&self) -> bool {
        self.precision() != PrecisionProfile::Fp32
    }
}

pub(crate) struct ContinuousRunState {
    pub(crate) significance_matrix: Option<CpuPrecisionMatrix>,
    pub(crate) backend: CompiledContinuousBackend,
    pub(crate) primary: Option<PreparedContinuousExecution>,
    screened: Option<ScreenedContinuousExecution>,
    interaction_diagnostics_cache: Option<InteractionDiagnosticsCache>,
    pub(crate) result_capacity: u64,
    pub(crate) result_max_arity: u32,
    pub(crate) result_metric_count: u32,
}

struct InteractionDiagnosticsCache {
    row_count: usize,
    max_arity: usize,
    combo_indices: Vec<u32>,
    diagnostics: Option<Vec<InteractionPrecisionDiagnostic>>,
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
        self.interaction_diagnostics_cache = None;
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
        let bytes = screened_candidate_storage_bytes(
            PrecisionProfile::Fp32,
            10_000_000,
            0,
            10_000_000,
            5,
            1,
            0,
        );

        assert_eq!(bytes, 720_000_000);
        assert!(bytes > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES);
    }

    #[test]
    fn gpu_budget_preflight_rejects_before_payload_load_or_matrix_allocation() {
        let rows = 1_024u64;
        let cols = 512u32;
        let mut config = EngineConfig {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = u64::from(cols);
        config.budget.vram_budget_mb = 1;
        let input = OwnedNumericInput::from_f32(
            config.precision,
            vec![0.0; rows as usize * cols as usize],
            vec![0.0; rows as usize],
        )
        .unwrap();

        let error = match build_continuous_state(&config, rows, cols, input) {
            Ok(_) => panic!("oversized fp32 matrix unexpectedly passed static admission"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("continuous plan device footprint exceeds budget.vram_budget_mb"));
    }

    #[test]
    fn zero_vram_budget_skips_duplicate_gpu_plan_preflight() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            ..EngineConfig::default()
        };
        config.budget.vram_budget_mb = 0;
        assert!(!requires_gpu_budget_preflight(&config));

        config.budget.vram_budget_mb = 1;
        assert!(requires_gpu_budget_preflight(&config));

        config.backend_kind = GAFIME_BACKEND_CPU;
        assert!(!requires_gpu_budget_preflight(&config));
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

        assert_eq!(report.precision, PrecisionProfile::Mixed);
        assert_eq!(report.rows, 4);
        assert_eq!(report.cols, 3);
        assert_eq!(report.len(), 3);
        assert_eq!(report.combo(0).unwrap(), vec![0]);
        assert!((report.metric_values(0).unwrap().as_f64().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((report.metric_values(1).unwrap().as_f64().unwrap()[0] + 1.0).abs() < 1e-6);
        assert_eq!(report.metric_values(2).unwrap().as_f64().unwrap()[0], 0.0);
    }

    #[test]
    fn profile_aware_cpu_boundary_routes_typed_inputs_and_results() {
        for precision in [PrecisionProfile::Fp32, PrecisionProfile::Mixed] {
            let report = analyze_continuous_cpu_rows_with_precision(
                precision,
                4,
                1,
                ContinuousCpuInput::F32 {
                    features: vec![1.0, 2.0, 3.0, 4.0],
                    target: vec![1.0, 2.0, 3.0, 4.0],
                },
                1,
                1,
                vec![GAFIME_METRIC_PEARSON],
            )
            .unwrap();

            assert_eq!(report.precision, precision);
            let metric_values = report.metric_values(0).unwrap();
            let value = match precision {
                PrecisionProfile::Fp32 => f64::from(metric_values.as_f32().unwrap()[0]),
                PrecisionProfile::Mixed => metric_values.as_f64().unwrap()[0],
                PrecisionProfile::Fp64 => unreachable!(),
            };
            assert!((value - 1.0).abs() < 1.0e-6);
            match (precision, report.table) {
                (PrecisionProfile::Fp32, PrecisionOwnedResultTable::Fp32(_))
                | (
                    PrecisionProfile::Mixed,
                    PrecisionOwnedResultTable::F64 {
                        profile: PrecisionProfile::Mixed,
                        ..
                    },
                ) => {}
                (precision, table) => {
                    panic!("unexpected result lane for {precision:?}: {table:?}")
                }
            }
        }

        let delta = 2.0_f64.powi(-30);
        let values = (0..16)
            .map(|index| 1.0 + f64::from(index) * delta)
            .collect::<Vec<_>>();
        let report = analyze_continuous_cpu_rows_with_precision(
            PrecisionProfile::Fp64,
            16,
            1,
            ContinuousCpuInput::F64 {
                features: values.clone(),
                target: values,
            },
            1,
            1,
            vec![GAFIME_METRIC_PEARSON],
        )
        .unwrap();

        assert_eq!(report.precision, PrecisionProfile::Fp64);
        assert!((report.metric_values(0).unwrap().as_f64().unwrap()[0] - 1.0).abs() < 1.0e-12);
        assert!(matches!(
            report.table,
            PrecisionOwnedResultTable::F64 {
                profile: PrecisionProfile::Fp64,
                ..
            }
        ));
    }

    #[test]
    fn profile_aware_cpu_boundary_rejects_typed_lane_mismatch() {
        let error = analyze_continuous_cpu_rows_with_precision(
            PrecisionProfile::Fp64,
            2,
            1,
            ContinuousCpuInput::F32 {
                features: vec![1.0, 2.0],
                target: vec![1.0, 2.0],
            },
            1,
            1,
            vec![GAFIME_METRIC_PEARSON],
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("fp64 precision cannot ingest an intermediate f32 buffer"));
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
    fn compiled_diagnostics_cache_reuses_exact_shape_and_plan_changes_invalidate() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 10;
        config.budget.top_features_for_higher_k = 3;
        let features = vec![1.0, 4.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 0.0, 4.0, 1.0, -1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let mut compiled = compile_continuous_rows(config.clone(), 4, 3, features, target).unwrap();

        let first = execute_compiled_artifact(&mut compiled).unwrap();
        assert!(first
            .interaction_diagnostics
            .as_ref()
            .is_some_and(|diagnostics| !diagnostics.is_empty()));
        compiled
            .state
            .as_mut()
            .unwrap()
            .interaction_diagnostics_cache
            .as_mut()
            .unwrap()
            .diagnostics
            .as_mut()
            .unwrap()[0]
            .overflow_row_count = 7;

        let repeated = execute_compiled_artifact(&mut compiled).unwrap();
        assert_eq!(
            repeated.interaction_diagnostics.as_ref().unwrap()[0].overflow_row_count,
            7
        );

        let replacement = {
            let state = compiled.state.as_ref().unwrap();
            prepare_screened_continuous_execution(&config, 4, 3, &state.backend).unwrap()
        };
        let state = compiled.state.as_mut().unwrap();
        state.replace_plans(replacement);
        assert!(state.interaction_diagnostics_cache.is_none());
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
            precision: PrecisionProfile::Fp32,
            metric_ids: vec![GAFIME_METRIC_R2, GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = 8;
        let input = OwnedNumericInput::from_f32(
            config.precision,
            vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let state = build_continuous_state(&config, 4, 2, input).unwrap();
        let prepared = state.primary.as_ref().unwrap();

        assert_eq!(
            execute_ranked_metric_extremum::<f32>(
                &state.backend,
                prepared,
                GAFIME_METRIC_PEARSON,
                true,
            )
            .unwrap(),
            1.0_f32
        );
        assert_eq!(
            execute_ranked_metric_extremum::<f32>(
                &state.backend,
                prepared,
                GAFIME_METRIC_PEARSON,
                false,
            )
            .unwrap(),
            -1.0_f32
        );
    }

    #[test]
    fn ranked_extremum_accepts_a_valid_empty_device_result() {
        let table = OwnedResultTable::new(1, 5, 1);

        assert_eq!(
            ranked_metric_value::<f32>(&table, 0).unwrap(),
            f32::NEG_INFINITY
        );
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

    #[test]
    fn fp32_ranking_excludes_nonfinite_public_scores() {
        let mut table = OwnedResultTable::new(3, 1, 2);
        let raw = table.raw_mut();
        // SAFETY: the owner allocated three rows of two f32 metrics and three
        // candidate identifiers; these writes stay within those live buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(
                [f32::NAN, f32::NAN, -0.25, f32::NAN, f32::NAN, 0.75].as_ptr(),
                raw.metric_values,
                6,
            );
            std::ptr::copy_nonoverlapping([30_u64, 20, 10].as_ptr(), raw.candidate_ids, 3);
        }
        raw.row_count = 3;

        assert_eq!(
            bounded_ranked_indices(
                &table,
                &[GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
                None,
                true,
                3,
            ),
            vec![2, 1]
        );
        assert_eq!(
            bounded_ranked_indices(
                &table,
                &[GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
                Some(0),
                true,
                3,
            ),
            vec![1]
        );
    }

    #[test]
    fn f64_ranking_excludes_nonfinite_public_scores() {
        let mut table = OwnedResultTableF64::new(3, 1, 1);
        table
            .metric_values_mut()
            .copy_from_slice(&[f64::NAN, 0.5, f64::INFINITY]);
        table.raw_mut().row_count = 3;

        assert_eq!(
            bounded_ranked_indices(&table, &[GAFIME_METRIC_R2], Some(0), true, 3),
            vec![1]
        );
        assert!(rank_value_at(&table, &[GAFIME_METRIC_R2], 0, Some(0)).is_none());
        assert!(rank_value_at(&table, &[GAFIME_METRIC_R2], 2, Some(0)).is_none());
    }
}
