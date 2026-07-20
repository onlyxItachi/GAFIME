use std::{cell::RefCell, collections::HashSet, error::Error, ffi::CString, fmt, sync::Arc};

mod legacy_helpers;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Fields};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use gafime_cpu::{
    kernels::MetricKernel,
    matrix::CpuMatrix,
    result::OwnedResultTable,
    significance::{self, AdaptiveSearchSpec, SignificanceParams},
    simd::{finite_dispatch_isa, IsaLevel},
    CpuBackend,
};
use gafime_gpu_sys::{
    GpuArchitectureClass, GpuBackend, GpuDeviceProfile, GpuSysError, OwnedGpuMatrix,
};
use gafime_orchestrator::{
    config::EngineConfig,
    plan::combos::{
        legacy_higher_feature_order, legacy_unary_feature_order,
        DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES,
    },
    prepare_continuous_execution_for_feature_orders, OrchestratorError,
    PreparedContinuousExecution,
};
use gafime_types::{
    GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimeRankSpec, GAFIME_BACKEND_CPU,
    GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_GRAPH_HOST_REPLAY,
    GAFIME_GRAPH_STREAM_CAPTURE, GAFIME_GRAPH_UNSUPPORTED, GAFIME_METRIC_MUTUAL_INFO,
    GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
    GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes, PyCapsule, PyDict},
};

use legacy_helpers::{
    PyBatchScheduler, PyCacheAwareScheduler, PyDataQualityAnalyzer, PyOTSEncoder, PySmartScheduler,
};

trait ResultTableView {
    fn row_count(&self) -> usize;
    fn metric_count(&self) -> usize;
    fn max_arity(&self) -> usize;
    fn combo_indices(&self) -> &[u32];
    fn metric_values(&self) -> &[f32];
    fn ranks(&self) -> &[u32];
    fn candidate_ids(&self) -> &[u64];
}

impl ResultTableView for OwnedResultTable {
    fn row_count(&self) -> usize {
        OwnedResultTable::row_count(self)
    }

    fn metric_count(&self) -> usize {
        OwnedResultTable::metric_count(self)
    }

    fn max_arity(&self) -> usize {
        OwnedResultTable::max_arity(self)
    }

    fn combo_indices(&self) -> &[u32] {
        OwnedResultTable::combo_indices(self)
    }

    fn metric_values(&self) -> &[f32] {
        OwnedResultTable::metric_values(self)
    }

    fn ranks(&self) -> &[u32] {
        OwnedResultTable::ranks(self)
    }

    fn candidate_ids(&self) -> &[u64] {
        OwnedResultTable::candidate_ids(self)
    }
}

#[derive(Debug)]
struct SendOwnedResultTable(OwnedResultTable);

// SAFETY: this wrapper is private and is constructed only by consuming a
// completed synchronous execution result. OwnedResultTable owns primitive Vec
// buffers; its raw C descriptor only aliases those stable heap allocations and
// has no destructor. Report methods expose immutable slices and never call
// raw()/raw_mut(), so neither the descriptor nor the buffers can be rebound or
// mutated after wrapping. Moving or dropping this owner on another thread is
// therefore equivalent to moving or dropping the owned Vec buffers themselves.
unsafe impl Send for SendOwnedResultTable {}

impl ResultTableView for SendOwnedResultTable {
    fn row_count(&self) -> usize {
        self.0.row_count()
    }

    fn metric_count(&self) -> usize {
        self.0.metric_count()
    }

    fn max_arity(&self) -> usize {
        self.0.max_arity()
    }

    fn combo_indices(&self) -> &[u32] {
        self.0.combo_indices()
    }

    fn metric_values(&self) -> &[f32] {
        self.0.metric_values()
    }

    fn ranks(&self) -> &[u32] {
        self.0.ranks()
    }

    fn candidate_ids(&self) -> &[u64] {
        self.0.candidate_ids()
    }
}

/// Build a zero-copy-to-consumer Arrow `StructArray` over the compact result
/// table. Columns: `candidate_id` (u64), `rank` (u32), `combo`
/// (FixedSizeList<u32>[max_arity]), `metrics` (FixedSizeList<f32>[metric_count]).
/// The only copy is the compact (top-K) table into Arrow-owned buffers; the
/// Arrow -> framework (Polars/torch/pyarrow) handoff is then zero-copy, and the
/// FFI release callbacks are owned by arrow-rs (no hand-rolled unsafe).
fn result_table_view_to_arrow(table: &impl ResultTableView) -> StructArray {
    let rows = table.row_count();
    let metric_count = table.metric_count();
    let max_arity = table.max_arity();

    let candidate_id =
        Arc::new(UInt64Array::from(table.candidate_ids()[..rows].to_vec())) as ArrayRef;
    let rank = Arc::new(UInt32Array::from(table.ranks()[..rows].to_vec())) as ArrayRef;

    let combo_item = Arc::new(Field::new("item", DataType::UInt32, false));
    let combo_child = Arc::new(UInt32Array::from(
        table.combo_indices()[..rows * max_arity].to_vec(),
    )) as ArrayRef;
    let combo = Arc::new(FixedSizeListArray::new(
        combo_item.clone(),
        max_arity as i32,
        combo_child,
        None,
    )) as ArrayRef;

    let metric_item = Arc::new(Field::new("item", DataType::Float32, false));
    let metric_child = Arc::new(Float32Array::from(
        table.metric_values()[..rows * metric_count].to_vec(),
    )) as ArrayRef;
    let metrics = Arc::new(FixedSizeListArray::new(
        metric_item.clone(),
        metric_count as i32,
        metric_child,
        None,
    )) as ArrayRef;

    let fields = Fields::from(vec![
        Field::new("candidate_id", DataType::UInt64, false),
        Field::new("rank", DataType::UInt32, false),
        Field::new(
            "combo",
            DataType::FixedSizeList(combo_item, max_arity as i32),
            false,
        ),
        Field::new(
            "metrics",
            DataType::FixedSizeList(metric_item, metric_count as i32),
            false,
        ),
    ]);
    StructArray::new(fields, vec![candidate_id, rank, combo, metrics], None)
}

pub fn result_table_to_arrow(table: &OwnedResultTable) -> StructArray {
    result_table_view_to_arrow(table)
}

pub const BOUNDARY_NAME: &str = "gafime-py";

fn cargo_version_to_python(cargo_version: &str) -> String {
    let Some((release, prerelease)) = cargo_version.split_once('-') else {
        return cargo_version.to_string();
    };
    let mut prerelease_parts = prerelease.split('.');
    let Some(label) = prerelease_parts.next() else {
        return cargo_version.to_string();
    };
    let Some(serial) = prerelease_parts.next() else {
        return cargo_version.to_string();
    };
    if prerelease_parts.next().is_some()
        || serial.is_empty()
        || !serial.bytes().all(|byte| byte.is_ascii_digit())
    {
        return cargo_version.to_string();
    }
    let pep440_label = match label {
        "alpha" => "a",
        "beta" => "b",
        "rc" => "rc",
        _ => return cargo_version.to_string(),
    };
    format!("{release}{pep440_label}{serial}")
}

pub fn public_package_version() -> String {
    cargo_version_to_python(env!("CARGO_PKG_VERSION"))
}

#[pyfunction]
fn native_version() -> String {
    public_package_version()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousRecord {
    pub combo: Vec<u32>,
    pub metrics: Vec<f32>,
    pub candidate_id: u64,
}

/// Per-candidate significance for one surfaced report row: permutation p-values
/// and bootstrap stability (mean/std) per metric, aligned to `metric_ids`.
#[derive(Clone, Debug, PartialEq)]
pub struct SignificanceEntry {
    pub row: usize,
    pub pvalues: Vec<f32>,
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct DecisionPathResultParams {
    feature_index: u32,
    features: Vec<u32>,
    thresholds: Vec<f32>,
    signs: Vec<i8>,
    gain: f32,
    support: u32,
    round_id: u32,
}

impl DecisionPathResultParams {
    fn from_path(feature_index: u32, path: &gafime_cpu::decision_path::DecisionPath) -> Self {
        let mut features = Vec::with_capacity(path.nodes.len());
        let mut thresholds = Vec::with_capacity(path.nodes.len());
        let mut signs = Vec::with_capacity(path.nodes.len());
        for node in &path.nodes {
            features.push(node.feature);
            thresholds.push(node.threshold);
            signs.push(match node.sign {
                gafime_cpu::decision_path::SplitSign::Le => -1,
                gafime_cpu::decision_path::SplitSign::Gt => 1,
            });
        }
        Self {
            feature_index,
            features,
            thresholds,
            signs,
            gain: path.gain,
            support: path.support,
            round_id: path.round,
        }
    }
}

#[derive(Debug)]
pub struct ContinuousReport {
    pub rows: u64,
    pub cols: u32,
    pub max_arity: u32,
    pub metric_ids: Vec<u32>,
    pub backend_kind: u32,
    pub graph_replayed: bool,
    table: OwnedResultTable,
    significance: Vec<SignificanceEntry>,
}

impl ContinuousReport {
    pub fn len(&self) -> usize {
        self.table.row_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn combo(&self, index: usize) -> Option<Vec<u32>> {
        combo_from_table(&self.table, index)
    }

    pub fn metric_values(&self, index: usize) -> Option<Vec<f32>> {
        metric_values_from_table(&self.table, index)
    }

    pub fn candidate_id(&self, index: usize) -> Option<u64> {
        self.table.candidate_ids().get(index).copied()
    }
}

#[derive(Debug)]
pub enum PyBoundaryError {
    InvalidInput(String),
    UnsupportedFeature(String),
    Orchestrator(OrchestratorError),
    Gpu(GpuSysError),
}

impl fmt::Display for PyBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid v1 boundary input: {message}"),
            Self::UnsupportedFeature(message) => write!(f, "unsupported v1 feature: {message}"),
            Self::Orchestrator(error) => write!(f, "v1 orchestrator error: {error:?}"),
            Self::Gpu(error) => write!(f, "v1 GPU boundary error: {error}"),
        }
    }
}

impl Error for PyBoundaryError {}

impl From<OrchestratorError> for PyBoundaryError {
    fn from(value: OrchestratorError) -> Self {
        Self::Orchestrator(value)
    }
}

impl From<GpuSysError> for PyBoundaryError {
    fn from(value: GpuSysError) -> Self {
        Self::Gpu(value)
    }
}

impl From<PyBoundaryError> for PyErr {
    fn from(value: PyBoundaryError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

pub fn boundary_name() -> &'static str {
    BOUNDARY_NAME
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

fn continuous_config_for_cpu(
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
    let mut config = EngineConfig::default();
    config.backend_kind = GAFIME_BACKEND_CPU;
    config.metric_ids = validate_metric_ids(metric_ids)?;
    config.budget.max_comb_size = max_arity;
    config.budget.max_combinations_per_k = max_combinations_per_k;
    // Significance is opt-in via the full-config `compile_continuous` path; the
    // low-level convenience entrypoints stay raw/fast (no permutation/stability).
    config.permutation_tests = 0;
    config.num_repeats = 1;
    Ok(config)
}

#[cfg(test)]
fn compile_continuous_cpu_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    if config.backend_kind != GAFIME_BACKEND_CPU {
        return Err(PyBoundaryError::InvalidInput(
            "compile_continuous_cpu_rows requires a CPU backend config".to_string(),
        ));
    }
    compile_continuous_rows(config, rows, cols, features, target)
}

fn compile_continuous_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    validate_shape(rows, cols, features.len(), target.len())?;
    let state = build_continuous_state(&config, rows, cols, features, target)?;
    let max_arity = state.result_max_arity;
    Ok(PyCompiledContinuousArtifact {
        config: config.clone(),
        rows,
        cols,
        max_arity,
        metric_ids: config.metric_ids.clone(),
        significance_top_n: config.significance_top_n,
        state: Some(state),
        runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
        decision_path_params: Vec::new(),
        target_updates_supported: true,
        closed: false,
    })
}

fn analyze_continuous_rows_once(
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

fn build_continuous_state(
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

struct PreparedContinuousRun {
    /// Direct execution plan, or the complete screened family retained only for
    /// graph/significance protocols. Stateless screened runs do not build it.
    primary: Option<PreparedContinuousExecution>,
    screened: Option<ScreenedContinuousExecution>,
    result_capacity: u64,
    result_max_arity: u32,
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

fn prepare_screened_continuous_execution(
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

fn unary_strengths_from_table(
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

fn execute_compiled_artifact(
    artifact: &mut PyCompiledContinuousArtifact,
) -> Result<ContinuousReport, PyBoundaryError> {
    if artifact.closed {
        return Err(PyBoundaryError::InvalidInput(
            "compiled artifact is closed".to_string(),
        ));
    }
    let result = {
        let state = artifact.state.as_mut().ok_or_else(|| {
            PyBoundaryError::InvalidInput("compiled artifact is closed".to_string())
        })?;
        execute_continuous_state(
            &artifact.config,
            artifact.rows,
            artifact.cols,
            &artifact.metric_ids,
            artifact.significance_top_n,
            state,
            &artifact.runtime_cache_counters,
        )
    };
    if result.is_err() && backend_is_gpu(artifact.backend_kind()) {
        artifact.state = None;
        artifact.closed = true;
    }
    result
}

fn execute_continuous_state(
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

    if table.row_count() == 0 {
        return Ok(report_from_table(
            rows,
            cols,
            state.result_max_arity,
            metric_ids.to_vec(),
            config.backend_kind,
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

fn compare_ranked_rows(
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

fn bounded_ranked_indices(
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

fn validate_shape(
    rows: u64,
    cols: u32,
    feature_len: usize,
    target_len: usize,
) -> Result<(), PyBoundaryError> {
    if rows == 0 || cols == 0 {
        return Err(PyBoundaryError::InvalidInput(
            "rows and cols must both be nonzero".to_string(),
        ));
    }
    let rows_usize = usize::try_from(rows)
        .map_err(|_| PyBoundaryError::InvalidInput("rows exceed host address space".to_string()))?;
    let cols_usize = usize::try_from(cols)
        .map_err(|_| PyBoundaryError::InvalidInput("cols exceed host address space".to_string()))?;
    let expected_features = rows_usize.checked_mul(cols_usize).ok_or_else(|| {
        PyBoundaryError::InvalidInput("rows*cols exceed host address space".to_string())
    })?;
    if feature_len != expected_features {
        return Err(PyBoundaryError::InvalidInput(
            "feature buffer length does not match rows*cols".to_string(),
        ));
    }
    if target_len != rows_usize {
        return Err(PyBoundaryError::InvalidInput(
            "target length does not match rows".to_string(),
        ));
    }
    Ok(())
}

fn decode_f32_le(bytes: &[u8], label: &str) -> Result<Vec<f32>, PyBoundaryError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(PyBoundaryError::InvalidInput(format!(
            "{label} byte length is not divisible by four"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn flatten_continuous_rows(
    features: Vec<Vec<f32>>,
    target: Vec<f32>,
) -> Result<(u64, u32, Vec<f32>, Vec<f32>), PyBoundaryError> {
    let rows = u64::try_from(features.len()).map_err(|_| {
        PyBoundaryError::InvalidInput("X row count exceeds the supported range".to_string())
    })?;
    let cols = features.first().map_or(0usize, Vec::len);
    let cols = u32::try_from(cols).map_err(|_| {
        PyBoundaryError::InvalidInput("X feature count exceeds the supported range".to_string())
    })?;
    let capacity = features.len().checked_mul(cols as usize).ok_or_else(|| {
        PyBoundaryError::InvalidInput("X shape exceeds the supported range".to_string())
    })?;
    validate_shape(rows, cols, capacity, target.len())?;

    let mut flat = Vec::with_capacity(capacity);
    for (row_index, row) in features.into_iter().enumerate() {
        if row.len() != cols as usize {
            return Err(PyBoundaryError::InvalidInput(format!(
                "X row {row_index} has length {}; expected {cols}",
                row.len()
            )));
        }
        flat.extend(row);
    }
    Ok((rows, cols, flat, target))
}

fn validate_metric_ids(metric_ids: Vec<u32>) -> Result<Vec<u32>, PyBoundaryError> {
    if metric_ids.is_empty() {
        return Err(PyBoundaryError::InvalidInput(
            "metric_names must contain at least one metric".to_string(),
        ));
    }
    for metric_id in &metric_ids {
        match *metric_id {
            GAFIME_METRIC_PEARSON
            | GAFIME_METRIC_SPEARMAN
            | GAFIME_METRIC_MUTUAL_INFO
            | GAFIME_METRIC_R2 => {}
            _ => {
                return Err(PyBoundaryError::InvalidInput(format!(
                    "unknown metric id {metric_id}"
                )))
            }
        }
    }
    Ok(metric_ids)
}

fn report_from_table(
    rows: u64,
    cols: u32,
    max_arity: u32,
    metric_ids: Vec<u32>,
    backend_kind: u32,
    table: OwnedResultTable,
    significance: Vec<SignificanceEntry>,
) -> ContinuousReport {
    let graph_replayed = (table.raw().flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0;
    ContinuousReport {
        rows,
        cols,
        max_arity,
        metric_ids,
        backend_kind,
        graph_replayed,
        table,
        significance,
    }
}

fn combo_from_table(table: &impl ResultTableView, index: usize) -> Option<Vec<u32>> {
    if index >= table.row_count() {
        return None;
    }
    let max_arity = table.max_arity();
    let combo_base = index.checked_mul(max_arity)?;
    Some(
        table.combo_indices()[combo_base..combo_base + max_arity]
            .iter()
            .copied()
            .filter(|&feature| feature != u32::MAX)
            .collect(),
    )
}

fn metric_values_from_table(table: &impl ResultTableView, index: usize) -> Option<Vec<f32>> {
    if index >= table.row_count() {
        return None;
    }
    let metric_count = table.metric_count();
    let metric_base = index.checked_mul(metric_count)?;
    Some(table.metric_values()[metric_base..metric_base + metric_count].to_vec())
}

fn parse_engine_config(config: &Bound<'_, PyDict>) -> PyResult<EngineConfig> {
    validate_family_flags(
        get_bool(config, "enable_time_series_functions", false)?,
        get_bool(config, "enable_decision_path_functions", false)?,
    )
    .map_err(PyErr::from)?;

    let mut out = EngineConfig::default();
    let backend_name = get_string(config, "backend", "auto")?;
    out.device_id = get_u32(config, "device_id", 0)?;
    out.backend_kind = backend_kind_from_name(&backend_name, out.device_id)?;
    out.metric_ids = metric_ids_from_names(get_vec_string(config, "metric_names")?)?;
    out.num_repeats = get_u32(config, "num_repeats", 3)?;
    out.permutation_tests = get_u32(config, "permutation_tests", 25)?;
    out.significance_top_n = get_u32(config, "significance_top_n", 50)?;
    let (random_seed, planning_seed_words) = get_python_integer_seed(config, "random_seed")?;
    out.random_seed = random_seed;
    out.planning_seed_words = planning_seed_words;
    out.mi_bins = get_u32(config, "mi_bins", 96)?;
    out.mi_approximate = get_bool(config, "mi_approximate", false)?;
    if let Some(flags) = get_optional_dict(config, "compile_flags")? {
        out.graph_requested = get_bool(&flags, "graph", false)?;
    }

    if let Some(budget) = get_optional_dict(config, "budget")? {
        out.budget.max_comb_size = get_u32(&budget, "max_comb_size", 2)?;
        out.budget.max_combinations_per_k = get_u64(&budget, "max_combinations_per_k", 5_000)?;
        out.budget.top_features_for_higher_k = get_u32(&budget, "top_features_for_higher_k", 50)?;
        out.budget.max_generated_features = get_u32(&budget, "max_generated_features", 0)?;
        out.budget.max_time_series_candidates =
            get_u64(&budget, "max_time_series_candidates", 100_000)?;
        out.budget.top_k_features_for_time_series =
            get_u32(&budget, "top_k_features_for_time_series", 50)?;
        out.budget.max_feature_candidate = match get_optional_i64(&budget, "max_feature_candidate")?
        {
            None => -2,
            Some(value) if value >= -1 => value,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "budget.max_feature_candidate must be >= 0 or -1 for power-user mode",
                ))
            }
        };
        out.budget.vram_budget_mb = get_u64(&budget, "vram_budget_mb", 6_144)?;
    }

    if out.budget.max_comb_size == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "budget.max_comb_size must be greater than zero".to_string(),
        )));
    }
    if out.budget.max_combinations_per_k == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "budget.max_combinations_per_k must be greater than zero".to_string(),
        )));
    }
    if out.significance_top_n == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "significance_top_n must be greater than zero".to_string(),
        )));
    }
    if out.mi_bins < 2 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "mi_bins must be at least 2".to_string(),
        )));
    }
    Ok(out)
}

fn validate_family_flags(
    enable_time_series: bool,
    enable_decision_path: bool,
) -> Result<(), PyBoundaryError> {
    if enable_time_series {
        return Err(PyBoundaryError::UnsupportedFeature(
            "time-series families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        ));
    }
    if enable_decision_path {
        return Err(PyBoundaryError::UnsupportedFeature(
            "decision-path families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        ));
    }
    Ok(())
}

fn backend_kind_from_name(name: &str, device_id: u32) -> PyResult<u32> {
    backend_kind_from_name_result(name, device_id).map_err(PyErr::from)
}

fn backend_kind_from_name_result(name: &str, device_id: u32) -> Result<u32, PyBoundaryError> {
    match name {
        "auto" => Ok(resolve_auto_backend(device_id)),
        "cpu" | "core" | "rust" | "v1-rust-cpu" => Ok(GAFIME_BACKEND_CPU),
        "cuda" => Ok(GAFIME_BACKEND_CUDA),
        "gpu" => Err(PyBoundaryError::UnsupportedFeature(
            "backend \"gpu\" is ambiguous in v1; request backend \"cuda\", \"rocm\", or \"metal\" explicitly"
                .to_string(),
        )),
        "rocm" | "hip" => Ok(GAFIME_BACKEND_ROCM),
        "metal" => Ok(GAFIME_BACKEND_METAL),
        other => Err(PyBoundaryError::InvalidInput(format!(
            "unknown backend {other:?}"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AutoBackendCandidate {
    kind: u32,
    score: i64,
}

#[derive(Clone, Copy, Debug, Default)]
struct RuntimeCacheCounters {
    metric_hits: u64,
    metric_builds: u64,
    candidate_table_hits: u64,
}

fn resolve_auto_backend(device_id: u32) -> u32 {
    [
        probe_gpu_candidate(GAFIME_BACKEND_CUDA, device_id),
        probe_gpu_candidate(GAFIME_BACKEND_ROCM, device_id),
        probe_gpu_candidate(GAFIME_BACKEND_METAL, device_id),
    ]
    .into_iter()
    .flatten()
    .max_by_key(|candidate| (candidate.score, backend_tie_breaker(candidate.kind)))
    .map(|candidate| candidate.kind)
    .unwrap_or_else(|| {
        let _ = cpu_isa_rank(finite_dispatch_isa());
        GAFIME_BACKEND_CPU
    })
}

#[derive(Clone, Debug)]
struct GpuRuntimeProbe {
    kind: u32,
    info: GafimeGpuDeviceInfo,
    graph: GafimeGpuGraphCapability,
    supports_permutation_pvalues: bool,
    supports_decision_path_membership: bool,
    supports_decision_path_score: bool,
    library_path: Option<String>,
}

fn probe_gpu_runtime(kind: u32, device_id: u32) -> Result<GpuRuntimeProbe, GpuSysError> {
    let backend = match kind {
        GAFIME_BACKEND_CUDA => GpuBackend::cuda_from_env(device_id),
        GAFIME_BACKEND_ROCM => GpuBackend::rocm_from_env(device_id),
        GAFIME_BACKEND_METAL => GpuBackend::metal_from_env(device_id),
        _ => return Err(GpuSysError::InvalidInput("unsupported GPU backend kind")),
    }?;
    let library_path = backend
        .loaded_library_path()
        .map(|path| path.display().to_string());
    let info = backend.device_info()?;
    let graph = backend.graph_capability()?;
    Ok(GpuRuntimeProbe {
        kind,
        info,
        graph,
        supports_permutation_pvalues: backend.supports_permutation_pvalues(),
        supports_decision_path_membership: backend.supports_decision_path_membership(),
        supports_decision_path_score: backend.supports_decision_path_score(),
        library_path,
    })
}

fn probe_gpu_candidate(kind: u32, device_id: u32) -> Option<AutoBackendCandidate> {
    // A payload is eligible for automatic selection only when its required
    // identity/capability query succeeds. Allocation success is not a substitute:
    // older or mismatched payloads can allocate while exposing an incompatible ABI.
    let probe = probe_gpu_runtime(kind, device_id).ok()?;
    Some(AutoBackendCandidate {
        kind,
        score: gpu_device_score(&probe.info),
    })
}

fn gpu_device_score(info: &GafimeGpuDeviceInfo) -> i64 {
    let profile = GpuDeviceProfile::from_info(info);
    let mut score = 1_000_000i64;
    score += match profile.architecture {
        GpuArchitectureClass::NvidiaBlackwell => 80_000,
        GpuArchitectureClass::NvidiaHopper => 75_000,
        GpuArchitectureClass::NvidiaAda => 70_000,
        GpuArchitectureClass::NvidiaAmpere => 62_000,
        GpuArchitectureClass::NvidiaTuring => 52_000,
        GpuArchitectureClass::AmdCdna => 68_000,
        GpuArchitectureClass::AmdRdna => 58_000,
        GpuArchitectureClass::Apple => 54_000,
        GpuArchitectureClass::VendorSpecific(value) => 30_000 + (value.min(20_000) as i64),
        GpuArchitectureClass::Unknown => 20_000,
    };
    if profile.discrete {
        score += 12_000;
    }
    if profile.high_bandwidth {
        score += 8_000;
    }
    if profile.unified_memory {
        score += 4_000;
    }
    if profile.integrated {
        score += 2_000;
    }
    if profile.managed_memory {
        score += 1_000;
    }
    score += (info.total_global_mem_bytes / (1024 * 1024 * 512)).min(256) as i64;
    score += (info.multiprocessor_count as i64) * 64;
    score += (info.compute_major as i64) * 128 + (info.compute_minor as i64);
    score
}

fn backend_tie_breaker(kind: u32) -> i64 {
    match kind {
        GAFIME_BACKEND_CUDA => 30,
        GAFIME_BACKEND_ROCM => 20,
        GAFIME_BACKEND_METAL => 10,
        _ => 0,
    }
}

fn cpu_isa_rank(isa: IsaLevel) -> i64 {
    match isa {
        IsaLevel::Avx512 => 50_000,
        IsaLevel::Avx2 => 40_000,
        IsaLevel::Sse42 | IsaLevel::Neon => 30_000,
        IsaLevel::Scalar => 10_000,
    }
}

fn normalize_runtime_backend(name: &str) -> Result<&'static str, PyBoundaryError> {
    match name {
        "auto" => Ok("auto"),
        "cpu" | "core" | "rust" | "v1-rust-cpu" => Ok("core"),
        "cuda" => Ok("cuda"),
        "rocm" | "hip" => Ok("rocm"),
        "metal" => Ok("metal"),
        "gpu" => Err(PyBoundaryError::UnsupportedFeature(
            "backend \"gpu\" is ambiguous in v1; request backend \"cuda\", \"rocm\", or \"metal\" explicitly"
                .to_string(),
        )),
        other => Err(PyBoundaryError::InvalidInput(format!(
            "unknown backend {other:?}"
        ))),
    }
}

fn backend_kind_for_runtime_name(name: &str) -> u32 {
    match name {
        "cuda" => GAFIME_BACKEND_CUDA,
        "rocm" => GAFIME_BACKEND_ROCM,
        "metal" => GAFIME_BACKEND_METAL,
        _ => GAFIME_BACKEND_CPU,
    }
}

fn device_name(info: &GafimeGpuDeviceInfo) -> String {
    let length = info
        .name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(info.name.len());
    String::from_utf8_lossy(&info.name[..length]).into_owned()
}

fn graph_mode_name(mode: u32) -> &'static str {
    match mode {
        GAFIME_GRAPH_UNSUPPORTED => "unsupported",
        GAFIME_GRAPH_STREAM_CAPTURE => "stream_capture",
        GAFIME_GRAPH_HOST_REPLAY => "host_replay",
        _ => "vendor_specific",
    }
}

fn runtime_probe_to_python<'py>(
    py: Python<'py>,
    probe: &GpuRuntimeProbe,
) -> PyResult<Bound<'py, PyDict>> {
    let profile = GpuDeviceProfile::from_info(&probe.info);
    let device = PyDict::new_bound(py);
    let name = device_name(&probe.info);
    if name.is_empty() {
        device.set_item("name", py.None())?;
    } else {
        device.set_item("name", name)?;
    }
    device.set_item("device_id", probe.info.device_id)?;
    device.set_item("flags", probe.info.flags)?;
    device.set_item("architecture_class", probe.info.reserved[0])?;
    device.set_item("total_global_mem_bytes", probe.info.total_global_mem_bytes)?;
    device.set_item("multiprocessor_count", probe.info.multiprocessor_count)?;
    device.set_item("warp_size", probe.info.warp_size)?;
    device.set_item("compute_major", probe.info.compute_major)?;
    device.set_item("compute_minor", probe.info.compute_minor)?;
    device.set_item("driver_version", probe.info.driver_version)?;
    device.set_item("runtime_version", probe.info.runtime_version)?;
    device.set_item("unified_memory", profile.unified_memory)?;
    device.set_item("integrated", profile.integrated)?;
    device.set_item("discrete", profile.discrete)?;
    device.set_item("managed_memory", profile.managed_memory)?;
    device.set_item("high_bandwidth", profile.high_bandwidth)?;

    let graph = PyDict::new_bound(py);
    graph.set_item(
        "supported",
        probe.graph.graph_mode != GAFIME_GRAPH_UNSUPPORTED,
    )?;
    graph.set_item("mode", graph_mode_name(probe.graph.graph_mode))?;
    graph.set_item("flags", probe.graph.flags)?;
    graph.set_item(
        "supports_memcpy_nodes",
        probe.graph.supports_memcpy_nodes != 0,
    )?;
    graph.set_item(
        "supports_kernel_param_update",
        probe.graph.supports_kernel_param_update != 0,
    )?;
    graph.set_item(
        "supports_device_ranking",
        probe.graph.supports_device_ranking != 0,
    )?;
    graph.set_item("max_captured_nodes", probe.graph.max_captured_nodes)?;
    graph.set_item("stable_pointer_flags", probe.graph.stable_pointer_flags)?;

    let significance = PyDict::new_bound(py);
    significance.set_item(
        "permutation_pvalues_abi",
        probe.supports_permutation_pvalues,
    )?;

    let rt = PyDict::new_bound(py);
    rt.set_item("available", profile.optix_rt)?;
    rt.set_item(
        "decision_path_membership_abi",
        probe.supports_decision_path_membership,
    )?;
    rt.set_item(
        "decision_path_score_abi",
        probe.supports_decision_path_score,
    )?;

    let runtime = PyDict::new_bound(py);
    runtime.set_item("backend", backend_capability_name_for_kind(probe.kind))?;
    runtime.set_item("device", device)?;
    runtime.set_item("graph", graph)?;
    runtime.set_item("significance", significance)?;
    runtime.set_item("rt", rt)?;
    match &probe.library_path {
        Some(path) => runtime.set_item("library_path", path)?,
        None => runtime.set_item("library_path", py.None())?,
    }
    Ok(runtime)
}

fn runtime_probe_error_to_python<'py>(
    py: Python<'py>,
    error: &GpuSysError,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("status", "unavailable")?;
    result.set_item("detail", error.to_string())?;
    Ok(result)
}

/// Runtime-only facts for public Python capability reporting. The function uses
/// the same `GpuBackend::*_from_env` loader seam as normal execution; payload
/// discovery can evolve behind that seam without changing the public shape.
#[pyfunction]
#[pyo3(signature = (backend="auto", device_id=0, probe=false))]
fn runtime_capabilities(
    py: Python<'_>,
    backend: &str,
    device_id: u32,
    probe: bool,
) -> PyResult<Py<PyDict>> {
    let backend = normalize_runtime_backend(backend).map_err(PyErr::from)?;
    let result = PyDict::new_bound(py);
    let candidates = PyDict::new_bound(py);
    result.set_item("configured_backend", backend)?;
    result.set_item("probe_performed", probe)?;
    result.set_item("native_version", public_package_version())?;
    result.set_item("boundary_name", BOUNDARY_NAME)?;
    result.set_item("candidates", &candidates)?;
    result.set_item("runtime", py.None())?;

    if backend == "core" {
        result.set_item("status", "available")?;
        result.set_item("selected_backend", "core")?;
        result.set_item("detail", "Core is built into the native boundary.")?;
        return Ok(result.unbind());
    }

    if !probe {
        result.set_item("status", "not_probed")?;
        result.set_item("selected_backend", py.None())?;
        if backend == "auto" {
            result.set_item(
                "detail",
                "automatic selection was not probed; no backend was selected",
            )?;
        } else {
            result.set_item(
                "detail",
                format!("{backend} was configured but runtime payload probing is disabled"),
            )?;
        }
        return Ok(result.unbind());
    }

    if backend != "auto" {
        let kind = backend_kind_for_runtime_name(backend);
        match probe_gpu_runtime(kind, device_id) {
            Ok(probe_result) => {
                let candidate = PyDict::new_bound(py);
                candidate.set_item("status", "available")?;
                candidate.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
                candidates.set_item(backend, candidate)?;
                result.set_item("status", "available")?;
                result.set_item("selected_backend", backend)?;
                result.set_item("detail", "explicit backend passed the runtime ABI probe")?;
                result.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
            }
            Err(error) => {
                candidates.set_item(backend, runtime_probe_error_to_python(py, &error)?)?;
                result.set_item("status", "unavailable")?;
                result.set_item("selected_backend", py.None())?;
                result.set_item("detail", error.to_string())?;
            }
        }
        return Ok(result.unbind());
    }

    let mut probes = Vec::new();
    for kind in [
        GAFIME_BACKEND_CUDA,
        GAFIME_BACKEND_ROCM,
        GAFIME_BACKEND_METAL,
    ] {
        let name = backend_capability_name_for_kind(kind);
        match probe_gpu_runtime(kind, device_id) {
            Ok(probe_result) => {
                let candidate = PyDict::new_bound(py);
                candidate.set_item("status", "available")?;
                candidate.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
                candidates.set_item(name, candidate)?;
                probes.push(probe_result);
            }
            Err(error) => {
                candidates.set_item(name, runtime_probe_error_to_python(py, &error)?)?;
            }
        }
    }
    let selected = probes.iter().max_by_key(|candidate| {
        (
            gpu_device_score(&candidate.info),
            backend_tie_breaker(candidate.kind),
        )
    });
    match selected {
        Some(probe_result) => {
            let name = backend_capability_name_for_kind(probe_result.kind);
            result.set_item("status", "available")?;
            result.set_item("selected_backend", name)?;
            result.set_item(
                "detail",
                format!("auto selected {name} after runtime ABI probes"),
            )?;
            result.set_item("runtime", runtime_probe_to_python(py, probe_result)?)?;
        }
        None => {
            result.set_item("status", "available")?;
            result.set_item("selected_backend", "core")?;
            result.set_item(
                "detail",
                "auto selected core because no GPU payload passed the runtime ABI probe",
            )?;
        }
    }
    Ok(result.unbind())
}

fn metric_ids_from_names(names: Vec<String>) -> PyResult<Vec<u32>> {
    metric_ids_from_names_result(names).map_err(PyErr::from)
}

fn metric_ids_from_names_result(names: Vec<String>) -> Result<Vec<u32>, PyBoundaryError> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        ids.push(match name.as_str() {
            "pearson" => GAFIME_METRIC_PEARSON,
            "spearman" => GAFIME_METRIC_SPEARMAN,
            "mutual_info" => GAFIME_METRIC_MUTUAL_INFO,
            "r2" => GAFIME_METRIC_R2,
            other => {
                return Err(PyBoundaryError::InvalidInput(format!(
                    "unsupported metric {other:?}"
                )))
            }
        });
    }
    validate_metric_ids(ids)
}

fn get_optional_dict<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.downcast_into::<PyDict>()?)),
        _ => Ok(None),
    }
}

fn get_string(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<String>(),
        _ => Ok(default.to_string()),
    }
}

fn get_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<bool>(),
        _ => Ok(default),
    }
}

fn get_u32(dict: &Bound<'_, PyDict>, key: &str, default: u32) -> PyResult<u32> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<u32>(),
        _ => Ok(default),
    }
}

fn get_u64(dict: &Bound<'_, PyDict>, key: &str, default: u64) -> PyResult<u64> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<u64>(),
        _ => Ok(default),
    }
}

fn get_optional_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<i64>().map(Some),
        _ => Ok(None),
    }
}

fn get_python_integer_seed(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<(u64, Vec<u32>)> {
    let Some(value) = dict.get_item(key)? else {
        return Ok((0, vec![0]));
    };
    if value.is_none() {
        return Ok((0, vec![0]));
    }
    parse_python_integer_seed(&value)
}

fn parse_python_integer_seed(value: &Bound<'_, PyAny>) -> PyResult<(u64, Vec<u32>)> {
    let absolute = value.call_method0("__abs__")?;
    let bit_length = absolute.call_method0("bit_length")?.extract::<usize>()?;
    let byte_length = bit_length.div_ceil(8).max(1);
    let bytes = absolute
        .call_method1("to_bytes", (byte_length, "little"))?
        .extract::<Vec<u8>>()?;
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
    let significance_seed = significance_seed_from_words(&words);
    Ok((significance_seed, words))
}

fn significance_seed_from_words(words: &[u32]) -> u64 {
    let low = u64::from(words.first().copied().unwrap_or(0))
        | (u64::from(words.get(1).copied().unwrap_or(0)) << 32);
    if words.len() <= 2 {
        return low;
    }

    // Preserve the historical u64 stream for ordinary seeds, but fold every
    // higher Python integer word into significance scheduling. FNV-1a is used
    // only as a stable identity compressor; planning still consumes all words.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash ^ (words.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn get_vec_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<Vec<String>>(),
        _ => Ok(vec![
            "pearson".to_string(),
            "spearman".to_string(),
            "mutual_info".to_string(),
            "r2".to_string(),
        ]),
    }
}

#[pyclass(name = "ContinuousRecord")]
#[derive(Clone)]
struct PyContinuousRecord {
    #[pyo3(get)]
    combo: Vec<u32>,
    #[pyo3(get)]
    metrics: Vec<f32>,
    #[pyo3(get)]
    candidate_id: u64,
}

#[pyclass(name = "ContinuousReport")]
struct PyContinuousReport {
    #[pyo3(get)]
    rows: u64,
    #[pyo3(get)]
    cols: u32,
    #[pyo3(get)]
    max_arity: u32,
    #[pyo3(get)]
    metric_ids: Vec<u32>,
    #[pyo3(get)]
    backend_kind: u32,
    #[pyo3(get)]
    graph_replayed: bool,
    table: SendOwnedResultTable,
    significance: Vec<SignificanceEntry>,
    decision_path_params: Vec<DecisionPathResultParams>,
}

#[pymethods]
impl PyContinuousReport {
    #[getter]
    fn backend_name(&self) -> &'static str {
        backend_name_for_kind(self.backend_kind)
    }

    #[getter]
    fn device(&self) -> &'static str {
        backend_device_for_kind(self.backend_kind)
    }

    #[getter]
    fn is_gpu(&self) -> bool {
        backend_is_gpu(self.backend_kind)
    }

    #[getter]
    fn selected_backend(&self) -> &'static str {
        backend_capability_name_for_kind(self.backend_kind)
    }

    #[getter]
    fn execution_placement(&self) -> &'static str {
        execution_placement_for_kind(self.backend_kind)
    }

    fn __len__(&self) -> usize {
        self.table.row_count()
    }

    fn record(&self, index: usize) -> PyResult<PyContinuousRecord> {
        Ok(PyContinuousRecord {
            combo: self.combo(index)?,
            metrics: self.metric_values(index)?,
            candidate_id: self.candidate_id(index)?,
        })
    }

    fn combo(&self, index: usize) -> PyResult<Vec<u32>> {
        combo_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn metric_values(&self, index: usize) -> PyResult<Vec<f32>> {
        metric_values_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn candidate_id(&self, index: usize) -> PyResult<u64> {
        self.table
            .candidate_ids()
            .get(index)
            .copied()
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn interaction_components(&self, index: usize) -> PyResult<(Vec<u32>, Vec<f32>, u64)> {
        let combo = combo_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))?;
        let metrics = metric_values_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))?;
        let candidate_id = self
            .table
            .candidate_ids()
            .get(index)
            .copied()
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))?;
        Ok((combo, metrics, candidate_id))
    }

    fn interaction_components_batch(
        &self,
        start: usize,
        limit: usize,
    ) -> PyResult<Vec<(Vec<u32>, Vec<f32>, u64)>> {
        if start > self.table.row_count() {
            return Err(PyValueError::new_err(
                "continuous report batch start is out of range",
            ));
        }
        let end = start.saturating_add(limit).min(self.table.row_count());
        let mut components = Vec::with_capacity(end - start);
        for index in start..end {
            components.push((
                combo_from_table(&self.table, index).unwrap_or_default(),
                metric_values_from_table(&self.table, index).unwrap_or_default(),
                self.table.candidate_ids()[index],
            ));
        }
        Ok(components)
    }

    #[pyo3(signature = (*, metric_index=None, descending=true, limit=None))]
    fn ranked_indices(
        &self,
        metric_index: Option<usize>,
        descending: bool,
        limit: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        if let Some(metric_index) = metric_index {
            if metric_index >= self.metric_ids.len() {
                return Err(PyValueError::new_err("metric_index is out of range"));
            }
        }
        if let Some(limit) = limit {
            return Ok(bounded_ranked_indices(
                &self.table,
                &self.metric_ids,
                metric_index,
                descending,
                limit,
            ));
        }
        let mut indices = (0..self.table.row_count()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            compare_ranked_rows(
                &self.table,
                &self.metric_ids,
                left,
                right,
                metric_index,
                descending,
            )
        });
        Ok(indices)
    }

    fn records(&self) -> Vec<PyContinuousRecord> {
        (0..self.table.row_count())
            .map(|index| PyContinuousRecord {
                combo: combo_from_table(&self.table, index).unwrap_or_default(),
                metrics: metric_values_from_table(&self.table, index).unwrap_or_default(),
                candidate_id: self.table.candidate_ids()[index],
            })
            .collect()
    }

    /// P-A significance surface: whether permutation/stability were computed, and
    /// the parallel per-row payloads (aligned to `significance_rows()`; inner
    /// vectors aligned to `metric_ids`). Empty when significance was not run
    /// (e.g. GPU reports, raw convenience paths, or `permutation_tests == 0`).
    fn has_significance(&self) -> bool {
        !self.significance.is_empty()
    }

    fn significance_rows(&self) -> Vec<usize> {
        self.significance.iter().map(|entry| entry.row).collect()
    }

    fn significance_pvalues(&self) -> Vec<Vec<f32>> {
        self.significance
            .iter()
            .map(|entry| entry.pvalues.clone())
            .collect()
    }

    fn significance_means(&self) -> Vec<Vec<f32>> {
        self.significance
            .iter()
            .map(|entry| entry.means.clone())
            .collect()
    }

    fn significance_stds(&self) -> Vec<Vec<f32>> {
        self.significance
            .iter()
            .map(|entry| entry.stds.clone())
            .collect()
    }

    fn decision_path_params<'py>(
        &self,
        py: Python<'py>,
        feature_index: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(params) = self
            .decision_path_params
            .iter()
            .find(|params| params.feature_index == feature_index)
        else {
            return Ok(None);
        };
        let out = PyDict::new_bound(py);
        out.set_item("kind", "decision_path")?;
        out.set_item("features", &params.features)?;
        out.set_item("thresholds", &params.thresholds)?;
        out.set_item("signs", &params.signs)?;
        out.set_item("gain", params.gain)?;
        out.set_item("support", params.support)?;
        out.set_item("round_id", params.round_id)?;
        Ok(Some(out))
    }

    /// Arrow PyCapsule Interface (Polars >= 1.3, pyarrow, etc. consume this
    /// zero-copy). Returns the (schema, array) capsule pair; arrow-rs owns the
    /// FFI release callbacks, so there is no hand-rolled unsafe lifetime logic.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema; // schema is fixed; no cast negotiation
        let data = result_table_view_to_arrow(&self.table).into_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data)
            .map_err(|err| PyValueError::new_err(format!("arrow ffi export failed: {err}")))?;
        let schema_name = CString::new("arrow_schema").expect("static capsule name");
        let array_name = CString::new("arrow_array").expect("static capsule name");
        let schema_capsule = PyCapsule::new_bound(py, ffi_schema, Some(schema_name))?;
        let array_capsule = PyCapsule::new_bound(py, ffi_array, Some(array_name))?;
        Ok((schema_capsule, array_capsule))
    }
}

impl From<ContinuousReport> for PyContinuousReport {
    fn from(value: ContinuousReport) -> Self {
        Self {
            rows: value.rows,
            cols: value.cols,
            max_arity: value.max_arity,
            metric_ids: value.metric_ids,
            backend_kind: value.backend_kind,
            graph_replayed: value.graph_replayed,
            table: SendOwnedResultTable(value.table),
            significance: value.significance,
            decision_path_params: Vec::new(),
        }
    }
}

fn backend_name_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "v1-cuda-cabi",
        GAFIME_BACKEND_ROCM => "v1-rocm-cabi",
        GAFIME_BACKEND_METAL => "v1-metal-cabi",
        _ => "v1-rust-cpu",
    }
}

fn backend_capability_name_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "cuda",
        GAFIME_BACKEND_ROCM => "rocm",
        GAFIME_BACKEND_METAL => "metal",
        _ => "core",
    }
}

fn execution_placement_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CPU => "gafime_cpu",
        _ => backend_capability_name_for_kind(backend_kind),
    }
}

fn backend_device_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "cuda",
        GAFIME_BACKEND_ROCM => "rocm",
        GAFIME_BACKEND_METAL => "metal",
        _ => "cpu",
    }
}

fn backend_is_gpu(backend_kind: u32) -> bool {
    matches!(
        backend_kind,
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
    )
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

enum CompiledContinuousBackend {
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

struct ContinuousRunState {
    significance_matrix: Option<CpuMatrix>,
    backend: CompiledContinuousBackend,
    primary: Option<PreparedContinuousExecution>,
    screened: Option<ScreenedContinuousExecution>,
    result_capacity: u64,
    result_max_arity: u32,
    result_metric_count: u32,
}

impl ContinuousRunState {
    fn complete_family(&self) -> Result<&PreparedContinuousExecution, PyBoundaryError> {
        self.primary.as_ref().ok_or_else(|| {
            PyBoundaryError::InvalidInput(
                "continuous significance protocol is missing its complete family".to_string(),
            )
        })
    }

    fn replace_plans(&mut self, run: PreparedContinuousRun) {
        self.primary = run.primary;
        self.screened = run.screened;
        self.result_capacity = run.result_capacity;
        self.result_max_arity = run.result_max_arity;
        self.result_metric_count = run.result_metric_count;
    }
}

#[pyclass(name = "CompiledContinuousArtifact", unsendable)]
struct PyCompiledContinuousArtifact {
    config: EngineConfig,
    #[pyo3(get)]
    rows: u64,
    #[pyo3(get)]
    cols: u32,
    #[pyo3(get)]
    max_arity: u32,
    #[pyo3(get)]
    metric_ids: Vec<u32>,
    significance_top_n: u32,
    state: Option<ContinuousRunState>,
    runtime_cache_counters: RefCell<RuntimeCacheCounters>,
    decision_path_params: Vec<DecisionPathResultParams>,
    target_updates_supported: bool,
    closed: bool,
}

impl PyCompiledContinuousArtifact {
    fn backend_kind(&self) -> u32 {
        self.config.backend_kind
    }

    fn replace_target(&mut self, target: Vec<f32>) -> PyResult<()> {
        if self.closed {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        if !self.target_updates_supported {
            return Err(PyValueError::new_err(
                "update_target is unsupported for compiled target-derived decision-path features; recompile with the new target",
            ));
        }
        if target.len() as u64 != self.rows {
            return Err(PyValueError::new_err(
                "target length must match the compiled matrix rows",
            ));
        }
        let backend_kind = self.backend_kind();
        let update_result = {
            let Some(state) = self.state.as_mut() else {
                return Err(PyValueError::new_err("compiled artifact is closed"));
            };
            match &mut state.backend {
                CompiledContinuousBackend::Cpu { matrix } => matrix
                    .set_target(target.clone())
                    .map_err(PyBoundaryError::from),
                CompiledContinuousBackend::Cuda { matrix, .. }
                | CompiledContinuousBackend::Rocm { matrix, .. }
                | CompiledContinuousBackend::Metal { matrix, .. } => {
                    matrix.update_target(&target).map_err(PyBoundaryError::from)
                }
            }
        };
        if let Err(error) = update_result {
            if backend_is_gpu(backend_kind) {
                self.state = None;
                self.closed = true;
            }
            return Err(PyErr::from(error));
        }
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("compiled artifact is closed"))?;
        if let Some(significance_matrix) = &mut state.significance_matrix {
            if let Err(error) = significance_matrix.set_target(target) {
                if backend_is_gpu(backend_kind) {
                    self.state = None;
                    self.closed = true;
                }
                return Err(PyErr::from(PyBoundaryError::from(error)));
            }
        }
        let run = match prepare_screened_continuous_execution(
            &self.config,
            self.rows,
            self.cols,
            &state.backend,
        ) {
            Ok(value) => value,
            Err(error) => {
                self.state = None;
                self.closed = true;
                return Err(PyErr::from(error));
            }
        };
        self.max_arity = run.result_max_arity;
        state.replace_plans(run);
        Ok(())
    }
}

#[pymethods]
impl PyCompiledContinuousArtifact {
    #[getter]
    fn backend_name(&self) -> &'static str {
        backend_name_for_kind(self.backend_kind())
    }

    #[getter]
    fn device(&self) -> &'static str {
        backend_device_for_kind(self.backend_kind())
    }

    #[getter]
    fn is_gpu(&self) -> bool {
        backend_is_gpu(self.backend_kind())
    }

    #[getter]
    fn selected_backend(&self) -> &'static str {
        backend_capability_name_for_kind(self.backend_kind())
    }

    #[getter]
    fn execution_placement(&self) -> &'static str {
        execution_placement_for_kind(self.backend_kind())
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    #[getter]
    fn graph_requested(&self) -> bool {
        self.state
            .as_ref()
            .and_then(|state| state.primary.as_ref())
            .is_some_and(|prepared| prepared.schedule().decision().graph_requested)
    }

    fn analyze(&mut self) -> PyResult<PyContinuousReport> {
        let report = execute_compiled_artifact(self).map_err(PyErr::from)?;
        let mut report = PyContinuousReport::from(report);
        report
            .decision_path_params
            .clone_from(&self.decision_path_params);
        Ok(report)
    }

    #[getter]
    fn continuous_metric_cache_hits(&self) -> u64 {
        self.runtime_cache_counters.borrow().metric_hits
    }

    #[getter]
    fn continuous_metric_cache_builds(&self) -> u64 {
        self.runtime_cache_counters.borrow().metric_builds
    }

    #[getter]
    fn candidate_table_cache_hits(&self) -> u64 {
        self.runtime_cache_counters.borrow().candidate_table_hits
    }

    /// Refresh stochastic planning and significance streams without uploading
    /// the resident feature matrix again. The Python wrapper uses this for
    /// `random_seed=None`, matching the legacy fresh-entropy-per-analysis rule.
    fn reseed(&mut self, seed: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.closed {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        if self.state.is_none() {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        let (random_seed, planning_seed_words) = parse_python_integer_seed(seed)?;
        let mut config = self.config.clone();
        config.random_seed = random_seed;
        config.planning_seed_words = planning_seed_words;
        let run_result = match self.state.as_ref() {
            Some(state) => {
                prepare_screened_continuous_execution(&config, self.rows, self.cols, &state.backend)
            }
            None => return Err(PyValueError::new_err("compiled artifact is closed")),
        };
        let run = match run_result {
            Ok(run) => run,
            Err(error) => {
                if backend_is_gpu(self.backend_kind()) {
                    self.state = None;
                    self.closed = true;
                }
                return Err(PyErr::from(error));
            }
        };
        self.max_arity = run.result_max_arity;
        self.config = config;
        let Some(state) = self.state.as_mut() else {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        };
        state.replace_plans(run);
        Ok(())
    }

    /// Resident-session reuse: swap the target in place and re-analyze without
    /// re-uploading the features. On GPU the resident device matrix keeps its
    /// feature buffers and only `y` is refreshed (the permutation/repeat pattern);
    /// on CPU the held matrix's target is replaced. The host significance matrix
    /// (GPU runs) is updated too so a subsequent analyze scores against the new y.
    fn update_target(&mut self, target: Vec<f32>) -> PyResult<()> {
        self.replace_target(target)
    }

    fn update_target_buffer(&mut self, target: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.replace_target(decode_f32_le(target.as_bytes(), "target")?)
    }

    fn close(&mut self) {
        self.state = None;
        self.closed = true;
    }
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
fn compile_continuous(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    compile_continuous_rows(config, rows, cols, features, target).map_err(PyErr::from)
}

#[pyfunction(name = "compile_continuous_rows")]
fn compile_continuous_nested(
    config: &Bound<'_, PyDict>,
    features: Vec<Vec<f32>>,
    target: Vec<f32>,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let (rows, cols, features, target) = flatten_continuous_rows(features, target)?;
    compile_continuous_rows(config, rows, cols, features, target).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
fn compile_continuous_buffers(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyBytes>,
    target: &Bound<'_, PyBytes>,
    rows: u64,
    cols: u32,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let features = decode_f32_le(features.as_bytes(), "feature")?;
    let target = decode_f32_le(target.as_bytes(), "target")?;
    compile_continuous_rows(config, rows, cols, features, target).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
fn analyze_continuous(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    analyze_continuous_rows_once(config, rows, cols, features, target)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction(name = "analyze_continuous_rows")]
fn analyze_continuous_nested(
    config: &Bound<'_, PyDict>,
    features: Vec<Vec<f32>>,
    target: Vec<f32>,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    let (rows, cols, features, target) = flatten_continuous_rows(features, target)?;
    analyze_continuous_rows_once(config, rows, cols, features, target)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
fn analyze_continuous_buffers(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyBytes>,
    target: &Bound<'_, PyBytes>,
    rows: u64,
    cols: u32,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    let features = decode_f32_le(features.as_bytes(), "feature")?;
    let target = decode_f32_le(target.as_bytes(), "target")?;
    analyze_continuous_rows_once(config, rows, cols, features, target)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (features, target, max_arity=2, max_combinations_per_k=5000, metric_ids=None))]
fn analyze_continuous_cpu(
    _py: Python<'_>,
    features: Vec<Vec<f32>>,
    target: Vec<f32>,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Option<Vec<u32>>,
) -> PyResult<PyContinuousReport> {
    let rows = features.len() as u64;
    let cols = features.first().map_or(0u32, |row| row.len() as u32);
    if features.iter().any(|row| row.len() != cols as usize) {
        return Err(PyValueError::new_err(
            "feature rows must all have the same length",
        ));
    }
    let flat_features = features.into_iter().flatten().collect::<Vec<_>>();
    analyze_continuous_cpu_rows(
        rows,
        cols,
        flat_features,
        target,
        max_arity,
        max_combinations_per_k,
        metric_ids.unwrap_or_default(),
    )
    .map(PyContinuousReport::from)
    .map_err(PyErr::from)
}

/// Import a Python object exposing the Arrow C stream interface
/// (`__arrow_c_stream__`, e.g. a Polars DataFrame) into a single Arrow
/// `StructArray`, zero-copy. We move the stream out of the capsule and leave an
/// empty no-op stream so the capsule destructor doesn't double-release; arrow-rs
/// owns the rest of the FFI lifecycle. Callers should `.rechunk()` so the frame
/// arrives as one record batch.
fn import_arrow_struct(obj: &Bound<'_, PyAny>) -> PyResult<StructArray> {
    let capsule = obj.call_method0("__arrow_c_stream__")?;
    let cap: Bound<'_, PyCapsule> = capsule.extract()?;
    let ptr = cap.pointer() as *mut FFI_ArrowArrayStream;
    if ptr.is_null() {
        return Err(PyValueError::new_err("null Arrow stream capsule pointer"));
    }
    let stream = unsafe { std::ptr::replace(ptr, FFI_ArrowArrayStream::empty()) };
    let reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(|err| PyValueError::new_err(format!("arrow stream import failed: {err}")))?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|err| PyValueError::new_err(format!("arrow batch: {err}")))?);
    }
    if batches.is_empty() {
        return Err(PyValueError::new_err("empty Arrow stream"));
    }
    if batches.len() > 1 {
        return Err(PyValueError::new_err(
            "multi-chunk Arrow input; call .rechunk() before ingest",
        ));
    }
    Ok(StructArray::from(batches.into_iter().next().unwrap()))
}

/// Transpose an Arrow struct of Float32 columns into a row-major f32 buffer in
/// Rust (one pass, no Python-object materialization). This is the zero-copy
/// ingest twin of the Arrow result export.
fn struct_to_row_major_f32(sa: &StructArray) -> PyResult<(u64, u32, Vec<f32>)> {
    let rows = sa.len();
    let cols = sa.num_columns();
    if cols == 0 || rows == 0 {
        return Err(PyValueError::new_err("empty Arrow input"));
    }
    let mut columns: Vec<&Float32Array> = Vec::with_capacity(cols);
    for c in 0..cols {
        let col = sa
            .column(c)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                PyValueError::new_err("feature columns must be Float32 (cast in the loader)")
            })?;
        if col.null_count() != 0 {
            return Err(PyValueError::new_err(
                "null feature values are not supported",
            ));
        }
        columns.push(col);
    }
    let mut flat = vec![0.0f32; rows * cols];
    for (c, column) in columns.iter().enumerate() {
        for r in 0..rows {
            flat[r * cols + c] = column.value(r);
        }
    }
    Ok((rows as u64, cols as u32, flat))
}

#[pyfunction]
#[pyo3(signature = (features, target, max_arity=2, max_combinations_per_k=5000, metric_ids=None))]
fn analyze_continuous_arrow(
    _py: Python<'_>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Option<Vec<u32>>,
) -> PyResult<PyContinuousReport> {
    let features = import_arrow_struct(features)?;
    let (rows, cols, flat) = struct_to_row_major_f32(&features)?;
    let target = import_arrow_struct(target)?;
    if target.num_columns() != 1 {
        return Err(PyValueError::new_err(
            "target must contain exactly one column",
        ));
    }
    let target_col = target
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PyValueError::new_err("target column must be Float32"))?;
    if target_col.null_count() != 0 {
        return Err(PyValueError::new_err(
            "null target values are not supported",
        ));
    }
    if target_col.len() as u64 != rows {
        return Err(PyValueError::new_err(
            "target length must match feature rows",
        ));
    }
    let y = (0..target_col.len())
        .map(|i| target_col.value(i))
        .collect::<Vec<f32>>();
    analyze_continuous_cpu_rows(
        rows,
        cols,
        flat,
        y,
        max_arity,
        max_combinations_per_k,
        metric_ids.unwrap_or_default(),
    )
    .map(PyContinuousReport::from)
    .map_err(PyErr::from)
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

fn row_major_feature_selection(
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
    let mut selected = Vec::with_capacity(rows.saturating_mul(selected_features.len()));
    for row in 0..rows {
        let source = row * source_cols;
        selected.extend(
            selected_features
                .iter()
                .map(|&feature| features[source + feature as usize]),
        );
    }
    Ok(selected)
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
            if velocity {
                if !push(TimeSeriesOp::Delta(lag))
                    || !push(TimeSeriesOp::Velocity(lag))
                    || (lag_rows.saturating_mul(2) < rows && !push(TimeSeriesOp::Acceleration(lag)))
                {
                    break 'features;
                }
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
    if base_candidate_cols > source_cols
        || discovery_features
            .iter()
            .any(|&feature| feature as usize >= base_candidate_cols)
    {
        return Err(PyBoundaryError::InvalidInput(
            "decision-path source feature is outside the candidate prefix".to_string(),
        ));
    }
    let base_features = row_major_feature_prefix(features, rows, source_cols, base_candidate_cols);
    if discovery_features.is_empty() || params.max_paths == 0 {
        return Ok((base_features, base_candidate_cols, Vec::new()));
    }
    let discovery_matrix =
        row_major_feature_selection(features, rows, source_cols, discovery_features)?;
    let discovery_cols = discovery_features.len();
    let (discovered, discovered_cols, mut paths) = gafime_cpu::decision_path::expand_row_major(
        &discovery_matrix,
        target,
        rows,
        discovery_cols,
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
    let expanded_cols = base_candidate_cols + paths.len();
    let mut expanded = vec![0.0f32; rows * expanded_cols];
    for row in 0..rows {
        let base_source = row * base_candidate_cols;
        let destination = row * expanded_cols;
        expanded[destination..destination + base_candidate_cols]
            .copy_from_slice(&base_features[base_source..base_source + base_candidate_cols]);
        let generated_source = row * discovered_cols + discovery_cols;
        expanded[destination + base_candidate_cols..destination + expanded_cols]
            .copy_from_slice(&discovered[generated_source..generated_source + paths.len()]);
    }
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
fn analyze_time_series(
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
fn compile_time_series(
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

/// decision_path family: discover depth-k GBDT conjunction paths (with residual
/// boosting) natively, append their hard-AND membership columns after the base
/// features, then mine the expanded matrix through the normal continuous path
/// (CPU or GPU per config). Mirrors `analyze_time_series`. Returns
/// (report, all_feature_names = base ++ path labels).
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, max_bins, min_leaf, learning_rate))]
#[allow(clippy::too_many_arguments)]
fn analyze_decision_path(
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
    let (expanded, expanded_cols, paths) = if base_candidate_cols == 0 {
        (features, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_decision_path_bounded(
            &features,
            &target,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?
    };
    let mut report = analyze_continuous_rows_once(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
        target,
    )
    .map(PyContinuousReport::from)
    .map_err(PyErr::from)?;
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

/// decision_path compile path: discover native GBDT conjunction paths, append
/// membership columns, then return a resident compiled continuous artifact over
/// that expanded matrix.
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, max_bins, min_leaf, learning_rate))]
#[allow(clippy::too_many_arguments)]
fn compile_decision_path(
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
    let (expanded, expanded_cols, paths) = if base_candidate_cols == 0 {
        (features, cols_usize, Vec::new())
    } else {
        parsed.budget.max_feature_candidate = -2;
        expand_decision_path_bounded(
            &features,
            &target,
            rows_usize,
            cols_usize,
            base_candidate_cols,
            &discovery_features,
            &params,
        )
        .map_err(PyErr::from)?
    };
    let mut artifact = compile_continuous_rows(
        parsed,
        rows,
        expanded_column_count(expanded_cols)?,
        expanded,
        target,
    )
    .map_err(PyErr::from)?;
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

#[pymodule]
fn gafime_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", public_package_version())?;
    m.add("BOUNDARY_NAME", BOUNDARY_NAME)?;
    m.add_class::<PyCompiledContinuousArtifact>()?;
    m.add_class::<PyContinuousRecord>()?;
    m.add_class::<PyContinuousReport>()?;
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyCacheAwareScheduler>()?;
    m.add_class::<PyDataQualityAnalyzer>()?;
    m.add_class::<PySmartScheduler>()?;
    m.add_function(wrap_pyfunction!(compile_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(compile_continuous_nested, m)?)?;
    m.add_function(wrap_pyfunction!(compile_continuous_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_nested, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(compile_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(compile_decision_path, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_decision_path, m)?)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(runtime_capabilities, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_name_is_stable() {
        assert_eq!(boundary_name(), "gafime-py");
    }

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
    fn screened_host_admission_counts_cached_and_combined_result_owners() {
        let bytes = screened_candidate_storage_bytes(10_000_000, 0, 10_000_000, 5, 1, 0);

        assert_eq!(bytes, 720_000_000);
        assert!(bytes > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES);
    }

    #[test]
    fn cargo_prerelease_version_maps_to_python_public_version() {
        assert_eq!(cargo_version_to_python("1.0.0-alpha.0"), "1.0.0a0");
        assert_eq!(cargo_version_to_python("1.0.0-beta.2"), "1.0.0b2");
        assert_eq!(cargo_version_to_python("1.0.0-rc.3"), "1.0.0rc3");
        assert_eq!(cargo_version_to_python("1.0.0-dev.1"), "1.0.0-dev.1");
        assert_eq!(public_package_version(), "1.0.0a0");
    }

    #[test]
    fn continuous_report_preserves_native_graph_replay_flag() {
        let mut table = OwnedResultTable::new(1, 1, 1);
        table.raw_mut().flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;

        let report = report_from_table(
            4,
            1,
            1,
            vec![GAFIME_METRIC_PEARSON],
            GAFIME_BACKEND_CUDA,
            table,
            Vec::new(),
        );

        assert!(report.graph_replayed);
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
    fn result_table_exports_to_arrow_struct_and_roundtrips_ffi() {
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

        let array = result_table_to_arrow(&report.table);
        assert_eq!(array.len(), report.len());
        assert_eq!(array.num_columns(), 4);

        // The Arrow C Data Interface export must round-trip — this is the exact
        // path Polars/pyarrow/torch consume, validated without a Python wheel.
        let data = array.into_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data).unwrap();
        let restored = unsafe { arrow::ffi::from_ffi(ffi_array, &ffi_schema) }.unwrap();
        assert_eq!(restored.len(), report.len());
        assert_eq!(restored.child_data().len(), 4);
    }

    #[test]
    fn arrow_struct_imports_to_row_major_f32() {
        let c0 = Arc::new(Float32Array::from(vec![1.0f32, 3.0, 5.0])) as ArrayRef;
        let c1 = Arc::new(Float32Array::from(vec![2.0f32, 4.0, 6.0])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float32, false),
        ]);
        let sa = StructArray::new(fields, vec![c0, c1], None);
        let (rows, cols, flat) = struct_to_row_major_f32(&sa).unwrap();
        assert_eq!((rows, cols), (3, 2));
        // row-major: [r0c0, r0c1, r1c0, r1c1, ...]
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rust_config_boundary_rejects_unknown_metric() {
        let error = metric_ids_from_names_result(vec!["pearson".to_string(), "bogus".to_string()])
            .unwrap_err();

        assert!(error.to_string().contains("unsupported metric"));
    }

    #[test]
    fn rust_config_boundary_accepts_explicit_cuda() {
        assert_eq!(
            backend_kind_from_name_result("cuda", 0).unwrap(),
            GAFIME_BACKEND_CUDA
        );
    }

    #[test]
    fn rust_config_boundary_accepts_explicit_metal() {
        assert_eq!(
            backend_kind_from_name_result("metal", 0).unwrap(),
            GAFIME_BACKEND_METAL
        );
    }

    #[test]
    fn rust_config_boundary_rejects_ambiguous_gpu_without_python_fallback() {
        let error = backend_kind_from_name_result("gpu", 0).unwrap_err();

        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn auto_backend_resolver_returns_supported_backend_kind() {
        assert!(matches!(
            backend_kind_from_name_result("auto", 0).unwrap(),
            GAFIME_BACKEND_CPU | GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ));
    }

    #[test]
    fn auto_backend_prefers_configured_usable_gpu_payload() {
        let has_gpu_payload = std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_some()
            || std::env::var_os(gafime_gpu_sys::ROCM_LIBRARY_ENV).is_some()
            || std::env::var_os(gafime_gpu_sys::METAL_LIBRARY_ENV).is_some();
        if !has_gpu_payload {
            return;
        }

        assert_ne!(
            backend_kind_from_name_result("auto", 0).unwrap(),
            GAFIME_BACKEND_CPU
        );
    }

    #[test]
    fn auto_rank_places_gpu_above_cpu_vector_isa() {
        let mut info = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_METAL,
            flags: gafime_types::GAFIME_GPU_DEVICE_FLAG_INTEGRATED
                | gafime_types::GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
                | gafime_types::GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
            total_global_mem_bytes: 8 * 1024 * 1024 * 1024,
            multiprocessor_count: 8,
            compute_major: 1,
            reserved: [0; 8],
            ..Default::default()
        };
        info.reserved[0] = gafime_types::GAFIME_GPU_ARCH_APPLE;

        assert!(gpu_device_score(&info) > cpu_isa_rank(IsaLevel::Avx512));
        assert!(
            cpu_isa_rank(IsaLevel::Avx512) > cpu_isa_rank(IsaLevel::Avx2)
                && cpu_isa_rank(IsaLevel::Avx2) > cpu_isa_rank(IsaLevel::Sse42)
                && cpu_isa_rank(IsaLevel::Sse42) > cpu_isa_rank(IsaLevel::Scalar)
        );
    }

    #[test]
    fn auto_rank_distinguishes_devices_within_same_gpu_architecture() {
        let mut laptop_ada = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_CUDA,
            flags: gafime_types::GAFIME_GPU_DEVICE_FLAG_DISCRETE,
            total_global_mem_bytes: 8 * 1024 * 1024 * 1024,
            multiprocessor_count: 24,
            compute_major: 8,
            compute_minor: 9,
            reserved: [0; 8],
            ..Default::default()
        };
        laptop_ada.reserved[0] = gafime_types::GAFIME_GPU_ARCH_NVIDIA_ADA;
        let mut desktop_ada = laptop_ada;
        desktop_ada.flags |= gafime_types::GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
        desktop_ada.total_global_mem_bytes = 24 * 1024 * 1024 * 1024;
        desktop_ada.multiprocessor_count = 128;

        assert!(gpu_device_score(&desktop_ada) > gpu_device_score(&laptop_ada));
    }

    #[test]
    fn explicit_cuda_requires_configured_cabi_payload() {
        if std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_some() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_CUDA;

        let error = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(_) => panic!("CUDA compile unexpectedly succeeded without a configured payload"),
            Err(error) => error,
        };

        assert!(error.to_string().contains(gafime_gpu_sys::CUDA_LIBRARY_ENV));
    }

    #[test]
    fn explicit_metal_requires_configured_cabi_payload() {
        if std::env::var_os(gafime_gpu_sys::METAL_LIBRARY_ENV).is_some() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_METAL;

        let error = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(_) => panic!("Metal compile unexpectedly succeeded without a configured payload"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains(gafime_gpu_sys::METAL_LIBRARY_ENV));
    }

    #[test]
    fn explicit_cuda_executes_when_cabi_payload_is_available() {
        if std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_none() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.permutation_tests = 0;

        let mut artifact = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(artifact) => artifact,
            Err(error) => panic!("CUDA compile failed despite configured payload: {error}"),
        };

        assert_eq!(artifact.backend_name(), "v1-cuda-cabi");
        assert_eq!(artifact.device(), "cuda");
        assert!(artifact.is_gpu());
        let report = execute_compiled_artifact(&mut artifact).unwrap();
        assert_eq!(report.len(), 2);
        assert_eq!(report.combo(0).unwrap(), vec![0]);
        assert!((report.metric_values(0).unwrap()[0] - 1.0).abs() < 1.0e-5);
        assert!((report.metric_values(1).unwrap()[0] + 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn rust_config_boundary_rejects_unwired_families() {
        let error = validate_family_flags(true, false).unwrap_err();

        assert!(error.to_string().contains("time-series families"));
    }

    #[test]
    fn rust_input_boundary_validates_flat_row_major_dimensions() {
        let config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();

        let error = match compile_continuous_cpu_rows(config, 4, 2, vec![1.0; 7], vec![1.0; 4]) {
            Ok(_) => panic!("invalid shape unexpectedly compiled"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("feature buffer length does not match rows*cols"));
    }

    #[test]
    fn rust_input_boundary_rejects_shape_overflow_without_panicking() {
        let error = validate_shape(1u64 << 63, 2, 0, 0).unwrap_err();
        assert!(error
            .to_string()
            .contains("rows*cols exceed host address space"));
    }

    #[test]
    fn little_endian_f32_buffer_decode_preserves_nonfinite_bits() {
        let values = [1.25f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let decoded = decode_f32_le(&bytes, "test").unwrap();

        assert_eq!(decoded[0].to_bits(), values[0].to_bits());
        assert_eq!(decoded[1].to_bits(), values[1].to_bits());
        assert_eq!(decoded[2].to_bits(), values[2].to_bits());
        assert_eq!(decoded[3].to_bits(), values[3].to_bits());
        assert!(decode_f32_le(&bytes[..bytes.len() - 1], "test").is_err());
    }

    #[test]
    fn nested_input_boundary_preserves_non_finite_fp32_values() {
        let (_, _, features, target) = flatten_continuous_rows(
            vec![vec![1.0, f32::INFINITY], vec![2.0, 3.0]],
            vec![1.0, f32::NEG_INFINITY],
        )
        .unwrap();
        assert!(features[1].is_infinite() && features[1].is_sign_positive());
        assert!(target[1].is_infinite() && target[1].is_sign_negative());
    }

    #[test]
    fn one_shot_and_compiled_cpu_execution_are_exact() {
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.permutation_tests = 0;
        config.num_repeats = 1;
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
    fn significance_seed_uses_python_integer_words_above_u64() {
        assert_eq!(significance_seed_from_words(&[7]), 7);
        assert_eq!(significance_seed_from_words(&[7, 0]), 7);
        assert_ne!(
            significance_seed_from_words(&[7]),
            significance_seed_from_words(&[7, 0, 1])
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
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_R2, GAFIME_METRIC_PEARSON];
        config.permutation_tests = 0;
        config.num_repeats = 1;
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
    fn python_report_table_moves_without_copy_and_is_send() {
        fn assert_send<T: Send>() {}

        assert_send::<PyContinuousReport>();

        let report = analyze_continuous_cpu_rows(
            4,
            2,
            vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
            1,
            8,
            vec![GAFIME_METRIC_PEARSON],
        )
        .unwrap();
        let combo_ptr = report.table.combo_indices().as_ptr();
        let metric_ptr = report.table.metric_values().as_ptr();
        let rank_ptr = report.table.ranks().as_ptr();
        let candidate_id_ptr = report.table.candidate_ids().as_ptr();

        let python_report = PyContinuousReport::from(report);

        assert_eq!(python_report.table.combo_indices().as_ptr(), combo_ptr);
        assert_eq!(python_report.table.metric_values().as_ptr(), metric_ptr);
        assert_eq!(python_report.table.ranks().as_ptr(), rank_ptr);
        assert_eq!(
            python_report.table.candidate_ids().as_ptr(),
            candidate_id_ptr
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
    fn v047_unranked_power_user_plan_above_one_million_rows_is_admitted() {
        const COLS: u32 = 1_450;
        const PAIR_ROWS: u64 = 1_050_525;
        const EXPECTED_ROWS: u64 = 1_051_975;

        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON];
        config.permutation_tests = 0;
        config.num_repeats = 1;
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = PAIR_ROWS;
        config.budget.top_features_for_higher_k = COLS;
        let features = (0..4)
            .flat_map(|row| std::iter::repeat(row as f32).take(COLS as usize))
            .collect();

        let artifact =
            compile_continuous_cpu_rows(config, 4, COLS, features, vec![0.0, 1.0, 2.0, 3.0])
                .unwrap();
        let state = artifact.state.as_ref().unwrap();

        assert_eq!(PAIR_ROWS, u64::from(COLS) * u64::from(COLS - 1) / 2);
        assert_eq!(state.result_capacity, EXPECTED_ROWS);
        assert_eq!(state.result_max_arity, 2);
    }

    #[test]
    fn pathological_unranked_plan_still_fails_storage_admission() {
        const COLS: u32 = 20_000;
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON];
        config.permutation_tests = 0;
        config.num_repeats = 1;
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 100_000_000;
        config.budget.top_features_for_higher_k = COLS;
        let features = (0..2)
            .flat_map(|row| std::iter::repeat(row as f32).take(COLS as usize))
            .collect();

        let error = compile_continuous_cpu_rows(config, 2, COLS, features, vec![0.0, 1.0])
            .err()
            .expect("pathological unranked plan must fail storage admission");

        assert!(error
            .to_string()
            .contains("unranked continuous candidate storage exceeds the host-memory budget"));
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
        let mut config = EngineConfig::default();
        config.permutation_tests = 1;
        let error = validate_decision_path_permutation_config(&config).unwrap_err();
        assert!(error.to_string().contains("rediscovery"));

        config.permutation_tests = 0;
        validate_decision_path_permutation_config(&config).unwrap();
    }

    #[test]
    fn close_drops_compiled_native_state_immediately() {
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON];
        config.permutation_tests = 0;
        config.num_repeats = 1;
        config.budget.max_comb_size = 1;
        let mut artifact =
            compile_continuous_rows(config, 3, 1, vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0])
                .unwrap();

        assert!(artifact.state.is_some());
        artifact.close();

        assert!(artifact.state.is_none());
        assert!(execute_compiled_artifact(&mut artifact).is_err());
    }
}
