use std::{cell::RefCell, error::Error, ffi::CString, fmt, sync::Arc};

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Fields};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use gafime_cpu::{
    kernels::MetricKernel,
    matrix::CpuMatrix,
    result::OwnedResultTable,
    significance::{self, SignificanceParams},
    simd::{finite_dispatch_isa, IsaLevel},
    CpuBackend,
};
use gafime_gpu_sys::{
    GpuArchitectureClass, GpuBackend, GpuDeviceProfile, GpuSysError, OwnedGpuMatrix,
};
use gafime_orchestrator::{
    config::EngineConfig, execute_plan, prepare_continuous_execution, OrchestratorError,
    PreparedContinuousExecution,
};
use gafime_types::{
    GafimeGpuDeviceInfo, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_METRIC_SPEARMAN, GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyCapsule, PyDict},
};

/// Build a zero-copy-to-consumer Arrow `StructArray` over the compact result
/// table. Columns: `candidate_id` (u64), `rank` (u32), `combo`
/// (FixedSizeList<u32>[max_arity]), `metrics` (FixedSizeList<f32>[metric_count]).
/// The only copy is the compact (top-K) table into Arrow-owned buffers; the
/// Arrow -> framework (Polars/torch/pyarrow) handoff is then zero-copy, and the
/// FFI release callbacks are owned by arrow-rs (no hand-rolled unsafe).
pub fn result_table_to_arrow(table: &OwnedResultTable) -> StructArray {
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

pub const BOUNDARY_NAME: &str = "gafime-py";

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
    let artifact = compile_continuous_cpu_rows(config, rows, cols, features, target)?;
    execute_compiled_artifact(&artifact)
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
    let prepared = prepare_continuous_execution(&config, rows, cols)?;
    let needs_significance = config.permutation_tests > 0 || config.num_repeats > 1;
    // For GPU runs that still need CPU-side significance, build the host copy by
    // MOVING the ingest buffers into CpuMatrix after device upload borrows them.
    // CUDA payloads exposing the optional permutation p-value ABI can skip that
    // copy when bootstrap stability is not requested; the future WHILE-node graph
    // can then replace CUDA's host-driven permutation loop under the same surface.
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
                    && config.permutation_tests > 0
                    && config.num_repeats <= 1
                    && backend.supports_permutation_pvalues();
                let significance_matrix = if needs_significance && !use_device_pvalues {
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
    Ok(PyCompiledContinuousArtifact {
        rows,
        cols,
        max_arity: prepared.result_max_arity(),
        metric_ids: config.metric_ids.clone(),
        permutation_tests: config.permutation_tests,
        num_repeats: config.num_repeats,
        random_seed: config.random_seed,
        mi_bins: config.mi_bins,
        mi_approximate: config.mi_approximate,
        significance_top_n: config.budget.top_features_for_higher_k,
        significance_matrix,
        backend,
        prepared,
        runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
        closed: false,
    })
}

fn execute_compiled_artifact(
    artifact: &PyCompiledContinuousArtifact,
) -> Result<ContinuousReport, PyBoundaryError> {
    if artifact.closed {
        return Err(PyBoundaryError::InvalidInput(
            "compiled artifact is closed".to_string(),
        ));
    }
    let mut table = OwnedResultTable::new(
        artifact.prepared.result_capacity(),
        artifact.prepared.result_max_arity(),
        artifact.prepared.result_metric_count(),
    );
    *artifact.runtime_cache_counters.borrow_mut() = RuntimeCacheCounters::default();
    match &artifact.backend {
        CompiledContinuousBackend::Cpu { matrix } => {
            let mut backend = CpuBackend;
            execute_plan(
                &mut backend,
                &matrix.handle(),
                artifact.prepared.plan(),
                table.raw_mut(),
            )?;
        }
        CompiledContinuousBackend::Cuda { backend, matrix }
        | CompiledContinuousBackend::Rocm { backend, matrix }
        | CompiledContinuousBackend::Metal { backend, matrix } => {
            let mut backend = backend.borrow_mut();
            execute_plan(
                &mut *backend,
                matrix.handle(),
                artifact.prepared.plan(),
                table.raw_mut(),
            )?;
        }
    }

    let significance = compute_significance(artifact, &table)?;

    Ok(report_from_table(
        artifact.rows,
        artifact.cols,
        artifact.prepared.result_max_arity(),
        artifact.metric_ids.clone(),
        artifact.backend_kind(),
        table,
        significance,
    ))
}

fn compute_significance(
    artifact: &PyCompiledContinuousArtifact,
    table: &OwnedResultTable,
) -> Result<Vec<SignificanceEntry>, PyBoundaryError> {
    if let Some(significance) = compute_gpu_permutation_pvalues(artifact, table)? {
        return Ok(significance);
    }
    compute_cpu_significance(artifact, table)
}

fn compute_gpu_permutation_pvalues(
    artifact: &PyCompiledContinuousArtifact,
    table: &OwnedResultTable,
) -> Result<Option<Vec<SignificanceEntry>>, PyBoundaryError> {
    if artifact.permutation_tests == 0 || artifact.num_repeats > 1 {
        return Ok(None);
    }
    let CompiledContinuousBackend::Cuda { backend, matrix } = &artifact.backend else {
        return Ok(None);
    };
    if !backend.borrow().supports_permutation_pvalues() {
        return Ok(None);
    }
    let row_count = table.row_count();
    if row_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut order: Vec<usize> = (0..row_count).collect();
    order.sort_by(|&left, &right| {
        let left_value = rank_value_at(table, &artifact.metric_ids, left, None);
        let right_value = rank_value_at(table, &artifact.metric_ids, right, None);
        compare_rank_values(left_value, right_value, true)
            .then_with(|| table.candidate_ids()[left].cmp(&table.candidate_ids()[right]))
    });
    let cap = (artifact.significance_top_n.max(1) as usize).min(row_count);
    order.truncate(cap);

    let metric_count = artifact.metric_ids.len();
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
    let null_family_rows = artifact.prepared.plan().planned_row_count();
    *artifact.runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(artifact.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: candidate_ids.len() as u64,
    };
    let pvalues = backend
        .borrow_mut()
        .permutation_pvalues(
            handle,
            artifact.prepared.plan().protocol(),
            &candidate_ids,
            &observed_flat,
            metric_count as u32,
        )?
        .ok_or_else(|| {
            PyBoundaryError::UnsupportedFeature(
                "CUDA payload does not expose gafime_gpu_permutation_pvalues".to_string(),
            )
        })?;
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
/// reported rows stay bounded, while each maxT permutation streams every
/// candidate in the compiled plan so screening does not shrink the null family.
/// CUDA can provide the same full-family reduction through its optional GPU ABI.
fn compute_cpu_significance(
    artifact: &PyCompiledContinuousArtifact,
    table: &OwnedResultTable,
) -> Result<Vec<SignificanceEntry>, PyBoundaryError> {
    if artifact.permutation_tests == 0 && artifact.num_repeats <= 1 {
        return Ok(Vec::new());
    }
    // CPU uses the backend's matrix directly; a GPU run uses the retained host copy.
    // Only selected rows are reported/bootstrap-resampled, but maxT streams the
    // complete compact plan for every permutation.
    let matrix = match &artifact.backend {
        CompiledContinuousBackend::Cpu { matrix } => matrix,
        CompiledContinuousBackend::Cuda { .. }
        | CompiledContinuousBackend::Rocm { .. }
        | CompiledContinuousBackend::Metal { .. } => match &artifact.significance_matrix {
            Some(matrix) => matrix,
            None => return Ok(Vec::new()),
        },
    };
    let row_count = table.row_count();
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let mut order: Vec<usize> = (0..row_count).collect();
    order.sort_by(|&left, &right| {
        let left_value = rank_value_at(table, &artifact.metric_ids, left, None);
        let right_value = rank_value_at(table, &artifact.metric_ids, right, None);
        compare_rank_values(left_value, right_value, true)
            .then_with(|| table.candidate_ids()[left].cmp(&table.candidate_ids()[right]))
    });
    let cap = (artifact.significance_top_n.max(1) as usize).min(row_count);
    order.truncate(cap);

    let kernels = artifact
        .metric_ids
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

    let null_family_rows = artifact.prepared.plan().planned_row_count();
    *artifact.runtime_cache_counters.borrow_mut() = RuntimeCacheCounters {
        metric_hits: null_family_rows.saturating_mul(u64::from(artifact.permutation_tests)),
        metric_builds: null_family_rows,
        candidate_table_hits: combos.len() as u64,
    };

    let backend_kind = artifact.backend_kind();
    let params = SignificanceParams {
        permutation_tests: artifact.permutation_tests,
        num_repeats: artifact.num_repeats,
        random_seed: artifact.random_seed,
        mi_bins: artifact.mi_bins,
        backend_kind,
        mi_approximate: artifact.mi_approximate,
    };
    let evaluated = significance::evaluate_with_null_family(
        matrix,
        &combos,
        &observed,
        artifact.prepared.plan(),
        &kernels,
        &params,
    )?;
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
    if feature_len != rows as usize * cols as usize {
        return Err(PyBoundaryError::InvalidInput(
            "feature buffer length does not match rows*cols".to_string(),
        ));
    }
    if target_len != rows as usize {
        return Err(PyBoundaryError::InvalidInput(
            "target length does not match rows".to_string(),
        ));
    }
    Ok(())
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

fn combo_from_table(table: &OwnedResultTable, index: usize) -> Option<Vec<u32>> {
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

fn metric_values_from_table(table: &OwnedResultTable, index: usize) -> Option<Vec<f32>> {
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
    out.random_seed = get_optional_u64(config, "random_seed")?.unwrap_or(0);
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

fn probe_gpu_candidate(kind: u32, device_id: u32) -> Option<AutoBackendCandidate> {
    let backend = match kind {
        GAFIME_BACKEND_CUDA => GpuBackend::cuda_from_env(device_id),
        GAFIME_BACKEND_ROCM => GpuBackend::rocm_from_env(device_id),
        GAFIME_BACKEND_METAL => GpuBackend::metal_from_env(device_id),
        _ => return None,
    }
    .ok()?;
    // A payload is eligible for automatic selection only when its required
    // identity/capability query succeeds. Allocation success is not a substitute:
    // older or mismatched payloads can allocate while exposing an incompatible ABI.
    let info = backend.device_info().ok()?;
    Some(AutoBackendCandidate {
        kind,
        score: gpu_device_score(&info),
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

fn get_optional_u64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<u64>().map(Some),
        _ => Ok(None),
    }
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

#[pyclass(name = "ContinuousReport", unsendable)]
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
    table: OwnedResultTable,
    significance: Vec<SignificanceEntry>,
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
        let mut indices = (0..self.table.row_count()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            let left_value = rank_value_at(&self.table, &self.metric_ids, left, metric_index);
            let right_value = rank_value_at(&self.table, &self.metric_ids, right, metric_index);
            compare_rank_values(left_value, right_value, descending).then_with(|| {
                self.table.candidate_ids()[left].cmp(&self.table.candidate_ids()[right])
            })
        });
        if let Some(limit) = limit {
            indices.truncate(limit);
        }
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
        let data = result_table_to_arrow(&self.table).into_data();
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
            table: value.table,
            significance: value.significance,
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
    table: &OwnedResultTable,
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

#[pyclass(name = "CompiledContinuousArtifact", unsendable)]
struct PyCompiledContinuousArtifact {
    #[pyo3(get)]
    rows: u64,
    #[pyo3(get)]
    cols: u32,
    #[pyo3(get)]
    max_arity: u32,
    #[pyo3(get)]
    metric_ids: Vec<u32>,
    permutation_tests: u32,
    num_repeats: u32,
    random_seed: u64,
    mi_bins: u32,
    mi_approximate: bool,
    significance_top_n: u32,
    // Host matrix retained for the significance pass on a GPU run (None for CPU,
    // which uses the backend's own matrix, and when significance is not requested).
    significance_matrix: Option<CpuMatrix>,
    backend: CompiledContinuousBackend,
    prepared: PreparedContinuousExecution,
    runtime_cache_counters: RefCell<RuntimeCacheCounters>,
    closed: bool,
}

impl PyCompiledContinuousArtifact {
    fn backend_kind(&self) -> u32 {
        match &self.backend {
            CompiledContinuousBackend::Cpu { .. } => GAFIME_BACKEND_CPU,
            CompiledContinuousBackend::Cuda { .. } => GAFIME_BACKEND_CUDA,
            CompiledContinuousBackend::Rocm { .. } => GAFIME_BACKEND_ROCM,
            CompiledContinuousBackend::Metal { .. } => GAFIME_BACKEND_METAL,
        }
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
    fn graph_requested(&self) -> bool {
        self.prepared.schedule().decision().graph_requested
    }

    fn analyze(&self) -> PyResult<PyContinuousReport> {
        execute_compiled_artifact(self)
            .map(PyContinuousReport::from)
            .map_err(PyErr::from)
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

    /// Resident-session reuse: swap the target in place and re-analyze without
    /// re-uploading the features. On GPU the resident device matrix keeps its
    /// feature buffers and only `y` is refreshed (the permutation/repeat pattern);
    /// on CPU the held matrix's target is replaced. The host significance matrix
    /// (GPU runs) is updated too so a subsequent analyze scores against the new y.
    fn update_target(&mut self, target: Vec<f32>) -> PyResult<()> {
        if self.closed {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        if target.len() as u64 != self.rows {
            return Err(PyValueError::new_err(
                "target length must match the compiled matrix rows",
            ));
        }
        match &mut self.backend {
            CompiledContinuousBackend::Cpu { matrix } => {
                matrix
                    .set_target(target.clone())
                    .map_err(|error| PyErr::from(PyBoundaryError::from(error)))?;
            }
            CompiledContinuousBackend::Cuda { matrix, .. }
            | CompiledContinuousBackend::Rocm { matrix, .. }
            | CompiledContinuousBackend::Metal { matrix, .. } => {
                matrix
                    .update_target(&target)
                    .map_err(|error| PyErr::from(PyBoundaryError::from(error)))?;
            }
        }
        if let Some(significance_matrix) = &mut self.significance_matrix {
            significance_matrix
                .set_target(target)
                .map_err(|error| PyErr::from(PyBoundaryError::from(error)))?;
        }
        Ok(())
    }

    fn close(&mut self) {
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

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
fn analyze_continuous(
    config: &Bound<'_, PyDict>,
    features: Vec<f32>,
    target: Vec<f32>,
    rows: u64,
    cols: u32,
) -> PyResult<PyContinuousReport> {
    let artifact = compile_continuous(config, features, target, rows, cols)?;
    artifact.analyze()
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
    let (expanded, ecols, descriptors) = gafime_cpu::time_series::expand_row_major(
        &features,
        rows as usize,
        cols as usize,
        &lags,
        &windows,
        velocity,
    );
    let report = analyze_continuous(config, expanded, target, rows, ecols as u32)?;
    let mut names = base_names.clone();
    for descriptor in &descriptors {
        let base = base_names
            .get(descriptor.base_feature as usize)
            .map(String::as_str)
            .unwrap_or("feature");
        names.push(gafime_cpu::time_series::feature_label(base, descriptor.op));
    }
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
    let (expanded, ecols, descriptors) = gafime_cpu::time_series::expand_row_major(
        &features,
        rows as usize,
        cols as usize,
        &lags,
        &windows,
        velocity,
    );
    let artifact = compile_continuous(config, expanded, target, rows, ecols as u32)?;
    let mut names = base_names.clone();
    for descriptor in &descriptors {
        let base = base_names
            .get(descriptor.base_feature as usize)
            .map(String::as_str)
            .unwrap_or("feature");
        names.push(gafime_cpu::time_series::feature_label(base, descriptor.op));
    }
    Ok((artifact, names))
}

/// decision_path family: discover depth-k GBDT conjunction paths (with residual
/// boosting) natively, append their hard-AND membership columns after the base
/// features, then mine the expanded matrix through the normal continuous path
/// (CPU or GPU per config). Mirrors `analyze_time_series`. Returns
/// (report, all_feature_names = base ++ path labels).
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, min_leaf, learning_rate))]
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
    min_leaf: u32,
    learning_rate: f32,
) -> PyResult<(PyContinuousReport, Vec<String>)> {
    let params = gafime_cpu::decision_path::DecisionPathParams {
        max_depth,
        rounds,
        max_paths,
        min_leaf,
        learning_rate,
    };
    let (expanded, ecols, paths) = gafime_cpu::decision_path::expand_row_major(
        &features,
        &target,
        rows as usize,
        cols as usize,
        &params,
    );
    let report = analyze_continuous(config, expanded, target, rows, ecols as u32)?;
    let mut names = base_names.clone();
    for path in &paths {
        names.push(gafime_cpu::decision_path::path_label(
            &base_names,
            &path.nodes,
        ));
    }
    Ok((report, names))
}

/// decision_path compile path: discover native GBDT conjunction paths, append
/// membership columns, then return a resident compiled continuous artifact over
/// that expanded matrix.
#[pyfunction]
#[pyo3(signature = (config, features, target, rows, cols, base_names, max_depth, rounds, max_paths, min_leaf, learning_rate))]
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
    min_leaf: u32,
    learning_rate: f32,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    let params = gafime_cpu::decision_path::DecisionPathParams {
        max_depth,
        rounds,
        max_paths,
        min_leaf,
        learning_rate,
    };
    let (expanded, ecols, paths) = gafime_cpu::decision_path::expand_row_major(
        &features,
        &target,
        rows as usize,
        cols as usize,
        &params,
    );
    let artifact = compile_continuous(config, expanded, target, rows, ecols as u32)?;
    let mut names = base_names.clone();
    for path in &paths {
        names.push(gafime_cpu::decision_path::path_label(
            &base_names,
            &path.nodes,
        ));
    }
    Ok((artifact, names))
}

#[pymodule]
fn gafime_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("BOUNDARY_NAME", BOUNDARY_NAME)?;
    m.add_class::<PyCompiledContinuousArtifact>()?;
    m.add_class::<PyContinuousRecord>()?;
    m.add_class::<PyContinuousReport>()?;
    m.add_function(wrap_pyfunction!(compile_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(compile_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(compile_decision_path, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_decision_path, m)?)?;
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

        let artifact = match compile_continuous_rows(
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
        let report = execute_compiled_artifact(&artifact).unwrap();
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
}
