use std::{error::Error, fmt};

use gafime_cpu::{matrix::CpuMatrix, result::OwnedResultTable, CpuBackend};
use gafime_orchestrator::{
    config::EngineConfig, execute_plan, prepare_continuous_execution, OrchestratorError,
    PreparedContinuousExecution,
};
use gafime_types::{
    GAFIME_BACKEND_CPU, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_METRIC_SPEARMAN,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub const BOUNDARY_NAME: &str = "gafime-py";

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousRecord {
    pub combo: Vec<u32>,
    pub metrics: Vec<f32>,
    pub candidate_id: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousReport {
    pub rows: u64,
    pub cols: u32,
    pub max_arity: u32,
    pub metric_ids: Vec<u32>,
    pub records: Vec<ContinuousRecord>,
}

#[derive(Debug)]
pub enum PyBoundaryError {
    InvalidInput(String),
    UnsupportedFeature(String),
    Orchestrator(OrchestratorError),
}

impl fmt::Display for PyBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid v1 boundary input: {message}"),
            Self::UnsupportedFeature(message) => write!(f, "unsupported v1 feature: {message}"),
            Self::Orchestrator(error) => write!(f, "v1 orchestrator error: {error:?}"),
        }
    }
}

impl Error for PyBoundaryError {}

impl From<OrchestratorError> for PyBoundaryError {
    fn from(value: OrchestratorError) -> Self {
        Self::Orchestrator(value)
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
    Ok(config)
}

fn compile_continuous_cpu_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    validate_shape(rows, cols, features.len(), target.len())?;
    let matrix = CpuMatrix::from_row_major(rows, cols, features, target)?;
    let prepared = prepare_continuous_execution(&config, rows, cols)?;
    Ok(PyCompiledContinuousArtifact {
        rows,
        cols,
        max_arity: prepared.result_max_arity(),
        metric_ids: config.metric_ids.clone(),
        matrix,
        prepared,
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
    let mut backend = CpuBackend;
    execute_plan(
        &mut backend,
        &artifact.matrix.handle(),
        artifact.prepared.plan(),
        table.raw_mut(),
    )?;

    Ok(report_from_table(
        artifact.rows,
        artifact.cols,
        artifact.prepared.result_max_arity(),
        artifact.metric_ids.clone(),
        &table,
    ))
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
    table: &OwnedResultTable,
) -> ContinuousReport {
    let raw = table.raw();
    let row_count = raw.row_count as usize;
    let max_arity_usize = raw.max_arity as usize;
    let metric_count = raw.metric_count as usize;
    let combos = table.combo_indices();
    let metrics = table.metric_values();
    let mut records = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let combo_base = row * max_arity_usize;
        let metric_base = row * metric_count;
        let combo = combos[combo_base..combo_base + max_arity_usize]
            .iter()
            .copied()
            .filter(|&feature| feature != u32::MAX)
            .collect();
        let row_metrics = metrics[metric_base..metric_base + metric_count].to_vec();
        records.push(ContinuousRecord {
            combo,
            metrics: row_metrics,
            candidate_id: row as u64,
        });
    }
    ContinuousReport {
        rows,
        cols,
        max_arity,
        metric_ids,
        records,
    }
}

fn parse_engine_config(config: &Bound<'_, PyDict>) -> PyResult<EngineConfig> {
    if get_bool(config, "enable_time_series_functions", false)? {
        return Err(PyErr::from(PyBoundaryError::UnsupportedFeature(
            "time-series families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        )));
    }
    if get_bool(config, "enable_decision_path_functions", false)? {
        return Err(PyErr::from(PyBoundaryError::UnsupportedFeature(
            "decision-path families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        )));
    }

    let mut out = EngineConfig::default();
    out.backend_kind = backend_kind_from_name(&get_string(config, "backend", "auto")?)?;
    out.device_id = get_u32(config, "device_id", 0)?;
    out.metric_ids = metric_ids_from_names(get_vec_string(config, "metric_names")?)?;
    out.num_repeats = get_u32(config, "num_repeats", 3)?;
    out.permutation_tests = get_u32(config, "permutation_tests", 25)?;
    out.random_seed = get_optional_u64(config, "random_seed")?.unwrap_or(0);
    out.mi_bins = get_u32(config, "mi_bins", 96)?;

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

fn backend_kind_from_name(name: &str) -> PyResult<u32> {
    match name {
        "auto" | "cpu" | "core" | "rust" | "v1-rust-cpu" => Ok(GAFIME_BACKEND_CPU),
        "cuda" | "gpu" | "rocm" | "hip" | "metal" => Err(PyErr::from(
            PyBoundaryError::UnsupportedFeature(format!(
                "backend {name:?} requires native gafime-gpu-sys execution and will not fall back to Python"
            )),
        )),
        other => Err(PyErr::from(PyBoundaryError::InvalidInput(format!(
            "unknown backend {other:?}"
        )))),
    }
}

fn metric_ids_from_names(names: Vec<String>) -> PyResult<Vec<u32>> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        ids.push(match name.as_str() {
            "pearson" => GAFIME_METRIC_PEARSON,
            "spearman" => GAFIME_METRIC_SPEARMAN,
            "mutual_info" => GAFIME_METRIC_MUTUAL_INFO,
            "r2" => GAFIME_METRIC_R2,
            other => {
                return Err(PyErr::from(PyBoundaryError::InvalidInput(format!(
                    "unsupported metric {other:?}"
                ))))
            }
        });
    }
    validate_metric_ids(ids).map_err(PyErr::from)
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
    records: Vec<PyContinuousRecord>,
}

#[pymethods]
impl PyContinuousReport {
    fn __len__(&self) -> usize {
        self.records.len()
    }

    fn record(&self, index: usize) -> PyResult<PyContinuousRecord> {
        self.records
            .get(index)
            .cloned()
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn combo(&self, index: usize) -> PyResult<Vec<u32>> {
        Ok(self.record(index)?.combo)
    }

    fn metric_values(&self, index: usize) -> PyResult<Vec<f32>> {
        Ok(self.record(index)?.metrics)
    }

    fn candidate_id(&self, index: usize) -> PyResult<u64> {
        Ok(self.record(index)?.candidate_id)
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
        let mut indices = (0..self.records.len()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            let left_value = rank_value(&self.records[left], &self.metric_ids, metric_index);
            let right_value = rank_value(&self.records[right], &self.metric_ids, metric_index);
            compare_rank_values(left_value, right_value, descending).then_with(|| {
                self.records[left]
                    .candidate_id
                    .cmp(&self.records[right].candidate_id)
            })
        });
        if let Some(limit) = limit {
            indices.truncate(limit);
        }
        Ok(indices)
    }

    fn records(&self) -> Vec<PyContinuousRecord> {
        self.records.clone()
    }
}

impl From<ContinuousReport> for PyContinuousReport {
    fn from(value: ContinuousReport) -> Self {
        Self {
            rows: value.rows,
            cols: value.cols,
            max_arity: value.max_arity,
            metric_ids: value.metric_ids,
            records: value
                .records
                .into_iter()
                .map(|record| PyContinuousRecord {
                    combo: record.combo,
                    metrics: record.metrics,
                    candidate_id: record.candidate_id,
                })
                .collect(),
        }
    }
}

fn rank_value(
    record: &PyContinuousRecord,
    metric_ids: &[u32],
    metric_index: Option<usize>,
) -> Option<f32> {
    if let Some(metric_index) = metric_index {
        return record.metrics.get(metric_index).copied();
    }
    record
        .metrics
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
    matrix: CpuMatrix,
    prepared: PreparedContinuousExecution,
    closed: bool,
}

#[pymethods]
impl PyCompiledContinuousArtifact {
    #[getter]
    fn backend_name(&self) -> &'static str {
        "v1-rust-cpu"
    }

    #[getter]
    fn device(&self) -> &'static str {
        "cpu"
    }

    #[getter]
    fn is_gpu(&self) -> bool {
        false
    }

    fn analyze(&self) -> PyResult<PyContinuousReport> {
        execute_compiled_artifact(self)
            .map(PyContinuousReport::from)
            .map_err(PyErr::from)
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
    compile_continuous_cpu_rows(config, rows, cols, features, target).map_err(PyErr::from)
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
    py: Python<'_>,
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
    py.allow_threads(move || {
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
    })
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
        assert_eq!(report.records.len(), 3);
        assert_eq!(report.records[0].combo, vec![0]);
        assert!((report.records[0].metrics[0] - 1.0).abs() < 1e-6);
        assert!((report.records[1].metrics[0] + 1.0).abs() < 1e-6);
        assert_eq!(report.records[2].metrics[0], 0.0);
    }
}
