use std::{error::Error, fmt};

use gafime_cpu::{matrix::CpuMatrix, result::OwnedResultTable, CpuBackend};
use gafime_orchestrator::{
    config::EngineConfig, execute_plan, prepare_continuous_execution, OrchestratorError,
};
use gafime_types::{GAFIME_BACKEND_CPU, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};
use pyo3::{exceptions::PyValueError, prelude::*};

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
    InvalidInput(&'static str),
    Orchestrator(OrchestratorError),
}

impl fmt::Display for PyBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid v1 boundary input: {message}"),
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
    if rows == 0 || cols == 0 {
        return Err(PyBoundaryError::InvalidInput(
            "rows and cols must both be nonzero",
        ));
    }
    if features.len() != rows as usize * cols as usize {
        return Err(PyBoundaryError::InvalidInput(
            "feature buffer length does not match rows*cols",
        ));
    }
    if target.len() != rows as usize {
        return Err(PyBoundaryError::InvalidInput(
            "target length does not match rows",
        ));
    }
    let metric_ids = if metric_ids.is_empty() {
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2]
    } else {
        metric_ids
    };

    let matrix = CpuMatrix::from_row_major(rows, cols, features, target)?;
    let mut config = EngineConfig::default();
    config.backend_kind = GAFIME_BACKEND_CPU;
    config.metric_ids = metric_ids.clone();
    config.budget.max_comb_size = max_arity;
    config.budget.max_combinations_per_k = max_combinations_per_k;

    let prepared = prepare_continuous_execution(&config, rows, cols)?;
    let mut table = OwnedResultTable::new(
        prepared.result_capacity(),
        prepared.result_max_arity(),
        prepared.result_metric_count(),
    );
    let mut backend = CpuBackend;
    execute_plan(
        &mut backend,
        &matrix.handle(),
        prepared.plan(),
        table.raw_mut(),
    )?;

    Ok(report_from_table(
        rows,
        cols,
        prepared.result_max_arity(),
        metric_ids,
        &table,
    ))
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
    m.add_class::<PyContinuousRecord>()?;
    m.add_class::<PyContinuousReport>()?;
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
