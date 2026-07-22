use std::{error::Error, fmt, sync::Arc};

use arrow::array::{
    ArrayRef, FixedSizeListArray, Float32Array, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Fields};
use gafime_cpu::result::OwnedResultTable;
use gafime_gpu_sys::GpuSysError;
use gafime_orchestrator::OrchestratorError;
use gafime_types::{
    GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
    GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use pyo3::{exceptions::PyValueError, prelude::*};

pub(crate) trait ResultTableView {
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
pub(crate) struct SendOwnedResultTable(pub(crate) OwnedResultTable);

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
pub(crate) fn result_table_view_to_arrow(table: &impl ResultTableView) -> StructArray {
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

pub(crate) fn cargo_version_to_python(cargo_version: &str) -> String {
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
pub(crate) fn native_version() -> String {
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
pub(crate) struct DecisionPathResultParams {
    pub(crate) feature_index: u32,
    pub(crate) features: Vec<u32>,
    pub(crate) thresholds: Vec<f32>,
    pub(crate) signs: Vec<i8>,
    pub(crate) gain: f32,
    pub(crate) support: u32,
    pub(crate) round_id: u32,
}

impl DecisionPathResultParams {
    pub(crate) fn from_path(
        feature_index: u32,
        path: &gafime_cpu::decision_path::DecisionPath,
    ) -> Self {
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
    pub mi_accumulation_fp64: bool,
    pub(crate) table: OwnedResultTable,
    pub(crate) significance: Vec<SignificanceEntry>,
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

pub(crate) fn validate_shape(
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

pub(crate) fn decode_f32_le(bytes: &[u8], label: &str) -> Result<Vec<f32>, PyBoundaryError> {
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

pub(crate) fn flatten_continuous_rows(
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

pub(crate) fn validate_metric_ids(metric_ids: Vec<u32>) -> Result<Vec<u32>, PyBoundaryError> {
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

pub(crate) fn report_from_table(
    rows: u64,
    cols: u32,
    max_arity: u32,
    metric_ids: Vec<u32>,
    backend_kind: u32,
    mi_accumulation_fp64: bool,
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
        mi_accumulation_fp64,
        table,
        significance,
    }
}

pub(crate) fn combo_from_table(table: &impl ResultTableView, index: usize) -> Option<Vec<u32>> {
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

pub(crate) fn metric_values_from_table(
    table: &impl ResultTableView,
    index: usize,
) -> Option<Vec<f32>> {
    if index >= table.row_count() {
        return None;
    }
    let metric_count = table.metric_count();
    let metric_base = index.checked_mul(metric_count)?;
    Some(table.metric_values()[metric_base..metric_base + metric_count].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use gafime_types::GAFIME_BACKEND_CUDA;

    use crate::continuous::analyze_continuous_cpu_rows;

    #[test]
    fn boundary_name_is_stable() {
        assert_eq!(boundary_name(), "gafime-py");
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
            false,
            table,
            Vec::new(),
        );

        assert!(report.graph_replayed);
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
}
