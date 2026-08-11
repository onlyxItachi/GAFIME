use std::{error::Error, fmt, sync::Arc};

use arrow::array::{
    ArrayRef, FixedSizeListArray, Float32Array, Float64Array, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Fields};
use gafime_cpu::{
    precision::{CpuPrecisionScalar, CpuPrecisionValues},
    result::{OwnedResultTable, OwnedResultTableF64, PrecisionOwnedResultTable},
};
use gafime_gpu_sys::GpuSysError;
use gafime_orchestrator::OrchestratorError;
use gafime_types::{
    PrecisionProfile, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_METRIC_SPEARMAN, GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use pyo3::{exceptions::PyValueError, prelude::*};

pub(crate) trait ResultTableView {
    fn row_count(&self) -> usize;
    fn metric_count(&self) -> usize;
    fn max_arity(&self) -> usize;
    fn combo_indices(&self) -> &[u32];
    fn metric_values(&self) -> MetricValuesRef<'_>;
    fn ranks(&self) -> &[u32];
    fn candidate_ids(&self) -> &[u64];
    fn flags(&self) -> u32;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MetricValuesRef<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl MetricValuesRef<'_> {
    fn to_arrow(self) -> (Arc<Field>, ArrayRef) {
        match self {
            Self::F32(values) => {
                let item = Arc::new(Field::new("item", DataType::Float32, false));
                (
                    item,
                    Arc::new(Float32Array::from(values.to_vec())) as ArrayRef,
                )
            }
            Self::F64(values) => {
                let item = Arc::new(Field::new("item", DataType::Float64, false));
                (
                    item,
                    Arc::new(Float64Array::from(values.to_vec())) as ArrayRef,
                )
            }
        }
    }

    pub(crate) fn range_to_owned(self, start: usize, end: usize) -> CpuPrecisionValues {
        match self {
            Self::F32(values) => CpuPrecisionValues::F32(values[start..end].to_vec()),
            Self::F64(values) => CpuPrecisionValues::F64(values[start..end].to_vec()),
        }
    }
}

pub(crate) fn precision_values_to_python_array(
    py: Python<'_>,
    values: &CpuPrecisionValues,
) -> PyResult<Py<PyAny>> {
    let array = py.import("array")?.getattr("array")?;
    let array = match values {
        CpuPrecisionValues::F32(values) => array.call1(("f", values))?,
        CpuPrecisionValues::F64(values) => array.call1(("d", values))?,
    };
    Ok(array.unbind())
}

pub(crate) fn precision_scalar_to_python(
    py: Python<'_>,
    value: CpuPrecisionScalar,
) -> PyResult<Py<PyAny>> {
    let scalar = match value {
        CpuPrecisionScalar::F32(value) => value.into_pyobject(py)?.into_any().unbind(),
        CpuPrecisionScalar::F64(value) => value.into_pyobject(py)?.into_any().unbind(),
    };
    Ok(scalar)
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

    fn metric_values(&self) -> MetricValuesRef<'_> {
        MetricValuesRef::F32(OwnedResultTable::metric_values(self))
    }

    fn ranks(&self) -> &[u32] {
        OwnedResultTable::ranks(self)
    }

    fn candidate_ids(&self) -> &[u64] {
        OwnedResultTable::candidate_ids(self)
    }

    fn flags(&self) -> u32 {
        self.raw().flags
    }
}

#[derive(Debug)]
pub(crate) struct SendOwnedResultTable(PrecisionOwnedResultTable);

impl SendOwnedResultTable {
    pub(crate) fn new(table: PrecisionOwnedResultTable) -> Self {
        Self(table)
    }
}

// SAFETY: this wrapper is private and is constructed only by consuming a
// completed synchronous execution result. OwnedResultTable owns primitive Vec
// buffers; its raw C descriptor only aliases those stable heap allocations and
// has no destructor. Report methods expose immutable slices and never call
// raw()/raw_mut(), so neither the descriptor nor the buffers can be rebound or
// mutated after wrapping. Moving or dropping this owner on another thread is
// therefore equivalent to moving or dropping the owned Vec buffers themselves.
unsafe impl Send for SendOwnedResultTable {}

// SAFETY: the same post-execution immutability invariant documented above also
// makes shared report access read-only. The private wrapper exposes no mutable
// table or raw-descriptor access after construction.
unsafe impl Sync for SendOwnedResultTable {}

impl ResultTableView for SendOwnedResultTable {
    fn row_count(&self) -> usize {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.row_count(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.row_count(),
        }
    }

    fn metric_count(&self) -> usize {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.metric_count(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.metric_count(),
        }
    }

    fn max_arity(&self) -> usize {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.max_arity(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.max_arity(),
        }
    }

    fn combo_indices(&self) -> &[u32] {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.combo_indices(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.combo_indices(),
        }
    }

    fn metric_values(&self) -> MetricValuesRef<'_> {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => MetricValuesRef::F32(table.metric_values()),
            PrecisionOwnedResultTable::F64 { table, .. } => {
                MetricValuesRef::F64(table.metric_values())
            }
        }
    }

    fn ranks(&self) -> &[u32] {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.ranks(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.ranks(),
        }
    }

    fn candidate_ids(&self) -> &[u64] {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.candidate_ids(),
            PrecisionOwnedResultTable::F64 { table, .. } => table.candidate_ids(),
        }
    }

    fn flags(&self) -> u32 {
        match &self.0 {
            PrecisionOwnedResultTable::Fp32(table) => table.raw().flags,
            PrecisionOwnedResultTable::F64 { table, .. } => table.raw().flags,
        }
    }
}

impl ResultTableView for OwnedResultTableF64 {
    fn row_count(&self) -> usize {
        OwnedResultTableF64::row_count(self)
    }

    fn metric_count(&self) -> usize {
        OwnedResultTableF64::metric_count(self)
    }

    fn max_arity(&self) -> usize {
        OwnedResultTableF64::max_arity(self)
    }

    fn combo_indices(&self) -> &[u32] {
        OwnedResultTableF64::combo_indices(self)
    }

    fn metric_values(&self) -> MetricValuesRef<'_> {
        MetricValuesRef::F64(OwnedResultTableF64::metric_values(self))
    }

    fn ranks(&self) -> &[u32] {
        OwnedResultTableF64::ranks(self)
    }

    fn candidate_ids(&self) -> &[u64] {
        OwnedResultTableF64::candidate_ids(self)
    }

    fn flags(&self) -> u32 {
        self.raw().flags
    }
}

impl ResultTableView for PrecisionOwnedResultTable {
    fn row_count(&self) -> usize {
        match self {
            Self::Fp32(table) => table.row_count(),
            Self::F64 { table, .. } => table.row_count(),
        }
    }

    fn metric_count(&self) -> usize {
        match self {
            Self::Fp32(table) => table.metric_count(),
            Self::F64 { table, .. } => table.metric_count(),
        }
    }

    fn max_arity(&self) -> usize {
        match self {
            Self::Fp32(table) => table.max_arity(),
            Self::F64 { table, .. } => table.max_arity(),
        }
    }

    fn combo_indices(&self) -> &[u32] {
        match self {
            Self::Fp32(table) => table.combo_indices(),
            Self::F64 { table, .. } => table.combo_indices(),
        }
    }

    fn metric_values(&self) -> MetricValuesRef<'_> {
        match self {
            Self::Fp32(table) => MetricValuesRef::F32(table.metric_values()),
            Self::F64 { table, .. } => MetricValuesRef::F64(table.metric_values()),
        }
    }

    fn ranks(&self) -> &[u32] {
        match self {
            Self::Fp32(table) => table.ranks(),
            Self::F64 { table, .. } => table.ranks(),
        }
    }

    fn candidate_ids(&self) -> &[u64] {
        match self {
            Self::Fp32(table) => table.candidate_ids(),
            Self::F64 { table, .. } => table.candidate_ids(),
        }
    }

    fn flags(&self) -> u32 {
        match self {
            Self::Fp32(table) => table.raw().flags,
            Self::F64 { table, .. } => table.raw().flags,
        }
    }
}

/// Build a zero-copy-to-consumer Arrow `StructArray` over the compact result
/// table. Columns: `candidate_id` (u64), `rank` (u32), `combo`
/// (FixedSizeList<u32>[max_arity]), `metrics`
/// (FixedSizeList<f32>[metric_count] for fp32 or
/// FixedSizeList<f64>[metric_count] for mixed/fp64).
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

    let (metric_item, metric_child) = match table.metric_values() {
        MetricValuesRef::F32(values) => {
            MetricValuesRef::F32(&values[..rows * metric_count]).to_arrow()
        }
        MetricValuesRef::F64(values) => {
            MetricValuesRef::F64(&values[..rows * metric_count]).to_arrow()
        }
    };
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

pub(crate) fn cargo_version_to_python(cargo_version: &str) -> Result<String, &'static str> {
    let Some((release, prerelease)) = cargo_version.split_once('-') else {
        let mut release_parts = cargo_version.split('.');
        if release_parts.by_ref().count() != 3
            || !cargo_version.split('.').all(canonical_numeric_component)
        {
            return Err("Cargo version must be canonical MAJOR.MINOR.PATCH SemVer");
        }
        return Ok(cargo_version.to_string());
    };
    if release.split('.').count() != 3 || !release.split('.').all(canonical_numeric_component) {
        return Err("Cargo release must be canonical MAJOR.MINOR.PATCH");
    }
    let mut prerelease_parts = prerelease.split('.');
    let Some(label) = prerelease_parts.next() else {
        return Err("Cargo prerelease label is missing");
    };
    let Some(serial) = prerelease_parts.next() else {
        return Err("Cargo prerelease serial is missing");
    };
    if prerelease_parts.next().is_some() || !canonical_numeric_component(serial) {
        return Err("Cargo prerelease must contain one canonical numeric serial");
    }
    let pep440_label = match label {
        "alpha" => "a",
        "beta" => "b",
        "rc" => "rc",
        _ => return Err("Cargo prerelease label must be alpha, beta, or rc"),
    };
    Ok(format!("{release}{pep440_label}{serial}"))
}

fn canonical_numeric_component(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && (value == "0" || !value.starts_with('0'))
}

pub fn public_package_version() -> String {
    cargo_version_to_python(env!("CARGO_PKG_VERSION"))
        .expect("CARGO_PKG_VERSION must satisfy the GAFIME release-version policy")
}

#[pyfunction]
pub(crate) fn native_version() -> String {
    public_package_version()
}
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousRecord {
    pub combo: Vec<u32>,
    pub metrics: CpuPrecisionValues,
    pub candidate_id: u64,
}

/// Per-candidate significance for one surfaced report row: permutation p-values
/// and bootstrap stability (mean/std) per metric, aligned to `metric_ids`.
#[derive(Clone, Debug, PartialEq)]
pub struct SignificanceEntry {
    pub row: usize,
    pub pvalues: CpuPrecisionValues,
    pub means: CpuPrecisionValues,
    pub stds: CpuPrecisionValues,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InteractionPrecisionDiagnostic {
    pub overflow_row_count: u64,
    pub source_nonfinite: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecisionPathResultParams {
    pub(crate) feature_index: u32,
    pub(crate) features: Vec<u32>,
    pub(crate) thresholds: CpuPrecisionValues,
    pub(crate) signs: Vec<i8>,
    pub(crate) gain: CpuPrecisionScalar,
    pub(crate) support: u32,
    pub(crate) round_id: u32,
}

impl DecisionPathResultParams {
    pub(crate) fn from_precision_path(
        feature_index: u32,
        precision: PrecisionProfile,
        path: &gafime_cpu::decision_path::PrecisionDecisionPath,
    ) -> Result<Self, PyBoundaryError> {
        use gafime_cpu::precision::CpuPrecisionScalar;

        let mut features = Vec::with_capacity(path.nodes.len());
        let mut signs = Vec::with_capacity(path.nodes.len());
        for node in &path.nodes {
            features.push(node.feature);
            signs.push(match node.sign {
                gafime_cpu::decision_path::SplitSign::Le => -1,
                gafime_cpu::decision_path::SplitSign::Gt => 1,
            });
        }
        let thresholds = match precision {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => CpuPrecisionValues::F32(
                path.nodes
                    .iter()
                    .map(|node| {
                        node.threshold.f32().ok_or_else(|| {
                            PyBoundaryError::InvalidInput(
                                "fp32/mixed decision-path has an fp64 threshold".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            PrecisionProfile::Fp64 => CpuPrecisionValues::F64(
                path.nodes
                    .iter()
                    .map(|node| {
                        node.threshold.f64().ok_or_else(|| {
                            PyBoundaryError::InvalidInput(
                                "fp64 decision-path has an fp32 threshold".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        let gain_matches_profile = matches!(
            (precision, path.gain),
            (PrecisionProfile::Fp32, CpuPrecisionScalar::F32(_))
                | (
                    PrecisionProfile::Mixed | PrecisionProfile::Fp64,
                    CpuPrecisionScalar::F64(_)
                )
        );
        if !gain_matches_profile {
            return Err(PyBoundaryError::InvalidInput(format!(
                "{precision:?} decision-path gain has the wrong public result dtype"
            )));
        }
        Ok(Self {
            feature_index,
            features,
            thresholds,
            signs,
            gain: path.gain,
            support: path.support,
            round_id: path.round,
        })
    }
}

#[derive(Debug)]
pub struct ContinuousReport {
    pub rows: u64,
    pub cols: u32,
    pub max_arity: u32,
    pub metric_ids: Vec<u32>,
    pub backend_kind: u32,
    pub precision: PrecisionProfile,
    pub graph_replayed: bool,
    pub mi_accumulation_fp64: bool,
    pub interaction_diagnostics: Option<Vec<InteractionPrecisionDiagnostic>>,
    pub(crate) table: PrecisionOwnedResultTable,
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

    pub fn metric_values(&self, index: usize) -> Option<CpuPrecisionValues> {
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
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(PyBoundaryError::InvalidInput(format!(
            "{label} byte length is not divisible by four"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(crate) fn decode_f64_le(bytes: &[u8], label: &str) -> Result<Vec<f64>, PyBoundaryError> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f64>()) {
        return Err(PyBoundaryError::InvalidInput(format!(
            "{label} byte length is not divisible by eight"
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OwnedNumericInput {
    F32 {
        features: Vec<f32>,
        target: Vec<f32>,
    },
    F64 {
        features: Vec<f64>,
        target: Vec<f64>,
    },
}

impl OwnedNumericInput {
    pub(crate) fn from_f32(
        precision: PrecisionProfile,
        features: Vec<f32>,
        target: Vec<f32>,
    ) -> Result<Self, PyBoundaryError> {
        if precision == PrecisionProfile::Fp64 {
            return Err(PyBoundaryError::InvalidInput(
                "fp64 precision cannot ingest an intermediate f32 buffer".to_string(),
            ));
        }
        Ok(Self::F32 { features, target })
    }

    pub(crate) fn from_f64(
        precision: PrecisionProfile,
        features: Vec<f64>,
        target: Vec<f64>,
    ) -> Result<Self, PyBoundaryError> {
        if precision != PrecisionProfile::Fp64 {
            return Err(PyBoundaryError::InvalidInput(
                "f64 resident input requires precision=\"fp64\"".to_string(),
            ));
        }
        Ok(Self::F64 { features, target })
    }

    pub(crate) fn feature_len(&self) -> usize {
        match self {
            Self::F32 { features, .. } => features.len(),
            Self::F64 { features, .. } => features.len(),
        }
    }

    pub(crate) fn target_len(&self) -> usize {
        match self {
            Self::F32 { target, .. } => target.len(),
            Self::F64 { target, .. } => target.len(),
        }
    }
}

pub(crate) fn decode_precision_input(
    precision: PrecisionProfile,
    feature_bytes: &[u8],
    target_bytes: &[u8],
) -> Result<OwnedNumericInput, PyBoundaryError> {
    match precision {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => OwnedNumericInput::from_f32(
            precision,
            decode_f32_le(feature_bytes, "feature")?,
            decode_f32_le(target_bytes, "target")?,
        ),
        PrecisionProfile::Fp64 => OwnedNumericInput::from_f64(
            precision,
            decode_f64_le(feature_bytes, "feature")?,
            decode_f64_le(target_bytes, "target")?,
        ),
    }
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

pub(crate) fn flatten_continuous_rows_f64(
    features: Vec<Vec<f64>>,
    target: Vec<f64>,
) -> Result<(u64, u32, Vec<f64>, Vec<f64>), PyBoundaryError> {
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

#[allow(
    clippy::too_many_arguments,
    reason = "report construction keeps dimensions, metric identity, backend precision, owned results, and significance explicit"
)]
pub(crate) fn report_from_table(
    rows: u64,
    cols: u32,
    max_arity: u32,
    metric_ids: Vec<u32>,
    backend_kind: u32,
    precision: PrecisionProfile,
    mi_accumulation_fp64: bool,
    interaction_diagnostics: Option<Vec<InteractionPrecisionDiagnostic>>,
    table: PrecisionOwnedResultTable,
    significance: Vec<SignificanceEntry>,
) -> ContinuousReport {
    let graph_replayed = (table.flags() & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0;
    ContinuousReport {
        rows,
        cols,
        max_arity,
        metric_ids,
        backend_kind,
        precision,
        graph_replayed,
        mi_accumulation_fp64,
        interaction_diagnostics,
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
) -> Option<CpuPrecisionValues> {
    if index >= table.row_count() {
        return None;
    }
    let metric_count = table.metric_count();
    let metric_base = index.checked_mul(metric_count)?;
    Some(
        table
            .metric_values()
            .range_to_owned(metric_base, metric_base + metric_count),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use gafime_cpu::decision_path::{PrecisionDecisionPath, PrecisionPathNode, SplitSign};
    use gafime_cpu::precision::CpuPrecisionScalar;
    use gafime_types::GAFIME_BACKEND_CUDA;

    use crate::continuous::analyze_continuous_cpu_rows;

    #[test]
    fn boundary_name_is_stable() {
        assert_eq!(boundary_name(), "gafime-py");
    }

    #[test]
    fn decision_path_public_params_reject_mixed_scalar_widths() {
        let path = PrecisionDecisionPath {
            nodes: vec![PrecisionPathNode {
                feature: 0,
                threshold: CpuPrecisionScalar::F64(0.5),
                sign: SplitSign::Le,
            }],
            gain: CpuPrecisionScalar::F32(0.25),
            support: 3,
            round: 0,
        };
        let error = DecisionPathResultParams::from_precision_path(2, PrecisionProfile::Fp32, &path)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("fp32/mixed decision-path has an fp64 threshold"));

        let mixed_path = PrecisionDecisionPath {
            nodes: vec![PrecisionPathNode {
                feature: 0,
                threshold: CpuPrecisionScalar::F32(0.5),
                sign: SplitSign::Gt,
            }],
            gain: CpuPrecisionScalar::F64(0.25),
            support: 3,
            round: 0,
        };
        let params =
            DecisionPathResultParams::from_precision_path(2, PrecisionProfile::Mixed, &mixed_path)
                .unwrap();
        assert_eq!(params.thresholds.as_f32(), Some(&[0.5][..]));
        assert_eq!(params.gain, CpuPrecisionScalar::F64(0.25));
    }

    #[test]
    fn cargo_prerelease_version_maps_to_python_public_version() {
        assert_eq!(
            cargo_version_to_python("1.0.0-alpha.0"),
            Ok("1.0.0a0".to_string())
        );
        assert_eq!(
            cargo_version_to_python("1.0.0-beta.12"),
            Ok("1.0.0b12".to_string())
        );
        assert_eq!(
            cargo_version_to_python("1.0.0-rc.3"),
            Ok("1.0.0rc3".to_string())
        );
        assert_eq!(cargo_version_to_python("2.3.4"), Ok("2.3.4".to_string()));
        for invalid in [
            "1.0",
            "01.0.0",
            "1.0.0-beta",
            "1.0.0-beta.01",
            "1.0.0-dev.1",
            "1.0.0+local",
        ] {
            assert!(cargo_version_to_python(invalid).is_err(), "{invalid}");
        }
        assert_eq!(public_package_version(), "1.0.0b2");
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
            PrecisionProfile::Fp32,
            false,
            Some(Vec::new()),
            PrecisionOwnedResultTable::Fp32(table),
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

        let array = result_table_view_to_arrow(&report.table);
        assert_eq!(array.len(), report.len());
        assert_eq!(array.num_columns(), 4);

        // The Arrow C Data Interface export must round-trip — this is the exact
        // path Polars/pyarrow/torch consume, validated without a Python wheel.
        let data = array.into_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data).unwrap();
        // SAFETY: to_ffi produced this matching array/schema pair from `data`;
        // both values are consumed exactly once by from_ffi while the schema
        // reference remains live for the call.
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
