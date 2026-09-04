use std::ffi::CString;

use arrow::array::{Array, Float32Array, Float64Array, StructArray};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use gafime_cpu::precision::CpuPrecisionValues;
use gafime_types::PrecisionProfile;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes, PyCapsule, PyDict},
};

use crate::artifact::{compile_continuous_input, PyCompiledContinuousArtifact};
use crate::common::{
    combo_from_table, decode_precision_input, flatten_continuous_rows, flatten_continuous_rows_f64,
    metric_values_from_table, precision_scalar_to_python, precision_values_to_python_array,
    result_table_view_to_arrow, ContinuousReport, DecisionPathResultParams,
    InteractionPrecisionDiagnostic, OwnedNumericInput, ResultTableView, SendOwnedResultTable,
    SignificanceEntry,
};
use crate::continuous::{
    analyze_continuous_cpu_input_rows, analyze_continuous_input_once, bounded_ranked_indices,
    compare_ranked_rows, continuous_config_for_cpu, row_is_rankable,
};
use crate::runtime::{
    backend_capability_name_for_kind, backend_device_for_kind, backend_is_gpu,
    backend_name_for_kind, execution_placement_for_kind, parse_engine_config,
};

type InteractionComponents = (Vec<u32>, Py<PyAny>, u64);

#[pyclass(name = "ContinuousRecord")]
#[derive(Clone)]
pub(crate) struct PyContinuousRecord {
    #[pyo3(get)]
    combo: Vec<u32>,
    metrics: CpuPrecisionValues,
    #[pyo3(get)]
    candidate_id: u64,
}

#[pymethods]
impl PyContinuousRecord {
    #[getter]
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        precision_values_to_python_array(py, &self.metrics)
    }
}

#[pyclass(name = "ContinuousReport")]
pub(crate) struct PyContinuousReport {
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
    precision: PrecisionProfile,
    mi_accumulation_fp64: bool,
    interaction_diagnostics: Option<Vec<InteractionPrecisionDiagnostic>>,
    table: SendOwnedResultTable,
    significance: Vec<SignificanceEntry>,
    pub(crate) decision_path_params: Vec<DecisionPathResultParams>,
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

    #[getter]
    fn precision(&self) -> &'static str {
        precision_name(self.precision)
    }

    #[getter]
    fn storage_dtype(&self) -> &'static str {
        if self.precision == PrecisionProfile::Fp64 {
            "float64"
        } else {
            "float32"
        }
    }

    #[getter]
    fn interaction_arithmetic(&self) -> &'static str {
        self.storage_dtype()
    }

    #[getter]
    fn result_dtype(&self) -> &'static str {
        if self.precision == PrecisionProfile::Fp32 {
            "float32"
        } else {
            "float64"
        }
    }

    #[getter]
    fn mi_accumulation_dtype(&self) -> &'static str {
        if self.mi_accumulation_fp64 {
            "float64"
        } else {
            "float32"
        }
    }

    #[getter]
    fn interaction_diagnostics_available(&self) -> bool {
        self.interaction_diagnostics.is_some()
    }

    #[getter]
    fn interaction_diagnostics(&self) -> Option<Vec<(u64, bool)>> {
        self.interaction_diagnostics.as_ref().map(|diagnostics| {
            diagnostics
                .iter()
                .map(|diagnostic| (diagnostic.overflow_row_count, diagnostic.source_nonfinite))
                .collect()
        })
    }

    #[getter]
    fn interaction_overflow_candidate_count(&self) -> usize {
        self.interaction_diagnostics
            .as_ref()
            .map_or(0, |diagnostics| {
                diagnostics
                    .iter()
                    .filter(|diagnostic| diagnostic.overflow_row_count != 0)
                    .count()
            })
    }

    #[getter]
    fn interaction_overflow_max_rows(&self) -> u64 {
        self.interaction_diagnostics
            .as_ref()
            .and_then(|diagnostics| {
                diagnostics
                    .iter()
                    .map(|diagnostic| diagnostic.overflow_row_count)
                    .max()
            })
            .unwrap_or(0)
    }

    fn interaction_diagnostic(&self, index: usize) -> PyResult<Option<(u64, bool)>> {
        let Some(diagnostics) = self.interaction_diagnostics.as_ref() else {
            return Ok(None);
        };
        diagnostics
            .get(index)
            .map(|diagnostic| Some((diagnostic.overflow_row_count, diagnostic.source_nonfinite)))
            .ok_or_else(|| PyValueError::new_err("interaction diagnostic index out of range"))
    }

    fn interaction_diagnostics_batch(
        &self,
        start: usize,
        limit: usize,
    ) -> PyResult<Option<Vec<(u64, bool)>>> {
        let Some(diagnostics) = self.interaction_diagnostics.as_ref() else {
            return Ok(None);
        };
        if start > diagnostics.len() {
            return Err(PyValueError::new_err(
                "interaction diagnostic batch start is out of range",
            ));
        }
        let end = start.saturating_add(limit).min(diagnostics.len());
        Ok(Some(
            diagnostics[start..end]
                .iter()
                .map(|diagnostic| (diagnostic.overflow_row_count, diagnostic.source_nonfinite))
                .collect(),
        ))
    }

    fn __len__(&self) -> usize {
        self.table.row_count()
    }

    fn record(&self, index: usize) -> PyResult<PyContinuousRecord> {
        Ok(PyContinuousRecord {
            combo: self.combo(index)?,
            metrics: metric_values_from_table(&self.table, index)
                .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))?,
            candidate_id: self.candidate_id(index)?,
        })
    }

    fn combo(&self, index: usize) -> PyResult<Vec<u32>> {
        combo_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn metric_values(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let values = metric_values_from_table(&self.table, index)
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))?;
        precision_values_to_python_array(py, &values)
    }

    fn candidate_id(&self, index: usize) -> PyResult<u64> {
        self.table
            .candidate_ids()
            .get(index)
            .copied()
            .ok_or_else(|| PyValueError::new_err("continuous report index out of range"))
    }

    fn interaction_components(
        &self,
        py: Python<'_>,
        index: usize,
    ) -> PyResult<InteractionComponents> {
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
        Ok((
            combo,
            precision_values_to_python_array(py, &metrics)?,
            candidate_id,
        ))
    }

    fn interaction_components_batch(
        &self,
        py: Python<'_>,
        start: usize,
        limit: usize,
    ) -> PyResult<Vec<InteractionComponents>> {
        if start > self.table.row_count() {
            return Err(PyValueError::new_err(
                "continuous report batch start is out of range",
            ));
        }
        let end = start.saturating_add(limit).min(self.table.row_count());
        let mut components = Vec::with_capacity(end - start);
        for index in start..end {
            let values = metric_values_from_table(&self.table, index).ok_or_else(|| {
                PyValueError::new_err("continuous report batch row is out of range")
            })?;
            components.push((
                combo_from_table(&self.table, index).unwrap_or_default(),
                precision_values_to_python_array(py, &values)?,
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
        let mut indices = (0..self.table.row_count())
            .filter(|&row| row_is_rankable(&self.table, &self.metric_ids, row, metric_index))
            .collect::<Vec<_>>();
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

    #[pyo3(signature = (indices, *, metric_index=None, descending=true, limit=None))]
    fn ranked_indices_subset(
        &self,
        indices: Vec<usize>,
        metric_index: Option<usize>,
        descending: bool,
        limit: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        if let Some(metric_index) = metric_index {
            if metric_index >= self.metric_ids.len() {
                return Err(PyValueError::new_err("metric_index is out of range"));
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(indices.len());
        if indices
            .iter()
            .any(|&index| index >= self.table.row_count() || !seen.insert(index))
        {
            return Err(PyValueError::new_err(
                "ranked subset indices must be unique and in range",
            ));
        }
        let mut ranked = indices
            .into_iter()
            .filter(|&row| row_is_rankable(&self.table, &self.metric_ids, row, metric_index))
            .collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| {
            compare_ranked_rows(
                &self.table,
                &self.metric_ids,
                left,
                right,
                metric_index,
                descending,
            )
        });
        if let Some(limit) = limit {
            ranked.truncate(limit);
        }
        Ok(ranked)
    }

    fn records(&self) -> Vec<PyContinuousRecord> {
        (0..self.table.row_count())
            .map(|index| PyContinuousRecord {
                combo: combo_from_table(&self.table, index).unwrap_or_default(),
                metrics: metric_values_from_table(&self.table, index)
                    .expect("records() iterates only in-range result rows"),
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

    fn significance_pvalues(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.significance
            .iter()
            .map(|entry| precision_values_to_python_array(py, &entry.pvalues))
            .collect()
    }

    fn significance_means(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.significance
            .iter()
            .map(|entry| precision_values_to_python_array(py, &entry.means))
            .collect()
    }

    fn significance_stds(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.significance
            .iter()
            .map(|entry| precision_values_to_python_array(py, &entry.stds))
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
        let out = PyDict::new(py);
        out.set_item("kind", "decision_path")?;
        out.set_item("features", &params.features)?;
        out.set_item(
            "thresholds",
            precision_values_to_python_array(py, &params.thresholds)?,
        )?;
        out.set_item("signs", &params.signs)?;
        out.set_item("gain", precision_scalar_to_python(py, params.gain)?)?;
        out.set_item("support", params.support)?;
        out.set_item("round_id", params.round_id)?;
        Ok(Some(out))
    }

    /// Arrow PyCapsule Interface (Polars 1.x >= 1.3, pyarrow, etc. consume this
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
        let schema_capsule = PyCapsule::new(py, ffi_schema, Some(schema_name))?;
        let array_capsule = PyCapsule::new(py, ffi_array, Some(array_name))?;
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
            precision: value.precision,
            mi_accumulation_fp64: value.mi_accumulation_fp64,
            interaction_diagnostics: value.interaction_diagnostics,
            table: SendOwnedResultTable::new(value.table),
            significance: value.significance,
            decision_path_params: Vec::new(),
        }
    }
}

fn precision_name(precision: PrecisionProfile) -> &'static str {
    match precision {
        PrecisionProfile::Fp32 => "fp32",
        PrecisionProfile::Mixed => "mixed",
        PrecisionProfile::Fp64 => "fp64",
    }
}

fn parse_precision_name(value: &str) -> PyResult<PrecisionProfile> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp32" => Ok(PrecisionProfile::Fp32),
        "mixed" => Ok(PrecisionProfile::Mixed),
        "fp64" => Ok(PrecisionProfile::Fp64),
        _ => Err(PyValueError::new_err(
            "precision must be one of: fp32, mixed, fp64",
        )),
    }
}

fn extract_flat_input(
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

fn extract_nested_input(
    precision: PrecisionProfile,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<(u64, u32, OwnedNumericInput)> {
    match precision {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            let (rows, cols, features, target) = flatten_continuous_rows(
                features.extract::<Vec<Vec<f32>>>()?,
                target.extract::<Vec<f32>>()?,
            )?;
            Ok((
                rows,
                cols,
                OwnedNumericInput::from_f32(precision, features, target)?,
            ))
        }
        PrecisionProfile::Fp64 => {
            let (rows, cols, features, target) = flatten_continuous_rows_f64(
                features.extract::<Vec<Vec<f64>>>()?,
                target.extract::<Vec<f64>>()?,
            )?;
            Ok((
                rows,
                cols,
                OwnedNumericInput::from_f64(precision, features, target)?,
            ))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
pub(crate) fn compile_continuous(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let input = extract_flat_input(config.precision, features, target)?;
    compile_continuous_input(config, rows, cols, input).map_err(PyErr::from)
}

#[pyfunction(name = "compile_continuous_rows")]
pub(crate) fn compile_continuous_nested(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let (rows, cols, input) = extract_nested_input(config.precision, features, target)?;
    compile_continuous_input(config, rows, cols, input).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
pub(crate) fn compile_continuous_buffers(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyBytes>,
    target: &Bound<'_, PyBytes>,
    rows: u64,
    cols: u32,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let input = decode_precision_input(config.precision, features.as_bytes(), target.as_bytes())?;
    compile_continuous_input(config, rows, cols, input).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
pub(crate) fn analyze_continuous(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    rows: u64,
    cols: u32,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    let input = extract_flat_input(config.precision, features, target)?;
    analyze_continuous_input_once(config, rows, cols, input)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction(name = "analyze_continuous_rows")]
pub(crate) fn analyze_continuous_nested(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    let (rows, cols, input) = extract_nested_input(config.precision, features, target)?;
    analyze_continuous_input_once(config, rows, cols, input)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (config, features, target, *, rows, cols))]
pub(crate) fn analyze_continuous_buffers(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyBytes>,
    target: &Bound<'_, PyBytes>,
    rows: u64,
    cols: u32,
) -> PyResult<PyContinuousReport> {
    let config = parse_engine_config(config)?;
    let input = decode_precision_input(config.precision, features.as_bytes(), target.as_bytes())?;
    analyze_continuous_input_once(config, rows, cols, input)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (features, target, max_arity=2, max_combinations_per_k=5000, metric_ids=None, *, precision="mixed"))]
pub(crate) fn analyze_continuous_cpu(
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Option<Vec<u32>>,
    precision: &str,
) -> PyResult<PyContinuousReport> {
    // Parse the profile before extracting or converting caller-owned values.
    let precision = parse_precision_name(precision)?;
    let (rows, cols, input) = extract_nested_input(precision, features, target)?;
    analyze_continuous_cpu_input_rows(
        precision,
        rows,
        cols,
        input,
        max_arity,
        max_combinations_per_k,
        metric_ids.unwrap_or_default(),
    )
    .map(PyContinuousReport::from)
    .map_err(PyErr::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Float32Array, StructArray},
        datatypes::{DataType, Field, Fields},
    };
    use gafime_types::GAFIME_METRIC_PEARSON;

    use crate::common::MetricValuesRef;
    use crate::continuous::analyze_continuous_cpu_rows;

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
    fn arrow_struct_imports_to_row_major_f64_without_fp32_collapse() {
        let base = 1.0f64;
        let adjacent = f64::from_bits(base.to_bits() + 1);
        assert_eq!(base as f32, adjacent as f32);
        let c0 = Arc::new(Float64Array::from(vec![base, adjacent])) as ArrayRef;
        let c1 = Arc::new(Float64Array::from(vec![2.0f64, 3.0])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]);
        let sa = StructArray::new(fields, vec![c0, c1], None);
        let (rows, cols, flat) = struct_to_row_major_f64(&sa).unwrap();

        assert_eq!((rows, cols), (2, 2));
        assert_eq!(flat[0].to_bits(), base.to_bits());
        assert_eq!(flat[2].to_bits(), adjacent.to_bits());
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
        let metric_ptr = match report.table.metric_values() {
            MetricValuesRef::F32(values) => values.as_ptr().cast::<()>(),
            MetricValuesRef::F64(values) => values.as_ptr().cast::<()>(),
        };
        let rank_ptr = report.table.ranks().as_ptr();
        let candidate_id_ptr = report.table.candidate_ids().as_ptr();

        let python_report = PyContinuousReport::from(report);

        assert_eq!(python_report.table.combo_indices().as_ptr(), combo_ptr);
        let python_metric_ptr = match python_report.table.metric_values() {
            MetricValuesRef::F32(values) => values.as_ptr().cast::<()>(),
            MetricValuesRef::F64(values) => values.as_ptr().cast::<()>(),
        };
        assert_eq!(python_metric_ptr, metric_ptr);
        assert_eq!(python_report.table.ranks().as_ptr(), rank_ptr);
        assert_eq!(
            python_report.table.candidate_ids().as_ptr(),
            candidate_id_ptr
        );
    }
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
    let ptr = cap
        .pointer_checked(Some(c"arrow_array_stream"))?
        .cast::<FFI_ArrowArrayStream>()
        .as_ptr();
    // SAFETY: PyCapsule::pointer returned a non-null Arrow C stream pointer
    // created by __arrow_c_stream__. Replacing it moves ownership exactly once
    // into arrow-rs and leaves an empty stream for the capsule destructor.
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

/// Transpose Arrow Float64 columns directly into row-major f64 storage. This
/// path intentionally has no f32 staging allocation, so adjacent binary64
/// values remain distinct through fp64 ingest.
fn struct_to_row_major_f64(sa: &StructArray) -> PyResult<(u64, u32, Vec<f64>)> {
    let rows = sa.len();
    let cols = sa.num_columns();
    if cols == 0 || rows == 0 {
        return Err(PyValueError::new_err("empty Arrow input"));
    }
    let mut columns: Vec<&Float64Array> = Vec::with_capacity(cols);
    for c in 0..cols {
        let col = sa
            .column(c)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                PyValueError::new_err("feature columns must be Float64 for precision=\"fp64\"")
            })?;
        if col.null_count() != 0 {
            return Err(PyValueError::new_err(
                "null feature values are not supported",
            ));
        }
        columns.push(col);
    }
    let mut flat = vec![0.0f64; rows * cols];
    for (c, column) in columns.iter().enumerate() {
        for r in 0..rows {
            flat[r * cols + c] = column.value(r);
        }
    }
    Ok((rows as u64, cols as u32, flat))
}

#[pyfunction]
#[pyo3(signature = (features, target, *, precision="mixed", max_arity=2, max_combinations_per_k=5000, metric_ids=None))]
pub(crate) fn analyze_continuous_arrow(
    _py: Python<'_>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    precision: &str,
    max_arity: u32,
    max_combinations_per_k: u64,
    metric_ids: Option<Vec<u32>>,
) -> PyResult<PyContinuousReport> {
    // Validate the profile before importing an Arrow stream. This ordering is
    // important: a rejected request must not trigger conversion or allocation.
    let precision = parse_precision_name(precision)?;
    let features = import_arrow_struct(features)?;
    let target = import_arrow_struct(target)?;
    if target.num_columns() != 1 {
        return Err(PyValueError::new_err(
            "target must contain exactly one column",
        ));
    }

    let (rows, cols, input) = match precision {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            let (rows, cols, flat) = struct_to_row_major_f32(&features)?;
            let target_col = target
                .column(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "target column must be Float32 for precision=\"fp32\" or \"mixed\"",
                    )
                })?;
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
            let target = (0..target_col.len())
                .map(|index| target_col.value(index))
                .collect::<Vec<f32>>();
            (
                rows,
                cols,
                OwnedNumericInput::from_f32(precision, flat, target)?,
            )
        }
        PrecisionProfile::Fp64 => {
            let (rows, cols, flat) = struct_to_row_major_f64(&features)?;
            let target_col = target
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    PyValueError::new_err("target column must be Float64 for precision=\"fp64\"")
                })?;
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
            let target = (0..target_col.len())
                .map(|index| target_col.value(index))
                .collect::<Vec<f64>>();
            (
                rows,
                cols,
                OwnedNumericInput::from_f64(precision, flat, target)?,
            )
        }
    };
    let metric_ids = metric_ids.unwrap_or_else(|| {
        vec![
            gafime_types::GAFIME_METRIC_PEARSON,
            gafime_types::GAFIME_METRIC_R2,
        ]
    });
    let mut config = continuous_config_for_cpu(max_arity, max_combinations_per_k, metric_ids)?;
    config.precision = precision;
    analyze_continuous_input_once(config, rows, cols, input)
        .map(PyContinuousReport::from)
        .map_err(PyErr::from)
}
