//! Copy-on-ingest boundary. Python exporters remain caller-owned; semantic
//! snapshots never alias mutable Python numeric storage after this call.
use super::{error, profile_name};
use arrow::array::{Array, Float32Array, Float64Array};
use gafime_orchestrator::semantic::{
    EvaluationRole, FeatureFrame, GraphEdge, LabelSet, NeighborGraph, NumericColumn,
};
use gafime_types::PrecisionProfile;
use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn bounded_len(value: &Bound<'_, PyAny>, limit: usize, what: &str) -> PyResult<usize> {
    let n = value.len()?;
    if n > limit {
        return Err(PyValueError::new_err(format!(
            "{what} exceeds bounded input limit"
        )));
    }
    Ok(n)
}

// Index only the admitted prefix instead of trusting an arbitrary iterator's
// length hint. A Python exporter cannot turn a bounded declaration into an
// unbounded native extraction by returning a short __len__ and infinite __iter__.
pub(super) fn bounded_items<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    limit: usize,
    what: &str,
) -> PyResult<impl Iterator<Item = PyResult<Bound<'py, PyAny>>> + 'a> {
    let n = bounded_len(value, limit, what)?;
    Ok((0..n).map(move |i| value.get_item(i)))
}

pub(super) fn bounded_text(value: &str, limit: usize, what: &str) -> PyResult<String> {
    if value.is_empty() || value.len() > limit {
        return Err(PyValueError::new_err(format!(
            "{what} must be nonempty and fit its byte limit"
        )));
    }
    Ok(value.to_owned())
}

pub(super) fn execution_intent(
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<(String, PrecisionProfile, i32)> {
    let Some(config) = config else {
        return Ok(("auto".into(), PrecisionProfile::Mixed, 0));
    };
    let class = config
        .py()
        .import("gafime.config")?
        .getattr("EngineConfig")?;
    if !config.is_instance(&class)? {
        return Err(PyValueError::new_err("config must be EngineConfig"));
    }
    // EngineConfig remains the single public precision selector. Other legacy
    // request controls are not silently interpreted as semantic policy.
    let defaults = class.call0()?;
    for field in defaults.getattr("__dataclass_fields__")?.try_iter()? {
        let name: String = field?.extract()?;
        if ["backend", "precision", "device_id"].contains(&name.as_str()) {
            continue;
        }
        if !config
            .getattr(name.as_str())?
            .eq(defaults.getattr(name.as_str())?)?
        {
            return Err(PyValueError::new_err(format!("EngineConfig.{name} is not a tabular semantic control; declare evidence, proposal and policy explicitly")));
        }
    }
    let backend = config.getattr("backend")?;
    let backend = crate::runtime::normalize_runtime_backend(
        &bounded_text(backend.extract::<&str>()?, 32, "backend")?
            .trim()
            .to_ascii_lowercase(),
    )
    .map_err(PyErr::from)?
    .to_owned();
    let device: i64 = config.getattr("device_id")?.extract()?;
    if !(0..=i64::from(i32::MAX)).contains(&device) {
        return Err(PyValueError::new_err(
            "device_id must fit a nonnegative native device ordinal",
        ));
    }
    let profile = match config.getattr("precision")?.extract::<String>()?.as_str() {
        "fp32" => PrecisionProfile::Fp32,
        "mixed" => PrecisionProfile::Mixed,
        "fp64" => PrecisionProfile::Fp64,
        _ => return Err(PyValueError::new_err("unsupported precision")),
    };
    Ok((backend, profile, device as i32))
}

fn role(value: &str) -> PyResult<EvaluationRole> {
    match value {
        "discovery" => Ok(EvaluationRole::Discovery),
        "holdout" => Ok(EvaluationRole::Holdout),
        "inference" => Ok(EvaluationRole::Inference),
        _ => Err(PyValueError::new_err(
            "role must be discovery, holdout or inference",
        )),
    }
}

fn shape(rows: usize, cols: usize, width: usize) -> PyResult<()> {
    if rows == 0
        || rows > 1_000_000
        || cols == 0
        || cols > 65_536
        || rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_add(rows * 8))
            .is_none_or(|n| n > 256 * 1024 * 1024)
    {
        return Err(PyValueError::new_err(
            "snapshot exceeds bounded row/schema/numeric storage limits",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn snapshot(
    data: &Bound<'_, PyAny>,
    profile: PrecisionProfile,
    names: Option<&Bound<'_, PyAny>>,
    keys: &Bound<'_, PyAny>,
    domain: &str,
    evaluation_role: &str,
    provenance: &str,
) -> PyResult<FeatureFrame> {
    let domain = bounded_text(domain, 4096, "row domain")?;
    let provenance = bounded_text(provenance, 4096, "provenance")?;
    let names = names
        .map(|value| {
            bounded_items(value, 65_536, "feature names")?
                .map(|item| {
                    let item = item?;
                    bounded_text(item.extract::<&str>()?, 256, "feature name")
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let keys: Vec<u64> = bounded_items(keys, 1_000_000, "row keys")?
        .map(|x| x?.extract())
        .collect::<PyResult<_>>()?;
    let width = if profile == PrecisionProfile::Fp64 {
        8
    } else {
        4
    };
    let (schema, columns) = if data.hasattr("__arrow_c_stream__")? {
        let table = crate::py_api::import_arrow_struct(data)?;
        shape(table.len(), table.num_columns(), width)?;
        if table.null_count() != 0 {
            return Err(PyValueError::new_err("null struct rows are unsupported"));
        }
        let schema: Vec<_> = table
            .fields()
            .iter()
            .map(|f| bounded_text(f.name(), 256, "feature name"))
            .collect::<PyResult<_>>()?;
        if names.as_ref().is_some_and(|n| n != &schema) {
            return Err(PyValueError::new_err(
                "feature_names must match Arrow schema exactly",
            ));
        }
        let columns: PyResult<Vec<_>> = table
            .columns()
            .iter()
            .map(|column| {
                if column.null_count() != 0 {
                    return Err(PyValueError::new_err("null feature values are unsupported"));
                }
                if profile == PrecisionProfile::Fp64 {
                    let column =
                        column
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                PyValueError::new_err(
                                    "fp64 snapshot requires Float64 Arrow columns",
                                )
                            })?;
                    Ok(NumericColumn::from(column.values().to_vec()))
                } else {
                    let column =
                        column
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| {
                                PyValueError::new_err(
                                    "fp32/mixed snapshot requires Float32 Arrow columns",
                                )
                            })?;
                    Ok(NumericColumn::from(column.values().to_vec()))
                }
            })
            .collect();
        (schema, columns?)
    } else {
        let schema = names.ok_or_else(|| {
            PyValueError::new_err("feature_names are required for a numeric buffer")
        })?;
        if profile == PrecisionProfile::Fp64 {
            let buffer = PyBuffer::<f64>::get(data)?;
            if buffer.dimensions() != 2 {
                return Err(PyValueError::new_err(
                    "features must be a two-dimensional typed buffer",
                ));
            }
            let (rows, cols) = (buffer.shape()[0], buffer.shape()[1]);
            shape(rows, cols, width)?;
            let flat = buffer.to_vec(data.py())?;
            let columns = (0..cols)
                .map(|c| {
                    NumericColumn::from((0..rows).map(|r| flat[r * cols + c]).collect::<Vec<_>>())
                })
                .collect();
            (schema, columns)
        } else {
            let buffer = PyBuffer::<f32>::get(data)?;
            if buffer.dimensions() != 2 {
                return Err(PyValueError::new_err(
                    "features must be a two-dimensional typed buffer",
                ));
            }
            let (rows, cols) = (buffer.shape()[0], buffer.shape()[1]);
            shape(rows, cols, width)?;
            let flat = buffer.to_vec(data.py())?;
            let columns = (0..cols)
                .map(|c| {
                    NumericColumn::from((0..rows).map(|r| flat[r * cols + c]).collect::<Vec<_>>())
                })
                .collect();
            (schema, columns)
        }
    };
    FeatureFrame::with_profile(
        profile,
        schema,
        domain,
        keys,
        role(evaluation_role)?,
        provenance,
        columns,
    )
    .map_err(error)
}

/// Immutable Rust-owned numeric input with explicit row alignment and provenance.
/// Created by TabularSession; mutating the original exporter has no effect.
#[pyclass(name = "Snapshot", module = "gafime.semantic", frozen)]
pub(crate) struct PySnapshot {
    pub(super) frame: Arc<FeatureFrame>,
}

#[pymethods]
impl PySnapshot {
    #[getter]
    fn rows(&self) -> usize {
        self.frame.rows()
    }
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.frame.schema().to_vec()
    }
    #[getter]
    fn row_keys(&self) -> Vec<u64> {
        self.frame.row_keys().to_vec()
    }
    #[getter]
    fn row_domain(&self) -> &str {
        self.frame.row_domain()
    }
    #[getter]
    fn provenance(&self) -> &str {
        self.frame.provenance()
    }
    #[getter]
    fn precision(&self) -> &'static str {
        profile_name(self.frame.profile())
    }
    #[getter]
    fn role(&self) -> &'static str {
        match self.frame.role() {
            EvaluationRole::Discovery => "discovery",
            EvaluationRole::Holdout => "holdout",
            EvaluationRole::Inference => "inference",
        }
    }

    /// Bind actual labels by unique row key, not positional coincidence. Omitted
    /// keys are unlabeled; duplicate/foreign keys and nonfinite labels error.
    #[pyo3(signature=(*, row_keys, values, provenance))]
    fn labels(
        &self,
        row_keys: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        provenance: &str,
    ) -> PyResult<PyLabels> {
        let provenance = bounded_text(provenance, 4096, "label provenance")?;
        let keys: Vec<u64> = bounded_items(row_keys, self.frame.rows(), "label keys")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        let values: Vec<f64> = bounded_items(values, self.frame.rows(), "label values")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        if keys.len() != values.len() {
            return Err(PyValueError::new_err("label keys/values length mismatch"));
        }
        let index = self.index();
        let pairs: PyResult<Vec<_>> = keys
            .into_iter()
            .zip(values)
            .map(|(key, value)| {
                Ok((
                    *index
                        .get(&key)
                        .ok_or_else(|| PyValueError::new_err("label key absent from snapshot"))?,
                    value,
                ))
            })
            .collect();
        let labels = if self.frame.profile() == PrecisionProfile::Fp64 {
            LabelSet::new_f64(&self.frame, pairs?, provenance)
        } else {
            LabelSet::new(
                &self.frame,
                pairs?.into_iter().map(|(r, v)| (r, v as f32)).collect(),
                provenance,
            )
        }
        .map_err(error)?;
        Ok(PyLabels {
            labels: Arc::new(labels),
        })
    }

    /// Bind an undirected weighted edge multiset by row keys. Duplicate edges
    /// count repeatedly; self-loops, absent keys and nonpositive weights error.
    /// The caller supplies the graph; no neighbor learning or graph quality is
    /// inferred by GAFIME.
    #[pyo3(signature=(*, left_keys, right_keys, weights, provenance))]
    fn graph(
        &self,
        left_keys: &Bound<'_, PyAny>,
        right_keys: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        provenance: &str,
    ) -> PyResult<PyGraph> {
        let provenance = bounded_text(provenance, 4096, "graph provenance")?;
        let n = bounded_len(left_keys, 1_000_000, "graph edges")?;
        if bounded_len(right_keys, 1_000_000, "graph edges")? != n
            || bounded_len(weights, 1_000_000, "graph weights")? != n
        {
            return Err(PyValueError::new_err("graph edge/weight length mismatch"));
        }
        let left: Vec<u64> = bounded_items(left_keys, n, "graph edges")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        let right: Vec<u64> = bounded_items(right_keys, n, "graph edges")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        let weights: Vec<f64> = bounded_items(weights, n, "graph weights")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        if left.len() != n || right.len() != n || weights.len() != n {
            return Err(PyValueError::new_err(
                "graph inputs changed length during extraction",
            ));
        }
        let index = self.index();
        let edges: PyResult<Vec<_>> = left
            .into_iter()
            .zip(right)
            .zip(weights)
            .map(|((a, b), weight)| {
                Ok(GraphEdge {
                    left: *index
                        .get(&a)
                        .ok_or_else(|| PyValueError::new_err("graph key absent from snapshot"))?,
                    right: *index
                        .get(&b)
                        .ok_or_else(|| PyValueError::new_err("graph key absent from snapshot"))?,
                    weight,
                })
            })
            .collect();
        Ok(PyGraph {
            graph: Arc::new(NeighborGraph::new(&self.frame, edges?, provenance).map_err(error)?),
        })
    }
}
impl PySnapshot {
    fn index(&self) -> BTreeMap<u64, usize> {
        self.frame
            .row_keys()
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i))
            .collect()
    }
}

/// Immutable actual-label subset bound to one snapshot. Missing labels are not
/// replaced by zero, anchors, or pseudo-labels.
#[pyclass(name = "Labels", module = "gafime.semantic", frozen)]
pub(crate) struct PyLabels {
    pub(super) labels: Arc<LabelSet>,
}
#[pymethods]
impl PyLabels {
    #[getter]
    fn support(&self) -> usize {
        self.labels.rows().len()
    }
    #[getter]
    fn provenance(&self) -> &str {
        self.labels.provenance()
    }
}

/// Immutable caller-supplied weighted graph bound to exact snapshot rows.
#[pyclass(name = "Graph", module = "gafime.semantic", frozen)]
pub(crate) struct PyGraph {
    pub(super) graph: Arc<NeighborGraph>,
}
#[pymethods]
impl PyGraph {
    #[getter]
    fn edges(&self) -> usize {
        self.graph.edges().len()
    }
    #[getter]
    fn provenance(&self) -> &str {
        self.graph.provenance()
    }
}
