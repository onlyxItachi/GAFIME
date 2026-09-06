//! Arrow-owned presentation buffers deliberately outlive their session. Opaque
//! program handles, not output column ordinals, remain semantic authority.
use super::request::PyEvidence;
use super::{error, profile_name, PyCandidate, PyCandidateSet};
use arrow::{
    array::{Array, ArrayRef, Float32Array, Float64Array, StringArray, StructArray, UInt64Array},
    datatypes::{DataType, Field},
};
use gafime_orchestrator::semantic::{
    AcceptedFeature, EvidenceTable, EvidenceValue, FeatureFrame, MaterializedColumns,
    NumericColumn, UnavailableReason,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyCapsule, PyDict},
};
use std::{ffi::CString, sync::Arc};

pub(super) fn reason(value: UnavailableReason) -> &'static str {
    match value {
        UnavailableReason::MissingLabels => "missing_labels",
        UnavailableReason::InsufficientSupport => "insufficient_support",
        UnavailableReason::ConstantOperand => "constant_operand",
        UnavailableReason::DegenerateReduction => "degenerate_reduction",
        UnavailableReason::NonFiniteReduction => "nonfinite_reduction",
    }
}
fn capsules<'py>(
    py: Python<'py>,
    table: &StructArray,
    requested: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    if requested.is_some_and(|x| !x.is_none()) {
        return Err(PyValueError::new_err(
            "Arrow schema casting is not supported; consume the declared schema",
        ));
    }
    let (array, schema) =
        arrow::ffi::to_ffi(&table.to_data()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyCapsule::new(
            py,
            schema,
            Some(CString::new("arrow_schema").expect("literal")),
        )?,
        PyCapsule::new(
            py,
            array,
            Some(CString::new("arrow_array").expect("literal")),
        )?,
    ))
}
fn column(name: &str, values: ArrayRef) -> (Arc<Field>, ArrayRef) {
    (
        Arc::new(Field::new(
            name,
            values.data_type().clone(),
            values.null_count() != 0,
        )),
        values,
    )
}

/// Immutable evidence records, including availability/support and original
/// evaluation context. value() retrieves one record; Arrow export delivers the
/// bounded long-form table without a Python row loop. No composite quality or
/// confidence is inferred. Old records remain readable after session closure.
#[pyclass(name = "EvidenceReport", module = "gafime.semantic", frozen)]
pub(crate) struct PyEvidenceReport {
    pub(super) table: EvidenceTable,
}
#[pymethods]
impl PyEvidenceReport {
    #[getter]
    fn backend(&self) -> &str {
        self.table.backend()
    }
    #[getter]
    fn precision(&self) -> &str {
        self.table.precision()
    }
    #[getter]
    fn provenance(&self) -> &str {
        self.table.frame().provenance()
    }
    #[getter]
    fn candidates(&self) -> PyCandidateSet {
        PyCandidateSet {
            ids: self.table.candidates().to_vec(),
        }
    }
    /// Original immutable evaluation declarations. Provenance is caller-supplied
    /// context, not certification of holdout independence or statistical validity.
    #[getter]
    fn context<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("row_domain", self.table.frame().row_domain())?;
        out.set_item("provenance", self.table.frame().provenance())?;
        out.set_item("rows", self.table.frame().rows())?;
        out.set_item(
            "role",
            format!("{:?}", self.table.frame().role()).to_lowercase(),
        )?;
        let channels = pyo3::types::PyList::empty(py);
        for channel in self.table.channels() {
            let entry = PyDict::new(py);
            entry.set_item("name", channel.name())?;
            entry.set_item("semantics", channel.definition().semantic_name())?;
            match channel.definition() {
                gafime_orchestrator::semantic::EvidenceDefinition::Association {
                    statistic,
                    context,
                } => {
                    entry.set_item("bins", statistic.fixed_nmi_bins())?;
                    match context {
                        gafime_orchestrator::semantic::AssociationContext::Reference {
                            reference,
                        } => {
                            entry.set_item("kind", "reference")?;
                            entry.set_item("reference", PyCandidate { id: *reference })?;
                        }
                        gafime_orchestrator::semantic::AssociationContext::PairedView { view } => {
                            entry.set_item("kind", "paired_view")?;
                            entry.set_item("provenance", view.provenance())?;
                        }
                        gafime_orchestrator::semantic::AssociationContext::Labels { labels } => {
                            entry.set_item("kind", "labels")?;
                            entry
                                .set_item("provenance", labels.as_ref().map(|x| x.provenance()))?;
                        }
                    }
                }
                gafime_orchestrator::semantic::EvidenceDefinition::GraphEnergy { graph } => {
                    entry.set_item("kind", "graph")?;
                    entry.set_item("provenance", graph.provenance())?;
                }
            }
            channels.append(entry)?;
        }
        out.set_item("channels", channels)?;
        Ok(out)
    }
    /// Return one original record; missing values have value=None and a reason.
    fn value<'py>(
        &self,
        py: Python<'py>,
        candidate: PyRef<'_, PyCandidate>,
        channel: PyRef<'_, PyEvidence>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        match self
            .table
            .value(candidate.id, channel.channel.id())
            .map_err(error)?
        {
            EvidenceValue::Measured { value, support } => {
                out.set_item("state", "measured")?;
                out.set_item("value", value)?;
                out.set_item("support", support)?;
                out.set_item("reason", py.None())?;
            }
            EvidenceValue::Unavailable {
                reason: why,
                support,
            } => {
                out.set_item("state", "unavailable")?;
                out.set_item("value", py.None())?;
                out.set_item("support", support)?;
                out.set_item("reason", reason(why))?;
            }
        }
        Ok(out)
    }
    /// Export the typed long-form evidence table through Arrow C Data. Candidate
    /// ordinal is local presentation, not a reusable program ID; use candidates.
    #[pyo3(signature=(requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let mut ordinals = Vec::new();
        let mut names = Vec::new();
        let mut meanings = Vec::new();
        let mut bins = Vec::new();
        let mut values = Vec::new();
        let mut supports = Vec::new();
        let mut reasons = Vec::new();
        for (row, candidate) in self.table.candidates().iter().enumerate() {
            for channel in self.table.channels() {
                ordinals.push(row as u64);
                names.push(channel.name());
                meanings.push(channel.definition().semantic_name());
                bins.push(match channel.definition() {
                    gafime_orchestrator::semantic::EvidenceDefinition::Association {
                        statistic,
                        ..
                    } => statistic.fixed_nmi_bins().map(u64::from),
                    _ => None,
                });
                match self.table.value(*candidate, channel.id()).map_err(error)? {
                    EvidenceValue::Measured { value, support } => {
                        values.push(Some(value));
                        supports.push(support as u64);
                        reasons.push(None);
                    }
                    EvidenceValue::Unavailable {
                        reason: why,
                        support,
                    } => {
                        values.push(None);
                        supports.push(support as u64);
                        reasons.push(Some(reason(why)));
                    }
                }
            }
        }
        let numeric: ArrayRef = if self.table.precision() == "fp32" {
            Arc::new(Float32Array::from(
                values
                    .into_iter()
                    .map(|v| v.map(|x| x as f32))
                    .collect::<Vec<_>>(),
            ))
        } else {
            Arc::new(Float64Array::from(values))
        };
        let table = StructArray::from(vec![
            column("candidate_ordinal", Arc::new(UInt64Array::from(ordinals))),
            column("channel", Arc::new(StringArray::from(names))),
            column("semantics", Arc::new(StringArray::from(meanings))),
            column("bins", Arc::new(UInt64Array::from(bins))),
            column("value", numeric),
            column("support", Arc::new(UInt64Array::from(supports))),
            column("unavailable_reason", Arc::new(StringArray::from(reasons))),
        ]);
        capsules(py, &table, requested_schema)
    }
}

/// Materialized accepted features in accepted-set order, plus explicit row keys.
/// Values use the profile's storage dtype. Arrow buffers remain valid after
/// close; generated column ordinals are not serialized semantic identities.
#[pyclass(name = "FeatureTable", module = "gafime.semantic", frozen)]
pub(crate) struct PyFeatureTable {
    table: StructArray,
    names: Vec<String>,
    keys: Vec<u64>,
    profile: gafime_types::PrecisionProfile,
}
#[pymethods]
impl PyFeatureTable {
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.names.clone()
    }
    #[getter]
    fn row_keys(&self) -> Vec<u64> {
        self.keys.clone()
    }
    #[getter]
    fn precision(&self) -> &'static str {
        profile_name(self.profile)
    }
    #[getter]
    fn rows(&self) -> usize {
        self.table.len()
    }
    /// Export row keys and numeric columns; no cast negotiation or Python rows.
    #[pyo3(signature=(requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        capsules(py, &self.table, requested_schema)
    }
}
pub(super) fn features(
    values: &MaterializedColumns,
    frame: &FeatureFrame,
    accepted: &[AcceptedFeature],
) -> PyResult<PyFeatureTable> {
    let keys = frame.row_keys().to_vec();
    let mut columns: Vec<(Arc<Field>, ArrayRef)> = vec![(
        Arc::new(Field::new("__gafime_row_key__", DataType::UInt64, false)),
        Arc::new(UInt64Array::from(keys.clone())),
    )];
    let mut names = Vec::with_capacity(accepted.len());
    for (index, feature) in accepted.iter().enumerate() {
        let name = format!("feature_{index}");
        let data: ArrayRef = match values.get_typed(feature.feature()).map_err(error)? {
            NumericColumn::F32(v) => Arc::new(Float32Array::from(v.as_ref().clone())),
            NumericColumn::F64(v) => Arc::new(Float64Array::from(v.as_ref().clone())),
        };
        columns.push(column(&name, data));
        names.push(name);
    }
    Ok(PyFeatureTable {
        table: StructArray::from(columns),
        names,
        keys,
        profile: frame.profile(),
    })
}
