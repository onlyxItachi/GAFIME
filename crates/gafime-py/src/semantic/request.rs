//! Bounded public requests delegate meaning to the orchestrator's closed keys.
use super::input::{PyGraph, PyLabels, PySnapshot};
use super::{error, PyCandidate};
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, Direction, EvidenceChannel, EvidenceConstraint,
    EvidenceDefinition, MissingEvidence, SelectionPolicy,
};
use pyo3::{exceptions::PyValueError, prelude::*};

fn statistic(name: &str, bins: Option<u32>) -> PyResult<AssociationStatistic> {
    match (name, bins) {
        ("pearson", None) => Ok(AssociationStatistic::Pearson),
        ("spearman", None) => Ok(AssociationStatistic::Spearman),
        ("fixed_nmi", Some(bins)) => Ok(AssociationStatistic::FixedCorrectedNmi { bins }),
        _ => Err(PyValueError::new_err("statistic must be pearson or spearman without bins, or fixed_nmi with an exact supported bins value")),
    }
}
fn missing(name: &str, optional: bool) -> PyResult<MissingEvidence> {
    match name {
        "reject" => Ok(MissingEvidence::RejectCandidate),
        "error" => Ok(MissingEvidence::Error),
        "ignore" if optional => Ok(MissingEvidence::IgnoreConstraint),
        _ => Err(PyValueError::new_err(
            "missing must be reject/error (or ignore for an optional constraint)",
        )),
    }
}

/// Immutable, named measurement and contextual binding. Constructors distinguish
/// reference, aligned view, actual labels and graph inputs. Correlations are
/// absolute for reference/labels and signed for paired views. Fixed NMI measures
/// dependence, not invariance; bins are exact, not EngineConfig.mi_bins maxima.
/// No evidence value is a calibrated universal feature quality or p-value.
#[pyclass(name = "Evidence", module = "gafime.semantic", frozen)]
pub(crate) struct PyEvidence {
    pub(super) channel: EvidenceChannel,
}
impl PyEvidence {
    fn association(
        name: &str,
        context: AssociationContext,
        kind: &str,
        bins: Option<u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            channel: EvidenceChannel::new(
                super::input::bounded_text(name, 256, "evidence name")?,
                EvidenceDefinition::Association {
                    statistic: statistic(kind, bins)?,
                    context,
                },
            )
            .map_err(error)?,
        })
    }
    fn rebind(&self, context: AssociationContext) -> PyResult<Self> {
        let EvidenceDefinition::Association { statistic, .. } = self.channel.definition() else {
            return Err(PyValueError::new_err("cannot rebind graph as association"));
        };
        Ok(Self {
            channel: self
                .channel
                .rebind(EvidenceDefinition::Association {
                    statistic: *statistic,
                    context,
                })
                .map_err(error)?,
        })
    }
}
#[pymethods]
impl PyEvidence {
    /// Ask for association with an explicit reference feature, never a label.
    #[staticmethod]
    #[pyo3(signature=(name, candidate, *, statistic="pearson", bins=None))]
    fn reference(
        name: &str,
        candidate: PyRef<'_, PyCandidate>,
        statistic: &str,
        bins: Option<u32>,
    ) -> PyResult<Self> {
        Self::association(
            name,
            AssociationContext::Reference {
                reference: candidate.id,
            },
            statistic,
            bins,
        )
    }
    /// Evaluate the same frozen program on an exactly aligned paired snapshot.
    /// Pearson/Spearman retain sign; fixed NMI is nonlinear dependence only.
    #[staticmethod]
    #[pyo3(signature=(name, view, *, statistic="pearson", bins=None))]
    fn paired(
        name: &str,
        view: PyRef<'_, PySnapshot>,
        statistic: &str,
        bins: Option<u32>,
    ) -> PyResult<Self> {
        Self::association(
            name,
            AssociationContext::PairedView {
                view: view.frame.clone(),
            },
            statistic,
            bins,
        )
    }
    /// Bind actual optionally partial labels. None means MissingLabels evidence,
    /// not a zero vector or an automatically generated target.
    #[staticmethod]
    #[pyo3(signature=(name, labels=None, *, statistic="pearson", bins=None))]
    fn labels(
        name: &str,
        labels: Option<PyRef<'_, PyLabels>>,
        statistic: &str,
        bins: Option<u32>,
    ) -> PyResult<Self> {
        Self::association(
            name,
            AssociationContext::Labels {
                labels: labels.map(|x| x.labels.clone()),
            },
            statistic,
            bins,
        )
    }
    /// Ask the uncentered weighted edge-energy ratio. This is translation-sensitive
    /// and is not a centered Laplacian score or launch-graph capability.
    #[staticmethod]
    fn graph(name: &str, graph: PyRef<'_, PyGraph>) -> PyResult<Self> {
        Ok(Self {
            channel: EvidenceChannel::new(
                super::input::bounded_text(name, 256, "evidence name")?,
                EvidenceDefinition::GraphEnergy {
                    graph: graph.graph.clone(),
                },
            )
            .map_err(error)?,
        })
    }
    /// Rebind a reference while preserving the exact estimator and channel ID.
    fn rebind_reference(&self, candidate: PyRef<'_, PyCandidate>) -> PyResult<Self> {
        self.rebind(AssociationContext::Reference {
            reference: candidate.id,
        })
    }
    /// Rebind an aligned view without changing earlier immutable observations.
    fn rebind_paired(&self, view: PyRef<'_, PySnapshot>) -> PyResult<Self> {
        self.rebind(AssociationContext::PairedView {
            view: view.frame.clone(),
        })
    }
    /// Rebind the actual labeled subset; changed labels do not change programs.
    #[pyo3(signature=(labels=None))]
    fn rebind_labels(&self, labels: Option<PyRef<'_, PyLabels>>) -> PyResult<Self> {
        self.rebind(AssociationContext::Labels {
            labels: labels.map(|x| x.labels.clone()),
        })
    }
    /// Rebind graph context; changing a measurement kind is rejected.
    fn rebind_graph(&self, graph: PyRef<'_, PyGraph>) -> PyResult<Self> {
        Ok(Self {
            channel: self
                .channel
                .rebind(EvidenceDefinition::GraphEnergy {
                    graph: graph.graph.clone(),
                })
                .map_err(error)?,
        })
    }
    #[getter]
    fn name(&self) -> &str {
        self.channel.name()
    }
    #[getter]
    fn semantics(&self) -> &'static str {
        self.channel.definition().semantic_name()
    }
    #[getter]
    fn bins(&self) -> Option<u32> {
        match self.channel.definition() {
            EvidenceDefinition::Association { statistic, .. } => statistic.fixed_nmi_bins(),
            _ => None,
        }
    }
}

/// Inclusive bounds in one channel's units. missing=None inherits policy;
/// missing='ignore' explicitly permits absent optional evidence, never primary.
#[pyclass(name = "Constraint", module = "gafime.semantic", frozen)]
pub(crate) struct PyConstraint {
    constraint: EvidenceConstraint,
}
#[pymethods]
impl PyConstraint {
    #[new]
    #[pyo3(signature=(evidence, *, minimum=None, maximum=None, missing=None))]
    fn new(
        evidence: PyRef<'_, PyEvidence>,
        minimum: Option<f64>,
        maximum: Option<f64>,
        missing: Option<&str>,
    ) -> PyResult<Self> {
        if minimum.is_some_and(|x| !x.is_finite())
            || maximum.is_some_and(|x| !x.is_finite())
            || matches!((minimum,maximum),(Some(a),Some(b)) if a>b)
        {
            return Err(PyValueError::new_err(
                "constraint requires finite ordered bounds",
            ));
        }
        Ok(Self {
            constraint: EvidenceConstraint {
                channel: evidence.channel.id(),
                minimum,
                maximum,
                missing: missing.map(|s| self::missing(s, true)).transpose()?,
            },
        })
    }
}

/// Explicit primary-channel order plus constraints, not a weighted quality score.
/// direction is maximize/minimize; missing is reject/error. Ties are resolved
/// by canonical program order. limit may be zero. fp32 thresholds quantize to
/// the profile's result dtype; nonfinite/overflowed thresholds fail closed.
#[pyclass(name = "SelectionPolicy", module = "gafime.semantic", frozen)]
pub(crate) struct PySelectionPolicy {
    pub(super) policy: SelectionPolicy,
}
#[pymethods]
impl PySelectionPolicy {
    #[new]
    #[pyo3(signature=(primary, *, direction="maximize", limit=10, missing="reject", constraints=None))]
    fn new(
        primary: PyRef<'_, PyEvidence>,
        direction: &str,
        limit: usize,
        missing: &str,
        constraints: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let constraints = match constraints {
            None => Vec::new(),
            Some(value) => super::input::bounded_items(value, 32, "policy constraints")?
                .map(|c| Ok(c?.extract::<PyRef<'_, PyConstraint>>()?.constraint.clone()))
                .collect::<PyResult<Vec<_>>>()?,
        };
        if limit > 65_536 {
            return Err(PyValueError::new_err(
                "policy exceeds bounded constraints/result limit",
            ));
        }
        let direction = match direction {
            "maximize" => Direction::Maximize,
            "minimize" => Direction::Minimize,
            _ => {
                return Err(PyValueError::new_err(
                    "direction must be maximize or minimize",
                ))
            }
        };
        Ok(Self {
            policy: SelectionPolicy {
                primary: primary.channel.id(),
                direction,
                limit,
                missing: self::missing(missing, false)?,
                constraints,
            },
        })
    }
}
