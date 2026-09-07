//! Public declarations are translated once into the canonical Rust semantic
//! objects. No Python callback, candidate evaluator or policy implementation
//! exists on this boundary. Handles deliberately have no integer constructor.

mod execution;
mod input;
mod output;
mod request;

use execution::TabularExecutor;
use gafime_orchestrator::semantic::{
    AcceptedFeature, CandidateRegistry, EvidenceTable, FeatureId, FeatureOp, FrozenMeans,
    ProgramLimits, ProposalOperator, SemanticError, SemanticSession, SessionLimits,
};
use gafime_types::PrecisionProfile;
use input::{PyGraph, PyLabels, PySnapshot};
use output::{PyEvidenceReport, PyFeatureTable};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
};
use request::{PyConstraint, PyEvidence, PySelectionPolicy};
use std::sync::Arc;

fn error(value: SemanticError) -> PyErr {
    match value {
        SemanticError::Closed => PyRuntimeError::new_err(value.to_string()),
        SemanticError::Unsupported(_) => {
            pyo3::exceptions::PyNotImplementedError::new_err(value.to_string())
        }
        _ => PyValueError::new_err(value.to_string()),
    }
}

/// Immutable session-local program handle. Equality means the same canonical
/// program in the same session, not equal scores or a cross-process identity.
/// Obtain handles from a session or accepted/candidate set; do not serialize.
#[pyclass(name = "Candidate", module = "gafime.semantic", frozen, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCandidate {
    id: FeatureId,
}

/// Bounded native candidate collection in deterministic declaration order.
/// Indexing returns opaque handles; it does not execute candidate arithmetic.
#[pyclass(name = "CandidateSet", module = "gafime.semantic", frozen)]
pub(crate) struct PyCandidateSet {
    ids: Vec<FeatureId>,
}

#[pymethods]
impl PyCandidateSet {
    fn __len__(&self) -> usize {
        self.ids.len()
    }
    fn __getitem__(&self, index: isize) -> PyResult<PyCandidate> {
        Ok(PyCandidate {
            id: *self
                .ids
                .get(index_of(index, self.ids.len())?)
                .ok_or_else(|| PyIndexError::new_err("candidate index out of range"))?,
        })
    }
}

/// Immutable accepted-program collection with original contextual decisions.
/// Pass the complete set to begin_round to authorize later-round reuse. Values
/// are context-bound; transform on new snapshots never refits or consults labels.
#[pyclass(name = "AcceptedSet", module = "gafime.semantic", frozen)]
pub(crate) struct PyAcceptedSet {
    values: Vec<AcceptedFeature>,
}

#[pymethods]
impl PyAcceptedSet {
    fn __len__(&self) -> usize {
        self.values.len()
    }
    fn __getitem__(&self, index: isize) -> PyResult<PyCandidate> {
        Ok(PyCandidate {
            id: self.values[index_of(index, self.values.len())?].feature(),
        })
    }
}

fn index_of(index: isize, len: usize) -> PyResult<usize> {
    let index = if index < 0 {
        len as isize + index
    } else {
        index
    };
    if index < 0 || index as usize >= len {
        return Err(PyIndexError::new_err("index out of range"));
    }
    Ok(index as usize)
}

fn candidate_ids(value: &Bound<'_, PyAny>) -> PyResult<Vec<FeatureId>> {
    if let Ok(batch) = value.extract::<PyRef<'_, PyCandidateSet>>() {
        return Ok(batch.ids.clone());
    }
    if let Ok(batch) = value.extract::<PyRef<'_, PyAcceptedSet>>() {
        return Ok(batch.values.iter().map(AcceptedFeature::feature).collect());
    }
    input::bounded_items(value, 65_536, "candidate handles")?
        .map(|item| Ok(item?.extract::<PyRef<'_, PyCandidate>>()?.id))
        .collect()
}

/// Synchronous, thread-affine tabular discovery session with Rust-owned inputs.
///
/// config is an EngineConfig; only backend, device_id and precision apply here.
/// Evidence, proposal and selection are explicit, so supervised metric/family/
/// significance defaults are not inherited. NumPy buffers or one Arrow record
/// batch must have the exact profile storage dtype and finite non-null values.
/// row_keys, row_domain and provenance are required alignment declarations.
///
/// Resource ceilings bound numeric working/retained storage and structural work,
/// not process RSS. close is terminal/idempotent; exported Arrow outputs remain
/// owned independently. Session/program persistence and concurrent calls are
/// unsupported. Arithmetic runs in the selected native lowering, not in Python.
#[pyclass(name = "TabularSession", module = "gafime.semantic")]
pub(crate) struct PyTabularSession {
    session: SemanticSession,
    executor: Option<TabularExecutor>,
    frame: Option<Arc<gafime_orchestrator::semantic::FeatureFrame>>,
    profile: PrecisionProfile,
    configured_backend: String,
    owner_thread: std::thread::ThreadId,
    device_id: i32,
}

#[pymethods]
impl PyTabularSession {
    #[new]
    #[pyo3(signature=(data, *, row_keys, row_domain, provenance, config=None, feature_names=None, role="discovery", max_bytes=67108864, max_retained_bytes=None, max_work=128000000, max_rounds=64, max_nodes=65536, max_logical_arity=8, max_source_arity=8, max_depth=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data: &Bound<'_, PyAny>,
        row_keys: &Bound<'_, PyAny>,
        row_domain: &str,
        provenance: &str,
        config: Option<&Bound<'_, PyAny>>,
        feature_names: Option<&Bound<'_, PyAny>>,
        role: &str,
        max_bytes: usize,
        max_retained_bytes: Option<usize>,
        max_work: usize,
        max_rounds: u64,
        max_nodes: usize,
        max_logical_arity: usize,
        max_source_arity: usize,
        max_depth: usize,
    ) -> PyResult<Self> {
        let (backend, profile, device_id) = input::execution_intent(config)?;
        let executor = TabularExecutor::new(data.py(), &backend, profile, device_id)?;
        let frame = Arc::new(input::snapshot(
            data,
            profile,
            feature_names,
            row_keys,
            row_domain,
            role,
            provenance,
        )?);
        let registry = CandidateRegistry::new(
            frame.schema().to_vec(),
            profile,
            ProgramLimits {
                max_nodes,
                max_logical_arity,
                max_source_arity,
                max_depth,
            },
        )
        .map_err(error)?;
        let session = SemanticSession::with_limits(
            registry,
            executor.backend_kind(),
            SessionLimits {
                max_bytes,
                max_retained_bytes: max_retained_bytes.unwrap_or(max_bytes / 4),
                max_work,
                max_rounds,
            },
        )
        .map_err(error)?;
        Ok(Self {
            session,
            executor: Some(executor),
            frame: Some(frame),
            profile,
            configured_backend: backend,
            owner_thread: std::thread::current().id(),
            device_id,
        })
    }

    /// The immutable discovery snapshot owned at construction.
    #[getter]
    fn frame(&self) -> PyResult<PySnapshot> {
        Ok(PySnapshot {
            frame: self.active_frame()?.clone(),
        })
    }
    #[getter]
    fn configured_backend(&self) -> PyResult<&str> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        Ok(&self.configured_backend)
    }
    #[getter]
    fn selected_backend(&self) -> PyResult<&'static str> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        Ok(self.executor_ref()?.name())
    }
    #[getter]
    fn precision(&self) -> PyResult<&'static str> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        Ok(profile_name(self.profile))
    }
    #[getter]
    fn retained_bytes(&self) -> PyResult<usize> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        Ok(self.session.retained_bytes())
    }

    /// Operation-specific capability snapshot, separate from v1 supervised
    /// backend availability. Auto requires the complete declared vocabulary;
    /// selecting Core here is never reported as GPU execution.
    #[getter]
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        self.executor_ref()?.capabilities(
            py,
            &self.configured_backend,
            self.device_id,
            self.precision()?,
        )
    }

    /// Core exposes observed cumulative work, not modeled cache hits. GPU
    /// exposes residency with an explicit unavailable-work-counter marker.
    /// Input snapshots and explicit Arrow output copies are separate.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        self.executor_ref()?
            .diagnostics(py, self.session.retained_bytes())
    }

    /// Snapshot another frame in the same precision domain. Equal lengths do not
    /// establish pairing: schema, row keys/domain and role must align explicitly.
    #[pyo3(signature=(data, *, row_keys, row_domain, provenance, feature_names=None, role="inference"))]
    fn snapshot(
        &self,
        data: &Bound<'_, PyAny>,
        row_keys: &Bound<'_, PyAny>,
        row_domain: &str,
        provenance: &str,
        feature_names: Option<&Bound<'_, PyAny>>,
        role: &str,
    ) -> PyResult<PySnapshot> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        Ok(PySnapshot {
            frame: Arc::new(input::snapshot(
                data,
                self.profile,
                feature_names,
                row_keys,
                row_domain,
                role,
                provenance,
            )?),
        })
    }

    /// Start a bounded discovery round with sources and explicitly accepted atoms.
    /// accepted is one AcceptedSet or an indexed sequence of AcceptedSet batches;
    /// candidate IDs alone cannot manufacture acceptance authority. Old evaluation
    /// tables cannot be selected after advancing the round.
    #[pyo3(signature=(accepted=None))]
    fn begin_round(&mut self, accepted: Option<&Bound<'_, PyAny>>) -> PyResult<u64> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        let mut accepted_values = Vec::new();
        if let Some(value) = accepted {
            if let Ok(batch) = value.extract::<PyRef<'_, PyAcceptedSet>>() {
                accepted_values.clone_from(&batch.values);
            } else {
                for item in input::bounded_items(value, 65_536, "accepted batches")? {
                    let item = item?;
                    let batch = item.extract::<PyRef<'_, PyAcceptedSet>>()?;
                    if accepted_values.len().saturating_add(batch.values.len()) > 65_536 {
                        return Err(PyValueError::new_err(
                            "accepted union exceeds bounded feature limit",
                        ));
                    }
                    accepted_values.extend(batch.values.iter().cloned());
                }
            }
        }
        self.session.begin_round(&accepted_values).map_err(error)?;
        Ok(self.session.round())
    }
    /// Resolve an exact source-column name to its session-owned program.
    fn source(&self, name: &str) -> PyResult<PyCandidate> {
        self.check_thread()?;
        let r = self.session.registry().map_err(error)?;
        let index = r
            .schema()
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| PyValueError::new_err("unknown source feature"))?;
        Ok(PyCandidate {
            id: r.source(index).map_err(error)?,
        })
    }
    /// Inspect mathematical intent and transitive source dependencies. Returned
    /// handles remain process-local; this metadata is not a serialization ABI.
    fn describe<'py>(
        &self,
        py: Python<'py>,
        candidate: PyRef<'_, PyCandidate>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.check_thread()?;
        let registry = self.session.registry().map_err(error)?;
        let program = registry.program(candidate.id).map_err(error)?;
        let out = pyo3::types::PyDict::new(py);
        out.set_item("logical_arity", program.logical_arity())?;
        out.set_item("source_arity", program.source_arity())?;
        out.set_item("depth", program.depth())?;
        out.set_item(
            "sources",
            program
                .source_dependencies()
                .iter()
                .map(|i| registry.schema()[*i as usize].clone())
                .collect::<Vec<_>>(),
        )?;
        let (operation, operands) = match program.op() {
            FeatureOp::Source(_) => ("source", Vec::new()),
            FeatureOp::AbsoluteDifference(a, b) => ("absolute_difference", vec![*a, *b]),
            FeatureOp::Softsign(a) => ("softsign", vec![*a]),
            FeatureOp::CenteredProduct {
                operands,
                mean_bits,
            } => {
                let means: Vec<f64> = match mean_bits {
                    FrozenMeans::F32(bits) => {
                        bits.iter().map(|b| f64::from(f32::from_bits(*b))).collect()
                    }
                    FrozenMeans::F64(bits) => bits.iter().map(|b| f64::from_bits(*b)).collect(),
                };
                out.set_item("means", means)?;
                ("centered_product", operands.clone())
            }
        };
        out.set_item("operation", operation)?;
        out.set_item("operands", PyCandidateSet { ids: operands })?;
        out.set_item("precision", self.precision()?)?;
        Ok(out)
    }
    /// Declare abs(a-b) in the pointwise dtype. Requires an active round and
    /// eligible distinct operands; finite-input overflow fails evaluation closed.
    fn absolute_difference(
        &mut self,
        a: PyRef<'_, PyCandidate>,
        b: PyRef<'_, PyCandidate>,
    ) -> PyResult<PyCandidate> {
        self.check_thread()?;
        Ok(PyCandidate {
            id: self
                .session
                .current_round()
                .map_err(error)?
                .abs_difference(a.id, b.id)
                .map_err(error)?,
        })
    }
    /// Declare x/(1+abs(x)) in the selected pointwise dtype, without fitting.
    fn softsign(&mut self, operand: PyRef<'_, PyCandidate>) -> PyResult<PyCandidate> {
        self.check_thread()?;
        Ok(PyCandidate {
            id: self
                .session
                .current_round()
                .map_err(error)?
                .softsign(operand.id)
                .map_err(error)?,
        })
    }
    /// Declare an ordered product of (operand-mean), with explicit frozen means.
    /// Means become profile-native bits in identity; inference never recomputes
    /// them. Their fitting origin/leakage is the caller's responsibility.
    fn centered_product(
        &mut self,
        operands: &Bound<'_, PyAny>,
        means: &Bound<'_, PyAny>,
    ) -> PyResult<PyCandidate> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        let ids = candidate_ids(operands)?;
        let means: Vec<f64> = input::bounded_items(means, ids.len(), "frozen means")?
            .map(|x| x?.extract())
            .collect::<PyResult<_>>()?;
        let profile = self.profile;
        let mut round = self.session.current_round().map_err(error)?;
        let id = if profile == PrecisionProfile::Fp64 {
            round.centered_product_f64(ids, means)
        } else {
            round.centered_product(ids, means.into_iter().map(|x| x as f32).collect())
        }
        .map_err(error)?;
        Ok(PyCandidate { id })
    }
    /// Propose a deterministic bounded prefix using built-in operators in the
    /// requested order and canonical atom order. With atoms=None use raw sources;
    /// otherwise use explicit eligible handles, including accepted later-round
    /// atoms. Supported tokens: source, softsign, absolute_difference. Frozen
    /// centered products require manual means and are not automatically fitted.
    /// Invalid batches roll back new declarations; no partial catalog escapes.
    #[pyo3(signature=(operators, *, atoms=None, limit=256))]
    fn propose(
        &mut self,
        operators: &Bound<'_, PyAny>,
        atoms: Option<&Bound<'_, PyAny>>,
        limit: usize,
    ) -> PyResult<PyCandidateSet> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        let operators: Vec<String> = input::bounded_items(operators, 3, "proposal operators")?
            .map(|x| {
                let x = x?;
                input::bounded_text(x.extract::<&str>()?, 32, "proposal operator")
            })
            .collect::<PyResult<_>>()?;
        let operators: PyResult<Vec<_>> = operators
            .iter()
            .map(|op| match op.as_str() {
                "source" => Ok(ProposalOperator::Source),
                "softsign" => Ok(ProposalOperator::Softsign),
                "absolute_difference" => Ok(ProposalOperator::AbsoluteDifference),
                _ => Err(PyValueError::new_err("unknown built-in proposal operator")),
            })
            .collect();
        let atoms = match atoms {
            Some(value) => candidate_ids(value)?,
            None => {
                let r = self.session.registry().map_err(error)?;
                (0..r.schema().len())
                    .map(|i| r.source(i).map_err(error))
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        Ok(PyCandidateSet {
            ids: self
                .session
                .current_round()
                .map_err(error)?
                .propose(&operators?, &atoms, limit)
                .map_err(error)?,
        })
    }
    /// Evaluate explicit evidence against candidates on the discovery snapshot or
    /// a supplied compatible snapshot. No channel becomes an implicit target.
    #[pyo3(signature=(candidates, channels, *, frame=None))]
    fn evaluate(
        &mut self,
        candidates: &Bound<'_, PyAny>,
        channels: &Bound<'_, PyAny>,
        frame: Option<PyRef<'_, PySnapshot>>,
    ) -> PyResult<PyEvidenceReport> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        let ids = candidate_ids(candidates)?;
        let channels: Vec<_> = input::bounded_items(channels, 32, "evidence channels")?
            .map(|c| Ok(c?.extract::<PyRef<'_, PyEvidence>>()?.channel.clone()))
            .collect::<PyResult<_>>()?;
        let frame = match frame {
            Some(f) => f.frame.clone(),
            None => self.active_frame()?.clone(),
        };
        let table: EvidenceTable = self
            .session
            .evaluate(
                self.executor
                    .as_mut()
                    .ok_or_else(|| error(SemanticError::Closed))?
                    .native(),
                frame,
                &ids,
                &channels,
            )
            .map_err(error)?;
        Ok(PyEvidenceReport { table })
    }
    /// Apply the explicit canonical Rust policy and retain selected values within
    /// budget. Foreign-session or previous-round observations are rejected.
    fn select(
        &mut self,
        report: PyRef<'_, PyEvidenceReport>,
        policy: PyRef<'_, PySelectionPolicy>,
    ) -> PyResult<PyAcceptedSet> {
        self.check_thread()?;
        Ok(PyAcceptedSet {
            values: self
                .session
                .accept_with(
                    self.executor
                        .as_mut()
                        .ok_or_else(|| error(SemanticError::Closed))?
                        .native(),
                    &report.table,
                    &policy.policy,
                )
                .map_err(error)?,
        })
    }
    /// Execute accepted frozen programs on unlabeled rows. Output columns follow
    /// accepted-set order, carry row keys, and own their Arrow buffers after close.
    fn transform(
        &mut self,
        accepted: PyRef<'_, PyAcceptedSet>,
        frame: PyRef<'_, PySnapshot>,
    ) -> PyResult<PyFeatureTable> {
        self.check_thread()?;
        let values = self
            .session
            .materialize_accepted(
                self.executor
                    .as_mut()
                    .ok_or_else(|| error(SemanticError::Closed))?
                    .native(),
                &frame.frame,
                &accepted.values,
            )
            .map_err(error)?;
        if values.is_resident() {
            let host = self
                .session
                .download_materialization(
                    self.executor
                        .as_mut()
                        .ok_or_else(|| error(SemanticError::Closed))?
                        .native(),
                    &frame.frame,
                    &values,
                )
                .map_err(error)?;
            output::features(&host, &frame.frame, &accepted.values)
        } else {
            output::features(&values, &frame.frame, &accepted.values)
        }
    }
    /// Drop retained values without revoking accepted program identity.
    fn clear_materializations(&mut self) -> PyResult<()> {
        self.check_thread()?;
        self.session.clear_materializations().map_err(error)
    }
    /// Release session-owned programs and retained values; later work errors.
    fn close(&mut self) -> PyResult<()> {
        self.check_thread()?;
        self.session.close();
        self.frame = None;
        self.executor = None;
        Ok(())
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.check_thread()?;
        slf.session.registry().map_err(error)?;
        Ok(slf)
    }
    fn __exit__(
        &mut self,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close()
    }
}

impl PyTabularSession {
    fn executor_ref(&self) -> PyResult<&TabularExecutor> {
        self.executor
            .as_ref()
            .ok_or_else(|| error(SemanticError::Closed))
    }
    fn check_thread(&self) -> PyResult<()> {
        // PyO3's unsendable marker panics before a method can return an ordinary
        // Python exception. The owned Rust fields are Send + Sync; enforce this
        // public lifecycle policy explicitly without unsafe marker overrides.
        if std::thread::current().id() != self.owner_thread {
            return Err(PyRuntimeError::new_err(
                "TabularSession is thread-affine; use it on its creating thread",
            ));
        }
        Ok(())
    }
    fn active_frame(&self) -> PyResult<&Arc<gafime_orchestrator::semantic::FeatureFrame>> {
        self.check_thread()?;
        self.session.registry().map_err(error)?;
        self.frame
            .as_ref()
            .ok_or_else(|| error(SemanticError::Closed))
    }
}

fn profile_name(profile: PrecisionProfile) -> &'static str {
    match profile {
        PrecisionProfile::Fp32 => "fp32",
        PrecisionProfile::Mixed => "mixed",
        PrecisionProfile::Fp64 => "fp64",
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTabularSession>()?;
    m.add_class::<PySnapshot>()?;
    m.add_class::<PyCandidate>()?;
    m.add_class::<PyCandidateSet>()?;
    m.add_class::<PyAcceptedSet>()?;
    m.add_class::<PyLabels>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyEvidence>()?;
    m.add_class::<PyEvidenceReport>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PySelectionPolicy>()?;
    m.add_class::<PyFeatureTable>()?;
    Ok(())
}
