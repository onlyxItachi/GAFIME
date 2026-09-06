use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use gafime_types::GAFIME_BACKEND_CPU;

use super::{
    next_identity, CandidateRegistry, EvidenceChannel, EvidenceDefinition, EvidenceRecord,
    EvidenceTable, EvidenceValue, FeatureFrame, FeatureId, SelectionPolicy, SemanticError,
    SemanticResult,
};

/// Context-bound values, distinct from durable candidate programs. Only native
/// execution constructs these; this internal safe interface validates shape and
/// ownership, but cannot certify the arithmetic of an arbitrary executor.
#[derive(Clone, Debug)]
pub struct MaterializedColumns {
    frame_id: u64,
    columns: BTreeMap<FeatureId, Arc<[f32]>>,
}

impl MaterializedColumns {
    pub fn from_columns(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        columns: BTreeMap<FeatureId, Arc<[f32]>>,
    ) -> SemanticResult<Self> {
        if registry.schema() != frame.schema() {
            return Err(SemanticError::Invalid("materialization schema mismatch"));
        }
        for (&id, values) in &columns {
            registry.program(id)?;
            if values.len() != frame.rows() || values.iter().any(|v| !v.is_finite()) {
                return Err(SemanticError::Invalid(
                    "materialized values must be finite and row-aligned",
                ));
            }
        }
        Ok(Self {
            frame_id: frame.id(),
            columns,
        })
    }
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub fn get(&self, id: FeatureId) -> SemanticResult<&[f32]> {
        self.columns
            .get(&id)
            .map(AsRef::as_ref)
            .ok_or(SemanticError::Invalid("feature is not materialized"))
    }
    pub fn columns(&self) -> &BTreeMap<FeatureId, Arc<[f32]>> {
        &self.columns
    }
}

/// Native kernels own arithmetic and candidate-level parallelism. The
/// orchestrator owns validation, dependency/context planning and selection.
/// Session validation precedes these lowering calls. This Rust interface is
/// not an independent user input boundary, C ABI or serialization format.
pub trait NativeEvidenceExecutor {
    fn backend_kind(&self) -> u32;
    fn materialize(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        candidates: &[FeatureId],
        retained: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns>;
    fn evaluate_channel(
        &mut self,
        definition: &EvidenceDefinition,
        candidates: &[FeatureId],
        values: &MaterializedColumns,
        paired: Option<&MaterializedColumns>,
    ) -> SemanticResult<Vec<EvidenceValue>>;
}

/// A recorded decision, not a promise of usefulness on another dataset. The
/// program remains in the session registry; this record preserves which input,
/// channels and explicit policy admitted it. Contexts remain alive with it.
#[derive(Clone, Debug)]
pub struct AcceptedFeature {
    owner: u64,
    feature: FeatureId,
    evaluation: u64,
    frame: Arc<FeatureFrame>,
    policy: SelectionPolicy,
    channels: Vec<EvidenceChannel>,
    evidence: Vec<EvidenceRecord>,
}

impl AcceptedFeature {
    pub fn feature(&self) -> FeatureId {
        self.feature
    }
    pub fn evaluation(&self) -> u64 {
        self.evaluation
    }
    pub fn frame(&self) -> &FeatureFrame {
        &self.frame
    }
    pub fn policy(&self) -> &SelectionPolicy {
        &self.policy
    }
    pub fn channels(&self) -> &[EvidenceChannel] {
        &self.channels
    }
    pub fn evidence(&self) -> &[EvidenceRecord] {
        &self.evidence
    }
}

/// Bounded internal discovery lifecycle. One context of accepted values is
/// retained; switching input contexts drops that cache, not the programs.
/// Evidence is always re-evaluated. Close releases owned state and fails closed.
pub struct SemanticSession {
    id: u64,
    registry: Option<CandidateRegistry>,
    cache: Option<MaterializedColumns>,
    max_bytes: usize,
}

impl SemanticSession {
    pub fn new(
        registry: CandidateRegistry,
        backend: u32,
        max_bytes: usize,
    ) -> SemanticResult<Self> {
        if backend != GAFIME_BACKEND_CPU {
            return Err(SemanticError::Unsupported(
                "semantic evidence currently requires explicit Core; no GPU or auto fallback",
            ));
        }
        if max_bytes == 0 || max_bytes > 512 * 1024 * 1024 {
            return Err(SemanticError::Invalid(
                "semantic execution budget must be 1..=512 MiB",
            ));
        }
        Ok(Self {
            id: next_identity()?,
            registry: Some(registry),
            cache: None,
            max_bytes,
        })
    }
    pub fn registry(&self) -> SemanticResult<&CandidateRegistry> {
        self.registry.as_ref().ok_or(SemanticError::Closed)
    }
    pub fn registry_mut(&mut self) -> SemanticResult<&mut CandidateRegistry> {
        self.registry.as_mut().ok_or(SemanticError::Closed)
    }
    pub fn close(&mut self) {
        self.cache = None;
        self.registry = None;
    }

    fn validate_executor(&self, executor: &impl NativeEvidenceExecutor) -> SemanticResult<()> {
        self.registry()?;
        if executor.backend_kind() != GAFIME_BACKEND_CPU {
            return Err(SemanticError::Unsupported(
                "semantic executor does not match explicit Core selection",
            ));
        }
        Ok(())
    }

    pub fn evaluate(
        &mut self,
        executor: &mut impl NativeEvidenceExecutor,
        frame: Arc<FeatureFrame>,
        candidates: &[FeatureId],
        channels: &[EvidenceChannel],
    ) -> SemanticResult<EvidenceTable> {
        self.validate_executor(executor)?;
        let registry = self.registry()?;
        if frame.schema() != registry.schema()
            || candidates.is_empty()
            || channels.is_empty()
            || channels.len() > 32
            || candidates
                .len()
                .checked_mul(channels.len())
                .is_none_or(|n| n > 65_536)
        {
            return Err(SemanticError::Invalid(
                "invalid or oversized semantic evaluation",
            ));
        }
        let ordered: BTreeSet<_> = candidates.iter().copied().collect();
        let ids: BTreeSet<_> = channels.iter().map(EvidenceChannel::id).collect();
        let names: BTreeSet<_> = channels.iter().map(EvidenceChannel::name).collect();
        if ordered.len() != candidates.len()
            || ids.len() != channels.len()
            || names.len() != channels.len()
        {
            return Err(SemanticError::Invalid(
                "duplicate candidate or evidence channel",
            ));
        }
        let candidates: Vec<_> = ordered.into_iter().collect();
        let mut roots = candidates.clone();
        for &candidate in &candidates {
            registry.program(candidate)?;
        }
        for channel in channels {
            channel.definition().validate(registry, &frame)?;
            if let EvidenceDefinition::Redundancy { reference } = channel.definition() {
                roots.push(*reference);
            }
        }
        roots.sort();
        roots.dedup();
        let retained = self.cache.as_ref().filter(|c| c.frame_id == frame.id());
        // Reserve half for the paired view; only one such view lives at a time.
        // The native budget includes its dependency bank and worker allocations.
        let budget = self.max_bytes / 2;
        let materialized = executor.materialize(registry, &frame, &roots, retained, budget)?;
        validate_output(registry, &frame, &roots, &materialized)?;
        let mut channel_values = Vec::with_capacity(channels.len());
        for channel in channels {
            let paired =
                if let EvidenceDefinition::PairedConsistency { view } = channel.definition() {
                    let result = executor.materialize(registry, view, &candidates, None, budget)?;
                    validate_output(registry, view, &candidates, &result)?;
                    Some(result)
                } else {
                    None
                };
            let values = executor.evaluate_channel(
                channel.definition(),
                &candidates,
                &materialized,
                paired.as_ref(),
            )?;
            if values.len() != candidates.len()
                || values
                    .iter()
                    .any(|v| matches!(v, EvidenceValue::Measured{value,..} if !value.is_finite()))
            {
                return Err(SemanticError::Invalid(
                    "native evidence output violates shape or finiteness",
                ));
            }
            channel_values.push(values);
        }
        let mut records = Vec::with_capacity(candidates.len() * channels.len());
        for (row, &candidate) in candidates.iter().enumerate() {
            for (col, channel) in channels.iter().enumerate() {
                records.push(EvidenceRecord {
                    candidate,
                    channel: channel.id(),
                    value: channel_values[col][row],
                });
            }
        }
        Ok(EvidenceTable {
            owner: self.id,
            id: next_identity()?,
            frame,
            candidates,
            channels: channels.to_vec(),
            records,
            materialized,
        })
    }

    pub fn accept(
        &mut self,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
    ) -> SemanticResult<Vec<AcceptedFeature>> {
        self.registry()?;
        if table.owner != self.id {
            return Err(SemanticError::ForeignIdentity);
        }
        let selected = policy.select(table)?;
        let mut retained = BTreeMap::new();
        let mut accepted = Vec::with_capacity(selected.len());
        for feature in selected {
            retained.insert(feature, Arc::clone(&table.materialized.columns[&feature]));
            accepted.push(AcceptedFeature {
                owner: self.id,
                feature,
                evaluation: table.id,
                frame: Arc::clone(&table.frame),
                policy: policy.clone(),
                channels: table.channels.clone(),
                evidence: table
                    .records
                    .iter()
                    .filter(|r| r.candidate == feature)
                    .cloned()
                    .collect(),
            });
        }
        self.cache = Some(MaterializedColumns {
            frame_id: table.frame.id(),
            columns: retained,
        });
        Ok(accepted)
    }

    /// Execute frozen accepted programs on new rows without consulting labels
    /// or reapplying their discovery policy. Same-context columns may be reused.
    pub fn materialize_accepted(
        &mut self,
        executor: &mut impl NativeEvidenceExecutor,
        frame: &FeatureFrame,
        accepted: &[AcceptedFeature],
    ) -> SemanticResult<MaterializedColumns> {
        self.validate_executor(executor)?;
        if frame.schema() != self.registry()?.schema() {
            return Err(SemanticError::Invalid(
                "accepted program input schema mismatch",
            ));
        }
        if accepted.len() > self.registry()?.limits().max_nodes
            || accepted.iter().any(|a| a.owner != self.id)
        {
            return Err(SemanticError::ForeignIdentity);
        }
        let ids: BTreeSet<_> = accepted.iter().map(AcceptedFeature::feature).collect();
        let ids: Vec<_> = ids.into_iter().collect();
        let retained = self.cache.as_ref().filter(|c| c.frame_id == frame.id());
        let result =
            executor.materialize(self.registry()?, frame, &ids, retained, self.max_bytes)?;
        validate_output(self.registry()?, frame, &ids, &result)?;
        self.cache = Some(result.clone());
        Ok(result)
    }
}

fn validate_output(
    registry: &CandidateRegistry,
    frame: &FeatureFrame,
    roots: &[FeatureId],
    output: &MaterializedColumns,
) -> SemanticResult<()> {
    if output.frame_id != frame.id() {
        return Err(SemanticError::Invalid("native output context mismatch"));
    }
    for &root in roots {
        registry.program(root)?;
        output.get(root)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::{EvaluationRole, ProgramLimits};
    use gafime_types::PrecisionProfile;

    struct MustNotExecute;
    impl NativeEvidenceExecutor for MustNotExecute {
        fn backend_kind(&self) -> u32 {
            GAFIME_BACKEND_CPU
        }
        fn materialize(
            &mut self,
            _: &CandidateRegistry,
            _: &FeatureFrame,
            _: &[FeatureId],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<MaterializedColumns> {
            panic!("orchestrator must reject incompatible schemas before native lowering")
        }
        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("unexpected evidence execution")
        }
    }

    #[test]
    fn inference_schema_gate_is_owned_by_orchestrator_not_executor() {
        let registry = CandidateRegistry::new(
            vec!["a".into()],
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1024).unwrap();
        let different = FeatureFrame::new(
            vec!["b".into()],
            "new".into(),
            vec![0, 1],
            EvaluationRole::Inference,
            "new rows".into(),
            vec![vec![1.0, 2.0]],
        )
        .unwrap();
        assert_eq!(
            session
                .materialize_accepted(&mut MustNotExecute, &different, &[])
                .unwrap_err(),
            SemanticError::Invalid("accepted program input schema mismatch")
        );
    }
}
