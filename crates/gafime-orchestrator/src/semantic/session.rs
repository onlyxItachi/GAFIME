use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

use super::{
    next_identity, CandidateRegistry, EvidenceChannel, EvidenceDefinition, EvidenceRecord,
    EvidenceTable, EvidenceValue, FeatureFrame, FeatureId, FeatureOp, NumericColumn,
    SelectionPolicy, SemanticError, SemanticResult,
};

/// Context-bound values, distinct from durable candidate programs. Only native
/// execution constructs these; this internal safe interface validates shape and
/// ownership, but cannot certify the arithmetic of an arbitrary executor.
#[derive(Clone, Debug)]
pub struct MaterializedColumns {
    frame_id: u64,
    profile: PrecisionProfile,
    columns: BTreeMap<FeatureId, NumericColumn>,
}

impl MaterializedColumns {
    pub fn from_columns(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        columns: BTreeMap<FeatureId, NumericColumn>,
    ) -> SemanticResult<Self> {
        if registry.schema() != frame.schema() || registry.precision() != frame.profile() {
            return Err(SemanticError::Invalid("materialization schema mismatch"));
        }
        for (&id, values) in &columns {
            registry.program(id)?;
            if values.len() != frame.rows()
                || !values.finite()
                || !values.supports_profile(frame.profile())
            {
                return Err(SemanticError::Invalid(
                    "materialized values must be finite and row-aligned",
                ));
            }
        }
        Ok(Self {
            frame_id: frame.id(),
            profile: frame.profile(),
            columns,
        })
    }
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub fn get(&self, id: FeatureId) -> SemanticResult<&[f32]> {
        self.get_typed(id)?.as_f32()
    }
    pub fn get_typed(&self, id: FeatureId) -> SemanticResult<&NumericColumn> {
        self.columns
            .get(&id)
            .ok_or(SemanticError::Invalid("feature is not materialized"))
    }
    pub fn profile(&self) -> PrecisionProfile {
        self.profile
    }
    pub fn bytes(&self) -> usize {
        self.columns
            .values()
            .fold(0, |sum, column| sum.saturating_add(column.bytes()))
    }
    pub fn columns(&self) -> &BTreeMap<FeatureId, NumericColumn> {
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
        max_bytes: usize,
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

/// Caller-selected resource ceilings, not heuristic unlimited cache growth.
/// Budgets cover native working banks and retained columns owned by this session;
/// immutable input frames and caller-retained tables/acceptance records are separate.
#[derive(Clone, Copy, Debug)]
pub struct SessionLimits {
    pub max_bytes: usize,
    pub max_retained_bytes: usize,
    /// Structural candidate-row/edge units across dependencies and channels.
    /// This is admission control, not actual kernel visits, FLOPs or elapsed time.
    pub max_work: usize,
    pub max_rounds: u64,
}

impl SessionLimits {
    pub fn for_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            max_retained_bytes: max_bytes / 4,
            max_work: 128_000_000,
            max_rounds: 64,
        }
    }
    fn validate(self) -> SemanticResult<Self> {
        if self.max_bytes == 0
            || self.max_bytes > 512 * 1024 * 1024
            || self.max_retained_bytes > self.max_bytes / 2
            || self.max_work == 0
            || self.max_work > 1_000_000_000
            || self.max_rounds == 0
            || self.max_rounds > 1024
        {
            return Err(SemanticError::Invalid(
                "invalid semantic session resource limits",
            ));
        }
        Ok(self)
    }
}

/// Exclusive declaration scope for a discovery round. Only raw sources,
/// explicitly supplied accepted atoms, and programs constructed in this round
/// can be operands. Acceptance is not a decorative record beside a mutable registry.
pub struct DiscoveryRound<'a> {
    registry: &'a mut CandidateRegistry,
    eligible: &'a mut BTreeSet<FeatureId>,
}

impl DiscoveryRound<'_> {
    fn operand(&self, id: FeatureId) -> SemanticResult<()> {
        self.registry.program(id)?;
        if !self.eligible.contains(&id) {
            return Err(SemanticError::Invalid(
                "operand is not an eligible atom in this round",
            ));
        }
        Ok(())
    }
    pub fn source(&self, index: usize) -> SemanticResult<FeatureId> {
        self.registry.source(index)
    }
    pub fn abs_difference(&mut self, a: FeatureId, b: FeatureId) -> SemanticResult<FeatureId> {
        self.operand(a)?;
        self.operand(b)?;
        let id = self.registry.abs_difference(a, b)?;
        self.eligible.insert(id);
        Ok(id)
    }
    pub fn softsign(&mut self, a: FeatureId) -> SemanticResult<FeatureId> {
        self.operand(a)?;
        let id = self.registry.softsign(a)?;
        self.eligible.insert(id);
        Ok(id)
    }
    pub fn centered_product(
        &mut self,
        operands: Vec<FeatureId>,
        means: Vec<f32>,
    ) -> SemanticResult<FeatureId> {
        for &id in &operands {
            self.operand(id)?;
        }
        let id = self.registry.centered_product(operands, means)?;
        self.eligible.insert(id);
        Ok(id)
    }
    pub fn centered_product_f64(
        &mut self,
        operands: Vec<FeatureId>,
        means: Vec<f64>,
    ) -> SemanticResult<FeatureId> {
        for &id in &operands {
            self.operand(id)?;
        }
        let id = self.registry.centered_product_f64(operands, means)?;
        self.eligible.insert(id);
        Ok(id)
    }
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
    limits: SessionLimits,
    round: u64,
    eligible: Option<BTreeSet<FeatureId>>,
}

impl SemanticSession {
    pub fn new(
        registry: CandidateRegistry,
        backend: u32,
        max_bytes: usize,
    ) -> SemanticResult<Self> {
        Self::with_limits(registry, backend, SessionLimits::for_budget(max_bytes))
    }
    pub fn with_limits(
        registry: CandidateRegistry,
        backend: u32,
        limits: SessionLimits,
    ) -> SemanticResult<Self> {
        if backend != GAFIME_BACKEND_CPU {
            return Err(SemanticError::Unsupported(
                "semantic evidence currently requires explicit Core; no GPU or auto fallback",
            ));
        }
        let limits = limits.validate()?;
        Ok(Self {
            id: next_identity()?,
            registry: Some(registry),
            cache: None,
            limits,
            round: 0,
            eligible: None,
        })
    }
    pub fn registry(&self) -> SemanticResult<&CandidateRegistry> {
        self.registry.as_ref().ok_or(SemanticError::Closed)
    }
    pub fn begin_round(
        &mut self,
        accepted: &[AcceptedFeature],
    ) -> SemanticResult<DiscoveryRound<'_>> {
        let registry = self.registry()?;
        if self.round >= self.limits.max_rounds || accepted.len() > registry.limits().max_nodes {
            return Err(SemanticError::Invalid(
                "discovery round resource limit exceeded",
            ));
        }
        if accepted.iter().any(|a| a.owner != self.id) {
            return Err(SemanticError::ForeignIdentity);
        }
        let mut eligible = BTreeSet::new();
        for index in 0..registry.schema().len() {
            eligible.insert(registry.source(index)?);
        }
        for a in accepted {
            registry.program(a.feature)?;
            eligible.insert(a.feature);
        }
        self.round += 1;
        self.eligible = Some(eligible);
        Ok(DiscoveryRound {
            registry: self.registry.as_mut().ok_or(SemanticError::Closed)?,
            eligible: self.eligible.as_mut().expect("round initialized"),
        })
    }
    pub fn round(&self) -> u64 {
        self.round
    }
    pub fn retained_bytes(&self) -> usize {
        self.cache.as_ref().map_or(0, MaterializedColumns::bytes)
    }
    pub fn clear_materializations(&mut self) -> SemanticResult<()> {
        self.registry()?;
        self.cache = None;
        Ok(())
    }
    fn eligible(&self, id: FeatureId) -> SemanticResult<()> {
        self.registry()?.program(id)?;
        if self.eligible.as_ref().is_none_or(|set| !set.contains(&id)) {
            return Err(SemanticError::Invalid(
                "candidate is not eligible in the current round",
            ));
        }
        Ok(())
    }
    fn switch_context(&mut self, frame: &FeatureFrame) {
        if self
            .cache
            .as_ref()
            .is_some_and(|c| c.frame_id != frame.id())
        {
            self.cache = None;
        }
    }
    pub fn close(&mut self) {
        self.cache = None;
        self.registry = None;
        self.eligible = None;
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
            || frame.profile() != registry.precision()
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
            self.eligible(candidate)?;
        }
        for channel in channels {
            channel.definition().validate(registry, &frame)?;
            if let EvidenceDefinition::Redundancy { reference } = channel.definition() {
                self.eligible(*reference)?;
                roots.push(*reference);
            }
        }
        roots.sort();
        roots.dedup();
        let mut work = dependency_work(registry, frame.rows(), &roots)?;
        for (index, channel) in channels.iter().enumerate() {
            if channels[..index].iter().any(|old| old.same_work(channel)) {
                continue;
            }
            let rows = match channel.definition() {
                EvidenceDefinition::GraphEnergy { graph } => graph.edges().len(),
                EvidenceDefinition::LabeledAssociation {
                    labels: Some(labels),
                } => labels.rows().len(),
                EvidenceDefinition::LabeledAssociation { labels: None } => 0,
                EvidenceDefinition::PairedConsistency { view } => {
                    work = work
                        .checked_add(dependency_work(registry, view.rows(), &candidates)?)
                        .ok_or(SemanticError::Invalid("semantic work count overflow"))?;
                    frame.rows()
                }
                _ => frame.rows(),
            };
            work = work
                .checked_add(
                    rows.checked_mul(candidates.len())
                        .ok_or(SemanticError::Invalid("semantic work count overflow"))?,
                )
                .ok_or(SemanticError::Invalid("semantic work count overflow"))?;
        }
        if work > self.limits.max_work {
            return Err(SemanticError::Invalid(
                "semantic evaluation work limit exceeded",
            ));
        }
        self.switch_context(&frame);
        let registry = self.registry()?;
        let retained = self.cache.as_ref().filter(|c| c.frame_id == frame.id());
        // Reserve half for the paired view; only one such view lives at a time.
        // The native budget includes its dependency bank and worker allocations.
        let budget = (self.limits.max_bytes - self.retained_bytes()) / 2;
        let materialized = executor.materialize(registry, &frame, &roots, retained, budget)?;
        validate_output(registry, &frame, &roots, &materialized, budget)?;
        let mut channel_values: Vec<Vec<EvidenceValue>> = Vec::with_capacity(channels.len());
        for (index, channel) in channels.iter().enumerate() {
            if let Some(previous) = channels[..index]
                .iter()
                .position(|old| old.same_work(channel))
            {
                channel_values.push(channel_values[previous].clone());
                continue;
            }
            let paired =
                if let EvidenceDefinition::PairedConsistency { view } = channel.definition() {
                    let result = executor.materialize(registry, view, &candidates, None, budget)?;
                    validate_output(registry, view, &candidates, &result, budget)?;
                    Some(result)
                } else {
                    None
                };
            let values = executor.evaluate_channel(
                channel.definition(),
                &candidates,
                &materialized,
                paired.as_ref(),
                self.limits
                    .max_bytes
                    .checked_sub(self.retained_bytes())
                    .and_then(|n| n.checked_sub(materialized.bytes()))
                    .and_then(|n| {
                        n.checked_sub(paired.as_ref().map_or(0, MaterializedColumns::bytes))
                    })
                    .ok_or(SemanticError::Invalid(
                        "native materialization exceeds session budget",
                    ))?,
            )?;
            if values.len() != candidates.len()
                || values
                    .iter()
                    .any(|v| matches!(v, EvidenceValue::Measured{value,..} if !value.is_finite() || (frame.profile() == PrecisionProfile::Fp32 && f64::from(*value as f32) != *value)))
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
            round: self.round,
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
        if table.round != self.round {
            return Err(SemanticError::Invalid(
                "cannot accept evidence from a previous discovery round",
            ));
        }
        let selected = policy.select(table)?;
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let mut retained = self
            .cache
            .as_ref()
            .filter(|c| c.frame_id == table.frame.id())
            .map_or_else(BTreeMap::new, |c| c.columns.clone());
        for &feature in &selected {
            retained.insert(feature, table.materialized.get_typed(feature)?.clone());
        }
        let retained_bytes = retained
            .values()
            .fold(0usize, |sum, column| sum.saturating_add(column.bytes()));
        if retained_bytes > self.limits.max_retained_bytes {
            return Err(SemanticError::Invalid("accepted materialization retention limit exceeded; clear retained values explicitly"));
        }
        let mut accepted = Vec::with_capacity(selected.len());
        for feature in selected {
            let row = table
                .candidates
                .binary_search(&feature)
                .map_err(|_| SemanticError::ForeignIdentity)?;
            let start = row * table.channels.len();
            accepted.push(AcceptedFeature {
                owner: self.id,
                feature,
                evaluation: table.id,
                frame: Arc::clone(&table.frame),
                policy: policy.clone(),
                channels: table.channels.clone(),
                evidence: table.records[start..start + table.channels.len()].to_vec(),
            });
        }
        self.cache = Some(MaterializedColumns {
            frame_id: table.frame.id(),
            profile: table.frame.profile(),
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
        if frame.schema() != self.registry()?.schema()
            || frame.profile() != self.registry()?.precision()
        {
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
        if dependency_work(self.registry()?, frame.rows(), &ids)? > self.limits.max_work {
            return Err(SemanticError::Invalid(
                "semantic inference work limit exceeded",
            ));
        }
        self.switch_context(frame);
        let retained = self.cache.as_ref().filter(|c| c.frame_id == frame.id());
        let result = executor.materialize(
            self.registry()?,
            frame,
            &ids,
            retained,
            self.limits.max_bytes - self.retained_bytes(),
        )?;
        validate_output(
            self.registry()?,
            frame,
            &ids,
            &result,
            self.limits.max_bytes - self.retained_bytes(),
        )?;
        let mut columns = self
            .cache
            .as_ref()
            .map_or_else(BTreeMap::new, |cache| cache.columns.clone());
        columns.extend(result.columns.clone());
        let cache = MaterializedColumns {
            frame_id: frame.id(),
            profile: frame.profile(),
            columns,
        };
        if cache.bytes() > self.limits.max_retained_bytes {
            return Err(SemanticError::Invalid(
                "accepted inference retention limit exceeded",
            ));
        }
        self.cache = Some(cache);
        Ok(result)
    }
}

fn validate_output(
    registry: &CandidateRegistry,
    frame: &FeatureFrame,
    roots: &[FeatureId],
    output: &MaterializedColumns,
    budget: usize,
) -> SemanticResult<()> {
    if output.frame_id != frame.id() || output.profile != frame.profile() || output.bytes() > budget
    {
        return Err(SemanticError::Invalid("native output context mismatch"));
    }
    for &root in roots {
        registry.program(root)?;
        output.get_typed(root)?;
    }
    Ok(())
}

fn dependency_work(
    registry: &CandidateRegistry,
    rows: usize,
    roots: &[FeatureId],
) -> SemanticResult<usize> {
    let mut visited = BTreeSet::new();
    let mut pending = roots.to_vec();
    let mut units = 0usize;
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let program = registry.program(id)?;
        units = units
            .checked_add(program.logical_arity())
            .ok_or(SemanticError::Invalid("semantic work count overflow"))?;
        match program.op() {
            FeatureOp::Source(_) => {}
            FeatureOp::AbsoluteDifference(a, b) => pending.extend([*a, *b]),
            FeatureOp::Softsign(a) => pending.push(*a),
            FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
        }
    }
    units
        .checked_mul(rows)
        .ok_or(SemanticError::Invalid("semantic work count overflow"))
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
            _: usize,
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

    #[test]
    fn prebuilt_programs_require_explicit_round_declaration() {
        let mut registry = CandidateRegistry::new(
            vec!["a".into()],
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let source = registry.source(0).unwrap();
        let prebuilt = registry.softsign(source).unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1024).unwrap();
        assert!(session.eligible(prebuilt).is_err());
        {
            let mut round = session.begin_round(&[]).unwrap();
            assert!(round.softsign(prebuilt).is_err());
            assert_eq!(
                round.softsign(source).unwrap(),
                prebuilt,
                "canonical redeclaration is legitimate current-round authority"
            );
        }
        assert!(session.eligible(prebuilt).is_ok());
        session.begin_round(&[]).unwrap();
        assert!(session.eligible(prebuilt).is_err());
    }

    #[test]
    fn executor_output_cannot_expand_the_admitted_bank_budget() {
        let registry = CandidateRegistry::new(
            vec!["a".into()],
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let source = registry.source(0).unwrap();
        let frame = FeatureFrame::new(
            vec!["a".into()],
            "rows".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "test".into(),
            vec![vec![0.0, 1.0]],
        )
        .unwrap();
        let output = MaterializedColumns::from_columns(
            &registry,
            &frame,
            BTreeMap::from([(source, NumericColumn::from(vec![0.0f32, 1.0]))]),
        )
        .unwrap();
        assert!(validate_output(&registry, &frame, &[source], &output, 7).is_err());
        assert!(validate_output(&registry, &frame, &[source], &output, 8).is_ok());
    }
}
