use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

use gafime_types::{
    BackendKind, PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM,
};

use super::{
    next_identity, CandidateRegistry, EvidenceChannel, EvidenceDefinition, EvidenceRecord,
    EvidenceTable, EvidenceValue, FeatureFrame, FeatureId, FeatureOp, FrozenMeans, NumericColumn,
    PredicateComparator, SelectionPolicy, SemanticError, SemanticResult, TrainingBinding,
};

/// Context-bound values, distinct from durable candidate programs. Only native
/// execution constructs these; this internal safe interface validates shape and
/// ownership, but cannot certify the arithmetic of an arbitrary executor.
#[derive(Clone)]
pub struct MaterializedColumns {
    frame_id: u64,
    profile: PrecisionProfile,
    backend: BackendKind,
    storage: MaterializedStorage,
}

/// An executor-owned, process-local residency lease.  The orchestrator never
/// interprets the object behind this handle: it only retains it with the
/// physical slot map and passes it back to the same backend executor.
pub type ResidentMaterializationLease = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
enum MaterializedStorage {
    Host(BTreeMap<FeatureId, NumericColumn>),
    Resident {
        slots: BTreeMap<FeatureId, u32>,
        bytes: usize,
        // An empty transform result has no physical bank.  It still carries
        // selected-backend/context identity rather than becoming fake Core
        // storage or forcing a dummy device allocation.
        lease: Option<ResidentMaterializationLease>,
    },
}

impl fmt::Debug for MaterializedColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = f.debug_struct("MaterializedColumns");
        output
            .field("frame_id", &self.frame_id)
            .field("profile", &self.profile)
            .field("backend", &self.backend);
        match &self.storage {
            MaterializedStorage::Host(columns) => {
                output.field("storage", &"host").field("columns", columns);
            }
            MaterializedStorage::Resident { slots, bytes, .. } => {
                output
                    .field("storage", &"resident")
                    .field("slots", slots)
                    .field("bytes", bytes);
            }
        }
        output.finish()
    }
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
            backend: GAFIME_BACKEND_CPU,
            storage: MaterializedStorage::Host(columns),
        })
    }

    /// Construct host-readable values that were explicitly downloaded from a
    /// non-Core executor.  The values remain tagged with the producing
    /// backend: callers may consume them as host output, but may not present
    /// them to that backend as resident input for further native arithmetic.
    pub fn from_downloaded(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        backend: BackendKind,
        columns: BTreeMap<FeatureId, NumericColumn>,
    ) -> SemanticResult<Self> {
        if !matches!(
            backend,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) {
            return Err(SemanticError::Invalid(
                "downloaded semantic output must retain a non-Core backend origin",
            ));
        }
        let mut output = Self::from_columns(registry, frame, columns)?;
        output.backend = backend;
        Ok(output)
    }

    /// Construct a context-bound resident bank after the backend has already
    /// validated its physical allocation and slot map.  Semantic identities
    /// remain Rust-owned: this method verifies that each logical feature is
    /// registry-owned before accepting an opaque native lease.
    pub fn from_resident(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        backend: BackendKind,
        slots: BTreeMap<FeatureId, u32>,
        bytes: usize,
        lease: ResidentMaterializationLease,
    ) -> SemanticResult<Self> {
        if !matches!(
            backend,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) || registry.schema() != frame.schema()
            || registry.precision() != frame.profile()
            || slots.is_empty()
            || bytes == 0
        {
            return Err(SemanticError::Invalid(
                "invalid resident semantic materialization",
            ));
        }
        let mut seen_slots = BTreeSet::new();
        for (&id, &slot) in &slots {
            registry.program(id)?;
            if !seen_slots.insert(slot) {
                return Err(SemanticError::Invalid(
                    "resident materialization maps multiple features to one slot",
                ));
            }
        }
        Ok(Self {
            frame_id: frame.id(),
            profile: frame.profile(),
            backend,
            storage: MaterializedStorage::Resident {
                slots,
                bytes,
                lease: Some(lease),
            },
        })
    }

    /// A truthful zero-column resident result for an empty accepted set.  No
    /// native bank is allocated and no host values are fabricated.
    pub fn empty_resident(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        backend: BackendKind,
    ) -> SemanticResult<Self> {
        if !matches!(
            backend,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) || registry.schema() != frame.schema()
            || registry.precision() != frame.profile()
        {
            return Err(SemanticError::Invalid(
                "invalid empty resident semantic materialization",
            ));
        }
        Ok(Self {
            frame_id: frame.id(),
            profile: frame.profile(),
            backend,
            storage: MaterializedStorage::Resident {
                slots: BTreeMap::new(),
                bytes: 0,
                lease: None,
            },
        })
    }
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub fn get(&self, id: FeatureId) -> SemanticResult<&[f32]> {
        self.get_typed(id)?.as_f32()
    }
    pub fn get_typed(&self, id: FeatureId) -> SemanticResult<&NumericColumn> {
        match &self.storage {
            MaterializedStorage::Host(columns) => columns
                .get(&id)
                .ok_or(SemanticError::Invalid("feature is not materialized")),
            MaterializedStorage::Resident { .. } => Err(SemanticError::Unsupported(
                "resident materialization requires an explicit backend download",
            )),
        }
    }
    pub fn profile(&self) -> PrecisionProfile {
        self.profile
    }
    pub fn backend_kind(&self) -> BackendKind {
        self.backend
    }
    pub fn contains(&self, id: FeatureId) -> bool {
        match &self.storage {
            MaterializedStorage::Host(columns) => columns.contains_key(&id),
            MaterializedStorage::Resident { slots, .. } => slots.contains_key(&id),
        }
    }
    pub fn bytes(&self) -> usize {
        match &self.storage {
            MaterializedStorage::Host(columns) => columns
                .values()
                .fold(0, |sum, column| sum.saturating_add(column.bytes())),
            MaterializedStorage::Resident { bytes, .. } => *bytes,
        }
    }
    pub fn columns(&self) -> SemanticResult<&BTreeMap<FeatureId, NumericColumn>> {
        match &self.storage {
            MaterializedStorage::Host(columns) => Ok(columns),
            MaterializedStorage::Resident { .. } => Err(SemanticError::Unsupported(
                "resident materialization requires an explicit backend download",
            )),
        }
    }
    pub fn resident_slots(&self) -> SemanticResult<&BTreeMap<FeatureId, u32>> {
        match &self.storage {
            MaterializedStorage::Resident { slots, .. } => Ok(slots),
            MaterializedStorage::Host(_) => Err(SemanticError::Invalid(
                "host materialization has no resident slot map",
            )),
        }
    }
    pub fn resident_lease(&self) -> SemanticResult<&ResidentMaterializationLease> {
        match &self.storage {
            MaterializedStorage::Resident {
                lease: Some(lease), ..
            } => Ok(lease),
            MaterializedStorage::Resident { lease: None, .. } => Err(SemanticError::Invalid(
                "empty resident materialization has no native bank",
            )),
            MaterializedStorage::Host(_) => Err(SemanticError::Invalid(
                "host materialization has no resident lease",
            )),
        }
    }
    pub fn is_resident(&self) -> bool {
        matches!(self.storage, MaterializedStorage::Resident { .. })
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

    /// Validate backend capabilities and return a conservative additional work
    /// charge before materialization. The session aggregates unique channels:
    /// granting each channel the entire call budget would admit unbounded sums.
    /// Core uses the ordinary structural charge; accelerator sorting/ranking
    /// may require additional backend-specific admission without substitution.
    fn validate_evidence_admission(
        &self,
        _definition: &EvidenceDefinition,
        _pair_count: usize,
        _support_rows: usize,
    ) -> SemanticResult<usize> {
        Ok(0)
    }

    /// Fit one ordered profile-native mean for every requested materialized
    /// candidate.  The returned typed bit vector is in exactly `candidates`
    /// order; it is a narrow arithmetic primitive, not a host-download or
    /// Python fitting fallback.
    fn fit_means(
        &mut self,
        _values: &MaterializedColumns,
        _candidates: &[FeatureId],
        _max_bytes: usize,
    ) -> SemanticResult<FrozenMeans> {
        Err(SemanticError::Unsupported(
            "semantic executor does not support fitted centered interaction means",
        ))
    }

    /// Retain only accepted values, optionally merging a same-context retained
    /// bank.  Backends must validate frame, profile, schema and backend
    /// identity before allocating; `max_live_bytes` covers all old, source,
    /// output and temporary resident allocations while this operation runs.
    fn retain(
        &mut self,
        _registry: &CandidateRegistry,
        _frame: &FeatureFrame,
        _source: &MaterializedColumns,
        _prior: Option<&MaterializedColumns>,
        _selected: &[FeatureId],
        _max_live_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        Err(SemanticError::Unsupported(
            "semantic executor does not support retained materializations",
        ))
    }

    /// Materialize resident values into an explicit host-owned representation.
    /// This is an output transfer, never a CPU fallback for native arithmetic.
    fn download(
        &mut self,
        _registry: &CandidateRegistry,
        _frame: &FeatureFrame,
        _source: &MaterializedColumns,
        _max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        Err(SemanticError::Unsupported(
            "semantic executor does not support resident materialization download",
        ))
    }
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
    training_bindings: Vec<Arc<TrainingBinding>>,
    contextual_training: Arc<[super::evidence::TrainingLineage]>,
}

/// Caller-selected resource ceilings, not heuristic unlimited cache growth.
/// Budgets cover native working banks and retained columns owned by this session;
/// immutable input frames and caller-retained tables/acceptance records are separate.
#[derive(Clone, Copy, Debug)]
pub struct SessionLimits {
    pub max_bytes: usize,
    pub max_retained_bytes: usize,
    /// Structural candidate-row/edge units across dependencies and channels,
    /// plus preflighted fitting-lineage metadata construction. This is
    /// admission control, not actual kernel visits, FLOPs or elapsed time.
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

/// Closed built-in proposal operators. Centered products require caller-owned
/// frozen means and therefore remain explicit declarations rather than bulk
/// proposals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProposalOperator {
    Source,
    Softsign,
    AbsoluteDifference,
}

const MAX_PROPOSAL_CANDIDATES: usize = 65_536;

/// Exclusive declaration scope for a discovery round. Only raw sources,
/// explicitly supplied accepted atoms, and programs constructed in this round
/// can be operands. Acceptance is not a decorative record beside a mutable registry.
pub struct DiscoveryRound<'a> {
    registry: &'a mut CandidateRegistry,
    eligible: &'a mut BTreeSet<FeatureId>,
    // Predicate leaves deliberately have a narrower operand authority than
    // generic algebraic proposals: only raw sources and explicitly accepted
    // prior-round programs are atoms. Newly proposed current-round algebraic
    // nodes may compose through their own operators, but cannot silently turn
    // into an unfitted decision threshold input.
    predicate_atoms: &'a BTreeSet<FeatureId>,
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
    fn predicate_atom(&self, id: FeatureId) -> SemanticResult<()> {
        self.registry.program(id)?;
        if !self.predicate_atoms.contains(&id) {
            return Err(SemanticError::Invalid(
                "hard predicate input is not a raw or accepted atom in this round",
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

    /// Declare one exact hard predicate over a raw source or explicitly
    /// accepted/current-round atom in an f32-storage profile.
    pub fn hard_predicate(
        &mut self,
        input: FeatureId,
        comparison: PredicateComparator,
        threshold: f32,
    ) -> SemanticResult<FeatureId> {
        self.predicate_atom(input)?;
        let id = self.registry.hard_predicate(input, comparison, threshold)?;
        self.eligible.insert(id);
        Ok(id)
    }

    /// Declare one exact hard predicate over a raw source or explicitly
    /// accepted/current-round atom in an fp64 profile.
    pub fn hard_predicate_f64(
        &mut self,
        input: FeatureId,
        comparison: PredicateComparator,
        threshold: f64,
    ) -> SemanticResult<FeatureId> {
        self.predicate_atom(input)?;
        let id = self
            .registry
            .hard_predicate_f64(input, comparison, threshold)?;
        self.eligible.insert(id);
        Ok(id)
    }

    /// Declare a canonical flattened hard-AND region from eligible hard
    /// predicates and/or eligible previously accepted regions.
    pub fn decision_region(&mut self, terms: Vec<FeatureId>) -> SemanticResult<FeatureId> {
        for &term in &terms {
            self.operand(term)?;
        }
        let id = self.registry.decision_region(terms)?;
        self.eligible.insert(id);
        Ok(id)
    }

    /// Deterministically declare a bounded built-in catalog from eligible
    /// atoms. Supplied operator order is retained, while atoms are sorted by
    /// `FeatureId`; duplicate atoms are harmless, duplicate operators are not.
    /// A failed batch rolls back every newly appended registry program before
    /// returning, so no unreturned identity becomes eligible or observable.
    pub fn propose(
        &mut self,
        operators: &[ProposalOperator],
        atoms: &[FeatureId],
        max_candidates: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        if operators.is_empty() {
            return Err(SemanticError::Invalid(
                "proposal requires at least one operator",
            ));
        }
        if atoms.is_empty() {
            return Err(SemanticError::Invalid(
                "proposal requires at least one atom",
            ));
        }
        if max_candidates == 0 || max_candidates > MAX_PROPOSAL_CANDIDATES {
            return Err(SemanticError::Invalid(
                "proposal candidate limit must be between one and 65536",
            ));
        }
        if operators.iter().copied().collect::<BTreeSet<_>>().len() != operators.len() {
            return Err(SemanticError::Invalid("proposal operators must be unique"));
        }
        let atoms = atoms.iter().copied().collect::<BTreeSet<_>>();
        for &atom in &atoms {
            self.operand(atom)?;
        }
        let atoms = atoms.into_iter().collect::<Vec<_>>();

        let checkpoint = self.registry.mutation_checkpoint();
        let result: SemanticResult<Vec<FeatureId>> = (|| {
            let mut proposed = Vec::new();
            let mut seen = BTreeSet::new();
            'operators: for &operator in operators {
                match operator {
                    ProposalOperator::Source => {
                        for &atom in &atoms {
                            if !matches!(self.registry.program(atom)?.op(), FeatureOp::Source(_)) {
                                continue;
                            }
                            push_proposal(&mut proposed, &mut seen, atom, max_candidates);
                            if proposed.len() == max_candidates {
                                break 'operators;
                            }
                        }
                    }
                    ProposalOperator::Softsign => {
                        for &atom in &atoms {
                            let candidate = self.registry.softsign(atom)?;
                            push_proposal(&mut proposed, &mut seen, candidate, max_candidates);
                            if proposed.len() == max_candidates {
                                break 'operators;
                            }
                        }
                    }
                    ProposalOperator::AbsoluteDifference => {
                        for (index, &left) in atoms.iter().enumerate() {
                            for &right in &atoms[index + 1..] {
                                let candidate = self.registry.abs_difference(left, right)?;
                                push_proposal(&mut proposed, &mut seen, candidate, max_candidates);
                                if proposed.len() == max_candidates {
                                    break 'operators;
                                }
                            }
                        }
                    }
                }
            }
            Ok(proposed)
        })();
        match result {
            Ok(proposed) => {
                self.eligible.extend(proposed.iter().copied());
                Ok(proposed)
            }
            Err(error) => {
                self.registry.rollback_mutations(checkpoint);
                Err(error)
            }
        }
    }

    /// Internal half of a session-owned fitted interaction proposal.  The
    /// session obtains profile-native means from the selected executor first;
    /// this scope then performs only deterministic registry mutation and
    /// eligibility admission.
    fn propose_fitted_centered_interactions(
        &mut self,
        atoms: &[FeatureId],
        arities: &[usize],
        means: &FrozenMeans,
        binding: Arc<TrainingBinding>,
        max_candidates: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        if atoms.is_empty()
            || arities.is_empty()
            || means.len() != atoms.len()
            || max_candidates == 0
            || max_candidates > MAX_PROPOSAL_CANDIDATES
        {
            return Err(SemanticError::Invalid(
                "invalid fitted centered interaction proposal",
            ));
        }
        if atoms.windows(2).any(|pair| pair[0] >= pair[1])
            || arities.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(SemanticError::Invalid(
                "fitted interaction atoms and arities must be unique and ordered",
            ));
        }
        for &atom in atoms {
            self.operand(atom)?;
        }
        for &arity in arities {
            if arity < 2 || arity > self.registry.limits().max_logical_arity {
                return Err(SemanticError::Invalid(
                    "fitted centered interaction arity is outside registry bounds",
                ));
            }
        }

        let checkpoint = self.registry.mutation_checkpoint();
        let result: SemanticResult<Vec<FeatureId>> = (|| {
            let mut proposed = Vec::new();
            let mut seen = BTreeSet::new();
            let mut indices = Vec::new();
            for &arity in arities {
                if emit_centered_combinations(
                    self.registry,
                    atoms,
                    means,
                    arity,
                    0,
                    &mut indices,
                    max_candidates,
                    &mut proposed,
                    &mut seen,
                )? {
                    break;
                }
            }
            self.registry.attach_training_binding(&proposed, binding)?;
            Ok(proposed)
        })();
        match result {
            Ok(proposed) => {
                self.eligible.extend(proposed.iter().copied());
                Ok(proposed)
            }
            Err(error) => {
                self.registry.rollback_mutations(checkpoint);
                Err(error)
            }
        }
    }
}

fn push_proposal(
    proposed: &mut Vec<FeatureId>,
    seen: &mut BTreeSet<FeatureId>,
    candidate: FeatureId,
    limit: usize,
) {
    if proposed.len() < limit && seen.insert(candidate) {
        proposed.push(candidate);
    }
}

/// Enumerate lexicographic combinations without materializing the full
/// combinatorial catalog.  The selected order is intentional: a bulk proposal
/// emits one canonical operand order per atom set, while an explicit caller
/// may still declare a different ordered centered product.
#[allow(clippy::too_many_arguments)]
fn emit_centered_combinations(
    registry: &mut CandidateRegistry,
    atoms: &[FeatureId],
    means: &FrozenMeans,
    arity: usize,
    start: usize,
    indices: &mut Vec<usize>,
    limit: usize,
    proposed: &mut Vec<FeatureId>,
    seen: &mut BTreeSet<FeatureId>,
) -> SemanticResult<bool> {
    if proposed.len() == limit {
        return Ok(true);
    }
    if indices.len() == arity {
        let operands = indices.iter().map(|&index| atoms[index]).collect();
        let frozen = match means {
            FrozenMeans::F32(values) => {
                FrozenMeans::F32(indices.iter().map(|&index| values[index]).collect())
            }
            FrozenMeans::F64(values) => {
                FrozenMeans::F64(indices.iter().map(|&index| values[index]).collect())
            }
        };
        let candidate = registry.centered_product_from_frozen(operands, frozen)?;
        push_proposal(proposed, seen, candidate, limit);
        return Ok(proposed.len() == limit);
    }
    let remaining = arity
        .checked_sub(indices.len())
        .ok_or(SemanticError::Invalid("fitted interaction arity underflow"))?;
    let last_start = atoms
        .len()
        .checked_sub(remaining)
        .ok_or(SemanticError::Invalid(
            "fitted interaction arity exceeds atom count",
        ))?;
    for index in start..=last_start {
        indices.push(index);
        if emit_centered_combinations(
            registry,
            atoms,
            means,
            arity,
            index + 1,
            indices,
            limit,
            proposed,
            seen,
        )? {
            indices.pop();
            return Ok(true);
        }
        indices.pop();
    }
    Ok(false)
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
    /// Immutable fitting lineage captured from the evaluated candidate when it
    /// was accepted.  Future equal-state fits cannot mutate this record.
    pub fn training_bindings(&self) -> &[Arc<TrainingBinding>] {
        &self.training_bindings
    }
    /// Fitting origins of the evaluated program AND contextual reference
    /// programs. These audit the complete evidence set, not just policy-used
    /// channels, and never become fitted state of the accepted program itself.
    pub fn evaluation_training_bindings(&self) -> Vec<Arc<TrainingBinding>> {
        super::evidence::evaluation_training_bindings(
            &self.training_bindings,
            &self.contextual_training,
        )
    }
}

/// Bounded internal discovery lifecycle. One context of accepted values is
/// retained; switching input contexts drops that cache, not the programs.
/// Evidence is always re-evaluated. Close releases owned state and fails closed.
pub struct SemanticSession {
    id: u64,
    registry: Option<CandidateRegistry>,
    cache: Option<MaterializedColumns>,
    backend: BackendKind,
    limits: SessionLimits,
    round: u64,
    eligible: Option<BTreeSet<FeatureId>>,
    predicate_atoms: Option<BTreeSet<FeatureId>>,
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
        if !matches!(
            backend,
            GAFIME_BACKEND_CPU | GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) {
            return Err(SemanticError::Invalid(
                "semantic session requires one concrete backend kind",
            ));
        }
        let limits = limits.validate()?;
        Ok(Self {
            id: next_identity()?,
            registry: Some(registry),
            cache: None,
            backend,
            limits,
            round: 0,
            eligible: None,
            predicate_atoms: None,
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
        // Bound declaration input independently of the unique accepted union:
        // overlapping batches confer the same authority, not extra programs.
        if self.round >= self.limits.max_rounds || accepted.len() > 65_536 {
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
        if eligible.len() > registry.limits().max_nodes {
            return Err(SemanticError::Invalid(
                "discovery round resource limit exceeded",
            ));
        }
        let predicate_atoms = eligible.clone();
        self.round += 1;
        self.eligible = Some(eligible);
        self.predicate_atoms = Some(predicate_atoms);
        Ok(DiscoveryRound {
            registry: self.registry.as_mut().ok_or(SemanticError::Closed)?,
            eligible: self.eligible.as_mut().expect("round initialized"),
            predicate_atoms: self
                .predicate_atoms
                .as_ref()
                .expect("predicate atoms initialized"),
        })
    }

    /// Borrow the currently active declaration scope without beginning another
    /// round or changing its eligible atom set. This is for one round's
    /// independent declarations; it is not an authority to revive an old one.
    pub fn current_round(&mut self) -> SemanticResult<DiscoveryRound<'_>> {
        if self.registry.is_none() {
            return Err(SemanticError::Closed);
        }
        if self.eligible.is_none() || self.predicate_atoms.is_none() {
            return Err(SemanticError::Invalid(
                "no active discovery round is available",
            ));
        }
        Ok(DiscoveryRound {
            registry: self.registry.as_mut().expect("open registry checked"),
            eligible: self.eligible.as_mut().expect("active round checked"),
            predicate_atoms: self
                .predicate_atoms
                .as_ref()
                .expect("active predicate atoms checked"),
        })
    }

    pub fn round(&self) -> u64 {
        self.round
    }
    /// The explicit execution backend selected before this session was
    /// constructed.  Operation-level capability negotiation is performed by
    /// its executor before each native lowering; no implicit fallback exists.
    pub const fn backend_kind(&self) -> BackendKind {
        self.backend
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
    fn declared_atom(&self, id: FeatureId) -> SemanticResult<()> {
        self.registry()?.program(id)?;
        if self
            .predicate_atoms
            .as_ref()
            .is_none_or(|set| !set.contains(&id))
        {
            return Err(SemanticError::Invalid(
                "fitted interaction atom is not a raw or accepted atom in this round",
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
        self.predicate_atoms = None;
    }

    fn validate_executor(&self, executor: &dyn NativeEvidenceExecutor) -> SemanticResult<()> {
        self.registry()?;
        if executor.backend_kind() != self.backend {
            return Err(SemanticError::Invalid(
                "semantic executor does not match the selected backend",
            ));
        }
        Ok(())
    }

    /// Fit and declare a bounded bulk of ordered centered interactions from an
    /// immutable discovery snapshot.  Means come from the selected native
    /// executor through [`NativeEvidenceExecutor::fit_means`]; this method
    /// never downloads a resident bank or substitutes Core/Python arithmetic.
    pub fn propose_centered_interactions(
        &mut self,
        executor: &mut dyn NativeEvidenceExecutor,
        training: &FeatureFrame,
        atoms: &[FeatureId],
        arities: &[usize],
        max_candidates: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        self.validate_executor(executor)?;
        let registry = self.registry()?;
        if training.role() != super::EvaluationRole::Discovery
            || training.schema() != registry.schema()
            || training.profile() != registry.precision()
            || atoms.is_empty()
            || arities.is_empty()
            || max_candidates == 0
            || max_candidates > MAX_PROPOSAL_CANDIDATES
        {
            return Err(SemanticError::Invalid(
                "fitted centered interactions require a bounded matching discovery frame",
            ));
        }
        let atoms = atoms.iter().copied().collect::<BTreeSet<_>>();
        let atoms = atoms.into_iter().collect::<Vec<_>>();
        let arity_set = arities.iter().copied().collect::<BTreeSet<_>>();
        if arity_set.len() != arities.len() {
            return Err(SemanticError::Invalid(
                "fitted centered interaction arities must be unique",
            ));
        }
        let arities = arity_set.into_iter().collect::<Vec<_>>();
        if arities.iter().any(|&arity| {
            arity < 2 || arity > registry.limits().max_logical_arity || arity > atoms.len()
        }) {
            return Err(SemanticError::Invalid(
                "fitted centered interaction arity is outside available atom bounds",
            ));
        }
        for &atom in &atoms {
            self.declared_atom(atom)?;
        }
        let work = dependency_work(registry, training.rows(), &atoms)?
            .checked_add(
                training
                    .rows()
                    .checked_mul(atoms.len())
                    .ok_or(SemanticError::Invalid("semantic fitting work overflow"))?,
            )
            .and_then(|work| work.checked_add(max_candidates))
            .ok_or(SemanticError::Invalid("semantic fitting work overflow"))?;
        if work > self.limits.max_work {
            return Err(SemanticError::Invalid(
                "fitted centered interaction work limit exceeded",
            ));
        }

        // Do not switch or evict the session cache while merely fitting a
        // declaration batch.  A failed proposal must leave lifecycle state
        // exactly as it was; same-frame retention remains reusable here.
        let retained = self
            .cache
            .as_ref()
            .filter(|cache| cache.frame_id() == training.id());
        let budget = self
            .limits
            .max_bytes
            .checked_sub(self.retained_bytes())
            .ok_or(SemanticError::Invalid(
                "retained semantic values exceed session budget",
            ))?;
        let materialized = executor.materialize(registry, training, &atoms, retained, budget)?;
        validate_output(
            registry,
            training,
            self.backend,
            &atoms,
            &materialized,
            budget,
        )?;
        let fit_budget = budget
            .checked_sub(materialized.bytes())
            .ok_or(SemanticError::Invalid(
                "semantic mean fitting exceeds session budget",
            ))?;
        let means = executor.fit_means(&materialized, &atoms, fit_budget)?;
        validate_fitted_means(training.profile(), &means, atoms.len())?;
        let binding = Arc::new(TrainingBinding::from_discovery_frame(
            training.id(),
            Arc::from(training.row_domain()),
            Arc::from(training.provenance()),
        ));
        self.current_round()?.propose_fitted_centered_interactions(
            &atoms,
            &arities,
            &means,
            binding,
            max_candidates,
        )
    }

    pub fn evaluate(
        &mut self,
        executor: &mut dyn NativeEvidenceExecutor,
        frame: Arc<FeatureFrame>,
        candidates: &[FeatureId],
        channels: &[EvidenceChannel],
    ) -> SemanticResult<EvidenceTable> {
        self.validate_executor(executor)?;
        let registry = self.registry()?;
        // Inference executes already-accepted programs. It is not an admission
        // context that may mint new acceptance records, even when labels exist.
        if frame.role() == super::EvaluationRole::Inference {
            return Err(SemanticError::Invalid(
                "inference frames cannot be used for evidence evaluation",
            ));
        }
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
            if let Some(reference) = channel.definition().reference() {
                self.eligible(reference)?;
                roots.push(reference);
            }
        }
        roots.sort();
        roots.dedup();
        let mut work = dependency_work(registry, frame.rows(), &roots)?;
        for (index, channel) in channels.iter().enumerate() {
            if channels[..index].iter().any(|old| old.same_work(channel)) {
                continue;
            }
            let definition = channel.definition();
            let rows = if let Some(graph) = definition.graph() {
                graph.edges().len()
            } else if let Some(labels) = definition.labels() {
                labels.as_ref().map_or(0, |labels| labels.rows().len())
            } else if let Some(view) = definition.paired_view() {
                work = work
                    .checked_add(dependency_work(registry, view.rows(), &candidates)?)
                    .ok_or(SemanticError::Invalid("semantic work count overflow"))?;
                frame.rows()
            } else {
                frame.rows()
            };
            // Missing labels deliberately produce explicit unavailable values
            // without a native statistic request. Every other unique channel
            // gives a selected backend one bounded admission opportunity
            // before materialization or evidence dispatch.
            if !matches!(definition.labels(), Some(None)) {
                work = work
                    .checked_add(executor.validate_evidence_admission(
                        definition,
                        candidates.len(),
                        rows,
                    )?)
                    .ok_or(SemanticError::Invalid("semantic work count overflow"))?;
            }
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
        // Provenance snapshots are not numeric banks and therefore are not
        // covered by `max_bytes`. Preflight their structurally deduplicated
        // expansion before allocating an EvidenceTable, charging both passes
        // against the remaining session work ceiling.
        let lineage_budget =
            self.limits
                .max_work
                .checked_sub(work)
                .ok_or(SemanticError::Invalid(
                    "semantic fitting lineage metadata work limit exceeded",
                ))?;
        let lineage_snapshot = registry.snapshot_training_lineages(&roots, lineage_budget)?;
        work = work
            .checked_add(lineage_snapshot.metadata_work)
            .ok_or(SemanticError::Invalid(
                "semantic fitting lineage metadata work overflow",
            ))?;
        // Shared per-root arrays avoid candidate x reference lineage expansion.
        // Acceptance retains the same immutable contextual snapshot, whereas
        // later program construction consults only mathematical dependencies.
        let handle_work = if lineage_snapshot.metadata_work == 0 {
            0
        } else {
            roots.len().saturating_mul(2)
        };
        work = work.checked_add(handle_work).ok_or(SemanticError::Invalid(
            "semantic fitting lineage metadata work overflow",
        ))?;
        if work > self.limits.max_work {
            return Err(SemanticError::Invalid(
                "semantic fitting lineage metadata work limit exceeded",
            ));
        }
        // Empty lineages share one allocation. The ordinary unfitted path
        // must not acquire a per-candidate heap object just for provenance.
        let empty_lineage: super::evidence::TrainingLineage = Arc::from([]);
        let lineages: Vec<super::evidence::TrainingLineage> = lineage_snapshot
            .lineages
            .into_iter()
            .map(|lineage| {
                if lineage.is_empty() {
                    Arc::clone(&empty_lineage)
                } else {
                    Arc::from(lineage)
                }
            })
            .collect();
        let lineage_for = |id: &FeatureId| {
            Arc::clone(&lineages[roots.binary_search(id).expect("validated evaluation root")])
        };
        let training_lineage = candidates.iter().map(lineage_for).collect();
        let reference_ids: BTreeSet<_> = channels
            .iter()
            .filter_map(|channel| channel.definition().reference())
            .collect();
        let contextual_training = reference_ids
            .iter()
            .map(lineage_for)
            .collect::<Vec<_>>()
            .into();
        self.switch_context(&frame);
        let registry = self.registry()?;
        let retained = self.cache.as_ref().filter(|c| c.frame_id == frame.id());
        // Reserve half for the paired view; only one such view lives at a time.
        // The native budget includes its dependency bank and worker allocations.
        let budget = (self.limits.max_bytes - self.retained_bytes()) / 2;
        let materialized = executor.materialize(registry, &frame, &roots, retained, budget)?;
        validate_output(
            registry,
            &frame,
            self.backend,
            &roots,
            &materialized,
            budget,
        )?;
        let mut channel_values: Vec<Vec<EvidenceValue>> = Vec::with_capacity(channels.len());
        for (index, channel) in channels.iter().enumerate() {
            if let Some(previous) = channels[..index]
                .iter()
                .position(|old| old.same_work(channel))
            {
                channel_values.push(channel_values[previous].clone());
                continue;
            }
            let paired = if let Some(view) = channel.definition().paired_view() {
                let result = executor.materialize(registry, view, &candidates, None, budget)?;
                validate_output(registry, view, self.backend, &candidates, &result, budget)?;
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
            training_lineage,
            contextual_training,
            materialized,
            backend: self.backend,
        })
    }

    /// Apply selection once and retain accepted values through the selected
    /// backend.  A resident executor must preserve device ownership during
    /// retention; it must not download values merely to rebuild a cache.
    pub fn accept_with(
        &mut self,
        executor: &mut dyn NativeEvidenceExecutor,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
    ) -> SemanticResult<Vec<AcceptedFeature>> {
        self.validate_executor(executor)?;
        let selected = self.select_for_accept(table, policy)?;
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let registry = self.registry()?;
        let prior = self
            .cache
            .as_ref()
            .filter(|cache| cache.frame_id == table.frame.id());
        let retained = executor.retain(
            registry,
            &table.frame,
            &table.materialized,
            prior,
            &selected,
            self.limits.max_bytes,
        )?;
        self.finish_accept(table, policy, selected, retained)
    }

    /// Compatibility adapter for existing Core callers.  It shares the same
    /// selection and acceptance path as [`Self::accept_with`], but is limited
    /// to host columns and therefore never turns resident GPU values into an
    /// undocumented CPU path.
    pub fn accept(
        &mut self,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
    ) -> SemanticResult<Vec<AcceptedFeature>> {
        if table.materialized.is_resident() {
            return Err(SemanticError::Unsupported(
                "resident materializations require accept_with on their selected backend",
            ));
        }
        if self.backend != GAFIME_BACKEND_CPU {
            return Err(SemanticError::Unsupported(
                "non-Core semantic sessions require accept_with",
            ));
        }
        let selected = self.select_for_accept(table, policy)?;
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let registry = self.registry()?;
        let mut retained = match self
            .cache
            .as_ref()
            .filter(|cache| cache.frame_id == table.frame.id())
        {
            Some(cache) => cache.columns()?.clone(),
            None => BTreeMap::new(),
        };
        let source = table.materialized.columns()?;
        for &feature in &selected {
            retained.insert(
                feature,
                source
                    .get(&feature)
                    .ok_or(SemanticError::ForeignIdentity)?
                    .shared_clone(),
            );
        }
        let retained = MaterializedColumns::from_columns(registry, &table.frame, retained)?;
        self.finish_accept(table, policy, selected, retained)
    }

    fn select_for_accept(
        &self,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
    ) -> SemanticResult<Vec<FeatureId>> {
        self.registry()?;
        if table.owner != self.id {
            return Err(SemanticError::ForeignIdentity);
        }
        if table.backend != self.backend || table.materialized.backend_kind() != self.backend {
            return Err(SemanticError::Invalid(
                "evidence table backend does not match the selected session backend",
            ));
        }
        if table.round != self.round {
            return Err(SemanticError::Invalid(
                "cannot accept evidence from a previous discovery round",
            ));
        }
        policy.select(table, self.limits.max_work)
    }

    fn finish_accept(
        &mut self,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
        selected: Vec<FeatureId>,
        retained: MaterializedColumns,
    ) -> SemanticResult<Vec<AcceptedFeature>> {
        if retained.frame_id != table.frame.id()
            || retained.profile != table.frame.profile()
            || retained.backend != self.backend
            || (self.backend != GAFIME_BACKEND_CPU && !retained.is_resident())
            || retained.bytes() > self.limits.max_retained_bytes
            || selected.iter().any(|&feature| !retained.contains(feature))
        {
            return Err(SemanticError::Invalid("accepted materialization retention limit exceeded; clear retained values explicitly"));
        }
        let mut accepted = Vec::with_capacity(selected.len());
        for &feature in &selected {
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
                training_bindings: table.training_bindings(feature)?.to_vec(),
                contextual_training: Arc::clone(&table.contextual_training),
            });
        }
        self.cache = Some(retained);
        Ok(accepted)
    }

    /// Execute frozen accepted programs on new rows without consulting labels
    /// or reapplying their discovery policy. Same-context columns may be reused.
    pub fn materialize_accepted(
        &mut self,
        executor: &mut dyn NativeEvidenceExecutor,
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
            self.backend,
            &ids,
            &result,
            self.limits.max_bytes - self.retained_bytes(),
        )?;
        let cache = executor.retain(
            self.registry()?,
            frame,
            &result,
            self.cache
                .as_ref()
                .filter(|cache| cache.frame_id == frame.id()),
            &ids,
            self.limits.max_bytes,
        )?;
        if cache.frame_id != frame.id()
            || cache.profile != frame.profile()
            || cache.backend != self.backend
            || (self.backend != GAFIME_BACKEND_CPU && !cache.is_resident())
            || cache.bytes() > self.limits.max_retained_bytes
        {
            return Err(SemanticError::Invalid(
                "accepted inference retention limit exceeded",
            ));
        }
        self.cache = Some(cache);
        Ok(result)
    }

    /// Explicitly transfer a resident materialization to host-owned output.
    /// This is deliberately outside native arithmetic: the returned values
    /// retain their producing backend tag and cannot be supplied to a non-Core
    /// executor as a resident materialization on a later call.
    pub fn download_materialization(
        &self,
        executor: &mut dyn NativeEvidenceExecutor,
        frame: &FeatureFrame,
        source: &MaterializedColumns,
    ) -> SemanticResult<MaterializedColumns> {
        self.validate_executor(executor)?;
        let registry = self.registry()?;
        if frame.schema() != registry.schema()
            || frame.profile() != registry.precision()
            || source.frame_id() != frame.id()
            || source.profile() != frame.profile()
            || source.backend_kind() != self.backend
            || !source.is_resident()
        {
            return Err(SemanticError::Invalid(
                "resident materialization download context or backend mismatch",
            ));
        }
        let source_slots = source.resident_slots()?;
        for &feature in source_slots.keys() {
            registry.program(feature)?;
        }

        // The retained bank, source bank and explicit host output coexist for
        // this transfer.  The executor receives only the remaining host-output
        // allowance; it cannot treat a download as an unaccounted copy.
        let output_budget = self
            .limits
            .max_bytes
            .checked_sub(self.retained_bytes())
            .and_then(|remaining| remaining.checked_sub(source.bytes()))
            .ok_or(SemanticError::Invalid(
                "resident materialization download exceeds session budget",
            ))?;
        let output = executor.download(registry, frame, source, output_budget)?;
        if output.frame_id() != frame.id()
            || output.profile() != frame.profile()
            || output.backend_kind() != self.backend
            || output.is_resident()
            || output.bytes() > output_budget
        {
            return Err(SemanticError::Invalid(
                "downloaded materialization violates context, storage, or budget",
            ));
        }
        let columns = output.columns()?;
        if columns.len() != source_slots.len()
            || source_slots
                .keys()
                .any(|feature| !columns.contains_key(feature))
        {
            return Err(SemanticError::Invalid(
                "downloaded materialization omitted or added a resident feature",
            ));
        }
        Ok(output)
    }
}

fn validate_output(
    registry: &CandidateRegistry,
    frame: &FeatureFrame,
    backend: BackendKind,
    roots: &[FeatureId],
    output: &MaterializedColumns,
    budget: usize,
) -> SemanticResult<()> {
    if output.frame_id != frame.id()
        || output.profile != frame.profile()
        || output.backend != backend
        || (backend != GAFIME_BACKEND_CPU && !output.is_resident())
        || output.bytes() > budget
    {
        return Err(SemanticError::Invalid("native output context mismatch"));
    }
    for &root in roots {
        registry.program(root)?;
        if !output.contains(root) {
            return Err(SemanticError::Invalid(
                "native output omitted a requested feature",
            ));
        }
    }
    Ok(())
}

fn validate_fitted_means(
    profile: PrecisionProfile,
    means: &FrozenMeans,
    expected: usize,
) -> SemanticResult<()> {
    if means.len() != expected {
        return Err(SemanticError::Invalid(
            "native mean fitting returned the wrong candidate count",
        ));
    }
    match (profile, means) {
        (PrecisionProfile::Fp32 | PrecisionProfile::Mixed, FrozenMeans::F32(bits))
            if bits.iter().all(|bits| f32::from_bits(*bits).is_finite()) =>
        {
            Ok(())
        }
        (PrecisionProfile::Fp64, FrozenMeans::F64(bits))
            if bits.iter().all(|bits| f64::from_bits(*bits).is_finite()) =>
        {
            Ok(())
        }
        (PrecisionProfile::Fp32 | PrecisionProfile::Mixed, FrozenMeans::F64(_)) => Err(
            SemanticError::Invalid("native mean fitting returned f64 state for f32 storage"),
        ),
        (PrecisionProfile::Fp64, FrozenMeans::F32(_)) => Err(SemanticError::Invalid(
            "native mean fitting returned f32 state for fp64 storage",
        )),
        _ => Err(SemanticError::Invalid(
            "native mean fitting returned nonfinite frozen state",
        )),
    }
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
            FeatureOp::Softsign(a) | FeatureOp::HardPredicate { input: a, .. } => pending.push(*a),
            FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
            FeatureOp::DecisionRegion { terms } => pending.extend(terms),
        }
    }
    units
        .checked_mul(rows)
        .ok_or(SemanticError::Invalid("semantic work count overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::{
        AssociationContext, AssociationStatistic, EvaluationRole, ProgramLimits,
    };
    use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CUDA};

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

    struct ChargedAdmission;
    impl NativeEvidenceExecutor for ChargedAdmission {
        fn backend_kind(&self) -> u32 {
            GAFIME_BACKEND_CPU
        }
        fn validate_evidence_admission(
            &self,
            _: &EvidenceDefinition,
            _: usize,
            _: usize,
        ) -> SemanticResult<usize> {
            Ok(60)
        }
        fn materialize(
            &mut self,
            _: &CandidateRegistry,
            _: &FeatureFrame,
            _: &[FeatureId],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<MaterializedColumns> {
            Err(SemanticError::Invalid("fixture reached materialization"))
        }
        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("admission fixture must not score")
        }
    }

    #[test]
    fn unique_backend_channel_costs_share_one_call_budget() {
        let frame = Arc::new(
            FeatureFrame::new(
                vec!["a".into()],
                "rows".into(),
                vec![0, 1],
                EvaluationRole::Discovery,
                "work admission".into(),
                vec![vec![0.0, 1.0]],
            )
            .unwrap(),
        );
        let registry = CandidateRegistry::new(
            frame.schema().to_vec(),
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let mut limits = SessionLimits::for_budget(1024);
        limits.max_work = 100;
        let mut session =
            SemanticSession::with_limits(registry, GAFIME_BACKEND_CPU, limits).unwrap();
        let source = session.begin_round(&[]).unwrap().source(0).unwrap();
        let make = |name: &str, statistic| {
            EvidenceChannel::new(
                name.into(),
                EvidenceDefinition::Association {
                    statistic,
                    context: AssociationContext::Reference { reference: source },
                },
            )
            .unwrap()
        };
        let pearson = make("pearson", AssociationStatistic::Pearson);
        let duplicate_work = make("another-name", AssociationStatistic::Pearson);
        let spearman = make("spearman", AssociationStatistic::Spearman);
        let error = session
            .evaluate(
                &mut ChargedAdmission,
                Arc::clone(&frame),
                &[source],
                &[pearson.clone(), spearman],
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            SemanticError::Invalid("semantic evaluation work limit exceeded")
        );
        // Identical mathematics/context is charged once even when exposed under
        // two channel names; the admitted call reaches the fixture boundary.
        let error = session
            .evaluate(
                &mut ChargedAdmission,
                frame,
                &[source],
                &[pearson, duplicate_work],
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            SemanticError::Invalid("fixture reached materialization")
        );
        assert_eq!(session.retained_bytes(), 0);
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
    fn current_round_reuses_active_scope_without_advancing_it() {
        let registry = CandidateRegistry::new(
            vec!["a".into()],
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1024).unwrap();
        assert!(matches!(
            session.current_round(),
            Err(SemanticError::Invalid(_))
        ));
        session.begin_round(&[]).unwrap();
        let first_round = session.round();
        {
            let round = session.current_round().unwrap();
            assert!(round.source(0).is_ok());
        }
        assert_eq!(session.round(), first_round);
        session.close();
        assert!(matches!(
            session.current_round(),
            Err(SemanticError::Closed)
        ));
    }

    #[test]
    fn bounded_proposal_is_deterministic_and_rolls_back_failed_batches() {
        let registry = CandidateRegistry::new(
            vec!["a".into(), "b".into(), "c".into()],
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1024).unwrap();
        let proposed = {
            let mut round = session.begin_round(&[]).unwrap();
            let a = round.source(0).unwrap();
            let b = round.source(1).unwrap();
            let c = round.source(2).unwrap();
            round
                .propose(
                    &[
                        ProposalOperator::AbsoluteDifference,
                        ProposalOperator::Source,
                        ProposalOperator::Softsign,
                    ],
                    &[c, a, b, a],
                    16,
                )
                .unwrap()
        };
        assert_eq!(proposed.len(), 9);
        let registry = session.registry().unwrap();
        assert!(matches!(
            registry.program(proposed[0]).unwrap().op(),
            FeatureOp::AbsoluteDifference(_, _)
        ));
        assert!(matches!(
            registry.program(proposed[1]).unwrap().op(),
            FeatureOp::AbsoluteDifference(_, _)
        ));
        assert!(matches!(
            registry.program(proposed[2]).unwrap().op(),
            FeatureOp::AbsoluteDifference(_, _)
        ));
        assert!(matches!(
            registry.program(proposed[3]).unwrap().op(),
            FeatureOp::Source(0)
        ));
        assert!(matches!(
            registry.program(proposed[6]).unwrap().op(),
            FeatureOp::Softsign(_)
        ));
        assert_eq!(session.round(), 1);

        let limited = CandidateRegistry::new(
            vec!["a".into(), "b".into()],
            PrecisionProfile::Mixed,
            ProgramLimits {
                max_nodes: 3,
                ..ProgramLimits::default()
            },
        )
        .unwrap();
        let mut limited = SemanticSession::new(limited, GAFIME_BACKEND_CPU, 1024).unwrap();
        let (a, b) = {
            let round = limited.begin_round(&[]).unwrap();
            (round.source(0).unwrap(), round.source(1).unwrap())
        };
        assert!(limited
            .current_round()
            .unwrap()
            .propose(&[ProposalOperator::Softsign], &[a, b], 2)
            .is_err());
        let recovered = limited
            .current_round()
            .unwrap()
            .propose(&[ProposalOperator::AbsoluteDifference], &[a, b], 1)
            .unwrap();
        assert_eq!(recovered.len(), 1, "failed proposal must not retain a node");
        assert!(limited
            .current_round()
            .unwrap()
            .propose(
                &[ProposalOperator::Source, ProposalOperator::Source],
                &[a],
                1,
            )
            .is_err());
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
        assert!(
            validate_output(&registry, &frame, GAFIME_BACKEND_CPU, &[source], &output, 7,).is_err()
        );
        assert!(
            validate_output(&registry, &frame, GAFIME_BACKEND_CPU, &[source], &output, 8,).is_ok()
        );
    }

    #[test]
    fn downloaded_gpu_values_remain_host_output_not_resident_gpu_input() {
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
        let downloaded = MaterializedColumns::from_downloaded(
            &registry,
            &frame,
            GAFIME_BACKEND_CUDA,
            BTreeMap::from([(source, NumericColumn::from(vec![0.0f32, 1.0]))]),
        )
        .unwrap();

        assert_eq!(downloaded.backend_kind(), GAFIME_BACKEND_CUDA);
        assert!(!downloaded.is_resident());
        assert_eq!(downloaded.get(source).unwrap(), &[0.0, 1.0]);
        assert!(validate_output(
            &registry,
            &frame,
            GAFIME_BACKEND_CUDA,
            &[source],
            &downloaded,
            downloaded.bytes(),
        )
        .is_err());
    }

    struct DownloadOnlyCuda {
        calls: usize,
        output_budget: Option<usize>,
    }

    impl NativeEvidenceExecutor for DownloadOnlyCuda {
        fn backend_kind(&self) -> u32 {
            GAFIME_BACKEND_CUDA
        }

        fn materialize(
            &mut self,
            _: &CandidateRegistry,
            _: &FeatureFrame,
            _: &[FeatureId],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<MaterializedColumns> {
            panic!("download fixture must not materialize")
        }

        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("download fixture must not evaluate evidence")
        }

        fn download(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            source: &MaterializedColumns,
            max_bytes: usize,
        ) -> SemanticResult<MaterializedColumns> {
            self.calls += 1;
            self.output_budget = Some(max_bytes);
            let columns = source
                .resident_slots()?
                .keys()
                .map(|&feature| (feature, NumericColumn::from(vec![1.0f32; frame.rows()])))
                .collect();
            MaterializedColumns::from_downloaded(registry, frame, GAFIME_BACKEND_CUDA, columns)
        }
    }

    #[test]
    fn explicit_download_is_budgeted_host_output_and_cannot_reenter_cuda() {
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
        let resident = MaterializedColumns::from_resident(
            &registry,
            &frame,
            GAFIME_BACKEND_CUDA,
            BTreeMap::from([(source, 0)]),
            8,
            Arc::new(()),
        )
        .unwrap();
        let session = SemanticSession::new(registry, GAFIME_BACKEND_CUDA, 64).unwrap();
        let mut executor = DownloadOnlyCuda {
            calls: 0,
            output_budget: None,
        };

        let downloaded = session
            .download_materialization(&mut executor, &frame, &resident)
            .unwrap();
        assert_eq!(executor.calls, 1);
        assert_eq!(executor.output_budget, Some(56));
        assert_eq!(downloaded.get(source).unwrap(), &[1.0, 1.0]);
        assert!(!downloaded.is_resident());

        assert!(session
            .download_materialization(&mut executor, &frame, &downloaded)
            .is_err());
        assert_eq!(executor.calls, 1);
    }

    #[test]
    fn lineage_snapshot_admission_rejects_shared_descendants_before_native_execution() {
        let frame = Arc::new(
            FeatureFrame::new(
                vec!["a".into(), "b".into()],
                "training".into(),
                vec![0, 1, 2, 3],
                EvaluationRole::Discovery,
                "lineage fixture".into(),
                vec![vec![-2.0, -1.0, 1.0, 2.0], vec![-3.0, -1.0, 1.0, 3.0]],
            )
            .unwrap(),
        );
        let mut registry = CandidateRegistry::new(
            frame.schema().to_vec(),
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let fitted = registry
            .centered_product(vec![a, b], vec![0.0, 0.0])
            .unwrap();
        // Model many independently fitted frames reaching this exact frozen
        // state. The centered branches below are shared descendants, not eight
        // independent origin ledgers.
        for index in 0..16 {
            registry
                .attach_training_binding(
                    &[fitted],
                    Arc::new(TrainingBinding::from_discovery_frame(
                        10_000 + index,
                        Arc::from("training"),
                        Arc::<str>::from(format!("same-state refit {index}")),
                    )),
                )
                .unwrap();
        }
        let descendants = (0..8)
            .map(|index| registry.centered_product(vec![fitted, a], vec![0.0, index as f32 - 4.0]))
            .collect::<SemanticResult<Vec<_>>>()
            .unwrap();

        let mut limits = SessionLimits::for_budget(1 << 20);
        // Numeric dependency/evidence work is 112 here; the remaining 18 is
        // deliberately insufficient for the preflighted 8 x 16 lineage copy.
        limits.max_work = 130;
        let mut session =
            SemanticSession::with_limits(registry, GAFIME_BACKEND_CPU, limits).unwrap();
        let declared = {
            let mut round = session.begin_round(&[]).unwrap();
            let a = round.source(0).unwrap();
            let b = round.source(1).unwrap();
            let fitted = round.centered_product(vec![a, b], vec![0.0, 0.0]).unwrap();
            (0..8)
                .map(|index| round.centered_product(vec![fitted, a], vec![0.0, index as f32 - 4.0]))
                .collect::<SemanticResult<Vec<_>>>()
                .unwrap()
        };
        assert_eq!(declared, descendants);
        let channel = EvidenceChannel::new(
            "reference".into(),
            EvidenceDefinition::Association {
                statistic: AssociationStatistic::Pearson,
                context: AssociationContext::Reference {
                    reference: session.registry().unwrap().source(0).unwrap(),
                },
            },
        )
        .unwrap();

        let error = match session.evaluate(
            &mut MustNotExecute,
            frame,
            &declared,
            std::slice::from_ref(&channel),
        ) {
            Err(error) => error,
            Ok(_) => panic!("lineage admission must reject before native materialization"),
        };
        assert_eq!(
            error,
            SemanticError::Invalid("semantic fitting lineage metadata work limit exceeded")
        );
        assert_eq!(session.retained_bytes(), 0);
        assert_eq!(session.registry().unwrap().training_binding_count(), 16);
        assert_eq!(
            session
                .registry()
                .unwrap()
                .training_lineage(declared[0])
                .unwrap()
                .len(),
            16
        );
    }
}
