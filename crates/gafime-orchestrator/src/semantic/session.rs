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

    /// True when every stored feature is among `allowed`.  This stays local to
    /// the orchestration boundary: native executors never receive a mutable
    /// candidate identity map and cannot use this as a second semantic
    /// registry.
    fn contains_only(&self, allowed: &BTreeSet<FeatureId>) -> bool {
        match &self.storage {
            MaterializedStorage::Host(columns) => columns.keys().all(|id| allowed.contains(id)),
            MaterializedStorage::Resident { slots, .. } => {
                slots.keys().all(|id| allowed.contains(id))
            }
        }
    }
}

/// One atomic compact-evidence result.  A compact executor may return only
/// dependency columns instead of a dense column for every evaluated candidate,
/// but it must return evidence for every requested channel in the same call.
/// Candidate identity, context validation, selection, and later chosen-column
/// materialization remain owned by the session.
pub struct CompactEvidenceBatch {
    dependencies: MaterializedColumns,
    channel_values: Vec<Vec<EvidenceValue>>,
    /// Conservative executor-owned peak for the compact call, including its
    /// returned dependency bank and any persistent local query state it keeps
    /// live after returning.  It excludes the session's separately retained
    /// bank, which the caller accounts before invoking the hook.
    explicit_peak_bytes: usize,
}

impl CompactEvidenceBatch {
    /// Construct a local executor result.  The session validates context,
    /// feature ownership, channel shape, finiteness, and the reported peak
    /// before it records any evidence.
    pub fn new(
        dependencies: MaterializedColumns,
        channel_values: Vec<Vec<EvidenceValue>>,
        explicit_peak_bytes: usize,
    ) -> Self {
        Self {
            dependencies,
            channel_values,
            explicit_peak_bytes,
        }
    }

    pub fn dependencies(&self) -> &MaterializedColumns {
        &self.dependencies
    }

    pub fn channel_values(&self) -> &[Vec<EvidenceValue>] {
        &self.channel_values
    }

    pub const fn explicit_peak_bytes(&self) -> usize {
        self.explicit_peak_bytes
    }
}

/// Arithmetic-only coordinates for an optional Pareto-frontier lowering.
///
/// Coordinates are physical, column-major fp32 values: axis `d` and candidate
/// row `r` is stored at `d * candidate_count + r`.  Rust has already oriented
/// every axis so smaller is no worse, but does not expose the original evidence
/// channel, candidate identity, policy, or acceptance authority to the
/// executor.  `None` deliberately carries no f64/mixed substitute; an explicit
/// local route can use it to reject an unsupported profile before any numeric
/// narrowing or coordinate allocation.
#[derive(Clone, Copy, Debug)]
pub struct ParetoFrontierRequest<'a> {
    profile: PrecisionProfile,
    objective_count: usize,
    candidate_count: usize,
    fp32_coordinates: Option<&'a [f32]>,
}

impl<'a> ParetoFrontierRequest<'a> {
    pub(crate) const fn new(
        profile: PrecisionProfile,
        objective_count: usize,
        candidate_count: usize,
        fp32_coordinates: Option<&'a [f32]>,
    ) -> Self {
        Self {
            profile,
            objective_count,
            candidate_count,
            fp32_coordinates,
        }
    }

    pub const fn profile(&self) -> PrecisionProfile {
        self.profile
    }

    pub const fn objective_count(&self) -> usize {
        self.objective_count
    }

    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    pub const fn fp32_coordinates(&self) -> Option<&'a [f32]> {
        self.fp32_coordinates
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

    /// Optionally evaluate a complete compact batch before the ordinary dense
    /// candidate materialization route.  `Some` is atomic over `channels`:
    /// an executor must not return a mixture of compact and ordinary channel
    /// values.  Returning `None` preserves the established materialize then
    /// evaluate path exactly.  This is a local Rust seam, not a semantic ABI,
    /// feature IR, target protocol, or backend policy catalog.
    fn evaluate_compact(
        &mut self,
        _registry: &CandidateRegistry,
        _frame: &FeatureFrame,
        _candidates: &[FeatureId],
        _channels: &[EvidenceChannel],
        _retained: Option<&MaterializedColumns>,
        _max_bytes: usize,
    ) -> SemanticResult<Option<CompactEvidenceBatch>> {
        Ok(None)
    }

    /// True only for an explicitly selected local arithmetic route that can
    /// answer a bounded Pareto frontier.  Keeping the default false means the
    /// ordinary Core and GPU paths do not construct a coordinate buffer merely
    /// to discover that no local RT query exists.
    fn wants_pareto_frontier(&self) -> bool {
        false
    }

    /// Optionally count weak dominators for physical Pareto coordinates.
    ///
    /// Returned count `i` includes candidate `i` and every exact-equal
    /// coordinate vector.  Rust alone subtracts that equal-vector cardinality,
    /// determines strict dominance, ranks the frontier, and accepts features.
    /// The request deliberately has no evidence IDs, feature IDs, directions,
    /// or selection policy.  A local executor that negotiated this route must
    /// return `Some` or a fail-closed error; `None` is the default seam for
    /// ordinary executors.
    fn pareto_weak_dominator_counts(
        &mut self,
        _request: ParetoFrontierRequest<'_>,
        _max_bytes: usize,
    ) -> SemanticResult<Option<Vec<u64>>> {
        Ok(None)
    }

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

    /// Declare one target-free count of distinct frozen decision-region
    /// memberships.  Region identity and dependency validation remain in the
    /// canonical registry; this scope only enforces current-round authority.
    pub fn region_count(&mut self, regions: Vec<FeatureId>) -> SemanticResult<FeatureId> {
        for &region in &regions {
            self.operand(region)?;
        }
        let id = self.registry.region_count(regions)?;
        self.eligible.insert(id);
        Ok(id)
    }

    /// Declare one profile-native weighted sum of distinct canonical decision
    /// regions.  The registry owns canonicalization and exact frozen bits;
    /// this round only enforces the same current-round operand authority as
    /// [`Self::region_count`].
    pub fn region_weighted_sum(
        &mut self,
        regions: Vec<FeatureId>,
        weights: Vec<f32>,
    ) -> SemanticResult<FeatureId> {
        for &region in &regions {
            self.operand(region)?;
        }
        let id = self.registry.region_weighted_sum(regions, weights)?;
        self.eligible.insert(id);
        Ok(id)
    }

    /// fp64 counterpart of [`Self::region_weighted_sum`].  It remains a
    /// declaration forwarder and does not introduce an fp32 intermediate.
    pub fn region_weighted_sum_f64(
        &mut self,
        regions: Vec<FeatureId>,
        weights: Vec<f64>,
    ) -> SemanticResult<FeatureId> {
        for &region in &regions {
            self.operand(region)?;
        }
        let id = self.registry.region_weighted_sum_f64(regions, weights)?;
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
            channel
                .definition()
                .validate_candidates(registry, &candidates)?;
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
        let compact =
            executor.evaluate_compact(registry, &frame, &candidates, channels, retained, budget)?;
        let (materialized, channel_values) = if let Some(compact) = compact {
            let allowed = dependency_ids(registry, &roots)?;
            validate_compact_batch(
                registry,
                &frame,
                self.backend,
                CompactBatchRequest {
                    candidates: &candidates,
                    channels,
                    allowed_dependencies: &allowed,
                    budget,
                },
                &compact,
            )?;
            (compact.dependencies, compact.channel_values)
        } else {
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
                validate_evidence_values(
                    &values,
                    candidates.len(),
                    frame.profile(),
                    channel.definition(),
                    &frame,
                )?;
                channel_values.push(values);
            }
            (materialized, channel_values)
        };
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
        let selected = if !policy.pareto_objectives.is_empty() && executor.wants_pareto_frontier() {
            // The evidence dependency bank and same-frame retained bank remain
            // live throughout the optional local frontier query. Reserve both
            // before arithmetic allocates coordinates, query state, or counts.
            let frontier_budget = self
                .limits
                .max_bytes
                .checked_sub(table.materialized.bytes())
                .and_then(|remaining| {
                    remaining.checked_sub(
                        self.cache
                            .as_ref()
                            .filter(|cache| cache.frame_id == table.frame.id())
                            .map_or(0, MaterializedColumns::bytes),
                    )
                })
                .ok_or(SemanticError::Invalid(
                    "local RT Pareto dependencies and retained values exceed session budget",
                ))?;
            self.select_for_accept_with_executor(executor, table, policy, frontier_budget)?
        } else {
            self.select_for_accept(table, policy)?
        };
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let registry = self.registry()?;
        let prior = self
            .cache
            .as_ref()
            .filter(|cache| cache.frame_id == table.frame.id());
        let deferred = selected
            .iter()
            .any(|&feature| !table.materialized.contains(feature));
        let selected_materialized;
        let source = if deferred {
            // The evidence table remains caller-owned and live through this
            // call.  Reserve its dependency bank and any prior accepted bank
            // before asking the backend for the selected dense columns.
            let materialize_budget = self
                .limits
                .max_bytes
                .checked_sub(table.materialized.bytes())
                .and_then(|remaining| {
                    remaining.checked_sub(prior.map_or(0, MaterializedColumns::bytes))
                })
                .ok_or(SemanticError::Invalid(
                    "compact evidence dependencies and retained values exceed session budget",
                ))?;
            let dependencies =
                if table.materialized.is_resident() && table.materialized.bytes() == 0 {
                    None
                } else {
                    Some(&table.materialized)
                };
            selected_materialized = executor.materialize(
                registry,
                &table.frame,
                &selected,
                dependencies,
                materialize_budget,
            )?;
            validate_output(
                registry,
                &table.frame,
                self.backend,
                &selected,
                &selected_materialized,
                materialize_budget,
            )?;
            &selected_materialized
        } else {
            &table.materialized
        };
        // A deferred selected bank is distinct from the compact dependency
        // bank retained by the table, so reserve that table bank once before
        // the backend accounts source, old retention and its output.  The
        // established full-materialization path keeps its exact old budget.
        let retain_budget = if deferred {
            self.limits
                .max_bytes
                .checked_sub(table.materialized.bytes())
                .ok_or(SemanticError::Invalid(
                    "compact evidence dependency bank exceeds session budget",
                ))?
        } else {
            self.limits.max_bytes
        };
        let retained = executor.retain(
            registry,
            &table.frame,
            source,
            prior,
            &selected,
            retain_budget,
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
        if selected
            .iter()
            .any(|&feature| !table.materialized.contains(feature))
        {
            return Err(SemanticError::Unsupported(
                "compact evidence tables with deferred selected columns require accept_with",
            ));
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

    fn select_for_accept_with_executor(
        &self,
        executor: &mut dyn NativeEvidenceExecutor,
        table: &EvidenceTable,
        policy: &SelectionPolicy,
        max_bytes: usize,
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
        policy.select_with_executor(executor, table, self.limits.max_work, max_bytes)
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

fn validate_evidence_values(
    values: &[EvidenceValue],
    expected: usize,
    profile: PrecisionProfile,
    definition: &EvidenceDefinition,
    frame: &FeatureFrame,
) -> SemanticResult<()> {
    let support_bound = if let Some(labels) = definition.labels() {
        labels.as_ref().map_or(0, |labels| labels.rows().len())
    } else if let Some(graph) = definition.graph() {
        graph.edges().len()
    } else {
        frame.rows()
    };
    let missing_labels = matches!(definition.labels(), Some(None));
    if values.len() != expected
        || values.iter().any(|value| match value {
            EvidenceValue::Measured { value, support } => {
                missing_labels
                    || *support > support_bound
                    || !value.is_finite()
                    || (profile == PrecisionProfile::Fp32 && f64::from(*value as f32) != *value)
            }
            EvidenceValue::Unavailable { reason, support } => {
                *support > support_bound
                    || (missing_labels
                        && (*reason != super::UnavailableReason::MissingLabels || *support != 0))
                    || (*reason == super::UnavailableReason::MissingLabels && !missing_labels)
            }
        })
    {
        return Err(SemanticError::Invalid(
            "native evidence output violates shape, support, or missing-label semantics",
        ));
    }
    Ok(())
}

struct CompactBatchRequest<'a> {
    candidates: &'a [FeatureId],
    channels: &'a [EvidenceChannel],
    allowed_dependencies: &'a BTreeSet<FeatureId>,
    budget: usize,
}

fn validate_compact_batch(
    registry: &CandidateRegistry,
    frame: &FeatureFrame,
    backend: BackendKind,
    request: CompactBatchRequest<'_>,
    compact: &CompactEvidenceBatch,
) -> SemanticResult<()> {
    let dependencies = &compact.dependencies;
    if dependencies.frame_id != frame.id()
        || dependencies.profile != frame.profile()
        || dependencies.backend != backend
        || (backend != GAFIME_BACKEND_CPU && !dependencies.is_resident())
        || dependencies.bytes() > request.budget
        || compact.explicit_peak_bytes < dependencies.bytes()
        || compact.explicit_peak_bytes > request.budget
        || !dependencies.contains_only(request.allowed_dependencies)
    {
        return Err(SemanticError::Invalid(
            "compact evidence dependencies or peak violate session context, ownership, or budget",
        ));
    }
    // Recheck every output identity after the native call.  A native executor
    // receives registry references only for lowering; it never gains the
    // authority to introduce a feature outside this evaluated dependency DAG.
    for &id in request.allowed_dependencies {
        registry.program(id)?;
    }
    if compact.channel_values.len() != request.channels.len() {
        return Err(SemanticError::Invalid(
            "compact evidence must return every requested channel atomically",
        ));
    }
    for (channel, values) in request.channels.iter().zip(&compact.channel_values) {
        validate_evidence_values(
            values,
            request.candidates.len(),
            frame.profile(),
            channel.definition(),
            frame,
        )?;
    }
    Ok(())
}

fn dependency_ids(
    registry: &CandidateRegistry,
    roots: &[FeatureId],
) -> SemanticResult<BTreeSet<FeatureId>> {
    let mut ids = BTreeSet::new();
    let mut pending = roots.to_vec();
    while let Some(id) = pending.pop() {
        if !ids.insert(id) {
            continue;
        }
        match registry.program(id)?.op() {
            FeatureOp::Source(_) => {}
            FeatureOp::AbsoluteDifference(left, right) => pending.extend([*left, *right]),
            FeatureOp::Softsign(input) | FeatureOp::HardPredicate { input, .. } => {
                pending.push(*input)
            }
            FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
            FeatureOp::DecisionRegion { terms } => pending.extend(terms),
            FeatureOp::RegionCount { regions } | FeatureOp::RegionWeightedSum { regions, .. } => {
                pending.extend(regions)
            }
        }
    }
    Ok(ids)
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
            FeatureOp::RegionCount { regions } | FeatureOp::RegionWeightedSum { regions, .. } => {
                pending.extend(regions)
            }
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

    #[derive(Default)]
    struct CompactLifecycleMock {
        compact_calls: usize,
        materialize_calls: usize,
        retain_calls: usize,
        last_materialize_budget: Option<usize>,
        last_retain_budget: Option<usize>,
    }

    impl NativeEvidenceExecutor for CompactLifecycleMock {
        fn backend_kind(&self) -> u32 {
            GAFIME_BACKEND_CPU
        }

        fn materialize(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            candidates: &[FeatureId],
            retained: Option<&MaterializedColumns>,
            max_bytes: usize,
        ) -> SemanticResult<MaterializedColumns> {
            self.materialize_calls += 1;
            self.last_materialize_budget = Some(max_bytes);
            assert!(
                retained.is_some(),
                "deferred selected materialization receives dependencies"
            );
            let columns = candidates
                .iter()
                .map(|&candidate| (candidate, NumericColumn::from(vec![0.0f32, 0.0, 1.0, 1.0])))
                .collect();
            MaterializedColumns::from_columns(registry, frame, columns)
        }

        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("an atomic compact result must not fall through per-channel evaluation")
        }

        fn evaluate_compact(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            candidates: &[FeatureId],
            channels: &[EvidenceChannel],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Option<CompactEvidenceBatch>> {
            self.compact_calls += 1;
            let source = registry.source(0)?;
            let dependencies = MaterializedColumns::from_columns(
                registry,
                frame,
                BTreeMap::from([(source, frame.column_typed(0)?.shared_clone())]),
            )?;
            let peak = dependencies.bytes();
            Ok(Some(CompactEvidenceBatch::new(
                dependencies,
                channels
                    .iter()
                    .map(|_| vec![EvidenceValue::measured(1.0, frame.rows()); candidates.len()])
                    .collect(),
                peak,
            )))
        }

        fn retain(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            source: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            selected: &[FeatureId],
            max_live_bytes: usize,
        ) -> SemanticResult<MaterializedColumns> {
            self.retain_calls += 1;
            self.last_retain_budget = Some(max_live_bytes);
            let source = source.columns()?;
            let columns = selected
                .iter()
                .map(|&candidate| {
                    Ok((
                        candidate,
                        source
                            .get(&candidate)
                            .ok_or(SemanticError::Invalid(
                                "deferred selected candidate was not materialized",
                            ))?
                            .shared_clone(),
                    ))
                })
                .collect::<SemanticResult<BTreeMap<_, _>>>()?;
            MaterializedColumns::from_columns(registry, frame, columns)
        }
    }

    fn compact_fixture() -> (
        Arc<FeatureFrame>,
        SemanticSession,
        FeatureId,
        EvidenceChannel,
    ) {
        let frame = Arc::new(
            FeatureFrame::new(
                vec!["x".into()],
                "compact-rows".into(),
                vec![0, 1, 2, 3],
                EvaluationRole::Discovery,
                "compact fixture".into(),
                vec![vec![-1.0, 0.0, 1.0, 2.0]],
            )
            .unwrap(),
        );
        let registry = CandidateRegistry::new(
            frame.schema().to_vec(),
            PrecisionProfile::Mixed,
            ProgramLimits::default(),
        )
        .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 256).unwrap();
        let region = {
            let mut round = session.begin_round(&[]).unwrap();
            let source = round.source(0).unwrap();
            let predicate = round
                .hard_predicate(source, PredicateComparator::GreaterThan, 0.0)
                .unwrap();
            round.decision_region(vec![predicate]).unwrap()
        };
        let channel =
            EvidenceChannel::new("occupancy".into(), EvidenceDefinition::BinaryOccupancy).unwrap();
        (frame, session, region, channel)
    }

    #[test]
    fn compact_batch_defers_selected_dense_materialization_to_accept_with() {
        let (frame, mut session, region, channel) = compact_fixture();
        let mut executor = CompactLifecycleMock::default();
        let table = session
            .evaluate(
                &mut executor,
                Arc::clone(&frame),
                &[region],
                std::slice::from_ref(&channel),
            )
            .unwrap();
        assert_eq!(executor.compact_calls, 1);
        assert_eq!(executor.materialize_calls, 0);
        assert!(!table.materialized.contains(region));
        assert!(table
            .materialized
            .contains(session.registry().unwrap().source(0).unwrap()));

        let policy = SelectionPolicy {
            primary: channel.id(),
            pareto_objectives: Vec::new(),
            direction: crate::semantic::Direction::Maximize,
            constraints: Vec::new(),
            missing: crate::semantic::MissingEvidence::Error,
            limit: 1,
        };
        assert_eq!(
            session.accept(&table, &policy).unwrap_err(),
            SemanticError::Unsupported(
                "compact evidence tables with deferred selected columns require accept_with"
            )
        );
        let accepted = session.accept_with(&mut executor, &table, &policy).unwrap();
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].feature(), region);
        assert_eq!(executor.materialize_calls, 1);
        assert_eq!(executor.retain_calls, 1);
        // Compact dependencies consume 16 bytes; the selected materialization
        // therefore receives the remaining 240-byte budget. Retention sees
        // that dependency bank reserved once rather than an unbounded call.
        assert_eq!(executor.last_materialize_budget, Some(240));
        assert_eq!(executor.last_retain_budget, Some(240));
    }

    struct MalformedCompact;

    impl NativeEvidenceExecutor for MalformedCompact {
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
            panic!("malformed compact output must reject before ordinary materialization")
        }

        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("malformed compact output must reject atomically")
        }

        fn evaluate_compact(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            _: &[FeatureId],
            _: &[EvidenceChannel],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Option<CompactEvidenceBatch>> {
            let source = registry.source(0)?;
            let dependencies = MaterializedColumns::from_columns(
                registry,
                frame,
                BTreeMap::from([(source, frame.column_typed(0)?.shared_clone())]),
            )?;
            // Deliberately omit the required channel vector.  The session must
            // reject this rather than silently falling back to ordinary work.
            Ok(Some(CompactEvidenceBatch::new(
                dependencies,
                Vec::new(),
                16,
            )))
        }
    }

    #[test]
    fn compact_batch_requires_all_channels_atomically() {
        let (frame, mut session, region, channel) = compact_fixture();
        let error = match session.evaluate(
            &mut MalformedCompact,
            frame,
            &[region],
            std::slice::from_ref(&channel),
        ) {
            Ok(_) => panic!("malformed compact batch must fail closed"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            SemanticError::Invalid(
                "compact evidence must return every requested channel atomically"
            )
        );
    }

    struct InvalidCompactValue {
        value: EvidenceValue,
        peak: usize,
    }

    impl NativeEvidenceExecutor for InvalidCompactValue {
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
            panic!("invalid compact output must reject before ordinary materialization")
        }

        fn evaluate_channel(
            &mut self,
            _: &EvidenceDefinition,
            _: &[FeatureId],
            _: &MaterializedColumns,
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Vec<EvidenceValue>> {
            panic!("invalid compact output must reject atomically")
        }

        fn evaluate_compact(
            &mut self,
            registry: &CandidateRegistry,
            frame: &FeatureFrame,
            _: &[FeatureId],
            _: &[EvidenceChannel],
            _: Option<&MaterializedColumns>,
            _: usize,
        ) -> SemanticResult<Option<CompactEvidenceBatch>> {
            let source = registry.source(0)?;
            let dependencies = MaterializedColumns::from_columns(
                registry,
                frame,
                BTreeMap::from([(source, frame.column_typed(0)?.shared_clone())]),
            )?;
            Ok(Some(CompactEvidenceBatch::new(
                dependencies,
                vec![vec![self.value]],
                self.peak,
            )))
        }
    }

    #[test]
    fn compact_batch_bounds_support_and_cannot_fabricate_missing_labels() {
        let (frame, mut session, region, channel) = compact_fixture();
        let error = match session.evaluate(
            &mut InvalidCompactValue {
                value: EvidenceValue::measured(1.0, frame.rows() + 1),
                peak: 16,
            },
            Arc::clone(&frame),
            &[region],
            std::slice::from_ref(&channel),
        ) {
            Err(error) => error,
            Ok(_) => panic!("oversized compact support must fail closed"),
        };
        assert_eq!(
            error,
            SemanticError::Invalid(
                "native evidence output violates shape, support, or missing-label semantics"
            )
        );

        let (frame, mut session, region, _) = compact_fixture();
        let missing = EvidenceChannel::new(
            "missing".into(),
            EvidenceDefinition::BinaryLabeledGiniGain { labels: None },
        )
        .unwrap();
        let error = match session.evaluate(
            &mut InvalidCompactValue {
                value: EvidenceValue::measured(0.0, 0),
                peak: 16,
            },
            frame,
            &[region],
            std::slice::from_ref(&missing),
        ) {
            Err(error) => error,
            Ok(_) => panic!("missing-label compact evidence must not be fabricated"),
        };
        assert_eq!(
            error,
            SemanticError::Invalid(
                "native evidence output violates shape, support, or missing-label semantics"
            )
        );
    }

    #[test]
    fn compact_batch_rejects_an_unbudgeted_executor_peak() {
        let (frame, mut session, region, channel) = compact_fixture();
        let error = match session.evaluate(
            &mut InvalidCompactValue {
                value: EvidenceValue::measured(1.0, frame.rows()),
                // Evaluation reserves half of this 256-byte session for
                // a compact call; 129 must not hide a persistent query.
                peak: 129,
            },
            frame,
            &[region],
            std::slice::from_ref(&channel),
        ) {
            Err(error) => error,
            Ok(_) => panic!("unbudgeted compact peak must fail closed"),
        };
        assert_eq!(
            error,
            SemanticError::Invalid(
                "compact evidence dependencies or peak violate session context, ownership, or budget"
            )
        );
    }
}
