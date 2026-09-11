//! Canonical, target-free candidate programs for the first semantic vertical
//! slice.  This registry deliberately has no ABI or Python representation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use gafime_types::PrecisionProfile;

use super::{SemanticError, SemanticResult};

static NEXT_REGISTRY_TOKEN: AtomicU64 = AtomicU64::new(1);

// These are internal work bounds, not a serialized candidate-IR contract. They
// keep the registry's metadata sets and copied operator state bounded before
// callers can trigger their allocation paths. The node cap also stays far below
// the `u32` feature-id slot representation.
const MAX_PROGRAM_NODES: usize = 65_536;
const MAX_PROGRAM_ARITY: usize = 64;
const MAX_PROGRAM_DEPTH: usize = 64;
// Provenance is metadata rather than numeric materialization, so it has an
// independent admission ceiling.  This caps all Arc entries copied into one
// evidence snapshot and prevents a shared fitted ancestor from multiplying
// report storage by every descendant.
const MAX_TRAINING_LINEAGE_SNAPSHOT_BINDINGS: usize = 65_536;
/// A region may carry two predicates for one semantic atom (for example an
/// open lower and closed upper interval), so this independent physical-term
/// limit must not be conflated with logical or source arity.
pub const MAX_REGION_TERMS: usize = 64;

/// An opaque identity owned by one [`CandidateRegistry`].
///
/// The slot is meaningful only together with its registry token.  Constructors
/// remain private so callers cannot turn legacy execution ordinals into
/// semantic identities.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FeatureId {
    registry: u64,
    slot: u32,
}

/// Independent structural bounds for a semantic candidate DAG.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramLimits {
    /// Maximum source and derived programs held by one registry.
    pub max_nodes: usize,
    /// Maximum number of distinct immediate inputs to one derived program.
    pub max_logical_arity: usize,
    /// Maximum number of unique transitive source columns for one program.
    pub max_source_arity: usize,
    /// Maximum number of derived edges from a source program.
    pub max_depth: usize,
}

impl Default for ProgramLimits {
    fn default() -> Self {
        Self {
            max_nodes: 65_536,
            max_logical_arity: 8,
            max_source_arity: 8,
            max_depth: 8,
        }
    }
}

impl ProgramLimits {
    fn validate(self, source_count: usize) -> SemanticResult<()> {
        if self.max_nodes == 0 {
            return Err(SemanticError::Invalid(
                "semantic candidate node limit must be non-zero",
            ));
        }
        if self.max_logical_arity == 0 {
            return Err(SemanticError::Invalid(
                "semantic logical arity limit must be non-zero",
            ));
        }
        if self.max_source_arity == 0 {
            return Err(SemanticError::Invalid(
                "semantic source arity limit must be non-zero",
            ));
        }
        if self.max_nodes > MAX_PROGRAM_NODES {
            return Err(SemanticError::Unsupported(
                "semantic candidate node limit exceeds the bounded vertical slice",
            ));
        }
        if self.max_logical_arity > MAX_PROGRAM_ARITY {
            return Err(SemanticError::Unsupported(
                "semantic logical arity limit exceeds the bounded vertical slice",
            ));
        }
        if self.max_source_arity > MAX_PROGRAM_ARITY {
            return Err(SemanticError::Unsupported(
                "semantic source arity limit exceeds the bounded vertical slice",
            ));
        }
        if self.max_depth > MAX_PROGRAM_DEPTH {
            return Err(SemanticError::Unsupported(
                "semantic depth limit exceeds the bounded vertical slice",
            ));
        }
        if source_count > self.max_nodes {
            return Err(SemanticError::Unsupported(
                "source schema exceeds semantic candidate node limit",
            ));
        }
        Ok(())
    }
}

/// Exact frozen centering constants bound to one candidate profile.
///
/// The raw IEEE bit pattern participates in candidate identity. In particular,
/// signed zero and adjacent f64 values must not be collapsed by a registry.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FrozenMeans {
    F32(Vec<u32>),
    F64(Vec<u64>),
}

/// Exact finite weights for a canonical weighted sum of decision regions.
///
/// The raw IEEE bit pattern is part of candidate identity.  In particular,
/// signed zero and adjacent finite values must not be silently quantized,
/// coalesced, or converted through another precision profile.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FrozenRegionWeights {
    F32(Vec<u32>),
    F64(Vec<u64>),
}

/// Exact threshold storage for a hard predicate.  Like frozen means, the raw
/// IEEE bits are part of the mathematical candidate identity; fitting history
/// is deliberately kept outside [`FeatureOp`] so later context cannot fork or
/// rewrite that identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FrozenThreshold {
    F32(u32),
    F64(u64),
}

impl FrozenThreshold {
    pub fn as_f32_bits(&self) -> SemanticResult<u32> {
        match self {
            Self::F32(bits) => Ok(*bits),
            Self::F64(_) => Err(SemanticError::Invalid("frozen threshold is not f32 bits")),
        }
    }

    pub fn as_f64_bits(&self) -> SemanticResult<u64> {
        match self {
            Self::F32(_) => Err(SemanticError::Invalid("frozen threshold is not f64 bits")),
            Self::F64(bits) => Ok(*bits),
        }
    }
}

/// Exact hard-predicate relation.  The two relations form a deterministic
/// partition for finite values without inventing epsilon semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PredicateComparator {
    LessEqual,
    GreaterThan,
}

/// Immutable provenance for a state fitted from one discovery snapshot.
///
/// This is intentionally not a candidate-ID component: equal frozen bits and
/// identical program structure remain one reusable candidate even when a
/// later fitting context independently reaches the same state.  The registry,
/// evidence table and accepted feature own snapshots of this record instead.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TrainingBinding {
    frame_id: u64,
    row_domain: Arc<str>,
    provenance: Arc<str>,
}

impl TrainingBinding {
    pub(crate) fn from_discovery_frame(
        frame_id: u64,
        row_domain: Arc<str>,
        provenance: Arc<str>,
    ) -> Self {
        Self {
            frame_id,
            row_domain,
            provenance,
        }
    }

    /// Exact immutable snapshot used to fit this state.
    pub const fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Caller-declared row domain retained for provenance inspection.
    pub fn row_domain(&self) -> &str {
        &self.row_domain
    }

    /// Caller-declared input provenance retained for provenance inspection.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
}

impl FrozenMeans {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(bits) => bits.len(),
            Self::F64(bits) => bits.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_bits(&self) -> SemanticResult<&[u32]> {
        match self {
            Self::F32(bits) => Ok(bits),
            Self::F64(_) => Err(SemanticError::Invalid("frozen means are not f32 bits")),
        }
    }

    pub fn as_f64_bits(&self) -> SemanticResult<&[u64]> {
        match self {
            Self::F32(_) => Err(SemanticError::Invalid("frozen means are not f64 bits")),
            Self::F64(bits) => Ok(bits),
        }
    }
}

impl FrozenRegionWeights {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(bits) => bits.len(),
            Self::F64(bits) => bits.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_bits(&self) -> SemanticResult<&[u32]> {
        match self {
            Self::F32(bits) => Ok(bits),
            Self::F64(_) => Err(SemanticError::Invalid(
                "frozen region weights are not f32 bits",
            )),
        }
    }

    pub fn as_f64_bits(&self) -> SemanticResult<&[u64]> {
        match self {
            Self::F32(_) => Err(SemanticError::Invalid(
                "frozen region weights are not f64 bits",
            )),
            Self::F64(bits) => Ok(bits),
        }
    }
}

/// A target-free, precision-bound candidate operation.
///
/// `CenteredProduct` preserves operand order because sequential multiplication
/// in the pointwise dtype is not generally associative.  Frozen means are
/// caller-declared constants stored as exact profile-bound bits rather than
/// recomputed from a later frame.  Fitted-state provenance is held separately
/// by the registry/session lifecycle so it cannot change canonical identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FeatureOp {
    Source(u32),
    AbsoluteDifference(FeatureId, FeatureId),
    Softsign(FeatureId),
    CenteredProduct {
        operands: Vec<FeatureId>,
        mean_bits: FrozenMeans,
    },
    /// Exact 0/1 membership for one finite semantic atom and frozen threshold.
    HardPredicate {
        input: FeatureId,
        comparison: PredicateComparator,
        threshold_bits: FrozenThreshold,
    },
    /// Canonical flattened hard-AND of [`FeatureOp::HardPredicate`] leaves.
    /// The terms are program identities rather than a second descriptor/IR.
    DecisionRegion {
        terms: Vec<FeatureId>,
    },
    /// Target-free per-row coverage count across distinct canonical decision
    /// regions.  The operand order is canonicalized because integer addition
    /// is commutative before its one profile-native pointwise conversion.
    /// This is a bounded candidate form, not a tree, rule learner, or another
    /// predicate language.
    RegionCount {
        regions: Vec<FeatureId>,
    },
    /// A profile-native, target-free weighted sum across distinct canonical
    /// decision-region memberships.  Regions and weights are stored together
    /// in canonical region-identity order, so each row accumulates from +0
    /// in one declared deterministic order.  This remains distinct from
    /// [`FeatureOp::RegionCount`], including when every weight is one.
    RegionWeightedSum {
        regions: Vec<FeatureId>,
        weight_bits: FrozenRegionWeights,
    },
}

/// One immutable semantic program in a [`CandidateRegistry`].
#[derive(Clone, Debug)]
pub struct FeatureProgram {
    id: FeatureId,
    op: FeatureOp,
    source_dependencies: Vec<u32>,
    logical_arity: usize,
    region_term_count: usize,
    depth: usize,
}

impl FeatureProgram {
    /// Return this registry-owned semantic identity.
    pub const fn id(&self) -> FeatureId {
        self.id
    }

    /// Return the canonical, immutable operation descriptor.
    pub const fn op(&self) -> &FeatureOp {
        &self.op
    }

    /// Return sorted, unique source-column dependencies.
    pub fn source_dependencies(&self) -> &[u32] {
        &self.source_dependencies
    }

    /// Return the number of direct logical inputs (one for a source program).
    pub const fn logical_arity(&self) -> usize {
        self.logical_arity
    }

    /// Return the number of unique transitive source columns.
    pub fn source_arity(&self) -> usize {
        self.source_dependencies.len()
    }

    /// Number of flattened hard-predicate terms, or zero for a non-region
    /// program.  This is intentionally distinct from logical arity: a closed
    /// interval has two terms but only one semantic input atom.
    pub const fn region_term_count(&self) -> usize {
        self.region_term_count
    }

    /// Return canonical flattened predicate leaves for a decision region.
    pub fn decision_region_terms(&self) -> Option<&[FeatureId]> {
        match &self.op {
            FeatureOp::DecisionRegion { terms } => Some(terms),
            _ => None,
        }
    }

    /// Return the canonical region memberships summed by a coverage-count
    /// program.  This remains a program dependency list rather than a second
    /// native semantic descriptor.
    pub fn region_count_regions(&self) -> Option<&[FeatureId]> {
        match &self.op {
            FeatureOp::RegionCount { regions } => Some(regions),
            _ => None,
        }
    }

    /// Return the canonical region memberships and exact profile-native
    /// weights used by a weighted regional sum.  The aligned slices are
    /// ordered by canonical region identity rather than declaration order.
    pub fn region_weighted_sum(&self) -> Option<(&[FeatureId], &FrozenRegionWeights)> {
        match &self.op {
            FeatureOp::RegionWeightedSum {
                regions,
                weight_bits,
            } => Some((regions, weight_bits)),
            _ => None,
        }
    }

    /// Return the frozen relation and threshold carried by one hard predicate.
    /// This is presentation metadata for the canonical program, not a mutable
    /// training-context record.
    pub fn hard_predicate(&self) -> Option<(FeatureId, PredicateComparator, &FrozenThreshold)> {
        match &self.op {
            FeatureOp::HardPredicate {
                input,
                comparison,
                threshold_bits,
            } => Some((*input, *comparison, threshold_bits)),
            _ => None,
        }
    }

    /// Return the number of derived edges from a source program.
    pub const fn depth(&self) -> usize {
        self.depth
    }
}

/// Canonical owner of semantic candidate identities and their bounded DAG.
///
/// A registry is deliberately not cloneable: a clone would fork mutable ID
/// authority.  Programs are immutable and may be cloned by consumers instead.
pub struct CandidateRegistry {
    token: u64,
    source_names: Vec<String>,
    precision: PrecisionProfile,
    limits: ProgramLimits,
    source_ids: Vec<FeatureId>,
    programs: Vec<FeatureProgram>,
    by_operation: BTreeMap<FeatureOp, FeatureId>,
    /// Per-feature, immutable fitting records.  These do not participate in
    /// `by_operation`: fitting the same frozen state in another context adds a
    /// record instead of manufacturing a competing candidate identity.
    training_bindings: BTreeMap<FeatureId, Vec<Arc<TrainingBinding>>>,
    training_binding_count: usize,
}

/// Internal mutation boundary used by bounded bulk declaration. A failed batch
/// never exposes newly allocated identities and may therefore safely reclaim
/// its appended program slots without cloning/forking registry authority.
#[derive(Clone, Copy)]
pub(crate) struct RegistryCheckpoint(usize);

struct DerivedProgramMetadata {
    source_dependencies: Vec<u32>,
    logical_arity: usize,
    region_term_count: usize,
    depth: usize,
}

/// A fully preflighted immutable provenance snapshot.  It remains crate-local:
/// callers observe lineages through evidence tables and accepted features, not
/// through another public lifecycle object.
pub(crate) struct TrainingLineageSnapshot {
    pub(crate) lineages: Vec<Vec<Arc<TrainingBinding>>>,
    pub(crate) metadata_work: usize,
}

impl CandidateRegistry {
    /// Create a profile-bound registry for one named source schema.
    pub fn new(
        source_names: Vec<String>,
        precision: PrecisionProfile,
        limits: ProgramLimits,
    ) -> SemanticResult<Self> {
        if source_names.is_empty() {
            return Err(SemanticError::Invalid(
                "semantic source schema must not be empty",
            ));
        }
        let source_count = u32::try_from(source_names.len()).map_err(|_| {
            SemanticError::Unsupported("source schema exceeds semantic feature-id capacity")
        })?;
        limits.validate(source_names.len())?;
        validate_source_names(&source_names)?;

        let token = allocate_registry_token()?;
        let mut registry = Self {
            token,
            source_names,
            precision,
            limits,
            source_ids: Vec::new(),
            programs: Vec::new(),
            by_operation: BTreeMap::new(),
            training_bindings: BTreeMap::new(),
            training_binding_count: 0,
        };
        for source in 0..source_count {
            let id = FeatureId {
                registry: token,
                slot: source,
            };
            let operation = FeatureOp::Source(source);
            registry.source_ids.push(id);
            registry.programs.push(FeatureProgram {
                id,
                op: operation.clone(),
                source_dependencies: vec![source],
                logical_arity: 1,
                region_term_count: 0,
                depth: 0,
            });
            registry.by_operation.insert(operation, id);
        }
        Ok(registry)
    }

    /// Return the source schema used to construct this registry.
    pub fn schema(&self) -> &[String] {
        &self.source_names
    }

    /// Return the source names used to construct this registry.
    pub fn source_names(&self) -> &[String] {
        self.schema()
    }

    /// Return the selected pointwise precision identity.
    pub const fn precision(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return this registry's immutable structural limits.
    pub const fn limits(&self) -> ProgramLimits {
        self.limits
    }

    /// Return whether an identity belongs to this live registry and slot range.
    pub fn owns(&self, id: FeatureId) -> bool {
        id.registry == self.token
            && usize::try_from(id.slot)
                .ok()
                .is_some_and(|slot| slot < self.programs.len())
    }

    /// Resolve a source-schema position to its semantic identity.
    pub fn source(&self, index: usize) -> SemanticResult<FeatureId> {
        self.source_ids
            .get(index)
            .copied()
            .ok_or(SemanticError::Invalid(
                "source feature index is out of bounds",
            ))
    }

    /// Add or resolve the canonical absolute difference of two programs.
    pub fn abs_difference(
        &mut self,
        left: FeatureId,
        right: FeatureId,
    ) -> SemanticResult<FeatureId> {
        self.program(left)?;
        self.program(right)?;
        let (left, right) = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        self.add_derived(FeatureOp::AbsoluteDifference(left, right), &[left, right])
    }

    /// Add or resolve the canonical softsign of one program.
    pub fn softsign(&mut self, input: FeatureId) -> SemanticResult<FeatureId> {
        self.add_derived(FeatureOp::Softsign(input), &[input])
    }

    /// Add or resolve an ordered f32-storage centered product.
    pub fn centered_product(
        &mut self,
        operands: Vec<FeatureId>,
        frozen_means: Vec<f32>,
    ) -> SemanticResult<FeatureId> {
        if self.precision == PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f32 frozen means do not match an fp64 candidate registry",
            ));
        }
        self.centered_product_from_frozen(
            operands,
            FrozenMeans::F32(frozen_means.into_iter().map(f32::to_bits).collect()),
        )
    }

    /// Add or resolve an ordered f64-storage centered product.
    pub fn centered_product_f64(
        &mut self,
        operands: Vec<FeatureId>,
        frozen_means: Vec<f64>,
    ) -> SemanticResult<FeatureId> {
        if self.precision != PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f64 frozen means require an fp64 candidate registry",
            ));
        }
        self.centered_product_from_frozen(
            operands,
            FrozenMeans::F64(frozen_means.into_iter().map(f64::to_bits).collect()),
        )
    }

    /// Add or resolve a centered product from profile-native frozen bits.  This
    /// is the lifecycle-owned fitting seam; callers cannot use it to change a
    /// program's mathematical identity after construction.
    pub(crate) fn centered_product_from_frozen(
        &mut self,
        operands: Vec<FeatureId>,
        mean_bits: FrozenMeans,
    ) -> SemanticResult<FeatureId> {
        let metadata = self.centered_product_metadata(&operands, mean_bits.len())?;
        self.validate_frozen_means(&mean_bits, operands.len())?;
        self.insert_derived(
            FeatureOp::CenteredProduct {
                operands,
                mean_bits,
            },
            metadata,
        )
    }

    /// Add or resolve a profile-native hard predicate with a caller-declared
    /// frozen f32 threshold.
    pub fn hard_predicate(
        &mut self,
        input: FeatureId,
        comparison: PredicateComparator,
        threshold: f32,
    ) -> SemanticResult<FeatureId> {
        if self.precision == PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f32 frozen threshold does not match an fp64 candidate registry",
            ));
        }
        if !threshold.is_finite() {
            return Err(SemanticError::Invalid(
                "hard predicate frozen threshold must be finite",
            ));
        }
        self.add_derived(
            FeatureOp::HardPredicate {
                input,
                comparison,
                threshold_bits: FrozenThreshold::F32(threshold.to_bits()),
            },
            &[input],
        )
    }

    /// Add or resolve a profile-native hard predicate with a caller-declared
    /// frozen f64 threshold.
    pub fn hard_predicate_f64(
        &mut self,
        input: FeatureId,
        comparison: PredicateComparator,
        threshold: f64,
    ) -> SemanticResult<FeatureId> {
        if self.precision != PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f64 frozen threshold requires an fp64 candidate registry",
            ));
        }
        if !threshold.is_finite() {
            return Err(SemanticError::Invalid(
                "hard predicate frozen threshold must be finite",
            ));
        }
        self.add_derived(
            FeatureOp::HardPredicate {
                input,
                comparison,
                threshold_bits: FrozenThreshold::F64(threshold.to_bits()),
            },
            &[input],
        )
    }

    /// Add or resolve a canonical flattened hard-AND region.  Inputs may be
    /// hard predicates or existing regions; the latter are flattened so
    /// lowering sees one compact term list instead of a second program form.
    pub fn decision_region(&mut self, terms: Vec<FeatureId>) -> SemanticResult<FeatureId> {
        let terms = self.flatten_region_terms(&terms)?;
        let metadata = self.decision_region_metadata(&terms)?;
        self.insert_derived(FeatureOp::DecisionRegion { terms }, metadata)
    }

    /// Add or resolve a canonical target-free coverage count across frozen
    /// decision-region memberships.  One region is intentionally rejected:
    /// it would duplicate an existing binary region under a second candidate
    /// identity rather than establish a new mathematical form.
    pub fn region_count(&mut self, regions: Vec<FeatureId>) -> SemanticResult<FeatureId> {
        let regions = self.canonical_region_count_regions(&regions)?;
        let metadata = self.derived_metadata_from_valid_inputs(&regions)?;
        self.insert_derived(FeatureOp::RegionCount { regions }, metadata)
    }

    /// Add or resolve a canonical weighted sum of distinct frozen decision
    /// regions for an fp32-storage profile.  Weights are finite f32 values,
    /// including signed zero, and remain exact frozen candidate state rather
    /// than a later scoring or selection policy.
    pub fn region_weighted_sum(
        &mut self,
        regions: Vec<FeatureId>,
        weights: Vec<f32>,
    ) -> SemanticResult<FeatureId> {
        if self.precision == PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f32 region weights do not match an fp64 candidate registry",
            ));
        }
        self.region_weighted_sum_from_frozen(
            regions,
            FrozenRegionWeights::F32(weights.into_iter().map(f32::to_bits).collect()),
        )
    }

    /// Add or resolve a canonical weighted sum of distinct frozen decision
    /// regions for an fp64 profile.  No f32 intermediate is introduced into
    /// either the candidate identity or the Core pointwise arithmetic lane.
    pub fn region_weighted_sum_f64(
        &mut self,
        regions: Vec<FeatureId>,
        weights: Vec<f64>,
    ) -> SemanticResult<FeatureId> {
        if self.precision != PrecisionProfile::Fp64 {
            return Err(SemanticError::Invalid(
                "f64 region weights require an fp64 candidate registry",
            ));
        }
        self.region_weighted_sum_from_frozen(
            regions,
            FrozenRegionWeights::F64(weights.into_iter().map(f64::to_bits).collect()),
        )
    }

    /// Resolve one registry-owned identity to its immutable semantic program.
    pub fn program(&self, id: FeatureId) -> SemanticResult<&FeatureProgram> {
        if id.registry != self.token {
            return Err(SemanticError::ForeignIdentity);
        }
        let slot = usize::try_from(id.slot)
            .map_err(|_| SemanticError::Invalid("feature id slot exceeds platform bounds"))?;
        self.programs
            .get(slot)
            .ok_or(SemanticError::Invalid("feature id slot is out of bounds"))
    }

    /// Return a snapshot of direct fitting records for one candidate.  The
    /// empty result means the program has only caller-declared frozen state or
    /// no fitted state at all.
    pub fn training_bindings(&self, id: FeatureId) -> SemanticResult<Vec<Arc<TrainingBinding>>> {
        self.program(id)?;
        Ok(self.training_bindings.get(&id).cloned().unwrap_or_default())
    }

    /// Return a deterministic, deduplicated snapshot of all fitting records
    /// reachable through one candidate program's dependency DAG.
    pub fn training_lineage(&self, id: FeatureId) -> SemanticResult<Vec<Arc<TrainingBinding>>> {
        // A diagnostic query has no session work budget, but it still cannot
        // turn the bounded registry into an unbounded transitive expansion.
        let query_work = self
            .limits
            .max_nodes
            .saturating_mul(MAX_PROGRAM_ARITY.saturating_add(2))
            .saturating_mul(2)
            .saturating_add(MAX_TRAINING_LINEAGE_SNAPSHOT_BINDINGS);
        let mut snapshot = self.snapshot_training_lineages(&[id], query_work)?;
        snapshot
            .lineages
            .pop()
            .ok_or(SemanticError::Invalid("missing training lineage snapshot"))
    }

    /// Preflight every transitive origin before allocating report-facing
    /// vectors.  Each root traversal deduplicates its DAG structurally; the
    /// caller's work ceiling additionally bounds repeated shared-descendant
    /// traversal across roots.  The second pass runs only after the total Arc
    /// snapshot count and both traversal passes are known to fit admission.
    pub(crate) fn snapshot_training_lineages(
        &self,
        roots: &[FeatureId],
        max_work: usize,
    ) -> SemanticResult<TrainingLineageSnapshot> {
        if roots.len() > self.limits.max_nodes {
            return Err(SemanticError::Invalid(
                "semantic fitting lineage metadata work limit exceeded",
            ));
        }
        // Even the zero-origin fast path must preserve opaque registry
        // identity validation for diagnostic callers.
        for &root in roots {
            self.program(root)?;
        }
        // The overwhelmingly common non-fitted path has no reachable origin
        // by construction. Preserve its existing work admission and avoid a
        // gratuitous DAG scan while still returning row-aligned empty records.
        if self.training_binding_count == 0 {
            return Ok(TrainingLineageSnapshot {
                lineages: (0..roots.len()).map(|_| Vec::new()).collect(),
                metadata_work: 0,
            });
        }
        if max_work == 0 {
            return Err(SemanticError::Invalid(
                "semantic fitting lineage metadata work limit exceeded",
            ));
        }
        // We make two deterministic passes: one admission pass, then one to
        // clone Arc handles into table-owned vectors. Reserving half first
        // guarantees the second structural traversal is charged before any
        // report-facing allocation begins.
        let scan_limit = max_work / 2;
        if scan_limit == 0 {
            return Err(SemanticError::Invalid(
                "semantic fitting lineage metadata work limit exceeded",
            ));
        }
        let mut scan_work = 0usize;
        let mut expected_lengths = Vec::with_capacity(roots.len());
        let mut snapshot_entries = 0usize;
        for &root in roots {
            let lineage =
                self.collect_training_lineage_bindings(root, &mut scan_work, scan_limit)?;
            snapshot_entries =
                snapshot_entries
                    .checked_add(lineage.len())
                    .ok_or(SemanticError::Invalid(
                        "semantic fitting lineage snapshot count overflow",
                    ))?;
            if snapshot_entries > MAX_TRAINING_LINEAGE_SNAPSHOT_BINDINGS {
                return Err(SemanticError::Unsupported(
                    "semantic fitting lineage snapshot limit exceeded",
                ));
            }
            expected_lengths.push(lineage.len());
        }
        let metadata_work = scan_work
            .checked_mul(2)
            .and_then(|work| work.checked_add(snapshot_entries))
            .ok_or(SemanticError::Invalid(
                "semantic fitting lineage metadata work overflow",
            ))?;
        if metadata_work > max_work {
            return Err(SemanticError::Invalid(
                "semantic fitting lineage metadata work limit exceeded",
            ));
        }

        let mut build_work = 0usize;
        let mut lineages = Vec::with_capacity(roots.len());
        for (&root, &expected) in roots.iter().zip(&expected_lengths) {
            let lineage =
                self.collect_training_lineage_bindings(root, &mut build_work, scan_limit)?;
            debug_assert_eq!(lineage.len(), expected);
            lineages.push(lineage.into_iter().collect());
        }
        debug_assert_eq!(build_work, scan_work);
        Ok(TrainingLineageSnapshot {
            lineages,
            metadata_work,
        })
    }

    fn collect_training_lineage_bindings(
        &self,
        root: FeatureId,
        work: &mut usize,
        max_work: usize,
    ) -> SemanticResult<BTreeSet<Arc<TrainingBinding>>> {
        let mut pending = vec![root];
        let mut visited = BTreeSet::new();
        let mut lineage = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            charge_training_lineage_work(work, 1, max_work)?;
            let program = self.program(current)?;
            if let Some(bindings) = self.training_bindings.get(&current) {
                charge_training_lineage_work(work, bindings.len(), max_work)?;
                lineage.extend(bindings.iter().cloned());
            }
            match program.op() {
                FeatureOp::Source(_) => {}
                FeatureOp::AbsoluteDifference(left, right) => {
                    charge_training_lineage_work(work, 2, max_work)?;
                    pending.extend([*left, *right]);
                }
                FeatureOp::Softsign(input) | FeatureOp::HardPredicate { input, .. } => {
                    charge_training_lineage_work(work, 1, max_work)?;
                    pending.push(*input);
                }
                FeatureOp::CenteredProduct { operands, .. } => {
                    charge_training_lineage_work(work, operands.len(), max_work)?;
                    pending.extend(operands);
                }
                FeatureOp::DecisionRegion { terms } => {
                    charge_training_lineage_work(work, terms.len(), max_work)?;
                    pending.extend(terms);
                }
                FeatureOp::RegionCount { regions } => {
                    charge_training_lineage_work(work, regions.len(), max_work)?;
                    pending.extend(regions);
                }
                FeatureOp::RegionWeightedSum { regions, .. } => {
                    charge_training_lineage_work(work, regions.len(), max_work)?;
                    pending.extend(regions);
                }
            }
        }
        Ok(lineage)
    }

    /// Number of bounded direct fitting records currently owned by this
    /// registry.  This is diagnostic metadata, not candidate-node count.
    pub const fn training_binding_count(&self) -> usize {
        self.training_binding_count
    }

    /// Attach one immutable training binding to every returned candidate in a
    /// fitted declaration batch.  This is atomic: if the bounded ledger cannot
    /// admit all new records, no candidate gains a partial provenance update.
    pub(crate) fn attach_training_binding(
        &mut self,
        candidates: &[FeatureId],
        binding: Arc<TrainingBinding>,
    ) -> SemanticResult<()> {
        let candidates = candidates.iter().copied().collect::<BTreeSet<_>>();
        for &candidate in &candidates {
            self.program(candidate)?;
        }
        let additions = candidates
            .iter()
            .filter(|candidate| {
                self.training_bindings
                    .get(candidate)
                    .is_none_or(|bindings| {
                        !bindings.iter().any(|old| old.as_ref() == binding.as_ref())
                    })
            })
            .count();
        if self
            .training_binding_count
            .checked_add(additions)
            .is_none_or(|count| count > self.limits.max_nodes)
        {
            return Err(SemanticError::Unsupported(
                "semantic fitting provenance limit exceeded",
            ));
        }
        for candidate in candidates {
            let bindings = self.training_bindings.entry(candidate).or_default();
            if !bindings.iter().any(|old| old.as_ref() == binding.as_ref()) {
                bindings.push(Arc::clone(&binding));
                bindings.sort();
                self.training_binding_count += 1;
            }
        }
        Ok(())
    }

    pub(crate) fn mutation_checkpoint(&self) -> RegistryCheckpoint {
        RegistryCheckpoint(self.programs.len())
    }

    pub(crate) fn rollback_mutations(&mut self, checkpoint: RegistryCheckpoint) {
        debug_assert!(checkpoint.0 <= self.programs.len());
        while self.programs.len() > checkpoint.0 {
            let program = self
                .programs
                .pop()
                .expect("program length was checked before rollback");
            self.by_operation.remove(program.op());
            if let Some(bindings) = self.training_bindings.remove(&program.id()) {
                self.training_binding_count = self
                    .training_binding_count
                    .checked_sub(bindings.len())
                    .expect("training binding count tracks registry entries");
            }
        }
    }

    fn add_derived(
        &mut self,
        operation: FeatureOp,
        inputs: &[FeatureId],
    ) -> SemanticResult<FeatureId> {
        let metadata = self.derived_metadata(inputs)?;
        self.insert_derived(operation, metadata)
    }

    fn centered_product_metadata(
        &self,
        operands: &[FeatureId],
        mean_len: usize,
    ) -> SemanticResult<DerivedProgramMetadata> {
        if operands.len() < 2 {
            return Err(SemanticError::Invalid(
                "centered product requires at least two operands",
            ));
        }
        self.validate_derived_inputs(operands)?;
        if operands.len() != mean_len {
            return Err(SemanticError::Invalid(
                "centered product operands and frozen means must have equal lengths",
            ));
        }
        self.derived_metadata_from_valid_inputs(operands)
    }

    fn validate_frozen_means(&self, means: &FrozenMeans, expected: usize) -> SemanticResult<()> {
        if means.len() != expected {
            return Err(SemanticError::Invalid(
                "centered product operands and frozen means must have equal lengths",
            ));
        }
        match (self.precision, means) {
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
            (PrecisionProfile::Fp32 | PrecisionProfile::Mixed, FrozenMeans::F64(_)) => {
                Err(SemanticError::Invalid(
                    "f64 frozen means do not match the selected candidate profile",
                ))
            }
            (PrecisionProfile::Fp64, FrozenMeans::F32(_)) => Err(SemanticError::Invalid(
                "f32 frozen means do not match an fp64 candidate registry",
            )),
            _ => Err(SemanticError::Invalid(
                "centered product frozen means must be finite",
            )),
        }
    }

    fn flatten_region_terms(&self, terms: &[FeatureId]) -> SemanticResult<Vec<FeatureId>> {
        if terms.is_empty() {
            return Err(SemanticError::Invalid(
                "decision region requires at least one hard predicate",
            ));
        }
        let mut flattened = Vec::new();
        for &term in terms {
            match self.program(term)?.op() {
                FeatureOp::HardPredicate { .. } => flattened.push(term),
                FeatureOp::DecisionRegion { terms } => flattened.extend_from_slice(terms),
                _ => {
                    return Err(SemanticError::Invalid(
                        "decision region terms must be hard predicates or regions",
                    ))
                }
            }
        }
        flattened.sort();
        if flattened.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SemanticError::Invalid(
                "decision region repeats a hard predicate term",
            ));
        }
        if flattened.len() > MAX_REGION_TERMS {
            return Err(SemanticError::Unsupported(
                "decision region exceeds bounded predicate term limit",
            ));
        }
        Ok(flattened)
    }

    fn decision_region_metadata(
        &self,
        terms: &[FeatureId],
    ) -> SemanticResult<DerivedProgramMetadata> {
        debug_assert!(!terms.is_empty());
        let mut atoms = BTreeSet::new();
        let mut source_dependencies = BTreeSet::new();
        let mut deepest = 0usize;
        let mut intervals: BTreeMap<FeatureId, (Option<FrozenThreshold>, Option<FrozenThreshold>)> =
            BTreeMap::new();
        for &term in terms {
            let predicate = self.program(term)?;
            let FeatureOp::HardPredicate {
                input,
                comparison,
                threshold_bits,
            } = predicate.op()
            else {
                return Err(SemanticError::Invalid(
                    "decision region did not flatten to hard predicate terms",
                ));
            };
            let input_program = self.program(*input)?;
            atoms.insert(*input);
            source_dependencies.extend(input_program.source_dependencies().iter().copied());
            deepest = deepest.max(predicate.depth());
            let entry = intervals.entry(*input).or_default();
            let slot = match comparison {
                PredicateComparator::GreaterThan => &mut entry.0,
                PredicateComparator::LessEqual => &mut entry.1,
            };
            if slot.replace(threshold_bits.clone()).is_some() {
                return Err(SemanticError::Invalid(
                    "decision region repeats a same-direction atom bound",
                ));
            }
        }
        if atoms.len() > self.limits.max_logical_arity {
            return Err(SemanticError::Unsupported(
                "decision region exceeds logical atom arity limit",
            ));
        }
        if source_dependencies.len() > self.limits.max_source_arity {
            return Err(SemanticError::Unsupported(
                "semantic program exceeds source arity limit",
            ));
        }
        for (_, (lower, upper)) in intervals {
            if let (Some(lower), Some(upper)) = (lower, upper) {
                let order = self.compare_thresholds(&lower, &upper)?;
                if !order.is_lt() {
                    return Err(SemanticError::Invalid(
                        "decision region contains an empty or inverted interval",
                    ));
                }
            }
        }
        let depth = deepest.checked_add(1).ok_or(SemanticError::Unsupported(
            "semantic program depth overflow",
        ))?;
        if depth > self.limits.max_depth {
            return Err(SemanticError::Unsupported(
                "semantic program exceeds depth limit",
            ));
        }
        Ok(DerivedProgramMetadata {
            source_dependencies: source_dependencies.into_iter().collect(),
            logical_arity: atoms.len(),
            region_term_count: terms.len(),
            depth,
        })
    }

    fn canonical_region_count_regions(
        &self,
        regions: &[FeatureId],
    ) -> SemanticResult<Vec<FeatureId>> {
        if regions.len() < 2 {
            return Err(SemanticError::Invalid(
                "region coverage requires at least two distinct decision regions",
            ));
        }
        if regions.len() > self.limits.max_logical_arity {
            return Err(SemanticError::Unsupported(
                "region coverage exceeds logical arity limit",
            ));
        }
        let mut regions = regions.to_vec();
        regions.sort();
        if regions.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SemanticError::Invalid(
                "region coverage repeats a decision region",
            ));
        }
        for &region in &regions {
            match self.program(region)?.op() {
                FeatureOp::DecisionRegion { .. } => {}
                _ => {
                    return Err(SemanticError::Invalid(
                        "region coverage inputs must be canonical decision regions",
                    ))
                }
            }
        }
        self.validate_derived_inputs(&regions)?;
        Ok(regions)
    }

    fn region_weighted_sum_from_frozen(
        &mut self,
        regions: Vec<FeatureId>,
        weights: FrozenRegionWeights,
    ) -> SemanticResult<FeatureId> {
        let canonical_regions = self.canonical_region_count_regions(&regions)?;
        self.validate_region_weights(&weights, regions.len())?;
        let weight_bits = self.canonical_region_weights(&regions, &canonical_regions, weights)?;
        let metadata = self.derived_metadata_from_valid_inputs(&canonical_regions)?;
        self.insert_derived(
            FeatureOp::RegionWeightedSum {
                regions: canonical_regions,
                weight_bits,
            },
            metadata,
        )
    }

    fn validate_region_weights(
        &self,
        weights: &FrozenRegionWeights,
        expected: usize,
    ) -> SemanticResult<()> {
        if weights.len() != expected {
            return Err(SemanticError::Invalid(
                "region weighted sum regions and weights must have equal lengths",
            ));
        }
        match (self.precision, weights) {
            (PrecisionProfile::Fp32 | PrecisionProfile::Mixed, FrozenRegionWeights::F32(bits))
                if bits.iter().all(|bits| f32::from_bits(*bits).is_finite()) =>
            {
                Ok(())
            }
            (PrecisionProfile::Fp64, FrozenRegionWeights::F64(bits))
                if bits.iter().all(|bits| f64::from_bits(*bits).is_finite()) =>
            {
                Ok(())
            }
            (PrecisionProfile::Fp32 | PrecisionProfile::Mixed, FrozenRegionWeights::F64(_)) => {
                Err(SemanticError::Invalid(
                    "f64 region weights do not match the selected candidate profile",
                ))
            }
            (PrecisionProfile::Fp64, FrozenRegionWeights::F32(_)) => Err(SemanticError::Invalid(
                "f32 region weights do not match an fp64 candidate registry",
            )),
            _ => Err(SemanticError::Invalid(
                "region weighted sum weights must be finite",
            )),
        }
    }

    fn canonical_region_weights(
        &self,
        submitted_regions: &[FeatureId],
        canonical_regions: &[FeatureId],
        weights: FrozenRegionWeights,
    ) -> SemanticResult<FrozenRegionWeights> {
        // `canonical_region_count_regions` has already proved the submitted
        // regions distinct and sorted.  The raw weight bits travel with their
        // matching region while we establish that one canonical order; unlike
        // a multiset reduction, duplicate terms are rejected rather than
        // merged or summed.
        match weights {
            FrozenRegionWeights::F32(bits) => {
                let mut by_region = BTreeMap::new();
                for (&region, bits) in submitted_regions.iter().zip(bits) {
                    if by_region.insert(region, bits).is_some() {
                        return Err(SemanticError::Invalid(
                            "region weighted sum repeats a decision region",
                        ));
                    }
                }
                Ok(FrozenRegionWeights::F32(
                    canonical_regions
                        .iter()
                        .map(|region| {
                            by_region.get(region).copied().ok_or(SemanticError::Invalid(
                                "region weighted sum lost a canonical region weight",
                            ))
                        })
                        .collect::<SemanticResult<Vec<_>>>()?,
                ))
            }
            FrozenRegionWeights::F64(bits) => {
                let mut by_region = BTreeMap::new();
                for (&region, bits) in submitted_regions.iter().zip(bits) {
                    if by_region.insert(region, bits).is_some() {
                        return Err(SemanticError::Invalid(
                            "region weighted sum repeats a decision region",
                        ));
                    }
                }
                Ok(FrozenRegionWeights::F64(
                    canonical_regions
                        .iter()
                        .map(|region| {
                            by_region.get(region).copied().ok_or(SemanticError::Invalid(
                                "region weighted sum lost a canonical region weight",
                            ))
                        })
                        .collect::<SemanticResult<Vec<_>>>()?,
                ))
            }
        }
    }

    fn compare_thresholds(
        &self,
        left: &FrozenThreshold,
        right: &FrozenThreshold,
    ) -> SemanticResult<std::cmp::Ordering> {
        match self.precision {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                Ok(f32::from_bits(left.as_f32_bits()?)
                    .partial_cmp(&f32::from_bits(right.as_f32_bits()?))
                    .expect("finite frozen predicate thresholds were validated"))
            }
            PrecisionProfile::Fp64 => Ok(f64::from_bits(left.as_f64_bits()?)
                .partial_cmp(&f64::from_bits(right.as_f64_bits()?))
                .expect("finite frozen predicate thresholds were validated")),
        }
    }

    fn validate_derived_inputs(&self, inputs: &[FeatureId]) -> SemanticResult<()> {
        if inputs.is_empty() {
            return Err(SemanticError::Invalid(
                "derived semantic program requires an input",
            ));
        }
        if inputs.len() > self.limits.max_logical_arity {
            return Err(SemanticError::Unsupported(
                "semantic program exceeds logical arity limit",
            ));
        }
        for input in inputs {
            self.program(*input)?;
        }
        Ok(())
    }

    fn derived_metadata(&self, inputs: &[FeatureId]) -> SemanticResult<DerivedProgramMetadata> {
        self.validate_derived_inputs(inputs)?;
        self.derived_metadata_from_valid_inputs(inputs)
    }

    fn derived_metadata_from_valid_inputs(
        &self,
        inputs: &[FeatureId],
    ) -> SemanticResult<DerivedProgramMetadata> {
        let mut unique_inputs = BTreeSet::new();
        let mut source_dependencies = BTreeSet::new();
        let mut deepest = 0usize;
        for input in inputs {
            let program = self.program(*input)?;
            if !unique_inputs.insert(*input) {
                return Err(SemanticError::Invalid(
                    "semantic program repeats an immediate input",
                ));
            }
            source_dependencies.extend(program.source_dependencies.iter().copied());
            deepest = deepest.max(program.depth);
        }
        let source_dependencies = source_dependencies.into_iter().collect::<Vec<_>>();
        if source_dependencies.len() > self.limits.max_source_arity {
            return Err(SemanticError::Unsupported(
                "semantic program exceeds source arity limit",
            ));
        }
        let depth = deepest.checked_add(1).ok_or(SemanticError::Unsupported(
            "semantic program depth overflow",
        ))?;
        if depth > self.limits.max_depth {
            return Err(SemanticError::Unsupported(
                "semantic program exceeds depth limit",
            ));
        }
        Ok(DerivedProgramMetadata {
            source_dependencies,
            logical_arity: inputs.len(),
            region_term_count: 0,
            depth,
        })
    }

    fn insert_derived(
        &mut self,
        operation: FeatureOp,
        metadata: DerivedProgramMetadata,
    ) -> SemanticResult<FeatureId> {
        if let Some(&existing) = self.by_operation.get(&operation) {
            return Ok(existing);
        }
        if self.programs.len() >= self.limits.max_nodes {
            return Err(SemanticError::Unsupported(
                "semantic candidate node limit exceeded",
            ));
        }
        let slot = u32::try_from(self.programs.len()).map_err(|_| {
            SemanticError::Unsupported("semantic candidate registry exhausted feature-id slots")
        })?;
        let id = FeatureId {
            registry: self.token,
            slot,
        };
        self.programs.push(FeatureProgram {
            id,
            op: operation.clone(),
            source_dependencies: metadata.source_dependencies,
            logical_arity: metadata.logical_arity,
            region_term_count: metadata.region_term_count,
            depth: metadata.depth,
        });
        self.by_operation.insert(operation, id);
        Ok(id)
    }
}

fn charge_training_lineage_work(
    work: &mut usize,
    amount: usize,
    max_work: usize,
) -> SemanticResult<()> {
    *work = work.checked_add(amount).ok_or(SemanticError::Invalid(
        "semantic fitting lineage metadata work overflow",
    ))?;
    if *work > max_work {
        return Err(SemanticError::Invalid(
            "semantic fitting lineage metadata work limit exceeded",
        ));
    }
    Ok(())
}

fn allocate_registry_token() -> SemanticResult<u64> {
    NEXT_REGISTRY_TOKEN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |token| {
            token.checked_add(1)
        })
        .map_err(|_| SemanticError::Closed)
}

fn validate_source_names(source_names: &[String]) -> SemanticResult<()> {
    if source_names
        .iter()
        .any(|name| name.is_empty() || name.len() > 256)
    {
        return Err(SemanticError::Invalid(
            "semantic source names must be nonempty and at most 256 bytes",
        ));
    }
    let mut names = BTreeSet::new();
    if source_names.iter().any(|name| !names.insert(name.as_str())) {
        return Err(SemanticError::Invalid(
            "semantic source names must be unique",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with(limits: ProgramLimits) -> CandidateRegistry {
        registry_with_profile(PrecisionProfile::Mixed, limits)
    }

    fn registry_with_profile(
        precision: PrecisionProfile,
        limits: ProgramLimits,
    ) -> CandidateRegistry {
        CandidateRegistry::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            precision,
            limits,
        )
        .unwrap()
    }

    #[test]
    fn all_profiles_have_a_profile_bound_semantic_registry() {
        for precision in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let registry =
                CandidateRegistry::new(vec!["a".into()], precision, ProgramLimits::default())
                    .unwrap();
            assert_eq!(registry.precision(), precision);
        }
    }

    #[test]
    fn source_schema_and_source_bounds_fail_closed() {
        for source_names in [
            vec![],
            vec![String::new()],
            vec!["a".into(), "a".into()],
            vec!["a".repeat(257)],
        ] {
            assert!(matches!(
                CandidateRegistry::new(
                    source_names,
                    PrecisionProfile::Mixed,
                    ProgramLimits::default()
                ),
                Err(SemanticError::Invalid(_))
            ));
        }
        let registry = registry_with(ProgramLimits::default());
        assert!(matches!(registry.source(4), Err(SemanticError::Invalid(_))));
        assert_eq!(registry.schema(), registry.source_names());
    }

    #[test]
    fn bounded_limits_precede_schema_deduplication() {
        let source_count_limit = ProgramLimits {
            max_nodes: 2,
            ..ProgramLimits::default()
        };
        assert!(matches!(
            CandidateRegistry::new(
                vec!["same".into(), "same".into(), "third".into()],
                PrecisionProfile::Mixed,
                source_count_limit,
            ),
            Err(SemanticError::Unsupported(_))
        ));

        for limits in [
            ProgramLimits {
                max_nodes: MAX_PROGRAM_NODES + 1,
                ..ProgramLimits::default()
            },
            ProgramLimits {
                max_logical_arity: MAX_PROGRAM_ARITY + 1,
                ..ProgramLimits::default()
            },
            ProgramLimits {
                max_source_arity: MAX_PROGRAM_ARITY + 1,
                ..ProgramLimits::default()
            },
            ProgramLimits {
                max_depth: MAX_PROGRAM_DEPTH + 1,
                ..ProgramLimits::default()
            },
        ] {
            assert!(matches!(
                CandidateRegistry::new(vec!["a".into()], PrecisionProfile::Mixed, limits),
                Err(SemanticError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn registry_identity_rejects_foreign_and_out_of_range_ids() {
        let mut registry = registry_with(ProgramLimits::default());
        let foreign_registry = registry_with(ProgramLimits::default());
        let foreign = foreign_registry.source(0).unwrap();
        assert!(!registry.owns(foreign));
        assert!(matches!(
            registry.program(foreign),
            Err(SemanticError::ForeignIdentity)
        ));
        assert!(matches!(
            registry.softsign(foreign),
            Err(SemanticError::ForeignIdentity)
        ));
        assert!(matches!(
            registry.training_lineage(foreign),
            Err(SemanticError::ForeignIdentity)
        ));

        let local = registry.source(0).unwrap();
        let out_of_range = FeatureId {
            registry: local.registry,
            slot: u32::MAX,
        };
        assert!(!registry.owns(out_of_range));
        assert!(matches!(
            registry.program(out_of_range),
            Err(SemanticError::Invalid(_))
        ));
    }

    #[test]
    fn absolute_difference_is_unordered_but_repeated_inputs_fail() {
        let mut registry = registry_with(ProgramLimits::default());
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let forward = registry.abs_difference(a, b).unwrap();
        let reverse = registry.abs_difference(b, a).unwrap();
        assert_eq!(forward, reverse);
        assert!(matches!(
            registry.program(forward).unwrap().op(),
            FeatureOp::AbsoluteDifference(left, right) if *left == a && *right == b
        ));
        assert!(matches!(
            registry.abs_difference(a, a),
            Err(SemanticError::Invalid(_))
        ));
    }

    #[test]
    fn centered_product_keeps_order_and_exact_frozen_mean_bits() {
        let mut registry = registry_with(ProgramLimits::default());
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let forward = registry
            .centered_product(vec![a, b], vec![-0.0, 1.25])
            .unwrap();
        let reversed = registry
            .centered_product(vec![b, a], vec![-0.0, 1.25])
            .unwrap();
        assert_ne!(forward, reversed);
        let FeatureOp::CenteredProduct {
            operands,
            mean_bits,
        } = registry.program(forward).unwrap().op()
        else {
            panic!("expected centered product");
        };
        assert_eq!(operands.as_slice(), &[a, b]);
        assert_eq!(
            mean_bits.as_f32_bits().unwrap(),
            &[(-0.0f32).to_bits(), 1.25f32.to_bits()]
        );
        assert!(matches!(
            registry.centered_product(vec![a, b], vec![f32::NAN, 0.0]),
            Err(SemanticError::Invalid(_))
        ));
        assert!(matches!(
            registry.centered_product(vec![a, b], vec![0.0]),
            Err(SemanticError::Invalid(_))
        ));
        assert!(matches!(
            registry.centered_product(vec![a, a], vec![0.0, 0.0]),
            Err(SemanticError::Invalid(_))
        ));
    }

    #[test]
    fn centered_product_checks_arity_before_mean_conversion() {
        let limits = ProgramLimits {
            max_logical_arity: 2,
            ..ProgramLimits::default()
        };
        let mut registry = registry_with(limits);
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let c = registry.source(2).unwrap();
        assert!(matches!(
            registry.centered_product(vec![a, b, c], vec![f32::NAN, f32::NAN, f32::NAN]),
            Err(SemanticError::Unsupported(_))
        ));
    }

    #[test]
    fn f64_frozen_means_preserve_bits_and_reject_f32_profile_crossing() {
        let mut fp64 = registry_with_profile(PrecisionProfile::Fp64, ProgramLimits::default());
        let a = fp64.source(0).unwrap();
        let b = fp64.source(1).unwrap();
        let one = 1.0f64;
        let next = f64::from_bits(one.to_bits() + 1);
        let first = fp64
            .centered_product_f64(vec![a, b], vec![one, -0.0])
            .unwrap();
        let distinct = fp64
            .centered_product_f64(vec![a, b], vec![next, -0.0])
            .unwrap();
        assert_ne!(first, distinct);
        let FeatureOp::CenteredProduct { mean_bits, .. } = fp64.program(first).unwrap().op() else {
            panic!("expected centered product");
        };
        assert_eq!(
            mean_bits.as_f64_bits().unwrap(),
            &[one.to_bits(), (-0.0f64).to_bits()]
        );
        assert!(fp64.centered_product(vec![a, b], vec![0.0, 0.0]).is_err());

        let mut mixed = registry_with(ProgramLimits::default());
        let a = mixed.source(0).unwrap();
        let b = mixed.source(1).unwrap();
        assert!(mixed
            .centered_product_f64(vec![a, b], vec![0.0, 0.0])
            .is_err());
    }

    #[test]
    fn transitive_dependencies_and_logical_arity_are_distinct() {
        let mut registry = registry_with(ProgramLimits::default());
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let c = registry.source(2).unwrap();
        let d = registry.source(3).unwrap();
        let left = registry.abs_difference(a, b).unwrap();
        let right = registry.abs_difference(c, d).unwrap();
        let joined = registry.abs_difference(left, right).unwrap();
        let program = registry.program(joined).unwrap();
        assert_eq!(program.logical_arity(), 2);
        assert_eq!(program.source_arity(), 4);
        assert_eq!(program.source_dependencies(), &[0, 1, 2, 3]);
        assert_eq!(program.depth(), 2);
    }

    #[test]
    fn logical_source_depth_and_node_limits_are_independent() {
        let logical = ProgramLimits {
            max_logical_arity: 2,
            max_source_arity: 4,
            ..ProgramLimits::default()
        };
        let mut logical_registry = registry_with(logical);
        let a = logical_registry.source(0).unwrap();
        let b = logical_registry.source(1).unwrap();
        let c = logical_registry.source(2).unwrap();
        assert!(matches!(
            logical_registry.centered_product(vec![a, b, c], vec![0.0, 0.0, 0.0]),
            Err(SemanticError::Unsupported(_))
        ));

        let source = ProgramLimits {
            max_source_arity: 3,
            ..ProgramLimits::default()
        };
        let mut source_registry = registry_with(source);
        let a = source_registry.source(0).unwrap();
        let b = source_registry.source(1).unwrap();
        let c = source_registry.source(2).unwrap();
        let d = source_registry.source(3).unwrap();
        let left = source_registry.abs_difference(a, b).unwrap();
        let right = source_registry.abs_difference(c, d).unwrap();
        assert!(matches!(
            source_registry.abs_difference(left, right),
            Err(SemanticError::Unsupported(_))
        ));

        let depth = ProgramLimits {
            max_depth: 1,
            ..ProgramLimits::default()
        };
        let mut depth_registry = registry_with(depth);
        let a = depth_registry.source(0).unwrap();
        let first = depth_registry.softsign(a).unwrap();
        assert!(matches!(
            depth_registry.softsign(first),
            Err(SemanticError::Unsupported(_))
        ));

        let nodes = ProgramLimits {
            max_nodes: 5,
            ..ProgramLimits::default()
        };
        let mut node_registry = registry_with(nodes);
        let a = node_registry.source(0).unwrap();
        let b = node_registry.source(1).unwrap();
        node_registry.abs_difference(a, b).unwrap();
        assert!(matches!(
            node_registry.softsign(a),
            Err(SemanticError::Unsupported(_))
        ));
    }
}
