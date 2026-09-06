//! Canonical, target-free candidate programs for the first semantic vertical
//! slice.  This registry deliberately has no ABI or Python representation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// A target-free, precision-bound candidate operation.
///
/// `CenteredProduct` preserves operand order because sequential multiplication
/// in the pointwise dtype is not generally associative.  Frozen means are
/// caller-declared constants stored as exact profile-bound bits rather than
/// recomputed from a later frame. This layer neither fits nor estimates them and makes no
/// declaration about their split or origin; that provenance is outside the
/// mathematical identity and belongs to the evaluation/acceptance context.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FeatureOp {
    Source(u32),
    AbsoluteDifference(FeatureId, FeatureId),
    Softsign(FeatureId),
    CenteredProduct {
        operands: Vec<FeatureId>,
        mean_bits: FrozenMeans,
    },
}

/// One immutable semantic program in a [`CandidateRegistry`].
#[derive(Clone, Debug)]
pub struct FeatureProgram {
    id: FeatureId,
    op: FeatureOp,
    source_dependencies: Vec<u32>,
    logical_arity: usize,
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
}

/// Internal mutation boundary used by bounded bulk declaration. A failed batch
/// never exposes newly allocated identities and may therefore safely reclaim
/// its appended program slots without cloning/forking registry authority.
#[derive(Clone, Copy)]
pub(crate) struct RegistryCheckpoint(usize);

struct DerivedProgramMetadata {
    source_dependencies: Vec<u32>,
    logical_arity: usize,
    depth: usize,
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
        let metadata = self.centered_product_metadata(&operands, frozen_means.len())?;
        if frozen_means.iter().any(|mean| !mean.is_finite()) {
            return Err(SemanticError::Invalid(
                "centered product frozen means must be finite",
            ));
        }
        let mean_bits = FrozenMeans::F32(frozen_means.into_iter().map(f32::to_bits).collect());
        self.insert_derived(
            FeatureOp::CenteredProduct {
                operands,
                mean_bits,
            },
            metadata,
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
        let metadata = self.centered_product_metadata(&operands, frozen_means.len())?;
        if frozen_means.iter().any(|mean| !mean.is_finite()) {
            return Err(SemanticError::Invalid(
                "centered product frozen means must be finite",
            ));
        }
        let mean_bits = FrozenMeans::F64(frozen_means.into_iter().map(f64::to_bits).collect());
        self.insert_derived(
            FeatureOp::CenteredProduct {
                operands,
                mean_bits,
            },
            metadata,
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
            depth: metadata.depth,
        });
        self.by_operation.insert(operation, id);
        Ok(id)
    }
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
