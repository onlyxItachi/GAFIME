use std::sync::Arc;

use super::{
    next_identity, CandidateRegistry, FeatureFrame, FeatureId, LabelSet, MaterializedColumns,
    NeighborGraph, SemanticError, SemanticResult,
};

/// Distinct evaluation channels do not become candidate identities or metric IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceId(u64);

/// Versioned measurements with explicit operands, not fabricated targets.
#[derive(Clone, Debug)]
pub enum EvidenceDefinition {
    Redundancy { reference: FeatureId },
    PairedConsistency { view: Arc<FeatureFrame> },
    GraphEnergy { graph: Arc<NeighborGraph> },
    LabeledAssociation { labels: Option<Arc<LabelSet>> },
}

impl EvidenceDefinition {
    pub fn semantic_name(&self) -> &'static str {
        match self {
            Self::Redundancy { .. } => "absolute-pearson/reference/v1",
            Self::PairedConsistency { .. } => "signed-pearson/aligned-view/v1",
            Self::GraphEnergy { .. } => "uncentered-edge-energy-ratio/v1",
            Self::LabeledAssociation { .. } => "absolute-pearson/labeled-subset/v1",
        }
    }

    pub(crate) fn validate(
        &self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
    ) -> SemanticResult<()> {
        match self {
            Self::Redundancy { reference } => {
                registry.program(*reference)?;
            }
            Self::PairedConsistency { view } if !frame.aligned_with(view) => {
                return Err(SemanticError::Invalid(
                    "paired view schema, row keys, domain and role must align",
                ));
            }
            Self::GraphEnergy { graph } if graph.frame_id() != frame.id() => {
                return Err(SemanticError::Invalid(
                    "graph belongs to another input context",
                ));
            }
            Self::LabeledAssociation {
                labels: Some(labels),
            } if labels.frame_id() != frame.id() => {
                return Err(SemanticError::Invalid(
                    "labels belong to another input context",
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Stable, process-local measurement specification. Policy names this identity,
/// not a particular set of labels or rows. Binding new operands preserves the
/// question being asked; it never changes an already recorded evaluation.
#[derive(Clone, Debug)]
pub struct EvidenceSpec {
    id: EvidenceId,
    name: String,
    semantics: &'static str,
}

impl EvidenceSpec {
    pub fn id(&self) -> EvidenceId {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn semantic_name(&self) -> &'static str {
        self.semantics
    }
    pub fn bind(&self, definition: EvidenceDefinition) -> SemanticResult<EvidenceChannel> {
        if self.semantics != definition.semantic_name() {
            return Err(SemanticError::Invalid(
                "evidence binding changes measurement semantics",
            ));
        }
        Ok(EvidenceChannel {
            spec: self.clone(),
            definition,
        })
    }
}

/// An immutable binding of a specification to actual evaluation operands.
/// Old tables and acceptances retain old bindings when a specification is rebound.
#[derive(Clone, Debug)]
pub struct EvidenceChannel {
    spec: EvidenceSpec,
    definition: EvidenceDefinition,
}

impl EvidenceChannel {
    pub fn new(name: String, definition: EvidenceDefinition) -> SemanticResult<Self> {
        if name.is_empty() || name.len() > 256 {
            return Err(SemanticError::Invalid(
                "evidence channel requires a bounded nonempty name",
            ));
        }
        EvidenceSpec {
            id: EvidenceId(next_identity()?),
            name,
            semantics: definition.semantic_name(),
        }
        .bind(definition)
    }
    pub fn id(&self) -> EvidenceId {
        self.spec.id
    }
    pub fn name(&self) -> &str {
        &self.spec.name
    }
    pub fn definition(&self) -> &EvidenceDefinition {
        &self.definition
    }
    pub fn spec(&self) -> &EvidenceSpec {
        &self.spec
    }
    pub fn rebind(&self, definition: EvidenceDefinition) -> SemanticResult<Self> {
        self.spec.bind(definition)
    }

    // Within one evaluation, identical immutable operands can share native work
    // even when users retain separate channel names for distinct policy roles.
    pub(crate) fn same_work(&self, other: &Self) -> bool {
        match (&self.definition, &other.definition) {
            (
                EvidenceDefinition::Redundancy { reference: a },
                EvidenceDefinition::Redundancy { reference: b },
            ) => a == b,
            (
                EvidenceDefinition::PairedConsistency { view: a },
                EvidenceDefinition::PairedConsistency { view: b },
            ) => Arc::ptr_eq(a, b),
            (
                EvidenceDefinition::GraphEnergy { graph: a },
                EvidenceDefinition::GraphEnergy { graph: b },
            ) => Arc::ptr_eq(a, b),
            (
                EvidenceDefinition::LabeledAssociation { labels: a },
                EvidenceDefinition::LabeledAssociation { labels: b },
            ) => match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnavailableReason {
    MissingLabels,
    InsufficientSupport,
    ConstantOperand,
    NonFiniteReduction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvidenceValue {
    Measured {
        value: f64,
        support: usize,
    },
    Unavailable {
        reason: UnavailableReason,
        support: usize,
    },
}

impl EvidenceValue {
    /// Exact widening for storage only: fp32 arithmetic and selection thresholds
    /// remain f32. This shared report container does not grant extra precision.
    pub fn measured_f32(value: f32, support: usize) -> Self {
        Self::measured(f64::from(value), support)
    }
    pub fn measured(value: f64, support: usize) -> Self {
        if value.is_finite() {
            Self::Measured { value, support }
        } else {
            Self::Unavailable {
                reason: UnavailableReason::NonFiniteReduction,
                support,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvidenceRecord {
    pub(crate) candidate: FeatureId,
    pub(crate) channel: EvidenceId,
    pub(crate) value: EvidenceValue,
}

impl EvidenceRecord {
    pub fn candidate(&self) -> FeatureId {
        self.candidate
    }
    pub fn channel(&self) -> EvidenceId {
        self.channel
    }
    pub fn value(&self) -> EvidenceValue {
        self.value
    }
}

/// One immutable evaluation, retaining context/provenance and actual operands.
/// Evidence is not cached across contexts. Numeric values follow the frame's
/// Core profile, not calibrated cross-paradigm utilities or significance claims.
pub struct EvidenceTable {
    pub(crate) owner: u64,
    pub(crate) id: u64,
    pub(crate) round: u64,
    pub(crate) frame: Arc<FeatureFrame>,
    pub(crate) candidates: Vec<FeatureId>,
    pub(crate) channels: Vec<EvidenceChannel>,
    pub(crate) records: Vec<EvidenceRecord>,
    pub(crate) materialized: MaterializedColumns,
}

impl EvidenceTable {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn frame(&self) -> &FeatureFrame {
        &self.frame
    }
    pub fn candidates(&self) -> &[FeatureId] {
        &self.candidates
    }
    pub fn channels(&self) -> &[EvidenceChannel] {
        &self.channels
    }
    pub fn records(&self) -> &[EvidenceRecord] {
        &self.records
    }
    pub fn backend(&self) -> &'static str {
        "core"
    }
    pub fn precision(&self) -> &'static str {
        match self.frame.profile() {
            gafime_types::PrecisionProfile::Fp32 => "fp32",
            gafime_types::PrecisionProfile::Mixed => "mixed",
            gafime_types::PrecisionProfile::Fp64 => "fp64",
        }
    }
    pub fn value(
        &self,
        candidate: FeatureId,
        channel: EvidenceId,
    ) -> SemanticResult<EvidenceValue> {
        let row = self
            .candidates
            .binary_search(&candidate)
            .map_err(|_| SemanticError::ForeignIdentity)?;
        let col = self
            .channels
            .iter()
            .position(|c| c.id() == channel)
            .ok_or(SemanticError::ForeignIdentity)?;
        Ok(self.records[row * self.channels.len() + col].value)
    }
}
