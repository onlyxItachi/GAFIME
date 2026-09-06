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

/// Immutable channel identity includes its actual contextual operands. A new
/// label set/view/graph requires a new channel; changing a display name cannot
/// redirect an existing evidence record.
#[derive(Clone, Debug)]
pub struct EvidenceChannel {
    id: EvidenceId,
    name: String,
    definition: EvidenceDefinition,
}

impl EvidenceChannel {
    pub fn new(name: String, definition: EvidenceDefinition) -> SemanticResult<Self> {
        if name.is_empty() || name.len() > 256 {
            return Err(SemanticError::Invalid(
                "evidence channel requires a bounded nonempty name",
            ));
        }
        Ok(Self {
            id: EvidenceId(next_identity()?),
            name,
            definition,
        })
    }
    pub fn id(&self) -> EvidenceId {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn definition(&self) -> &EvidenceDefinition {
        &self.definition
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
/// Evidence is not cached across contexts. Numeric values are mixed-route Core
/// evidence, not calibrated cross-paradigm utilities or significance claims.
pub struct EvidenceTable {
    pub(crate) owner: u64,
    pub(crate) id: u64,
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
        "mixed"
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
            .position(|c| c.id == channel)
            .ok_or(SemanticError::ForeignIdentity)?;
        Ok(self.records[row * self.channels.len() + col].value)
    }
}
