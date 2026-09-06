use std::sync::Arc;

use crate::plan::combos::MI_TEMPLATE_BIN_LEVELS;
use gafime_types::{
    BackendKind, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM,
};

use super::{
    next_identity, CandidateRegistry, FeatureFrame, FeatureId, LabelSet, MaterializedColumns,
    NeighborGraph, SemanticError, SemanticResult,
};

/// Distinct evaluation channels do not become candidate identities or metric IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceId(u64);

/// The closed numerical estimator vocabulary for contextual association.
/// `FixedCorrectedNmi` records an exact fixed histogram shape; it is not the
/// legacy adaptive-MI maximum parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssociationStatistic {
    Pearson,
    Spearman,
    FixedCorrectedNmi { bins: u32 },
}

impl AssociationStatistic {
    pub fn fixed_nmi_bins(self) -> Option<u32> {
        match self {
            Self::FixedCorrectedNmi { bins } => Some(bins),
            Self::Pearson | Self::Spearman => None,
        }
    }

    fn validate(self) -> SemanticResult<()> {
        if let Self::FixedCorrectedNmi { bins } = self {
            if !MI_TEMPLATE_BIN_LEVELS.contains(&bins) {
                return Err(SemanticError::Invalid(
                    "fixed corrected NMI requires an exact supported bin count",
                ));
            }
        }
        Ok(())
    }
}

/// The kind of contextual operand that participates in an association.
/// Operand instances are deliberately excluded from `EvidenceSemanticKey` so
/// an immutable specification can be rebound to new same-kind context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssociationContextKind {
    Reference,
    PairedView,
    Labels,
}

/// Explicit contextual operands for an association. Labels stay optional so
/// absence is represented as unavailable evidence rather than a pseudo-target.
#[derive(Clone, Debug)]
pub enum AssociationContext {
    Reference { reference: FeatureId },
    PairedView { view: Arc<FeatureFrame> },
    Labels { labels: Option<Arc<LabelSet>> },
}

impl AssociationContext {
    pub fn kind(&self) -> AssociationContextKind {
        match self {
            Self::Reference { .. } => AssociationContextKind::Reference,
            Self::PairedView { .. } => AssociationContextKind::PairedView,
            Self::Labels { .. } => AssociationContextKind::Labels,
        }
    }

    pub fn reference(&self) -> Option<FeatureId> {
        match self {
            Self::Reference { reference } => Some(*reference),
            Self::PairedView { .. } | Self::Labels { .. } => None,
        }
    }

    pub fn paired_view(&self) -> Option<&Arc<FeatureFrame>> {
        match self {
            Self::PairedView { view } => Some(view),
            Self::Reference { .. } | Self::Labels { .. } => None,
        }
    }

    pub fn labels(&self) -> Option<&Option<Arc<LabelSet>>> {
        match self {
            Self::Labels { labels } => Some(labels),
            Self::Reference { .. } | Self::PairedView { .. } => None,
        }
    }

    fn same_work(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Reference { reference: a }, Self::Reference { reference: b }) => a == b,
            (Self::PairedView { view: a }, Self::PairedView { view: b }) => Arc::ptr_eq(a, b),
            (Self::Labels { labels: a }, Self::Labels { labels: b }) => match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, Some(_)) | (Some(_), None) => false,
            },
            _ => false,
        }
    }
}

/// Closed, owned measurement identity. The exact statistic, fixed-bin shape
/// and contextual operand kind all participate; display names do not decide
/// whether an existing specification can be rebound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceSemanticKey {
    Association {
        statistic: AssociationStatistic,
        context: AssociationContextKind,
    },
    GraphEnergy,
}

/// Versioned measurements with explicit operands, not fabricated targets.
#[derive(Clone, Debug)]
pub enum EvidenceDefinition {
    Association {
        statistic: AssociationStatistic,
        context: AssociationContext,
    },
    GraphEnergy {
        graph: Arc<NeighborGraph>,
    },
}

impl EvidenceDefinition {
    pub fn semantic_key(&self) -> EvidenceSemanticKey {
        match self {
            Self::Association { statistic, context } => EvidenceSemanticKey::Association {
                statistic: *statistic,
                context: context.kind(),
            },
            Self::GraphEnergy { .. } => EvidenceSemanticKey::GraphEnergy,
        }
    }

    pub fn semantic_name(&self) -> &'static str {
        semantic_name(self.semantic_key())
    }

    pub fn statistic(&self) -> Option<AssociationStatistic> {
        match self {
            Self::Association { statistic, .. } => Some(*statistic),
            Self::GraphEnergy { .. } => None,
        }
    }

    pub fn association_context(&self) -> Option<&AssociationContext> {
        match self {
            Self::Association { context, .. } => Some(context),
            Self::GraphEnergy { .. } => None,
        }
    }

    pub fn reference(&self) -> Option<FeatureId> {
        self.association_context()?.reference()
    }

    pub fn paired_view(&self) -> Option<&Arc<FeatureFrame>> {
        self.association_context()?.paired_view()
    }

    pub fn labels(&self) -> Option<&Option<Arc<LabelSet>>> {
        self.association_context()?.labels()
    }

    pub fn graph(&self) -> Option<&Arc<NeighborGraph>> {
        match self {
            Self::GraphEnergy { graph } => Some(graph),
            Self::Association { .. } => None,
        }
    }

    fn validate_definition(&self) -> SemanticResult<()> {
        if let Self::Association { statistic, .. } = self {
            statistic.validate()?;
        }
        Ok(())
    }

    pub(crate) fn validate(
        &self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
    ) -> SemanticResult<()> {
        self.validate_definition()?;
        match self {
            Self::Association {
                context: AssociationContext::Reference { reference },
                ..
            } => {
                registry.program(*reference)?;
            }
            Self::Association {
                context: AssociationContext::PairedView { view },
                ..
            } if !frame.aligned_with(view) => {
                return Err(SemanticError::Invalid(
                    "paired view schema, row keys, domain and role must align",
                ));
            }
            Self::Association {
                context:
                    AssociationContext::Labels {
                        labels: Some(labels),
                    },
                ..
            } if labels.frame_id() != frame.id() => {
                return Err(SemanticError::Invalid(
                    "labels belong to another input context",
                ));
            }
            Self::GraphEnergy { graph } if graph.frame_id() != frame.id() => {
                return Err(SemanticError::Invalid(
                    "graph belongs to another input context",
                ));
            }
            Self::Association { .. } | Self::GraphEnergy { .. } => {}
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
    semantics: EvidenceSemanticKey,
}

impl EvidenceSpec {
    pub fn id(&self) -> EvidenceId {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn semantic_name(&self) -> &'static str {
        semantic_name(self.semantics)
    }
    pub fn semantic_key(&self) -> EvidenceSemanticKey {
        self.semantics
    }
    pub fn bind(&self, definition: EvidenceDefinition) -> SemanticResult<EvidenceChannel> {
        definition.validate_definition()?;
        if self.semantics != definition.semantic_key() {
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
            semantics: definition.semantic_key(),
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
                EvidenceDefinition::Association {
                    statistic: a_statistic,
                    context: a_context,
                },
                EvidenceDefinition::Association {
                    statistic: b_statistic,
                    context: b_context,
                },
            ) => a_statistic == b_statistic && a_context.same_work(b_context),
            (
                EvidenceDefinition::GraphEnergy { graph: a },
                EvidenceDefinition::GraphEnergy { graph: b },
            ) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

fn semantic_name(key: EvidenceSemanticKey) -> &'static str {
    match key {
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContextKind::Reference,
        } => "absolute-pearson/reference/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContextKind::PairedView,
        } => "signed-pearson/aligned-view/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContextKind::Labels,
        } => "absolute-pearson/labeled-subset/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Spearman,
            context: AssociationContextKind::Reference,
        } => "absolute-spearman/reference/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Spearman,
            context: AssociationContextKind::PairedView,
        } => "signed-spearman/aligned-view/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::Spearman,
            context: AssociationContextKind::Labels,
        } => "absolute-spearman/labeled-subset/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::FixedCorrectedNmi { .. },
            context: AssociationContextKind::Reference,
        } => "normalized-fixed-corrected-mi/reference/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::FixedCorrectedNmi { .. },
            context: AssociationContextKind::PairedView,
        } => "normalized-fixed-corrected-mi/aligned-view/v1",
        EvidenceSemanticKey::Association {
            statistic: AssociationStatistic::FixedCorrectedNmi { .. },
            context: AssociationContextKind::Labels,
        } => "normalized-fixed-corrected-mi/labeled-subset/v1",
        EvidenceSemanticKey::GraphEnergy => "uncentered-edge-energy-ratio/v1",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnavailableReason {
    MissingLabels,
    InsufficientSupport,
    ConstantOperand,
    DegenerateReduction,
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
    pub(crate) backend: BackendKind,
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
        match self.backend {
            GAFIME_BACKEND_CPU => "core",
            GAFIME_BACKEND_CUDA => "cuda",
            GAFIME_BACKEND_ROCM => "rocm",
            GAFIME_BACKEND_METAL => "metal",
            _ => "unknown",
        }
    }
    pub const fn backend_kind(&self) -> BackendKind {
        self.backend
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
