//! Workspace-internal v1.1 candidate/evidence semantics, not a serialized IR or
//! a stable Python/Rust API. Native launch descriptors are lowerings, not IDs.
//!
//! A program survives evaluation and selection. Evidence is bound to immutable
//! input contexts; acceptance is a recorded decision, not a universal claim of
//! usefulness. Inference consumes only source features and frozen program state.

mod context;
mod evidence;
mod numeric;
mod ordering;
pub mod program;
mod selection;
mod session;
pub mod supervised;

pub use context::{
    EvaluationRole, FeatureFrame, GraphEdge, GraphEndpoints, LabelSet, NeighborGraph,
};
pub use evidence::{
    AssociationContext, AssociationContextKind, AssociationStatistic, EvidenceChannel,
    EvidenceDefinition, EvidenceId, EvidenceRecord, EvidenceSemanticKey, EvidenceSpec,
    EvidenceTable, EvidenceValue, UnavailableReason,
};
pub use numeric::NumericColumn;
pub use ordering::Direction;
pub use program::{
    CandidateRegistry, FeatureId, FeatureOp, FeatureProgram, FrozenMeans, ProgramLimits,
};
pub use selection::{EvidenceConstraint, MissingEvidence, SelectionPolicy};
pub use session::{
    AcceptedFeature, DiscoveryRound, MaterializedColumns, NativeEvidenceExecutor, ProposalOperator,
    ResidentMaterializationLease, SemanticSession, SessionLimits,
};

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticError {
    Invalid(&'static str),
    Unsupported(&'static str),
    Closed,
    ForeignIdentity,
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Unsupported(message) => f.write_str(message),
            Self::Closed => f.write_str("semantic session is closed"),
            Self::ForeignIdentity => f.write_str("identity belongs to a different owner"),
        }
    }
}

impl std::error::Error for SemanticError {}
pub type SemanticResult<T> = Result<T, SemanticError>;

// In-process ownership identity only. Never serialize these counters or use
// them as a cross-process content hash. Exhaustion fails rather than aliasing.
fn next_identity() -> SemanticResult<u64> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| SemanticError::Invalid("semantic identity space exhausted"))
}
