"""Tabular candidate/evidence discovery through one Rust-owned lifecycle.

Development API for the bounded v1.1 tabular milestone. Inputs are immutable
native snapshots, handles are session-local, and outputs implement Arrow C Data.
No Python callbacks or feature-computation loops participate in execution.
See ``docs/v1.1-tabular-semantic-product.md`` for the supported contract.
"""

from .gafime_py import (
    AcceptedSet,
    Candidate,
    CandidateSet,
    Constraint,
    Evidence,
    EvidenceReport,
    FeatureTable,
    Graph,
    Labels,
    SelectionPolicy,
    Snapshot,
    TabularSession,
)

__all__ = [
    "AcceptedSet",
    "Candidate",
    "CandidateSet",
    "Constraint",
    "Evidence",
    "EvidenceReport",
    "FeatureTable",
    "Graph",
    "Labels",
    "SelectionPolicy",
    "Snapshot",
    "TabularSession",
]
