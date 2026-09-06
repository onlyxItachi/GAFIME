use std::{collections::BTreeSet, sync::Arc};

use super::{next_identity, SemanticError, SemanticResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationRole {
    Discovery,
    Holdout,
    Inference,
}

/// Immutable, label-free, mixed-storage input. Row keys declare alignment in
/// one named row domain; equal lengths alone never establish paired views.
/// The caller is responsible for truthful dataset/split declarations.
#[derive(Clone, Debug)]
pub struct FeatureFrame {
    id: u64,
    schema: Arc<[String]>,
    row_domain: Arc<str>,
    row_keys: Arc<[u64]>,
    role: EvaluationRole,
    provenance: Arc<str>,
    columns: Arc<[Vec<f32>]>,
}

impl FeatureFrame {
    pub fn new(
        schema: Vec<String>,
        row_domain: String,
        row_keys: Vec<u64>,
        role: EvaluationRole,
        provenance: String,
        columns: Vec<Vec<f32>>,
    ) -> SemanticResult<Self> {
        // Bound matrix plus row-key storage before the uniqueness indexes
        // allocate. Strings/Vec headers and the caller's input allocation are
        // outside this numeric-storage budget; this is not an RSS guarantee.
        let bytes = row_keys
            .len()
            .checked_mul(columns.len())
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| {
                row_keys
                    .len()
                    .checked_mul(8)
                    .and_then(|keys| n.checked_add(keys))
            });
        if schema.len() > 65_536
            || row_keys.len() > 1_000_000
            || schema.iter().any(|s| s.len() > 256)
            || row_domain.len() > 4096
            || provenance.len() > 4096
            || bytes.is_none_or(|n| n > 256 * 1024 * 1024)
        {
            return Err(SemanticError::Invalid(
                "frame exceeds bounded schema, row or numeric-storage limits",
            ));
        }
        if schema.is_empty()
            || schema.len() != columns.len()
            || schema.iter().any(String::is_empty)
            || schema.iter().collect::<BTreeSet<_>>().len() != schema.len()
            || row_keys.len() < 2
            || row_domain.is_empty()
            || provenance.is_empty()
            || row_keys.iter().collect::<BTreeSet<_>>().len() != row_keys.len()
        {
            return Err(SemanticError::Invalid(
                "invalid frame schema, rows or provenance",
            ));
        }
        if columns
            .iter()
            .any(|c| c.len() != row_keys.len() || c.iter().any(|v| !v.is_finite()))
        {
            return Err(SemanticError::Invalid(
                "frame requires aligned finite f32 columns",
            ));
        }
        Ok(Self {
            id: next_identity()?,
            schema: schema.into(),
            row_domain: row_domain.into(),
            row_keys: row_keys.into(),
            role,
            provenance: provenance.into(),
            columns: columns.into(),
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn schema(&self) -> &[String] {
        &self.schema
    }
    pub fn row_domain(&self) -> &str {
        &self.row_domain
    }
    pub fn row_keys(&self) -> &[u64] {
        &self.row_keys
    }
    pub fn role(&self) -> EvaluationRole {
        self.role
    }
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
    pub fn rows(&self) -> usize {
        self.row_keys.len()
    }
    pub fn column(&self, index: usize) -> SemanticResult<&[f32]> {
        self.columns
            .get(index)
            .map(Vec::as_slice)
            .ok_or(SemanticError::Invalid("source column out of bounds"))
    }
    pub fn aligned_with(&self, other: &Self) -> bool {
        self.schema == other.schema
            && self.row_domain == other.row_domain
            && self.row_keys == other.row_keys
            && self.role == other.role
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GraphEdge {
    pub left: usize,
    pub right: usize,
    pub weight: f32,
}

/// Explicit undirected edge multiset. Duplicate edges contribute repeatedly;
/// self-loops and nonpositive/nonfinite weights fail instead of being rewritten.
#[derive(Clone, Debug)]
pub struct NeighborGraph {
    id: u64,
    frame_id: u64,
    edges: Arc<[GraphEdge]>,
    provenance: Arc<str>,
}

impl NeighborGraph {
    pub fn new(
        frame: &FeatureFrame,
        edges: Vec<GraphEdge>,
        provenance: String,
    ) -> SemanticResult<Self> {
        if edges.is_empty()
            || edges.len() > 1_000_000
            || provenance.is_empty()
            || provenance.len() > 4096
            || edges.iter().any(|e| {
                e.left >= frame.rows()
                    || e.right >= frame.rows()
                    || e.left == e.right
                    || !e.weight.is_finite()
                    || e.weight <= 0.0
            })
        {
            return Err(SemanticError::Invalid("invalid bounded neighbor graph"));
        }
        Ok(Self {
            id: next_identity()?,
            frame_id: frame.id,
            edges: edges.into(),
            provenance: provenance.into(),
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
}

/// Actual supplied labels on unique row positions, canonicalized by position.
/// Missing labels are absence, never an all-zero target. The frame identity
/// binds each position to its immutable row key.
#[derive(Clone, Debug)]
pub struct LabelSet {
    id: u64,
    frame_id: u64,
    rows: Arc<[usize]>,
    values: Arc<[f32]>,
    provenance: Arc<str>,
}

impl LabelSet {
    pub fn new(
        frame: &FeatureFrame,
        mut labels: Vec<(usize, f32)>,
        provenance: String,
    ) -> SemanticResult<Self> {
        if provenance.is_empty()
            || provenance.len() > 4096
            || labels.len() > frame.rows()
            || labels
                .iter()
                .any(|(i, v)| *i >= frame.rows() || !v.is_finite())
        {
            return Err(SemanticError::Invalid("invalid labeled-row context"));
        }
        labels.sort_by_key(|(i, _)| *i);
        if labels.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err(SemanticError::Invalid("duplicate labeled row"));
        }
        let (rows, values): (Vec<_>, Vec<_>) = labels.into_iter().unzip();
        Ok(Self {
            id: next_identity()?,
            frame_id: frame.id,
            rows: rows.into(),
            values: values.into(),
            provenance: provenance.into(),
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(keys: Vec<u64>, role: EvaluationRole) -> SemanticResult<FeatureFrame> {
        FeatureFrame::new(
            vec!["a".into()],
            "dataset-A".into(),
            keys,
            role,
            "declared fixture".into(),
            vec![vec![1.0, 2.0, 3.0]],
        )
    }

    #[test]
    fn immutable_context_ids_and_alignment_are_not_shape_only() {
        let a = frame(vec![0, 1, 2], EvaluationRole::Discovery).unwrap();
        let cloned = a.clone();
        assert_eq!(a.id(), cloned.id());
        let same = frame(vec![0, 1, 2], EvaluationRole::Discovery).unwrap();
        assert_ne!(a.id(), same.id());
        assert!(a.aligned_with(&same));
        assert!(!a.aligned_with(&frame(vec![1, 0, 2], EvaluationRole::Discovery).unwrap()));
        assert!(!a.aligned_with(&frame(vec![0, 1, 2], EvaluationRole::Holdout).unwrap()));
        assert!(frame(vec![0, 0, 2], EvaluationRole::Discovery).is_err());
    }

    #[test]
    fn invalid_numeric_inputs_and_unbounded_names_fail_closed() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(FeatureFrame::new(
                vec!["a".into()],
                "rows".into(),
                vec![0, 1],
                EvaluationRole::Discovery,
                "fixture".into(),
                vec![vec![0.0, value]]
            )
            .is_err());
        }
        assert!(FeatureFrame::new(
            vec!["a".repeat(257)],
            "rows".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "fixture".into(),
            vec![vec![0.0, 1.0]]
        )
        .is_err());
    }

    #[test]
    fn graph_and_label_inputs_preserve_exact_frame_binding() {
        let input = frame(vec![0, 1, 2], EvaluationRole::Discovery).unwrap();
        for edge in [
            GraphEdge {
                left: 0,
                right: 0,
                weight: 1.0,
            },
            GraphEdge {
                left: 0,
                right: 3,
                weight: 1.0,
            },
            GraphEdge {
                left: 0,
                right: 1,
                weight: 0.0,
            },
            GraphEdge {
                left: 0,
                right: 1,
                weight: f32::NAN,
            },
        ] {
            assert!(NeighborGraph::new(&input, vec![edge], "edges".into()).is_err());
        }
        let edge = GraphEdge {
            left: 0,
            right: 1,
            weight: 1.0,
        };
        let graph = NeighborGraph::new(&input, vec![edge, edge], "multiset".into()).unwrap();
        assert_eq!(graph.edges().len(), 2);
        assert_eq!(graph.frame_id(), input.id());
        let labels = LabelSet::new(&input, vec![(2, 3.0), (0, 1.0)], "supplied".into()).unwrap();
        assert_eq!(labels.rows(), &[0, 2]);
        assert_eq!(labels.values(), &[1.0, 3.0]);
        assert_eq!(labels.frame_id(), input.id());
        assert!(LabelSet::new(&input, vec![(0, 1.0), (0, 2.0)], "duplicate".into()).is_err());
        assert!(LabelSet::new(&input, vec![(3, 1.0)], "out-of-bounds".into()).is_err());
    }
}
