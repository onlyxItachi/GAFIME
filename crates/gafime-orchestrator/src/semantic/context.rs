use std::{collections::BTreeSet, sync::Arc};

use gafime_types::PrecisionProfile;

use super::{next_identity, NumericColumn, SemanticError, SemanticResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationRole {
    Discovery,
    Holdout,
    Inference,
}

/// Immutable, label-free, profile-bound input. Row keys declare alignment in
/// one named row domain; equal lengths alone never establish paired views.
/// The caller is responsible for truthful dataset/split declarations.
#[derive(Clone, Debug)]
pub struct FeatureFrame {
    id: u64,
    profile: PrecisionProfile,
    schema: Arc<[String]>,
    row_domain: Arc<str>,
    row_keys: Arc<[u64]>,
    role: EvaluationRole,
    provenance: Arc<str>,
    columns: Arc<[NumericColumn]>,
}

impl FeatureFrame {
    /// Mixed precision convenience construction with immutable f32 storage.
    pub fn new(
        schema: Vec<String>,
        row_domain: String,
        row_keys: Vec<u64>,
        role: EvaluationRole,
        provenance: String,
        columns: Vec<Vec<f32>>,
    ) -> SemanticResult<Self> {
        Self::with_profile(
            PrecisionProfile::Mixed,
            schema,
            row_domain,
            row_keys,
            role,
            provenance,
            columns.into_iter().map(NumericColumn::from).collect(),
        )
    }

    /// Fp64 convenience construction with immutable f64 storage.
    pub fn new_f64(
        schema: Vec<String>,
        row_domain: String,
        row_keys: Vec<u64>,
        role: EvaluationRole,
        provenance: String,
        columns: Vec<Vec<f64>>,
    ) -> SemanticResult<Self> {
        Self::with_profile(
            PrecisionProfile::Fp64,
            schema,
            row_domain,
            row_keys,
            role,
            provenance,
            columns.into_iter().map(NumericColumn::from).collect(),
        )
    }

    /// Construct one immutable frame without coercing its numeric storage.
    pub fn with_profile(
        profile: PrecisionProfile,
        schema: Vec<String>,
        row_domain: String,
        row_keys: Vec<u64>,
        role: EvaluationRole,
        provenance: String,
        columns: Vec<NumericColumn>,
    ) -> SemanticResult<Self> {
        // Bound matrix plus row-key storage before the uniqueness indexes
        // allocate. Strings/Vec headers and caller-owned input allocations are
        // outside this numeric-storage budget; this is not an RSS guarantee.
        let numeric_bytes = columns.iter().try_fold(0usize, |total, column| {
            total
                .checked_add(column.bytes())
                .ok_or(SemanticError::Invalid(
                    "frame numeric storage exceeds host address space",
                ))
        })?;
        let bytes = row_keys
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|keys| numeric_bytes.checked_add(keys));
        if schema.len() > 65_536
            || row_keys.len() > 1_000_000
            || schema.iter().any(|name| name.len() > 256)
            || row_domain.len() > 4096
            || provenance.len() > 4096
            || bytes.is_none_or(|total| total > 256 * 1024 * 1024)
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
        if columns.iter().any(|column| {
            column.len() != row_keys.len() || !column.finite() || !column.supports_profile(profile)
        }) {
            return Err(SemanticError::Invalid(
                "frame requires aligned finite profile-compatible columns",
            ));
        }
        Ok(Self {
            id: next_identity()?,
            profile,
            schema: schema.into(),
            row_domain: row_domain.into(),
            row_keys: row_keys.into(),
            role,
            provenance: provenance.into(),
            columns: columns.into(),
        })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }
    pub const fn profile(&self) -> PrecisionProfile {
        self.profile
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
    pub const fn role(&self) -> EvaluationRole {
        self.role
    }
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
    pub fn rows(&self) -> usize {
        self.row_keys.len()
    }
    pub fn column_typed(&self, index: usize) -> SemanticResult<&NumericColumn> {
        self.columns
            .get(index)
            .ok_or(SemanticError::Invalid("source column out of bounds"))
    }

    /// Read an f32 source only when the frame's profile permits f32 storage.
    pub fn column(&self, index: usize) -> SemanticResult<&[f32]> {
        self.column_typed(index)?.as_f32()
    }

    pub fn aligned_with(&self, other: &Self) -> bool {
        self.profile == other.profile
            && self.schema == other.schema
            && self.row_domain == other.row_domain
            && self.row_keys == other.row_keys
            && self.role == other.role
    }
}

/// Public graph-construction input. The frame determines whether validated
/// weights are stored as f32 or f64; f64 input never forces mixed storage.
#[derive(Clone, Copy, Debug)]
pub struct GraphEdge {
    pub left: usize,
    pub right: usize,
    pub weight: f64,
}

/// Stored graph topology, deliberately separate from profile-bound weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphEndpoints {
    pub left: usize,
    pub right: usize,
}

/// Explicit undirected edge multiset. Duplicate edges contribute repeatedly;
/// self-loops and nonpositive/nonfinite weights fail instead of being rewritten.
#[derive(Clone, Debug)]
pub struct NeighborGraph {
    id: u64,
    frame_id: u64,
    profile: PrecisionProfile,
    endpoints: Arc<[GraphEndpoints]>,
    weights: NumericColumn,
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
            || edges.iter().any(|edge| {
                edge.left >= frame.rows()
                    || edge.right >= frame.rows()
                    || edge.left == edge.right
                    || !edge.weight.is_finite()
                    || edge.weight <= 0.0
            })
        {
            return Err(SemanticError::Invalid("invalid bounded neighbor graph"));
        }

        let mut endpoints = Vec::with_capacity(edges.len());
        let weights = match frame.profile() {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                let mut weights = Vec::with_capacity(edges.len());
                for edge in edges {
                    let weight = edge.weight as f32;
                    if !weight.is_finite() || weight <= 0.0 {
                        return Err(SemanticError::Invalid(
                            "graph weight cannot be represented in f32 storage",
                        ));
                    }
                    endpoints.push(GraphEndpoints {
                        left: edge.left,
                        right: edge.right,
                    });
                    weights.push(weight);
                }
                NumericColumn::from(weights)
            }
            PrecisionProfile::Fp64 => {
                let mut weights = Vec::with_capacity(edges.len());
                for edge in edges {
                    endpoints.push(GraphEndpoints {
                        left: edge.left,
                        right: edge.right,
                    });
                    weights.push(edge.weight);
                }
                NumericColumn::from(weights)
            }
        };
        Ok(Self {
            id: next_identity()?,
            frame_id: frame.id,
            profile: frame.profile,
            endpoints: endpoints.into(),
            weights,
            provenance: provenance.into(),
        })
    }
    pub const fn id(&self) -> u64 {
        self.id
    }
    pub const fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub const fn profile(&self) -> PrecisionProfile {
        self.profile
    }
    pub fn edges(&self) -> &[GraphEndpoints] {
        &self.endpoints
    }
    pub fn weights_typed(&self) -> &NumericColumn {
        &self.weights
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
    profile: PrecisionProfile,
    rows: Arc<[usize]>,
    values: NumericColumn,
    provenance: Arc<str>,
}

impl LabelSet {
    pub fn new(
        frame: &FeatureFrame,
        mut labels: Vec<(usize, f32)>,
        provenance: String,
    ) -> SemanticResult<Self> {
        if labels.len() > frame.rows()
            || labels
                .iter()
                .any(|(row, value)| *row >= frame.rows() || !value.is_finite())
        {
            return Err(SemanticError::Invalid("invalid labeled-row context"));
        }
        labels.sort_by_key(|(row, _)| *row);
        if labels.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(SemanticError::Invalid("duplicate labeled row"));
        }
        let (rows, values): (Vec<_>, Vec<_>) = labels.into_iter().unzip();
        Self::from_typed(frame, rows, NumericColumn::from(values), provenance)
    }

    pub fn new_f64(
        frame: &FeatureFrame,
        mut labels: Vec<(usize, f64)>,
        provenance: String,
    ) -> SemanticResult<Self> {
        if labels.len() > frame.rows()
            || labels
                .iter()
                .any(|(row, value)| *row >= frame.rows() || !value.is_finite())
        {
            return Err(SemanticError::Invalid("invalid labeled-row context"));
        }
        labels.sort_by_key(|(row, _)| *row);
        if labels.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(SemanticError::Invalid("duplicate labeled row"));
        }
        let (rows, values): (Vec<_>, Vec<_>) = labels.into_iter().unzip();
        Self::from_typed(frame, rows, NumericColumn::from(values), provenance)
    }

    fn from_typed(
        frame: &FeatureFrame,
        rows: Vec<usize>,
        values: NumericColumn,
        provenance: String,
    ) -> SemanticResult<Self> {
        if provenance.is_empty()
            || provenance.len() > 4096
            || rows.len() != values.len()
            || rows.iter().any(|row| *row >= frame.rows())
            || !values.finite()
            || !values.supports_profile(frame.profile())
        {
            return Err(SemanticError::Invalid("invalid labeled-row context"));
        }
        Ok(Self {
            id: next_identity()?,
            frame_id: frame.id,
            profile: frame.profile,
            rows: rows.into(),
            values,
            provenance: provenance.into(),
        })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }
    pub const fn frame_id(&self) -> u64 {
        self.frame_id
    }
    pub const fn profile(&self) -> PrecisionProfile {
        self.profile
    }
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
    pub fn values_typed(&self) -> &NumericColumn {
        &self.values
    }
    /// Read f32 labels only from an f32-storage profile.
    pub fn values(&self) -> SemanticResult<&[f32]> {
        self.values.as_f32()
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
    fn profile_storage_and_alignment_do_not_narrow_fp64() {
        let f64_frame = FeatureFrame::new_f64(
            vec!["a".into()],
            "dataset-A".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "f64 fixture".into(),
            vec![vec![1.0, f64::from_bits(1.0f64.to_bits() + 1)]],
        )
        .unwrap();
        assert_eq!(f64_frame.profile(), PrecisionProfile::Fp64);
        assert!(f64_frame.column(0).is_err());
        assert_eq!(
            f64_frame.column_typed(0).unwrap().as_f64().unwrap()[1].to_bits(),
            1.0f64.to_bits() + 1
        );
        let f32_profile = FeatureFrame::with_profile(
            PrecisionProfile::Fp32,
            vec!["a".into()],
            "dataset-A".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "f32 fixture".into(),
            vec![NumericColumn::from(vec![1.0f32, 2.0])],
        )
        .unwrap();
        assert!(!f64_frame.aligned_with(&f32_profile));
        assert!(FeatureFrame::with_profile(
            PrecisionProfile::Fp64,
            vec!["a".into()],
            "dataset-A".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "wrong storage".into(),
            vec![NumericColumn::from(vec![1.0f32, 2.0])],
        )
        .is_err());
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
        assert!(FeatureFrame::new_f64(
            vec!["a".into()],
            "rows".into(),
            vec![0, 1],
            EvaluationRole::Discovery,
            "fixture".into(),
            vec![vec![0.0, f64::NAN]]
        )
        .is_err());
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
    fn graph_and_label_inputs_preserve_exact_frame_binding_and_profile() {
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
                weight: f64::NAN,
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
        assert_eq!(graph.edges()[0], GraphEndpoints { left: 0, right: 1 });
        assert_eq!(graph.weights_typed().as_f32().unwrap(), &[1.0, 1.0]);
        assert_eq!(graph.frame_id(), input.id());
        let labels = LabelSet::new(&input, vec![(2, 3.0), (0, 1.0)], "supplied".into()).unwrap();
        assert_eq!(labels.rows(), &[0, 2]);
        assert_eq!(labels.values().unwrap(), &[1.0, 3.0]);
        assert_eq!(labels.frame_id(), input.id());
        assert!(LabelSet::new(&input, vec![(0, 1.0), (0, 2.0)], "duplicate".into()).is_err());
        assert!(LabelSet::new(&input, vec![(3, 1.0)], "out-of-bounds".into()).is_err());

        let f64_frame = FeatureFrame::new_f64(
            vec!["a".into()],
            "dataset-A".into(),
            vec![10, 11, 12],
            EvaluationRole::Discovery,
            "f64 graph".into(),
            vec![vec![1.0, 2.0, 3.0]],
        )
        .unwrap();
        let f64_graph = NeighborGraph::new(&f64_frame, vec![edge], "f64 edge".into()).unwrap();
        assert_eq!(f64_graph.weights_typed().as_f64().unwrap(), &[1.0]);
        let f64_labels = LabelSet::new_f64(
            &f64_frame,
            vec![(1, f64::from_bits(1.0f64.to_bits() + 1))],
            "f64 labels".into(),
        )
        .unwrap();
        assert!(f64_labels.values().is_err());
        assert_eq!(
            f64_labels.values_typed().as_f64().unwrap()[0].to_bits(),
            1.0f64.to_bits() + 1
        );
        assert!(LabelSet::new(&f64_frame, vec![(0, 1.0)], "wrong label storage".into()).is_err());
    }
}
