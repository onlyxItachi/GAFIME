use std::collections::HashSet;

use gafime_cpu::{kernels::precision::pearson_mixed, simd::pearson_sums};
use rayon::prelude::*;

/// This spike deliberately accepts only the existing Core mixed route.
#[allow(dead_code)] // The selector enumerates routes that this probe must reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProbeBackend {
    Core,
    Cuda,
    Rocm,
    Metal,
    Auto,
}

#[allow(dead_code)] // The selector enumerates profiles that this probe must reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProbePrecision {
    Fp32,
    Mixed,
    Fp64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selector {
    backend: ProbeBackend,
    precision: ProbePrecision,
}

impl Selector {
    pub(super) const fn core_mixed() -> Self {
        Self::new(ProbeBackend::Core, ProbePrecision::Mixed)
    }

    pub(super) const fn new(backend: ProbeBackend, precision: ProbePrecision) -> Self {
        Self { backend, precision }
    }
}

#[derive(Clone, Debug)]
pub(super) struct GraphEdge {
    left: usize,
    right: usize,
    weight: f32,
}

impl GraphEdge {
    pub(super) const fn new(left: usize, right: usize, weight: f32) -> Self {
        Self {
            left,
            right,
            weight,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SparseGraph {
    edges: Vec<GraphEdge>,
}

impl SparseGraph {
    pub(super) fn new(edges: Vec<GraphEdge>) -> Self {
        Self { edges }
    }
}

/// Labels are optional evidence, not a mandatory target vector.
#[derive(Clone, Debug)]
pub(super) struct LabeledRows {
    rows: Vec<usize>,
    values: Vec<f32>,
}

impl LabeledRows {
    pub(super) fn new(rows: Vec<usize>, values: Vec<f32>) -> Self {
        Self { rows, values }
    }
}

/// Validated immutable inputs for the three fixed experimental candidates.
#[derive(Clone, Debug)]
pub(super) struct ProbeInput {
    columns: Vec<Vec<f32>>,
    aligned_view: Vec<Vec<f32>>,
    anchor_column: usize,
    graph: Option<SparseGraph>,
    labels: Option<LabeledRows>,
}

impl ProbeInput {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        columns: Vec<Vec<f32>>,
        aligned_view: Vec<Vec<f32>>,
        anchor_column: usize,
        graph: Option<SparseGraph>,
        labels: Option<LabeledRows>,
    ) -> Result<Self, ProbeError> {
        let input = Self {
            columns,
            aligned_view,
            anchor_column,
            graph,
            labels,
        };
        validate_input(&input)?;
        Ok(input)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatrixSide {
    Original,
    AlignedView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ProbeError {
    UnsupportedBackend(ProbeBackend),
    UnsupportedPrecision(ProbePrecision),
    EmptyColumns,
    TooFewColumns {
        found: usize,
    },
    EmptyRows,
    TooFewRows {
        found: usize,
    },
    ColumnCount {
        side: MatrixSide,
        expected: usize,
        found: usize,
    },
    ColumnShape {
        side: MatrixSide,
        column: usize,
        expected: usize,
        found: usize,
    },
    NonFiniteInput {
        side: MatrixSide,
        column: usize,
        row: usize,
    },
    AnchorOutOfBounds {
        anchor: usize,
        columns: usize,
    },
    EmptyGraph,
    GraphEndpointOutOfBounds {
        edge: usize,
        rows: usize,
    },
    GraphSelfLoop {
        edge: usize,
    },
    GraphWeightInvalid {
        edge: usize,
    },
    LabelShape {
        rows: usize,
        values: usize,
    },
    EmptyLabels,
    TooFewLabels {
        found: usize,
    },
    LabelIndexOutOfBounds {
        label: usize,
        row: usize,
        rows: usize,
    },
    DuplicateLabelRow {
        label: usize,
        row: usize,
    },
    NonFiniteLabel {
        label: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnavailableReason {
    NotRequested,
    CandidateNonFinite,
    AlternateViewNonFinite,
    ConstantCandidate,
    ConstantAnchor,
    ConstantAlternateView,
    ConstantLabels,
    ZeroGraphEnergy,
    InsufficientPairs,
    NonFiniteStatistics,
    NonPositiveVariance,
    ArithmeticFailure,
}

impl UnavailableReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::CandidateNonFinite => "candidate_nonfinite",
            Self::AlternateViewNonFinite => "alternate_view_nonfinite",
            Self::ConstantCandidate => "constant_candidate",
            Self::ConstantAnchor => "constant_anchor",
            Self::ConstantAlternateView => "constant_alternate_view",
            Self::ConstantLabels => "constant_labels",
            Self::ZeroGraphEnergy => "zero_graph_energy",
            Self::InsufficientPairs => "insufficient_pairs",
            Self::NonFiniteStatistics => "nonfinite_statistics",
            Self::NonPositiveVariance => "nonpositive_variance",
            Self::ArithmeticFailure => "arithmetic_failure",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum EvidenceValue {
    Value(f64),
    Unavailable(UnavailableReason),
}

impl EvidenceValue {
    fn absolute(self) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.abs()),
            unavailable => unavailable,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct EvidenceRow {
    pub(super) candidate: &'static str,
    pub(super) redundancy_abs_pearson: EvidenceValue,
    pub(super) paired_view_consistency: EvidenceValue,
    pub(super) graph_normalized_dirichlet: EvidenceValue,
    pub(super) hybrid_labeled_pearson: EvidenceValue,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ProbeReport {
    pub(super) rows: Vec<EvidenceRow>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Candidate {
    Identity { column: usize },
    AbsoluteDifference { left: usize, right: usize },
    Softsign { column: usize },
}

impl Candidate {
    pub(super) const fn catalog() -> [Self; 3] {
        [
            Self::Identity { column: 0 },
            Self::AbsoluteDifference { left: 0, right: 1 },
            Self::Softsign { column: 0 },
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Identity { .. } => "identity_col0",
            Self::AbsoluteDifference { .. } => "abs_difference_col0_col1",
            Self::Softsign { .. } => "softsign_col0",
        }
    }

    pub(super) fn materialize_into(self, columns: &[Vec<f32>], output: &mut Vec<f32>) {
        output.clear();
        let rows = columns[0].len();
        output.reserve(rows);
        match self {
            Self::Identity { column } => output.extend_from_slice(&columns[column]),
            Self::AbsoluteDifference { left, right } => {
                for (&left_value, &right_value) in columns[left].iter().zip(&columns[right]) {
                    output.push((left_value - right_value).abs());
                }
            }
            Self::Softsign { column } => {
                for &value in &columns[column] {
                    output.push(value / (1.0f32 + value.abs()));
                }
            }
        }
    }
}

struct ProbePlan<'a> {
    input: &'a ProbeInput,
    candidates: [Candidate; 3],
}

impl<'a> ProbePlan<'a> {
    fn build(input: &'a ProbeInput, selector: Selector) -> Result<Self, ProbeError> {
        if selector.backend != ProbeBackend::Core {
            return Err(ProbeError::UnsupportedBackend(selector.backend));
        }
        if selector.precision != ProbePrecision::Mixed {
            return Err(ProbeError::UnsupportedPrecision(selector.precision));
        }
        Ok(Self {
            input,
            candidates: Candidate::catalog(),
        })
    }

    fn evaluate(&self) -> ProbeReport {
        let anchor = &self.input.columns[self.input.anchor_column];
        let rows = self
            .candidates
            .par_iter()
            .map_init(ProbeScratch::default, |scratch, candidate| {
                evaluate_candidate(*candidate, self.input, anchor, scratch)
            })
            .collect();
        ProbeReport { rows }
    }
}

/// Runs the fixed catalog with f32 materialization and f64 mixed reductions.
pub(super) fn run(input: &ProbeInput, selector: Selector) -> Result<ProbeReport, ProbeError> {
    ProbePlan::build(input, selector).map(|plan| plan.evaluate())
}

#[derive(Default)]
struct ProbeScratch {
    original: Vec<f32>,
    aligned: Vec<f32>,
    labeled_candidate: Vec<f32>,
    labeled_values: Vec<f32>,
}

fn evaluate_candidate(
    candidate: Candidate,
    input: &ProbeInput,
    anchor: &[f32],
    scratch: &mut ProbeScratch,
) -> EvidenceRow {
    candidate.materialize_into(&input.columns, &mut scratch.original);
    candidate.materialize_into(&input.aligned_view, &mut scratch.aligned);
    let original_state = signal_state(&scratch.original);
    let aligned_state = signal_state(&scratch.aligned);
    let redundancy_abs_pearson = match original_state {
        SignalState::Varying => mixed_pearson(
            &scratch.original,
            anchor,
            UnavailableReason::ConstantCandidate,
            UnavailableReason::ConstantAnchor,
        )
        .absolute(),
        SignalState::Constant => EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate),
        SignalState::NonFinite | SignalState::Empty => {
            EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
        }
    };
    let paired_view_consistency = match (original_state, aligned_state) {
        (SignalState::Varying, SignalState::Varying) => mixed_pearson(
            &scratch.original,
            &scratch.aligned,
            UnavailableReason::ConstantCandidate,
            UnavailableReason::ConstantAlternateView,
        ),
        (SignalState::Constant, _) => {
            EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate)
        }
        (_, SignalState::Constant) => {
            EvidenceValue::Unavailable(UnavailableReason::ConstantAlternateView)
        }
        (SignalState::NonFinite | SignalState::Empty, _) => {
            EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
        }
        (_, SignalState::NonFinite | SignalState::Empty) => {
            EvidenceValue::Unavailable(UnavailableReason::AlternateViewNonFinite)
        }
    };
    let graph_normalized_dirichlet = match (&input.graph, original_state) {
        (None, _) => EvidenceValue::Unavailable(UnavailableReason::NotRequested),
        (Some(_), SignalState::Constant) => {
            EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate)
        }
        (Some(_), SignalState::NonFinite | SignalState::Empty) => {
            EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
        }
        (Some(graph), SignalState::Varying) => normalized_dirichlet(&scratch.original, graph),
    };
    let hybrid_labeled_pearson = match (&input.labels, original_state) {
        (None, _) => EvidenceValue::Unavailable(UnavailableReason::NotRequested),
        (Some(_), SignalState::Constant) => {
            EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate)
        }
        (Some(_), SignalState::NonFinite | SignalState::Empty) => {
            EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
        }
        (Some(labels), SignalState::Varying) => labeled_pearson(
            &scratch.original,
            labels,
            &mut scratch.labeled_candidate,
            &mut scratch.labeled_values,
        ),
    };
    EvidenceRow {
        candidate: candidate.name(),
        redundancy_abs_pearson,
        paired_view_consistency,
        graph_normalized_dirichlet,
        hybrid_labeled_pearson,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignalState {
    Empty,
    NonFinite,
    Constant,
    Varying,
}

fn signal_state(values: &[f32]) -> SignalState {
    let Some(&first) = values.first() else {
        return SignalState::Empty;
    };
    if !first.is_finite() {
        return SignalState::NonFinite;
    }
    let mut varying = false;
    for &value in &values[1..] {
        if !value.is_finite() {
            return SignalState::NonFinite;
        }
        if value != first {
            varying = true;
        }
    }
    if varying {
        SignalState::Varying
    } else {
        SignalState::Constant
    }
}

fn mixed_pearson(
    left: &[f32],
    right: &[f32],
    constant_left: UnavailableReason,
    constant_right: UnavailableReason,
) -> EvidenceValue {
    // The public SIMD reduction exposes f64 means and centered sums. Inspect
    // them before calling pearson_mixed because its constant-input sentinel is 0.
    let sums = pearson_sums(left, right);
    if sums.n < 2 {
        return EvidenceValue::Unavailable(UnavailableReason::InsufficientPairs);
    }
    if !sums.sx.is_finite()
        || !sums.sy.is_finite()
        || !sums.sxx.is_finite()
        || !sums.syy.is_finite()
        || !sums.sxy.is_finite()
    {
        return EvidenceValue::Unavailable(UnavailableReason::NonFiniteStatistics);
    }
    if sums.sxx == 0.0 {
        return EvidenceValue::Unavailable(constant_left);
    }
    if sums.syy == 0.0 {
        return EvidenceValue::Unavailable(constant_right);
    }
    if sums.sxx < 0.0 || sums.syy < 0.0 {
        return EvidenceValue::Unavailable(UnavailableReason::NonPositiveVariance);
    }
    let value = pearson_mixed(left, right);
    if value.is_finite() {
        EvidenceValue::Value(value)
    } else {
        EvidenceValue::Unavailable(UnavailableReason::ArithmeticFailure)
    }
}

/// Edge-normalized Dirichlet energy: sum(w * (xi-xj)^2) / sum(w * (xi^2+xj^2)).
fn normalized_dirichlet(signal: &[f32], graph: &SparseGraph) -> EvidenceValue {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for edge in &graph.edges {
        let left = f64::from(signal[edge.left]);
        let right = f64::from(signal[edge.right]);
        let weight = f64::from(edge.weight);
        let difference = left - right;
        numerator += weight * difference * difference;
        denominator += weight * (left * left + right * right);
    }
    if !numerator.is_finite() || !denominator.is_finite() {
        return EvidenceValue::Unavailable(UnavailableReason::ArithmeticFailure);
    }
    if denominator <= 0.0 {
        return EvidenceValue::Unavailable(UnavailableReason::ZeroGraphEnergy);
    }
    let value = numerator / denominator;
    if value.is_finite() {
        EvidenceValue::Value(value)
    } else {
        EvidenceValue::Unavailable(UnavailableReason::ArithmeticFailure)
    }
}

fn labeled_pearson(
    signal: &[f32],
    labels: &LabeledRows,
    labeled_candidate: &mut Vec<f32>,
    labeled_values: &mut Vec<f32>,
) -> EvidenceValue {
    labeled_candidate.clear();
    labeled_values.clear();
    labeled_candidate.reserve(labels.rows.len());
    labeled_values.reserve(labels.values.len());
    for (&row, &value) in labels.rows.iter().zip(&labels.values) {
        labeled_candidate.push(signal[row]);
        labeled_values.push(value);
    }
    mixed_pearson(
        labeled_candidate,
        labeled_values,
        UnavailableReason::ConstantCandidate,
        UnavailableReason::ConstantLabels,
    )
}

fn validate_input(input: &ProbeInput) -> Result<(), ProbeError> {
    if input.columns.is_empty() {
        return Err(ProbeError::EmptyColumns);
    }
    if input.columns.len() < 2 {
        return Err(ProbeError::TooFewColumns {
            found: input.columns.len(),
        });
    }
    let rows = input.columns[0].len();
    if rows == 0 {
        return Err(ProbeError::EmptyRows);
    }
    if rows < 2 {
        return Err(ProbeError::TooFewRows { found: rows });
    }
    validate_matrix(&input.columns, MatrixSide::Original, rows)?;
    if input.aligned_view.len() != input.columns.len() {
        return Err(ProbeError::ColumnCount {
            side: MatrixSide::AlignedView,
            expected: input.columns.len(),
            found: input.aligned_view.len(),
        });
    }
    validate_matrix(&input.aligned_view, MatrixSide::AlignedView, rows)?;
    if input.anchor_column >= input.columns.len() {
        return Err(ProbeError::AnchorOutOfBounds {
            anchor: input.anchor_column,
            columns: input.columns.len(),
        });
    }
    if let Some(graph) = &input.graph {
        validate_graph(graph, rows)?;
    }
    if let Some(labels) = &input.labels {
        validate_labels(labels, rows)?;
    }
    Ok(())
}

fn validate_matrix(
    matrix: &[Vec<f32>],
    side: MatrixSide,
    expected_rows: usize,
) -> Result<(), ProbeError> {
    for (column_index, column) in matrix.iter().enumerate() {
        if column.len() != expected_rows {
            return Err(ProbeError::ColumnShape {
                side,
                column: column_index,
                expected: expected_rows,
                found: column.len(),
            });
        }
        for (row, &value) in column.iter().enumerate() {
            if !value.is_finite() {
                return Err(ProbeError::NonFiniteInput {
                    side,
                    column: column_index,
                    row,
                });
            }
        }
    }
    Ok(())
}

fn validate_graph(graph: &SparseGraph, rows: usize) -> Result<(), ProbeError> {
    if graph.edges.is_empty() {
        return Err(ProbeError::EmptyGraph);
    }
    for (edge_index, edge) in graph.edges.iter().enumerate() {
        if edge.left >= rows || edge.right >= rows {
            return Err(ProbeError::GraphEndpointOutOfBounds {
                edge: edge_index,
                rows,
            });
        }
        if edge.left == edge.right {
            return Err(ProbeError::GraphSelfLoop { edge: edge_index });
        }
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err(ProbeError::GraphWeightInvalid { edge: edge_index });
        }
    }
    Ok(())
}

fn validate_labels(labels: &LabeledRows, rows: usize) -> Result<(), ProbeError> {
    if labels.rows.len() != labels.values.len() {
        return Err(ProbeError::LabelShape {
            rows: labels.rows.len(),
            values: labels.values.len(),
        });
    }
    if labels.rows.is_empty() {
        return Err(ProbeError::EmptyLabels);
    }
    if labels.rows.len() < 2 {
        return Err(ProbeError::TooFewLabels {
            found: labels.rows.len(),
        });
    }
    let mut seen = HashSet::with_capacity(labels.rows.len());
    for (label_index, (&row, &value)) in labels.rows.iter().zip(&labels.values).enumerate() {
        if row >= rows {
            return Err(ProbeError::LabelIndexOutOfBounds {
                label: label_index,
                row,
                rows,
            });
        }
        if !seen.insert(row) {
            return Err(ProbeError::DuplicateLabelRow {
                label: label_index,
                row,
            });
        }
        if !value.is_finite() {
            return Err(ProbeError::NonFiniteLabel { label: label_index });
        }
    }
    Ok(())
}
