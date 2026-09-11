use std::{collections::BTreeMap, sync::Arc};

use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

use super::*;
use crate::semantic::{
    CandidateRegistry, EvaluationRole, EvidenceChannel, EvidenceDefinition, EvidenceRecord,
    FeatureFrame, MaterializedColumns, NativeEvidenceExecutor, NumericColumn, ProgramLimits,
    SemanticResult,
};

struct ExactWeakCounts {
    calls: usize,
    saw_no_coordinates: bool,
    coordinates: Vec<f32>,
}

impl ExactWeakCounts {
    fn new() -> Self {
        Self {
            calls: 0,
            saw_no_coordinates: false,
            coordinates: Vec::new(),
        }
    }
}

impl NativeEvidenceExecutor for ExactWeakCounts {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    fn materialize(
        &mut self,
        _: &CandidateRegistry,
        _: &FeatureFrame,
        _: &[FeatureId],
        _: Option<&MaterializedColumns>,
        _: usize,
    ) -> SemanticResult<MaterializedColumns> {
        panic!("selection-only mock must not materialize")
    }

    fn evaluate_channel(
        &mut self,
        _: &EvidenceDefinition,
        _: &[FeatureId],
        _: &MaterializedColumns,
        _: Option<&MaterializedColumns>,
        _: usize,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        panic!("selection-only mock must not evaluate evidence")
    }

    fn wants_pareto_frontier(&self) -> bool {
        true
    }

    /// Deliberately quadratic test oracle only. Production selection consumes
    /// native weak counts and never repeats this pairwise dominance scan.
    fn pareto_weak_dominator_counts(
        &mut self,
        request: ParetoFrontierRequest<'_>,
        _: usize,
    ) -> SemanticResult<Option<Vec<u64>>> {
        self.calls += 1;
        let Some(coordinates) = request.fp32_coordinates() else {
            self.saw_no_coordinates = true;
            return Ok(Some(Vec::new()));
        };
        assert_eq!(request.profile(), PrecisionProfile::Fp32);
        assert_eq!(
            coordinates.len(),
            request.objective_count() * request.candidate_count()
        );
        self.coordinates = coordinates.to_vec();
        let mut counts = vec![0u64; request.candidate_count()];
        for candidate in 0..request.candidate_count() {
            for other in 0..request.candidate_count() {
                if (0..request.objective_count()).all(|axis| {
                    coordinates[axis * request.candidate_count() + other]
                        <= coordinates[axis * request.candidate_count() + candidate]
                }) {
                    counts[candidate] += 1;
                }
            }
        }
        Ok(Some(counts))
    }
}

struct NoLocal;

impl NativeEvidenceExecutor for NoLocal {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    fn materialize(
        &mut self,
        _: &CandidateRegistry,
        _: &FeatureFrame,
        _: &[FeatureId],
        _: Option<&MaterializedColumns>,
        _: usize,
    ) -> SemanticResult<MaterializedColumns> {
        panic!("selection-only mock must not materialize")
    }

    fn evaluate_channel(
        &mut self,
        _: &EvidenceDefinition,
        _: &[FeatureId],
        _: &MaterializedColumns,
        _: Option<&MaterializedColumns>,
        _: usize,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        panic!("selection-only mock must not evaluate evidence")
    }

    fn pareto_weak_dominator_counts(
        &mut self,
        _: ParetoFrontierRequest<'_>,
        _: usize,
    ) -> SemanticResult<Option<Vec<u64>>> {
        panic!("ordinary executor must not receive local Pareto coordinates")
    }
}

fn table_f32(
    profile: PrecisionProfile,
    primary: &[f32],
    objective_axes: &[&[f32]],
) -> (EvidenceTable, Vec<FeatureId>, Vec<EvidenceChannel>) {
    assert!(!primary.is_empty());
    assert!(objective_axes
        .iter()
        .all(|axis| axis.len() == primary.len()));
    let schema = (0..primary.len())
        .map(|index| format!("candidate-{index}"))
        .collect::<Vec<_>>();
    let frame = Arc::new(
        FeatureFrame::with_profile(
            profile,
            schema.clone(),
            "selection-test-domain".into(),
            vec![10, 20],
            EvaluationRole::Discovery,
            "selection test fixture".into(),
            schema
                .iter()
                .map(|_| NumericColumn::from(vec![1.0f32, 2.0]))
                .collect(),
        )
        .unwrap(),
    );
    let registry = CandidateRegistry::new(schema, profile, ProgramLimits::default()).unwrap();
    let candidates = (0..primary.len())
        .map(|index| registry.source(index).unwrap())
        .collect::<Vec<_>>();
    let materialized = MaterializedColumns::from_columns(
        &registry,
        &frame,
        candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                (
                    *candidate,
                    frame.column_typed(index).unwrap().shared_clone(),
                )
            })
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    let mut channels =
        vec![EvidenceChannel::new("primary".into(), EvidenceDefinition::BinaryOccupancy).unwrap()];
    channels.extend((0..objective_axes.len()).map(|axis| {
        EvidenceChannel::new(
            format!("objective-{axis}"),
            EvidenceDefinition::BinaryOccupancy,
        )
        .unwrap()
    }));
    let mut records = Vec::with_capacity(candidates.len() * channels.len());
    for (row, &candidate) in candidates.iter().enumerate() {
        records.push(EvidenceRecord {
            candidate,
            channel: channels[0].id(),
            value: EvidenceValue::measured_f32(primary[row], 2),
        });
        for (axis, values) in objective_axes.iter().enumerate() {
            records.push(EvidenceRecord {
                candidate,
                channel: channels[axis + 1].id(),
                value: EvidenceValue::measured_f32(values[row], 2),
            });
        }
    }
    (
        EvidenceTable {
            owner: 1,
            id: 2,
            round: 3,
            frame,
            candidates: candidates.clone(),
            channels: channels.clone(),
            records,
            training_lineage: (0..candidates.len()).map(|_| Arc::from([])).collect(),
            contextual_training: Arc::from([]),
            materialized,
            backend: GAFIME_BACKEND_CPU,
        },
        candidates,
        channels,
    )
}

fn policy(channels: &[EvidenceChannel], directions: &[Direction]) -> SelectionPolicy {
    SelectionPolicy {
        primary: channels[0].id(),
        direction: Direction::Maximize,
        constraints: Vec::new(),
        missing: MissingEvidence::Error,
        limit: 16,
        pareto_objectives: directions
            .iter()
            .enumerate()
            .map(|(axis, direction)| EvidenceObjective {
                channel: channels[axis + 1].id(),
                direction: *direction,
            })
            .collect(),
    }
}

#[test]
fn exact_local_counts_match_the_core_frontier_and_preserve_primary_ranking() {
    let primary = [0.2, 0.9, 0.6, 0.8, 0.7, 0.1];
    let minimum = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0];
    let maximum = [1.0, 4.0, 3.0, 0.0, 1.0, 2.0];
    let (table, candidates, channels) =
        table_f32(PrecisionProfile::Fp32, &primary, &[&minimum, &maximum]);
    let policy = policy(&channels, &[Direction::Minimize, Direction::Maximize]);
    let core = policy.select(&table, 10_000).unwrap();
    let mut local = ExactWeakCounts::new();
    let lowered = policy
        .select_with_executor(&mut local, &table, 10_000, 4096)
        .unwrap();

    assert_eq!(lowered, core);
    assert_eq!(lowered, vec![candidates[1], candidates[2]]);
    assert_eq!(local.calls, 1);
    // Axis 0 is minimization unchanged; axis 1 is maximization negated.
    assert_eq!(&local.coordinates[..primary.len()], &minimum);
    assert_eq!(
        &local.coordinates[primary.len()..],
        &[-1.0, -4.0, -3.0, -0.0, -1.0, -2.0]
    );
}

#[test]
fn equal_signed_zero_vectors_have_zero_strict_dominators() {
    let primary = [0.2, 0.9, 0.1];
    let minimum = [-0.0, 0.0, 1.0];
    let maximum = [0.0, -0.0, -1.0];
    let (table, candidates, channels) =
        table_f32(PrecisionProfile::Fp32, &primary, &[&minimum, &maximum]);
    let policy = policy(&channels, &[Direction::Minimize, Direction::Maximize]);
    let mut local = ExactWeakCounts::new();
    let lowered = policy
        .select_with_executor(&mut local, &table, 10_000, 4096)
        .unwrap();

    assert_eq!(lowered, vec![candidates[1], candidates[0]]);
    assert_eq!(lowered, policy.select(&table, 10_000).unwrap());
    assert!(local.coordinates[0]
        .partial_cmp(&local.coordinates[1])
        .is_some_and(|order| order.is_eq()));
    assert!(local.coordinates[3]
        .partial_cmp(&local.coordinates[4])
        .is_some_and(|order| order.is_eq()));
}

#[test]
fn ordinary_executor_keeps_the_core_path_without_coordinate_allocation() {
    let primary = [0.4, 0.9, 0.2];
    let first = [1.0, 2.0, 1.0];
    let second = [3.0, 1.0, 2.0];
    let (table, _, channels) = table_f32(PrecisionProfile::Fp32, &primary, &[&first, &second]);
    let policy = policy(&channels, &[Direction::Minimize, Direction::Maximize]);
    let mut ordinary = NoLocal;
    assert_eq!(
        policy
            .select_with_executor(&mut ordinary, &table, 10_000, 0)
            .unwrap(),
        policy.select(&table, 10_000).unwrap(),
    );
}

#[test]
fn explicit_local_request_rejects_mixed_before_any_score_narrowing() {
    let primary = [0.4, 0.9, 0.2];
    let first = [1.0, 2.0, 1.0];
    let second = [3.0, 1.0, 2.0];
    let (table, _, channels) = table_f32(PrecisionProfile::Mixed, &primary, &[&first, &second]);
    let policy = policy(&channels, &[Direction::Minimize, Direction::Maximize]);
    let mut local = ExactWeakCounts::new();
    assert_eq!(
        policy
            .select_with_executor(&mut local, &table, 10_000, 4096)
            .unwrap_err(),
        SemanticError::Unsupported(
            "local RT Pareto requires fp32 evidence with two or three objectives"
        ),
    );
    assert_eq!(local.calls, 1);
    assert!(local.saw_no_coordinates);
    assert!(local.coordinates.is_empty());
}

#[test]
fn explicit_local_request_rejects_subnormal_objectives_before_allocation() {
    let primary = [0.4, 0.9, 0.2];
    let first = [f32::from_bits(1), 2.0, 1.0];
    let second = [3.0, 1.0, 2.0];
    let (table, _, channels) = table_f32(PrecisionProfile::Fp32, &primary, &[&first, &second]);
    let policy = policy(&channels, &[Direction::Minimize, Direction::Maximize]);
    let mut local = ExactWeakCounts::new();
    assert_eq!(
        policy
            .select_with_executor(&mut local, &table, 10_000, 4096)
            .unwrap_err(),
        SemanticError::Invalid(
            "local RT Pareto requires finite non-subnormal exact fp32 objectives"
        ),
    );
    assert_eq!(local.calls, 0);
}
