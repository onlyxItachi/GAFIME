use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    AcceptedFeature, AssociationContext, AssociationStatistic, CandidateRegistry, Direction,
    EvaluationRole, EvidenceChannel, EvidenceDefinition, FeatureFrame, FeatureId, FeatureOp,
    FrozenMeans, MissingEvidence, NumericColumn, PredicateComparator, ProgramLimits,
    SelectionPolicy, SemanticError, SemanticSession,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

fn frame(
    profile: PrecisionProfile,
    role: EvaluationRole,
    domain: &str,
    provenance: &str,
    values: Vec<Vec<f64>>,
) -> Arc<FeatureFrame> {
    let rows = values.first().map_or(0, Vec::len);
    let columns: Vec<NumericColumn> = values
        .into_iter()
        .map(|column| match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => NumericColumn::from(
                column
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>(),
            ),
            PrecisionProfile::Fp64 => NumericColumn::from(column),
        })
        .collect();
    Arc::new(
        FeatureFrame::with_profile(
            profile,
            (0..columns.len())
                .map(|index| format!("x{index}"))
                .collect(),
            domain.into(),
            (0..rows as u64).collect(),
            role,
            provenance.into(),
            columns,
        )
        .unwrap(),
    )
}

fn self_channel(reference: FeatureId) -> EvidenceChannel {
    EvidenceChannel::new(
        "self".into(),
        EvidenceDefinition::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContext::Reference { reference },
        },
    )
    .unwrap()
}

fn select_one(channel: &EvidenceChannel) -> SelectionPolicy {
    SelectionPolicy {
        primary: channel.id(),
        pareto_objectives: Vec::new(),
        direction: Direction::Maximize,
        constraints: Vec::new(),
        missing: MissingEvidence::Error,
        limit: 1,
    }
}

fn raw_bits(column: &NumericColumn) -> Vec<u64> {
    match column {
        NumericColumn::F32(values) => values
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        NumericColumn::F64(values) => values.iter().map(|value| value.to_bits()).collect(),
    }
}

fn ordered_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &value in values {
        sum += value;
    }
    sum / values.len() as f32
}

fn ordered_mixed(values: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &value in values {
        sum += f64::from(value);
    }
    (sum / values.len() as f64) as f32
}

fn ordered_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &value in values {
        sum += value;
    }
    sum / values.len() as f64
}

fn expected_frozen_and_product(
    profile: PrecisionProfile,
    input: &[Vec<f64>],
) -> (Vec<u64>, Vec<u64>) {
    match profile {
        PrecisionProfile::Fp32 => {
            let left = input[0]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let right = input[1]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let left_mean = ordered_f32(&left);
            let right_mean = ordered_f32(&right);
            (
                vec![
                    u64::from(left_mean.to_bits()),
                    u64::from(right_mean.to_bits()),
                ],
                left.iter()
                    .zip(&right)
                    .map(|(&left, &right)| ((left - left_mean) * (right - right_mean)).to_bits())
                    .map(u64::from)
                    .collect(),
            )
        }
        PrecisionProfile::Mixed => {
            let left = input[0]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let right = input[1]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let left_mean = ordered_mixed(&left);
            let right_mean = ordered_mixed(&right);
            (
                vec![
                    u64::from(left_mean.to_bits()),
                    u64::from(right_mean.to_bits()),
                ],
                left.iter()
                    .zip(&right)
                    .map(|(&left, &right)| ((left - left_mean) * (right - right_mean)).to_bits())
                    .map(u64::from)
                    .collect(),
            )
        }
        PrecisionProfile::Fp64 => {
            let left = &input[0];
            let right = &input[1];
            let left_mean = ordered_f64(left);
            let right_mean = ordered_f64(right);
            (
                vec![left_mean.to_bits(), right_mean.to_bits()],
                left.iter()
                    .zip(right)
                    .map(|(&left, &right)| ((left - left_mean) * (right - right_mean)).to_bits())
                    .collect(),
            )
        }
    }
}

fn run_fitted_interaction(profile: PrecisionProfile, workers: usize) -> (Vec<u64>, Vec<u64>) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap()
        .install(|| {
            let input = vec![
                vec![-3.0, -1.0, 0.5, 2.0, 4.0],
                vec![-2.0, 3.0, -1.0, 4.0, 1.0],
            ];
            let training = frame(
                profile,
                EvaluationRole::Discovery,
                "training",
                "ordered mean fixture",
                input,
            );
            let registry = CandidateRegistry::new(
                training.schema().to_vec(),
                profile,
                ProgramLimits::default(),
            )
            .unwrap();
            let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
            let (left, right) = {
                let round = session.begin_round(&[]).unwrap();
                (round.source(0).unwrap(), round.source(1).unwrap())
            };
            let mut core = CoreEvidenceExecutor::default();
            let candidate = session
                .propose_centered_interactions(&mut core, &training, &[left, right], &[2], 1)
                .unwrap()[0];
            let means = match session.registry().unwrap().program(candidate).unwrap().op() {
                FeatureOp::CenteredProduct { mean_bits, .. } => match mean_bits {
                    FrozenMeans::F32(bits) => bits.iter().map(|bits| u64::from(*bits)).collect(),
                    FrozenMeans::F64(bits) => bits.clone(),
                },
                _ => panic!("bulk proposal did not create a centered product"),
            };
            let channel = self_channel(candidate);
            let table = session
                .evaluate(
                    &mut core,
                    Arc::clone(&training),
                    &[candidate],
                    std::slice::from_ref(&channel),
                )
                .unwrap();
            let accepted = session.accept(&table, &select_one(&channel)).unwrap();
            assert_eq!(accepted.len(), 1);
            let output = session
                .materialize_accepted(&mut core, &training, &accepted)
                .unwrap();
            (means, raw_bits(output.get_typed(candidate).unwrap()))
        })
}

#[test]
fn fitted_means_are_ordered_profile_native_and_rayon_deterministic() {
    let input = vec![
        vec![-3.0, -1.0, 0.5, 2.0, 4.0],
        vec![-2.0, 3.0, -1.0, 4.0, 1.0],
    ];
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let expected = expected_frozen_and_product(profile, &input);
        let serial = run_fitted_interaction(profile, 1);
        let parallel = run_fitted_interaction(profile, 4);
        assert_eq!(
            serial, expected,
            "ordered arithmetic oracle for {profile:?}"
        );
        assert_eq!(parallel, serial, "Rayon worker count for {profile:?}");
    }
}

#[test]
fn frozen_region_has_distinct_term_arity_and_exact_inference_membership() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let training = frame(
            profile,
            EvaluationRole::Discovery,
            "training",
            "region fixture",
            vec![vec![-1.0, 0.0, 1.0, 2.0]],
        );
        let registry = CandidateRegistry::new(
            training.schema().to_vec(),
            profile,
            ProgramLimits {
                max_logical_arity: 1,
                max_source_arity: 1,
                ..ProgramLimits::default()
            },
        )
        .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
        let region = {
            let mut round = session.begin_round(&[]).unwrap();
            let atom = round.source(0).unwrap();
            let unaccepted_current = round.softsign(atom).unwrap();
            assert!(if profile == PrecisionProfile::Fp64 {
                round
                    .hard_predicate_f64(unaccepted_current, PredicateComparator::GreaterThan, 0.0)
                    .is_err()
            } else {
                round
                    .hard_predicate(unaccepted_current, PredicateComparator::GreaterThan, 0.0)
                    .is_err()
            });
            let lower = if profile == PrecisionProfile::Fp64 {
                round
                    .hard_predicate_f64(atom, PredicateComparator::GreaterThan, 0.0)
                    .unwrap()
            } else {
                round
                    .hard_predicate(atom, PredicateComparator::GreaterThan, 0.0)
                    .unwrap()
            };
            let upper = if profile == PrecisionProfile::Fp64 {
                round
                    .hard_predicate_f64(atom, PredicateComparator::LessEqual, 1.0)
                    .unwrap()
            } else {
                round
                    .hard_predicate(atom, PredicateComparator::LessEqual, 1.0)
                    .unwrap()
            };
            let region = round.decision_region(vec![upper, lower]).unwrap();
            assert_eq!(round.decision_region(vec![lower, upper]).unwrap(), region);
            assert!(round.decision_region(vec![lower, lower]).is_err());
            let inverted_upper = if profile == PrecisionProfile::Fp64 {
                round
                    .hard_predicate_f64(atom, PredicateComparator::LessEqual, 0.0)
                    .unwrap()
            } else {
                round
                    .hard_predicate(atom, PredicateComparator::LessEqual, 0.0)
                    .unwrap()
            };
            assert!(round.decision_region(vec![lower, inverted_upper]).is_err());
            region
        };
        let program = session.registry().unwrap().program(region).unwrap();
        assert_eq!(program.logical_arity(), 1);
        assert_eq!(program.source_arity(), 1);
        assert_eq!(program.region_term_count(), 2);
        assert_eq!(program.depth(), 2);

        let channel = self_channel(region);
        let mut core = CoreEvidenceExecutor::default();
        let table = session
            .evaluate(
                &mut core,
                Arc::clone(&training),
                &[region],
                std::slice::from_ref(&channel),
            )
            .unwrap();
        let accepted = session.accept(&table, &select_one(&channel)).unwrap();
        let inference = frame(
            profile,
            EvaluationRole::Inference,
            "inference",
            "never fitted",
            vec![vec![-1.0, 0.5, 1.0, 2.0]],
        );
        let output = session
            .materialize_accepted(&mut core, &inference, &accepted)
            .unwrap();
        match output.get_typed(region).unwrap() {
            NumericColumn::F32(values) => assert_eq!(values.as_slice(), &[0.0, 1.0, 1.0, 0.0]),
            NumericColumn::F64(values) => assert_eq!(values.as_slice(), &[0.0, 1.0, 1.0, 0.0]),
        }
    }
}

#[test]
fn fitted_proposal_counts_distinct_atoms_and_reuses_retained_sources() {
    let training = frame(
        PrecisionProfile::Mixed,
        EvaluationRole::Discovery,
        "training",
        "retained source fixture",
        vec![vec![-2.0, -1.0, 1.0, 2.0], vec![-3.0, -1.0, 1.0, 3.0]],
    );
    let registry = CandidateRegistry::new(
        training.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (left, right) = {
        let round = session.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let mut core = CoreEvidenceExecutor::default();
    let source_channel = self_channel(left);
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&training),
            &[left],
            std::slice::from_ref(&source_channel),
        )
        .unwrap();
    let accepted = session
        .accept(&table, &select_one(&source_channel))
        .unwrap();
    assert_eq!(accepted.len(), 1);
    let retained_before = session.retained_bytes();
    let hits_before = core.retained_hits();
    let mean_columns_before = core.fitted_mean_columns();
    let mean_rows_before = core.fitted_mean_rows();

    // Duplicate caller atoms collapse before native work. Only the retained
    // source is reused; the other raw source still has one ordered mean scan.
    let proposed = session
        .propose_centered_interactions(&mut core, &training, &[right, left, left], &[2], 1)
        .unwrap();
    assert_eq!(proposed.len(), 1);
    assert_eq!(core.retained_hits() - hits_before, 1);
    assert_eq!(core.fitted_mean_columns() - mean_columns_before, 2);
    assert_eq!(
        core.fitted_mean_rows() - mean_rows_before,
        2 * training.rows()
    );
    assert_eq!(session.retained_bytes(), retained_before);

    // New current-round algebraic nodes remain legal for legacy composition,
    // but the fitted bulk's explicit atom contract rejects them before any
    // mean materialization. The separate ledger-full regression below covers
    // failure after a successful native fit as well.
    let unaccepted_current = session.current_round().unwrap().softsign(left).unwrap();
    assert!(session
        .propose_centered_interactions(&mut core, &training, &[unaccepted_current, right], &[2], 1,)
        .is_err());
    assert_eq!(session.retained_bytes(), retained_before);
    assert_eq!(core.fitted_mean_columns() - mean_columns_before, 2);
}

#[test]
fn accepted_atoms_authorize_new_predicates_and_fitted_bulk() {
    let training = frame(
        PrecisionProfile::Mixed,
        EvaluationRole::Discovery,
        "training",
        "accepted atom fixture",
        vec![vec![-2.0, -1.0, 1.0, 2.0], vec![-3.0, -1.0, 1.0, 3.0]],
    );
    let registry = CandidateRegistry::new(
        training.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (left, right, derived) = {
        let mut round = session.begin_round(&[]).unwrap();
        let left = round.source(0).unwrap();
        let right = round.source(1).unwrap();
        let derived = round.softsign(left).unwrap();
        (left, right, derived)
    };
    let channel = self_channel(derived);
    let mut core = CoreEvidenceExecutor::default();
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&training),
            &[derived],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    let accepted = session.accept(&table, &select_one(&channel)).unwrap();
    assert_eq!(accepted[0].feature(), derived);

    {
        let mut round = session.begin_round(&accepted).unwrap();
        assert_eq!(
            round
                .hard_predicate(derived, PredicateComparator::GreaterThan, 0.0)
                .unwrap(),
            round
                .hard_predicate(derived, PredicateComparator::GreaterThan, 0.0)
                .unwrap()
        );
    }
    let proposed = session
        .propose_centered_interactions(&mut core, &training, &[derived, right], &[2], 1)
        .unwrap();
    assert_eq!(proposed.len(), 1);
    assert_eq!(
        session
            .registry()
            .unwrap()
            .training_lineage(proposed[0])
            .unwrap()
            .len(),
        1
    );
    assert!(session.registry().unwrap().program(left).is_ok());
}

#[test]
fn fitting_ledger_full_failure_rolls_back_registry_round_cache_and_lineage() {
    let base_values = vec![vec![-2.0, -1.0, 1.0, 2.0], vec![-3.0, -1.0, 1.0, 3.0]];
    let first = frame(
        PrecisionProfile::Mixed,
        EvaluationRole::Discovery,
        "first-training",
        "first equal-state fit",
        base_values.clone(),
    );
    let registry = CandidateRegistry::new(
        first.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits {
            max_nodes: 4,
            ..ProgramLimits::default()
        },
    )
    .unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (left, right) = {
        let round = session.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let mut core = CoreEvidenceExecutor::default();
    let candidate = session
        .propose_centered_interactions(&mut core, &first, &[left, right], &[2], 1)
        .unwrap()[0];
    let channel = self_channel(candidate);
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&first),
            &[candidate],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    let accepted: Vec<AcceptedFeature> = session.accept(&table, &select_one(&channel)).unwrap();
    assert_eq!(accepted.len(), 1);
    let retained_before = session.retained_bytes();
    assert!(retained_before > 0);
    assert_eq!(table.training_bindings(candidate).unwrap().len(), 1);
    assert_eq!(accepted[0].training_bindings().len(), 1);

    for index in 1..4 {
        let same_state = frame(
            PrecisionProfile::Mixed,
            EvaluationRole::Discovery,
            &format!("same-state-{index}"),
            &format!("independent equal-state fit {index}"),
            base_values.clone(),
        );
        assert_eq!(
            session
                .propose_centered_interactions(&mut core, &same_state, &[left, right], &[2], 1)
                .unwrap(),
            vec![candidate]
        );
    }
    assert_eq!(session.registry().unwrap().training_binding_count(), 4);
    assert_eq!(
        session
            .registry()
            .unwrap()
            .training_lineage(candidate)
            .unwrap()
            .len(),
        4
    );

    let different_state = frame(
        PrecisionProfile::Mixed,
        EvaluationRole::Discovery,
        "different-state",
        "would allocate a distinct frozen program",
        vec![vec![-1.0, 0.0, 2.0, 3.0], vec![-1.0, 1.0, 3.0, 5.0]],
    );
    assert_eq!(
        session
            .propose_centered_interactions(&mut core, &different_state, &[left, right], &[2], 1)
            .unwrap_err(),
        SemanticError::Unsupported("semantic fitting provenance limit exceeded")
    );
    assert_eq!(session.round(), 1);
    assert_eq!(session.retained_bytes(), retained_before);
    assert_eq!(session.registry().unwrap().training_binding_count(), 4);
    assert_eq!(table.training_bindings(candidate).unwrap().len(), 1);
    assert_eq!(accepted[0].training_bindings().len(), 1);

    // The failed state was appended before provenance admission, so this
    // exact manual declaration proves both the operation map and program slot
    // rolled back rather than leaving a stale, unusable identity behind.
    let recovered = session
        .current_round()
        .unwrap()
        .centered_product(vec![left, right], vec![1.0, 2.0])
        .unwrap();
    assert!(session.registry().unwrap().program(recovered).is_ok());
    assert!(session
        .registry()
        .unwrap()
        .training_bindings(recovered)
        .unwrap()
        .is_empty());
    assert!(session
        .materialize_accepted(&mut core, &first, &accepted)
        .unwrap()
        .contains(candidate));
}
