//! Independent count-oracle regressions for the bounded binary semantic
//! evidence vocabulary.  These tests intentionally calculate integer count
//! tables locally instead of reusing the Core evidence helpers.

use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    BinaryPairedStatistic, CandidateRegistry, EvaluationRole, EvidenceChannel, EvidenceDefinition,
    EvidenceTable, EvidenceValue, FeatureFrame, LabelSet, NativeEvidenceExecutor, NumericColumn,
    PredicateComparator, ProgramLimits, SemanticError, SemanticSession, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

const BUDGET: usize = 1 << 20;

#[test]
fn region_count_respects_default_and_explicit_logical_arity_limits() {
    for limits in [
        ProgramLimits::default(),
        ProgramLimits {
            max_logical_arity: 64,
            ..ProgramLimits::default()
        },
    ] {
        let mut registry =
            CandidateRegistry::new(vec!["x".into()], PrecisionProfile::Fp32, limits).unwrap();
        let source = registry.source(0).unwrap();
        let regions: Vec<_> = (0..65)
            .map(|index| {
                let predicate = registry
                    .hard_predicate(source, PredicateComparator::GreaterThan, index as f32)
                    .unwrap();
                registry.decision_region(vec![predicate]).unwrap()
            })
            .collect();
        registry
            .region_count(regions[..limits.max_logical_arity].to_vec())
            .unwrap();
        assert!(registry
            .region_count(regions[..=limits.max_logical_arity].to_vec())
            .is_err());
    }
}

fn frame(profile: PrecisionProfile, values: Vec<f64>, provenance: &str) -> Arc<FeatureFrame> {
    let rows = values.len();
    let column = match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => NumericColumn::from(
            values
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ),
        PrecisionProfile::Fp64 => NumericColumn::from(values),
    };
    Arc::new(
        FeatureFrame::with_profile(
            profile,
            vec!["x".into()],
            "binary-evidence-rows".into(),
            (0..rows as u64).collect(),
            EvaluationRole::Discovery,
            provenance.into(),
            vec![column],
        )
        .unwrap(),
    )
}

fn session(input: &FeatureFrame) -> SemanticSession {
    SemanticSession::new(
        CandidateRegistry::new(
            input.schema().to_vec(),
            input.profile(),
            ProgramLimits::default(),
        )
        .unwrap(),
        GAFIME_BACKEND_CPU,
        BUDGET,
    )
    .unwrap()
}

fn labels(frame: &FeatureFrame, values: &[f64]) -> Arc<LabelSet> {
    match frame.profile() {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => Arc::new(
            LabelSet::new(
                frame,
                values
                    .iter()
                    .enumerate()
                    .map(|(row, &value)| (row, value as f32))
                    .collect(),
                "actual binary labels".into(),
            )
            .unwrap(),
        ),
        PrecisionProfile::Fp64 => Arc::new(
            LabelSet::new_f64(
                frame,
                values.iter().copied().enumerate().collect(),
                "actual binary labels".into(),
            )
            .unwrap(),
        ),
    }
}

fn measured(
    table: &EvidenceTable,
    candidate: gafime_orchestrator::semantic::FeatureId,
    channel: &EvidenceChannel,
) -> (f64, usize) {
    match table.value(candidate, channel.id()).unwrap() {
        EvidenceValue::Measured { value, support } => (value, support),
        value => panic!("expected measured evidence, got {value:?}"),
    }
}

fn unavailable(
    table: &EvidenceTable,
    candidate: gafime_orchestrator::semantic::FeatureId,
    channel: &EvidenceChannel,
    reason: UnavailableReason,
    support: usize,
) {
    assert_eq!(
        table.value(candidate, channel.id()).unwrap(),
        EvidenceValue::Unavailable { reason, support }
    );
}

fn gini_f32(zero: usize, one: usize) -> f32 {
    let total = (zero + one) as f32;
    let zero = zero as f32 / total;
    let one = one as f32 / total;
    1.0 - zero * zero - one * one
}

fn gini_f64(zero: usize, one: usize) -> f64 {
    let total = (zero + one) as f64;
    let zero = zero as f64 / total;
    let one = one as f64 / total;
    1.0 - zero * zero - one * one
}

fn expected_gain(profile: PrecisionProfile) -> f64 {
    // outside={0:3,1:1}; inside={0:1,1:3}.  This is a fixed frozen split,
    // not an induced decision tree.
    match profile {
        PrecisionProfile::Fp32 => {
            let parent = gini_f32(4, 4);
            let outside = gini_f32(3, 1);
            let inside = gini_f32(1, 3);
            f64::from((parent - (4.0f32 / 8.0) * outside) - (4.0f32 / 8.0) * inside)
        }
        PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
            let parent = gini_f64(4, 4);
            let outside = gini_f64(3, 1);
            let inside = gini_f64(1, 3);
            (parent - (4.0f64 / 8.0) * outside) - (4.0f64 / 8.0) * inside
        }
    }
}

#[test]
fn binary_counts_finalize_in_the_declared_profile_and_preserve_contexts() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(
            profile,
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "primary binary frame",
        );
        let paired = frame(
            profile,
            vec![-2.0, -1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 5.0],
            "paired binary frame",
        );
        let mut session = session(&input);
        let region = {
            let mut round = session.begin_round(&[]).unwrap();
            let source = round.source(0).unwrap();
            let predicate = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => round
                    .hard_predicate(source, PredicateComparator::GreaterThan, 1.0)
                    .unwrap(),
                PrecisionProfile::Fp64 => round
                    .hard_predicate_f64(source, PredicateComparator::GreaterThan, 1.0)
                    .unwrap(),
            };
            round.decision_region(vec![predicate]).unwrap()
        };
        let occupancy =
            EvidenceChannel::new("occupancy".into(), EvidenceDefinition::BinaryOccupancy).unwrap();
        let agreement = EvidenceChannel::new(
            "agreement".into(),
            EvidenceDefinition::BinaryPaired {
                statistic: BinaryPairedStatistic::Agreement,
                view: Arc::clone(&paired),
            },
        )
        .unwrap();
        let iou = EvidenceChannel::new(
            "iou".into(),
            EvidenceDefinition::BinaryPaired {
                statistic: BinaryPairedStatistic::IntersectionOverUnion,
                view: paired,
            },
        )
        .unwrap();
        let gain = EvidenceChannel::new(
            "gini".into(),
            EvidenceDefinition::BinaryLabeledGiniGain {
                labels: Some(labels(&input, &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])),
            },
        )
        .unwrap();
        let mut core = CoreEvidenceExecutor::default();
        let table = session
            .evaluate(
                &mut core,
                Arc::clone(&input),
                &[region],
                &[
                    occupancy.clone(),
                    agreement.clone(),
                    iou.clone(),
                    gain.clone(),
                ],
            )
            .unwrap();

        let expected_scalar = |value_f32: f32, value_f64: f64| match profile {
            PrecisionProfile::Fp32 => f64::from(value_f32),
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => value_f64,
        };
        assert_eq!(
            measured(&table, region, &occupancy),
            (expected_scalar(0.5, 0.5), 8)
        );
        assert_eq!(
            measured(&table, region, &agreement),
            (expected_scalar(0.75, 0.75), 8)
        );
        assert_eq!(
            measured(&table, region, &iou),
            (expected_scalar(0.6, 0.6), 8)
        );
        assert_eq!(measured(&table, region, &gain), (expected_gain(profile), 8));
        assert_eq!(core.evidence_kernel_calls(), 4);
        assert_eq!(
            occupancy.spec().semantic_name(),
            "binary-occupancy/region-membership/v1"
        );
        assert_eq!(
            agreement.spec().semantic_name(),
            "binary-agreement/aligned-view/v1"
        );
        assert_eq!(
            iou.spec().semantic_name(),
            "binary-intersection-over-union/aligned-view/v1"
        );
        assert_eq!(
            gain.spec().semantic_name(),
            "binary-gini-split-gain/labeled-subset/v1"
        );
    }
}

#[test]
fn binary_evidence_keeps_missing_constant_and_support_states_explicit() {
    let input = frame(
        PrecisionProfile::Mixed,
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "state fixture",
    );
    let paired_zero = frame(
        PrecisionProfile::Mixed,
        vec![-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0],
        "empty paired union",
    );
    let mut session = session(&input);
    let (region, all_zero) = {
        let mut round = session.begin_round(&[]).unwrap();
        let source = round.source(0).unwrap();
        let region_term = round
            .hard_predicate(source, PredicateComparator::GreaterThan, 1.0)
            .unwrap();
        let region = round.decision_region(vec![region_term]).unwrap();
        let all_zero_term = round
            .hard_predicate(source, PredicateComparator::GreaterThan, 100.0)
            .unwrap();
        let all_zero = round.decision_region(vec![all_zero_term]).unwrap();
        (region, all_zero)
    };
    let missing = EvidenceChannel::new(
        "missing".into(),
        EvidenceDefinition::BinaryLabeledGiniGain { labels: None },
    )
    .unwrap();
    let constant_labels = EvidenceChannel::new(
        "constant-labels".into(),
        EvidenceDefinition::BinaryLabeledGiniGain {
            labels: Some(labels(&input, &[0.0; 8])),
        },
    )
    .unwrap();
    let iou = EvidenceChannel::new(
        "iou".into(),
        EvidenceDefinition::BinaryPaired {
            statistic: BinaryPairedStatistic::IntersectionOverUnion,
            view: paired_zero,
        },
    )
    .unwrap();
    let valid_labels = EvidenceChannel::new(
        "valid-labels".into(),
        EvidenceDefinition::BinaryLabeledGiniGain {
            labels: Some(labels(&input, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
        },
    )
    .unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&input),
            &[region, all_zero],
            &[
                missing.clone(),
                constant_labels.clone(),
                iou.clone(),
                valid_labels.clone(),
            ],
        )
        .unwrap();
    unavailable(
        &table,
        region,
        &missing,
        UnavailableReason::MissingLabels,
        0,
    );
    unavailable(
        &table,
        region,
        &constant_labels,
        UnavailableReason::ConstantOperand,
        8,
    );
    unavailable(
        &table,
        all_zero,
        &iou,
        UnavailableReason::ConstantOperand,
        8,
    );
    unavailable(
        &table,
        all_zero,
        &valid_labels,
        UnavailableReason::ConstantOperand,
        8,
    );
}

#[test]
fn binary_evidence_rejects_a_non_membership_candidate_before_core_execution() {
    let input = frame(
        PrecisionProfile::Mixed,
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "candidate eligibility fixture",
    );
    let mut session = session(&input);
    let source = session.begin_round(&[]).unwrap().source(0).unwrap();
    let occupancy =
        EvidenceChannel::new("occupancy".into(), EvidenceDefinition::BinaryOccupancy).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let error = match session.evaluate(&mut core, input, &[source], &[occupancy]) {
        Ok(_) => panic!("non-membership candidate must reject before Core execution"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        SemanticError::Unsupported(
            "binary evidence requires canonical hard-predicate or decision-region candidates"
        )
    );
    assert_eq!(core.materialized_nodes(), 0);
}

#[test]
fn region_coverage_is_a_canonical_nonbinary_count_with_exact_integer_accumulation() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(
            profile,
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "coverage fixture",
        );
        let mut session = session(&input);
        let (first, second, coverage) = {
            let mut round = session.begin_round(&[]).unwrap();
            let source = round.source(0).unwrap();
            let first_term = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => round
                    .hard_predicate(source, PredicateComparator::GreaterThan, 0.0)
                    .unwrap(),
                PrecisionProfile::Fp64 => round
                    .hard_predicate_f64(source, PredicateComparator::GreaterThan, 0.0)
                    .unwrap(),
            };
            let second_term = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => round
                    .hard_predicate(source, PredicateComparator::GreaterThan, 2.0)
                    .unwrap(),
                PrecisionProfile::Fp64 => round
                    .hard_predicate_f64(source, PredicateComparator::GreaterThan, 2.0)
                    .unwrap(),
            };
            let first = round.decision_region(vec![first_term]).unwrap();
            let second = round.decision_region(vec![second_term]).unwrap();
            let coverage = round.region_count(vec![second, first]).unwrap();
            assert_eq!(round.region_count(vec![first, second]).unwrap(), coverage);
            assert!(round.region_count(vec![first]).is_err());
            assert!(round.region_count(vec![first, first]).is_err());
            assert!(round.region_count(vec![source, first]).is_err());
            (first, second, coverage)
        };
        assert!(session
            .registry()
            .unwrap()
            .program(coverage)
            .unwrap()
            .region_count_regions()
            .is_some());
        let mut core = CoreEvidenceExecutor::default();
        let output = core
            .materialize(
                session.registry().unwrap(),
                &input,
                &[coverage],
                None,
                BUDGET,
            )
            .unwrap();
        let expected_f32 = [0.0f32, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                assert_eq!(
                    output.get_typed(coverage).unwrap().as_f32().unwrap(),
                    expected_f32
                )
            }
            PrecisionProfile::Fp64 => assert_eq!(
                output.get_typed(coverage).unwrap().as_f64().unwrap(),
                &[0.0f64, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
            ),
        }
        let occupancy =
            EvidenceChannel::new("occupancy".into(), EvidenceDefinition::BinaryOccupancy).unwrap();
        let error = match session.evaluate(&mut core, input, &[coverage], &[occupancy]) {
            Ok(_) => panic!("coverage count must never be reinterpreted as binary evidence"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            SemanticError::Unsupported(
                "binary evidence requires canonical hard-predicate or decision-region candidates"
            )
        );
        assert_ne!(first, second);
    }
}
