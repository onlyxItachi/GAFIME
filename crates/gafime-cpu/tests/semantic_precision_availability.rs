//! Tiny finite inputs can be nonconstant while their declared reduction lane
//! underflows both centered variances to zero.

use std::sync::Arc;

use gafime_cpu::{
    kernels::precision::{pearson_f32, pearson_f64},
    semantic::CoreEvidenceExecutor,
};
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, EvaluationRole, EvidenceChannel,
    EvidenceDefinition, EvidenceValue, FeatureFrame, GraphEdge, NeighborGraph, NumericColumn,
    ProgramLimits, SemanticSession, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

fn evaluate_redundancy(profile: PrecisionProfile, columns: Vec<NumericColumn>) -> EvidenceValue {
    let frame = Arc::new(
        FeatureFrame::with_profile(
            profile,
            vec!["candidate".into(), "anchor".into()],
            "tiny-finite-rows".into(),
            vec![0, 1, 2, 3],
            EvaluationRole::Discovery,
            "underflow availability fixture".into(),
            columns,
        )
        .unwrap(),
    );
    let registry =
        CandidateRegistry::new(frame.schema().to_vec(), profile, ProgramLimits::default()).unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (candidate, anchor) = {
        let round = session.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let channel = EvidenceChannel::new(
        "tiny finite redundancy".into(),
        EvidenceDefinition::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContext::Reference { reference: anchor },
        },
    )
    .unwrap();
    let table = session
        .evaluate(
            &mut CoreEvidenceExecutor::default(),
            frame,
            &[candidate],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    table.value(candidate, channel.id()).unwrap()
}

fn evaluate_graph(profile: PrecisionProfile, column: NumericColumn) -> EvidenceValue {
    let frame = Arc::new(
        FeatureFrame::with_profile(
            profile,
            vec!["candidate".into()],
            "graph-overflow-rows".into(),
            vec![0, 1, 2],
            EvaluationRole::Discovery,
            "graph overflow availability fixture".into(),
            vec![column],
        )
        .unwrap(),
    );
    let graph = Arc::new(
        NeighborGraph::new(
            &frame,
            vec![GraphEdge {
                left: 0,
                right: 1,
                weight: 1.0,
            }],
            "same-valued edge".into(),
        )
        .unwrap(),
    );
    let registry =
        CandidateRegistry::new(frame.schema().to_vec(), profile, ProgramLimits::default()).unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let candidate = session.begin_round(&[]).unwrap().source(0).unwrap();
    let channel = EvidenceChannel::new(
        "overflow graph energy".into(),
        EvidenceDefinition::GraphEnergy { graph },
    )
    .unwrap();
    let table = session
        .evaluate(
            &mut CoreEvidenceExecutor::default(),
            frame,
            &[candidate],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    table.value(candidate, channel.id()).unwrap()
}

#[test]
fn finite_nonconstant_underflow_is_not_semantic_evidence() {
    let f32_values = vec![0.0f32, 1.0e-30, 2.0e-30, 3.0e-30];
    assert_eq!(pearson_f32(&f32_values, &f32_values), 0.0);
    let f32_evidence = evaluate_redundancy(
        PrecisionProfile::Fp32,
        vec![
            NumericColumn::from(f32_values.clone()),
            NumericColumn::from(f32_values),
        ],
    );

    let f64_values = vec![0.0f64, 1.0e-200, 2.0e-200, 3.0e-200];
    assert_eq!(pearson_f64(&f64_values, &f64_values), 0.0);
    let f64_evidence = evaluate_redundancy(
        PrecisionProfile::Fp64,
        vec![
            NumericColumn::from(f64_values.clone()),
            NumericColumn::from(f64_values),
        ],
    );

    assert_eq!(
        f32_evidence,
        EvidenceValue::Unavailable {
            reason: UnavailableReason::DegenerateReduction,
            support: 4,
        },
        "fp32 finite underflow must not become a measured zero"
    );
    assert_eq!(
        f64_evidence,
        EvidenceValue::Unavailable {
            reason: UnavailableReason::DegenerateReduction,
            support: 4,
        },
        "fp64 finite underflow must not become a measured zero"
    );
}

#[test]
fn nonfinite_pearson_reduction_is_distinct_from_degenerate() {
    let f32_left = vec![1.0e20f32, -1.0e20, 1.0e20, -1.0e20];
    let f32_right = vec![1.0f32, -1.0, -1.0, 1.0];
    assert!(pearson_f32(&f32_left, &f32_right).is_nan());
    let f32_evidence = evaluate_redundancy(
        PrecisionProfile::Fp32,
        vec![
            NumericColumn::from(f32_left),
            NumericColumn::from(f32_right),
        ],
    );

    let f64_left = vec![1.0e200f64, -1.0e200, 1.0e200, -1.0e200];
    let f64_right = vec![1.0f64, -1.0, -1.0, 1.0];
    assert!(pearson_f64(&f64_left, &f64_right).is_nan());
    let f64_evidence = evaluate_redundancy(
        PrecisionProfile::Fp64,
        vec![
            NumericColumn::from(f64_left),
            NumericColumn::from(f64_right),
        ],
    );

    for evidence in [f32_evidence, f64_evidence] {
        assert_eq!(
            evidence,
            EvidenceValue::Unavailable {
                reason: UnavailableReason::NonFiniteReduction,
                support: 4,
            }
        );
    }
}

#[test]
fn graph_nonfinite_intermediate_is_not_laundered_to_zero() {
    let f32_evidence = evaluate_graph(
        PrecisionProfile::Fp32,
        NumericColumn::from(vec![1.0e20f32, 1.0e20, 2.0e20]),
    );
    let mixed_evidence = evaluate_graph(
        PrecisionProfile::Mixed,
        NumericColumn::from(vec![1.0e20f32, 1.0e20, 2.0e20]),
    );
    let f64_evidence = evaluate_graph(
        PrecisionProfile::Fp64,
        NumericColumn::from(vec![1.0e200f64, 1.0e200, 2.0e200]),
    );

    assert_eq!(
        f32_evidence,
        EvidenceValue::Unavailable {
            reason: UnavailableReason::NonFiniteReduction,
            support: 1,
        },
        "fp32 infinity in the graph denominator must not become zero"
    );
    assert_eq!(
        mixed_evidence,
        EvidenceValue::Measured {
            value: 0.0,
            support: 1,
        },
        "mixed widens its f32 input and weight reductions to f64"
    );
    assert_eq!(
        f64_evidence,
        EvidenceValue::Unavailable {
            reason: UnavailableReason::NonFiniteReduction,
            support: 1,
        },
        "fp64 infinity in the graph denominator must not become zero"
    );
}
