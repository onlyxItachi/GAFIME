use std::sync::Arc;

use gafime_cpu::{
    kernels::precision::{pearson_f32, pearson_f64, pearson_mixed},
    semantic::CoreEvidenceExecutor,
};
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, Direction, EvaluationRole,
    EvidenceChannel, EvidenceConstraint, EvidenceDefinition, EvidenceTable, EvidenceValue,
    FeatureFrame, FeatureId, GraphEdge, LabelSet, MissingEvidence, NeighborGraph, NumericColumn,
    ProgramLimits, SelectionPolicy, SemanticSession, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

const BUDGET: usize = 1 << 20;

fn pearson_reference(reference: FeatureId) -> EvidenceDefinition {
    EvidenceDefinition::Association {
        statistic: AssociationStatistic::Pearson,
        context: AssociationContext::Reference { reference },
    }
}

fn pearson_paired(view: Arc<FeatureFrame>) -> EvidenceDefinition {
    EvidenceDefinition::Association {
        statistic: AssociationStatistic::Pearson,
        context: AssociationContext::PairedView { view },
    }
}

fn pearson_labels(labels: Option<Arc<LabelSet>>) -> EvidenceDefinition {
    EvidenceDefinition::Association {
        statistic: AssociationStatistic::Pearson,
        context: AssociationContext::Labels { labels },
    }
}

fn association(statistic: AssociationStatistic, context: AssociationContext) -> EvidenceDefinition {
    EvidenceDefinition::Association { statistic, context }
}

fn frame(
    profile: PrecisionProfile,
    schema: &[&str],
    values: Vec<Vec<f64>>,
    role: EvaluationRole,
    provenance: &str,
) -> Arc<FeatureFrame> {
    let rows = values.first().expect("test frame has a column").len();
    let columns = match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => values
            .into_iter()
            .map(|column| {
                NumericColumn::from(
                    column
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
        PrecisionProfile::Fp64 => values.into_iter().map(NumericColumn::from).collect(),
    };
    Arc::new(
        FeatureFrame::with_profile(
            profile,
            schema.iter().map(|name| (*name).to_owned()).collect(),
            "policy-row-domain".into(),
            (0..u64::try_from(rows).unwrap()).collect(),
            role,
            provenance.into(),
            columns,
        )
        .unwrap(),
    )
}

fn session(frame: &FeatureFrame) -> SemanticSession {
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        frame.profile(),
        ProgramLimits::default(),
    )
    .unwrap();
    SemanticSession::new(registry, GAFIME_BACKEND_CPU, BUDGET).unwrap()
}

fn measured(table: &EvidenceTable, feature: FeatureId, channel: &EvidenceChannel) -> f64 {
    match table.value(feature, channel.id()).unwrap() {
        EvidenceValue::Measured { value, .. } => value,
        value => panic!("expected measured evidence, got {value:?}"),
    }
}

fn selection(
    primary: &EvidenceChannel,
    direction: Direction,
    constraints: Vec<EvidenceConstraint>,
    missing: MissingEvidence,
) -> SelectionPolicy {
    SelectionPolicy {
        primary: primary.id(),
        direction,
        constraints,
        missing,
        limit: 8,
    }
}

fn graph_ratio_f32(values: &[f32], edges: &[(usize, usize, f32)]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for &(left, right, weight) in edges {
        let left = values[left];
        let right = values[right];
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    numerator / denominator
}

fn graph_ratio_mixed(values: &[f32], edges: &[(usize, usize, f32)]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for &(left, right, weight) in edges {
        let left = f64::from(values[left]);
        let right = f64::from(values[right]);
        let weight = f64::from(weight);
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    numerator / denominator
}

fn graph_ratio_f64(values: &[f64], edges: &[(usize, usize, f64)]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for &(left, right, weight) in edges {
        let left = values[left];
        let right = values[right];
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    numerator / denominator
}

#[test]
fn rebinding_a_spec_keeps_its_policy_identity_and_old_table() {
    let discovery = frame(
        PrecisionProfile::Mixed,
        &["signal"],
        vec![vec![0.0, 2.0, 3.0, 5.0]],
        EvaluationRole::Discovery,
        "first declared label context",
    );
    let later = frame(
        PrecisionProfile::Mixed,
        &["signal"],
        vec![vec![5.0, 7.0, 10.0, 14.0]],
        EvaluationRole::Discovery,
        "later declared label context",
    );
    let mut semantic = session(&discovery);
    let source = semantic.begin_round(&[]).unwrap().source(0).unwrap();
    let unbound =
        EvidenceChannel::new("optional ground truth".into(), pearson_labels(None)).unwrap();
    let policy = selection(
        &unbound,
        Direction::Maximize,
        vec![],
        MissingEvidence::RejectCandidate,
    );
    let mut core = CoreEvidenceExecutor::default();
    let old_table = semantic
        .evaluate(
            &mut core,
            Arc::clone(&discovery),
            &[source],
            std::slice::from_ref(&unbound),
        )
        .unwrap();
    assert!(matches!(
        old_table.value(source, unbound.id()).unwrap(),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::MissingLabels,
            support: 0,
        }
    ));
    assert!(semantic.accept(&old_table, &policy).unwrap().is_empty());

    let labels = Arc::new(
        LabelSet::new(
            &later,
            vec![(0, 5.0), (1, 7.0), (2, 10.0), (3, 14.0)],
            "actual later labels".into(),
        )
        .unwrap(),
    );
    let rebound = unbound.rebind(pearson_labels(Some(labels))).unwrap();
    assert_eq!(rebound.id(), unbound.id());
    assert_eq!(rebound.spec().id(), policy.primary);
    assert_eq!(
        rebound.spec().semantic_name(),
        unbound.spec().semantic_name()
    );

    let new_table = semantic
        .evaluate(
            &mut core,
            Arc::clone(&later),
            &[source],
            std::slice::from_ref(&rebound),
        )
        .unwrap();
    assert_eq!(measured(&new_table, source, &rebound), 1.0);
    assert_eq!(
        semantic
            .accept(&new_table, &policy)
            .unwrap()
            .iter()
            .map(|accepted| accepted.feature())
            .collect::<Vec<_>>(),
        vec![source]
    );
    assert_ne!(old_table.id(), new_table.id());
    assert!(matches!(
        old_table.value(source, unbound.id()).unwrap(),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::MissingLabels,
            ..
        }
    ));
    assert!(unbound.rebind(pearson_reference(source)).is_err());
}

#[test]
fn selection_separates_channel_bounds_from_optional_missingness() {
    let input = frame(
        PrecisionProfile::Mixed,
        &["aligned", "zigzag", "anchor"],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, -1.0, -1.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        EvaluationRole::Discovery,
        "selection policy fixture",
    );
    let mut semantic = session(&input);
    let (aligned, zigzag, anchor) = {
        let round = semantic.begin_round(&[]).unwrap();
        (
            round.source(0).unwrap(),
            round.source(1).unwrap(),
            round.source(2).unwrap(),
        )
    };
    let labels = Arc::new(
        LabelSet::new(
            &input,
            vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)],
            "complete policy labels".into(),
        )
        .unwrap(),
    );
    let redundancy =
        EvidenceChannel::new("anchor redundancy".into(), pearson_reference(anchor)).unwrap();
    let association =
        EvidenceChannel::new("label association".into(), pearson_labels(Some(labels))).unwrap();
    let optional =
        EvidenceChannel::new("not collected labels".into(), pearson_labels(None)).unwrap();
    let channels = vec![redundancy.clone(), association.clone(), optional.clone()];
    let mut core = CoreEvidenceExecutor::default();
    let table = semantic
        .evaluate(&mut core, Arc::clone(&input), &[aligned, zigzag], &channels)
        .unwrap();

    let high_association = selection(
        &redundancy,
        Direction::Minimize,
        vec![EvidenceConstraint {
            channel: association.id(),
            minimum: Some(0.99),
            maximum: Some(1.0),
            missing: None,
        }],
        MissingEvidence::Error,
    );
    assert_eq!(
        semantic
            .accept(&table, &high_association)
            .unwrap()
            .iter()
            .map(|accepted| accepted.feature())
            .collect::<Vec<_>>(),
        vec![aligned],
        "a maximum/minimum constraint can override a minimizing primary channel"
    );

    let low_association = selection(
        &redundancy,
        Direction::Minimize,
        vec![EvidenceConstraint {
            channel: association.id(),
            minimum: Some(0.0),
            maximum: Some(0.01),
            missing: None,
        }],
        MissingEvidence::Error,
    );
    assert_eq!(
        semantic
            .accept(&table, &low_association)
            .unwrap()
            .iter()
            .map(|accepted| accepted.feature())
            .collect::<Vec<_>>(),
        vec![zigzag]
    );

    let ignored_optional = selection(
        &redundancy,
        Direction::Minimize,
        vec![EvidenceConstraint {
            channel: optional.id(),
            minimum: Some(0.0),
            maximum: Some(1.0),
            missing: Some(MissingEvidence::IgnoreConstraint),
        }],
        MissingEvidence::Error,
    );
    assert_eq!(
        semantic
            .accept(&table, &ignored_optional)
            .unwrap()
            .iter()
            .map(|accepted| accepted.feature())
            .collect::<Vec<_>>(),
        vec![zigzag, aligned]
    );

    let inherited_error = selection(
        &redundancy,
        Direction::Minimize,
        vec![EvidenceConstraint {
            channel: optional.id(),
            minimum: Some(0.0),
            maximum: Some(1.0),
            missing: None,
        }],
        MissingEvidence::Error,
    );
    assert!(semantic.accept(&table, &inherited_error).is_err());
    assert!(semantic
        .accept(
            &table,
            &selection(
                &redundancy,
                Direction::Minimize,
                vec![],
                MissingEvidence::IgnoreConstraint,
            ),
        )
        .is_err());
}

#[test]
fn native_pearson_and_graph_follow_each_profile_lane_and_selection_precision() {
    let raw_signal = vec![0.25, 1.5, -2.0, 3.25, 0.75, 5.0];
    let raw_anchor = vec![2.0, -0.5, 1.0, 4.0, -3.0, 0.5];
    let raw_edges = vec![
        (0, 1, 0.1),
        (1, 2, 0.35),
        (2, 3, 0.75),
        (3, 4, 1.1),
        (4, 5, 1.6),
    ];
    let stored = EvidenceValue::measured_f32(f32::from_bits(0x3f12_3456), 4);
    assert_eq!(
        match stored {
            EvidenceValue::Measured { value, .. } => value.to_bits(),
            EvidenceValue::Unavailable { .. } => panic!("finite f32 was not stored"),
        },
        f64::from(f32::from_bits(0x3f12_3456)).to_bits(),
        "the shared f64 container must not invent fp32 precision"
    );

    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(
            profile,
            &["signal", "anchor"],
            vec![raw_signal.clone(), raw_anchor.clone()],
            EvaluationRole::Discovery,
            "profile arithmetic fixture",
        );
        let graph = Arc::new(
            NeighborGraph::new(
                &input,
                raw_edges
                    .iter()
                    .map(|&(left, right, weight)| GraphEdge {
                        left,
                        right,
                        weight,
                    })
                    .collect(),
                "ordered weighted graph".into(),
            )
            .unwrap(),
        );
        let mut semantic = session(&input);
        let (candidate, anchor) = {
            let round = semantic.begin_round(&[]).unwrap();
            (round.source(0).unwrap(), round.source(1).unwrap())
        };
        let pearson =
            EvidenceChannel::new("direct native pearson".into(), pearson_reference(anchor))
                .unwrap();
        let energy = EvidenceChannel::new(
            "ordered graph energy".into(),
            EvidenceDefinition::GraphEnergy { graph },
        )
        .unwrap();
        let channels = vec![pearson.clone(), energy.clone()];
        let mut core = CoreEvidenceExecutor::default();
        let table = semantic
            .evaluate(&mut core, Arc::clone(&input), &[candidate], &channels)
            .unwrap();
        let expected_pearson = match profile {
            PrecisionProfile::Fp32 => {
                let signal = raw_signal
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let anchor = raw_anchor
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                f64::from(pearson_f32(&signal, &anchor).abs())
            }
            PrecisionProfile::Mixed => {
                let signal = raw_signal
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let anchor = raw_anchor
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                pearson_mixed(&signal, &anchor).abs()
            }
            PrecisionProfile::Fp64 => pearson_f64(&raw_signal, &raw_anchor).abs(),
        };
        let expected_energy = match profile {
            PrecisionProfile::Fp32 => {
                let signal = raw_signal
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let edges = raw_edges
                    .iter()
                    .map(|&(left, right, weight)| (left, right, weight as f32))
                    .collect::<Vec<_>>();
                f64::from(graph_ratio_f32(&signal, &edges))
            }
            PrecisionProfile::Mixed => {
                let signal = raw_signal
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let edges = raw_edges
                    .iter()
                    .map(|&(left, right, weight)| (left, right, weight as f32))
                    .collect::<Vec<_>>();
                graph_ratio_mixed(&signal, &edges)
            }
            PrecisionProfile::Fp64 => graph_ratio_f64(&raw_signal, &raw_edges),
        };
        assert_eq!(
            measured(&table, candidate, &pearson).to_bits(),
            expected_pearson.to_bits(),
            "{profile:?} must use the existing direct Pearson primitive"
        );
        assert_eq!(
            measured(&table, candidate, &energy).to_bits(),
            expected_energy.to_bits(),
            "{profile:?} graph arithmetic must retain its declared reduction lane"
        );

        let value = measured(&table, candidate, &pearson);
        match profile {
            PrecisionProfile::Fp32 => {
                let stored = value as f32;
                let next = f32::from_bits(stored.to_bits() + 1);
                let unquantized_threshold = value + (f64::from(next) - value) / 4.0;
                assert_eq!(
                    f64::from(unquantized_threshold as f32).to_bits(),
                    value.to_bits()
                );
                let accepts_after_fp32_quantization = selection(
                    &pearson,
                    Direction::Maximize,
                    vec![EvidenceConstraint {
                        channel: pearson.id(),
                        minimum: Some(unquantized_threshold),
                        maximum: None,
                        missing: None,
                    }],
                    MissingEvidence::Error,
                );
                assert_eq!(
                    semantic
                        .accept(&table, &accepts_after_fp32_quantization)
                        .unwrap()
                        .len(),
                    1
                );
            }
            PrecisionProfile::Fp64 => {
                let next_f64 = f64::from_bits(value.to_bits() + 1);
                let rejects_at_the_next_f64 = selection(
                    &pearson,
                    Direction::Maximize,
                    vec![EvidenceConstraint {
                        channel: pearson.id(),
                        minimum: Some(next_f64),
                        maximum: None,
                        missing: None,
                    }],
                    MissingEvidence::Error,
                );
                assert!(semantic
                    .accept(&table, &rejects_at_the_next_f64)
                    .unwrap()
                    .is_empty());
            }
            PrecisionProfile::Mixed => {}
        }
    }
}

#[test]
fn canonical_rank_and_fixed_nmi_associations_preserve_context_and_key_identity() {
    let ascending = (0..32).map(f64::from).collect::<Vec<_>>();
    let descending = ascending.iter().rev().copied().collect::<Vec<_>>();

    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(
            profile,
            &["candidate", "reference"],
            vec![ascending.clone(), descending.clone()],
            EvaluationRole::Discovery,
            "rank and fixed-NMI discovery rows",
        );
        let paired = frame(
            profile,
            &["candidate", "reference"],
            vec![descending.clone(), ascending.clone()],
            EvaluationRole::Discovery,
            "rank and fixed-NMI aligned view",
        );
        let labels = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => Arc::new(
                LabelSet::new(
                    &input,
                    descending
                        .iter()
                        .enumerate()
                        .map(|(row, value)| (row, *value as f32))
                        .collect(),
                    "rank labels".into(),
                )
                .unwrap(),
            ),
            PrecisionProfile::Fp64 => Arc::new(
                LabelSet::new_f64(
                    &input,
                    descending.iter().copied().enumerate().collect(),
                    "rank labels".into(),
                )
                .unwrap(),
            ),
        };
        let mut semantic = session(&input);
        let (candidate, reference) = {
            let round = semantic.begin_round(&[]).unwrap();
            (round.source(0).unwrap(), round.source(1).unwrap())
        };
        let rank_reference = EvidenceChannel::new(
            "absolute rank reference".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::Reference { reference },
            ),
        )
        .unwrap();
        let rank_paired = EvidenceChannel::new(
            "signed rank paired view".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::PairedView {
                    view: Arc::clone(&paired),
                },
            ),
        )
        .unwrap();
        let rank_labels = EvidenceChannel::new(
            "absolute rank labels".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::Labels {
                    labels: Some(labels),
                },
            ),
        )
        .unwrap();
        let fixed_nmi = EvidenceChannel::new(
            "fixed two-bin NMI".into(),
            association(
                AssociationStatistic::FixedCorrectedNmi { bins: 2 },
                AssociationContext::Reference { reference },
            ),
        )
        .unwrap();
        let table = semantic
            .evaluate(
                &mut CoreEvidenceExecutor::default(),
                Arc::clone(&input),
                &[candidate],
                &[
                    rank_reference.clone(),
                    rank_paired.clone(),
                    rank_labels.clone(),
                    fixed_nmi.clone(),
                ],
            )
            .unwrap();
        assert!(
            measured(&table, candidate, &rank_reference) > 0.999_999,
            "{profile:?} reference correlations are absolute"
        );
        assert!(
            measured(&table, candidate, &rank_paired) < -0.999_999,
            "{profile:?} paired correlations retain their sign"
        );
        assert!(
            measured(&table, candidate, &rank_labels) > 0.999_999,
            "{profile:?} labeled correlations are absolute"
        );
        assert!(
            measured(&table, candidate, &fixed_nmi) > 0.9,
            "{profile:?} fixed corrected NMI measures the declared two-bin dependence"
        );
        assert!(fixed_nmi
            .rebind(association(
                AssociationStatistic::FixedCorrectedNmi { bins: 4 },
                AssociationContext::Reference { reference },
            ))
            .is_err());
    }

    assert!(EvidenceChannel::new(
        "unsupported NMI bins".into(),
        association(
            AssociationStatistic::FixedCorrectedNmi { bins: 3 },
            AssociationContext::Labels { labels: None },
        ),
    )
    .is_err());
}

#[test]
fn constant_insufficient_and_nonfinite_evidence_remain_explicitly_unavailable() {
    let input = frame(
        PrecisionProfile::Mixed,
        &["constant", "varying"],
        vec![vec![2.0, 2.0, 2.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]],
        EvaluationRole::Discovery,
        "unavailable evidence fixture",
    );
    let graph = Arc::new(
        NeighborGraph::new(
            &input,
            vec![
                GraphEdge {
                    left: 0,
                    right: 1,
                    weight: 1.0,
                },
                GraphEdge {
                    left: 1,
                    right: 2,
                    weight: 1.0,
                },
                GraphEdge {
                    left: 2,
                    right: 3,
                    weight: 1.0,
                },
            ],
            "constant graph fixture".into(),
        )
        .unwrap(),
    );
    let labels =
        Arc::new(LabelSet::new(&input, vec![(0, 1.0)], "one supplied label".into()).unwrap());
    let mut semantic = session(&input);
    let (constant, varying) = {
        let round = semantic.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let redundancy =
        EvidenceChannel::new("constant redundancy".into(), pearson_reference(varying)).unwrap();
    let energy = EvidenceChannel::new(
        "constant graph energy".into(),
        EvidenceDefinition::GraphEnergy { graph },
    )
    .unwrap();
    let association =
        EvidenceChannel::new("one label association".into(), pearson_labels(Some(labels))).unwrap();
    let channels = vec![redundancy.clone(), energy.clone(), association.clone()];
    let table = semantic
        .evaluate(
            &mut CoreEvidenceExecutor::default(),
            input,
            &[constant],
            &channels,
        )
        .unwrap();
    assert!(matches!(
        table.value(constant, redundancy.id()).unwrap(),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::ConstantOperand,
            support: 4,
        }
    ));
    assert!(matches!(
        table.value(constant, energy.id()).unwrap(),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::ConstantOperand,
            support: 3,
        }
    ));
    assert!(matches!(
        table.value(constant, association.id()).unwrap(),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::InsufficientSupport,
            support: 1,
        }
    ));
    assert_eq!(
        EvidenceValue::measured(f64::NAN, 7),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::NonFiniteReduction,
            support: 7,
        }
    );
    assert_eq!(
        EvidenceValue::measured(f64::INFINITY, 9),
        EvidenceValue::Unavailable {
            reason: UnavailableReason::NonFiniteReduction,
            support: 9,
        }
    );
}

#[test]
fn rounds_must_be_declared_and_context_mismatch_fails_before_native_work() {
    let input = frame(
        PrecisionProfile::Mixed,
        &["signal", "anchor"],
        vec![vec![0.0, 1.0, 3.0, 6.0], vec![2.0, 0.0, 4.0, 1.0]],
        EvaluationRole::Discovery,
        "declared discovery frame",
    );
    let mut semantic = session(&input);
    let source = semantic.registry().unwrap().source(0).unwrap();
    let predeclared =
        EvidenceChannel::new("pre-round reference".into(), pearson_reference(source)).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    assert!(semantic
        .evaluate(
            &mut core,
            Arc::clone(&input),
            &[source],
            std::slice::from_ref(&predeclared),
        )
        .is_err());

    let declared = {
        let mut round = semantic.begin_round(&[]).unwrap();
        round.softsign(source).unwrap()
    };
    let self_reference =
        EvidenceChannel::new("declared reference".into(), pearson_reference(declared)).unwrap();
    let table = semantic
        .evaluate(
            &mut core,
            Arc::clone(&input),
            &[declared],
            std::slice::from_ref(&self_reference),
        )
        .unwrap();
    let accepted = semantic
        .accept(
            &table,
            &selection(
                &self_reference,
                Direction::Maximize,
                vec![],
                MissingEvidence::Error,
            ),
        )
        .unwrap();
    assert_eq!(accepted.len(), 1);

    let next = {
        let mut round = semantic.begin_round(&accepted).unwrap();
        round.abs_difference(accepted[0].feature(), source).unwrap()
    };
    let stale = frame(
        PrecisionProfile::Mixed,
        &["signal", "anchor"],
        vec![vec![0.0, 1.0, 3.0, 6.0], vec![2.0, 0.0, 4.0, 1.0]],
        EvaluationRole::Discovery,
        "different immutable context",
    );
    let stale_labels =
        Arc::new(LabelSet::new(&stale, vec![(0, 0.0), (1, 1.0)], "stale labels".into()).unwrap());
    let stale_channel = EvidenceChannel::new(
        "stale label binding".into(),
        pearson_labels(Some(stale_labels)),
    )
    .unwrap();
    assert!(semantic
        .evaluate(
            &mut core,
            Arc::clone(&input),
            &[next],
            std::slice::from_ref(&stale_channel),
        )
        .is_err());

    let wrong_schema = frame(
        PrecisionProfile::Mixed,
        &["different", "anchor"],
        vec![vec![0.0, 1.0, 3.0, 6.0], vec![2.0, 0.0, 4.0, 1.0]],
        EvaluationRole::Discovery,
        "different schema",
    );
    assert!(semantic
        .evaluate(
            &mut core,
            wrong_schema,
            &[next],
            std::slice::from_ref(&self_reference),
        )
        .is_err());

    let wrong_profile = frame(
        PrecisionProfile::Fp64,
        &["signal", "anchor"],
        vec![vec![0.0, 1.0, 3.0, 6.0], vec![2.0, 0.0, 4.0, 1.0]],
        EvaluationRole::Discovery,
        "different numeric profile",
    );
    assert!(semantic
        .evaluate(
            &mut core,
            wrong_profile,
            &[next],
            std::slice::from_ref(&self_reference),
        )
        .is_err());

    let holdout = frame(
        PrecisionProfile::Mixed,
        &["signal", "anchor"],
        vec![vec![0.0, 1.0, 3.0, 6.0], vec![2.0, 0.0, 4.0, 1.0]],
        EvaluationRole::Holdout,
        "role-mismatched paired frame",
    );
    let paired = EvidenceChannel::new(
        "role mismatched paired evidence".into(),
        pearson_paired(holdout),
    )
    .unwrap();
    assert!(semantic
        .evaluate(&mut core, input, &[next], std::slice::from_ref(&paired),)
        .is_err());
}
