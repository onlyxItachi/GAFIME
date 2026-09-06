use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::*;
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

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

#[test]
fn independent_acceptance_rounds_retain_both_atoms_for_composition() {
    let frame = Arc::new(
        FeatureFrame::new(
            vec!["a".into(), "b".into()],
            "rows".into(),
            vec![0, 1, 2],
            EvaluationRole::Discovery,
            "retention counterexample".into(),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 1.0, 0.0]],
        )
        .unwrap(),
    );
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (first, second) = {
        let mut round = session.begin_round(&[]).unwrap();
        let a = round.source(0).unwrap();
        let b = round.source(1).unwrap();
        (
            round.abs_difference(a, b).unwrap(),
            round.softsign(a).unwrap(),
        )
    };
    let mut core = CoreEvidenceExecutor::default();
    let mut atoms = Vec::new();
    for id in [first, second] {
        let channel =
            EvidenceChannel::new("self association".into(), pearson_reference(id)).unwrap();
        let table = session
            .evaluate(
                &mut core,
                Arc::clone(&frame),
                &[id],
                std::slice::from_ref(&channel),
            )
            .unwrap();
        let accepted = session
            .accept(
                &table,
                &SelectionPolicy {
                    primary: channel.id(),
                    direction: Direction::Maximize,
                    constraints: vec![],
                    missing: MissingEvidence::Error,
                    limit: 1,
                },
            )
            .unwrap();
        assert_eq!(accepted.len(), 1);
        atoms.extend(accepted);
    }
    let combined = session
        .begin_round(&atoms)
        .unwrap()
        .abs_difference(first, second)
        .unwrap();
    let before = core.materialized_nodes();
    let reused = core.reused_nodes();
    let channel =
        EvidenceChannel::new("combined association".into(), pearson_reference(combined)).unwrap();
    session
        .evaluate(&mut core, frame, &[combined], &[channel])
        .unwrap();
    assert_eq!(
        core.materialized_nodes() - before,
        1,
        "only the new composite should execute"
    );
    assert_eq!(
        core.reused_nodes() - reused,
        2,
        "both accepted atoms should remain resident"
    );
}

fn typed_frame(profile: PrecisionProfile, role: EvaluationRole, offset: f64) -> Arc<FeatureFrame> {
    let columns = vec![
        vec![
            0.25 + offset,
            1.5 + offset,
            -2.0 + offset,
            3.25 + offset,
            0.75 + offset,
            5.0 + offset,
        ],
        vec![2.0, -0.5, 1.0, 4.0, -3.0, 0.5],
        vec![1.0, 0.0, 3.0, -2.0, 2.0, 4.0],
    ]
    .into_iter()
    .map(|values| {
        if profile == PrecisionProfile::Fp64 {
            NumericColumn::from(values)
        } else {
            NumericColumn::from(values.into_iter().map(|v| v as f32).collect::<Vec<_>>())
        }
    })
    .collect();
    Arc::new(
        FeatureFrame::with_profile(
            profile,
            vec!["a".into(), "b".into(), "anchor".into()],
            "aligned-population".into(),
            (0..6).collect(),
            role,
            format!("declared view {offset}"),
            columns,
        )
        .unwrap(),
    )
}

fn channels(
    frame: &Arc<FeatureFrame>,
    view: &Arc<FeatureFrame>,
    anchor: FeatureId,
) -> Vec<EvidenceChannel> {
    let graph = Arc::new(
        NeighborGraph::new(
            frame,
            (0..5)
                .map(|i| GraphEdge {
                    left: i,
                    right: i + 1,
                    weight: 1.0 + (i as f64) / 4.0,
                })
                .collect(),
            "declared weighted chain".into(),
        )
        .unwrap(),
    );
    vec![
        EvidenceChannel::new("redundancy".into(), pearson_reference(anchor)).unwrap(),
        EvidenceChannel::new("consistency".into(), pearson_paired(Arc::clone(view))).unwrap(),
        EvidenceChannel::new("graph".into(), EvidenceDefinition::GraphEnergy { graph }).unwrap(),
        EvidenceChannel::new("optional labels".into(), pearson_labels(None)).unwrap(),
    ]
}

fn policy(channels: &[EvidenceChannel]) -> SelectionPolicy {
    SelectionPolicy {
        primary: channels[2].id(),
        direction: Direction::Minimize,
        constraints: vec![
            EvidenceConstraint {
                channel: channels[1].id(),
                minimum: Some(-1.0),
                maximum: Some(1.0),
                missing: None,
            },
            EvidenceConstraint {
                channel: channels[3].id(),
                minimum: Some(0.0),
                maximum: Some(1.0),
                missing: Some(MissingEvidence::IgnoreConstraint),
            },
        ],
        missing: MissingEvidence::Error,
        limit: 2,
    }
}

fn bits(column: &NumericColumn) -> Vec<u64> {
    match column {
        NumericColumn::F32(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
        NumericColumn::F64(values) => values.iter().map(|v| v.to_bits()).collect(),
    }
}

fn multiround(profile: PrecisionProfile, workers: usize) -> Vec<u64> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap()
        .install(|| {
            let frame = typed_frame(profile, EvaluationRole::Discovery, 0.0);
            let view = typed_frame(profile, EvaluationRole::Discovery, 0.125);
            let registry =
                CandidateRegistry::new(frame.schema().to_vec(), profile, ProgramLimits::default())
                    .unwrap();
            let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
            let (anchor, candidates) = {
                let mut round = session.begin_round(&[]).unwrap();
                let a = round.source(0).unwrap();
                let b = round.source(1).unwrap();
                let anchor = round.source(2).unwrap();
                let product = if profile == PrecisionProfile::Fp64 {
                    round
                        .centered_product_f64(vec![a, b], vec![0.125, -0.75])
                        .unwrap()
                } else {
                    round
                        .centered_product(vec![a, b], vec![0.125, -0.75])
                        .unwrap()
                };
                (
                    anchor,
                    vec![
                        round.abs_difference(a, b).unwrap(),
                        round.softsign(a).unwrap(),
                        product,
                    ],
                )
            };
            let channels = channels(&frame, &view, anchor);
            let policy = policy(&channels);
            let mut core = CoreEvidenceExecutor::default();
            let table = session
                .evaluate(&mut core, Arc::clone(&frame), &candidates, &channels)
                .unwrap();
            let mut fingerprint = Vec::new();
            for record in table.records() {
                match record.value() {
                    EvidenceValue::Measured { value, support } => {
                        fingerprint.extend([value.to_bits(), support as u64]);
                    }
                    EvidenceValue::Unavailable { support, .. } => {
                        fingerprint.extend([u64::MAX, support as u64])
                    }
                }
            }
            let accepted = session.accept(&table, &policy).unwrap();
            assert_eq!(accepted.len(), 2);
            fingerprint.extend(
                accepted
                    .iter()
                    .map(|a| candidates.iter().position(|id| *id == a.feature()).unwrap() as u64),
            );
            let first_outputs = session
                .materialize_accepted(&mut core, &frame, &accepted)
                .unwrap();
            let child = {
                let mut round = session.begin_round(&accepted).unwrap();
                round
                    .abs_difference(accepted[0].feature(), accepted[1].feature())
                    .unwrap()
            };
            assert_eq!(
                session
                    .registry()
                    .unwrap()
                    .program(child)
                    .unwrap()
                    .logical_arity(),
                2
            );
            assert!(
                session
                    .registry()
                    .unwrap()
                    .program(child)
                    .unwrap()
                    .source_arity()
                    <= 2
            );
            let second = session
                .evaluate(&mut core, Arc::clone(&frame), &[child], &channels)
                .unwrap();
            let second_accepted = session.accept(&second, &policy).unwrap();
            assert_eq!(second_accepted.len(), 1);
            let materialized = session
                .materialize_accepted(&mut core, &frame, &second_accepted)
                .unwrap();
            let expected = match (
                first_outputs.get_typed(accepted[0].feature()).unwrap(),
                first_outputs.get_typed(accepted[1].feature()).unwrap(),
            ) {
                (NumericColumn::F32(a), NumericColumn::F32(b)) => NumericColumn::from(
                    a.iter()
                        .zip(b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>(),
                ),
                (NumericColumn::F64(a), NumericColumn::F64(b)) => NumericColumn::from(
                    a.iter()
                        .zip(b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>(),
                ),
                _ => panic!("profile changed"),
            };
            assert_eq!(
                bits(materialized.get_typed(child).unwrap()),
                bits(&expected)
            );
            fingerprint.extend(bits(&expected));
            let inference = typed_frame(profile, EvaluationRole::Inference, 8.0);
            let output = session
                .materialize_accepted(&mut core, &inference, &second_accepted)
                .unwrap();
            let before_repeat = core.materialized_nodes();
            let repeat = session
                .materialize_accepted(&mut core, &inference, &second_accepted)
                .unwrap();
            assert_eq!(
                core.materialized_nodes(),
                before_repeat,
                "same-frame inference must retain accepted outputs"
            );
            assert_eq!(
                bits(output.get_typed(child).unwrap()),
                bits(repeat.get_typed(child).unwrap())
            );
            // Independent composition oracle: execute the already accepted parents,
            // then compare their pointwise difference to the frozen composed program.
            let parents = session
                .materialize_accepted(&mut core, &inference, &accepted)
                .unwrap();
            let expected = match (
                parents.get_typed(accepted[0].feature()).unwrap(),
                parents.get_typed(accepted[1].feature()).unwrap(),
            ) {
                (NumericColumn::F32(a), NumericColumn::F32(b)) => NumericColumn::from(
                    a.iter()
                        .zip(b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>(),
                ),
                (NumericColumn::F64(a), NumericColumn::F64(b)) => NumericColumn::from(
                    a.iter()
                        .zip(b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>(),
                ),
                _ => panic!("profile changed"),
            };
            assert_eq!(bits(output.get_typed(child).unwrap()), bits(&expected));
            fingerprint.extend(bits(&expected));
            assert!(
                session.accept(&table, &policy).is_err(),
                "stale-round evidence cannot admit new atoms"
            );
            fingerprint
        })
}

#[test]
fn heterogeneous_two_round_inference_is_bit_identical_for_every_core_profile() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        assert_eq!(
            multiround(profile, 1),
            multiround(profile, 4),
            "{profile:?}"
        );
    }
}

#[test]
fn stable_policy_rebinds_partial_labels_and_context_without_changing_program() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let frame = typed_frame(profile, EvaluationRole::Discovery, 0.0);
        let registry =
            CandidateRegistry::new(frame.schema().to_vec(), profile, ProgramLimits::default())
                .unwrap();
        let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
        let candidate = session.begin_round(&[]).unwrap().source(0).unwrap();
        let absent = EvidenceChannel::new("labeled".into(), pearson_labels(None)).unwrap();
        let policy = SelectionPolicy {
            primary: absent.id(),
            direction: Direction::Maximize,
            constraints: vec![],
            missing: MissingEvidence::RejectCandidate,
            limit: 1,
        };
        let mut core = CoreEvidenceExecutor::default();
        let old = session
            .evaluate(
                &mut core,
                Arc::clone(&frame),
                &[candidate],
                std::slice::from_ref(&absent),
            )
            .unwrap();
        assert!(session.accept(&old, &policy).unwrap().is_empty());
        for context in [
            Arc::clone(&frame),
            typed_frame(profile, EvaluationRole::Discovery, 2.0),
        ] {
            let labels = if profile == PrecisionProfile::Fp64 {
                LabelSet::new_f64(
                    &context,
                    vec![(0, 1.0), (2, 3.0), (4, -1.0)],
                    "actual partial observations".into(),
                )
                .unwrap()
            } else {
                LabelSet::new(
                    &context,
                    vec![(0, 1.0), (2, 3.0), (4, -1.0)],
                    "actual partial observations".into(),
                )
                .unwrap()
            };
            let rebound = absent
                .rebind(pearson_labels(Some(Arc::new(labels))))
                .unwrap();
            assert_eq!(absent.id(), rebound.id());
            let table = session
                .evaluate(
                    &mut core,
                    context,
                    &[candidate],
                    std::slice::from_ref(&rebound),
                )
                .unwrap();
            let accepted = session.accept(&table, &policy).unwrap();
            assert_eq!(accepted[0].feature(), candidate);
            assert!(matches!(
                accepted[0].evidence()[0].value(),
                EvidenceValue::Measured { support: 3, .. }
            ));
            assert_ne!(old.id(), table.id());
        }
        assert!(matches!(
            old.value(candidate, absent.id()).unwrap(),
            EvidenceValue::Unavailable {
                reason: UnavailableReason::MissingLabels,
                ..
            }
        ));
        assert!(absent.rebind(pearson_reference(candidate)).is_err());
    }
}

#[test]
fn round_atom_admission_rejects_unaccepted_foreign_and_exhausted_state() {
    let frame = typed_frame(PrecisionProfile::Mixed, EvaluationRole::Discovery, 0.0);
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut limits = SessionLimits::for_budget(1 << 20);
    limits.max_rounds = 2;
    let mut session = SemanticSession::with_limits(registry, GAFIME_BACKEND_CPU, limits).unwrap();
    let (rejected, accepted_id) = {
        let mut round = session.begin_round(&[]).unwrap();
        let a = round.source(0).unwrap();
        let b = round.source(1).unwrap();
        (
            round.abs_difference(a, b).unwrap(),
            round.softsign(a).unwrap(),
        )
    };
    let ch = EvidenceChannel::new("reference".into(), pearson_reference(accepted_id)).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&frame),
            &[accepted_id],
            std::slice::from_ref(&ch),
        )
        .unwrap();
    let policy = SelectionPolicy {
        primary: ch.id(),
        direction: Direction::Maximize,
        constraints: vec![],
        missing: MissingEvidence::Error,
        limit: 1,
    };
    let accepted = session.accept(&table, &policy).unwrap();
    {
        let mut round = session.begin_round(&accepted).unwrap();
        assert!(round.softsign(rejected).is_err());
        assert!(round.softsign(accepted_id).is_ok());
    }
    assert!(session
        .evaluate(
            &mut core,
            Arc::clone(&frame),
            &[rejected],
            std::slice::from_ref(&ch)
        )
        .is_err());
    assert!(session.begin_round(&accepted).is_err());
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut foreign = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    assert!(matches!(
        foreign.begin_round(&accepted),
        Err(SemanticError::ForeignIdentity)
    ));
    assert_eq!(foreign.round(), 0, "failed admission must be atomic");
    session.close();
    assert!(matches!(
        session.begin_round(&accepted),
        Err(SemanticError::Closed)
    ));
}

#[test]
fn work_and_retention_admission_are_bounded_without_silent_eviction() {
    let frame = typed_frame(PrecisionProfile::Mixed, EvaluationRole::Discovery, 0.0);
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let ids = [registry.source(0).unwrap(), registry.source(1).unwrap()];
    let mut limits = SessionLimits::for_budget(1 << 20);
    limits.max_work = 1;
    let mut session = SemanticSession::with_limits(registry, GAFIME_BACKEND_CPU, limits).unwrap();
    session.begin_round(&[]).unwrap();
    let ch = EvidenceChannel::new("self".into(), pearson_reference(ids[0])).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    assert!(session
        .evaluate(
            &mut core,
            Arc::clone(&frame),
            &ids,
            std::slice::from_ref(&ch)
        )
        .is_err());
    assert_eq!(core.materialized_nodes(), 0);
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let ids = [registry.source(0).unwrap(), registry.source(1).unwrap()];
    let mut limits = SessionLimits::for_budget(1 << 20);
    limits.max_retained_bytes = 6 * 4;
    let mut session = SemanticSession::with_limits(registry, GAFIME_BACKEND_CPU, limits).unwrap();
    session.begin_round(&[]).unwrap();
    let ch = EvidenceChannel::new("anchor".into(), pearson_reference(ids[0])).unwrap();
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&frame),
            &ids,
            std::slice::from_ref(&ch),
        )
        .unwrap();
    let mut policy = SelectionPolicy {
        primary: ch.id(),
        direction: Direction::Maximize,
        constraints: vec![],
        missing: MissingEvidence::Error,
        limit: 1,
    };
    let accepted = session.accept(&table, &policy).unwrap();
    assert_eq!(session.retained_bytes(), 24);
    policy.limit = 2;
    assert!(session.accept(&table, &policy).is_err());
    assert_eq!(
        session.retained_bytes(),
        24,
        "failed admission preserves prior cache"
    );
    session.clear_materializations().unwrap();
    assert_eq!(session.retained_bytes(), 0);
    assert!(
        session
            .materialize_accepted(&mut core, &frame, &accepted)
            .is_ok(),
        "eviction does not revoke accepted semantics"
    );
}

#[test]
fn accepted_atoms_do_not_hide_transitive_source_limits() {
    let frame = typed_frame(PrecisionProfile::Mixed, EvaluationRole::Discovery, 0.0);
    let limits = ProgramLimits {
        max_source_arity: 2,
        max_logical_arity: 2,
        ..ProgramLimits::default()
    };
    let registry =
        CandidateRegistry::new(frame.schema().to_vec(), PrecisionProfile::Mixed, limits).unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let (a, b, c, feature) = {
        let mut round = session.begin_round(&[]).unwrap();
        let a = round.source(0).unwrap();
        let b = round.source(1).unwrap();
        let c = round.source(2).unwrap();
        (a, b, c, round.abs_difference(a, b).unwrap())
    };
    let channel = EvidenceChannel::new("reference".into(), pearson_reference(a)).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let table = session
        .evaluate(&mut core, frame, &[feature], std::slice::from_ref(&channel))
        .unwrap();
    let accepted = session
        .accept(
            &table,
            &SelectionPolicy {
                primary: channel.id(),
                direction: Direction::Maximize,
                constraints: vec![],
                missing: MissingEvidence::Error,
                limit: 1,
            },
        )
        .unwrap();
    let mut round = session.begin_round(&accepted).unwrap();
    assert!(
        round.abs_difference(feature, c).is_err(),
        "two logical atoms still touch three sources"
    );
    assert!(
        round.abs_difference(feature, b).is_ok(),
        "overlapping transitive sources remain legal"
    );
}

#[test]
fn repeated_discovery_rounds_stop_at_registry_budget_without_losing_accepted_inference() {
    let frame = typed_frame(PrecisionProfile::Mixed, EvaluationRole::Discovery, 0.0);
    let limits = ProgramLimits {
        max_nodes: 6,
        ..ProgramLimits::default()
    };
    let registry =
        CandidateRegistry::new(frame.schema().to_vec(), PrecisionProfile::Mixed, limits).unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let mut atoms = Vec::new();
    let mut parent = session.registry().unwrap().source(0).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    for _ in 0..3 {
        let child = session
            .begin_round(&atoms)
            .unwrap()
            .softsign(parent)
            .unwrap();
        let channel =
            EvidenceChannel::new("self measurement".into(), pearson_reference(child)).unwrap();
        let table = session
            .evaluate(
                &mut core,
                Arc::clone(&frame),
                &[child],
                std::slice::from_ref(&channel),
            )
            .unwrap();
        atoms = session
            .accept(
                &table,
                &SelectionPolicy {
                    primary: channel.id(),
                    direction: Direction::Maximize,
                    constraints: vec![],
                    missing: MissingEvidence::Error,
                    limit: 1,
                },
            )
            .unwrap();
        parent = child;
    }
    assert!(session
        .begin_round(&atoms)
        .unwrap()
        .softsign(parent)
        .is_err());
    assert!(
        session
            .materialize_accepted(&mut core, &frame, &atoms)
            .is_ok(),
        "resource rejection must not revoke immutable accepted programs"
    );
}
