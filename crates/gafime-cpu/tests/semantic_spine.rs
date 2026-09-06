//! End-to-end assertions for the bounded internal semantic spine.
//!
//! These cases deliberately use small hand-calculated fixtures.  They exercise
//! the production Core session/executor boundary without creating a second
//! candidate or evidence implementation in the test target.

use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, Direction, EvaluationRole,
    EvidenceChannel, EvidenceConstraint, EvidenceDefinition, EvidenceId, EvidenceTable,
    EvidenceValue, FeatureFrame, FeatureId, GraphEdge, LabelSet, MissingEvidence, NeighborGraph,
    ProgramLimits, SelectionPolicy, SemanticError, SemanticSession, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA};

const EXECUTION_BUDGET: usize = 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
enum EvidenceBits {
    Measured {
        bits: u64,
        support: usize,
    },
    Unavailable {
        reason: UnavailableReason,
        support: usize,
    },
}

fn frame(
    schema: &[&str],
    row_keys: Vec<u64>,
    columns: Vec<Vec<f32>>,
    provenance: &str,
) -> Arc<FeatureFrame> {
    frame_with_role(
        schema,
        row_keys,
        columns,
        EvaluationRole::Discovery,
        provenance,
    )
}

fn frame_with_role(
    schema: &[&str],
    row_keys: Vec<u64>,
    columns: Vec<Vec<f32>>,
    role: EvaluationRole,
    provenance: &str,
) -> Arc<FeatureFrame> {
    Arc::new(
        FeatureFrame::new(
            schema.iter().map(|name| (*name).to_owned()).collect(),
            "fixture-rows".to_owned(),
            row_keys,
            role,
            provenance.to_owned(),
            columns,
        )
        .unwrap(),
    )
}

fn session(schema: &[&str]) -> SemanticSession {
    session_with_budget(schema, EXECUTION_BUDGET)
}

fn session_with_budget(schema: &[&str], budget: usize) -> SemanticSession {
    let registry = CandidateRegistry::new(
        schema.iter().map(|name| (*name).to_owned()).collect(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    SemanticSession::new(registry, GAFIME_BACKEND_CPU, budget).unwrap()
}

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

fn measured(value: EvidenceValue, expected: f64, support: usize) {
    match value {
        EvidenceValue::Measured {
            value: actual,
            support: actual_support,
        } => {
            assert_eq!(actual_support, support);
            assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            );
        }
        other => panic!("expected measured evidence, got {other:?}"),
    }
}

fn unavailable(value: EvidenceValue, reason: UnavailableReason, support: usize) {
    assert_eq!(
        value,
        EvidenceValue::Unavailable { reason, support },
        "evidence availability is part of the semantic result"
    );
}

fn record_bits(table: &EvidenceTable) -> Vec<(FeatureId, EvidenceId, EvidenceBits)> {
    table
        .records()
        .iter()
        .map(|record| {
            let value = evidence_bits(record.value());
            (record.candidate(), record.channel(), value)
        })
        .collect()
}

fn evidence_bits(value: EvidenceValue) -> EvidenceBits {
    match value {
        EvidenceValue::Measured { value, support } => EvidenceBits::Measured {
            bits: value.to_bits(),
            support,
        },
        EvidenceValue::Unavailable { reason, support } => {
            EvidenceBits::Unavailable { reason, support }
        }
    }
}

#[test]
fn core_session_evaluates_all_channels_accepts_and_reuses_programs_on_new_rows() {
    let schema = ["a", "b", "reference"];
    // abs(a - b) is [3, 1, 1, 3].  Its Pearson association with the
    // increasing reference is 0, while labels below are the actual same
    // values and therefore have association +1.
    let discovery = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "discovery-v1",
    );
    // The same frozen program yields [1, 3, 3, 1], an affine reversal of the
    // discovery candidate, so paired signed Pearson is -1.
    let paired = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 4.0, 5.0, 4.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "paired-view-v1",
    );

    let mut semantic = session(&schema);
    let (difference, reference) = {
        let mut registry = semantic.begin_round(&[]).unwrap();
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let reference = registry.source(2).unwrap();
        (registry.abs_difference(a, b).unwrap(), reference)
    };

    let graph = Arc::new(
        NeighborGraph::new(
            discovery.as_ref(),
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
            "unit-chain".to_owned(),
        )
        .unwrap(),
    );
    let labels = Arc::new(
        LabelSet::new(
            discovery.as_ref(),
            vec![(0, 3.0), (1, 1.0), (2, 1.0), (3, 3.0)],
            "actual-discovery-labels".to_owned(),
        )
        .unwrap(),
    );
    let redundancy =
        EvidenceChannel::new("redundancy".to_owned(), pearson_reference(reference)).unwrap();
    let paired_consistency =
        EvidenceChannel::new("paired".to_owned(), pearson_paired(Arc::clone(&paired))).unwrap();
    let graph_energy = EvidenceChannel::new(
        "graph".to_owned(),
        EvidenceDefinition::GraphEnergy {
            graph: Arc::clone(&graph),
        },
    )
    .unwrap();
    let labeled = EvidenceChannel::new(
        "labels".to_owned(),
        pearson_labels(Some(Arc::clone(&labels))),
    )
    .unwrap();
    let missing_labels =
        EvidenceChannel::new("missing-labels".to_owned(), pearson_labels(None)).unwrap();
    let channels = vec![
        redundancy.clone(),
        paired_consistency.clone(),
        graph_energy.clone(),
        labeled.clone(),
        missing_labels.clone(),
    ];

    let mut executor = CoreEvidenceExecutor::default();
    let table = semantic
        .evaluate(
            &mut executor,
            Arc::clone(&discovery),
            &[difference],
            &channels,
        )
        .unwrap();
    measured(table.value(difference, redundancy.id()).unwrap(), 0.0, 4);
    measured(
        table.value(difference, paired_consistency.id()).unwrap(),
        -1.0,
        4,
    );
    // Chain numerator is 4 + 0 + 4 = 8; denominator is 10 + 2 + 10 = 22.
    measured(
        table.value(difference, graph_energy.id()).unwrap(),
        4.0 / 11.0,
        3,
    );
    measured(table.value(difference, labeled.id()).unwrap(), 1.0, 4);
    unavailable(
        table.value(difference, missing_labels.id()).unwrap(),
        UnavailableReason::MissingLabels,
        0,
    );

    let policy = SelectionPolicy {
        primary: labeled.id(),
        direction: Direction::Maximize,
        constraints: vec![EvidenceConstraint {
            channel: redundancy.id(),
            minimum: None,
            maximum: Some(0.01),
            missing: None,
        }],
        missing: MissingEvidence::RejectCandidate,
        limit: 1,
    };
    let accepted = semantic.accept(&table, &policy).unwrap();
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].feature(), difference);
    assert_eq!(accepted[0].frame().id(), discovery.id());
    assert_eq!(accepted[0].evidence().len(), channels.len());
    assert!(accepted[0].evidence().iter().any(|record| {
        record.channel() == missing_labels.id()
            && record.value()
                == (EvidenceValue::Unavailable {
                    reason: UnavailableReason::MissingLabels,
                    support: 0,
                })
    }));

    // An accepted program stays registry-owned and can be the frozen input of
    // a new program.  Its source arity remains transitive through abs-diff.
    let child = semantic
        .begin_round(&accepted)
        .unwrap()
        .softsign(difference)
        .unwrap();
    let child_program = semantic.registry().unwrap().program(child).unwrap();
    assert_eq!(child_program.logical_arity(), 1);
    assert_eq!(child_program.source_dependencies(), &[0, 1]);
    assert_eq!(child_program.source_arity(), 2);
    assert_eq!(child_program.depth(), 2);

    let child_channel =
        EvidenceChannel::new("child-redundancy".to_owned(), pearson_reference(reference)).unwrap();
    let nodes_before_child = executor.materialized_nodes();
    let reused_before_child = executor.reused_nodes();
    let child_table = semantic
        .evaluate(
            &mut executor,
            Arc::clone(&discovery),
            &[child],
            std::slice::from_ref(&child_channel),
        )
        .unwrap();
    measured(
        child_table.value(child, child_channel.id()).unwrap(),
        0.0,
        4,
    );
    // The accepted abs-diff is reused; only the reference source and softsign
    // child are newly materialized for this evaluation.
    assert_eq!(executor.materialized_nodes(), nodes_before_child + 2);
    assert_eq!(executor.reused_nodes(), reused_before_child + 1);

    let nodes_before_same_context = executor.materialized_nodes();
    let reused_before_same_context = executor.reused_nodes();
    let same_context = semantic
        .materialize_accepted(&mut executor, discovery.as_ref(), &accepted)
        .unwrap();
    assert_eq!(same_context.get(difference).unwrap(), &[3.0, 1.0, 1.0, 3.0]);
    assert_eq!(executor.materialized_nodes(), nodes_before_same_context);
    assert_eq!(executor.reused_nodes(), reused_before_same_context + 1);

    let later_rows = frame_with_role(
        &schema,
        vec![100, 101, 102, 103],
        vec![
            vec![10.0, 11.0, 12.0, 13.0],
            vec![14.0, 13.0, 12.0, 11.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        EvaluationRole::Inference,
        "later-inference-v1",
    );
    assert_eq!(later_rows.role(), EvaluationRole::Inference);
    let nodes_before_new_context = executor.materialized_nodes();
    let reused_before_new_context = executor.reused_nodes();
    let rematerialized = semantic
        .materialize_accepted(&mut executor, later_rows.as_ref(), &accepted)
        .unwrap();
    assert_eq!(
        rematerialized.get(difference).unwrap(),
        &[4.0, 2.0, 0.0, 2.0]
    );
    // A new immutable frame invalidates the retained value: a, b, and abs-diff
    // execute again, while the accepted program identity remains reusable.
    assert_eq!(executor.materialized_nodes(), nodes_before_new_context + 3);
    assert_eq!(executor.reused_nodes(), reused_before_new_context);
}

#[test]
fn selection_thresholds_missingness_and_ties_are_explicit_and_deterministic() {
    let schema = ["a", "b", "reference"];
    let input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "selection-fixture",
    );
    let mut semantic = session(&schema);
    let (a, b, reference) = {
        let registry = semantic.begin_round(&[]).unwrap();
        (
            registry.source(0).unwrap(),
            registry.source(1).unwrap(),
            registry.source(2).unwrap(),
        )
    };
    let strength =
        EvidenceChannel::new("strength".to_owned(), pearson_reference(reference)).unwrap();
    let absent = EvidenceChannel::new("optional-labels".to_owned(), pearson_labels(None)).unwrap();
    let mut executor = CoreEvidenceExecutor::default();
    // Input order must not choose a winner: evidence rows are canonicalized by
    // FeatureId before the deterministic tie break is applied.
    let table = semantic
        .evaluate(
            &mut executor,
            input,
            &[b, a],
            &[strength.clone(), absent.clone()],
        )
        .unwrap();
    measured(table.value(a, strength.id()).unwrap(), 1.0, 4);
    measured(table.value(b, strength.id()).unwrap(), 1.0, 4);

    let ties = SelectionPolicy {
        primary: strength.id(),
        direction: Direction::Maximize,
        constraints: vec![],
        missing: MissingEvidence::RejectCandidate,
        limit: 2,
    };
    let accepted = semantic.accept(&table, &ties).unwrap();
    assert_eq!(
        accepted
            .iter()
            .map(|feature| feature.feature())
            .collect::<Vec<_>>(),
        vec![a, b]
    );

    let threshold_rejects = SelectionPolicy {
        primary: strength.id(),
        direction: Direction::Maximize,
        constraints: vec![EvidenceConstraint {
            channel: strength.id(),
            minimum: None,
            maximum: Some(0.99),
            missing: None,
        }],
        missing: MissingEvidence::RejectCandidate,
        limit: 2,
    };
    assert!(semantic
        .accept(&table, &threshold_rejects)
        .unwrap()
        .is_empty());

    let missing_rejects = SelectionPolicy {
        primary: strength.id(),
        direction: Direction::Maximize,
        constraints: vec![EvidenceConstraint {
            channel: absent.id(),
            minimum: None,
            maximum: None,
            missing: None,
        }],
        missing: MissingEvidence::RejectCandidate,
        limit: 2,
    };
    assert!(semantic
        .accept(&table, &missing_rejects)
        .unwrap()
        .is_empty());

    let missing_is_an_error = SelectionPolicy {
        missing: MissingEvidence::Error,
        ..missing_rejects
    };
    assert!(matches!(
        semantic.accept(&table, &missing_is_an_error),
        Err(SemanticError::Invalid(_))
    ));
}

#[test]
fn changed_labels_only_change_labeled_evidence_for_the_same_program_and_frame() {
    let schema = ["a", "b", "reference"];
    let input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "same-primary-frame",
    );
    let mut semantic = session(&schema);
    let (difference, reference) = {
        let mut registry = semantic.begin_round(&[]).unwrap();
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let reference = registry.source(2).unwrap();
        (registry.abs_difference(a, b).unwrap(), reference)
    };
    let redundancy = EvidenceChannel::new(
        "target-free-redundancy".to_owned(),
        pearson_reference(reference),
    )
    .unwrap();
    let matching_labels = Arc::new(
        LabelSet::new(
            input.as_ref(),
            vec![(0, 3.0), (1, 1.0), (2, 1.0), (3, 3.0)],
            "matching-actual-labels".to_owned(),
        )
        .unwrap(),
    );
    let different_labels = Arc::new(
        LabelSet::new(
            input.as_ref(),
            vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)],
            "different-actual-labels".to_owned(),
        )
        .unwrap(),
    );
    assert_ne!(matching_labels.id(), different_labels.id());
    let first_labels = EvidenceChannel::new(
        "matching-label-evidence".to_owned(),
        pearson_labels(Some(matching_labels)),
    )
    .unwrap();
    let second_labels = EvidenceChannel::new(
        "different-label-evidence".to_owned(),
        pearson_labels(Some(different_labels)),
    )
    .unwrap();
    assert_ne!(first_labels.id(), second_labels.id());

    let mut executor = CoreEvidenceExecutor::default();
    let first = semantic
        .evaluate(
            &mut executor,
            Arc::clone(&input),
            &[difference],
            &[redundancy.clone(), first_labels.clone()],
        )
        .unwrap();
    measured(first.value(difference, redundancy.id()).unwrap(), 0.0, 4);
    measured(first.value(difference, first_labels.id()).unwrap(), 1.0, 4);
    let policy = SelectionPolicy {
        primary: redundancy.id(),
        direction: Direction::Minimize,
        constraints: vec![],
        missing: MissingEvidence::RejectCandidate,
        limit: 1,
    };
    let accepted = semantic.accept(&first, &policy).unwrap();
    let initial_materialization = semantic
        .materialize_accepted(&mut executor, input.as_ref(), &accepted)
        .unwrap();
    assert_eq!(
        initial_materialization.get(difference).unwrap(),
        &[3.0, 1.0, 1.0, 3.0]
    );

    let second = semantic
        .evaluate(
            &mut executor,
            Arc::clone(&input),
            &[difference],
            &[redundancy.clone(), second_labels.clone()],
        )
        .unwrap();
    assert_eq!(first.candidates(), second.candidates());
    assert_eq!(accepted[0].feature(), difference);
    assert_eq!(
        evidence_bits(first.value(difference, redundancy.id()).unwrap()),
        evidence_bits(second.value(difference, redundancy.id()).unwrap())
    );
    measured(
        second.value(difference, second_labels.id()).unwrap(),
        0.0,
        4,
    );

    let nodes_before_pure_reuse = executor.materialized_nodes();
    let after_label_change = semantic
        .materialize_accepted(&mut executor, input.as_ref(), &accepted)
        .unwrap();
    assert_eq!(
        after_label_change.get(difference).unwrap(),
        initial_materialization.get(difference).unwrap()
    );
    assert_eq!(executor.materialized_nodes(), nodes_before_pure_reuse);
}

#[test]
fn frozen_centered_product_uses_declared_means_on_new_inference_rows() {
    let schema = ["a", "b"];
    let discovery = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![vec![2.0, 3.0, 4.0, 5.0], vec![12.0, 14.0, 16.0, 18.0]],
        "centered-product-discovery",
    );
    let inference = frame_with_role(
        &schema,
        vec![100, 101, 102, 103],
        vec![
            vec![100.0, 101.0, 102.0, 103.0],
            vec![12.0, 14.0, 16.0, 18.0],
        ],
        EvaluationRole::Inference,
        "centered-product-inference",
    );
    let mut semantic = session(&schema);
    let (product, reference) = {
        let mut registry = semantic.begin_round(&[]).unwrap();
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        (
            registry
                .centered_product(vec![a, b], vec![1.0, 10.0])
                .unwrap(),
            a,
        )
    };
    let channel =
        EvidenceChannel::new("product-reference".to_owned(), pearson_reference(reference)).unwrap();
    let mut executor = CoreEvidenceExecutor::default();
    let table = semantic
        .evaluate(
            &mut executor,
            discovery,
            &[product],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    let policy = SelectionPolicy {
        primary: channel.id(),
        direction: Direction::Maximize,
        constraints: vec![],
        missing: MissingEvidence::RejectCandidate,
        limit: 1,
    };
    let accepted = semantic.accept(&table, &policy).unwrap();
    let materialized = semantic
        .materialize_accepted(&mut executor, inference.as_ref(), &accepted)
        .unwrap();
    // Frozen means are 1 and 10: (100-1)*(12-10), ... .  Re-centering on
    // inference rows would produce a different vector and is not permitted.
    assert_eq!(
        materialized.get(product).unwrap(),
        &[198.0, 400.0, 606.0, 816.0]
    );
    assert_eq!(inference.role(), EvaluationRole::Inference);
}

#[test]
fn core_budget_rejects_before_materialization_accounting_changes() {
    let schema = ["a", "b"];
    let input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![vec![0.0, 1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0, 0.0]],
        "budget-fixture",
    );
    let mut semantic = session_with_budget(&schema, 1);
    let (candidate, reference) = {
        let registry = semantic.begin_round(&[]).unwrap();
        (registry.source(0).unwrap(), registry.source(1).unwrap())
    };
    let channel =
        EvidenceChannel::new("budget-reference".to_owned(), pearson_reference(reference)).unwrap();
    let mut executor = CoreEvidenceExecutor::default();
    assert!(matches!(
        semantic.evaluate(
            &mut executor,
            input,
            &[candidate],
            std::slice::from_ref(&channel)
        ),
        Err(SemanticError::Invalid(_))
    ));
    assert_eq!(executor.materialized_nodes(), 0);
    assert_eq!(executor.reused_nodes(), 0);
}

#[test]
fn core_evidence_records_are_bit_identical_with_one_or_four_rayon_workers() {
    let schema = ["a", "b", "reference"];
    let input = frame(
        &schema,
        vec![0, 1, 2, 3, 4, 5],
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        "rayon-determinism",
    );
    let paired = frame(
        &schema,
        vec![0, 1, 2, 3, 4, 5],
        vec![
            vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        "rayon-paired",
    );
    let mut semantic = session(&schema);
    let (difference, softsign, reference) = {
        let mut registry = semantic.begin_round(&[]).unwrap();
        let a = registry.source(0).unwrap();
        let b = registry.source(1).unwrap();
        let reference = registry.source(2).unwrap();
        (
            registry.abs_difference(a, b).unwrap(),
            registry.softsign(a).unwrap(),
            reference,
        )
    };
    let graph = Arc::new(
        NeighborGraph::new(
            input.as_ref(),
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
                GraphEdge {
                    left: 3,
                    right: 4,
                    weight: 1.0,
                },
                GraphEdge {
                    left: 4,
                    right: 5,
                    weight: 1.0,
                },
            ],
            "rayon-chain".to_owned(),
        )
        .unwrap(),
    );
    let labels = Arc::new(
        LabelSet::new(
            input.as_ref(),
            vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)],
            "rayon-labels".to_owned(),
        )
        .unwrap(),
    );
    let channels = vec![
        EvidenceChannel::new("redundancy".to_owned(), pearson_reference(reference)).unwrap(),
        EvidenceChannel::new("paired".to_owned(), pearson_paired(Arc::clone(&paired))).unwrap(),
        EvidenceChannel::new(
            "graph".to_owned(),
            EvidenceDefinition::GraphEnergy { graph },
        )
        .unwrap(),
        EvidenceChannel::new("labels".to_owned(), pearson_labels(Some(labels))).unwrap(),
    ];
    let mut one_worker = CoreEvidenceExecutor::default();
    let one_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let one = one_pool.install(|| {
        semantic
            .evaluate(
                &mut one_worker,
                Arc::clone(&input),
                &[difference, softsign],
                &channels,
            )
            .unwrap()
    });
    let mut four_workers = CoreEvidenceExecutor::default();
    let four_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let four = four_pool.install(|| {
        semantic
            .evaluate(
                &mut four_workers,
                Arc::clone(&input),
                &[difference, softsign],
                &channels,
            )
            .unwrap()
    });

    // This is a numerical determinism check only; it makes no timing claim.
    assert_eq!(one.candidates(), four.candidates());
    assert_eq!(record_bits(&one), record_bits(&four));
}

#[test]
fn backend_context_identity_and_closed_sessions_fail_closed() {
    let schema = ["a", "b", "reference"];
    let registry = CandidateRegistry::new(
        schema.iter().map(|name| (*name).to_owned()).collect(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        SemanticSession::new(registry, GAFIME_BACKEND_CUDA, EXECUTION_BUDGET),
        Err(SemanticError::Unsupported(_))
    ));

    let input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "context-fixture",
    );
    let mut semantic = session(&schema);
    let candidate = semantic.registry().unwrap().source(0).unwrap();
    let mut executor = CoreEvidenceExecutor::default();

    let misaligned_rows = frame(
        &schema,
        vec![10, 11, 12, 13],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "different-row-identity",
    );
    let misaligned_channel =
        EvidenceChannel::new("misaligned".to_owned(), pearson_paired(misaligned_rows)).unwrap();
    assert!(matches!(
        semantic.evaluate(
            &mut executor,
            Arc::clone(&input),
            &[candidate],
            &[misaligned_channel]
        ),
        Err(SemanticError::Invalid(_))
    ));

    let schema_mismatch = frame(
        &["a", "other", "reference"],
        vec![0, 1, 2, 3],
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ],
        "different-schema",
    );
    let schema_channel = EvidenceChannel::new(
        "schema-mismatch".to_owned(),
        pearson_paired(schema_mismatch),
    )
    .unwrap();
    assert!(matches!(
        semantic.evaluate(
            &mut executor,
            Arc::clone(&input),
            &[candidate],
            &[schema_channel]
        ),
        Err(SemanticError::Invalid(_))
    ));

    let mut owner = session(&schema);
    let (owner_candidate, owner_reference) = {
        let registry = owner.begin_round(&[]).unwrap();
        (registry.source(0).unwrap(), registry.source(2).unwrap())
    };
    let channel = EvidenceChannel::new(
        "owner-strength".to_owned(),
        pearson_reference(owner_reference),
    )
    .unwrap();
    let table = owner
        .evaluate(
            &mut executor,
            Arc::clone(&input),
            &[owner_candidate],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    let policy = SelectionPolicy {
        primary: channel.id(),
        direction: Direction::Maximize,
        constraints: vec![],
        missing: MissingEvidence::RejectCandidate,
        limit: 1,
    };
    let accepted = owner.accept(&table, &policy).unwrap();

    let mut foreign_owner = session(&schema);
    assert!(matches!(
        foreign_owner.materialize_accepted(&mut executor, input.as_ref(), &accepted),
        Err(SemanticError::ForeignIdentity)
    ));
    owner.close();
    assert!(matches!(
        owner.materialize_accepted(&mut executor, input.as_ref(), &accepted),
        Err(SemanticError::Closed)
    ));
}

#[test]
fn core_evidence_keeps_unavailability_and_uncentered_graph_translation_visible() {
    let schema = ["a", "b"];
    let constant_input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![vec![2.0, 2.0, 2.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]],
        "constant-fixture",
    );
    let mut semantic = session(&schema);
    let (constant, reference) = {
        let registry = semantic.begin_round(&[]).unwrap();
        (registry.source(0).unwrap(), registry.source(1).unwrap())
    };
    let one_label = Arc::new(
        LabelSet::new(
            constant_input.as_ref(),
            vec![(0, 5.0)],
            "single-actual-label".to_owned(),
        )
        .unwrap(),
    );
    let constant_channel =
        EvidenceChannel::new("constant".to_owned(), pearson_reference(reference)).unwrap();
    let insufficient_channel =
        EvidenceChannel::new("one-label".to_owned(), pearson_labels(Some(one_label))).unwrap();
    let mut executor = CoreEvidenceExecutor::default();
    let table = semantic
        .evaluate(
            &mut executor,
            constant_input,
            &[constant],
            &[constant_channel.clone(), insufficient_channel.clone()],
        )
        .unwrap();
    unavailable(
        table.value(constant, constant_channel.id()).unwrap(),
        UnavailableReason::ConstantOperand,
        4,
    );
    unavailable(
        table.value(constant, insufficient_channel.id()).unwrap(),
        UnavailableReason::InsufficientSupport,
        1,
    );

    // All frame inputs are finite, but the frozen centered product overflows
    // binary32.  Production materialization must reject it rather than turn it
    // into a plausible numerical evidence value.
    let overflow_input = frame(
        &schema,
        vec![0, 1],
        vec![vec![f32::MAX, f32::MAX], vec![f32::MAX, f32::MAX]],
        "finite-input-overflow",
    );
    let mut overflow_session = session(&schema);
    let (left, right) = {
        let registry = overflow_session.begin_round(&[]).unwrap();
        (registry.source(0).unwrap(), registry.source(1).unwrap())
    };
    let product = overflow_session
        .begin_round(&[])
        .unwrap()
        .centered_product(vec![left, right], vec![0.0, 0.0])
        .unwrap();
    let overflow_channel =
        EvidenceChannel::new("overflow".to_owned(), pearson_reference(left)).unwrap();
    assert!(matches!(
        overflow_session.evaluate(
            &mut executor,
            overflow_input,
            &[product],
            &[overflow_channel]
        ),
        Err(SemanticError::Invalid(_))
    ));

    let graph_input = frame(
        &schema,
        vec![0, 1, 2, 3],
        vec![vec![0.0, 1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0, 0.0]],
        "graph-origin",
    );
    let translated_input = frame(
        &schema,
        vec![10, 11, 12, 13],
        vec![vec![10.0, 11.0, 12.0, 13.0], vec![3.0, 2.0, 1.0, 0.0]],
        "graph-translated",
    );
    let mut graph_session = session(&schema);
    let graph_candidate = graph_session.registry().unwrap().source(0).unwrap();
    let origin_graph = Arc::new(
        NeighborGraph::new(
            graph_input.as_ref(),
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
            "origin-chain".to_owned(),
        )
        .unwrap(),
    );
    let translated_graph = Arc::new(
        NeighborGraph::new(
            translated_input.as_ref(),
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
            "translated-chain".to_owned(),
        )
        .unwrap(),
    );
    let origin_channel = EvidenceChannel::new(
        "origin-energy".to_owned(),
        EvidenceDefinition::GraphEnergy {
            graph: origin_graph,
        },
    )
    .unwrap();
    let translated_channel = EvidenceChannel::new(
        "translated-energy".to_owned(),
        EvidenceDefinition::GraphEnergy {
            graph: translated_graph,
        },
    )
    .unwrap();
    // For [0, 1, 2, 3], chain energy is 3 / 19.  Translating every value by
    // ten keeps the numerator 3 but changes the declared uncentered
    // denominator to 799, giving 3 / 799 rather than a translation-invariant
    // Laplacian-style score.
    graph_session.begin_round(&[]).unwrap();
    let origin = graph_session
        .evaluate(
            &mut executor,
            graph_input,
            &[graph_candidate],
            std::slice::from_ref(&origin_channel),
        )
        .unwrap();
    let translated = graph_session
        .evaluate(
            &mut executor,
            translated_input,
            &[graph_candidate],
            std::slice::from_ref(&translated_channel),
        )
        .unwrap();
    measured(
        origin.value(graph_candidate, origin_channel.id()).unwrap(),
        3.0 / 19.0,
        3,
    );
    measured(
        translated
            .value(graph_candidate, translated_channel.id())
            .unwrap(),
        3.0 / 799.0,
        3,
    );
}
