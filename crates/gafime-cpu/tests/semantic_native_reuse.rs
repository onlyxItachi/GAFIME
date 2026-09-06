//! Native accounting checks for the internal semantic Core lowering.

use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, Direction, EvaluationRole,
    EvidenceChannel, EvidenceDefinition, FeatureFrame, MissingEvidence, NativeEvidenceExecutor,
    NumericColumn, ProgramLimits, SelectionPolicy, SemanticSession,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

fn frame() -> Arc<FeatureFrame> {
    Arc::new(
        FeatureFrame::new(
            vec!["a".into()],
            "rows".into(),
            vec![0, 1, 2, 3],
            EvaluationRole::Discovery,
            "native-reuse fixture".into(),
            vec![vec![0.0, 1.0, 2.0, 3.0]],
        )
        .unwrap(),
    )
}

#[test]
fn source_materialization_shares_payload_without_a_derived_allocation() {
    let frame = frame();
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let source = registry.source(0).unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let materialized = core
        .materialize(&registry, &frame, &[source], None, 1024)
        .unwrap();

    match (
        frame.column_typed(0).unwrap(),
        materialized.get_typed(source).unwrap(),
    ) {
        (NumericColumn::F32(input), NumericColumn::F32(shared)) => {
            assert!(Arc::ptr_eq(input, shared));
        }
        _ => panic!("mixed source materialization changed the f32 storage type"),
    }
    assert_eq!(core.materialized_nodes(), 1);
    assert_eq!(core.source_shares(), 1);
    assert_eq!(core.output_allocations(), 0);
    assert_eq!(core.output_bytes(), 0);
}

#[test]
fn duplicate_bound_channel_runs_one_numerical_primitive_and_acceptance_hits_cache() {
    let frame = frame();
    let registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .unwrap();
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 1 << 20).unwrap();
    let candidate = {
        let mut round = session.begin_round(&[]).unwrap();
        let source = round.source(0).unwrap();
        round.softsign(source).unwrap()
    };
    let first = EvidenceChannel::new(
        "first self Pearson".into(),
        EvidenceDefinition::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContext::Reference {
                reference: candidate,
            },
        },
    )
    .unwrap();
    let duplicate = EvidenceChannel::new(
        "duplicate self Pearson".into(),
        EvidenceDefinition::Association {
            statistic: AssociationStatistic::Pearson,
            context: AssociationContext::Reference {
                reference: candidate,
            },
        },
    )
    .unwrap();
    let mut core = CoreEvidenceExecutor::default();
    let calls_before = core.evidence_kernel_calls();
    let table = session
        .evaluate(
            &mut core,
            Arc::clone(&frame),
            &[candidate],
            &[first.clone(), duplicate.clone()],
        )
        .unwrap();
    assert_eq!(core.evidence_kernel_calls() - calls_before, 1);
    assert_eq!(
        table.value(candidate, first.id()).unwrap(),
        table.value(candidate, duplicate.id()).unwrap()
    );

    let accepted = session
        .accept(
            &table,
            &SelectionPolicy {
                primary: first.id(),
                direction: Direction::Maximize,
                constraints: Vec::new(),
                missing: MissingEvidence::RejectCandidate,
                limit: 1,
            },
        )
        .unwrap();
    assert_eq!(accepted.len(), 1);
    let nodes_before = core.materialized_nodes();
    let hits_before = core.retained_hits();
    let again = session
        .materialize_accepted(&mut core, &frame, &accepted)
        .unwrap();
    assert!(again.get_typed(candidate).is_ok());
    assert_eq!(core.materialized_nodes(), nodes_before);
    assert_eq!(core.retained_hits(), hits_before + 1);
}
