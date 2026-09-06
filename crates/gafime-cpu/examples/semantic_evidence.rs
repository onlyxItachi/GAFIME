//! Runnable internal cookbook for target-free Core semantic evidence.
//!
//! This demonstrates candidate construction, explicit evidence, selection and
//! unlabeled new-row materialization. It is not a benchmark or a public API.

use std::sync::Arc;

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    CandidateRegistry, Direction, EvaluationRole, EvidenceChannel, EvidenceDefinition,
    FeatureFrame, GraphEdge, MissingEvidence, NeighborGraph, ProgramLimits, SelectionPolicy,
    SemanticError, SemanticSession,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

fn frame(
    row_domain: &str,
    row_keys: Vec<u64>,
    role: EvaluationRole,
    provenance: &str,
    columns: Vec<Vec<f32>>,
) -> Result<Arc<FeatureFrame>, SemanticError> {
    Ok(Arc::new(FeatureFrame::new(
        vec!["signal".into(), "anchor".into()],
        row_domain.into(),
        row_keys,
        role,
        provenance.into(),
        columns,
    )?))
}

fn chain_graph(frame: &FeatureFrame) -> Result<Arc<NeighborGraph>, SemanticError> {
    Ok(Arc::new(NeighborGraph::new(
        frame,
        (0..frame.rows() - 1)
            .map(|left| GraphEdge {
                left,
                right: left + 1,
                weight: 1.0,
            })
            .collect(),
        "declared-consecutive-neighbor-graph".into(),
    )?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = frame(
        "cookbook-row-domain/v1",
        vec![10, 11, 12, 13, 14, 15],
        EvaluationRole::Discovery,
        "declared-discovery-view",
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 1.5, 2.4, 3.5, 4.6, 5.7],
        ],
    )?;
    let alternate = frame(
        "cookbook-row-domain/v1",
        vec![10, 11, 12, 13, 14, 15],
        EvaluationRole::Discovery,
        "declared-aligned-alternate-view",
        vec![
            vec![0.1, 1.1, 1.9, 3.1, 3.9, 5.1],
            vec![1.05, 1.55, 2.35, 3.55, 4.55, 5.65],
        ],
    )?;
    let graph = chain_graph(&discovery)?;

    let registry = CandidateRegistry::new(
        vec!["signal".into(), "anchor".into()],
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )?;
    let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 2 * 1024 * 1024)?;
    let (anchor, absolute_difference, softsign) = {
        let registry = session.registry_mut()?;
        let signal = registry.source(0)?;
        let anchor = registry.source(1)?;
        (
            anchor,
            registry.abs_difference(signal, anchor)?,
            registry.softsign(signal)?,
        )
    };

    let consistency = EvidenceChannel::new(
        "paired_view_consistency".into(),
        EvidenceDefinition::PairedConsistency {
            view: Arc::clone(&alternate),
        },
    )?;
    let policy = SelectionPolicy {
        primary: consistency.id(),
        direction: Direction::Maximize,
        constraints: Vec::new(),
        missing: MissingEvidence::RejectCandidate,
        limit: 1,
    };

    let mut executor = CoreEvidenceExecutor::default();
    let table = session.evaluate(
        &mut executor,
        Arc::clone(&discovery),
        &[absolute_difference, softsign],
        &[
            EvidenceChannel::new(
                "redundancy_against_anchor".into(),
                EvidenceDefinition::Redundancy { reference: anchor },
            )?,
            consistency.clone(),
            EvidenceChannel::new(
                "declared_graph_energy".into(),
                EvidenceDefinition::GraphEnergy { graph },
            )?,
            EvidenceChannel::new(
                "optional_labels_absent".into(),
                EvidenceDefinition::LabeledAssociation { labels: None },
            )?,
        ],
    )?;
    println!(
        "semantic_evidence backend={} precision={} target=none table={} discovery_frame={} provenance={} paired_provenance={}",
        table.backend(),
        table.precision(),
        table.id(),
        discovery.id(),
        discovery.provenance(),
        alternate.provenance(),
    );
    for record in table.records() {
        println!(
            "evidence candidate={:?} channel={:?} result={:?}",
            record.candidate(),
            record.channel(),
            record.value()
        );
    }

    let accepted = session.accept(&table, &policy)?;
    let parent = accepted
        .first()
        .ok_or_else(|| std::io::Error::other("explicit policy selected no candidate"))?
        .feature();
    println!(
        "accepted feature={parent:?} evaluation={} provenance={}",
        accepted[0].evaluation(),
        accepted[0].frame().provenance(),
    );

    let child = session.registry_mut()?.softsign(parent)?;
    let child_table = session.evaluate(
        &mut executor,
        Arc::clone(&discovery),
        &[child],
        std::slice::from_ref(&consistency),
    )?;
    let child_accepted = session.accept(&child_table, &policy)?;
    println!(
        "derived child={child:?} parent={parent:?}; accepted={}",
        child_accepted.len()
    );

    let inference = frame(
        "unlabeled-new-rows/v1",
        vec![100, 101, 102, 103, 104],
        EvaluationRole::Inference,
        "declared-unlabeled-inference-view",
        vec![vec![0.5, 1.5, 2.5, 3.5, 4.5], vec![1.2, 2.0, 2.9, 4.1, 5.0]],
    )?;
    let materialized =
        session.materialize_accepted(&mut executor, inference.as_ref(), &child_accepted)?;
    println!(
        "unlabeled_materialization frame={} provenance={} feature={child:?} values={:?}",
        materialized.frame_id(),
        inference.provenance(),
        materialized.get(child)?,
    );
    println!("scope=Core mixed target-free semantic cookbook; no speed claim");
    Ok(())
}
