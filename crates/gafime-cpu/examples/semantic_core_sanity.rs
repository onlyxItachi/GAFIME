//! Bounded internal Core lifecycle diagnostic, not public GAFIME throughput.
//! Cold and accepted-resident evaluations use identical programs and evidence.
use std::{sync::Arc, time::Instant};

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::*;
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

fn frame(profile: PrecisionProfile, offset: f64) -> SemanticResult<Arc<FeatureFrame>> {
    let rows = 8192;
    let columns = (0..12)
        .map(|column| {
            let values: Vec<_> = (0..rows)
                .map(|row| ((row * (column + 3) + column * 17) % 997) as f64 / 997.0 + offset)
                .collect();
            if profile == PrecisionProfile::Fp64 {
                NumericColumn::from(values)
            } else {
                NumericColumn::from(values.into_iter().map(|v| v as f32).collect::<Vec<_>>())
            }
        })
        .collect();
    Ok(Arc::new(FeatureFrame::with_profile(
        profile,
        (0..12).map(|i| format!("x{i}")).collect(),
        "deterministic-synthetic".into(),
        (0..rows as u64).collect(),
        EvaluationRole::Discovery,
        format!("fixed modular fixture offset={offset}"),
        columns,
    )?))
}

fn run(profile: PrecisionProfile, workers: usize) -> SemanticResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|_| SemanticError::Invalid("thread pool construction failed"))?
        .install(|| {
            let frame = frame(profile, 0.0)?;
            let view = self::frame(profile, 0.125)?;
            let registry =
                CandidateRegistry::new(frame.schema().to_vec(), profile, ProgramLimits::default())?;
            let mut session = SemanticSession::new(registry, GAFIME_BACKEND_CPU, 128 << 20)?;
            let mut candidates = Vec::new();
            let anchor;
            {
                let mut round = session.begin_round(&[])?;
                anchor = round.source(0)?;
                for a in 0..12 {
                    candidates.push(round.softsign(round.source(a)?)?);
                    for b in a + 1..12 {
                        candidates.push(round.abs_difference(round.source(a)?, round.source(b)?)?);
                    }
                }
            }
            let graph = Arc::new(NeighborGraph::new(
                &frame,
                (0..frame.rows() - 1)
                    .map(|i| GraphEdge {
                        left: i,
                        right: i + 1,
                        weight: 1.0,
                    })
                    .collect(),
                "ordered chain".into(),
            )?);
            let redundancy = EvidenceChannel::new(
                "reference".into(),
                EvidenceDefinition::Redundancy { reference: anchor },
            )?;
            let channels = vec![
                redundancy.clone(),
                EvidenceChannel::new(
                    "paired".into(),
                    EvidenceDefinition::PairedConsistency { view },
                )?,
                EvidenceChannel::new("graph".into(), EvidenceDefinition::GraphEnergy { graph })?,
                EvidenceChannel::new(
                    "same work distinct role".into(),
                    redundancy.definition().clone(),
                )?,
            ];
            let policy = SelectionPolicy {
                primary: redundancy.id(),
                direction: Direction::Maximize,
                constraints: vec![],
                missing: MissingEvidence::Error,
                limit: candidates.len(),
            };
            let mut core = CoreEvidenceExecutor::default();
            // Untimed correctness/warm-up precedes samples. No hardware-wide speed claim.
            let reference =
                session.evaluate(&mut core, Arc::clone(&frame), &candidates, &channels)?;
            let expected: Vec<_> = reference
                .records()
                .iter()
                .map(EvidenceRecord::value)
                .collect();
            for sample in 0..5 {
                session.clear_materializations()?;
                for mode in ["cold", "accepted-resident"] {
                    let nodes = core.materialized_nodes();
                    let hits = core.reused_nodes();
                    let started = Instant::now();
                    let table =
                        session.evaluate(&mut core, Arc::clone(&frame), &candidates, &channels)?;
                    let elapsed = started.elapsed().as_nanos();
                    let actual: Vec<_> =
                        table.records().iter().map(EvidenceRecord::value).collect();
                    assert_eq!(actual, expected, "residency changed evidence");
                    let accepted = session.accept(&table, &policy)?;
                    assert_eq!(accepted.len(), candidates.len());
                    println!(
                        "{profile:?},{workers},{sample},{mode},{},{},{elapsed},{},{},{}",
                        frame.rows(),
                        candidates.len(),
                        core.materialized_nodes() - nodes,
                        core.reused_nodes() - hits,
                        session.retained_bytes()
                    );
                }
            }
            Ok(())
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("profile,workers,sample,mode,rows,candidates,evaluation_ns,materialized_nodes,retained_hits,retained_bytes");
    let mut workers = vec![1, 4, rayon::current_num_threads()];
    workers.sort_unstable();
    workers.dedup();
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        for &workers in &workers {
            run(profile, workers)?;
        }
    }
    Ok(())
}
