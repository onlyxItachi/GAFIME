//! Native Core lowering of bounded semantic programs. Reuses the production
//! mixed Pearson SIMD reduction; new pointwise operations use f32 arithmetic.
//! Candidate-parallel levels preserve each program's ordered arithmetic and
//! collect into deterministic IDs. No target buffer or metric ID is fabricated.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use gafime_orchestrator::semantic::{
    CandidateRegistry, EvidenceDefinition, EvidenceValue, FeatureFrame, FeatureId, FeatureOp,
    MaterializedColumns, NativeEvidenceExecutor, SemanticError, SemanticResult, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};
use rayon::prelude::*;

use crate::kernels::precision::pearson_mixed;

#[derive(Default)]
pub struct CoreEvidenceExecutor {
    materialized_nodes: usize,
    reused_nodes: usize,
}

impl CoreEvidenceExecutor {
    /// Observed dispatch accounting, not an inferred cache/performance claim.
    pub fn materialized_nodes(&self) -> usize {
        self.materialized_nodes
    }
    pub fn reused_nodes(&self) -> usize {
        self.reused_nodes
    }
}

impl NativeEvidenceExecutor for CoreEvidenceExecutor {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    fn materialize(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        candidates: &[FeatureId],
        retained: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        if registry.precision() != PrecisionProfile::Mixed || registry.schema() != frame.schema() {
            return Err(SemanticError::Invalid(
                "Core semantic lowering requires exact schema and mixed precision",
            ));
        }
        if candidates.len() > registry.limits().max_nodes {
            return Err(SemanticError::Invalid(
                "materialization root count exceeds node budget",
            ));
        }
        if retained.is_some_and(|c| c.frame_id() != frame.id()) {
            return Err(SemanticError::Invalid(
                "retained values belong to another input context",
            ));
        }
        let mut needed = BTreeSet::new();
        let mut pending = candidates.to_vec();
        let mut bank = BTreeMap::new();
        while let Some(id) = pending.pop() {
            let program = registry.program(id)?;
            if !needed.insert(id) {
                continue;
            }
            if let Some(value) = retained.and_then(|c| c.columns().get(&id)) {
                bank.insert(id, Arc::clone(value));
                continue;
            }
            match program.op() {
                FeatureOp::Source(_) => {}
                FeatureOp::AbsoluteDifference(a, b) => {
                    pending.push(*a);
                    pending.push(*b);
                }
                FeatureOp::Softsign(a) => pending.push(*a),
                FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
            }
        }
        // Three banks conservatively cover retained outputs, per-worker Vecs
        // and Vec -> Arc conversion. Labels gather at most one extra bank.
        // Frames/evidence metadata and caller-retained result tables are separate.
        let bytes = needed
            .len()
            .checked_mul(frame.rows())
            .and_then(|n| n.checked_mul(4 * 3));
        if bytes.is_none_or(|n| n > max_bytes) {
            return Err(SemanticError::Invalid(
                "semantic dependency working set exceeds execution budget",
            ));
        }
        self.reused_nodes = self.reused_nodes.saturating_add(bank.len());
        let mut levels: BTreeMap<usize, Vec<FeatureId>> = BTreeMap::new();
        for &id in &needed {
            if !bank.contains_key(&id) {
                levels
                    .entry(registry.program(id)?.depth())
                    .or_default()
                    .push(id);
            }
        }
        for ids in levels.values() {
            let outputs: SemanticResult<Vec<_>> = ids
                .par_iter()
                .map(|&id| {
                    let operand = |id| {
                        bank.get(&id)
                            .map(AsRef::as_ref)
                            .ok_or(SemanticError::Invalid(
                                "semantic dependency was not materialized",
                            ))
                    };
                    let values = match registry.program(id)?.op() {
                        FeatureOp::Source(index) => frame.column(*index as usize)?.to_vec(),
                        FeatureOp::AbsoluteDifference(a, b) => operand(*a)?
                            .iter()
                            .zip(operand(*b)?)
                            .map(|(&a, &b): (&f32, &f32)| (a - b).abs())
                            .collect(),
                        FeatureOp::Softsign(a) => {
                            operand(*a)?.iter().map(|&a| a / (1.0 + a.abs())).collect()
                        }
                        FeatureOp::CenteredProduct {
                            operands,
                            mean_bits,
                        } => {
                            let mut values = vec![1.0f32; frame.rows()];
                            for (&input, &bits) in operands.iter().zip(mean_bits) {
                                let mean = f32::from_bits(bits);
                                for (value, &input) in values.iter_mut().zip(operand(input)?) {
                                    *value *= input - mean;
                                }
                            }
                            values
                        }
                    };
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(SemanticError::Invalid(
                            "candidate arithmetic produced nonfinite values",
                        ));
                    }
                    Ok((id, Arc::<[f32]>::from(values)))
                })
                .collect();
            let outputs = outputs?;
            self.materialized_nodes = self.materialized_nodes.saturating_add(outputs.len());
            bank.extend(outputs);
        }
        let mut requested = BTreeMap::new();
        for &id in candidates {
            requested.insert(
                id,
                Arc::clone(
                    bank.get(&id)
                        .ok_or(SemanticError::Invalid("missing candidate output"))?,
                ),
            );
        }
        MaterializedColumns::from_columns(registry, frame, requested)
    }

    fn evaluate_channel(
        &mut self,
        definition: &EvidenceDefinition,
        candidates: &[FeatureId],
        values: &MaterializedColumns,
        paired: Option<&MaterializedColumns>,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        candidates
            .par_iter()
            .map(|&candidate| {
                let x = values.get(candidate)?;
                Ok(match definition {
                    EvidenceDefinition::Redundancy { reference } => {
                        correlation(x, values.get(*reference)?, true)
                    }
                    EvidenceDefinition::PairedConsistency { view } => {
                        let paired = paired.ok_or(SemanticError::Invalid(
                            "paired evidence requires materialized view",
                        ))?;
                        if paired.frame_id() != view.id() {
                            return Err(SemanticError::Invalid(
                                "paired materialization context mismatch",
                            ));
                        }
                        correlation(x, paired.get(candidate)?, false)
                    }
                    EvidenceDefinition::LabeledAssociation { labels: None } => {
                        unavailable(UnavailableReason::MissingLabels, 0)
                    }
                    EvidenceDefinition::LabeledAssociation {
                        labels: Some(labels),
                    } => {
                        if labels.frame_id() != values.frame_id() {
                            return Err(SemanticError::Invalid("label context mismatch"));
                        }
                        let subset: SemanticResult<Vec<_>> = labels
                            .rows()
                            .iter()
                            .map(|&i| {
                                x.get(i)
                                    .copied()
                                    .ok_or(SemanticError::Invalid("label row out of bounds"))
                            })
                            .collect();
                        correlation(&subset?, labels.values(), true)
                    }
                    EvidenceDefinition::GraphEnergy { graph } => {
                        if graph.frame_id() != values.frame_id() {
                            return Err(SemanticError::Invalid("graph context mismatch"));
                        }
                        if x.iter().all(|v| *v == x[0]) {
                            unavailable(UnavailableReason::ConstantOperand, graph.edges().len())
                        } else {
                            let mut numerator = 0.0f64;
                            let mut denominator = 0.0f64;
                            for edge in graph.edges() {
                                let a = f64::from(*x.get(edge.left).ok_or(
                                    SemanticError::Invalid("graph endpoint out of bounds"),
                                )?);
                                let b = f64::from(*x.get(edge.right).ok_or(
                                    SemanticError::Invalid("graph endpoint out of bounds"),
                                )?);
                                let w = f64::from(edge.weight);
                                numerator += w * (a - b) * (a - b);
                                denominator += w * (a * a + b * b);
                            }
                            // Deliberately uncentered, translation-sensitive and
                            // deterministic in declared edge order. Not Laplacian score.
                            EvidenceValue::measured(numerator / denominator, graph.edges().len())
                        }
                    }
                })
            })
            .collect()
    }
}

fn unavailable(reason: UnavailableReason, support: usize) -> EvidenceValue {
    EvidenceValue::Unavailable { reason, support }
}

fn correlation(x: &[f32], y: &[f32], absolute: bool) -> EvidenceValue {
    if x.len() != y.len() || x.len() < 2 {
        return unavailable(UnavailableReason::InsufficientSupport, x.len().min(y.len()));
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return unavailable(UnavailableReason::NonFiniteReduction, x.len());
    }
    if x.iter().all(|v| *v == x[0]) || y.iter().all(|v| *v == y[0]) {
        return unavailable(UnavailableReason::ConstantOperand, x.len());
    }
    let value = pearson_mixed(x, y);
    EvidenceValue::measured(if absolute { value.abs() } else { value }, x.len())
}
