//! Native Core lowering of bounded semantic programs.
//!
//! Candidate programs remain owned by the orchestrator. This module evaluates
//! their pointwise operations in the declared profile, reuses the production
//! Pearson kernels, and records only work it actually completes. It does not
//! fabricate targets, cache identities, or a cross-backend performance claim.

use std::collections::{BTreeMap, BTreeSet};

use gafime_orchestrator::semantic::{
    CandidateRegistry, EvidenceDefinition, EvidenceValue, FeatureFrame, FeatureId, FeatureOp,
    FrozenMeans, LabelSet, MaterializedColumns, NativeEvidenceExecutor, NeighborGraph,
    NumericColumn, SemanticError, SemanticResult, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};
use rayon::prelude::*;

use crate::kernels::precision::{
    multiply_centered_into_f32, multiply_centered_into_f64, pearson_f32_checked,
    pearson_f64_checked, pearson_mixed_checked,
};

const MAX_NATIVE_BYTES: usize = 512 * 1024 * 1024;

/// Native counters describe completed work, not a timing or cache model.
#[derive(Default)]
pub struct CoreEvidenceExecutor {
    materialized_nodes: usize,
    retained_hits: usize,
    source_shares: usize,
    output_allocations: usize,
    output_bytes: usize,
    evidence_kernel_calls: usize,
}

impl CoreEvidenceExecutor {
    /// Number of program nodes completed by this executor, including sources.
    pub fn materialized_nodes(&self) -> usize {
        self.materialized_nodes
    }

    /// Compatibility name for observed retained-bank hits.
    pub fn reused_nodes(&self) -> usize {
        self.retained_hits
    }

    /// Number of values actually obtained from the retained materialization.
    pub fn retained_hits(&self) -> usize {
        self.retained_hits
    }

    /// Number of source columns shared by cloning their immutable ownership
    /// handle instead of copying their numeric payload.
    pub fn source_shares(&self) -> usize {
        self.source_shares
    }

    /// Number of derived pointwise output vectors allocated by this executor.
    pub fn output_allocations(&self) -> usize {
        self.output_allocations
    }

    /// Bytes in those derived output vectors, excluding frame-owned sources,
    /// metadata, backend copies, and temporary label scratch.
    pub fn output_bytes(&self) -> usize {
        self.output_bytes
    }

    /// Completed candidate-level numerical evidence primitives. Missing or
    /// unavailable evidence that never reaches a numerical primitive is not
    /// counted as a kernel call.
    pub fn evidence_kernel_calls(&self) -> usize {
        self.evidence_kernel_calls
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
        validate_budget(max_bytes)?;
        if registry.schema() != frame.schema() || registry.precision() != frame.profile() {
            return Err(SemanticError::Invalid(
                "Core semantic lowering requires exact schema and precision",
            ));
        }
        if candidates.len() > registry.limits().max_nodes {
            return Err(SemanticError::Invalid(
                "materialization root count exceeds node budget",
            ));
        }
        if retained.is_some_and(|columns| {
            columns.frame_id() != frame.id() || columns.profile() != frame.profile()
        }) {
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
            if let Some(value) = retained.and_then(|columns| columns.columns().get(&id)) {
                bank.insert(id, value.shared_clone());
                continue;
            }
            match program.op() {
                FeatureOp::Source(_) => {}
                FeatureOp::AbsoluteDifference(left, right) => pending.extend([*left, *right]),
                FeatureOp::Softsign(input) => pending.push(*input),
                FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
            }
        }

        // Admission reserves one profile-typed logical bank for every missing
        // node. Source nodes share their frame payload below, but still appear
        // in the returned materialization and remain conservatively bounded.
        let missing = needed.iter().filter(|id| !bank.contains_key(id)).count();
        let planned_bytes = checked_bytes(missing, frame.rows(), element_bytes(frame.profile()))?;
        if planned_bytes > max_bytes {
            return Err(SemanticError::Invalid(
                "semantic dependency bank exceeds execution budget",
            ));
        }
        self.retained_hits = self.retained_hits.saturating_add(bank.len());

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
                .map(|&id| materialize_node(registry, frame, id, &bank))
                .collect();
            for output in outputs? {
                self.materialized_nodes = self.materialized_nodes.saturating_add(1);
                if output.source_shared {
                    self.source_shares = self.source_shares.saturating_add(1);
                } else {
                    self.output_allocations = self.output_allocations.saturating_add(1);
                    self.output_bytes = self.output_bytes.saturating_add(output.values.bytes());
                }
                bank.insert(output.id, output.values);
            }
        }

        let mut requested = BTreeMap::new();
        for &id in candidates {
            let values = bank
                .get(&id)
                .ok_or(SemanticError::Invalid("missing candidate output"))?;
            requested.insert(id, values.shared_clone());
        }
        MaterializedColumns::from_columns(registry, frame, requested)
    }

    fn evaluate_channel(
        &mut self,
        definition: &EvidenceDefinition,
        candidates: &[FeatureId],
        values: &MaterializedColumns,
        paired: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        validate_budget(max_bytes)?;
        validate_bank_profile(values, candidates)?;
        let (evidence, calls) = match definition {
            EvidenceDefinition::Redundancy { reference } => {
                parallel_correlation(candidates, values, values, *reference, true)?
            }
            EvidenceDefinition::PairedConsistency { view } => {
                let paired = paired.ok_or(SemanticError::Invalid(
                    "paired evidence requires materialized view",
                ))?;
                if paired.frame_id() != view.id()
                    || paired.profile() != view.profile()
                    || paired.profile() != values.profile()
                {
                    return Err(SemanticError::Invalid(
                        "paired materialization context or precision mismatch",
                    ));
                }
                validate_bank_profile(paired, candidates)?;
                parallel_correlation_between(candidates, values, paired, false)?
            }
            EvidenceDefinition::LabeledAssociation { labels: None } => (
                vec![unavailable(UnavailableReason::MissingLabels, 0); candidates.len()],
                0,
            ),
            EvidenceDefinition::LabeledAssociation {
                labels: Some(labels),
            } => {
                if labels.frame_id() != values.frame_id() || labels.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "label context or precision mismatch",
                    ));
                }
                labeled_association(candidates, values, labels, max_bytes)?
            }
            EvidenceDefinition::GraphEnergy { graph } => {
                if graph.frame_id() != values.frame_id() || graph.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "graph context or precision mismatch",
                    ));
                }
                parallel_graph_energy(candidates, values, graph)?
            }
        };
        self.evidence_kernel_calls = self.evidence_kernel_calls.saturating_add(calls);
        Ok(evidence)
    }
}

struct NodeMaterialization {
    id: FeatureId,
    values: NumericColumn,
    source_shared: bool,
}

fn validate_budget(max_bytes: usize) -> SemanticResult<()> {
    if max_bytes > MAX_NATIVE_BYTES {
        return Err(SemanticError::Invalid(
            "semantic native execution budget exceeds bounded Core limit",
        ));
    }
    Ok(())
}

fn checked_bytes(count: usize, rows: usize, width: usize) -> SemanticResult<usize> {
    count
        .checked_mul(rows)
        .and_then(|values| values.checked_mul(width))
        .ok_or(SemanticError::Invalid(
            "semantic numeric storage exceeds host address space",
        ))
}

const fn element_bytes(profile: PrecisionProfile) -> usize {
    match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => std::mem::size_of::<f32>(),
        PrecisionProfile::Fp64 => std::mem::size_of::<f64>(),
    }
}

fn materialize_node(
    registry: &CandidateRegistry,
    frame: &FeatureFrame,
    id: FeatureId,
    bank: &BTreeMap<FeatureId, NumericColumn>,
) -> SemanticResult<NodeMaterialization> {
    let program = registry.program(id)?;
    let (values, source_shared) = match program.op() {
        FeatureOp::Source(index) => (frame.column_typed(*index as usize)?.shared_clone(), true),
        FeatureOp::AbsoluteDifference(left, right) => (
            absolute_difference(
                frame.profile(),
                operand(bank, *left)?,
                operand(bank, *right)?,
                frame.rows(),
            )?,
            false,
        ),
        FeatureOp::Softsign(input) => (
            softsign(frame.profile(), operand(bank, *input)?, frame.rows())?,
            false,
        ),
        FeatureOp::CenteredProduct {
            operands,
            mean_bits,
        } => (
            centered_product(frame.profile(), operands, mean_bits, bank, frame.rows())?,
            false,
        ),
    };
    if values.len() != frame.rows() || !values.finite() || !values.supports_profile(frame.profile())
    {
        return Err(SemanticError::Invalid(
            "candidate arithmetic produced nonfinite or profile-incompatible values",
        ));
    }
    Ok(NodeMaterialization {
        id,
        values,
        source_shared,
    })
}

fn operand(
    bank: &BTreeMap<FeatureId, NumericColumn>,
    id: FeatureId,
) -> SemanticResult<&NumericColumn> {
    bank.get(&id).ok_or(SemanticError::Invalid(
        "semantic dependency was not materialized",
    ))
}

fn absolute_difference(
    profile: PrecisionProfile,
    left: &NumericColumn,
    right: &NumericColumn,
    rows: usize,
) -> SemanticResult<NumericColumn> {
    match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            let left = left.as_f32()?;
            let right = right.as_f32()?;
            if left.len() != rows || right.len() != rows {
                return Err(SemanticError::Invalid("unaligned pointwise operands"));
            }
            Ok(NumericColumn::from(
                left.iter()
                    .zip(right)
                    .map(|(&left, &right)| (left - right).abs())
                    .collect::<Vec<f32>>(),
            ))
        }
        PrecisionProfile::Fp64 => {
            let left = left.as_f64()?;
            let right = right.as_f64()?;
            if left.len() != rows || right.len() != rows {
                return Err(SemanticError::Invalid("unaligned pointwise operands"));
            }
            Ok(NumericColumn::from(
                left.iter()
                    .zip(right)
                    .map(|(&left, &right)| (left - right).abs())
                    .collect::<Vec<f64>>(),
            ))
        }
    }
}

fn softsign(
    profile: PrecisionProfile,
    input: &NumericColumn,
    rows: usize,
) -> SemanticResult<NumericColumn> {
    match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            let input = input.as_f32()?;
            if input.len() != rows {
                return Err(SemanticError::Invalid("unaligned pointwise operand"));
            }
            Ok(NumericColumn::from(
                input
                    .iter()
                    .map(|&value| value / (1.0 + value.abs()))
                    .collect::<Vec<f32>>(),
            ))
        }
        PrecisionProfile::Fp64 => {
            let input = input.as_f64()?;
            if input.len() != rows {
                return Err(SemanticError::Invalid("unaligned pointwise operand"));
            }
            Ok(NumericColumn::from(
                input
                    .iter()
                    .map(|&value| value / (1.0 + value.abs()))
                    .collect::<Vec<f64>>(),
            ))
        }
    }
}

fn centered_product(
    profile: PrecisionProfile,
    operands: &[FeatureId],
    means: &FrozenMeans,
    bank: &BTreeMap<FeatureId, NumericColumn>,
    rows: usize,
) -> SemanticResult<NumericColumn> {
    match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            let means = means.as_f32_bits()?;
            if operands.len() != means.len() {
                return Err(SemanticError::Invalid(
                    "centered product means do not match operands",
                ));
            }
            let mut output = vec![1.0f32; rows];
            for (&operand_id, &bits) in operands.iter().zip(means) {
                let input = operand(bank, operand_id)?.as_f32()?;
                if input.len() != rows {
                    return Err(SemanticError::Invalid("unaligned pointwise operand"));
                }
                let mean = f32::from_bits(bits);
                multiply_centered_into_f32(&mut output, input, mean);
            }
            Ok(NumericColumn::from(output))
        }
        PrecisionProfile::Fp64 => {
            let means = means.as_f64_bits()?;
            if operands.len() != means.len() {
                return Err(SemanticError::Invalid(
                    "centered product means do not match operands",
                ));
            }
            let mut output = vec![1.0f64; rows];
            for (&operand_id, &bits) in operands.iter().zip(means) {
                let input = operand(bank, operand_id)?.as_f64()?;
                if input.len() != rows {
                    return Err(SemanticError::Invalid("unaligned pointwise operand"));
                }
                let mean = f64::from_bits(bits);
                multiply_centered_into_f64(&mut output, input, mean);
            }
            Ok(NumericColumn::from(output))
        }
    }
}

fn validate_bank_profile(
    values: &MaterializedColumns,
    candidates: &[FeatureId],
) -> SemanticResult<()> {
    for &candidate in candidates {
        if !values
            .get_typed(candidate)?
            .supports_profile(values.profile())
        {
            return Err(SemanticError::Invalid(
                "materialized candidate storage does not match its precision",
            ));
        }
    }
    Ok(())
}

fn parallel_correlation(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    reference_owner: &MaterializedColumns,
    reference: FeatureId,
    absolute: bool,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let reference_values = reference_owner.get_typed(reference)?;
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .map(|&candidate| {
            correlation(
                values.profile(),
                values.get_typed(candidate)?,
                reference_values,
                absolute,
            )
        })
        .collect();
    Ok(split_evidence(results?))
}

fn parallel_correlation_between(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    paired: &MaterializedColumns,
    absolute: bool,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .map(|&candidate| {
            correlation(
                values.profile(),
                values.get_typed(candidate)?,
                paired.get_typed(candidate)?,
                absolute,
            )
        })
        .collect();
    Ok(split_evidence(results?))
}

fn correlation(
    profile: PrecisionProfile,
    left: &NumericColumn,
    right: &NumericColumn,
    absolute: bool,
) -> SemanticResult<(EvidenceValue, bool)> {
    match profile {
        PrecisionProfile::Fp32 => Ok(correlation_f32(left.as_f32()?, right.as_f32()?, absolute)),
        PrecisionProfile::Mixed => Ok(correlation_mixed(left.as_f32()?, right.as_f32()?, absolute)),
        PrecisionProfile::Fp64 => Ok(correlation_f64(left.as_f64()?, right.as_f64()?, absolute)),
    }
}

fn correlation_f32(left: &[f32], right: &[f32], absolute: bool) -> (EvidenceValue, bool) {
    if left.len() != right.len() || left.len() < 2 {
        return (
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        );
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return (
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        );
    }
    if is_constant_f32(left) || is_constant_f32(right) {
        return (
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        );
    }
    match pearson_f32_checked(left, right) {
        Some(value) => (
            EvidenceValue::measured_f32(if absolute { value.abs() } else { value }, left.len()),
            true,
        ),
        None => (
            unavailable(UnavailableReason::DegenerateReduction, left.len()),
            true,
        ),
    }
}

fn correlation_mixed(left: &[f32], right: &[f32], absolute: bool) -> (EvidenceValue, bool) {
    if left.len() != right.len() || left.len() < 2 {
        return (
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        );
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return (
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        );
    }
    if is_constant_f32(left) || is_constant_f32(right) {
        return (
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        );
    }
    match pearson_mixed_checked(left, right) {
        Some(value) => (
            EvidenceValue::measured(if absolute { value.abs() } else { value }, left.len()),
            true,
        ),
        None => (
            unavailable(UnavailableReason::DegenerateReduction, left.len()),
            true,
        ),
    }
}

fn correlation_f64(left: &[f64], right: &[f64], absolute: bool) -> (EvidenceValue, bool) {
    if left.len() != right.len() || left.len() < 2 {
        return (
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        );
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return (
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        );
    }
    if is_constant_f64(left) || is_constant_f64(right) {
        return (
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        );
    }
    match pearson_f64_checked(left, right) {
        Some(value) => (
            EvidenceValue::measured(if absolute { value.abs() } else { value }, left.len()),
            true,
        ),
        None => (
            unavailable(UnavailableReason::DegenerateReduction, left.len()),
            true,
        ),
    }
}

fn labeled_association(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    max_bytes: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    if labels.values_typed().len() != labels.rows().len()
        || !labels.values_typed().supports_profile(values.profile())
    {
        return Err(SemanticError::Invalid(
            "label rows and values are not aligned",
        ));
    }
    if labels.rows().len() < 2 {
        return Ok((
            vec![
                unavailable(UnavailableReason::InsufficientSupport, labels.rows().len());
                candidates.len()
            ],
            0,
        ));
    }
    if candidates.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let workers = rayon::current_num_threads();
    let scratch_bytes = checked_bytes(
        workers,
        labels.rows().len(),
        element_bytes(values.profile()),
    )?;
    if scratch_bytes > max_bytes {
        return Err(SemanticError::Invalid(
            "label gather scratch exceeds execution budget",
        ));
    }

    // Rayon initializes map_init once per job rather than per worker. The
    // indexed minimum keeps its number of jobs within this worker-count budget.
    let minimum_job_len = candidates.len().div_ceil(workers);
    match values.profile() {
        PrecisionProfile::Fp32 => {
            labeled_association_f32(candidates, values, labels, minimum_job_len)
        }
        PrecisionProfile::Mixed => {
            labeled_association_mixed(candidates, values, labels, minimum_job_len)
        }
        PrecisionProfile::Fp64 => {
            labeled_association_f64(candidates, values, labels, minimum_job_len)
        }
    }
}

fn labeled_association_f32(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f32()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || Vec::<f32>::with_capacity(rows),
            |scratch, &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f32()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                Ok(correlation_f32(scratch, label_values, true))
            },
        )
        .collect();
    Ok(split_evidence(results?))
}

fn labeled_association_mixed(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f32()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || Vec::<f32>::with_capacity(rows),
            |scratch, &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f32()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                Ok(correlation_mixed(scratch, label_values, true))
            },
        )
        .collect();
    Ok(split_evidence(results?))
}

fn labeled_association_f64(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f64()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || Vec::<f64>::with_capacity(rows),
            |scratch, &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f64()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                Ok(correlation_f64(scratch, label_values, true))
            },
        )
        .collect();
    Ok(split_evidence(results?))
}

fn parallel_graph_energy(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    graph: &NeighborGraph,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .map(|&candidate| graph_energy(values.profile(), values.get_typed(candidate)?, graph))
        .collect();
    Ok(split_evidence(results?))
}

fn graph_energy(
    profile: PrecisionProfile,
    values: &NumericColumn,
    graph: &NeighborGraph,
) -> SemanticResult<(EvidenceValue, bool)> {
    match profile {
        PrecisionProfile::Fp32 => graph_energy_f32(values.as_f32()?, graph),
        PrecisionProfile::Mixed => graph_energy_mixed(values.as_f32()?, graph),
        PrecisionProfile::Fp64 => graph_energy_f64(values.as_f64()?, graph),
    }
}

fn graph_energy_f32(
    values: &[f32],
    graph: &NeighborGraph,
) -> SemanticResult<(EvidenceValue, bool)> {
    graph_energy_ordered_f32(values, graph, graph.weights_typed().as_f32()?)
}

fn graph_energy_mixed(
    values: &[f32],
    graph: &NeighborGraph,
) -> SemanticResult<(EvidenceValue, bool)> {
    let weights = graph.weights_typed().as_f32()?;
    if values.is_empty() {
        return Ok((
            unavailable(UnavailableReason::InsufficientSupport, 0),
            false,
        ));
    }
    if is_constant_f32(values) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, graph.edges().len()),
            false,
        ));
    }
    if graph.edges().len() != weights.len() {
        return Err(SemanticError::Invalid(
            "graph topology and weights differ in length",
        ));
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (edge, &weight) in graph.edges().iter().zip(weights) {
        let left = f64::from(
            *values
                .get(edge.left)
                .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?,
        );
        let right = f64::from(
            *values
                .get(edge.right)
                .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?,
        );
        let weight = f64::from(weight);
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    if !numerator.is_finite() || !denominator.is_finite() {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, graph.edges().len()),
            true,
        ));
    }
    Ok((
        EvidenceValue::measured(numerator / denominator, graph.edges().len()),
        true,
    ))
}

fn graph_energy_f64(
    values: &[f64],
    graph: &NeighborGraph,
) -> SemanticResult<(EvidenceValue, bool)> {
    let weights = graph.weights_typed().as_f64()?;
    if values.is_empty() {
        return Ok((
            unavailable(UnavailableReason::InsufficientSupport, 0),
            false,
        ));
    }
    if is_constant_f64(values) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, graph.edges().len()),
            false,
        ));
    }
    if graph.edges().len() != weights.len() {
        return Err(SemanticError::Invalid(
            "graph topology and weights differ in length",
        ));
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (edge, &weight) in graph.edges().iter().zip(weights) {
        let left = *values
            .get(edge.left)
            .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?;
        let right = *values
            .get(edge.right)
            .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?;
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    if !numerator.is_finite() || !denominator.is_finite() {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, graph.edges().len()),
            true,
        ));
    }
    Ok((
        EvidenceValue::measured(numerator / denominator, graph.edges().len()),
        true,
    ))
}

fn graph_energy_ordered_f32(
    values: &[f32],
    graph: &NeighborGraph,
    weights: &[f32],
) -> SemanticResult<(EvidenceValue, bool)> {
    if values.is_empty() {
        return Ok((
            unavailable(UnavailableReason::InsufficientSupport, 0),
            false,
        ));
    }
    if is_constant_f32(values) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, graph.edges().len()),
            false,
        ));
    }
    if graph.edges().len() != weights.len() {
        return Err(SemanticError::Invalid(
            "graph topology and weights differ in length",
        ));
    }
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (edge, &weight) in graph.edges().iter().zip(weights) {
        let left = *values
            .get(edge.left)
            .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?;
        let right = *values
            .get(edge.right)
            .ok_or(SemanticError::Invalid("graph endpoint out of bounds"))?;
        numerator += weight * (left - right) * (left - right);
        denominator += weight * (left * left + right * right);
    }
    if !numerator.is_finite() || !denominator.is_finite() {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, graph.edges().len()),
            true,
        ));
    }
    Ok((
        EvidenceValue::measured_f32(numerator / denominator, graph.edges().len()),
        true,
    ))
}

fn split_evidence(values: Vec<(EvidenceValue, bool)>) -> (Vec<EvidenceValue>, usize) {
    let calls = values.iter().filter(|(_, called)| *called).count();
    (values.into_iter().map(|(value, _)| value).collect(), calls)
}

fn unavailable(reason: UnavailableReason, support: usize) -> EvidenceValue {
    EvidenceValue::Unavailable { reason, support }
}

fn is_constant_f32(values: &[f32]) -> bool {
    values
        .first()
        .is_some_and(|first| values.iter().all(|value| value == first))
}

fn is_constant_f64(values: &[f64]) -> bool {
    values
        .first()
        .is_some_and(|first| values.iter().all(|value| value == first))
}
