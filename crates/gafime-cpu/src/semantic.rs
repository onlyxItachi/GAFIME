//! Native Core lowering of bounded semantic programs.
//!
//! Candidate programs remain owned by the orchestrator. This module evaluates
//! their pointwise operations in the declared profile, reuses the production
//! Pearson kernels, and records only work it actually completes. It does not
//! fabricate targets, cache identities, or a cross-backend performance claim.

use std::collections::{BTreeMap, BTreeSet};

use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, EvidenceDefinition, EvidenceValue,
    FeatureFrame, FeatureId, FeatureOp, FrozenMeans, LabelSet, MaterializedColumns,
    NativeEvidenceExecutor, NeighborGraph, NumericColumn, SemanticError, SemanticResult,
    UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};
use rayon::prelude::*;

use crate::kernels::precision::{
    fixed_corrected_nmi_f32_checked, fixed_corrected_nmi_f64_checked,
    fixed_corrected_nmi_mixed_checked, multiply_centered_into_f32, multiply_centered_into_f64,
    pearson_f32_checked, pearson_f64_checked, pearson_mixed_checked, spearman_f32_checked,
    spearman_f64_checked, spearman_mixed_checked, FixedMiScratch, FixedNmiOutcome,
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
            if let Some(value) = retained
                .map(|columns| columns.columns().map(|values| values.get(&id)))
                .transpose()?
                .flatten()
            {
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
            EvidenceDefinition::Association {
                statistic,
                context: AssociationContext::Reference { reference },
            } => parallel_association(
                candidates, values, values, *reference, *statistic, true, max_bytes,
            )?,
            EvidenceDefinition::Association {
                statistic,
                context: AssociationContext::PairedView { view },
            } => {
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
                parallel_association_between(
                    candidates, values, paired, *statistic, false, max_bytes,
                )?
            }
            EvidenceDefinition::Association {
                context: AssociationContext::Labels { labels: None },
                ..
            } => (
                vec![unavailable(UnavailableReason::MissingLabels, 0); candidates.len()],
                0,
            ),
            EvidenceDefinition::Association {
                statistic,
                context:
                    AssociationContext::Labels {
                        labels: Some(labels),
                    },
            } => {
                if labels.frame_id() != values.frame_id() || labels.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "label context or precision mismatch",
                    ));
                }
                labeled_association(candidates, values, labels, *statistic, max_bytes)?
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

    fn retain(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        source: &MaterializedColumns,
        prior: Option<&MaterializedColumns>,
        selected: &[FeatureId],
        max_live_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        if registry.schema() != frame.schema()
            || registry.precision() != frame.profile()
            || source.frame_id() != frame.id()
            || source.profile() != frame.profile()
            || source.backend_kind() != GAFIME_BACKEND_CPU
            || prior.is_some_and(|values| {
                values.frame_id() != frame.id()
                    || values.profile() != frame.profile()
                    || values.backend_kind() != GAFIME_BACKEND_CPU
            })
        {
            return Err(SemanticError::Invalid(
                "Core retained materialization context or backend mismatch",
            ));
        }
        let source_columns = source.columns()?;
        let prior_columns = match prior {
            Some(values) => Some(values.columns()?),
            None => None,
        };
        let mut merged = prior_columns.cloned().unwrap_or_default();
        for &feature in selected {
            registry.program(feature)?;
            let values = source_columns.get(&feature).ok_or(SemanticError::Invalid(
                "selected feature is absent from materialized source",
            ))?;
            merged.insert(feature, values.shared_clone());
        }
        let result = MaterializedColumns::from_columns(registry, frame, merged)?;
        let live = source
            .bytes()
            .checked_add(prior.map_or(0, MaterializedColumns::bytes))
            .ok_or(SemanticError::Invalid(
                "Core retained materialization byte count overflow",
            ))?;
        if live > max_live_bytes || result.bytes() > max_live_bytes {
            return Err(SemanticError::Invalid(
                "Core retained materialization exceeds live byte budget",
            ));
        }
        Ok(result)
    }

    fn download(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        source: &MaterializedColumns,
        max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        if source.frame_id() != frame.id()
            || source.profile() != frame.profile()
            || source.backend_kind() != GAFIME_BACKEND_CPU
            || source.bytes() > max_bytes
        {
            return Err(SemanticError::Invalid(
                "Core materialization download context or budget mismatch",
            ));
        }
        MaterializedColumns::from_columns(registry, frame, source.columns()?.clone())
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

#[derive(Clone, Copy)]
enum AssociationRight<'a> {
    Reference(&'a NumericColumn),
    Paired(&'a MaterializedColumns),
}

impl<'a> AssociationRight<'a> {
    fn column(self, candidate: FeatureId) -> SemanticResult<&'a NumericColumn> {
        match self {
            Self::Reference(values) => Ok(values),
            Self::Paired(values) => values.get_typed(candidate),
        }
    }
}

fn parallel_association(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    reference_owner: &MaterializedColumns,
    reference: FeatureId,
    statistic: AssociationStatistic,
    absolute: bool,
    max_bytes: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    parallel_association_with_right(
        candidates,
        values,
        AssociationRight::Reference(reference_owner.get_typed(reference)?),
        statistic,
        absolute,
        max_bytes,
    )
}

fn parallel_association_between(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    paired: &MaterializedColumns,
    statistic: AssociationStatistic,
    absolute: bool,
    max_bytes: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    parallel_association_with_right(
        candidates,
        values,
        AssociationRight::Paired(paired),
        statistic,
        absolute,
        max_bytes,
    )
}

fn parallel_association_with_right(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    right: AssociationRight<'_>,
    statistic: AssociationStatistic,
    absolute: bool,
    max_bytes: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    if candidates.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let workers = rayon::current_num_threads().min(candidates.len());
    let rows = values.get_typed(candidates[0])?.len();
    let scratch_bytes = association_scratch_bytes(statistic, values.profile(), rows, workers)?;
    if scratch_bytes > max_bytes {
        return Err(SemanticError::Invalid(
            "association scratch exceeds execution budget",
        ));
    }
    // `map_init` is per Rayon job. This indexed minimum bounds concurrent
    // scratch instances by the worker-count reservation above.
    let minimum_job_len = candidates.len().div_ceil(workers);
    let results: SemanticResult<Vec<_>> = match values.profile() {
        PrecisionProfile::Fp32 => candidates
            .par_iter()
            .with_min_len(minimum_job_len)
            .map_init(FixedMiScratch::default, |scratch, &candidate| {
                association_f32(
                    values.get_typed(candidate)?.as_f32()?,
                    right.column(candidate)?.as_f32()?,
                    statistic,
                    absolute,
                    scratch,
                )
            })
            .collect(),
        PrecisionProfile::Mixed => candidates
            .par_iter()
            .with_min_len(minimum_job_len)
            .map_init(FixedMiScratch::default, |scratch, &candidate| {
                association_mixed(
                    values.get_typed(candidate)?.as_f32()?,
                    right.column(candidate)?.as_f32()?,
                    statistic,
                    absolute,
                    scratch,
                )
            })
            .collect(),
        PrecisionProfile::Fp64 => candidates
            .par_iter()
            .with_min_len(minimum_job_len)
            .map_init(FixedMiScratch::default, |scratch, &candidate| {
                association_f64(
                    values.get_typed(candidate)?.as_f64()?,
                    right.column(candidate)?.as_f64()?,
                    statistic,
                    absolute,
                    scratch,
                )
            })
            .collect(),
    };
    Ok(split_evidence(results?))
}

fn association_f32(
    left: &[f32],
    right: &[f32],
    statistic: AssociationStatistic,
    absolute: bool,
    scratch: &mut FixedMiScratch,
) -> SemanticResult<(EvidenceValue, bool)> {
    if left.len() != right.len() || left.len() < 2 {
        return Ok((
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        ));
    }
    if is_constant_f32(left) || is_constant_f32(right) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        ));
    }
    let measured = |value: f32| {
        EvidenceValue::measured_f32(if absolute { value.abs() } else { value }, left.len())
    };
    let result = match statistic {
        AssociationStatistic::Pearson => pearson_f32_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::Spearman => spearman_f32_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::FixedCorrectedNmi { bins } => {
            if left.len() < fixed_nmi_support(bins) {
                (
                    unavailable(UnavailableReason::InsufficientSupport, left.len()),
                    false,
                )
            } else {
                match fixed_corrected_nmi_f32_checked(left, right, bins, scratch) {
                    FixedNmiOutcome::Measured(value) => {
                        (EvidenceValue::measured_f32(value, left.len()), true)
                    }
                    FixedNmiOutcome::Degenerate => (
                        unavailable(UnavailableReason::DegenerateReduction, left.len()),
                        true,
                    ),
                    FixedNmiOutcome::NonFinite => (
                        unavailable(UnavailableReason::NonFiniteReduction, left.len()),
                        true,
                    ),
                }
            }
        }
    };
    Ok(result)
}

fn association_mixed(
    left: &[f32],
    right: &[f32],
    statistic: AssociationStatistic,
    absolute: bool,
    scratch: &mut FixedMiScratch,
) -> SemanticResult<(EvidenceValue, bool)> {
    if left.len() != right.len() || left.len() < 2 {
        return Ok((
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        ));
    }
    if is_constant_f32(left) || is_constant_f32(right) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        ));
    }
    let measured = |value: f64| {
        EvidenceValue::measured(if absolute { value.abs() } else { value }, left.len())
    };
    let result = match statistic {
        AssociationStatistic::Pearson => pearson_mixed_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::Spearman => spearman_mixed_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::FixedCorrectedNmi { bins } => {
            if left.len() < fixed_nmi_support(bins) {
                (
                    unavailable(UnavailableReason::InsufficientSupport, left.len()),
                    false,
                )
            } else {
                match fixed_corrected_nmi_mixed_checked(left, right, bins, scratch) {
                    FixedNmiOutcome::Measured(value) => {
                        (EvidenceValue::measured(value, left.len()), true)
                    }
                    FixedNmiOutcome::Degenerate => (
                        unavailable(UnavailableReason::DegenerateReduction, left.len()),
                        true,
                    ),
                    FixedNmiOutcome::NonFinite => (
                        unavailable(UnavailableReason::NonFiniteReduction, left.len()),
                        true,
                    ),
                }
            }
        }
    };
    Ok(result)
}

fn association_f64(
    left: &[f64],
    right: &[f64],
    statistic: AssociationStatistic,
    absolute: bool,
    scratch: &mut FixedMiScratch,
) -> SemanticResult<(EvidenceValue, bool)> {
    if left.len() != right.len() || left.len() < 2 {
        return Ok((
            unavailable(
                UnavailableReason::InsufficientSupport,
                left.len().min(right.len()),
            ),
            false,
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Ok((
            unavailable(UnavailableReason::NonFiniteReduction, left.len()),
            false,
        ));
    }
    if is_constant_f64(left) || is_constant_f64(right) {
        return Ok((
            unavailable(UnavailableReason::ConstantOperand, left.len()),
            false,
        ));
    }
    let measured = |value: f64| {
        EvidenceValue::measured(if absolute { value.abs() } else { value }, left.len())
    };
    let result = match statistic {
        AssociationStatistic::Pearson => pearson_f64_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::Spearman => spearman_f64_checked(left, right)
            .map(|value| (measured(value), true))
            .unwrap_or_else(|| {
                (
                    unavailable(UnavailableReason::DegenerateReduction, left.len()),
                    true,
                )
            }),
        AssociationStatistic::FixedCorrectedNmi { bins } => {
            if left.len() < fixed_nmi_support(bins) {
                (
                    unavailable(UnavailableReason::InsufficientSupport, left.len()),
                    false,
                )
            } else {
                match fixed_corrected_nmi_f64_checked(left, right, bins, scratch) {
                    FixedNmiOutcome::Measured(value) => {
                        (EvidenceValue::measured(value, left.len()), true)
                    }
                    FixedNmiOutcome::Degenerate => (
                        unavailable(UnavailableReason::DegenerateReduction, left.len()),
                        true,
                    ),
                    FixedNmiOutcome::NonFinite => (
                        unavailable(UnavailableReason::NonFiniteReduction, left.len()),
                        true,
                    ),
                }
            }
        }
    };
    Ok(result)
}

fn fixed_nmi_support(bins: u32) -> usize {
    usize::try_from(bins)
        .ok()
        .and_then(|bins| bins.checked_mul(bins))
        .and_then(|joint| joint.checked_mul(8))
        .unwrap_or(usize::MAX)
}

fn association_scratch_bytes(
    statistic: AssociationStatistic,
    profile: PrecisionProfile,
    rows: usize,
    workers: usize,
) -> SemanticResult<usize> {
    let per_worker = match statistic {
        AssociationStatistic::Pearson => 0,
        // Two rank-position vectors plus two typed rank vectors are a
        // conservative reservation for the existing checked rank primitive.
        AssociationStatistic::Spearman => {
            let rank_value_bytes = match profile {
                PrecisionProfile::Fp32 => std::mem::size_of::<f32>(),
                PrecisionProfile::Mixed | PrecisionProfile::Fp64 => std::mem::size_of::<f64>(),
            };
            rows.checked_mul(2 * std::mem::size_of::<u64>() + 2 * rank_value_bytes)
                .ok_or(SemanticError::Invalid(
                    "association scratch exceeds host address space",
                ))?
        }
        AssociationStatistic::FixedCorrectedNmi { bins } => {
            let bins = usize::try_from(bins)
                .map_err(|_| SemanticError::Invalid("fixed NMI bins exceed host address space"))?;
            let histogram_cells = bins
                .checked_mul(bins)
                .and_then(|joint| joint.checked_add(bins.checked_mul(2)?))
                .ok_or(SemanticError::Invalid(
                    "fixed NMI histogram exceeds host address space",
                ))?;
            histogram_cells
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(SemanticError::Invalid(
                    "fixed NMI histogram exceeds host address space",
                ))?
        }
    };
    workers
        .checked_mul(per_worker)
        .ok_or(SemanticError::Invalid(
            "association scratch exceeds host address space",
        ))
}

fn labeled_association(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    statistic: AssociationStatistic,
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
    let workers = rayon::current_num_threads().min(candidates.len());
    let gather_bytes = checked_bytes(
        workers,
        labels.rows().len(),
        element_bytes(values.profile()),
    )?;
    let association_bytes =
        association_scratch_bytes(statistic, values.profile(), labels.rows().len(), workers)?;
    let scratch_bytes =
        gather_bytes
            .checked_add(association_bytes)
            .ok_or(SemanticError::Invalid(
                "label association scratch exceeds host address space",
            ))?;
    if scratch_bytes > max_bytes {
        return Err(SemanticError::Invalid(
            "label association scratch exceeds execution budget",
        ));
    }

    // Rayon initializes map_init once per job rather than per worker. The
    // indexed minimum keeps concurrent scratch within the reservation above.
    let minimum_job_len = candidates.len().div_ceil(workers);
    match values.profile() {
        PrecisionProfile::Fp32 => {
            labeled_association_f32(candidates, values, labels, statistic, minimum_job_len)
        }
        PrecisionProfile::Mixed => {
            labeled_association_mixed(candidates, values, labels, statistic, minimum_job_len)
        }
        PrecisionProfile::Fp64 => {
            labeled_association_f64(candidates, values, labels, statistic, minimum_job_len)
        }
    }
}

fn labeled_association_f32(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    statistic: AssociationStatistic,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f32()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || (Vec::<f32>::with_capacity(rows), FixedMiScratch::default()),
            |(scratch, fixed_mi), &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f32()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                association_f32(scratch, label_values, statistic, true, fixed_mi)
            },
        )
        .collect();
    Ok(split_evidence(results?))
}

fn labeled_association_mixed(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    statistic: AssociationStatistic,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f32()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || (Vec::<f32>::with_capacity(rows), FixedMiScratch::default()),
            |(scratch, fixed_mi), &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f32()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                association_mixed(scratch, label_values, statistic, true, fixed_mi)
            },
        )
        .collect();
    Ok(split_evidence(results?))
}

fn labeled_association_f64(
    candidates: &[FeatureId],
    values: &MaterializedColumns,
    labels: &LabelSet,
    statistic: AssociationStatistic,
    minimum_job_len: usize,
) -> SemanticResult<(Vec<EvidenceValue>, usize)> {
    let label_rows = labels.rows();
    let label_values = labels.values_typed().as_f64()?;
    let rows = label_values.len();
    let results: SemanticResult<Vec<_>> = candidates
        .par_iter()
        .with_min_len(minimum_job_len)
        .map_init(
            || (Vec::<f64>::with_capacity(rows), FixedMiScratch::default()),
            |(scratch, fixed_mi), &candidate| {
                scratch.clear();
                let candidate = values.get_typed(candidate)?.as_f64()?;
                for &row in label_rows {
                    scratch.push(
                        *candidate
                            .get(row)
                            .ok_or(SemanticError::Invalid("label row out of bounds"))?,
                    );
                }
                association_f64(scratch, label_values, statistic, true, fixed_mi)
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
