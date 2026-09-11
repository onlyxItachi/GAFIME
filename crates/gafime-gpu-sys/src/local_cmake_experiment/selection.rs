//! Local-only arithmetic lowering for a bounded fp32 Pareto frontier.
//!
//! This module receives no semantic candidate/evidence IDs and no policy. It
//! owns only a temporary physical coordinate bank, one finite `<=` region per
//! coordinate row, and exact weak-dominator occupancy counts.

use gafime_orchestrator::semantic::{ParetoFrontierRequest, SemanticError, SemanticResult};
use gafime_types::{
    GafimeSemanticFrozenRegionTerm, PrecisionProfile, GAFIME_BACKEND_CUDA,
    GAFIME_SEMANTIC_REGION_LESS_EQUAL,
};

use super::{
    local_compact::{checked_add, remaining, Functions, Query, OCCUPANCY},
    GpuNativeEvidenceExecutor,
};

const MAX_LOCAL_PARETO_CANDIDATES: usize = 8_192;

/// Execute the physical weak-dominator count query for already-oriented fp32
/// coordinates. `coordinates` are column-major and contain no semantic ID or
/// selection-policy information. The caller retains equality grouping and all
/// strict-dominance/acceptance decisions.
pub(super) fn pareto_weak_dominator_counts(
    executor: &mut GpuNativeEvidenceExecutor,
    request: ParetoFrontierRequest<'_>,
    max_bytes: usize,
) -> SemanticResult<Vec<u64>> {
    let dimensions = request.objective_count();
    let candidates = request.candidate_count();
    if executor.backend.kind != GAFIME_BACKEND_CUDA
        || request.profile() != PrecisionProfile::Fp32
        || !(2..=3).contains(&dimensions)
    {
        return Err(SemanticError::Unsupported(
            "local RT Pareto requires CUDA fp32 evidence with two or three objectives",
        ));
    }
    if candidates == 0 || candidates > MAX_LOCAL_PARETO_CANDIDATES {
        return Err(SemanticError::Unsupported(
            "local RT Pareto candidate count exceeds the compact query envelope",
        ));
    }
    let coordinates = request
        .fp32_coordinates()
        .ok_or(SemanticError::Unsupported(
            "local RT Pareto requires physical fp32 objective coordinates",
        ))?;
    let coordinate_count = candidates
        .checked_mul(dimensions)
        .ok_or(SemanticError::Invalid(
            "local RT Pareto coordinate count overflow",
        ))?;
    if coordinates.len() != coordinate_count
        || coordinates
            .iter()
            .any(|value| !value.is_finite() || value.is_subnormal())
    {
        return Err(SemanticError::Invalid(
            "local RT Pareto rejects nonfinite, subnormal, or malformed fp32 coordinates",
        ));
    }
    let source_slots = u32::try_from(dimensions)
        .map_err(|_| SemanticError::Invalid("local RT Pareto axis count overflows u32"))?;
    let region_count = u32::try_from(candidates)
        .map_err(|_| SemanticError::Invalid("local RT Pareto candidate count overflows u32"))?;
    let term_count = coordinate_count;
    let descriptor_bytes = term_count
        .checked_mul(std::mem::size_of::<GafimeSemanticFrozenRegionTerm>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto descriptor bytes overflow",
        ))?;
    let offset_bytes = candidates
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(SemanticError::Invalid(
            "local RT Pareto offset bytes overflow",
        ))?;
    let bank_bytes = coordinate_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto bank bytes overflow",
        ))?;
    // Reserve result storage before creating the query. It is allocated while
    // the native `ExactStats` record vector is still live after execute.
    let count_bytes =
        candidates
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(SemanticError::Invalid(
                "local RT Pareto count bytes overflow",
            ))?;
    let fixed_bytes = checked_add(
        checked_add(descriptor_bytes, offset_bytes)?,
        checked_add(bank_bytes, count_bytes)?,
    )?;
    if fixed_bytes > max_bytes {
        return Err(SemanticError::Invalid(
            "local RT Pareto host and bank reservation exceeds selection budget",
        ));
    }
    let local = executor.backend.functions.local_cmake_experiment;
    let missing = || SemanticError::Unsupported("local RT Pareto compact-query ABI is incomplete");
    let functions = Functions {
        create: local.semantic_region_query_create_rt.ok_or_else(missing)?,
        execute: local.semantic_region_query_execute_rt.ok_or_else(missing)?,
        free: local.semantic_region_query_free_rt.ok_or_else(missing)?,
        coverage: local
            .semantic_region_query_materialize_coverage_rt
            .ok_or_else(missing)?,
    };

    // All descriptor, offset, bank, and returned-count capacities were
    // checked above before any allocation. Each candidate produces exactly one
    // frozen `<=` box across the original coordinate axes.
    let mut terms = Vec::with_capacity(term_count);
    let mut offsets = Vec::with_capacity(candidates + 1);
    offsets.push(0);
    for candidate in 0..candidates {
        for axis in 0..dimensions {
            terms.push(GafimeSemanticFrozenRegionTerm {
                input_slot: u32::try_from(axis)
                    .map_err(|_| SemanticError::Invalid("local RT Pareto axis overflows u32"))?,
                relation: GAFIME_SEMANTIC_REGION_LESS_EQUAL,
                threshold_bits: u64::from(coordinates[axis * candidates + candidate].to_bits()),
            });
        }
        offsets.push(
            u32::try_from(terms.len())
                .map_err(|_| SemanticError::Invalid("local RT Pareto term offsets overflow"))?,
        );
    }
    let actual_descriptor_bytes = terms
        .capacity()
        .checked_mul(std::mem::size_of::<GafimeSemanticFrozenRegionTerm>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto descriptor allocation bytes overflow",
        ))?;
    let actual_offset_bytes = offsets
        .capacity()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto offset allocation bytes overflow",
        ))?;
    if actual_descriptor_bytes != descriptor_bytes || actual_offset_bytes != offset_bytes {
        return Err(SemanticError::Invalid(
            "local RT Pareto descriptor allocation exceeded its reservation",
        ));
    }

    let bank = executor
        .backend
        .allocate_semantic_bank(
            PrecisionProfile::Fp32,
            candidates,
            source_slots,
            source_slots,
        )
        .map_err(GpuNativeEvidenceExecutor::semantic_error)?;
    if usize::try_from(bank.bytes()).map_err(|_| {
        SemanticError::Invalid("local RT Pareto bank bytes exceed host address space")
    })? > bank_bytes
    {
        return Err(SemanticError::Invalid(
            "local RT Pareto bank allocation exceeded its reservation",
        ));
    }
    bank.upload_f32(coordinates)
        .map_err(GpuNativeEvidenceExecutor::semantic_error)?;

    let query_budget = remaining(max_bytes, fixed_bytes)?;
    let mut query = Query::create(functions, bank, None, &terms, &[], &offsets, query_budget)?;
    let committed = checked_add(fixed_bytes, query.persistent_bytes)?;
    let execute_budget = remaining(max_bytes, committed)?;
    let (records, execution_peak) = query.execute(OCCUPANCY, 0, None, execute_budget)?;
    let record_bytes = records
        .capacity()
        .checked_mul(std::mem::size_of::<super::local_compact::ExactStats>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto record allocation bytes overflow",
        ))?;
    let expected_record_bytes = candidates
        .checked_mul(std::mem::size_of::<super::local_compact::ExactStats>())
        .ok_or(SemanticError::Invalid(
            "local RT Pareto record reservation bytes overflow",
        ))?;
    if records.len() != candidates
        || record_bytes != expected_record_bytes
        || records.iter().any(|record| {
            record.row_count != u64::from(region_count)
                || record.occupancy_inside > u64::from(region_count)
        })
        || checked_add(committed, execution_peak)? > max_bytes
    {
        return Err(SemanticError::Invalid(
            "local RT Pareto compact counts violate the bounded query request",
        ));
    }

    let mut counts = Vec::with_capacity(candidates);
    if counts.capacity().checked_mul(std::mem::size_of::<u64>()) != Some(count_bytes) {
        return Err(SemanticError::Invalid(
            "local RT Pareto count allocation exceeded its reservation",
        ));
    }
    counts.extend(records.iter().map(|record| record.occupancy_inside));
    if counts.len() != candidates {
        return Err(SemanticError::Invalid(
            "local RT Pareto count output violates its reserved shape",
        ));
    }
    Ok(counts)
}
