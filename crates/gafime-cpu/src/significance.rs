//! Native permutation-test + stability significance (P-A).
//!
//! GAFIME's core differentiator is *is a discovered interaction real, or noise?*
//! This module answers it natively (no Python loop, `rayon`-parallel), for the
//! survivors surfaced in a report:
//!
//! * **Permutation p-value** — shuffle `y`, re-score the (y-independent)
//!   interaction signal, count how often the permuted metric is at least as
//!   extreme as the observed metric. `p = (exceedances + 1) / (permutations + 1)`.
//! * **Stability** — bootstrap-resample the rows, recompute the metric, and take
//!   the mean/std across `num_repeats` resamples (low std = stable).
//!
//! The interaction *signal* (centered-product) is independent of `y`, so fixed
//! families only re-run metric reduction against a shuffled target. Adaptive
//! higher-order families also repeat unary screening and stream the resulting
//! permutation-specific combinations. Determinism comes from a seeded splitmix64
//! stream per permutation/repeat, so results are reproducible and parallel-safe.

use std::collections::HashSet;

use rayon::prelude::*;

use gafime_orchestrator::{
    plan::{
        combos::{
            legacy_higher_feature_order, legacy_higher_feature_order_f64,
            legacy_unary_feature_order, select_adaptive_mi_bins_for_backend,
        },
        DescriptorBatchSource, DEFAULT_DESCRIPTOR_BATCH_WORDS,
    },
    CompiledPlan, OrchestratorError, OrchestratorResult,
};
use gafime_types::{BackendKind, PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_FAMILY_CONTINUOUS};

use crate::matrix::CpuMatrix;
use crate::precision::{
    CpuPrecisionMatrix, CpuPrecisionScalar, CpuPrecisionSlice, CpuPrecisionValues,
};
use crate::simd;
use crate::{
    decision_path::{
        find_decision_paths_precision, path_membership_precision, DecisionPathParams,
        PrecisionDecisionPath,
    },
    kernels::{self, precision, MetricKernel},
};

/// Inputs that drive the significance passes (mirrors the significance-relevant
/// fields of `EngineConfig`).
#[derive(Clone, Copy, Debug)]
pub struct SignificanceParams {
    pub permutation_tests: u32,
    pub num_repeats: u32,
    pub random_seed: u64,
    pub mi_bins: u32,
    /// Backend that produced `observed`; its MI template ceiling must also govern
    /// the CPU fallback null and stability passes.
    pub backend_kind: BackendKind,
    /// CPU-only request for fixed-width MI. GPU observations always force the
    /// fixed-width estimator regardless of this configured value.
    pub mi_approximate: bool,
}

/// The target-dependent portion of continuous candidate generation. Unary
/// candidates are fixed by planning, but their scores select and order the
/// feature pool used for higher arities. maxT must repeat this selection for
/// every permuted target rather than replay the observed-target shortlist.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveSearchSpec<'a> {
    pub unary_features: &'a [u32],
    pub candidate_feature_count: u32,
    pub max_arity: u32,
    pub max_combinations_per_arity: u64,
    pub top_features_for_higher_arity: u32,
    pub planning_seed_words: &'a [u32],
}

/// Structural recipe for the continuous family produced after decision-path
/// discovery expands the base feature prefix.
///
/// The public candidate rows use `base_candidate_cols + observed_paths.len()`
/// columns.  A target permutation can discover a different number of paths,
/// so a compiled observed descriptor plan is deliberately *not* reused for a
/// null pass.  Instead the null pass starts from the same source feature
/// prefix, rediscovers the paths with its shuffled target, then re-applies this
/// exact continuous planning recipe to its own expanded column set.
///
/// Feature ids, descriptor limits, arities, and the planning seed remain
/// structural integer/control values. Only the numeric storage and metric
/// execution use the selected precision profile.
#[derive(Clone, Copy, Debug)]
pub struct ExpandedDecisionPathSearchSpec<'a> {
    /// Prefix of the original row-major source matrix that participates in the
    /// expanded continuous family. Path nodes are remapped into this prefix.
    pub base_candidate_cols: u32,
    /// Original-prefix feature ids used for target-dependent path discovery.
    /// They may be a strict subset of `0..base_candidate_cols`.
    pub discovery_features: &'a [u32],
    /// Target-dependent CART/residual discovery controls.
    pub discovery: DecisionPathParams,
    /// Continuous planner's maximum arity (the v1 surface exercises 1–5).
    pub max_arity: u32,
    /// Per-arity continuous descriptor limit.
    pub max_combinations_per_arity: u64,
    /// Number of unary survivors available to higher-order planning.
    pub top_features_for_higher_arity: u32,
    /// Existing structural planning-seed words.
    pub planning_seed_words: &'a [u32],
}

/// Per-candidate significance, one value per metric (aligned to the metric set).
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSignificance {
    /// Permutation p-values (NaN when `permutation_tests == 0`).
    pub pvalues: Vec<f32>,
    /// Bootstrap metric means across resamples.
    pub means: Vec<f32>,
    /// Bootstrap metric standard deviations across resamples.
    pub stds: Vec<f32>,
}

/// Typed significance values for one shortlisted candidate.  fp32 retains f32
/// p-values/means/stds; mixed and fp64 retain f64 values through observed
/// scoring, permutation comparison, bootstrap statistics, and public output.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionCandidateSignificance {
    /// Stable candidate identity supplied by the structural planner. It never
    /// participates in floating-point arithmetic.
    pub candidate_id: u64,
    pub pvalues: CpuPrecisionValues,
    pub means: CpuPrecisionValues,
    pub stds: CpuPrecisionValues,
}

/// One materialized family candidate supplied to the typed significance path.
///
/// Time-series and decision-path descriptors remain outside this value because
/// they are structural metadata. This object carries only the profile-typed
/// numeric signal and the existing integer candidate identity.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionSignificanceSignal {
    pub candidate_id: u64,
    pub values: CpuPrecisionValues,
}

/// Evaluate a fixed shortlist under a single precision profile.
///
/// The adaptive planner remains structural and owns candidate enumeration.
/// This CPU entrypoint receives those candidate ids/combinations after planning,
/// materializes each interaction in the correct pointwise storage dtype once,
/// then performs observed, permutation/maxT, and bootstrap arithmetic in the
/// declared profile reduction/result dtype.
pub fn evaluate_precision_shortlist(
    matrix: &CpuPrecisionMatrix,
    combos: &[Vec<u32>],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let candidate_ids = (0..combos.len())
        .map(|candidate| candidate as u64)
        .collect::<Vec<_>>();
    evaluate_precision_continuous_shortlist(matrix, combos, &candidate_ids, metrics, params)
}

/// Evaluate a continuous-combo shortlist while preserving the planner's stable
/// candidate ids in the typed significance result.
pub fn evaluate_precision_continuous_shortlist(
    matrix: &CpuPrecisionMatrix,
    combos: &[Vec<u32>],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if candidate_ids.len() != combos.len() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous significance candidate ids do not match its shortlist",
        ));
    }
    let signals = combos
        .iter()
        .enumerate()
        .map(|(candidate_id, result)| {
            precision::materialize_precision_combo(matrix, result).map(|values| {
                PrecisionSignificanceSignal {
                    candidate_id: candidate_ids[candidate_id],
                    values,
                }
            })
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    evaluate_precision_signals(matrix.profile(), matrix.target(), &signals, metrics, params)
}

/// Evaluate any fixed, materialized numeric family with the selected profile.
///
/// This is the shared bootstrap/permutation/maxT implementation for continuous
/// interactions, generated time-series columns, and a fixed shortlist of
/// decision-path memberships. It validates the profile before inspecting a
/// candidate value so an fp64 signal cannot be narrowed on the way into a
/// significance pass.
pub fn evaluate_precision_signals(
    profile: PrecisionProfile,
    target: CpuPrecisionSlice<'_>,
    signals: &[PrecisionSignificanceSignal],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let candidate_ids = signals
        .iter()
        .map(|signal| signal.candidate_id)
        .collect::<Vec<_>>();
    match profile {
        PrecisionProfile::Fp32 => {
            let CpuPrecisionSlice::F32(target) = target else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp32 significance target is not f32",
                ));
            };
            let score_signals = signals
                .iter()
                .map(|signal| match signal {
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F32(values),
                        ..
                    } if values.len() == target.len() => Ok(values.as_slice()),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F32(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "fp32 significance signal length does not match target",
                    )),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F64(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "fp32 significance received f64 interaction storage",
                    )),
                })
                .collect::<OrchestratorResult<Vec<_>>>()?;
            let mut result = evaluate_precision_f32(&score_signals, target, metrics, params);
            assign_precision_candidate_ids(&mut result, &candidate_ids);
            Ok(result)
        }
        PrecisionProfile::Mixed => {
            let CpuPrecisionSlice::F32(target) = target else {
                return Err(OrchestratorError::InvalidPlan(
                    "mixed significance target is not f32",
                ));
            };
            let score_signals = signals
                .iter()
                .map(|signal| match signal {
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F32(values),
                        ..
                    } if values.len() == target.len() => Ok(values.as_slice()),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F32(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "mixed significance signal length does not match target",
                    )),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F64(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "mixed significance received f64 interaction storage",
                    )),
                })
                .collect::<OrchestratorResult<Vec<_>>>()?;
            let mut result = evaluate_precision_mixed(&score_signals, target, metrics, params);
            assign_precision_candidate_ids(&mut result, &candidate_ids);
            Ok(result)
        }
        PrecisionProfile::Fp64 => {
            let CpuPrecisionSlice::F64(target) = target else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 significance target is not f64",
                ));
            };
            let score_signals = signals
                .iter()
                .map(|signal| match signal {
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F32(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "fp64 significance received f32 interaction storage",
                    )),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F64(values),
                        ..
                    } if values.len() == target.len() => Ok(values.as_slice()),
                    PrecisionSignificanceSignal {
                        values: CpuPrecisionValues::F64(_),
                        ..
                    } => Err(OrchestratorError::InvalidPlan(
                        "fp64 significance signal length does not match target",
                    )),
                })
                .collect::<OrchestratorResult<Vec<_>>>()?;
            let mut result = evaluate_precision_f64(&score_signals, target, metrics, params);
            assign_precision_candidate_ids(&mut result, &candidate_ids);
            Ok(result)
        }
    }
}

fn assign_precision_candidate_ids(
    results: &mut [PrecisionCandidateSignificance],
    candidate_ids: &[u64],
) {
    for (result, &candidate_id) in results.iter_mut().zip(candidate_ids) {
        result.candidate_id = candidate_id;
    }
}

/// Evaluate a materialized time-series shortlist. `columns` is column-major,
/// with `rows` entries for each selected generated descriptor. Lags, windows,
/// source feature ids, and descriptor ordering stay outside the numeric values
/// and therefore remain their existing structural types.
pub fn evaluate_precision_time_series_columns(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    target: CpuPrecisionSlice<'_>,
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if rows == 0 {
        if !columns.is_empty() || !candidate_ids.is_empty() {
            return Err(OrchestratorError::InvalidPlan(
                "time-series significance cannot describe columns with zero rows",
            ));
        }
        return evaluate_precision_signals(profile, target, &[], metrics, params);
    }
    if !columns.len().is_multiple_of(rows) || candidate_ids.len() != columns.len() / rows {
        return Err(OrchestratorError::InvalidPlan(
            "time-series significance descriptors do not match generated columns",
        ));
    }
    // Reject an impossible profile/storage request before copying generated
    // columns into per-candidate significance buffers.
    match (profile, columns, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        )
        | (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        ) => {
            let signals = columns
                .chunks_exact(rows)
                .zip(candidate_ids)
                .map(|(values, &candidate_id)| PrecisionSignificanceSignal {
                    candidate_id,
                    values: CpuPrecisionValues::F32(values.to_vec()),
                })
                .collect::<Vec<_>>();
            evaluate_precision_signals(
                profile,
                CpuPrecisionSlice::F32(target),
                &signals,
                metrics,
                params,
            )
        }
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(columns),
            CpuPrecisionSlice::F64(target),
        ) => {
            let signals = columns
                .chunks_exact(rows)
                .zip(candidate_ids)
                .map(|(values, &candidate_id)| PrecisionSignificanceSignal {
                    candidate_id,
                    values: CpuPrecisionValues::F64(values.to_vec()),
                })
                .collect::<Vec<_>>();
            evaluate_precision_signals(
                profile,
                CpuPrecisionSlice::F64(target),
                &signals,
                metrics,
                params,
            )
        }
        _ => Err(OrchestratorError::InvalidPlan(
            "time-series significance storage dtype does not match the requested precision profile",
        )),
    }
}

/// Evaluate a decision-path family with target-aware maxT null construction.
///
/// Unlike continuous interactions and time-series transforms, decision paths
/// are discovered from the target/residual.  Each permutation therefore
/// rediscovers its own path family before its metric maximum is compared to the
/// observed paths. This prevents a target-dependent observed path from being
/// treated as a fixed null feature. Bootstrap mean/std remain aligned to each
/// observed candidate identity, as they are for the established shortlist API.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_precision_decision_path_family(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    target: CpuPrecisionSlice<'_>,
    observed_paths: &[PrecisionDecisionPath],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    discovery: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "decision-path significance shape exceeds host address space",
        ))?;
    if columns.len() != expected || target.len() != rows {
        return Err(OrchestratorError::InvalidPlan(
            "decision-path significance input does not match the declared shape",
        ));
    }
    if observed_paths.len() != candidate_ids.len() {
        return Err(OrchestratorError::InvalidPlan(
            "decision-path significance candidate ids do not match observed paths",
        ));
    }
    match (profile, columns, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        ) => evaluate_precision_decision_paths_f32(
            columns,
            rows,
            cols,
            target,
            observed_paths,
            candidate_ids,
            metrics,
            params,
            discovery,
        ),
        (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        ) => evaluate_precision_decision_paths_mixed(
            columns,
            rows,
            cols,
            target,
            observed_paths,
            candidate_ids,
            metrics,
            params,
            discovery,
        ),
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(columns),
            CpuPrecisionSlice::F64(target),
        ) => evaluate_precision_decision_paths_f64(
            columns,
            rows,
            cols,
            target,
            observed_paths,
            candidate_ids,
            metrics,
            params,
            discovery,
        ),
        _ => Err(OrchestratorError::InvalidPlan(
            "decision-path significance storage dtype does not match the requested precision profile",
        )),
    }
}

/// Discover observed paths and immediately run the complete profile-aware
/// decision-path significance family. The ordinal ids are structural emission
/// positions; integrations with external candidate ids can call
/// [`evaluate_precision_decision_path_family`] directly.
#[allow(clippy::too_many_arguments)]
pub fn discover_and_evaluate_precision_decision_paths(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    target: CpuPrecisionSlice<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    discovery: &DecisionPathParams,
) -> OrchestratorResult<(
    Vec<PrecisionDecisionPath>,
    Vec<PrecisionCandidateSignificance>,
)> {
    let paths = find_decision_paths_precision(profile, columns, rows, cols, target, discovery)?;
    let candidate_ids = (0..paths.len())
        .map(|candidate| candidate as u64)
        .collect::<Vec<_>>();
    let significance = evaluate_precision_decision_path_family(
        profile,
        columns,
        rows,
        cols,
        target,
        &paths,
        &candidate_ids,
        metrics,
        params,
        discovery,
    )?;
    Ok((paths, significance))
}

/// Evaluate selected rows from the complete, expanded decision-path family.
///
/// Generated decision-path execution first appends target-discovered path
/// memberships to the base feature prefix, then sends that expanded matrix
/// through ordinary continuous planning.  A static null plan over the observed
/// expansion is invalid: a permutation may discover a different path count,
/// ordering, and thresholds.  This entrypoint reconstructs the base plus paths
/// expansion for every permuted target and re-runs the continuous unary and
/// higher-order planning schedule before calculating maxT.
///
/// `selected_combos`, `selected_observed`, and `candidate_ids` must be aligned
/// to the surfaced observed rows. `candidate_ids` are copied through unchanged;
/// null candidates intentionally have no correspondence with observed paths or
/// path-column positions.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_precision_expanded_decision_path_family(
    profile: PrecisionProfile,
    source_features: CpuPrecisionSlice<'_>,
    rows: usize,
    source_cols: usize,
    target: CpuPrecisionSlice<'_>,
    observed_paths: &[PrecisionDecisionPath],
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    validate_expanded_decision_path_request(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        observed_paths,
        selected_combos,
        selected_observed,
        candidate_ids,
        metrics,
        search,
    )?;
    if selected_combos.is_empty() {
        return Ok(Vec::new());
    }

    match (profile, source_features, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(source_features),
            CpuPrecisionSlice::F32(target),
        ) => evaluate_expanded_decision_path_f32(
            PrecisionProfile::Fp32,
            source_features,
            rows,
            source_cols,
            target,
            observed_paths,
            selected_combos,
            selected_observed,
            candidate_ids,
            metrics,
            params,
            search,
        ),
        (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(source_features),
            CpuPrecisionSlice::F32(target),
        ) => evaluate_expanded_decision_path_mixed(
            source_features,
            rows,
            source_cols,
            target,
            observed_paths,
            selected_combos,
            selected_observed,
            candidate_ids,
            metrics,
            params,
            search,
        ),
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(source_features),
            CpuPrecisionSlice::F64(target),
        ) => evaluate_expanded_decision_path_f64(
            source_features,
            rows,
            source_cols,
            target,
            observed_paths,
            selected_combos,
            selected_observed,
            candidate_ids,
            metrics,
            params,
            search,
        ),
        _ => Err(OrchestratorError::InvalidPlan(
            "expanded decision-path significance storage dtype does not match the requested precision profile",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_expanded_decision_path_request(
    profile: PrecisionProfile,
    source_features: CpuPrecisionSlice<'_>,
    rows: usize,
    source_cols: usize,
    target: CpuPrecisionSlice<'_>,
    observed_paths: &[PrecisionDecisionPath],
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<()> {
    if rows == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path significance requires at least one row",
        ));
    }
    let expected = rows
        .checked_mul(source_cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "expanded decision-path source shape exceeds host address space",
        ))?;
    if source_features.len() != expected || target.len() != rows {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path source input does not match the declared shape",
        ));
    }
    let base_candidate_cols = usize::try_from(search.base_candidate_cols).map_err(|_| {
        OrchestratorError::InvalidPlan(
            "expanded decision-path base candidate count exceeds host address space",
        )
    })?;
    if base_candidate_cols == 0 || base_candidate_cols > source_cols {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path base candidate prefix is outside the source matrix",
        ));
    }
    if search.max_arity == 0 || search.max_combinations_per_arity == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path continuous planner requires non-zero arity and limit",
        ));
    }
    let mut seen_discovery_features = HashSet::with_capacity(search.discovery_features.len());
    if search.discovery_features.iter().any(|&feature| {
        feature as usize >= base_candidate_cols || !seen_discovery_features.insert(feature)
    }) {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path discovery feature is invalid or duplicated",
        ));
    }

    let observed_cols = base_candidate_cols
        .checked_add(observed_paths.len())
        .ok_or(OrchestratorError::InvalidPlan(
            "expanded decision-path observed column count overflows",
        ))?;
    if observed_cols > u32::MAX as usize {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path observed column count exceeds u32",
        ));
    }
    for path in observed_paths {
        if path.nodes.is_empty()
            || path.nodes.iter().any(|node| {
                node.feature as usize >= base_candidate_cols
                    || !precision_path_threshold_matches(profile, node.threshold)
            })
            || !precision_path_gain_matches(profile, path.gain)
        {
            return Err(OrchestratorError::InvalidPlan(
                "expanded decision-path observed path does not match the profile/base prefix",
            ));
        }
    }

    if selected_combos.len() != selected_observed.len()
        || selected_combos.len() != candidate_ids.len()
    {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path selected combos, observed metrics, and candidate ids differ in length",
        ));
    }
    if !selected_combos.is_empty() && metrics.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path selected rows require metrics",
        ));
    }
    let mut seen_candidate_ids = HashSet::with_capacity(candidate_ids.len());
    if candidate_ids
        .iter()
        .any(|&candidate_id| !seen_candidate_ids.insert(candidate_id))
    {
        return Err(OrchestratorError::InvalidPlan(
            "expanded decision-path selected candidate ids must be unique",
        ));
    }
    let unary_features = legacy_unary_feature_order(
        observed_cols as u32,
        search.max_combinations_per_arity,
        search.planning_seed_words,
    );
    let higher_enabled = expanded_decision_path_uses_higher_order(search, &unary_features);
    for (combo, observed) in selected_combos.iter().zip(selected_observed) {
        let mut combo_features = HashSet::with_capacity(combo.len());
        if combo.is_empty()
            || combo.len() > search.max_arity.min(observed_cols as u32) as usize
            || combo.iter().any(|&feature| {
                feature as usize >= observed_cols || !combo_features.insert(feature)
            })
        {
            return Err(OrchestratorError::InvalidPlan(
                "expanded decision-path selected combo is outside the observed continuous family",
            ));
        }
        if combo.len() == 1 && !unary_features.contains(&combo[0]) {
            return Err(OrchestratorError::InvalidPlan(
                "expanded decision-path selected unary is absent from the observed plan",
            ));
        }
        if combo.len() > 1 && !higher_enabled {
            return Err(OrchestratorError::InvalidPlan(
                "expanded decision-path selected higher-order combo is absent from the observed plan",
            ));
        }
        let observed_width_matches_profile = match (profile, observed) {
            (PrecisionProfile::Fp32, CpuPrecisionValues::F32(values)) => {
                values.len() == metrics.len()
            }
            (PrecisionProfile::Mixed | PrecisionProfile::Fp64, CpuPrecisionValues::F64(values)) => {
                values.len() == metrics.len()
            }
            _ => false,
        };
        if !observed_width_matches_profile {
            return Err(OrchestratorError::InvalidPlan(
                "expanded decision-path observed metric oracle does not match the profile result lane",
            ));
        }
    }

    match (profile, source_features, target) {
        (PrecisionProfile::Fp32, CpuPrecisionSlice::F32(_), CpuPrecisionSlice::F32(_))
        | (PrecisionProfile::Mixed, CpuPrecisionSlice::F32(_), CpuPrecisionSlice::F32(_))
        | (PrecisionProfile::Fp64, CpuPrecisionSlice::F64(_), CpuPrecisionSlice::F64(_)) => Ok(()),
        _ => Err(OrchestratorError::InvalidPlan(
            "expanded decision-path input dtype does not match the requested precision profile",
        )),
    }
}

fn precision_path_threshold_matches(profile: PrecisionProfile, value: CpuPrecisionScalar) -> bool {
    matches!(
        (profile, value),
        (
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed,
            CpuPrecisionScalar::F32(_)
        ) | (PrecisionProfile::Fp64, CpuPrecisionScalar::F64(_))
    )
}

fn precision_path_gain_matches(profile: PrecisionProfile, value: CpuPrecisionScalar) -> bool {
    matches!(
        (profile, value),
        (PrecisionProfile::Fp32, CpuPrecisionScalar::F32(_))
            | (
                PrecisionProfile::Mixed | PrecisionProfile::Fp64,
                CpuPrecisionScalar::F64(_)
            )
    )
}

fn expanded_decision_path_uses_higher_order(
    search: &ExpandedDecisionPathSearchSpec<'_>,
    unary_features: &[u32],
) -> bool {
    search.max_arity >= 2 && search.top_features_for_higher_arity >= 2 && unary_features.len() >= 2
}

#[allow(clippy::too_many_arguments)]
fn evaluate_expanded_decision_path_f32(
    profile: PrecisionProfile,
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    observed_paths: &[PrecisionDecisionPath],
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let observed_matrix = build_expanded_decision_path_matrix_f32(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        observed_paths,
        search,
    )?;
    let signals = precision_continuous_signals_f32(&observed_matrix, selected_combos)?;
    let observed = precision_observed_f32(selected_observed)?;
    finish_precision_f32(
        &signals,
        target,
        &observed,
        candidate_ids,
        metrics,
        params,
        |permuted| {
            let null_matrix = rediscover_expanded_decision_path_matrix_f32(
                profile,
                source_features,
                rows,
                source_cols,
                permuted,
                search,
            )?;
            expanded_decision_path_maxima_f32(&null_matrix, permuted, metrics, params, search)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_expanded_decision_path_mixed(
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    observed_paths: &[PrecisionDecisionPath],
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let observed_matrix = build_expanded_decision_path_matrix_f32(
        PrecisionProfile::Mixed,
        source_features,
        rows,
        source_cols,
        target,
        observed_paths,
        search,
    )?;
    let signals = precision_continuous_signals_f32(&observed_matrix, selected_combos)?;
    let observed = precision_observed_f64(selected_observed)?;
    finish_precision_mixed(
        &signals,
        target,
        &observed,
        candidate_ids,
        metrics,
        params,
        |permuted| {
            let null_matrix = rediscover_expanded_decision_path_matrix_f32(
                PrecisionProfile::Mixed,
                source_features,
                rows,
                source_cols,
                permuted,
                search,
            )?;
            expanded_decision_path_maxima_mixed(&null_matrix, permuted, metrics, params, search)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_expanded_decision_path_f64(
    source_features: &[f64],
    rows: usize,
    source_cols: usize,
    target: &[f64],
    observed_paths: &[PrecisionDecisionPath],
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let observed_matrix = build_expanded_decision_path_matrix_f64(
        source_features,
        rows,
        source_cols,
        target,
        observed_paths,
        search,
    )?;
    let signals = precision_continuous_signals_f64(&observed_matrix, selected_combos)?;
    let observed = precision_observed_f64(selected_observed)?;
    finish_precision_f64(
        &signals,
        target,
        &observed,
        candidate_ids,
        metrics,
        params,
        |permuted| {
            let null_matrix = rediscover_expanded_decision_path_matrix_f64(
                source_features,
                rows,
                source_cols,
                permuted,
                search,
            )?;
            expanded_decision_path_maxima_f64(&null_matrix, permuted, metrics, params, search)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn rediscover_expanded_decision_path_matrix_f32(
    profile: PrecisionProfile,
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<CpuPrecisionMatrix> {
    let paths = rediscover_expanded_decision_paths_f32(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        search,
    )?;
    build_expanded_decision_path_matrix_f32(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        &paths,
        search,
    )
}

#[allow(clippy::too_many_arguments)]
fn rediscover_expanded_decision_path_matrix_f64(
    source_features: &[f64],
    rows: usize,
    source_cols: usize,
    target: &[f64],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<CpuPrecisionMatrix> {
    let paths =
        rediscover_expanded_decision_paths_f64(source_features, rows, source_cols, target, search)?;
    build_expanded_decision_path_matrix_f64(
        source_features,
        rows,
        source_cols,
        target,
        &paths,
        search,
    )
}

#[allow(clippy::too_many_arguments)]
fn rediscover_expanded_decision_paths_f32(
    profile: PrecisionProfile,
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionDecisionPath>> {
    if search.discovery_features.is_empty() || search.discovery.max_paths == 0 {
        return Ok(Vec::new());
    }
    let mut discovery_columns = Vec::with_capacity(
        rows.checked_mul(search.discovery_features.len())
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path discovery selection exceeds host address space",
            ))?,
    );
    for &feature in search.discovery_features {
        let feature = feature as usize;
        for row in 0..rows {
            discovery_columns.push(source_features[row * source_cols + feature]);
        }
    }
    let mut paths = find_decision_paths_precision(
        profile,
        CpuPrecisionSlice::F32(&discovery_columns),
        rows,
        search.discovery_features.len(),
        CpuPrecisionSlice::F32(target),
        &search.discovery,
    )?;
    remap_expanded_decision_path_nodes(&mut paths, search.discovery_features)?;
    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
fn rediscover_expanded_decision_paths_f64(
    source_features: &[f64],
    rows: usize,
    source_cols: usize,
    target: &[f64],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionDecisionPath>> {
    if search.discovery_features.is_empty() || search.discovery.max_paths == 0 {
        return Ok(Vec::new());
    }
    let mut discovery_columns = Vec::with_capacity(
        rows.checked_mul(search.discovery_features.len())
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path discovery selection exceeds host address space",
            ))?,
    );
    for &feature in search.discovery_features {
        let feature = feature as usize;
        for row in 0..rows {
            discovery_columns.push(source_features[row * source_cols + feature]);
        }
    }
    let mut paths = find_decision_paths_precision(
        PrecisionProfile::Fp64,
        CpuPrecisionSlice::F64(&discovery_columns),
        rows,
        search.discovery_features.len(),
        CpuPrecisionSlice::F64(target),
        &search.discovery,
    )?;
    remap_expanded_decision_path_nodes(&mut paths, search.discovery_features)?;
    Ok(paths)
}

fn remap_expanded_decision_path_nodes(
    paths: &mut [PrecisionDecisionPath],
    discovery_features: &[u32],
) -> OrchestratorResult<()> {
    for path in paths.iter_mut() {
        for node in &mut path.nodes {
            node.feature = discovery_features
                .get(node.feature as usize)
                .copied()
                .ok_or(OrchestratorError::InvalidPlan(
                    "expanded decision-path discovery returned an out-of-range feature",
                ))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_expanded_decision_path_matrix_f32(
    profile: PrecisionProfile,
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    paths: &[PrecisionDecisionPath],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<CpuPrecisionMatrix> {
    let base_candidate_cols = search.base_candidate_cols as usize;
    let expanded_cols =
        base_candidate_cols
            .checked_add(paths.len())
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path column count overflows",
            ))?;
    let capacity = rows
        .checked_mul(expanded_cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "expanded decision-path matrix exceeds host address space",
        ))?;
    let mut base_columns = Vec::with_capacity(rows.checked_mul(base_candidate_cols).ok_or(
        OrchestratorError::InvalidPlan(
            "expanded decision-path base matrix exceeds host address space",
        ),
    )?);
    for feature in 0..base_candidate_cols {
        for row in 0..rows {
            base_columns.push(source_features[row * source_cols + feature]);
        }
    }
    let memberships = paths
        .iter()
        .map(|path| {
            let CpuPrecisionValues::F32(values) = path_membership_precision(
                profile,
                CpuPrecisionSlice::F32(&base_columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "f32 expanded decision-path membership received f64 storage",
                ));
            };
            Ok(values)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let mut expanded = Vec::with_capacity(capacity);
    for row in 0..rows {
        let source_start = row * source_cols;
        expanded
            .extend_from_slice(&source_features[source_start..source_start + base_candidate_cols]);
        for membership in &memberships {
            expanded.push(membership[row]);
        }
    }
    let rows = u64::try_from(rows).map_err(|_| {
        OrchestratorError::InvalidPlan("expanded decision-path row count exceeds u64")
    })?;
    let cols = u32::try_from(expanded_cols).map_err(|_| {
        OrchestratorError::InvalidPlan("expanded decision-path column count exceeds u32")
    })?;
    CpuPrecisionMatrix::from_row_major_f32(profile, rows, cols, expanded, target.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn build_expanded_decision_path_matrix_f64(
    source_features: &[f64],
    rows: usize,
    source_cols: usize,
    target: &[f64],
    paths: &[PrecisionDecisionPath],
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<CpuPrecisionMatrix> {
    let base_candidate_cols = search.base_candidate_cols as usize;
    let expanded_cols =
        base_candidate_cols
            .checked_add(paths.len())
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path column count overflows",
            ))?;
    let capacity = rows
        .checked_mul(expanded_cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "expanded decision-path matrix exceeds host address space",
        ))?;
    let mut base_columns = Vec::with_capacity(rows.checked_mul(base_candidate_cols).ok_or(
        OrchestratorError::InvalidPlan(
            "expanded decision-path base matrix exceeds host address space",
        ),
    )?);
    for feature in 0..base_candidate_cols {
        for row in 0..rows {
            base_columns.push(source_features[row * source_cols + feature]);
        }
    }
    let memberships = paths
        .iter()
        .map(|path| {
            let CpuPrecisionValues::F64(values) = path_membership_precision(
                PrecisionProfile::Fp64,
                CpuPrecisionSlice::F64(&base_columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 expanded decision-path membership received f32 storage",
                ));
            };
            Ok(values)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let mut expanded = Vec::with_capacity(capacity);
    for row in 0..rows {
        let source_start = row * source_cols;
        expanded
            .extend_from_slice(&source_features[source_start..source_start + base_candidate_cols]);
        for membership in &memberships {
            expanded.push(membership[row]);
        }
    }
    let rows = u64::try_from(rows).map_err(|_| {
        OrchestratorError::InvalidPlan("expanded decision-path row count exceeds u64")
    })?;
    let cols = u32::try_from(expanded_cols).map_err(|_| {
        OrchestratorError::InvalidPlan("expanded decision-path column count exceeds u32")
    })?;
    CpuPrecisionMatrix::from_row_major_f64(
        PrecisionProfile::Fp64,
        rows,
        cols,
        expanded,
        target.to_vec(),
    )
}

fn expanded_decision_path_maxima_f32(
    matrix: &CpuPrecisionMatrix,
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<f32>> {
    let unary_features = legacy_unary_feature_order(
        matrix.cols(),
        search.max_combinations_per_arity,
        search.planning_seed_words,
    );
    if expanded_decision_path_uses_higher_order(search, &unary_features) {
        let adaptive = AdaptiveSearchSpec {
            unary_features: &unary_features,
            candidate_feature_count: matrix.cols(),
            max_arity: search.max_arity,
            max_combinations_per_arity: search.max_combinations_per_arity,
            top_features_for_higher_arity: search.top_features_for_higher_arity,
            planning_seed_words: search.planning_seed_words,
        };
        return adaptive_precision_maxima_f32(matrix, target, metrics, params, &adaptive);
    }
    let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
    for feature in unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 expanded decision-path null has f64 resident columns",
            ))?;
        let scores = score_all_f32(PrecisionProfile::Fp32, signal, target, metrics, params);
        update_precision_max_f32(&mut maxima, &scores, metrics);
    }
    Ok(maxima)
}

fn expanded_decision_path_maxima_mixed(
    matrix: &CpuPrecisionMatrix,
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<f64>> {
    let unary_features = legacy_unary_feature_order(
        matrix.cols(),
        search.max_combinations_per_arity,
        search.planning_seed_words,
    );
    if expanded_decision_path_uses_higher_order(search, &unary_features) {
        let adaptive = AdaptiveSearchSpec {
            unary_features: &unary_features,
            candidate_feature_count: matrix.cols(),
            max_arity: search.max_arity,
            max_combinations_per_arity: search.max_combinations_per_arity,
            top_features_for_higher_arity: search.top_features_for_higher_arity,
            planning_seed_words: search.planning_seed_words,
        };
        return adaptive_precision_maxima_mixed(matrix, target, metrics, params, &adaptive);
    }
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    for feature in unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "mixed expanded decision-path null has f64 resident columns",
            ))?;
        let scores =
            score_all_f64_from_f32(PrecisionProfile::Mixed, signal, target, metrics, params);
        update_precision_max_f64(&mut maxima, &scores, metrics);
    }
    Ok(maxima)
}

fn expanded_decision_path_maxima_f64(
    matrix: &CpuPrecisionMatrix,
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &ExpandedDecisionPathSearchSpec<'_>,
) -> OrchestratorResult<Vec<f64>> {
    let unary_features = legacy_unary_feature_order(
        matrix.cols(),
        search.max_combinations_per_arity,
        search.planning_seed_words,
    );
    if expanded_decision_path_uses_higher_order(search, &unary_features) {
        let adaptive = AdaptiveSearchSpec {
            unary_features: &unary_features,
            candidate_feature_count: matrix.cols(),
            max_arity: search.max_arity,
            max_combinations_per_arity: search.max_combinations_per_arity,
            top_features_for_higher_arity: search.top_features_for_higher_arity,
            planning_seed_words: search.planning_seed_words,
        };
        return adaptive_precision_maxima_f64(matrix, target, metrics, params, &adaptive);
    }
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    for feature in unary_features {
        let signal = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 expanded decision-path null has f32 resident columns",
            ))?;
        let scores = score_all_f64(PrecisionProfile::Fp64, signal, target, metrics, params);
        update_precision_max_f64(&mut maxima, &scores, metrics);
    }
    Ok(maxima)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_precision_decision_paths_f32(
    columns: &[f32],
    rows: usize,
    cols: usize,
    target: &[f32],
    observed_paths: &[PrecisionDecisionPath],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    discovery: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let signals = observed_paths
        .iter()
        .map(|path| {
            let CpuPrecisionValues::F32(values) = path_membership_precision(
                PrecisionProfile::Fp32,
                CpuPrecisionSlice::F32(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp32 decision-path significance received f64 membership",
                ));
            };
            Ok(values)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let observed = signals
        .iter()
        .map(|signal| score_all_f32(PrecisionProfile::Fp32, signal, target, metrics, params))
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f32(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let null_paths = find_decision_paths_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(columns),
            rows,
            cols,
            CpuPrecisionSlice::F32(&permuted),
            discovery,
        )?;
        let mut maximums = vec![f32::NEG_INFINITY; metrics.len()];
        for path in &null_paths {
            let CpuPrecisionValues::F32(signal) = path_membership_precision(
                PrecisionProfile::Fp32,
                CpuPrecisionSlice::F32(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp32 decision-path null received f64 membership",
                ));
            };
            let scores = score_all_f32(PrecisionProfile::Fp32, &signal, &permuted, metrics, params);
            for (maximum, (&score, &metric)) in maximums.iter_mut().zip(scores.iter().zip(metrics))
            {
                *maximum = maximum.max(extremeness_f32(score, metric));
            }
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maximums[metric_index] >= extremeness_f32(observed_scores[metric_index], metric)
                {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f32(
                PrecisionProfile::Fp32,
                signal,
                target,
                metrics,
                params,
                candidate_ids[candidate],
            );
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F32(if params.permutation_tests == 0 {
                    vec![f32::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f32 / (params.permutation_tests + 1) as f32)
                        .collect()
                }),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            }
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_precision_decision_paths_mixed(
    columns: &[f32],
    rows: usize,
    cols: usize,
    target: &[f32],
    observed_paths: &[PrecisionDecisionPath],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    discovery: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let signals = observed_paths
        .iter()
        .map(|path| {
            let CpuPrecisionValues::F32(values) = path_membership_precision(
                PrecisionProfile::Mixed,
                CpuPrecisionSlice::F32(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "mixed decision-path significance received f64 membership",
                ));
            };
            Ok(values)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let observed = signals
        .iter()
        .map(|signal| {
            score_all_f64_from_f32(PrecisionProfile::Mixed, signal, target, metrics, params)
        })
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f32(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let null_paths = find_decision_paths_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(columns),
            rows,
            cols,
            CpuPrecisionSlice::F32(&permuted),
            discovery,
        )?;
        let mut maximums = vec![f64::NEG_INFINITY; metrics.len()];
        for path in &null_paths {
            let CpuPrecisionValues::F32(signal) = path_membership_precision(
                PrecisionProfile::Mixed,
                CpuPrecisionSlice::F32(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "mixed decision-path null received f64 membership",
                ));
            };
            let scores = score_all_f64_from_f32(
                PrecisionProfile::Mixed,
                &signal,
                &permuted,
                metrics,
                params,
            );
            for (maximum, (&score, &metric)) in maximums.iter_mut().zip(scores.iter().zip(metrics))
            {
                *maximum = maximum.max(extremeness_f64(score, metric));
            }
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maximums[metric_index] >= extremeness_f64(observed_scores[metric_index], metric)
                {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f64_from_f32(
                PrecisionProfile::Mixed,
                signal,
                target,
                metrics,
                params,
                candidate_ids[candidate],
            );
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_precision_decision_paths_f64(
    columns: &[f64],
    rows: usize,
    cols: usize,
    target: &[f64],
    observed_paths: &[PrecisionDecisionPath],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    discovery: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let signals = observed_paths
        .iter()
        .map(|path| {
            let CpuPrecisionValues::F64(values) = path_membership_precision(
                PrecisionProfile::Fp64,
                CpuPrecisionSlice::F64(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 decision-path significance received f32 membership",
                ));
            };
            Ok(values)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let observed = signals
        .iter()
        .map(|signal| score_all_f64(PrecisionProfile::Fp64, signal, target, metrics, params))
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f64(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let null_paths = find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(columns),
            rows,
            cols,
            CpuPrecisionSlice::F64(&permuted),
            discovery,
        )?;
        let mut maximums = vec![f64::NEG_INFINITY; metrics.len()];
        for path in &null_paths {
            let CpuPrecisionValues::F64(signal) = path_membership_precision(
                PrecisionProfile::Fp64,
                CpuPrecisionSlice::F64(columns),
                rows,
                &path.nodes,
            )?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 decision-path null received f32 membership",
                ));
            };
            let scores = score_all_f64(PrecisionProfile::Fp64, &signal, &permuted, metrics, params);
            for (maximum, (&score, &metric)) in maximums.iter_mut().zip(scores.iter().zip(metrics))
            {
                *maximum = maximum.max(extremeness_f64(score, metric));
            }
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maximums[metric_index] >= extremeness_f64(observed_scores[metric_index], metric)
                {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f64(
                PrecisionProfile::Fp64,
                signal,
                target,
                metrics,
                params,
                candidate_ids[candidate],
            );
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

fn evaluate_precision_f32(
    signals: &[&[f32]],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<PrecisionCandidateSignificance> {
    let observed = signals
        .iter()
        .map(|signal| score_all_f32(PrecisionProfile::Fp32, signal, target, metrics, params))
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f32(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let scores = signals
            .iter()
            .map(|signal| score_all_f32(PrecisionProfile::Fp32, signal, &permuted, metrics, params))
            .collect::<Vec<_>>();
        for (metric_index, &metric) in metrics.iter().enumerate() {
            let maximum = scores
                .iter()
                .map(|score| extremeness_f32(score[metric_index], metric))
                .fold(f32::NEG_INFINITY, f32::max);
            for candidate in 0..signals.len() {
                if maximum >= extremeness_f32(observed[candidate][metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f32(
                PrecisionProfile::Fp32,
                signal,
                target,
                metrics,
                params,
                candidate as u64,
            );
            let pvalues = if params.permutation_tests == 0 {
                vec![f32::NAN; metrics.len()]
            } else {
                exceedances[candidate]
                    .iter()
                    .map(|&count| (count + 1) as f32 / (params.permutation_tests + 1) as f32)
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: 0,
                pvalues: CpuPrecisionValues::F32(pvalues),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            }
        })
        .collect()
}

fn evaluate_precision_mixed(
    signals: &[&[f32]],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<PrecisionCandidateSignificance> {
    let observed = signals
        .iter()
        .map(|signal| {
            score_all_f64_from_f32(PrecisionProfile::Mixed, signal, target, metrics, params)
        })
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f32(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let scores = signals
            .iter()
            .map(|signal| {
                score_all_f64_from_f32(PrecisionProfile::Mixed, signal, &permuted, metrics, params)
            })
            .collect::<Vec<_>>();
        for (metric_index, &metric) in metrics.iter().enumerate() {
            let maximum = scores
                .iter()
                .map(|score| extremeness_f64(score[metric_index], metric))
                .fold(f64::NEG_INFINITY, f64::max);
            for candidate in 0..signals.len() {
                if maximum >= extremeness_f64(observed[candidate][metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f64_from_f32(
                PrecisionProfile::Mixed,
                signal,
                target,
                metrics,
                params,
                candidate as u64,
            );
            let pvalues = if params.permutation_tests == 0 {
                vec![f64::NAN; metrics.len()]
            } else {
                exceedances[candidate]
                    .iter()
                    .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: 0,
                pvalues: CpuPrecisionValues::F64(pvalues),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect()
}

fn evaluate_precision_f64(
    signals: &[&[f64]],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<PrecisionCandidateSignificance> {
    let observed = signals
        .iter()
        .map(|signal| score_all_f64(PrecisionProfile::Fp64, signal, target, metrics, params))
        .collect::<Vec<_>>();
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = shuffle_f64(
            target,
            mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
        );
        let scores = signals
            .iter()
            .map(|signal| score_all_f64(PrecisionProfile::Fp64, signal, &permuted, metrics, params))
            .collect::<Vec<_>>();
        for (metric_index, &metric) in metrics.iter().enumerate() {
            let maximum = scores
                .iter()
                .map(|score| extremeness_f64(score[metric_index], metric))
                .fold(f64::NEG_INFINITY, f64::max);
            for candidate in 0..signals.len() {
                if maximum >= extremeness_f64(observed[candidate][metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = bootstrap_f64(
                PrecisionProfile::Fp64,
                signal,
                target,
                metrics,
                params,
                candidate as u64,
            );
            let pvalues = if params.permutation_tests == 0 {
                vec![f64::NAN; metrics.len()]
            } else {
                exceedances[candidate]
                    .iter()
                    .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: 0,
                pvalues: CpuPrecisionValues::F64(pvalues),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect()
}

fn score_all_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f32> {
    metrics
        .iter()
        .map(|&metric| {
            let CpuPrecisionScalar::F32(value) = precision::score_precision_signal(
                profile,
                CpuPrecisionSlice::F32(signal),
                CpuPrecisionSlice::F32(target),
                metric,
                params.mi_bins,
                params.mi_approximate,
            )
            .expect("profile-specialized fp32 significance signal is valid") else {
                unreachable!("fp32 profile must return fp32 scores")
            };
            value
        })
        .collect()
}

fn score_all_f64_from_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f64> {
    metrics
        .iter()
        .map(|&metric| {
            let CpuPrecisionScalar::F64(value) = precision::score_precision_signal(
                profile,
                CpuPrecisionSlice::F32(signal),
                CpuPrecisionSlice::F32(target),
                metric,
                params.mi_bins,
                params.mi_approximate,
            )
            .expect("profile-specialized mixed significance signal is valid") else {
                unreachable!("mixed profile must return fp64 scores")
            };
            value
        })
        .collect()
}

fn score_all_f64(
    profile: PrecisionProfile,
    signal: &[f64],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f64> {
    metrics
        .iter()
        .map(|&metric| {
            let CpuPrecisionScalar::F64(value) = precision::score_precision_signal(
                profile,
                CpuPrecisionSlice::F64(signal),
                CpuPrecisionSlice::F64(target),
                metric,
                params.mi_bins,
                params.mi_approximate,
            )
            .expect("profile-specialized fp64 significance signal is valid") else {
                unreachable!("fp64 profile must return fp64 scores")
            };
            value
        })
        .collect()
}

fn bootstrap_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate: u64,
) -> (Vec<f32>, Vec<f32>) {
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return (vec![f32::NAN; metrics.len()], vec![f32::NAN; metrics.len()]);
    }
    let mut samples = vec![vec![0.0f32; repeats]; metrics.len()];
    for repeat in 0..repeats {
        let indices = bootstrap_indices(
            target.len(),
            mix_seed(params.random_seed, 0xB007_57A9 ^ candidate, repeat as u64),
        );
        let x = indices
            .iter()
            .map(|&index| signal[index])
            .collect::<Vec<_>>();
        let y = indices
            .iter()
            .map(|&index| target[index])
            .collect::<Vec<_>>();
        let score = score_all_f32(profile, &x, &y, metrics, params);
        for (metric_samples, &value) in samples.iter_mut().zip(&score) {
            metric_samples[repeat] = value;
        }
    }
    mean_std_f32(&samples)
}

fn bootstrap_f64_from_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate: u64,
) -> (Vec<f64>, Vec<f64>) {
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return (vec![f64::NAN; metrics.len()], vec![f64::NAN; metrics.len()]);
    }
    let mut samples = vec![vec![0.0f64; repeats]; metrics.len()];
    for repeat in 0..repeats {
        let indices = bootstrap_indices(
            target.len(),
            mix_seed(params.random_seed, 0xB007_57A9 ^ candidate, repeat as u64),
        );
        let x = indices
            .iter()
            .map(|&index| signal[index])
            .collect::<Vec<_>>();
        let y = indices
            .iter()
            .map(|&index| target[index])
            .collect::<Vec<_>>();
        let score = score_all_f64_from_f32(profile, &x, &y, metrics, params);
        for (metric_samples, &value) in samples.iter_mut().zip(&score) {
            metric_samples[repeat] = value;
        }
    }
    mean_std_f64(&samples)
}

fn bootstrap_f64(
    profile: PrecisionProfile,
    signal: &[f64],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate: u64,
) -> (Vec<f64>, Vec<f64>) {
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return (vec![f64::NAN; metrics.len()], vec![f64::NAN; metrics.len()]);
    }
    let mut samples = vec![vec![0.0f64; repeats]; metrics.len()];
    for repeat in 0..repeats {
        let indices = bootstrap_indices(
            target.len(),
            mix_seed(params.random_seed, 0xB007_57A9 ^ candidate, repeat as u64),
        );
        let x = indices
            .iter()
            .map(|&index| signal[index])
            .collect::<Vec<_>>();
        let y = indices
            .iter()
            .map(|&index| target[index])
            .collect::<Vec<_>>();
        let score = score_all_f64(profile, &x, &y, metrics, params);
        for (metric_samples, &value) in samples.iter_mut().zip(&score) {
            metric_samples[repeat] = value;
        }
    }
    mean_std_f64(&samples)
}

fn mean_std_f32(samples: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let means = samples
        .iter()
        .map(|values| values.iter().copied().sum::<f32>() / values.len() as f32)
        .collect::<Vec<_>>();
    let stds = samples
        .iter()
        .zip(&means)
        .map(|(values, &mean)| {
            (values
                .iter()
                .map(|&value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / values.len() as f32)
                .sqrt()
        })
        .collect();
    (means, stds)
}

fn mean_std_f64(samples: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let means = samples
        .iter()
        .map(|values| values.iter().copied().sum::<f64>() / values.len() as f64)
        .collect::<Vec<_>>();
    let stds = samples
        .iter()
        .zip(&means)
        .map(|(values, &mean)| {
            (values
                .iter()
                .map(|&value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / values.len() as f64)
                .sqrt()
        })
        .collect();
    (means, stds)
}

fn shuffle_f32(values: &[f32], seed: u64) -> Vec<f32> {
    let mut result = values.to_vec();
    shuffle_in_place(&mut result, seed);
    result
}

fn shuffle_f64(values: &[f64], seed: u64) -> Vec<f64> {
    let mut result = values.to_vec();
    shuffle_in_place(&mut result, seed);
    result
}

fn shuffle_in_place<T>(values: &mut [T], seed: u64) {
    let mut state = seed;
    for index in (1..values.len()).rev() {
        let swap = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
        values.swap(index, swap);
    }
}

fn extremeness_f32(value: f32, kernel: MetricKernel) -> f32 {
    if !value.is_finite() {
        return f32::NEG_INFINITY;
    }
    match kernel {
        MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
        MetricKernel::R2 | MetricKernel::MutualInfo => value,
    }
}

fn extremeness_f64(value: f64, kernel: MetricKernel) -> f64 {
    if !value.is_finite() {
        return f64::NEG_INFINITY;
    }
    match kernel {
        MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
        MetricKernel::R2 | MetricKernel::MutualInfo => value,
    }
}

/// Association strength of a metric value. Correlation-style metrics are
/// two-sided (magnitude); variance/MI are one-sided (raw, higher = stronger).
/// Non-finite values map to negative infinity so they never inflate a max.
fn extremeness(value: f32, kernel: MetricKernel) -> f32 {
    if !value.is_finite() {
        return f32::NEG_INFINITY;
    }
    match kernel {
        MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
        MetricKernel::R2 | MetricKernel::MutualInfo => value,
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive an independent, deterministic seed for stream `idx` within `domain`.
fn mix_seed(base: u64, domain: u64, idx: u64) -> u64 {
    let mut state =
        base ^ domain.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ idx.wrapping_mul(0xD1B5_4A32_D192_ED03);
    splitmix64(&mut state)
}

/// Fisher-Yates shuffle of a copy of `target`, seeded deterministically.
fn shuffled_target(target: &[f32], seed: u64) -> Vec<f32> {
    let mut y = target.to_vec();
    let mut state = seed;
    for i in (1..y.len()).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        y.swap(i, j);
    }
    y
}

/// Build the deterministic target permutation shared by host-orchestrated GPU
/// adaptive search and the native CPU maxT implementation.
pub fn permutation_target(target: &[f32], base_seed: u64, permutation_index: u32) -> Vec<f32> {
    let seed = mix_seed(base_seed, 0xA5A5_A5A5, u64::from(permutation_index));
    shuffled_target(target, seed)
}

/// Build a deterministic profile-typed target permutation.
///
/// Permutation schedule words and positions remain integer/control values; the
/// returned target preserves the resident profile dtype so fp64 never visits an
/// f32 shuffle buffer and mixed remains f32 until its reduction phase.
pub fn precision_permutation_target(
    profile: PrecisionProfile,
    target: CpuPrecisionSlice<'_>,
    base_seed: u64,
    permutation_index: u32,
) -> OrchestratorResult<CpuPrecisionValues> {
    match (profile, target) {
        (PrecisionProfile::Fp32, CpuPrecisionSlice::F32(target))
        | (PrecisionProfile::Mixed, CpuPrecisionSlice::F32(target)) => Ok(CpuPrecisionValues::F32(
            precision_permutation_target_f32(target, base_seed, permutation_index),
        )),
        (PrecisionProfile::Fp64, CpuPrecisionSlice::F64(target)) => Ok(CpuPrecisionValues::F64(
            precision_permutation_target_f64(target, base_seed, permutation_index),
        )),
        _ => Err(OrchestratorError::InvalidPlan(
            "precision permutation target storage dtype does not match the requested profile",
        )),
    }
}

fn precision_permutation_target_f32(
    target: &[f32],
    base_seed: u64,
    permutation_index: u32,
) -> Vec<f32> {
    shuffle_f32(
        target,
        mix_seed(base_seed, 0xA5A5_A5A5, u64::from(permutation_index)),
    )
}

fn precision_permutation_target_f64(
    target: &[f64],
    base_seed: u64,
    permutation_index: u32,
) -> Vec<f64> {
    shuffle_f64(
        target,
        mix_seed(base_seed, 0xA5A5_A5A5, u64::from(permutation_index)),
    )
}

/// `n` bootstrap row indices sampled with replacement, seeded deterministically.
fn bootstrap_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    (0..n)
        .map(|_| (splitmix64(&mut state) % n as u64) as usize)
        .collect()
}

/// The y-independent interaction signal for a combo: a single column for arity 1,
/// or the elementwise product of mean-centered columns for higher arity. Matches
/// `kernels::build_interaction_vector_into` so significance scores the same signal
/// the primary pass did.
fn interaction_signal_into(matrix: &CpuMatrix, combo: &[u32], out: &mut Vec<f32>) {
    let rows = matrix.rows() as usize;
    out.clear();
    out.resize(rows, 1.0);
    for &feature in combo {
        let col = matrix.column(feature as usize);
        let mean = matrix.column_mean(feature as usize);
        for (product, &value) in out.iter_mut().zip(col) {
            *product *= value - mean;
        }
    }
}

struct ResampledColumn {
    values: Vec<f32>,
    mean: f32,
}

const BOOTSTRAP_COLUMN_CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Gather each selected feature once per bootstrap repeat. Candidate scoring then
/// reuses these columns while preserving the legacy f64 mean order and each
/// combo's multiplication order.
fn gather_resampled_columns(
    matrix: &CpuMatrix,
    features: &[u32],
    indices: &[usize],
) -> Vec<ResampledColumn> {
    features
        .iter()
        .map(|&feature| {
            let source = matrix.column(feature as usize);
            let values = indices
                .iter()
                .map(|&index| source[index])
                .collect::<Vec<_>>();
            let mean = if values.is_empty() {
                0.0
            } else {
                (values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64) as f32
            };
            ResampledColumn { values, mean }
        })
        .collect()
}

fn resampled_signal_into(
    combo: &[u32],
    selected_features: &[u32],
    columns: &[ResampledColumn],
    out: &mut Vec<f32>,
) {
    let first_index = selected_features
        .binary_search(&combo[0])
        .expect("selected bootstrap feature was gathered");
    if combo.len() == 1 {
        out.clear();
        out.extend_from_slice(&columns[first_index].values);
        return;
    }

    out.clear();
    out.resize(columns[first_index].values.len(), 1.0);
    for &feature in combo {
        let index = selected_features
            .binary_search(&feature)
            .expect("selected bootstrap feature was gathered");
        let column = &columns[index];
        for (product, &value) in out.iter_mut().zip(&column.values) {
            *product *= value - column.mean;
        }
    }
}

fn resampled_signal_from_matrix_into(
    matrix: &CpuMatrix,
    combo: &[u32],
    indices: &[usize],
    out: &mut Vec<f32>,
) {
    if combo.len() == 1 {
        let column = matrix.column(combo[0] as usize);
        out.clear();
        out.extend(indices.iter().map(|&index| column[index]));
        return;
    }

    out.clear();
    out.resize(indices.len(), 1.0);
    let mut gathered = Vec::with_capacity(indices.len());
    for &feature in combo {
        let column = matrix.column(feature as usize);
        gathered.clear();
        gathered.extend(indices.iter().map(|&index| column[index]));
        let mean = if gathered.is_empty() {
            0.0
        } else {
            (gathered.iter().map(|&value| value as f64).sum::<f64>() / gathered.len() as f64) as f32
        };
        for (product, &value) in out.iter_mut().zip(&gathered) {
            *product *= value - mean;
        }
    }
}

fn score_signal_into(
    signal: &[f32],
    y: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.extend(metrics.iter().map(|metric| match metric {
        MetricKernel::Pearson => kernels::pearson(signal, y),
        MetricKernel::Spearman => kernels::spearman(signal, y),
        MetricKernel::MutualInfo => {
            if mi_approximate {
                kernels::mutual_info_fixed(signal, y, mi_bins)
            } else {
                kernels::mutual_info(signal, y, mi_bins)
            }
        }
        MetricKernel::R2 => simd::r2_score(signal, y),
    }));
}

#[cfg(test)]
fn score_signal(
    signal: &[f32],
    y: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(metrics.len());
    score_signal_into(signal, y, metrics, mi_bins, mi_approximate, &mut out);
    out
}

/// Population mean/std of a small sample (std = 0 for fewer than two points).
fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    if n < 2 {
        return (mean as f32, 0.0);
    }
    let var = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    (mean as f32, var.sqrt() as f32)
}

fn significance_mi_bins(rows: u64, params: &SignificanceParams) -> u32 {
    select_adaptive_mi_bins_for_backend(params.backend_kind, rows, params.mi_bins)
}

fn significance_uses_fixed_width_mi(params: &SignificanceParams) -> bool {
    params.mi_approximate || params.backend_kind != GAFIME_BACKEND_CPU
}

#[derive(Clone)]
enum NullFamily<'a> {
    Selected(&'a [Vec<u32>]),
    Plan(DescriptorBatchSource),
}

impl NullFamily<'_> {
    fn for_each_combo(&self, visit: impl FnMut(&[u32])) {
        self.for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, visit);
    }

    fn for_each_combo_with_batch_words(
        &self,
        max_descriptor_words: usize,
        mut visit: impl FnMut(&[u32]),
    ) {
        match self {
            Self::Selected(combos) => {
                for combo in *combos {
                    visit(combo);
                }
            }
            Self::Plan(source) => {
                let batches = source
                    .descriptor_batches(max_descriptor_words)
                    .expect("validated null-family plan produces descriptor batches");
                for batch in batches {
                    for combo in batch.combo_indices().chunks_exact(batch.arity() as usize) {
                        visit(combo);
                    }
                }
            }
        }
    }

    fn try_for_each_combo_with_batch_words(
        &self,
        max_descriptor_words: usize,
        mut visit: impl FnMut(&[u32]) -> OrchestratorResult<()>,
    ) -> OrchestratorResult<()> {
        match self {
            Self::Selected(combos) => {
                for combo in *combos {
                    visit(combo)?;
                }
            }
            Self::Plan(source) => {
                let batches = source.descriptor_batches(max_descriptor_words)?;
                for batch in batches {
                    for combo in batch.combo_indices().chunks_exact(batch.arity() as usize) {
                        visit(combo)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
enum PermutationFamily<'a> {
    Fixed(NullFamily<'a>),
    Adaptive(AdaptiveSearchSpec<'a>),
}

fn validate_adaptive_search(
    matrix: &CpuMatrix,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<()> {
    if search.unary_features.len() < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance search requires at least two unary features",
        ));
    }
    if search.max_arity < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance search requires higher-order candidates",
        ));
    }
    if search.max_combinations_per_arity == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance search requires a non-zero combination limit",
        ));
    }
    if search.top_features_for_higher_arity < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance search requires a shortlist of at least two features",
        ));
    }
    if search.candidate_feature_count > matrix.cols() {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance candidate feature count exceeds matrix features",
        ));
    }
    if search
        .unary_features
        .iter()
        .any(|&feature| feature >= search.candidate_feature_count)
    {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance unary feature exceeds candidate features",
        ));
    }
    let mut seen = HashSet::with_capacity(search.unary_features.len());
    if search
        .unary_features
        .iter()
        .any(|&feature| !seen.insert(feature))
    {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive significance unary features contain a duplicate",
        ));
    }
    Ok(())
}

fn screening_strength(scores: &[f32], metrics: &[MetricKernel]) -> f32 {
    let mut strength = None::<f32>;
    for (&value, &metric) in scores.iter().zip(metrics) {
        let candidate = match metric {
            MetricKernel::Pearson | MetricKernel::Spearman => value.abs(),
            MetricKernel::R2 | MetricKernel::MutualInfo => value,
        };
        strength = Some(strength.map_or(candidate, |current| current.max(candidate)));
    }
    strength.unwrap_or(0.0)
}

/// Visit the same prefix of lexicographic combinations emitted by the
/// continuous planner, without materializing a descriptor buffer per
/// permutation.
fn for_each_combination_limited(
    features: &[u32],
    arity: usize,
    limit: u64,
    mut visit: impl FnMut(&[u32]),
) {
    if arity == 0 || arity > features.len() || limit == 0 {
        return;
    }
    let mut positions = (0..arity).collect::<Vec<_>>();
    let mut descriptor = vec![0u32; arity];
    let mut generated = 0u64;
    loop {
        for (slot, &position) in descriptor.iter_mut().zip(&positions) {
            *slot = features[position];
        }
        visit(&descriptor);
        generated += 1;
        if generated >= limit {
            return;
        }

        let mut pivot = arity;
        loop {
            if pivot == 0 {
                return;
            }
            pivot -= 1;
            if positions[pivot] != pivot + features.len() - arity {
                break;
            }
        }
        positions[pivot] += 1;
        for index in pivot + 1..arity {
            positions[index] = positions[index - 1] + 1;
        }
    }
}

fn validate_null_family_plan(
    matrix: &CpuMatrix,
    plan: &CompiledPlan,
    backend_kind: BackendKind,
) -> OrchestratorResult<()> {
    plan.validate()?;
    if plan.protocol().n_samples != matrix.rows() || plan.protocol().n_features != matrix.cols() {
        return Err(OrchestratorError::InvalidPlan(
            "significance null family does not match the CPU matrix",
        ));
    }
    if plan.protocol().backend_kind != backend_kind {
        return Err(OrchestratorError::InvalidPlan(
            "significance null family backend does not match the observed backend",
        ));
    }
    if plan.planned_row_count() == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "significance null family has no candidates",
        ));
    }

    for chunk in plan.chunks() {
        if chunk.family != GAFIME_FAMILY_CONTINUOUS {
            return Err(OrchestratorError::InvalidPlan(
                "significance null family must be continuous",
            ));
        }
    }
    Ok(())
}

fn validate_selected_rows(
    matrix: &CpuMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[Vec<f32>],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
) -> OrchestratorResult<()> {
    if selected_combos.len() != selected_observed.len() {
        return Err(OrchestratorError::InvalidPlan(
            "selected significance combos and observed rows differ in length",
        ));
    }
    if selected_combos.is_empty() {
        return Ok(());
    }
    if metrics.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "selected significance rows require metrics",
        ));
    }
    for (combo, observed) in selected_combos.iter().zip(selected_observed) {
        if combo.is_empty() {
            return Err(OrchestratorError::InvalidPlan(
                "selected significance combo is empty",
            ));
        }
        if combo.iter().any(|&feature| feature >= matrix.cols()) {
            return Err(OrchestratorError::InvalidPlan(
                "selected significance combo exceeds matrix features",
            ));
        }
        if observed.len() != metrics.len() {
            return Err(OrchestratorError::InvalidPlan(
                "selected significance metric row has the wrong width",
            ));
        }
    }

    // Selected output is bounded, so this set stays bounded by report size while
    // the compact plan is streamed exactly once. No full-family combo copy is made.
    let mut missing: HashSet<&[u32]> = selected_combos.iter().map(Vec::as_slice).collect();
    null_family.for_each_combo(|combo| {
        missing.remove(combo);
    });
    if !missing.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "selected significance combo is absent from the null family",
        ));
    }
    Ok(())
}

fn validate_precision_null_family_plan(
    matrix: &CpuPrecisionMatrix,
    plan: &CompiledPlan,
    backend_kind: BackendKind,
) -> OrchestratorResult<()> {
    plan.validate()?;
    if plan.protocol().n_samples != matrix.rows() || plan.protocol().n_features != matrix.cols() {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance null family does not match the CPU matrix",
        ));
    }
    if plan.protocol().backend_kind != backend_kind {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance null family backend does not match the observed backend",
        ));
    }
    if plan.planned_row_count() == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance null family has no candidates",
        ));
    }
    if plan
        .chunks()
        .iter()
        .any(|chunk| chunk.family != GAFIME_FAMILY_CONTINUOUS)
    {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance null family must be continuous",
        ));
    }
    Ok(())
}

fn validate_precision_adaptive_search(
    matrix: &CpuPrecisionMatrix,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<()> {
    if search.unary_features.len() < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance requires at least two unary features",
        ));
    }
    if search.max_arity < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance requires higher-order candidates",
        ));
    }
    if search.max_combinations_per_arity == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance requires a non-zero combination limit",
        ));
    }
    if search.top_features_for_higher_arity < 2 {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance requires a shortlist of at least two features",
        ));
    }
    if search.candidate_feature_count > matrix.cols() {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance candidate feature count exceeds matrix features",
        ));
    }
    if search
        .unary_features
        .iter()
        .any(|&feature| feature >= search.candidate_feature_count)
    {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance unary feature exceeds candidate features",
        ));
    }
    let mut seen = HashSet::with_capacity(search.unary_features.len());
    if search
        .unary_features
        .iter()
        .any(|&feature| !seen.insert(feature))
    {
        return Err(OrchestratorError::InvalidPlan(
            "adaptive precision significance unary features contain a duplicate",
        ));
    }
    Ok(())
}

fn validate_precision_selected_oracles(
    matrix: &CpuPrecisionMatrix,
    profile: PrecisionProfile,
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
) -> OrchestratorResult<()> {
    if matrix.profile() != profile {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance profile does not match resident matrix identity",
        ));
    }
    if selected_combos.len() != selected_observed.len()
        || selected_combos.len() != candidate_ids.len()
    {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance combos, observed rows, and candidate ids differ in length",
        ));
    }
    if selected_combos.is_empty() {
        return Ok(());
    }
    if metrics.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance selected rows require metrics",
        ));
    }
    for (combo, observed) in selected_combos.iter().zip(selected_observed) {
        if combo.is_empty() {
            return Err(OrchestratorError::InvalidPlan(
                "precision significance selected combo is empty",
            ));
        }
        if combo.iter().any(|&feature| feature >= matrix.cols()) {
            return Err(OrchestratorError::InvalidPlan(
                "precision significance selected combo exceeds matrix features",
            ));
        }
        let valid_oracle = match (profile, observed) {
            (PrecisionProfile::Fp32, CpuPrecisionValues::F32(values)) => {
                values.len() == metrics.len()
            }
            (PrecisionProfile::Mixed | PrecisionProfile::Fp64, CpuPrecisionValues::F64(values)) => {
                values.len() == metrics.len()
            }
            _ => false,
        };
        if !valid_oracle {
            return Err(OrchestratorError::InvalidPlan(
                "precision significance observed oracle does not match the profile metric lane",
            ));
        }
    }
    let mut missing: HashSet<&[u32]> = selected_combos.iter().map(Vec::as_slice).collect();
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        missing.remove(combo);
        Ok(())
    })?;
    if !missing.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance selected combo is absent from the null family",
        ));
    }
    Ok(())
}

/// Profile-aware version of [`evaluate_with_null_family`].
///
/// The selected observed scores are typed oracles from the actual profile
/// execution (f32 only for fp32, f64 for mixed/fp64). Each permutation streams
/// the entire compiled continuous null family without converting profile scores
/// through the legacy f32 table.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_precision_with_null_family(
    matrix: &CpuPrecisionMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    null_family: &CompiledPlan,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let profile = matrix.profile();
    validate_precision_null_family_plan(matrix, null_family, params.backend_kind)?;
    let null_family = NullFamily::Plan(null_family.descriptor_batch_source()?);
    validate_precision_selected_oracles(
        matrix,
        profile,
        selected_combos,
        selected_observed,
        candidate_ids,
        &null_family,
        metrics,
    )?;
    match profile {
        PrecisionProfile::Fp32 => {
            let observed = precision_observed_f32(selected_observed)?;
            let signals = precision_continuous_signals_f32(matrix, selected_combos)?;
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "fp32 significance matrix has non-f32 target",
            ))?;
            finish_precision_f32(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| {
                    fixed_precision_maxima_f32(matrix, permuted, &null_family, metrics, params)
                },
            )
        }
        PrecisionProfile::Mixed => {
            let observed = precision_observed_f64(selected_observed)?;
            let signals = precision_continuous_signals_f32(matrix, selected_combos)?;
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "mixed significance matrix has non-f32 target",
            ))?;
            finish_precision_mixed(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| {
                    fixed_precision_maxima_mixed(matrix, permuted, &null_family, metrics, params)
                },
            )
        }
        PrecisionProfile::Fp64 => {
            let observed = precision_observed_f64(selected_observed)?;
            let signals = precision_continuous_signals_f64(matrix, selected_combos)?;
            let target = matrix.target_f64().ok_or(OrchestratorError::InvalidPlan(
                "fp64 significance matrix has non-f64 target",
            ))?;
            finish_precision_f64(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| {
                    fixed_precision_maxima_f64(matrix, permuted, &null_family, metrics, params)
                },
            )
        }
    }
}

/// Profile-aware version of [`evaluate_with_adaptive_search`].
///
/// Every permutation scores the profile-typed unary pool, ranks it in the
/// visible profile lane, rebuilds the higher-order feature schedule, and then
/// takes maxT maxima over that rebuilt family.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_precision_with_adaptive_search(
    matrix: &CpuPrecisionMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[CpuPrecisionValues],
    candidate_ids: &[u64],
    observed_family: &CompiledPlan,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    let profile = matrix.profile();
    validate_precision_null_family_plan(matrix, observed_family, params.backend_kind)?;
    let observed_family = NullFamily::Plan(observed_family.descriptor_batch_source()?);
    validate_precision_selected_oracles(
        matrix,
        profile,
        selected_combos,
        selected_observed,
        candidate_ids,
        &observed_family,
        metrics,
    )?;
    validate_precision_adaptive_search(matrix, search)?;
    match profile {
        PrecisionProfile::Fp32 => {
            let observed = precision_observed_f32(selected_observed)?;
            let signals = precision_continuous_signals_f32(matrix, selected_combos)?;
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "fp32 significance matrix has non-f32 target",
            ))?;
            finish_precision_f32(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| adaptive_precision_maxima_f32(matrix, permuted, metrics, params, search),
            )
        }
        PrecisionProfile::Mixed => {
            let observed = precision_observed_f64(selected_observed)?;
            let signals = precision_continuous_signals_f32(matrix, selected_combos)?;
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "mixed significance matrix has non-f32 target",
            ))?;
            finish_precision_mixed(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| {
                    adaptive_precision_maxima_mixed(matrix, permuted, metrics, params, search)
                },
            )
        }
        PrecisionProfile::Fp64 => {
            let observed = precision_observed_f64(selected_observed)?;
            let signals = precision_continuous_signals_f64(matrix, selected_combos)?;
            let target = matrix.target_f64().ok_or(OrchestratorError::InvalidPlan(
                "fp64 significance matrix has non-f64 target",
            ))?;
            finish_precision_f64(
                &signals,
                target,
                &observed,
                candidate_ids,
                metrics,
                params,
                |permuted| adaptive_precision_maxima_f64(matrix, permuted, metrics, params, search),
            )
        }
    }
}

fn precision_observed_f32(observed: &[CpuPrecisionValues]) -> OrchestratorResult<Vec<&[f32]>> {
    observed
        .iter()
        .map(|values| {
            values.as_f32().ok_or(OrchestratorError::InvalidPlan(
                "fp32 precision significance observed oracle is not f32",
            ))
        })
        .collect()
}

fn precision_observed_f64(observed: &[CpuPrecisionValues]) -> OrchestratorResult<Vec<&[f64]>> {
    observed
        .iter()
        .map(|values| {
            values.as_f64().ok_or(OrchestratorError::InvalidPlan(
                "mixed/fp64 precision significance observed oracle is not f64",
            ))
        })
        .collect()
}

fn precision_continuous_signals_f32(
    matrix: &CpuPrecisionMatrix,
    combos: &[Vec<u32>],
) -> OrchestratorResult<Vec<Vec<f32>>> {
    combos
        .iter()
        .map(|combo| {
            let CpuPrecisionValues::F32(values) =
                precision::materialize_precision_combo(matrix, combo)?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "f32 precision significance materialized an f64 interaction",
                ));
            };
            Ok(values)
        })
        .collect()
}

fn precision_continuous_signals_f64(
    matrix: &CpuPrecisionMatrix,
    combos: &[Vec<u32>],
) -> OrchestratorResult<Vec<Vec<f64>>> {
    combos
        .iter()
        .map(|combo| {
            let CpuPrecisionValues::F64(values) =
                precision::materialize_precision_combo(matrix, combo)?
            else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 precision significance materialized an f32 interaction",
                ));
            };
            Ok(values)
        })
        .collect()
}

fn finish_precision_f32(
    signals: &[Vec<f32>],
    target: &[f32],
    observed: &[&[f32]],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    mut maxima_for_permutation: impl FnMut(&[f32]) -> OrchestratorResult<Vec<f32>>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = precision_permutation_target_f32(target, params.random_seed, permutation);
        let maxima = maxima_for_permutation(&permuted)?;
        if maxima.len() != metrics.len() {
            return Err(OrchestratorError::InvalidPlan(
                "fp32 precision significance maxT width does not match metrics",
            ));
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maxima[metric_index] >= extremeness_f32(observed_scores[metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = if params.num_repeats > 1 {
                bootstrap_f32(
                    PrecisionProfile::Fp32,
                    signal,
                    target,
                    metrics,
                    params,
                    candidate_ids[candidate],
                )
            } else {
                (observed[candidate].to_vec(), vec![0.0f32; metrics.len()])
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F32(if params.permutation_tests == 0 {
                    vec![f32::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f32 / (params.permutation_tests + 1) as f32)
                        .collect()
                }),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            }
        })
        .collect())
}

fn finish_precision_mixed(
    signals: &[Vec<f32>],
    target: &[f32],
    observed: &[&[f64]],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    mut maxima_for_permutation: impl FnMut(&[f32]) -> OrchestratorResult<Vec<f64>>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = precision_permutation_target_f32(target, params.random_seed, permutation);
        let maxima = maxima_for_permutation(&permuted)?;
        if maxima.len() != metrics.len() {
            return Err(OrchestratorError::InvalidPlan(
                "mixed precision significance maxT width does not match metrics",
            ));
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maxima[metric_index] >= extremeness_f64(observed_scores[metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = if params.num_repeats > 1 {
                bootstrap_f64_from_f32(
                    PrecisionProfile::Mixed,
                    signal,
                    target,
                    metrics,
                    params,
                    candidate_ids[candidate],
                )
            } else {
                (observed[candidate].to_vec(), vec![0.0f64; metrics.len()])
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

fn finish_precision_f64(
    signals: &[Vec<f64>],
    target: &[f64],
    observed: &[&[f64]],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    mut maxima_for_permutation: impl FnMut(&[f64]) -> OrchestratorResult<Vec<f64>>,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let mut exceedances = vec![vec![0u32; metrics.len()]; signals.len()];
    for permutation in 0..params.permutation_tests {
        let permuted = precision_permutation_target_f64(target, params.random_seed, permutation);
        let maxima = maxima_for_permutation(&permuted)?;
        if maxima.len() != metrics.len() {
            return Err(OrchestratorError::InvalidPlan(
                "fp64 precision significance maxT width does not match metrics",
            ));
        }
        for (candidate, observed_scores) in observed.iter().enumerate() {
            for (metric_index, &metric) in metrics.iter().enumerate() {
                if maxima[metric_index] >= extremeness_f64(observed_scores[metric_index], metric) {
                    exceedances[candidate][metric_index] += 1;
                }
            }
        }
    }
    Ok(signals
        .iter()
        .enumerate()
        .map(|(candidate, signal)| {
            let (means, stds) = if params.num_repeats > 1 {
                bootstrap_f64(
                    PrecisionProfile::Fp64,
                    signal,
                    target,
                    metrics,
                    params,
                    candidate_ids[candidate],
                )
            } else {
                (observed[candidate].to_vec(), vec![0.0f64; metrics.len()])
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate]
                        .iter()
                        .map(|&count| (count + 1) as f64 / (params.permutation_tests + 1) as f64)
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

fn score_precision_combo_f32(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f32>> {
    let CpuPrecisionValues::F32(signal) = precision::materialize_precision_combo(matrix, combo)?
    else {
        return Err(OrchestratorError::InvalidPlan(
            "fp32 significance attempted to score an f64 interaction",
        ));
    };
    Ok(score_all_f32(
        PrecisionProfile::Fp32,
        &signal,
        target,
        metrics,
        params,
    ))
}

fn score_precision_combo_mixed(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f64>> {
    let CpuPrecisionValues::F32(signal) = precision::materialize_precision_combo(matrix, combo)?
    else {
        return Err(OrchestratorError::InvalidPlan(
            "mixed significance attempted to score an f64 interaction storage",
        ));
    };
    Ok(score_all_f64_from_f32(
        PrecisionProfile::Mixed,
        &signal,
        target,
        metrics,
        params,
    ))
}

fn score_precision_combo_f64(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f64>> {
    let CpuPrecisionValues::F64(signal) = precision::materialize_precision_combo(matrix, combo)?
    else {
        return Err(OrchestratorError::InvalidPlan(
            "fp64 significance attempted to score an f32 interaction",
        ));
    };
    Ok(score_all_f64(
        PrecisionProfile::Fp64,
        &signal,
        target,
        metrics,
        params,
    ))
}

fn update_precision_max_f32(maxima: &mut [f32], scores: &[f32], metrics: &[MetricKernel]) {
    for (maximum, (&score, &metric)) in maxima.iter_mut().zip(scores.iter().zip(metrics)) {
        *maximum = (*maximum).max(extremeness_f32(score, metric));
    }
}

fn update_precision_max_f64(maxima: &mut [f64], scores: &[f64], metrics: &[MetricKernel]) {
    for (maximum, (&score, &metric)) in maxima.iter_mut().zip(scores.iter().zip(metrics)) {
        *maximum = (*maximum).max(extremeness_f64(score, metric));
    }
}

fn fixed_precision_maxima_f32(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f32],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f32>> {
    let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores = score_precision_combo_f32(matrix, combo, shuffled, metrics, params)?;
        update_precision_max_f32(&mut maxima, &scores, metrics);
        Ok(())
    })?;
    Ok(maxima)
}

fn fixed_precision_maxima_mixed(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f32],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f64>> {
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores = score_precision_combo_mixed(matrix, combo, shuffled, metrics, params)?;
        update_precision_max_f64(&mut maxima, &scores, metrics);
        Ok(())
    })?;
    Ok(maxima)
}

fn fixed_precision_maxima_f64(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f64],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f64>> {
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores = score_precision_combo_f64(matrix, combo, shuffled, metrics, params)?;
        update_precision_max_f64(&mut maxima, &scores, metrics);
        Ok(())
    })?;
    Ok(maxima)
}

fn precision_screening_strength_f64(scores: &[f64], metrics: &[MetricKernel]) -> f64 {
    scores
        .iter()
        .zip(metrics)
        .map(|(&score, &metric)| extremeness_f64(score, metric))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn adaptive_precision_maxima_f32(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<Vec<f32>> {
    let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
    let mut unary_strengths = Vec::with_capacity(search.unary_features.len());
    for &feature in search.unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 adaptive significance received fp64 resident columns",
            ))?;
        let scores = score_all_f32(PrecisionProfile::Fp32, signal, shuffled, metrics, params);
        update_precision_max_f32(&mut maxima, &scores, metrics);
        unary_strengths.push((feature, screening_strength(&scores, metrics)));
    }
    if params.backend_kind == GAFIME_BACKEND_CPU {
        // Preserve established stable score ties on Core before the structural
        // scheduling shuffle selects the higher-order input order.
        unary_strengths.sort_by_key(|(feature, _)| *feature);
    }
    let higher_features = legacy_higher_feature_order(
        search.candidate_feature_count,
        search.max_combinations_per_arity,
        search.top_features_for_higher_arity,
        search.planning_seed_words,
        &unary_strengths,
    );
    let max_arity = search.max_arity.min(search.candidate_feature_count) as usize;
    for arity in 2..=max_arity {
        let mut failure = None;
        for_each_combination_limited(
            &higher_features,
            arity,
            search.max_combinations_per_arity,
            |combo| {
                if failure.is_some() {
                    return;
                }
                match score_precision_combo_f32(matrix, combo, shuffled, metrics, params) {
                    Ok(scores) => update_precision_max_f32(&mut maxima, &scores, metrics),
                    Err(error) => failure = Some(error),
                }
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
    }
    Ok(maxima)
}

fn adaptive_precision_maxima_mixed(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<Vec<f64>> {
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    let mut unary_strengths = Vec::with_capacity(search.unary_features.len());
    for &feature in search.unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "mixed adaptive significance received fp64 resident columns",
            ))?;
        let scores =
            score_all_f64_from_f32(PrecisionProfile::Mixed, signal, shuffled, metrics, params);
        update_precision_max_f64(&mut maxima, &scores, metrics);
        unary_strengths.push((feature, precision_screening_strength_f64(&scores, metrics)));
    }
    if params.backend_kind == GAFIME_BACKEND_CPU {
        unary_strengths.sort_by_key(|(feature, _)| *feature);
    }
    let higher_features = legacy_higher_feature_order_f64(
        search.candidate_feature_count,
        search.max_combinations_per_arity,
        search.top_features_for_higher_arity,
        search.planning_seed_words,
        &unary_strengths,
    );
    let max_arity = search.max_arity.min(search.candidate_feature_count) as usize;
    for arity in 2..=max_arity {
        let mut failure = None;
        for_each_combination_limited(
            &higher_features,
            arity,
            search.max_combinations_per_arity,
            |combo| {
                if failure.is_some() {
                    return;
                }
                match score_precision_combo_mixed(matrix, combo, shuffled, metrics, params) {
                    Ok(scores) => update_precision_max_f64(&mut maxima, &scores, metrics),
                    Err(error) => failure = Some(error),
                }
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
    }
    Ok(maxima)
}

fn adaptive_precision_maxima_f64(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<Vec<f64>> {
    let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
    let mut unary_strengths = Vec::with_capacity(search.unary_features.len());
    for &feature in search.unary_features {
        let signal = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 adaptive significance received f32 resident columns",
            ))?;
        let scores = score_all_f64(PrecisionProfile::Fp64, signal, shuffled, metrics, params);
        update_precision_max_f64(&mut maxima, &scores, metrics);
        unary_strengths.push((feature, precision_screening_strength_f64(&scores, metrics)));
    }
    if params.backend_kind == GAFIME_BACKEND_CPU {
        unary_strengths.sort_by_key(|(feature, _)| *feature);
    }
    let higher_features = legacy_higher_feature_order_f64(
        search.candidate_feature_count,
        search.max_combinations_per_arity,
        search.top_features_for_higher_arity,
        search.planning_seed_words,
        &unary_strengths,
    );
    let max_arity = search.max_arity.min(search.candidate_feature_count) as usize;
    for arity in 2..=max_arity {
        let mut failure = None;
        for_each_combination_limited(
            &higher_features,
            arity,
            search.max_combinations_per_arity,
            |combo| {
                if failure.is_some() {
                    return;
                }
                match score_precision_combo_f64(matrix, combo, shuffled, metrics, params) {
                    Ok(scores) => update_precision_max_f64(&mut maxima, &scores, metrics),
                    Err(error) => failure = Some(error),
                }
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
    }
    Ok(maxima)
}

/// Evaluate permutation p-values + bootstrap stability for a set of candidate
/// `combos` whose observed metric values are `observed` (aligned to `combos`,
/// each inner slice aligned to `metrics`). Returns one `CandidateSignificance`
/// per combo, in the same order. This compatibility entrypoint uses the selected
/// combos as the permutation family. Callers with a larger fixed family must use
/// [`evaluate_with_null_family`]; target-dependent screening must use
/// [`evaluate_with_adaptive_search`].
pub fn evaluate(
    matrix: &CpuMatrix,
    combos: &[Vec<u32>],
    observed: &[Vec<f32>],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<CandidateSignificance> {
    evaluate_impl(
        matrix,
        combos,
        observed,
        PermutationFamily::Fixed(NullFamily::Selected(combos)),
        metrics,
        params,
    )
}

/// Evaluate selected report rows while deriving each permutation's maxT null
/// statistic from every candidate in `null_family`. The plan stays in compact,
/// borrowed descriptor form: permutations stream its chunks and retain only one
/// maximum per metric. This path is valid only when candidate generation is
/// target-independent. Bootstrap stability remains selected-only.
pub fn evaluate_with_null_family(
    matrix: &CpuMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[Vec<f32>],
    null_family: &CompiledPlan,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<CandidateSignificance>> {
    validate_null_family_plan(matrix, null_family, params.backend_kind)?;
    let null_family = NullFamily::Plan(null_family.descriptor_batch_source()?);
    validate_selected_rows(
        matrix,
        selected_combos,
        selected_observed,
        &null_family,
        metrics,
    )?;
    Ok(evaluate_impl(
        matrix,
        selected_combos,
        selected_observed,
        PermutationFamily::Fixed(null_family),
        metrics,
        params,
    ))
}

/// Evaluate selected report rows against the same adaptive search that produced
/// the observed higher-order family. `observed_family` validates candidate
/// identity, while each permutation independently re-scores the unary pool and
/// rebuilds its own higher-order shortlist before taking maxT maxima.
pub fn evaluate_with_adaptive_search(
    matrix: &CpuMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[Vec<f32>],
    observed_family: &CompiledPlan,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    search: &AdaptiveSearchSpec<'_>,
) -> OrchestratorResult<Vec<CandidateSignificance>> {
    validate_null_family_plan(matrix, observed_family, params.backend_kind)?;
    let observed_family = NullFamily::Plan(observed_family.descriptor_batch_source()?);
    validate_selected_rows(
        matrix,
        selected_combos,
        selected_observed,
        &observed_family,
        metrics,
    )?;
    validate_adaptive_search(matrix, search)?;
    Ok(evaluate_impl(
        matrix,
        selected_combos,
        selected_observed,
        PermutationFamily::Adaptive(*search),
        metrics,
        params,
    ))
}

fn update_permutation_max(maxima: &mut [f32], scores: &[f32], metrics: &[MetricKernel]) {
    for (metric_index, &value) in scores.iter().enumerate() {
        let strength = extremeness(value, metrics[metric_index]);
        if strength > maxima[metric_index] {
            maxima[metric_index] = strength;
        }
    }
}

fn fixed_permutation_maxima(
    matrix: &CpuMatrix,
    shuffled: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    null_family: &NullFamily<'_>,
    max_descriptor_words: usize,
) -> Vec<f32> {
    let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
    let mut interaction_scratch = Vec::with_capacity(matrix.rows() as usize);
    let mut score_scratch = Vec::with_capacity(metrics.len());
    null_family.for_each_combo_with_batch_words(max_descriptor_words, |combo| {
        if combo.len() == 1 {
            score_signal_into(
                matrix.column(combo[0] as usize),
                shuffled,
                metrics,
                mi_bins,
                mi_approximate,
                &mut score_scratch,
            );
        } else {
            interaction_signal_into(matrix, combo, &mut interaction_scratch);
            score_signal_into(
                &interaction_scratch,
                shuffled,
                metrics,
                mi_bins,
                mi_approximate,
                &mut score_scratch,
            );
        }
        update_permutation_max(&mut maxima, &score_scratch, metrics);
    });
    maxima
}

fn adaptive_permutation_maxima(
    matrix: &CpuMatrix,
    shuffled: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    backend_kind: BackendKind,
    search: AdaptiveSearchSpec<'_>,
) -> Vec<f32> {
    let metric_count = metrics.len();
    let mut maxima = vec![f32::NEG_INFINITY; metric_count];
    let mut unary_strengths = Vec::with_capacity(search.unary_features.len());
    let mut score_scratch = Vec::with_capacity(metric_count);

    for &feature in search.unary_features {
        score_signal_into(
            matrix.column(feature as usize),
            shuffled,
            metrics,
            mi_bins,
            mi_approximate,
            &mut score_scratch,
        );
        update_permutation_max(&mut maxima, &score_scratch, metrics);
        unary_strengths.push((feature, screening_strength(&score_scratch, metrics)));
    }
    if backend_kind == GAFIME_BACKEND_CPU {
        // Match observed Core insertion order before the stable score sort.
        unary_strengths.sort_by_key(|(feature, _)| *feature);
    }
    let higher_features = legacy_higher_feature_order(
        search.candidate_feature_count,
        search.max_combinations_per_arity,
        search.top_features_for_higher_arity,
        search.planning_seed_words,
        &unary_strengths,
    );

    let max_arity = search.max_arity.min(search.candidate_feature_count) as usize;
    let mut interaction_scratch = Vec::with_capacity(matrix.rows() as usize);
    for arity in 2..=max_arity {
        for_each_combination_limited(
            &higher_features,
            arity,
            search.max_combinations_per_arity,
            |combo| {
                interaction_signal_into(matrix, combo, &mut interaction_scratch);
                score_signal_into(
                    &interaction_scratch,
                    shuffled,
                    metrics,
                    mi_bins,
                    mi_approximate,
                    &mut score_scratch,
                );
                update_permutation_max(&mut maxima, &score_scratch, metrics);
            },
        );
    }
    maxima
}

fn evaluate_impl(
    matrix: &CpuMatrix,
    combos: &[Vec<u32>],
    observed: &[Vec<f32>],
    permutation_family: PermutationFamily<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<CandidateSignificance> {
    let candidate_count = combos.len();
    let metric_count = metrics.len();
    if candidate_count == 0 || metric_count == 0 {
        return Vec::new();
    }
    let mi_bins = significance_mi_bins(matrix.rows(), params);
    let mi_approximate = significance_uses_fixed_width_mi(params);

    // Permutation pass (Westfall-Young maxT). Each permutation is independent ->
    // rayon-parallel, with a per-permutation seeded target shuffle. The null
    // statistic per metric is the MAX association across the full screened family;
    // counting each candidate's observed association against this max-null yields
    // family-wise multiplicity-corrected p-values. This is what stops screening
    // many candidates from manufacturing "significant" hits out of pure noise:
    // the winning candidate is judged against the distribution of the winner under
    // the null, not against its own marginal null.
    let permutations = params.permutation_tests as usize;
    let mut counts = vec![0u32; candidate_count * metric_count];
    if permutations > 0 {
        let target = matrix.target();
        let observed_ext: Vec<f32> = (0..candidate_count)
            .flat_map(|ci| (0..metric_count).map(move |mi| (ci, mi)))
            .map(|(ci, mi)| extremeness(observed[ci][mi], metrics[mi]))
            .collect();
        counts = (0..permutations)
            .into_par_iter()
            .map(|p| {
                let shuffled = permutation_target(target, params.random_seed, p as u32);
                let perm_max = match &permutation_family {
                    PermutationFamily::Fixed(null_family) => fixed_permutation_maxima(
                        matrix,
                        &shuffled,
                        metrics,
                        mi_bins,
                        mi_approximate,
                        null_family,
                        DEFAULT_DESCRIPTOR_BATCH_WORDS,
                    ),
                    PermutationFamily::Adaptive(search) => adaptive_permutation_maxima(
                        matrix,
                        &shuffled,
                        metrics,
                        mi_bins,
                        mi_approximate,
                        params.backend_kind,
                        *search,
                    ),
                };
                let mut local = vec![0u32; candidate_count * metric_count];
                for ci in 0..candidate_count {
                    for mi in 0..metric_count {
                        if perm_max[mi] >= observed_ext[ci * metric_count + mi] {
                            local[ci * metric_count + mi] += 1;
                        }
                    }
                }
                local
            })
            .reduce(
                || vec![0u32; candidate_count * metric_count],
                |mut acc, local| {
                    for (slot, value) in acc.iter_mut().zip(local) {
                        *slot += value;
                    }
                    acc
                },
            );
    }

    // Stability pass: each bootstrap resample is independent -> rayon-parallel.
    // Collect the metric grid per repeat, then fold into per-candidate mean/std.
    let repeats = params.num_repeats as usize;
    let rows = matrix.rows() as usize;
    let mut selected_features = combos
        .iter()
        .flat_map(|combo| combo.iter().copied())
        .collect::<Vec<_>>();
    selected_features.sort_unstable();
    selected_features.dedup();
    let parallel_repeats = repeats.min(rayon::current_num_threads()).max(1);
    let per_repeat_cache_budget = BOOTSTRAP_COLUMN_CACHE_BUDGET_BYTES / parallel_repeats;
    let column_cache_bytes = selected_features
        .len()
        .saturating_mul(rows)
        .saturating_mul(std::mem::size_of::<f32>());
    let cache_resampled_columns = column_cache_bytes <= per_repeat_cache_budget;
    let repeat_grids: Vec<Vec<f32>> = if repeats <= 1 {
        Vec::new()
    } else {
        (0..repeats)
            .into_par_iter()
            .map(|r| {
                let seed = mix_seed(params.random_seed, 0x5A5A_5A5A, r as u64);
                let indices = bootstrap_indices(rows, seed);
                let y = indices
                    .iter()
                    .map(|&index| matrix.target()[index])
                    .collect::<Vec<_>>();
                let columns = cache_resampled_columns
                    .then(|| gather_resampled_columns(matrix, &selected_features, &indices));
                let mut signal = Vec::with_capacity(rows);
                let mut scores = Vec::with_capacity(metric_count);
                let mut flat = vec![0.0f32; candidate_count * metric_count];
                for (ci, combo) in combos.iter().enumerate() {
                    if let Some(columns) = &columns {
                        resampled_signal_into(combo, &selected_features, columns, &mut signal);
                    } else {
                        resampled_signal_from_matrix_into(matrix, combo, &indices, &mut signal);
                    }
                    score_signal_into(&signal, &y, metrics, mi_bins, mi_approximate, &mut scores);
                    for (mi, &value) in scores.iter().enumerate() {
                        flat[ci * metric_count + mi] = value;
                    }
                }
                flat
            })
            .collect()
    };

    (0..candidate_count)
        .map(|ci| {
            let mut pvalues = vec![f32::NAN; metric_count];
            let mut means = vec![0.0f32; metric_count];
            let mut stds = vec![0.0f32; metric_count];
            for mi in 0..metric_count {
                if permutations > 0 {
                    let exceedances = counts[ci * metric_count + mi] as f32;
                    pvalues[mi] = (exceedances + 1.0) / (permutations as f32 + 1.0);
                }
                if repeats > 1 {
                    let samples: Vec<f32> = repeat_grids
                        .iter()
                        .map(|grid| grid[ci * metric_count + mi])
                        .collect();
                    let (mean, std) = mean_std(&samples);
                    means[mi] = mean;
                    stds[mi] = std;
                } else {
                    means[mi] = observed[ci][mi];
                }
            }
            CandidateSignificance {
                pvalues,
                means,
                stds,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_orchestrator::plan::combos::{
        build_continuous_plan, build_continuous_plan_for_feature_orders, ContinuousPlanRequest,
    };
    use gafime_types::{
        GafimeRankSpec, GAFIME_BACKEND_METAL, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    };

    fn pearson_r2() -> Vec<MetricKernel> {
        vec![MetricKernel::Pearson, MetricKernel::R2]
    }

    #[test]
    fn precision_significance_keeps_each_profile_public_dtype() {
        let params = SignificanceParams {
            permutation_tests: 3,
            num_repeats: 3,
            random_seed: 19,
            mi_bins: 2,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: true,
        };
        let combos = vec![vec![0]];
        let metrics = vec![MetricKernel::Pearson, MetricKernel::MutualInfo];
        let fp32 = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Fp32,
            5,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let mixed = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Mixed,
            5,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let fp32_values = evaluate_precision_shortlist(&fp32, &combos, &metrics, &params).unwrap();
        let mixed_values =
            evaluate_precision_shortlist(&mixed, &combos, &metrics, &params).unwrap();
        assert!(matches!(fp32_values[0].pvalues, CpuPrecisionValues::F32(_)));
        assert!(matches!(fp32_values[0].means, CpuPrecisionValues::F32(_)));
        assert!(matches!(fp32_values[0].stds, CpuPrecisionValues::F32(_)));
        assert!(matches!(
            mixed_values[0].pvalues,
            CpuPrecisionValues::F64(_)
        ));
        assert!(matches!(mixed_values[0].means, CpuPrecisionValues::F64(_)));
        assert!(matches!(mixed_values[0].stds, CpuPrecisionValues::F64(_)));
    }

    #[test]
    fn fp64_significance_preserves_non_f32_target_distinctions() {
        let base = 1.0f64;
        let values = (0..6)
            .map(|offset| f64::from_bits(base.to_bits() + offset * 4096))
            .collect::<Vec<_>>();
        let target = (0..6)
            .map(|offset| f64::from_bits(base.to_bits() + offset * 8192))
            .collect::<Vec<_>>();
        assert!(values
            .windows(2)
            .all(|pair| pair[0] as f32 == pair[1] as f32));
        let matrix =
            CpuPrecisionMatrix::from_row_major_f64(PrecisionProfile::Fp64, 6, 1, values, target)
                .unwrap();
        let params = SignificanceParams {
            permutation_tests: 2,
            num_repeats: 2,
            random_seed: 7,
            mi_bins: 2,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let values = evaluate_precision_shortlist(
            &matrix,
            &[vec![0]],
            &[MetricKernel::Pearson, MetricKernel::Spearman],
            &params,
        )
        .unwrap();
        let CpuPrecisionValues::F64(means) = &values[0].means else {
            panic!("fp64 significance must return f64 means")
        };
        assert!(means.iter().all(|value| value.is_finite()));
    }

    fn typed_significance_params() -> SignificanceParams {
        SignificanceParams {
            permutation_tests: 3,
            num_repeats: 3,
            random_seed: 0x51A7,
            mi_bins: 2,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: true,
        }
    }

    #[test]
    fn time_series_significance_keeps_candidate_ids_and_profile_result_lanes() {
        let source = [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0];
        let (generated, descriptors) = crate::time_series::time_series_columns_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&source),
            source.len(),
            1,
            &[1],
            &[],
            false,
        )
        .unwrap();
        let CpuPrecisionValues::F32(generated) = generated else {
            panic!("mixed time-series storage must remain f32")
        };
        let candidate_ids = (0..descriptors.len())
            .map(|index| 40 + index as u64)
            .collect::<Vec<_>>();
        let target = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let results = evaluate_precision_time_series_columns(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&generated),
            source.len(),
            CpuPrecisionSlice::F32(&target),
            &candidate_ids,
            &[MetricKernel::Pearson, MetricKernel::Spearman],
            &typed_significance_params(),
        )
        .unwrap();
        assert_eq!(results.len(), descriptors.len());
        assert_eq!(
            results
                .iter()
                .map(|result| result.candidate_id)
                .collect::<Vec<_>>(),
            candidate_ids
        );
        assert!(results.iter().all(|result| {
            matches!(result.pvalues, CpuPrecisionValues::F64(_))
                && matches!(result.means, CpuPrecisionValues::F64(_))
                && matches!(result.stds, CpuPrecisionValues::F64(_))
        }));
    }

    #[test]
    fn fp64_decision_path_maxt_rediscovery_preserves_non_f32_values_and_public_f64() {
        let base = 1.0f64;
        let columns = [
            base,
            f64::from_bits(base.to_bits() + 8),
            f64::from_bits(base.to_bits() + 16),
            f64::from_bits(base.to_bits() + 24),
        ];
        assert!(columns
            .windows(2)
            .all(|pair| (pair[0] as f32).to_bits() == (pair[1] as f32).to_bits()));
        let target = [0.0f64, 0.0, 1.0, 1.0];
        let discovery = DecisionPathParams {
            max_depth: 1,
            rounds: 1,
            max_paths: 4,
            max_bins: 0,
            min_leaf: 1,
            learning_rate: 1.0,
        };
        let (paths, results) = discover_and_evaluate_precision_decision_paths(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&columns),
            4,
            1,
            CpuPrecisionSlice::F64(&target),
            &[MetricKernel::Pearson, MetricKernel::R2],
            &typed_significance_params(),
            &discovery,
        )
        .unwrap();
        assert!(!paths.is_empty());
        assert_eq!(results.len(), paths.len());
        assert!(results.iter().all(|result| {
            matches!(result.pvalues, CpuPrecisionValues::F64(_))
                && matches!(result.means, CpuPrecisionValues::F64(_))
                && matches!(result.stds, CpuPrecisionValues::F64(_))
        }));
        assert!(paths.iter().all(|path| {
            matches!(path.gain, CpuPrecisionScalar::F64(_))
                && path
                    .nodes
                    .iter()
                    .all(|node| matches!(node.threshold, CpuPrecisionScalar::F64(_)))
        }));
    }

    #[test]
    fn typed_significance_rejects_fp64_f32_signals_before_resampling() {
        let error = evaluate_precision_signals(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F32(&[0.0, 1.0]),
            &[PrecisionSignificanceSignal {
                candidate_id: 9,
                values: CpuPrecisionValues::F32(vec![0.0, 1.0]),
            }],
            &[MetricKernel::Pearson],
            &typed_significance_params(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            OrchestratorError::InvalidPlan("fp64 significance target is not f64")
        );
    }

    fn precision_plan(rows: u64, cols: u32, max_arity: u32) -> CompiledPlan {
        build_continuous_plan(ContinuousPlanRequest {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CPU,
            n_samples: rows,
            n_features: cols,
            max_arity,
            max_combinations_per_arity: 16,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            mi_bins: 2,
            rank: GafimeRankSpec::default(),
        })
        .unwrap()
    }

    #[test]
    fn typed_null_family_preserves_fp32_oracles_and_candidate_identity() {
        let matrix = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Fp32,
            6,
            3,
            vec![
                0.0, 4.0, 1.0, 1.0, 3.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 0.0, 4.0, 0.0, 1.0, 5.0,
                -1.0, 0.0,
            ],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let combos = vec![vec![0], vec![0, 1]];
        let metrics = vec![MetricKernel::Pearson, MetricKernel::R2];
        let observed = combos
            .iter()
            .map(|combo| {
                precision::score_precision_continuous_combo(&matrix, combo, &metrics, 2, true)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let params = typed_significance_params();
        let output = evaluate_precision_with_null_family(
            &matrix,
            &combos,
            &observed,
            &[91, 92],
            &precision_plan(6, 3, 2),
            &metrics,
            &params,
        )
        .unwrap();
        assert_eq!(
            output
                .iter()
                .map(|result| result.candidate_id)
                .collect::<Vec<_>>(),
            vec![91, 92]
        );
        assert!(output.iter().all(|result| {
            matches!(result.pvalues, CpuPrecisionValues::F32(_))
                && matches!(result.means, CpuPrecisionValues::F32(_))
                && matches!(result.stds, CpuPrecisionValues::F32(_))
        }));
    }

    #[test]
    fn typed_adaptive_significance_rescreens_mixed_in_f64_result_lane() {
        let matrix = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Mixed,
            6,
            3,
            vec![
                0.0, 4.0, 1.0, 1.0, 3.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 0.0, 4.0, 0.0, 1.0, 5.0,
                -1.0, 0.0,
            ],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let combos = vec![vec![0], vec![0, 1]];
        let metrics = vec![MetricKernel::Pearson, MetricKernel::R2];
        let observed = combos
            .iter()
            .map(|combo| {
                precision::score_precision_continuous_combo(&matrix, combo, &metrics, 2, true)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let search = AdaptiveSearchSpec {
            unary_features: &[0, 1, 2],
            candidate_feature_count: 3,
            max_arity: 2,
            max_combinations_per_arity: 3,
            top_features_for_higher_arity: 2,
            planning_seed_words: &[17],
        };
        let output = evaluate_precision_with_adaptive_search(
            &matrix,
            &combos,
            &observed,
            &[101, 102],
            &precision_plan(6, 3, 2),
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        assert_eq!(
            output
                .iter()
                .map(|result| result.candidate_id)
                .collect::<Vec<_>>(),
            vec![101, 102]
        );
        assert!(output.iter().all(|result| {
            matches!(result.pvalues, CpuPrecisionValues::F64(_))
                && matches!(result.means, CpuPrecisionValues::F64(_))
                && matches!(result.stds, CpuPrecisionValues::F64(_))
        }));
    }

    #[test]
    fn fp64_adaptive_oracle_and_permutation_never_visit_f32() {
        let base = 1.0f64;
        let feature = [
            base,
            f64::from_bits(base.to_bits() + 8),
            f64::from_bits(base.to_bits() + 16),
            f64::from_bits(base.to_bits() + 24),
            f64::from_bits(base.to_bits() + 32),
            f64::from_bits(base.to_bits() + 40),
        ];
        assert!(feature
            .windows(2)
            .all(|pair| (pair[0] as f32).to_bits() == (pair[1] as f32).to_bits()));
        let mut features = Vec::new();
        for (index, &value) in feature.iter().enumerate() {
            features.push(value);
            features.push(index as f64);
        }
        let target = (0..feature.len())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let matrix = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            feature.len() as u64,
            2,
            features,
            target,
        )
        .unwrap();
        let metrics = vec![MetricKernel::Pearson, MetricKernel::Spearman];
        let observed =
            vec![
                precision::score_precision_continuous_combo(&matrix, &[0], &metrics, 2, false)
                    .unwrap(),
            ];
        let search = AdaptiveSearchSpec {
            unary_features: &[0, 1],
            candidate_feature_count: 2,
            max_arity: 2,
            max_combinations_per_arity: 1,
            top_features_for_higher_arity: 2,
            planning_seed_words: &[23],
        };
        let output = evaluate_precision_with_adaptive_search(
            &matrix,
            &[vec![0]],
            &observed,
            &[77],
            &precision_plan(feature.len() as u64, 2, 2),
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        let CpuPrecisionValues::F64(means) = &output[0].means else {
            panic!("fp64 adaptive significance must expose f64 means")
        };
        assert!(means.iter().all(|value| value.is_finite()));
        let permuted = precision_permutation_target(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(matrix.target_f64().unwrap()),
            99,
            1,
        )
        .unwrap();
        let CpuPrecisionValues::F64(permuted) = permuted else {
            panic!("fp64 permutation target must remain f64")
        };
        let mut expected = matrix
            .target_f64()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let mut actual = permuted
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    fn expanded_decision_path_search() -> ExpandedDecisionPathSearchSpec<'static> {
        ExpandedDecisionPathSearchSpec {
            base_candidate_cols: 3,
            discovery_features: &[0, 1],
            discovery: DecisionPathParams {
                max_depth: 1,
                rounds: 1,
                max_paths: 2,
                max_bins: 0,
                min_leaf: 1,
                learning_rate: 1.0,
            },
            max_arity: 5,
            max_combinations_per_arity: 32,
            top_features_for_higher_arity: 5,
            planning_seed_words: &[0xD15C_0A7A],
        }
    }

    fn expanded_f32_paths() -> Vec<PrecisionDecisionPath> {
        use crate::decision_path::{PrecisionPathNode, SplitSign};

        vec![
            PrecisionDecisionPath {
                nodes: vec![PrecisionPathNode {
                    feature: 0,
                    threshold: CpuPrecisionScalar::F32(4.5),
                    sign: SplitSign::Le,
                }],
                gain: CpuPrecisionScalar::F32(1.0),
                support: 4,
                round: 0,
            },
            PrecisionDecisionPath {
                nodes: vec![PrecisionPathNode {
                    feature: 1,
                    threshold: CpuPrecisionScalar::F32(4.5),
                    sign: SplitSign::Gt,
                }],
                gain: CpuPrecisionScalar::F32(1.0),
                support: 4,
                round: 0,
            },
        ]
    }

    fn expanded_f64_paths() -> Vec<PrecisionDecisionPath> {
        use crate::decision_path::{PrecisionPathNode, SplitSign};

        let base = 1.0f64;
        vec![
            PrecisionDecisionPath {
                nodes: vec![PrecisionPathNode {
                    feature: 0,
                    threshold: CpuPrecisionScalar::F64(f64::from_bits(base.to_bits() + 24)),
                    sign: SplitSign::Le,
                }],
                gain: CpuPrecisionScalar::F64(1.0),
                support: 4,
                round: 0,
            },
            PrecisionDecisionPath {
                nodes: vec![PrecisionPathNode {
                    feature: 1,
                    threshold: CpuPrecisionScalar::F64(4.5),
                    sign: SplitSign::Gt,
                }],
                gain: CpuPrecisionScalar::F64(1.0),
                support: 4,
                round: 0,
            },
        ]
    }

    fn expanded_f32_source() -> Vec<f32> {
        vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0, 6.0, 7.0,
            6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0,
        ]
    }

    #[test]
    fn expanded_decision_path_maxt_enumerates_full_arity_five_family() {
        let source = expanded_f32_source();
        let paths = expanded_f32_paths();
        let search = expanded_decision_path_search();
        let combo = vec![0, 1, 2, 3, 4];
        let preliminary = build_expanded_decision_path_matrix_f32(
            PrecisionProfile::Fp32,
            &source,
            8,
            3,
            &[0.0; 8],
            &paths,
            &search,
        )
        .unwrap();
        let CpuPrecisionValues::F32(target) =
            precision::materialize_precision_combo(&preliminary, &combo).unwrap()
        else {
            panic!("fp32 expanded interaction must remain f32")
        };
        let matrix = build_expanded_decision_path_matrix_f32(
            PrecisionProfile::Fp32,
            &source,
            8,
            3,
            &target,
            &paths,
            &search,
        )
        .unwrap();
        let metrics = [MetricKernel::Pearson];
        let unary_peak = (0..matrix.cols())
            .map(|feature| {
                score_precision_combo_f32(
                    &matrix,
                    &[feature],
                    &target,
                    &metrics,
                    &typed_significance_params(),
                )
                .unwrap()[0]
                    .abs()
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let maximum = expanded_decision_path_maxima_f32(
            &matrix,
            &target,
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        assert!(
            unary_peak < 0.999,
            "fixture must require the arity-five signal"
        );
        assert!(
            maximum[0] > 0.999,
            "full maxT family must score the arity-five base-plus-path interaction"
        );
    }

    #[test]
    fn expanded_decision_path_significance_preserves_profile_lanes_and_ids() {
        let source = expanded_f32_source();
        let paths = expanded_f32_paths();
        let search = expanded_decision_path_search();
        let combo = vec![0, 1, 2, 3, 4];
        let metrics = [MetricKernel::Pearson, MetricKernel::Spearman];

        let fp32_matrix = build_expanded_decision_path_matrix_f32(
            PrecisionProfile::Fp32,
            &source,
            8,
            3,
            &[0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0],
            &paths,
            &search,
        )
        .unwrap();
        let fp32_observed =
            precision::score_precision_continuous_combo(&fp32_matrix, &combo, &metrics, 2, true)
                .unwrap();
        let fp32 = evaluate_precision_expanded_decision_path_family(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&source),
            8,
            3,
            CpuPrecisionSlice::F32(&[0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0]),
            &paths,
            std::slice::from_ref(&combo),
            std::slice::from_ref(&fp32_observed),
            &[701],
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        assert_eq!(fp32[0].candidate_id, 701);
        assert!(matches!(fp32[0].pvalues, CpuPrecisionValues::F32(_)));

        let mixed_matrix = build_expanded_decision_path_matrix_f32(
            PrecisionProfile::Mixed,
            &source,
            8,
            3,
            &[0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0],
            &paths,
            &search,
        )
        .unwrap();
        let mixed_paths = paths
            .iter()
            .cloned()
            .map(|mut path| {
                path.gain = CpuPrecisionScalar::F64(1.0);
                path
            })
            .collect::<Vec<_>>();
        let mixed_observed =
            precision::score_precision_continuous_combo(&mixed_matrix, &combo, &metrics, 2, true)
                .unwrap();
        let mixed = evaluate_precision_expanded_decision_path_family(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&source),
            8,
            3,
            CpuPrecisionSlice::F32(&[0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0]),
            &mixed_paths,
            std::slice::from_ref(&combo),
            std::slice::from_ref(&mixed_observed),
            &[702],
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        assert_eq!(mixed[0].candidate_id, 702);
        assert!(matches!(mixed[0].pvalues, CpuPrecisionValues::F64(_)));
        assert!(matches!(mixed[0].means, CpuPrecisionValues::F64(_)));

        let base = 1.0f64;
        let mut fp64_source = Vec::with_capacity(24);
        for row in 0..8 {
            fp64_source.push(f64::from_bits(base.to_bits() + row * 8));
            fp64_source.push(row as f64 + 1.0);
            fp64_source.push((row * row + 1) as f64);
        }
        assert!(fp64_source
            .chunks_exact(3)
            .map(|row| row[0] as f32)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|pair| pair[0].to_bits() == pair[1].to_bits()));
        let fp64_paths = expanded_f64_paths();
        let fp64_target = [0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0];
        let fp64_matrix = build_expanded_decision_path_matrix_f64(
            &fp64_source,
            8,
            3,
            &fp64_target,
            &fp64_paths,
            &search,
        )
        .unwrap();
        assert_eq!(
            fp64_matrix.column_f64(0).unwrap()[3].to_bits(),
            fp64_source[3 * 3].to_bits(),
            "the expanded fp64 source must not visit f32 storage"
        );
        let fp64_observed =
            precision::score_precision_continuous_combo(&fp64_matrix, &combo, &metrics, 2, false)
                .unwrap();
        let fp64 = evaluate_precision_expanded_decision_path_family(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&fp64_source),
            8,
            3,
            CpuPrecisionSlice::F64(&fp64_target),
            &fp64_paths,
            std::slice::from_ref(&combo),
            std::slice::from_ref(&fp64_observed),
            &[703],
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();
        assert_eq!(fp64[0].candidate_id, 703);
        assert!(matches!(fp64[0].pvalues, CpuPrecisionValues::F64(_)));
        assert!(matches!(fp64[0].stds, CpuPrecisionValues::F64(_)));
    }

    #[test]
    fn expanded_decision_path_significance_preserves_planner_combo_order() {
        let source = expanded_f32_source();
        let paths = expanded_f32_paths();
        let search = expanded_decision_path_search();
        // Screened planner output is ordered by the selected feature ranking,
        // not by numeric feature id. Significance must preserve that identity
        // instead of requiring a canonicalized ascending tuple.
        let combo = vec![4, 2];
        let target = [0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0];
        let matrix = build_expanded_decision_path_matrix_f32(
            PrecisionProfile::Fp32,
            &source,
            8,
            3,
            &target,
            &paths,
            &search,
        )
        .unwrap();
        let metrics = [MetricKernel::Pearson];
        let observed =
            precision::score_precision_continuous_combo(&matrix, &combo, &metrics, 2, true)
                .unwrap();

        let evaluated = evaluate_precision_expanded_decision_path_family(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&source),
            8,
            3,
            CpuPrecisionSlice::F32(&target),
            &paths,
            std::slice::from_ref(&combo),
            std::slice::from_ref(&observed),
            &[704],
            &metrics,
            &typed_significance_params(),
            &search,
        )
        .unwrap();

        assert_eq!(evaluated[0].candidate_id, 704);
    }

    #[test]
    fn splitmix_shuffle_is_deterministic() {
        let y = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(shuffled_target(&y, 42), shuffled_target(&y, 42));
    }

    #[test]
    fn adaptive_combination_stream_matches_the_planner_prefix() {
        let higher_features = vec![4u32, 0, 5, 2];
        let plan = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                precision: PrecisionProfile::Fp32,
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 32,
                n_features: 6,
                max_arity: 3,
                max_combinations_per_arity: 4,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec::default(),
            },
            &[],
            &higher_features,
            false,
        )
        .unwrap();

        for chunk in plan.chunks() {
            let arity = chunk.arity as usize;
            let start = chunk.descriptor_offset as usize;
            let end = start + chunk.combo_count as usize * arity;
            let mut streamed = Vec::new();
            for_each_combination_limited(&higher_features, arity, chunk.combo_count, |combo| {
                streamed.extend_from_slice(combo)
            });
            assert_eq!(streamed, plan.combo_indices()[start..end]);
        }
    }

    #[test]
    fn hundred_million_empty_selection_returns_without_materializing_descriptors() {
        let higher_features = (0..20_000).collect::<Vec<_>>();
        let plan = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                precision: PrecisionProfile::Fp32,
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 2,
                n_features: 20_000,
                max_arity: 2,
                max_combinations_per_arity: 100_000_000,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec {
                    top_k: 1,
                    primary_metric: GAFIME_METRIC_PEARSON,
                    descending: 1,
                    include_ties: 0,
                    reserved: [0; 4],
                },
            },
            &[],
            &higher_features,
            false,
        )
        .unwrap();
        let matrix =
            CpuMatrix::from_row_major(2, 20_000, vec![0.0; 40_000], vec![0.0, 0.0]).unwrap();
        let params = SignificanceParams {
            permutation_tests: 1,
            num_repeats: 1,
            random_seed: 7,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };

        let significance =
            evaluate_with_null_family(&matrix, &[], &[], &plan, &[MetricKernel::Pearson], &params)
                .unwrap();

        assert!(significance.is_empty());
        assert_eq!(plan.planned_row_count(), 100_000_000);
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn generated_null_family_maxt_scans_every_descriptor_batch() {
        let cols = 1_025u32;
        let target = vec![1.0f32, -1.0, 1.0, -1.0];
        let shuffled = permutation_target(&target, 7, 0);
        let positive = shuffled
            .iter()
            .position(|&value| value > 0.0)
            .expect("test target contains a positive value");
        let negative = shuffled
            .iter()
            .position(|&value| value < 0.0)
            .expect("test target contains a negative value");
        let mut left = vec![-1.0f32; target.len()];
        left[positive] = 1.0;
        left[negative] = 1.0;
        let right = shuffled
            .iter()
            .zip(&left)
            .map(|(&value, &sign)| value * sign)
            .collect::<Vec<_>>();
        assert_eq!(left.iter().sum::<f32>(), 0.0);
        assert_eq!(right.iter().sum::<f32>(), 0.0);

        let mut features = vec![0.0f32; target.len() * cols as usize];
        for row in 0..target.len() {
            features[row * cols as usize + cols as usize - 2] = left[row];
            features[row * cols as usize + cols as usize - 1] = right[row];
        }
        let matrix =
            CpuMatrix::from_row_major(target.len() as u64, cols, features, target).unwrap();
        let higher_features = (0..cols).collect::<Vec<_>>();
        let plan = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                precision: PrecisionProfile::Fp32,
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: matrix.rows(),
                n_features: cols,
                max_arity: 2,
                max_combinations_per_arity: u64::MAX,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec {
                    top_k: 1,
                    primary_metric: GAFIME_METRIC_PEARSON,
                    descending: 1,
                    include_ties: 0,
                    reserved: [0; 4],
                },
            },
            &[],
            &higher_features,
            false,
        )
        .unwrap();
        assert!(plan.uses_generated_descriptors());
        assert_eq!(plan.materialized_descriptor_words(), 0);

        let source = plan.descriptor_batch_source().unwrap();
        let batches = source
            .descriptor_batches(DEFAULT_DESCRIPTOR_BATCH_WORDS)
            .unwrap()
            .map(|batch| (batch.logical_row_offset(), batch.combo_count()))
            .collect::<Vec<_>>();
        assert_eq!(batches, vec![(0, 524_288), (524_288, 512)]);

        let maxima = fixed_permutation_maxima(
            &matrix,
            &shuffled,
            &[MetricKernel::Pearson],
            96,
            false,
            &NullFamily::Plan(plan.descriptor_batch_source().unwrap()),
            DEFAULT_DESCRIPTOR_BATCH_WORDS,
        );
        assert!(maxima[0] > 0.999_999);
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn cached_bootstrap_columns_match_bounded_fallback_bitwise() {
        let matrix = CpuMatrix::from_row_major(
            5,
            3,
            vec![
                1.0, 10.0, -2.0, 3.0, 8.0, -1.0, 2.0, 7.0, 4.0, 9.0, 6.0, 3.0, 5.0, 4.0, 0.0,
            ],
            vec![0.0; 5],
        )
        .unwrap();
        let indices = vec![4usize, 1, 4, 0, 2];
        let selected_features = vec![0u32, 1, 2];
        let columns = gather_resampled_columns(&matrix, &selected_features, &indices);

        for combo in [vec![0u32], vec![2], vec![0, 2], vec![2, 1, 0]] {
            let mut cached = Vec::new();
            let mut fallback = Vec::new();
            resampled_signal_into(&combo, &selected_features, &columns, &mut cached);
            resampled_signal_from_matrix_into(&matrix, &combo, &indices, &mut fallback);
            assert_eq!(
                cached
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                fallback
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn strong_linear_signal_has_low_pvalue() {
        // y == x0 exactly -> pearson/r2 == 1.0; permutations should almost never
        // reach that, so the p-value is near the (1)/(P+1) floor.
        let n = 64usize;
        let mut features = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f32;
            features.push(v); // single feature, arity 1
            target.push(v);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, 1, features, target).unwrap();
        let metrics = pearson_r2();
        let combos = vec![vec![0u32]];
        let observed = vec![score_signal(
            matrix.column(0),
            matrix.target(),
            &metrics,
            96,
            false,
        )];

        let params = SignificanceParams {
            permutation_tests: 200,
            num_repeats: 5,
            random_seed: 7,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let out = evaluate(&matrix, &combos, &observed, &metrics, &params);
        assert_eq!(out.len(), 1);
        // pearson p-value at/near the permutation floor
        assert!(out[0].pvalues[0] <= 0.02, "pearson p={}", out[0].pvalues[0]);
        // a perfect signal is highly stable across resamples
        assert!(out[0].stds[0] < 0.05, "pearson std={}", out[0].stds[0]);
        // stability mean recovers ~1.0
        assert!((out[0].means[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn pure_noise_has_high_pvalue() {
        // target uncorrelated with the feature -> observed ~0, permutations reach
        // it easily -> large p-value (not significant).
        let n = 96usize;
        let mut features = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            features.push((i % 7) as f32);
            target.push(((i * 13 + 5) % 11) as f32);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, 1, features, target).unwrap();
        let metrics = pearson_r2();
        let combos = vec![vec![0u32]];
        let observed = vec![score_signal(
            matrix.column(0),
            matrix.target(),
            &metrics,
            96,
            false,
        )];

        let params = SignificanceParams {
            permutation_tests: 200,
            num_repeats: 5,
            random_seed: 11,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let out = evaluate(&matrix, &combos, &observed, &metrics, &params);
        assert!(
            out[0].pvalues[0] > 0.05,
            "noise pearson p should be non-significant, got {}",
            out[0].pvalues[0]
        );
    }

    #[test]
    fn zero_permutations_yields_nan_pvalues_but_still_stability() {
        let n = 32usize;
        let features: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let target: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
        let matrix = CpuMatrix::from_row_major(n as u64, 1, features, target).unwrap();
        let metrics = vec![MetricKernel::Pearson];
        let combos = vec![vec![0u32]];
        let observed = vec![score_signal(
            matrix.column(0),
            matrix.target(),
            &metrics,
            96,
            false,
        )];
        let params = SignificanceParams {
            permutation_tests: 0,
            num_repeats: 4,
            random_seed: 1,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let out = evaluate(&matrix, &combos, &observed, &metrics, &params);
        assert!(out[0].pvalues[0].is_nan());
        assert!(out[0].stds[0].is_finite());
    }

    #[test]
    fn static_compiled_family_matches_the_selected_family_path() {
        let n = 48usize;
        let cols = 3u32;
        let mut features = Vec::with_capacity(n * cols as usize);
        let mut target = Vec::with_capacity(n);
        for row in 0..n {
            features.extend_from_slice(&[
                row as f32,
                ((row * 7 + 3) % 19) as f32,
                ((row * 11 + 5) % 23) as f32,
            ]);
            target.push(((row * 13 + 1) % 29) as f32);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, cols, features, target).unwrap();
        let metrics = vec![MetricKernel::Pearson];
        let combos = (0..cols).map(|feature| vec![feature]).collect::<Vec<_>>();
        let observed = combos
            .iter()
            .map(|combo| {
                score_signal(
                    matrix.column(combo[0] as usize),
                    matrix.target(),
                    &metrics,
                    96,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let plan = build_continuous_plan(ContinuousPlanRequest {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CPU,
            n_samples: n as u64,
            n_features: cols,
            max_arity: 1,
            max_combinations_per_arity: u64::MAX,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            mi_bins: 96,
            rank: GafimeRankSpec::default(),
        })
        .unwrap();
        let params = SignificanceParams {
            permutation_tests: 31,
            num_repeats: 3,
            random_seed: 19,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };

        assert_eq!(
            evaluate_with_null_family(&matrix, &combos, &observed, &plan, &metrics, &params)
                .unwrap(),
            evaluate(&matrix, &combos, &observed, &metrics, &params)
        );
    }

    fn uniform_stream(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| (splitmix64(&mut state) >> 11) as f32 / (1u64 << 53) as f32)
            .collect()
    }

    #[test]
    fn maxt_controls_false_positives_across_many_noise_candidates() {
        // 8 independent noise features vs an independent noise target. Family-wise
        // maxT must give even the best-looking candidate a non-tiny p-value, i.e.
        // searching many candidates must not manufacture a "signal".
        let n = 128usize;
        let cols = 8u32;
        let draws = uniform_stream(0x1234_5678, n * (cols as usize + 1));
        let mut features = Vec::with_capacity(n * cols as usize);
        let mut target = Vec::with_capacity(n);
        for row in 0..n {
            let base = row * (cols as usize + 1);
            for c in 0..cols as usize {
                features.push(draws[base + c]);
            }
            target.push(draws[base + cols as usize]);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, cols, features, target).unwrap();
        let metrics = vec![MetricKernel::Pearson];
        let combos: Vec<Vec<u32>> = (0..cols).map(|c| vec![c]).collect();
        let observed: Vec<Vec<f32>> = combos
            .iter()
            .map(|c| {
                score_signal(
                    matrix.column(c[0] as usize),
                    matrix.target(),
                    &metrics,
                    96,
                    false,
                )
            })
            .collect();
        let params = SignificanceParams {
            permutation_tests: 200,
            num_repeats: 3,
            random_seed: 3,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let out = evaluate(&matrix, &combos, &observed, &metrics, &params);
        let best_p = out
            .iter()
            .map(|c| c.pvalues[0])
            .fold(f32::INFINITY, f32::min);
        assert!(
            best_p > 0.05,
            "maxT should not flag pure noise, best p={best_p}"
        );
    }

    #[test]
    fn maxt_detects_a_real_signal_among_noise() {
        // Feature 0 drives the target; five others are noise. Family-wise maxT
        // must still call feature 0 significant.
        let n = 128usize;
        let cols = 6u32;
        let draws = uniform_stream(0x0000_BEEF, n * (cols as usize + 1));
        let mut features = Vec::with_capacity(n * cols as usize);
        let mut target = Vec::with_capacity(n);
        for row in 0..n {
            let base = row * (cols as usize + 1);
            let f0 = draws[base];
            for c in 0..cols as usize {
                features.push(draws[base + c]);
            }
            target.push(3.0 * f0 + 0.01 * draws[base + cols as usize]);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, cols, features, target).unwrap();
        let metrics = vec![MetricKernel::Pearson];
        let combos: Vec<Vec<u32>> = (0..cols).map(|c| vec![c]).collect();
        let observed: Vec<Vec<f32>> = combos
            .iter()
            .map(|c| {
                score_signal(
                    matrix.column(c[0] as usize),
                    matrix.target(),
                    &metrics,
                    96,
                    false,
                )
            })
            .collect();
        let params = SignificanceParams {
            permutation_tests: 200,
            num_repeats: 3,
            random_seed: 5,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };
        let out = evaluate(&matrix, &combos, &observed, &metrics, &params);
        assert!(
            out[0].pvalues[0] <= 0.05,
            "real signal should survive family-wise correction, p={}",
            out[0].pvalues[0]
        );
    }

    #[test]
    fn hidden_null_family_candidates_change_selected_row_pvalue() {
        let n = 128usize;
        let cols = 8u32;
        let draws = uniform_stream(0xC0DE_CAFE, n * (cols as usize + 1));
        let mut features = Vec::with_capacity(n * cols as usize);
        let mut target = Vec::with_capacity(n);
        for row in 0..n {
            let base = row * (cols as usize + 1);
            features.extend_from_slice(&draws[base..base + cols as usize]);
            target.push(draws[base + cols as usize]);
        }
        let matrix = CpuMatrix::from_row_major(n as u64, cols, features, target).unwrap();
        let plan = build_continuous_plan(ContinuousPlanRequest {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CPU,
            n_samples: n as u64,
            n_features: cols,
            max_arity: 2,
            max_combinations_per_arity: u64::MAX,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            mi_bins: 96,
            rank: GafimeRankSpec::default(),
        })
        .unwrap();

        // Model a bounded report by selecting only the strongest observed row.
        // Every other unary/pairwise plan row remains hidden but must still
        // contribute to each permutation's family maximum.
        let family = NullFamily::Plan(plan.descriptor_batch_source().unwrap());
        let mut selected_combo = Vec::new();
        let mut selected_value = 0.0f32;
        let mut selected_strength = f32::NEG_INFINITY;
        let mut interaction_scratch = Vec::new();
        family.for_each_combo(|combo| {
            let value = if combo.len() == 1 {
                kernels::pearson(matrix.column(combo[0] as usize), matrix.target())
            } else {
                interaction_signal_into(&matrix, combo, &mut interaction_scratch);
                kernels::pearson(&interaction_scratch, matrix.target())
            };
            if value.abs() > selected_strength {
                selected_combo = combo.to_vec();
                selected_value = value;
                selected_strength = value.abs();
            }
        });

        let metrics = vec![MetricKernel::Pearson];
        let selected_combos = vec![selected_combo];
        let selected_observed = vec![vec![selected_value]];
        let params = SignificanceParams {
            permutation_tests: 255,
            num_repeats: 3,
            random_seed: 0x5151,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: false,
        };

        let selected_only = evaluate(
            &matrix,
            &selected_combos,
            &selected_observed,
            &metrics,
            &params,
        );
        let full_family = evaluate_with_null_family(
            &matrix,
            &selected_combos,
            &selected_observed,
            &plan,
            &metrics,
            &params,
        )
        .unwrap();

        assert!(
            full_family[0].pvalues[0] > selected_only[0].pvalues[0],
            "hidden candidates must raise maxT p: selected-only={}, full-family={}",
            selected_only[0].pvalues[0],
            full_family[0].pvalues[0]
        );
        assert_eq!(full_family[0].means, selected_only[0].means);
        assert_eq!(full_family[0].stds, selected_only[0].stds);
    }

    #[test]
    fn fixed_mi_significance_uses_the_observed_adaptive_template() {
        let n = 1_152usize;
        let features = (0..n)
            .map(|index| index as f32 / (n - 1) as f32)
            .collect::<Vec<_>>();
        let target = features
            .iter()
            .map(|&value| if value > 0.55 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let matrix =
            CpuMatrix::from_row_major(n as u64, 1, features.clone(), target.clone()).unwrap();
        let metrics = vec![MetricKernel::MutualInfo];
        let combos = vec![vec![0u32]];
        let observed = vec![vec![kernels::mutual_info_fixed(&features, &target, 12)]];
        let params = |mi_bins| SignificanceParams {
            permutation_tests: 2,
            num_repeats: 2,
            random_seed: 17,
            mi_bins,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: true,
        };

        assert_eq!(
            evaluate(&matrix, &combos, &observed, &metrics, &params(96)),
            evaluate(&matrix, &combos, &observed, &metrics, &params(12))
        );
    }

    #[test]
    fn significance_mi_bins_follow_the_observed_backend_ceiling() {
        let params = |backend_kind| SignificanceParams {
            permutation_tests: 1,
            num_repeats: 1,
            random_seed: 1,
            mi_bins: 96,
            backend_kind,
            mi_approximate: true,
        };

        assert_eq!(
            significance_mi_bins(73_728, &params(GAFIME_BACKEND_CPU)),
            96
        );
        assert_eq!(
            significance_mi_bins(73_728, &params(GAFIME_BACKEND_METAL)),
            48
        );
    }

    #[test]
    fn gpu_observations_force_fixed_width_mi_for_the_cpu_fallback() {
        let params = |backend_kind, mi_approximate| SignificanceParams {
            permutation_tests: 1,
            num_repeats: 1,
            random_seed: 1,
            mi_bins: 96,
            backend_kind,
            mi_approximate,
        };

        assert!(!significance_uses_fixed_width_mi(&params(
            GAFIME_BACKEND_CPU,
            false
        )));
        assert!(significance_uses_fixed_width_mi(&params(
            GAFIME_BACKEND_CPU,
            true
        )));
        assert!(significance_uses_fixed_width_mi(&params(
            GAFIME_BACKEND_METAL,
            false
        )));
    }

    #[test]
    fn metric_id_kernels_map_as_expected() {
        assert_eq!(
            MetricKernel::try_from(GAFIME_METRIC_PEARSON).unwrap(),
            MetricKernel::Pearson
        );
        assert_eq!(
            MetricKernel::try_from(GAFIME_METRIC_R2).unwrap(),
            MetricKernel::R2
        );
    }
}
