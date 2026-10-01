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
        combos::{legacy_unary_feature_order, select_adaptive_mi_bins_for_backend},
        DescriptorBatchSource, DEFAULT_DESCRIPTOR_BATCH_WORDS,
    },
    semantic::supervised::SupervisedStrengths,
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
        .par_iter()
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
            evaluate_precision_f32(&score_signals, target, &candidate_ids, metrics, params)
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
            evaluate_precision_mixed(&score_signals, target, &candidate_ids, metrics, params)
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
            evaluate_precision_f64(&score_signals, target, &candidate_ids, metrics, params)
        }
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
            let signals = candidate_ids
                .par_iter()
                .enumerate()
                .map(|(candidate, &candidate_id)| {
                    let start = candidate * rows;
                    PrecisionSignificanceSignal {
                        candidate_id,
                        values: CpuPrecisionValues::F32(columns[start..start + rows].to_vec()),
                    }
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
            let signals = candidate_ids
                .par_iter()
                .enumerate()
                .map(|(candidate, &candidate_id)| {
                    let start = candidate * rows;
                    PrecisionSignificanceSignal {
                        candidate_id,
                        values: CpuPrecisionValues::F64(columns[start..start + rows].to_vec()),
                    }
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
    build_expanded_decision_path_matrix_f32_with_parallelism(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        &paths,
        search,
        false,
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
    build_expanded_decision_path_matrix_f64_with_parallelism(
        source_features,
        rows,
        source_cols,
        target,
        &paths,
        search,
        false,
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
    build_expanded_decision_path_matrix_f32_with_parallelism(
        profile,
        source_features,
        rows,
        source_cols,
        target,
        paths,
        search,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_expanded_decision_path_matrix_f32_with_parallelism(
    profile: PrecisionProfile,
    source_features: &[f32],
    rows: usize,
    source_cols: usize,
    target: &[f32],
    paths: &[PrecisionDecisionPath],
    search: &ExpandedDecisionPathSearchSpec<'_>,
    parallelize: bool,
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
    let base_capacity =
        rows.checked_mul(base_candidate_cols)
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path base matrix exceeds host address space",
            ))?;
    let mut base_columns = vec![0.0f32; base_capacity];
    if parallelize && rows > 0 {
        base_columns
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(feature, column)| {
                for (row, value) in column.iter_mut().enumerate() {
                    *value = source_features[row * source_cols + feature];
                }
            });
    } else {
        for feature in 0..base_candidate_cols {
            for row in 0..rows {
                base_columns[feature * rows + row] = source_features[row * source_cols + feature];
            }
        }
    }
    let materialize_membership = |path: &PrecisionDecisionPath| {
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
    };
    let memberships = if parallelize {
        paths
            .par_iter()
            .map(materialize_membership)
            .collect::<OrchestratorResult<Vec<_>>>()?
    } else {
        paths
            .iter()
            .map(materialize_membership)
            .collect::<OrchestratorResult<Vec<_>>>()?
    };
    let mut expanded = vec![0.0f32; capacity];
    let fill_row = |row: usize, output: &mut [f32]| {
        let source_start = row * source_cols;
        output[..base_candidate_cols]
            .copy_from_slice(&source_features[source_start..source_start + base_candidate_cols]);
        for (slot, membership) in output[base_candidate_cols..].iter_mut().zip(&memberships) {
            *slot = membership[row];
        }
    };
    if parallelize && expanded_cols > 0 {
        expanded
            .par_chunks_mut(expanded_cols)
            .enumerate()
            .for_each(|(row, output)| fill_row(row, output));
    } else {
        for (row, output) in expanded.chunks_mut(expanded_cols.max(1)).enumerate() {
            if expanded_cols > 0 {
                fill_row(row, output);
            }
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
    build_expanded_decision_path_matrix_f64_with_parallelism(
        source_features,
        rows,
        source_cols,
        target,
        paths,
        search,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_expanded_decision_path_matrix_f64_with_parallelism(
    source_features: &[f64],
    rows: usize,
    source_cols: usize,
    target: &[f64],
    paths: &[PrecisionDecisionPath],
    search: &ExpandedDecisionPathSearchSpec<'_>,
    parallelize: bool,
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
    let base_capacity =
        rows.checked_mul(base_candidate_cols)
            .ok_or(OrchestratorError::InvalidPlan(
                "expanded decision-path base matrix exceeds host address space",
            ))?;
    let mut base_columns = vec![0.0f64; base_capacity];
    if parallelize && rows > 0 {
        base_columns
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(feature, column)| {
                for (row, value) in column.iter_mut().enumerate() {
                    *value = source_features[row * source_cols + feature];
                }
            });
    } else {
        for feature in 0..base_candidate_cols {
            for row in 0..rows {
                base_columns[feature * rows + row] = source_features[row * source_cols + feature];
            }
        }
    }
    let materialize_membership = |path: &PrecisionDecisionPath| {
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
    };
    let memberships = if parallelize {
        paths
            .par_iter()
            .map(materialize_membership)
            .collect::<OrchestratorResult<Vec<_>>>()?
    } else {
        paths
            .iter()
            .map(materialize_membership)
            .collect::<OrchestratorResult<Vec<_>>>()?
    };
    let mut expanded = vec![0.0f64; capacity];
    let fill_row = |row: usize, output: &mut [f64]| {
        let source_start = row * source_cols;
        output[..base_candidate_cols]
            .copy_from_slice(&source_features[source_start..source_start + base_candidate_cols]);
        for (slot, membership) in output[base_candidate_cols..].iter_mut().zip(&memberships) {
            *slot = membership[row];
        }
    };
    if parallelize && expanded_cols > 0 {
        expanded
            .par_chunks_mut(expanded_cols)
            .enumerate()
            .for_each(|(row, output)| fill_row(row, output));
    } else {
        for (row, output) in expanded.chunks_mut(expanded_cols.max(1)).enumerate() {
            if expanded_cols > 0 {
                fill_row(row, output);
            }
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
    let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32);
    for feature in unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 expanded decision-path null has f64 resident columns",
            ))?;
        let scores = score_all_f32_into(
            PrecisionProfile::Fp32,
            signal,
            target,
            metrics,
            params,
            &mut score_scratch,
        );
        update_precision_max_f32(&mut maxima, scores, metrics);
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
    let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed);
    for feature in unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "mixed expanded decision-path null has f64 resident columns",
            ))?;
        let scores = score_all_f64_from_f32_into(
            PrecisionProfile::Mixed,
            signal,
            target,
            metrics,
            params,
            &mut score_scratch,
        );
        update_precision_max_f64(&mut maxima, scores, metrics);
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
    let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64);
    for feature in unary_features {
        let signal = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 expanded decision-path null has f32 resident columns",
            ))?;
        let scores = score_all_f64_into(
            PrecisionProfile::Fp64,
            signal,
            target,
            metrics,
            params,
            &mut score_scratch,
        );
        update_precision_max_f64(&mut maxima, scores, metrics);
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
        .par_iter()
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
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32),
            |scratch, signal| {
                score_all_f32_into(
                    PrecisionProfile::Fp32,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f32(
        params.permutation_tests,
        &observed,
        metrics,
        "fp32 decision-path significance maxT width does not match metrics",
        |permutation| {
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
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32);
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
                let scores = score_all_f32_into(
                    PrecisionProfile::Fp32,
                    &signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f32(&mut maximums, scores, metrics);
            }
            Ok(maximums)
        },
    )?;
    let bootstrap = bootstrap_all_f32(
        PrecisionProfile::Fp32,
        &signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F32(if params.permutation_tests == 0 {
                    vec![f32::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f32(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            },
        )
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
        .par_iter()
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
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed),
            |scratch, signal| {
                score_all_f64_from_f32_into(
                    PrecisionProfile::Mixed,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        &observed,
        metrics,
        "mixed decision-path significance maxT width does not match metrics",
        |permutation| {
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
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed);
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
                let scores = score_all_f64_from_f32_into(
                    PrecisionProfile::Mixed,
                    &signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f64(&mut maximums, scores, metrics);
            }
            Ok(maximums)
        },
    )?;
    let bootstrap = bootstrap_all_f64_from_f32(
        PrecisionProfile::Mixed,
        &signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            },
        )
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
        .par_iter()
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
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64),
            |scratch, signal| {
                score_all_f64_into(
                    PrecisionProfile::Fp64,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        &observed,
        metrics,
        "fp64 decision-path significance maxT width does not match metrics",
        |permutation| {
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
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64);
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
                let scores = score_all_f64_into(
                    PrecisionProfile::Fp64,
                    &signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f64(&mut maximums, scores, metrics);
            }
            Ok(maximums)
        },
    )?;
    let bootstrap = bootstrap_all_f64(
        PrecisionProfile::Fp64,
        &signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            },
        )
        .collect())
}

fn evaluate_precision_f32(
    signals: &[&[f32]],
    target: &[f32],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let observed = signals
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32),
            |scratch, signal| {
                score_all_f32_into(
                    PrecisionProfile::Fp32,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f32(
        params.permutation_tests,
        &observed,
        metrics,
        "fp32 precision significance maxT width does not match metrics",
        |permutation| {
            let permuted = shuffle_f32(
                target,
                mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
            );
            let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32);
            for signal in signals {
                let scores = score_all_f32_into(
                    PrecisionProfile::Fp32,
                    signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f32(&mut maxima, scores, metrics);
            }
            Ok(maxima)
        },
    )?;
    let bootstrap = bootstrap_all_f32(
        PrecisionProfile::Fp32,
        signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(|(candidate, (means, stds))| {
            let pvalues = if params.permutation_tests == 0 {
                vec![f32::NAN; metrics.len()]
            } else {
                exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                    .iter()
                    .map(|&count| permutation_pvalue_f32(count, params.permutation_tests))
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F32(pvalues),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            }
        })
        .collect())
}

fn evaluate_precision_mixed(
    signals: &[&[f32]],
    target: &[f32],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let observed = signals
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed),
            |scratch, signal| {
                score_all_f64_from_f32_into(
                    PrecisionProfile::Mixed,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        &observed,
        metrics,
        "mixed precision significance maxT width does not match metrics",
        |permutation| {
            let permuted = shuffle_f32(
                target,
                mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
            );
            let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed);
            for signal in signals {
                let scores = score_all_f64_from_f32_into(
                    PrecisionProfile::Mixed,
                    signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f64(&mut maxima, scores, metrics);
            }
            Ok(maxima)
        },
    )?;
    let bootstrap = bootstrap_all_f64_from_f32(
        PrecisionProfile::Mixed,
        signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(|(candidate, (means, stds))| {
            let pvalues = if params.permutation_tests == 0 {
                vec![f64::NAN; metrics.len()]
            } else {
                exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                    .iter()
                    .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(pvalues),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

fn evaluate_precision_f64(
    signals: &[&[f64]],
    target: &[f64],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let observed = signals
        .par_iter()
        .map_init(
            || precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64),
            |scratch, signal| {
                score_all_f64_into(
                    PrecisionProfile::Fp64,
                    signal,
                    target,
                    metrics,
                    params,
                    scratch,
                )
                .to_vec()
            },
        )
        .collect::<Vec<_>>();
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        &observed,
        metrics,
        "fp64 precision significance maxT width does not match metrics",
        |permutation| {
            let permuted = shuffle_f64(
                target,
                mix_seed(params.random_seed, 0xA5A5_A5A5, u64::from(permutation)),
            );
            let mut maxima = vec![f64::NEG_INFINITY; metrics.len()];
            let mut score_scratch = precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64);
            for signal in signals {
                let scores = score_all_f64_into(
                    PrecisionProfile::Fp64,
                    signal,
                    &permuted,
                    metrics,
                    params,
                    &mut score_scratch,
                );
                update_precision_max_f64(&mut maxima, scores, metrics);
            }
            Ok(maxima)
        },
    )?;
    let bootstrap = bootstrap_all_f64(
        PrecisionProfile::Fp64,
        signals,
        target,
        metrics,
        params,
        candidate_ids,
    )?;
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(|(candidate, (means, stds))| {
            let pvalues = if params.permutation_tests == 0 {
                vec![f64::NAN; metrics.len()]
            } else {
                exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                    .iter()
                    .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                    .collect()
            };
            PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(pvalues),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            }
        })
        .collect())
}

#[cfg(test)]
fn score_all_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f32> {
    let mut scratch = precision::PrecisionScoreScratch::new(profile);
    score_all_f32_into(profile, signal, target, metrics, params, &mut scratch).to_vec()
}

fn score_all_f32_into<'a>(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut precision::PrecisionScoreScratch,
) -> &'a [f32] {
    let CpuPrecisionSlice::F32(scores) = precision::score_precision_signal_metrics_into(
        profile,
        CpuPrecisionSlice::F32(signal),
        CpuPrecisionSlice::F32(target),
        metrics,
        significance_mi_bins(signal.len() as u64, params),
        significance_uses_fixed_width_mi(params),
        scratch,
    )
    .expect("profile-specialized fp32 significance signal is valid") else {
        unreachable!("fp32 profile must return fp32 scores")
    };
    scores
}

#[cfg(test)]
fn score_all_f64_from_f32(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f64> {
    let mut scratch = precision::PrecisionScoreScratch::new(profile);
    score_all_f64_from_f32_into(profile, signal, target, metrics, params, &mut scratch).to_vec()
}

fn score_all_f64_from_f32_into<'a>(
    profile: PrecisionProfile,
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut precision::PrecisionScoreScratch,
) -> &'a [f64] {
    let CpuPrecisionSlice::F64(scores) = precision::score_precision_signal_metrics_into(
        profile,
        CpuPrecisionSlice::F32(signal),
        CpuPrecisionSlice::F32(target),
        metrics,
        significance_mi_bins(signal.len() as u64, params),
        significance_uses_fixed_width_mi(params),
        scratch,
    )
    .expect("profile-specialized mixed significance signal is valid") else {
        unreachable!("mixed profile must return fp64 scores")
    };
    scores
}

#[cfg(test)]
fn score_all_f64(
    profile: PrecisionProfile,
    signal: &[f64],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> Vec<f64> {
    let mut scratch = precision::PrecisionScoreScratch::new(profile);
    score_all_f64_into(profile, signal, target, metrics, params, &mut scratch).to_vec()
}

fn score_all_f64_into<'a>(
    profile: PrecisionProfile,
    signal: &[f64],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut precision::PrecisionScoreScratch,
) -> &'a [f64] {
    let CpuPrecisionSlice::F64(scores) = precision::score_precision_signal_metrics_into(
        profile,
        CpuPrecisionSlice::F64(signal),
        CpuPrecisionSlice::F64(target),
        metrics,
        significance_mi_bins(signal.len() as u64, params),
        significance_uses_fixed_width_mi(params),
        scratch,
    )
    .expect("profile-specialized fp64 significance signal is valid") else {
        unreachable!("fp64 profile must return fp64 scores")
    };
    scores
}

struct BootstrapScratchF32 {
    indices: Vec<usize>,
    signal: Vec<f32>,
    target: Vec<f32>,
    score: precision::PrecisionScoreScratch,
    samples: Vec<Vec<f32>>,
}

impl Default for BootstrapScratchF32 {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            signal: Vec::new(),
            target: Vec::new(),
            score: precision::PrecisionScoreScratch::new(PrecisionProfile::Fp32),
            samples: Vec::new(),
        }
    }
}

struct BootstrapScratchMixed {
    indices: Vec<usize>,
    signal: Vec<f32>,
    target: Vec<f32>,
    score: precision::PrecisionScoreScratch,
    samples: Vec<Vec<f64>>,
}

impl Default for BootstrapScratchMixed {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            signal: Vec::new(),
            target: Vec::new(),
            score: precision::PrecisionScoreScratch::new(PrecisionProfile::Mixed),
            samples: Vec::new(),
        }
    }
}

struct BootstrapScratchF64 {
    indices: Vec<usize>,
    signal: Vec<f64>,
    target: Vec<f64>,
    score: precision::PrecisionScoreScratch,
    samples: Vec<Vec<f64>>,
}

impl Default for BootstrapScratchF64 {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            signal: Vec::new(),
            target: Vec::new(),
            score: precision::PrecisionScoreScratch::new(PrecisionProfile::Fp64),
            samples: Vec::new(),
        }
    }
}

fn prepare_bootstrap_buffer<T>(buffer: &mut Vec<T>, required: usize) -> OrchestratorResult<()> {
    buffer.clear();
    if buffer.capacity() < required {
        buffer
            .try_reserve_exact(required)
            .map_err(|_| OrchestratorError::InvalidPlan("bootstrap scratch capacity is invalid"))?;
    }
    Ok(())
}

fn prepare_bootstrap_samples<T>(
    samples: &mut Vec<Vec<T>>,
    metric_count: usize,
    repeats: usize,
) -> OrchestratorResult<()> {
    metric_count
        .checked_mul(repeats)
        .ok_or(OrchestratorError::InvalidPlan(
            "bootstrap metric/repeat shape exceeds host address space",
        ))?;
    if samples.len() < metric_count {
        samples
            .try_reserve_exact(metric_count - samples.len())
            .map_err(|_| {
                OrchestratorError::InvalidPlan("bootstrap metric scratch capacity is invalid")
            })?;
        samples.resize_with(metric_count, Vec::new);
    } else {
        samples.truncate(metric_count);
    }
    for metric_samples in samples {
        prepare_bootstrap_buffer(metric_samples, repeats)?;
    }
    Ok(())
}

fn validate_bootstrap_candidates(
    signal_count: usize,
    candidate_ids: &[u64],
) -> OrchestratorResult<()> {
    if signal_count != candidate_ids.len() {
        return Err(OrchestratorError::InvalidPlan(
            "bootstrap candidate ids do not match significance signals",
        ));
    }
    Ok(())
}

fn bootstrap_all_f32<S: AsRef<[f32]> + Sync>(
    profile: PrecisionProfile,
    signals: &[S],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate_ids: &[u64],
) -> OrchestratorResult<Vec<(Vec<f32>, Vec<f32>)>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return Ok(signals
            .iter()
            .map(|_| (vec![f32::NAN; metrics.len()], vec![f32::NAN; metrics.len()]))
            .collect());
    }
    signals
        .par_iter()
        .enumerate()
        .map_init(
            BootstrapScratchF32::default,
            |scratch, (candidate, signal)| {
                prepare_bootstrap_buffer(&mut scratch.indices, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.signal, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.target, target.len())?;
                prepare_bootstrap_samples(&mut scratch.samples, metrics.len(), repeats)?;
                let signal = signal.as_ref();
                for repeat in 0..repeats {
                    bootstrap_indices_into(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                        &mut scratch.indices,
                    );
                    scratch.signal.clear();
                    scratch
                        .signal
                        .extend(scratch.indices.iter().map(|&index| signal[index]));
                    scratch.target.clear();
                    scratch
                        .target
                        .extend(scratch.indices.iter().map(|&index| target[index]));
                    let scores = score_all_f32_into(
                        profile,
                        &scratch.signal,
                        &scratch.target,
                        metrics,
                        params,
                        &mut scratch.score,
                    );
                    for (metric_samples, &value) in scratch.samples.iter_mut().zip(scores) {
                        metric_samples.push(value);
                    }
                }
                Ok(mean_std_f32(&scratch.samples))
            },
        )
        .collect()
}

fn bootstrap_all_f64_from_f32<S: AsRef<[f32]> + Sync>(
    profile: PrecisionProfile,
    signals: &[S],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate_ids: &[u64],
) -> OrchestratorResult<Vec<(Vec<f64>, Vec<f64>)>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return Ok(signals
            .iter()
            .map(|_| (vec![f64::NAN; metrics.len()], vec![f64::NAN; metrics.len()]))
            .collect());
    }
    signals
        .par_iter()
        .enumerate()
        .map_init(
            BootstrapScratchMixed::default,
            |scratch, (candidate, signal)| {
                prepare_bootstrap_buffer(&mut scratch.indices, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.signal, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.target, target.len())?;
                prepare_bootstrap_samples(&mut scratch.samples, metrics.len(), repeats)?;
                let signal = signal.as_ref();
                for repeat in 0..repeats {
                    bootstrap_indices_into(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                        &mut scratch.indices,
                    );
                    scratch.signal.clear();
                    scratch
                        .signal
                        .extend(scratch.indices.iter().map(|&index| signal[index]));
                    scratch.target.clear();
                    scratch
                        .target
                        .extend(scratch.indices.iter().map(|&index| target[index]));
                    let scores = score_all_f64_from_f32_into(
                        profile,
                        &scratch.signal,
                        &scratch.target,
                        metrics,
                        params,
                        &mut scratch.score,
                    );
                    for (metric_samples, &value) in scratch.samples.iter_mut().zip(scores) {
                        metric_samples.push(value);
                    }
                }
                Ok(mean_std_f64(&scratch.samples))
            },
        )
        .collect()
}

fn bootstrap_all_f64<S: AsRef<[f64]> + Sync>(
    profile: PrecisionProfile,
    signals: &[S],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    candidate_ids: &[u64],
) -> OrchestratorResult<Vec<(Vec<f64>, Vec<f64>)>> {
    validate_bootstrap_candidates(signals.len(), candidate_ids)?;
    let repeats = params.num_repeats as usize;
    if repeats == 0 {
        return Ok(signals
            .iter()
            .map(|_| (vec![f64::NAN; metrics.len()], vec![f64::NAN; metrics.len()]))
            .collect());
    }
    signals
        .par_iter()
        .enumerate()
        .map_init(
            BootstrapScratchF64::default,
            |scratch, (candidate, signal)| {
                prepare_bootstrap_buffer(&mut scratch.indices, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.signal, target.len())?;
                prepare_bootstrap_buffer(&mut scratch.target, target.len())?;
                prepare_bootstrap_samples(&mut scratch.samples, metrics.len(), repeats)?;
                let signal = signal.as_ref();
                for repeat in 0..repeats {
                    bootstrap_indices_into(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                        &mut scratch.indices,
                    );
                    scratch.signal.clear();
                    scratch
                        .signal
                        .extend(scratch.indices.iter().map(|&index| signal[index]));
                    scratch.target.clear();
                    scratch
                        .target
                        .extend(scratch.indices.iter().map(|&index| target[index]));
                    let scores = score_all_f64_into(
                        profile,
                        &scratch.signal,
                        &scratch.target,
                        metrics,
                        params,
                        &mut scratch.score,
                    );
                    for (metric_samples, &value) in scratch.samples.iter_mut().zip(scores) {
                        metric_samples.push(value);
                    }
                }
                Ok(mean_std_f64(&scratch.samples))
            },
        )
        .collect()
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

fn widened_permutation_pvalue_counts(exceedances: u32, permutation_tests: u32) -> (u64, u64) {
    (u64::from(exceedances) + 1, u64::from(permutation_tests) + 1)
}

fn permutation_pvalue_f32(exceedances: u32, permutation_tests: u32) -> f32 {
    let (numerator, denominator) =
        widened_permutation_pvalue_counts(exceedances, permutation_tests);
    numerator as f32 / denominator as f32
}

fn permutation_pvalue_f64(exceedances: u32, permutation_tests: u32) -> f64 {
    let (numerator, denominator) =
        widened_permutation_pvalue_counts(exceedances, permutation_tests);
    numerator as f64 / denominator as f64
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
    let mut indices = Vec::with_capacity(n);
    bootstrap_indices_into(n, seed, &mut indices);
    indices
}

fn bootstrap_indices_into(n: usize, seed: u64, indices: &mut Vec<usize>) {
    indices.clear();
    indices.reserve(n);
    let mut state = seed;
    for _ in 0..n {
        indices.push((splitmix64(&mut state) % n as u64) as usize);
    }
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
        .par_iter()
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
        .par_iter()
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
    maxima_for_permutation: impl Fn(&[f32]) -> OrchestratorResult<Vec<f32>> + Sync,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let exceedances = parallel_precision_exceedances_f32(
        params.permutation_tests,
        observed,
        metrics,
        "fp32 precision significance maxT width does not match metrics",
        |permutation| {
            let permuted =
                precision_permutation_target_f32(target, params.random_seed, permutation);
            maxima_for_permutation(&permuted)
        },
    )?;
    let bootstrap = if params.num_repeats > 1 {
        bootstrap_all_f32(
            PrecisionProfile::Fp32,
            signals,
            target,
            metrics,
            params,
            candidate_ids,
        )?
    } else {
        observed
            .iter()
            .map(|scores| (scores.to_vec(), vec![0.0f32; metrics.len()]))
            .collect()
    };
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F32(if params.permutation_tests == 0 {
                    vec![f32::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f32(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F32(means),
                stds: CpuPrecisionValues::F32(stds),
            },
        )
        .collect())
}

fn finish_precision_mixed(
    signals: &[Vec<f32>],
    target: &[f32],
    observed: &[&[f64]],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    maxima_for_permutation: impl Fn(&[f32]) -> OrchestratorResult<Vec<f64>> + Sync,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        observed,
        metrics,
        "mixed precision significance maxT width does not match metrics",
        |permutation| {
            let permuted =
                precision_permutation_target_f32(target, params.random_seed, permutation);
            maxima_for_permutation(&permuted)
        },
    )?;
    let bootstrap = if params.num_repeats > 1 {
        bootstrap_all_f64_from_f32(
            PrecisionProfile::Mixed,
            signals,
            target,
            metrics,
            params,
            candidate_ids,
        )?
    } else {
        observed
            .iter()
            .map(|scores| (scores.to_vec(), vec![0.0f64; metrics.len()]))
            .collect()
    };
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            },
        )
        .collect())
}

fn finish_precision_f64(
    signals: &[Vec<f64>],
    target: &[f64],
    observed: &[&[f64]],
    candidate_ids: &[u64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    maxima_for_permutation: impl Fn(&[f64]) -> OrchestratorResult<Vec<f64>> + Sync,
) -> OrchestratorResult<Vec<PrecisionCandidateSignificance>> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let exceedances = parallel_precision_exceedances_f64(
        params.permutation_tests,
        observed,
        metrics,
        "fp64 precision significance maxT width does not match metrics",
        |permutation| {
            let permuted =
                precision_permutation_target_f64(target, params.random_seed, permutation);
            maxima_for_permutation(&permuted)
        },
    )?;
    let bootstrap = if params.num_repeats > 1 {
        bootstrap_all_f64(
            PrecisionProfile::Fp64,
            signals,
            target,
            metrics,
            params,
            candidate_ids,
        )?
    } else {
        observed
            .iter()
            .map(|scores| (scores.to_vec(), vec![0.0f64; metrics.len()]))
            .collect()
    };
    Ok(bootstrap
        .into_iter()
        .enumerate()
        .map(
            |(candidate, (means, stds))| PrecisionCandidateSignificance {
                candidate_id: candidate_ids[candidate],
                pvalues: CpuPrecisionValues::F64(if params.permutation_tests == 0 {
                    vec![f64::NAN; metrics.len()]
                } else {
                    exceedances[candidate * metrics.len()..(candidate + 1) * metrics.len()]
                        .iter()
                        .map(|&count| permutation_pvalue_f64(count, params.permutation_tests))
                        .collect()
                }),
                means: CpuPrecisionValues::F64(means),
                stds: CpuPrecisionValues::F64(stds),
            },
        )
        .collect())
}

struct PrecisionSignificanceScratch {
    interaction_f32: Vec<f32>,
    interaction_f64: Vec<f64>,
    score: precision::PrecisionScoreScratch,
}

impl PrecisionSignificanceScratch {
    fn new(profile: PrecisionProfile) -> Self {
        Self {
            interaction_f32: Vec::new(),
            interaction_f64: Vec::new(),
            score: precision::PrecisionScoreScratch::new(profile),
        }
    }
}

fn build_precision_interaction_f32_into(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    interaction: &mut Vec<f32>,
) -> OrchestratorResult<()> {
    if combo.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance combo is empty",
        ));
    }
    interaction.clear();
    interaction.resize(matrix.rows() as usize, 1.0f32);
    for &feature in combo {
        let column = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "f32 precision significance combo exceeds resident columns",
            ))?;
        let mean =
            matrix
                .column_mean_f32(feature as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "f32 precision significance combo has no resident mean",
                ))?;
        for (product, &value) in interaction.iter_mut().zip(column) {
            *product *= value - mean;
        }
    }
    Ok(())
}

fn build_precision_interaction_f64_into(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    interaction: &mut Vec<f64>,
) -> OrchestratorResult<()> {
    if combo.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "precision significance combo is empty",
        ));
    }
    interaction.clear();
    interaction.resize(matrix.rows() as usize, 1.0f64);
    for &feature in combo {
        let column = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 precision significance combo exceeds resident columns",
            ))?;
        let mean =
            matrix
                .column_mean_f64(feature as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "fp64 precision significance combo has no resident mean",
                ))?;
        for (product, &value) in interaction.iter_mut().zip(column) {
            *product *= value - mean;
        }
    }
    Ok(())
}

fn score_precision_combo_f32_into<'a>(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut PrecisionSignificanceScratch,
) -> OrchestratorResult<&'a [f32]> {
    let signal = if combo.len() == 1 {
        matrix
            .column_f32(combo[0] as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 significance combo exceeds resident columns",
            ))?
    } else {
        build_precision_interaction_f32_into(matrix, combo, &mut scratch.interaction_f32)?;
        &scratch.interaction_f32
    };
    Ok(score_all_f32_into(
        PrecisionProfile::Fp32,
        signal,
        target,
        metrics,
        params,
        &mut scratch.score,
    ))
}

fn score_precision_combo_mixed_into<'a>(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f32],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut PrecisionSignificanceScratch,
) -> OrchestratorResult<&'a [f64]> {
    let signal = if combo.len() == 1 {
        matrix
            .column_f32(combo[0] as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "mixed significance combo exceeds resident columns",
            ))?
    } else {
        build_precision_interaction_f32_into(matrix, combo, &mut scratch.interaction_f32)?;
        &scratch.interaction_f32
    };
    Ok(score_all_f64_from_f32_into(
        PrecisionProfile::Mixed,
        signal,
        target,
        metrics,
        params,
        &mut scratch.score,
    ))
}

fn score_precision_combo_f64_into<'a>(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    target: &[f64],
    metrics: &[MetricKernel],
    params: &SignificanceParams,
    scratch: &'a mut PrecisionSignificanceScratch,
) -> OrchestratorResult<&'a [f64]> {
    let signal = if combo.len() == 1 {
        matrix
            .column_f64(combo[0] as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 significance combo exceeds resident columns",
            ))?
    } else {
        build_precision_interaction_f64_into(matrix, combo, &mut scratch.interaction_f64)?;
        &scratch.interaction_f64
    };
    Ok(score_all_f64_into(
        PrecisionProfile::Fp64,
        signal,
        target,
        metrics,
        params,
        &mut scratch.score,
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

fn precision_exceedance_slots(
    candidate_count: usize,
    metric_count: usize,
) -> OrchestratorResult<usize> {
    candidate_count
        .checked_mul(metric_count)
        .ok_or(OrchestratorError::InvalidPlan(
            "precision significance exceedance table exceeds host address space",
        ))
}

// Retain only a bounded window of per-permutation metric maxima. Candidate
// comparisons accumulate into the one final count table after each parallel
// window completes, so temporary count storage never multiplies by workers.
const PRECISION_MAXT_PERMUTATION_BATCH_SIZE: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrecisionExceedanceMemoryShape {
    final_count_slots: usize,
    maximum_batch_permutations: usize,
    maximum_batch_value_slots: usize,
}

fn precision_exceedance_memory_shape(
    candidate_count: usize,
    metric_count: usize,
    permutation_tests: u32,
) -> OrchestratorResult<PrecisionExceedanceMemoryShape> {
    let final_count_slots = precision_exceedance_slots(candidate_count, metric_count)?;
    let maximum_batch_permutations = usize::try_from(
        permutation_tests.min(PRECISION_MAXT_PERMUTATION_BATCH_SIZE),
    )
    .map_err(|_| {
        OrchestratorError::InvalidPlan(
            "precision significance permutation batch exceeds host address space",
        )
    })?;
    let maximum_batch_value_slots = maximum_batch_permutations.checked_mul(metric_count).ok_or(
        OrchestratorError::InvalidPlan(
            "precision significance permutation maxima batch exceeds host address space",
        ),
    )?;
    Ok(PrecisionExceedanceMemoryShape {
        final_count_slots,
        maximum_batch_permutations,
        maximum_batch_value_slots,
    })
}

fn parallel_precision_exceedances_f32<O, F>(
    permutation_tests: u32,
    observed: &[O],
    metrics: &[MetricKernel],
    width_error: &'static str,
    maxima_for_permutation: F,
) -> OrchestratorResult<Vec<u32>>
where
    O: AsRef<[f32]> + Sync,
    F: Fn(u32) -> OrchestratorResult<Vec<f32>> + Sync,
{
    if metrics.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "fp32 precision significance requires at least one metric",
        ));
    }
    if observed
        .iter()
        .any(|scores| scores.as_ref().len() != metrics.len())
    {
        return Err(OrchestratorError::InvalidPlan(
            "fp32 precision significance observed width does not match metrics",
        ));
    }
    let memory_shape =
        precision_exceedance_memory_shape(observed.len(), metrics.len(), permutation_tests)?;
    let mut counts = vec![0u32; memory_shape.final_count_slots];
    let mut batch_start = 0u32;
    while batch_start < permutation_tests {
        let batch_end = batch_start
            .saturating_add(PRECISION_MAXT_PERMUTATION_BATCH_SIZE)
            .min(permutation_tests);
        let maxima_batch = (batch_start..batch_end)
            .into_par_iter()
            .map(&maxima_for_permutation)
            .collect::<OrchestratorResult<Vec<_>>>()?;
        if maxima_batch.len() > memory_shape.maximum_batch_permutations {
            return Err(OrchestratorError::InvalidPlan(
                "precision significance permutation batch exceeded its bounded shape",
            ));
        }
        let batch_value_slots = maxima_batch.iter().try_fold(0usize, |slots, maxima| {
            slots
                .checked_add(maxima.len())
                .ok_or(OrchestratorError::InvalidPlan(
                    "precision significance permutation maxima batch exceeds host address space",
                ))
        })?;
        if batch_value_slots > memory_shape.maximum_batch_value_slots {
            return Err(OrchestratorError::InvalidPlan(width_error));
        }
        if maxima_batch
            .iter()
            .any(|maxima| maxima.len() != metrics.len())
        {
            return Err(OrchestratorError::InvalidPlan(width_error));
        }
        // The permutation maxima phase above has completed before this phase
        // begins, so this is not nested Rayon. Each worker owns one disjoint
        // candidate row in the final count table and visits the bounded maxima
        // batch in permutation order, retaining deterministic checked sums
        // without atomics or a worker-sized C x M temporary table.
        counts
            .par_chunks_mut(metrics.len())
            .enumerate()
            .try_for_each(|(candidate, candidate_counts)| {
                let observed_scores = observed[candidate].as_ref();
                for maxima in &maxima_batch {
                    for (metric_index, (&maximum, &metric)) in
                        maxima.iter().zip(metrics).enumerate()
                    {
                        if maximum >= extremeness_f32(observed_scores[metric_index], metric) {
                            candidate_counts[metric_index] = candidate_counts[metric_index]
                                .checked_add(1)
                                .ok_or(OrchestratorError::InvalidPlan(
                                    "precision significance exceedance count overflowed u32",
                                ))?;
                        }
                    }
                }
                Ok(())
            })?;
        batch_start = batch_end;
    }
    Ok(counts)
}

fn parallel_precision_exceedances_f64<O, F>(
    permutation_tests: u32,
    observed: &[O],
    metrics: &[MetricKernel],
    width_error: &'static str,
    maxima_for_permutation: F,
) -> OrchestratorResult<Vec<u32>>
where
    O: AsRef<[f64]> + Sync,
    F: Fn(u32) -> OrchestratorResult<Vec<f64>> + Sync,
{
    if metrics.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "mixed/fp64 precision significance requires at least one metric",
        ));
    }
    if observed
        .iter()
        .any(|scores| scores.as_ref().len() != metrics.len())
    {
        return Err(OrchestratorError::InvalidPlan(
            "mixed/fp64 precision significance observed width does not match metrics",
        ));
    }
    let memory_shape =
        precision_exceedance_memory_shape(observed.len(), metrics.len(), permutation_tests)?;
    let mut counts = vec![0u32; memory_shape.final_count_slots];
    let mut batch_start = 0u32;
    while batch_start < permutation_tests {
        let batch_end = batch_start
            .saturating_add(PRECISION_MAXT_PERMUTATION_BATCH_SIZE)
            .min(permutation_tests);
        let maxima_batch = (batch_start..batch_end)
            .into_par_iter()
            .map(&maxima_for_permutation)
            .collect::<OrchestratorResult<Vec<_>>>()?;
        if maxima_batch.len() > memory_shape.maximum_batch_permutations {
            return Err(OrchestratorError::InvalidPlan(
                "precision significance permutation batch exceeded its bounded shape",
            ));
        }
        let batch_value_slots = maxima_batch.iter().try_fold(0usize, |slots, maxima| {
            slots
                .checked_add(maxima.len())
                .ok_or(OrchestratorError::InvalidPlan(
                    "precision significance permutation maxima batch exceeds host address space",
                ))
        })?;
        if batch_value_slots > memory_shape.maximum_batch_value_slots {
            return Err(OrchestratorError::InvalidPlan(width_error));
        }
        if maxima_batch
            .iter()
            .any(|maxima| maxima.len() != metrics.len())
        {
            return Err(OrchestratorError::InvalidPlan(width_error));
        }
        counts
            .par_chunks_mut(metrics.len())
            .enumerate()
            .try_for_each(|(candidate, candidate_counts)| {
                let observed_scores = observed[candidate].as_ref();
                for maxima in &maxima_batch {
                    for (metric_index, (&maximum, &metric)) in
                        maxima.iter().zip(metrics).enumerate()
                    {
                        if maximum >= extremeness_f64(observed_scores[metric_index], metric) {
                            candidate_counts[metric_index] = candidate_counts[metric_index]
                                .checked_add(1)
                                .ok_or(OrchestratorError::InvalidPlan(
                                    "precision significance exceedance count overflowed u32",
                                ))?;
                        }
                    }
                }
                Ok(())
            })?;
        batch_start = batch_end;
    }
    Ok(counts)
}

fn fixed_precision_maxima_f32(
    matrix: &CpuPrecisionMatrix,
    shuffled: &[f32],
    null_family: &NullFamily<'_>,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<f32>> {
    let mut maxima = vec![f32::NEG_INFINITY; metrics.len()];
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Fp32);
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores =
            score_precision_combo_f32_into(matrix, combo, shuffled, metrics, params, &mut scratch)?;
        update_precision_max_f32(&mut maxima, scores, metrics);
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
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Mixed);
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores = score_precision_combo_mixed_into(
            matrix,
            combo,
            shuffled,
            metrics,
            params,
            &mut scratch,
        )?;
        update_precision_max_f64(&mut maxima, scores, metrics);
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
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Fp64);
    null_family.try_for_each_combo_with_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS, |combo| {
        let scores =
            score_precision_combo_f64_into(matrix, combo, shuffled, metrics, params, &mut scratch)?;
        update_precision_max_f64(&mut maxima, scores, metrics);
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

/// Apply the canonical supervised shortlist ordering to strengths that this
/// significance path has already reduced.  The reducers above intentionally
/// remain local: in particular, the fp64 path maps nonfinite scores to
/// negative infinity before its maximum, unlike public unary-screening
/// compatibility reduction.
fn canonical_higher_feature_order(
    mut strengths: SupervisedStrengths,
    search: &AdaptiveSearchSpec<'_>,
    backend_kind: BackendKind,
) -> Vec<u32> {
    if backend_kind == GAFIME_BACKEND_CPU {
        // Published Core inserted scheduler results by ascending feature ID,
        // while GPU mappings retained unary-plan order. Stable score sorting
        // therefore uses this order only as the Core tie-break contract.
        strengths.sort_by_feature();
    }
    strengths.higher_feature_order(
        search.candidate_feature_count,
        search.max_combinations_per_arity,
        search.top_features_for_higher_arity,
        search.planning_seed_words,
    )
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
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Fp32);
    for &feature in search.unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 adaptive significance received fp64 resident columns",
            ))?;
        let scores = score_all_f32_into(
            PrecisionProfile::Fp32,
            signal,
            shuffled,
            metrics,
            params,
            &mut scratch.score,
        );
        update_precision_max_f32(&mut maxima, scores, metrics);
        unary_strengths.push((feature, screening_strength(scores, metrics)));
    }
    let higher_features = canonical_higher_feature_order(
        SupervisedStrengths::F32(unary_strengths),
        search,
        params.backend_kind,
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
                match score_precision_combo_f32_into(
                    matrix,
                    combo,
                    shuffled,
                    metrics,
                    params,
                    &mut scratch,
                ) {
                    Ok(scores) => update_precision_max_f32(&mut maxima, scores, metrics),
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
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Mixed);
    for &feature in search.unary_features {
        let signal = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "mixed adaptive significance received fp64 resident columns",
            ))?;
        let scores = score_all_f64_from_f32_into(
            PrecisionProfile::Mixed,
            signal,
            shuffled,
            metrics,
            params,
            &mut scratch.score,
        );
        update_precision_max_f64(&mut maxima, scores, metrics);
        unary_strengths.push((feature, precision_screening_strength_f64(scores, metrics)));
    }
    let higher_features = canonical_higher_feature_order(
        SupervisedStrengths::F64(unary_strengths),
        search,
        params.backend_kind,
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
                match score_precision_combo_mixed_into(
                    matrix,
                    combo,
                    shuffled,
                    metrics,
                    params,
                    &mut scratch,
                ) {
                    Ok(scores) => update_precision_max_f64(&mut maxima, scores, metrics),
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
    let mut scratch = PrecisionSignificanceScratch::new(PrecisionProfile::Fp64);
    for &feature in search.unary_features {
        let signal = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp64 adaptive significance received f32 resident columns",
            ))?;
        let scores = score_all_f64_into(
            PrecisionProfile::Fp64,
            signal,
            shuffled,
            metrics,
            params,
            &mut scratch.score,
        );
        update_precision_max_f64(&mut maxima, scores, metrics);
        unary_strengths.push((feature, precision_screening_strength_f64(scores, metrics)));
    }
    let higher_features = canonical_higher_feature_order(
        SupervisedStrengths::F64(unary_strengths),
        search,
        params.backend_kind,
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
                match score_precision_combo_f64_into(
                    matrix,
                    combo,
                    shuffled,
                    metrics,
                    params,
                    &mut scratch,
                ) {
                    Ok(scores) => update_precision_max_f64(&mut maxima, scores, metrics),
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
    let higher_features = canonical_higher_feature_order(
        SupervisedStrengths::F32(unary_strengths),
        &search,
        backend_kind,
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
        GafimeRankSpec, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_R2,
    };

    fn pearson_r2() -> Vec<MetricKernel> {
        vec![MetricKernel::Pearson, MetricKernel::R2]
    }

    fn precision_profiles() -> [PrecisionProfile; 3] {
        [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ]
    }

    fn precision_parity_metrics() -> [MetricKernel; 4] {
        [
            MetricKernel::Pearson,
            MetricKernel::Spearman,
            MetricKernel::MutualInfo,
            MetricKernel::R2,
        ]
    }

    fn precision_parity_params() -> SignificanceParams {
        SignificanceParams {
            permutation_tests: 5,
            num_repeats: 4,
            random_seed: 0x51A7_7E57,
            mi_bins: 4,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: true,
        }
    }

    #[test]
    fn adaptive_shortlist_keeps_producer_order_and_existing_fp64_reduction() {
        let search = AdaptiveSearchSpec {
            unary_features: &[0, 1, 2, 3],
            candidate_feature_count: 4,
            max_arity: 2,
            max_combinations_per_arity: 10,
            top_features_for_higher_arity: 3,
            planning_seed_words: &[7],
        };
        let producer_order = vec![(3, 1.0), (2, 1.0), (1, 1.0), (0, 1.0)];

        let core = canonical_higher_feature_order(
            SupervisedStrengths::F32(producer_order.clone()),
            &search,
            GAFIME_BACKEND_CPU,
        );
        let cuda = canonical_higher_feature_order(
            SupervisedStrengths::F32(producer_order),
            &search,
            GAFIME_BACKEND_CUDA,
        );
        assert_eq!(core, vec![2, 0, 1]);
        assert_eq!(cuda, vec![1, 3, 2]);

        // This path intentionally retains its pre-existing nonfinite reducer;
        // only its final ordering migrated to SupervisedStrengths.
        assert_eq!(
            precision_screening_strength_f64(&[f64::NAN], &[MetricKernel::Pearson]),
            f64::NEG_INFINITY
        );
    }

    fn precision_parity_data(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let features = (0..rows * cols)
            .map(|index| {
                let row = (index / cols) as f32 + 1.0;
                let column = (index % cols) as f32 + 1.0;
                (row * column * 0.071).sin()
                    + (row * (column + 0.5) * 0.037).cos()
                    + row * column * 0.000_7
            })
            .collect();
        let target = (0..rows)
            .map(|row| {
                let row = row as f32 + 1.0;
                (row * 0.113).sin() + (row * 0.047).cos() + row * 0.001_3
            })
            .collect();
        (features, target)
    }

    fn precision_parity_matrix(
        profile: PrecisionProfile,
        rows: usize,
        cols: usize,
    ) -> CpuPrecisionMatrix {
        let (features, target) = precision_parity_data(rows, cols);
        match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                CpuPrecisionMatrix::from_row_major_f32(
                    profile,
                    rows as u64,
                    cols as u32,
                    features,
                    target,
                )
                .unwrap()
            }
            PrecisionProfile::Fp64 => CpuPrecisionMatrix::from_row_major_f64(
                profile,
                rows as u64,
                cols as u32,
                features.into_iter().map(f64::from).collect(),
                target.into_iter().map(f64::from).collect(),
            )
            .unwrap(),
        }
    }

    fn column_major_f32(row_major: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut columns = Vec::with_capacity(row_major.len());
        for column in 0..cols {
            for row in 0..rows {
                columns.push(row_major[row * cols + column]);
            }
        }
        columns
    }

    fn precision_test_pools() -> (rayon::ThreadPool, rayon::ThreadPool) {
        (
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap(),
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
        )
    }

    fn assert_precision_values_bit_equal(
        single: &CpuPrecisionValues,
        parallel: &CpuPrecisionValues,
        context: &str,
    ) {
        match (single, parallel) {
            (CpuPrecisionValues::F32(single), CpuPrecisionValues::F32(parallel)) => {
                assert_eq!(single.len(), parallel.len(), "{context}");
                assert_eq!(
                    single
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    parallel
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{context}"
                );
            }
            (CpuPrecisionValues::F64(single), CpuPrecisionValues::F64(parallel)) => {
                assert_eq!(single.len(), parallel.len(), "{context}");
                assert_eq!(
                    single
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    parallel
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{context}"
                );
            }
            _ => panic!("precision result dtype changed: {context}"),
        }
    }

    fn assert_precision_results_bit_equal(
        single: &[PrecisionCandidateSignificance],
        parallel: &[PrecisionCandidateSignificance],
        context: &str,
    ) {
        assert_eq!(single.len(), parallel.len(), "{context}");
        for (row, (single, parallel)) in single.iter().zip(parallel).enumerate() {
            assert_eq!(
                single.candidate_id, parallel.candidate_id,
                "{context} row={row}"
            );
            assert_precision_values_bit_equal(
                &single.pvalues,
                &parallel.pvalues,
                &format!("{context} row={row} pvalues"),
            );
            assert_precision_values_bit_equal(
                &single.means,
                &parallel.means,
                &format!("{context} row={row} means"),
            );
            assert_precision_values_bit_equal(
                &single.stds,
                &parallel.stds,
                &format!("{context} row={row} stds"),
            );
        }
    }

    fn expanded_paths_for_profile(profile: PrecisionProfile) -> Vec<PrecisionDecisionPath> {
        let mut paths = expanded_f32_paths();
        match profile {
            PrecisionProfile::Fp32 => paths,
            PrecisionProfile::Mixed => {
                for path in &mut paths {
                    let CpuPrecisionScalar::F32(gain) = path.gain else {
                        unreachable!()
                    };
                    path.gain = CpuPrecisionScalar::F64(f64::from(gain));
                }
                paths
            }
            PrecisionProfile::Fp64 => {
                for path in &mut paths {
                    for node in &mut path.nodes {
                        let CpuPrecisionScalar::F32(threshold) = node.threshold else {
                            unreachable!()
                        };
                        node.threshold = CpuPrecisionScalar::F64(f64::from(threshold));
                    }
                    let CpuPrecisionScalar::F32(gain) = path.gain else {
                        unreachable!()
                    };
                    path.gain = CpuPrecisionScalar::F64(f64::from(gain));
                }
                paths
            }
        }
    }

    fn serial_bootstrap_reference_f32(
        signals: &[Vec<f32>],
        target: &[f32],
        metrics: &[MetricKernel],
        params: &SignificanceParams,
        candidate_ids: &[u64],
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let repeats = params.num_repeats as usize;
        signals
            .iter()
            .enumerate()
            .map(|(candidate, signal)| {
                let mut samples = vec![Vec::with_capacity(repeats); metrics.len()];
                for repeat in 0..repeats {
                    let indices = bootstrap_indices(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                    );
                    let x = indices
                        .iter()
                        .map(|&index| signal[index])
                        .collect::<Vec<_>>();
                    let y = indices
                        .iter()
                        .map(|&index| target[index])
                        .collect::<Vec<_>>();
                    let scores = score_all_f32(PrecisionProfile::Fp32, &x, &y, metrics, params);
                    for (metric_samples, &value) in samples.iter_mut().zip(&scores) {
                        metric_samples.push(value);
                    }
                }
                mean_std_f32(&samples)
            })
            .collect()
    }

    fn serial_bootstrap_reference_mixed(
        signals: &[Vec<f32>],
        target: &[f32],
        metrics: &[MetricKernel],
        params: &SignificanceParams,
        candidate_ids: &[u64],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let repeats = params.num_repeats as usize;
        signals
            .iter()
            .enumerate()
            .map(|(candidate, signal)| {
                let mut samples = vec![Vec::with_capacity(repeats); metrics.len()];
                for repeat in 0..repeats {
                    let indices = bootstrap_indices(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                    );
                    let x = indices
                        .iter()
                        .map(|&index| signal[index])
                        .collect::<Vec<_>>();
                    let y = indices
                        .iter()
                        .map(|&index| target[index])
                        .collect::<Vec<_>>();
                    let scores =
                        score_all_f64_from_f32(PrecisionProfile::Mixed, &x, &y, metrics, params);
                    for (metric_samples, &value) in samples.iter_mut().zip(&scores) {
                        metric_samples.push(value);
                    }
                }
                mean_std_f64(&samples)
            })
            .collect()
    }

    fn serial_bootstrap_reference_f64(
        signals: &[Vec<f64>],
        target: &[f64],
        metrics: &[MetricKernel],
        params: &SignificanceParams,
        candidate_ids: &[u64],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let repeats = params.num_repeats as usize;
        signals
            .iter()
            .enumerate()
            .map(|(candidate, signal)| {
                let mut samples = vec![Vec::with_capacity(repeats); metrics.len()];
                for repeat in 0..repeats {
                    let indices = bootstrap_indices(
                        target.len(),
                        mix_seed(
                            params.random_seed,
                            0xB007_57A9 ^ candidate_ids[candidate],
                            repeat as u64,
                        ),
                    );
                    let x = indices
                        .iter()
                        .map(|&index| signal[index])
                        .collect::<Vec<_>>();
                    let y = indices
                        .iter()
                        .map(|&index| target[index])
                        .collect::<Vec<_>>();
                    let scores = score_all_f64(PrecisionProfile::Fp64, &x, &y, metrics, params);
                    for (metric_samples, &value) in samples.iter_mut().zip(&scores) {
                        metric_samples.push(value);
                    }
                }
                mean_std_f64(&samples)
            })
            .collect()
    }

    fn assert_bootstrap_summary_f32_bits(
        expected: &[(Vec<f32>, Vec<f32>)],
        actual: &[(Vec<f32>, Vec<f32>)],
    ) {
        assert_eq!(expected.len(), actual.len());
        for ((expected_mean, expected_std), (actual_mean, actual_std)) in
            expected.iter().zip(actual)
        {
            assert_precision_values_bit_equal(
                &CpuPrecisionValues::F32(expected_mean.clone()),
                &CpuPrecisionValues::F32(actual_mean.clone()),
                "bootstrap fp32 mean",
            );
            assert_precision_values_bit_equal(
                &CpuPrecisionValues::F32(expected_std.clone()),
                &CpuPrecisionValues::F32(actual_std.clone()),
                "bootstrap fp32 std",
            );
        }
    }

    fn assert_bootstrap_summary_f64_bits(
        expected: &[(Vec<f64>, Vec<f64>)],
        actual: &[(Vec<f64>, Vec<f64>)],
    ) {
        assert_eq!(expected.len(), actual.len());
        for ((expected_mean, expected_std), (actual_mean, actual_std)) in
            expected.iter().zip(actual)
        {
            assert_precision_values_bit_equal(
                &CpuPrecisionValues::F64(expected_mean.clone()),
                &CpuPrecisionValues::F64(actual_mean.clone()),
                "bootstrap f64 mean",
            );
            assert_precision_values_bit_equal(
                &CpuPrecisionValues::F64(expected_std.clone()),
                &CpuPrecisionValues::F64(actual_std.clone()),
                "bootstrap f64 std",
            );
        }
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
    fn precision_significance_is_identical_in_single_and_multi_worker_pools() {
        const ROWS: usize = 127;
        const COLS: usize = 24;
        let f32_values = (0..ROWS * COLS)
            .map(|index| {
                let row = (index / COLS) as f32;
                let column = (index % COLS) as f32 + 1.0;
                (row * (0.011 * column)).sin() + row * column * 0.000_3
            })
            .collect::<Vec<_>>();
        let f32_target = (0..ROWS)
            .map(|row| {
                let row = row as f32;
                (row * 0.037).cos() + row * 0.002
            })
            .collect::<Vec<_>>();
        let combos = (0..COLS)
            .map(|column| vec![column as u32])
            .collect::<Vec<_>>();
        let metrics = [
            MetricKernel::Pearson,
            MetricKernel::Spearman,
            MetricKernel::MutualInfo,
            MetricKernel::R2,
        ];
        let params = SignificanceParams {
            permutation_tests: 8,
            num_repeats: 8,
            random_seed: 0xC011_AB1E,
            mi_bins: 8,
            backend_kind: GAFIME_BACKEND_CPU,
            mi_approximate: true,
        };
        let one_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let multi_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let matrix = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                    CpuPrecisionMatrix::from_row_major_f32(
                        profile,
                        ROWS as u64,
                        COLS as u32,
                        f32_values.clone(),
                        f32_target.clone(),
                    )
                    .unwrap()
                }
                PrecisionProfile::Fp64 => CpuPrecisionMatrix::from_row_major_f64(
                    profile,
                    ROWS as u64,
                    COLS as u32,
                    f32_values.iter().map(|&value| f64::from(value)).collect(),
                    f32_target.iter().map(|&value| f64::from(value)).collect(),
                )
                .unwrap(),
            };
            let single = one_worker
                .install(|| evaluate_precision_shortlist(&matrix, &combos, &metrics, &params))
                .unwrap();
            let parallel = multi_worker
                .install(|| evaluate_precision_shortlist(&matrix, &combos, &metrics, &params))
                .unwrap();
            assert_eq!(single, parallel, "profile={profile:?}");
            assert_eq!(
                parallel
                    .iter()
                    .map(|result| result.candidate_id)
                    .collect::<Vec<_>>(),
                (0..COLS as u64).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn precision_fixed_null_family_is_bit_identical_across_worker_counts() {
        const ROWS: usize = 24;
        const COLS: usize = 5;
        let metrics = precision_parity_metrics();
        let params = precision_parity_params();
        let combos = vec![vec![0], vec![1], vec![0, 2], vec![1, 3, 4]];
        let candidate_ids = [101, 207, 309, 411];
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let matrix = precision_parity_matrix(profile, ROWS, COLS);
            let observed = combos
                .iter()
                .map(|combo| {
                    precision::score_precision_continuous_combo(
                        &matrix,
                        combo,
                        &metrics,
                        params.mi_bins,
                        params.mi_approximate,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let single = one_worker
                .install(|| {
                    let plan = precision_plan(ROWS as u64, COLS as u32, 3);
                    evaluate_precision_with_null_family(
                        &matrix,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &plan,
                        &metrics,
                        &params,
                    )
                })
                .unwrap();
            let parallel = four_workers
                .install(|| {
                    let plan = precision_plan(ROWS as u64, COLS as u32, 3);
                    evaluate_precision_with_null_family(
                        &matrix,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &plan,
                        &metrics,
                        &params,
                    )
                })
                .unwrap();
            assert_precision_results_bit_equal(
                &single,
                &parallel,
                &format!("fixed-null profile={profile:?}"),
            );
        }
    }

    #[test]
    fn precision_adaptive_family_is_bit_identical_across_worker_counts() {
        const ROWS: usize = 24;
        const COLS: usize = 5;
        let metrics = precision_parity_metrics();
        let params = precision_parity_params();
        let combos = vec![vec![0], vec![2], vec![0, 1], vec![1, 3, 4]];
        let candidate_ids = [501, 607, 709, 811];
        let unary_features = [0, 1, 2, 3, 4];
        let planning_seed_words = [0x0ADA_B71E];
        let search = AdaptiveSearchSpec {
            unary_features: &unary_features,
            candidate_feature_count: COLS as u32,
            max_arity: 3,
            max_combinations_per_arity: 8,
            top_features_for_higher_arity: 4,
            planning_seed_words: &planning_seed_words,
        };
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let matrix = precision_parity_matrix(profile, ROWS, COLS);
            let observed = combos
                .iter()
                .map(|combo| {
                    precision::score_precision_continuous_combo(
                        &matrix,
                        combo,
                        &metrics,
                        params.mi_bins,
                        params.mi_approximate,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let single = one_worker
                .install(|| {
                    let plan = precision_plan(ROWS as u64, COLS as u32, 3);
                    evaluate_precision_with_adaptive_search(
                        &matrix,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &plan,
                        &metrics,
                        &params,
                        &search,
                    )
                })
                .unwrap();
            let parallel = four_workers
                .install(|| {
                    let plan = precision_plan(ROWS as u64, COLS as u32, 3);
                    evaluate_precision_with_adaptive_search(
                        &matrix,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &plan,
                        &metrics,
                        &params,
                        &search,
                    )
                })
                .unwrap();
            assert_precision_results_bit_equal(
                &single,
                &parallel,
                &format!("adaptive profile={profile:?}"),
            );
        }
    }

    #[test]
    fn precision_time_series_significance_is_bit_identical_across_worker_counts() {
        const ROWS: usize = 24;
        const COLS: usize = 3;
        let (row_major, target) = precision_parity_data(ROWS, COLS);
        let columns = column_major_f32(&row_major, ROWS, COLS);
        let columns_f64 = columns.iter().copied().map(f64::from).collect::<Vec<_>>();
        let target_f64 = target.iter().copied().map(f64::from).collect::<Vec<_>>();
        let metrics = precision_parity_metrics();
        let params = precision_parity_params();
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let source = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                    CpuPrecisionSlice::F32(&columns)
                }
                PrecisionProfile::Fp64 => CpuPrecisionSlice::F64(&columns_f64),
            };
            let (generated, descriptors) = crate::time_series::time_series_columns_precision(
                profile,
                source,
                ROWS,
                COLS,
                &[1, 3],
                &[4],
                true,
            )
            .unwrap();
            let candidate_ids = (0..descriptors.len())
                .map(|candidate| 1_000 + candidate as u64 * 7)
                .collect::<Vec<_>>();
            let generated_slice = match &generated {
                CpuPrecisionValues::F32(values) => CpuPrecisionSlice::F32(values),
                CpuPrecisionValues::F64(values) => CpuPrecisionSlice::F64(values),
            };
            let single = one_worker
                .install(|| {
                    evaluate_precision_time_series_columns(
                        profile,
                        generated_slice,
                        ROWS,
                        match profile {
                            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                                CpuPrecisionSlice::F32(&target)
                            }
                            PrecisionProfile::Fp64 => CpuPrecisionSlice::F64(&target_f64),
                        },
                        &candidate_ids,
                        &metrics,
                        &params,
                    )
                })
                .unwrap();
            let parallel = four_workers
                .install(|| {
                    evaluate_precision_time_series_columns(
                        profile,
                        generated_slice,
                        ROWS,
                        match profile {
                            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                                CpuPrecisionSlice::F32(&target)
                            }
                            PrecisionProfile::Fp64 => CpuPrecisionSlice::F64(&target_f64),
                        },
                        &candidate_ids,
                        &metrics,
                        &params,
                    )
                })
                .unwrap();
            assert_precision_results_bit_equal(
                &single,
                &parallel,
                &format!("time-series profile={profile:?}"),
            );
        }
    }

    #[test]
    fn precision_decision_path_significance_is_bit_identical_across_worker_counts() {
        const ROWS: usize = 24;
        const COLS: usize = 4;
        let (row_major, target) = precision_parity_data(ROWS, COLS);
        let columns = column_major_f32(&row_major, ROWS, COLS);
        let columns_f64 = columns.iter().copied().map(f64::from).collect::<Vec<_>>();
        let target_f64 = target.iter().copied().map(f64::from).collect::<Vec<_>>();
        let metrics = precision_parity_metrics();
        let params = precision_parity_params();
        let discovery = DecisionPathParams {
            max_depth: 2,
            rounds: 2,
            max_paths: 6,
            max_bins: 8,
            min_leaf: 2,
            learning_rate: 0.75,
        };
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let (source, target_slice) = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => (
                    CpuPrecisionSlice::F32(&columns),
                    CpuPrecisionSlice::F32(&target),
                ),
                PrecisionProfile::Fp64 => (
                    CpuPrecisionSlice::F64(&columns_f64),
                    CpuPrecisionSlice::F64(&target_f64),
                ),
            };
            let paths = find_decision_paths_precision(
                profile,
                source,
                ROWS,
                COLS,
                target_slice,
                &discovery,
            )
            .unwrap();
            assert!(!paths.is_empty(), "profile={profile:?}");
            let candidate_ids = (0..paths.len())
                .map(|candidate| 2_000 + candidate as u64 * 11)
                .collect::<Vec<_>>();
            let single = one_worker
                .install(|| {
                    evaluate_precision_decision_path_family(
                        profile,
                        source,
                        ROWS,
                        COLS,
                        target_slice,
                        &paths,
                        &candidate_ids,
                        &metrics,
                        &params,
                        &discovery,
                    )
                })
                .unwrap();
            let parallel = four_workers
                .install(|| {
                    evaluate_precision_decision_path_family(
                        profile,
                        source,
                        ROWS,
                        COLS,
                        target_slice,
                        &paths,
                        &candidate_ids,
                        &metrics,
                        &params,
                        &discovery,
                    )
                })
                .unwrap();
            assert_precision_results_bit_equal(
                &single,
                &parallel,
                &format!("decision-path profile={profile:?}"),
            );
        }
    }

    #[test]
    fn precision_expanded_decision_path_is_bit_identical_across_worker_counts() {
        const ROWS: usize = 8;
        const COLS: usize = 3;
        let source = expanded_f32_source();
        let source_f64 = source.iter().copied().map(f64::from).collect::<Vec<_>>();
        let target = [0.0f32, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0];
        let target_f64 = target.map(f64::from);
        let discovery_features = [0, 1];
        let planning_seed_words = [0xD15C_0A7A];
        let search = ExpandedDecisionPathSearchSpec {
            base_candidate_cols: COLS as u32,
            discovery_features: &discovery_features,
            discovery: DecisionPathParams {
                max_depth: 1,
                rounds: 1,
                max_paths: 2,
                max_bins: 4,
                min_leaf: 1,
                learning_rate: 1.0,
            },
            max_arity: 2,
            max_combinations_per_arity: 8,
            top_features_for_higher_arity: 5,
            planning_seed_words: &planning_seed_words,
        };
        let combo = vec![0, 3];
        let combos = vec![combo.clone()];
        let candidate_ids = [3_001];
        let metrics = precision_parity_metrics();
        let params = precision_parity_params();
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let paths = expanded_paths_for_profile(profile);
            let (source_slice, target_slice, observed_matrix) = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => (
                    CpuPrecisionSlice::F32(&source),
                    CpuPrecisionSlice::F32(&target),
                    build_expanded_decision_path_matrix_f32(
                        profile, &source, ROWS, COLS, &target, &paths, &search,
                    )
                    .unwrap(),
                ),
                PrecisionProfile::Fp64 => (
                    CpuPrecisionSlice::F64(&source_f64),
                    CpuPrecisionSlice::F64(&target_f64),
                    build_expanded_decision_path_matrix_f64(
                        &source_f64,
                        ROWS,
                        COLS,
                        &target_f64,
                        &paths,
                        &search,
                    )
                    .unwrap(),
                ),
            };
            let observed = vec![precision::score_precision_continuous_combo(
                &observed_matrix,
                &combo,
                &metrics,
                params.mi_bins,
                params.mi_approximate,
            )
            .unwrap()];
            let single = one_worker
                .install(|| {
                    evaluate_precision_expanded_decision_path_family(
                        profile,
                        source_slice,
                        ROWS,
                        COLS,
                        target_slice,
                        &paths,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &metrics,
                        &params,
                        &search,
                    )
                })
                .unwrap();
            let parallel = four_workers
                .install(|| {
                    evaluate_precision_expanded_decision_path_family(
                        profile,
                        source_slice,
                        ROWS,
                        COLS,
                        target_slice,
                        &paths,
                        &combos,
                        &observed,
                        &candidate_ids,
                        &metrics,
                        &params,
                        &search,
                    )
                })
                .unwrap();
            assert_precision_results_bit_equal(
                &single,
                &parallel,
                &format!("expanded-decision-path profile={profile:?}"),
            );
        }
    }

    #[test]
    fn precision_parallel_significance_edges_are_deterministic_and_fail_closed() {
        const ROWS: usize = 8;
        const COLS: usize = 2;
        let metrics = precision_parity_metrics();
        let combos = vec![vec![0], vec![1]];
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let matrix = precision_parity_matrix(profile, ROWS, COLS);
            for (permutation_tests, num_repeats) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let params = SignificanceParams {
                    permutation_tests,
                    num_repeats,
                    ..precision_parity_params()
                };
                let single = one_worker
                    .install(|| evaluate_precision_shortlist(&matrix, &combos, &metrics, &params))
                    .unwrap();
                let parallel = four_workers
                    .install(|| evaluate_precision_shortlist(&matrix, &combos, &metrics, &params))
                    .unwrap();
                assert_precision_results_bit_equal(
                    &single,
                    &parallel,
                    &format!(
                        "edge profile={profile:?} permutations={permutation_tests} repeats={num_repeats}"
                    ),
                );
            }

            let empty = four_workers
                .install(|| {
                    evaluate_precision_shortlist(&matrix, &[], &metrics, &precision_parity_params())
                })
                .unwrap();
            assert!(empty.is_empty(), "profile={profile:?}");

            let single_error = one_worker
                .install(|| {
                    evaluate_precision_continuous_shortlist(
                        &matrix,
                        &combos,
                        &[17],
                        &metrics,
                        &precision_parity_params(),
                    )
                })
                .unwrap_err();
            let parallel_error = four_workers
                .install(|| {
                    evaluate_precision_continuous_shortlist(
                        &matrix,
                        &combos,
                        &[17],
                        &metrics,
                        &precision_parity_params(),
                    )
                })
                .unwrap_err();
            assert_eq!(
                format!("{single_error:?}"),
                format!("{parallel_error:?}"),
                "profile={profile:?}"
            );
        }
    }

    #[test]
    fn candidate_parallel_bootstrap_matches_serial_repeat_order_bitwise() {
        const ROWS: usize = 19;
        const COLS: usize = 4;
        let metrics = precision_parity_metrics();
        let params = SignificanceParams {
            num_repeats: 7,
            ..precision_parity_params()
        };
        let combos = vec![vec![0], vec![1, 2], vec![0, 2, 3]];
        let candidate_ids = [91, 207, 4_009];
        let (one_worker, four_workers) = precision_test_pools();

        for profile in precision_profiles() {
            let matrix = precision_parity_matrix(profile, ROWS, COLS);
            match profile {
                PrecisionProfile::Fp32 => {
                    let signals = precision_continuous_signals_f32(&matrix, &combos).unwrap();
                    let target = matrix.target_f32().unwrap();
                    let expected = serial_bootstrap_reference_f32(
                        &signals,
                        target,
                        &metrics,
                        &params,
                        &candidate_ids,
                    );
                    let single = one_worker
                        .install(|| {
                            bootstrap_all_f32(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    let parallel = four_workers
                        .install(|| {
                            bootstrap_all_f32(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    assert_bootstrap_summary_f32_bits(&expected, &single);
                    assert_bootstrap_summary_f32_bits(&expected, &parallel);
                }
                PrecisionProfile::Mixed => {
                    let signals = precision_continuous_signals_f32(&matrix, &combos).unwrap();
                    let target = matrix.target_f32().unwrap();
                    let expected = serial_bootstrap_reference_mixed(
                        &signals,
                        target,
                        &metrics,
                        &params,
                        &candidate_ids,
                    );
                    let single = one_worker
                        .install(|| {
                            bootstrap_all_f64_from_f32(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    let parallel = four_workers
                        .install(|| {
                            bootstrap_all_f64_from_f32(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    assert_bootstrap_summary_f64_bits(&expected, &single);
                    assert_bootstrap_summary_f64_bits(&expected, &parallel);
                }
                PrecisionProfile::Fp64 => {
                    let signals = precision_continuous_signals_f64(&matrix, &combos).unwrap();
                    let target = matrix.target_f64().unwrap();
                    let expected = serial_bootstrap_reference_f64(
                        &signals,
                        target,
                        &metrics,
                        &params,
                        &candidate_ids,
                    );
                    let single = one_worker
                        .install(|| {
                            bootstrap_all_f64(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    let parallel = four_workers
                        .install(|| {
                            bootstrap_all_f64(
                                profile,
                                &signals,
                                target,
                                &metrics,
                                &params,
                                &candidate_ids,
                            )
                        })
                        .unwrap();
                    assert_bootstrap_summary_f64_bits(&expected, &single);
                    assert_bootstrap_summary_f64_bits(&expected, &parallel);
                }
            }
        }

        let error = bootstrap_all_f32(
            PrecisionProfile::Fp32,
            &[vec![1.0f32, 2.0]],
            &[1.0f32, 2.0],
            &metrics,
            &params,
            &[],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OrchestratorError::InvalidPlan(
                "bootstrap candidate ids do not match significance signals"
            )
        );

        let overflow =
            prepare_bootstrap_samples::<f32>(&mut Vec::new(), usize::MAX, 2).unwrap_err();
        assert_eq!(
            overflow,
            OrchestratorError::InvalidPlan(
                "bootstrap metric/repeat shape exceeds host address space"
            )
        );
    }

    #[test]
    fn precision_exceedance_memory_is_one_count_table_plus_bounded_maxima() {
        let small = precision_exceedance_memory_shape(3, 4, u32::MAX).unwrap();
        let large = precision_exceedance_memory_shape(30_000, 4, u32::MAX).unwrap();

        assert_eq!(small.final_count_slots, 12);
        assert_eq!(large.final_count_slots, 120_000);
        assert_eq!(small.maximum_batch_permutations, 32);
        assert_eq!(large.maximum_batch_permutations, 32);
        assert_eq!(small.maximum_batch_value_slots, 32 * 4);
        assert_eq!(large.maximum_batch_value_slots, 32 * 4);

        // The temporary maxima shape is independent of candidate count and of
        // Rayon worker count. Only the one final table scales with candidates.
        let (one_worker, four_workers) = precision_test_pools();
        let one_worker_shape = one_worker
            .install(|| precision_exceedance_memory_shape(30_000, 4, 65))
            .unwrap();
        let four_worker_shape = four_workers
            .install(|| precision_exceedance_memory_shape(30_000, 4, 65))
            .unwrap();
        assert_eq!(one_worker_shape, four_worker_shape);
        assert_eq!(one_worker_shape.maximum_batch_value_slots, 32 * 4);
    }

    #[test]
    fn bounded_precision_exceedance_batches_preserve_exact_counts() {
        const PERMUTATIONS: u32 = 65;
        let metrics = [MetricKernel::Pearson, MetricKernel::R2];
        let observed_f32 = [vec![2.0f32, 2.0], vec![4.0, 4.0]];
        let observed_f64 = [vec![2.0f64, 2.0], vec![4.0, 4.0]];
        let mut expected = vec![0u32; observed_f32.len() * metrics.len()];
        for permutation in 0..PERMUTATIONS {
            let maxima = [(permutation % 7) as f64, (permutation % 5) as f64];
            for (candidate, observed) in observed_f64.iter().enumerate() {
                for (metric_index, &metric) in metrics.iter().enumerate() {
                    if maxima[metric_index] >= extremeness_f64(observed[metric_index], metric) {
                        expected[candidate * metrics.len() + metric_index] += 1;
                    }
                }
            }
        }

        let (_, four_workers) = precision_test_pools();
        let actual_f32 = four_workers
            .install(|| {
                parallel_precision_exceedances_f32(
                    PERMUTATIONS,
                    &observed_f32,
                    &metrics,
                    "test fp32 maxima width",
                    |permutation| Ok(vec![(permutation % 7) as f32, (permutation % 5) as f32]),
                )
            })
            .unwrap();
        let actual_f64 = four_workers
            .install(|| {
                parallel_precision_exceedances_f64(
                    PERMUTATIONS,
                    &observed_f64,
                    &metrics,
                    "test f64 maxima width",
                    |permutation| Ok(vec![(permutation % 7) as f64, (permutation % 5) as f64]),
                )
            })
            .unwrap();
        assert_eq!(actual_f32, expected);
        assert_eq!(actual_f64, expected);
    }

    #[test]
    fn planner_candidate_ids_bind_bootstrap_streams_across_reordering() {
        const ROWS: usize = 29;
        let target_f32 = (0..ROWS)
            .map(|row| {
                let row = row as f32 + 1.0;
                (row * 0.131).sin() + (row * 0.043).cos() + row * 0.002
            })
            .collect::<Vec<_>>();
        let signal_a_f32 = (0..ROWS)
            .map(|row| {
                let row = row as f32 + 1.0;
                (row * 0.077).sin() + row * 0.011
            })
            .collect::<Vec<_>>();
        let signal_b_f32 = (0..ROWS)
            .map(|row| {
                let row = row as f32 + 1.0;
                (row * 0.193).cos() - row * 0.004
            })
            .collect::<Vec<_>>();
        let target_f64 = target_f32
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>();
        let metrics = precision_parity_metrics();
        let params = SignificanceParams {
            permutation_tests: 3,
            num_repeats: 7,
            random_seed: 0x1D5_B007,
            ..precision_parity_params()
        };

        for profile in precision_profiles() {
            let values = |values: &[f32]| match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                    CpuPrecisionValues::F32(values.to_vec())
                }
                PrecisionProfile::Fp64 => {
                    CpuPrecisionValues::F64(values.iter().copied().map(f64::from).collect())
                }
            };
            let original = vec![
                PrecisionSignificanceSignal {
                    candidate_id: 9_001,
                    values: values(&signal_a_f32),
                },
                PrecisionSignificanceSignal {
                    candidate_id: 17,
                    values: values(&signal_b_f32),
                },
            ];
            let reordered = vec![original[1].clone(), original[0].clone()];
            let target = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                    CpuPrecisionSlice::F32(&target_f32)
                }
                PrecisionProfile::Fp64 => CpuPrecisionSlice::F64(&target_f64),
            };

            let original_results =
                evaluate_precision_signals(profile, target, &original, &metrics, &params).unwrap();
            let reordered_results =
                evaluate_precision_signals(profile, target, &reordered, &metrics, &params).unwrap();
            assert_eq!(
                original_results
                    .iter()
                    .map(|result| result.candidate_id)
                    .collect::<Vec<_>>(),
                vec![9_001, 17],
                "profile={profile:?}"
            );
            assert_eq!(
                reordered_results
                    .iter()
                    .map(|result| result.candidate_id)
                    .collect::<Vec<_>>(),
                vec![17, 9_001],
                "profile={profile:?}"
            );
            for original_result in &original_results {
                let reordered_result = reordered_results
                    .iter()
                    .find(|result| result.candidate_id == original_result.candidate_id)
                    .unwrap();
                assert_precision_values_bit_equal(
                    &original_result.pvalues,
                    &reordered_result.pvalues,
                    &format!(
                        "profile={profile:?} candidate={} pvalues",
                        original_result.candidate_id
                    ),
                );
                assert_precision_values_bit_equal(
                    &original_result.means,
                    &reordered_result.means,
                    &format!(
                        "profile={profile:?} candidate={} means",
                        original_result.candidate_id
                    ),
                );
                assert_precision_values_bit_equal(
                    &original_result.stds,
                    &reordered_result.stds,
                    &format!(
                        "profile={profile:?} candidate={} stds",
                        original_result.candidate_id
                    ),
                );
            }
        }
    }

    #[test]
    fn permutation_pvalue_pseudocount_widens_before_u32_max_boundary() {
        let widened = widened_permutation_pvalue_counts(u32::MAX, u32::MAX);
        assert_eq!(widened, (1u64 << 32, 1u64 << 32));
        assert_eq!(
            permutation_pvalue_f32(u32::MAX, u32::MAX).to_bits(),
            1.0f32.to_bits()
        );
        assert_eq!(
            permutation_pvalue_f64(u32::MAX, u32::MAX).to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(
            permutation_pvalue_f32(0, u32::MAX).to_bits(),
            (1.0f32 / 4_294_967_296.0f32).to_bits()
        );
        assert_eq!(
            permutation_pvalue_f64(0, u32::MAX).to_bits(),
            (1.0f64 / 4_294_967_296.0f64).to_bits()
        );
        assert_eq!(
            permutation_pvalue_f32(1, 5).to_bits(),
            (2.0f32 / 6.0f32).to_bits()
        );
        assert_eq!(
            permutation_pvalue_f64(1, 5).to_bits(),
            (2.0f64 / 6.0f64).to_bits()
        );
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
                score_all_f32(
                    PrecisionProfile::Fp32,
                    matrix.column_f32(feature as usize).unwrap(),
                    &target,
                    &metrics,
                    &typed_significance_params(),
                )[0]
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
    fn typed_gpu_observations_execute_the_same_fixed_mi_fallback_for_every_profile() {
        let rows = 1_152usize;
        let signal_f32 = (0..rows)
            .map(|index| {
                let unit = index as f32 / (rows - 1) as f32;
                (unit * 13.0).sin() + unit * 0.25
            })
            .collect::<Vec<_>>();
        let target_f32 = signal_f32
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if index % 11 == 0 {
                    value * -0.5
                } else if value > 0.35 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let signal_f64 = signal_f32
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>();
        let target_f64 = target_f32
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>();
        let metrics = [MetricKernel::MutualInfo];
        let params = |mi_approximate| SignificanceParams {
            permutation_tests: 3,
            num_repeats: 3,
            random_seed: 0xC0DE_71A1,
            mi_bins: 96,
            backend_kind: GAFIME_BACKEND_CUDA,
            mi_approximate,
        };

        for profile in precision_profiles() {
            let (target, values) = match profile {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => (
                    CpuPrecisionSlice::F32(&target_f32),
                    CpuPrecisionValues::F32(signal_f32.clone()),
                ),
                PrecisionProfile::Fp64 => (
                    CpuPrecisionSlice::F64(&target_f64),
                    CpuPrecisionValues::F64(signal_f64.clone()),
                ),
            };
            let signals = [PrecisionSignificanceSignal {
                candidate_id: 91,
                values,
            }];
            let forced =
                evaluate_precision_signals(profile, target, &signals, &metrics, &params(false))
                    .unwrap();
            let explicit =
                evaluate_precision_signals(profile, target, &signals, &metrics, &params(true))
                    .unwrap();
            assert_eq!(forced, explicit, "profile={profile:?}");
            assert_eq!(forced[0].candidate_id, 91);
        }
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
