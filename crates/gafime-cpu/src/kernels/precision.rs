//! Profile-specialized Core continuous kernels.
//!
//! Dispatch happens once at the public entrypoint.  The hot loops below are
//! split by profile, so the fp32 lane contains no hidden f64 accumulator or
//! result conversion, while mixed deliberately widens only statistical work
//! after its f32 interaction vector has been materialized.

use core::cmp::Ordering;

use gafime_orchestrator::{
    plan::combos::{
        sanitize_mi_bins_for_backend, MI_SAMPLES_PER_JOINT_BIN, MI_TEMPLATE_BIN_LEVELS,
    },
    OrchestratorError, OrchestratorResult,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

use super::MetricKernel;
use crate::precision::{
    CpuPrecisionMatrix, CpuPrecisionScalar, CpuPrecisionSlice, CpuPrecisionValues,
};

/// Reusable profile-typed materialization and result buffers for a worker.
///
/// A caller creates it for one profile and retains it through a candidate
/// batch.  This avoids allocation in the hot candidate loop and documents that
/// a task cannot switch precision mid-artifact.
#[derive(Debug)]
pub struct PrecisionScoreScratch {
    profile: PrecisionProfile,
    interaction_f32: Vec<f32>,
    interaction_f64: Vec<f64>,
    scores_f32: Vec<f32>,
    scores_f64: Vec<f64>,
    fixed_mi: FixedMiScratch,
}

#[derive(Debug, Default)]
struct FixedMiScratch {
    finite_f32_x: Vec<f32>,
    finite_f32_y: Vec<f32>,
    finite_f64_x: Vec<f64>,
    finite_f64_y: Vec<f64>,
    hist_x: Vec<u32>,
    hist_y: Vec<u32>,
    joint: Vec<u32>,
}

impl PrecisionScoreScratch {
    pub fn new(profile: PrecisionProfile) -> Self {
        Self {
            profile,
            interaction_f32: Vec::new(),
            interaction_f64: Vec::new(),
            scores_f32: Vec::new(),
            scores_f64: Vec::new(),
            fixed_mi: FixedMiScratch::default(),
        }
    }

    pub fn profile(&self) -> PrecisionProfile {
        self.profile
    }

    fn ensure_profile(&self, profile: PrecisionProfile) -> OrchestratorResult<()> {
        if self.profile != profile {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision scratch profile does not match its matrix profile",
            ));
        }
        Ok(())
    }
}

/// Score one continuous candidate in its profile's public result dtype.
pub fn score_precision_continuous_combo(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
) -> OrchestratorResult<CpuPrecisionValues> {
    let mut scratch = PrecisionScoreScratch::new(matrix.profile());
    match score_precision_continuous_combo_into(
        matrix,
        combo,
        metrics,
        mi_bins,
        mi_approximate,
        &mut scratch,
    )? {
        CpuPrecisionSlice::F32(values) => Ok(CpuPrecisionValues::F32(values.to_vec())),
        CpuPrecisionSlice::F64(values) => Ok(CpuPrecisionValues::F64(values.to_vec())),
    }
}

/// Materialize a unary or interaction signal in the profile's pointwise dtype.
/// Significance uses this once per shortlisted candidate and then only reruns
/// profile-specialized statistical reductions for each resample/permutation.
pub fn materialize_precision_combo(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
) -> OrchestratorResult<CpuPrecisionValues> {
    if combo.is_empty() {
        return Err(OrchestratorError::InvalidPlan("continuous combo is empty"));
    }
    if combo.iter().any(|&feature| feature >= matrix.cols()) {
        return Err(OrchestratorError::InvalidPlan(
            "continuous combo feature is out of bounds",
        ));
    }
    match matrix.profile() {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
            if combo.len() == 1 {
                return matrix
                    .column_f32(combo[0] as usize)
                    .map(|values| CpuPrecisionValues::F32(values.to_vec()))
                    .ok_or(OrchestratorError::InvalidPlan(
                        "f32 precision matrix has non-f32 resident columns",
                    ));
            }
            let mut interaction = Vec::new();
            build_interaction_f32(matrix, combo, &mut interaction)?;
            Ok(CpuPrecisionValues::F32(interaction))
        }
        PrecisionProfile::Fp64 => {
            if combo.len() == 1 {
                return matrix
                    .column_f64(combo[0] as usize)
                    .map(|values| CpuPrecisionValues::F64(values.to_vec()))
                    .ok_or(OrchestratorError::InvalidPlan(
                        "fp64 precision matrix has non-f64 resident columns",
                    ));
            }
            let mut interaction = Vec::new();
            build_interaction_f64(matrix, combo, &mut interaction)?;
            Ok(CpuPrecisionValues::F64(interaction))
        }
    }
}

/// Score one continuous candidate using caller-owned, profile-fixed scratch.
///
/// The returned slice borrows `scratch`, allowing an executor to write it into
/// a same-typed result table without an intermediate f32/f64 conversion.
#[allow(clippy::too_many_arguments)]
pub fn score_precision_continuous_combo_into<'a>(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    scratch: &'a mut PrecisionScoreScratch,
) -> OrchestratorResult<CpuPrecisionSlice<'a>> {
    if combo.is_empty() {
        return Err(OrchestratorError::InvalidPlan("continuous combo is empty"));
    }
    if combo.iter().any(|&feature| feature >= matrix.cols()) {
        return Err(OrchestratorError::InvalidPlan(
            "continuous combo feature is out of bounds",
        ));
    }
    scratch.ensure_profile(matrix.profile())?;

    // Profile selection occurs once per candidate.  The individual calls only
    // contain the one permitted numeric type in their arithmetic loops.
    match matrix.profile() {
        PrecisionProfile::Fp32 => {
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "fp32 CPU matrix has non-f32 resident target",
            ))?;
            let PrecisionScoreScratch {
                interaction_f32,
                scores_f32,
                fixed_mi,
                ..
            } = scratch;
            let signal = if combo.len() == 1 {
                matrix
                    .column_f32(combo[0] as usize)
                    .ok_or(OrchestratorError::InvalidPlan(
                        "fp32 CPU matrix has non-f32 resident columns",
                    ))?
            } else {
                build_interaction_f32(matrix, combo, interaction_f32)?;
                interaction_f32
            };
            score_f32(
                signal,
                target,
                metrics,
                mi_bins,
                mi_approximate,
                scores_f32,
                fixed_mi,
            );
            Ok(CpuPrecisionSlice::F32(scores_f32))
        }
        PrecisionProfile::Mixed => {
            let target = matrix.target_f32().ok_or(OrchestratorError::InvalidPlan(
                "mixed CPU matrix has non-f32 resident target",
            ))?;
            let PrecisionScoreScratch {
                interaction_f32,
                scores_f64,
                fixed_mi,
                ..
            } = scratch;
            let signal = if combo.len() == 1 {
                matrix
                    .column_f32(combo[0] as usize)
                    .ok_or(OrchestratorError::InvalidPlan(
                        "mixed CPU matrix has non-f32 resident columns",
                    ))?
            } else {
                build_interaction_f32(matrix, combo, interaction_f32)?;
                interaction_f32
            };
            score_mixed(
                signal,
                target,
                metrics,
                mi_bins,
                mi_approximate,
                scores_f64,
                fixed_mi,
            );
            Ok(CpuPrecisionSlice::F64(scores_f64))
        }
        PrecisionProfile::Fp64 => {
            let target = matrix.target_f64().ok_or(OrchestratorError::InvalidPlan(
                "fp64 CPU matrix has non-f64 resident target",
            ))?;
            let PrecisionScoreScratch {
                interaction_f64,
                scores_f64,
                fixed_mi,
                ..
            } = scratch;
            let signal = if combo.len() == 1 {
                matrix
                    .column_f64(combo[0] as usize)
                    .ok_or(OrchestratorError::InvalidPlan(
                        "fp64 CPU matrix has non-f64 resident columns",
                    ))?
            } else {
                build_interaction_f64(matrix, combo, interaction_f64)?;
                interaction_f64
            };
            score_f64(
                signal,
                target,
                metrics,
                mi_bins,
                mi_approximate,
                scores_f64,
                fixed_mi,
            );
            Ok(CpuPrecisionSlice::F64(scores_f64))
        }
    }
}

/// Score a pre-materialized signal with the profile's statistical and public
/// result domain.  Significance, time-series, and decision-path callers use
/// this instead of routing through legacy f32 metric helpers.
pub fn score_precision_signal(
    profile: PrecisionProfile,
    signal: CpuPrecisionSlice<'_>,
    target: CpuPrecisionSlice<'_>,
    metric: MetricKernel,
    mi_bins: u32,
    mi_approximate: bool,
) -> OrchestratorResult<CpuPrecisionScalar> {
    let mut scratch = PrecisionScoreScratch::new(profile);
    match score_precision_signal_metrics_into(
        profile,
        signal,
        target,
        &[metric],
        mi_bins,
        mi_approximate,
        &mut scratch,
    )? {
        CpuPrecisionSlice::F32(values) => Ok(CpuPrecisionScalar::F32(values[0])),
        CpuPrecisionSlice::F64(values) => Ok(CpuPrecisionScalar::F64(values[0])),
    }
}

/// Score a pre-materialized signal for several metrics with worker-owned
/// scratch. This is the allocation-bounded counterpart used by parallel
/// significance phases; Pearson and R2 also share one covariance evaluation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_precision_signal_metrics_into<'a>(
    profile: PrecisionProfile,
    signal: CpuPrecisionSlice<'_>,
    target: CpuPrecisionSlice<'_>,
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    scratch: &'a mut PrecisionScoreScratch,
) -> OrchestratorResult<CpuPrecisionSlice<'a>> {
    scratch.ensure_profile(profile)?;
    let PrecisionScoreScratch {
        scores_f32,
        scores_f64,
        fixed_mi,
        ..
    } = scratch;
    match (profile, signal, target) {
        (PrecisionProfile::Fp32, CpuPrecisionSlice::F32(x), CpuPrecisionSlice::F32(y)) => {
            score_f32(x, y, metrics, mi_bins, mi_approximate, scores_f32, fixed_mi);
            Ok(CpuPrecisionSlice::F32(scores_f32))
        }
        (PrecisionProfile::Mixed, CpuPrecisionSlice::F32(x), CpuPrecisionSlice::F32(y)) => {
            score_mixed(x, y, metrics, mi_bins, mi_approximate, scores_f64, fixed_mi);
            Ok(CpuPrecisionSlice::F64(scores_f64))
        }
        (PrecisionProfile::Fp64, CpuPrecisionSlice::F64(x), CpuPrecisionSlice::F64(y)) => {
            score_f64(x, y, metrics, mi_bins, mi_approximate, scores_f64, fixed_mi);
            Ok(CpuPrecisionSlice::F64(scores_f64))
        }
        _ => Err(OrchestratorError::InvalidPlan(
            "CPU precision signal dtype does not match the requested profile",
        )),
    }
}

fn build_interaction_f32(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    out: &mut Vec<f32>,
) -> OrchestratorResult<()> {
    out.clear();
    out.resize(matrix.rows() as usize, 1.0f32);
    for &feature in combo {
        let column = matrix
            .column_f32(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "f32 interaction requested from non-f32 CPU matrix",
            ))?;
        let mean =
            matrix
                .column_mean_f32(feature as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "f32 interaction requested from non-f32 CPU matrix",
                ))?;
        for (product, &value) in out.iter_mut().zip(column) {
            *product *= value - mean;
        }
    }
    Ok(())
}

fn build_interaction_f64(
    matrix: &CpuPrecisionMatrix,
    combo: &[u32],
    out: &mut Vec<f64>,
) -> OrchestratorResult<()> {
    out.clear();
    out.resize(matrix.rows() as usize, 1.0f64);
    for &feature in combo {
        let column = matrix
            .column_f64(feature as usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "f64 interaction requested from non-f64 CPU matrix",
            ))?;
        let mean =
            matrix
                .column_mean_f64(feature as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "f64 interaction requested from non-f64 CPU matrix",
                ))?;
        for (product, &value) in out.iter_mut().zip(column) {
            *product *= value - mean;
        }
    }
    Ok(())
}

fn score_f32(
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    out: &mut Vec<f32>,
    fixed_mi: &mut FixedMiScratch,
) {
    out.clear();
    out.reserve(metrics.len());
    let mut cached_pearson = None;
    for &metric in metrics {
        let value = match metric {
            MetricKernel::Pearson => {
                *cached_pearson.get_or_insert_with(|| pearson_f32(signal, target))
            }
            MetricKernel::Spearman => spearman_f32(signal, target),
            MetricKernel::MutualInfo => {
                if mi_approximate {
                    mutual_info_fixed_f32_with_scratch(signal, target, mi_bins, fixed_mi)
                } else {
                    mutual_info_f32(signal, target, mi_bins)
                }
            }
            MetricKernel::R2 => {
                let corr = *cached_pearson.get_or_insert_with(|| pearson_f32(signal, target));
                finalize_r2_f32(corr)
            }
        };
        out.push(value);
    }
}

fn score_mixed(
    signal: &[f32],
    target: &[f32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    out: &mut Vec<f64>,
    fixed_mi: &mut FixedMiScratch,
) {
    out.clear();
    out.reserve(metrics.len());
    let mut cached_pearson = None;
    for &metric in metrics {
        let value = match metric {
            MetricKernel::Pearson => {
                *cached_pearson.get_or_insert_with(|| pearson_mixed(signal, target))
            }
            MetricKernel::Spearman => spearman_mixed(signal, target),
            MetricKernel::MutualInfo => {
                if mi_approximate {
                    mutual_info_fixed_mixed_with_scratch(signal, target, mi_bins, fixed_mi)
                } else {
                    mutual_info_mixed(signal, target, mi_bins)
                }
            }
            MetricKernel::R2 => {
                let corr = *cached_pearson.get_or_insert_with(|| pearson_mixed(signal, target));
                crate::simd::finalize_r2_f64(corr)
            }
        };
        out.push(value);
    }
}

fn score_f64(
    signal: &[f64],
    target: &[f64],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    out: &mut Vec<f64>,
    fixed_mi: &mut FixedMiScratch,
) {
    out.clear();
    out.reserve(metrics.len());
    let mut cached_pearson = None;
    for &metric in metrics {
        let value = match metric {
            MetricKernel::Pearson => {
                *cached_pearson.get_or_insert_with(|| pearson_f64(signal, target))
            }
            MetricKernel::Spearman => spearman_f64(signal, target),
            MetricKernel::MutualInfo => {
                if mi_approximate {
                    mutual_info_fixed_f64_with_scratch(signal, target, mi_bins, fixed_mi)
                } else {
                    mutual_info_f64(signal, target, mi_bins)
                }
            }
            MetricKernel::R2 => {
                let corr = *cached_pearson.get_or_insert_with(|| pearson_f64(signal, target));
                crate::simd::finalize_r2_f64(corr)
            }
        };
        out.push(value);
    }
}

fn finalize_r2_f32(correlation: f32) -> f32 {
    if correlation.is_finite() {
        (correlation * correlation).clamp(0.0, 1.0)
    } else {
        f32::NAN
    }
}

/// Full fp32 Pearson: sums, centering, normalization, and public result all
/// stay binary32. The dedicated SIMD routine is physically separate from the
/// historical stable reduction ladder and never widens an arithmetic lane.
pub fn pearson_f32(x: &[f32], y: &[f32]) -> f32 {
    crate::simd::pearson_corr_f32(x, y)
}

/// Mixed Pearson uses binary32 inputs and interactions, then widens exactly at
/// the reduction boundary and keeps the binary64 result public.
pub fn pearson_mixed(x: &[f32], y: &[f32]) -> f64 {
    let sums = crate::simd::pearson_sums(x, y);
    if sums.n == 0 {
        return 0.0;
    }
    crate::simd::finalize_correlation_f64(sums.sxx, sums.syy, sums.sxy)
}

/// Full fp64 Pearson with no f32 conversion on any numeric input, intermediate,
/// or result path. The Core SIMD dispatch loads and reduces f64 lanes directly.
pub fn pearson_f64(x: &[f64], y: &[f64]) -> f64 {
    crate::simd::pearson_corr_f64(x, y)
}

pub fn spearman_f32(x: &[f32], y: &[f32]) -> f32 {
    let (x_values, y_values) = finite_pairs_f32(x, y);
    if x_values.is_empty() {
        return 0.0;
    }
    let x_positions = rank_positions_twice(&x_values);
    let y_positions = rank_positions_twice(&y_values);
    let x_ranks = x_positions
        .into_iter()
        .map(|position| position as f32 * 0.5)
        .collect::<Vec<_>>();
    let y_ranks = y_positions
        .into_iter()
        .map(|position| position as f32 * 0.5)
        .collect::<Vec<_>>();
    pearson_f32(&x_ranks, &y_ranks)
}

pub fn spearman_mixed(x: &[f32], y: &[f32]) -> f64 {
    let (x_values, y_values) = finite_pairs_f32(x, y);
    if x_values.is_empty() {
        return 0.0;
    }
    let x_positions = rank_positions_twice(&x_values);
    let y_positions = rank_positions_twice(&y_values);
    let x_ranks = x_positions
        .into_iter()
        .map(|position| position as f64 * 0.5)
        .collect::<Vec<_>>();
    let y_ranks = y_positions
        .into_iter()
        .map(|position| position as f64 * 0.5)
        .collect::<Vec<_>>();
    pearson_f64(&x_ranks, &y_ranks)
}

pub fn spearman_f64(x: &[f64], y: &[f64]) -> f64 {
    let (x_values, y_values) = finite_pairs_f64(x, y);
    if x_values.is_empty() {
        return 0.0;
    }
    let x_positions = rank_positions_twice(&x_values);
    let y_positions = rank_positions_twice(&y_values);
    let x_ranks = x_positions
        .into_iter()
        .map(|position| position as f64 * 0.5)
        .collect::<Vec<_>>();
    let y_ranks = y_positions
        .into_iter()
        .map(|position| position as f64 * 0.5)
        .collect::<Vec<_>>();
    pearson_f64(&x_ranks, &y_ranks)
}

pub fn mutual_info_f32(x: &[f32], y: &[f32], max_bins: u32) -> f32 {
    let (x_values, y_values) = finite_pairs_f32(x, y);
    if x_values.len() <= 1 || constant_f32(&x_values) || constant_f32(&y_values) {
        return 0.0;
    }
    let bins = select_adaptive_mi_bins(x_values.len(), max_bins);
    let (x_bins, x_count) = adaptive_bin_indices(&x_values, bins, true);
    let (y_bins, y_count) = adaptive_bin_indices(&y_values, bins, true);
    if x_count < 2 || y_count < 2 {
        return 0.0;
    }
    let mut joint = vec![0u32; x_count * y_count];
    for (&x_bin, &y_bin) in x_bins.iter().zip(&y_bins) {
        joint[x_bin * y_count + y_bin] += 1;
    }
    corrected_mi_f32(&joint, x_count, y_count, false)
}

pub fn mutual_info_mixed(x: &[f32], y: &[f32], max_bins: u32) -> f64 {
    let (x_values, y_values) = finite_pairs_f32(x, y);
    if x_values.len() <= 1 || constant_f32(&x_values) || constant_f32(&y_values) {
        return 0.0;
    }
    let bins = select_adaptive_mi_bins(x_values.len(), max_bins);
    let (x_bins, x_count) = adaptive_bin_indices(&x_values, bins, true);
    let (y_bins, y_count) = adaptive_bin_indices(&y_values, bins, true);
    if x_count < 2 || y_count < 2 {
        return 0.0;
    }
    let mut joint = vec![0u32; x_count * y_count];
    for (&x_bin, &y_bin) in x_bins.iter().zip(&y_bins) {
        joint[x_bin * y_count + y_bin] += 1;
    }
    corrected_mi_f64(&joint, x_count, y_count, false)
}

pub fn mutual_info_f64(x: &[f64], y: &[f64], max_bins: u32) -> f64 {
    let (x_values, y_values) = finite_pairs_f64(x, y);
    if x_values.len() <= 1 || constant_f64(&x_values) || constant_f64(&y_values) {
        return 0.0;
    }
    let bins = select_adaptive_mi_bins(x_values.len(), max_bins);
    let (x_bins, x_count) = adaptive_bin_indices(&x_values, bins, true);
    let (y_bins, y_count) = adaptive_bin_indices(&y_values, bins, true);
    if x_count < 2 || y_count < 2 {
        return 0.0;
    }
    let mut joint = vec![0u32; x_count * y_count];
    for (&x_bin, &y_bin) in x_bins.iter().zip(&y_bins) {
        joint[x_bin * y_count + y_bin] += 1;
    }
    corrected_mi_f64(&joint, x_count, y_count, false)
}

pub fn mutual_info_fixed_f32(x: &[f32], y: &[f32], bins: u32) -> f32 {
    let mut scratch = FixedMiScratch::default();
    mutual_info_fixed_f32_with_scratch(x, y, bins, &mut scratch)
}

fn mutual_info_fixed_f32_with_scratch(
    x: &[f32],
    y: &[f32],
    bins: u32,
    scratch: &mut FixedMiScratch,
) -> f32 {
    let bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, bins) as usize;
    let FixedMiScratch {
        finite_f32_x,
        finite_f32_y,
        hist_x,
        hist_y,
        joint,
        ..
    } = scratch;
    finite_pairs_f32_into(x, y, finite_f32_x, finite_f32_y);
    if finite_f32_x.len() <= 1 {
        return 0.0;
    }
    let (min_x, max_x) = min_max_f32(finite_f32_x);
    let (min_y, max_y) = min_max_f32(finite_f32_y);
    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }
    let inv_x = bins as f32 / (max_x - min_x);
    let inv_y = bins as f32 / (max_y - min_y);
    resize_fixed_histograms(hist_x, hist_y, joint, bins);
    crate::simd::fixed_bin_histogram2d(
        finite_f32_x,
        finite_f32_y,
        min_x,
        inv_x,
        min_y,
        inv_y,
        bins as u32,
        hist_x,
        hist_y,
        joint,
    );
    corrected_mi_f32_with_marginals(joint, hist_x, hist_y, true)
}

pub fn mutual_info_fixed_mixed(x: &[f32], y: &[f32], bins: u32) -> f64 {
    let mut scratch = FixedMiScratch::default();
    mutual_info_fixed_mixed_with_scratch(x, y, bins, &mut scratch)
}

fn mutual_info_fixed_mixed_with_scratch(
    x: &[f32],
    y: &[f32],
    bins: u32,
    scratch: &mut FixedMiScratch,
) -> f64 {
    let bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, bins) as usize;
    let FixedMiScratch {
        finite_f32_x,
        finite_f32_y,
        hist_x,
        hist_y,
        joint,
        ..
    } = scratch;
    finite_pairs_f32_into(x, y, finite_f32_x, finite_f32_y);
    if finite_f32_x.len() <= 1 {
        return 0.0;
    }
    let (min_x, max_x) = min_max_f32(finite_f32_x);
    let (min_y, max_y) = min_max_f32(finite_f32_y);
    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }
    // Binning is pointwise arithmetic, so mixed intentionally retains f32
    // bounds and mapping before its f64 probability/logarithm phase.
    let inv_x = bins as f32 / (max_x - min_x);
    let inv_y = bins as f32 / (max_y - min_y);
    resize_fixed_histograms(hist_x, hist_y, joint, bins);
    crate::simd::fixed_bin_histogram2d(
        finite_f32_x,
        finite_f32_y,
        min_x,
        inv_x,
        min_y,
        inv_y,
        bins as u32,
        hist_x,
        hist_y,
        joint,
    );
    corrected_mi_f64_with_marginals(joint, hist_x, hist_y, true)
}

pub fn mutual_info_fixed_f64(x: &[f64], y: &[f64], bins: u32) -> f64 {
    let mut scratch = FixedMiScratch::default();
    mutual_info_fixed_f64_with_scratch(x, y, bins, &mut scratch)
}

fn mutual_info_fixed_f64_with_scratch(
    x: &[f64],
    y: &[f64],
    bins: u32,
    scratch: &mut FixedMiScratch,
) -> f64 {
    let bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, bins) as usize;
    let FixedMiScratch {
        finite_f64_x,
        finite_f64_y,
        hist_x,
        hist_y,
        joint,
        ..
    } = scratch;
    finite_pairs_f64_into(x, y, finite_f64_x, finite_f64_y);
    if finite_f64_x.len() <= 1 {
        return 0.0;
    }
    let (min_x, max_x) = min_max_f64(finite_f64_x);
    let (min_y, max_y) = min_max_f64(finite_f64_y);
    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }
    let inv_x = bins as f64 / (max_x - min_x);
    let inv_y = bins as f64 / (max_y - min_y);
    resize_fixed_histograms(hist_x, hist_y, joint, bins);
    clear_fixed_histograms(hist_x, hist_y, joint);
    for (&x_value, &y_value) in finite_f64_x.iter().zip(finite_f64_y.iter()) {
        let x_bin = fixed_bin_f64((x_value - min_x) * inv_x, bins) as usize;
        let y_bin = fixed_bin_f64((y_value - min_y) * inv_y, bins) as usize;
        hist_x[x_bin] += 1;
        hist_y[y_bin] += 1;
        joint[x_bin * bins + y_bin] += 1;
    }
    corrected_mi_f64_with_marginals(joint, hist_x, hist_y, true)
}

fn resize_fixed_histograms(
    hist_x: &mut Vec<u32>,
    hist_y: &mut Vec<u32>,
    joint: &mut Vec<u32>,
    bins: usize,
) {
    hist_x.resize(bins, 0);
    hist_y.resize(bins, 0);
    joint.resize(
        bins.checked_mul(bins)
            .expect("sanitized fixed-bin histogram dimensions fit usize"),
        0,
    );
}

fn clear_fixed_histograms(hist_x: &mut [u32], hist_y: &mut [u32], joint: &mut [u32]) {
    hist_x.fill(0);
    hist_y.fill(0);
    joint.fill(0);
}

fn finite_pairs_f32(x: &[f32], y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut out_x = Vec::with_capacity(x.len());
    let mut out_y = Vec::with_capacity(y.len());
    finite_pairs_f32_into(x, y, &mut out_x, &mut out_y);
    (out_x, out_y)
}

fn finite_pairs_f32_into(x: &[f32], y: &[f32], out_x: &mut Vec<f32>, out_y: &mut Vec<f32>) {
    out_x.clear();
    out_y.clear();
    if x.len() != y.len() {
        return;
    }
    out_x.reserve(x.len());
    out_y.reserve(y.len());
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            out_x.push(x_value);
            out_y.push(y_value);
        }
    }
}

fn finite_pairs_f64(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut out_x = Vec::with_capacity(x.len());
    let mut out_y = Vec::with_capacity(y.len());
    finite_pairs_f64_into(x, y, &mut out_x, &mut out_y);
    (out_x, out_y)
}

fn finite_pairs_f64_into(x: &[f64], y: &[f64], out_x: &mut Vec<f64>, out_y: &mut Vec<f64>) {
    out_x.clear();
    out_y.clear();
    if x.len() != y.len() {
        return;
    }
    out_x.reserve(x.len());
    out_y.reserve(y.len());
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            out_x.push(x_value);
            out_y.push(y_value);
        }
    }
}

/// Return twice each average rank.  The rank position itself stays an exact
/// integer (`u64`), including ties; only the covariance phase converts it to
/// the selected reduction dtype.
fn rank_positions_twice<T: PartialOrd + PartialEq>(values: &[T]) -> Vec<u64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(&right))
    });
    let mut positions = vec![0u64; values.len()];
    let mut begin = 0usize;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let rank_twice = begin as u64 + (end - 1) as u64;
        for &row in &order[begin..end] {
            positions[row] = rank_twice;
        }
        begin = end;
    }
    positions
}

fn select_adaptive_mi_bins(n_samples: usize, max_bins: u32) -> u32 {
    let max_bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, max_bins);
    let mut best = 2u32;
    for &level in MI_TEMPLATE_BIN_LEVELS {
        if level > max_bins {
            break;
        }
        let required =
            (MI_SAMPLES_PER_JOINT_BIN as usize).saturating_mul(level.saturating_pow(2) as usize);
        if n_samples >= required {
            best = level;
        }
    }
    best
}

fn adaptive_bin_indices<T: PartialOrd + PartialEq + Copy>(
    values: &[T],
    max_bins: u32,
    exact_low_cardinality: bool,
) -> (Vec<usize>, usize) {
    let n = values.len();
    let max_bins = max_bins.clamp(2, *MI_TEMPLATE_BIN_LEVELS.last().unwrap()) as usize;
    if n == 0 {
        return (Vec::new(), 0);
    }
    let mut unique = values.to_vec();
    unique.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    unique.dedup_by(|left, right| *left == *right);
    if unique.len() <= 1 {
        return (vec![0; n], 1);
    }
    if exact_low_cardinality && unique.len() <= max_bins {
        let bins = values
            .iter()
            .map(|value| {
                unique
                    .binary_search_by(|probe| probe.partial_cmp(value).unwrap_or(Ordering::Equal))
                    .expect("unique values contain every input value")
            })
            .collect();
        return (bins, unique.len());
    }

    let bins = max_bins.min(n);
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(&right))
    });
    let mut out = vec![0usize; n];
    let mut block_start = 0usize;
    let mut previous_raw_bin = None;
    let mut compact_bin = 0usize;
    while block_start < n {
        let mut block_end = block_start + 1;
        while block_end < n && values[order[block_end]] == values[order[block_start]] {
            block_end += 1;
        }
        let midpoint_twice = block_start as u128 + (block_end - 1) as u128;
        let raw_bin = ((midpoint_twice * bins as u128) / (2 * n as u128)) as usize;
        if previous_raw_bin.is_some_and(|previous| previous != raw_bin) {
            compact_bin += 1;
        }
        previous_raw_bin = Some(raw_bin);
        for position in block_start..block_end {
            out[order[position]] = compact_bin;
        }
        block_start = block_end;
    }
    (out, compact_bin + 1)
}

fn corrected_mi_f32(joint: &[u32], row_count: usize, col_count: usize, normalize: bool) -> f32 {
    if row_count == 0 || col_count == 0 || joint.is_empty() {
        return 0.0;
    }
    let mut rows = vec![0u32; row_count];
    let mut cols = vec![0u32; col_count];
    let mut total = 0u32;
    for row in 0..row_count {
        for col in 0..col_count {
            let count = joint[row * col_count + col];
            rows[row] += count;
            cols[col] += count;
            total += count;
        }
    }
    corrected_mi_f32_with_marginals_and_total(joint, &rows, &cols, total, normalize)
}

fn corrected_mi_f32_with_marginals(
    joint: &[u32],
    rows: &[u32],
    cols: &[u32],
    normalize: bool,
) -> f32 {
    let Some(required) = rows.len().checked_mul(cols.len()) else {
        return 0.0;
    };
    if rows.is_empty() || cols.is_empty() || joint.len() < required {
        return 0.0;
    }
    let mut total = 0u32;
    for row in 0..rows.len() {
        for col in 0..cols.len() {
            total += joint[row * cols.len() + col];
        }
    }
    corrected_mi_f32_with_marginals_and_total(joint, rows, cols, total, normalize)
}

fn corrected_mi_f32_with_marginals_and_total(
    joint: &[u32],
    rows: &[u32],
    cols: &[u32],
    total: u32,
    normalize: bool,
) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let nonzero_rows = rows.iter().filter(|&&count| count != 0).count() as u32;
    let nonzero_cols = cols.iter().filter(|&&count| count != 0).count() as u32;
    if nonzero_rows < 2 || nonzero_cols < 2 {
        return 0.0;
    }
    let total_f = total as f32;
    let mut mi = 0.0f32;
    for row in 0..rows.len() {
        for col in 0..cols.len() {
            let count = joint[row * cols.len() + col];
            if count == 0 {
                continue;
            }
            let pxy = count as f32 / total_f;
            let px = rows[row] as f32 / total_f;
            let py = cols[col] as f32 / total_f;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    let correction = ((nonzero_rows - 1) as f32 * (nonzero_cols - 1) as f32) / (2.0 * total_f);
    let corrected = (mi - correction).max(0.0);
    if !normalize {
        return corrected;
    }
    let normalizer = (nonzero_rows.min(nonzero_cols) as f32).ln();
    if normalizer > 0.0 {
        corrected / normalizer
    } else {
        0.0
    }
}

fn corrected_mi_f64(joint: &[u32], row_count: usize, col_count: usize, normalize: bool) -> f64 {
    if row_count == 0 || col_count == 0 || joint.is_empty() {
        return 0.0;
    }
    let mut rows = vec![0u32; row_count];
    let mut cols = vec![0u32; col_count];
    let mut total = 0u32;
    for row in 0..row_count {
        for col in 0..col_count {
            let count = joint[row * col_count + col];
            rows[row] += count;
            cols[col] += count;
            total += count;
        }
    }
    corrected_mi_f64_with_marginals_and_total(joint, &rows, &cols, total, normalize)
}

fn corrected_mi_f64_with_marginals(
    joint: &[u32],
    rows: &[u32],
    cols: &[u32],
    normalize: bool,
) -> f64 {
    let Some(required) = rows.len().checked_mul(cols.len()) else {
        return 0.0;
    };
    if rows.is_empty() || cols.is_empty() || joint.len() < required {
        return 0.0;
    }
    let mut total = 0u32;
    for row in 0..rows.len() {
        for col in 0..cols.len() {
            total += joint[row * cols.len() + col];
        }
    }
    corrected_mi_f64_with_marginals_and_total(joint, rows, cols, total, normalize)
}

fn corrected_mi_f64_with_marginals_and_total(
    joint: &[u32],
    rows: &[u32],
    cols: &[u32],
    total: u32,
    normalize: bool,
) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let nonzero_rows = rows.iter().filter(|&&count| count != 0).count() as u32;
    let nonzero_cols = cols.iter().filter(|&&count| count != 0).count() as u32;
    if nonzero_rows < 2 || nonzero_cols < 2 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut mi = 0.0f64;
    for row in 0..rows.len() {
        for col in 0..cols.len() {
            let count = joint[row * cols.len() + col];
            if count == 0 {
                continue;
            }
            let pxy = count as f64 / total_f;
            let px = rows[row] as f64 / total_f;
            let py = cols[col] as f64 / total_f;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    let correction = ((nonzero_rows - 1) as f64 * (nonzero_cols - 1) as f64) / (2.0 * total_f);
    let corrected = (mi - correction).max(0.0);
    if !normalize {
        return corrected;
    }
    let normalizer = (nonzero_rows.min(nonzero_cols) as f64).ln();
    if normalizer > 0.0 {
        corrected / normalizer
    } else {
        0.0
    }
}

fn min_max_f32(values: &[f32]) -> (f32, f32) {
    let mut min = values[0];
    let mut max = values[0];
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

fn min_max_f64(values: &[f64]) -> (f64, f64) {
    let mut min = values[0];
    let mut max = values[0];
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

#[cfg(test)]
fn fixed_joint_f32(
    x: &[f32],
    y: &[f32],
    min_x: f32,
    inv_x: f32,
    min_y: f32,
    inv_y: f32,
    bins: usize,
) -> Vec<u32> {
    // This standalone helper exists for the direct numerical fixtures. The
    // production executor uses `FixedMiScratch`, retaining these exact-size
    // buffers per Rayon worker instead of allocating them per candidate.
    let mut hist_x = vec![0u32; bins];
    let mut hist_y = vec![0u32; bins];
    let mut joint = vec![0u32; bins * bins];
    crate::simd::fixed_bin_histogram2d(
        x,
        y,
        min_x,
        inv_x,
        min_y,
        inv_y,
        bins as u32,
        &mut hist_x,
        &mut hist_y,
        &mut joint,
    );
    joint
}

#[cfg(test)]
fn fixed_bin_f32(scaled: f32, bins: usize) -> u32 {
    let max_bin = bins.saturating_sub(1) as u32;
    if scaled.is_nan() || scaled <= 0.0 {
        0
    } else if !scaled.is_finite() || scaled >= max_bin as f32 {
        max_bin
    } else {
        scaled as u32
    }
}

fn fixed_bin_f64(scaled: f64, bins: usize) -> u32 {
    let max_bin = bins.saturating_sub(1) as u32;
    if scaled.is_nan() || scaled <= 0.0 {
        0
    } else if !scaled.is_finite() || scaled >= max_bin as f64 {
        max_bin
    } else {
        scaled as u32
    }
}

fn constant_f32(values: &[f32]) -> bool {
    values
        .first()
        .is_none_or(|&first| values.iter().all(|&value| value == first))
}

fn constant_f64(values: &[f64]) -> bool {
    values
        .first()
        .is_none_or(|&first| values.iter().all(|&value| value == first))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pearson_mixed_scalar_oracle(x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let mut n = 0u64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                n += 1;
                sum_x += f64::from(x_value);
                sum_y += f64::from(y_value);
            }
        }
        if n == 0 {
            return 0.0;
        }
        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;
        let mut variance_x = 0.0f64;
        let mut variance_y = 0.0f64;
        let mut covariance = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                let dx = f64::from(x_value) - mean_x;
                let dy = f64::from(y_value) - mean_y;
                variance_x += dx * dx;
                variance_y += dy * dy;
                covariance += dx * dy;
            }
        }
        crate::simd::finalize_correlation_f64(variance_x, variance_y, covariance)
    }

    fn pearson_f64_scalar_oracle(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let mut n = 0u64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                n += 1;
                sum_x += x_value;
                sum_y += y_value;
            }
        }
        if n == 0 {
            return 0.0;
        }
        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;
        let mut variance_x = 0.0f64;
        let mut variance_y = 0.0f64;
        let mut covariance = 0.0f64;
        for (&x_value, &y_value) in x.iter().zip(y) {
            if x_value.is_finite() && y_value.is_finite() {
                let dx = x_value - mean_x;
                let dy = y_value - mean_y;
                variance_x += dx * dx;
                variance_y += dy * dy;
                covariance += dx * dy;
            }
        }
        crate::simd::finalize_correlation_f64(variance_x, variance_y, covariance)
    }

    fn fixed_joint_f32_scalar_oracle(
        x: &[f32],
        y: &[f32],
        min_x: f32,
        inv_x: f32,
        min_y: f32,
        inv_y: f32,
        bins: usize,
    ) -> Vec<u32> {
        let mut joint = vec![0u32; bins * bins];
        for (&x_value, &y_value) in x.iter().zip(y) {
            let x_bin = fixed_bin_f32((x_value - min_x) * inv_x, bins) as usize;
            let y_bin = fixed_bin_f32((y_value - min_y) * inv_y, bins) as usize;
            joint[x_bin * bins + y_bin] += 1;
        }
        joint
    }

    fn matrix_f32(
        profile: PrecisionProfile,
        columns: &[Vec<f32>],
        target: Vec<f32>,
    ) -> CpuPrecisionMatrix {
        let rows = target.len();
        let mut features = Vec::with_capacity(rows * columns.len());
        for row in 0..rows {
            for column in columns {
                features.push(column[row]);
            }
        }
        CpuPrecisionMatrix::from_row_major_f32(
            profile,
            rows as u64,
            columns.len() as u32,
            features,
            target,
        )
        .unwrap()
    }

    #[test]
    fn fp32_scores_and_results_are_binary32_without_f64_reduction() {
        let matrix = matrix_f32(
            PrecisionProfile::Fp32,
            &[vec![16_777_216.0, 1.0, -16_777_216.0, 2.0]],
            vec![1.0, 0.0, -1.0, 3.0],
        );
        let scores = score_precision_continuous_combo(
            &matrix,
            &[0],
            &[
                MetricKernel::Pearson,
                MetricKernel::R2,
                MetricKernel::Spearman,
                MetricKernel::MutualInfo,
            ],
            2,
            true,
        )
        .unwrap();
        assert!(matches!(scores, CpuPrecisionValues::F32(_)));
        let CpuPrecisionValues::F32(values) = scores else {
            unreachable!()
        };
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(
            values[0].to_bits(),
            pearson_f32(matrix.column_f32(0).unwrap(), matrix.target_f32().unwrap()).to_bits()
        );
    }

    #[test]
    fn mixed_pearson_simd_matches_independent_scalar_f64_oracle() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..100usize {
            let len = 257 + case;
            let mut x = Vec::with_capacity(len);
            let mut y = Vec::with_capacity(len);
            for row in 0..len {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 40) as i32 - (1 << 23)) as f32 * 1.0e-7;
                let value = (row as f32 * 0.013_7).sin() + noise;
                x.push(value);
                y.push((value * 0.71 + (row as f32 * 0.009_1).cos()).sin());
            }
            let expected = pearson_mixed_scalar_oracle(&x, &y);
            let actual = pearson_mixed(&x, &y);
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "case={case} expected={expected:.17e} actual={actual:.17e}"
            );
        }
    }

    #[test]
    fn mixed_pearson_simd_preserves_scalar_edge_classification() {
        let cases = [
            (vec![], vec![]),
            (vec![1.0, 2.0], vec![1.0]),
            (vec![2.0; 33], (0..33).map(|value| value as f32).collect()),
            (vec![f32::NAN; 33], vec![1.0; 33]),
            (
                vec![1.0, 2.0, f32::NAN, 4.0, f32::INFINITY, 6.0],
                vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ),
        ];
        for (x, y) in cases {
            let expected = pearson_mixed_scalar_oracle(&x, &y);
            let actual = pearson_mixed(&x, &y);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn fp64_pearson_simd_matches_independent_scalar_oracle_across_tails() {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for case in 0..100usize {
            let len = 257 + case;
            let mut x = Vec::with_capacity(len);
            let mut y = Vec::with_capacity(len);
            for row in 0..len {
                state = state
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .wrapping_add(3_037_000_493);
                let noise = ((state >> 12) as i64 as f64) * 1.0e-20;
                let value = (row as f64 * 0.013_7).sin() + noise;
                x.push(value);
                y.push((value * 0.71 + (row as f64 * 0.009_1).cos()).sin());
            }
            let expected = pearson_f64_scalar_oracle(&x, &y);
            let actual = pearson_f64(&x, &y);
            assert!(
                (actual - expected).abs() <= crate::simd::FP64_SIMD_REGROUPING_TOLERANCE,
                "case={case} expected={expected:.17e} actual={actual:.17e}"
            );
            let expected_r2 = crate::simd::finalize_r2_f64(expected);
            let actual_r2 = crate::simd::finalize_r2_f64(actual);
            assert!(
                (actual_r2 - expected_r2).abs() <= crate::simd::FP64_SIMD_REGROUPING_TOLERANCE,
                "case={case} expected_r2={expected_r2:.17e} actual_r2={actual_r2:.17e}"
            );
        }
    }

    #[test]
    fn mixed_and_fp64_spearman_use_binary64_covariance_for_long_tied_ranks() {
        let x_f32 = (0..1027)
            .map(|row| ((row * 37) % 29) as f32)
            .collect::<Vec<_>>();
        let y_f32 = (0..1027)
            .map(|row| ((row * 19 + row / 7) % 31) as f32)
            .collect::<Vec<_>>();
        let x_positions = rank_positions_twice(&x_f32);
        let y_positions = rank_positions_twice(&y_f32);
        let x_ranks = x_positions
            .into_iter()
            .map(|position| position as f64 * 0.5)
            .collect::<Vec<_>>();
        let y_ranks = y_positions
            .into_iter()
            .map(|position| position as f64 * 0.5)
            .collect::<Vec<_>>();
        let expected = pearson_f64_scalar_oracle(&x_ranks, &y_ranks);

        let mixed = spearman_mixed(&x_f32, &y_f32);
        let x_f64 = x_f32
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let y_f64 = y_f32
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let fp64 = spearman_f64(&x_f64, &y_f64);
        assert!((mixed - expected).abs() <= crate::simd::FP64_SIMD_REGROUPING_TOLERANCE);
        assert!((fp64 - expected).abs() <= crate::simd::FP64_SIMD_REGROUPING_TOLERANCE);
    }

    #[test]
    fn mixed_materializes_f32_interaction_but_keeps_f64_public_scores() {
        let matrix = matrix_f32(
            PrecisionProfile::Mixed,
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 1.0, 4.0, 3.0, 6.0]],
            vec![0.0, 2.0, 1.0, 4.0, 3.0],
        );
        let mut scratch = PrecisionScoreScratch::new(PrecisionProfile::Mixed);
        let scores = score_precision_continuous_combo_into(
            &matrix,
            &[0, 1],
            &[
                MetricKernel::Pearson,
                MetricKernel::Spearman,
                MetricKernel::MutualInfo,
                MetricKernel::R2,
            ],
            2,
            true,
            &mut scratch,
        )
        .unwrap();
        let CpuPrecisionSlice::F64(values) = scores else {
            unreachable!()
        };
        let score_values = values.to_vec();
        assert_eq!(scratch.interaction_f32.len(), 5);
        assert!(scratch.interaction_f64.is_empty());
        assert_eq!(score_values.len(), 4);
        let expected = pearson_mixed(&scratch.interaction_f32, matrix.target_f32().unwrap());
        assert_eq!(score_values[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn fp64_distinguishes_values_that_collapse_in_fp32_for_all_metrics() {
        let base = 1.0f64;
        // Far enough apart for the independent f64 covariance oracle to avoid
        // cancellation, but still far below one f32 ULP at one.
        let next = f64::from_bits(base.to_bits() + 1024);
        let next_two = f64::from_bits(base.to_bits() + 2048);
        let values = vec![base, next, next_two, f64::from_bits(base.to_bits() + 3072)];
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            4,
            1,
            values.clone(),
            target.clone(),
        )
        .unwrap();
        let scores = score_precision_continuous_combo(
            &matrix,
            &[0],
            &[
                MetricKernel::Pearson,
                MetricKernel::Spearman,
                MetricKernel::MutualInfo,
                MetricKernel::R2,
            ],
            2,
            false,
        )
        .unwrap();
        let CpuPrecisionValues::F64(scores) = scores else {
            panic!("fp64 must expose f64 results")
        };
        assert!((scores[0] - 1.0).abs() < 1.0e-12);
        assert!((scores[1] - 1.0).abs() < 1.0e-12);
        assert!((scores[3] - 1.0).abs() < 1.0e-12);
        assert!(scores[2] > 0.0);
        let collapsed = values
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert!(collapsed.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn profile_dispatch_rejects_mismatched_storage_before_metric_execution() {
        let matrix = matrix_f32(PrecisionProfile::Mixed, &[vec![1.0, 2.0]], vec![1.0, 2.0]);
        assert!(score_precision_signal(
            PrecisionProfile::Fp64,
            matrix.column(0),
            matrix.target(),
            MetricKernel::Pearson,
            2,
            false,
        )
        .is_err());
    }

    #[test]
    fn fp32_mi_keeps_integer_histograms_at_boundary_and_nonfinite_inputs_are_skipped() {
        let x = [f32::MIN_POSITIVE, 0.0, 1.0, 2.0, f32::INFINITY, f32::NAN];
        let y = [0.0, 1.0, 1.0, 0.0, 2.0, 3.0];
        let fp32 = mutual_info_fixed_f32(&x, &y, 2);
        let mixed = mutual_info_fixed_mixed(&x, &y, 2);
        assert!(fp32.is_finite());
        assert!(mixed.is_finite());
        assert_eq!(fixed_bin_f32(f32::NEG_INFINITY, 2), 0);
        assert_eq!(fixed_bin_f32(f32::INFINITY, 2), 1);
    }

    #[test]
    fn mixed_fixed_mi_uses_f32_bin_mapping_at_an_exact_boundary() {
        let minimum = f32::from_bits(0x44e8_3674);
        let boundary = f32::from_bits(0x4500_0abe);
        let maximum = f32::from_bits(0x4553_975a);
        let inverse = 8.0f32 / (maximum - minimum);
        let scaled_f32 = (boundary - minimum) * inverse;

        assert_eq!(scaled_f32.to_bits(), 1.0f32.to_bits());
        assert_eq!(fixed_bin_f32(scaled_f32, 8), 1);

        // Widening the already-stored f32 samples before bin placement would
        // cross the boundary and is therefore observably the wrong contract.
        let promoted_scaled = (f64::from(boundary) - f64::from(minimum))
            * (8.0f64 / (f64::from(maximum) - f64::from(minimum)));
        assert!(promoted_scaled < 1.0);
        assert_eq!(fixed_bin_f64(promoted_scaled, 8), 0);
    }

    #[test]
    fn precision_fixed_mi_histogram_matches_scalar_counts_at_f32_boundaries_and_tail() {
        let minimum = f32::from_bits(0x44e8_3674);
        let boundary = f32::from_bits(0x4500_0abe);
        let maximum = f32::from_bits(0x4553_975a);
        let bins = 8usize;
        let inv_x = bins as f32 / (maximum - minimum);
        let min_y = -1.0f32;
        let inv_y = bins as f32 / 2.0;

        // Seventeen finite pairs intentionally exercises an AVX2 tail after two
        // complete eight-lane blocks. The first three values pin the exact f32
        // lower, interior-boundary, and upper-bin behavior.
        let mut x = vec![minimum, boundary, maximum];
        x.extend((0..14).map(|index| {
            minimum + (maximum - minimum) * ((index % bins) as f32 / (bins - 1) as f32)
        }));
        let y = (0..x.len())
            .map(|index| -1.0f32 + 2.0 * ((index * 3 % bins) as f32 / (bins - 1) as f32))
            .collect::<Vec<_>>();
        assert_eq!(x.len(), 17);
        assert_eq!(y.len(), 17);

        let expected = fixed_joint_f32_scalar_oracle(&x, &y, minimum, inv_x, min_y, inv_y, bins);
        let actual = fixed_joint_f32(&x, &y, minimum, inv_x, min_y, inv_y, bins);
        assert_eq!(actual, expected);
        assert_eq!(actual.iter().sum::<u32>(), x.len() as u32);
        assert_eq!(fixed_bin_f32((boundary - minimum) * inv_x, bins), 1);
    }

    #[test]
    fn precision_fixed_mi_keeps_nonfinite_filtering_and_profile_finalization_after_simd_binning() {
        let bins = 8usize;
        let x = [
            0.0,
            0.125,
            0.25,
            0.5,
            0.75,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.375,
            0.625,
        ];
        let y = [
            1.0,
            0.875,
            0.75,
            0.5,
            0.25,
            0.0,
            0.5,
            0.5,
            0.5,
            f32::NAN,
            0.375,
        ];
        let (finite_x, finite_y) = finite_pairs_f32(&x, &y);
        assert_eq!(finite_x.len(), 7);
        assert_eq!(finite_y.len(), 7);
        let (min_x, max_x) = min_max_f32(&finite_x);
        let (min_y, max_y) = min_max_f32(&finite_y);
        let inv_x = bins as f32 / (max_x - min_x);
        let inv_y = bins as f32 / (max_y - min_y);
        let expected_joint =
            fixed_joint_f32_scalar_oracle(&finite_x, &finite_y, min_x, inv_x, min_y, inv_y, bins);
        let actual_joint = fixed_joint_f32(&finite_x, &finite_y, min_x, inv_x, min_y, inv_y, bins);
        assert_eq!(actual_joint, expected_joint);
        assert_eq!(actual_joint.iter().sum::<u32>(), finite_x.len() as u32);

        let expected_fp32 = corrected_mi_f32(&expected_joint, bins, bins, true);
        let expected_mixed = corrected_mi_f64(&expected_joint, bins, bins, true);
        assert_eq!(
            mutual_info_fixed_f32(&x, &y, bins as u32).to_bits(),
            expected_fp32.to_bits()
        );
        assert_eq!(
            mutual_info_fixed_mixed(&x, &y, bins as u32).to_bits(),
            expected_mixed.to_bits()
        );
    }

    #[test]
    fn production_fixed_mi_reuses_worker_owned_buffers_across_candidates() {
        let x = (0..257)
            .map(|row| ((row * 17) % 101) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let y = (0..257)
            .map(|row| ((row * 29 + 7) % 97) as f32 * 0.0625)
            .collect::<Vec<_>>();
        let mut scratch = FixedMiScratch::default();

        let first_fp32 = mutual_info_fixed_f32_with_scratch(&x, &y, 32, &mut scratch);
        let f32_pointers = (
            scratch.finite_f32_x.as_ptr(),
            scratch.finite_f32_y.as_ptr(),
            scratch.hist_x.as_ptr(),
            scratch.hist_y.as_ptr(),
            scratch.joint.as_ptr(),
        );
        let f32_capacities = (
            scratch.finite_f32_x.capacity(),
            scratch.finite_f32_y.capacity(),
            scratch.hist_x.capacity(),
            scratch.hist_y.capacity(),
            scratch.joint.capacity(),
        );
        let second_fp32 = mutual_info_fixed_f32_with_scratch(&y, &x, 32, &mut scratch);
        assert_eq!(
            f32_pointers,
            (
                scratch.finite_f32_x.as_ptr(),
                scratch.finite_f32_y.as_ptr(),
                scratch.hist_x.as_ptr(),
                scratch.hist_y.as_ptr(),
                scratch.joint.as_ptr(),
            )
        );
        assert_eq!(
            f32_capacities,
            (
                scratch.finite_f32_x.capacity(),
                scratch.finite_f32_y.capacity(),
                scratch.hist_x.capacity(),
                scratch.hist_y.capacity(),
                scratch.joint.capacity(),
            )
        );
        assert_eq!(
            first_fp32.to_bits(),
            mutual_info_fixed_f32(&x, &y, 32).to_bits()
        );
        assert_eq!(
            second_fp32.to_bits(),
            mutual_info_fixed_f32(&y, &x, 32).to_bits()
        );

        let x64 = x.iter().map(|&value| f64::from(value)).collect::<Vec<_>>();
        let y64 = y.iter().map(|&value| f64::from(value)).collect::<Vec<_>>();
        let first_fp64 = mutual_info_fixed_f64_with_scratch(&x64, &y64, 32, &mut scratch);
        let f64_pointers = (
            scratch.finite_f64_x.as_ptr(),
            scratch.finite_f64_y.as_ptr(),
            scratch.hist_x.as_ptr(),
            scratch.hist_y.as_ptr(),
            scratch.joint.as_ptr(),
        );
        let second_fp64 = mutual_info_fixed_f64_with_scratch(&y64, &x64, 32, &mut scratch);
        assert_eq!(
            f64_pointers,
            (
                scratch.finite_f64_x.as_ptr(),
                scratch.finite_f64_y.as_ptr(),
                scratch.hist_x.as_ptr(),
                scratch.hist_y.as_ptr(),
                scratch.joint.as_ptr(),
            )
        );
        assert_eq!(
            first_fp64.to_bits(),
            mutual_info_fixed_f64(&x64, &y64, 32).to_bits()
        );
        assert_eq!(
            second_fp64.to_bits(),
            mutual_info_fixed_f64(&y64, &x64, 32).to_bits()
        );
    }

    #[test]
    fn correlation_finalization_preserves_arithmetic_failure_as_nan() {
        let x_f32 = [1.0e20f32, -1.0e20, 1.0e20, -1.0e20];
        let y_f32 = [1.0f32, -1.0, -1.0, 1.0];
        let fp32 = pearson_f32(&x_f32, &y_f32);
        assert!(fp32.is_nan());
        assert!(finalize_r2_f32(fp32).is_nan());

        let x_f64 = [1.0e200f64, -1.0e200, 1.0e200, -1.0e200];
        let y_f64 = [1.0f64, -1.0, -1.0, 1.0];
        let fp64 = pearson_f64(&x_f64, &y_f64);
        assert!(fp64.is_nan());
        assert!(crate::simd::finalize_r2_f64(fp64).is_nan());

        assert!(crate::simd::finalize_correlation_f64(f64::INFINITY, 1.0, 0.0).is_nan());
        assert_eq!(pearson_mixed(&[2.0, 2.0], &[1.0, 3.0]), 0.0);
        assert_eq!(pearson_f32(&[2.0, 2.0], &[1.0, 3.0]), 0.0);
        assert_eq!(pearson_f64(&[2.0, 2.0], &[1.0, 3.0]), 0.0);
    }

    #[test]
    fn fixed_mi_profiles_preserve_the_gpu_compatible_normalized_estimator() {
        let x = (0..64).map(|index| (index % 8) as f32).collect::<Vec<_>>();
        let y = (0..64)
            .map(|index| ((index / 2) % 8) as f32)
            .collect::<Vec<_>>();
        let legacy = crate::kernels::mutual_info_fixed(&x, &y, 8);
        let fp32 = mutual_info_fixed_f32(&x, &y, 8);
        let mixed = mutual_info_fixed_mixed(&x, &y, 8);
        let x_f64 = x.iter().map(|&value| value as f64).collect::<Vec<_>>();
        let y_f64 = y.iter().map(|&value| value as f64).collect::<Vec<_>>();
        let fp64 = mutual_info_fixed_f64(&x_f64, &y_f64, 8);

        assert!(legacy > 0.0 && legacy <= 1.0);
        assert!((fp32 - legacy).abs() <= 2.0e-6);
        assert!((mixed - f64::from(legacy)).abs() <= f64::from(f32::EPSILON));
        assert!((fp64 - mixed).abs() <= f64::EPSILON);
    }

    #[test]
    fn spearman_rank_positions_remain_integer_until_selected_reduction() {
        let ranks = rank_positions_twice(&[1.0f32, 1.0, 2.0, 5.0]);
        assert_eq!(ranks, vec![1, 1, 4, 6]);
        assert_eq!(
            spearman_f32(&[1.0, 1.0, 2.0, 5.0], &[4.0, 4.0, 3.0, 2.0]),
            -1.0
        );
        assert_eq!(
            spearman_mixed(&[1.0, 1.0, 2.0, 5.0], &[4.0, 4.0, 3.0, 2.0]),
            -1.0
        );
    }
}
