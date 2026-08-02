//! Decision-path (GBDT-method) split finding for the `decision_path` family.
//!
//! Core primitive: the CART/GBDT variance-reduction best-split of a feature vs
//! the target (or residual). A decision_path candidate is a conjunction of such
//! splits (a root→leaf path); its materialized feature is the membership
//! indicator of that region, scored by the continuous engine. depth-k recursion
//! and residual boosting build on `best_variance_split`.

use core::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

use gafime_orchestrator::{OrchestratorError, OrchestratorResult};
use gafime_types::PrecisionProfile;

use crate::precision::{CpuPrecisionScalar, CpuPrecisionSlice, CpuPrecisionValues};

/// A single threshold split and its variance-reduction gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    pub threshold: f32,
    pub gain: f32,
}

/// Precision-aware decision-path split.  Its feature index is structural; its
/// threshold and gain use the profile's numeric execution contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecisionSplit {
    pub threshold: CpuPrecisionScalar,
    pub gain: CpuPrecisionScalar,
}

/// A precision-aware path predicate.  The threshold has the storage/pointwise
/// dtype (f32 for fp32 and mixed; f64 for fp64), rather than a global f64
/// planning scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecisionPathNode {
    pub feature: u32,
    pub threshold: CpuPrecisionScalar,
    pub sign: SplitSign,
}

/// Find the best variance-reduction split using the requested lane's numeric
/// contract.  Mixed uses f32 resident feature/target values and f32 comparison
/// thresholds, then uses f64 only for its variance/reduction statistics.
pub fn best_variance_split_precision(
    profile: PrecisionProfile,
    feature: CpuPrecisionSlice<'_>,
    target: CpuPrecisionSlice<'_>,
) -> OrchestratorResult<Option<PrecisionSplit>> {
    match (profile, feature, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(feature),
            CpuPrecisionSlice::F32(target),
        ) => Ok(
            best_variance_split_f32(feature, target).map(|(threshold, gain)| PrecisionSplit {
                threshold: CpuPrecisionScalar::F32(threshold),
                gain: CpuPrecisionScalar::F32(gain),
            }),
        ),
        (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(feature),
            CpuPrecisionSlice::F32(target),
        ) => Ok(
            best_variance_split_mixed(feature, target).map(|(threshold, gain)| PrecisionSplit {
                threshold: CpuPrecisionScalar::F32(threshold),
                gain: CpuPrecisionScalar::F64(gain),
            }),
        ),
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(feature),
            CpuPrecisionSlice::F64(target),
        ) => Ok(
            best_variance_split_f64(feature, target).map(|(threshold, gain)| PrecisionSplit {
                threshold: CpuPrecisionScalar::F64(threshold),
                gain: CpuPrecisionScalar::F64(gain),
            }),
        ),
        _ => Err(OrchestratorError::InvalidPlan(
            "decision-path storage dtype does not match the requested precision profile",
        )),
    }
}

/// Materialize profile-typed hard-AND path membership.  Feature indices and
/// path ordering are untouched; only numeric comparisons and the materialized
/// membership vector use the selected storage/pointwise dtype.
pub fn path_membership_precision(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    nodes: &[PrecisionPathNode],
) -> OrchestratorResult<CpuPrecisionValues> {
    match (profile, columns) {
        (PrecisionProfile::Fp32, CpuPrecisionSlice::F32(columns))
        | (PrecisionProfile::Mixed, CpuPrecisionSlice::F32(columns)) => {
            let mut out = vec![1.0f32; rows];
            for (row, output) in out.iter_mut().enumerate().take(rows) {
                let mut member = true;
                let mut undetermined = false;
                for node in nodes {
                    let CpuPrecisionScalar::F32(threshold) = node.threshold else {
                        return Err(OrchestratorError::InvalidPlan(
                            "f32 decision-path membership received an f64 threshold",
                        ));
                    };
                    let value = columns
                        .get(node.feature as usize * rows + row)
                        .copied()
                        .ok_or(OrchestratorError::InvalidPlan(
                            "decision-path feature index exceeds column storage",
                        ))?;
                    if value.is_nan() {
                        undetermined = true;
                        continue;
                    }
                    let holds = match node.sign {
                        SplitSign::Le => value <= threshold,
                        SplitSign::Gt => value > threshold,
                    };
                    if !holds {
                        member = false;
                    }
                }
                *output = if !member {
                    0.0
                } else if undetermined {
                    f32::NAN
                } else {
                    1.0
                };
            }
            Ok(CpuPrecisionValues::F32(out))
        }
        (PrecisionProfile::Fp64, CpuPrecisionSlice::F64(columns)) => {
            let mut out = vec![1.0f64; rows];
            for (row, output) in out.iter_mut().enumerate().take(rows) {
                let mut member = true;
                let mut undetermined = false;
                for node in nodes {
                    let CpuPrecisionScalar::F64(threshold) = node.threshold else {
                        return Err(OrchestratorError::InvalidPlan(
                            "fp64 decision-path membership received an f32 threshold",
                        ));
                    };
                    let value = columns
                        .get(node.feature as usize * rows + row)
                        .copied()
                        .ok_or(OrchestratorError::InvalidPlan(
                            "decision-path feature index exceeds column storage",
                        ))?;
                    if value.is_nan() {
                        undetermined = true;
                        continue;
                    }
                    let holds = match node.sign {
                        SplitSign::Le => value <= threshold,
                        SplitSign::Gt => value > threshold,
                    };
                    if !holds {
                        member = false;
                    }
                }
                *output = if !member {
                    0.0
                } else if undetermined {
                    f64::NAN
                } else {
                    1.0
                };
            }
            Ok(CpuPrecisionValues::F64(out))
        }
        _ => Err(OrchestratorError::InvalidPlan(
            "decision-path storage dtype does not match the requested precision profile",
        )),
    }
}

fn best_variance_split_f32(feature: &[f32], target: &[f32]) -> Option<(f32, f32)> {
    let mut pairs = feature
        .iter()
        .copied()
        .zip(target.iter().copied())
        .filter(|(feature, target)| feature.is_finite() && target.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let total = pairs.len() as f32;
    let sum_all = pairs.iter().map(|pair| pair.1).sum::<f32>();
    let sum2_all = pairs.iter().map(|pair| pair.1 * pair.1).sum::<f32>();
    if !sum_all.is_finite() || !sum2_all.is_finite() {
        return None;
    }
    let parent_variance = sum2_all / total - (sum_all / total) * (sum_all / total);
    if !parent_variance.is_finite() || parent_variance <= 0.0 {
        return None;
    }
    let mut left_sum = 0.0f32;
    let mut left_sum2 = 0.0f32;
    let mut best = None;
    for index in 0..pairs.len() - 1 {
        left_sum += pairs[index].1;
        left_sum2 += pairs[index].1 * pairs[index].1;
        if pairs[index].0 == pairs[index + 1].0 {
            continue;
        }
        if !left_sum.is_finite() || !left_sum2.is_finite() {
            continue;
        }
        let n_left = (index + 1) as f32;
        let n_right = total - n_left;
        let Some(left_variance) = finite_nonnegative_path_stat(
            left_sum2 / n_left - (left_sum / n_left) * (left_sum / n_left),
        ) else {
            continue;
        };
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        if !right_sum.is_finite() || !right_sum2.is_finite() {
            continue;
        }
        let Some(right_variance) = finite_nonnegative_path_stat(
            right_sum2 / n_right - (right_sum / n_right) * (right_sum / n_right),
        ) else {
            continue;
        };
        let weighted = (n_left * left_variance + n_right * right_variance) / total;
        let gain = parent_variance - weighted;
        if !weighted.is_finite() || !gain.is_finite() {
            continue;
        }
        let threshold = pairs[index].0 * 0.5 + pairs[index + 1].0 * 0.5;
        if best.is_none_or(|(_, current_gain)| gain > current_gain) {
            best = Some((threshold, gain));
        }
    }
    best
}

fn best_variance_split_mixed(feature: &[f32], target: &[f32]) -> Option<(f32, f64)> {
    let mut pairs = feature
        .iter()
        .copied()
        .zip(target.iter().copied())
        .filter(|(feature, target)| feature.is_finite() && target.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let total = pairs.len() as f64;
    let sum_all = pairs.iter().map(|pair| pair.1 as f64).sum::<f64>();
    let sum2_all = pairs
        .iter()
        .map(|pair| {
            let target = pair.1 as f64;
            target * target
        })
        .sum::<f64>();
    if !sum_all.is_finite() || !sum2_all.is_finite() {
        return None;
    }
    let parent_variance = sum2_all / total - (sum_all / total) * (sum_all / total);
    if !parent_variance.is_finite() || parent_variance <= 0.0 {
        return None;
    }
    let mut left_sum = 0.0f64;
    let mut left_sum2 = 0.0f64;
    let mut best = None;
    for index in 0..pairs.len() - 1 {
        let target = pairs[index].1 as f64;
        left_sum += target;
        left_sum2 += target * target;
        if pairs[index].0 == pairs[index + 1].0 {
            continue;
        }
        if !left_sum.is_finite() || !left_sum2.is_finite() {
            continue;
        }
        let n_left = (index + 1) as f64;
        let n_right = total - n_left;
        let Some(left_variance) = finite_nonnegative_path_stat(
            left_sum2 / n_left - (left_sum / n_left) * (left_sum / n_left),
        ) else {
            continue;
        };
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        if !right_sum.is_finite() || !right_sum2.is_finite() {
            continue;
        }
        let Some(right_variance) = finite_nonnegative_path_stat(
            right_sum2 / n_right - (right_sum / n_right) * (right_sum / n_right),
        ) else {
            continue;
        };
        let weighted = (n_left * left_variance + n_right * right_variance) / total;
        let gain = parent_variance - weighted;
        if !weighted.is_finite() || !gain.is_finite() {
            continue;
        }
        // This midpoint defines a pointwise membership boundary, therefore it
        // remains f32 even though its gain calculation is f64.
        let threshold = pairs[index].0 * 0.5 + pairs[index + 1].0 * 0.5;
        if best.is_none_or(|(_, current_gain)| gain > current_gain) {
            best = Some((threshold, gain));
        }
    }
    best
}

fn best_variance_split_f64(feature: &[f64], target: &[f64]) -> Option<(f64, f64)> {
    let mut pairs = feature
        .iter()
        .copied()
        .zip(target.iter().copied())
        .filter(|(feature, target)| feature.is_finite() && target.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let total = pairs.len() as f64;
    let sum_all = pairs.iter().map(|pair| pair.1).sum::<f64>();
    let sum2_all = pairs.iter().map(|pair| pair.1 * pair.1).sum::<f64>();
    if !sum_all.is_finite() || !sum2_all.is_finite() {
        return None;
    }
    let parent_variance = sum2_all / total - (sum_all / total) * (sum_all / total);
    if !parent_variance.is_finite() || parent_variance <= 0.0 {
        return None;
    }
    let mut left_sum = 0.0f64;
    let mut left_sum2 = 0.0f64;
    let mut best = None;
    for index in 0..pairs.len() - 1 {
        left_sum += pairs[index].1;
        left_sum2 += pairs[index].1 * pairs[index].1;
        if pairs[index].0 == pairs[index + 1].0 {
            continue;
        }
        if !left_sum.is_finite() || !left_sum2.is_finite() {
            continue;
        }
        let n_left = (index + 1) as f64;
        let n_right = total - n_left;
        let Some(left_variance) = finite_nonnegative_path_stat(
            left_sum2 / n_left - (left_sum / n_left) * (left_sum / n_left),
        ) else {
            continue;
        };
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        if !right_sum.is_finite() || !right_sum2.is_finite() {
            continue;
        }
        let Some(right_variance) = finite_nonnegative_path_stat(
            right_sum2 / n_right - (right_sum / n_right) * (right_sum / n_right),
        ) else {
            continue;
        };
        let weighted = (n_left * left_variance + n_right * right_variance) / total;
        let gain = parent_variance - weighted;
        if !weighted.is_finite() || !gain.is_finite() {
            continue;
        }
        let threshold = pairs[index].0 * 0.5 + pairs[index + 1].0 * 0.5;
        if best.is_none_or(|(_, current_gain)| gain > current_gain) {
            best = Some((threshold, gain));
        }
    }
    best
}

/// Find the threshold on `feature` that maximizes variance reduction of `y`.
/// O(n log n): sort by feature, sweep boundaries maintaining running
/// left/right (sum, sum²) for incremental child variances. Returns `None` for a
/// constant feature, fewer than 2 finite pairs, or zero parent variance.
pub fn best_variance_split(feature: &[f32], y: &[f32]) -> Option<Split> {
    let n = feature.len().min(y.len());
    let mut pairs: Vec<(f32, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let (x, t) = (feature[i], y[i]);
        if x.is_finite() && t.is_finite() {
            pairs.push((x, t as f64));
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let total = pairs.len() as f64;
    let sum_all: f64 = pairs.iter().map(|p| p.1).sum();
    let sum2_all: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let parent_var = sum2_all / total - (sum_all / total).powi(2);
    if parent_var <= 0.0 {
        return None;
    }

    let mut left_sum = 0.0f64;
    let mut left_sum2 = 0.0f64;
    let mut best: Option<Split> = None;
    for i in 0..pairs.len() - 1 {
        left_sum += pairs[i].1;
        left_sum2 += pairs[i].1 * pairs[i].1;
        if pairs[i].0 == pairs[i + 1].0 {
            continue; // can't split between equal feature values
        }
        let n_left = (i + 1) as f64;
        let n_right = total - n_left;
        let var_left = (left_sum2 / n_left - (left_sum / n_left).powi(2)).max(0.0);
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        let var_right = (right_sum2 / n_right - (right_sum / n_right).powi(2)).max(0.0);
        let weighted = (n_left * var_left + n_right * var_right) / total;
        let gain = (parent_var - weighted) as f32;
        let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
        if best.is_none_or(|b| gain > b.gain) {
            best = Some(Split { threshold, gain });
        }
    }
    best
}

/// Materialize a split's membership indicator into `out`: 1.0 where
/// `feature >= threshold`, 0.0 where `< threshold`, NaN where the feature is NaN
/// (so the finite-pair scoring skips it).
pub fn split_indicator(feature: &[f32], threshold: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(feature.len());
    for &x in feature {
        out.push(if x.is_nan() {
            f32::NAN
        } else if x >= threshold {
            1.0
        } else {
            0.0
        });
    }
}

/// Side of a threshold split taken by a path node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitSign {
    /// `feature <= threshold`
    Le,
    /// `feature > threshold`
    Gt,
}

/// One node of a root->leaf conjunction path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathNode {
    pub feature: u32,
    pub threshold: f32,
    pub sign: SplitSign,
}

/// A decision-path candidate: a conjunction of threshold conditions (an
/// axis-aligned region), with the variance-reduction proxy `gain`, the number of
/// training rows it covers (`support`), and the boosting `round` that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionPath {
    pub nodes: Vec<PathNode>,
    pub gain: f32,
    pub support: u32,
    pub round: u32,
}

/// Depth-k recursion + residual-boosting controls for `find_decision_paths`.
#[derive(Clone, Copy, Debug)]
pub struct DecisionPathParams {
    pub max_depth: u32,
    pub rounds: u32,
    pub max_paths: u32,
    pub max_bins: u32,
    pub min_leaf: u32,
    /// Numeric execution input. Profile-specialized discovery converts this
    /// once to its statistical lane: f32 for `fp32`, f64 for `mixed`/`fp64`.
    pub learning_rate: f64,
}

/// A complete decision-path candidate in one canonical precision profile.
///
/// `nodes`, `support`, and `round` are structural metadata and retain their
/// integer representations.  `gain` is a public statistical value in the
/// profile's result dtype: f32 for `fp32`, f64 for `mixed` and `fp64`.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionDecisionPath {
    pub nodes: Vec<PrecisionPathNode>,
    pub gain: CpuPrecisionScalar,
    pub support: u32,
    pub round: u32,
}

/// Arithmetic used by a profile-specialized decision-path statistical lane.
/// The generic helpers below are monomorphized for f32 and f64; profile
/// dispatch happens at `find_decision_paths_precision`, not inside the tree
/// growth hot loops.
trait PathStat:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn from_usize(value: usize) -> Self;
    fn from_f64(value: f64) -> Self;
    fn finite(value: Self) -> bool;
    fn public(value: Self) -> CpuPrecisionScalar;
}

impl PathStat for f32 {
    fn zero() -> Self {
        0.0
    }

    fn from_usize(value: usize) -> Self {
        value as f32
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn finite(value: Self) -> bool {
        value.is_finite()
    }

    fn public(value: Self) -> CpuPrecisionScalar {
        CpuPrecisionScalar::F32(value)
    }
}

impl PathStat for f64 {
    fn zero() -> Self {
        0.0
    }

    fn from_usize(value: usize) -> Self {
        value as f64
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn finite(value: Self) -> bool {
        value.is_finite()
    }

    fn public(value: Self) -> CpuPrecisionScalar {
        CpuPrecisionScalar::F64(value)
    }
}

fn finite_nonnegative_path_stat<T: PathStat>(value: T) -> Option<T> {
    if !T::finite(value) {
        None
    } else if value > T::zero() {
        Some(value)
    } else {
        Some(T::zero())
    }
}

/// Resident/pointwise numeric type used for split comparisons and membership.
trait PathStorage: Copy + PartialOrd + PartialEq {
    fn finite(value: Self) -> bool;
    fn midpoint(left: Self, right: Self) -> Self;
    fn public(value: Self) -> CpuPrecisionScalar;
}

impl PathStorage for f32 {
    fn finite(value: Self) -> bool {
        value.is_finite()
    }

    fn midpoint(left: Self, right: Self) -> Self {
        // Split the scale before addition so finite dynamic-range inputs do
        // not overflow merely while constructing their boundary.
        left * 0.5 + right * 0.5
    }

    fn public(value: Self) -> CpuPrecisionScalar {
        CpuPrecisionScalar::F32(value)
    }
}

impl PathStorage for f64 {
    fn finite(value: Self) -> bool {
        value.is_finite()
    }

    fn midpoint(left: Self, right: Self) -> Self {
        left * 0.5 + right * 0.5
    }

    fn public(value: Self) -> CpuPrecisionScalar {
        CpuPrecisionScalar::F64(value)
    }
}

/// A compile-time-selected numeric decision-path lane.
trait DecisionPathLane {
    type Storage: PathStorage;
    type Stat: PathStat;

    fn stat_from_storage(value: Self::Storage) -> Self::Stat;
}

struct Fp32DecisionPathLane;
struct MixedDecisionPathLane;
struct Fp64DecisionPathLane;

impl DecisionPathLane for Fp32DecisionPathLane {
    type Storage = f32;
    type Stat = f32;

    fn stat_from_storage(value: Self::Storage) -> Self::Stat {
        value
    }
}

impl DecisionPathLane for MixedDecisionPathLane {
    type Storage = f32;
    type Stat = f64;

    fn stat_from_storage(value: Self::Storage) -> Self::Stat {
        value as f64
    }
}

impl DecisionPathLane for Fp64DecisionPathLane {
    type Storage = f64;
    type Stat = f64;

    fn stat_from_storage(value: Self::Storage) -> Self::Stat {
        value
    }
}

struct PrecisionLeaf<L: DecisionPathLane> {
    nodes: Vec<PrecisionPathNode>,
    indices: Vec<usize>,
    mean: L::Stat,
}

struct PrecisionPathWork<L: DecisionPathLane> {
    nodes: Vec<PrecisionPathNode>,
    gain: L::Stat,
    support: u32,
    round: u32,
}

/// Find all profile-aware paths using the same depth/round/min-leaf structural
/// planner as the established path family.
///
/// * fp32: resident values, residuals, covariance/variance, gains, ranking and
///   public values are f32.
/// * mixed: resident comparisons and materialized memberships are f32 while
///   residuals, split statistics, gains, ranking and public values are f64.
/// * fp64: values never enter an f32 buffer or arithmetic operation.
pub fn find_decision_paths_precision(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    target: CpuPrecisionSlice<'_>,
    params: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionDecisionPath>> {
    validate_precision_path_shape(columns, rows, cols, target)?;
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    // This is the one profile dispatch for a complete path-discovery artifact.
    // Each helper below is statically specialized by `L` and has no runtime
    // profile branch in its candidate, split, or residual loops.
    match (profile, columns, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        ) => find_decision_paths_lane::<Fp32DecisionPathLane>(columns, rows, cols, target, params),
        (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(columns),
            CpuPrecisionSlice::F32(target),
        ) => find_decision_paths_lane::<MixedDecisionPathLane>(columns, rows, cols, target, params),
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(columns),
            CpuPrecisionSlice::F64(target),
        ) => find_decision_paths_lane::<Fp64DecisionPathLane>(columns, rows, cols, target, params),
        _ => Err(OrchestratorError::InvalidPlan(
            "decision-path storage dtype does not match the requested precision profile",
        )),
    }
}

fn validate_precision_path_shape(
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    target: CpuPrecisionSlice<'_>,
) -> OrchestratorResult<()> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "decision-path matrix shape exceeds host address space",
        ))?;
    if columns.len() != expected {
        return Err(OrchestratorError::InvalidPlan(
            "decision-path column storage does not match the declared shape",
        ));
    }
    if target.len() != rows {
        return Err(OrchestratorError::InvalidPlan(
            "decision-path target storage does not match the declared rows",
        ));
    }
    Ok(())
}

fn find_decision_paths_lane<L: DecisionPathLane>(
    columns: &[L::Storage],
    rows: usize,
    cols: usize,
    target: &[L::Storage],
    params: &DecisionPathParams,
) -> OrchestratorResult<Vec<PrecisionDecisionPath>> {
    let max_depth = params.max_depth.max(1) as usize;
    let min_leaf = params.min_leaf.max(1) as usize;
    let rounds = params.rounds.max(1);
    // `learning_rate` is numeric execution input, not structural planner
    // metadata. Convert it exactly once when a profile-specialized executor is
    // built so fp64 never passes through f32 and fp32 still rounds to its lane.
    let learning_rate = L::Stat::from_f64(if params.learning_rate > 0.0 {
        params.learning_rate
    } else {
        1.0
    });
    let mut residual = target
        .iter()
        .copied()
        .map(L::stat_from_storage)
        .collect::<Vec<_>>();
    let all = (0..rows).collect::<Vec<_>>();
    let mut collected = Vec::<PrecisionPathWork<L>>::new();

    for round in 0..rounds {
        let mut leaves = Vec::<PrecisionLeaf<L>>::new();
        let mut prefix = Vec::<PrecisionPathNode>::new();
        grow_precision_path_lane::<L>(
            columns,
            rows,
            cols,
            &residual,
            &all,
            0,
            max_depth,
            min_leaf,
            params.max_bins,
            &mut prefix,
            &mut leaves,
        );

        let mut produced_path = false;
        for leaf in &leaves {
            if leaf.nodes.is_empty() {
                continue;
            }
            produced_path = true;
            let support = u32::try_from(leaf.indices.len())
                .map_err(|_| OrchestratorError::InvalidPlan("decision-path support exceeds u32"))?;
            let gain = L::Stat::from_usize(leaf.indices.len()) * leaf.mean * leaf.mean;
            collected.push(PrecisionPathWork::<L> {
                nodes: leaf.nodes.clone(),
                gain,
                support,
                round,
            });
        }
        if !produced_path {
            break;
        }
        // Leaves partition the row indices; `residual` therefore remains a
        // lane-typed statistical vector through every boosting round.
        for leaf in &leaves {
            let shrink = learning_rate * leaf.mean;
            for &row in &leaf.indices {
                residual[row] = residual[row] - shrink;
            }
        }
    }

    collected.sort_by(|left, right| {
        right
            .gain
            .partial_cmp(&left.gain)
            .unwrap_or(Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for path in collected {
        if seen.insert(precision_path_key(&path.nodes)) {
            paths.push(PrecisionDecisionPath {
                nodes: path.nodes,
                gain: L::Stat::public(path.gain),
                support: path.support,
                round: path.round,
            });
        }
    }
    paths.truncate(params.max_paths.max(1) as usize);
    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
fn grow_precision_path_lane<L: DecisionPathLane>(
    columns: &[L::Storage],
    rows: usize,
    cols: usize,
    residual: &[L::Stat],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    max_bins: u32,
    prefix: &mut Vec<PrecisionPathNode>,
    leaves: &mut Vec<PrecisionLeaf<L>>,
) {
    if depth >= max_depth || indices.len() < min_leaf.saturating_mul(2) {
        leaves.push(PrecisionLeaf::<L> {
            nodes: prefix.clone(),
            indices: indices.to_vec(),
            mean: precision_leaf_mean::<L>(residual, indices),
        });
        return;
    }

    let mut best: Option<(u32, L::Storage, L::Stat)> = None;
    let mut pairs = Vec::<(L::Storage, L::Stat)>::with_capacity(indices.len());
    for feature in 0..cols {
        let start = feature * rows;
        let column = &columns[start..start + rows];
        pairs.clear();
        for &row in indices {
            let value = column[row];
            let response = residual[row];
            if L::Storage::finite(value) && L::Stat::finite(response) {
                pairs.push((value, response));
            }
        }
        if let Some((threshold, gain)) =
            best_precision_split_subset::<L>(&mut pairs, min_leaf, max_bins)
        {
            if best
                .as_ref()
                .is_none_or(|(_, _, current_gain)| gain > *current_gain)
            {
                best = Some((feature as u32, threshold, gain));
            }
        }
    }

    let (feature, threshold, _) = match best {
        Some(found) if found.2 > L::Stat::zero() => found,
        _ => {
            leaves.push(PrecisionLeaf::<L> {
                nodes: prefix.clone(),
                indices: indices.to_vec(),
                mean: precision_leaf_mean::<L>(residual, indices),
            });
            return;
        }
    };
    let column = &columns[feature as usize * rows..(feature as usize + 1) * rows];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &row in indices {
        // NaN values preserve the established deterministic `>` branch. They
        // remain non-finite for scoring/membership rather than becoming a
        // floating-point planner surrogate.
        if column[row] <= threshold {
            left.push(row);
        } else {
            right.push(row);
        }
    }
    if left.len() < min_leaf || right.len() < min_leaf {
        leaves.push(PrecisionLeaf::<L> {
            nodes: prefix.clone(),
            indices: indices.to_vec(),
            mean: precision_leaf_mean::<L>(residual, indices),
        });
        return;
    }

    prefix.push(PrecisionPathNode {
        feature,
        threshold: L::Storage::public(threshold),
        sign: SplitSign::Le,
    });
    grow_precision_path_lane::<L>(
        columns,
        rows,
        cols,
        residual,
        &left,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();

    prefix.push(PrecisionPathNode {
        feature,
        threshold: L::Storage::public(threshold),
        sign: SplitSign::Gt,
    });
    grow_precision_path_lane::<L>(
        columns,
        rows,
        cols,
        residual,
        &right,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();
}

fn precision_leaf_mean<L: DecisionPathLane>(residual: &[L::Stat], indices: &[usize]) -> L::Stat {
    let mut sum = L::Stat::zero();
    let mut count = 0usize;
    for &row in indices {
        let value = residual[row];
        if L::Stat::finite(value) {
            sum = sum + value;
            count += 1;
        }
    }
    if count == 0 {
        L::Stat::zero()
    } else {
        sum / L::Stat::from_usize(count)
    }
}

fn best_precision_split_subset<L: DecisionPathLane>(
    pairs: &mut [(L::Storage, L::Stat)],
    min_leaf: usize,
    max_bins: u32,
) -> Option<(L::Storage, L::Stat)> {
    let count = pairs.len();
    let min_leaf = min_leaf.max(1);
    if count < min_leaf.saturating_mul(2) {
        return None;
    }
    pairs.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let total = L::Stat::from_usize(count);
    let sum_all = pairs.iter().fold(L::Stat::zero(), |sum, pair| sum + pair.1);
    let sum2_all = pairs
        .iter()
        .fold(L::Stat::zero(), |sum, pair| sum + pair.1 * pair.1);
    if !L::Stat::finite(total) || !L::Stat::finite(sum_all) || !L::Stat::finite(sum2_all) {
        return None;
    }
    let parent_variance = sum2_all / total - (sum_all / total) * (sum_all / total);
    if !L::Stat::finite(parent_variance)
        || parent_variance.partial_cmp(&L::Stat::zero()) != Some(Ordering::Greater)
    {
        return None;
    }
    let mut best = None;
    let mut consider = |index: usize, left_sum: L::Stat, left_sum2: L::Stat| {
        let left_count = index + 1;
        let right_count = count - left_count;
        if left_count < min_leaf || right_count < min_leaf || pairs[index].0 == pairs[index + 1].0 {
            return;
        }
        if !L::Stat::finite(left_sum) || !L::Stat::finite(left_sum2) {
            return;
        }
        let left_n = L::Stat::from_usize(left_count);
        let right_n = L::Stat::from_usize(right_count);
        let Some(left_variance) = finite_nonnegative_path_stat(
            left_sum2 / left_n - (left_sum / left_n) * (left_sum / left_n),
        ) else {
            return;
        };
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        if !L::Stat::finite(right_sum) || !L::Stat::finite(right_sum2) {
            return;
        }
        let Some(right_variance) = finite_nonnegative_path_stat(
            right_sum2 / right_n - (right_sum / right_n) * (right_sum / right_n),
        ) else {
            return;
        };
        let weighted = (left_n * left_variance + right_n * right_variance) / total;
        let gain = parent_variance - weighted;
        if !L::Stat::finite(weighted) || !L::Stat::finite(gain) {
            return;
        }
        if best
            .as_ref()
            .is_none_or(|(_, current_gain)| gain > *current_gain)
        {
            best = Some((
                L::Storage::midpoint(pairs[index].0, pairs[index + 1].0),
                gain,
            ));
        }
    };

    if max_bins == 0 {
        let mut left_sum = L::Stat::zero();
        let mut left_sum2 = L::Stat::zero();
        for (index, pair) in pairs.iter().enumerate().take(count - 1) {
            left_sum = left_sum + pair.1;
            left_sum2 = left_sum2 + pair.1 * pair.1;
            consider(index, left_sum, left_sum2);
        }
        return best;
    }

    let mut prefix_sum = Vec::with_capacity(count + 1);
    let mut prefix_sum2 = Vec::with_capacity(count + 1);
    prefix_sum.push(L::Stat::zero());
    prefix_sum2.push(L::Stat::zero());
    for pair in pairs.iter() {
        prefix_sum.push(prefix_sum.last().copied().unwrap_or_else(L::Stat::zero) + pair.1);
        prefix_sum2
            .push(prefix_sum2.last().copied().unwrap_or_else(L::Stat::zero) + pair.1 * pair.1);
    }
    let valid = (0..count - 1)
        .filter(|&index| {
            pairs[index].0 != pairs[index + 1].0
                && index + 1 >= min_leaf
                && count - (index + 1) >= min_leaf
        })
        .collect::<Vec<_>>();
    let cap = max_bins as usize;
    if valid.len() <= cap {
        for index in valid {
            consider(index, prefix_sum[index + 1], prefix_sum2[index + 1]);
        }
    } else if cap == 1 {
        let index = valid[valid.len() / 2];
        consider(index, prefix_sum[index + 1], prefix_sum2[index + 1]);
    } else {
        for position in 0..cap {
            let index = valid[position * (valid.len() - 1) / (cap - 1)];
            consider(index, prefix_sum[index + 1], prefix_sum2[index + 1]);
        }
    }
    best
}

fn precision_path_key(nodes: &[PrecisionPathNode]) -> String {
    let mut parts = nodes
        .iter()
        .map(|node| {
            let sign = match node.sign {
                SplitSign::Le => 'L',
                SplitSign::Gt => 'G',
            };
            match node.threshold {
                CpuPrecisionScalar::F32(value) => {
                    format!("{}:{sign}:f32:{:08x}", node.feature, value.to_bits())
                }
                CpuPrecisionScalar::F64(value) => {
                    format!("{}:{sign}:f64:{:016x}", node.feature, value.to_bits())
                }
            }
        })
        .collect::<Vec<_>>();
    parts.sort();
    parts.join("&")
}

/// Expand a row-major matrix with discovered profile-aware path memberships.
/// The base and appended numeric columns retain the profile storage dtype;
/// descriptors and output ordering remain structural integers/ordering.
pub fn expand_row_major_precision(
    profile: PrecisionProfile,
    features: CpuPrecisionSlice<'_>,
    target: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    params: &DecisionPathParams,
) -> OrchestratorResult<(CpuPrecisionValues, usize, Vec<PrecisionDecisionPath>)> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OrchestratorError::InvalidPlan(
            "decision-path expansion shape exceeds host address space",
        ))?;
    if features.len() != expected || target.len() != rows {
        return Err(OrchestratorError::InvalidPlan(
            "decision-path expansion input does not match the declared shape",
        ));
    }
    match (profile, features, target) {
        (
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(features),
            CpuPrecisionSlice::F32(target),
        )
        | (
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(features),
            CpuPrecisionSlice::F32(target),
        ) => expand_row_major_precision_f32(profile, features, target, rows, cols, params),
        (
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(features),
            CpuPrecisionSlice::F64(target),
        ) => expand_row_major_precision_f64(features, target, rows, cols, params),
        _ => Err(OrchestratorError::InvalidPlan(
            "decision-path storage dtype does not match the requested precision profile",
        )),
    }
}

fn expand_row_major_precision_f32(
    profile: PrecisionProfile,
    features: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    params: &DecisionPathParams,
) -> OrchestratorResult<(CpuPrecisionValues, usize, Vec<PrecisionDecisionPath>)> {
    let mut columns = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            columns[col * rows + row] = features[row * cols + col];
        }
    }
    let paths = find_decision_paths_precision(
        profile,
        CpuPrecisionSlice::F32(&columns),
        rows,
        cols,
        CpuPrecisionSlice::F32(target),
        params,
    )?;
    let memberships = paths
        .iter()
        .map(|path| {
            path_membership_precision(profile, CpuPrecisionSlice::F32(&columns), rows, &path.nodes)
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let expanded_cols = cols
        .checked_add(paths.len())
        .ok_or(OrchestratorError::InvalidPlan(
            "decision-path expanded column count overflows",
        ))?;
    let mut expanded = vec![
        0.0f32;
        rows.checked_mul(expanded_cols).ok_or(
            OrchestratorError::InvalidPlan(
                "decision-path expanded matrix exceeds host address space"
            ),
        )?
    ];
    for row in 0..rows {
        let destination = row * expanded_cols;
        expanded[destination..destination + cols]
            .copy_from_slice(&features[row * cols..row * cols + cols]);
        for (path_index, membership) in memberships.iter().enumerate() {
            let CpuPrecisionValues::F32(membership) = membership else {
                return Err(OrchestratorError::InvalidPlan(
                    "f32 decision-path expansion received f64 membership",
                ));
            };
            expanded[destination + cols + path_index] = membership[row];
        }
    }
    Ok((CpuPrecisionValues::F32(expanded), expanded_cols, paths))
}

fn expand_row_major_precision_f64(
    features: &[f64],
    target: &[f64],
    rows: usize,
    cols: usize,
    params: &DecisionPathParams,
) -> OrchestratorResult<(CpuPrecisionValues, usize, Vec<PrecisionDecisionPath>)> {
    let mut columns = vec![0.0f64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            columns[col * rows + row] = features[row * cols + col];
        }
    }
    let paths = find_decision_paths_precision(
        PrecisionProfile::Fp64,
        CpuPrecisionSlice::F64(&columns),
        rows,
        cols,
        CpuPrecisionSlice::F64(target),
        params,
    )?;
    let memberships = paths
        .iter()
        .map(|path| {
            path_membership_precision(
                PrecisionProfile::Fp64,
                CpuPrecisionSlice::F64(&columns),
                rows,
                &path.nodes,
            )
        })
        .collect::<OrchestratorResult<Vec<_>>>()?;
    let expanded_cols = cols
        .checked_add(paths.len())
        .ok_or(OrchestratorError::InvalidPlan(
            "decision-path expanded column count overflows",
        ))?;
    let mut expanded = vec![
        0.0f64;
        rows.checked_mul(expanded_cols).ok_or(
            OrchestratorError::InvalidPlan(
                "decision-path expanded matrix exceeds host address space"
            ),
        )?
    ];
    for row in 0..rows {
        let destination = row * expanded_cols;
        expanded[destination..destination + cols]
            .copy_from_slice(&features[row * cols..row * cols + cols]);
        for (path_index, membership) in memberships.iter().enumerate() {
            let CpuPrecisionValues::F64(membership) = membership else {
                return Err(OrchestratorError::InvalidPlan(
                    "fp64 decision-path expansion received f32 membership",
                ));
            };
            expanded[destination + cols + path_index] = membership[row];
        }
    }
    Ok((CpuPrecisionValues::F64(expanded), expanded_cols, paths))
}

/// Best variance-reduction split of a pre-gathered `(value, target)` subset,
/// enforcing at least `min_leaf` rows on each side. `pairs` is sorted in place.
fn best_split_subset(pairs: &mut [(f32, f64)], min_leaf: usize, max_bins: u32) -> Option<Split> {
    let n = pairs.len();
    let min_leaf = min_leaf.max(1);
    if n < 2 * min_leaf {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    let total = n as f64;
    let sum_all: f64 = pairs.iter().map(|p| p.1).sum();
    let sum2_all: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let parent_var = sum2_all / total - (sum_all / total).powi(2);
    if parent_var <= 0.0 {
        return None;
    }
    let mut best: Option<Split> = None;
    let mut consider = |i: usize, left_sum: f64, left_sum2: f64| {
        let n_left = i + 1usize;
        let n_right = n - n_left;
        if n_left < min_leaf || n_right < min_leaf {
            return;
        }
        let nl = n_left as f64;
        let nr = n_right as f64;
        let var_left = (left_sum2 / nl - (left_sum / nl).powi(2)).max(0.0);
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        let var_right = (right_sum2 / nr - (right_sum / nr).powi(2)).max(0.0);
        let weighted = (nl * var_left + nr * var_right) / total;
        let gain = (parent_var - weighted) as f32;
        let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
        if best.is_none_or(|b| gain > b.gain) {
            best = Some(Split { threshold, gain });
        }
    };

    if max_bins == 0 {
        let mut left_sum = 0.0f64;
        let mut left_sum2 = 0.0f64;
        for i in 0..n - 1 {
            left_sum += pairs[i].1;
            left_sum2 += pairs[i].1 * pairs[i].1;
            if pairs[i].0 != pairs[i + 1].0 {
                consider(i, left_sum, left_sum2);
            }
        }
        return best;
    }

    let mut prefix_sum = Vec::with_capacity(n + 1);
    let mut prefix_sum2 = Vec::with_capacity(n + 1);
    prefix_sum.push(0.0f64);
    prefix_sum2.push(0.0f64);
    for pair in pairs.iter() {
        prefix_sum.push(prefix_sum.last().copied().unwrap_or_default() + pair.1);
        prefix_sum2.push(prefix_sum2.last().copied().unwrap_or_default() + pair.1 * pair.1);
    }
    let valid = (0..n - 1)
        .filter(|&i| pairs[i].0 != pairs[i + 1].0 && i + 1 >= min_leaf && n - (i + 1) >= min_leaf)
        .collect::<Vec<_>>();
    let cap = max_bins as usize;
    if valid.len() <= cap {
        for i in valid {
            consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
        }
    } else if cap == 1 {
        let i = valid[valid.len() / 2];
        consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
    } else {
        for position in 0..cap {
            let i = valid[position * (valid.len() - 1) / (cap - 1)];
            consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
        }
    }
    best
}

struct LeafAcc {
    path: Vec<PathNode>,
    indices: Vec<usize>,
    mean: f32,
}

fn column(columns: &[f32], rows: usize, feature: usize) -> &[f32] {
    &columns[feature * rows..(feature + 1) * rows]
}

fn leaf_mean(residual: &[f32], indices: &[usize]) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for &i in indices {
        let r = residual[i];
        if r.is_finite() {
            sum += r as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Greedy CART growth against a fixed `residual`. Emits one `LeafAcc` per
/// root->leaf region (leaves partition the rows). `prefix` carries the current
/// conjunction; `scratch` is reused to gather per-feature subsets.
#[allow(clippy::too_many_arguments)]
fn grow(
    columns: &[f32],
    rows: usize,
    cols: usize,
    residual: &[f32],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    max_bins: u32,
    prefix: &mut Vec<PathNode>,
    leaves: &mut Vec<LeafAcc>,
) {
    if depth >= max_depth || indices.len() < 2 * min_leaf {
        leaves.push(LeafAcc {
            path: prefix.clone(),
            mean: leaf_mean(residual, indices),
            indices: indices.to_vec(),
        });
        return;
    }

    let mut best: Option<(u32, Split)> = None;
    let mut pairs: Vec<(f32, f64)> = Vec::with_capacity(indices.len());
    for feature in 0..cols {
        let col = column(columns, rows, feature);
        pairs.clear();
        for &i in indices {
            let x = col[i];
            let r = residual[i];
            if x.is_finite() && r.is_finite() {
                pairs.push((x, r as f64));
            }
        }
        if let Some(split) = best_split_subset(&mut pairs, min_leaf, max_bins) {
            if best.is_none_or(|(_, current)| split.gain > current.gain) {
                best = Some((feature as u32, split));
            }
        }
    }

    let (feature, split) = match best {
        Some(found) if found.1.gain > 0.0 => found,
        _ => {
            leaves.push(LeafAcc {
                path: prefix.clone(),
                mean: leaf_mean(residual, indices),
                indices: indices.to_vec(),
            });
            return;
        }
    };

    let col = column(columns, rows, feature as usize);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in indices {
        // NaN feature values follow the ">" branch deterministically; they carry
        // no split information but must land somewhere consistent.
        if col[i] <= split.threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.len() < min_leaf || right.len() < min_leaf {
        leaves.push(LeafAcc {
            path: prefix.clone(),
            mean: leaf_mean(residual, indices),
            indices: indices.to_vec(),
        });
        return;
    }

    prefix.push(PathNode {
        feature,
        threshold: split.threshold,
        sign: SplitSign::Le,
    });
    grow(
        columns,
        rows,
        cols,
        residual,
        &left,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();

    prefix.push(PathNode {
        feature,
        threshold: split.threshold,
        sign: SplitSign::Gt,
    });
    grow(
        columns,
        rows,
        cols,
        residual,
        &right,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();
}

fn path_key(nodes: &[PathNode]) -> String {
    let mut parts: Vec<String> = nodes
        .iter()
        .map(|n| {
            let sign = match n.sign {
                SplitSign::Le => 'L',
                SplitSign::Gt => 'G',
            };
            // Quantize the threshold so numerically-identical splits dedup.
            format!("{}:{}:{:.5}", n.feature, sign, n.threshold)
        })
        .collect();
    parts.sort();
    parts.join("&")
}

/// Discover decision-path conjunctions via depth-k greedy CART with residual
/// boosting. Each boosting round fits a depth-`max_depth` tree to the current
/// residual, records its leaf regions as candidate paths, then subtracts
/// `learning_rate * leaf_mean` from the residual (standard gradient boosting on
/// squared error). Paths are deduplicated and the top `max_paths` by `gain` are
/// returned. `columns` is column-major (`columns[f*rows + i]`).
pub fn find_decision_paths(
    columns: &[f32],
    rows: usize,
    cols: usize,
    target: &[f32],
    params: &DecisionPathParams,
) -> Vec<DecisionPath> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let max_depth = params.max_depth.max(1) as usize;
    let min_leaf = params.min_leaf.max(1) as usize;
    let rounds = params.rounds.max(1);
    let learning_rate = if params.learning_rate > 0.0 {
        params.learning_rate as f32
    } else {
        1.0
    };

    let mut residual: Vec<f32> = target.to_vec();
    let mut collected: Vec<DecisionPath> = Vec::new();
    let all: Vec<usize> = (0..rows).collect();

    for round in 0..rounds {
        let mut leaves: Vec<LeafAcc> = Vec::new();
        let mut prefix: Vec<PathNode> = Vec::new();
        grow(
            columns,
            rows,
            cols,
            &residual,
            &all,
            0,
            max_depth,
            min_leaf,
            params.max_bins,
            &mut prefix,
            &mut leaves,
        );

        let mut produced_path = false;
        for leaf in &leaves {
            if leaf.path.is_empty() {
                continue; // degenerate (no split found) -> not a usable feature
            }
            produced_path = true;
            let support = leaf.indices.len() as u32;
            let gain = support as f32 * leaf.mean * leaf.mean;
            collected.push(DecisionPath {
                nodes: leaf.path.clone(),
                gain,
                support,
                round,
            });
        }
        if !produced_path {
            break; // nothing splits any more -> boosting has converged
        }
        // Residual update: leaves partition the rows, so each row is adjusted by
        // exactly one leaf mean.
        for leaf in &leaves {
            let shrink = learning_rate * leaf.mean;
            for &i in &leaf.indices {
                residual[i] -= shrink;
            }
        }
    }

    collected.sort_by(|a, b| {
        b.gain
            .partial_cmp(&a.gain)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for path in collected {
        if seen.insert(path_key(&path.nodes)) {
            unique.push(path);
        }
    }
    unique.truncate(params.max_paths.max(1) as usize);
    unique
}

/// Materialize a path's hard-AND membership indicator (column-major input): 1.0
/// where every condition holds, 0.0 where any concrete condition fails, NaN only
/// when membership is undetermined because a still-satisfied path hits a NaN
/// feature (so finite-pair scoring skips it).
pub fn path_membership(columns: &[f32], rows: usize, nodes: &[PathNode]) -> Vec<f32> {
    let mut out = vec![1.0f32; rows];
    for i in 0..rows {
        let mut member = 1.0f32;
        let mut undetermined = false;
        for node in nodes {
            let x = columns[node.feature as usize * rows + i];
            if x.is_nan() {
                undetermined = true;
                continue;
            }
            let holds = match node.sign {
                SplitSign::Le => x <= node.threshold,
                SplitSign::Gt => x > node.threshold,
            };
            if !holds {
                member = 0.0;
            }
        }
        out[i] = if member == 0.0 {
            0.0
        } else if undetermined {
            f32::NAN
        } else {
            1.0
        };
    }
    out
}

/// Expand a row-major feature matrix with decision-path membership columns
/// appended after the `cols` base features, mirroring `time_series::expand_row_major`
/// so the continuous engine mines base + path features on any backend. Returns
/// (expanded row-major, expanded column count, discovered paths in appended order).
pub fn expand_row_major(
    features: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    params: &DecisionPathParams,
) -> (Vec<f32>, usize, Vec<DecisionPath>) {
    if rows == 0 || cols == 0 {
        return (features.to_vec(), cols, Vec::new());
    }
    let mut colmajor = vec![0.0f32; rows * cols];
    for t in 0..rows {
        let base = t * cols;
        for c in 0..cols {
            colmajor[c * rows + t] = features[base + c];
        }
    }
    let paths = find_decision_paths(&colmajor, rows, cols, target, params);
    let n_new = paths.len();
    let membership: Vec<Vec<f32>> = paths
        .iter()
        .map(|path| path_membership(&colmajor, rows, &path.nodes))
        .collect();

    let ecols = cols + n_new;
    let mut expanded = vec![0.0f32; rows * ecols];
    for t in 0..rows {
        let src = t * cols;
        let dst = t * ecols;
        expanded[dst..dst + cols].copy_from_slice(&features[src..src + cols]);
        for (j, column) in membership.iter().enumerate() {
            expanded[dst + cols + j] = column[t];
        }
    }
    (expanded, ecols, paths)
}

/// Human-readable label for a path, e.g. `path[f0>3.5000 & f2<=1.2000]`.
pub fn path_label(feature_names: &[String], nodes: &[PathNode]) -> String {
    let parts: Vec<String> = nodes
        .iter()
        .map(|node| {
            let name = feature_names
                .get(node.feature as usize)
                .map(String::as_str)
                .unwrap_or("f");
            let op = match node.sign {
                SplitSign::Le => "<=",
                SplitSign::Gt => ">",
            };
            format!("{name}{op}{:.4}", node.threshold)
        })
        .collect();
    format!("path[{}]", parts.join(" & "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_split_profiles_keep_their_declared_numeric_lanes() {
        let feature = [1.0f32, 2.0, 3.0, 4.0];
        let target = [0.0f32, 0.0, 10.0, 10.0];
        let fp32 = best_variance_split_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&feature),
            CpuPrecisionSlice::F32(&target),
        )
        .unwrap()
        .unwrap();
        let mixed = best_variance_split_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&feature),
            CpuPrecisionSlice::F32(&target),
        )
        .unwrap()
        .unwrap();
        assert!(matches!(fp32.threshold, CpuPrecisionScalar::F32(_)));
        assert!(matches!(fp32.gain, CpuPrecisionScalar::F32(_)));
        assert!(matches!(mixed.threshold, CpuPrecisionScalar::F32(_)));
        assert!(matches!(mixed.gain, CpuPrecisionScalar::F64(_)));
        assert_eq!(fp32.threshold.f32(), Some(2.5));
        assert_eq!(mixed.threshold.f32(), Some(2.5));
    }

    #[test]
    fn precision_split_profiles_fail_closed_on_nonfinite_variance_arithmetic() {
        let feature_f32 = [0.0f32, 1.0, 2.0, 3.0];
        let target_f32 = [2.0e19f32, 2.0e19, -2.0e19, -2.0e19];
        assert!(best_variance_split_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&feature_f32),
            CpuPrecisionSlice::F32(&target_f32),
        )
        .unwrap()
        .is_none());

        let mixed = best_variance_split_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&feature_f32),
            CpuPrecisionSlice::F32(&target_f32),
        )
        .unwrap()
        .expect("mixed f64 statistics keep the finite f32 workload evaluable");
        assert!(mixed.gain.f64().is_some_and(f64::is_finite));

        let feature_f64 = [0.0f64, 1.0, 2.0, 3.0];
        let target_f64 = [2.0e200f64, 2.0e200, -2.0e200, -2.0e200];
        assert!(best_variance_split_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&feature_f64),
            CpuPrecisionSlice::F64(&target_f64),
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn precision_split_midpoints_do_not_overflow_finite_feature_ranges() {
        let feature_f32 = [f32::MAX / 2.0, f32::MAX];
        let target_f32 = [0.0f32, 1.0];
        for profile in [PrecisionProfile::Fp32, PrecisionProfile::Mixed] {
            let split = best_variance_split_precision(
                profile,
                CpuPrecisionSlice::F32(&feature_f32),
                CpuPrecisionSlice::F32(&target_f32),
            )
            .unwrap()
            .unwrap();
            assert!(split.threshold.f32().is_some_and(f32::is_finite));
        }

        let feature_f64 = [f64::MAX / 2.0, f64::MAX];
        let target_f64 = [0.0f64, 1.0];
        let split = best_variance_split_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&feature_f64),
            CpuPrecisionSlice::F64(&target_f64),
        )
        .unwrap()
        .unwrap();
        assert!(split.threshold.f64().is_some_and(f64::is_finite));
    }

    #[test]
    fn fp64_decision_path_keeps_adjacent_binary64_thresholds_and_membership() {
        let base = 1.0f64;
        let values = [
            base,
            f64::from_bits(base.to_bits() + 8),
            f64::from_bits(base.to_bits() + 16),
            f64::from_bits(base.to_bits() + 24),
        ];
        let target = [0.0f64, 0.0, 1.0, 1.0];
        let split = best_variance_split_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&values),
            CpuPrecisionSlice::F64(&target),
        )
        .unwrap()
        .unwrap();
        assert!(matches!(split.threshold, CpuPrecisionScalar::F64(_)));
        let threshold = split.threshold.f64().unwrap();
        assert!(threshold > values[1] && threshold < values[2]);
        let membership = path_membership_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&values),
            4,
            &[PrecisionPathNode {
                feature: 0,
                threshold: split.threshold,
                sign: SplitSign::Gt,
            }],
        )
        .unwrap();
        let CpuPrecisionValues::F64(membership) = membership else {
            panic!("fp64 membership must stay f64")
        };
        assert_eq!(membership, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn precision_membership_rejects_f64_profile_over_f32_columns() {
        assert!(path_membership_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F32(&[1.0, 2.0]),
            2,
            &[],
        )
        .is_err());
    }

    fn precision_path_params() -> DecisionPathParams {
        DecisionPathParams {
            max_depth: 1,
            rounds: 1,
            max_paths: 4,
            max_bins: 0,
            min_leaf: 1,
            learning_rate: 1.0,
        }
    }

    #[test]
    fn full_precision_path_discovery_dispatches_all_three_lanes_once() {
        let columns = [1.0f32, 2.0, 3.0, 4.0];
        let target = [0.0f32, 0.0, 1.0, 1.0];
        let fp32 = find_decision_paths_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&columns),
            4,
            1,
            CpuPrecisionSlice::F32(&target),
            &precision_path_params(),
        )
        .unwrap();
        let mixed = find_decision_paths_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&columns),
            4,
            1,
            CpuPrecisionSlice::F32(&target),
            &precision_path_params(),
        )
        .unwrap();
        assert!(!fp32.is_empty());
        assert_eq!(fp32.len(), mixed.len());
        for path in &fp32 {
            assert!(matches!(path.gain, CpuPrecisionScalar::F32(_)));
            assert!(path
                .nodes
                .iter()
                .all(|node| matches!(node.threshold, CpuPrecisionScalar::F32(_))));
            assert!(path.support > 0);
        }
        for path in &mixed {
            assert!(matches!(path.gain, CpuPrecisionScalar::F64(_)));
            assert!(path
                .nodes
                .iter()
                .all(|node| matches!(node.threshold, CpuPrecisionScalar::F32(_))));
        }
    }

    #[test]
    fn full_precision_path_discovery_rejects_fabricated_nonfinite_gain() {
        let feature_f32 = [0.0f32, 1.0, 2.0, 3.0];
        let target_f32 = [2.0e19f32, 2.0e19, -2.0e19, -2.0e19];
        let params = DecisionPathParams {
            max_bins: 1,
            ..precision_path_params()
        };
        let fp32 = find_decision_paths_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&feature_f32),
            4,
            1,
            CpuPrecisionSlice::F32(&target_f32),
            &params,
        )
        .unwrap();
        assert!(fp32.is_empty());

        let mixed = find_decision_paths_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&feature_f32),
            4,
            1,
            CpuPrecisionSlice::F32(&target_f32),
            &params,
        )
        .unwrap();
        assert!(!mixed.is_empty());
        assert!(mixed
            .iter()
            .all(|path| path.gain.f64().is_some_and(f64::is_finite)));

        let feature_f64 = [0.0f64, 1.0, 2.0, 3.0];
        let target_f64 = [2.0e200f64, 2.0e200, -2.0e200, -2.0e200];
        let fp64 = find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&feature_f64),
            4,
            1,
            CpuPrecisionSlice::F64(&target_f64),
            &params,
        )
        .unwrap();
        assert!(fp64.is_empty());
    }

    #[test]
    fn fp32_path_statistics_do_not_widen_the_public_gain() {
        // The unit is lost when a binary32 residual accumulates around this
        // dynamic range. The path result must nevertheless remain f32 rather
        // than exposing a hidden f64 reduction as an fp32 public lane.
        let columns = [0.0f32, 1.0, 2.0, 3.0];
        let target = [16_777_216.0f32, 1.0, -16_777_216.0, 2.0];
        let paths = find_decision_paths_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&columns),
            4,
            1,
            CpuPrecisionSlice::F32(&target),
            &precision_path_params(),
        )
        .unwrap();
        assert!(paths
            .iter()
            .all(|path| matches!(path.gain, CpuPrecisionScalar::F32(_))));
        assert!(paths.iter().all(|path| path.gain.f64().is_none()));
    }

    #[test]
    fn fp64_full_discovery_and_expansion_preserve_values_that_collapse_to_fp32() {
        let base = 1.0f64;
        let features = [
            base,
            f64::from_bits(base.to_bits() + 8),
            f64::from_bits(base.to_bits() + 16),
            f64::from_bits(base.to_bits() + 24),
        ];
        assert!(features
            .windows(2)
            .all(|pair| (pair[0] as f32).to_bits() == (pair[1] as f32).to_bits()));
        let target = [0.0f64, 0.0, 1.0, 1.0];
        let paths = find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&features),
            4,
            1,
            CpuPrecisionSlice::F64(&target),
            &precision_path_params(),
        )
        .unwrap();
        assert!(!paths.is_empty());
        let top = &paths[0];
        assert!(matches!(top.gain, CpuPrecisionScalar::F64(_)));
        let threshold = top.nodes[0].threshold.f64().unwrap();
        assert!(threshold > features[1] && threshold < features[2]);

        let (expanded, expanded_cols, expanded_paths) = expand_row_major_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&features),
            CpuPrecisionSlice::F64(&target),
            4,
            1,
            &precision_path_params(),
        )
        .unwrap();
        assert_eq!(expanded_paths, paths);
        assert!(expanded_cols > 1);
        let CpuPrecisionValues::F64(expanded) = expanded else {
            panic!("fp64 decision-path expansion must stay f64")
        };
        for row in 0..4 {
            assert_eq!(
                expanded[row * expanded_cols].to_bits(),
                features[row].to_bits()
            );
        }
    }

    #[test]
    fn fp64_learning_rate_is_not_quantized_through_binary32() {
        let columns = [
            0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 3.0, 8.0, 1.0, 10.0,
            5.0, 0.0, 11.0, 4.0, 9.0, 2.0, 7.0, 6.0,
        ];
        let target = [
            -2.0f64, 1.0, 4.0, -1.0, 6.0, 2.0, 9.0, 3.0, 8.0, 0.0, 7.0, 5.0,
        ];
        let base = DecisionPathParams {
            max_depth: 1,
            rounds: 3,
            max_paths: 12,
            max_bins: 0,
            min_leaf: 1,
            learning_rate: 1.0,
        };
        let precise = DecisionPathParams {
            learning_rate: 1.0 + 2.0f64.powi(-40),
            ..base
        };
        assert_eq!(precise.learning_rate as f32, base.learning_rate as f32);

        let base_paths = find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&columns),
            12,
            2,
            CpuPrecisionSlice::F64(&target),
            &base,
        )
        .unwrap();
        let precise_paths = find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&columns),
            12,
            2,
            CpuPrecisionSlice::F64(&target),
            &precise,
        )
        .unwrap();

        assert_ne!(precise_paths, base_paths);
    }

    #[test]
    fn full_precision_path_discovery_rejects_mismatched_storage_before_growth() {
        assert!(find_decision_paths_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F32(&[1.0, 2.0]),
            2,
            1,
            CpuPrecisionSlice::F32(&[0.0, 1.0]),
            &precision_path_params(),
        )
        .is_err());
    }

    #[test]
    fn finds_the_obvious_threshold() {
        // clean step: y jumps from 0 to 10 at x=3.5
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0f32, 0.0, 0.0, 10.0, 10.0, 10.0];
        let s = best_variance_split(&x, &y).unwrap();
        assert!(
            (s.threshold - 3.5).abs() < 1e-6,
            "threshold={}",
            s.threshold
        );
        assert!(s.gain > 0.0);
    }

    #[test]
    fn constant_feature_has_no_split() {
        let x = vec![2.0f32, 2.0, 2.0, 2.0];
        let y = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn constant_target_has_no_split() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 5.0, 5.0, 5.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn split_boundary_cap_samples_candidates_without_subsampling_rows() {
        let pairs = (0..10)
            .map(|value| (value as f32, if value >= 2 { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let mut exhaustive_pairs = pairs.clone();
        let mut capped_pairs = pairs;

        let exhaustive = best_split_subset(&mut exhaustive_pairs, 1, 0).unwrap();
        let capped = best_split_subset(&mut capped_pairs, 1, 1).unwrap();

        assert_eq!(exhaustive.threshold, 1.5);
        assert_eq!(capped.threshold, 4.5);
        assert!(exhaustive.gain > capped.gain);
    }

    #[test]
    fn indicator_materializes_membership_and_skips_nan() {
        let x = vec![1.0f32, 4.0, f32::NAN, 5.0];
        let mut out = Vec::new();
        split_indicator(&x, 3.5, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert!(out[2].is_nan());
        assert_eq!(out[3], 1.0);
    }

    fn params(max_depth: u32) -> DecisionPathParams {
        DecisionPathParams {
            max_depth,
            rounds: 1,
            max_paths: 8,
            max_bins: 0,
            min_leaf: 2,
            learning_rate: 1.0,
        }
    }

    #[test]
    fn finds_a_depth2_and_conjunction() {
        // y is high only in the (f0 high AND f1 high) quadrant -> a depth-2 tree
        // must recover that 2-condition conjunction as its top-gain path.
        let mut f0 = Vec::new();
        let mut f1 = Vec::new();
        let mut y = Vec::new();
        for q0 in 0..2 {
            for q1 in 0..2 {
                for k in 0..10 {
                    f0.push((if q0 == 0 { 0.2 } else { 0.8 }) + 0.001 * k as f32);
                    f1.push((if q1 == 0 { 0.2 } else { 0.8 }) + 0.001 * k as f32);
                    y.push(if q0 == 1 && q1 == 1 { 5.0 } else { 0.0 });
                }
            }
        }
        let rows = y.len();
        let mut columns = Vec::with_capacity(rows * 2);
        columns.extend_from_slice(&f0);
        columns.extend_from_slice(&f1);

        let paths = find_decision_paths(&columns, rows, 2, &y, &params(2));
        assert!(!paths.is_empty());
        let top = &paths[0];
        assert_eq!(top.nodes.len(), 2, "top path should be a 2-way conjunction");
        // The top region must select exactly the high-y rows.
        let member = path_membership(&columns, rows, &top.nodes);
        for i in 0..rows {
            if member[i] == 1.0 {
                assert_eq!(y[i], 5.0, "row {i} in top region should be high-y");
            }
        }
        let selected = member.iter().filter(|&&m| m == 1.0).count();
        assert_eq!(
            selected, 10,
            "top region should be the 10-row high quadrant"
        );
    }

    #[test]
    fn path_membership_is_hard_and_with_nan_skip() {
        // 4 rows, 2 features (column-major).
        let columns = vec![
            0.0, 1.0, 0.0, 1.0, // f0
            0.0, 0.0, 1.0, 1.0, // f1
        ];
        let nodes = vec![
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ];
        let member = path_membership(&columns, 4, &nodes);
        assert_eq!(member, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn expand_appends_membership_columns_and_labels() {
        // 8-row AND structure -> at least one path column appended.
        let features = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let p = DecisionPathParams {
            max_depth: 2,
            rounds: 1,
            max_paths: 4,
            max_bins: 0,
            min_leaf: 1,
            learning_rate: 1.0,
        };
        let (expanded, ecols, paths) = expand_row_major(&features, &target, 8, 2, &p);
        assert!(!paths.is_empty());
        assert_eq!(ecols, 2 + paths.len());
        assert_eq!(expanded.len(), 8 * ecols);
        let label = path_label(&["f0".to_string(), "f1".to_string()], &paths[0].nodes);
        assert!(label.starts_with("path["), "label={label}");
    }

    #[test]
    fn boosting_dedups_and_caps_paths() {
        // Repeated rounds on the same structure must not emit duplicate paths, and
        // max_paths caps the output.
        let mut f0 = Vec::new();
        let mut y = Vec::new();
        for k in 0..40 {
            let v = k as f32 / 40.0;
            f0.push(v);
            y.push(if v > 0.5 { 3.0 } else { 0.0 });
        }
        let rows = y.len();
        let p = DecisionPathParams {
            max_depth: 1,
            rounds: 5,
            max_paths: 2,
            max_bins: 0,
            min_leaf: 2,
            learning_rate: 0.5,
        };
        let paths = find_decision_paths(&f0, rows, 1, &y, &p);
        assert!(paths.len() <= 2, "max_paths must cap output");
        let mut keys: Vec<String> = paths.iter().map(|p| path_key(&p.nodes)).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), paths.len(), "paths must be deduplicated");
    }
}
