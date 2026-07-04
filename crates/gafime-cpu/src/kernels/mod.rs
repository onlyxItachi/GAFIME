use gafime_orchestrator::{OrchestratorError, OrchestratorResult};
use gafime_types::{
    MetricId, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_METRIC_SPEARMAN,
};

use crate::matrix::CpuMatrix;
use crate::simd;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKernel {
    Pearson,
    Spearman,
    MutualInfo,
    R2,
}

impl TryFrom<MetricId> for MetricKernel {
    type Error = OrchestratorError;

    fn try_from(value: MetricId) -> Result<Self, Self::Error> {
        match value {
            GAFIME_METRIC_PEARSON => Ok(Self::Pearson),
            GAFIME_METRIC_SPEARMAN => Ok(Self::Spearman),
            GAFIME_METRIC_MUTUAL_INFO => Ok(Self::MutualInfo),
            GAFIME_METRIC_R2 => Ok(Self::R2),
            _ => Err(OrchestratorError::Unsupported("unknown metric id")),
        }
    }
}

pub fn planned_kernel_names() -> &'static [&'static str] {
    &["pearson", "spearman", "mutual_info", "r2"]
}

#[derive(Debug, Default)]
pub struct ContinuousScoreScratch {
    interaction: Vec<f32>,
    scores: Vec<f32>,
}

pub fn score_continuous_combo(
    matrix: &CpuMatrix,
    combo: &[u32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
) -> OrchestratorResult<Vec<f32>> {
    let mut scratch = ContinuousScoreScratch::default();
    Ok(score_continuous_combo_into(
        matrix,
        combo,
        metrics,
        mi_bins,
        mi_approximate,
        &mut scratch,
    )?
    .to_vec())
}

pub fn score_continuous_combo_into<'a>(
    matrix: &CpuMatrix,
    combo: &[u32],
    metrics: &[MetricKernel],
    mi_bins: u32,
    mi_approximate: bool,
    scratch: &'a mut ContinuousScoreScratch,
) -> OrchestratorResult<&'a [f32]> {
    if combo.is_empty() {
        return Err(OrchestratorError::InvalidPlan("continuous combo is empty"));
    }
    for &feature in combo {
        if feature >= matrix.cols() {
            return Err(OrchestratorError::InvalidPlan(
                "continuous combo feature is out of bounds",
            ));
        }
    }

    let signal = if combo.len() == 1 {
        matrix.column(combo[0] as usize)
    } else {
        build_interaction_vector_into(matrix, combo, &mut scratch.interaction);
        &scratch.interaction
    };
    scratch.scores.clear();
    scratch.scores.reserve(metrics.len());
    for metric in metrics {
        let value = match metric {
            MetricKernel::Pearson => pearson(signal, matrix.target()),
            MetricKernel::Spearman => spearman(signal, matrix.target()),
            MetricKernel::MutualInfo => {
                if mi_approximate {
                    mutual_info_fixed(signal, matrix.target(), mi_bins)
                } else {
                    mutual_info(signal, matrix.target(), mi_bins)
                }
            }
            MetricKernel::R2 => simd::r2_score(signal, matrix.target()),
        };
        scratch.scores.push(value);
    }
    Ok(&scratch.scores)
}

pub fn pearson(x: &[f32], y: &[f32]) -> f32 {
    simd::pearson_corr(x, y)
}

pub fn spearman(x: &[f32], y: &[f32]) -> f32 {
    let (x_values, y_values) = finite_pairs(x, y);
    if x_values.is_empty() {
        return 0.0;
    }
    let x_ranks = rankdata(&x_values);
    let y_ranks = rankdata(&y_values);
    pearson(&x_ranks, &y_ranks)
}

pub fn mutual_info(x: &[f32], y: &[f32], max_bins: u32) -> f32 {
    let (x_values, y_values) = finite_pairs(x, y);
    let n = x_values.len();
    if n <= 1 || constant(&x_values) || constant(&y_values) {
        return 0.0;
    }
    let actual_bins = select_adaptive_mi_bins(n, max_bins, 8, 2);
    if actual_bins < 2 {
        return 0.0;
    }
    let (x_bins, x_count) = adaptive_bin_indices(&x_values, actual_bins, true);
    let (y_bins, y_count) = adaptive_bin_indices(&y_values, actual_bins, true);
    if x_count < 2 || y_count < 2 {
        return 0.0;
    }

    let mut joint = vec![0.0f64; x_count * y_count];
    for (&x_bin, &y_bin) in x_bins.iter().zip(&y_bins) {
        joint[x_bin * y_count + y_bin] += 1.0;
    }
    corrected_mi_from_joint(&joint, x_count, y_count) as f32
}

/// Fixed equal-width-bin mutual information (the opt-in "approximation backend"):
/// the exact algorithm the CUDA/ROCm MI kernel uses — equal-width bins over
/// [min,max], finite-sample-corrected, normalized by log(min(active_x, active_y)).
/// Unlike `mutual_info` (adaptive quantile bins) this needs no sort, so the bin
/// mapping vectorizes (`simd::fixed_bin_histogram2d`); the unavoidable
/// data-dependent histogram scatter is fed from SIMD lane bins. Chosen only when
/// MI approximation is requested; adaptive stays default.
pub fn mutual_info_fixed(x: &[f32], y: &[f32], bins: u32) -> f32 {
    const MAX_FIXED_MI_BINS: usize = 96;

    let bins = bins.clamp(2, 96) as usize;
    let (x_values, y_values) = finite_pairs(x, y);
    let n = x_values.len();
    if n <= 1 {
        return 0.0;
    }
    let mut min_x = x_values[0];
    let mut max_x = x_values[0];
    let mut min_y = y_values[0];
    let mut max_y = y_values[0];
    for i in 0..n {
        min_x = min_x.min(x_values[i]);
        max_x = max_x.max(x_values[i]);
        min_y = min_y.min(y_values[i]);
        max_y = max_y.max(y_values[i]);
    }
    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }
    let inv_x = bins as f32 / (max_x - min_x);
    let inv_y = bins as f32 / (max_y - min_y);

    let mut hist_x = [0u32; MAX_FIXED_MI_BINS];
    let mut hist_y = [0u32; MAX_FIXED_MI_BINS];
    let mut joint = [0u32; MAX_FIXED_MI_BINS * MAX_FIXED_MI_BINS];
    simd::fixed_bin_histogram2d(
        &x_values,
        &y_values,
        min_x,
        inv_x,
        min_y,
        inv_y,
        bins as u32,
        &mut hist_x,
        &mut hist_y,
        &mut joint,
    );

    let total = n as f64;
    let mut mi = 0.0f64;
    let mut active_x = 0u32;
    let mut active_y = 0u32;
    for a in 0..bins {
        if hist_x[a] == 0 {
            continue;
        }
        active_x += 1;
        let px = hist_x[a] as f64 / total;
        for b in 0..bins {
            let count = joint[a * bins + b];
            if count == 0 || hist_y[b] == 0 {
                continue;
            }
            let py = hist_y[b] as f64 / total;
            let pxy = count as f64 / total;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    for b in 0..bins {
        if hist_y[b] != 0 {
            active_y += 1;
        }
    }
    let correction = if active_x > 0 && active_y > 0 {
        ((active_x - 1) as f64 * (active_y - 1) as f64) / (2.0 * total)
    } else {
        0.0
    };
    let corrected = (mi - correction).max(0.0);
    let normalizer_bins = active_x.min(active_y);
    let normalizer = if normalizer_bins > 1 {
        (normalizer_bins as f64).ln()
    } else {
        0.0
    };
    if normalizer > 0.0 {
        (corrected / normalizer) as f32
    } else {
        0.0
    }
}

fn build_interaction_vector_into(matrix: &CpuMatrix, combo: &[u32], out: &mut Vec<f32>) {
    let rows = matrix.rows() as usize;
    out.clear();
    out.resize(rows, 1.0);

    for &feature in combo {
        let col = feature as usize;
        let mean = matrix.column_mean(col);
        for (product, &value) in out.iter_mut().zip(matrix.column(col)) {
            *product *= value - mean;
        }
    }
}

fn finite_pairs(x: &[f32], y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if x.len() != y.len() {
        return (Vec::new(), Vec::new());
    }
    let mut out_x = Vec::with_capacity(x.len());
    let mut out_y = Vec::with_capacity(y.len());
    for (&x_value, &y_value) in x.iter().zip(y) {
        if x_value.is_finite() && y_value.is_finite() {
            out_x.push(x_value);
            out_y.push(y_value);
        }
    }
    (out_x, out_y)
}

fn rankdata(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(left.cmp(&right))
    });

    let mut ranks = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        let avg_rank = 0.5f32 * (i + j - 1) as f32;
        for pos in i..j {
            ranks[order[pos]] = avg_rank;
        }
        i = j;
    }
    ranks
}

const ADAPTIVE_MI_BIN_LEVELS: &[u32] = &[2, 4, 8, 16, 32, 64, 96];

fn select_adaptive_mi_bins(
    n_samples: usize,
    max_bins: u32,
    samples_per_bin: usize,
    dimensions: u32,
) -> u32 {
    let max_bins = max_bins.clamp(2, *ADAPTIVE_MI_BIN_LEVELS.last().unwrap());
    let samples_per_bin = samples_per_bin.max(1);
    let dimensions = dimensions.max(1);
    let mut best = 2u32;
    for &level in ADAPTIVE_MI_BIN_LEVELS {
        if level > max_bins {
            break;
        }
        let required = samples_per_bin.saturating_mul(level.saturating_pow(dimensions) as usize);
        if n_samples >= required {
            best = level;
        }
    }
    best
}

fn adaptive_bin_indices(
    values: &[f32],
    max_bins: u32,
    exact_low_cardinality: bool,
) -> (Vec<usize>, usize) {
    let n = values.len();
    let max_bins = max_bins.clamp(2, *ADAPTIVE_MI_BIN_LEVELS.last().unwrap()) as usize;
    if n == 0 {
        return (Vec::new(), 0);
    }

    let mut unique = values.to_vec();
    unique.sort_by(|left, right| {
        left.partial_cmp(right)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    unique.dedup_by(|left, right| *left == *right);
    if unique.len() <= 1 {
        return (vec![0; n], 1);
    }
    if exact_low_cardinality && unique.len() <= max_bins {
        let bins = values
            .iter()
            .map(|value| {
                unique
                    .binary_search_by(|probe| {
                        probe
                            .partial_cmp(value)
                            .unwrap_or(core::cmp::Ordering::Equal)
                    })
                    .unwrap()
            })
            .collect();
        return (bins, unique.len());
    }

    let bins = max_bins.min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(left.cmp(&right))
    });
    let mut out = vec![0usize; n];
    for (pos, &idx) in order.iter().enumerate() {
        let bin_id = (pos * bins) / n;
        out[idx] = bin_id.min(bins - 1);
    }
    (out, bins)
}

fn corrected_mi_from_joint(joint: &[f64], row_count: usize, col_count: usize) -> f64 {
    if row_count == 0 || col_count == 0 || joint.is_empty() {
        return 0.0;
    }
    let total: f64 = joint.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut px = vec![0.0f64; row_count];
    let mut py = vec![0.0f64; col_count];
    for row in 0..row_count {
        for col in 0..col_count {
            let count = joint[row * col_count + col];
            px[row] += count;
            py[col] += count;
        }
    }
    let nonzero_rows = px.iter().filter(|&&value| value > 0.0).count();
    let nonzero_cols = py.iter().filter(|&&value| value > 0.0).count();
    if nonzero_rows < 2 || nonzero_cols < 2 {
        return 0.0;
    }

    let inv_total = 1.0 / total;
    let mut mi = 0.0f64;
    for row in 0..row_count {
        for col in 0..col_count {
            let count = joint[row * col_count + col];
            if count <= 0.0 {
                continue;
            }
            let pxy = count * inv_total;
            let expected = (px[row] * inv_total) * (py[col] * inv_total);
            if expected > 0.0 {
                mi += pxy * (pxy / expected).ln();
            }
        }
    }
    let bias = ((nonzero_rows - 1) * (nonzero_cols - 1)) as f64 / (2.0 * total);
    (mi - bias).max(0.0)
}

fn constant(values: &[f32]) -> bool {
    if values.is_empty() {
        return true;
    }
    let mut min = values[0];
    let mut max = values[0];
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    min == max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_and_r2_match_simple_linear_signal() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];

        assert!((pearson(&x, &y) - 1.0).abs() < 1e-6);
        assert!((pearson(&x, &[-2.0, -4.0, -6.0, -8.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn spearman_uses_average_tie_ranks() {
        let x = [1.0, 1.0, 2.0, 3.0];
        let ranks = rankdata(&x);
        assert_eq!(ranks, vec![0.5, 0.5, 2.0, 3.0]);
    }

    #[test]
    fn mutual_info_ignores_constant_inputs() {
        assert_eq!(mutual_info(&[1.0, 1.0, 1.0], &[0.0, 1.0, 0.0], 96), 0.0);
        assert_eq!(
            mutual_info_fixed(&[1.0, 1.0, 1.0], &[0.0, 1.0, 0.0], 96),
            0.0
        );
    }

    #[test]
    fn score_continuous_combo_selects_fixed_bin_mi_when_requested() {
        let rows = 16u64;
        let features = (0..rows)
            .flat_map(|row| {
                let v = row as f32;
                [v.sin() + v * 0.07, (v % 5.0) * 0.25 + v * 0.01]
            })
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| {
                let v = row as f32;
                v.cos() * 0.5 + v * 0.03
            })
            .collect::<Vec<_>>();
        let matrix = CpuMatrix::from_row_major(rows, 2, features, target).unwrap();
        let metrics = [MetricKernel::MutualInfo];

        let adaptive = score_continuous_combo(&matrix, &[0], &metrics, 12, false).unwrap();
        let fixed = score_continuous_combo(&matrix, &[0], &metrics, 12, true).unwrap();

        assert_eq!(
            adaptive[0],
            mutual_info(matrix.column(0), matrix.target(), 12)
        );
        assert_eq!(
            fixed[0],
            mutual_info_fixed(matrix.column(0), matrix.target(), 12)
        );
    }
}
