pub mod precision;

use gafime_orchestrator::{
    plan::combos::{
        sanitize_mi_bins_for_backend, MI_SAMPLES_PER_JOINT_BIN, MI_TEMPLATE_BIN_LEVELS,
    },
    OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    MetricId, GAFIME_BACKEND_CPU, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON,
    GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
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
    let mut cached_pearson = None;

    for metric in metrics {
        let value = match metric {
            MetricKernel::Pearson => {
                *cached_pearson.get_or_insert_with(|| pearson(signal, matrix.target()))
            }
            MetricKernel::Spearman => spearman(signal, matrix.target()),
            MetricKernel::MutualInfo => {
                if mi_approximate {
                    mutual_info_fixed(signal, matrix.target(), mi_bins)
                } else {
                    mutual_info(signal, matrix.target(), mi_bins)
                }
            }
            MetricKernel::R2 => {
                let corr = *cached_pearson.get_or_insert_with(|| pearson(signal, matrix.target()));
                (corr * corr).clamp(0.0, 1.0)
            }
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
    let actual_bins = select_adaptive_mi_bins(n, max_bins, MI_SAMPLES_PER_JOINT_BIN as usize, 2);
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
/// equal-width bins over [min,max], finite-sample-corrected, and normalized by
/// log(min(active_x, active_y)). Bin arithmetic intentionally stays in `f32`
/// and defines overflow exactly like the CUDA/ROCm/Metal kernels.
/// Unlike `mutual_info` (adaptive quantile bins) this needs no sort, so the bin
/// mapping vectorizes (`simd::fixed_bin_histogram2d`); the unavoidable
/// data-dependent histogram scatter is fed from SIMD lane bins. Chosen only when
/// MI approximation is requested; adaptive stays default.
pub fn mutual_info_fixed(x: &[f32], y: &[f32], bins: u32) -> f32 {
    const MAX_FIXED_MI_BINS: usize = 96;

    let bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, bins) as usize;
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
    let active_y = hist_y
        .iter()
        .take(bins)
        .filter(|&&count| count != 0)
        .count() as u32;
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

fn select_adaptive_mi_bins(
    n_samples: usize,
    max_bins: u32,
    samples_per_bin: usize,
    dimensions: u32,
) -> u32 {
    let max_bins = sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, max_bins);
    let samples_per_bin = samples_per_bin.max(1);
    let dimensions = dimensions.max(1);
    let mut best = 2u32;
    for &level in MI_TEMPLATE_BIN_LEVELS {
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
    let max_bins = max_bins.clamp(2, *MI_TEMPLATE_BIN_LEVELS.last().unwrap()) as usize;
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
    let mut block_start = 0usize;
    let mut previous_raw_bin = None;
    let mut compact_bin = 0usize;
    while block_start < n {
        let mut block_end = block_start + 1;
        while block_end < n && values[order[block_end]] == values[order[block_start]] {
            block_end += 1;
        }

        // Assign the entire equal-valued block by its mid-rank. u128 keeps the
        // rank arithmetic defined even for theoretical usize-sized inputs.
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
    fn adaptive_mi_bins_use_intermediate_dense_histogram_shapes() {
        assert_eq!(select_adaptive_mi_bins(1_151, 96, 8, 2), 8);
        assert_eq!(select_adaptive_mi_bins(1_152, 96, 8, 2), 12);
        assert_eq!(select_adaptive_mi_bins(4_608, 96, 8, 2), 24);
        assert_eq!(select_adaptive_mi_bins(18_432, 96, 8, 2), 48);
        assert_eq!(select_adaptive_mi_bins(100_000, 20, 8, 2), 16);
    }

    #[test]
    fn adaptive_mi_keeps_ties_together_and_is_row_permutation_invariant() {
        let x = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0];
        let (x_bins, x_count) = adaptive_bin_indices(&x, 2, true);
        assert_eq!(x_count, 2);
        assert!(x_bins[..5].iter().all(|&bin| bin == x_bins[0]));

        let permutation = [4usize, 5, 0, 1, 2, 3, 6, 7];
        let permuted_x = permutation.map(|index| x[index]);
        let permuted_y = permutation.map(|index| y[index]);
        assert_eq!(
            mutual_info(&x, &y, 2).to_bits(),
            mutual_info(&permuted_x, &permuted_y, 2).to_bits()
        );
    }

    #[test]
    fn fixed_mi_sanitizes_unsupported_bin_ceilings_downward() {
        let x: Vec<f32> = (0..512).map(|index| (index % 97) as f32).collect();
        let y: Vec<f32> = (0..512)
            .map(|index| ((index * 17 + index / 7) % 89) as f32)
            .collect();

        assert_eq!(
            mutual_info_fixed(&x, &y, 20).to_bits(),
            mutual_info_fixed(&x, &y, 16).to_bits()
        );
        assert_eq!(
            mutual_info_fixed(&x, &y, 0).to_bits(),
            mutual_info_fixed(&x, &y, 2).to_bits()
        );
    }

    #[test]
    fn fixed_mi_is_finite_for_extreme_wide_and_subnormal_ranges() {
        let mut wide_x = Vec::new();
        let mut wide_y = Vec::new();
        let wide_pattern = [
            -f32::MAX,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        for _ in 0..32 {
            for (index, &value) in wide_pattern.iter().enumerate() {
                wide_x.push(value);
                wide_y.push((index % 4) as f32);
            }
        }
        let wide_score = mutual_info_fixed(&wide_x, &wide_y, 8);
        assert!(wide_score.is_finite());
        assert_eq!(wide_score, 0.0);

        let subnormal_x: Vec<f32> = (0..256)
            .map(|index| f32::from_bits((index % 9) as u32))
            .collect();
        let subnormal_y: Vec<f32> = (0..256).map(|index| (index % 9) as f32).collect();
        let subnormal_score = mutual_info_fixed(&subnormal_x, &subnormal_y, 8);
        assert!(subnormal_score.is_finite());
        assert!(subnormal_score > 0.0);
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

    #[test]
    fn pearson_and_r2_match_scalar_and_materialized_references_for_arities_one_through_five() {
        let rows = 97usize;
        let matrix = matrix_from_columns(&test_columns(rows), test_target(rows));
        let metrics = [MetricKernel::Pearson, MetricKernel::R2];

        for arity in 1..=5 {
            let combo: Vec<u32> = (0..arity as u32).collect();
            let (scalar, materialized) = materialized_pearson_r2_references(&matrix, &combo);
            let mut scratch = ContinuousScoreScratch::default();
            let actual =
                score_continuous_combo_into(&matrix, &combo, &metrics, 12, false, &mut scratch)
                    .unwrap()
                    .to_vec();

            assert_eq!(
                actual[0].to_bits(),
                materialized[0].to_bits(),
                "arity={arity}"
            );
            assert_eq!(
                actual[1].to_bits(),
                materialized[1].to_bits(),
                "arity={arity}"
            );
            if arity > 1 {
                assert_eq!(scratch.interaction.len(), rows, "arity={arity}");
            }
            assert!(
                (actual[0] - scalar[0]).abs() <= 5.0e-5,
                "arity={arity}, actual={}, scalar={}",
                actual[0],
                scalar[0]
            );
            assert!(
                (actual[1] - scalar[1]).abs() <= 1.0e-4,
                "arity={arity}, actual={}, scalar={}",
                actual[1],
                scalar[1]
            );
        }
    }

    #[test]
    fn pearson_and_r2_preserve_non_finite_and_constant_column_behavior() {
        let rows = 41usize;
        let metrics = [MetricKernel::Pearson, MetricKernel::R2];

        let mut target_with_non_finite = test_target(rows);
        target_with_non_finite[5] = f32::NAN;
        target_with_non_finite[13] = f32::INFINITY;
        target_with_non_finite[29] = f32::NEG_INFINITY;
        assert_exact_reference_scores(
            &matrix_from_columns(&test_columns(rows), target_with_non_finite),
            &metrics,
        );

        let mut columns_with_non_finite = test_columns(rows);
        columns_with_non_finite[0][7] = f32::NAN;
        columns_with_non_finite[0][23] = f32::INFINITY;
        assert_exact_reference_scores(
            &matrix_from_columns(&columns_with_non_finite, test_target(rows)),
            &metrics,
        );

        let mut columns_with_constant = test_columns(rows);
        columns_with_constant[0].fill(3.5);
        let matrix = matrix_from_columns(&columns_with_constant, test_target(rows));
        assert_exact_reference_scores(&matrix, &metrics);
        for arity in 1..=5 {
            let combo: Vec<u32> = (0..arity as u32).collect();
            let scores = score_continuous_combo(&matrix, &combo, &metrics, 12, false).unwrap();
            assert_eq!(scores, vec![0.0, 0.0], "arity={arity}");
        }
    }

    #[test]
    fn mixed_metrics_materialize_once_for_all_slice_kernels() {
        let rows = 73usize;
        let matrix = matrix_from_columns(&test_columns(rows), test_target(rows));
        let combo = [0, 1];
        let metrics = [
            MetricKernel::Pearson,
            MetricKernel::Spearman,
            MetricKernel::MutualInfo,
            MetricKernel::R2,
        ];
        let mut interaction = Vec::new();
        build_interaction_vector_into(&matrix, &combo, &mut interaction);
        let expected = [
            pearson(&interaction, matrix.target()),
            spearman(&interaction, matrix.target()),
            mutual_info_fixed(&interaction, matrix.target(), 12),
            simd::r2_score(&interaction, matrix.target()),
        ];
        let mut scratch = ContinuousScoreScratch::default();
        let actual = score_continuous_combo_into(&matrix, &combo, &metrics, 12, true, &mut scratch)
            .unwrap()
            .to_vec();

        assert_eq!(scratch.interaction, interaction);
        for (metric, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "metric={metric}");
        }
    }

    fn assert_exact_reference_scores(matrix: &CpuMatrix, metrics: &[MetricKernel]) {
        for arity in 1..=5 {
            let combo: Vec<u32> = (0..arity as u32).collect();
            let (scalar, materialized) = materialized_pearson_r2_references(matrix, &combo);
            let actual = score_continuous_combo(matrix, &combo, metrics, 12, false).unwrap();

            assert_eq!(actual[0].to_bits(), scalar[0].to_bits(), "arity={arity}");
            assert_eq!(actual[1].to_bits(), scalar[1].to_bits(), "arity={arity}");
            assert_eq!(
                actual[0].to_bits(),
                materialized[0].to_bits(),
                "arity={arity}"
            );
            assert_eq!(
                actual[1].to_bits(),
                materialized[1].to_bits(),
                "arity={arity}"
            );
        }
    }

    fn materialized_pearson_r2_references(
        matrix: &CpuMatrix,
        combo: &[u32],
    ) -> ([f32; 2], [f32; 2]) {
        let mut interaction = Vec::new();
        let signal = if combo.len() == 1 {
            matrix.column(combo[0] as usize)
        } else {
            build_interaction_vector_into(matrix, combo, &mut interaction);
            interaction.as_slice()
        };
        let scalar_pearson = simd::pearson_sums_scalar(signal, matrix.target()).pearson();
        let scalar_r2 = (scalar_pearson * scalar_pearson).clamp(0.0, 1.0);
        (
            [scalar_pearson, scalar_r2],
            [
                pearson(signal, matrix.target()),
                simd::r2_score(signal, matrix.target()),
            ],
        )
    }

    fn matrix_from_columns(columns: &[Vec<f32>], target: Vec<f32>) -> CpuMatrix {
        let rows = target.len();
        assert!(columns.iter().all(|column| column.len() == rows));
        let mut features = Vec::with_capacity(rows * columns.len());
        for row in 0..rows {
            for column in columns {
                features.push(column[row]);
            }
        }
        CpuMatrix::from_row_major(rows as u64, columns.len() as u32, features, target).unwrap()
    }

    fn test_columns(rows: usize) -> Vec<Vec<f32>> {
        (0..5)
            .map(|feature| {
                (0..rows)
                    .map(|row| {
                        let t = row as f32;
                        let phase = feature as f32 + 1.0;
                        (t * (0.071 * phase)).sin()
                            + (t * (0.113 + 0.017 * phase)).cos() * 0.5
                            + t * (0.003 * phase)
                    })
                    .collect()
            })
            .collect()
    }

    fn test_target(rows: usize) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let t = row as f32;
                (t * 0.049).sin() * 0.75 + (t * 0.127).cos() * 0.25 + t * 0.011
            })
            .collect()
    }
}
